//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-tcp/Conversion/TorchToTcp/TorchToTcp.h"

#include "mlir-tcp/Dialect/IR/TcpDialect.h"
#include "mlir-tcp/Dialect/IR/TcpOps.h"

#include "PopulatePatterns.h"
#include "Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

using namespace mlir;
using namespace mlir::tcp;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

template <typename AtenOpT>
bool checkZerosOnesOpAttributes(AtenOpT op, RankedTensorType outType) {
  // check output type
  if (!outType)
    return false;
  if (!outType.getElementType().isIntOrFloat())
    return false;

  // check default layout
  int64_t memoryLayout;
  if (!isa<Torch::NoneType>(op.getLayout().getType()) &&
      (!matchPattern(op.getLayout(), m_TorchConstantInt(&memoryLayout)) ||
       memoryLayout != 0)) {
    return false;
  }

  // check default pin_memory
  bool pinMemory;
  if (!isa<Torch::NoneType>(op.getPinMemory().getType()) &&
      (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)) ||
       pinMemory)) {
    return false;
  }

  return true;
}

template <typename AtenOpT>
class ConvertAtenBroadcastLikeOps : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    RankedTensorType inputType = dyn_cast<RankedTensorType>(input.getType());

    ArrayRef<int64_t> inputShape = inputType.getShape();

    SmallVector<Value> newDimSizes;
    if (!getListConstructElements(op.getSize(), newDimSizes))
      return rewriter.notifyMatchFailure(
          op, "Broadcasted shape must be a list of scalars");

    int64_t newLeadingDims = newDimSizes.size() - inputType.getRank();
    if (newLeadingDims > 0)
      input = torch_to_tcp::broadcastRankInLeadingDims(rewriter, input,
                                                       newLeadingDims);

    SmallVector<int64_t> axes;
    SmallVector<Value> resultShape;
    ArrayRef<int64_t> newInputShape =
        dyn_cast<RankedTensorType>(input.getType()).getShape();
    for (int64_t i = 0; i < static_cast<int64_t>(newDimSizes.size()); ++i) {
      Value newDimSize = newDimSizes[i];

      // Per PyTorch, "Tensor can be also expanded to a larger number of
      // dimensions, and the new ones will be appended at the front. For
      // the new dimensions, the size cannot be set to -1.", so this dim is
      // always broadcasted (no need to check for `isDimSizePreserved` below)
      bool isNewDim = i < newLeadingDims;
      // PyTorch semantics allow for a broadcast to a dynamic dimension;
      // for example: `torch.broadcast_to(x, y.size())` where `y.size()` has
      // dynamic shapes. When broadcasting to dynamic size, matchPattern for
      // `torch.constant.int` fails, and we do not read `staticDimSize` (an int)
      // out of `newDimSize` (an mlir::Value). In the "static" case
      // (`torch.broadcast_to(x, (3, 3))`) matchPattern will succeed.
      int64_t staticDimSize;
      bool isDynamicDim =
          !matchPattern(newDimSize, m_TorchConstantInt(&staticDimSize));
      // Per PyTorch, "passing -1 as the size for a dimension means not
      // changing the size of that dimension". Don't evaluate if `isDynamicDim`
      // (as `staticDimSize` won't have a valid value).
      bool isDimSizePreserved = isDynamicDim ? false : staticDimSize == -1;
      // Don't evaluate if `isNewDim` (to prevent out of bounds access on
      // `inputShape`) or if `isDynamicDim` (as `staticDimSize` won't have
      // a valid value).
      bool doesDimSizeChange =
          (isDynamicDim || isNewDim)
              ? true
              : staticDimSize != inputShape[i - newLeadingDims];

      bool isInputDimBroadcastable = newInputShape[i] == 1;
      // Note: The order of checks in this boolean expression matters!
      bool isOutputDimBroadcastable =
          isNewDim || isDynamicDim ||
          (!isDimSizePreserved && doesDimSizeChange);
      if (isInputDimBroadcastable && isOutputDimBroadcastable) {
        axes.push_back(i);
        newDimSize = rewriter.create<torch::TorchConversion::ToI64Op>(
            op->getLoc(), newDimSize);
        resultShape.push_back(rewriter.create<arith::IndexCastOp>(
            op->getLoc(), rewriter.getIndexType(), newDimSize));
      }
    }

    // fold the broadcast if no axes are found
    if (axes.size() == 0) {
      rewriter.replaceOp(op, input);
      return success();
    }
    RankedTensorType resultType = cast<RankedTensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op->getResult(0).getType()));
    auto axesAttr = rewriter.getI64ArrayAttr(axes);
    rewriter.replaceOpWithNewOp<tcp::BroadcastOp>(op, resultType, input,
                                                  resultShape, axesAttr);
    return success();
  }
};

class ConvertValueTensorLiteralOp
    : public OpConversionPattern<ValueTensorLiteralOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ValueTensorLiteralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

    if (auto elements = dyn_cast<DenseIntElementsAttr>(op.getValueAttr())) {
      Type elementType = resultType.getElementType();
      auto denseIntAttr = elements.mapValues(elementType, [&](const APInt &v) {
        return APInt(elementType.getIntOrFloatBitWidth(), v.getSExtValue());
      });
      rewriter.replaceOpWithNewOp<tcp::ConstOp>(op, resultType, denseIntAttr);
      return success();
    }
    if (auto elements =
            dyn_cast<DenseResourceElementsAttr>(op.getValueAttr())) {
      if (resultType.getElementType().isInteger() &&
          resultType != adaptor.getValue().getType()) {
        auto attr =
            DenseResourceElementsAttr::get(resultType, elements.getRawHandle());
        rewriter.replaceOpWithNewOp<tcp::ConstOp>(op, resultType, attr);
        return success();
      }
    }

    rewriter.replaceOpWithNewOp<tcp::ConstOp>(op, resultType,
                                              adaptor.getValue());
    return success();
  }
};

class ConvertAtenSizeIntOp : public OpConversionPattern<AtenSizeIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenSizeIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value self = adaptor.getSelf();
    auto type = cast<RankedTensorType>(self.getType());
    if (!isa<ConstantIntOp>(op->getOperand(1).getDefiningOp())) {
      return rewriter.notifyMatchFailure(op, "dim must be a constant int");
    }
    auto constIntOp =
        dyn_cast<ConstantIntOp>(op->getOperand(1).getDefiningOp());
    int idxVal = constIntOp.getValueAttr().getValue().getSExtValue();
    if (idxVal < 0 || idxVal >= type.getRank()) {
      return rewriter.notifyMatchFailure(op, "dim must be in range");
    }
    auto idxOp = rewriter.create<arith::ConstantIndexOp>(loc, idxVal);
    auto dimOp = rewriter.create<tensor::DimOp>(loc, self, idxOp);
    auto result =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), dimOp);

    rewriter.replaceOp(op, result);

    return success();
  }
};

template <typename AtenOpT, int fillVal>
class ConvertAtenZerosOnesOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto outType = dyn_cast<RankedTensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));
    Type outElemTy = outType.getElementType();

    if (!checkZerosOnesOpAttributes<AtenOpT>(op, outType)) {
      return rewriter.notifyMatchFailure(op, "Attribute checks failed");
    }

    Value constOp;
    if (!torch_to_tcp::getConstTensorWithType(rewriter, op, constOp, outElemTy,
                                              fillVal)) {
      return rewriter.notifyMatchFailure(op, "Unsupported output element type");
    }

    Operation *primListOp = op.getSize().getDefiningOp();
    auto listConstruct = dyn_cast<Torch::PrimListConstructOp>(primListOp);
    if (!listConstruct) {
      return rewriter.notifyMatchFailure(
          op, "Size must come from PrimListConstructOp");
    }
    SmallVector<Value> primListVal;
    for (Value value : listConstruct.getElements()) {
      primListVal.push_back(value);
    }

    SmallVector<int64_t> resultShape =
        torch_to_tcp::getShapeFromPrimList(primListVal);
    Value resultOp = torch_to_tcp::broadcast0DOr1DFromShape(
        rewriter, constOp, primListVal, resultShape);

    rewriter.replaceOp(op, resultOp);

    return success();
  }
};

template <typename AtenOpT, int fillVal>
class ConvertAtenZerosOnesLikeOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto outType = dyn_cast<RankedTensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));
    Type outElemTy = outType.getElementType();

    // TODO: Check the attribute for input vtensor
    if (!isa<Torch::NoneType>(op.getMemoryFormat().getType()))
      return rewriter.notifyMatchFailure(
          op, "Only default memory format is supported");

    if (!checkZerosOnesOpAttributes<AtenOpT>(op, outType)) {
      return rewriter.notifyMatchFailure(op, "Attribute checks failed");
    }

    Value constOp;
    if (!torch_to_tcp::getConstTensorWithType(rewriter, op, constOp, outElemTy,
                                              fillVal)) {
      return rewriter.notifyMatchFailure(op, "Unsupported output element type");
    }

    Value resultOp = torch_to_tcp::broadcast0DOr1DToNDAndMatchShape(
        rewriter, constOp, input,
        cast<RankedTensorType>(constOp.getType()).getElementType());

    rewriter.replaceOp(op, resultOp);

    return success();
  }
};

class ConvertSymbolicIntOp : public OpConversionPattern<Torch::SymbolicIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Torch::SymbolicIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());

    rewriter.replaceOpWithNewOp<tcp::SymbolicIntOp>(
        op, resultType, adaptor.getSymbolNameAttr(), adaptor.getMinValAttr(),
        adaptor.getMaxValAttr());
    return success();
  }
};

class ConvertBindSymbolicShapeOp
    : public OpConversionPattern<Torch::BindSymbolicShapeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Torch::BindSymbolicShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<tcp::BindSymbolicShapeOp>(
        op, adaptor.getOperand(), adaptor.getShapeSymbols(),
        adaptor.getShapeExpressionsAttr());
    return success();
  }
};

} // namespace

void torch_to_tcp::populateMiscPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const llvm::StringSet<> &convertTorchOpsSet) {

  torch_to_tcp::addPatternIfOpInConvertTorchOpsSet<ConvertSymbolicIntOp,
                                                   Torch::SymbolicIntOp>(
      typeConverter, patterns, target, convertTorchOpsSet);
  torch_to_tcp::addPatternIfOpInConvertTorchOpsSet<ConvertBindSymbolicShapeOp,
                                                   Torch::BindSymbolicShapeOp>(
      typeConverter, patterns, target, convertTorchOpsSet);

#define INSERT_ATEN_MISC_OP_PATTERN(AtenOp)                                    \
  torch_to_tcp::addPatternIfOpInConvertTorchOpsSet<Convert##AtenOp, AtenOp>(   \
      typeConverter, patterns, target, convertTorchOpsSet)
  INSERT_ATEN_MISC_OP_PATTERN(ValueTensorLiteralOp);
  INSERT_ATEN_MISC_OP_PATTERN(AtenSizeIntOp);
#undef INSERT_ATEN_MISC_OP_PATTERN

#define INSERT_ATEN_BROADCAST_PATTERN(AtenOp)                                  \
  torch_to_tcp::addPatternIfOpInConvertTorchOpsSet<                            \
      ConvertAtenBroadcastLikeOps<AtenOp>, AtenOp>(typeConverter, patterns,    \
                                                   target, convertTorchOpsSet)
  INSERT_ATEN_BROADCAST_PATTERN(AtenBroadcastToOp);
  INSERT_ATEN_BROADCAST_PATTERN(AtenExpandOp);
#undef INSERT_ATEN_BROADCAST_PATTERN

#define INSERT_ATEN_ZEROS_ONES_PATTERN(ConvertAtenOpPattern, AtenOp, Val)      \
  torch_to_tcp::addPatternIfOpInConvertTorchOpsSet<                            \
      ConvertAtenOpPattern<AtenOp, Val>, AtenOp>(typeConverter, patterns,      \
                                                 target, convertTorchOpsSet)
  INSERT_ATEN_ZEROS_ONES_PATTERN(ConvertAtenZerosOnesOp, AtenZerosOp, 0);
  INSERT_ATEN_ZEROS_ONES_PATTERN(ConvertAtenZerosOnesOp, AtenOnesOp, 1);
  INSERT_ATEN_ZEROS_ONES_PATTERN(ConvertAtenZerosOnesLikeOp, AtenZerosLikeOp,
                                 0);
  INSERT_ATEN_ZEROS_ONES_PATTERN(ConvertAtenZerosOnesLikeOp, AtenOnesLikeOp, 1);
#undef INSERT_ATEN_ZEROS_ONES_PATTERN
}
