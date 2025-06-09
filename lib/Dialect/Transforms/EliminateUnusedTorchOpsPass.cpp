//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-tcp/Dialect/Transforms/EliminateUnusedTorchOpsPass.h"

#include "./PassDetail.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace mlir::tcp {

namespace {

bool isTargetOpInEraselist(Operation *op) {
  if (isa<torch::Torch::TorchDialect>(op->getDialect()))
    return true;
  return false;
}

class RemoveTargetedTorchOps : public RewritePattern {
public:
  RemoveTargetedTorchOps(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!op->use_empty())
      return failure();
    if (!isTargetOpInEraselist(op))
      return failure();
    // These contain dynamic shape annotations, do not DCE
    if (isa<torch::Torch::BindSymbolicShapeOp>(op))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

class EliminateUnusedTorchOpsPass
    : public EliminateUnusedTorchOpsBase<EliminateUnusedTorchOpsPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<RemoveTargetedTorchOps>(context);

    if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createEliminateUnusedTorchOpsPass() {
  return std::make_unique<EliminateUnusedTorchOpsPass>();
}

} // namespace mlir::tcp
