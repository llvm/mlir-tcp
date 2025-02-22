//===-------------------------------------------------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TCP_OPS
#define TCP_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"

include "mlir-tcp/Dialect/IR/TcpBase.td"
include "mlir-tcp/Dialect/IR/TcpEnums.td"
include "mlir-tcp/Dialect/IR/TcpTypes.td"

def Tcp_TanhOp : Tcp_UnaryElementwiseOp<"tanh", [SameOperandsAndResultElementType]> {
  let summary = "Computes tanh of input, elementwise";

  let description = [{
    Computes elementwise hyperbolic tangent of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatTensor:$in
  );

  let results = (outs
    Tcp_FloatTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_ClampOp : Tcp_UnaryElementwiseOp<"clamp", [SameOperandsAndResultElementType]> {
  let summary = "Clamps input tensor to the given min and/or max";

  let description = [{
    Clamps the input tensor, elementwise, to the given min and/or max values.

    The min and max values specified as attributes should correspond to the
    element type of the input tensor.
     * If the input is a float tensor, only `min_float` and/or `max_float` must
       be specified.
     * If the input is an int tensor, only `min_int` and/or `max_int` must be
       specified.
  }];

  let arguments = (ins
    Tcp_FloatOrIntTensor:$in,
    OptionalAttr<F32Attr>:$min_float,
    OptionalAttr<F32Attr>:$max_float,
    OptionalAttr<I64Attr>:$min_int,
    OptionalAttr<I64Attr>:$max_int
  );

  let results = (outs
    Tcp_FloatOrIntTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";

  let hasVerifier = 1;
}

def Tcp_SigmoidOp : Tcp_UnaryElementwiseOp<"sigmoid", [SameOperandsAndResultElementType]> {
  let summary = "Computes sigmoid of input, elementwise";

  let description = [{
    Computes elementwise sigmoid of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatTensor:$in
  );

  let results = (outs
    Tcp_FloatTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_AddOp : Tcp_BinaryElementwiseOp<"add", [SameOperandsAndResultElementType]> {
  let summary = "Computes elementwise addition";

  let description = [{
    Computes the elementwise addition of `in1` and `in2`.
  }];

  let arguments = (ins
    Tcp_Tensor:$in1,
    Tcp_Tensor:$in2
  );

  let results = (outs
    Tcp_Tensor:$out
  );

  let assemblyFormat = "$in1 `,` $in2 attr-dict `:` type($in1) `,` type($in2) `->` type($out)";
}

def Tcp_SubOp : Tcp_BinaryElementwiseOp<"sub", [SameOperandsAndResultElementType]> {
  let summary = "Computes elementwise subtraction";

  let description = [{
    Computes the elementwise subtraction of `in2` from `in1`.
  }];

  let arguments = (ins
    Tcp_Tensor:$in1,
    Tcp_Tensor:$in2
  );

  let results = (outs
    Tcp_Tensor:$out
  );

  let assemblyFormat = "$in1 `,` $in2 attr-dict `:` type($in1) `,` type($in2) `->` type($out)";
}

def Tcp_MulOp : Tcp_BinaryElementwiseOp<"mul", [SameOperandsAndResultElementType]> {
  let summary = "Computes elementwise multiplication";

  let description = [{
    Computes the elementwise multiplication of `in1` and `in2`.
  }];

  let arguments = (ins
    Tcp_FloatOrIntTensor:$in1,
    Tcp_FloatOrIntTensor:$in2
  );

  let results = (outs
    Tcp_FloatOrIntTensor:$out
  );

  let assemblyFormat = "$in1 `,` $in2 attr-dict `:` type($in1) `,` type($in2) `->` type($out)";
}

def Tcp_DivFOp : Tcp_BinaryElementwiseOp<"divf", [SameOperandsAndResultElementType]> {
  let summary = "Computes elementwise floating point division";

  let description = [{
    Computes the elementwise floating point division of `in1` and `in2`.
  }];

  let arguments = (ins
    Tcp_FloatTensor:$in1,
    Tcp_FloatTensor:$in2
  );

  let results = (outs
    Tcp_FloatTensor:$out
  );

  let assemblyFormat = "$in1 `,` $in2 attr-dict `:` type($in1) `,` type($in2) `->` type($out)";
}

def Tcp_DivSIOp : Tcp_BinaryElementwiseOp<"divsi", [SameOperandsAndResultElementType]> {
  let summary = "Computes elementwise signed integer division";

  let description = [{
    Computes the signed integer division of `in1` and `in2`.
  }];

  let arguments = (ins
    Tcp_IntTensor:$in1,
    Tcp_IntTensor:$in2,
    Tcp_RoundingModeAttr:$rounding_mode
  );

  let results = (outs
    Tcp_IntTensor:$out
  );

  let assemblyFormat = "$in1 `,` $in2 attr-dict `:` type($in1) `,` type($in2) `->` type($out)";
}

def Tcp_DivUIOp : Tcp_BinaryElementwiseOp<"divui", [SameOperandsAndResultElementType]> {
  let summary = "Computes elementwise unsigned integer division";

  let description = [{
    Computes the unsigned integer division of `in1` and `in2`.
  }];

  let arguments = (ins
    Tcp_IntTensor:$in1,
    Tcp_IntTensor:$in2,
    Tcp_RoundingModeAttr:$rounding_mode
  );

  let results = (outs
    Tcp_IntTensor:$out
  );

  let assemblyFormat = "$in1 `,` $in2 attr-dict `:` type($in1) `,` type($in2) `->` type($out)";
}

def Tcp_ConstOp : Tcp_Op<"const", [ConstantLike, Pure]> {
  let summary = "Constant op";

  let description = [{
    An op that produces constant tensor (of floating point or signless integer type).
  }];

  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    Tcp_Tensor:$out
  );

  let assemblyFormat = "attr-dict `:` type($out)";

  let hasFolder = 1;
  let hasVerifier = 1;
}

def Tcp_BroadcastOp : Tcp_Op<"broadcast", [
          AllRanksMatch<["in", "out"]>,
          AllElementTypesMatch<["in", "out"]>]> {
  let summary = "Broadcasts input to a given shape";

  let description = [{
    Broadcasts `in` to a given shape by replicating dimensions with bound `1`.

    All of the dimensions listed in `axes` must have a static size of `1` in
    `in`.  The dimesion `axis[i]` will be replicated `new_dim_sizes[i]` times in
    the output.

    This op does not change the rank of the input.

    The attribute `$axes` must be sorted and not have any duplicates. The size
    of `$axes` must be equal to the size of the variadic argument `$new_dim_sizes`.
  }];

  let arguments = (ins
    Tcp_Tensor:$in,
    Variadic<Index>:$new_dim_sizes,
    I64ArrayAttr:$axes
  );

  let results = (outs
    Tcp_Tensor:$out
  );

  let assemblyFormat = "$in `,` $new_dim_sizes attr-dict `:` type($in) `,` type($new_dim_sizes) `->` type($out)";

  let hasVerifier = 1;
}

def Tcp_YieldOp : Tcp_Op<"yield", [Terminator, Pure]> {
  let summary = "Yields values from regions in Tcp";

  let description = [{
    Terminates and yields values from regions in Tcp.

    The values that are yielded should correspond to the parent operation's
    results.

    The results can be `AnyType` since Tcp grouping ops can be used with ops
    from any dialect (not just Tcp).
  }];

  let arguments = (ins
    Variadic<AnyType>:$ins
  );

  let builders = [
    OpBuilder<(ins), [{ }]>,
  ];

  let assemblyFormat = "$ins attr-dict `:` type($ins)";
}

def Tcp_GroupOp : Tcp_Op<"group", [
          AffineScope,
          SingleBlockImplicitTerminator<"tcp::YieldOp">]> {
  let summary = "Groups ops into a region that is not isolated from above";

  let description = [{
    Provides a way to group ops into a region that is not isolated from above.

    Since this is not isolated from above, ops inside this region can refer to
    values in the outer scopes. So, this op does not have any input parameters.

    This op must not include control flow ops in its region. However, it can
    be present within regions that are inside control flow ops.

    The enclosed region will terminate with a `tcp.yield` op. This ops results
    should correspond to the values that are yielded by the terminating op.

    The results can be `AnyType` since Tcp grouping ops can be used with ops
    from any dialect (not just Tcp).
  }];

  let results = (outs
    Variadic<AnyType>:$outs
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let assemblyFormat = "attr-dict-with-keyword $body `:` type($outs)";

  let hasVerifier = 1;
}

def Tcp_IsolatedGroupOp : Tcp_Op<"isolated_group", [
          AffineScope,
          IsolatedFromAbove,
          SingleBlockImplicitTerminator<"tcp::YieldOp">]> {
  let summary = "Groups ops into a region that is isolated from above";

  let description = [{
    Provides a way to group ops into a region that is isolated from above.

    Since the enclosed region is isolated from above, any use of values from
    outer scopes requires passing that value as an input parameter to this op
    and using that parameter instead.

    This op must not include control flow ops in its region. However, it can
    be present within regions that are inside control flow ops.

    The enclosed region will terminate with a `tcp.yield` op. This ops results
    should correspond to the values that are yielded by the terminating op.

    The inputs and results can be `AnyType` since Tcp grouping ops can be used
    with ops from any dialect (not just Tcp).
  }];

  let arguments = (ins
    Variadic<AnyType>:$ins
  );

  let results = (outs
    Variadic<AnyType>:$outs
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let assemblyFormat = "$ins attr-dict-with-keyword $body `:` type($ins) `->` type($outs)";

  let hasVerifier = 1;
}

def Tcp_CustomOp : Tcp_Op<"custom_op", []> {
    let summary = "Custom opaque operation in Tcp dialect";

    let description = [{
        This is useful to represent and lower custom/opaque operations that have
        tensor inputs and outputs, but
          (1) lack a clear operational semantics or
          (2) have an operational semantic that isn't defined by Tcp (e.g user
              defined custom operations) or
          (3) need to "flow through" the middle-end because they target existing
              pre-written kernels available in the backend.
    }];

    let arguments = (ins
        Variadic<AnyType>:$inputs,
        StrAttr:$op_name
    );

    let results = (outs
        Variadic<AnyType>:$outputs
    );

    let assemblyFormat = "`(`$op_name`)` $inputs attr-dict `:` type($inputs) `->` type($outputs)";
}

def Tcp_SqrtOp : Tcp_UnaryElementwiseOp<"sqrt"> {
  let summary = "Computes square root of input, elementwise";

  let description = [{
    Computes elementwise square root of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatTensor:$in
  );

  let results = (outs
    Tcp_FloatTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_CeilOp : Tcp_UnaryElementwiseOp<"ceil", [SameOperandsAndResultElementType]> {
  let summary = "Computes ceil of input, elementwise";

  let description = [{
    Computes elementwise ceil of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatTensor:$in
  );

  let results = (outs
    Tcp_FloatTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_FloorOp : Tcp_UnaryElementwiseOp<"floor", [SameOperandsAndResultElementType]> {
  let summary = "Computes floor of input, elementwise";

  let description = [{
    Computes elementwise floor of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatTensor:$in
  );

  let results = (outs
    Tcp_FloatTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_RoundOp : Tcp_UnaryElementwiseOp<"round", [SameOperandsAndResultElementType]> {
  let summary = "Computes elementwise rounding of input, with ties rounded away from zero";

  let description = [{
    Computes elementwise rounding of the input float tensor to the nearest integer value
    in float, with halfway values rounded away from zero.
  }];

  let arguments = (ins
    Tcp_FloatTensor:$in
  );

  let results = (outs
    Tcp_FloatTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_RoundEvenOp : Tcp_UnaryElementwiseOp<"round_even", [SameOperandsAndResultElementType]> {
  let summary = "Computes elementwise rounding of input, with ties rounded to nearest even";

  let description = [{
    Computes elementwise rounding of the input float tensor to the nearest integer value
    in float, with halfway values rounded to the nearest even integer.
  }];

  let arguments = (ins
    Tcp_FloatTensor:$in
  );

  let results = (outs
    Tcp_FloatTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_CosOp : Tcp_UnaryElementwiseOp<"cos", [SameOperandsAndResultElementType]> {
  let summary = "Computes cosine of input, elementwise";

  let description = [{
    Computes elementwise cosine of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatTensor:$in
  );

  let results = (outs
    Tcp_FloatTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_SinOp : Tcp_UnaryElementwiseOp<"sin", [SameOperandsAndResultElementType]> {
  let summary = "Computes sine of input, elementwise";

  let description = [{
    Computes elementwise sine of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatTensor:$in
  );

  let results = (outs
    Tcp_FloatTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_AbsOp : Tcp_UnaryElementwiseOp<"abs", [SameOperandsAndResultElementType]> {
  let summary = "Computes absolute of input, elementwise";

  let description = [{
    Computes elementwise absolute value of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatOrIntTensor:$in
  );

  let results = (outs
    Tcp_FloatOrIntTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_LogOp : Tcp_UnaryElementwiseOp<"log", [SameOperandsAndResultElementType]> {
  let summary = "Computes natural logarithm of input, elementwise";

  let description = [{
    Computes elementwise natural logarithm value of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatTensor:$in
  );

  let results = (outs
    Tcp_FloatTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_NegOp : Tcp_UnaryElementwiseOp<"neg", [SameOperandsAndResultElementType]> {
  let summary = "Computes the negation of input, elementwise";

  let description = [{
    Computes elementwise negation value of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatTensor:$in
  );

  let results = (outs
    Tcp_FloatTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_AtanOp : Tcp_UnaryElementwiseOp<"atan", [SameOperandsAndResultElementType]> {
  let summary = "Computes the arcus tangent value of input, elementwise";

  let description = [{
    Computes elementwise arcus tangent value of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatTensor:$in
  );

  let results = (outs
    Tcp_FloatTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_Atan2Op : Tcp_BinaryElementwiseOp<"atan2", [SameOperandsAndResultElementType]> {
  let summary = "2-argument arcus tangent of the given values";

  let description = [{
    It is a generalization of the 1-argument arcus tangent which returns the angle 
    on the basis of the ratio in1 / in2, elementwise.
  }];

  let arguments = (ins
    Tcp_FloatTensor:$in1,
    Tcp_FloatTensor:$in2
  );

  let results = (outs
    Tcp_FloatTensor:$out
  );

  let assemblyFormat = "$in1 `,` $in2 attr-dict `:` type($in1) `,` type($in2) `->` type($out)";
}

def Tcp_CastOp:  Tcp_Op<"cast", [Pure, Elementwise, SameOperandsAndResultShape]> {

  let summary = "TCP Cast operation";

  // TODO: Make the cast op handle more corner cases
  // e.g. conversion overflows the destination type
  // e.g. convert negative number to unsigned destination type
  let description = [{
    Performs elementwise type conversion.

    Supported MLIR core types:
      INT: ui8,  si8,  i8, 
           ui16, si16, i16,
           ui32, si32, i32,
           ui64, si64, i64, 
           i1 (Boolean)
      FP: f16, bf16, f32, f64

    Supported Conversion Mode:
      FP -> FP
      INT -> FP
      FP -> INT
      INT -> INT
  }];

  let arguments = (ins
    Tcp_Tensor:$in,
    OptionalAttr<Tcp_SignednessAttr>:$in_int_signedness,
    OptionalAttr<Tcp_SignednessAttr>:$out_int_signedness
  );

  let results = (outs
    Tcp_Tensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";

  let hasVerifier = 1;
}

def Tcp_GatherOp : Tcp_Op<"gather", [Pure, AllElementTypesMatch<["input", "out"]>]> {

  let summary = "Gather elements from input based on indices";

  let description = [{
    Gathers elements from a given tensor based on indices that index along a given dim.

    More details regarding this op: docs/gather.md
  }];

  let arguments = (ins
    Tcp_Tensor:$input,
    Tcp_IntTensor:$indices,
    IndexAttr:$dim
  );

  let results = (outs
    Tcp_Tensor:$out
  );

  let assemblyFormat = "$input `,` $indices attr-dict `:` type($input) `,` type($indices) `->` type($out)";

  let hasVerifier = 1;
}

def Tcp_GatherNDOp : Tcp_Op<"gather_nd", [Pure, AllElementTypesMatch<["input", "out"]>]> {

  let summary = "Gather elements from input based on indices over multiple dimensions";

  let description = [{
    Gathers elements from a given tensor based on indices that index along multiple dimensions.

    More details regarding this op: docs/gather.md
  }];

  let arguments = (ins
    Tcp_Tensor:$input,
    Tcp_IntTensor:$indices
  );

  let results = (outs
    Tcp_Tensor:$out
  );

  let assemblyFormat = "$input `,` $indices attr-dict `:` type($input) `,` type($indices) `->` type($out)";

  let hasVerifier = 1;
}

def Tcp_SliceOp : Tcp_Op<"slice", [Pure, AllElementTypesMatch<["in", "out"]>, SameVariadicOperandSize]> {

  let summary = "Extracts a slice of the input tensor";

  let description = [{
    Extracts a slice of the input tensor based on the given starts, sizes, and strides.
  }];

  let arguments = (ins
    Tcp_Tensor:$in,
    Variadic<Index>:$starts,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides
  );

  let results = (outs
    Tcp_Tensor:$out
  );

  let assemblyFormat = "$in `starts` `(` $starts `)` `sizes` `(` $sizes `)` `strides` `(` $strides `)` attr-dict `:` type($in) `->` type($out)";
}

//===----------------------------------------------------------------------===//
// Symbolic shape modeling ops for TorchDynamo frontend.
//===----------------------------------------------------------------------===//

def Tcp_SymbolicIntOp : Tcp_Op<"symbolic_int", [Pure]> {

  let summary = "Symbolic int representing a dynamic dimension";

  let description = [{
    The `tcp.symbolic_int` operation captures a dynamic dimension on the
    global function arguments. It associates the shape symbols (i.e. "s0",
    "s1") with the global SSA values (i.e. `%0`, `%1`) that is then
    referenced to bind shapes on op results.

    Additionally, the operation annotates `min_val` and `max_val` attributes
    denoting the range constraints for the dynamic dimension. This may be
    useful for modeling runtime shape guards, or compile-time optimizations
    based on the shape bounds (min, opt, max) on results of ops / regions.

    Example:
    ```
    %0 = tcp.symbolic_int "s0" {min_val = 5, max_val = 10} : i64
    %1 = tcp.symbolic_int "s1" {min_val = 2, max_val = 20} : i64
    ```
  }];

  let arguments = (ins
    StrAttr:$symbol_name,
    I64Attr:$min_val,
    I64Attr:$max_val
  );
  let results = (outs
    AnySignlessInteger:$result
  );
  let assemblyFormat = [{
    $symbol_name ` ` `{` `min_val` `=` $min_val `,` `max_val` `=` $max_val `}` attr-dict `:` type($result)
  }];
}

def Tcp_BindSymbolicShapeOp : Tcp_Op<"bind_symbolic_shape", []> {
  let summary = "Binds shape expressions to tensors using an affine map indexed by shape symbols";
  let description = [{
    The `tcp.bind_symbolic_shape` operation binds shape expressions
    useful to compute the dynamic dimensions of a tensor. It takes a
    variadic of SSA symbols that map 1:1 to the local symbols declared
    in the affine map. The affine map contains a list of affine shape
    expressions for each dim where the terminals are from the declared
    symbols.

    Example:
    ```
    tcp.bind_symbolic_shape %arg0, [%0, %1], affine_map<()[s0, s1] -> (s0, s1, 3)> : tensor<?x?x3xf32>
    tcp.bind_symbolic_shape %out0, [%0, %1, %2], affine_map<()[s0, s1, s2] -> (s0, s1 * 2 + s2, 3)> : tensor<?x?x3xf32>
    ```
  }];
  let arguments = (ins
    Tcp_Tensor:$operand,
    Variadic<AnySignlessInteger>:$shape_symbols,
    Builtin_AffineMapAttr:$shape_expressions
  );
  let results = (outs);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // TCP_OPS
