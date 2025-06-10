// RUN: tcp-opt %s -eliminate-unused-torch-ops | FileCheck %s

// CHECK-LABEL:  func.func @test_eliminate_unused_torch_ops(
// CHECK-SAME:    %[[ARG0:.*]]: !torch.vtensor<[?,3],f32>, %[[ARG1:.*]]: !torch.vtensor<[?,3],f32>) -> !torch.vtensor<[?,3],f32> {
// CHECK:         %[[CONST2:.*]] = torch.constant.int 2
// CHECK:         %[[S0:.*]] = torch.symbolic_int "s35" {min_val = 0, max_val = 9223372036854775807} : !torch.int
// CHECK:         torch.bind_symbolic_shape %[[ARG0]], [%[[S0]]], affine_map<()[s0] -> (s0, 3)> : !torch.vtensor<[?,3],f32>
// CHECK:         torch.bind_symbolic_shape %[[ARG1]], [%[[S0]]], affine_map<()[s0] -> (s0, 3)> : !torch.vtensor<[?,3],f32>
// CHECK:         %[[SUB:.*]] = torch.aten.sub.Tensor %[[ARG0]], %[[ARG1]], %[[CONST2]] : !torch.vtensor<[?,3],f32>, !torch.vtensor<[?,3],f32>, !torch.int -> !torch.vtensor<[?,3],f32>
// CHECK:         torch.bind_symbolic_shape %[[SUB]], [%[[S0]]], affine_map<()[s0] -> (s0, 3)> : !torch.vtensor<[?,3],f32>
// CHECK:         return %[[SUB]] : !torch.vtensor<[?,3],f32>
func.func @test_eliminate_unused_torch_ops(%arg0: !torch.vtensor<[?,3],f32>, %arg1: !torch.vtensor<[?,3],f32>) -> !torch.vtensor<[?,3],f32> {
%int2 = torch.constant.int 2
%int0 = torch.constant.int 0
%0 = torch.symbolic_int "s35" {min_val = 0, max_val = 9223372036854775807} : !torch.int
torch.bind_symbolic_shape %arg0, [%0], affine_map<()[s0] -> (s0, 3)> : !torch.vtensor<[?,3],f32>
torch.bind_symbolic_shape %arg1, [%0], affine_map<()[s0] -> (s0, 3)> : !torch.vtensor<[?,3],f32>
%1 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,3],f32>, !torch.int -> !torch.int
%2 = torch.aten.size.int %arg1, %int0 : !torch.vtensor<[?,3],f32>, !torch.int -> !torch.int
%3 = torch.aten.sub.Tensor %arg0, %arg1, %int2 : !torch.vtensor<[?,3],f32>, !torch.vtensor<[?,3],f32>, !torch.int -> !torch.vtensor<[?,3],f32>
torch.bind_symbolic_shape %3, [%0], affine_map<()[s0] -> (s0, 3)> : !torch.vtensor<[?,3],f32>
%4 = torch.aten.eq.int %1, %2 : !torch.int, !torch.int -> !torch.bool
%5 = torch.aten.Int.bool %4 : !torch.bool -> !torch.int
%6 = torch.aten.Bool.int %5 : !torch.int -> !torch.bool
torch.runtime.assert %6, "Runtime assertion failed for expression Eq(s35, s58) on node 'eq_2'"
return %3 : !torch.vtensor<[?,3],f32>
}
