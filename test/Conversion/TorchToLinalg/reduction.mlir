// RUN: torch-mlir-opt -convert-torch-to-linalg %s | FileCheck %s

// CHECK-LABEL: func.func @torch.aten.any.dim$negative_dim
// CHECK:         %[[ARG0:.*]] = torch_c.to_builtin_tensor
// CHECK:         %[[FALSE:.*]] = arith.constant false
// CHECK:         %[[INIT:.*]] = tensor.empty
// CHECK:         %[[FILL:.*]] = linalg.fill ins(%[[FALSE]] : i1) outs(%[[INIT]]
// CHECK:         %[[REDUCE:.*]] = linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK:           arith.ori
// CHECK:           linalg.yield
// CHECK:         tensor.cast %[[REDUCE]]
// CHECK-NOT:     torch.aten.any.dim
func.func @torch.aten.any.dim$negative_dim(%arg0: !torch.vtensor<[1,2,3,3],i1>) -> !torch.vtensor<[1,2,3,1],i1> {
  %int-1 = torch.constant.int -1
  %true = torch.constant.bool true
  %0 = torch.aten.any.dim %arg0, %int-1, %true : !torch.vtensor<[1,2,3,3],i1>, !torch.int, !torch.bool -> !torch.vtensor<[1,2,3,1],i1>
  return %0 : !torch.vtensor<[1,2,3,1],i1>
}
