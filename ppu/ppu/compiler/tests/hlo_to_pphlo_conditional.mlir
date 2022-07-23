// RUN: mlir-pphlo-opt -scfhlo-convert-controlflow -hlo-legalize-to-pphlo %s --split-input-file  | FileCheck %s

func @main(%arg0: tensor<f32>) -> tensor<f32> {
  //CHECK: %2 = "pphlo.if"(%1, %arg0, %arg0) ( {
  //CHECK: ^bb0(%arg1: tensor<!pphlo.pfxp>):  // no predecessors
  //CHECK:   %3 = "pphlo.log"(%arg1) : (tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
  //CHECK:   "pphlo.return"(%3) : (tensor<!pphlo.pfxp>) -> ()
  //CHECK: },  {
  //CHECK: ^bb0(%arg1: tensor<!pphlo.pfxp>):  // no predecessors
  //CHECK:   %3 = "pphlo.exponential"(%arg1) : (tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
  //CHECK:   "pphlo.return"(%3) : (tensor<!pphlo.pfxp>) -> ()
  //CHECK: }) {operand_segment_sizes = dense<1> : vector<3xi32>} : (tensor<!pphlo.pint>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
  %cst = mhlo.constant dense<1.000000e+01> : tensor<f32>
  %0 = "mhlo.compare"(%arg0, %cst) {comparison_direction = "LT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %1 = "mhlo.if"(%0, %arg0, %arg0) ( {
  ^bb0(%arg1: tensor<f32>):
    %2 = "mhlo.log"(%arg1) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  },  {
  ^bb0(%arg1: tensor<f32>):
    %2 = "mhlo.exponential"(%arg1) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}