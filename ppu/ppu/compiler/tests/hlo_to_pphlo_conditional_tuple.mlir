// RUN: mlir-pphlo-opt -scfhlo-convert-controlflow -hlo-legalize-to-pphlo %s --split-input-file  | FileCheck %s


func @main(%arg0: tensor<i64>) -> tensor<i64> {
  %c0 = mhlo.constant dense<1> : tensor<i64>
  %c1 = mhlo.constant dense<2> : tensor<i64>
  %0 = "mhlo.compare"(%arg0, %c0) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %1 = "mhlo.tuple"(%arg0) : (tensor<i64>) -> tuple<tensor<i64>>
  //CHECK: %3 = "pphlo.if"(%2, %arg0, %arg0) ( {
  //CHECK: ^bb0(%arg1: tensor<!pphlo.pint>):  // no predecessors
  //CHECK:   %4 = "pphlo.add"(%arg1, %0) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
  //CHECK:   "pphlo.return"(%4) : (tensor<!pphlo.pint>) -> ()
  //CHECK: },  {
  //CHECK: ^bb0(%arg1: tensor<!pphlo.pint>):  // no predecessors
  //CHECK:   %4 = "pphlo.add"(%arg1, %1) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
  //CHECK:   "pphlo.return"(%4) : (tensor<!pphlo.pint>) -> ()
  //CHECK: }) {operand_segment_sizes = dense<1> : vector<3xi32>} : (tensor<!pphlo.pint>, tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
  %2 = "mhlo.if"(%0, %1, %1) ( {
  ^bb0(%arg1: tuple<tensor<i64>>):
    %3 = "mhlo.get_tuple_element"(%arg1) {index = 0 : i32} : (tuple<tensor<i64>>) -> tensor<i64>
    %4 = mhlo.add %c0, %3 : tensor<i64>
    %5 = "mhlo.tuple"(%4) : (tensor<i64>) -> tuple<tensor<i64>>
    "mhlo.return"(%5) : (tuple<tensor<i64>>) -> ()
  },  {
  ^bb0(%arg1: tuple<tensor<i64>>):
    %6 = "mhlo.get_tuple_element"(%arg1) {index = 0 : i32} : (tuple<tensor<i64>>) -> tensor<i64>
    %7 = mhlo.add %c1, %6 : tensor<i64>
    %8 = "mhlo.tuple"(%7) : (tensor<i64>) -> tuple<tensor<i64>>
    "mhlo.return"(%8) : (tuple<tensor<i64>>) -> ()
  }) : (tensor<i1>, tuple<tensor<i64>>, tuple<tensor<i64>>) -> tuple<tensor<i64>>
  %9 = "mhlo.get_tuple_element"(%2) {index = 0 : i32} : (tuple<tensor<i64>>) -> tensor<i64>
  return %9 : tensor<i64>
}