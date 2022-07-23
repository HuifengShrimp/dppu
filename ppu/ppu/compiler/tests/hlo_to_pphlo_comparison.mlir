// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo --split-input-file %s | FileCheck %s

func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi1>) {
    // CHECK: "pphlo.equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "EQ"} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    // CHECK: "pphlo.not_equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %1 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "NE"} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    // CHECK: "pphlo.less"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %2 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "LT"} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    // CHECK: "pphlo.greater_equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %3 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "GE"} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    // CHECK: "pphlo.less_equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %4 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "LE"} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
     // CHECK: "pphlo.greater"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %5 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    return %0 : tensor<2x2xi1>
}
