// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo --split-input-file %s | FileCheck %s

func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    // CHECK: %0 = "pphlo.subtract"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %0 = "mhlo.subtract"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    // CHECK: %1 = "pphlo.maximum"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %1 = "mhlo.maximum"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    // CHECK: %2 = "pphlo.minimum"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %2 = "mhlo.minimum"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    // CHECK: %3 = "pphlo.divide"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %3 = "mhlo.divide"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    // CHECK: %4 = "pphlo.add"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %4 = "mhlo.add"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    // CHECK: %5 = "pphlo.multiply"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %5 = "mhlo.multiply"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
     // CHECK: %6 = "pphlo.power"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %6 = "mhlo.power"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    // CHECK: %7 = "pphlo.and"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %7 = "mhlo.and"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
}

