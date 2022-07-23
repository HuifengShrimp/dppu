// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo='io-visibility-json={"inputs":["VIS_PUBLIC"]}' %s --split-input-file  | FileCheck %s

func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xui8>) {
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xui8>} : () -> tensor<2x2x!pphlo.pint> 
    %0 = mhlo.constant dense<1> : tensor<2x2xui8>
    // CHECK: %1 = "pphlo.sqrt"(%arg0) : (tensor<2x2x!pphlo.pfxp>) -> tensor<2x2x!pphlo.pfxp>
    %1 = "mhlo.sqrt"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK: %2 = "pphlo.add"(%arg0, %arg1) : (tensor<2x2x!pphlo.pfxp>, tensor<2x2x!pphlo.pfxp>) -> tensor<2x2x!pphlo.pfxp>
    %2 = "mhlo.add"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xui8>
}
