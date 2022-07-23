// RUN: mlir-pphlo-opt --decompose-sqrt --split-input-file %s | FileCheck %s

func @sqrt(%arg0: tensor<2x2x!pphlo.pfxp>) -> (tensor<2x2x!pphlo.pfxp>) {
    //CHECK: %0 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<2x2xf32>} : () -> tensor<2x2x!pphlo.pfxp>
    //CHECK: %1 = "pphlo.power"(%arg0, %0) : (tensor<2x2x!pphlo.pfxp>, tensor<2x2x!pphlo.pfxp>) -> tensor<2x2x!pphlo.pfxp>
    %0 = "pphlo.sqrt"(%arg0) : (tensor<2x2x!pphlo.pfxp>) -> tensor<2x2x!pphlo.pfxp>
    return %0 : tensor<2x2x!pphlo.pfxp>
}

