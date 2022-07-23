// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo='io-visibility-json={"inputs":["VIS_SECRET","VIS_SECRET"]}' %s --split-input-file  | FileCheck %s

// CHECK: func @main(%arg0: tensor<2x2x!pphlo.sfxp>, %arg1: tensor<2x2x!pphlo.sfxp>) -> tensor<2x2x!pphlo.sfxp> {
func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    // CHECK: %0 = "pphlo.sqrt"(%arg0) : (tensor<2x2x!pphlo.sfxp>) -> tensor<2x2x!pphlo.sfxp>
    %0 = "mhlo.sqrt"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK: %1 = "pphlo.add"(%arg0, %arg1) : (tensor<2x2x!pphlo.sfxp>, tensor<2x2x!pphlo.sfxp>) -> tensor<2x2x!pphlo.sfxp>
    %1 = "mhlo.add"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK: return %0 : tensor<2x2x!pphlo.sfxp>
    return %0 : tensor<2x2xf32>
}
