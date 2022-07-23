// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo --split-input-file %s | FileCheck %s

func @main(%arg0: tensor<16xf32>,%arg1: tensor<1024x1xi1>, %arg2: tensor<1024x1xf32>, %arg3: tensor<1024x1xf32>, %arg4: tensor<3x4xi32>) -> (tensor<1024x16xf32>) {
    // CHECK: %0 = "pphlo.broadcast"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<16x!pphlo.pfxp>) -> tensor<1024x16x!pphlo.pfxp>
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<16xf32>) -> tensor<1024x16xf32>    
    // CHECK: %1 = "pphlo.reshape"(%arg0) : (tensor<16x!pphlo.pfxp>) -> tensor<1x16x!pphlo.pfxp>
    %1 = "mhlo.reshape"(%arg0) : (tensor<16xf32>) -> tensor<1x16xf32>
    // CHECK: %2 = "pphlo.transpose"(%1) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1x16x!pphlo.pfxp>) -> tensor<16x1x!pphlo.pfxp>
    %2 = "mhlo.transpose"(%1) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1x16xf32>) -> tensor<16x1xf32>
    // CHECK: %3 = "pphlo.dot"(%0, %2) : (tensor<1024x16x!pphlo.pfxp>, tensor<16x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %3 = "mhlo.dot"(%0, %2) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1024x16xf32>, tensor<16x1xf32>) -> tensor<1024x1xf32> 
    // CHECK: %4 = "pphlo.concatenate"(%0, %3) {dimension = 1 : i64} : (tensor<1024x16x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x17x!pphlo.pfxp>
    %4 = "mhlo.concatenate"(%0, %3) {dimension = 1 : i64} : (tensor<1024x16xf32>, tensor<1024x1xf32>) -> tensor<1024x17xf32>
    // CHECK: %5 = "pphlo.select"(%arg1, %arg2, %arg3) : (tensor<1024x1x!pphlo.pint>, tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %5 = "mhlo.select"(%arg1, %arg2, %arg3) : (tensor<1024x1xi1>, tensor<1024x1xf32>, tensor<1024x1xf32>) -> tensor<1024x1xf32>
    // CHECK: %6 = "pphlo.slice"(%arg4) {limit_indices = dense<[2, 4]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<3x4x!pphlo.pint>) -> tensor<1x2x!pphlo.pint>
    %6 = "mhlo.slice"(%arg4) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<3x4xi32>) -> tensor<1x2xi32>
    return %0 : tensor<1024x16xf32>
}
