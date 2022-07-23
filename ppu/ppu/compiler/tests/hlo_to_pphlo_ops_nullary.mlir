// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo --split-input-file %s | FileCheck %s

func @main() -> (tensor<2x2xi1>) {
    // CHECK: "pphlo.constant"() {value = dense<true> : tensor<2x2xi1>} : () -> tensor<2x2x!pphlo.pint> 
    %0 = mhlo.constant dense<true> : tensor<2x2xi1>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xi8>} : () -> tensor<2x2x!pphlo.pint> 
    %1 = mhlo.constant dense<1> : tensor<2x2xi8>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xi16>} : () -> tensor<2x2x!pphlo.pint> 
    %2 = mhlo.constant dense<1> : tensor<2x2xi16>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xi32>} : () -> tensor<2x2x!pphlo.pint> 
    %3 = mhlo.constant dense<1> : tensor<2x2xi32>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xi64>} : () -> tensor<2x2x!pphlo.pint> 
    %4 = mhlo.constant dense<1> : tensor<2x2xi64>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xui8>} : () -> tensor<2x2x!pphlo.pint> 
    %5 = mhlo.constant dense<1> : tensor<2x2xui8>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xui16>} : () -> tensor<2x2x!pphlo.pint> 
    %6 = mhlo.constant dense<1> : tensor<2x2xui16>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xui32>} : () -> tensor<2x2x!pphlo.pint> 
    %7 = mhlo.constant dense<1> : tensor<2x2xui32>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xui64>} : () -> tensor<2x2x!pphlo.pint> 
    %8 = mhlo.constant dense<1> : tensor<2x2xui64>
    // CHECK: "pphlo.constant"() {value = dense<1.000000e+00> : tensor<2x2xf32>} : () -> tensor<2x2x!pphlo.pfxp> 
    %9 = mhlo.constant dense<1.0> : tensor<2x2xf32>
    // CHECK: "pphlo.constant"() {value = dense<1.000000e+00> : tensor<2x2xf64>} : () -> tensor<2x2x!pphlo.pfxp> 
    %10 = mhlo.constant dense<1.0> : tensor<2x2xf64>
    return %0 : tensor<2x2xi1>
}


