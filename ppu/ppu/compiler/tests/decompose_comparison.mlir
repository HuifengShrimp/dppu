// RUN: mlir-pphlo-opt --decompose-comparison --cse --split-input-file %s | FileCheck %s

func @ne(%arg0: tensor<2x2x!pphlo.pint>, %arg1: tensor<2x2x!pphlo.pint>) -> (tensor<2x2x!pphlo.pint>) {
    // CHECK: %0 = "pphlo.equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    // CHECK: %1 = "pphlo.constant"() {value = dense<1> : tensor<2x2xi32>} : () -> tensor<2x2x!pphlo.pint>
    // CHECK: %2 = "pphlo.subtract"(%1, %0) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %0 = "pphlo.not_equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    return %0 : tensor<2x2x!pphlo.pint>
}

func @le(%arg0: tensor<2x2x!pphlo.pint>, %arg1: tensor<2x2x!pphlo.pint>) -> (tensor<2x2x!pphlo.pint>) {
    // CHECK: %0 = "pphlo.greater"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    // CHECK: %1 = "pphlo.constant"() {value = dense<1> : tensor<2x2xi32>} : () -> tensor<2x2x!pphlo.pint>
    // CHECK: %2 = "pphlo.subtract"(%1, %0) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %0 = "pphlo.less_equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    return %0 : tensor<2x2x!pphlo.pint>
}

func @ge(%arg0: tensor<2x2x!pphlo.pint>, %arg1: tensor<2x2x!pphlo.pint>) -> (tensor<2x2x!pphlo.pint>) {
    // CHECK: %0 = "pphlo.less"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    // CHECK: %1 = "pphlo.constant"() {value = dense<1> : tensor<2x2xi32>} : () -> tensor<2x2x!pphlo.pint>
    // CHECK: %2 = "pphlo.subtract"(%1, %0) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %0 = "pphlo.greater_equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    return %0 : tensor<2x2x!pphlo.pint>
}

func @all(%arg0: tensor<2x2x!pphlo.pint>, %arg1: tensor<2x2x!pphlo.pint>) -> (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) {
    //CHECK: %0 = "pphlo.constant"() {value = dense<1> : tensor<2x2xi32>} : () -> tensor<2x2x!pphlo.pint>
    //CHECK: %1 = "pphlo.equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    //CHECK: %2 = "pphlo.subtract"(%0, %1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    //CHECK: %3 = "pphlo.less"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    //CHECK: %4 = "pphlo.greater"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    //CHECK: %5 = "pphlo.subtract"(%0, %4) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    //CHECK: %6 = "pphlo.subtract"(%0, %3) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %0 = "pphlo.constant"() {value = dense<1> : tensor<2x2xi32>} : () -> tensor<2x2x!pphlo.pint>
    %1 = "pphlo.equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %2 = "pphlo.not_equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %3 = "pphlo.less"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %4 = "pphlo.greater"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %5 = "pphlo.less_equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %6 = "pphlo.greater_equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    return %1, %2, %3, %4, %5, %6, %0 : tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>
  }
  