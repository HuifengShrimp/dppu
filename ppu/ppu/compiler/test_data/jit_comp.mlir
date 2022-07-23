module @a_inference_comp_27__.20  {
  func @main(%arg0: tensor<2x2x!pphlo.pint>, %arg1: tensor<2x2x!pphlo.pint>) -> (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) {
    %0 = "pphlo.constant"() {value = dense<true> : tensor<2x2xi1>} : () -> tensor<2x2x!pphlo.pint>
    %1 = "pphlo.equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %2 = "pphlo.constant"() {value = dense<1> : tensor<2x2xi32>} : () -> tensor<2x2x!pphlo.pint>
    %3 = "pphlo.subtract"(%2, %1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %4 = "pphlo.less"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %5 = "pphlo.greater"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %6 = "pphlo.subtract"(%2, %5) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %7 = "pphlo.subtract"(%2, %4) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    return %1, %3, %4, %5, %6, %7, %0 : tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>
  }
}
