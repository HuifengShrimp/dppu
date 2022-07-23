module @xla_computation_selu.26  {
  func @main(%arg0: tensor<100x!pphlo.pfxp>) -> tensor<100x!pphlo.pfxp> {
    %0 = "pphlo.constant"() {value = dense<1.050000e+00> : tensor<100xf32>} : () -> tensor<100x!pphlo.pfxp>
    %1 = "pphlo.constant"() {value = dense<-1.670000e+00> : tensor<100xf32>} : () -> tensor<100x!pphlo.pfxp>
    %2 = "pphlo.constant"() {value = dense<1.670000e+00> : tensor<100xf32>} : () -> tensor<100x!pphlo.pfxp>
    %3 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100xf32>} : () -> tensor<100x!pphlo.pfxp>
    %4 = "pphlo.greater"(%arg0, %3) : (tensor<100x!pphlo.pfxp>, tensor<100x!pphlo.pfxp>) -> tensor<100x!pphlo.pint>
    %5 = "pphlo.exponential"(%arg0) : (tensor<100x!pphlo.pfxp>) -> tensor<100x!pphlo.pfxp>
    %6 = "pphlo.multiply"(%5, %2) : (tensor<100x!pphlo.pfxp>, tensor<100x!pphlo.pfxp>) -> tensor<100x!pphlo.pfxp>
    %7 = "pphlo.add"(%6, %1) : (tensor<100x!pphlo.pfxp>, tensor<100x!pphlo.pfxp>) -> tensor<100x!pphlo.pfxp>
    %8 = "pphlo.subtract"(%arg0, %7) : (tensor<100x!pphlo.pfxp>, tensor<100x!pphlo.pfxp>) -> tensor<100x!pphlo.pfxp>
    %9 = "pphlo.multiply"(%4, %8) : (tensor<100x!pphlo.pint>, tensor<100x!pphlo.pfxp>) -> tensor<100x!pphlo.pfxp>
    %10 = "pphlo.add"(%9, %7) : (tensor<100x!pphlo.pfxp>, tensor<100x!pphlo.pfxp>) -> tensor<100x!pphlo.pfxp>
    %11 = "pphlo.multiply"(%10, %0) : (tensor<100x!pphlo.pfxp>, tensor<100x!pphlo.pfxp>) -> tensor<100x!pphlo.pfxp>
    return %11 : tensor<100x!pphlo.pfxp>
  }
}
