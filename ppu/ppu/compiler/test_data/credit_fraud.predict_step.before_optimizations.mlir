module @cluster_0__XlaCompiledKernel_true__XlaHasReferenceVars_false__XlaNumConstantArgs_6__XlaNumResourceArgs_1_.73  {
  func @main(%arg0: tensor<100x!pphlo.pint>, %arg1: tensor<100x1x!pphlo.pfxp>, %arg2: tensor<!pphlo.pfxp>) -> (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) {
    %0 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %1 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x1xf32>} : () -> tensor<100x1x!pphlo.pfxp>
    %2 = "pphlo.constant"() {value = dense<1.000000e+02> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %3 = "pphlo.less"(%arg1, %1) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pint>
    %4 = "pphlo.constant"() {value = dense<1> : tensor<100x1xi32>} : () -> tensor<100x1x!pphlo.pint>
    %5 = "pphlo.subtract"(%4, %3) : (tensor<100x1x!pphlo.pint>, tensor<100x1x!pphlo.pint>) -> tensor<100x1x!pphlo.pint>
    %6 = "pphlo.negate"(%arg1) : (tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %7 = "pphlo.subtract"(%6, %arg1) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %8 = "pphlo.multiply"(%5, %7) : (tensor<100x1x!pphlo.pint>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %9 = "pphlo.add"(%8, %arg1) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %10 = "pphlo.exponential"(%9) : (tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %11 = "pphlo.log_plus_one"(%10) : (tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %12 = "pphlo.subtract"(%arg1, %1) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %13 = "pphlo.multiply"(%5, %12) : (tensor<100x1x!pphlo.pint>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %14 = "pphlo.add"(%13, %1) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %15 = "pphlo.convert"(%arg0) : (tensor<100x!pphlo.pint>) -> tensor<100x!pphlo.pfxp>
    %16 = "pphlo.reshape"(%15) : (tensor<100x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %17 = "pphlo.multiply"(%16, %arg1) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %18 = "pphlo.subtract"(%14, %17) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %19 = "pphlo.add"(%11, %18) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %20 = "pphlo.reduce"(%19, %0) ( {
    ^bb0(%arg3: tensor<!pphlo.pfxp>, %arg4: tensor<!pphlo.pfxp>):  // no predecessors
      %22 = "pphlo.add"(%arg3, %arg4) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%22) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<100x1x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %21 = "pphlo.add"(%arg2, %20) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    return %2, %21 : tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>
  }
}
