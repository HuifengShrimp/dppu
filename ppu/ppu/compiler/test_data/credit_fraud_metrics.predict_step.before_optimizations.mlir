module @cluster_0__XlaCompiledKernel_true__XlaHasReferenceVars_false__XlaNumConstantArgs_21__XlaNumResourceArgs_5_.200  {
  func @main(%arg0: tensor<100x!pphlo.pint>, %arg1: tensor<100x1x!pphlo.pfxp>, %arg2: tensor<!pphlo.pfxp>, %arg3: tensor<1x!pphlo.pfxp>, %arg4: tensor<1x!pphlo.pfxp>, %arg5: tensor<1x!pphlo.pfxp>, %arg6: tensor<1x!pphlo.pfxp>) -> (tensor<!pphlo.pint>, tensor<100x1x!pphlo.pfxp>, tensor<!pphlo.pint>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) {
    %0 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %1 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x1xf32>} : () -> tensor<100x1x!pphlo.pfxp>
    %2 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<100x1xf32>} : () -> tensor<100x1x!pphlo.pfxp>
    %3 = "pphlo.constant"() {value = dense<1.000000e+02> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %4 = "pphlo.constant"() {value = dense<true> : tensor<i1>} : () -> tensor<!pphlo.pint>
    %5 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<100x1xf32>} : () -> tensor<100x1x!pphlo.pfxp>
    %6 = "pphlo.logistic"(%arg1) : (tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %7 = "pphlo.less"(%6, %1) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pint>
    %8 = "pphlo.constant"() {value = dense<1> : tensor<100x1xi32>} : () -> tensor<100x1x!pphlo.pint>
    %9 = "pphlo.subtract"(%8, %7) : (tensor<100x1x!pphlo.pint>, tensor<100x1x!pphlo.pint>) -> tensor<100x1x!pphlo.pint>
    %10 = "pphlo.reduce"(%9, %4) ( {
    ^bb0(%arg7: tensor<!pphlo.pint>, %arg8: tensor<!pphlo.pint>):  // no predecessors
      %52 = "pphlo.and"(%arg7, %arg8) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
      "pphlo.return"(%52) : (tensor<!pphlo.pint>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<100x1x!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %11 = "pphlo.greater"(%6, %5) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pint>
    %12 = "pphlo.subtract"(%8, %11) : (tensor<100x1x!pphlo.pint>, tensor<100x1x!pphlo.pint>) -> tensor<100x1x!pphlo.pint>
    %13 = "pphlo.reduce"(%12, %4) ( {
    ^bb0(%arg7: tensor<!pphlo.pint>, %arg8: tensor<!pphlo.pint>):  // no predecessors
      %52 = "pphlo.and"(%arg7, %arg8) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
      "pphlo.return"(%52) : (tensor<!pphlo.pint>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<100x1x!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %14 = "pphlo.less"(%arg1, %1) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pint>
    %15 = "pphlo.subtract"(%8, %14) : (tensor<100x1x!pphlo.pint>, tensor<100x1x!pphlo.pint>) -> tensor<100x1x!pphlo.pint>
    %16 = "pphlo.negate"(%arg1) : (tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %17 = "pphlo.subtract"(%16, %arg1) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %18 = "pphlo.multiply"(%15, %17) : (tensor<100x1x!pphlo.pint>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %19 = "pphlo.add"(%18, %arg1) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %20 = "pphlo.exponential"(%19) : (tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %21 = "pphlo.log_plus_one"(%20) : (tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %22 = "pphlo.subtract"(%arg1, %1) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %23 = "pphlo.multiply"(%15, %22) : (tensor<100x1x!pphlo.pint>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %24 = "pphlo.add"(%23, %1) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %25 = "pphlo.convert"(%arg0) : (tensor<100x!pphlo.pint>) -> tensor<100x!pphlo.pfxp>
    %26 = "pphlo.reshape"(%25) : (tensor<100x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %27 = "pphlo.multiply"(%26, %arg1) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %28 = "pphlo.subtract"(%24, %27) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %29 = "pphlo.add"(%21, %28) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %30 = "pphlo.reduce"(%29, %0) ( {
    ^bb0(%arg7: tensor<!pphlo.pfxp>, %arg8: tensor<!pphlo.pfxp>):  // no predecessors
      %52 = "pphlo.add"(%arg7, %arg8) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%52) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<100x1x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %31 = "pphlo.add"(%arg2, %30) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %32 = "pphlo.greater"(%6, %2) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pint>
    %33 = "pphlo.reshape"(%32) : (tensor<100x1x!pphlo.pint>) -> tensor<1x100x!pphlo.pint>
    %34 = "pphlo.equal"(%26, %1) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pint>
    %35 = "pphlo.subtract"(%8, %34) : (tensor<100x1x!pphlo.pint>, tensor<100x1x!pphlo.pint>) -> tensor<100x1x!pphlo.pint>
    %36 = "pphlo.reshape"(%35) : (tensor<100x1x!pphlo.pint>) -> tensor<1x100x!pphlo.pint>
    %37 = "pphlo.and"(%33, %36) : (tensor<1x100x!pphlo.pint>, tensor<1x100x!pphlo.pint>) -> tensor<1x100x!pphlo.pint>
    %38 = "pphlo.convert"(%37) : (tensor<1x100x!pphlo.pint>) -> tensor<1x100x!pphlo.pfxp>
    %39 = "pphlo.reduce"(%38, %0) ( {
    ^bb0(%arg7: tensor<!pphlo.pfxp>, %arg8: tensor<!pphlo.pfxp>):  // no predecessors
      %52 = "pphlo.add"(%arg7, %arg8) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%52) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x100x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %40 = "pphlo.add"(%arg3, %39) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %41 = "pphlo.add"(%arg4, %39) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %42 = "pphlo.not"(%36) : (tensor<1x100x!pphlo.pint>) -> tensor<1x100x!pphlo.pint>
    %43 = "pphlo.and"(%33, %42) : (tensor<1x100x!pphlo.pint>, tensor<1x100x!pphlo.pint>) -> tensor<1x100x!pphlo.pint>
    %44 = "pphlo.convert"(%43) : (tensor<1x100x!pphlo.pint>) -> tensor<1x100x!pphlo.pfxp>
    %45 = "pphlo.reduce"(%44, %0) ( {
    ^bb0(%arg7: tensor<!pphlo.pfxp>, %arg8: tensor<!pphlo.pfxp>):  // no predecessors
      %52 = "pphlo.add"(%arg7, %arg8) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%52) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x100x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %46 = "pphlo.add"(%arg5, %45) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %47 = "pphlo.not"(%33) : (tensor<1x100x!pphlo.pint>) -> tensor<1x100x!pphlo.pint>
    %48 = "pphlo.and"(%47, %36) : (tensor<1x100x!pphlo.pint>, tensor<1x100x!pphlo.pint>) -> tensor<1x100x!pphlo.pint>
    %49 = "pphlo.convert"(%48) : (tensor<1x100x!pphlo.pint>) -> tensor<1x100x!pphlo.pfxp>
    %50 = "pphlo.reduce"(%49, %0) ( {
    ^bb0(%arg7: tensor<!pphlo.pfxp>, %arg8: tensor<!pphlo.pfxp>):  // no predecessors
      %52 = "pphlo.add"(%arg7, %arg8) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%52) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x100x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %51 = "pphlo.add"(%arg6, %50) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    return %10, %6, %13, %3, %31, %40, %41, %46, %51 : tensor<!pphlo.pint>, tensor<100x1x!pphlo.pfxp>, tensor<!pphlo.pint>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>
  }
}
