module @a_inference_train_step_2176__.301  {
  func @main(%arg0: tensor<1024x16x!pphlo.pfxp>, %arg1: tensor<1024x7x!pphlo.pfxp>, %arg2: tensor<1024x1x!pphlo.pfxp>, %arg3: tensor<16x!pphlo.pfxp>, %arg4: tensor<16x!pphlo.pfxp>, %arg5: tensor<7x!pphlo.pfxp>, %arg6: tensor<7x!pphlo.pfxp>, %arg7: tensor<23x1x!pphlo.pfxp>, %arg8: tensor<1x!pphlo.pfxp>, %arg9: tensor<!pphlo.pfxp>, %arg10: tensor<!pphlo.pfxp>, %arg11: tensor<!pphlo.pint>, %arg12: tensor<!pphlo.pfxp>) -> (tensor<!pphlo.pfxp>, tensor<23x1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pint>) {
    %0 = "pphlo.constant"() {value = dense<1> : tensor<i64>} : () -> tensor<!pphlo.pint>
    %1 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %2 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<1024x1xf32>} : () -> tensor<1024x1x!pphlo.pfxp>
    %3 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<1024x1xf32>} : () -> tensor<1024x1x!pphlo.pfxp>
    %4 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %5 = "pphlo.constant"() {value = dense<9.765625E-4> : tensor<1024x1xf32>} : () -> tensor<1024x1x!pphlo.pfxp>
    %6 = "pphlo.constant"() {value = dense<-9.765625E-4> : tensor<1024x1xf32>} : () -> tensor<1024x1x!pphlo.pfxp>
    %7 = "pphlo.constant"() {value = dense<1.000000e-01> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %8 = "pphlo.constant"() {value = dense<0.899999976> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %9 = "pphlo.constant"() {value = dense<-1.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %10 = "pphlo.constant"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %11 = "pphlo.constant"() {value = dense<0.0018248175> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %12 = "pphlo.constant"() {value = dense<0.00364963501> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %13 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<7xf32>} : () -> tensor<7x!pphlo.pfxp>
    %14 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<16xf32>} : () -> tensor<16x!pphlo.pfxp>
    %15 = "pphlo.constant"() {value = dense<1.024000e+03> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %16 = "pphlo.add"(%arg10, %15) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %17 = "pphlo.equal"(%16, %1) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pint>
    %18 = "pphlo.broadcast"(%arg3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<16x!pphlo.pfxp>) -> tensor<1024x16x!pphlo.pfxp>
    %19 = "pphlo.subtract"(%arg0, %18) : (tensor<1024x16x!pphlo.pfxp>, tensor<1024x16x!pphlo.pfxp>) -> tensor<1024x16x!pphlo.pfxp>
    %20 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<16xf32>} : () -> tensor<16x!pphlo.pfxp>
    %21 = "pphlo.power"(%arg4, %20) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %22 = "pphlo.maximum"(%21, %14) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %23 = "pphlo.broadcast"(%22) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<16x!pphlo.pfxp>) -> tensor<1024x16x!pphlo.pfxp>
    %24 = "pphlo.reciprocal"(%23) : (tensor<1024x16x!pphlo.pfxp>) -> tensor<1024x16x!pphlo.pfxp>
    %25 = "pphlo.multiply"(%19, %24) : (tensor<1024x16x!pphlo.pfxp>, tensor<1024x16x!pphlo.pfxp>) -> tensor<1024x16x!pphlo.pfxp>
    %26 = "pphlo.broadcast"(%arg5) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<7x!pphlo.pfxp>) -> tensor<1024x7x!pphlo.pfxp>
    %27 = "pphlo.subtract"(%arg1, %26) : (tensor<1024x7x!pphlo.pfxp>, tensor<1024x7x!pphlo.pfxp>) -> tensor<1024x7x!pphlo.pfxp>
    %28 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<7xf32>} : () -> tensor<7x!pphlo.pfxp>
    %29 = "pphlo.power"(%arg6, %28) : (tensor<7x!pphlo.pfxp>, tensor<7x!pphlo.pfxp>) -> tensor<7x!pphlo.pfxp>
    %30 = "pphlo.maximum"(%29, %13) : (tensor<7x!pphlo.pfxp>, tensor<7x!pphlo.pfxp>) -> tensor<7x!pphlo.pfxp>
    %31 = "pphlo.broadcast"(%30) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<7x!pphlo.pfxp>) -> tensor<1024x7x!pphlo.pfxp>
    %32 = "pphlo.reciprocal"(%31) : (tensor<1024x7x!pphlo.pfxp>) -> tensor<1024x7x!pphlo.pfxp>
    %33 = "pphlo.multiply"(%27, %32) : (tensor<1024x7x!pphlo.pfxp>, tensor<1024x7x!pphlo.pfxp>) -> tensor<1024x7x!pphlo.pfxp>
    %34 = "pphlo.concatenate"(%25, %33) {dimension = 1 : i64} : (tensor<1024x16x!pphlo.pfxp>, tensor<1024x7x!pphlo.pfxp>) -> tensor<1024x23x!pphlo.pfxp>
    %35 = "pphlo.reshape"(%arg7) : (tensor<23x1x!pphlo.pfxp>) -> tensor<23x!pphlo.pfxp>
    %36 = "pphlo.broadcast"(%35) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<23x!pphlo.pfxp>) -> tensor<1024x23x!pphlo.pfxp>
    %37 = "pphlo.multiply"(%34, %36) : (tensor<1024x23x!pphlo.pfxp>, tensor<1024x23x!pphlo.pfxp>) -> tensor<1024x23x!pphlo.pfxp>
    %38 = "pphlo.reduce"(%37, %1) ( {
    ^bb0(%arg13: tensor<!pphlo.pfxp>, %arg14: tensor<!pphlo.pfxp>):  // no predecessors
      %115 = "pphlo.add"(%arg13, %arg14) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%115) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1024x23x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<1024x!pphlo.pfxp>
    %39 = "pphlo.reshape"(%arg8) : (tensor<1x!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %40 = "pphlo.broadcast"(%39) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<1024x!pphlo.pfxp>
    %41 = "pphlo.add"(%38, %40) : (tensor<1024x!pphlo.pfxp>, tensor<1024x!pphlo.pfxp>) -> tensor<1024x!pphlo.pfxp>
    %42 = "pphlo.reshape"(%41) : (tensor<1024x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %43 = "pphlo.less"(%42, %2) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pint>
    %44 = "pphlo.constant"() {value = dense<1> : tensor<1024x1xi32>} : () -> tensor<1024x1x!pphlo.pint>
    %45 = "pphlo.subtract"(%44, %43) : (tensor<1024x1x!pphlo.pint>, tensor<1024x1x!pphlo.pint>) -> tensor<1024x1x!pphlo.pint>
    %46 = "pphlo.subtract"(%42, %2) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %47 = "pphlo.multiply"(%45, %46) : (tensor<1024x1x!pphlo.pint>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %48 = "pphlo.add"(%47, %2) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %49 = "pphlo.multiply"(%42, %arg2) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %50 = "pphlo.subtract"(%48, %49) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %51 = "pphlo.negate"(%42) : (tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %52 = "pphlo.subtract"(%51, %42) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %53 = "pphlo.multiply"(%45, %52) : (tensor<1024x1x!pphlo.pint>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %54 = "pphlo.add"(%53, %42) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %55 = "pphlo.exponential"(%54) : (tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %56 = "pphlo.log_plus_one"(%55) : (tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %57 = "pphlo.add"(%50, %56) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %58 = "pphlo.reduce"(%57, %1) ( {
    ^bb0(%arg13: tensor<!pphlo.pfxp>, %arg14: tensor<!pphlo.pfxp>):  // no predecessors
      %115 = "pphlo.add"(%arg13, %arg14) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%115) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1024x1x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %59 = "pphlo.add"(%arg9, %58) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %60 = "pphlo.reciprocal"(%16) : (tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %61 = "pphlo.multiply"(%59, %60) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %62 = "pphlo.subtract"(%1, %61) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %63 = "pphlo.multiply"(%17, %62) : (tensor<!pphlo.pint>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %64 = "pphlo.add"(%63, %61) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %65 = "pphlo.convert"(%arg11) : (tensor<!pphlo.pint>) -> tensor<!pphlo.pfxp>
    %66 = "pphlo.multiply"(%65, %12) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %67 = "pphlo.multiply"(%65, %11) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %68 = "pphlo.add"(%67, %4) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %69 = "pphlo.floor"(%68) : (tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %70 = "pphlo.multiply"(%69, %10) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %71 = "pphlo.subtract"(%66, %70) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %72 = "pphlo.add"(%71, %4) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %73 = "pphlo.abs"(%72) : (tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %74 = "pphlo.subtract"(%4, %73) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %75 = "pphlo.maximum"(%74, %1) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %76 = "pphlo.add"(%69, %9) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %77 = "pphlo.negate"(%76) : (tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %78 = "pphlo.power"(%10, %77) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %79 = "pphlo.multiply"(%75, %78) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %80 = "pphlo.multiply"(%79, %8) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %81 = "pphlo.add"(%80, %7) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %82 = "pphlo.broadcast"(%81) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<23x!pphlo.pfxp>
    %83 = "pphlo.transpose"(%34) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[23,1024]{0,1}"} : (tensor<1024x23x!pphlo.pfxp>) -> tensor<23x1024x!pphlo.pfxp>
    %84 = "pphlo.subtract"(%5, %2) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %85 = "pphlo.multiply"(%45, %84) : (tensor<1024x1x!pphlo.pint>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %86 = "pphlo.add"(%85, %2) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %87 = "pphlo.multiply"(%arg2, %6) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %88 = "pphlo.add"(%86, %87) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %89 = "pphlo.add"(%55, %3) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %90 = "pphlo.reciprocal"(%89) : (tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %91 = "pphlo.multiply"(%90, %3) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %92 = "pphlo.multiply"(%91, %5) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %93 = "pphlo.multiply"(%92, %55) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %94 = "pphlo.subtract"(%2, %93) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %95 = "pphlo.multiply"(%45, %94) : (tensor<1024x1x!pphlo.pint>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %96 = "pphlo.add"(%95, %93) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %97 = "pphlo.add"(%88, %96) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %98 = "pphlo.subtract"(%93, %2) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %99 = "pphlo.multiply"(%45, %98) : (tensor<1024x1x!pphlo.pint>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %100 = "pphlo.add"(%99, %2) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %101 = "pphlo.negate"(%100) : (tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %102 = "pphlo.add"(%97, %101) : (tensor<1024x1x!pphlo.pfxp>, tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x1x!pphlo.pfxp>
    %103 = "pphlo.reshape"(%102) : (tensor<1024x1x!pphlo.pfxp>) -> tensor<1024x!pphlo.pfxp>
    %104 = "pphlo.broadcast"(%103) {broadcast_dimensions = dense<1> : tensor<1xi64>, xla_shape = "f32[23,1024]{0,1}"} : (tensor<1024x!pphlo.pfxp>) -> tensor<23x1024x!pphlo.pfxp>
    %105 = "pphlo.multiply"(%83, %104) {xla_shape = "f32[23,1024]{0,1}"} : (tensor<23x1024x!pphlo.pfxp>, tensor<23x1024x!pphlo.pfxp>) -> tensor<23x1024x!pphlo.pfxp>
    %106 = "pphlo.reduce"(%105, %1) ( {
    ^bb0(%arg13: tensor<!pphlo.pfxp>, %arg14: tensor<!pphlo.pfxp>):  // no predecessors
      %115 = "pphlo.add"(%arg13, %arg14) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%115) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<23x1024x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<23x!pphlo.pfxp>
    %107 = "pphlo.multiply"(%82, %106) : (tensor<23x!pphlo.pfxp>, tensor<23x!pphlo.pfxp>) -> tensor<23x!pphlo.pfxp>
    %108 = "pphlo.reshape"(%107) : (tensor<23x!pphlo.pfxp>) -> tensor<23x1x!pphlo.pfxp>
    %109 = "pphlo.subtract"(%arg7, %108) : (tensor<23x1x!pphlo.pfxp>, tensor<23x1x!pphlo.pfxp>) -> tensor<23x1x!pphlo.pfxp>
    %110 = "pphlo.reshape"(%81) : (tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %111 = "pphlo.reduce"(%102, %1) ( {
    ^bb0(%arg13: tensor<!pphlo.pfxp>, %arg14: tensor<!pphlo.pfxp>):  // no predecessors
      %115 = "pphlo.add"(%arg13, %arg14) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%115) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<1024x1x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %112 = "pphlo.multiply"(%110, %111) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %113 = "pphlo.subtract"(%arg8, %112) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %114 = "pphlo.add"(%arg11, %0) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    return %64, %109, %113, %59, %16, %114 : tensor<!pphlo.pfxp>, tensor<23x1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pint>
  }
}
