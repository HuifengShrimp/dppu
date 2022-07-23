module @a_inference_test_step_172960__XlaMustCompile_true_config_proto___n_007_n_003CPU_020_001_n_007_n_003GPU_020_0002_002J_0008_001_202_001_000__executor_type____.407  {
  func @main(%arg0: tensor<100x3x!pphlo.pfxp>, %arg1: tensor<100x26x!pphlo.pfxp>, %arg2: tensor<100x!pphlo.pint>, %arg3: tensor<29x16x!pphlo.pfxp>, %arg4: tensor<16x!pphlo.pfxp>, %arg5: tensor<16x24x!pphlo.pfxp>, %arg6: tensor<24x!pphlo.pfxp>, %arg7: tensor<24x20x!pphlo.pfxp>, %arg8: tensor<20x!pphlo.pfxp>, %arg9: tensor<20x24x!pphlo.pfxp>, %arg10: tensor<24x!pphlo.pfxp>, %arg11: tensor<24x1x!pphlo.pfxp>, %arg12: tensor<1x!pphlo.pfxp>, %arg13: tensor<!pphlo.pfxp>, %arg14: tensor<!pphlo.pfxp>, %arg15: tensor<1x!pphlo.pfxp>, %arg16: tensor<1x!pphlo.pfxp>, %arg17: tensor<1x!pphlo.pfxp>, %arg18: tensor<1x!pphlo.pfxp>) -> (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) {
    %0 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1x!pphlo.pfxp>
    %1 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %2 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<100x1xf32>} : () -> tensor<100x1x!pphlo.pfxp>
    %3 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100xf32>} : () -> tensor<100x!pphlo.pfxp>
    %4 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x1xf32>} : () -> tensor<100x1x!pphlo.pfxp>
    %5 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x24xf32>} : () -> tensor<100x24x!pphlo.pfxp>
    %6 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x20xf32>} : () -> tensor<100x20x!pphlo.pfxp>
    %7 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x16xf32>} : () -> tensor<100x16x!pphlo.pfxp>
    %8 = "pphlo.constant"() {value = dense<1.000000e+02> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %9 = "pphlo.add"(%arg14, %8) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %10 = "pphlo.equal"(%9, %1) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pint>
    %11 = "pphlo.concatenate"(%arg1, %arg0) {dimension = 1 : i64} : (tensor<100x26x!pphlo.pfxp>, tensor<100x3x!pphlo.pfxp>) -> tensor<100x29x!pphlo.pfxp>
    %12 = "pphlo.dot"(%11, %arg3) : (tensor<100x29x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<100x16x!pphlo.pfxp>
    %13 = "pphlo.broadcast"(%arg4) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<16x!pphlo.pfxp>) -> tensor<100x16x!pphlo.pfxp>
    %14 = "pphlo.add"(%12, %13) : (tensor<100x16x!pphlo.pfxp>, tensor<100x16x!pphlo.pfxp>) -> tensor<100x16x!pphlo.pfxp>
    %15 = "pphlo.maximum"(%14, %7) : (tensor<100x16x!pphlo.pfxp>, tensor<100x16x!pphlo.pfxp>) -> tensor<100x16x!pphlo.pfxp>
    %16 = "pphlo.dot"(%15, %arg5) : (tensor<100x16x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %17 = "pphlo.broadcast"(%arg6) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %18 = "pphlo.add"(%16, %17) : (tensor<100x24x!pphlo.pfxp>, tensor<100x24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %19 = "pphlo.maximum"(%18, %5) : (tensor<100x24x!pphlo.pfxp>, tensor<100x24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %20 = "pphlo.dot"(%19, %arg7) : (tensor<100x24x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<100x20x!pphlo.pfxp>
    %21 = "pphlo.broadcast"(%arg8) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<20x!pphlo.pfxp>) -> tensor<100x20x!pphlo.pfxp>
    %22 = "pphlo.add"(%20, %21) : (tensor<100x20x!pphlo.pfxp>, tensor<100x20x!pphlo.pfxp>) -> tensor<100x20x!pphlo.pfxp>
    %23 = "pphlo.maximum"(%22, %6) : (tensor<100x20x!pphlo.pfxp>, tensor<100x20x!pphlo.pfxp>) -> tensor<100x20x!pphlo.pfxp>
    %24 = "pphlo.dot"(%23, %arg9) : (tensor<100x20x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %25 = "pphlo.broadcast"(%arg10) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %26 = "pphlo.add"(%24, %25) : (tensor<100x24x!pphlo.pfxp>, tensor<100x24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %27 = "pphlo.maximum"(%26, %5) : (tensor<100x24x!pphlo.pfxp>, tensor<100x24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %28 = "pphlo.reshape"(%arg11) : (tensor<24x1x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %29 = "pphlo.broadcast"(%28) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %30 = "pphlo.multiply"(%27, %29) : (tensor<100x24x!pphlo.pfxp>, tensor<100x24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %31 = "pphlo.reduce"(%30, %1) ( {
    ^bb0(%arg19: tensor<!pphlo.pfxp>, %arg20: tensor<!pphlo.pfxp>):  // no predecessors
      %99 = "pphlo.add"(%arg19, %arg20) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%99) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<100x24x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<100x!pphlo.pfxp>
    %32 = "pphlo.reshape"(%arg12) : (tensor<1x!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %33 = "pphlo.broadcast"(%32) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<100x!pphlo.pfxp>
    %34 = "pphlo.add"(%31, %33) : (tensor<100x!pphlo.pfxp>, tensor<100x!pphlo.pfxp>) -> tensor<100x!pphlo.pfxp>
    %35 = "pphlo.reshape"(%34) : (tensor<100x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %36 = "pphlo.less"(%35, %4) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pint>
    %37 = "pphlo.constant"() {value = dense<1> : tensor<100x1xi32>} : () -> tensor<100x1x!pphlo.pint>
    %38 = "pphlo.subtract"(%37, %36) : (tensor<100x1x!pphlo.pint>, tensor<100x1x!pphlo.pint>) -> tensor<100x1x!pphlo.pint>
    %39 = "pphlo.subtract"(%35, %4) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %40 = "pphlo.multiply"(%38, %39) : (tensor<100x1x!pphlo.pint>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %41 = "pphlo.add"(%40, %4) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %42 = "pphlo.convert"(%arg2) : (tensor<100x!pphlo.pint>) -> tensor<100x!pphlo.pfxp>
    %43 = "pphlo.reshape"(%42) : (tensor<100x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %44 = "pphlo.multiply"(%35, %43) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %45 = "pphlo.subtract"(%41, %44) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %46 = "pphlo.negate"(%35) : (tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %47 = "pphlo.subtract"(%46, %35) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %48 = "pphlo.multiply"(%38, %47) : (tensor<100x1x!pphlo.pint>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %49 = "pphlo.add"(%48, %35) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %50 = "pphlo.exponential"(%49) : (tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %51 = "pphlo.log_plus_one"(%50) : (tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %52 = "pphlo.add"(%45, %51) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %53 = "pphlo.reduce"(%52, %1) ( {
    ^bb0(%arg19: tensor<!pphlo.pfxp>, %arg20: tensor<!pphlo.pfxp>):  // no predecessors
      %99 = "pphlo.add"(%arg19, %arg20) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%99) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<100x1x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %54 = "pphlo.add"(%arg13, %53) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %55 = "pphlo.reciprocal"(%9) : (tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %56 = "pphlo.multiply"(%54, %55) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %57 = "pphlo.subtract"(%1, %56) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %58 = "pphlo.multiply"(%10, %57) : (tensor<!pphlo.pint>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %59 = "pphlo.add"(%58, %56) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %60 = "pphlo.equal"(%42, %3) : (tensor<100x!pphlo.pfxp>, tensor<100x!pphlo.pfxp>) -> tensor<100x!pphlo.pint>
    %61 = "pphlo.constant"() {value = dense<1> : tensor<100xi32>} : () -> tensor<100x!pphlo.pint>
    %62 = "pphlo.subtract"(%61, %60) : (tensor<100x!pphlo.pint>, tensor<100x!pphlo.pint>) -> tensor<100x!pphlo.pint>
    %63 = "pphlo.reshape"(%62) : (tensor<100x!pphlo.pint>) -> tensor<1x100x!pphlo.pint>
    %64 = "pphlo.logistic"(%35) : (tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %65 = "pphlo.greater"(%64, %2) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pint>
    %66 = "pphlo.reshape"(%65) : (tensor<100x1x!pphlo.pint>) -> tensor<1x100x!pphlo.pint>
    %67 = "pphlo.and"(%63, %66) : (tensor<1x100x!pphlo.pint>, tensor<1x100x!pphlo.pint>) -> tensor<1x100x!pphlo.pint>
    %68 = "pphlo.convert"(%67) : (tensor<1x100x!pphlo.pint>) -> tensor<1x100x!pphlo.pfxp>
    %69 = "pphlo.reduce"(%68, %1) ( {
    ^bb0(%arg19: tensor<!pphlo.pfxp>, %arg20: tensor<!pphlo.pfxp>):  // no predecessors
      %99 = "pphlo.add"(%arg19, %arg20) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%99) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x100x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %70 = "pphlo.add"(%arg15, %69) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %71 = "pphlo.not"(%63) : (tensor<1x100x!pphlo.pint>) -> tensor<1x100x!pphlo.pint>
    %72 = "pphlo.and"(%71, %66) : (tensor<1x100x!pphlo.pint>, tensor<1x100x!pphlo.pint>) -> tensor<1x100x!pphlo.pint>
    %73 = "pphlo.convert"(%72) : (tensor<1x100x!pphlo.pint>) -> tensor<1x100x!pphlo.pfxp>
    %74 = "pphlo.reduce"(%73, %1) ( {
    ^bb0(%arg19: tensor<!pphlo.pfxp>, %arg20: tensor<!pphlo.pfxp>):  // no predecessors
      %99 = "pphlo.add"(%arg19, %arg20) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%99) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x100x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %75 = "pphlo.add"(%arg16, %74) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %76 = "pphlo.add"(%70, %75) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %77 = "pphlo.equal"(%76, %0) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pint>
    %78 = "pphlo.reciprocal"(%76) : (tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %79 = "pphlo.multiply"(%70, %78) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %80 = "pphlo.subtract"(%0, %79) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %81 = "pphlo.multiply"(%77, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %82 = "pphlo.add"(%81, %79) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %83 = "pphlo.reshape"(%82) : (tensor<1x!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %84 = "pphlo.reduce"(%68, %1) ( {
    ^bb0(%arg19: tensor<!pphlo.pfxp>, %arg20: tensor<!pphlo.pfxp>):  // no predecessors
      %99 = "pphlo.add"(%arg19, %arg20) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%99) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x100x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %85 = "pphlo.add"(%arg17, %84) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %86 = "pphlo.not"(%66) : (tensor<1x100x!pphlo.pint>) -> tensor<1x100x!pphlo.pint>
    %87 = "pphlo.and"(%63, %86) : (tensor<1x100x!pphlo.pint>, tensor<1x100x!pphlo.pint>) -> tensor<1x100x!pphlo.pint>
    %88 = "pphlo.convert"(%87) : (tensor<1x100x!pphlo.pint>) -> tensor<1x100x!pphlo.pfxp>
    %89 = "pphlo.reduce"(%88, %1) ( {
    ^bb0(%arg19: tensor<!pphlo.pfxp>, %arg20: tensor<!pphlo.pfxp>):  // no predecessors
      %99 = "pphlo.add"(%arg19, %arg20) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%99) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x100x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %90 = "pphlo.add"(%arg18, %89) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %91 = "pphlo.add"(%85, %90) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %92 = "pphlo.equal"(%91, %0) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pint>
    %93 = "pphlo.reciprocal"(%91) : (tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %94 = "pphlo.multiply"(%85, %93) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %95 = "pphlo.subtract"(%0, %94) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %96 = "pphlo.multiply"(%92, %95) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %97 = "pphlo.add"(%96, %94) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %98 = "pphlo.reshape"(%97) : (tensor<1x!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    return %59, %83, %98, %54, %9, %70, %75, %85, %90 : tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>
  }
}
