module @a_inference_train_step_2047__XlaMustCompile_true_config_proto___n_007_n_003CPU_020_001_n_007_n_003GPU_020_0002_002J_0008_001_202_001_000__executor_type____.970  {
  func @main(%arg0: tensor<15x3x!pphlo.pfxp>, %arg1: tensor<15x26x!pphlo.pfxp>, %arg2: tensor<15x!pphlo.pint>, %arg3: tensor<15x!pphlo.pint>, %arg4: tensor<29x16x!pphlo.pfxp>, %arg5: tensor<16x!pphlo.pfxp>, %arg6: tensor<16x24x!pphlo.pfxp>, %arg7: tensor<24x!pphlo.pfxp>, %arg8: tensor<24x20x!pphlo.pfxp>, %arg9: tensor<20x!pphlo.pfxp>, %arg10: tensor<20x24x!pphlo.pfxp>, %arg11: tensor<24x!pphlo.pfxp>, %arg12: tensor<24x1x!pphlo.pfxp>, %arg13: tensor<1x!pphlo.pfxp>, %arg14: tensor<!pphlo.pfxp>, %arg15: tensor<!pphlo.pfxp>, %arg16: tensor<!pphlo.pfxp>, %arg17: tensor<!pphlo.pint>, %arg18: tensor<!pphlo.pfxp>, %arg19: tensor<!pphlo.pfxp>, %arg20: tensor<29x16x!pphlo.pfxp>, %arg21: tensor<29x16x!pphlo.pfxp>, %arg22: tensor<16x!pphlo.pfxp>, %arg23: tensor<16x!pphlo.pfxp>, %arg24: tensor<16x24x!pphlo.pfxp>, %arg25: tensor<16x24x!pphlo.pfxp>, %arg26: tensor<24x!pphlo.pfxp>, %arg27: tensor<24x!pphlo.pfxp>, %arg28: tensor<24x20x!pphlo.pfxp>, %arg29: tensor<24x20x!pphlo.pfxp>, %arg30: tensor<20x!pphlo.pfxp>, %arg31: tensor<20x!pphlo.pfxp>, %arg32: tensor<20x24x!pphlo.pfxp>, %arg33: tensor<20x24x!pphlo.pfxp>, %arg34: tensor<24x!pphlo.pfxp>, %arg35: tensor<24x!pphlo.pfxp>, %arg36: tensor<24x1x!pphlo.pfxp>, %arg37: tensor<24x1x!pphlo.pfxp>, %arg38: tensor<1x!pphlo.pfxp>, %arg39: tensor<1x!pphlo.pfxp>, %arg40: tensor<1x!pphlo.pfxp>, %arg41: tensor<1x!pphlo.pfxp>, %arg42: tensor<1x!pphlo.pfxp>, %arg43: tensor<1x!pphlo.pfxp>) -> (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pint>, tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) {
    %0 = "pphlo.constant"() {value = dense<1> : tensor<i64>} : () -> tensor<!pphlo.pint>
    %1 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<1xf32>} : () -> tensor<1x!pphlo.pfxp>
    %2 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %3 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %4 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<24x1xf32>} : () -> tensor<24x1x!pphlo.pfxp>
    %5 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<24xf32>} : () -> tensor<24x!pphlo.pfxp>
    %6 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<20x24xf32>} : () -> tensor<20x24x!pphlo.pfxp>
    %7 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<20xf32>} : () -> tensor<20x!pphlo.pfxp>
    %8 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<24x20xf32>} : () -> tensor<24x20x!pphlo.pfxp>
    %9 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<16x24xf32>} : () -> tensor<16x24x!pphlo.pfxp>
    %10 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<16xf32>} : () -> tensor<16x!pphlo.pfxp>
    %11 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<29x16xf32>} : () -> tensor<29x16x!pphlo.pfxp>
    %12 = "pphlo.constant"() {value = dense<2.000000e+00> : tensor<15x24xf32>} : () -> tensor<15x24x!pphlo.pfxp>
    %13 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<15x24xf32>} : () -> tensor<15x24x!pphlo.pfxp>
    %14 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<15x1xf32>} : () -> tensor<15x1x!pphlo.pfxp>
    %15 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<15x1xf32>} : () -> tensor<15x1x!pphlo.pfxp>
    %16 = "pphlo.constant"() {value = dense<0.0666666701> : tensor<15xf32>} : () -> tensor<15x!pphlo.pfxp>
    %17 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<15x20xf32>} : () -> tensor<15x20x!pphlo.pfxp>
    %18 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<15x16xf32>} : () -> tensor<15x16x!pphlo.pfxp>
    %19 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1x!pphlo.pfxp>
    %20 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<15x1xf32>} : () -> tensor<15x1x!pphlo.pfxp>
    %21 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<15xf32>} : () -> tensor<15x!pphlo.pfxp>
    %22 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<15x24xf32>} : () -> tensor<15x24x!pphlo.pfxp>
    %23 = "pphlo.constant"() {value = dense<1.500000e+01> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %24 = "pphlo.add"(%arg15, %23) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %25 = "pphlo.equal"(%24, %3) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pint>
    %26 = "pphlo.rng_uniform"(%3, %2) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %27 = "pphlo.less"(%26, %22) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pint>
    %28 = "pphlo.constant"() {value = dense<1> : tensor<15x24xi32>} : () -> tensor<15x24x!pphlo.pint>
    %29 = "pphlo.subtract"(%28, %27) : (tensor<15x24x!pphlo.pint>, tensor<15x24x!pphlo.pint>) -> tensor<15x24x!pphlo.pint>
    %30 = "pphlo.concatenate"(%arg1, %arg0) {dimension = 1 : i64} : (tensor<15x26x!pphlo.pfxp>, tensor<15x3x!pphlo.pfxp>) -> tensor<15x29x!pphlo.pfxp>
    %31 = "pphlo.dot"(%30, %arg4) : (tensor<15x29x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<15x16x!pphlo.pfxp>
    %32 = "pphlo.broadcast"(%arg5) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<16x!pphlo.pfxp>) -> tensor<15x16x!pphlo.pfxp>
    %33 = "pphlo.add"(%31, %32) : (tensor<15x16x!pphlo.pfxp>, tensor<15x16x!pphlo.pfxp>) -> tensor<15x16x!pphlo.pfxp>
    %34 = "pphlo.maximum"(%33, %18) : (tensor<15x16x!pphlo.pfxp>, tensor<15x16x!pphlo.pfxp>) -> tensor<15x16x!pphlo.pfxp>
    %35 = "pphlo.dot"(%34, %arg6) : (tensor<15x16x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %36 = "pphlo.broadcast"(%arg7) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %37 = "pphlo.add"(%35, %36) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %38 = "pphlo.maximum"(%37, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %39 = "pphlo.multiply"(%38, %12) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %40 = "pphlo.subtract"(%39, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %41 = "pphlo.multiply"(%29, %40) : (tensor<15x24x!pphlo.pint>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %42 = "pphlo.add"(%41, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %43 = "pphlo.dot"(%42, %arg8) : (tensor<15x24x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<15x20x!pphlo.pfxp>
    %44 = "pphlo.broadcast"(%arg9) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<20x!pphlo.pfxp>) -> tensor<15x20x!pphlo.pfxp>
    %45 = "pphlo.add"(%43, %44) : (tensor<15x20x!pphlo.pfxp>, tensor<15x20x!pphlo.pfxp>) -> tensor<15x20x!pphlo.pfxp>
    %46 = "pphlo.maximum"(%45, %17) : (tensor<15x20x!pphlo.pfxp>, tensor<15x20x!pphlo.pfxp>) -> tensor<15x20x!pphlo.pfxp>
    %47 = "pphlo.dot"(%46, %arg10) : (tensor<15x20x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %48 = "pphlo.broadcast"(%arg11) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %49 = "pphlo.add"(%47, %48) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %50 = "pphlo.maximum"(%49, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %51 = "pphlo.reshape"(%arg12) : (tensor<24x1x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %52 = "pphlo.broadcast"(%51) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %53 = "pphlo.multiply"(%50, %52) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %54 = "pphlo.reduce"(%53, %3) ( {
    ^bb0(%arg44: tensor<!pphlo.pfxp>, %arg45: tensor<!pphlo.pfxp>):  // no predecessors
      %373 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%373) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<15x24x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<15x!pphlo.pfxp>
    %55 = "pphlo.reshape"(%arg13) : (tensor<1x!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %56 = "pphlo.broadcast"(%55) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<15x!pphlo.pfxp>
    %57 = "pphlo.add"(%54, %56) : (tensor<15x!pphlo.pfxp>, tensor<15x!pphlo.pfxp>) -> tensor<15x!pphlo.pfxp>
    %58 = "pphlo.reshape"(%57) : (tensor<15x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %59 = "pphlo.less"(%58, %15) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pint>
    %60 = "pphlo.constant"() {value = dense<1> : tensor<15x1xi32>} : () -> tensor<15x1x!pphlo.pint>
    %61 = "pphlo.subtract"(%60, %59) : (tensor<15x1x!pphlo.pint>, tensor<15x1x!pphlo.pint>) -> tensor<15x1x!pphlo.pint>
    %62 = "pphlo.subtract"(%58, %15) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %63 = "pphlo.multiply"(%61, %62) : (tensor<15x1x!pphlo.pint>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %64 = "pphlo.add"(%63, %15) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %65 = "pphlo.convert"(%arg2) : (tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pfxp>
    %66 = "pphlo.reshape"(%65) : (tensor<15x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %67 = "pphlo.multiply"(%58, %66) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %68 = "pphlo.subtract"(%64, %67) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %69 = "pphlo.negate"(%58) : (tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %70 = "pphlo.subtract"(%69, %58) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %71 = "pphlo.multiply"(%61, %70) : (tensor<15x1x!pphlo.pint>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %72 = "pphlo.add"(%71, %58) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %73 = "pphlo.exponential"(%72) : (tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %74 = "pphlo.log_plus_one"(%73) : (tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %75 = "pphlo.add"(%68, %74) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %76 = "pphlo.reshape"(%75) : (tensor<15x1x!pphlo.pfxp>) -> tensor<15x!pphlo.pfxp>
    %77 = "pphlo.convert"(%arg3) : (tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pfxp>
    %78 = "pphlo.multiply"(%76, %77) : (tensor<15x!pphlo.pfxp>, tensor<15x!pphlo.pfxp>) -> tensor<15x!pphlo.pfxp>
    %79 = "pphlo.reduce"(%78, %3) ( {
    ^bb0(%arg44: tensor<!pphlo.pfxp>, %arg45: tensor<!pphlo.pfxp>):  // no predecessors
      %373 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%373) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %80 = "pphlo.add"(%arg14, %79) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %81 = "pphlo.reciprocal"(%24) : (tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %82 = "pphlo.multiply"(%80, %81) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %83 = "pphlo.subtract"(%3, %82) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %84 = "pphlo.multiply"(%25, %83) : (tensor<!pphlo.pint>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %85 = "pphlo.add"(%84, %82) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %86 = "pphlo.equal"(%65, %21) : (tensor<15x!pphlo.pfxp>, tensor<15x!pphlo.pfxp>) -> tensor<15x!pphlo.pint>
    %87 = "pphlo.constant"() {value = dense<1> : tensor<15xi32>} : () -> tensor<15x!pphlo.pint>
    %88 = "pphlo.subtract"(%87, %86) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %89 = "pphlo.reshape"(%88) : (tensor<15x!pphlo.pint>) -> tensor<1x15x!pphlo.pint>
    %90 = "pphlo.logistic"(%58) : (tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %91 = "pphlo.greater"(%90, %20) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pint>
    %92 = "pphlo.reshape"(%91) : (tensor<15x1x!pphlo.pint>) -> tensor<1x15x!pphlo.pint>
    %93 = "pphlo.and"(%89, %92) : (tensor<1x15x!pphlo.pint>, tensor<1x15x!pphlo.pint>) -> tensor<1x15x!pphlo.pint>
    %94 = "pphlo.convert"(%93) : (tensor<1x15x!pphlo.pint>) -> tensor<1x15x!pphlo.pfxp>
    %95 = "pphlo.reduce"(%94, %3) ( {
    ^bb0(%arg44: tensor<!pphlo.pfxp>, %arg45: tensor<!pphlo.pfxp>):  // no predecessors
      %373 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%373) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x15x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %96 = "pphlo.add"(%arg40, %95) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %97 = "pphlo.not"(%89) : (tensor<1x15x!pphlo.pint>) -> tensor<1x15x!pphlo.pint>
    %98 = "pphlo.and"(%97, %92) : (tensor<1x15x!pphlo.pint>, tensor<1x15x!pphlo.pint>) -> tensor<1x15x!pphlo.pint>
    %99 = "pphlo.convert"(%98) : (tensor<1x15x!pphlo.pint>) -> tensor<1x15x!pphlo.pfxp>
    %100 = "pphlo.reduce"(%99, %3) ( {
    ^bb0(%arg44: tensor<!pphlo.pfxp>, %arg45: tensor<!pphlo.pfxp>):  // no predecessors
      %373 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%373) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x15x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %101 = "pphlo.add"(%arg41, %100) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %102 = "pphlo.add"(%96, %101) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %103 = "pphlo.equal"(%102, %19) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pint>
    %104 = "pphlo.reciprocal"(%102) : (tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %105 = "pphlo.multiply"(%96, %104) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %106 = "pphlo.subtract"(%19, %105) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %107 = "pphlo.multiply"(%103, %106) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %108 = "pphlo.add"(%107, %105) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %109 = "pphlo.reshape"(%108) : (tensor<1x!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %110 = "pphlo.reduce"(%94, %3) ( {
    ^bb0(%arg44: tensor<!pphlo.pfxp>, %arg45: tensor<!pphlo.pfxp>):  // no predecessors
      %373 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%373) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x15x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %111 = "pphlo.add"(%arg42, %110) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %112 = "pphlo.not"(%92) : (tensor<1x15x!pphlo.pint>) -> tensor<1x15x!pphlo.pint>
    %113 = "pphlo.and"(%89, %112) : (tensor<1x15x!pphlo.pint>, tensor<1x15x!pphlo.pint>) -> tensor<1x15x!pphlo.pint>
    %114 = "pphlo.convert"(%113) : (tensor<1x15x!pphlo.pint>) -> tensor<1x15x!pphlo.pfxp>
    %115 = "pphlo.reduce"(%114, %3) ( {
    ^bb0(%arg44: tensor<!pphlo.pfxp>, %arg45: tensor<!pphlo.pfxp>):  // no predecessors
      %373 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%373) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x15x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %116 = "pphlo.add"(%arg43, %115) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %117 = "pphlo.add"(%111, %116) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %118 = "pphlo.equal"(%117, %19) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pint>
    %119 = "pphlo.reciprocal"(%117) : (tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %120 = "pphlo.multiply"(%111, %119) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %121 = "pphlo.subtract"(%19, %120) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %122 = "pphlo.multiply"(%118, %121) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %123 = "pphlo.add"(%122, %120) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %124 = "pphlo.reshape"(%123) : (tensor<1x!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %125 = "pphlo.greater"(%34, %18) : (tensor<15x16x!pphlo.pfxp>, tensor<15x16x!pphlo.pfxp>) -> tensor<15x16x!pphlo.pint>
    %126 = "pphlo.greater"(%38, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pint>
    %127 = "pphlo.greater"(%46, %17) : (tensor<15x20x!pphlo.pfxp>, tensor<15x20x!pphlo.pfxp>) -> tensor<15x20x!pphlo.pint>
    %128 = "pphlo.greater"(%50, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pint>
    %129 = "pphlo.multiply"(%77, %16) : (tensor<15x!pphlo.pfxp>, tensor<15x!pphlo.pfxp>) -> tensor<15x!pphlo.pfxp>
    %130 = "pphlo.reshape"(%129) : (tensor<15x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %131 = "pphlo.subtract"(%130, %15) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %132 = "pphlo.multiply"(%61, %131) : (tensor<15x1x!pphlo.pint>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %133 = "pphlo.add"(%132, %15) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %134 = "pphlo.negate"(%130) : (tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %135 = "pphlo.multiply"(%134, %66) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %136 = "pphlo.add"(%133, %135) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %137 = "pphlo.add"(%73, %14) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %138 = "pphlo.reciprocal"(%137) : (tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %139 = "pphlo.multiply"(%138, %14) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %140 = "pphlo.multiply"(%130, %139) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %141 = "pphlo.multiply"(%140, %73) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %142 = "pphlo.subtract"(%15, %141) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %143 = "pphlo.multiply"(%61, %142) : (tensor<15x1x!pphlo.pint>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %144 = "pphlo.add"(%143, %141) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %145 = "pphlo.add"(%136, %144) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %146 = "pphlo.subtract"(%141, %15) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %147 = "pphlo.multiply"(%61, %146) : (tensor<15x1x!pphlo.pint>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %148 = "pphlo.add"(%147, %15) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %149 = "pphlo.negate"(%148) : (tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %150 = "pphlo.add"(%145, %149) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %151 = "pphlo.reshape"(%150) : (tensor<15x1x!pphlo.pfxp>) -> tensor<15x!pphlo.pfxp>
    %152 = "pphlo.broadcast"(%151) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<15x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %153 = "pphlo.multiply"(%152, %52) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %154 = "pphlo.subtract"(%153, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %155 = "pphlo.multiply"(%128, %154) : (tensor<15x24x!pphlo.pint>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %156 = "pphlo.add"(%155, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %157 = "pphlo.transpose"(%arg10) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<20x24x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %158 = "pphlo.dot"(%156, %157) : (tensor<15x24x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<15x20x!pphlo.pfxp>
    %159 = "pphlo.subtract"(%158, %17) : (tensor<15x20x!pphlo.pfxp>, tensor<15x20x!pphlo.pfxp>) -> tensor<15x20x!pphlo.pfxp>
    %160 = "pphlo.multiply"(%127, %159) : (tensor<15x20x!pphlo.pint>, tensor<15x20x!pphlo.pfxp>) -> tensor<15x20x!pphlo.pfxp>
    %161 = "pphlo.add"(%160, %17) : (tensor<15x20x!pphlo.pfxp>, tensor<15x20x!pphlo.pfxp>) -> tensor<15x20x!pphlo.pfxp>
    %162 = "pphlo.transpose"(%arg8) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<24x20x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %163 = "pphlo.dot"(%161, %162) : (tensor<15x20x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %164 = "pphlo.subtract"(%163, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %165 = "pphlo.multiply"(%29, %164) : (tensor<15x24x!pphlo.pint>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %166 = "pphlo.add"(%165, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %167 = "pphlo.multiply"(%166, %12) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %168 = "pphlo.subtract"(%167, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %169 = "pphlo.multiply"(%126, %168) : (tensor<15x24x!pphlo.pint>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %170 = "pphlo.add"(%169, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %171 = "pphlo.transpose"(%arg6) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16x24x!pphlo.pfxp>) -> tensor<24x16x!pphlo.pfxp>
    %172 = "pphlo.dot"(%170, %171) : (tensor<15x24x!pphlo.pfxp>, tensor<24x16x!pphlo.pfxp>) -> tensor<15x16x!pphlo.pfxp>
    %173 = "pphlo.subtract"(%172, %18) : (tensor<15x16x!pphlo.pfxp>, tensor<15x16x!pphlo.pfxp>) -> tensor<15x16x!pphlo.pfxp>
    %174 = "pphlo.multiply"(%125, %173) : (tensor<15x16x!pphlo.pint>, tensor<15x16x!pphlo.pfxp>) -> tensor<15x16x!pphlo.pfxp>
    %175 = "pphlo.add"(%174, %18) : (tensor<15x16x!pphlo.pfxp>, tensor<15x16x!pphlo.pfxp>) -> tensor<15x16x!pphlo.pfxp>
    %176 = "pphlo.transpose"(%30) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<15x29x!pphlo.pfxp>) -> tensor<29x15x!pphlo.pfxp>
    %177 = "pphlo.dot"(%176, %175) : (tensor<29x15x!pphlo.pfxp>, tensor<15x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %178 = "pphlo.subtract"(%177, %arg20) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %179 = "pphlo.subtract"(%2, %arg18) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %180 = "pphlo.broadcast"(%179) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %181 = "pphlo.multiply"(%178, %180) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %182 = "pphlo.add"(%arg20, %181) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %183 = "pphlo.add"(%arg17, %0) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %184 = "pphlo.convert"(%183) : (tensor<!pphlo.pint>) -> tensor<!pphlo.pfxp>
    %185 = "pphlo.power"(%arg19, %184) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %186 = "pphlo.subtract"(%2, %185) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %187 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %188 = "pphlo.power"(%186, %187) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %189 = "pphlo.multiply"(%arg16, %188) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %190 = "pphlo.power"(%arg18, %184) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %191 = "pphlo.subtract"(%2, %190) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %192 = "pphlo.reciprocal"(%191) : (tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %193 = "pphlo.multiply"(%189, %192) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %194 = "pphlo.broadcast"(%193) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %195 = "pphlo.multiply"(%182, %194) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %196 = "pphlo.multiply"(%177, %177) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %197 = "pphlo.subtract"(%196, %arg21) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %198 = "pphlo.subtract"(%2, %arg19) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %199 = "pphlo.broadcast"(%198) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %200 = "pphlo.multiply"(%197, %199) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %201 = "pphlo.add"(%arg21, %200) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %202 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<29x16xf32>} : () -> tensor<29x16x!pphlo.pfxp>
    %203 = "pphlo.power"(%201, %202) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %204 = "pphlo.add"(%203, %11) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %205 = "pphlo.reciprocal"(%204) : (tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %206 = "pphlo.multiply"(%195, %205) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %207 = "pphlo.subtract"(%arg4, %206) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %208 = "pphlo.reduce"(%175, %3) ( {
    ^bb0(%arg44: tensor<!pphlo.pfxp>, %arg45: tensor<!pphlo.pfxp>):  // no predecessors
      %373 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%373) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x16x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %209 = "pphlo.subtract"(%208, %arg22) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %210 = "pphlo.broadcast"(%179) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %211 = "pphlo.multiply"(%209, %210) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %212 = "pphlo.add"(%arg22, %211) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %213 = "pphlo.broadcast"(%193) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %214 = "pphlo.multiply"(%212, %213) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %215 = "pphlo.multiply"(%208, %208) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %216 = "pphlo.subtract"(%215, %arg23) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %217 = "pphlo.broadcast"(%198) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %218 = "pphlo.multiply"(%216, %217) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %219 = "pphlo.add"(%arg23, %218) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %220 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<16xf32>} : () -> tensor<16x!pphlo.pfxp>
    %221 = "pphlo.power"(%219, %220) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %222 = "pphlo.add"(%221, %10) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %223 = "pphlo.reciprocal"(%222) : (tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %224 = "pphlo.multiply"(%214, %223) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %225 = "pphlo.subtract"(%arg5, %224) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %226 = "pphlo.transpose"(%34) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<15x16x!pphlo.pfxp>) -> tensor<16x15x!pphlo.pfxp>
    %227 = "pphlo.dot"(%226, %170) : (tensor<16x15x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %228 = "pphlo.subtract"(%227, %arg24) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %229 = "pphlo.broadcast"(%179) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %230 = "pphlo.multiply"(%228, %229) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %231 = "pphlo.add"(%arg24, %230) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %232 = "pphlo.broadcast"(%193) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %233 = "pphlo.multiply"(%231, %232) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %234 = "pphlo.multiply"(%227, %227) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %235 = "pphlo.subtract"(%234, %arg25) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %236 = "pphlo.broadcast"(%198) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %237 = "pphlo.multiply"(%235, %236) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %238 = "pphlo.add"(%arg25, %237) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %239 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<16x24xf32>} : () -> tensor<16x24x!pphlo.pfxp>
    %240 = "pphlo.power"(%238, %239) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %241 = "pphlo.add"(%240, %9) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %242 = "pphlo.reciprocal"(%241) : (tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %243 = "pphlo.multiply"(%233, %242) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %244 = "pphlo.subtract"(%arg6, %243) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %245 = "pphlo.reduce"(%170, %3) ( {
    ^bb0(%arg44: tensor<!pphlo.pfxp>, %arg45: tensor<!pphlo.pfxp>):  // no predecessors
      %373 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%373) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x24x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %246 = "pphlo.subtract"(%245, %arg26) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %247 = "pphlo.broadcast"(%179) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %248 = "pphlo.multiply"(%246, %247) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %249 = "pphlo.add"(%arg26, %248) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %250 = "pphlo.broadcast"(%193) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %251 = "pphlo.multiply"(%249, %250) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %252 = "pphlo.multiply"(%245, %245) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %253 = "pphlo.subtract"(%252, %arg27) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %254 = "pphlo.broadcast"(%198) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %255 = "pphlo.multiply"(%253, %254) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %256 = "pphlo.add"(%arg27, %255) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %257 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<24xf32>} : () -> tensor<24x!pphlo.pfxp>
    %258 = "pphlo.power"(%256, %257) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %259 = "pphlo.add"(%258, %5) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %260 = "pphlo.reciprocal"(%259) : (tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %261 = "pphlo.multiply"(%251, %260) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %262 = "pphlo.subtract"(%arg7, %261) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %263 = "pphlo.transpose"(%42) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<15x24x!pphlo.pfxp>) -> tensor<24x15x!pphlo.pfxp>
    %264 = "pphlo.dot"(%263, %161) : (tensor<24x15x!pphlo.pfxp>, tensor<15x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %265 = "pphlo.subtract"(%264, %arg28) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %266 = "pphlo.broadcast"(%179) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %267 = "pphlo.multiply"(%265, %266) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %268 = "pphlo.add"(%arg28, %267) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %269 = "pphlo.broadcast"(%193) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %270 = "pphlo.multiply"(%268, %269) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %271 = "pphlo.multiply"(%264, %264) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %272 = "pphlo.subtract"(%271, %arg29) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %273 = "pphlo.broadcast"(%198) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %274 = "pphlo.multiply"(%272, %273) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %275 = "pphlo.add"(%arg29, %274) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %276 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<24x20xf32>} : () -> tensor<24x20x!pphlo.pfxp>
    %277 = "pphlo.power"(%275, %276) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %278 = "pphlo.add"(%277, %8) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %279 = "pphlo.reciprocal"(%278) : (tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %280 = "pphlo.multiply"(%270, %279) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %281 = "pphlo.subtract"(%arg8, %280) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %282 = "pphlo.reduce"(%161, %3) ( {
    ^bb0(%arg44: tensor<!pphlo.pfxp>, %arg45: tensor<!pphlo.pfxp>):  // no predecessors
      %373 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%373) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x20x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %283 = "pphlo.subtract"(%282, %arg30) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %284 = "pphlo.broadcast"(%179) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %285 = "pphlo.multiply"(%283, %284) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %286 = "pphlo.add"(%arg30, %285) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %287 = "pphlo.broadcast"(%193) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %288 = "pphlo.multiply"(%286, %287) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %289 = "pphlo.multiply"(%282, %282) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %290 = "pphlo.subtract"(%289, %arg31) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %291 = "pphlo.broadcast"(%198) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %292 = "pphlo.multiply"(%290, %291) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %293 = "pphlo.add"(%arg31, %292) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %294 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<20xf32>} : () -> tensor<20x!pphlo.pfxp>
    %295 = "pphlo.power"(%293, %294) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %296 = "pphlo.add"(%295, %7) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %297 = "pphlo.reciprocal"(%296) : (tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %298 = "pphlo.multiply"(%288, %297) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %299 = "pphlo.subtract"(%arg9, %298) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %300 = "pphlo.transpose"(%46) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<15x20x!pphlo.pfxp>) -> tensor<20x15x!pphlo.pfxp>
    %301 = "pphlo.dot"(%300, %156) : (tensor<20x15x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %302 = "pphlo.subtract"(%301, %arg32) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %303 = "pphlo.broadcast"(%179) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %304 = "pphlo.multiply"(%302, %303) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %305 = "pphlo.add"(%arg32, %304) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %306 = "pphlo.broadcast"(%193) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %307 = "pphlo.multiply"(%305, %306) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %308 = "pphlo.multiply"(%301, %301) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %309 = "pphlo.subtract"(%308, %arg33) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %310 = "pphlo.broadcast"(%198) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %311 = "pphlo.multiply"(%309, %310) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %312 = "pphlo.add"(%arg33, %311) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %313 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<20x24xf32>} : () -> tensor<20x24x!pphlo.pfxp>
    %314 = "pphlo.power"(%312, %313) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %315 = "pphlo.add"(%314, %6) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %316 = "pphlo.reciprocal"(%315) : (tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %317 = "pphlo.multiply"(%307, %316) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %318 = "pphlo.subtract"(%arg10, %317) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %319 = "pphlo.reduce"(%156, %3) ( {
    ^bb0(%arg44: tensor<!pphlo.pfxp>, %arg45: tensor<!pphlo.pfxp>):  // no predecessors
      %373 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%373) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x24x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %320 = "pphlo.subtract"(%319, %arg34) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %321 = "pphlo.multiply"(%320, %247) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %322 = "pphlo.add"(%arg34, %321) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %323 = "pphlo.multiply"(%322, %250) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %324 = "pphlo.multiply"(%319, %319) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %325 = "pphlo.subtract"(%324, %arg35) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %326 = "pphlo.multiply"(%325, %254) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %327 = "pphlo.add"(%arg35, %326) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %328 = "pphlo.power"(%327, %257) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %329 = "pphlo.add"(%328, %5) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %330 = "pphlo.reciprocal"(%329) : (tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %331 = "pphlo.multiply"(%323, %330) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %332 = "pphlo.subtract"(%arg11, %331) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %333 = "pphlo.transpose"(%50) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[24,15]{0,1}"} : (tensor<15x24x!pphlo.pfxp>) -> tensor<24x15x!pphlo.pfxp>
    %334 = "pphlo.broadcast"(%151) {broadcast_dimensions = dense<1> : tensor<1xi64>, xla_shape = "f32[24,15]{0,1}"} : (tensor<15x!pphlo.pfxp>) -> tensor<24x15x!pphlo.pfxp>
    %335 = "pphlo.multiply"(%333, %334) {xla_shape = "f32[24,15]{0,1}"} : (tensor<24x15x!pphlo.pfxp>, tensor<24x15x!pphlo.pfxp>) -> tensor<24x15x!pphlo.pfxp>
    %336 = "pphlo.reduce"(%335, %3) ( {
    ^bb0(%arg44: tensor<!pphlo.pfxp>, %arg45: tensor<!pphlo.pfxp>):  // no predecessors
      %373 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%373) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<24x15x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %337 = "pphlo.reshape"(%336) : (tensor<24x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %338 = "pphlo.subtract"(%337, %arg36) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %339 = "pphlo.broadcast"(%179) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %340 = "pphlo.multiply"(%338, %339) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %341 = "pphlo.add"(%arg36, %340) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %342 = "pphlo.broadcast"(%193) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %343 = "pphlo.multiply"(%341, %342) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %344 = "pphlo.multiply"(%337, %337) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %345 = "pphlo.subtract"(%344, %arg37) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %346 = "pphlo.broadcast"(%198) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %347 = "pphlo.multiply"(%345, %346) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %348 = "pphlo.add"(%arg37, %347) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %349 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<24x1xf32>} : () -> tensor<24x1x!pphlo.pfxp>
    %350 = "pphlo.power"(%348, %349) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %351 = "pphlo.add"(%350, %4) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %352 = "pphlo.reciprocal"(%351) : (tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %353 = "pphlo.multiply"(%343, %352) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %354 = "pphlo.subtract"(%arg12, %353) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %355 = "pphlo.reduce"(%150, %3) ( {
    ^bb0(%arg44: tensor<!pphlo.pfxp>, %arg45: tensor<!pphlo.pfxp>):  // no predecessors
      %373 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%373) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x1x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %356 = "pphlo.subtract"(%355, %arg38) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %357 = "pphlo.reshape"(%179) : (tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %358 = "pphlo.multiply"(%356, %357) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %359 = "pphlo.add"(%arg38, %358) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %360 = "pphlo.reshape"(%193) : (tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %361 = "pphlo.multiply"(%359, %360) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %362 = "pphlo.multiply"(%355, %355) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %363 = "pphlo.subtract"(%362, %arg39) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %364 = "pphlo.reshape"(%198) : (tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %365 = "pphlo.multiply"(%363, %364) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %366 = "pphlo.add"(%arg39, %365) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %367 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<1xf32>} : () -> tensor<1x!pphlo.pfxp>
    %368 = "pphlo.power"(%366, %367) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %369 = "pphlo.add"(%368, %1) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %370 = "pphlo.reciprocal"(%369) : (tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %371 = "pphlo.multiply"(%361, %370) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %372 = "pphlo.subtract"(%arg13, %371) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    return %85, %109, %124, %207, %225, %244, %262, %281, %299, %318, %332, %354, %372, %80, %24, %183, %182, %201, %212, %219, %231, %238, %249, %256, %268, %275, %286, %293, %305, %312, %322, %327, %341, %348, %359, %366, %96, %101, %111, %116 : tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pint>, tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>
  }
}
