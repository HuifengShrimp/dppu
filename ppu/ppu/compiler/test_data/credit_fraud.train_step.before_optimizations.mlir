module @a_inference_train_step_1585__XlaMustCompile_true_config_proto___n_007_n_003CPU_020_001_n_007_n_003GPU_020_0002_002J_0008_001_202_001_000__executor_type____.683  {
  func @main(%arg0: tensor<15x3x!pphlo.pfxp>, %arg1: tensor<15x26x!pphlo.pfxp>, %arg2: tensor<15x!pphlo.pint>, %arg3: tensor<15x!pphlo.pint>, %arg4: tensor<29x16x!pphlo.pfxp>, %arg5: tensor<16x!pphlo.pfxp>, %arg6: tensor<16x24x!pphlo.pfxp>, %arg7: tensor<24x!pphlo.pfxp>, %arg8: tensor<24x20x!pphlo.pfxp>, %arg9: tensor<20x!pphlo.pfxp>, %arg10: tensor<20x24x!pphlo.pfxp>, %arg11: tensor<24x!pphlo.pfxp>, %arg12: tensor<24x1x!pphlo.pfxp>, %arg13: tensor<1x!pphlo.pfxp>, %arg14: tensor<!pphlo.pfxp>, %arg15: tensor<!pphlo.pfxp>, %arg16: tensor<!pphlo.pfxp>, %arg17: tensor<!pphlo.pint>, %arg18: tensor<!pphlo.pfxp>, %arg19: tensor<!pphlo.pfxp>, %arg20: tensor<29x16x!pphlo.pfxp>, %arg21: tensor<29x16x!pphlo.pfxp>, %arg22: tensor<16x!pphlo.pfxp>, %arg23: tensor<16x!pphlo.pfxp>, %arg24: tensor<16x24x!pphlo.pfxp>, %arg25: tensor<16x24x!pphlo.pfxp>, %arg26: tensor<24x!pphlo.pfxp>, %arg27: tensor<24x!pphlo.pfxp>, %arg28: tensor<24x20x!pphlo.pfxp>, %arg29: tensor<24x20x!pphlo.pfxp>, %arg30: tensor<20x!pphlo.pfxp>, %arg31: tensor<20x!pphlo.pfxp>, %arg32: tensor<20x24x!pphlo.pfxp>, %arg33: tensor<20x24x!pphlo.pfxp>, %arg34: tensor<24x!pphlo.pfxp>, %arg35: tensor<24x!pphlo.pfxp>, %arg36: tensor<24x1x!pphlo.pfxp>, %arg37: tensor<24x1x!pphlo.pfxp>, %arg38: tensor<1x!pphlo.pfxp>, %arg39: tensor<1x!pphlo.pfxp>) -> (tensor<!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pint>, tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) {
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
    %19 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<15x24xf32>} : () -> tensor<15x24x!pphlo.pfxp>
    %20 = "pphlo.constant"() {value = dense<1.500000e+01> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %21 = "pphlo.add"(%arg15, %20) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %22 = "pphlo.equal"(%21, %3) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pint>
    %23 = "pphlo.rng_uniform"(%3, %2) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %24 = "pphlo.less"(%23, %19) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pint>
    %25 = "pphlo.constant"() {value = dense<1> : tensor<15x24xi32>} : () -> tensor<15x24x!pphlo.pint>
    %26 = "pphlo.subtract"(%25, %24) : (tensor<15x24x!pphlo.pint>, tensor<15x24x!pphlo.pint>) -> tensor<15x24x!pphlo.pint>
    %27 = "pphlo.concatenate"(%arg1, %arg0) {dimension = 1 : i64} : (tensor<15x26x!pphlo.pfxp>, tensor<15x3x!pphlo.pfxp>) -> tensor<15x29x!pphlo.pfxp>
    %28 = "pphlo.dot"(%27, %arg4) : (tensor<15x29x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<15x16x!pphlo.pfxp>
    %29 = "pphlo.broadcast"(%arg5) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<16x!pphlo.pfxp>) -> tensor<15x16x!pphlo.pfxp>
    %30 = "pphlo.add"(%28, %29) : (tensor<15x16x!pphlo.pfxp>, tensor<15x16x!pphlo.pfxp>) -> tensor<15x16x!pphlo.pfxp>
    %31 = "pphlo.maximum"(%30, %18) : (tensor<15x16x!pphlo.pfxp>, tensor<15x16x!pphlo.pfxp>) -> tensor<15x16x!pphlo.pfxp>
    %32 = "pphlo.dot"(%31, %arg6) : (tensor<15x16x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %33 = "pphlo.broadcast"(%arg7) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %34 = "pphlo.add"(%32, %33) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %35 = "pphlo.maximum"(%34, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %36 = "pphlo.multiply"(%35, %12) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %37 = "pphlo.subtract"(%36, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %38 = "pphlo.multiply"(%26, %37) : (tensor<15x24x!pphlo.pint>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %39 = "pphlo.add"(%38, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %40 = "pphlo.dot"(%39, %arg8) : (tensor<15x24x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<15x20x!pphlo.pfxp>
    %41 = "pphlo.broadcast"(%arg9) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<20x!pphlo.pfxp>) -> tensor<15x20x!pphlo.pfxp>
    %42 = "pphlo.add"(%40, %41) : (tensor<15x20x!pphlo.pfxp>, tensor<15x20x!pphlo.pfxp>) -> tensor<15x20x!pphlo.pfxp>
    %43 = "pphlo.maximum"(%42, %17) : (tensor<15x20x!pphlo.pfxp>, tensor<15x20x!pphlo.pfxp>) -> tensor<15x20x!pphlo.pfxp>
    %44 = "pphlo.dot"(%43, %arg10) : (tensor<15x20x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %45 = "pphlo.broadcast"(%arg11) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %46 = "pphlo.add"(%44, %45) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %47 = "pphlo.maximum"(%46, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %48 = "pphlo.reshape"(%arg12) : (tensor<24x1x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %49 = "pphlo.broadcast"(%48) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %50 = "pphlo.multiply"(%47, %49) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %51 = "pphlo.reduce"(%50, %3) ( {
    ^bb0(%arg40: tensor<!pphlo.pfxp>, %arg41: tensor<!pphlo.pfxp>):  // no predecessors
      %331 = "pphlo.add"(%arg40, %arg41) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%331) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<15x24x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<15x!pphlo.pfxp>
    %52 = "pphlo.reshape"(%arg13) : (tensor<1x!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %53 = "pphlo.broadcast"(%52) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<15x!pphlo.pfxp>
    %54 = "pphlo.add"(%51, %53) : (tensor<15x!pphlo.pfxp>, tensor<15x!pphlo.pfxp>) -> tensor<15x!pphlo.pfxp>
    %55 = "pphlo.reshape"(%54) : (tensor<15x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %56 = "pphlo.less"(%55, %15) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pint>
    %57 = "pphlo.constant"() {value = dense<1> : tensor<15x1xi32>} : () -> tensor<15x1x!pphlo.pint>
    %58 = "pphlo.subtract"(%57, %56) : (tensor<15x1x!pphlo.pint>, tensor<15x1x!pphlo.pint>) -> tensor<15x1x!pphlo.pint>
    %59 = "pphlo.subtract"(%55, %15) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %60 = "pphlo.multiply"(%58, %59) : (tensor<15x1x!pphlo.pint>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %61 = "pphlo.add"(%60, %15) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %62 = "pphlo.convert"(%arg2) : (tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pfxp>
    %63 = "pphlo.reshape"(%62) : (tensor<15x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %64 = "pphlo.multiply"(%55, %63) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %65 = "pphlo.subtract"(%61, %64) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %66 = "pphlo.negate"(%55) : (tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %67 = "pphlo.subtract"(%66, %55) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %68 = "pphlo.multiply"(%58, %67) : (tensor<15x1x!pphlo.pint>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %69 = "pphlo.add"(%68, %55) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %70 = "pphlo.exponential"(%69) : (tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %71 = "pphlo.log_plus_one"(%70) : (tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %72 = "pphlo.add"(%65, %71) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %73 = "pphlo.reshape"(%72) : (tensor<15x1x!pphlo.pfxp>) -> tensor<15x!pphlo.pfxp>
    %74 = "pphlo.convert"(%arg3) : (tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pfxp>
    %75 = "pphlo.multiply"(%73, %74) : (tensor<15x!pphlo.pfxp>, tensor<15x!pphlo.pfxp>) -> tensor<15x!pphlo.pfxp>
    %76 = "pphlo.reduce"(%75, %3) ( {
    ^bb0(%arg40: tensor<!pphlo.pfxp>, %arg41: tensor<!pphlo.pfxp>):  // no predecessors
      %331 = "pphlo.add"(%arg40, %arg41) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%331) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %77 = "pphlo.add"(%arg14, %76) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %78 = "pphlo.reciprocal"(%21) : (tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %79 = "pphlo.multiply"(%77, %78) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %80 = "pphlo.subtract"(%3, %79) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %81 = "pphlo.multiply"(%22, %80) : (tensor<!pphlo.pint>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %82 = "pphlo.add"(%81, %79) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %83 = "pphlo.greater"(%31, %18) : (tensor<15x16x!pphlo.pfxp>, tensor<15x16x!pphlo.pfxp>) -> tensor<15x16x!pphlo.pint>
    %84 = "pphlo.greater"(%35, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pint>
    %85 = "pphlo.greater"(%43, %17) : (tensor<15x20x!pphlo.pfxp>, tensor<15x20x!pphlo.pfxp>) -> tensor<15x20x!pphlo.pint>
    %86 = "pphlo.greater"(%47, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pint>
    %87 = "pphlo.multiply"(%74, %16) : (tensor<15x!pphlo.pfxp>, tensor<15x!pphlo.pfxp>) -> tensor<15x!pphlo.pfxp>
    %88 = "pphlo.reshape"(%87) : (tensor<15x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %89 = "pphlo.subtract"(%88, %15) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %90 = "pphlo.multiply"(%58, %89) : (tensor<15x1x!pphlo.pint>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %91 = "pphlo.add"(%90, %15) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %92 = "pphlo.negate"(%88) : (tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %93 = "pphlo.multiply"(%92, %63) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %94 = "pphlo.add"(%91, %93) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %95 = "pphlo.add"(%70, %14) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %96 = "pphlo.reciprocal"(%95) : (tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %97 = "pphlo.multiply"(%96, %14) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %98 = "pphlo.multiply"(%88, %97) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %99 = "pphlo.multiply"(%98, %70) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %100 = "pphlo.subtract"(%15, %99) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %101 = "pphlo.multiply"(%58, %100) : (tensor<15x1x!pphlo.pint>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %102 = "pphlo.add"(%101, %99) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %103 = "pphlo.add"(%94, %102) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %104 = "pphlo.subtract"(%99, %15) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %105 = "pphlo.multiply"(%58, %104) : (tensor<15x1x!pphlo.pint>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %106 = "pphlo.add"(%105, %15) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %107 = "pphlo.negate"(%106) : (tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %108 = "pphlo.add"(%103, %107) : (tensor<15x1x!pphlo.pfxp>, tensor<15x1x!pphlo.pfxp>) -> tensor<15x1x!pphlo.pfxp>
    %109 = "pphlo.reshape"(%108) : (tensor<15x1x!pphlo.pfxp>) -> tensor<15x!pphlo.pfxp>
    %110 = "pphlo.broadcast"(%109) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<15x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %111 = "pphlo.multiply"(%110, %49) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %112 = "pphlo.subtract"(%111, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %113 = "pphlo.multiply"(%86, %112) : (tensor<15x24x!pphlo.pint>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %114 = "pphlo.add"(%113, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %115 = "pphlo.transpose"(%arg10) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<20x24x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %116 = "pphlo.dot"(%114, %115) : (tensor<15x24x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<15x20x!pphlo.pfxp>
    %117 = "pphlo.subtract"(%116, %17) : (tensor<15x20x!pphlo.pfxp>, tensor<15x20x!pphlo.pfxp>) -> tensor<15x20x!pphlo.pfxp>
    %118 = "pphlo.multiply"(%85, %117) : (tensor<15x20x!pphlo.pint>, tensor<15x20x!pphlo.pfxp>) -> tensor<15x20x!pphlo.pfxp>
    %119 = "pphlo.add"(%118, %17) : (tensor<15x20x!pphlo.pfxp>, tensor<15x20x!pphlo.pfxp>) -> tensor<15x20x!pphlo.pfxp>
    %120 = "pphlo.transpose"(%arg8) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<24x20x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %121 = "pphlo.dot"(%119, %120) : (tensor<15x20x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %122 = "pphlo.subtract"(%121, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %123 = "pphlo.multiply"(%26, %122) : (tensor<15x24x!pphlo.pint>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %124 = "pphlo.add"(%123, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %125 = "pphlo.multiply"(%124, %12) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %126 = "pphlo.subtract"(%125, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %127 = "pphlo.multiply"(%84, %126) : (tensor<15x24x!pphlo.pint>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %128 = "pphlo.add"(%127, %13) : (tensor<15x24x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<15x24x!pphlo.pfxp>
    %129 = "pphlo.transpose"(%arg6) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16x24x!pphlo.pfxp>) -> tensor<24x16x!pphlo.pfxp>
    %130 = "pphlo.dot"(%128, %129) : (tensor<15x24x!pphlo.pfxp>, tensor<24x16x!pphlo.pfxp>) -> tensor<15x16x!pphlo.pfxp>
    %131 = "pphlo.subtract"(%130, %18) : (tensor<15x16x!pphlo.pfxp>, tensor<15x16x!pphlo.pfxp>) -> tensor<15x16x!pphlo.pfxp>
    %132 = "pphlo.multiply"(%83, %131) : (tensor<15x16x!pphlo.pint>, tensor<15x16x!pphlo.pfxp>) -> tensor<15x16x!pphlo.pfxp>
    %133 = "pphlo.add"(%132, %18) : (tensor<15x16x!pphlo.pfxp>, tensor<15x16x!pphlo.pfxp>) -> tensor<15x16x!pphlo.pfxp>
    %134 = "pphlo.transpose"(%27) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<15x29x!pphlo.pfxp>) -> tensor<29x15x!pphlo.pfxp>
    %135 = "pphlo.dot"(%134, %133) : (tensor<29x15x!pphlo.pfxp>, tensor<15x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %136 = "pphlo.subtract"(%135, %arg20) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %137 = "pphlo.subtract"(%2, %arg18) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %138 = "pphlo.broadcast"(%137) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %139 = "pphlo.multiply"(%136, %138) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %140 = "pphlo.add"(%arg20, %139) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %141 = "pphlo.add"(%arg17, %0) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %142 = "pphlo.convert"(%141) : (tensor<!pphlo.pint>) -> tensor<!pphlo.pfxp>
    %143 = "pphlo.power"(%arg19, %142) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %144 = "pphlo.subtract"(%2, %143) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %145 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %146 = "pphlo.power"(%144, %145) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %147 = "pphlo.multiply"(%arg16, %146) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %148 = "pphlo.power"(%arg18, %142) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %149 = "pphlo.subtract"(%2, %148) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %150 = "pphlo.reciprocal"(%149) : (tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %151 = "pphlo.multiply"(%147, %150) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %152 = "pphlo.broadcast"(%151) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %153 = "pphlo.multiply"(%140, %152) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %154 = "pphlo.multiply"(%135, %135) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %155 = "pphlo.subtract"(%154, %arg21) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %156 = "pphlo.subtract"(%2, %arg19) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %157 = "pphlo.broadcast"(%156) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %158 = "pphlo.multiply"(%155, %157) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %159 = "pphlo.add"(%arg21, %158) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %160 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<29x16xf32>} : () -> tensor<29x16x!pphlo.pfxp>
    %161 = "pphlo.power"(%159, %160) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %162 = "pphlo.add"(%161, %11) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %163 = "pphlo.reciprocal"(%162) : (tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %164 = "pphlo.multiply"(%153, %163) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %165 = "pphlo.subtract"(%arg4, %164) : (tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<29x16x!pphlo.pfxp>
    %166 = "pphlo.reduce"(%133, %3) ( {
    ^bb0(%arg40: tensor<!pphlo.pfxp>, %arg41: tensor<!pphlo.pfxp>):  // no predecessors
      %331 = "pphlo.add"(%arg40, %arg41) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%331) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x16x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %167 = "pphlo.subtract"(%166, %arg22) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %168 = "pphlo.broadcast"(%137) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %169 = "pphlo.multiply"(%167, %168) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %170 = "pphlo.add"(%arg22, %169) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %171 = "pphlo.broadcast"(%151) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %172 = "pphlo.multiply"(%170, %171) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %173 = "pphlo.multiply"(%166, %166) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %174 = "pphlo.subtract"(%173, %arg23) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %175 = "pphlo.broadcast"(%156) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %176 = "pphlo.multiply"(%174, %175) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %177 = "pphlo.add"(%arg23, %176) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %178 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<16xf32>} : () -> tensor<16x!pphlo.pfxp>
    %179 = "pphlo.power"(%177, %178) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %180 = "pphlo.add"(%179, %10) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %181 = "pphlo.reciprocal"(%180) : (tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %182 = "pphlo.multiply"(%172, %181) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %183 = "pphlo.subtract"(%arg5, %182) : (tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>) -> tensor<16x!pphlo.pfxp>
    %184 = "pphlo.transpose"(%31) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<15x16x!pphlo.pfxp>) -> tensor<16x15x!pphlo.pfxp>
    %185 = "pphlo.dot"(%184, %128) : (tensor<16x15x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %186 = "pphlo.subtract"(%185, %arg24) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %187 = "pphlo.broadcast"(%137) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %188 = "pphlo.multiply"(%186, %187) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %189 = "pphlo.add"(%arg24, %188) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %190 = "pphlo.broadcast"(%151) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %191 = "pphlo.multiply"(%189, %190) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %192 = "pphlo.multiply"(%185, %185) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %193 = "pphlo.subtract"(%192, %arg25) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %194 = "pphlo.broadcast"(%156) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %195 = "pphlo.multiply"(%193, %194) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %196 = "pphlo.add"(%arg25, %195) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %197 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<16x24xf32>} : () -> tensor<16x24x!pphlo.pfxp>
    %198 = "pphlo.power"(%196, %197) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %199 = "pphlo.add"(%198, %9) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %200 = "pphlo.reciprocal"(%199) : (tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %201 = "pphlo.multiply"(%191, %200) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %202 = "pphlo.subtract"(%arg6, %201) : (tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<16x24x!pphlo.pfxp>
    %203 = "pphlo.reduce"(%128, %3) ( {
    ^bb0(%arg40: tensor<!pphlo.pfxp>, %arg41: tensor<!pphlo.pfxp>):  // no predecessors
      %331 = "pphlo.add"(%arg40, %arg41) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%331) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x24x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %204 = "pphlo.subtract"(%203, %arg26) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %205 = "pphlo.broadcast"(%137) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %206 = "pphlo.multiply"(%204, %205) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %207 = "pphlo.add"(%arg26, %206) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %208 = "pphlo.broadcast"(%151) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %209 = "pphlo.multiply"(%207, %208) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %210 = "pphlo.multiply"(%203, %203) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %211 = "pphlo.subtract"(%210, %arg27) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %212 = "pphlo.broadcast"(%156) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %213 = "pphlo.multiply"(%211, %212) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %214 = "pphlo.add"(%arg27, %213) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %215 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<24xf32>} : () -> tensor<24x!pphlo.pfxp>
    %216 = "pphlo.power"(%214, %215) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %217 = "pphlo.add"(%216, %5) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %218 = "pphlo.reciprocal"(%217) : (tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %219 = "pphlo.multiply"(%209, %218) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %220 = "pphlo.subtract"(%arg7, %219) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %221 = "pphlo.transpose"(%39) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<15x24x!pphlo.pfxp>) -> tensor<24x15x!pphlo.pfxp>
    %222 = "pphlo.dot"(%221, %119) : (tensor<24x15x!pphlo.pfxp>, tensor<15x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %223 = "pphlo.subtract"(%222, %arg28) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %224 = "pphlo.broadcast"(%137) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %225 = "pphlo.multiply"(%223, %224) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %226 = "pphlo.add"(%arg28, %225) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %227 = "pphlo.broadcast"(%151) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %228 = "pphlo.multiply"(%226, %227) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %229 = "pphlo.multiply"(%222, %222) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %230 = "pphlo.subtract"(%229, %arg29) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %231 = "pphlo.broadcast"(%156) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %232 = "pphlo.multiply"(%230, %231) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %233 = "pphlo.add"(%arg29, %232) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %234 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<24x20xf32>} : () -> tensor<24x20x!pphlo.pfxp>
    %235 = "pphlo.power"(%233, %234) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %236 = "pphlo.add"(%235, %8) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %237 = "pphlo.reciprocal"(%236) : (tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %238 = "pphlo.multiply"(%228, %237) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %239 = "pphlo.subtract"(%arg8, %238) : (tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<24x20x!pphlo.pfxp>
    %240 = "pphlo.reduce"(%119, %3) ( {
    ^bb0(%arg40: tensor<!pphlo.pfxp>, %arg41: tensor<!pphlo.pfxp>):  // no predecessors
      %331 = "pphlo.add"(%arg40, %arg41) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%331) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x20x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %241 = "pphlo.subtract"(%240, %arg30) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %242 = "pphlo.broadcast"(%137) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %243 = "pphlo.multiply"(%241, %242) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %244 = "pphlo.add"(%arg30, %243) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %245 = "pphlo.broadcast"(%151) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %246 = "pphlo.multiply"(%244, %245) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %247 = "pphlo.multiply"(%240, %240) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %248 = "pphlo.subtract"(%247, %arg31) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %249 = "pphlo.broadcast"(%156) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %250 = "pphlo.multiply"(%248, %249) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %251 = "pphlo.add"(%arg31, %250) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %252 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<20xf32>} : () -> tensor<20x!pphlo.pfxp>
    %253 = "pphlo.power"(%251, %252) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %254 = "pphlo.add"(%253, %7) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %255 = "pphlo.reciprocal"(%254) : (tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %256 = "pphlo.multiply"(%246, %255) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %257 = "pphlo.subtract"(%arg9, %256) : (tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>) -> tensor<20x!pphlo.pfxp>
    %258 = "pphlo.transpose"(%43) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<15x20x!pphlo.pfxp>) -> tensor<20x15x!pphlo.pfxp>
    %259 = "pphlo.dot"(%258, %114) : (tensor<20x15x!pphlo.pfxp>, tensor<15x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %260 = "pphlo.subtract"(%259, %arg32) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %261 = "pphlo.broadcast"(%137) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %262 = "pphlo.multiply"(%260, %261) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %263 = "pphlo.add"(%arg32, %262) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %264 = "pphlo.broadcast"(%151) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %265 = "pphlo.multiply"(%263, %264) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %266 = "pphlo.multiply"(%259, %259) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %267 = "pphlo.subtract"(%266, %arg33) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %268 = "pphlo.broadcast"(%156) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %269 = "pphlo.multiply"(%267, %268) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %270 = "pphlo.add"(%arg33, %269) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %271 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<20x24xf32>} : () -> tensor<20x24x!pphlo.pfxp>
    %272 = "pphlo.power"(%270, %271) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %273 = "pphlo.add"(%272, %6) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %274 = "pphlo.reciprocal"(%273) : (tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %275 = "pphlo.multiply"(%265, %274) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %276 = "pphlo.subtract"(%arg10, %275) : (tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<20x24x!pphlo.pfxp>
    %277 = "pphlo.reduce"(%114, %3) ( {
    ^bb0(%arg40: tensor<!pphlo.pfxp>, %arg41: tensor<!pphlo.pfxp>):  // no predecessors
      %331 = "pphlo.add"(%arg40, %arg41) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%331) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x24x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %278 = "pphlo.subtract"(%277, %arg34) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %279 = "pphlo.multiply"(%278, %205) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %280 = "pphlo.add"(%arg34, %279) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %281 = "pphlo.multiply"(%280, %208) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %282 = "pphlo.multiply"(%277, %277) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %283 = "pphlo.subtract"(%282, %arg35) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %284 = "pphlo.multiply"(%283, %212) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %285 = "pphlo.add"(%arg35, %284) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %286 = "pphlo.power"(%285, %215) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %287 = "pphlo.add"(%286, %5) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %288 = "pphlo.reciprocal"(%287) : (tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %289 = "pphlo.multiply"(%281, %288) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %290 = "pphlo.subtract"(%arg11, %289) : (tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %291 = "pphlo.transpose"(%47) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[24,15]{0,1}"} : (tensor<15x24x!pphlo.pfxp>) -> tensor<24x15x!pphlo.pfxp>
    %292 = "pphlo.broadcast"(%109) {broadcast_dimensions = dense<1> : tensor<1xi64>, xla_shape = "f32[24,15]{0,1}"} : (tensor<15x!pphlo.pfxp>) -> tensor<24x15x!pphlo.pfxp>
    %293 = "pphlo.multiply"(%291, %292) {xla_shape = "f32[24,15]{0,1}"} : (tensor<24x15x!pphlo.pfxp>, tensor<24x15x!pphlo.pfxp>) -> tensor<24x15x!pphlo.pfxp>
    %294 = "pphlo.reduce"(%293, %3) ( {
    ^bb0(%arg40: tensor<!pphlo.pfxp>, %arg41: tensor<!pphlo.pfxp>):  // no predecessors
      %331 = "pphlo.add"(%arg40, %arg41) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%331) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<24x15x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %295 = "pphlo.reshape"(%294) : (tensor<24x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %296 = "pphlo.subtract"(%295, %arg36) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %297 = "pphlo.broadcast"(%137) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %298 = "pphlo.multiply"(%296, %297) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %299 = "pphlo.add"(%arg36, %298) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %300 = "pphlo.broadcast"(%151) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %301 = "pphlo.multiply"(%299, %300) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %302 = "pphlo.multiply"(%295, %295) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %303 = "pphlo.subtract"(%302, %arg37) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %304 = "pphlo.broadcast"(%156) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %305 = "pphlo.multiply"(%303, %304) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %306 = "pphlo.add"(%arg37, %305) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %307 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<24x1xf32>} : () -> tensor<24x1x!pphlo.pfxp>
    %308 = "pphlo.power"(%306, %307) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %309 = "pphlo.add"(%308, %4) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %310 = "pphlo.reciprocal"(%309) : (tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %311 = "pphlo.multiply"(%301, %310) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %312 = "pphlo.subtract"(%arg12, %311) : (tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>) -> tensor<24x1x!pphlo.pfxp>
    %313 = "pphlo.reduce"(%108, %3) ( {
    ^bb0(%arg40: tensor<!pphlo.pfxp>, %arg41: tensor<!pphlo.pfxp>):  // no predecessors
      %331 = "pphlo.add"(%arg40, %arg41) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%331) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x1x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %314 = "pphlo.subtract"(%313, %arg38) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %315 = "pphlo.reshape"(%137) : (tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %316 = "pphlo.multiply"(%314, %315) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %317 = "pphlo.add"(%arg38, %316) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %318 = "pphlo.reshape"(%151) : (tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %319 = "pphlo.multiply"(%317, %318) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %320 = "pphlo.multiply"(%313, %313) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %321 = "pphlo.subtract"(%320, %arg39) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %322 = "pphlo.reshape"(%156) : (tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %323 = "pphlo.multiply"(%321, %322) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %324 = "pphlo.add"(%arg39, %323) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %325 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<1xf32>} : () -> tensor<1x!pphlo.pfxp>
    %326 = "pphlo.power"(%324, %325) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %327 = "pphlo.add"(%326, %1) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %328 = "pphlo.reciprocal"(%327) : (tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %329 = "pphlo.multiply"(%319, %328) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    %330 = "pphlo.subtract"(%arg13, %329) : (tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
    return %82, %165, %183, %202, %220, %239, %257, %276, %290, %312, %330, %77, %21, %141, %140, %159, %170, %177, %189, %196, %207, %214, %226, %233, %244, %251, %263, %270, %280, %285, %299, %306, %317, %324 : tensor<!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pint>, tensor<29x16x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>, tensor<16x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>, tensor<20x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>, tensor<24x1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>, tensor<1x!pphlo.pfxp>
  }
}
