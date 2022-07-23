module @a_inference_test_step_123449__XlaMustCompile_true_config_proto___n_007_n_003CPU_020_001_n_007_n_003GPU_020_0002_002J_0008_001_202_001_000__executor_type____.121  {
  func @main(%arg0: tensor<100x3x!pphlo.pfxp>, %arg1: tensor<100x26x!pphlo.pfxp>, %arg2: tensor<100x!pphlo.pint>, %arg3: tensor<29x16x!pphlo.pfxp>, %arg4: tensor<16x!pphlo.pfxp>, %arg5: tensor<16x24x!pphlo.pfxp>, %arg6: tensor<24x!pphlo.pfxp>, %arg7: tensor<24x20x!pphlo.pfxp>, %arg8: tensor<20x!pphlo.pfxp>, %arg9: tensor<20x24x!pphlo.pfxp>, %arg10: tensor<24x!pphlo.pfxp>, %arg11: tensor<24x1x!pphlo.pfxp>, %arg12: tensor<1x!pphlo.pfxp>, %arg13: tensor<!pphlo.pfxp>, %arg14: tensor<!pphlo.pfxp>) -> (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) {
    %0 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %1 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x1xf32>} : () -> tensor<100x1x!pphlo.pfxp>
    %2 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x24xf32>} : () -> tensor<100x24x!pphlo.pfxp>
    %3 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x20xf32>} : () -> tensor<100x20x!pphlo.pfxp>
    %4 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x16xf32>} : () -> tensor<100x16x!pphlo.pfxp>
    %5 = "pphlo.constant"() {value = dense<1.000000e+02> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %6 = "pphlo.add"(%arg14, %5) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %7 = "pphlo.equal"(%6, %0) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pint>
    %8 = "pphlo.concatenate"(%arg1, %arg0) {dimension = 1 : i64} : (tensor<100x26x!pphlo.pfxp>, tensor<100x3x!pphlo.pfxp>) -> tensor<100x29x!pphlo.pfxp>
    %9 = "pphlo.dot"(%8, %arg3) : (tensor<100x29x!pphlo.pfxp>, tensor<29x16x!pphlo.pfxp>) -> tensor<100x16x!pphlo.pfxp>
    %10 = "pphlo.broadcast"(%arg4) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<16x!pphlo.pfxp>) -> tensor<100x16x!pphlo.pfxp>
    %11 = "pphlo.add"(%9, %10) : (tensor<100x16x!pphlo.pfxp>, tensor<100x16x!pphlo.pfxp>) -> tensor<100x16x!pphlo.pfxp>
    %12 = "pphlo.maximum"(%11, %4) : (tensor<100x16x!pphlo.pfxp>, tensor<100x16x!pphlo.pfxp>) -> tensor<100x16x!pphlo.pfxp>
    %13 = "pphlo.dot"(%12, %arg5) : (tensor<100x16x!pphlo.pfxp>, tensor<16x24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %14 = "pphlo.broadcast"(%arg6) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %15 = "pphlo.add"(%13, %14) : (tensor<100x24x!pphlo.pfxp>, tensor<100x24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %16 = "pphlo.maximum"(%15, %2) : (tensor<100x24x!pphlo.pfxp>, tensor<100x24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %17 = "pphlo.dot"(%16, %arg7) : (tensor<100x24x!pphlo.pfxp>, tensor<24x20x!pphlo.pfxp>) -> tensor<100x20x!pphlo.pfxp>
    %18 = "pphlo.broadcast"(%arg8) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<20x!pphlo.pfxp>) -> tensor<100x20x!pphlo.pfxp>
    %19 = "pphlo.add"(%17, %18) : (tensor<100x20x!pphlo.pfxp>, tensor<100x20x!pphlo.pfxp>) -> tensor<100x20x!pphlo.pfxp>
    %20 = "pphlo.maximum"(%19, %3) : (tensor<100x20x!pphlo.pfxp>, tensor<100x20x!pphlo.pfxp>) -> tensor<100x20x!pphlo.pfxp>
    %21 = "pphlo.dot"(%20, %arg9) : (tensor<100x20x!pphlo.pfxp>, tensor<20x24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %22 = "pphlo.broadcast"(%arg10) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %23 = "pphlo.add"(%21, %22) : (tensor<100x24x!pphlo.pfxp>, tensor<100x24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %24 = "pphlo.maximum"(%23, %2) : (tensor<100x24x!pphlo.pfxp>, tensor<100x24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %25 = "pphlo.reshape"(%arg11) : (tensor<24x1x!pphlo.pfxp>) -> tensor<24x!pphlo.pfxp>
    %26 = "pphlo.broadcast"(%25) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %27 = "pphlo.multiply"(%24, %26) : (tensor<100x24x!pphlo.pfxp>, tensor<100x24x!pphlo.pfxp>) -> tensor<100x24x!pphlo.pfxp>
    %28 = "pphlo.reduce"(%27, %0) ( {
    ^bb0(%arg15: tensor<!pphlo.pfxp>, %arg16: tensor<!pphlo.pfxp>):  // no predecessors
      %57 = "pphlo.add"(%arg15, %arg16) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%57) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<100x24x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<100x!pphlo.pfxp>
    %29 = "pphlo.reshape"(%arg12) : (tensor<1x!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %30 = "pphlo.broadcast"(%29) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pfxp>) -> tensor<100x!pphlo.pfxp>
    %31 = "pphlo.add"(%28, %30) : (tensor<100x!pphlo.pfxp>, tensor<100x!pphlo.pfxp>) -> tensor<100x!pphlo.pfxp>
    %32 = "pphlo.reshape"(%31) : (tensor<100x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %33 = "pphlo.less"(%32, %1) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pint>
    %34 = "pphlo.constant"() {value = dense<1> : tensor<100x1xi32>} : () -> tensor<100x1x!pphlo.pint>
    %35 = "pphlo.subtract"(%34, %33) : (tensor<100x1x!pphlo.pint>, tensor<100x1x!pphlo.pint>) -> tensor<100x1x!pphlo.pint>
    %36 = "pphlo.subtract"(%32, %1) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %37 = "pphlo.multiply"(%35, %36) : (tensor<100x1x!pphlo.pint>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %38 = "pphlo.add"(%37, %1) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %39 = "pphlo.convert"(%arg2) : (tensor<100x!pphlo.pint>) -> tensor<100x!pphlo.pfxp>
    %40 = "pphlo.reshape"(%39) : (tensor<100x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %41 = "pphlo.multiply"(%32, %40) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %42 = "pphlo.subtract"(%38, %41) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %43 = "pphlo.negate"(%32) : (tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %44 = "pphlo.subtract"(%43, %32) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %45 = "pphlo.multiply"(%35, %44) : (tensor<100x1x!pphlo.pint>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %46 = "pphlo.add"(%45, %32) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %47 = "pphlo.exponential"(%46) : (tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %48 = "pphlo.log_plus_one"(%47) : (tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %49 = "pphlo.add"(%42, %48) : (tensor<100x1x!pphlo.pfxp>, tensor<100x1x!pphlo.pfxp>) -> tensor<100x1x!pphlo.pfxp>
    %50 = "pphlo.reduce"(%49, %0) ( {
    ^bb0(%arg15: tensor<!pphlo.pfxp>, %arg16: tensor<!pphlo.pfxp>):  // no predecessors
      %57 = "pphlo.add"(%arg15, %arg16) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%57) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<100x1x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %51 = "pphlo.add"(%arg13, %50) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %52 = "pphlo.reciprocal"(%6) : (tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %53 = "pphlo.multiply"(%51, %52) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %54 = "pphlo.subtract"(%0, %53) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %55 = "pphlo.multiply"(%7, %54) : (tensor<!pphlo.pint>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %56 = "pphlo.add"(%55, %53) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    return %56, %51, %6 : tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>
  }
}
