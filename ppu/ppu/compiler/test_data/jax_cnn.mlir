module @xla_computation_train_and_evaluate.6651  {
  func @main(%arg0: tensor<30x28x28x1x!pphlo.pfxp>, %arg1: tensor<30x!pphlo.pint>, %arg2: tensor<50x28x28x1x!pphlo.pfxp>, %arg3: tensor<50x!pphlo.pint>) -> tensor<!pphlo.pfxp> {
    %0 = "pphlo.constant"() {value = dense<2.000000e-02> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %1 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %2 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<50x10xf32>} : () -> tensor<50x10x!pphlo.pfxp>
    %3 = "pphlo.constant"() {value = dense<0xFF800000> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
    %4 = "pphlo.constant"() {value = dense<-1.000000e-01> : tensor<10xf32>} : () -> tensor<10x!pphlo.pfxp>
    %5 = "pphlo.constant"() {value = dense<-1.000000e-01> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %6 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<50x256xf32>} : () -> tensor<50x256x!pphlo.pfxp>
    %7 = "pphlo.constant"() {value = dense<-1.000000e-01> : tensor<256xf32>} : () -> tensor<256x!pphlo.pfxp>
    %8 = "pphlo.constant"() {value = dense<-1.000000e-01> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %9 = "pphlo.constant"() {value = dense<2.500000e-01> : tensor<50x7x7x64xf32>} : () -> tensor<50x7x7x64x!pphlo.pfxp>
    %10 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<50x14x14x64xf32>} : () -> tensor<50x14x14x64x!pphlo.pfxp>
    %11 = "pphlo.constant"() {value = dense<-1.000000e-01> : tensor<64xf32>} : () -> tensor<64x!pphlo.pfxp>
    %12 = "pphlo.constant"() {value = dense<-1.000000e-01> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %13 = "pphlo.constant"() {value = dense<2.500000e-01> : tensor<50x14x14x32xf32>} : () -> tensor<50x14x14x32x!pphlo.pfxp>
    %14 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<50x28x28x32xf32>} : () -> tensor<50x28x28x32x!pphlo.pfxp>
    %15 = "pphlo.constant"() {value = dense<-1.000000e-01> : tensor<32xf32>} : () -> tensor<32x!pphlo.pfxp>
    %16 = "pphlo.constant"() {value = dense<-1.000000e-01> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %17 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<30x28x28x32xf32>} : () -> tensor<30x28x28x32x!pphlo.pfxp>
    %18 = "pphlo.constant"() {value = dense<2.500000e-01> : tensor<30x14x14x32xf32>} : () -> tensor<30x14x14x32x!pphlo.pfxp>
    %19 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<30x14x14x64xf32>} : () -> tensor<30x14x14x64x!pphlo.pfxp>
    %20 = "pphlo.constant"() {value = dense<2.500000e-01> : tensor<30x3136xf32>} : () -> tensor<30x3136x!pphlo.pfxp>
    %21 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<30x256xf32>} : () -> tensor<30x256x!pphlo.pfxp>
    %22 = "pphlo.constant"() {value = dense<0.0710529536> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %23 = "pphlo.constant"() {value = dense<1.99999988> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %24 = "pphlo.constant"() {value = dense<1.41421354> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %25 = "pphlo.constant"() {value = dense<-3.000000e+00> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %26 = "pphlo.constant"() {value = dense<-2.500000e+00> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %27 = "pphlo.constant"() {value = dense<-2.00214257E-4> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %28 = "pphlo.constant"() {value = dense<2.81022636E-8> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %29 = "pphlo.constant"() {value = dense<1.00950558E-4> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %30 = "pphlo.constant"() {value = dense<3.43273939E-7> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %31 = "pphlo.constant"() {value = dense<0.00134934322> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %32 = "pphlo.constant"() {value = dense<-3.5233877E-6> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %33 = "pphlo.constant"() {value = dense<-0.00367342844> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %34 = "pphlo.constant"() {value = dense<-4.39150654E-6> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %35 = "pphlo.constant"() {value = dense<0.00573950773> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %36 = "pphlo.constant"() {value = dense<2.1858087E-4> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %37 = "pphlo.constant"() {value = dense<-0.0076224613> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %38 = "pphlo.constant"() {value = dense<-0.00125372503> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %39 = "pphlo.constant"() {value = dense<0.00943887047> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %40 = "pphlo.constant"() {value = dense<-0.00417768164> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %41 = "pphlo.constant"() {value = dense<1.00167406> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %42 = "pphlo.constant"() {value = dense<0.246640727> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %43 = "pphlo.constant"() {value = dense<2.83297682> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %44 = "pphlo.constant"() {value = dense<1.50140941> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %45 = "pphlo.constant"() {value = dense<5.000000e+00> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %46 = "pphlo.constant"() {value = dense<0x7F800000> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %47 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %48 = "pphlo.constant"() {value = dense<-0.954499721> : tensor<2560xf32>} : () -> tensor<2560x!pphlo.pfxp>
    %49 = "pphlo.constant"() {value = dense<1.90899944> : tensor<2560xf32>} : () -> tensor<2560x!pphlo.pfxp>
    %50 = "pphlo.constant"() {value = dense<-1.000000e+00> : tensor<2560xf32>} : () -> tensor<2560x!pphlo.pfxp>
    %51 = "pphlo.constant"() {value = dense<1065353216> : tensor<2560xui32>} : () -> tensor<2560x!pphlo.pint>
    %52 = "pphlo.constant"() {value = dense<9> : tensor<2560xui32>} : () -> tensor<2560x!pphlo.pint>
    %53 = "pphlo.constant"() {value = dense<5> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pint>
    %54 = "pphlo.constant"() {value = dense<26> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pint>
    %55 = "pphlo.constant"() {value = dense<6> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pint>
    %56 = "pphlo.constant"() {value = dense<17> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pint>
    %57 = "pphlo.constant"() {value = dense<15> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pint>
    %58 = "pphlo.constant"() {value = dense<19> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pint>
    %59 = "pphlo.constant"() {value = dense<13> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pint>
    %60 = "pphlo.constant"() {value = dense<4> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pint>
    %61 = "pphlo.constant"() {value = dense<8> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pint>
    %62 = "pphlo.constant"() {value = dense<24> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pint>
    %63 = "pphlo.constant"() {value = dense<16> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pint>
    %64 = "pphlo.constant"() {value = dense<3> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pint>
    %65 = "pphlo.constant"() {value = dense<29> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pint>
    %66 = "pphlo.constant"() {value = dense<2> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pint>
    %67 = "pphlo.constant"() {value = dense<1> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pint>
    %68 = "pphlo.constant"() {value = dense<466688986> : tensor<ui32>} : () -> tensor<!pphlo.pint>
    %69 = "pphlo.constant"() {value = dense<5> : tensor<1xui32>} : () -> tensor<1x!pphlo.pint>
    %70 = "pphlo.constant"() {value = dense<26> : tensor<1xui32>} : () -> tensor<1x!pphlo.pint>
    %71 = "pphlo.constant"() {value = dense<6> : tensor<1xui32>} : () -> tensor<1x!pphlo.pint>
    %72 = "pphlo.constant"() {value = dense<17> : tensor<1xui32>} : () -> tensor<1x!pphlo.pint>
    %73 = "pphlo.constant"() {value = dense<15> : tensor<1xui32>} : () -> tensor<1x!pphlo.pint>
    %74 = "pphlo.constant"() {value = dense<19> : tensor<1xui32>} : () -> tensor<1x!pphlo.pint>
    %75 = "pphlo.constant"() {value = dense<13> : tensor<1xui32>} : () -> tensor<1x!pphlo.pint>
    %76 = "pphlo.constant"() {value = dense<4> : tensor<1xui32>} : () -> tensor<1x!pphlo.pint>
    %77 = "pphlo.constant"() {value = dense<8> : tensor<1xui32>} : () -> tensor<1x!pphlo.pint>
    %78 = "pphlo.constant"() {value = dense<24> : tensor<1xui32>} : () -> tensor<1x!pphlo.pint>
    %79 = "pphlo.constant"() {value = dense<16> : tensor<1xui32>} : () -> tensor<1x!pphlo.pint>
    %80 = "pphlo.constant"() {value = dense<3> : tensor<1xui32>} : () -> tensor<1x!pphlo.pint>
    %81 = "pphlo.constant"() {value = dense<29> : tensor<1xui32>} : () -> tensor<1x!pphlo.pint>
    %82 = "pphlo.constant"() {value = dense<2> : tensor<1xui32>} : () -> tensor<1x!pphlo.pint>
    %83 = "pphlo.constant"() {value = dense<1> : tensor<1xui32>} : () -> tensor<1x!pphlo.pint>
    %84 = "pphlo.constant"() {value = dense<3995620053> : tensor<1xui32>} : () -> tensor<1x!pphlo.pint>
    %85 = "pphlo.constant"() {value = dense<-1.99999988> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %86 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<30x10xf32>} : () -> tensor<30x10x!pphlo.pfxp>
    %87 = "pphlo.constant"() {value = dense<-0.0333333351> : tensor<30x10xf32>} : () -> tensor<30x10x!pphlo.pfxp>
    %88 = "pphlo.constant"() {value = dense<30> : tensor<30xi32>} : () -> tensor<30x!pphlo.pint>
    %89 = "pphlo.constant"() {value = dense<0> : tensor<30xi32>} : () -> tensor<30x!pphlo.pint>
    %90 = "pphlo.constant"() {value = dense<0.0203008428> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %91 = "pphlo.constant"() {value = dense<1.99999988> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %92 = "pphlo.constant"() {value = dense<1.41421354> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %93 = "pphlo.constant"() {value = dense<-3.000000e+00> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %94 = "pphlo.constant"() {value = dense<-2.500000e+00> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %95 = "pphlo.constant"() {value = dense<-2.00214257E-4> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %96 = "pphlo.constant"() {value = dense<2.81022636E-8> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %97 = "pphlo.constant"() {value = dense<1.00950558E-4> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %98 = "pphlo.constant"() {value = dense<3.43273939E-7> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %99 = "pphlo.constant"() {value = dense<0.00134934322> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %100 = "pphlo.constant"() {value = dense<-3.5233877E-6> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %101 = "pphlo.constant"() {value = dense<-0.00367342844> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %102 = "pphlo.constant"() {value = dense<-4.39150654E-6> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %103 = "pphlo.constant"() {value = dense<0.00573950773> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %104 = "pphlo.constant"() {value = dense<2.1858087E-4> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %105 = "pphlo.constant"() {value = dense<-0.0076224613> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %106 = "pphlo.constant"() {value = dense<-0.00125372503> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %107 = "pphlo.constant"() {value = dense<0.00943887047> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %108 = "pphlo.constant"() {value = dense<-0.00417768164> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %109 = "pphlo.constant"() {value = dense<1.00167406> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %110 = "pphlo.constant"() {value = dense<0.246640727> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %111 = "pphlo.constant"() {value = dense<2.83297682> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %112 = "pphlo.constant"() {value = dense<1.50140941> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %113 = "pphlo.constant"() {value = dense<5.000000e+00> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %114 = "pphlo.constant"() {value = dense<0x7F800000> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %115 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %116 = "pphlo.constant"() {value = dense<-0.954499721> : tensor<802816xf32>} : () -> tensor<802816x!pphlo.pfxp>
    %117 = "pphlo.constant"() {value = dense<1.90899944> : tensor<802816xf32>} : () -> tensor<802816x!pphlo.pfxp>
    %118 = "pphlo.constant"() {value = dense<-1.000000e+00> : tensor<802816xf32>} : () -> tensor<802816x!pphlo.pfxp>
    %119 = "pphlo.constant"() {value = dense<1065353216> : tensor<802816xui32>} : () -> tensor<802816x!pphlo.pint>
    %120 = "pphlo.constant"() {value = dense<9> : tensor<802816xui32>} : () -> tensor<802816x!pphlo.pint>
    %121 = "pphlo.constant"() {value = dense<5> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pint>
    %122 = "pphlo.constant"() {value = dense<26> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pint>
    %123 = "pphlo.constant"() {value = dense<6> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pint>
    %124 = "pphlo.constant"() {value = dense<17> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pint>
    %125 = "pphlo.constant"() {value = dense<15> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pint>
    %126 = "pphlo.constant"() {value = dense<19> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pint>
    %127 = "pphlo.constant"() {value = dense<13> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pint>
    %128 = "pphlo.constant"() {value = dense<4> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pint>
    %129 = "pphlo.constant"() {value = dense<8> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pint>
    %130 = "pphlo.constant"() {value = dense<24> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pint>
    %131 = "pphlo.constant"() {value = dense<16> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pint>
    %132 = "pphlo.constant"() {value = dense<3> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pint>
    %133 = "pphlo.constant"() {value = dense<29> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pint>
    %134 = "pphlo.constant"() {value = dense<2> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pint>
    %135 = "pphlo.constant"() {value = dense<1> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pint>
    %136 = "pphlo.constant"() {value = dense<706584679> : tensor<1xui32>} : () -> tensor<1x!pphlo.pint>
    %137 = "pphlo.constant"() {value = dense<-1.99999988> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %138 = "pphlo.constant"() {value = dense<2.500000e-01> : tensor<30x7x7x64xf32>} : () -> tensor<30x7x7x64x!pphlo.pfxp>
    %139 = "pphlo.constant"() {value = dense<0.0669893697> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %140 = "pphlo.constant"() {value = dense<1.99999988> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %141 = "pphlo.constant"() {value = dense<1.41421354> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %142 = "pphlo.constant"() {value = dense<-3.000000e+00> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %143 = "pphlo.constant"() {value = dense<-2.500000e+00> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %144 = "pphlo.constant"() {value = dense<-2.00214257E-4> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %145 = "pphlo.constant"() {value = dense<2.81022636E-8> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %146 = "pphlo.constant"() {value = dense<1.00950558E-4> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %147 = "pphlo.constant"() {value = dense<3.43273939E-7> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %148 = "pphlo.constant"() {value = dense<0.00134934322> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %149 = "pphlo.constant"() {value = dense<-3.5233877E-6> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %150 = "pphlo.constant"() {value = dense<-0.00367342844> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %151 = "pphlo.constant"() {value = dense<-4.39150654E-6> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %152 = "pphlo.constant"() {value = dense<0.00573950773> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %153 = "pphlo.constant"() {value = dense<2.1858087E-4> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %154 = "pphlo.constant"() {value = dense<-0.0076224613> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %155 = "pphlo.constant"() {value = dense<-0.00125372503> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %156 = "pphlo.constant"() {value = dense<0.00943887047> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %157 = "pphlo.constant"() {value = dense<-0.00417768164> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %158 = "pphlo.constant"() {value = dense<1.00167406> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %159 = "pphlo.constant"() {value = dense<0.246640727> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %160 = "pphlo.constant"() {value = dense<2.83297682> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %161 = "pphlo.constant"() {value = dense<1.50140941> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %162 = "pphlo.constant"() {value = dense<5.000000e+00> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %163 = "pphlo.constant"() {value = dense<0x7F800000> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %164 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %165 = "pphlo.constant"() {value = dense<-0.954499721> : tensor<18432xf32>} : () -> tensor<18432x!pphlo.pfxp>
    %166 = "pphlo.constant"() {value = dense<1.90899944> : tensor<18432xf32>} : () -> tensor<18432x!pphlo.pfxp>
    %167 = "pphlo.constant"() {value = dense<-1.000000e+00> : tensor<18432xf32>} : () -> tensor<18432x!pphlo.pfxp>
    %168 = "pphlo.constant"() {value = dense<1065353216> : tensor<18432xui32>} : () -> tensor<18432x!pphlo.pint>
    %169 = "pphlo.constant"() {value = dense<9> : tensor<18432xui32>} : () -> tensor<18432x!pphlo.pint>
    %170 = "pphlo.constant"() {value = dense<5> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pint>
    %171 = "pphlo.constant"() {value = dense<26> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pint>
    %172 = "pphlo.constant"() {value = dense<6> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pint>
    %173 = "pphlo.constant"() {value = dense<17> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pint>
    %174 = "pphlo.constant"() {value = dense<15> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pint>
    %175 = "pphlo.constant"() {value = dense<19> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pint>
    %176 = "pphlo.constant"() {value = dense<13> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pint>
    %177 = "pphlo.constant"() {value = dense<4> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pint>
    %178 = "pphlo.constant"() {value = dense<8> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pint>
    %179 = "pphlo.constant"() {value = dense<24> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pint>
    %180 = "pphlo.constant"() {value = dense<16> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pint>
    %181 = "pphlo.constant"() {value = dense<3> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pint>
    %182 = "pphlo.constant"() {value = dense<29> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pint>
    %183 = "pphlo.constant"() {value = dense<2> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pint>
    %184 = "pphlo.constant"() {value = dense<1> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pint>
    %185 = "pphlo.constant"() {value = dense<2095399837> : tensor<1xui32>} : () -> tensor<1x!pphlo.pint>
    %186 = "pphlo.constant"() {value = dense<-1.99999988> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %187 = "pphlo.constant"() {value = dense<5> : tensor<15xui32>} : () -> tensor<15x!pphlo.pint>
    %188 = "pphlo.constant"() {value = dense<26> : tensor<15xui32>} : () -> tensor<15x!pphlo.pint>
    %189 = "pphlo.constant"() {value = dense<6> : tensor<15xui32>} : () -> tensor<15x!pphlo.pint>
    %190 = "pphlo.constant"() {value = dense<17> : tensor<15xui32>} : () -> tensor<15x!pphlo.pint>
    %191 = "pphlo.constant"() {value = dense<15> : tensor<15xui32>} : () -> tensor<15x!pphlo.pint>
    %192 = "pphlo.constant"() {value = dense<19> : tensor<15xui32>} : () -> tensor<15x!pphlo.pint>
    %193 = "pphlo.constant"() {value = dense<13> : tensor<15xui32>} : () -> tensor<15x!pphlo.pint>
    %194 = "pphlo.constant"() {value = dense<4> : tensor<15xui32>} : () -> tensor<15x!pphlo.pint>
    %195 = "pphlo.constant"() {value = dense<8> : tensor<15xui32>} : () -> tensor<15x!pphlo.pint>
    %196 = "pphlo.constant"() {value = dense<24> : tensor<15xui32>} : () -> tensor<15x!pphlo.pint>
    %197 = "pphlo.constant"() {value = dense<16> : tensor<15xui32>} : () -> tensor<15x!pphlo.pint>
    %198 = "pphlo.constant"() {value = dense<3> : tensor<15xui32>} : () -> tensor<15x!pphlo.pint>
    %199 = "pphlo.constant"() {value = dense<29> : tensor<15xui32>} : () -> tensor<15x!pphlo.pint>
    %200 = "pphlo.constant"() {value = dense<2> : tensor<15xui32>} : () -> tensor<15x!pphlo.pint>
    %201 = "pphlo.constant"() {value = dense<1> : tensor<15xui32>} : () -> tensor<15x!pphlo.pint>
    %202 = "pphlo.constant"() {value = dense<5> : tensor<2xui32>} : () -> tensor<2x!pphlo.pint>
    %203 = "pphlo.constant"() {value = dense<26> : tensor<2xui32>} : () -> tensor<2x!pphlo.pint>
    %204 = "pphlo.constant"() {value = dense<6> : tensor<2xui32>} : () -> tensor<2x!pphlo.pint>
    %205 = "pphlo.constant"() {value = dense<17> : tensor<2xui32>} : () -> tensor<2x!pphlo.pint>
    %206 = "pphlo.constant"() {value = dense<15> : tensor<2xui32>} : () -> tensor<2x!pphlo.pint>
    %207 = "pphlo.constant"() {value = dense<19> : tensor<2xui32>} : () -> tensor<2x!pphlo.pint>
    %208 = "pphlo.constant"() {value = dense<13> : tensor<2xui32>} : () -> tensor<2x!pphlo.pint>
    %209 = "pphlo.constant"() {value = dense<4> : tensor<2xui32>} : () -> tensor<2x!pphlo.pint>
    %210 = "pphlo.constant"() {value = dense<8> : tensor<2xui32>} : () -> tensor<2x!pphlo.pint>
    %211 = "pphlo.constant"() {value = dense<24> : tensor<2xui32>} : () -> tensor<2x!pphlo.pint>
    %212 = "pphlo.constant"() {value = dense<16> : tensor<2xui32>} : () -> tensor<2x!pphlo.pint>
    %213 = "pphlo.constant"() {value = dense<3> : tensor<2xui32>} : () -> tensor<2x!pphlo.pint>
    %214 = "pphlo.constant"() {value = dense<29> : tensor<2xui32>} : () -> tensor<2x!pphlo.pint>
    %215 = "pphlo.constant"() {value = dense<2> : tensor<2xui32>} : () -> tensor<2x!pphlo.pint>
    %216 = "pphlo.constant"() {value = dense<1> : tensor<2xui32>} : () -> tensor<2x!pphlo.pint>
    %217 = "pphlo.constant"() {value = dense<0.378949106> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %218 = "pphlo.constant"() {value = dense<1.99999988> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %219 = "pphlo.constant"() {value = dense<1.41421354> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %220 = "pphlo.constant"() {value = dense<-3.000000e+00> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %221 = "pphlo.constant"() {value = dense<-2.500000e+00> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %222 = "pphlo.constant"() {value = dense<-2.00214257E-4> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %223 = "pphlo.constant"() {value = dense<2.81022636E-8> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %224 = "pphlo.constant"() {value = dense<1.00950558E-4> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %225 = "pphlo.constant"() {value = dense<3.43273939E-7> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %226 = "pphlo.constant"() {value = dense<0.00134934322> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %227 = "pphlo.constant"() {value = dense<-3.5233877E-6> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %228 = "pphlo.constant"() {value = dense<-0.00367342844> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %229 = "pphlo.constant"() {value = dense<-4.39150654E-6> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %230 = "pphlo.constant"() {value = dense<0.00573950773> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %231 = "pphlo.constant"() {value = dense<2.1858087E-4> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %232 = "pphlo.constant"() {value = dense<-0.0076224613> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %233 = "pphlo.constant"() {value = dense<-0.00125372503> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %234 = "pphlo.constant"() {value = dense<0.00943887047> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %235 = "pphlo.constant"() {value = dense<-0.00417768164> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %236 = "pphlo.constant"() {value = dense<1.00167406> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %237 = "pphlo.constant"() {value = dense<0.246640727> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %238 = "pphlo.constant"() {value = dense<2.83297682> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %239 = "pphlo.constant"() {value = dense<1.50140941> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %240 = "pphlo.constant"() {value = dense<5.000000e+00> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %241 = "pphlo.constant"() {value = dense<0x7F800000> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %242 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %243 = "pphlo.constant"() {value = dense<-0.954499721> : tensor<288xf32>} : () -> tensor<288x!pphlo.pfxp>
    %244 = "pphlo.constant"() {value = dense<1.90899944> : tensor<288xf32>} : () -> tensor<288x!pphlo.pfxp>
    %245 = "pphlo.constant"() {value = dense<-1.000000e+00> : tensor<288xf32>} : () -> tensor<288x!pphlo.pfxp>
    %246 = "pphlo.constant"() {value = dense<1065353216> : tensor<288xui32>} : () -> tensor<288x!pphlo.pint>
    %247 = "pphlo.constant"() {value = dense<9> : tensor<288xui32>} : () -> tensor<288x!pphlo.pint>
    %248 = "pphlo.constant"() {value = dense<5> : tensor<144xui32>} : () -> tensor<144x!pphlo.pint>
    %249 = "pphlo.constant"() {value = dense<26> : tensor<144xui32>} : () -> tensor<144x!pphlo.pint>
    %250 = "pphlo.constant"() {value = dense<6> : tensor<144xui32>} : () -> tensor<144x!pphlo.pint>
    %251 = "pphlo.constant"() {value = dense<17> : tensor<144xui32>} : () -> tensor<144x!pphlo.pint>
    %252 = "pphlo.constant"() {value = dense<15> : tensor<144xui32>} : () -> tensor<144x!pphlo.pint>
    %253 = "pphlo.constant"() {value = dense<19> : tensor<144xui32>} : () -> tensor<144x!pphlo.pint>
    %254 = "pphlo.constant"() {value = dense<13> : tensor<144xui32>} : () -> tensor<144x!pphlo.pint>
    %255 = "pphlo.constant"() {value = dense<4> : tensor<144xui32>} : () -> tensor<144x!pphlo.pint>
    %256 = "pphlo.constant"() {value = dense<8> : tensor<144xui32>} : () -> tensor<144x!pphlo.pint>
    %257 = "pphlo.constant"() {value = dense<24> : tensor<144xui32>} : () -> tensor<144x!pphlo.pint>
    %258 = "pphlo.constant"() {value = dense<16> : tensor<144xui32>} : () -> tensor<144x!pphlo.pint>
    %259 = "pphlo.constant"() {value = dense<3> : tensor<144xui32>} : () -> tensor<144x!pphlo.pint>
    %260 = "pphlo.constant"() {value = dense<29> : tensor<144xui32>} : () -> tensor<144x!pphlo.pint>
    %261 = "pphlo.constant"() {value = dense<2> : tensor<144xui32>} : () -> tensor<144x!pphlo.pint>
    %262 = "pphlo.constant"() {value = dense<1> : tensor<144xui32>} : () -> tensor<144x!pphlo.pint>
    %263 = "pphlo.constant"() {value = dense<3798891600> : tensor<1xui32>} : () -> tensor<1x!pphlo.pint>
    %264 = "pphlo.constant"() {value = dense<466688986> : tensor<2xui32>} : () -> tensor<2x!pphlo.pint>
    %265 = "pphlo.constant"() {value = dense<466688990> : tensor<2xui32>} : () -> tensor<2x!pphlo.pint>
    %266 = "pphlo.constant"() {value = dense<466688987> : tensor<2xui32>} : () -> tensor<2x!pphlo.pint>
    %267 = "pphlo.constant"() {value = dense<-1.99999988> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %268 = "pphlo.broadcast"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<50x!pphlo.pint>) -> tensor<50x10x!pphlo.pint>
    %269 = "pphlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<10x!pphlo.pint>
    %270 = "pphlo.broadcast"(%269) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<10x!pphlo.pint>) -> tensor<50x10x!pphlo.pint>
    %271 = "pphlo.equal"(%268, %270) : (tensor<50x10x!pphlo.pint>, tensor<50x10x!pphlo.pint>) -> tensor<50x10x!pphlo.pint>
    %272 = "pphlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<288x!pphlo.pint>
    %273 = "pphlo.slice"(%272) {limit_indices = dense<144> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<288x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %274 = "pphlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4x!pphlo.pint>
    %275 = "pphlo.slice"(%274) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %276 = "pphlo.slice"(%274) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %277 = "pphlo.add"(%275, %276) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %278 = "pphlo.shift_left"(%276, %208) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %279 = "pphlo.shift_right_logical"(%276, %207) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %280 = "pphlo.or"(%278, %279) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %281 = "pphlo.xor"(%277, %280) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %282 = "pphlo.add"(%277, %281) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %283 = "pphlo.shift_left"(%281, %206) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %284 = "pphlo.shift_right_logical"(%281, %205) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %285 = "pphlo.or"(%283, %284) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %286 = "pphlo.xor"(%282, %285) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %287 = "pphlo.add"(%282, %286) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %288 = "pphlo.shift_left"(%286, %203) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %289 = "pphlo.shift_right_logical"(%286, %204) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %290 = "pphlo.or"(%288, %289) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %291 = "pphlo.xor"(%287, %290) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %292 = "pphlo.add"(%287, %291) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %293 = "pphlo.shift_left"(%291, %204) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %294 = "pphlo.shift_right_logical"(%291, %203) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %295 = "pphlo.or"(%293, %294) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %296 = "pphlo.xor"(%292, %295) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %297 = "pphlo.add"(%296, %266) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %298 = "pphlo.add"(%292, %297) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %299 = "pphlo.shift_left"(%297, %205) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %300 = "pphlo.shift_right_logical"(%297, %206) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %301 = "pphlo.or"(%299, %300) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %302 = "pphlo.xor"(%298, %301) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %303 = "pphlo.add"(%298, %302) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %304 = "pphlo.shift_left"(%302, %214) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %305 = "pphlo.shift_right_logical"(%302, %213) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %306 = "pphlo.or"(%304, %305) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %307 = "pphlo.xor"(%303, %306) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %308 = "pphlo.add"(%303, %307) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %309 = "pphlo.shift_left"(%307, %212) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %310 = "pphlo.shift_right_logical"(%307, %212) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %311 = "pphlo.or"(%309, %310) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %312 = "pphlo.xor"(%308, %311) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %313 = "pphlo.add"(%308, %312) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %314 = "pphlo.add"(%313, %264) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %315 = "pphlo.shift_left"(%312, %211) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %316 = "pphlo.shift_right_logical"(%312, %210) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %317 = "pphlo.or"(%315, %316) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %318 = "pphlo.xor"(%313, %317) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %319 = "pphlo.add"(%318, %215) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %320 = "pphlo.add"(%314, %319) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %321 = "pphlo.shift_left"(%319, %208) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %322 = "pphlo.shift_right_logical"(%319, %207) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %323 = "pphlo.or"(%321, %322) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %324 = "pphlo.xor"(%320, %323) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %325 = "pphlo.add"(%320, %324) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %326 = "pphlo.shift_left"(%324, %206) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %327 = "pphlo.shift_right_logical"(%324, %205) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %328 = "pphlo.or"(%326, %327) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %329 = "pphlo.xor"(%325, %328) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %330 = "pphlo.add"(%325, %329) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %331 = "pphlo.shift_left"(%329, %203) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %332 = "pphlo.shift_right_logical"(%329, %204) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %333 = "pphlo.or"(%331, %332) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %334 = "pphlo.xor"(%330, %333) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %335 = "pphlo.add"(%330, %334) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %336 = "pphlo.shift_left"(%334, %204) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %337 = "pphlo.shift_right_logical"(%334, %203) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %338 = "pphlo.or"(%336, %337) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %339 = "pphlo.xor"(%335, %338) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %340 = "pphlo.add"(%339, %213) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %341 = "pphlo.add"(%335, %340) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %342 = "pphlo.shift_left"(%340, %205) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %343 = "pphlo.shift_right_logical"(%340, %206) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %344 = "pphlo.or"(%342, %343) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %345 = "pphlo.xor"(%341, %344) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %346 = "pphlo.add"(%341, %345) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %347 = "pphlo.shift_left"(%345, %214) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %348 = "pphlo.shift_right_logical"(%345, %213) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %349 = "pphlo.or"(%347, %348) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %350 = "pphlo.xor"(%346, %349) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %351 = "pphlo.add"(%346, %350) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %352 = "pphlo.shift_left"(%350, %212) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %353 = "pphlo.shift_right_logical"(%350, %212) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %354 = "pphlo.or"(%352, %353) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %355 = "pphlo.xor"(%351, %354) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %356 = "pphlo.add"(%351, %355) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %357 = "pphlo.shift_left"(%355, %211) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %358 = "pphlo.shift_right_logical"(%355, %210) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %359 = "pphlo.or"(%357, %358) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %360 = "pphlo.xor"(%356, %359) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %361 = "pphlo.add"(%360, %265) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %362 = "pphlo.add"(%356, %361) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %363 = "pphlo.shift_left"(%361, %208) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %364 = "pphlo.shift_right_logical"(%361, %207) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %365 = "pphlo.or"(%363, %364) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %366 = "pphlo.xor"(%362, %365) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %367 = "pphlo.add"(%362, %366) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %368 = "pphlo.shift_left"(%366, %206) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %369 = "pphlo.shift_right_logical"(%366, %205) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %370 = "pphlo.or"(%368, %369) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %371 = "pphlo.xor"(%367, %370) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %372 = "pphlo.add"(%367, %371) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %373 = "pphlo.shift_left"(%371, %203) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %374 = "pphlo.shift_right_logical"(%371, %204) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %375 = "pphlo.or"(%373, %374) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %376 = "pphlo.xor"(%372, %375) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %377 = "pphlo.add"(%372, %376) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %378 = "pphlo.add"(%377, %264) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %379 = "pphlo.shift_left"(%376, %204) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %380 = "pphlo.shift_right_logical"(%376, %203) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %381 = "pphlo.or"(%379, %380) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %382 = "pphlo.xor"(%377, %381) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %383 = "pphlo.add"(%382, %202) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %384 = "pphlo.concatenate"(%378, %383) {dimension = 0 : i64} : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<4x!pphlo.pint>
    %385 = "pphlo.reshape"(%384) : (tensor<4x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %386 = "pphlo.slice"(%385) {limit_indices = dense<[2, 1]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2x!pphlo.pint>) -> tensor<1x1x!pphlo.pint>
    %387 = "pphlo.reshape"(%386) : (tensor<1x1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %388 = "pphlo.slice"(%385) {limit_indices = dense<2> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2x!pphlo.pint>) -> tensor<1x2x!pphlo.pint>
    %389 = "pphlo.reshape"(%388) : (tensor<1x2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %390 = "pphlo.slice"(%389) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %391 = "pphlo.add"(%390, %263) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %392 = "pphlo.add"(%387, %391) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %393 = "pphlo.shift_left"(%391, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %394 = "pphlo.shift_right_logical"(%391, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %395 = "pphlo.or"(%393, %394) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %396 = "pphlo.xor"(%392, %395) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %397 = "pphlo.add"(%392, %396) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %398 = "pphlo.shift_left"(%396, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %399 = "pphlo.shift_right_logical"(%396, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %400 = "pphlo.or"(%398, %399) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %401 = "pphlo.xor"(%397, %400) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %402 = "pphlo.add"(%397, %401) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %403 = "pphlo.shift_left"(%401, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %404 = "pphlo.shift_right_logical"(%401, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %405 = "pphlo.or"(%403, %404) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %406 = "pphlo.xor"(%402, %405) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %407 = "pphlo.add"(%402, %406) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %408 = "pphlo.add"(%407, %390) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %409 = "pphlo.shift_left"(%406, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %410 = "pphlo.shift_right_logical"(%406, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %411 = "pphlo.or"(%409, %410) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %412 = "pphlo.xor"(%407, %411) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %413 = "pphlo.reshape"(%386) : (tensor<1x1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %414 = "pphlo.reshape"(%390) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %415 = "pphlo.xor"(%413, %414) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %416 = "pphlo.xor"(%415, %68) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %417 = "pphlo.reshape"(%416) : (tensor<!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %418 = "pphlo.add"(%412, %417) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %419 = "pphlo.add"(%418, %83) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %420 = "pphlo.add"(%408, %419) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %421 = "pphlo.shift_left"(%419, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %422 = "pphlo.shift_right_logical"(%419, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %423 = "pphlo.or"(%421, %422) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %424 = "pphlo.xor"(%420, %423) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %425 = "pphlo.add"(%420, %424) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %426 = "pphlo.shift_left"(%424, %81) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %427 = "pphlo.shift_right_logical"(%424, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %428 = "pphlo.or"(%426, %427) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %429 = "pphlo.xor"(%425, %428) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %430 = "pphlo.add"(%425, %429) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %431 = "pphlo.shift_left"(%429, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %432 = "pphlo.shift_right_logical"(%429, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %433 = "pphlo.or"(%431, %432) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %434 = "pphlo.xor"(%430, %433) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %435 = "pphlo.add"(%430, %434) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %436 = "pphlo.add"(%435, %417) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %437 = "pphlo.shift_left"(%434, %78) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %438 = "pphlo.shift_right_logical"(%434, %77) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %439 = "pphlo.or"(%437, %438) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %440 = "pphlo.xor"(%435, %439) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %441 = "pphlo.add"(%440, %387) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %442 = "pphlo.add"(%441, %82) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %443 = "pphlo.add"(%436, %442) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %444 = "pphlo.shift_left"(%442, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %445 = "pphlo.shift_right_logical"(%442, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %446 = "pphlo.or"(%444, %445) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %447 = "pphlo.xor"(%443, %446) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %448 = "pphlo.add"(%443, %447) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %449 = "pphlo.shift_left"(%447, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %450 = "pphlo.shift_right_logical"(%447, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %451 = "pphlo.or"(%449, %450) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %452 = "pphlo.xor"(%448, %451) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %453 = "pphlo.add"(%448, %452) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %454 = "pphlo.shift_left"(%452, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %455 = "pphlo.shift_right_logical"(%452, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %456 = "pphlo.or"(%454, %455) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %457 = "pphlo.xor"(%453, %456) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %458 = "pphlo.add"(%453, %457) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %459 = "pphlo.add"(%458, %387) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %460 = "pphlo.shift_left"(%457, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %461 = "pphlo.shift_right_logical"(%457, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %462 = "pphlo.or"(%460, %461) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %463 = "pphlo.xor"(%458, %462) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %464 = "pphlo.add"(%463, %390) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %465 = "pphlo.add"(%464, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %466 = "pphlo.add"(%459, %465) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %467 = "pphlo.shift_left"(%465, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %468 = "pphlo.shift_right_logical"(%465, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %469 = "pphlo.or"(%467, %468) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %470 = "pphlo.xor"(%466, %469) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %471 = "pphlo.add"(%466, %470) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %472 = "pphlo.shift_left"(%470, %81) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %473 = "pphlo.shift_right_logical"(%470, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %474 = "pphlo.or"(%472, %473) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %475 = "pphlo.xor"(%471, %474) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %476 = "pphlo.add"(%471, %475) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %477 = "pphlo.shift_left"(%475, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %478 = "pphlo.shift_right_logical"(%475, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %479 = "pphlo.or"(%477, %478) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %480 = "pphlo.xor"(%476, %479) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %481 = "pphlo.add"(%476, %480) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %482 = "pphlo.add"(%481, %390) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %483 = "pphlo.shift_left"(%480, %78) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %484 = "pphlo.shift_right_logical"(%480, %77) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %485 = "pphlo.or"(%483, %484) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %486 = "pphlo.xor"(%481, %485) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %487 = "pphlo.add"(%486, %417) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %488 = "pphlo.add"(%487, %76) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %489 = "pphlo.add"(%482, %488) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %490 = "pphlo.shift_left"(%488, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %491 = "pphlo.shift_right_logical"(%488, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %492 = "pphlo.or"(%490, %491) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %493 = "pphlo.xor"(%489, %492) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %494 = "pphlo.add"(%489, %493) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %495 = "pphlo.shift_left"(%493, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %496 = "pphlo.shift_right_logical"(%493, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %497 = "pphlo.or"(%495, %496) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %498 = "pphlo.xor"(%494, %497) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %499 = "pphlo.add"(%494, %498) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %500 = "pphlo.shift_left"(%498, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %501 = "pphlo.shift_right_logical"(%498, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %502 = "pphlo.or"(%500, %501) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %503 = "pphlo.xor"(%499, %502) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %504 = "pphlo.add"(%499, %503) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %505 = "pphlo.add"(%504, %417) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %506 = "pphlo.shift_left"(%503, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %507 = "pphlo.shift_right_logical"(%503, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %508 = "pphlo.or"(%506, %507) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %509 = "pphlo.xor"(%504, %508) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %510 = "pphlo.add"(%509, %387) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %511 = "pphlo.add"(%510, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %512 = "pphlo.add"(%505, %511) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %513 = "pphlo.shift_left"(%511, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %514 = "pphlo.shift_right_logical"(%511, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %515 = "pphlo.or"(%513, %514) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %516 = "pphlo.xor"(%512, %515) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %517 = "pphlo.add"(%512, %516) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %518 = "pphlo.shift_left"(%516, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %519 = "pphlo.shift_right_logical"(%516, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %520 = "pphlo.or"(%518, %519) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %521 = "pphlo.xor"(%517, %520) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %522 = "pphlo.add"(%517, %521) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %523 = "pphlo.shift_left"(%521, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %524 = "pphlo.shift_right_logical"(%521, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %525 = "pphlo.or"(%523, %524) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %526 = "pphlo.xor"(%522, %525) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %527 = "pphlo.add"(%522, %526) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %528 = "pphlo.add"(%510, %69) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %529 = "pphlo.add"(%527, %528) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %530 = "pphlo.shift_left"(%526, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %531 = "pphlo.shift_right_logical"(%526, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %532 = "pphlo.or"(%530, %531) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %533 = "pphlo.xor"(%527, %532) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %534 = "pphlo.reshape"(%505) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %535 = "pphlo.reshape"(%528) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %536 = "pphlo.xor"(%534, %535) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %537 = "pphlo.xor"(%536, %68) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %538 = "pphlo.reshape"(%537) : (tensor<!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %539 = "pphlo.add"(%533, %538) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %540 = "pphlo.add"(%539, %83) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %541 = "pphlo.add"(%529, %540) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %542 = "pphlo.shift_left"(%540, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %543 = "pphlo.shift_right_logical"(%540, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %544 = "pphlo.or"(%542, %543) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %545 = "pphlo.xor"(%541, %544) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %546 = "pphlo.add"(%541, %545) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %547 = "pphlo.shift_left"(%545, %81) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %548 = "pphlo.shift_right_logical"(%545, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %549 = "pphlo.or"(%547, %548) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %550 = "pphlo.xor"(%546, %549) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %551 = "pphlo.add"(%546, %550) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %552 = "pphlo.shift_left"(%550, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %553 = "pphlo.shift_right_logical"(%550, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %554 = "pphlo.or"(%552, %553) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %555 = "pphlo.xor"(%551, %554) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %556 = "pphlo.add"(%551, %555) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %557 = "pphlo.add"(%556, %538) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %558 = "pphlo.shift_left"(%555, %78) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %559 = "pphlo.shift_right_logical"(%555, %77) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %560 = "pphlo.or"(%558, %559) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %561 = "pphlo.xor"(%556, %560) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %562 = "pphlo.add"(%561, %505) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %563 = "pphlo.add"(%562, %82) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %564 = "pphlo.add"(%557, %563) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %565 = "pphlo.shift_left"(%563, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %566 = "pphlo.shift_right_logical"(%563, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %567 = "pphlo.or"(%565, %566) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %568 = "pphlo.xor"(%564, %567) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %569 = "pphlo.add"(%564, %568) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %570 = "pphlo.shift_left"(%568, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %571 = "pphlo.shift_right_logical"(%568, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %572 = "pphlo.or"(%570, %571) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %573 = "pphlo.xor"(%569, %572) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %574 = "pphlo.add"(%569, %573) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %575 = "pphlo.shift_left"(%573, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %576 = "pphlo.shift_right_logical"(%573, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %577 = "pphlo.or"(%575, %576) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %578 = "pphlo.xor"(%574, %577) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %579 = "pphlo.add"(%574, %578) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %580 = "pphlo.add"(%579, %505) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %581 = "pphlo.shift_left"(%578, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %582 = "pphlo.shift_right_logical"(%578, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %583 = "pphlo.or"(%581, %582) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %584 = "pphlo.xor"(%579, %583) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %585 = "pphlo.add"(%584, %528) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %586 = "pphlo.add"(%585, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %587 = "pphlo.add"(%580, %586) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %588 = "pphlo.shift_left"(%586, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %589 = "pphlo.shift_right_logical"(%586, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %590 = "pphlo.or"(%588, %589) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %591 = "pphlo.xor"(%587, %590) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %592 = "pphlo.add"(%587, %591) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %593 = "pphlo.shift_left"(%591, %81) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %594 = "pphlo.shift_right_logical"(%591, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %595 = "pphlo.or"(%593, %594) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %596 = "pphlo.xor"(%592, %595) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %597 = "pphlo.add"(%592, %596) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %598 = "pphlo.shift_left"(%596, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %599 = "pphlo.shift_right_logical"(%596, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %600 = "pphlo.or"(%598, %599) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %601 = "pphlo.xor"(%597, %600) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %602 = "pphlo.add"(%597, %601) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %603 = "pphlo.add"(%602, %528) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %604 = "pphlo.shift_left"(%601, %78) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %605 = "pphlo.shift_right_logical"(%601, %77) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %606 = "pphlo.or"(%604, %605) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %607 = "pphlo.xor"(%602, %606) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %608 = "pphlo.add"(%607, %538) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %609 = "pphlo.add"(%608, %76) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %610 = "pphlo.add"(%603, %609) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %611 = "pphlo.shift_left"(%609, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %612 = "pphlo.shift_right_logical"(%609, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %613 = "pphlo.or"(%611, %612) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %614 = "pphlo.xor"(%610, %613) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %615 = "pphlo.add"(%610, %614) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %616 = "pphlo.shift_left"(%614, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %617 = "pphlo.shift_right_logical"(%614, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %618 = "pphlo.or"(%616, %617) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %619 = "pphlo.xor"(%615, %618) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %620 = "pphlo.add"(%615, %619) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %621 = "pphlo.shift_left"(%619, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %622 = "pphlo.shift_right_logical"(%619, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %623 = "pphlo.or"(%621, %622) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %624 = "pphlo.xor"(%620, %623) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %625 = "pphlo.add"(%620, %624) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %626 = "pphlo.add"(%625, %538) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %627 = "pphlo.reshape"(%626) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %628 = "pphlo.broadcast"(%627) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %629 = "pphlo.add"(%273, %628) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %630 = "pphlo.slice"(%272) {limit_indices = dense<288> : tensor<1xi64>, start_indices = dense<144> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<288x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %631 = "pphlo.shift_left"(%624, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %632 = "pphlo.shift_right_logical"(%624, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %633 = "pphlo.or"(%631, %632) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %634 = "pphlo.xor"(%625, %633) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %635 = "pphlo.add"(%634, %505) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %636 = "pphlo.add"(%635, %69) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %637 = "pphlo.reshape"(%636) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %638 = "pphlo.broadcast"(%637) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %639 = "pphlo.add"(%630, %638) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %640 = "pphlo.add"(%629, %639) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %641 = "pphlo.shift_left"(%639, %254) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %642 = "pphlo.shift_right_logical"(%639, %253) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %643 = "pphlo.or"(%641, %642) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %644 = "pphlo.xor"(%640, %643) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %645 = "pphlo.add"(%640, %644) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %646 = "pphlo.shift_left"(%644, %252) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %647 = "pphlo.shift_right_logical"(%644, %251) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %648 = "pphlo.or"(%646, %647) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %649 = "pphlo.xor"(%645, %648) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %650 = "pphlo.add"(%645, %649) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %651 = "pphlo.shift_left"(%649, %249) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %652 = "pphlo.shift_right_logical"(%649, %250) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %653 = "pphlo.or"(%651, %652) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %654 = "pphlo.xor"(%650, %653) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %655 = "pphlo.add"(%650, %654) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %656 = "pphlo.add"(%655, %638) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %657 = "pphlo.shift_left"(%654, %250) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %658 = "pphlo.shift_right_logical"(%654, %249) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %659 = "pphlo.or"(%657, %658) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %660 = "pphlo.xor"(%655, %659) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %661 = "pphlo.xor"(%627, %637) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %662 = "pphlo.xor"(%661, %68) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %663 = "pphlo.broadcast"(%662) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %664 = "pphlo.add"(%660, %663) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %665 = "pphlo.add"(%664, %262) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %666 = "pphlo.add"(%656, %665) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %667 = "pphlo.shift_left"(%665, %251) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %668 = "pphlo.shift_right_logical"(%665, %252) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %669 = "pphlo.or"(%667, %668) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %670 = "pphlo.xor"(%666, %669) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %671 = "pphlo.add"(%666, %670) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %672 = "pphlo.shift_left"(%670, %260) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %673 = "pphlo.shift_right_logical"(%670, %259) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %674 = "pphlo.or"(%672, %673) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %675 = "pphlo.xor"(%671, %674) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %676 = "pphlo.add"(%671, %675) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %677 = "pphlo.shift_left"(%675, %258) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %678 = "pphlo.shift_right_logical"(%675, %258) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %679 = "pphlo.or"(%677, %678) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %680 = "pphlo.xor"(%676, %679) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %681 = "pphlo.add"(%676, %680) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %682 = "pphlo.add"(%681, %663) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %683 = "pphlo.shift_left"(%680, %257) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %684 = "pphlo.shift_right_logical"(%680, %256) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %685 = "pphlo.or"(%683, %684) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %686 = "pphlo.xor"(%681, %685) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %687 = "pphlo.add"(%686, %628) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %688 = "pphlo.add"(%687, %261) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %689 = "pphlo.add"(%682, %688) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %690 = "pphlo.shift_left"(%688, %254) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %691 = "pphlo.shift_right_logical"(%688, %253) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %692 = "pphlo.or"(%690, %691) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %693 = "pphlo.xor"(%689, %692) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %694 = "pphlo.add"(%689, %693) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %695 = "pphlo.shift_left"(%693, %252) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %696 = "pphlo.shift_right_logical"(%693, %251) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %697 = "pphlo.or"(%695, %696) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %698 = "pphlo.xor"(%694, %697) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %699 = "pphlo.add"(%694, %698) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %700 = "pphlo.shift_left"(%698, %249) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %701 = "pphlo.shift_right_logical"(%698, %250) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %702 = "pphlo.or"(%700, %701) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %703 = "pphlo.xor"(%699, %702) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %704 = "pphlo.add"(%699, %703) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %705 = "pphlo.add"(%704, %628) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %706 = "pphlo.shift_left"(%703, %250) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %707 = "pphlo.shift_right_logical"(%703, %249) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %708 = "pphlo.or"(%706, %707) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %709 = "pphlo.xor"(%704, %708) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %710 = "pphlo.add"(%709, %638) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %711 = "pphlo.add"(%710, %259) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %712 = "pphlo.add"(%705, %711) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %713 = "pphlo.shift_left"(%711, %251) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %714 = "pphlo.shift_right_logical"(%711, %252) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %715 = "pphlo.or"(%713, %714) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %716 = "pphlo.xor"(%712, %715) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %717 = "pphlo.add"(%712, %716) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %718 = "pphlo.shift_left"(%716, %260) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %719 = "pphlo.shift_right_logical"(%716, %259) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %720 = "pphlo.or"(%718, %719) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %721 = "pphlo.xor"(%717, %720) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %722 = "pphlo.add"(%717, %721) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %723 = "pphlo.shift_left"(%721, %258) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %724 = "pphlo.shift_right_logical"(%721, %258) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %725 = "pphlo.or"(%723, %724) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %726 = "pphlo.xor"(%722, %725) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %727 = "pphlo.add"(%722, %726) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %728 = "pphlo.add"(%727, %638) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %729 = "pphlo.shift_left"(%726, %257) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %730 = "pphlo.shift_right_logical"(%726, %256) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %731 = "pphlo.or"(%729, %730) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %732 = "pphlo.xor"(%727, %731) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %733 = "pphlo.add"(%732, %663) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %734 = "pphlo.add"(%733, %255) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %735 = "pphlo.add"(%728, %734) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %736 = "pphlo.shift_left"(%734, %254) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %737 = "pphlo.shift_right_logical"(%734, %253) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %738 = "pphlo.or"(%736, %737) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %739 = "pphlo.xor"(%735, %738) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %740 = "pphlo.add"(%735, %739) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %741 = "pphlo.shift_left"(%739, %252) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %742 = "pphlo.shift_right_logical"(%739, %251) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %743 = "pphlo.or"(%741, %742) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %744 = "pphlo.xor"(%740, %743) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %745 = "pphlo.add"(%740, %744) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %746 = "pphlo.shift_left"(%744, %249) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %747 = "pphlo.shift_right_logical"(%744, %250) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %748 = "pphlo.or"(%746, %747) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %749 = "pphlo.xor"(%745, %748) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %750 = "pphlo.add"(%745, %749) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %751 = "pphlo.add"(%750, %663) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %752 = "pphlo.shift_left"(%749, %250) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %753 = "pphlo.shift_right_logical"(%749, %249) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %754 = "pphlo.or"(%752, %753) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %755 = "pphlo.xor"(%750, %754) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %756 = "pphlo.add"(%755, %628) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %757 = "pphlo.add"(%756, %248) : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<144x!pphlo.pint>
    %758 = "pphlo.concatenate"(%751, %757) {dimension = 0 : i64} : (tensor<144x!pphlo.pint>, tensor<144x!pphlo.pint>) -> tensor<288x!pphlo.pint>
    %759 = "pphlo.shift_right_logical"(%758, %247) : (tensor<288x!pphlo.pint>, tensor<288x!pphlo.pint>) -> tensor<288x!pphlo.pint>
    %760 = "pphlo.or"(%759, %246) : (tensor<288x!pphlo.pint>, tensor<288x!pphlo.pint>) -> tensor<288x!pphlo.pint>
    %761 = "pphlo.bitcast_convert"(%760) {elsize = 32 : i64} : (tensor<288x!pphlo.pint>) -> tensor<288x!pphlo.pfxp>
    %762 = "pphlo.add"(%761, %245) : (tensor<288x!pphlo.pfxp>, tensor<288x!pphlo.pfxp>) -> tensor<288x!pphlo.pfxp>
    %763 = "pphlo.multiply"(%762, %244) : (tensor<288x!pphlo.pfxp>, tensor<288x!pphlo.pfxp>) -> tensor<288x!pphlo.pfxp>
    %764 = "pphlo.add"(%763, %243) : (tensor<288x!pphlo.pfxp>, tensor<288x!pphlo.pfxp>) -> tensor<288x!pphlo.pfxp>
    %765 = "pphlo.maximum"(%764, %243) : (tensor<288x!pphlo.pfxp>, tensor<288x!pphlo.pfxp>) -> tensor<288x!pphlo.pfxp>
    %766 = "pphlo.reshape"(%765) : (tensor<288x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %767 = "pphlo.abs"(%766) : (tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %768 = "pphlo.equal"(%767, %242) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pint>
    %769 = "pphlo.multiply"(%766, %241) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %770 = "pphlo.negate"(%766) : (tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %771 = "pphlo.multiply"(%770, %766) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %772 = "pphlo.log_plus_one"(%771) : (tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %773 = "pphlo.negate"(%772) : (tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %774 = "pphlo.less"(%773, %240) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pint>
    %775 = "pphlo.subtract"(%239, %238) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %776 = "pphlo.multiply"(%774, %775) : (tensor<3x3x1x32x!pphlo.pint>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %777 = "pphlo.add"(%776, %238) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %778 = "pphlo.subtract"(%237, %236) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %779 = "pphlo.multiply"(%774, %778) : (tensor<3x3x1x32x!pphlo.pint>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %780 = "pphlo.add"(%779, %236) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %781 = "pphlo.subtract"(%235, %234) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %782 = "pphlo.multiply"(%774, %781) : (tensor<3x3x1x32x!pphlo.pint>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %783 = "pphlo.add"(%782, %234) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %784 = "pphlo.subtract"(%233, %232) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %785 = "pphlo.multiply"(%774, %784) : (tensor<3x3x1x32x!pphlo.pint>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %786 = "pphlo.add"(%785, %232) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %787 = "pphlo.subtract"(%231, %230) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %788 = "pphlo.multiply"(%774, %787) : (tensor<3x3x1x32x!pphlo.pint>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %789 = "pphlo.add"(%788, %230) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %790 = "pphlo.subtract"(%229, %228) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %791 = "pphlo.multiply"(%774, %790) : (tensor<3x3x1x32x!pphlo.pint>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %792 = "pphlo.add"(%791, %228) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %793 = "pphlo.subtract"(%227, %226) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %794 = "pphlo.multiply"(%774, %793) : (tensor<3x3x1x32x!pphlo.pint>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %795 = "pphlo.add"(%794, %226) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %796 = "pphlo.subtract"(%225, %224) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %797 = "pphlo.multiply"(%774, %796) : (tensor<3x3x1x32x!pphlo.pint>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %798 = "pphlo.add"(%797, %224) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %799 = "pphlo.subtract"(%223, %222) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %800 = "pphlo.multiply"(%774, %799) : (tensor<3x3x1x32x!pphlo.pint>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %801 = "pphlo.add"(%800, %222) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %802 = "pphlo.add"(%773, %221) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %803 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pfxp>
    %804 = "pphlo.power"(%773, %803) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %805 = "pphlo.add"(%804, %220) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %806 = "pphlo.subtract"(%802, %805) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %807 = "pphlo.multiply"(%774, %806) : (tensor<3x3x1x32x!pphlo.pint>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %808 = "pphlo.add"(%807, %805) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %809 = "pphlo.multiply"(%801, %808) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %810 = "pphlo.add"(%798, %809) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %811 = "pphlo.multiply"(%810, %808) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %812 = "pphlo.add"(%795, %811) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %813 = "pphlo.multiply"(%812, %808) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %814 = "pphlo.add"(%792, %813) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %815 = "pphlo.multiply"(%814, %808) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %816 = "pphlo.add"(%789, %815) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %817 = "pphlo.multiply"(%816, %808) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %818 = "pphlo.add"(%786, %817) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %819 = "pphlo.multiply"(%818, %808) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %820 = "pphlo.add"(%783, %819) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %821 = "pphlo.multiply"(%820, %808) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %822 = "pphlo.add"(%780, %821) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %823 = "pphlo.multiply"(%822, %808) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %824 = "pphlo.add"(%777, %823) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %825 = "pphlo.multiply"(%824, %766) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %826 = "pphlo.subtract"(%769, %825) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %827 = "pphlo.multiply"(%768, %826) : (tensor<3x3x1x32x!pphlo.pint>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %828 = "pphlo.add"(%827, %825) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %829 = "pphlo.multiply"(%828, %219) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %830 = "pphlo.clamp"(%267, %829, %218) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %831 = "pphlo.multiply"(%830, %217) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %832 = "pphlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<30x!pphlo.pint>
    %833 = "pphlo.slice"(%832) {limit_indices = dense<15> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<30x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %834 = "pphlo.slice"(%378) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %835 = "pphlo.reshape"(%834) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %836 = "pphlo.broadcast"(%835) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %837 = "pphlo.add"(%275, %836) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %838 = "pphlo.slice"(%378) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %839 = "pphlo.reshape"(%838) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %840 = "pphlo.broadcast"(%839) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %841 = "pphlo.add"(%276, %840) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %842 = "pphlo.add"(%837, %841) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %843 = "pphlo.shift_left"(%841, %208) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %844 = "pphlo.shift_right_logical"(%841, %207) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %845 = "pphlo.or"(%843, %844) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %846 = "pphlo.xor"(%842, %845) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %847 = "pphlo.add"(%842, %846) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %848 = "pphlo.shift_left"(%846, %206) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %849 = "pphlo.shift_right_logical"(%846, %205) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %850 = "pphlo.or"(%848, %849) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %851 = "pphlo.xor"(%847, %850) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %852 = "pphlo.add"(%847, %851) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %853 = "pphlo.shift_left"(%851, %203) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %854 = "pphlo.shift_right_logical"(%851, %204) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %855 = "pphlo.or"(%853, %854) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %856 = "pphlo.xor"(%852, %855) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %857 = "pphlo.add"(%852, %856) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %858 = "pphlo.add"(%857, %840) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %859 = "pphlo.shift_left"(%856, %204) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %860 = "pphlo.shift_right_logical"(%856, %203) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %861 = "pphlo.or"(%859, %860) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %862 = "pphlo.xor"(%857, %861) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %863 = "pphlo.xor"(%835, %839) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %864 = "pphlo.xor"(%863, %68) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %865 = "pphlo.broadcast"(%864) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %866 = "pphlo.add"(%862, %865) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %867 = "pphlo.add"(%866, %216) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %868 = "pphlo.add"(%858, %867) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %869 = "pphlo.shift_left"(%867, %205) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %870 = "pphlo.shift_right_logical"(%867, %206) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %871 = "pphlo.or"(%869, %870) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %872 = "pphlo.xor"(%868, %871) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %873 = "pphlo.add"(%868, %872) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %874 = "pphlo.shift_left"(%872, %214) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %875 = "pphlo.shift_right_logical"(%872, %213) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %876 = "pphlo.or"(%874, %875) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %877 = "pphlo.xor"(%873, %876) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %878 = "pphlo.add"(%873, %877) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %879 = "pphlo.shift_left"(%877, %212) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %880 = "pphlo.shift_right_logical"(%877, %212) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %881 = "pphlo.or"(%879, %880) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %882 = "pphlo.xor"(%878, %881) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %883 = "pphlo.add"(%878, %882) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %884 = "pphlo.add"(%883, %865) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %885 = "pphlo.shift_left"(%882, %211) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %886 = "pphlo.shift_right_logical"(%882, %210) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %887 = "pphlo.or"(%885, %886) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %888 = "pphlo.xor"(%883, %887) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %889 = "pphlo.add"(%888, %836) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %890 = "pphlo.add"(%889, %215) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %891 = "pphlo.add"(%884, %890) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %892 = "pphlo.shift_left"(%890, %208) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %893 = "pphlo.shift_right_logical"(%890, %207) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %894 = "pphlo.or"(%892, %893) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %895 = "pphlo.xor"(%891, %894) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %896 = "pphlo.add"(%891, %895) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %897 = "pphlo.shift_left"(%895, %206) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %898 = "pphlo.shift_right_logical"(%895, %205) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %899 = "pphlo.or"(%897, %898) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %900 = "pphlo.xor"(%896, %899) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %901 = "pphlo.add"(%896, %900) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %902 = "pphlo.shift_left"(%900, %203) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %903 = "pphlo.shift_right_logical"(%900, %204) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %904 = "pphlo.or"(%902, %903) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %905 = "pphlo.xor"(%901, %904) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %906 = "pphlo.add"(%901, %905) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %907 = "pphlo.add"(%906, %836) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %908 = "pphlo.shift_left"(%905, %204) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %909 = "pphlo.shift_right_logical"(%905, %203) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %910 = "pphlo.or"(%908, %909) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %911 = "pphlo.xor"(%906, %910) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %912 = "pphlo.add"(%911, %840) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %913 = "pphlo.add"(%912, %213) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %914 = "pphlo.add"(%907, %913) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %915 = "pphlo.shift_left"(%913, %205) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %916 = "pphlo.shift_right_logical"(%913, %206) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %917 = "pphlo.or"(%915, %916) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %918 = "pphlo.xor"(%914, %917) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %919 = "pphlo.add"(%914, %918) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %920 = "pphlo.shift_left"(%918, %214) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %921 = "pphlo.shift_right_logical"(%918, %213) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %922 = "pphlo.or"(%920, %921) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %923 = "pphlo.xor"(%919, %922) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %924 = "pphlo.add"(%919, %923) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %925 = "pphlo.shift_left"(%923, %212) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %926 = "pphlo.shift_right_logical"(%923, %212) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %927 = "pphlo.or"(%925, %926) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %928 = "pphlo.xor"(%924, %927) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %929 = "pphlo.add"(%924, %928) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %930 = "pphlo.add"(%929, %840) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %931 = "pphlo.shift_left"(%928, %211) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %932 = "pphlo.shift_right_logical"(%928, %210) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %933 = "pphlo.or"(%931, %932) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %934 = "pphlo.xor"(%929, %933) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %935 = "pphlo.add"(%934, %865) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %936 = "pphlo.add"(%935, %209) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %937 = "pphlo.add"(%930, %936) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %938 = "pphlo.shift_left"(%936, %208) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %939 = "pphlo.shift_right_logical"(%936, %207) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %940 = "pphlo.or"(%938, %939) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %941 = "pphlo.xor"(%937, %940) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %942 = "pphlo.add"(%937, %941) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %943 = "pphlo.shift_left"(%941, %206) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %944 = "pphlo.shift_right_logical"(%941, %205) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %945 = "pphlo.or"(%943, %944) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %946 = "pphlo.xor"(%942, %945) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %947 = "pphlo.add"(%942, %946) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %948 = "pphlo.shift_left"(%946, %203) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %949 = "pphlo.shift_right_logical"(%946, %204) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %950 = "pphlo.or"(%948, %949) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %951 = "pphlo.xor"(%947, %950) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %952 = "pphlo.add"(%947, %951) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %953 = "pphlo.add"(%952, %865) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %954 = "pphlo.shift_left"(%951, %204) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %955 = "pphlo.shift_right_logical"(%951, %203) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %956 = "pphlo.or"(%954, %955) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %957 = "pphlo.xor"(%952, %956) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %958 = "pphlo.add"(%957, %836) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %959 = "pphlo.add"(%958, %202) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %960 = "pphlo.concatenate"(%953, %959) {dimension = 0 : i64} : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<4x!pphlo.pint>
    %961 = "pphlo.reshape"(%960) : (tensor<4x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %962 = "pphlo.slice"(%961) {limit_indices = dense<[2, 1]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2x!pphlo.pint>) -> tensor<1x1x!pphlo.pint>
    %963 = "pphlo.reshape"(%962) : (tensor<1x1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %964 = "pphlo.broadcast"(%963) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %965 = "pphlo.add"(%275, %964) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %966 = "pphlo.slice"(%961) {limit_indices = dense<2> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2x!pphlo.pint>) -> tensor<1x2x!pphlo.pint>
    %967 = "pphlo.reshape"(%966) : (tensor<1x2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %968 = "pphlo.slice"(%967) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %969 = "pphlo.reshape"(%968) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %970 = "pphlo.broadcast"(%969) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %971 = "pphlo.add"(%276, %970) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %972 = "pphlo.add"(%965, %971) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %973 = "pphlo.shift_left"(%971, %208) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %974 = "pphlo.shift_right_logical"(%971, %207) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %975 = "pphlo.or"(%973, %974) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %976 = "pphlo.xor"(%972, %975) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %977 = "pphlo.add"(%972, %976) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %978 = "pphlo.shift_left"(%976, %206) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %979 = "pphlo.shift_right_logical"(%976, %205) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %980 = "pphlo.or"(%978, %979) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %981 = "pphlo.xor"(%977, %980) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %982 = "pphlo.add"(%977, %981) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %983 = "pphlo.shift_left"(%981, %203) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %984 = "pphlo.shift_right_logical"(%981, %204) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %985 = "pphlo.or"(%983, %984) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %986 = "pphlo.xor"(%982, %985) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %987 = "pphlo.add"(%982, %986) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %988 = "pphlo.add"(%987, %970) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %989 = "pphlo.shift_left"(%986, %204) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %990 = "pphlo.shift_right_logical"(%986, %203) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %991 = "pphlo.or"(%989, %990) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %992 = "pphlo.xor"(%987, %991) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %993 = "pphlo.xor"(%963, %969) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %994 = "pphlo.xor"(%993, %68) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %995 = "pphlo.broadcast"(%994) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %996 = "pphlo.add"(%992, %995) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %997 = "pphlo.add"(%996, %216) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %998 = "pphlo.add"(%988, %997) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %999 = "pphlo.shift_left"(%997, %205) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1000 = "pphlo.shift_right_logical"(%997, %206) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1001 = "pphlo.or"(%999, %1000) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1002 = "pphlo.xor"(%998, %1001) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1003 = "pphlo.add"(%998, %1002) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1004 = "pphlo.shift_left"(%1002, %214) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1005 = "pphlo.shift_right_logical"(%1002, %213) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1006 = "pphlo.or"(%1004, %1005) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1007 = "pphlo.xor"(%1003, %1006) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1008 = "pphlo.add"(%1003, %1007) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1009 = "pphlo.shift_left"(%1007, %212) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1010 = "pphlo.shift_right_logical"(%1007, %212) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1011 = "pphlo.or"(%1009, %1010) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1012 = "pphlo.xor"(%1008, %1011) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1013 = "pphlo.add"(%1008, %1012) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1014 = "pphlo.add"(%1013, %995) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1015 = "pphlo.shift_left"(%1012, %211) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1016 = "pphlo.shift_right_logical"(%1012, %210) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1017 = "pphlo.or"(%1015, %1016) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1018 = "pphlo.xor"(%1013, %1017) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1019 = "pphlo.add"(%1018, %964) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1020 = "pphlo.add"(%1019, %215) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1021 = "pphlo.add"(%1014, %1020) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1022 = "pphlo.shift_left"(%1020, %208) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1023 = "pphlo.shift_right_logical"(%1020, %207) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1024 = "pphlo.or"(%1022, %1023) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1025 = "pphlo.xor"(%1021, %1024) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1026 = "pphlo.add"(%1021, %1025) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1027 = "pphlo.shift_left"(%1025, %206) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1028 = "pphlo.shift_right_logical"(%1025, %205) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1029 = "pphlo.or"(%1027, %1028) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1030 = "pphlo.xor"(%1026, %1029) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1031 = "pphlo.add"(%1026, %1030) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1032 = "pphlo.shift_left"(%1030, %203) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1033 = "pphlo.shift_right_logical"(%1030, %204) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1034 = "pphlo.or"(%1032, %1033) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1035 = "pphlo.xor"(%1031, %1034) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1036 = "pphlo.add"(%1031, %1035) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1037 = "pphlo.add"(%1036, %964) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1038 = "pphlo.shift_left"(%1035, %204) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1039 = "pphlo.shift_right_logical"(%1035, %203) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1040 = "pphlo.or"(%1038, %1039) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1041 = "pphlo.xor"(%1036, %1040) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1042 = "pphlo.add"(%1041, %970) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1043 = "pphlo.add"(%1042, %213) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1044 = "pphlo.add"(%1037, %1043) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1045 = "pphlo.shift_left"(%1043, %205) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1046 = "pphlo.shift_right_logical"(%1043, %206) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1047 = "pphlo.or"(%1045, %1046) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1048 = "pphlo.xor"(%1044, %1047) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1049 = "pphlo.add"(%1044, %1048) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1050 = "pphlo.shift_left"(%1048, %214) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1051 = "pphlo.shift_right_logical"(%1048, %213) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1052 = "pphlo.or"(%1050, %1051) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1053 = "pphlo.xor"(%1049, %1052) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1054 = "pphlo.add"(%1049, %1053) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1055 = "pphlo.shift_left"(%1053, %212) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1056 = "pphlo.shift_right_logical"(%1053, %212) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1057 = "pphlo.or"(%1055, %1056) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1058 = "pphlo.xor"(%1054, %1057) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1059 = "pphlo.add"(%1054, %1058) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1060 = "pphlo.add"(%1059, %970) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1061 = "pphlo.shift_left"(%1058, %211) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1062 = "pphlo.shift_right_logical"(%1058, %210) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1063 = "pphlo.or"(%1061, %1062) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1064 = "pphlo.xor"(%1059, %1063) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1065 = "pphlo.add"(%1064, %995) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1066 = "pphlo.add"(%1065, %209) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1067 = "pphlo.add"(%1060, %1066) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1068 = "pphlo.shift_left"(%1066, %208) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1069 = "pphlo.shift_right_logical"(%1066, %207) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1070 = "pphlo.or"(%1068, %1069) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1071 = "pphlo.xor"(%1067, %1070) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1072 = "pphlo.add"(%1067, %1071) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1073 = "pphlo.shift_left"(%1071, %206) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1074 = "pphlo.shift_right_logical"(%1071, %205) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1075 = "pphlo.or"(%1073, %1074) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1076 = "pphlo.xor"(%1072, %1075) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1077 = "pphlo.add"(%1072, %1076) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1078 = "pphlo.shift_left"(%1076, %203) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1079 = "pphlo.shift_right_logical"(%1076, %204) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1080 = "pphlo.or"(%1078, %1079) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1081 = "pphlo.xor"(%1077, %1080) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1082 = "pphlo.add"(%1077, %1081) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1083 = "pphlo.add"(%1082, %995) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1084 = "pphlo.shift_left"(%1081, %204) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1085 = "pphlo.shift_right_logical"(%1081, %203) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1086 = "pphlo.or"(%1084, %1085) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1087 = "pphlo.xor"(%1082, %1086) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1088 = "pphlo.add"(%1087, %964) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1089 = "pphlo.add"(%1088, %202) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1090 = "pphlo.concatenate"(%1083, %1089) {dimension = 0 : i64} : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<4x!pphlo.pint>
    %1091 = "pphlo.reshape"(%1090) : (tensor<4x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %1092 = "pphlo.slice"(%1091) {limit_indices = dense<[2, 1]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2x!pphlo.pint>) -> tensor<1x1x!pphlo.pint>
    %1093 = "pphlo.reshape"(%1092) : (tensor<1x1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %1094 = "pphlo.broadcast"(%1093) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1095 = "pphlo.add"(%833, %1094) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1096 = "pphlo.slice"(%832) {limit_indices = dense<30> : tensor<1xi64>, start_indices = dense<15> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<30x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1097 = "pphlo.slice"(%1091) {limit_indices = dense<2> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2x!pphlo.pint>) -> tensor<1x2x!pphlo.pint>
    %1098 = "pphlo.reshape"(%1097) : (tensor<1x2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    %1099 = "pphlo.slice"(%1098) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1100 = "pphlo.reshape"(%1099) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %1101 = "pphlo.broadcast"(%1100) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1102 = "pphlo.add"(%1096, %1101) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1103 = "pphlo.add"(%1095, %1102) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1104 = "pphlo.shift_left"(%1102, %193) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1105 = "pphlo.shift_right_logical"(%1102, %192) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1106 = "pphlo.or"(%1104, %1105) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1107 = "pphlo.xor"(%1103, %1106) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1108 = "pphlo.add"(%1103, %1107) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1109 = "pphlo.shift_left"(%1107, %191) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1110 = "pphlo.shift_right_logical"(%1107, %190) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1111 = "pphlo.or"(%1109, %1110) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1112 = "pphlo.xor"(%1108, %1111) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1113 = "pphlo.add"(%1108, %1112) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1114 = "pphlo.shift_left"(%1112, %188) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1115 = "pphlo.shift_right_logical"(%1112, %189) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1116 = "pphlo.or"(%1114, %1115) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1117 = "pphlo.xor"(%1113, %1116) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1118 = "pphlo.add"(%1113, %1117) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1119 = "pphlo.add"(%1118, %1101) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1120 = "pphlo.shift_left"(%1117, %189) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1121 = "pphlo.shift_right_logical"(%1117, %188) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1122 = "pphlo.or"(%1120, %1121) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1123 = "pphlo.xor"(%1118, %1122) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1124 = "pphlo.xor"(%1093, %1100) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %1125 = "pphlo.xor"(%1124, %68) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %1126 = "pphlo.broadcast"(%1125) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1127 = "pphlo.add"(%1123, %1126) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1128 = "pphlo.add"(%1127, %201) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1129 = "pphlo.add"(%1119, %1128) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1130 = "pphlo.shift_left"(%1128, %190) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1131 = "pphlo.shift_right_logical"(%1128, %191) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1132 = "pphlo.or"(%1130, %1131) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1133 = "pphlo.xor"(%1129, %1132) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1134 = "pphlo.add"(%1129, %1133) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1135 = "pphlo.shift_left"(%1133, %199) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1136 = "pphlo.shift_right_logical"(%1133, %198) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1137 = "pphlo.or"(%1135, %1136) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1138 = "pphlo.xor"(%1134, %1137) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1139 = "pphlo.add"(%1134, %1138) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1140 = "pphlo.shift_left"(%1138, %197) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1141 = "pphlo.shift_right_logical"(%1138, %197) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1142 = "pphlo.or"(%1140, %1141) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1143 = "pphlo.xor"(%1139, %1142) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1144 = "pphlo.add"(%1139, %1143) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1145 = "pphlo.add"(%1144, %1126) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1146 = "pphlo.shift_left"(%1143, %196) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1147 = "pphlo.shift_right_logical"(%1143, %195) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1148 = "pphlo.or"(%1146, %1147) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1149 = "pphlo.xor"(%1144, %1148) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1150 = "pphlo.add"(%1149, %1094) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1151 = "pphlo.add"(%1150, %200) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1152 = "pphlo.add"(%1145, %1151) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1153 = "pphlo.shift_left"(%1151, %193) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1154 = "pphlo.shift_right_logical"(%1151, %192) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1155 = "pphlo.or"(%1153, %1154) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1156 = "pphlo.xor"(%1152, %1155) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1157 = "pphlo.add"(%1152, %1156) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1158 = "pphlo.shift_left"(%1156, %191) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1159 = "pphlo.shift_right_logical"(%1156, %190) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1160 = "pphlo.or"(%1158, %1159) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1161 = "pphlo.xor"(%1157, %1160) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1162 = "pphlo.add"(%1157, %1161) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1163 = "pphlo.shift_left"(%1161, %188) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1164 = "pphlo.shift_right_logical"(%1161, %189) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1165 = "pphlo.or"(%1163, %1164) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1166 = "pphlo.xor"(%1162, %1165) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1167 = "pphlo.add"(%1162, %1166) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1168 = "pphlo.add"(%1167, %1094) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1169 = "pphlo.shift_left"(%1166, %189) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1170 = "pphlo.shift_right_logical"(%1166, %188) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1171 = "pphlo.or"(%1169, %1170) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1172 = "pphlo.xor"(%1167, %1171) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1173 = "pphlo.add"(%1172, %1101) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1174 = "pphlo.add"(%1173, %198) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1175 = "pphlo.add"(%1168, %1174) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1176 = "pphlo.shift_left"(%1174, %190) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1177 = "pphlo.shift_right_logical"(%1174, %191) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1178 = "pphlo.or"(%1176, %1177) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1179 = "pphlo.xor"(%1175, %1178) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1180 = "pphlo.add"(%1175, %1179) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1181 = "pphlo.shift_left"(%1179, %199) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1182 = "pphlo.shift_right_logical"(%1179, %198) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1183 = "pphlo.or"(%1181, %1182) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1184 = "pphlo.xor"(%1180, %1183) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1185 = "pphlo.add"(%1180, %1184) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1186 = "pphlo.shift_left"(%1184, %197) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1187 = "pphlo.shift_right_logical"(%1184, %197) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1188 = "pphlo.or"(%1186, %1187) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1189 = "pphlo.xor"(%1185, %1188) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1190 = "pphlo.add"(%1185, %1189) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1191 = "pphlo.add"(%1190, %1101) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1192 = "pphlo.shift_left"(%1189, %196) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1193 = "pphlo.shift_right_logical"(%1189, %195) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1194 = "pphlo.or"(%1192, %1193) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1195 = "pphlo.xor"(%1190, %1194) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1196 = "pphlo.add"(%1195, %1126) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1197 = "pphlo.add"(%1196, %194) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1198 = "pphlo.add"(%1191, %1197) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1199 = "pphlo.shift_left"(%1197, %193) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1200 = "pphlo.shift_right_logical"(%1197, %192) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1201 = "pphlo.or"(%1199, %1200) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1202 = "pphlo.xor"(%1198, %1201) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1203 = "pphlo.add"(%1198, %1202) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1204 = "pphlo.shift_left"(%1202, %191) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1205 = "pphlo.shift_right_logical"(%1202, %190) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1206 = "pphlo.or"(%1204, %1205) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1207 = "pphlo.xor"(%1203, %1206) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1208 = "pphlo.add"(%1203, %1207) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1209 = "pphlo.shift_left"(%1207, %188) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1210 = "pphlo.shift_right_logical"(%1207, %189) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1211 = "pphlo.or"(%1209, %1210) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1212 = "pphlo.xor"(%1208, %1211) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1213 = "pphlo.add"(%1208, %1212) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1214 = "pphlo.add"(%1213, %1126) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1215 = "pphlo.shift_left"(%1212, %189) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1216 = "pphlo.shift_right_logical"(%1212, %188) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1217 = "pphlo.or"(%1215, %1216) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1218 = "pphlo.xor"(%1213, %1217) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1219 = "pphlo.add"(%1218, %1094) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1220 = "pphlo.add"(%1219, %187) : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<15x!pphlo.pint>
    %1221 = "pphlo.concatenate"(%1214, %1220) {dimension = 0 : i64} : (tensor<15x!pphlo.pint>, tensor<15x!pphlo.pint>) -> tensor<30x!pphlo.pint>
    %1222:2 = "pphlo.sort"(%1221, %832) {dimension = 0 : i64, is_less = true, is_stable = true} : (tensor<30x!pphlo.pint>, tensor<30x!pphlo.pint>) -> (tensor<30x!pphlo.pint>, tensor<30x!pphlo.pint>)
    %1223 = "pphlo.less"(%1222#1, %89) : (tensor<30x!pphlo.pint>, tensor<30x!pphlo.pint>) -> tensor<30x!pphlo.pint>
    %1224 = "pphlo.add"(%1222#1, %88) : (tensor<30x!pphlo.pint>, tensor<30x!pphlo.pint>) -> tensor<30x!pphlo.pint>
    %1225 = "pphlo.subtract"(%1224, %1222#1) : (tensor<30x!pphlo.pint>, tensor<30x!pphlo.pint>) -> tensor<30x!pphlo.pint>
    %1226 = "pphlo.multiply"(%1223, %1225) : (tensor<30x!pphlo.pint>, tensor<30x!pphlo.pint>) -> tensor<30x!pphlo.pint>
    %1227 = "pphlo.add"(%1226, %1222#1) : (tensor<30x!pphlo.pint>, tensor<30x!pphlo.pint>) -> tensor<30x!pphlo.pint>
    %1228 = "pphlo.reshape"(%1227) : (tensor<30x!pphlo.pint>) -> tensor<30x1x!pphlo.pint>
    %1229 = "pphlo.gather"(%arg0, %1228) {dimension_numbers = #pphlo.gather<offset_dims = [1, 2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 28, 28, 1]> : tensor<4xi64>} : (tensor<30x28x28x1x!pphlo.pfxp>, tensor<30x1x!pphlo.pint>) -> tensor<30x28x28x1x!pphlo.pfxp>
    %1230 = pphlo.convolution(%1229, %831) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<30x28x28x1x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<30x28x28x32x!pphlo.pfxp>
    %1231 = "pphlo.greater"(%1230, %17) : (tensor<30x28x28x32x!pphlo.pfxp>, tensor<30x28x28x32x!pphlo.pfxp>) -> tensor<30x28x28x32x!pphlo.pint>
    %1232 = "pphlo.maximum"(%1230, %17) : (tensor<30x28x28x32x!pphlo.pfxp>, tensor<30x28x28x32x!pphlo.pfxp>) -> tensor<30x28x28x32x!pphlo.pfxp>
    %1233 = "pphlo.reduce_window"(%1232, %1) ( {
    ^bb0(%arg4: tensor<!pphlo.pfxp>, %arg5: tensor<!pphlo.pfxp>):  // no predecessors
      %2655 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%2655) : (tensor<!pphlo.pfxp>) -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<30x28x28x32x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<30x14x14x32x!pphlo.pfxp>
    %1234 = "pphlo.multiply"(%1233, %18) : (tensor<30x14x14x32x!pphlo.pfxp>, tensor<30x14x14x32x!pphlo.pfxp>) -> tensor<30x14x14x32x!pphlo.pfxp>
    %1235 = "pphlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<18432x!pphlo.pint>
    %1236 = "pphlo.slice"(%1235) {limit_indices = dense<9216> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<18432x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1237 = "pphlo.add"(%390, %185) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1238 = "pphlo.add"(%387, %1237) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1239 = "pphlo.shift_left"(%1237, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1240 = "pphlo.shift_right_logical"(%1237, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1241 = "pphlo.or"(%1239, %1240) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1242 = "pphlo.xor"(%1238, %1241) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1243 = "pphlo.add"(%1238, %1242) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1244 = "pphlo.shift_left"(%1242, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1245 = "pphlo.shift_right_logical"(%1242, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1246 = "pphlo.or"(%1244, %1245) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1247 = "pphlo.xor"(%1243, %1246) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1248 = "pphlo.add"(%1243, %1247) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1249 = "pphlo.shift_left"(%1247, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1250 = "pphlo.shift_right_logical"(%1247, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1251 = "pphlo.or"(%1249, %1250) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1252 = "pphlo.xor"(%1248, %1251) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1253 = "pphlo.add"(%1248, %1252) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1254 = "pphlo.add"(%1253, %390) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1255 = "pphlo.shift_left"(%1252, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1256 = "pphlo.shift_right_logical"(%1252, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1257 = "pphlo.or"(%1255, %1256) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1258 = "pphlo.xor"(%1253, %1257) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1259 = "pphlo.add"(%1258, %417) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1260 = "pphlo.add"(%1259, %83) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1261 = "pphlo.add"(%1254, %1260) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1262 = "pphlo.shift_left"(%1260, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1263 = "pphlo.shift_right_logical"(%1260, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1264 = "pphlo.or"(%1262, %1263) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1265 = "pphlo.xor"(%1261, %1264) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1266 = "pphlo.add"(%1261, %1265) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1267 = "pphlo.shift_left"(%1265, %81) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1268 = "pphlo.shift_right_logical"(%1265, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1269 = "pphlo.or"(%1267, %1268) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1270 = "pphlo.xor"(%1266, %1269) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1271 = "pphlo.add"(%1266, %1270) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1272 = "pphlo.shift_left"(%1270, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1273 = "pphlo.shift_right_logical"(%1270, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1274 = "pphlo.or"(%1272, %1273) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1275 = "pphlo.xor"(%1271, %1274) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1276 = "pphlo.add"(%1271, %1275) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1277 = "pphlo.add"(%1276, %417) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1278 = "pphlo.shift_left"(%1275, %78) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1279 = "pphlo.shift_right_logical"(%1275, %77) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1280 = "pphlo.or"(%1278, %1279) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1281 = "pphlo.xor"(%1276, %1280) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1282 = "pphlo.add"(%1281, %387) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1283 = "pphlo.add"(%1282, %82) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1284 = "pphlo.add"(%1277, %1283) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1285 = "pphlo.shift_left"(%1283, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1286 = "pphlo.shift_right_logical"(%1283, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1287 = "pphlo.or"(%1285, %1286) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1288 = "pphlo.xor"(%1284, %1287) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1289 = "pphlo.add"(%1284, %1288) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1290 = "pphlo.shift_left"(%1288, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1291 = "pphlo.shift_right_logical"(%1288, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1292 = "pphlo.or"(%1290, %1291) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1293 = "pphlo.xor"(%1289, %1292) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1294 = "pphlo.add"(%1289, %1293) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1295 = "pphlo.shift_left"(%1293, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1296 = "pphlo.shift_right_logical"(%1293, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1297 = "pphlo.or"(%1295, %1296) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1298 = "pphlo.xor"(%1294, %1297) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1299 = "pphlo.add"(%1294, %1298) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1300 = "pphlo.add"(%1299, %387) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1301 = "pphlo.shift_left"(%1298, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1302 = "pphlo.shift_right_logical"(%1298, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1303 = "pphlo.or"(%1301, %1302) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1304 = "pphlo.xor"(%1299, %1303) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1305 = "pphlo.add"(%1304, %390) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1306 = "pphlo.add"(%1305, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1307 = "pphlo.add"(%1300, %1306) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1308 = "pphlo.shift_left"(%1306, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1309 = "pphlo.shift_right_logical"(%1306, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1310 = "pphlo.or"(%1308, %1309) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1311 = "pphlo.xor"(%1307, %1310) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1312 = "pphlo.add"(%1307, %1311) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1313 = "pphlo.shift_left"(%1311, %81) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1314 = "pphlo.shift_right_logical"(%1311, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1315 = "pphlo.or"(%1313, %1314) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1316 = "pphlo.xor"(%1312, %1315) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1317 = "pphlo.add"(%1312, %1316) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1318 = "pphlo.shift_left"(%1316, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1319 = "pphlo.shift_right_logical"(%1316, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1320 = "pphlo.or"(%1318, %1319) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1321 = "pphlo.xor"(%1317, %1320) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1322 = "pphlo.add"(%1317, %1321) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1323 = "pphlo.add"(%1322, %390) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1324 = "pphlo.shift_left"(%1321, %78) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1325 = "pphlo.shift_right_logical"(%1321, %77) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1326 = "pphlo.or"(%1324, %1325) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1327 = "pphlo.xor"(%1322, %1326) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1328 = "pphlo.add"(%1327, %417) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1329 = "pphlo.add"(%1328, %76) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1330 = "pphlo.add"(%1323, %1329) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1331 = "pphlo.shift_left"(%1329, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1332 = "pphlo.shift_right_logical"(%1329, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1333 = "pphlo.or"(%1331, %1332) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1334 = "pphlo.xor"(%1330, %1333) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1335 = "pphlo.add"(%1330, %1334) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1336 = "pphlo.shift_left"(%1334, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1337 = "pphlo.shift_right_logical"(%1334, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1338 = "pphlo.or"(%1336, %1337) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1339 = "pphlo.xor"(%1335, %1338) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1340 = "pphlo.add"(%1335, %1339) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1341 = "pphlo.shift_left"(%1339, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1342 = "pphlo.shift_right_logical"(%1339, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1343 = "pphlo.or"(%1341, %1342) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1344 = "pphlo.xor"(%1340, %1343) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1345 = "pphlo.add"(%1340, %1344) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1346 = "pphlo.add"(%1345, %417) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1347 = "pphlo.shift_left"(%1344, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1348 = "pphlo.shift_right_logical"(%1344, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1349 = "pphlo.or"(%1347, %1348) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1350 = "pphlo.xor"(%1345, %1349) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1351 = "pphlo.add"(%1350, %387) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1352 = "pphlo.add"(%1351, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1353 = "pphlo.add"(%1346, %1352) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1354 = "pphlo.shift_left"(%1352, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1355 = "pphlo.shift_right_logical"(%1352, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1356 = "pphlo.or"(%1354, %1355) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1357 = "pphlo.xor"(%1353, %1356) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1358 = "pphlo.add"(%1353, %1357) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1359 = "pphlo.shift_left"(%1357, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1360 = "pphlo.shift_right_logical"(%1357, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1361 = "pphlo.or"(%1359, %1360) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1362 = "pphlo.xor"(%1358, %1361) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1363 = "pphlo.add"(%1358, %1362) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1364 = "pphlo.shift_left"(%1362, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1365 = "pphlo.shift_right_logical"(%1362, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1366 = "pphlo.or"(%1364, %1365) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1367 = "pphlo.xor"(%1363, %1366) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1368 = "pphlo.add"(%1363, %1367) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1369 = "pphlo.add"(%1351, %69) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1370 = "pphlo.add"(%1368, %1369) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1371 = "pphlo.shift_left"(%1367, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1372 = "pphlo.shift_right_logical"(%1367, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1373 = "pphlo.or"(%1371, %1372) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1374 = "pphlo.xor"(%1368, %1373) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1375 = "pphlo.reshape"(%1346) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %1376 = "pphlo.reshape"(%1369) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %1377 = "pphlo.xor"(%1375, %1376) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %1378 = "pphlo.xor"(%1377, %68) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %1379 = "pphlo.reshape"(%1378) : (tensor<!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1380 = "pphlo.add"(%1374, %1379) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1381 = "pphlo.add"(%1380, %83) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1382 = "pphlo.add"(%1370, %1381) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1383 = "pphlo.shift_left"(%1381, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1384 = "pphlo.shift_right_logical"(%1381, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1385 = "pphlo.or"(%1383, %1384) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1386 = "pphlo.xor"(%1382, %1385) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1387 = "pphlo.add"(%1382, %1386) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1388 = "pphlo.shift_left"(%1386, %81) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1389 = "pphlo.shift_right_logical"(%1386, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1390 = "pphlo.or"(%1388, %1389) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1391 = "pphlo.xor"(%1387, %1390) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1392 = "pphlo.add"(%1387, %1391) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1393 = "pphlo.shift_left"(%1391, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1394 = "pphlo.shift_right_logical"(%1391, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1395 = "pphlo.or"(%1393, %1394) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1396 = "pphlo.xor"(%1392, %1395) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1397 = "pphlo.add"(%1392, %1396) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1398 = "pphlo.add"(%1397, %1379) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1399 = "pphlo.shift_left"(%1396, %78) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1400 = "pphlo.shift_right_logical"(%1396, %77) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1401 = "pphlo.or"(%1399, %1400) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1402 = "pphlo.xor"(%1397, %1401) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1403 = "pphlo.add"(%1402, %1346) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1404 = "pphlo.add"(%1403, %82) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1405 = "pphlo.add"(%1398, %1404) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1406 = "pphlo.shift_left"(%1404, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1407 = "pphlo.shift_right_logical"(%1404, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1408 = "pphlo.or"(%1406, %1407) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1409 = "pphlo.xor"(%1405, %1408) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1410 = "pphlo.add"(%1405, %1409) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1411 = "pphlo.shift_left"(%1409, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1412 = "pphlo.shift_right_logical"(%1409, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1413 = "pphlo.or"(%1411, %1412) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1414 = "pphlo.xor"(%1410, %1413) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1415 = "pphlo.add"(%1410, %1414) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1416 = "pphlo.shift_left"(%1414, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1417 = "pphlo.shift_right_logical"(%1414, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1418 = "pphlo.or"(%1416, %1417) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1419 = "pphlo.xor"(%1415, %1418) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1420 = "pphlo.add"(%1415, %1419) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1421 = "pphlo.add"(%1420, %1346) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1422 = "pphlo.shift_left"(%1419, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1423 = "pphlo.shift_right_logical"(%1419, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1424 = "pphlo.or"(%1422, %1423) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1425 = "pphlo.xor"(%1420, %1424) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1426 = "pphlo.add"(%1425, %1369) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1427 = "pphlo.add"(%1426, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1428 = "pphlo.add"(%1421, %1427) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1429 = "pphlo.shift_left"(%1427, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1430 = "pphlo.shift_right_logical"(%1427, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1431 = "pphlo.or"(%1429, %1430) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1432 = "pphlo.xor"(%1428, %1431) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1433 = "pphlo.add"(%1428, %1432) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1434 = "pphlo.shift_left"(%1432, %81) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1435 = "pphlo.shift_right_logical"(%1432, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1436 = "pphlo.or"(%1434, %1435) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1437 = "pphlo.xor"(%1433, %1436) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1438 = "pphlo.add"(%1433, %1437) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1439 = "pphlo.shift_left"(%1437, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1440 = "pphlo.shift_right_logical"(%1437, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1441 = "pphlo.or"(%1439, %1440) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1442 = "pphlo.xor"(%1438, %1441) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1443 = "pphlo.add"(%1438, %1442) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1444 = "pphlo.add"(%1443, %1369) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1445 = "pphlo.shift_left"(%1442, %78) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1446 = "pphlo.shift_right_logical"(%1442, %77) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1447 = "pphlo.or"(%1445, %1446) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1448 = "pphlo.xor"(%1443, %1447) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1449 = "pphlo.add"(%1448, %1379) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1450 = "pphlo.add"(%1449, %76) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1451 = "pphlo.add"(%1444, %1450) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1452 = "pphlo.shift_left"(%1450, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1453 = "pphlo.shift_right_logical"(%1450, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1454 = "pphlo.or"(%1452, %1453) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1455 = "pphlo.xor"(%1451, %1454) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1456 = "pphlo.add"(%1451, %1455) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1457 = "pphlo.shift_left"(%1455, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1458 = "pphlo.shift_right_logical"(%1455, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1459 = "pphlo.or"(%1457, %1458) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1460 = "pphlo.xor"(%1456, %1459) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1461 = "pphlo.add"(%1456, %1460) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1462 = "pphlo.shift_left"(%1460, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1463 = "pphlo.shift_right_logical"(%1460, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1464 = "pphlo.or"(%1462, %1463) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1465 = "pphlo.xor"(%1461, %1464) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1466 = "pphlo.add"(%1461, %1465) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1467 = "pphlo.add"(%1466, %1379) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1468 = "pphlo.reshape"(%1467) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %1469 = "pphlo.broadcast"(%1468) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1470 = "pphlo.add"(%1236, %1469) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1471 = "pphlo.slice"(%1235) {limit_indices = dense<18432> : tensor<1xi64>, start_indices = dense<9216> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<18432x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1472 = "pphlo.shift_left"(%1465, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1473 = "pphlo.shift_right_logical"(%1465, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1474 = "pphlo.or"(%1472, %1473) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1475 = "pphlo.xor"(%1466, %1474) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1476 = "pphlo.add"(%1475, %1346) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1477 = "pphlo.add"(%1476, %69) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1478 = "pphlo.reshape"(%1477) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %1479 = "pphlo.broadcast"(%1478) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1480 = "pphlo.add"(%1471, %1479) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1481 = "pphlo.add"(%1470, %1480) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1482 = "pphlo.shift_left"(%1480, %176) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1483 = "pphlo.shift_right_logical"(%1480, %175) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1484 = "pphlo.or"(%1482, %1483) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1485 = "pphlo.xor"(%1481, %1484) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1486 = "pphlo.add"(%1481, %1485) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1487 = "pphlo.shift_left"(%1485, %174) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1488 = "pphlo.shift_right_logical"(%1485, %173) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1489 = "pphlo.or"(%1487, %1488) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1490 = "pphlo.xor"(%1486, %1489) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1491 = "pphlo.add"(%1486, %1490) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1492 = "pphlo.shift_left"(%1490, %171) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1493 = "pphlo.shift_right_logical"(%1490, %172) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1494 = "pphlo.or"(%1492, %1493) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1495 = "pphlo.xor"(%1491, %1494) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1496 = "pphlo.add"(%1491, %1495) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1497 = "pphlo.add"(%1496, %1479) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1498 = "pphlo.shift_left"(%1495, %172) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1499 = "pphlo.shift_right_logical"(%1495, %171) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1500 = "pphlo.or"(%1498, %1499) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1501 = "pphlo.xor"(%1496, %1500) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1502 = "pphlo.xor"(%1468, %1478) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %1503 = "pphlo.xor"(%1502, %68) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %1504 = "pphlo.broadcast"(%1503) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1505 = "pphlo.add"(%1501, %1504) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1506 = "pphlo.add"(%1505, %184) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1507 = "pphlo.add"(%1497, %1506) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1508 = "pphlo.shift_left"(%1506, %173) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1509 = "pphlo.shift_right_logical"(%1506, %174) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1510 = "pphlo.or"(%1508, %1509) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1511 = "pphlo.xor"(%1507, %1510) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1512 = "pphlo.add"(%1507, %1511) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1513 = "pphlo.shift_left"(%1511, %182) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1514 = "pphlo.shift_right_logical"(%1511, %181) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1515 = "pphlo.or"(%1513, %1514) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1516 = "pphlo.xor"(%1512, %1515) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1517 = "pphlo.add"(%1512, %1516) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1518 = "pphlo.shift_left"(%1516, %180) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1519 = "pphlo.shift_right_logical"(%1516, %180) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1520 = "pphlo.or"(%1518, %1519) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1521 = "pphlo.xor"(%1517, %1520) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1522 = "pphlo.add"(%1517, %1521) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1523 = "pphlo.add"(%1522, %1504) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1524 = "pphlo.shift_left"(%1521, %179) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1525 = "pphlo.shift_right_logical"(%1521, %178) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1526 = "pphlo.or"(%1524, %1525) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1527 = "pphlo.xor"(%1522, %1526) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1528 = "pphlo.add"(%1527, %1469) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1529 = "pphlo.add"(%1528, %183) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1530 = "pphlo.add"(%1523, %1529) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1531 = "pphlo.shift_left"(%1529, %176) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1532 = "pphlo.shift_right_logical"(%1529, %175) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1533 = "pphlo.or"(%1531, %1532) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1534 = "pphlo.xor"(%1530, %1533) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1535 = "pphlo.add"(%1530, %1534) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1536 = "pphlo.shift_left"(%1534, %174) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1537 = "pphlo.shift_right_logical"(%1534, %173) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1538 = "pphlo.or"(%1536, %1537) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1539 = "pphlo.xor"(%1535, %1538) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1540 = "pphlo.add"(%1535, %1539) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1541 = "pphlo.shift_left"(%1539, %171) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1542 = "pphlo.shift_right_logical"(%1539, %172) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1543 = "pphlo.or"(%1541, %1542) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1544 = "pphlo.xor"(%1540, %1543) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1545 = "pphlo.add"(%1540, %1544) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1546 = "pphlo.add"(%1545, %1469) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1547 = "pphlo.shift_left"(%1544, %172) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1548 = "pphlo.shift_right_logical"(%1544, %171) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1549 = "pphlo.or"(%1547, %1548) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1550 = "pphlo.xor"(%1545, %1549) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1551 = "pphlo.add"(%1550, %1479) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1552 = "pphlo.add"(%1551, %181) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1553 = "pphlo.add"(%1546, %1552) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1554 = "pphlo.shift_left"(%1552, %173) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1555 = "pphlo.shift_right_logical"(%1552, %174) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1556 = "pphlo.or"(%1554, %1555) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1557 = "pphlo.xor"(%1553, %1556) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1558 = "pphlo.add"(%1553, %1557) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1559 = "pphlo.shift_left"(%1557, %182) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1560 = "pphlo.shift_right_logical"(%1557, %181) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1561 = "pphlo.or"(%1559, %1560) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1562 = "pphlo.xor"(%1558, %1561) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1563 = "pphlo.add"(%1558, %1562) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1564 = "pphlo.shift_left"(%1562, %180) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1565 = "pphlo.shift_right_logical"(%1562, %180) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1566 = "pphlo.or"(%1564, %1565) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1567 = "pphlo.xor"(%1563, %1566) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1568 = "pphlo.add"(%1563, %1567) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1569 = "pphlo.add"(%1568, %1479) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1570 = "pphlo.shift_left"(%1567, %179) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1571 = "pphlo.shift_right_logical"(%1567, %178) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1572 = "pphlo.or"(%1570, %1571) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1573 = "pphlo.xor"(%1568, %1572) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1574 = "pphlo.add"(%1573, %1504) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1575 = "pphlo.add"(%1574, %177) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1576 = "pphlo.add"(%1569, %1575) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1577 = "pphlo.shift_left"(%1575, %176) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1578 = "pphlo.shift_right_logical"(%1575, %175) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1579 = "pphlo.or"(%1577, %1578) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1580 = "pphlo.xor"(%1576, %1579) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1581 = "pphlo.add"(%1576, %1580) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1582 = "pphlo.shift_left"(%1580, %174) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1583 = "pphlo.shift_right_logical"(%1580, %173) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1584 = "pphlo.or"(%1582, %1583) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1585 = "pphlo.xor"(%1581, %1584) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1586 = "pphlo.add"(%1581, %1585) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1587 = "pphlo.shift_left"(%1585, %171) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1588 = "pphlo.shift_right_logical"(%1585, %172) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1589 = "pphlo.or"(%1587, %1588) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1590 = "pphlo.xor"(%1586, %1589) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1591 = "pphlo.add"(%1586, %1590) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1592 = "pphlo.add"(%1591, %1504) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1593 = "pphlo.shift_left"(%1590, %172) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1594 = "pphlo.shift_right_logical"(%1590, %171) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1595 = "pphlo.or"(%1593, %1594) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1596 = "pphlo.xor"(%1591, %1595) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1597 = "pphlo.add"(%1596, %1469) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1598 = "pphlo.add"(%1597, %170) : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<9216x!pphlo.pint>
    %1599 = "pphlo.concatenate"(%1592, %1598) {dimension = 0 : i64} : (tensor<9216x!pphlo.pint>, tensor<9216x!pphlo.pint>) -> tensor<18432x!pphlo.pint>
    %1600 = "pphlo.shift_right_logical"(%1599, %169) : (tensor<18432x!pphlo.pint>, tensor<18432x!pphlo.pint>) -> tensor<18432x!pphlo.pint>
    %1601 = "pphlo.or"(%1600, %168) : (tensor<18432x!pphlo.pint>, tensor<18432x!pphlo.pint>) -> tensor<18432x!pphlo.pint>
    %1602 = "pphlo.bitcast_convert"(%1601) {elsize = 32 : i64} : (tensor<18432x!pphlo.pint>) -> tensor<18432x!pphlo.pfxp>
    %1603 = "pphlo.add"(%1602, %167) : (tensor<18432x!pphlo.pfxp>, tensor<18432x!pphlo.pfxp>) -> tensor<18432x!pphlo.pfxp>
    %1604 = "pphlo.multiply"(%1603, %166) : (tensor<18432x!pphlo.pfxp>, tensor<18432x!pphlo.pfxp>) -> tensor<18432x!pphlo.pfxp>
    %1605 = "pphlo.add"(%1604, %165) : (tensor<18432x!pphlo.pfxp>, tensor<18432x!pphlo.pfxp>) -> tensor<18432x!pphlo.pfxp>
    %1606 = "pphlo.maximum"(%1605, %165) : (tensor<18432x!pphlo.pfxp>, tensor<18432x!pphlo.pfxp>) -> tensor<18432x!pphlo.pfxp>
    %1607 = "pphlo.reshape"(%1606) : (tensor<18432x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1608 = "pphlo.abs"(%1607) : (tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1609 = "pphlo.equal"(%1608, %164) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pint>
    %1610 = "pphlo.multiply"(%1607, %163) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1611 = "pphlo.negate"(%1607) : (tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1612 = "pphlo.multiply"(%1611, %1607) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1613 = "pphlo.log_plus_one"(%1612) : (tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1614 = "pphlo.negate"(%1613) : (tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1615 = "pphlo.less"(%1614, %162) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pint>
    %1616 = "pphlo.subtract"(%161, %160) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1617 = "pphlo.multiply"(%1615, %1616) : (tensor<3x3x32x64x!pphlo.pint>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1618 = "pphlo.add"(%1617, %160) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1619 = "pphlo.subtract"(%159, %158) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1620 = "pphlo.multiply"(%1615, %1619) : (tensor<3x3x32x64x!pphlo.pint>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1621 = "pphlo.add"(%1620, %158) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1622 = "pphlo.subtract"(%157, %156) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1623 = "pphlo.multiply"(%1615, %1622) : (tensor<3x3x32x64x!pphlo.pint>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1624 = "pphlo.add"(%1623, %156) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1625 = "pphlo.subtract"(%155, %154) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1626 = "pphlo.multiply"(%1615, %1625) : (tensor<3x3x32x64x!pphlo.pint>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1627 = "pphlo.add"(%1626, %154) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1628 = "pphlo.subtract"(%153, %152) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1629 = "pphlo.multiply"(%1615, %1628) : (tensor<3x3x32x64x!pphlo.pint>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1630 = "pphlo.add"(%1629, %152) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1631 = "pphlo.subtract"(%151, %150) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1632 = "pphlo.multiply"(%1615, %1631) : (tensor<3x3x32x64x!pphlo.pint>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1633 = "pphlo.add"(%1632, %150) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1634 = "pphlo.subtract"(%149, %148) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1635 = "pphlo.multiply"(%1615, %1634) : (tensor<3x3x32x64x!pphlo.pint>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1636 = "pphlo.add"(%1635, %148) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1637 = "pphlo.subtract"(%147, %146) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1638 = "pphlo.multiply"(%1615, %1637) : (tensor<3x3x32x64x!pphlo.pint>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1639 = "pphlo.add"(%1638, %146) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1640 = "pphlo.subtract"(%145, %144) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1641 = "pphlo.multiply"(%1615, %1640) : (tensor<3x3x32x64x!pphlo.pint>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1642 = "pphlo.add"(%1641, %144) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1643 = "pphlo.add"(%1614, %143) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1644 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pfxp>
    %1645 = "pphlo.power"(%1614, %1644) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1646 = "pphlo.add"(%1645, %142) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1647 = "pphlo.subtract"(%1643, %1646) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1648 = "pphlo.multiply"(%1615, %1647) : (tensor<3x3x32x64x!pphlo.pint>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1649 = "pphlo.add"(%1648, %1646) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1650 = "pphlo.multiply"(%1642, %1649) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1651 = "pphlo.add"(%1639, %1650) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1652 = "pphlo.multiply"(%1651, %1649) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1653 = "pphlo.add"(%1636, %1652) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1654 = "pphlo.multiply"(%1653, %1649) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1655 = "pphlo.add"(%1633, %1654) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1656 = "pphlo.multiply"(%1655, %1649) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1657 = "pphlo.add"(%1630, %1656) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1658 = "pphlo.multiply"(%1657, %1649) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1659 = "pphlo.add"(%1627, %1658) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1660 = "pphlo.multiply"(%1659, %1649) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1661 = "pphlo.add"(%1624, %1660) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1662 = "pphlo.multiply"(%1661, %1649) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1663 = "pphlo.add"(%1621, %1662) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1664 = "pphlo.multiply"(%1663, %1649) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1665 = "pphlo.add"(%1618, %1664) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1666 = "pphlo.multiply"(%1665, %1607) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1667 = "pphlo.subtract"(%1610, %1666) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1668 = "pphlo.multiply"(%1609, %1667) : (tensor<3x3x32x64x!pphlo.pint>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1669 = "pphlo.add"(%1668, %1666) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1670 = "pphlo.multiply"(%1669, %141) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1671 = "pphlo.clamp"(%186, %1670, %140) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1672 = "pphlo.multiply"(%1671, %139) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %1673 = pphlo.convolution(%1234, %1672) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<30x14x14x32x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<30x14x14x64x!pphlo.pfxp>
    %1674 = "pphlo.greater"(%1673, %19) : (tensor<30x14x14x64x!pphlo.pfxp>, tensor<30x14x14x64x!pphlo.pfxp>) -> tensor<30x14x14x64x!pphlo.pint>
    %1675 = "pphlo.maximum"(%1673, %19) : (tensor<30x14x14x64x!pphlo.pfxp>, tensor<30x14x14x64x!pphlo.pfxp>) -> tensor<30x14x14x64x!pphlo.pfxp>
    %1676 = "pphlo.reduce_window"(%1675, %1) ( {
    ^bb0(%arg4: tensor<!pphlo.pfxp>, %arg5: tensor<!pphlo.pfxp>):  // no predecessors
      %2655 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%2655) : (tensor<!pphlo.pfxp>) -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<30x14x14x64x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<30x7x7x64x!pphlo.pfxp>
    %1677 = "pphlo.multiply"(%1676, %138) : (tensor<30x7x7x64x!pphlo.pfxp>, tensor<30x7x7x64x!pphlo.pfxp>) -> tensor<30x7x7x64x!pphlo.pfxp>
    %1678 = "pphlo.reshape"(%1677) : (tensor<30x7x7x64x!pphlo.pfxp>) -> tensor<30x3136x!pphlo.pfxp>
    %1679 = "pphlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<802816x!pphlo.pint>
    %1680 = "pphlo.slice"(%1679) {limit_indices = dense<401408> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<802816x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1681 = "pphlo.add"(%390, %136) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1682 = "pphlo.add"(%387, %1681) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1683 = "pphlo.shift_left"(%1681, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1684 = "pphlo.shift_right_logical"(%1681, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1685 = "pphlo.or"(%1683, %1684) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1686 = "pphlo.xor"(%1682, %1685) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1687 = "pphlo.add"(%1682, %1686) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1688 = "pphlo.shift_left"(%1686, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1689 = "pphlo.shift_right_logical"(%1686, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1690 = "pphlo.or"(%1688, %1689) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1691 = "pphlo.xor"(%1687, %1690) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1692 = "pphlo.add"(%1687, %1691) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1693 = "pphlo.shift_left"(%1691, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1694 = "pphlo.shift_right_logical"(%1691, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1695 = "pphlo.or"(%1693, %1694) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1696 = "pphlo.xor"(%1692, %1695) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1697 = "pphlo.add"(%1692, %1696) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1698 = "pphlo.add"(%1697, %390) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1699 = "pphlo.shift_left"(%1696, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1700 = "pphlo.shift_right_logical"(%1696, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1701 = "pphlo.or"(%1699, %1700) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1702 = "pphlo.xor"(%1697, %1701) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1703 = "pphlo.add"(%1702, %417) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1704 = "pphlo.add"(%1703, %83) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1705 = "pphlo.add"(%1698, %1704) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1706 = "pphlo.shift_left"(%1704, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1707 = "pphlo.shift_right_logical"(%1704, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1708 = "pphlo.or"(%1706, %1707) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1709 = "pphlo.xor"(%1705, %1708) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1710 = "pphlo.add"(%1705, %1709) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1711 = "pphlo.shift_left"(%1709, %81) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1712 = "pphlo.shift_right_logical"(%1709, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1713 = "pphlo.or"(%1711, %1712) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1714 = "pphlo.xor"(%1710, %1713) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1715 = "pphlo.add"(%1710, %1714) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1716 = "pphlo.shift_left"(%1714, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1717 = "pphlo.shift_right_logical"(%1714, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1718 = "pphlo.or"(%1716, %1717) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1719 = "pphlo.xor"(%1715, %1718) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1720 = "pphlo.add"(%1715, %1719) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1721 = "pphlo.add"(%1720, %417) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1722 = "pphlo.shift_left"(%1719, %78) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1723 = "pphlo.shift_right_logical"(%1719, %77) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1724 = "pphlo.or"(%1722, %1723) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1725 = "pphlo.xor"(%1720, %1724) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1726 = "pphlo.add"(%1725, %387) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1727 = "pphlo.add"(%1726, %82) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1728 = "pphlo.add"(%1721, %1727) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1729 = "pphlo.shift_left"(%1727, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1730 = "pphlo.shift_right_logical"(%1727, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1731 = "pphlo.or"(%1729, %1730) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1732 = "pphlo.xor"(%1728, %1731) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1733 = "pphlo.add"(%1728, %1732) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1734 = "pphlo.shift_left"(%1732, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1735 = "pphlo.shift_right_logical"(%1732, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1736 = "pphlo.or"(%1734, %1735) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1737 = "pphlo.xor"(%1733, %1736) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1738 = "pphlo.add"(%1733, %1737) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1739 = "pphlo.shift_left"(%1737, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1740 = "pphlo.shift_right_logical"(%1737, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1741 = "pphlo.or"(%1739, %1740) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1742 = "pphlo.xor"(%1738, %1741) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1743 = "pphlo.add"(%1738, %1742) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1744 = "pphlo.add"(%1743, %387) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1745 = "pphlo.shift_left"(%1742, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1746 = "pphlo.shift_right_logical"(%1742, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1747 = "pphlo.or"(%1745, %1746) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1748 = "pphlo.xor"(%1743, %1747) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1749 = "pphlo.add"(%1748, %390) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1750 = "pphlo.add"(%1749, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1751 = "pphlo.add"(%1744, %1750) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1752 = "pphlo.shift_left"(%1750, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1753 = "pphlo.shift_right_logical"(%1750, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1754 = "pphlo.or"(%1752, %1753) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1755 = "pphlo.xor"(%1751, %1754) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1756 = "pphlo.add"(%1751, %1755) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1757 = "pphlo.shift_left"(%1755, %81) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1758 = "pphlo.shift_right_logical"(%1755, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1759 = "pphlo.or"(%1757, %1758) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1760 = "pphlo.xor"(%1756, %1759) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1761 = "pphlo.add"(%1756, %1760) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1762 = "pphlo.shift_left"(%1760, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1763 = "pphlo.shift_right_logical"(%1760, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1764 = "pphlo.or"(%1762, %1763) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1765 = "pphlo.xor"(%1761, %1764) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1766 = "pphlo.add"(%1761, %1765) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1767 = "pphlo.add"(%1766, %390) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1768 = "pphlo.shift_left"(%1765, %78) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1769 = "pphlo.shift_right_logical"(%1765, %77) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1770 = "pphlo.or"(%1768, %1769) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1771 = "pphlo.xor"(%1766, %1770) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1772 = "pphlo.add"(%1771, %417) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1773 = "pphlo.add"(%1772, %76) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1774 = "pphlo.add"(%1767, %1773) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1775 = "pphlo.shift_left"(%1773, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1776 = "pphlo.shift_right_logical"(%1773, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1777 = "pphlo.or"(%1775, %1776) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1778 = "pphlo.xor"(%1774, %1777) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1779 = "pphlo.add"(%1774, %1778) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1780 = "pphlo.shift_left"(%1778, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1781 = "pphlo.shift_right_logical"(%1778, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1782 = "pphlo.or"(%1780, %1781) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1783 = "pphlo.xor"(%1779, %1782) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1784 = "pphlo.add"(%1779, %1783) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1785 = "pphlo.shift_left"(%1783, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1786 = "pphlo.shift_right_logical"(%1783, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1787 = "pphlo.or"(%1785, %1786) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1788 = "pphlo.xor"(%1784, %1787) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1789 = "pphlo.add"(%1784, %1788) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1790 = "pphlo.add"(%1789, %417) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1791 = "pphlo.shift_left"(%1788, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1792 = "pphlo.shift_right_logical"(%1788, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1793 = "pphlo.or"(%1791, %1792) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1794 = "pphlo.xor"(%1789, %1793) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1795 = "pphlo.add"(%1794, %387) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1796 = "pphlo.add"(%1795, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1797 = "pphlo.add"(%1790, %1796) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1798 = "pphlo.shift_left"(%1796, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1799 = "pphlo.shift_right_logical"(%1796, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1800 = "pphlo.or"(%1798, %1799) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1801 = "pphlo.xor"(%1797, %1800) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1802 = "pphlo.add"(%1797, %1801) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1803 = "pphlo.shift_left"(%1801, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1804 = "pphlo.shift_right_logical"(%1801, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1805 = "pphlo.or"(%1803, %1804) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1806 = "pphlo.xor"(%1802, %1805) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1807 = "pphlo.add"(%1802, %1806) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1808 = "pphlo.shift_left"(%1806, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1809 = "pphlo.shift_right_logical"(%1806, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1810 = "pphlo.or"(%1808, %1809) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1811 = "pphlo.xor"(%1807, %1810) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1812 = "pphlo.add"(%1807, %1811) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1813 = "pphlo.add"(%1795, %69) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1814 = "pphlo.add"(%1812, %1813) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1815 = "pphlo.shift_left"(%1811, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1816 = "pphlo.shift_right_logical"(%1811, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1817 = "pphlo.or"(%1815, %1816) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1818 = "pphlo.xor"(%1812, %1817) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1819 = "pphlo.reshape"(%1790) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %1820 = "pphlo.reshape"(%1813) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %1821 = "pphlo.xor"(%1819, %1820) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %1822 = "pphlo.xor"(%1821, %68) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %1823 = "pphlo.reshape"(%1822) : (tensor<!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1824 = "pphlo.add"(%1818, %1823) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1825 = "pphlo.add"(%1824, %83) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1826 = "pphlo.add"(%1814, %1825) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1827 = "pphlo.shift_left"(%1825, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1828 = "pphlo.shift_right_logical"(%1825, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1829 = "pphlo.or"(%1827, %1828) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1830 = "pphlo.xor"(%1826, %1829) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1831 = "pphlo.add"(%1826, %1830) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1832 = "pphlo.shift_left"(%1830, %81) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1833 = "pphlo.shift_right_logical"(%1830, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1834 = "pphlo.or"(%1832, %1833) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1835 = "pphlo.xor"(%1831, %1834) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1836 = "pphlo.add"(%1831, %1835) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1837 = "pphlo.shift_left"(%1835, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1838 = "pphlo.shift_right_logical"(%1835, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1839 = "pphlo.or"(%1837, %1838) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1840 = "pphlo.xor"(%1836, %1839) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1841 = "pphlo.add"(%1836, %1840) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1842 = "pphlo.add"(%1841, %1823) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1843 = "pphlo.shift_left"(%1840, %78) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1844 = "pphlo.shift_right_logical"(%1840, %77) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1845 = "pphlo.or"(%1843, %1844) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1846 = "pphlo.xor"(%1841, %1845) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1847 = "pphlo.add"(%1846, %1790) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1848 = "pphlo.add"(%1847, %82) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1849 = "pphlo.add"(%1842, %1848) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1850 = "pphlo.shift_left"(%1848, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1851 = "pphlo.shift_right_logical"(%1848, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1852 = "pphlo.or"(%1850, %1851) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1853 = "pphlo.xor"(%1849, %1852) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1854 = "pphlo.add"(%1849, %1853) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1855 = "pphlo.shift_left"(%1853, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1856 = "pphlo.shift_right_logical"(%1853, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1857 = "pphlo.or"(%1855, %1856) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1858 = "pphlo.xor"(%1854, %1857) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1859 = "pphlo.add"(%1854, %1858) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1860 = "pphlo.shift_left"(%1858, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1861 = "pphlo.shift_right_logical"(%1858, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1862 = "pphlo.or"(%1860, %1861) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1863 = "pphlo.xor"(%1859, %1862) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1864 = "pphlo.add"(%1859, %1863) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1865 = "pphlo.add"(%1864, %1790) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1866 = "pphlo.shift_left"(%1863, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1867 = "pphlo.shift_right_logical"(%1863, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1868 = "pphlo.or"(%1866, %1867) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1869 = "pphlo.xor"(%1864, %1868) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1870 = "pphlo.add"(%1869, %1813) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1871 = "pphlo.add"(%1870, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1872 = "pphlo.add"(%1865, %1871) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1873 = "pphlo.shift_left"(%1871, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1874 = "pphlo.shift_right_logical"(%1871, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1875 = "pphlo.or"(%1873, %1874) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1876 = "pphlo.xor"(%1872, %1875) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1877 = "pphlo.add"(%1872, %1876) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1878 = "pphlo.shift_left"(%1876, %81) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1879 = "pphlo.shift_right_logical"(%1876, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1880 = "pphlo.or"(%1878, %1879) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1881 = "pphlo.xor"(%1877, %1880) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1882 = "pphlo.add"(%1877, %1881) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1883 = "pphlo.shift_left"(%1881, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1884 = "pphlo.shift_right_logical"(%1881, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1885 = "pphlo.or"(%1883, %1884) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1886 = "pphlo.xor"(%1882, %1885) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1887 = "pphlo.add"(%1882, %1886) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1888 = "pphlo.add"(%1887, %1813) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1889 = "pphlo.shift_left"(%1886, %78) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1890 = "pphlo.shift_right_logical"(%1886, %77) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1891 = "pphlo.or"(%1889, %1890) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1892 = "pphlo.xor"(%1887, %1891) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1893 = "pphlo.add"(%1892, %1823) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1894 = "pphlo.add"(%1893, %76) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1895 = "pphlo.add"(%1888, %1894) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1896 = "pphlo.shift_left"(%1894, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1897 = "pphlo.shift_right_logical"(%1894, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1898 = "pphlo.or"(%1896, %1897) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1899 = "pphlo.xor"(%1895, %1898) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1900 = "pphlo.add"(%1895, %1899) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1901 = "pphlo.shift_left"(%1899, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1902 = "pphlo.shift_right_logical"(%1899, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1903 = "pphlo.or"(%1901, %1902) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1904 = "pphlo.xor"(%1900, %1903) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1905 = "pphlo.add"(%1900, %1904) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1906 = "pphlo.shift_left"(%1904, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1907 = "pphlo.shift_right_logical"(%1904, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1908 = "pphlo.or"(%1906, %1907) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1909 = "pphlo.xor"(%1905, %1908) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1910 = "pphlo.add"(%1905, %1909) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1911 = "pphlo.add"(%1910, %1823) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1912 = "pphlo.reshape"(%1911) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %1913 = "pphlo.broadcast"(%1912) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1914 = "pphlo.add"(%1680, %1913) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1915 = "pphlo.slice"(%1679) {limit_indices = dense<802816> : tensor<1xi64>, start_indices = dense<401408> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<802816x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1916 = "pphlo.shift_left"(%1909, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1917 = "pphlo.shift_right_logical"(%1909, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1918 = "pphlo.or"(%1916, %1917) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1919 = "pphlo.xor"(%1910, %1918) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1920 = "pphlo.add"(%1919, %1790) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1921 = "pphlo.add"(%1920, %69) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %1922 = "pphlo.reshape"(%1921) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %1923 = "pphlo.broadcast"(%1922) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1924 = "pphlo.add"(%1915, %1923) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1925 = "pphlo.add"(%1914, %1924) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1926 = "pphlo.shift_left"(%1924, %127) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1927 = "pphlo.shift_right_logical"(%1924, %126) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1928 = "pphlo.or"(%1926, %1927) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1929 = "pphlo.xor"(%1925, %1928) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1930 = "pphlo.add"(%1925, %1929) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1931 = "pphlo.shift_left"(%1929, %125) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1932 = "pphlo.shift_right_logical"(%1929, %124) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1933 = "pphlo.or"(%1931, %1932) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1934 = "pphlo.xor"(%1930, %1933) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1935 = "pphlo.add"(%1930, %1934) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1936 = "pphlo.shift_left"(%1934, %122) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1937 = "pphlo.shift_right_logical"(%1934, %123) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1938 = "pphlo.or"(%1936, %1937) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1939 = "pphlo.xor"(%1935, %1938) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1940 = "pphlo.add"(%1935, %1939) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1941 = "pphlo.add"(%1940, %1923) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1942 = "pphlo.shift_left"(%1939, %123) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1943 = "pphlo.shift_right_logical"(%1939, %122) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1944 = "pphlo.or"(%1942, %1943) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1945 = "pphlo.xor"(%1940, %1944) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1946 = "pphlo.xor"(%1912, %1922) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %1947 = "pphlo.xor"(%1946, %68) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %1948 = "pphlo.broadcast"(%1947) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1949 = "pphlo.add"(%1945, %1948) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1950 = "pphlo.add"(%1949, %135) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1951 = "pphlo.add"(%1941, %1950) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1952 = "pphlo.shift_left"(%1950, %124) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1953 = "pphlo.shift_right_logical"(%1950, %125) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1954 = "pphlo.or"(%1952, %1953) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1955 = "pphlo.xor"(%1951, %1954) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1956 = "pphlo.add"(%1951, %1955) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1957 = "pphlo.shift_left"(%1955, %133) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1958 = "pphlo.shift_right_logical"(%1955, %132) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1959 = "pphlo.or"(%1957, %1958) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1960 = "pphlo.xor"(%1956, %1959) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1961 = "pphlo.add"(%1956, %1960) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1962 = "pphlo.shift_left"(%1960, %131) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1963 = "pphlo.shift_right_logical"(%1960, %131) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1964 = "pphlo.or"(%1962, %1963) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1965 = "pphlo.xor"(%1961, %1964) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1966 = "pphlo.add"(%1961, %1965) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1967 = "pphlo.add"(%1966, %1948) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1968 = "pphlo.shift_left"(%1965, %130) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1969 = "pphlo.shift_right_logical"(%1965, %129) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1970 = "pphlo.or"(%1968, %1969) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1971 = "pphlo.xor"(%1966, %1970) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1972 = "pphlo.add"(%1971, %1913) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1973 = "pphlo.add"(%1972, %134) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1974 = "pphlo.add"(%1967, %1973) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1975 = "pphlo.shift_left"(%1973, %127) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1976 = "pphlo.shift_right_logical"(%1973, %126) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1977 = "pphlo.or"(%1975, %1976) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1978 = "pphlo.xor"(%1974, %1977) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1979 = "pphlo.add"(%1974, %1978) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1980 = "pphlo.shift_left"(%1978, %125) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1981 = "pphlo.shift_right_logical"(%1978, %124) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1982 = "pphlo.or"(%1980, %1981) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1983 = "pphlo.xor"(%1979, %1982) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1984 = "pphlo.add"(%1979, %1983) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1985 = "pphlo.shift_left"(%1983, %122) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1986 = "pphlo.shift_right_logical"(%1983, %123) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1987 = "pphlo.or"(%1985, %1986) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1988 = "pphlo.xor"(%1984, %1987) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1989 = "pphlo.add"(%1984, %1988) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1990 = "pphlo.add"(%1989, %1913) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1991 = "pphlo.shift_left"(%1988, %123) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1992 = "pphlo.shift_right_logical"(%1988, %122) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1993 = "pphlo.or"(%1991, %1992) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1994 = "pphlo.xor"(%1989, %1993) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1995 = "pphlo.add"(%1994, %1923) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1996 = "pphlo.add"(%1995, %132) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1997 = "pphlo.add"(%1990, %1996) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1998 = "pphlo.shift_left"(%1996, %124) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %1999 = "pphlo.shift_right_logical"(%1996, %125) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2000 = "pphlo.or"(%1998, %1999) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2001 = "pphlo.xor"(%1997, %2000) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2002 = "pphlo.add"(%1997, %2001) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2003 = "pphlo.shift_left"(%2001, %133) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2004 = "pphlo.shift_right_logical"(%2001, %132) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2005 = "pphlo.or"(%2003, %2004) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2006 = "pphlo.xor"(%2002, %2005) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2007 = "pphlo.add"(%2002, %2006) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2008 = "pphlo.shift_left"(%2006, %131) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2009 = "pphlo.shift_right_logical"(%2006, %131) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2010 = "pphlo.or"(%2008, %2009) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2011 = "pphlo.xor"(%2007, %2010) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2012 = "pphlo.add"(%2007, %2011) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2013 = "pphlo.add"(%2012, %1923) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2014 = "pphlo.shift_left"(%2011, %130) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2015 = "pphlo.shift_right_logical"(%2011, %129) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2016 = "pphlo.or"(%2014, %2015) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2017 = "pphlo.xor"(%2012, %2016) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2018 = "pphlo.add"(%2017, %1948) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2019 = "pphlo.add"(%2018, %128) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2020 = "pphlo.add"(%2013, %2019) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2021 = "pphlo.shift_left"(%2019, %127) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2022 = "pphlo.shift_right_logical"(%2019, %126) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2023 = "pphlo.or"(%2021, %2022) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2024 = "pphlo.xor"(%2020, %2023) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2025 = "pphlo.add"(%2020, %2024) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2026 = "pphlo.shift_left"(%2024, %125) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2027 = "pphlo.shift_right_logical"(%2024, %124) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2028 = "pphlo.or"(%2026, %2027) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2029 = "pphlo.xor"(%2025, %2028) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2030 = "pphlo.add"(%2025, %2029) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2031 = "pphlo.shift_left"(%2029, %122) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2032 = "pphlo.shift_right_logical"(%2029, %123) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2033 = "pphlo.or"(%2031, %2032) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2034 = "pphlo.xor"(%2030, %2033) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2035 = "pphlo.add"(%2030, %2034) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2036 = "pphlo.add"(%2035, %1948) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2037 = "pphlo.shift_left"(%2034, %123) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2038 = "pphlo.shift_right_logical"(%2034, %122) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2039 = "pphlo.or"(%2037, %2038) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2040 = "pphlo.xor"(%2035, %2039) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2041 = "pphlo.add"(%2040, %1913) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2042 = "pphlo.add"(%2041, %121) : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<401408x!pphlo.pint>
    %2043 = "pphlo.concatenate"(%2036, %2042) {dimension = 0 : i64} : (tensor<401408x!pphlo.pint>, tensor<401408x!pphlo.pint>) -> tensor<802816x!pphlo.pint>
    %2044 = "pphlo.shift_right_logical"(%2043, %120) : (tensor<802816x!pphlo.pint>, tensor<802816x!pphlo.pint>) -> tensor<802816x!pphlo.pint>
    %2045 = "pphlo.or"(%2044, %119) : (tensor<802816x!pphlo.pint>, tensor<802816x!pphlo.pint>) -> tensor<802816x!pphlo.pint>
    %2046 = "pphlo.bitcast_convert"(%2045) {elsize = 32 : i64} : (tensor<802816x!pphlo.pint>) -> tensor<802816x!pphlo.pfxp>
    %2047 = "pphlo.add"(%2046, %118) : (tensor<802816x!pphlo.pfxp>, tensor<802816x!pphlo.pfxp>) -> tensor<802816x!pphlo.pfxp>
    %2048 = "pphlo.multiply"(%2047, %117) : (tensor<802816x!pphlo.pfxp>, tensor<802816x!pphlo.pfxp>) -> tensor<802816x!pphlo.pfxp>
    %2049 = "pphlo.add"(%2048, %116) : (tensor<802816x!pphlo.pfxp>, tensor<802816x!pphlo.pfxp>) -> tensor<802816x!pphlo.pfxp>
    %2050 = "pphlo.maximum"(%2049, %116) : (tensor<802816x!pphlo.pfxp>, tensor<802816x!pphlo.pfxp>) -> tensor<802816x!pphlo.pfxp>
    %2051 = "pphlo.reshape"(%2050) : (tensor<802816x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2052 = "pphlo.abs"(%2051) : (tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2053 = "pphlo.equal"(%2052, %115) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pint>
    %2054 = "pphlo.multiply"(%2051, %114) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2055 = "pphlo.negate"(%2051) : (tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2056 = "pphlo.multiply"(%2055, %2051) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2057 = "pphlo.log_plus_one"(%2056) : (tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2058 = "pphlo.negate"(%2057) : (tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2059 = "pphlo.less"(%2058, %113) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pint>
    %2060 = "pphlo.subtract"(%112, %111) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2061 = "pphlo.multiply"(%2059, %2060) : (tensor<3136x256x!pphlo.pint>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2062 = "pphlo.add"(%2061, %111) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2063 = "pphlo.subtract"(%110, %109) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2064 = "pphlo.multiply"(%2059, %2063) : (tensor<3136x256x!pphlo.pint>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2065 = "pphlo.add"(%2064, %109) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2066 = "pphlo.subtract"(%108, %107) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2067 = "pphlo.multiply"(%2059, %2066) : (tensor<3136x256x!pphlo.pint>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2068 = "pphlo.add"(%2067, %107) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2069 = "pphlo.subtract"(%106, %105) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2070 = "pphlo.multiply"(%2059, %2069) : (tensor<3136x256x!pphlo.pint>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2071 = "pphlo.add"(%2070, %105) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2072 = "pphlo.subtract"(%104, %103) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2073 = "pphlo.multiply"(%2059, %2072) : (tensor<3136x256x!pphlo.pint>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2074 = "pphlo.add"(%2073, %103) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2075 = "pphlo.subtract"(%102, %101) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2076 = "pphlo.multiply"(%2059, %2075) : (tensor<3136x256x!pphlo.pint>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2077 = "pphlo.add"(%2076, %101) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2078 = "pphlo.subtract"(%100, %99) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2079 = "pphlo.multiply"(%2059, %2078) : (tensor<3136x256x!pphlo.pint>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2080 = "pphlo.add"(%2079, %99) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2081 = "pphlo.subtract"(%98, %97) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2082 = "pphlo.multiply"(%2059, %2081) : (tensor<3136x256x!pphlo.pint>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2083 = "pphlo.add"(%2082, %97) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2084 = "pphlo.subtract"(%96, %95) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2085 = "pphlo.multiply"(%2059, %2084) : (tensor<3136x256x!pphlo.pint>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2086 = "pphlo.add"(%2085, %95) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2087 = "pphlo.add"(%2058, %94) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2088 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pfxp>
    %2089 = "pphlo.power"(%2058, %2088) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2090 = "pphlo.add"(%2089, %93) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2091 = "pphlo.subtract"(%2087, %2090) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2092 = "pphlo.multiply"(%2059, %2091) : (tensor<3136x256x!pphlo.pint>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2093 = "pphlo.add"(%2092, %2090) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2094 = "pphlo.multiply"(%2086, %2093) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2095 = "pphlo.add"(%2083, %2094) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2096 = "pphlo.multiply"(%2095, %2093) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2097 = "pphlo.add"(%2080, %2096) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2098 = "pphlo.multiply"(%2097, %2093) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2099 = "pphlo.add"(%2077, %2098) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2100 = "pphlo.multiply"(%2099, %2093) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2101 = "pphlo.add"(%2074, %2100) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2102 = "pphlo.multiply"(%2101, %2093) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2103 = "pphlo.add"(%2071, %2102) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2104 = "pphlo.multiply"(%2103, %2093) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2105 = "pphlo.add"(%2068, %2104) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2106 = "pphlo.multiply"(%2105, %2093) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2107 = "pphlo.add"(%2065, %2106) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2108 = "pphlo.multiply"(%2107, %2093) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2109 = "pphlo.add"(%2062, %2108) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2110 = "pphlo.multiply"(%2109, %2051) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2111 = "pphlo.subtract"(%2054, %2110) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2112 = "pphlo.multiply"(%2053, %2111) : (tensor<3136x256x!pphlo.pint>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2113 = "pphlo.add"(%2112, %2110) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2114 = "pphlo.multiply"(%2113, %92) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2115 = "pphlo.clamp"(%137, %2114, %91) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2116 = "pphlo.multiply"(%2115, %90) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2117 = "pphlo.dot"(%1678, %2116) : (tensor<30x3136x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<30x256x!pphlo.pfxp>
    %2118 = "pphlo.greater"(%2117, %21) : (tensor<30x256x!pphlo.pfxp>, tensor<30x256x!pphlo.pfxp>) -> tensor<30x256x!pphlo.pint>
    %2119 = "pphlo.gather"(%arg1, %1228) {dimension_numbers = #pphlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<30x!pphlo.pint>, tensor<30x1x!pphlo.pint>) -> tensor<30x!pphlo.pint>
    %2120 = "pphlo.broadcast"(%2119) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<30x!pphlo.pint>) -> tensor<30x10x!pphlo.pint>
    %2121 = "pphlo.broadcast"(%269) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<10x!pphlo.pint>) -> tensor<30x10x!pphlo.pint>
    %2122 = "pphlo.equal"(%2120, %2121) : (tensor<30x10x!pphlo.pint>, tensor<30x10x!pphlo.pint>) -> tensor<30x10x!pphlo.pint>
    %2123 = "pphlo.subtract"(%87, %86) : (tensor<30x10x!pphlo.pfxp>, tensor<30x10x!pphlo.pfxp>) -> tensor<30x10x!pphlo.pfxp>
    %2124 = "pphlo.multiply"(%2122, %2123) : (tensor<30x10x!pphlo.pint>, tensor<30x10x!pphlo.pfxp>) -> tensor<30x10x!pphlo.pfxp>
    %2125 = "pphlo.add"(%2124, %86) : (tensor<30x10x!pphlo.pfxp>, tensor<30x10x!pphlo.pfxp>) -> tensor<30x10x!pphlo.pfxp>
    %2126 = "pphlo.negate"(%2125) : (tensor<30x10x!pphlo.pfxp>) -> tensor<30x10x!pphlo.pfxp>
    %2127 = "pphlo.reduce"(%2126, %1) ( {
    ^bb0(%arg4: tensor<!pphlo.pfxp>, %arg5: tensor<!pphlo.pfxp>):  // no predecessors
      %2655 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%2655) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<30x10x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<30x!pphlo.pfxp>
    %2128 = "pphlo.maximum"(%2117, %21) : (tensor<30x256x!pphlo.pfxp>, tensor<30x256x!pphlo.pfxp>) -> tensor<30x256x!pphlo.pfxp>
    %2129 = "pphlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<2560x!pphlo.pint>
    %2130 = "pphlo.slice"(%2129) {limit_indices = dense<1280> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2560x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2131 = "pphlo.add"(%390, %84) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2132 = "pphlo.add"(%387, %2131) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2133 = "pphlo.shift_left"(%2131, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2134 = "pphlo.shift_right_logical"(%2131, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2135 = "pphlo.or"(%2133, %2134) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2136 = "pphlo.xor"(%2132, %2135) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2137 = "pphlo.add"(%2132, %2136) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2138 = "pphlo.shift_left"(%2136, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2139 = "pphlo.shift_right_logical"(%2136, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2140 = "pphlo.or"(%2138, %2139) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2141 = "pphlo.xor"(%2137, %2140) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2142 = "pphlo.add"(%2137, %2141) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2143 = "pphlo.shift_left"(%2141, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2144 = "pphlo.shift_right_logical"(%2141, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2145 = "pphlo.or"(%2143, %2144) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2146 = "pphlo.xor"(%2142, %2145) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2147 = "pphlo.add"(%2142, %2146) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2148 = "pphlo.add"(%2147, %390) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2149 = "pphlo.shift_left"(%2146, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2150 = "pphlo.shift_right_logical"(%2146, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2151 = "pphlo.or"(%2149, %2150) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2152 = "pphlo.xor"(%2147, %2151) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2153 = "pphlo.add"(%2152, %417) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2154 = "pphlo.add"(%2153, %83) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2155 = "pphlo.add"(%2148, %2154) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2156 = "pphlo.shift_left"(%2154, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2157 = "pphlo.shift_right_logical"(%2154, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2158 = "pphlo.or"(%2156, %2157) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2159 = "pphlo.xor"(%2155, %2158) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2160 = "pphlo.add"(%2155, %2159) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2161 = "pphlo.shift_left"(%2159, %81) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2162 = "pphlo.shift_right_logical"(%2159, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2163 = "pphlo.or"(%2161, %2162) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2164 = "pphlo.xor"(%2160, %2163) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2165 = "pphlo.add"(%2160, %2164) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2166 = "pphlo.shift_left"(%2164, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2167 = "pphlo.shift_right_logical"(%2164, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2168 = "pphlo.or"(%2166, %2167) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2169 = "pphlo.xor"(%2165, %2168) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2170 = "pphlo.add"(%2165, %2169) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2171 = "pphlo.add"(%2170, %417) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2172 = "pphlo.shift_left"(%2169, %78) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2173 = "pphlo.shift_right_logical"(%2169, %77) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2174 = "pphlo.or"(%2172, %2173) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2175 = "pphlo.xor"(%2170, %2174) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2176 = "pphlo.add"(%2175, %387) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2177 = "pphlo.add"(%2176, %82) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2178 = "pphlo.add"(%2171, %2177) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2179 = "pphlo.shift_left"(%2177, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2180 = "pphlo.shift_right_logical"(%2177, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2181 = "pphlo.or"(%2179, %2180) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2182 = "pphlo.xor"(%2178, %2181) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2183 = "pphlo.add"(%2178, %2182) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2184 = "pphlo.shift_left"(%2182, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2185 = "pphlo.shift_right_logical"(%2182, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2186 = "pphlo.or"(%2184, %2185) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2187 = "pphlo.xor"(%2183, %2186) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2188 = "pphlo.add"(%2183, %2187) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2189 = "pphlo.shift_left"(%2187, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2190 = "pphlo.shift_right_logical"(%2187, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2191 = "pphlo.or"(%2189, %2190) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2192 = "pphlo.xor"(%2188, %2191) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2193 = "pphlo.add"(%2188, %2192) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2194 = "pphlo.add"(%2193, %387) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2195 = "pphlo.shift_left"(%2192, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2196 = "pphlo.shift_right_logical"(%2192, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2197 = "pphlo.or"(%2195, %2196) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2198 = "pphlo.xor"(%2193, %2197) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2199 = "pphlo.add"(%2198, %390) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2200 = "pphlo.add"(%2199, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2201 = "pphlo.add"(%2194, %2200) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2202 = "pphlo.shift_left"(%2200, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2203 = "pphlo.shift_right_logical"(%2200, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2204 = "pphlo.or"(%2202, %2203) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2205 = "pphlo.xor"(%2201, %2204) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2206 = "pphlo.add"(%2201, %2205) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2207 = "pphlo.shift_left"(%2205, %81) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2208 = "pphlo.shift_right_logical"(%2205, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2209 = "pphlo.or"(%2207, %2208) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2210 = "pphlo.xor"(%2206, %2209) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2211 = "pphlo.add"(%2206, %2210) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2212 = "pphlo.shift_left"(%2210, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2213 = "pphlo.shift_right_logical"(%2210, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2214 = "pphlo.or"(%2212, %2213) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2215 = "pphlo.xor"(%2211, %2214) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2216 = "pphlo.add"(%2211, %2215) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2217 = "pphlo.add"(%2216, %390) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2218 = "pphlo.shift_left"(%2215, %78) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2219 = "pphlo.shift_right_logical"(%2215, %77) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2220 = "pphlo.or"(%2218, %2219) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2221 = "pphlo.xor"(%2216, %2220) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2222 = "pphlo.add"(%2221, %417) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2223 = "pphlo.add"(%2222, %76) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2224 = "pphlo.add"(%2217, %2223) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2225 = "pphlo.shift_left"(%2223, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2226 = "pphlo.shift_right_logical"(%2223, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2227 = "pphlo.or"(%2225, %2226) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2228 = "pphlo.xor"(%2224, %2227) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2229 = "pphlo.add"(%2224, %2228) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2230 = "pphlo.shift_left"(%2228, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2231 = "pphlo.shift_right_logical"(%2228, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2232 = "pphlo.or"(%2230, %2231) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2233 = "pphlo.xor"(%2229, %2232) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2234 = "pphlo.add"(%2229, %2233) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2235 = "pphlo.shift_left"(%2233, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2236 = "pphlo.shift_right_logical"(%2233, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2237 = "pphlo.or"(%2235, %2236) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2238 = "pphlo.xor"(%2234, %2237) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2239 = "pphlo.add"(%2234, %2238) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2240 = "pphlo.add"(%2239, %417) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2241 = "pphlo.shift_left"(%2238, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2242 = "pphlo.shift_right_logical"(%2238, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2243 = "pphlo.or"(%2241, %2242) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2244 = "pphlo.xor"(%2239, %2243) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2245 = "pphlo.add"(%2244, %387) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2246 = "pphlo.add"(%2245, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2247 = "pphlo.add"(%2240, %2246) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2248 = "pphlo.shift_left"(%2246, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2249 = "pphlo.shift_right_logical"(%2246, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2250 = "pphlo.or"(%2248, %2249) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2251 = "pphlo.xor"(%2247, %2250) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2252 = "pphlo.add"(%2247, %2251) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2253 = "pphlo.shift_left"(%2251, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2254 = "pphlo.shift_right_logical"(%2251, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2255 = "pphlo.or"(%2253, %2254) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2256 = "pphlo.xor"(%2252, %2255) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2257 = "pphlo.add"(%2252, %2256) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2258 = "pphlo.shift_left"(%2256, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2259 = "pphlo.shift_right_logical"(%2256, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2260 = "pphlo.or"(%2258, %2259) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2261 = "pphlo.xor"(%2257, %2260) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2262 = "pphlo.add"(%2257, %2261) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2263 = "pphlo.add"(%2245, %69) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2264 = "pphlo.add"(%2262, %2263) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2265 = "pphlo.shift_left"(%2261, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2266 = "pphlo.shift_right_logical"(%2261, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2267 = "pphlo.or"(%2265, %2266) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2268 = "pphlo.xor"(%2262, %2267) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2269 = "pphlo.reshape"(%2240) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %2270 = "pphlo.reshape"(%2263) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %2271 = "pphlo.xor"(%2269, %2270) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %2272 = "pphlo.xor"(%2271, %68) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %2273 = "pphlo.reshape"(%2272) : (tensor<!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2274 = "pphlo.add"(%2268, %2273) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2275 = "pphlo.add"(%2274, %83) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2276 = "pphlo.add"(%2264, %2275) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2277 = "pphlo.shift_left"(%2275, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2278 = "pphlo.shift_right_logical"(%2275, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2279 = "pphlo.or"(%2277, %2278) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2280 = "pphlo.xor"(%2276, %2279) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2281 = "pphlo.add"(%2276, %2280) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2282 = "pphlo.shift_left"(%2280, %81) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2283 = "pphlo.shift_right_logical"(%2280, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2284 = "pphlo.or"(%2282, %2283) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2285 = "pphlo.xor"(%2281, %2284) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2286 = "pphlo.add"(%2281, %2285) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2287 = "pphlo.shift_left"(%2285, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2288 = "pphlo.shift_right_logical"(%2285, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2289 = "pphlo.or"(%2287, %2288) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2290 = "pphlo.xor"(%2286, %2289) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2291 = "pphlo.add"(%2286, %2290) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2292 = "pphlo.add"(%2291, %2273) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2293 = "pphlo.shift_left"(%2290, %78) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2294 = "pphlo.shift_right_logical"(%2290, %77) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2295 = "pphlo.or"(%2293, %2294) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2296 = "pphlo.xor"(%2291, %2295) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2297 = "pphlo.add"(%2296, %2240) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2298 = "pphlo.add"(%2297, %82) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2299 = "pphlo.add"(%2292, %2298) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2300 = "pphlo.shift_left"(%2298, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2301 = "pphlo.shift_right_logical"(%2298, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2302 = "pphlo.or"(%2300, %2301) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2303 = "pphlo.xor"(%2299, %2302) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2304 = "pphlo.add"(%2299, %2303) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2305 = "pphlo.shift_left"(%2303, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2306 = "pphlo.shift_right_logical"(%2303, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2307 = "pphlo.or"(%2305, %2306) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2308 = "pphlo.xor"(%2304, %2307) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2309 = "pphlo.add"(%2304, %2308) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2310 = "pphlo.shift_left"(%2308, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2311 = "pphlo.shift_right_logical"(%2308, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2312 = "pphlo.or"(%2310, %2311) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2313 = "pphlo.xor"(%2309, %2312) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2314 = "pphlo.add"(%2309, %2313) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2315 = "pphlo.add"(%2314, %2240) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2316 = "pphlo.shift_left"(%2313, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2317 = "pphlo.shift_right_logical"(%2313, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2318 = "pphlo.or"(%2316, %2317) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2319 = "pphlo.xor"(%2314, %2318) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2320 = "pphlo.add"(%2319, %2263) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2321 = "pphlo.add"(%2320, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2322 = "pphlo.add"(%2315, %2321) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2323 = "pphlo.shift_left"(%2321, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2324 = "pphlo.shift_right_logical"(%2321, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2325 = "pphlo.or"(%2323, %2324) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2326 = "pphlo.xor"(%2322, %2325) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2327 = "pphlo.add"(%2322, %2326) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2328 = "pphlo.shift_left"(%2326, %81) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2329 = "pphlo.shift_right_logical"(%2326, %80) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2330 = "pphlo.or"(%2328, %2329) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2331 = "pphlo.xor"(%2327, %2330) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2332 = "pphlo.add"(%2327, %2331) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2333 = "pphlo.shift_left"(%2331, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2334 = "pphlo.shift_right_logical"(%2331, %79) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2335 = "pphlo.or"(%2333, %2334) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2336 = "pphlo.xor"(%2332, %2335) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2337 = "pphlo.add"(%2332, %2336) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2338 = "pphlo.add"(%2337, %2263) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2339 = "pphlo.shift_left"(%2336, %78) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2340 = "pphlo.shift_right_logical"(%2336, %77) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2341 = "pphlo.or"(%2339, %2340) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2342 = "pphlo.xor"(%2337, %2341) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2343 = "pphlo.add"(%2342, %2273) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2344 = "pphlo.add"(%2343, %76) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2345 = "pphlo.add"(%2338, %2344) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2346 = "pphlo.shift_left"(%2344, %75) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2347 = "pphlo.shift_right_logical"(%2344, %74) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2348 = "pphlo.or"(%2346, %2347) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2349 = "pphlo.xor"(%2345, %2348) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2350 = "pphlo.add"(%2345, %2349) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2351 = "pphlo.shift_left"(%2349, %73) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2352 = "pphlo.shift_right_logical"(%2349, %72) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2353 = "pphlo.or"(%2351, %2352) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2354 = "pphlo.xor"(%2350, %2353) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2355 = "pphlo.add"(%2350, %2354) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2356 = "pphlo.shift_left"(%2354, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2357 = "pphlo.shift_right_logical"(%2354, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2358 = "pphlo.or"(%2356, %2357) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2359 = "pphlo.xor"(%2355, %2358) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2360 = "pphlo.add"(%2355, %2359) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2361 = "pphlo.add"(%2360, %2273) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2362 = "pphlo.reshape"(%2361) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %2363 = "pphlo.broadcast"(%2362) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2364 = "pphlo.add"(%2130, %2363) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2365 = "pphlo.slice"(%2129) {limit_indices = dense<2560> : tensor<1xi64>, start_indices = dense<1280> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2560x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2366 = "pphlo.shift_left"(%2359, %71) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2367 = "pphlo.shift_right_logical"(%2359, %70) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2368 = "pphlo.or"(%2366, %2367) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2369 = "pphlo.xor"(%2360, %2368) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2370 = "pphlo.add"(%2369, %2240) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2371 = "pphlo.add"(%2370, %69) : (tensor<1x!pphlo.pint>, tensor<1x!pphlo.pint>) -> tensor<1x!pphlo.pint>
    %2372 = "pphlo.reshape"(%2371) : (tensor<1x!pphlo.pint>) -> tensor<!pphlo.pint>
    %2373 = "pphlo.broadcast"(%2372) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2374 = "pphlo.add"(%2365, %2373) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2375 = "pphlo.add"(%2364, %2374) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2376 = "pphlo.shift_left"(%2374, %59) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2377 = "pphlo.shift_right_logical"(%2374, %58) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2378 = "pphlo.or"(%2376, %2377) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2379 = "pphlo.xor"(%2375, %2378) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2380 = "pphlo.add"(%2375, %2379) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2381 = "pphlo.shift_left"(%2379, %57) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2382 = "pphlo.shift_right_logical"(%2379, %56) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2383 = "pphlo.or"(%2381, %2382) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2384 = "pphlo.xor"(%2380, %2383) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2385 = "pphlo.add"(%2380, %2384) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2386 = "pphlo.shift_left"(%2384, %54) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2387 = "pphlo.shift_right_logical"(%2384, %55) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2388 = "pphlo.or"(%2386, %2387) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2389 = "pphlo.xor"(%2385, %2388) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2390 = "pphlo.add"(%2385, %2389) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2391 = "pphlo.add"(%2390, %2373) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2392 = "pphlo.shift_left"(%2389, %55) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2393 = "pphlo.shift_right_logical"(%2389, %54) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2394 = "pphlo.or"(%2392, %2393) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2395 = "pphlo.xor"(%2390, %2394) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2396 = "pphlo.xor"(%2362, %2372) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %2397 = "pphlo.xor"(%2396, %68) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    %2398 = "pphlo.broadcast"(%2397) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2399 = "pphlo.add"(%2395, %2398) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2400 = "pphlo.add"(%2399, %67) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2401 = "pphlo.add"(%2391, %2400) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2402 = "pphlo.shift_left"(%2400, %56) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2403 = "pphlo.shift_right_logical"(%2400, %57) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2404 = "pphlo.or"(%2402, %2403) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2405 = "pphlo.xor"(%2401, %2404) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2406 = "pphlo.add"(%2401, %2405) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2407 = "pphlo.shift_left"(%2405, %65) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2408 = "pphlo.shift_right_logical"(%2405, %64) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2409 = "pphlo.or"(%2407, %2408) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2410 = "pphlo.xor"(%2406, %2409) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2411 = "pphlo.add"(%2406, %2410) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2412 = "pphlo.shift_left"(%2410, %63) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2413 = "pphlo.shift_right_logical"(%2410, %63) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2414 = "pphlo.or"(%2412, %2413) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2415 = "pphlo.xor"(%2411, %2414) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2416 = "pphlo.add"(%2411, %2415) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2417 = "pphlo.add"(%2416, %2398) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2418 = "pphlo.shift_left"(%2415, %62) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2419 = "pphlo.shift_right_logical"(%2415, %61) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2420 = "pphlo.or"(%2418, %2419) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2421 = "pphlo.xor"(%2416, %2420) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2422 = "pphlo.add"(%2421, %2363) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2423 = "pphlo.add"(%2422, %66) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2424 = "pphlo.add"(%2417, %2423) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2425 = "pphlo.shift_left"(%2423, %59) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2426 = "pphlo.shift_right_logical"(%2423, %58) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2427 = "pphlo.or"(%2425, %2426) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2428 = "pphlo.xor"(%2424, %2427) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2429 = "pphlo.add"(%2424, %2428) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2430 = "pphlo.shift_left"(%2428, %57) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2431 = "pphlo.shift_right_logical"(%2428, %56) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2432 = "pphlo.or"(%2430, %2431) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2433 = "pphlo.xor"(%2429, %2432) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2434 = "pphlo.add"(%2429, %2433) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2435 = "pphlo.shift_left"(%2433, %54) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2436 = "pphlo.shift_right_logical"(%2433, %55) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2437 = "pphlo.or"(%2435, %2436) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2438 = "pphlo.xor"(%2434, %2437) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2439 = "pphlo.add"(%2434, %2438) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2440 = "pphlo.add"(%2439, %2363) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2441 = "pphlo.shift_left"(%2438, %55) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2442 = "pphlo.shift_right_logical"(%2438, %54) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2443 = "pphlo.or"(%2441, %2442) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2444 = "pphlo.xor"(%2439, %2443) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2445 = "pphlo.add"(%2444, %2373) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2446 = "pphlo.add"(%2445, %64) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2447 = "pphlo.add"(%2440, %2446) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2448 = "pphlo.shift_left"(%2446, %56) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2449 = "pphlo.shift_right_logical"(%2446, %57) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2450 = "pphlo.or"(%2448, %2449) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2451 = "pphlo.xor"(%2447, %2450) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2452 = "pphlo.add"(%2447, %2451) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2453 = "pphlo.shift_left"(%2451, %65) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2454 = "pphlo.shift_right_logical"(%2451, %64) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2455 = "pphlo.or"(%2453, %2454) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2456 = "pphlo.xor"(%2452, %2455) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2457 = "pphlo.add"(%2452, %2456) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2458 = "pphlo.shift_left"(%2456, %63) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2459 = "pphlo.shift_right_logical"(%2456, %63) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2460 = "pphlo.or"(%2458, %2459) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2461 = "pphlo.xor"(%2457, %2460) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2462 = "pphlo.add"(%2457, %2461) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2463 = "pphlo.add"(%2462, %2373) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2464 = "pphlo.shift_left"(%2461, %62) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2465 = "pphlo.shift_right_logical"(%2461, %61) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2466 = "pphlo.or"(%2464, %2465) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2467 = "pphlo.xor"(%2462, %2466) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2468 = "pphlo.add"(%2467, %2398) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2469 = "pphlo.add"(%2468, %60) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2470 = "pphlo.add"(%2463, %2469) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2471 = "pphlo.shift_left"(%2469, %59) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2472 = "pphlo.shift_right_logical"(%2469, %58) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2473 = "pphlo.or"(%2471, %2472) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2474 = "pphlo.xor"(%2470, %2473) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2475 = "pphlo.add"(%2470, %2474) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2476 = "pphlo.shift_left"(%2474, %57) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2477 = "pphlo.shift_right_logical"(%2474, %56) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2478 = "pphlo.or"(%2476, %2477) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2479 = "pphlo.xor"(%2475, %2478) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2480 = "pphlo.add"(%2475, %2479) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2481 = "pphlo.shift_left"(%2479, %54) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2482 = "pphlo.shift_right_logical"(%2479, %55) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2483 = "pphlo.or"(%2481, %2482) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2484 = "pphlo.xor"(%2480, %2483) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2485 = "pphlo.add"(%2480, %2484) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2486 = "pphlo.add"(%2485, %2398) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2487 = "pphlo.shift_left"(%2484, %55) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2488 = "pphlo.shift_right_logical"(%2484, %54) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2489 = "pphlo.or"(%2487, %2488) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2490 = "pphlo.xor"(%2485, %2489) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2491 = "pphlo.add"(%2490, %2363) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2492 = "pphlo.add"(%2491, %53) : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<1280x!pphlo.pint>
    %2493 = "pphlo.concatenate"(%2486, %2492) {dimension = 0 : i64} : (tensor<1280x!pphlo.pint>, tensor<1280x!pphlo.pint>) -> tensor<2560x!pphlo.pint>
    %2494 = "pphlo.shift_right_logical"(%2493, %52) : (tensor<2560x!pphlo.pint>, tensor<2560x!pphlo.pint>) -> tensor<2560x!pphlo.pint>
    %2495 = "pphlo.or"(%2494, %51) : (tensor<2560x!pphlo.pint>, tensor<2560x!pphlo.pint>) -> tensor<2560x!pphlo.pint>
    %2496 = "pphlo.bitcast_convert"(%2495) {elsize = 32 : i64} : (tensor<2560x!pphlo.pint>) -> tensor<2560x!pphlo.pfxp>
    %2497 = "pphlo.add"(%2496, %50) : (tensor<2560x!pphlo.pfxp>, tensor<2560x!pphlo.pfxp>) -> tensor<2560x!pphlo.pfxp>
    %2498 = "pphlo.multiply"(%2497, %49) : (tensor<2560x!pphlo.pfxp>, tensor<2560x!pphlo.pfxp>) -> tensor<2560x!pphlo.pfxp>
    %2499 = "pphlo.add"(%2498, %48) : (tensor<2560x!pphlo.pfxp>, tensor<2560x!pphlo.pfxp>) -> tensor<2560x!pphlo.pfxp>
    %2500 = "pphlo.maximum"(%2499, %48) : (tensor<2560x!pphlo.pfxp>, tensor<2560x!pphlo.pfxp>) -> tensor<2560x!pphlo.pfxp>
    %2501 = "pphlo.reshape"(%2500) : (tensor<2560x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2502 = "pphlo.abs"(%2501) : (tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2503 = "pphlo.equal"(%2502, %47) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pint>
    %2504 = "pphlo.multiply"(%2501, %46) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2505 = "pphlo.negate"(%2501) : (tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2506 = "pphlo.multiply"(%2505, %2501) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2507 = "pphlo.log_plus_one"(%2506) : (tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2508 = "pphlo.negate"(%2507) : (tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2509 = "pphlo.less"(%2508, %45) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pint>
    %2510 = "pphlo.subtract"(%44, %43) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2511 = "pphlo.multiply"(%2509, %2510) : (tensor<256x10x!pphlo.pint>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2512 = "pphlo.add"(%2511, %43) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2513 = "pphlo.subtract"(%42, %41) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2514 = "pphlo.multiply"(%2509, %2513) : (tensor<256x10x!pphlo.pint>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2515 = "pphlo.add"(%2514, %41) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2516 = "pphlo.subtract"(%40, %39) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2517 = "pphlo.multiply"(%2509, %2516) : (tensor<256x10x!pphlo.pint>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2518 = "pphlo.add"(%2517, %39) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2519 = "pphlo.subtract"(%38, %37) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2520 = "pphlo.multiply"(%2509, %2519) : (tensor<256x10x!pphlo.pint>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2521 = "pphlo.add"(%2520, %37) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2522 = "pphlo.subtract"(%36, %35) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2523 = "pphlo.multiply"(%2509, %2522) : (tensor<256x10x!pphlo.pint>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2524 = "pphlo.add"(%2523, %35) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2525 = "pphlo.subtract"(%34, %33) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2526 = "pphlo.multiply"(%2509, %2525) : (tensor<256x10x!pphlo.pint>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2527 = "pphlo.add"(%2526, %33) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2528 = "pphlo.subtract"(%32, %31) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2529 = "pphlo.multiply"(%2509, %2528) : (tensor<256x10x!pphlo.pint>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2530 = "pphlo.add"(%2529, %31) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2531 = "pphlo.subtract"(%30, %29) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2532 = "pphlo.multiply"(%2509, %2531) : (tensor<256x10x!pphlo.pint>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2533 = "pphlo.add"(%2532, %29) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2534 = "pphlo.subtract"(%28, %27) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2535 = "pphlo.multiply"(%2509, %2534) : (tensor<256x10x!pphlo.pint>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2536 = "pphlo.add"(%2535, %27) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2537 = "pphlo.add"(%2508, %26) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2538 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pfxp>
    %2539 = "pphlo.power"(%2508, %2538) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2540 = "pphlo.add"(%2539, %25) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2541 = "pphlo.subtract"(%2537, %2540) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2542 = "pphlo.multiply"(%2509, %2541) : (tensor<256x10x!pphlo.pint>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2543 = "pphlo.add"(%2542, %2540) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2544 = "pphlo.multiply"(%2536, %2543) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2545 = "pphlo.add"(%2533, %2544) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2546 = "pphlo.multiply"(%2545, %2543) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2547 = "pphlo.add"(%2530, %2546) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2548 = "pphlo.multiply"(%2547, %2543) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2549 = "pphlo.add"(%2527, %2548) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2550 = "pphlo.multiply"(%2549, %2543) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2551 = "pphlo.add"(%2524, %2550) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2552 = "pphlo.multiply"(%2551, %2543) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2553 = "pphlo.add"(%2521, %2552) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2554 = "pphlo.multiply"(%2553, %2543) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2555 = "pphlo.add"(%2518, %2554) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2556 = "pphlo.multiply"(%2555, %2543) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2557 = "pphlo.add"(%2515, %2556) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2558 = "pphlo.multiply"(%2557, %2543) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2559 = "pphlo.add"(%2512, %2558) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2560 = "pphlo.multiply"(%2559, %2501) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2561 = "pphlo.subtract"(%2504, %2560) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2562 = "pphlo.multiply"(%2503, %2561) : (tensor<256x10x!pphlo.pint>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2563 = "pphlo.add"(%2562, %2560) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2564 = "pphlo.multiply"(%2563, %24) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2565 = "pphlo.clamp"(%85, %2564, %23) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2566 = "pphlo.multiply"(%2565, %22) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2567 = "pphlo.dot"(%2128, %2566) : (tensor<30x256x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<30x10x!pphlo.pfxp>
    %2568 = "pphlo.reduce"(%2567, %3) ( {
    ^bb0(%arg4: tensor<!pphlo.pfxp>, %arg5: tensor<!pphlo.pfxp>):  // no predecessors
      %2655 = "pphlo.maximum"(%arg4, %arg5) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%2655) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<30x10x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<30x!pphlo.pfxp>
    %2569 = "pphlo.broadcast"(%2568) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<30x!pphlo.pfxp>) -> tensor<30x10x!pphlo.pfxp>
    %2570 = "pphlo.subtract"(%2567, %2569) : (tensor<30x10x!pphlo.pfxp>, tensor<30x10x!pphlo.pfxp>) -> tensor<30x10x!pphlo.pfxp>
    %2571 = "pphlo.exponential"(%2570) : (tensor<30x10x!pphlo.pfxp>) -> tensor<30x10x!pphlo.pfxp>
    %2572 = "pphlo.reduce"(%2571, %1) ( {
    ^bb0(%arg4: tensor<!pphlo.pfxp>, %arg5: tensor<!pphlo.pfxp>):  // no predecessors
      %2655 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%2655) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<30x10x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<30x!pphlo.pfxp>
    %2573 = "pphlo.reciprocal"(%2572) : (tensor<30x!pphlo.pfxp>) -> tensor<30x!pphlo.pfxp>
    %2574 = "pphlo.multiply"(%2127, %2573) : (tensor<30x!pphlo.pfxp>, tensor<30x!pphlo.pfxp>) -> tensor<30x!pphlo.pfxp>
    %2575 = "pphlo.broadcast"(%2574) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<30x!pphlo.pfxp>) -> tensor<30x10x!pphlo.pfxp>
    %2576 = "pphlo.multiply"(%2575, %2571) : (tensor<30x10x!pphlo.pfxp>, tensor<30x10x!pphlo.pfxp>) -> tensor<30x10x!pphlo.pfxp>
    %2577 = "pphlo.add"(%2125, %2576) : (tensor<30x10x!pphlo.pfxp>, tensor<30x10x!pphlo.pfxp>) -> tensor<30x10x!pphlo.pfxp>
    %2578 = "pphlo.transpose"(%2566) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x10x!pphlo.pfxp>) -> tensor<10x256x!pphlo.pfxp>
    %2579 = "pphlo.dot"(%2577, %2578) : (tensor<30x10x!pphlo.pfxp>, tensor<10x256x!pphlo.pfxp>) -> tensor<30x256x!pphlo.pfxp>
    %2580 = "pphlo.subtract"(%2579, %21) : (tensor<30x256x!pphlo.pfxp>, tensor<30x256x!pphlo.pfxp>) -> tensor<30x256x!pphlo.pfxp>
    %2581 = "pphlo.multiply"(%2118, %2580) : (tensor<30x256x!pphlo.pint>, tensor<30x256x!pphlo.pfxp>) -> tensor<30x256x!pphlo.pfxp>
    %2582 = "pphlo.add"(%2581, %21) : (tensor<30x256x!pphlo.pfxp>, tensor<30x256x!pphlo.pfxp>) -> tensor<30x256x!pphlo.pfxp>
    %2583 = "pphlo.transpose"(%2116) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<3136x256x!pphlo.pfxp>) -> tensor<256x3136x!pphlo.pfxp>
    %2584 = "pphlo.dot"(%2582, %2583) : (tensor<30x256x!pphlo.pfxp>, tensor<256x3136x!pphlo.pfxp>) -> tensor<30x3136x!pphlo.pfxp>
    %2585 = "pphlo.multiply"(%2584, %20) : (tensor<30x3136x!pphlo.pfxp>, tensor<30x3136x!pphlo.pfxp>) -> tensor<30x3136x!pphlo.pfxp>
    %2586 = "pphlo.reshape"(%2585) : (tensor<30x3136x!pphlo.pfxp>) -> tensor<30x7x7x64x!pphlo.pfxp>
    %2587 = "pphlo.reduce_window"(%2586, %1) ( {
    ^bb0(%arg4: tensor<!pphlo.pfxp>, %arg5: tensor<!pphlo.pfxp>):  // no predecessors
      %2655 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%2655) : (tensor<!pphlo.pfxp>) -> ()
    }) {base_dilations = dense<[1, 2, 2, 1]> : tensor<4xi64>, padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<30x7x7x64x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<30x14x14x64x!pphlo.pfxp>
    %2588 = "pphlo.subtract"(%2587, %19) : (tensor<30x14x14x64x!pphlo.pfxp>, tensor<30x14x14x64x!pphlo.pfxp>) -> tensor<30x14x14x64x!pphlo.pfxp>
    %2589 = "pphlo.multiply"(%1674, %2588) : (tensor<30x14x14x64x!pphlo.pint>, tensor<30x14x14x64x!pphlo.pfxp>) -> tensor<30x14x14x64x!pphlo.pfxp>
    %2590 = "pphlo.add"(%2589, %19) : (tensor<30x14x14x64x!pphlo.pfxp>, tensor<30x14x14x64x!pphlo.pfxp>) -> tensor<30x14x14x64x!pphlo.pfxp>
    %2591 = "pphlo.reverse"(%1672) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %2592 = pphlo.convolution(%2590, %2591) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<30x14x14x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<30x14x14x32x!pphlo.pfxp>
    %2593 = "pphlo.multiply"(%2592, %18) : (tensor<30x14x14x32x!pphlo.pfxp>, tensor<30x14x14x32x!pphlo.pfxp>) -> tensor<30x14x14x32x!pphlo.pfxp>
    %2594 = "pphlo.reduce_window"(%2593, %1) ( {
    ^bb0(%arg4: tensor<!pphlo.pfxp>, %arg5: tensor<!pphlo.pfxp>):  // no predecessors
      %2655 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%2655) : (tensor<!pphlo.pfxp>) -> ()
    }) {base_dilations = dense<[1, 2, 2, 1]> : tensor<4xi64>, padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<30x14x14x32x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<30x28x28x32x!pphlo.pfxp>
    %2595 = "pphlo.subtract"(%2594, %17) : (tensor<30x28x28x32x!pphlo.pfxp>, tensor<30x28x28x32x!pphlo.pfxp>) -> tensor<30x28x28x32x!pphlo.pfxp>
    %2596 = "pphlo.multiply"(%1231, %2595) : (tensor<30x28x28x32x!pphlo.pint>, tensor<30x28x28x32x!pphlo.pfxp>) -> tensor<30x28x28x32x!pphlo.pfxp>
    %2597 = "pphlo.add"(%2596, %17) : (tensor<30x28x28x32x!pphlo.pfxp>, tensor<30x28x28x32x!pphlo.pfxp>) -> tensor<30x28x28x32x!pphlo.pfxp>
    %2598 = pphlo.convolution(%1229, %2597) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<30x28x28x1x!pphlo.pfxp>, tensor<30x28x28x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %2599 = "pphlo.multiply"(%2598, %16) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %2600 = "pphlo.add"(%831, %2599) : (tensor<3x3x1x32x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<3x3x1x32x!pphlo.pfxp>
    %2601 = pphlo.convolution(%arg2, %2600) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<50x28x28x1x!pphlo.pfxp>, tensor<3x3x1x32x!pphlo.pfxp>) -> tensor<50x28x28x32x!pphlo.pfxp>
    %2602 = "pphlo.reduce"(%2597, %1) ( {
    ^bb0(%arg4: tensor<!pphlo.pfxp>, %arg5: tensor<!pphlo.pfxp>):  // no predecessors
      %2655 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%2655) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<30x28x28x32x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<32x!pphlo.pfxp>
    %2603 = "pphlo.multiply"(%2602, %15) : (tensor<32x!pphlo.pfxp>, tensor<32x!pphlo.pfxp>) -> tensor<32x!pphlo.pfxp>
    %2604 = "pphlo.broadcast"(%2603) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<32x!pphlo.pfxp>) -> tensor<50x28x28x32x!pphlo.pfxp>
    %2605 = "pphlo.add"(%2601, %2604) : (tensor<50x28x28x32x!pphlo.pfxp>, tensor<50x28x28x32x!pphlo.pfxp>) -> tensor<50x28x28x32x!pphlo.pfxp>
    %2606 = "pphlo.maximum"(%2605, %14) : (tensor<50x28x28x32x!pphlo.pfxp>, tensor<50x28x28x32x!pphlo.pfxp>) -> tensor<50x28x28x32x!pphlo.pfxp>
    %2607 = "pphlo.reduce_window"(%2606, %1) ( {
    ^bb0(%arg4: tensor<!pphlo.pfxp>, %arg5: tensor<!pphlo.pfxp>):  // no predecessors
      %2655 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%2655) : (tensor<!pphlo.pfxp>) -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<50x28x28x32x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<50x14x14x32x!pphlo.pfxp>
    %2608 = "pphlo.multiply"(%2607, %13) : (tensor<50x14x14x32x!pphlo.pfxp>, tensor<50x14x14x32x!pphlo.pfxp>) -> tensor<50x14x14x32x!pphlo.pfxp>
    %2609 = pphlo.convolution(%1234, %2590) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<30x14x14x32x!pphlo.pfxp>, tensor<30x14x14x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %2610 = "pphlo.multiply"(%2609, %12) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %2611 = "pphlo.add"(%1672, %2610) : (tensor<3x3x32x64x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<3x3x32x64x!pphlo.pfxp>
    %2612 = pphlo.convolution(%2608, %2611) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<50x14x14x32x!pphlo.pfxp>, tensor<3x3x32x64x!pphlo.pfxp>) -> tensor<50x14x14x64x!pphlo.pfxp>
    %2613 = "pphlo.reduce"(%2590, %1) ( {
    ^bb0(%arg4: tensor<!pphlo.pfxp>, %arg5: tensor<!pphlo.pfxp>):  // no predecessors
      %2655 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%2655) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<30x14x14x64x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<64x!pphlo.pfxp>
    %2614 = "pphlo.multiply"(%2613, %11) : (tensor<64x!pphlo.pfxp>, tensor<64x!pphlo.pfxp>) -> tensor<64x!pphlo.pfxp>
    %2615 = "pphlo.broadcast"(%2614) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<64x!pphlo.pfxp>) -> tensor<50x14x14x64x!pphlo.pfxp>
    %2616 = "pphlo.add"(%2612, %2615) : (tensor<50x14x14x64x!pphlo.pfxp>, tensor<50x14x14x64x!pphlo.pfxp>) -> tensor<50x14x14x64x!pphlo.pfxp>
    %2617 = "pphlo.maximum"(%2616, %10) : (tensor<50x14x14x64x!pphlo.pfxp>, tensor<50x14x14x64x!pphlo.pfxp>) -> tensor<50x14x14x64x!pphlo.pfxp>
    %2618 = "pphlo.reduce_window"(%2617, %1) ( {
    ^bb0(%arg4: tensor<!pphlo.pfxp>, %arg5: tensor<!pphlo.pfxp>):  // no predecessors
      %2655 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%2655) : (tensor<!pphlo.pfxp>) -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<50x14x14x64x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<50x7x7x64x!pphlo.pfxp>
    %2619 = "pphlo.multiply"(%2618, %9) : (tensor<50x7x7x64x!pphlo.pfxp>, tensor<50x7x7x64x!pphlo.pfxp>) -> tensor<50x7x7x64x!pphlo.pfxp>
    %2620 = "pphlo.reshape"(%2619) : (tensor<50x7x7x64x!pphlo.pfxp>) -> tensor<50x3136x!pphlo.pfxp>
    %2621 = "pphlo.transpose"(%1678) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<30x3136x!pphlo.pfxp>) -> tensor<3136x30x!pphlo.pfxp>
    %2622 = "pphlo.dot"(%2621, %2582) : (tensor<3136x30x!pphlo.pfxp>, tensor<30x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2623 = "pphlo.multiply"(%2622, %8) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2624 = "pphlo.add"(%2116, %2623) : (tensor<3136x256x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<3136x256x!pphlo.pfxp>
    %2625 = "pphlo.dot"(%2620, %2624) : (tensor<50x3136x!pphlo.pfxp>, tensor<3136x256x!pphlo.pfxp>) -> tensor<50x256x!pphlo.pfxp>
    %2626 = "pphlo.reduce"(%2582, %1) ( {
    ^bb0(%arg4: tensor<!pphlo.pfxp>, %arg5: tensor<!pphlo.pfxp>):  // no predecessors
      %2655 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%2655) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<30x256x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<256x!pphlo.pfxp>
    %2627 = "pphlo.multiply"(%2626, %7) : (tensor<256x!pphlo.pfxp>, tensor<256x!pphlo.pfxp>) -> tensor<256x!pphlo.pfxp>
    %2628 = "pphlo.broadcast"(%2627) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256x!pphlo.pfxp>) -> tensor<50x256x!pphlo.pfxp>
    %2629 = "pphlo.add"(%2625, %2628) : (tensor<50x256x!pphlo.pfxp>, tensor<50x256x!pphlo.pfxp>) -> tensor<50x256x!pphlo.pfxp>
    %2630 = "pphlo.maximum"(%2629, %6) : (tensor<50x256x!pphlo.pfxp>, tensor<50x256x!pphlo.pfxp>) -> tensor<50x256x!pphlo.pfxp>
    %2631 = "pphlo.transpose"(%2128) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<30x256x!pphlo.pfxp>) -> tensor<256x30x!pphlo.pfxp>
    %2632 = "pphlo.dot"(%2631, %2577) : (tensor<256x30x!pphlo.pfxp>, tensor<30x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2633 = "pphlo.multiply"(%2632, %5) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2634 = "pphlo.add"(%2566, %2633) : (tensor<256x10x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<256x10x!pphlo.pfxp>
    %2635 = "pphlo.dot"(%2630, %2634) : (tensor<50x256x!pphlo.pfxp>, tensor<256x10x!pphlo.pfxp>) -> tensor<50x10x!pphlo.pfxp>
    %2636 = "pphlo.reduce"(%2577, %1) ( {
    ^bb0(%arg4: tensor<!pphlo.pfxp>, %arg5: tensor<!pphlo.pfxp>):  // no predecessors
      %2655 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%2655) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<30x10x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<10x!pphlo.pfxp>
    %2637 = "pphlo.multiply"(%2636, %4) : (tensor<10x!pphlo.pfxp>, tensor<10x!pphlo.pfxp>) -> tensor<10x!pphlo.pfxp>
    %2638 = "pphlo.broadcast"(%2637) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<10x!pphlo.pfxp>) -> tensor<50x10x!pphlo.pfxp>
    %2639 = "pphlo.add"(%2635, %2638) : (tensor<50x10x!pphlo.pfxp>, tensor<50x10x!pphlo.pfxp>) -> tensor<50x10x!pphlo.pfxp>
    %2640 = "pphlo.reduce"(%2639, %3) ( {
    ^bb0(%arg4: tensor<!pphlo.pfxp>, %arg5: tensor<!pphlo.pfxp>):  // no predecessors
      %2655 = "pphlo.maximum"(%arg4, %arg5) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%2655) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<50x10x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<50x!pphlo.pfxp>
    %2641 = "pphlo.broadcast"(%2640) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<50x!pphlo.pfxp>) -> tensor<50x10x!pphlo.pfxp>
    %2642 = "pphlo.subtract"(%2639, %2641) : (tensor<50x10x!pphlo.pfxp>, tensor<50x10x!pphlo.pfxp>) -> tensor<50x10x!pphlo.pfxp>
    %2643 = "pphlo.exponential"(%2642) : (tensor<50x10x!pphlo.pfxp>) -> tensor<50x10x!pphlo.pfxp>
    %2644 = "pphlo.reduce"(%2643, %1) ( {
    ^bb0(%arg4: tensor<!pphlo.pfxp>, %arg5: tensor<!pphlo.pfxp>):  // no predecessors
      %2655 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%2655) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<50x10x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<50x!pphlo.pfxp>
    %2645 = "pphlo.log"(%2644) : (tensor<50x!pphlo.pfxp>) -> tensor<50x!pphlo.pfxp>
    %2646 = "pphlo.broadcast"(%2645) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<50x!pphlo.pfxp>) -> tensor<50x10x!pphlo.pfxp>
    %2647 = "pphlo.subtract"(%2642, %2646) : (tensor<50x10x!pphlo.pfxp>, tensor<50x10x!pphlo.pfxp>) -> tensor<50x10x!pphlo.pfxp>
    %2648 = "pphlo.subtract"(%2647, %2) : (tensor<50x10x!pphlo.pfxp>, tensor<50x10x!pphlo.pfxp>) -> tensor<50x10x!pphlo.pfxp>
    %2649 = "pphlo.multiply"(%271, %2648) : (tensor<50x10x!pphlo.pint>, tensor<50x10x!pphlo.pfxp>) -> tensor<50x10x!pphlo.pfxp>
    %2650 = "pphlo.add"(%2649, %2) : (tensor<50x10x!pphlo.pfxp>, tensor<50x10x!pphlo.pfxp>) -> tensor<50x10x!pphlo.pfxp>
    %2651 = "pphlo.reduce"(%2650, %1) ( {
    ^bb0(%arg4: tensor<!pphlo.pfxp>, %arg5: tensor<!pphlo.pfxp>):  // no predecessors
      %2655 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%2655) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<50x10x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<50x!pphlo.pfxp>
    %2652 = "pphlo.negate"(%2651) : (tensor<50x!pphlo.pfxp>) -> tensor<50x!pphlo.pfxp>
    %2653 = "pphlo.reduce"(%2652, %1) ( {
    ^bb0(%arg4: tensor<!pphlo.pfxp>, %arg5: tensor<!pphlo.pfxp>):  // no predecessors
      %2655 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
      "pphlo.return"(%2655) : (tensor<!pphlo.pfxp>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<50x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    %2654 = "pphlo.multiply"(%2653, %0) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    return %2654 : tensor<!pphlo.pfxp>
  }
}
