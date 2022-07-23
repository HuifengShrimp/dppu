// RUN: mlir-pphlo-opt --decompose-divide --split-input-file %s | FileCheck %s

func @divide_fxp(%arg0: tensor<2x2x!pphlo.pfxp>, %arg1: tensor<2x2x!pphlo.pfxp>) -> (tensor<2x2x!pphlo.pfxp>) {
    //CHECK: %0 = "pphlo.reciprocal"(%arg1) : (tensor<2x2x!pphlo.pfxp>) -> tensor<2x2x!pphlo.pfxp>
    //CHECK: %1 = "pphlo.multiply"(%arg0, %0) : (tensor<2x2x!pphlo.pfxp>, tensor<2x2x!pphlo.pfxp>) -> tensor<2x2x!pphlo.pfxp>
    %0 = "pphlo.divide"(%arg0, %arg1) : (tensor<2x2x!pphlo.pfxp>, tensor<2x2x!pphlo.pfxp>) -> tensor<2x2x!pphlo.pfxp>
    return %0 : tensor<2x2x!pphlo.pfxp>
}

func @divide_int(%arg0: tensor<2x2x!pphlo.pint>, %arg1: tensor<2x2x!pphlo.pint>) -> (tensor<2x2x!pphlo.pint>) {
    //CHECK: %0 = "pphlo.convert"(%arg0) : (tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pfxp>
    //CHECK: %1 = "pphlo.convert"(%arg1) : (tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pfxp>
    //CHECK: %2 = "pphlo.reciprocal"(%1) : (tensor<2x2x!pphlo.pfxp>) -> tensor<2x2x!pphlo.pfxp>
    //CHECK: %3 = "pphlo.multiply"(%0, %2) : (tensor<2x2x!pphlo.pfxp>, tensor<2x2x!pphlo.pfxp>) -> tensor<2x2x!pphlo.pfxp>
    //CHECK: %4 = "pphlo.convert"(%3) : (tensor<2x2x!pphlo.pfxp>) -> tensor<2x2x!pphlo.pint>
    %0 = "pphlo.divide"(%arg0, %arg1) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    return %0 : tensor<2x2x!pphlo.pint>
}

func @divide_mixed(%arg0:tensor<!pphlo.sfxp>, %arg1: tensor<!pphlo.pfxp>) -> (tensor<!pphlo.sfxp>) {
    //CHECK: %0 = "pphlo.reciprocal"(%arg1) : (tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    //CHECK: %1 = "pphlo.multiply"(%arg0, %0) : (tensor<!pphlo.sfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.sfxp>
    %0 = "pphlo.divide"(%arg0, %arg1) : (tensor<!pphlo.sfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.sfxp>
    return %0 : tensor<!pphlo.sfxp>
}
