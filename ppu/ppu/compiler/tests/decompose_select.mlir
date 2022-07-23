// RUN: mlir-pphlo-opt --decompose-select --split-input-file %s | FileCheck %s

func @select(%arg0: tensor<2x2x!pphlo.pint>, %arg1: tensor<2x2x!pphlo.pint>, %arg2: tensor<2x2x!pphlo.pint>) -> (tensor<2x2x!pphlo.pint>) {
    //CHECK: %0 = "pphlo.subtract"(%arg1, %arg2) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint> 
    //CHECK: %1 = "pphlo.multiply"(%arg0, %0) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    //CHECK: %2 = "pphlo.add"(%1, %arg2) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    %0 = "pphlo.select"(%arg0, %arg1, %arg2) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    return %0 : tensor<2x2x!pphlo.pint>
}

func @select1(%arg0: tensor<2x2x!pphlo.pint>, %arg1: tensor<2x2x!pphlo.pfxp>, %arg2: tensor<2x2x!pphlo.pfxp>) -> (tensor<2x2x!pphlo.pfxp>) {
    //CHECK: %0 = "pphlo.subtract"(%arg1, %arg2) : (tensor<2x2x!pphlo.pfxp>, tensor<2x2x!pphlo.pfxp>) -> tensor<2x2x!pphlo.pfxp>
    //CHECK: %1 = "pphlo.multiply"(%arg0, %0) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pfxp>) -> tensor<2x2x!pphlo.pfxp>
    //CHECK: %2 = "pphlo.add"(%1, %arg2) : (tensor<2x2x!pphlo.pfxp>, tensor<2x2x!pphlo.pfxp>) -> tensor<2x2x!pphlo.pfxp> 
    %0 = "pphlo.select"(%arg0, %arg1, %arg2) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pfxp>, tensor<2x2x!pphlo.pfxp>) -> tensor<2x2x!pphlo.pfxp>
    return %0 : tensor<2x2x!pphlo.pfxp>
}

func @select_mixed(%arg0: tensor<!pphlo.pint>, %arg1: tensor<!pphlo.pfxp>, %arg2: tensor<!pphlo.sfxp>) -> (tensor<!pphlo.sfxp>) {
    //CHECK: %0 = "pphlo.subtract"(%arg1, %arg2) : (tensor<!pphlo.pfxp>, tensor<!pphlo.sfxp>) -> tensor<!pphlo.sfxp>
    //CHECK: %1 = "pphlo.multiply"(%arg0, %0) : (tensor<!pphlo.pint>, tensor<!pphlo.sfxp>) -> tensor<!pphlo.sfxp>
    //CHECK: %2 = "pphlo.add"(%1, %arg2) : (tensor<!pphlo.sfxp>, tensor<!pphlo.sfxp>) -> tensor<!pphlo.sfxp>
    %0 = "pphlo.select"(%arg0, %arg1, %arg2) : (tensor<!pphlo.pint>, tensor<!pphlo.pfxp>, tensor<!pphlo.sfxp>) -> tensor<!pphlo.sfxp>
    return %0 : tensor<!pphlo.sfxp>
}
