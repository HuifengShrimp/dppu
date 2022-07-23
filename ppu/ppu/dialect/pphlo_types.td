//
// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

//===----------------------------------------------------------------------===//
//
// This file declares the Privacy-Preserving HLO dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef PPU_DIALECT_PPHLO_TYPES
#define PPU_DIALECT_PPHLO_TYPES

include "pphlo_dialect.td"
include "pphlo_base_enums.td"

//===----------------------------------------------------------------------===//
// PPHLO Types
//===----------------------------------------------------------------------===//
class PPHLO_TypeDef<string name, string typeMnemonic>
    : TypeDef<PPHlo_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def PPHLO_PINT : PPHLO_TypeDef<"PInteger", "pint"> {
  let summary = "PPHLO public int type";
}

def PPHLO_SINT : PPHLO_TypeDef<"SInteger", "sint"> {
  let summary = "PPHLO secret int type";
}

def PPHLO_PFXP : PPHLO_TypeDef<"PFixedpoint", "pfxp"> {
  let summary = "PPHLO public fixedpoint type";
}

def PPHLO_SFXP : PPHLO_TypeDef<"SFixedpoint", "sfxp"> {
  let summary = "PPHLO secret fixedpoint type";
}

def PPHLO_UINT : PPHLO_TypeDef<"UInteger", "uint"> {
  let summary = "PPHLO unkown int type";
}

def PPHLO_UFXP : PPHLO_TypeDef<"UFixedpoint", "ufxp"> {
  let summary = "PPHLO unkown fixedpoint type";
}

def PPHLO_INT : AnyTypeOf<[PPHLO_SINT, PPHLO_PINT]>;

def PPHLO_FXP : AnyTypeOf<[PPHLO_SFXP, PPHLO_PFXP]>;

def PPHLO_IntTensor : StaticShapeTensorOf<[PPHLO_INT]>;

def PPHLO_FxpTensor : StaticShapeTensorOf<[PPHLO_FXP]>;

def PPHLO_Tensor : StaticShapeTensorOf<[PPHLO_INT, PPHLO_FXP]>;

#endif  // PPU_DIALECT_PPHLO_TYPES
