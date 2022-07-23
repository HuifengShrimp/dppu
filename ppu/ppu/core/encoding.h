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


#pragma once

#include "ppu/core/type_util.h"
#include "ppu/utils/exception.h"

#include "ppu/ppu.pb.h"

namespace ppu {

// This module define encode/decode function from PtType to DataType and vice
// versa.
//
// DataType is much simpler than PtType, basically it support two types (FXP and
// INT, FLOAT maybe supported in the future), but it could be encoded in
// different storage types, DataType and underline storage type are conceptually
// independent, for example.
// - (I64, FXP): fixed point encoded on I64.
// - (I128, FXP): fixed point encoded on I128.
// - (I128, INT): integer encoded I128.
//
// Encode/Decode rules:
// - encoded DataType is deduced from source PtType, (float->FXP, int32_t->INT).
// - encoded storage type is specified by the caller.
// - decoded PtType is deduced from DataType, (FXP->float, INT->int64_t).
//
// Encode examples:
//  src_type, storage type  |   encoded type
//   int32_t + I64         ->   (I64,  INT)
//   int32_t + I128        ->   (I128, INT)
//   double  + I64         ->   (I64,  FXP)
//   float   + I128        ->   (I128, FXP)
//
// Decode example:
//   encoded type  -> cpp type
//   (I64,  INT)  ->  int64_t
//   (I128, INT)  ->  int64_t
//   (I64,  FXP)  ->  float
//   (I128, FXP)  ->  float
//
// Check GetEncodeType/GetDecodeType for more details.
//
// Common pitfall, since DataType is less than PtType, so convert PtType to
// DataType then convert back may loss type information. For example:
//   raw       encoded        decoded
//   double -> (I64, FXP) -> float       ; double convert back into float
//  int32_t -> (I64, INT) -> int64_t     ; int32_t convert back into int64_t
constexpr PtType GetDecodeType(DataType dtype) {
  switch (dtype) {
    case DT_FXP:
      return PT_F32;
    case DT_INT:
      return PT_I64;
    default:
      return PT_INVALID;
  }
}

size_t FxpFractionalBits(const RuntimeConfig& config);

}  // namespace ppu
