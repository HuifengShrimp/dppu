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


#include "ppu/core/encoding.h"

namespace ppu {
namespace {

size_t FxpDefaultFractionalBits(FieldType field) {
  switch (field) {
    case FieldType::FM32: {
      return 8;
    }
    case FieldType::FM64: {
      return 18;
    }
    case FieldType::FM128: {
      return 26;
    }
    default: {
      PPU_THROW("unsupported field={}", field);
    }
  }
}

}  // namespace

size_t FxpFractionalBits(const RuntimeConfig& config) {
  if (config.fxp_fraction_bits() == 0) {
    return FxpDefaultFractionalBits(config.field());
  }
  return config.fxp_fraction_bits();
}

}  // namespace ppu
