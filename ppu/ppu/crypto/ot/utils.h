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

#include <array>
#include <cstddef>
#include <string>
#include <type_traits>

#include "ppu/core/buffer.h"
#include "ppu/crypto/ot/block.h"
#include "ppu/utils/int128.h"

namespace ppu {

// TODO(shuyan.ycf): drop this after link support view type.
template <size_t N>
inline Buffer StringFromArray(const std::array<uint128_t, N>& array) {
  return {array.data(), static_cast<int64_t>(array.size() * sizeof(uint128_t))};
}

inline constexpr uint128_t GetBit(uint128_t i, size_t pos) {
  return (i >> pos) & 1;
}

// TODO(shuyan): check MP-SPDZ implementation of `EklundhTranspose128`
// TODO(shuyan): consider introduce 1024x128 SSE transpose in the future.
template <size_t N = 1>
inline void NaiveTranspose(std::array<uint128_t, 128 * N>* inout) {
  std::array<uint128_t, 128 * N> in = *inout;
  for (size_t i = 0; i < 128 * N; ++i) {
    uint128_t t = 0;
    for (size_t j = 0; j < 128; ++j) {
      t |= GetBit(in[j], i) << j;
    }
    (*inout)[i] = t;
  }
}

void EklundhTranspose128(std::array<uint128_t, 128>* inout);

void SseTranspose128(std::array<uint128_t, 128>* inout);

void SseTranspose128x1024(std::array<std::array<block, 8>, 128>& inout);
void SseTranspose128x1024(std::array<std::array<uint128_t, 8>, 128>* inout);
void EklundhTranspose128x1024(std::array<std::array<uint128_t, 8>, 128>* inout);

}  // namespace ppu
