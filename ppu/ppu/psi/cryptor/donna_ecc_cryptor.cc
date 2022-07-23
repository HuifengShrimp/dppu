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


#include "ppu/psi/cryptor/donna_ecc_cryptor.h"

extern "C" {
#include "curve25519.h"
}

#include <iostream>

#include "ppu/utils/parallel.h"

namespace ppu {

void DonnaEccCryptor::EccMask(absl::Span<const char> batch_points,
                              absl::Span<char> dest_points) const {
  PPU_ENFORCE(batch_points.size() % kEccKeySize == 0);

  using Item = std::array<unsigned char, kEccKeySize>;
  static_assert(sizeof(Item) == kEccKeySize);

  auto mask_functor = [this](const Item& in, Item& out) {
    PPU_ENFORCE(out.size() == kEccKeySize);
    PPU_ENFORCE(in.size() == kEccKeySize);

    curve25519_donna(out.data(), this->private_key_, in.data());
  };

  absl::Span<const Item> input(
      reinterpret_cast<const Item*>(batch_points.data()),
      batch_points.size() / sizeof(Item));
  absl::Span<Item> output(reinterpret_cast<Item*>(dest_points.data()),
                          dest_points.size() / sizeof(Item));

  parallel_for(0, input.size(), 1, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      mask_functor(input[idx], output[idx]);
    }
  });
}

}  // namespace ppu
