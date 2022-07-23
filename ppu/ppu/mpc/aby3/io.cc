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


#include "ppu/mpc/aby3/io.h"

#include "ppu/mpc/aby3/type.h"

namespace ppu::mpc::aby3 {

std::vector<NdArrayRef> Aby3Io::makeSecret(const NdArrayRef& raw) const {
  const auto field = raw.eltype().as<Ring2k>()->field();
  const auto splits = randAdditiveSplits(raw);
  PPU_ENFORCE(splits.size() == 3, "expect 3PC, got={}", splits.size());

  std::vector<NdArrayRef> shares;
  for (std::size_t i = 0; i < 3; i++) {
    auto buf =
        shares.emplace_back(makeType<AShrTy>(field), splits[i].shape()).buf();
    {
      const auto& x1 = splits[i];
      const auto& x2 = splits[(i + 1) % 3];

      PPU_ENFORCE(buf->size() == x1.buf()->size() + x2.buf()->size());
      PPU_ENFORCE(x1.elsize() == x2.elsize());

      size_t x1_offset = 0;
      size_t x2_offset = 0;
      size_t buf_offset = 0;
      size_t elsize = x1.elsize();
      for (int64_t idx = 0; idx < x1.numel(); ++idx) {
        std::memcpy(buf->data<char>() + buf_offset,
                    x1.buf()->data<char>() + x1_offset, elsize);
        buf_offset += elsize;
        x1_offset += elsize;
        std::memcpy(buf->data<char>() + buf_offset,
                    x2.buf()->data<char>() + x2_offset, elsize);
        buf_offset += elsize;
        x2_offset += elsize;
      }
    }
  }
  return shares;
}

NdArrayRef Aby3Io::reconstructSecret(
    const std::vector<NdArrayRef>& shares) const {
  const auto field = shares.at(0).eltype().as<Ring2k>()->field();

  std::vector<NdArrayRef> encoded;
  encoded.reserve(shares.size());
  for (size_t idx = 0; idx < shares.size(); idx++) {
    // view the first part of element.
    encoded.push_back(shares[idx].as(makePtType(GetStorageType(field)), true));
    PPU_ENFORCE(encoded[idx].shape() == shares[idx].shape());
    PPU_ENFORCE(encoded[idx].elsize() * 2 == shares[idx].elsize());
  }
  return sum(encoded).as(makeType<RingTy>(field));
}

std::unique_ptr<Aby3Io> makeAby3Io(size_t npc) {
  PPU_ENFORCE_EQ(npc, 3u, "aby3 is only for 3pc.");
  registerTypes();
  return std::make_unique<Aby3Io>(npc);
}

}  // namespace ppu::mpc::aby3
