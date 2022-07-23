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


#include "ppu/mpc/base2k/ring_io.h"

namespace ppu::mpc {

std::vector<NdArrayRef> RingIo::makePublic(const NdArrayRef& raw) const {
  const auto field = raw.eltype().as<Ring2k>()->field();
  const auto share = raw.as(makeType<Ring2kPublTy>(field));

  return std::vector<NdArrayRef>(world_size_, share);
}

NdArrayRef RingIo::reconstruct(const std::vector<NdArrayRef>& shares) const {
  PPU_ENFORCE(!shares.empty(), "got {}", shares.size());

  if (shares[0].eltype().isa<Public>()) {
    const auto field = shares[0].eltype().as<Ring2k>()->field();
    return shares[0].as(makeType<RingTy>(field));
  } else if (shares[0].eltype().isa<Secret>()) {
    return reconstructSecret(shares);
  } else {
    PPU_THROW("should not be here, eltype={}", shares[0].eltype());
  }
}

std::vector<NdArrayRef> RingIo::randAdditiveSplits(
    const NdArrayRef& arr) const {
  PPU_ENFORCE(world_size_ > 1, "split world_size_ber should be greater than 1",
              world_size_);

  const auto field = arr.eltype().as<Ring2k>()->field();
  const Type int_ty = makePtType(GetStorageType(field));

  std::vector<NdArrayRef> splits;

  for (size_t idx = 0; idx < world_size_; idx++) {
    splits.push_back(randint(int_ty, arr.shape()));
  }

  NdArrayRef s = sum(splits);
  splits[0] = add(splits[0], sub(arr.as(int_ty), s));

  return splits;
}

}  // namespace ppu::mpc
