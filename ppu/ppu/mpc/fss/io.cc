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


#include "ppu/mpc/fss/io.h"

#include "ppu/mpc/semi2k/type.h"


namespace ppu::mpc::fss {

std::vector<NdArrayRef> FSSIo::makeSecret(const NdArrayRef& raw) const {
  const auto field = raw.eltype().as<Ring2k>()->field();
  const auto splits = randAdditiveSplits(raw);
  std::vector<NdArrayRef> shares;
  const auto ty = makeType<semi2k::AShrTy>(field);
  for (const auto& split : splits) {
    shares.emplace_back(split.as(ty));
  }
  return shares;
}

NdArrayRef FSSIo::reconstructSecret(
    const std::vector<NdArrayRef>& shares) const {
  const auto field = shares.at(0).eltype().as<Ring2k>()->field();

  std::vector<NdArrayRef> encoded;
  encoded.reserve(shares.size());
  for (const auto& val : shares) {
    encoded.push_back(val.as(makePtType(GetStorageType(field))));
  }
  return sum(encoded).as(makeType<RingTy>(field));
}

std::unique_ptr<FSSIo> makeFSSIo(size_t npc) {
  semi2k::registerTypes();
  return std::make_unique<FSSIo>(npc);
}

}  // namespace ppu::mpc::fss
