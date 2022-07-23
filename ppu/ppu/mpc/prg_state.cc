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


#include "ppu/mpc/prg_state.h"

#include "ppu/crypto/pseudo_random_generator.h"
#include "ppu/utils/rand.h"
#include "ppu/utils/serialize.h"

namespace ppu::mpc {

PrgState::PrgState() {
  pub_seed_ = 0;
  pub_counter_ = 0;

  priv_seed_ = utils::RandSeed();
  priv_counter_ = 0;

  self_seed_ = 0;
  prev_seed_ = 0;
  prss_counter_ = 0;
}

PrgState::PrgState(std::shared_ptr<link::Context> lctx) {
  // synchronize public state.
  {
    uint128_t self_pk = utils::RandSeed();

    const auto all_buf =
        link::AllGather(lctx, utils::SerializeUint128(self_pk), "Random::PK");

    pub_seed_ = 0;
    for (const auto& buf : all_buf) {
      uint128_t seed = utils::DeserializeUint128(buf);
      pub_seed_ += seed;
    }

    pub_counter_ = 0;
  }

  // init private state.
  {
    priv_seed_ = utils::RandSeed();
    priv_counter_ = 0;
  }

  // init PRSS state.
  {
    self_seed_ = utils::RandSeed();

    constexpr char kCommTag[] = "Random:PRSS";

    // send seed to next party, receive seed from prev party
    lctx->SendAsync(lctx->NextRank(), utils::SerializeUint128(self_seed_),
                    kCommTag);
    prev_seed_ =
        utils::DeserializeUint128(lctx->Recv(lctx->PrevRank(), kCommTag));

    prss_counter_ = 0;
  }
}

std::pair<ArrayRef, ArrayRef> PrgState::genPrssPair(FieldType field,
                                                    size_t size) {
  const Type ty = makeType<RingTy>(field);

  ArrayRef r_prev(ty, size);
  ArrayRef r_self(ty, size);
  FillAesRandom(
      self_seed_, 0, prss_counter_,
      absl::MakeSpan(static_cast<char*>(r_self.data()), r_self.buf()->size()));
  prss_counter_ = FillAesRandom(
      prev_seed_, 0, prss_counter_,
      absl::MakeSpan(static_cast<char*>(r_prev.data()), r_prev.buf()->size()));

  return std::make_pair(r_prev, r_self);
}

ArrayRef PrgState::genPriv(FieldType field, size_t numel) {
  ArrayRef res(makeType<RingTy>(field), numel);
  priv_counter_ = FillAesRandom(
      priv_seed_, 0, priv_counter_,
      absl::MakeSpan(static_cast<char*>(res.data()), res.buf()->size()));

  return res;
}

ArrayRef PrgState::genPubl(FieldType field, size_t numel) {
  ArrayRef res(makeType<RingTy>(field), numel);
  pub_counter_ = FillAesRandom(
      pub_seed_, 0, pub_counter_,
      absl::MakeSpan(static_cast<char*>(res.data()), res.buf()->size()));

  return res;
}

}  // namespace ppu::mpc
