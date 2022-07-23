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


#include "ppu/mpc/aby3/ot.h"

#include "ppu/core/array_ref_util.h"
#include "ppu/core/shape_util.h"
#include "ppu/crypto/pseudo_random_generator.h"
#include "ppu/mpc/util/communicator.h"
#include "ppu/utils/rand.h"
#include "ppu/utils/serialize.h"

namespace ppu::mpc::aby3 {

// Referrence:
// 5.4.1 Semi-honest Security
// https://eprint.iacr.org/2018/403.pdf
//
// Latency: 1
OT3Party::OT3Party(std::shared_ptr<link::Context> lctx) : lctx_(lctx) {
  // parties synchronize shared seed between self and next.
  {
    PPU_ENFORCE(lctx_->WorldSize() == 3);
    constexpr char kCommTag[] = "Random:ShareKey";
    self_shared_seed_ = utils::RandSeed();

    // send seed to previous party.
    lctx_->SendAsync(lctx_->PrevRank(),
                     utils::SerializeUint128(self_shared_seed_), kCommTag);

    // receive seed from next party
    next_shared_seed_ =
        utils::DeserializeUint128(lctx_->Recv(lctx_->NextRank(), kCommTag));
  }
}

ArrayRef OT3Party::RandPSelfAndNext(FieldType field, size_t size,
                                    size_t rank_self, size_t rank_next) {
  return DISPATCH_ALL_FIELDS(field, "aby3.OT3Party.RandPSelfAndNext", [&]() {
    PPU_ENFORCE((rank_self + 1) % lctx_->WorldSize() == rank_next);

    ArrayRef out(makeType<Ring2kPublTy>(field), size);

    if (lctx_->Rank() == rank_self) {
      // sample the data with the synchronized shared seed.
      next_shared_counter_ = FillAesRandom(
          next_shared_seed_, 0, next_shared_counter_,
          absl::MakeSpan(static_cast<ring2k_t*>(out.data()), out.numel()));
    } else if (lctx_->Rank() == rank_next) {
      // sample the data with the synchronized shared seed.
      self_shared_counter_ = FillAesRandom(
          self_shared_seed_, 0, self_shared_counter_,
          absl::MakeSpan(static_cast<ring2k_t*>(out.data()), out.numel()));
    }

    return out;
  });
}

void OT3Party::OTSend(absl::Span<const ArrayRef> v0,
                      absl::Span<const ArrayRef> v1) {
  PPU_ENFORCE(v0.size() > 0);
  const auto field = v0[0].eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, "aby3.OT3Party.OTSend", [&]() {
    PPU_ENFORCE(v0.size() == v1.size());

    std::vector<Buffer> masked_msg_v0;
    std::vector<Buffer> masked_msg_v1;
    for (size_t i = 0; i < v0.size(); ++i) {
      const auto& m0 = xt_adapt<ring2k_t>(v0[i]);
      const auto& m1 = xt_adapt<ring2k_t>(v1[i]);
      size_t numel = v0[i].numel();
      ArrayRef p0 =
          RandPSelfAndNext(field, numel, lctx_->PrevRank(), lctx_->Rank());
      ArrayRef p1 =
          RandPSelfAndNext(field, numel, lctx_->PrevRank(), lctx_->Rank());
      const auto& mask0 = xt_adapt<ring2k_t>(p0);
      const auto& mask1 = xt_adapt<ring2k_t>(p1);

      // send to receiver
      const auto& masked_m0 = m0 ^ mask0;
      const auto& masked_m1 = m1 ^ mask1;
      masked_msg_v0.emplace_back(detail::SerializeXtensor(masked_m0));
      masked_msg_v1.emplace_back(detail::SerializeXtensor(masked_m1));
    }

    lctx_->SendAsync(lctx_->NextRank(),
                     utils::SerializeArrayOfBuffers(std::vector<Buffer>{
                         utils::SerializeArrayOfBuffers(masked_msg_v0),
                         utils::SerializeArrayOfBuffers(masked_msg_v1)}),
                     "OT3Party:SendMaskedMsg0");
  });
}

void OT3Party::OTRecv(absl::Span<const ArrayRef> choice,
                      absl::Span<ArrayRef> v) {
  PPU_ENFORCE(!choice.empty());

  const auto field = choice[0].eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, "aby3.OT3Party.OTRecv", [&]() {
    auto buf = lctx_->Recv(lctx_->PrevRank(), "OT3Party:SendMaskedMsg0");
    auto str2 = lctx_->Recv(lctx_->NextRank(), "OT3Party:SendMask");

    // deserialize masked msg
    std::vector<Buffer> str_v = utils::DeserializeArrayOfBuffers(buf);
    PPU_ENFORCE(str_v.size() == 2);
    std::vector<Buffer> masked_msg_v0 =
        utils::DeserializeArrayOfBuffers(str_v[0]);
    std::vector<Buffer> masked_msg_v1 =
        utils::DeserializeArrayOfBuffers(str_v[1]);
    PPU_ENFORCE(choice.size() == masked_msg_v0.size());
    PPU_ENFORCE(choice.size() == masked_msg_v1.size());

    // deserialize mask
    std::vector<Buffer> mask_v = utils::DeserializeArrayOfBuffers(str2);

    for (size_t i = 0; i < choice.size(); ++i) {
      const auto& c = xt_adapt<ring2k_t>(choice[i]);
      const auto& masked_m0 =
          detail::BuildXtensor<ring2k_t>(c.shape(), masked_msg_v0[i]);
      const auto& masked_m1 =
          detail::BuildXtensor<ring2k_t>(c.shape(), masked_msg_v1[i]);
      const auto& mask = detail::BuildXtensor<ring2k_t>(c.shape(), mask_v[i]);

      auto masked_m = masked_m0;
      for (size_t i = 0; i < c.size(); ++i) {
        masked_m(i) = (c(i) & 1) ? masked_m1(i) : masked_m0(i);
      }

      v[i] = make_array(mask ^ masked_m, makeType<Ring2kPublTy>(field));
    }
  });
}

void OT3Party::OTHelp(absl::Span<const ArrayRef> choice) {
  PPU_ENFORCE(!choice.empty());

  const auto field = choice[0].eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, "aby3.OT3Party.OTHelp", [&]() {
    std::vector<Buffer> mask_v;
    for (size_t i = 0; i < choice.size(); ++i) {
      const auto& c = xt_adapt<ring2k_t>(choice[i]);

      size_t numel = choice[i].numel();
      ArrayRef p0 =
          RandPSelfAndNext(field, numel, lctx_->Rank(), lctx_->NextRank());
      ArrayRef p1 =
          RandPSelfAndNext(field, numel, lctx_->Rank(), lctx_->NextRank());

      const auto& mask0 = xt_adapt<ring2k_t>(p0);
      const auto& mask1 = xt_adapt<ring2k_t>(p1);

      xt::xarray<ring2k_t> mask = mask0;
      for (size_t i = 0; i < c.size(); ++i) {
        mask(i) = (c(i) & 1) ? mask1(i) : mask0(i);
      }
      mask_v.emplace_back(detail::SerializeXtensor(mask));
    }
    lctx_->SendAsync(lctx_->PrevRank(), utils::SerializeArrayOfBuffers(mask_v),
                     "OT3Party:SendMask");
  });
}

}  // namespace ppu::mpc::aby3
