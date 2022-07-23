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

#include "ppu/core/array_ref.h"
#include "ppu/core/type_util.h"
#include "ppu/link/link.h"

namespace ppu::mpc::aby3 {

// ot for 3 parties
class OT3Party {
  std::shared_ptr<link::Context> lctx_;

  // shared seed, know to self party and previous party.
  uint128_t self_shared_seed_;

  // shared seed, know to self party and next party.
  uint128_t next_shared_seed_;

  // action counter.
  mutable uint64_t self_shared_counter_ = 0;
  mutable uint64_t next_shared_counter_ = 0;

  // share data between self and next
  ArrayRef RandPSelfAndNext(FieldType field, size_t size, size_t rank_self,
                            size_t rank_next);

 public:
  // the rank of sender, receiver and helper should be (0, 1, 2) or (1, 2, 0) or
  // (2, 0, 1)
  OT3Party(std::shared_ptr<link::Context> lctx);

  void OTSend(absl::Span<const ArrayRef> v0, absl::Span<const ArrayRef> v1);
  // OTRecv
  // input: choice
  // output: v
  void OTRecv(absl::Span<const ArrayRef> choice, absl::Span<ArrayRef> v);
  void OTHelp(absl::Span<const ArrayRef> choice);
};

}  // namespace ppu::mpc::aby3
