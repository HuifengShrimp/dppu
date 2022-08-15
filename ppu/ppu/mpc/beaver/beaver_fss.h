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

// #include "ppu/crypto/ot/silent/primitives.h"
#include "ppu/link/context.h"
#include "ppu/mpc/beaver/beaver.h"
#include "ppu/mpc/beaver/trusted_party.h"


namespace ppu::mpc {

// TFP Dealer implementation.
// Rank0 party owns TrustedParty directly. Check security implications before
// moving on.
class BeaverFss : public Beaver {
 protected:
  // Only for rank0 party.
  TrustedParty tp_;

 protected:
  std::shared_ptr<link::Context> lctx_;

  PrgSeed seed_;

  PrgCounter counter_;

 public:
  BeaverFss(std::shared_ptr<link::Context> lctx);

  Beaver::Triple Mul(FieldType field, size_t size) override;

  Beaver::Triple And(FieldType field, size_t size) override;

  Beaver::Triple Dot(FieldType field, size_t M, size_t N, size_t K) override;

  Beaver::LR_set lr(FieldType field, size_t M, size_t N) override;

  Beaver::Pair Trunc(FieldType field, size_t size, size_t bits) override;

  ArrayRef RandBit(FieldType field, size_t size) override;
};

}  // namespace ppu::mpc
