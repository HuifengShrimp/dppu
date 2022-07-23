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

#include <utility>

#include "ppu/core/type_util.h"
#include "ppu/mpc/dealer/dealer.h"

namespace ppu::mpc {

// reference dealer protocol implementation.
class DealerRef : public Dealer {
 public:
  Dealer::Triple Mul(FieldType field, size_t size) override;

  Dealer::Triple And(FieldType field, size_t size) override;

  Dealer::Triple Dot(FieldType field, size_t M, size_t N, size_t K) override;

  Dealer::LR_set lr(FieldType field, size_t M, size_t N) override;

  Dealer::Pair Trunc(FieldType field, size_t size, size_t bits) override;

  ArrayRef RandBit(FieldType field, size_t size) override;
};

}  // namespace ppu::mpc
