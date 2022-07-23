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

#include <random>

#include "ppu/core/array_ref.h"
#include "ppu/core/type_util.h"
#include "ppu/link/link.h"
#include "ppu/mpc/object.h"
#include "ppu/mpc/util/communicator.h"

namespace ppu::mpc::test {

// helper constants
constexpr Rank kAlice = 0;
constexpr Rank kBob = 1;
constexpr Rank kCarol = 2;

// Evaluate a function for a given world size.
void Eval(size_t world_size,
          std::function<void(std::shared_ptr<link::Context> lctx)> proc);

// Test two public equals
bool EqualsPP(FieldType field, const ArrayRef& px, const ArrayRef& py);

bool EqualsVP(Rank rank, Rank owner, FieldType field, const ArrayRef& vx,
              const ArrayRef& py);

inline bool EqualsVV(FieldType field, const ArrayRef& vx, const ArrayRef& vy) {
  return EqualsPP(field, vx, vy);
}

// Evaluates truncation.
bool SatisfyTruncateErrorPP(size_t world_size, FieldType field,
                            const ArrayRef& px, const ArrayRef& py);

// parties random a public together.
ArrayRef RandP(FieldType field, size_t size, std::mt19937::result_type seed = 0,
               int min = 0, int max = 100);

bool RingEqual(const ArrayRef& a, const ArrayRef& b, size_t abs_err = 0);

bool VerifyCost(Kernel* kernel, std::string_view name, FieldType field,
                size_t numel, size_t npc, const Communicator::Stats& cost);

using CreateComputeFn = std::function<std::unique_ptr<Object>(
    const std::shared_ptr<link::Context>& lctx)>;

}  // namespace ppu::mpc::test
