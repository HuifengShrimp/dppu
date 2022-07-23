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
#include "ppu/core/type.h"

namespace ppu::mpc {

ArrayRef ring_rand(FieldType field, size_t size, uint128_t prg_seed,
                   uint64_t* prg_counter);

ArrayRef ring_zeros(FieldType field, size_t size);

ArrayRef ring_ones(FieldType field, size_t size);

ArrayRef ring_others(FieldType field, size_t size, int32_t value);

ArrayRef assignment(FieldType field, ArrayRef &res, int32_t pos, int32_t value); 

ArrayRef ring_randbit(FieldType field, size_t size);

// signed 2's complement negation.
ArrayRef ring_neg(const ArrayRef& x);
void ring_neg_(ArrayRef& x);

ArrayRef ring_add(const ArrayRef& x, const ArrayRef& y);
void ring_add_(ArrayRef& x, const ArrayRef& y);

ArrayRef ring_sub(const ArrayRef& x, const ArrayRef& y);
void ring_sub_(ArrayRef& x, const ArrayRef& y);

ArrayRef ring_mul(const ArrayRef& x, const ArrayRef& y);
void ring_mul_(ArrayRef& x, const ArrayRef& y);

ArrayRef ring_mmul(const ArrayRef& x, const ArrayRef& y, int64_t M, int64_t N,
                   int64_t K);

ArrayRef ring_not(const ArrayRef& x);
void ring_not_(ArrayRef& x);

ArrayRef ring_and(const ArrayRef& x, const ArrayRef& y);
void ring_and_(ArrayRef& x, const ArrayRef& y);

ArrayRef ring_xor(const ArrayRef& x, const ArrayRef& y);
void ring_xor_(ArrayRef& x, const ArrayRef& y);

ArrayRef ring_arshift(const ArrayRef& x, size_t bits);
void ring_arshift_(ArrayRef& x, size_t bits);

ArrayRef ring_rshift(const ArrayRef& x, size_t bits);
void ring_rshift_(ArrayRef& x, size_t bits);

ArrayRef ring_lshift(const ArrayRef& x, size_t bits);
void ring_lshift_(ArrayRef& x, size_t bits);

ArrayRef ring_reverse_bits(const ArrayRef& x, size_t start, size_t end);
void ring_reverse_bits_(ArrayRef& x, size_t start, size_t end);

}  // namespace ppu::mpc
