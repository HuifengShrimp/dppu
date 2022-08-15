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

#include "ppu/hal/context.h"
#include "ppu/hal/value.h"

namespace ppu::hal {

// !!please read [README.md] for api naming conventions.

Value _p2s(HalContext* ctx, const Value& x);
Value _s2p(HalContext* ctx, const Value& x);

Value _negate_p(HalContext* ctx, const Value& x);
Value _negate_s(HalContext* ctx, const Value& x);

Value _eqz_p(HalContext* ctx, const Value& x);
Value _eqz_s(HalContext* ctx, const Value& x);

Value _lshift_p(HalContext* ctx, const Value& x, size_t bits);
Value _lshift_s(HalContext* ctx, const Value& x, size_t bits);

Value _rshift_p(HalContext* ctx, const Value& x, size_t bits);
Value _rshift_s(HalContext* ctx, const Value& x, size_t bits);

Value _arshift_p(HalContext* ctx, const Value& x, size_t bits);
Value _arshift_s(HalContext* ctx, const Value& x, size_t bits);
Value _truncpr_s(HalContext* ctx, const Value& x, size_t bits);

Value _add_pp(HalContext* ctx, const Value& x, const Value& y);
Value _add_sp(HalContext* ctx, const Value& x, const Value& y);
Value _add_ss(HalContext* ctx, const Value& x, const Value& y);

Value _mul_pp(HalContext* ctx, const Value& x, const Value& y);
Value _mul_sp(HalContext* ctx, const Value& x, const Value& y);
Value _mul_ss(HalContext* ctx, const Value& x, const Value& y);

Value _matmul_pp(HalContext* ctx, const Value& x, const Value& y);
Value _matmul_sp(HalContext* ctx, const Value& x, const Value& y);
Value _matmul_ss(HalContext* ctx, const Value& x, const Value& y);

Value _logreg_all(HalContext* ctx, const Value& x, const Value& w, const Value& y,size_t M, size_t N);

Value _and_pp(HalContext* ctx, const Value& x, const Value& y);
Value _and_sp(HalContext* ctx, const Value& x, const Value& y);
Value _and_ss(HalContext* ctx, const Value& x, const Value& y);

Value _xor_pp(HalContext* ctx, const Value& x, const Value& y);
Value _xor_sp(HalContext* ctx, const Value& x, const Value& y);
Value _xor_ss(HalContext* ctx, const Value& x, const Value& y);

Value _reverse_bits_p(HalContext* ctx, const Value& in, size_t start_idx,
                      size_t end_idx);

Value _reverse_bits_s(HalContext* ctx, const Value& in, size_t start_idx,
                      size_t end_idx);

// permutations is public.
Value _permute_p(HalContext* ctx, const Value& x, size_t dimension,
                 const Value& permutations);
// permutations is secret.
Value _permute_s(HalContext* ctx, const Value& x, size_t dimension,
                 const Value& permutations);

Value _msb_p(HalContext* ctx, const Value& x);
Value _msb_s(HalContext* ctx, const Value& x);

}  // namespace ppu::hal
