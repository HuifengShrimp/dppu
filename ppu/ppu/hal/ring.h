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

// this module implements ops x ring 2k space WITHOUT dtype check.
//
// for example, when multiply sfxp with sint, it first dispatch to `_mul`, which
// multiply the underline data x a typed unchecked way, then set the result
// dtype to fxp.
Value _negate(HalContext* ctx, const Value& x);

Value _add(HalContext* ctx, const Value& x, const Value& y);

Value _sub(HalContext* ctx, const Value& x, const Value& y);

Value _mul(HalContext* ctx, const Value& x, const Value& y);

// Note: no div (aka, multiplicative inverse), since protocol may works on 2^k
// ring.
// Value _div(HalContext* ctx, const Value& x, const Value& y);

Value _matmul(HalContext* ctx, const Value& x, const Value& y);

Value _and(HalContext* ctx, const Value& x, const Value& y);

Value _xor(HalContext* ctx, const Value& x, const Value& y);

Value _or(HalContext* ctx, const Value& x, const Value& y);

Value _msb(HalContext* ctx, const Value& x);

Value _eqz(HalContext* ctx, const Value& x);

Value _less(HalContext* ctx, const Value& x, const Value& y);

Value _lshift(HalContext* ctx, const Value& x, size_t bits);

Value _rshift(HalContext* ctx, const Value& x, size_t bits);

Value _arshift(HalContext* ctx, const Value& x, size_t bits);

Value _trunc(HalContext* ctx, const Value& x, size_t bits = 0);

// swap bits of [start_idx, end_idx)
// TODO(jint) use bit decomposition to do this.
Value _reverse_bits(HalContext* ctx, const Value&, size_t start_idx,
                    size_t end_idx);

Value _permute(HalContext* ctx, const Value& x, size_t dimension,
               const Value& permutations);

}  // namespace ppu::hal
