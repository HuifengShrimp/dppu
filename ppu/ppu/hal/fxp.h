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

Value f_negate(HalContext* ctx, const Value& x);

Value f_abs(HalContext* ctx, const Value& x);

Value f_reciprocal(HalContext* ctx, const Value& x);

Value f_add(HalContext* ctx, const Value& x, const Value& y);

Value f_sub(HalContext* ctx, const Value& x, const Value& y);

Value f_mul(HalContext* ctx, const Value& x, const Value& y);

Value f_matmul(HalContext* ctx, const Value& x, const Value& y);

Value f_div(HalContext* ctx, const Value& x, const Value& y);

Value f_square(HalContext* ctx, const Value& x);

Value f_exp(HalContext* ctx, const Value& x);

Value f_equal(HalContext* ctx, const Value& x, const Value& y);

Value f_less(HalContext* ctx, const Value& x, const Value& y);

Value f_log1p(HalContext* ctx, const Value& x);

Value f_log(HalContext* ctx, const Value& x);

Value f_floor(HalContext* ctx, const Value& x);

Value f_ceil(HalContext* ctx, const Value& x);

}  // namespace ppu::hal
