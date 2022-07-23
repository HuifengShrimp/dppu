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


#include "ppu/hal/integer.h"

#include "ppu/hal/prot_wrapper.h"
#include "ppu/hal/ring.h"

namespace ppu::hal {

#define DEF_UNARY_OP(Name, Fn2K)                \
  Value Name(HalContext* ctx, const Value& x) { \
    PPU_TRACE_OP(ctx, x);                       \
    PPU_ENFORCE(x.is_int());                    \
    return Fn2K(ctx, x).as_int();               \
  }

/*           name,     op_2k */
DEF_UNARY_OP(i_negate, _negate)

#undef DEF_UNARY_OP

#define DEF_BINARY_OP(Name, Fn2K)                               \
  Value Name(HalContext* ctx, const Value& x, const Value& y) { \
    PPU_TRACE_OP(ctx, x, y);                                    \
    PPU_ENFORCE(x.is_int());                                    \
    PPU_ENFORCE(y.is_int());                                    \
    return Fn2K(ctx, x, y).as_int();                            \
  }

DEF_BINARY_OP(i_add, _add)
DEF_BINARY_OP(i_mul, _mul)
DEF_BINARY_OP(i_matmul, _matmul)
DEF_BINARY_OP(i_less, _less)

#undef DEF_BINARY_OP

Value i_equal(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);

  PPU_ENFORCE(x.is_int());
  PPU_ENFORCE(y.is_int());

  return _eqz(ctx, i_sub(ctx, x, y)).as_int();
}

Value i_sub(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  PPU_ENFORCE(x.is_int());
  PPU_ENFORCE(y.is_int());
  return i_add(ctx, x, i_negate(ctx, y));
}

}  // namespace ppu::hal
