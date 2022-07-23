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


#include "ppu/hal/ring.h"

#include <array>

#include "ppu/hal/dispatch.h"
#include "ppu/hal/prot_wrapper.h"
#include "ppu/hal/shape_ops.h"

namespace ppu::hal {

#define DEF_UNARY_OP(Name, P, S)                    \
  Value Name(HalContext* ctx, const Value& x) {     \
    PPU_TRACE_OP(ctx, x);                           \
    return VtypeUnaryDispatch<P, S>(#Name, ctx, x); \
  }

DEF_UNARY_OP(_negate, _negate_p, _negate_s)

#undef DEF_UNARY_OP

#define DEF_BINARY_OP(Name, PP, SP, SS)                                  \
  Value Name(HalContext* ctx, const Value& x, const Value& y) {          \
    PPU_TRACE_OP(ctx, x, y);                                             \
    return VtypeCommutativeBinaryDispatch<PP, SP, SS>(#Name, ctx, x, y); \
  }

DEF_BINARY_OP(_add, _add_pp, _add_sp, _add_ss)
DEF_BINARY_OP(_mul, _mul_pp, _mul_sp, _mul_ss)
DEF_BINARY_OP(_and, _and_pp, _and_sp, _and_ss)
DEF_BINARY_OP(_xor, _xor_pp, _xor_sp, _xor_ss)

#undef DEF_BINARY_OP

Value _sub(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return _add(ctx, x, _negate(ctx, y));
}

Value _matmul(HalContext* ctx, const Value& x, const Value& y) {
  if (x.is_public() && y.is_public()) {
    return _matmul_pp(ctx, x, y);
  } else if (x.is_secret() && y.is_public()) {
    return _matmul_sp(ctx, x, y);
  } else if (x.is_public() && y.is_secret()) {
    return transpose(ctx,
                     _matmul_sp(ctx, transpose(ctx, y), transpose(ctx, x)));
  } else if (x.is_secret() && y.is_secret()) {
    return _matmul_ss(ctx, x, y);
  } else {
    PPU_THROW("unsupported op {} for x={}, y={}", "_matmul", x, y);
  }
}

Value _or(HalContext* ctx, const Value& x, const Value& y) {
  // X or Y = X xor Y xor (X and Y)
  return _xor(ctx, x, _xor(ctx, y, _and(ctx, x, y)));
}

Value _lshift(HalContext* ctx, const Value& x, size_t bits) {
  PPU_TRACE_OP(ctx, x, bits);
  if (x.is_public()) {
    return _lshift_p(ctx, x, bits);
  } else if (x.is_secret()) {
    return _lshift_s(ctx, x, bits);
  } else {
    PPU_THROW("unsupport unary op={} for {}", "_lshift", x);
  }
}

Value _rshift(HalContext* ctx, const Value& x, size_t bits) {
  PPU_TRACE_OP(ctx, x, bits);
  if (x.is_public()) {
    return _rshift_p(ctx, x, bits);
  } else if (x.is_secret()) {
    return _rshift_s(ctx, x, bits);
  } else {
    PPU_THROW("unsupport unary op={} for {}", "_rshift", x);
  }
}

Value _msb(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);

  Value msb = VtypeUnaryDispatch<_msb_p, _msb_s>("_msb", ctx, x);
  return msb.as_int();
}

Value _eqz(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);

  return VtypeUnaryDispatch<_eqz_p, _eqz_s>("_equal", ctx, x);
}

Value _less(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);

  // test msb(x-y) == 1
  return _msb(ctx, _sub(ctx, x, y));
}

Value _arshift(HalContext* ctx, const Value& x, size_t bits) {
  PPU_TRACE_OP(ctx, x, bits);
  bits = (bits == 0) ? ctx->FxpBits() : bits;

  if (x.is_public()) {
    return _arshift_p(ctx, x, bits);
  } else if (x.is_secret()) {
    return _arshift_s(ctx, x, bits);
  } else {
    PPU_THROW("unsupport unary op={} for {}", "_trunc", x);
  }
}

Value _trunc(HalContext* ctx, const Value& x, size_t bits) {
  PPU_TRACE_OP(ctx, x, bits);
  bits = (bits == 0) ? ctx->FxpBits() : bits;

  if (x.is_public()) {
    return _arshift_p(ctx, x, bits);
  } else if (x.is_secret()) {
    if (ctx->rt_config().disable_trunc_pr()) {
      return _arshift_s(ctx, x, bits);
    } else {
      return _truncpr_s(ctx, x, bits);
    }
  } else {
    PPU_THROW("unsupport unary op={} for {}", "_trunc", x);
  }
}

// swap bits of [start_idx, end_idx)
Value _reverse_bits(HalContext* ctx, const Value& x, size_t start_idx,
                    size_t end_idx) {
  PPU_TRACE_OP(ctx, x, start_idx, end_idx);

  if (x.is_public()) {
    return _reverse_bits_p(ctx, x, start_idx, end_idx);
  }
  if (x.is_secret()) {
    return _reverse_bits_s(ctx, x, start_idx, end_idx);
  }
  PPU_THROW("unsupport op={} for {}", "_reverse_bit", x);
}

Value _permute(HalContext* ctx, const Value& x, size_t dimension,
               const Value& permutations) {
  PPU_TRACE_OP(ctx, x, dimension, permutations);

  if (permutations.is_public()) {
    return _permute_p(ctx, x, dimension, permutations);
  }
  if (permutations.is_secret()) {
    return _permute_s(ctx, x, dimension, permutations);
  }
  PPU_THROW("unsupport op={} for {}", "_permute", permutations);
}

}  // namespace ppu::hal
