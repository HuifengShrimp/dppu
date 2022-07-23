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


#include "ppu/hal/fxp.h"

#include "absl/numeric/bits.h"

#include "ppu/hal/const_util.h"
#include "ppu/hal/integer.h"
#include "ppu/hal/public_intrinsic.h"
#include "ppu/hal/ring.h"
#include "ppu/hal/type_cast.h"

namespace ppu::hal {
namespace {

Value f_sign(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);
  PPU_ENFORCE(x.is_fxp());

  // is_negative = x < 0 ? 1 : 0;
  const Value is_negative = _less(ctx, x, shaped_const(ctx, 0.0f, x.shape()));

  // sign = 1 - 2 * is_negative
  //      = +1 ,if x >= 0
  //      = -1 ,if x < 0
  return f_sub(
      ctx, shaped_const(ctx, 1.0f, is_negative.shape()),
      _mul(ctx, shaped_const(ctx, 2.0f, is_negative.shape()), is_negative)
          .as_fxp());
}

// see:
// https://github.com/facebookresearch/CrypTen/blob/5b9cf7a161606dbf517cae670b749869597fb10a/crypten/common/functions/power.py#L63-L97
// Coefficients should be ordered from the order 1 (linear) term first, ending
// with the highest order term. (Constant is not included).
Value f_polynomial(HalContext* ctx, const Value& x,
                   const std::vector<Value>& coeffs) {
  PPU_TRACE_OP(ctx, x);
  PPU_ENFORCE(x.is_fxp());
  PPU_ENFORCE(!coeffs.empty());

  Value x_pow = x;
  Value res = f_mul(ctx, x_pow, coeffs[0]);

  for (size_t i = 1; i < coeffs.size(); i++) {
    x_pow = f_mul(ctx, x_pow, x);
    res = f_add(ctx, res, f_mul(ctx, x_pow, coeffs[i]));
  }

  return res;
}

// Fill all bits after msb to 1.
//
// Algorithm, lets consider the msb only, in each iteration we fill
// [msb-2^k, msb) to 1.
//   x0:  010000000   ; x0
//   x1:  011000000   ; x0 | (x0>>1)
//   x2:  011110000   ; x1 | (x1>>2)
//   x3:  011111111   ; x2 | (x2>>4)
//
Value prefix_or(HalContext* ctx, const Value& x) {
  auto b0 = x;
  const size_t bit_width = x.elsize() * 8;
  for (size_t idx = 0; idx < absl::bit_width(bit_width); idx++) {
    const size_t offset = 1UL << idx;
    auto b1 = _rshift(ctx, b0, offset);
    b0 = _or(ctx, b0, b1);
  }
  return b0;
}

// Extract the most significant bit.
// see
// https://docs.oracle.com/javase/7/docs/api/java/lang/Integer.html#highestOneBit(int)
Value highestOneBit(HalContext* ctx, const Value& x) {
  auto y = prefix_or(ctx, x);
  auto y1 = _rshift(ctx, y, 1);
  return _xor(ctx, y, y1);
}

// Reference:
//   Charpter 3.4 Division @ Secure Computation With Fixed Point Number
//   http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.221.1305&rep=rep1&type=pdf
//
// Symbols:
//   f: number of fractional bits in fixed point.
//   m: the highest position of bit with value == 1.
//   e: m - f
//
//
// Goldschmdit main idea:
//
//   x = c * 2^{m}, fixed point representation of x
//   c = normalize(x), c \in [0.5, 1)
//
//   Initial guess w = (2.9142 - 2*c) * 2^{-m}.
//
//   Let r = w, denotes result
//   Let e = 1 - x*w, denotes error
//
//   Iteration (Reduce error)
//     r = r(1 + e)
//     e = e * e
//
//   return r
//
[[maybe_unused]] Value reciprocal_goldschmdit(HalContext* ctx, const Value& x) {
  // TODO(jint) calculate sign/abs value togather.
  auto x_sign = f_sign(ctx, x);
  auto x_abs = f_mul(ctx, x_sign, x);

  const size_t num_fxp_bits = ctx->FxpBits();
  auto x_msb = highestOneBit(ctx, x_abs);

  // factor = 2^{2f-m} = 2^{f-m} * 2^f, the fixed point repr of 2^{f-m}
  auto factor = _reverse_bits(ctx, x_msb, 0, 2 * num_fxp_bits).as_fxp();

  // compute normalize x_abs, [0.5, 1)
  auto x_norm = f_mul(ctx, x_abs, factor);

  // initial guess::
  //   r = 1/c = 2.9142 - 2c when c >= 0.5 and c < 1
  auto r =
      f_mul(ctx,
            f_sub(ctx, shaped_const(ctx, 2.9142f, x_norm.shape()),
                  f_mul(ctx, shaped_const(ctx, 2.0f, x_norm.shape()), x_norm)),
            factor);

  // The error.
  const auto& ones = shaped_const(ctx, 1.0f, x_norm.shape());
  auto e = f_sub(ctx, ones, f_mul(ctx, x_abs, r));

  const size_t config_num_iters =
      ctx->rt_config().fxp_reciprocal_goldschmdit_iters();
  const size_t num_iters = config_num_iters == 0 ? 2 : config_num_iters;

  for (size_t itr = 0; itr < num_iters; itr++) {
    r = f_mul(ctx, r, f_add(ctx, e, ones));
    e = f_square(ctx, e);
  }

  return f_mul(ctx, r, x_sign);
}

[[maybe_unused]] Value reciprocal_newton(HalContext* ctx, const Value& x) {
  // Note(jint), this initialize guess result is far warse than the
  // normalized-goldschmidt method.
  //
  // see https://lvdmaaten.github.io/publications/papers/crypten.pdf
  // Newton-Raphson.
  // Initialization to a decent estimate (found by qualitative inspection):
  //   1/x = 3 * exp(0.5 - x) + 0.003
  Value sign = f_sign(ctx, x);
  Value abs_x = f_mul(ctx, sign, x);
  Value res = f_add(
      ctx,
      f_mul(ctx, shaped_const(ctx, 3.0f, x.shape()),
            f_exp(ctx, f_sub(ctx, shaped_const(ctx, 0.5f, x.shape()), abs_x))),
      shaped_const(ctx, 0.003f, x.shape()));
  const size_t kIters = 5;
  for (size_t i = 0; i < kIters; i++) {
    Value step = f_sub(ctx, res, f_mul(ctx, abs_x, f_square(ctx, res)));

    res = f_add(ctx, res, step);
  }
  res = f_mul(ctx, sign, res);
  return res;
}

}  // namespace

Value f_square(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);

  PPU_ENFORCE(x.is_fxp());
  // TODO(jint) optimize me.

  return f_mul(ctx, x, x);
}

Value f_exp(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);

  PPU_ENFORCE(x.is_fxp());

  if (x.is_public()) {
    return f_exp_p(ctx, x);
  }

  const size_t config_iters = ctx->rt_config().fxp_exp_iters();
  const size_t num_iters = config_iters == 0 ? 8 : config_iters;
  // see https://lvdmaaten.github.io/publications/papers/crypten.pdf
  //   exp(x) = (1 + x / n) ^ n, when n is infinite large.
  Value res = f_add(ctx, _trunc(ctx, x, num_iters).as_fxp(),
                    shaped_const(ctx, 1.0f, x.shape()));

  for (size_t i = 0; i < num_iters; i++) {
    res = f_square(ctx, res);
  }

  return res;
}

Value f_negate(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);
  PPU_ENFORCE(x.is_fxp());

  return _negate(ctx, x).as_fxp();
}

Value f_abs(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);
  PPU_ENFORCE(x.is_fxp());

  const Value sign = f_sign(ctx, x);

  return f_mul(ctx, sign, x);
}

Value f_reciprocal(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);
  PPU_ENFORCE(x.is_fxp());

  if (x.is_public()) {
    return f_reciprocal_p(ctx, x);
  }

  return reciprocal_goldschmdit(ctx, x);
}

Value f_add(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);

  PPU_ENFORCE(x.is_fxp());
  PPU_ENFORCE(y.is_fxp());

  return _add(ctx, x, y).as_fxp();
}

Value f_sub(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);

  PPU_ENFORCE(x.is_fxp());
  PPU_ENFORCE(y.is_fxp());
  return f_add(ctx, x, f_negate(ctx, y));
}

Value f_mul(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);

  PPU_ENFORCE(x.is_fxp());
  PPU_ENFORCE(y.is_fxp());

  return _trunc(ctx, _mul(ctx, x, y)).as_fxp();
}

Value f_matmul(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);

  PPU_ENFORCE(x.is_fxp());
  PPU_ENFORCE(y.is_fxp());

  return _trunc(ctx, _matmul(ctx, x, y)).as_fxp();
}

Value f_div(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);

  PPU_ENFORCE(x.is_fxp());
  PPU_ENFORCE(y.is_fxp());

  return f_mul(ctx, x, f_reciprocal(ctx, y));
}

Value f_equal(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);

  PPU_ENFORCE(x.is_fxp());
  PPU_ENFORCE(y.is_fxp());

  return _eqz(ctx, f_sub(ctx, x, y)).as_int();
}

Value f_less(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);

  PPU_ENFORCE(x.is_fxp());
  PPU_ENFORCE(y.is_fxp());

  return _less(ctx, x, y).as_int();
}

// See P11, A.2.4 Logarithm and Exponent,
// https://lvdmaaten.github.io/publications/papers/crypten.pdf
// https://github.com/facebookresearch/CrypTen/blob/master/crypten/common/functions/approximations.py#L55-L104
Value f_log(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);

  PPU_ENFORCE(x.is_fxp());

  if (x.is_public()) {
    return f_log_p(ctx, x);
  }

  Value term_1 = f_div(ctx, x, shaped_const(ctx, 120.0f, x.shape()));
  Value term_2 = f_mul(
      ctx,
      f_exp(ctx, f_negate(ctx, f_add(ctx,
                                     f_mul(ctx, x,
                                           shaped_const(ctx, 2.0f, x.shape())),
                                     shaped_const(ctx, 1.0f, x.shape())))),
      shaped_const(ctx, 20.0f, x.shape()));
  Value y = f_add(ctx, f_sub(ctx, term_1, term_2),
                  shaped_const(ctx, 3.0f, x.shape()));

  std::vector<Value> coeffs;
  const size_t config_orders = ctx->rt_config().fxp_log_orders();
  const size_t num_order = config_orders == 0 ? 8 : config_orders;
  for (size_t i = 0; i < num_order; i++) {
    coeffs.emplace_back(shaped_const(ctx, 1.0f / (1.0f + i), x.shape()));
  }

  const size_t config_iters = ctx->rt_config().fxp_log_iters();
  const size_t num_iters = config_iters == 0 ? 3 : config_iters;
  for (size_t i = 0; i < num_iters; i++) {
    Value h = f_sub(ctx, shaped_const(ctx, 1.0f, x.shape()),
                    f_mul(ctx, x, f_exp(ctx, f_negate(ctx, y))));
    y = f_sub(ctx, y, f_polynomial(ctx, h, coeffs));
  }

  return y;
}

Value f_log1p(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);

  PPU_ENFORCE(x.is_fxp());

  return f_log(ctx, f_add(ctx, shaped_const(ctx, 1.0f, x.shape()), x));
}

Value f_floor(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);

  PPU_ENFORCE(x.is_fxp());

  return _lshift(ctx, _arshift(ctx, x, ctx->FxpBits()), ctx->FxpBits())
      .as_fxp();
}

Value f_ceil(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);

  PPU_ENFORCE(x.is_fxp());

  return f_floor(
      ctx,
      f_add(ctx, x,
            shaped_const(ctx, 1.0 - (1.0 / (1 << ctx->FxpBits())), x.shape())
                .as_fxp()));
}

}  // namespace ppu::hal
