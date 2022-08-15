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


#include "ppu/hal/polymorphic.h"

#include "fmt/format.h"
#include "fmt/ostream.h"

#include "ppu/core/array_ref_util.h"
#include "ppu/hal/dispatch.h"
#include "ppu/hal/fxp.h"
#include "ppu/hal/integer.h"
#include "ppu/hal/io_ops.h"
#include "ppu/hal/ring.h"  // for fast fxp x int
#include "ppu/hal/shape_ops.h"
#include "ppu/hal/type_cast.h"
#include "ppu/utils/exception.h"

namespace ppu::hal {
namespace {

bool CrossIntFxp(const Value& x, const Value& y) {
  return (x.is_fxp() && y.is_int()) || (x.is_int() && y.is_fxp());
}

Value logisticMM1(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);

  // SigmoidMM1: f(x) = 0.5 + 0.125 * x
  const auto c1 = broadcast_to(ctx, make_public(ctx, 0.5f), x.shape());
  const auto c2 = broadcast_to(ctx, make_public(ctx, 0.125f), x.shape());
  return add(ctx, c1, mul(ctx, c2, x));
}

Value LogisticReal(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);

  // f(x) = 1/(1+exp(-x))
  const auto c1 = broadcast_to(ctx, make_public(ctx, 1.0f), x.shape());
  return reciprocal(ctx, add(ctx, c1, exp(ctx, negate(ctx, x))));
}

Value LogisticSEG3(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);

  // f(x) = 0.5 + 0.125x if -4 <= x <= 4
  //        1            if       x > 4
  //        0            if  -4 > x
  // Rounds = Gt + Mux*2 = 4 + Log(K)
  auto upper = broadcast_to(ctx, make_public(ctx, 1.0F), x.shape());
  auto lower = broadcast_to(ctx, make_public(ctx, 0.0F), x.shape());
  auto middle = logisticMM1(ctx, x);

  auto upper_bound = broadcast_to(ctx, make_public(ctx, 4.0F), x.shape());
  auto lower_bound = broadcast_to(ctx, make_public(ctx, -4.0F), x.shape());

  auto ret = select(ctx, greater(ctx, x, upper_bound), upper, middle);
  return select(ctx, less(ctx, x, lower_bound), lower, ret);
}

}  // namespace

Value identity(HalContext* ctx, const Value& x) {
  // This is a helper function, useful for lazy AB evaluation.
  auto zeros = broadcast_to(ctx, make_public(ctx, 0U), x.shape());
  return add(ctx, x, zeros);
}

Value add(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return DtypeBinaryDispatch<f_add, i_add, int2fxp>("add", ctx, x, y);
}

Value sub(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return DtypeBinaryDispatch<f_sub, i_sub, int2fxp>("sub", ctx, x, y);
}

Value mul(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  // fast dispath, avoid trunction cost
  if (CrossIntFxp(x, y)) {
    return _mul(ctx, x, y).as_fxp();
  }

  return DtypeBinaryDispatch<f_mul, i_mul, int2fxp>("mul", ctx, x, y);
}

Value matmul(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  // fast dispath, avoid trunction cost
  if (CrossIntFxp(x, y)) {
    return _matmul(ctx, x, y).as_fxp();
  }

  return DtypeBinaryDispatch<f_matmul, i_matmul, int2fxp>("matmul", ctx, x, y);
}

Value logreg(HalContext* ctx, const Value& x, const Value& w, const Value& y, size_t M, size_t N){
  PPU_TRACE_OP(ctx, x, w, y);

  //lj-todo
  return _logreg(ctx, x, w, y, M, N).as_fxp();
}

Value logical_not(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);

  auto ones = broadcast_to(ctx, make_public(ctx, 1U), x.shape());
  return i_sub(ctx, ones, x);
}

Value equal(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  PPU_ENFORCE(x.shape() == y.shape());

  // TODO(junfeng): Implement the real equal!
  return bitwise_and(ctx, logical_not(ctx, less(ctx, x, y)),
                     logical_not(ctx, less(ctx, y, x)));
}

Value not_equal(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  PPU_ENFORCE(x.shape() == y.shape());

  return logical_not(ctx, equal(ctx, x, y));
}

Value less(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  PPU_ENFORCE(x.shape() == y.shape());

  return DtypeBinaryDispatch<f_less, i_less, int2fxp>("less", ctx, x, y);
}

Value less_equal(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  PPU_ENFORCE(x.shape() == y.shape());

  // not (x > y)
  return logical_not(ctx, greater(ctx, x, y));
}

Value greater(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  PPU_ENFORCE(x.shape() == y.shape());

  return less(ctx, y, x);
}

Value greater_equal(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  PPU_ENFORCE(x.shape() == y.shape());

  // not (x < y)
  return logical_not(ctx, less(ctx, x, y));
}

Value negate(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);

  return DtypeUnaryDispatch<f_negate, i_negate>("negate", ctx, x);
}

Value abs(HalContext* ctx, const Value& in) {
  PPU_TRACE_OP(ctx, in);

  if (in.is_fxp()) {
    return f_abs(ctx, in);
  }
  return fxp2int(ctx, f_abs(ctx, int2fxp(ctx, in)));
}

Value exp(HalContext* ctx, const Value& a) {
  PPU_TRACE_OP(ctx, a);

  return f_exp(ctx, a.is_int() ? int2fxp(ctx, a) : a);
}

Value select(HalContext* ctx, const Value& pred, const Value& a,
             const Value& b) {
  PPU_TRACE_OP(ctx, pred, a, b);

  PPU_ENFORCE(pred.is_int());
  PPU_ENFORCE(a.shape() == b.shape());

  return add(ctx, b, mul(ctx, pred, sub(ctx, a, b)));
}

Value bitwise_and(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);

  PPU_ENFORCE(x.is_int() && y.is_int());
  PPU_ENFORCE(x.shape() == y.shape());

  return _and(ctx, x, y).as_int();
}

Value bitwise_xor(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  PPU_ENFORCE(x.is_int() && y.is_int());
  PPU_ENFORCE(x.shape() == y.shape());

  return _xor(ctx, x, y).as_int();
}

Value bitwise_or(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  PPU_ENFORCE(x.is_int() && y.is_int());
  PPU_ENFORCE(x.shape() == y.shape());

  return _or(ctx, x, y).as_int();
}

Value logistic(HalContext* ctx, const Value& in) {
  PPU_TRACE_OP(ctx, in);

  PPU_ENFORCE(in.is_fxp());

  // std::cout<<"*************************"<<std::endl;
  // std::cout<<ctx->rt_config().sigmoid_mode()<<std::endl;
  // std::cout<<"*************************"<<std::endl;

  switch (ctx->rt_config().sigmoid_mode()) {
    case ppu::SigmoidMode::DEFAULT:
    case MM1: {
      return logisticMM1(ctx, in);
    }
    case SEG3: {
      return LogisticSEG3(ctx, in);
    }
    case REAL: {
      return LogisticReal(ctx, in);
    }
    default: {
      PPU_THROW("Should not hit");
    }
  }
}

Value log(HalContext* ctx, const Value& in) {
  PPU_TRACE_OP(ctx, in);

  return f_log(ctx, in.is_int() ? int2fxp(ctx, in) : in);
}

Value log1p(HalContext* ctx, const Value& in) {
  PPU_TRACE_OP(ctx, in);

  return f_log1p(ctx, in.is_int() ? int2fxp(ctx, in) : in);
}

Value reciprocal(HalContext* ctx, const Value& in) {
  PPU_TRACE_OP(ctx, in);
  PPU_ENFORCE(in.is_fxp());

  return f_reciprocal(ctx, in);
}

Value floor(HalContext* ctx, const Value& in) {
  PPU_TRACE_OP(ctx, in);

  PPU_ENFORCE(in.is_fxp());

  return f_floor(ctx, in);
}

Value ceil(HalContext* ctx, const Value& in) {
  PPU_TRACE_OP(ctx, in);

  PPU_ENFORCE(in.is_fxp());

  return f_ceil(ctx, in);
}

Value max(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);

  PPU_ENFORCE(x.dtype() == y.dtype());

  return select(ctx, greater(ctx, x, y), x, y);
}

Value min(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);

  PPU_ENFORCE(x.dtype() == y.dtype());

  return select(ctx, less(ctx, x, y), x, y);
}

Value power(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);

  PPU_ENFORCE(x.dtype() == y.dtype());

  // x^y = e^(y*ln(x))
  auto ret = exp(ctx, mul(ctx, y, log(ctx, x)));
  if (x.is_int()) {
    return fxp2int(ctx, ret);
  }
  return ret;
}

Value div(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);

  if (y.is_int()) {
    return fxp2int(ctx, mul(ctx, x, reciprocal(ctx, int2fxp(ctx, y))));
  }
  return mul(ctx, x, reciprocal(ctx, y));
}

Value clamp(HalContext* ctx, const Value& minv, const Value& x,
            const Value& maxv) {
  PPU_TRACE_OP(ctx, minv, x, maxv);

  PPU_ENFORCE(minv.dtype() == maxv.dtype());
  PPU_ENFORCE(minv.dtype() == x.dtype());

  return min(ctx, max(ctx, minv, x), maxv);
}

Value bitcast(HalContext* ctx, const Value& x, DataType dtype, size_t elsize) {
  PPU_TRACE_OP(ctx, x, dtype);

  PPU_ENFORCE(x.is_public(), "bitcast a non-public is not supported yet");

  ppu::Type decode_type;
  ppu::Type reencode_type;
  switch (elsize) {
    case 32: {
      decode_type = (x.is_int() ? I32 : F32);
      reencode_type = (dtype == DT_FXP ? F32 : I32);
      break;
    }
    case 64: {
      decode_type = (x.is_int() ? I64 : F64);
      reencode_type = (dtype == DT_FXP ? F64 : I64);
      break;
    }
    default: {
      PPU_THROW("TODO");
    }
  }

  // decode->cast->encode
  auto t = decodeFromRing(
      {x.buf(), x.mpc_type(), x.shape(), x.strides(), x.offset()}, decode_type,
      ctx->FxpBits(), x.dtype());

  return makeValue(
      encodeToRing({t.buf(), reencode_type, t.shape(), t.strides(), t.offset()},
                   x.mpc_type(), ctx->FxpBits()),
      dtype);
}

Value left_sift(HalContext* ctx, const Value& x, size_t bits) {
  PPU_TRACE_OP(ctx, x, bits);

  return _lshift(ctx, x, bits).as_dtype(x.dtype());
}

Value right_shift_logical(HalContext* ctx, const Value& x, size_t bits) {
  PPU_TRACE_OP(ctx, x, bits);

  return _rshift(ctx, x, bits).as_dtype(x.dtype());
}

Value permute(HalContext* ctx, const Value& x, size_t dimension,
              const Value& permutations) {
  PPU_TRACE_OP(ctx, x, dimension, permutations);

  return _permute(ctx, x, dimension, permutations);
}



}  // namespace ppu::hal
