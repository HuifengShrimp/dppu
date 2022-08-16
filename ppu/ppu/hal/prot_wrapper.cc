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


#include "ppu/hal/prot_wrapper.h"

#include <cstddef>
#include <tuple>
#include <vector>

#include "xtensor/xoperation.hpp"

#include "ppu/core/array_ref_util.h"
#include "ppu/core/shape_util.h"
#include "ppu/core/type_util.h"
#include "ppu/hal/permute_util.h"
#include "ppu/mpc/interfaces.h"
#include "ppu/utils/exception.h"
#include "ppu/utils/int128.h"

namespace ppu::hal {
namespace {

Value arrayToValue(const ArrayRef& mpc_arr, std::vector<int64_t> shape,
                   DataType dtype = DT_INVALID) {
  const Type ty = makeType<ValueTy>(dtype, mpc_arr.eltype());
  auto strides = compactStrides(shape);
  return Value(mpc_arr.buf(), ty, std::move(shape), std::move(strides),
               mpc_arr.offset());
}

ArrayRef getArray(const Value& v) {
  if (v.isCompact()) {
    return {v.buf(), v.mpc_type(), v.numel(), 1, v.offset()};
  }
  // Create a compact clone
  auto compact = v.clone();
  return {compact.buf(), v.mpc_type(), v.numel(), 1, compact.offset()};
}

std::tuple<int64_t, int64_t, int64_t> DeduceParamsForMatMul(
    const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs) {
  PPU_ENFORCE(!lhs.empty() && lhs.size() <= 2);
  PPU_ENFORCE(!rhs.empty() && rhs.size() <= 2);

  if (lhs.size() == 1 && rhs.size() == 1) {
    PPU_ENFORCE(lhs[0] == rhs[0]);
    return std::make_tuple(1, 1, rhs[0]);
  }
  if (lhs.size() == 1 && rhs.size() == 2) {
    PPU_ENFORCE(lhs[0] == rhs[0]);
    return std::make_tuple(1, rhs[1], rhs[0]);
  }
  if (lhs.size() == 2 && rhs.size() == 1) {
    PPU_ENFORCE(lhs[1] == rhs[0]);
    return std::make_tuple(lhs[0], 1, rhs[0]);
  }
  PPU_ENFORCE(lhs[1] == rhs[0]);
  return std::make_tuple(lhs[0], rhs[1], rhs[0]);
}

}  // namespace

// FIXME: Rethink about how strides and offset works on execution results

static std::unique_ptr<mpc::ICompute> compute(HalContext* hctx) {
  return hctx->prot()->getInterface<mpc::ICompute>();
}

//////////////////////////////////////////////////////////////
// Unary ops.
//////////////////////////////////////////////////////////////
Value _p2s(HalContext* ctx, const Value& in) {
  PPU_TRACE_OP(ctx, in);
  auto ret = compute(ctx)->P2S(getArray(in));
  return arrayToValue(ret, in.shape());
}

Value _s2p(HalContext* ctx, const Value& in) {
  PPU_TRACE_OP(ctx, in);
  auto ret = compute(ctx)->S2P(getArray(in));
  return arrayToValue(ret, in.shape());
}

Value _negate_p(HalContext* ctx, const Value& in) {
  PPU_TRACE_OP(ctx, in);
  return arrayToValue(compute(ctx)->NegP(getArray(in)), in.shape());
}

Value _negate_s(HalContext* ctx, const Value& in) {
  PPU_TRACE_OP(ctx, in);
  return arrayToValue(compute(ctx)->NegS(getArray(in)), in.shape());
}

Value _eqz_p(HalContext* ctx, const Value& in) {
  PPU_TRACE_OP(ctx, in);
  return arrayToValue(compute(ctx)->EqzP(getArray(in)), in.shape());
}

Value _eqz_s(HalContext* ctx, const Value& in) {
  PPU_TRACE_OP(ctx, in);
  return arrayToValue(compute(ctx)->EqzS(getArray(in)), in.shape());
}

Value _lshift_p(HalContext* ctx, const Value& in, size_t bits) {
  PPU_TRACE_OP(ctx, in, bits);
  return arrayToValue(compute(ctx)->LShiftP(getArray(in), bits), in.shape());
}

Value _lshift_s(HalContext* ctx, const Value& in, size_t bits) {
  PPU_TRACE_OP(ctx, in, bits);
  return arrayToValue(compute(ctx)->LShiftS(getArray(in), bits), in.shape());
}

Value _rshift_p(HalContext* ctx, const Value& in, size_t bits) {
  PPU_TRACE_OP(ctx, in, bits);
  return arrayToValue(compute(ctx)->RShiftP(getArray(in), bits), in.shape());
}

Value _rshift_s(HalContext* ctx, const Value& in, size_t bits) {
  PPU_TRACE_OP(ctx, in, bits);
  return arrayToValue(compute(ctx)->RShiftS(getArray(in), bits), in.shape());
}

Value _arshift_p(HalContext* ctx, const Value& in, size_t bits) {
  PPU_TRACE_OP(ctx, in, bits);
  return arrayToValue(compute(ctx)->ARShiftP(getArray(in), bits), in.shape());
}

Value _arshift_s(HalContext* ctx, const Value& in, size_t bits) {
  PPU_TRACE_OP(ctx, in, bits);
  return arrayToValue(compute(ctx)->ARShiftS(getArray(in), bits), in.shape());
}

Value _truncpr_s(HalContext* ctx, const Value& in, size_t bits) {
  PPU_TRACE_OP(ctx, in, bits);
  return arrayToValue(compute(ctx)->TruncPrS(getArray(in), bits), in.shape());
}

Value _reverse_bits_p(HalContext* ctx, const Value& in, size_t start_idx,
                      size_t end_idx) {
  PPU_TRACE_OP(ctx, in, start_idx, end_idx);
  return arrayToValue(
      compute(ctx)->ReverseBitsP(getArray(in), start_idx, end_idx), in.shape());
}

Value _reverse_bits_s(HalContext* ctx, const Value& in, size_t start_idx,
                      size_t end_idx) {
  PPU_TRACE_OP(ctx, in, start_idx, end_idx);
  return arrayToValue(
      compute(ctx)->ReverseBitsS(getArray(in), start_idx, end_idx), in.shape());
}

//////////////////////////////////////////////////////////////
// Binary ops.
//////////////////////////////////////////////////////////////
Value _add_pp(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return arrayToValue(compute(ctx)->AddPP(getArray(x), getArray(y)), x.shape());
}

Value _add_sp(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return arrayToValue(compute(ctx)->AddSP(getArray(x), getArray(y)), x.shape());
}

Value _add_ss(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return arrayToValue(compute(ctx)->AddSS(getArray(x), getArray(y)), x.shape());
}

Value _mul_pp(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return arrayToValue(compute(ctx)->MulPP(getArray(x), getArray(y)), x.shape());
}

Value _mul_sp(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return arrayToValue(compute(ctx)->MulSP(getArray(x), getArray(y)), x.shape());
}

Value _mul_ss(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return arrayToValue(compute(ctx)->MulSS(getArray(x), getArray(y)), x.shape());
}

Value _and_pp(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return arrayToValue(compute(ctx)->AndPP(getArray(x), getArray(y)), x.shape());
}

Value _and_sp(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return arrayToValue(compute(ctx)->AndSP(getArray(x), getArray(y)), x.shape());
}

Value _and_ss(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return arrayToValue(compute(ctx)->AndSS(getArray(x), getArray(y)), x.shape());
}

Value _xor_pp(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return arrayToValue(compute(ctx)->XorPP(getArray(x), getArray(y)), x.shape());
}

Value _xor_sp(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return arrayToValue(compute(ctx)->XorSP(getArray(x), getArray(y)), x.shape());
}

Value _xor_ss(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return arrayToValue(compute(ctx)->XorSS(getArray(x), getArray(y)), x.shape());
}

Value _matmul_pp(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  auto [m, n, k] = DeduceParamsForMatMul(x.shape(), y.shape());
  return arrayToValue(compute(ctx)->MatMulPP(getArray(x), getArray(y), m, n, k),
                      DeduceDotShape(x.shape(), y.shape()));
}

Value _matmul_sp(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  auto [m, n, k] = DeduceParamsForMatMul(x.shape(), y.shape());
  return arrayToValue(compute(ctx)->MatMulSP(getArray(x), getArray(y), m, n, k),
                      DeduceDotShape(x.shape(), y.shape()));
}

Value _matmul_ss(HalContext* ctx, const Value& x, const Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  auto [m, n, k] = DeduceParamsForMatMul(x.shape(), y.shape());
  return arrayToValue(compute(ctx)->MatMulSS(getArray(x), getArray(y), m, n, k),
                      DeduceDotShape(x.shape(), y.shape()));
}

Value _logreg(HalContext* ctx, const Value& x, const Value& w, const Value& y, size_t M, size_t N){
  PPU_TRACE_OP(ctx, x, w, y);
  return arrayToValue(compute(ctx)->LogReg(getArray(x), getArray(w), getArray(y), M , N), w.shape());
}

// TODO: this should not be here, the function will not be dispatched to mpc
// layer.
Value _permute_p(HalContext* ctx, const Value& x, size_t dimension,
                 const Value& permutations) {
  PPU_TRACE_OP(ctx, x, dimension, permutations);

  PPU_ENFORCE(permutations.is_public());

  return DISPATCH_ALL_FIELDS(ctx->GetField(), "_permute_p", [&]() {
    using U = typename std::make_unsigned<ring2k_t>::type;

    const auto& permutations_ref = getArray(permutations);

    const auto& permutations_xt = xt_adapt<ring2k_t>(permutations_ref);

    auto permutations_xt_casted = xt::cast<U>(permutations_xt);

    return permute(ctx, x, dimension, xt::eval(permutations_xt_casted));
  });
}

Value _permute_s(HalContext* ctx, const Value& x, size_t dimension,
                 const Value& permutations) {
  PPU_THROW("unimplemented.");
}

Value _msb_p(HalContext* ctx, const Value& in) {
  PPU_TRACE_OP(ctx, in);
  return arrayToValue(compute(ctx)->MsbP(getArray(in)), in.shape());
}

Value _msb_s(HalContext* ctx, const Value& in) {
  PPU_TRACE_OP(ctx, in);
  return arrayToValue(compute(ctx)->MsbS(getArray(in)), in.shape());
}

}  // namespace ppu::hal
