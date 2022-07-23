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


#include "ppu/hal/public_intrinsic.h"

#include "ppu/core/array_ref_util.h"

namespace ppu::hal {
namespace {

Value applyFloatingPointFn(
    HalContext* ctx, const Value& in,
    std::function<NdArrayRef(const xt::xarray<float>&)> fn) {
  PPU_TRACE_OP(ctx, in);
  PPU_ENFORCE(in.is_public());
  PPU_ENFORCE(in.dtype() == DT_FXP, "expected fxp, got={}", in.dtype());

  const Type ring_ty = makeType<RingTy>(ctx->GetField());
  // decode to floating point
  const auto raw =
      decodeFromRing(in.as(ring_ty), F32, ctx->FxpBits(), in.dtype());

  DataType dtype;
  const auto out =
      encodeToRing(fn(xt_adapt<float>(raw)), ring_ty, ctx->FxpBits(), &dtype);
  return makeValue(out.as(in.mpc_type()), dtype);
}

}  // namespace

Value f_reciprocal_p(HalContext* ctx, const Value& in) {
  PPU_TRACE_OP(ctx, in);

  return applyFloatingPointFn(ctx, in, [&](const xt::xarray<float>& farr) {
    return make_ndarray(1.0f / farr);
  });
}

Value f_log_p(HalContext* ctx, const Value& in) {
  PPU_TRACE_OP(ctx, in);
  return applyFloatingPointFn(ctx, in, [&](const xt::xarray<float>& farr) {
    return make_ndarray(xt::log(farr));
  });
}

Value f_exp_p(HalContext* ctx, const Value& in) {
  PPU_TRACE_OP(ctx, in);
  return applyFloatingPointFn(ctx, in, [&](const xt::xarray<float>& farr) {
    return make_ndarray(xt::exp(farr));
  });
}

}  // namespace ppu::hal
