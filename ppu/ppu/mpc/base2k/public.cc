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


#include "ppu/mpc/base2k/public.h"

#include "ppu/core/array_ref_util.h"
#include "ppu/core/trace.h"
#include "ppu/mpc/prg_state.h"
#include "ppu/mpc/util/ring_ops.h"

namespace ppu::mpc::base2k {

void RandP::evaluate(KernelEvalContext* ctx) const {
  ctx->setOutput(
      proc(ctx, ctx->getParam<FieldType>(0), ctx->getParam<size_t>(1)));
}

ArrayRef RandP::proc(KernelEvalContext* ctx, FieldType field,
                     size_t size) const {
  PPU_TRACE_OP(this, size);
  auto* state = ctx->caller()->getState<PrgState>();
  return state->genPubl(field, size).as(makeType<Ring2kPublTy>(field));
}

ArrayRef NegP::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  PPU_TRACE_OP(this, in);
  const auto field = in.eltype().as<Ring2k>()->field();
  return ring_neg(in).as(makeType<Ring2kPublTy>(field));
}

ArrayRef EqzP::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  PPU_TRACE_OP(this, in);
  const auto field = in.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    const auto& in_xt = xt_adapt<ring2k_t>(in);
    auto r = xt::equal(in_xt, xt::zeros_like(in_xt));
    return make_array(r, makeType<Ring2kPublTy>(field));
  });
}

ArrayRef AddPP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);
  PPU_ENFORCE(lhs.eltype() == rhs.eltype());
  return ring_add(lhs, rhs).as(lhs.eltype());
}

ArrayRef MulPP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);
  PPU_ENFORCE(lhs.eltype() == rhs.eltype());
  return ring_mul(lhs, rhs).as(lhs.eltype());
}

ArrayRef MatMulPP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                        const ArrayRef& rhs, int64_t M, int64_t N,
                        int64_t K) const {
  PPU_TRACE_OP(this, lhs, rhs);
  PPU_ENFORCE(lhs.eltype() == rhs.eltype());
  return ring_mmul(lhs, rhs, M, N, K).as(lhs.eltype());
}

ArrayRef AndPP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);
  PPU_ENFORCE(lhs.eltype() == rhs.eltype());
  return ring_and(lhs, rhs).as(lhs.eltype());
}

ArrayRef XorPP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);
  PPU_ENFORCE(lhs.eltype() == rhs.eltype());
  return ring_xor(lhs, rhs).as(lhs.eltype());
}

ArrayRef LShiftP::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  PPU_TRACE_OP(this, in, bits);
  return ring_lshift(in, bits).as(in.eltype());
}

ArrayRef RShiftP::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  PPU_TRACE_OP(this, in, bits);
  return ring_rshift(in, bits).as(in.eltype());
}

ArrayRef ReverseBitsP::proc(KernelEvalContext* ctx, const ArrayRef& in,
                            size_t start, size_t end) const {
  PPU_TRACE_OP(this, in, start, end);
  return ring_reverse_bits(in, start, end).as(in.eltype());
}

ArrayRef ARShiftP::proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t bits) const {
  PPU_TRACE_OP(this, in, bits);
  return ring_arshift(in, bits).as(in.eltype());
}

ArrayRef MsbP::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  PPU_TRACE_OP(this, in);
  return ring_rshift(in, in.elsize() * 8 - 1).as(in.eltype());
}
}  // namespace ppu::mpc::base2k
