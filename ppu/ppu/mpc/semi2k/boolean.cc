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


#include "ppu/mpc/semi2k/boolean.h"

#include "ppu/core/array_ref_util.h"
#include "ppu/core/trace.h"
#include "ppu/mpc/interfaces.h"
#include "ppu/mpc/kernel.h"
#include "ppu/mpc/prg_state.h"
#include "ppu/mpc/semi2k/object.h"
#include "ppu/mpc/semi2k/type.h"
#include "ppu/mpc/util/communicator.h"
#include "ppu/mpc/util/ring_ops.h"

namespace ppu::mpc::semi2k {

ArrayRef ZeroB::proc(KernelEvalContext* ctx, FieldType field,
                     size_t size) const {
  PPU_TRACE_OP(this, size);

  auto* prg_state = ctx->caller()->getState<PrgState>();
  auto [r0, r1] = prg_state->genPrssPair(field, size);
  return ring_xor(r0, r1).as(makeType<BShrTy>(field));
}

ArrayRef B2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  PPU_TRACE_OP(this, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto out = comm->allReduce(ReduceOp::XOR, in, kName);
  return out.as(makeType<Ring2kPublTy>(field));
}

ArrayRef P2B::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  PPU_TRACE_OP(this, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto boolean = ctx->caller()->getInterface<IBoolean>();
  auto x = boolean->ZeroB(field, in.numel());

  if (comm->getRank() == 0) {
    ring_xor_(x, in);
  }

  return x.as(makeType<BShrTy>(field));
}

ArrayRef AndBP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);
  return ring_and(lhs, rhs).as(lhs.eltype());
}

ArrayRef AndBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* beaver = ctx->caller()->getState<Semi2kState>()->beaver();

  // generate beaver and triple.
  auto [a, b, c] = beaver->And(field, lhs.numel());

  // open x^a, y^b
  auto res =
      vectorize({ring_xor(lhs, a), ring_xor(rhs, b)}, [&](const ArrayRef& s) {
        return comm->allReduce(ReduceOp::XOR, s, kName);
      });
  auto x_a = std::move(res[0]);
  auto y_b = std::move(res[1]);

  // Zi = Ci ^ ((X ^ A) & Bi) ^ ((Y ^ B) & Ai) ^ <(X ^ A) & (Y ^ B)>
  auto z = ring_xor(ring_xor(ring_and(x_a, b), ring_and(y_b, a)), c);
  if (comm->getRank() == 0) {
    ring_xor_(z, ring_and(x_a, y_b));
  }

  return z.as(lhs.eltype());
}

ArrayRef XorBP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);

  auto* comm = ctx->caller()->getState<Communicator>();
  if (comm->getRank() == 0) {
    return ring_xor(lhs, rhs).as(lhs.eltype());
  }

  return lhs;
}

ArrayRef XorBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);
  return ring_xor(lhs, rhs).as(lhs.eltype());
}

ArrayRef LShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  PPU_TRACE_OP(this, in, bits);
  return ring_lshift(in, bits).as(in.eltype());
}

ArrayRef RShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  PPU_TRACE_OP(this, in, bits);
  return ring_rshift(in, bits).as(in.eltype());
}

ArrayRef ARShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t bits) const {
  PPU_TRACE_OP(this, in, bits);
  return ring_arshift(in, bits).as(in.eltype());
}

ArrayRef ReverseBitsB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                            size_t start, size_t end) const {
  PPU_TRACE_OP(this, in, start, end);
  return ring_reverse_bits(in, start, end).as(in.eltype());
}

}  // namespace ppu::mpc::semi2k
