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


#include "ppu/mpc/semi2k/conversion.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

#include "ppu/core/array_ref_util.h"
#include "ppu/core/trace.h"
#include "ppu/core/vectorize.h"
#include "ppu/mpc/interfaces.h"
#include "ppu/mpc/prg_state.h"
#include "ppu/mpc/semi2k/object.h"
#include "ppu/mpc/semi2k/type.h"
#include "ppu/mpc/util/circuits.h"
#include "ppu/mpc/util/communicator.h"
#include "ppu/mpc/util/ring_ops.h"

namespace ppu::mpc::semi2k {

ArrayRef AddBB::proc(KernelEvalContext* ctx, const ArrayRef& x,
                     const ArrayRef& y) const {
  PPU_TRACE_OP(this, x, y);
  auto boolean = ctx->caller()->getInterface<IBoolean>();

  CircuitBasicBlock<ArrayRef> cbb;
  {
    cbb.num_bits = x.elsize() * 8;
    cbb._xor = [&](ArrayRef const& lhs, ArrayRef const& rhs) -> ArrayRef {
      return boolean->XorBB(lhs, rhs);
    };
    cbb._and = [&](ArrayRef const& lhs, ArrayRef const& rhs) -> ArrayRef {
      return boolean->AndBB(lhs, rhs);
    };
    cbb.lshift = [&](ArrayRef const& x, size_t bits) -> ArrayRef {
      return boolean->LShiftB(x, bits);
    };
    cbb.rshift = [&](ArrayRef const& x, size_t bits) -> ArrayRef {
      return boolean->RShiftB(x, bits);
    };
  }

  return KoggleStoneAdder<ArrayRef>(x, y, cbb);
}

ArrayRef A2B::proc(KernelEvalContext* ctx, const ArrayRef& x) const {
  PPU_TRACE_OP(this, x);

  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto boolean = ctx->caller()->getInterface<IBoolean>();

  std::vector<ArrayRef> bshrs;
  const auto bty = makeType<BShrTy>(field);
  for (size_t idx = 0; idx < comm->getWorldSize(); idx++) {
    auto b = boolean->ZeroB(field, x.numel());
    if (idx == comm->getRank()) {
      ring_xor_(b, x);
    }
    bshrs.push_back(b.as(bty));
  }

  ArrayRef res = vectorizedReduce(bshrs.begin(), bshrs.end(),
                                  [&](const ArrayRef& xx, const ArrayRef& yy) {
                                    return boolean->AddBB(xx, yy);
                                  });
  return res.as(makeType<BShrTy>(field));
}

ArrayRef B2A::proc(KernelEvalContext* ctx, const ArrayRef& x) const {
  PPU_TRACE_OP(this, x);

  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* prg_state = ctx->caller()->getState<PrgState>();
  auto boolean = ctx->caller()->getInterface<IBoolean>();

  auto r_v = prg_state->genPriv(field, x.numel());
  auto r_a = r_v.as(makeType<AShrTy>(field));

  // convert r to boolean share.
  auto r_b = boolean->A2B(r_a);

  // evaluate adder circuit on x & r, and reveal x+r
  auto x_plus_r = comm->allReduce(ReduceOp::XOR, boolean->AddBB(x, r_b), kName);

  // compute -r + (x+r)
  ring_neg_(r_a);
  if (comm->getRank() == 0) {
    ring_add_(r_a, x_plus_r);
  }
  return r_a;
}

ArrayRef B2A_Randbit::proc(KernelEvalContext* ctx, const ArrayRef& x) const {
  PPU_TRACE_OP(this, x);

  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* beaver = ctx->caller()->getState<Semi2kState>()->beaver();

  // TODO: customize bits.
  const size_t nbits = SizeOf(field) * 8;
  const size_t numel = x.numel();

  auto randbits = beaver->RandBit(field, x.numel() * nbits);
  auto res = ArrayRef(makeType<AShrTy>(field), numel);

  DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using namespace xt::placeholders;
    auto xb = xt_adapt<ring2k_t>(x);
    auto ya = xt_mutable_adapt<ring2k_t>(res);
    auto ra = xt_adapt<ring2k_t>(randbits);

    auto open = [&](const auto& ss) -> xt::xarray<ring2k_t> {
      // return AllReduce(comm->lctx_, ReduceOp::XOR, ss, "_");
      // TODO: there are too many memory copies.
      return xt_adapt<ring2k_t>(comm->allReduce(
          ReduceOp::XOR, make_array(ss, makeType<RingTy>(field)), "open"));
    };

    // algorithm begins.
    // Ref: III.D @ https://eprint.iacr.org/2019/599.pdf (SPDZ-2K primitives)
    xt::xarray<ring2k_t> rb = xt::zeros<ring2k_t>({numel});
    for (size_t i = 0; i < nbits; i++) {
      auto ra_i = xt::view(ra, xt::range(i, _, nbits));
      // randbit A2B, leave lsb only, then pack it.
      rb += (ra_i & 0x1) << i;
    }

    // open c = x ^ r
    auto c = open(xb ^ rb);

    ya = xt::zeros<ring2k_t>({numel});
    for (size_t i = 0; i < nbits; i++) {
      auto ra_i = xt::view(ra, xt::range(i, _, nbits));
      auto c_i = (c >> i) & 0x1;

      // compute c + (1 - 2*c) * <r>, then stack it.
      if (comm->getRank() == 0) {
        ya += (c_i + (1 - c_i * 2) * ra_i) << i;
      } else {
        ya += ((1 - c_i * 2) * ra_i) << i;
      }
    }
  });

  return res;
}

}  // namespace ppu::mpc::semi2k
