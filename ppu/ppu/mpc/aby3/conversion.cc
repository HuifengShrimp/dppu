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


#include "ppu/mpc/aby3/conversion.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xvectorize.hpp"

#include "ppu/core/array_ref_util.h"
#include "ppu/core/trace.h"
#include "ppu/link/link.h"
#include "ppu/mpc/aby3/defs.h"
#include "ppu/mpc/aby3/type.h"
#include "ppu/mpc/interfaces.h"
#include "ppu/mpc/prg_state.h"
#include "ppu/mpc/util/circuits.h"
#include "ppu/mpc/util/communicator.h"

namespace ppu::mpc::aby3 {

// Referrence:
// ABY3: A Mixed Protocol Framework for Machine Learning
// P16 5.3 Share Conversions, Bit Decomposition
// https://eprint.iacr.org/2018/403.pdf
//
// Latency: 2 + log(nbits) from 1 rotate and 1 ppa.
//
// See:
// https://github.com/tf-encrypted/tf-encrypted/blob/master/tf_encrypted/protocol/aby3/aby3.py#L2889
ArrayRef A2B::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  PPU_TRACE_OP(this, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* prg_state = ctx->caller()->getState<PrgState>();
  auto boolean = ctx->caller()->getInterface<IBoolean>();

  // Let
  //   X = [(x0, x1), (x1, x2), (x2, x0)] as input.
  //   Z = (z0, z1, z2) as boolean zero share.
  //
  // Construct
  //   M = [((x0+x1)^z0, z1) (z1, z2), (z2, (x0+x1)^z0)]
  //   N = [(0, 0), (0, x2), (x2, 0)]
  // Then
  //   Y = PPA(M, N) as the output.
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    // in
    const auto& x = xt_adapt<share_t>(in);
    const auto& x1 = xt::real(x);
    const auto& x2 = xt::imag(x);

    // gen (z0, z1, z2)
    auto [r0, r1] = prg_state->genPrssPair(field, in.numel());
    auto z = xt_adapt<ring2k_t>(r0) ^ xt_adapt<ring2k_t>(r1);

    // build m = [((x0+x1)^z0, z1) (z1, z2), (z2, (x0+x1)^z0)]
    //       n = [(0, 0), (0, x2), (x2, 0)]
    xt::xarray<share_t> m = xt::zeros<share_t>({x.size()});
    auto m1 = xt::real(m);
    auto m2 = xt::imag(m);
    xt::xarray<share_t> n = xt::zeros<share_t>({x.size()});
    auto n1 = xt::real(n);
    auto n2 = xt::imag(n);

    m1 = z;
    if (comm->getRank() == 0) {
      m1 = z ^ (x1 + x2);
    } else if (comm->getRank() == 1) {
      n2 = x2;
    } else if (comm->getRank() == 2) {
      n1 = x1;
    }
    m2 = comm->rotate(m1, _kName);

    // Shr(x) = [in1, in2, 0]
    // Shr(y) = [0, 0, in3]
    auto ty = makeType<BShrTy>(field);
    return boolean->AddBB(make_array(m, ty), make_array(n, ty));
  });
}

// Referrence:
// IV.E Boolean to Arithmetic Sharing (B2A), extended to 3pc settings.
// https://encrypto.de/papers/DSZ15.pdf
//
// Latency: 4 + log(nbits) - 3 rotate + 1 send/rec + 1 ppa.
// TODO(junfeng): Optimize anount of comm.
ArrayRef B2A::proc(KernelEvalContext* ctx, const ArrayRef& x) const {
  PPU_TRACE_OP(this, x);

  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* prg_state = ctx->caller()->getState<PrgState>();
  auto* state = ctx->caller()->getState<Aby3State>();
  auto boolean = ctx->caller()->getInterface<IBoolean>();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    auto rra = xt::empty<share_t>({x.numel()});
    auto ra1 = xt::real(rra);
    auto ra2 = xt::imag(rra);

    auto rtmp = prg_state->genPriv(field, rra.size());
    ra1 = xt_adapt<ring2k_t>(rtmp);
    ra2 = comm->rotate(ra1, _kName);

    auto rb = xt::empty<share_t>(rra.shape());
    auto rb1 = xt::real(rb);
    auto rb2 = xt::imag(rb);

    auto [z0, z1] = prg_state->genPrssPair(field, x.numel());
    rb1 = xt_adapt<ring2k_t>(z0) ^ xt_adapt<ring2k_t>(z1);
    if (comm->getRank() == 1) {
      rb1 ^= (ra1 + ra2);
    }
    rb2 = comm->rotate(rb1, _kName);

    auto ty = makeType<BShrTy>(field);
    auto x_plus_r = boolean->AddBB(x, make_array(rb, ty));
    auto xx_plus_r = xt_adapt<share_t>(x_plus_r);
    const auto x_plus_r1 = xt::real(xx_plus_r);
    const auto x_plus_r2 = xt::imag(xx_plus_r);

    auto y = xt::empty<share_t>(rra.shape());
    auto y1 = xt::real(y);
    auto y2 = xt::imag(y);

    y1 = -ra1;
    if (state->lctx()->Rank() == 0) {
      auto buf = state->lctx()->Recv(2, _kName);
      const auto x_plus_r3 = detail::BuildXtensor<ring2k_t>(rra.shape(), buf);
      y1 = (x_plus_r1 ^ x_plus_r2 ^ x_plus_r3);
    } else if (state->lctx()->Rank() == 2) {
      state->lctx()->SendAsync(0, detail::SerializeXtensor(x_plus_r1), _kName);
    }

    y2 = comm->rotate(y1, _kName);

    return make_array(y, ty);
  });
}

// Referrence:
// 5.4.1 Semi-honest Security
// https://eprint.iacr.org/2018/403.pdf
//
// Latency: 2.
ArrayRef B2AByOT::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t valid_bits) const {
  PPU_TRACE_OP(this, in, valid_bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* state = ctx->caller()->getState<Aby3State>();
  auto* prg_state = ctx->caller()->getState<PrgState>();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    const auto& in_x = xt_adapt<share_t>(in);
    const auto& x1 = xt::real(in_x);
    const auto& x2 = xt::imag(in_x);

    // 1. zero share
    auto res = xt::empty<share_t>(in_x.shape());
    auto res_x1 = xt::real(res);
    auto res_x2 = xt::imag(res);

    auto [r0, r1] = prg_state->genPrssPair(field, in.numel());
    res_x1 = xt_adapt<ring2k_t>(r0) - xt_adapt<ring2k_t>(r1);

    // 2. process every valid bit from the least significant bit, generate
    // three-party (sender, receiver, helper) -- (rank0, rank1, rank2)
    std::vector<ArrayRef> p0_v;
    std::vector<ArrayRef> p1_v;
    std::vector<ArrayRef> c_v;
    auto publ_ty = makeType<Ring2kPublTy>(field);
    for (size_t i = 0; i < std::min(valid_bits, sizeof(ring2k_t) * 8); ++i) {
      const ring2k_t a = static_cast<ring2k_t>(1) << i;

      if (state->lctx()->Rank() == 0) {
        auto vr = prg_state->genPriv(field, in_x.size());
        const auto& r = xt_adapt<ring2k_t>(vr);
        auto t0 = ((x1 & a) ^ (x2 & a)) - r;
        auto t1 = (a ^ (x1 & a) ^ (x2 & a)) - r;

        res_x1 += r;
        p0_v.emplace_back(make_array(t0, publ_ty));
        p1_v.emplace_back(make_array(t1, publ_ty));
      } else if (state->lctx()->Rank() == 1) {
        auto t1 = (x2 >> i) & 1;
        c_v.emplace_back(make_array(t1, publ_ty));
      } else if (state->lctx()->Rank() == 2) {
        auto t1 = (x1 >> i) & 1;
        c_v.emplace_back(make_array(t1, publ_ty));
      }
    }  // end for

    // 3. call three-party ot
    if (state->lctx()->Rank() == 0) {
      state->ot()->OTSend(absl::Span<const ArrayRef>(p0_v.data(), p0_v.size()),
                          absl::Span<const ArrayRef>(p1_v.data(), p1_v.size()));
    } else if (state->lctx()->Rank() == 1) {
      std::vector<ArrayRef> res_v(c_v.size());
      state->ot()->OTRecv(absl::Span<const ArrayRef>(c_v.data(), c_v.size()),
                          absl::MakeSpan(res_v));

      for (size_t i = 0; i < res_v.size(); ++i) {
        res_x1 += xt_adapt<ring2k_t>(res_v[i]);
      }
    } else if (state->lctx()->Rank() == 2) {
      state->ot()->OTHelp(absl::Span<const ArrayRef>(c_v.data(), c_v.size()));
    }

    // 4. rotate
    res_x2 = comm->rotate(res_x1, _kName);

    auto ty = makeType<AShrTy>(field);
    return make_array(res, ty);
  });
}

ArrayRef AddBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto boolean = ctx->caller()->getInterface<IBoolean>();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    CircuitBasicBlock<ArrayRef> cbb;
    {
      cbb.num_bits = sizeof(share_t) * 8;
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

    return KoggleStoneAdder<ArrayRef>(lhs, rhs, cbb);
  });
}

}  // namespace ppu::mpc::aby3
