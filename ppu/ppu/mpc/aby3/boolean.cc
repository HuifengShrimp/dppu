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


#include "ppu/mpc/aby3/boolean.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xvectorize.hpp"

#include "ppu/core/array_ref.h"
#include "ppu/core/array_ref_util.h"
#include "ppu/core/trace.h"
#include "ppu/link/link.h"
#include "ppu/mpc/aby3/defs.h"
#include "ppu/mpc/aby3/type.h"
#include "ppu/mpc/interfaces.h"
#include "ppu/mpc/prg_state.h"
#include "ppu/mpc/util/communicator.h"

namespace ppu::mpc::aby3 {

ArrayRef B2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  PPU_TRACE_OP(this, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    PPU_ENFORCE(in.eltype().isa<BShare>());

    // in
    const auto x = xt_adapt<share_t>(in);
    const auto x1 = xt::real(x);
    const auto x2 = xt::imag(x);
    const auto x3 = comm->rotate(x2, _kName);

    // out
    auto ty = makeType<Ring2kPublTy>(field);
    return make_array(x1 ^ x2 ^ x3, ty);
  });
}

ArrayRef P2B::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  PPU_TRACE_OP(this, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* state = ctx->caller()->getState<Aby3State>();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    xt::xarray<share_t> x = xt::zeros<share_t>({in.numel()});
    auto x1 = xt::real(x);
    auto x2 = xt::imag(x);

    // make (p, 0, 0) share
    if (state->lctx()->Rank() == 0) {
      x1 = xt_adapt<ring2k_t>(in);
    } else if (state->lctx()->Rank() == 2) {
      x2 = xt_adapt<ring2k_t>(in);
    }

    return make_array(x, makeType<BShrTy>(field));
  });
}

ArrayRef AndBP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    // lhs
    const auto& lhs_x = xt_adapt<share_t>(lhs);
    const auto& x1 = xt::real(lhs_x);
    const auto& x2 = xt::imag(lhs_x);

    // rhs
    const auto y = xt_adapt<ring2k_t>(rhs);

    // ret
    auto z = xt::empty<share_t>(lhs_x.shape());
    auto z1 = xt::real(z);
    auto z2 = xt::imag(z);

    z1 = x1 & y;
    z2 = x2 & y;

    auto ty = makeType<BShrTy>(field);
    return make_array(z, ty);
  });
}

ArrayRef AndBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* prg_state = ctx->caller()->getState<PrgState>();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    // lhs
    const auto& x = xt_adapt<share_t>(lhs);
    const auto& x1 = xt::real(x);
    const auto& x2 = xt::imag(x);

    // rhs
    const auto& y = xt_adapt<share_t>(rhs);
    const auto& y1 = xt::real(y);
    const auto& y2 = xt::imag(y);

    // ret
    auto z = xt::empty<share_t>(x.shape());
    auto out1 = xt::real(z);
    auto out2 = xt::imag(z);

    auto [r0, r1] = prg_state->genPrssPair(field, lhs.numel());
    auto r = xt_adapt<ring2k_t>(r0) ^ xt_adapt<ring2k_t>(r1);

    // z1 := (x1&y1) ^ (x1&y2) ^ (x2&y1) ^ k1
    // z2 := (x2&y2) ^ (x2&y3) ^ (x3&y2) ^ k2
    // z3 := (x3&y3) ^ (x3&y1) ^ (x1&y3) ^ k3
    out1 = (x1 & y1) ^ (x1 & y2) ^ (x2 & y1) ^ r;
    out2 = comm->rotate(out1, _kName);

    auto ty = makeType<BShrTy>(field);
    return make_array(z, ty);
  });
}

ArrayRef XorBP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    // lhs
    const auto& lhs_x = xt_adapt<share_t>(lhs);
    const auto& x1 = xt::real(lhs_x);
    const auto& x2 = xt::imag(lhs_x);

    // rhs
    const auto& rhs_x = xt_adapt<ring2k_t>(rhs);

    // ret
    auto z = xt::empty<share_t>(lhs_x.shape());
    auto out1 = xt::real(z);
    auto out2 = xt::imag(z);

    out1 = x1 ^ rhs_x;
    out2 = x2 ^ rhs_x;

    auto ty = makeType<BShrTy>(field);
    return make_array(z, ty);
  });
}

ArrayRef XorBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    // lhs
    const auto& x = xt_adapt<share_t>(lhs);
    const auto& x1 = xt::real(x);
    const auto& x2 = xt::imag(x);

    // rhs
    const auto& y = xt_adapt<share_t>(rhs);
    const auto& y1 = xt::real(y);
    const auto& y2 = xt::imag(y);

    // ret
    auto z = xt::empty<share_t>(x.shape());
    auto out1 = xt::real(z);
    auto out2 = xt::imag(z);

    out1 = x1 ^ y1;
    out2 = x2 ^ y2;

    auto ty = makeType<BShrTy>(field);
    return make_array(z, ty);
  });
}

ArrayRef LShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  PPU_TRACE_OP(this, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    // in
    const auto& x = xt_adapt<share_t>(in);
    const auto& x1 = xt::real(x);
    const auto& x2 = xt::imag(x);

    // out
    auto out = xt::empty<share_t>(x.shape());
    auto out1 = xt::real(out);
    auto out2 = xt::imag(out);

    out1 = xt::left_shift(x1, bits);
    out2 = xt::left_shift(x2, bits);

    auto ty = makeType<BShrTy>(field);
    return make_array(out, ty);
  });
}

ArrayRef RShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  PPU_TRACE_OP(this, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    // in
    const auto& x = xt_adapt<share_t>(in);
    const auto& x1 = xt::real(x);
    const auto& x2 = xt::imag(x);

    // out
    auto out = xt::empty<share_t>(x.shape());
    auto out1 = xt::real(out);
    auto out2 = xt::imag(out);

    auto logic_right_shift = [&](ring2k_t x) {
      using U = typename std::make_unsigned<ring2k_t>::type;
      U y = *reinterpret_cast<U*>(&x);
      return y >> bits;
    };

    out1 = xt::vectorize(logic_right_shift)(x1);
    out2 = xt::vectorize(logic_right_shift)(x2);

    auto ty = makeType<BShrTy>(field);
    return make_array(out, ty);
  });
}

ArrayRef ARShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t bits) const {
  PPU_TRACE_OP(this, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    // in
    const auto& x = xt_adapt<share_t>(in);
    const auto& x1 = xt::real(x);
    const auto& x2 = xt::imag(x);

    // out
    auto out = xt::empty<share_t>(x.shape());
    auto out1 = xt::real(out);
    auto out2 = xt::imag(out);

    auto logic_right_shift = [&](ring2k_t x) {
      using U = typename std::make_signed<ring2k_t>::type;
      U y = *reinterpret_cast<U*>(&x);
      return y >> bits;
    };

    out1 = xt::vectorize(logic_right_shift)(x1);
    out2 = xt::vectorize(logic_right_shift)(x2);

    auto ty = makeType<BShrTy>(field);
    return make_array(out, ty);
  });
}

ArrayRef ReverseBitsB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                            size_t start, size_t end) const {
  PPU_TRACE_OP(this, in, start, end);

  const auto field = in.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    // in
    const auto& x = xt_adapt<share_t>(in);
    const auto& x1 = xt::real(x);
    const auto& x2 = xt::imag(x);

    // out
    auto out = xt::empty<share_t>(x.shape());
    auto out1 = xt::real(out);
    auto out2 = xt::imag(out);

    auto reverse_bits_fn = [&](ring2k_t x) {
      using U = typename std::make_unsigned<ring2k_t>::type;
      U y = *reinterpret_cast<U*>(&x);

      U tmp = 0U;
      for (size_t idx = start; idx < end; idx++) {
        if (y & ((U)1 << idx)) {
          tmp |= (U)1 << (end - 1 - idx);
        }
      }

      U mask = ((U)1U << end) - ((U)1U << start);
      return (y & ~mask) | tmp;
    };

    out1 = xt::vectorize(reverse_bits_fn)(x1);
    out2 = xt::vectorize(reverse_bits_fn)(x2);

    auto ty = makeType<BShrTy>(field);
    return make_array(out, ty);
  });
}

}  // namespace ppu::mpc::aby3
