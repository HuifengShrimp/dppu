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


#include "ppu/mpc/aby3/arithmetic.h"

#include "spdlog/spdlog.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xio.hpp"

#include "ppu/core/array_ref.h"
#include "ppu/core/array_ref_util.h"
#include "ppu/core/trace.h"
#include "ppu/link/link.h"
#include "ppu/mpc/aby3/defs.h"
#include "ppu/mpc/aby3/type.h"
#include "ppu/mpc/prg_state.h"
#include "ppu/mpc/util/communicator.h"
#include "ppu/mpc/util/linalg.h"

namespace ppu::mpc::aby3 {

ArrayRef A2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  PPU_TRACE_OP(this, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    const auto x = xt_adapt<share_t>(in);

    const auto x1 = xt::real(x);
    const auto x2 = xt::imag(x);
    const auto x3 = comm->rotate(x2, _kName);

    auto ty = makeType<Ring2kPublTy>(field);
    return make_array(x1 + x2 + x3, ty);
  });
}

ArrayRef P2A::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
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

    return make_array(x, makeType<AShrTy>(field));
  });
}

ArrayRef NegA::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  PPU_TRACE_OP(this, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    auto ty = makeType<AShrTy>(field);
    return make_array(-xt_adapt<share_t>(in), ty);
  });
}

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
ArrayRef AddAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* state = ctx->caller()->getState<Aby3State>();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    // lhs
    const auto& lhs_x = xt_adapt<share_t>(lhs);

    // rhs
    const auto& rhs_x = xt_adapt<ring2k_t>(rhs);

    // ret
    auto z = xt::empty<share_t>({lhs.numel()});
    auto z1 = xt::real(z);
    auto z2 = xt::imag(z);

    z = lhs_x;

    if (state->lctx()->Rank() == 0) {
      z2 += rhs_x;
    } else if (state->lctx()->Rank() == 1) {
      z1 += rhs_x;
    }

    auto ty = makeType<AShrTy>(field);
    return make_array(z, ty);
  });
}

ArrayRef AddAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    // lhs
    const auto& lhs_x = xt_adapt<share_t>(lhs);

    // rhs
    const auto& rhs_x = xt_adapt<share_t>(rhs);

    // ret
    auto ty = makeType<AShrTy>(field);
    return make_array(lhs_x + rhs_x, ty);
  });
}

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
ArrayRef MulAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    // lhs
    const auto& lhs_x = xt_adapt<share_t>(lhs);

    // rhs
    const auto& rhs_x = xt_adapt<ring2k_t>(rhs);

    // ret
    auto ty = makeType<AShrTy>(field);
    return make_array(lhs_x * rhs_x, ty);
  });
}

ArrayRef MulAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* prg_state = ctx->caller()->getState<PrgState>();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    // lhs
    const auto& lhs_x = xt_adapt<share_t>(lhs);
    const auto& x1 = xt::real(lhs_x);
    const auto& x2 = xt::imag(lhs_x);

    // rhs
    const auto& rhs_x = xt_adapt<share_t>(rhs);
    const auto& y1 = xt::real(rhs_x);
    const auto& y2 = xt::imag(rhs_x);

    // ret
    auto z = xt::empty<share_t>({lhs_x.size()});
    auto z1 = xt::real(z);
    auto z2 = xt::imag(z);

    auto [r0, r1] = prg_state->genPrssPair(field, lhs.numel());
    auto r = xt_adapt<ring2k_t>(r0) - xt_adapt<ring2k_t>(r1);

    // z1 := x1*y1 + x1*y2 + x2*y1 + k1
    // z2 := x2*y2 + x2*y3 + x3*y2 + k2
    // z3 := x3*y3 + x3*y1 + x1*y3 + k3
    z1 = x1 * y1 + x1 * y2 + x2 * y1 + r;
    z2 = comm->rotate(z1, _kName);

    auto ty = makeType<AShrTy>(field);
    return make_array(z, ty);
  });
}

////////////////////////////////////////////////////////////////////
// matmul family
////////////////////////////////////////////////////////////////////
ArrayRef MatMulAP::proc(KernelEvalContext* ctx, const ArrayRef& A,
                        const ArrayRef& B, int64_t M, int64_t N,
                        int64_t K) const {
  PPU_TRACE_OP(this, A, B);

  const auto field = A.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    // lhs
    const auto* x_r = reinterpret_cast<const ring2k_t*>(A.data());
    const auto* x_i = x_r + 1;

    // rhs
    const auto* y = reinterpret_cast<const ring2k_t*>(B.data());

    // ret
    auto ty = makeType<AShrTy>(field);
    ArrayRef z(ty, M * N);

    auto* z_r = reinterpret_cast<ring2k_t*>(z.data());
    auto* z_i = z_r + 1;

    const auto x_stride = 2 * A.stride();
    const auto y_stride = B.stride();
    const auto z_stride = 2 * z.stride();

    linalg::matmul(M, N, K, x_r, K * x_stride, x_stride, y, N * y_stride,
                   y_stride, z_r, N * z_stride, z_stride);

    linalg::matmul(M, N, K, x_i, K * x_stride, x_stride, y, N * y_stride,
                   y_stride, z_i, N * z_stride, z_stride);

    return z;
  });
}

ArrayRef MatMulAA::proc(KernelEvalContext* ctx, const ArrayRef& A,
                        const ArrayRef& B, int64_t M, int64_t N,
                        int64_t K) const {
  PPU_TRACE_OP(this, A, B);

  const auto field = A.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* prg_state = ctx->caller()->getState<PrgState>();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    // lhs
    const auto* x_r = static_cast<const ring2k_t*>(A.data());
    const auto* x_i = x_r + 1;

    // rhs
    const auto* y_r = static_cast<const ring2k_t*>(B.data());
    const auto* y_i = y_r + 1;

    auto [r0, r1] = prg_state->genPrssPair(field, M * N);
    auto r = xt_adapt<ring2k_t>(r0) - xt_adapt<ring2k_t>(r1);

    auto tmp1 = xt::empty<ring2k_t>({M * N});
    auto tmp2 = xt::empty<ring2k_t>({M * N});
    auto tmp3 = xt::empty<ring2k_t>({M * N});

    const auto x_stride = 2 * A.stride();
    const auto y_stride = 2 * B.stride();
    const auto tmp_stride = tmp1.strides()[0];

    linalg::matmul(M, N, K, x_r, K * x_stride, x_stride, y_r, N * y_stride,
                   y_stride, tmp1.data(), N * tmp_stride, tmp_stride);

    linalg::matmul(M, N, K, x_r, K * x_stride, x_stride, y_i, N * y_stride,
                   y_stride, tmp2.data(), N * tmp_stride, tmp_stride);

    linalg::matmul(M, N, K, x_i, K * x_stride, x_stride, y_r, N * y_stride,
                   y_stride, tmp3.data(), N * tmp_stride, tmp_stride);

    // ret
    auto z = xt::empty<share_t>({M, N});
    auto z1 = xt::real(z);
    auto z2 = xt::imag(z);

    z1 = xt::reshape_view(tmp1 + tmp2 + tmp3 + r, {M, N});
    z2 = comm->rotate(z1, _kName);

    auto ty = makeType<AShrTy>(field);
    return make_array(z, ty);
  });
}

// Refer to:
// Share Truncation I, 5.1 Fixed-point Arithmetic, P13,
// ABY3: A Mixed Protocol Framework for Machine Learning
// - https://eprint.iacr.org/2018/403.pdf
ArrayRef TruncPrA::proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t bits) const {
  PPU_TRACE_OP(this, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* state = ctx->caller()->getState<Aby3State>();
  auto* prg_state = ctx->caller()->getState<PrgState>();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using share_t = Share<ring2k_t>;

    auto z = xt::empty<share_t>({in.numel()});
    auto z1 = xt::real(z);
    auto z2 = xt::imag(z);

    const auto x = xt_adapt<share_t>(in);
    const auto x1 = xt::real(x);
    const auto x2 = xt::imag(x);

    const ring2k_t kScale = ring2k_t(1) << bits;

    auto [r0, r1] = prg_state->genPrssPair(field, in.numel());
    auto prev_r = xt_adapt<ring2k_t>(r0);
    auto self_r = xt_adapt<ring2k_t>(r1);

    if (state->lctx()->Rank() == 0) {
      auto buf = state->lctx()->Recv(1, _kName);

      z1 = x1 / kScale;
      z2 = detail::BuildXtensor<ring2k_t>(z.shape(), buf);
    } else if (state->lctx()->Rank() == 1) {
      z1 = (x1 + x2) / kScale - self_r;
      z2 = self_r;
      state->lctx()->SendAsync(0, detail::SerializeXtensor(z1), _kName);
    } else {
      z1 = prev_r;
      z2 = x2 / kScale;
    }

    auto ty = makeType<AShrTy>(field);
    return make_array(z, ty);
  });
}

}  // namespace ppu::mpc::aby3
