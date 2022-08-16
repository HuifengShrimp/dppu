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


#include "ppu/mpc/semi2k/arithmetic.h"

#include "ppu/core/array_ref_util.h"
#include "ppu/core/trace.h"
#include "ppu/core/vectorize.h"
#include "ppu/mpc/interfaces.h"
#include "ppu/mpc/prg_state.h"
#include "ppu/mpc/semi2k/object.h"
#include "ppu/mpc/semi2k/type.h"
#include "ppu/mpc/util/communicator.h"
#include "ppu/mpc/util/ring_ops.h"
#include "time.h"

namespace ppu::mpc::semi2k {

ArrayRef ZeroA::proc(KernelEvalContext* ctx, FieldType field,
                     size_t size) const {
  PPU_TRACE_OP(this, size);

  auto* prg_state = ctx->caller()->getState<PrgState>();

  auto [r0, r1] = prg_state->genPrssPair(field, size);
  return ring_sub(r0, r1).as(makeType<AShrTy>(field));
}

ArrayRef P2A::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  PPU_TRACE_OP(this, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto arith = ctx->caller()->getInterface<IArithmetic>();

  auto x = arith->ZeroA(field, in.numel());

  if (comm->getRank() == 0) {
    ring_add_(x, in);
  }

  return x.as(makeType<AShrTy>(field));
}

ArrayRef A2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  PPU_TRACE_OP(this, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto out = comm->allReduce(ReduceOp::ADD, in, kName);
  return out.as(makeType<Ring2kPublTy>(field));
}

ArrayRef NegA::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  PPU_TRACE_OP(this, in);
  return ring_neg(in).as(in.eltype());
}

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
ArrayRef AddAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);

  PPU_ENFORCE(lhs.numel() == rhs.numel());
  auto* comm = ctx->caller()->getState<Communicator>();

  if (comm->getRank() == 0) {
    return ring_add(lhs, rhs).as(lhs.eltype());
  }

  return lhs;
}

ArrayRef AddAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);

  PPU_ENFORCE(lhs.numel() == rhs.numel());
  PPU_ENFORCE(lhs.eltype() == rhs.eltype());

  return ring_add(lhs, rhs).as(lhs.eltype());
}

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
ArrayRef MulAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);

  return ring_mul(lhs, rhs).as(lhs.eltype());
}

ArrayRef MulAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);

  // std::cout<<"**********MulAA**********"<<std::endl;

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();

  // clock_t start1, start2, end1, end2;
  // start1 = clock();
  auto* beaver = ctx->caller()->getState<Semi2kState>()->beaver();
  auto [a, b, c] = beaver->Mul(field, lhs.numel());
  // end1 = clock();
  // std::cout<<"++++++++++beaver generation++++++++++"<<std::endl;
  // std::cout<<"++++++"<<(double)(end1-start1)/CLOCKS_PER_SEC<<"++++++"<<std::endl;
  // std::cout<<"+++++++++++++++++++++++++++++++++++"<<std::endl;


  // start2 = clock();
  // Open x-a & y-b
  auto res =
      vectorize({ring_sub(lhs, a), ring_sub(rhs, b)}, [&](const ArrayRef& s) {
        return comm->allReduce(ReduceOp::ADD, s, kName);
      });

  auto x_a = std::move(res[0]);
  auto y_b = std::move(res[1]);

  // Zi = Ci + (X - A) * Bi + (Y - B) * Ai + <(X - A) * (Y - B)>
  auto z = ring_add(ring_add(ring_mul(x_a, b), ring_mul(y_b, a)), c);
  if (comm->getRank() == 0) {
    // z += (X-A) * (Y-B);
    ring_add_(z, ring_mul(x_a, y_b));
  }

  // end2 = clock();

  // std::cout<<"++++++++++online++++++++++"<<std::endl;
  // std::cout<<"++++++"<<(double)(end2-start2)/CLOCKS_PER_SEC<<"++++++"<<std::endl;
  // std::cout<<"+++++++++++++++++++++++++++++++++++"<<std::endl;

  // std::cout<<"///////////////online//////////////"<<std::endl;
  // std::cout<<"//////"<<(double)(end2-start1)/CLOCKS_PER_SEC<<"//////"<<std::endl;
  // std::cout<<"/////////////////////////////////"<<std::endl;

  return z.as(lhs.eltype());
}

////////////////////////////////////////////////////////////////////
// matmul family
////////////////////////////////////////////////////////////////////
ArrayRef MatMulAP::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        const ArrayRef& y, int64_t M, int64_t N,
                        int64_t K) const {
  PPU_TRACE_OP(this, x, y);
  return ring_mmul(x, y, M, N, K).as(x.eltype());
}

ArrayRef MatMulAA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        const ArrayRef& y, int64_t M, int64_t N,
                        int64_t K) const {
  PPU_TRACE_OP(this, x, y);

  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();

  // clock_t start1, start2, end1, end2;
  // start1 = clock();

  auto* beaver = ctx->caller()->getState<Semi2kState>()->beaver();

  // generate beaver multiple triple.
  auto [a, b, c] = beaver->Dot(field, M, N, K);

  // end1 = clock();
  // std::cout<<"++++++++++beaver generation++++++++++"<<std::endl;
  // std::cout<<"++++++"<<(double)(end1-start1)/CLOCKS_PER_SEC<<"++++++"<<std::endl;
  // std::cout<<"+++++++++++++++++++++++++++++++++++"<<std::endl;

  // start2 = clock();
  // Open x-a & y-b
  auto res =
      vectorize({ring_sub(x, a), ring_sub(y, b)}, [&](const ArrayRef& s) {
        return comm->allReduce(ReduceOp::ADD, s, kName);
      });
  auto x_a = std::move(res[0]);
  auto y_b = std::move(res[1]);

  // Zi = Ci + (X - A) dot Bi + Ai dot (Y - B) + <(X - A) dot (Y - B)>
  auto z = ring_add(
      ring_add(ring_mmul(x_a, b, M, N, K), ring_mmul(a, y_b, M, N, K)), c);
  if (comm->getRank() == 0) {
    // z += (X-A) * (Y-B);
    ring_add_(z, ring_mmul(x_a, y_b, M, N, K));
  }
  // end2 = clock();
  // std::cout<<"++++++++++online++++++++++"<<std::endl;
  // std::cout<<"++++++"<<(double)(end2-start2)/CLOCKS_PER_SEC<<"++++++"<<std::endl;
  // std::cout<<"+++++++++++++++++++++++++++++++++++"<<std::endl;

  // std::cout<<"///////////////all//////////////"<<std::endl;
  // std::cout<<"//////"<<(double)(end2-start1)/CLOCKS_PER_SEC<<"//////"<<std::endl;
  // std::cout<<"/////////////////////////////////"<<std::endl;
  return z.as(x.eltype());
}

ArrayRef TruncPrA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        size_t bits) const {
  PPU_TRACE_OP(this, x, bits);
  auto* comm = ctx->caller()->getState<Communicator>();

  // TODO: add trunction method to options.
  if (comm->getWorldSize() == 2u) {
    // SecurlML, local trunction.
    // Ref: Theorem 1. https://eprint.iacr.org/2017/396.pdf
    return ring_arshift(x, bits).as(x.eltype());
  } else {
    // ABY3, truncation pair method.
    // Ref: Section 5.1.2 https://eprint.iacr.org/2018/403.pdf
    auto* beaver = ctx->caller()->getState<Semi2kState>()->beaver();

    const auto field = x.eltype().as<Ring2k>()->field();
    const auto& [r, rb] = beaver->Trunc(field, x.numel(), bits);

    // open x - r
    auto x_r = comm->allReduce(ReduceOp::ADD, ring_sub(x, r), kName);
    auto res = rb;
    if (comm->getRank() == 0) {
      ring_add_(res, ring_arshift(x_r, bits));
    }

    // res = [x-r] + [r], x which [*] is truncation operation.
    return res.as(x.eltype());
  }
}

////////////////////////////////////////////////////////////////////
// logistic regression family
////////////////////////////////////////////////////////////////////
ArrayRef LogReg::proc(KernelEvalContext* ctx, const ArrayRef& x,
                      const ArrayRef& w, const ArrayRef& y,
                      int64_t M, int64_t N) const {
  std::cout<<"******************logreg start***************************"<<std::endl;
  PPU_TRACE_OP(this, x, y, w);

  std::cout<<"*****************x.shape****************************"<<std::endl;
  std::cout<<x.elsize()<<std::endl;

  std::cout<<"*****************M****************************"<<std::endl;
  std::cout<<M<<std::endl;


  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();

  std::cout<<"************************BEAVER*********************"<<std::endl;

  //This beaver refers the trusted dealer, not using beaver triples.
  auto* beaver = ctx->caller()->getState<Semi2kState>()->beaver();

  std::cout<<"************************BEAVER is ready*********************"<<std::endl;


  //generate dealer correlated randomness.
  auto [r1, r2, r3, c1, c2, c3, c4, c5] = beaver->lr(field, M, N);

  std::cout<<"************************CR*********************"<<std::endl;

  std::cout<<"x.shape"<<x.elsize()<<std::endl;

  std::cout<<"r1.shape"<<r1.elsize()<<std::endl;

  std::cout<<"w.shape"<<w.elsize()<<std::endl;

  std::cout<<"r2.shape"<<r2.elsize()<<std::endl;

  std::cout<<"y.shape"<<y.elsize()<<std::endl;

  std::cout<<"r3.shape"<<r3.elsize()<<std::endl;

  //Open x-r1, w-r2, y-r3

  auto res =
      vectorize({ring_sub(x, r1), ring_sub(w, r2), ring_sub(y, r3)}, [&](const ArrayRef& s){
        return comm->allReduce(ReduceOp::ADD, s, kName);
      });

  // std::cout<<"vectorize"<<std::endl;

  
  auto x_r1 = std::move(res[0]);
  auto w_r2 = std::move(res[1]);
  auto y_r3 = std::move(res[2]);

  std::cout<<"************************REVEAL*********************"<<std::endl;

  //Transpose : x_r1^T
  size_t i = 0, index;
  ArrayRef x_r1T(makeType<RingTy>(field), N * M);
  for ( i = 0 ; i < r1.elsize() ; i++){
    index = (i % N) * M + (i / N);
    x_r1T.at<int32_t>(index) = x_r1.at<int32_t>(i);
    std::cout<<x_r1.at<int32_t>(i)<<std::endl;
  }

  std::cout<<"************************TRANSPOSE*********************"<<std::endl;

  //Transpose : r1^T
  ArrayRef r1T(makeType<RingTy>(field), N * M);
  for ( i = 0 ; i < r1.elsize() ; i++){
    index = (i % N) * M + (i / N);
    r1T.at<int32_t>(index) = r1.at<int32_t>(i);
  }

  //activating criteria
  auto wx = ring_mmul(w_r2, x_r1T, 1, M, N);

  ArrayRef iden2(makeType<RingTy>(field), M) ;
  std::memset(iden2.data(), 2, iden2.buf()->size());
  
  
  //Delta
  auto w1 = ring_sub(ring_sub(ring_sub(ring_sub(
  ring_add(ring_add(ring_add(ring_add(ring_add(ring_add(ring_add(ring_add(ring_add(
  ring_mmul(iden2, x_r1, 1, N, M), ring_mmul(iden2, r1, 1, N, M)),
  ring_mmul(ring_mmul(w_r2, x_r1T, 1, M, N), x_r1, 1, N, M)),
  ring_mmul(ring_mmul(w_r2, r1, 1, M, N), x_r1, 1, N, M)),
  ring_mmul(ring_mmul(r2, x_r1T, 1, M, N), x_r1, 1, N, M)),
  ring_mmul(c1, x_r1, 1, N, M)),
  ring_mmul(ring_mmul(w_r2, x_r1T, 1, M, N), r1, 1, N, M)),
  ring_mmul(w_r2, c5, 1, N, N)),
  ring_mmul(c2, x_r1T, 1, M, N)),
  c3),
  ring_mmul(y_r3, x_r1, 1, N, M)),
  ring_mmul(y_r3, r1, 1, N, M)),
  ring_mmul(r3, x_r1, 1, N, M)),
  c4);

  std::cout<<"************************DELTA*********************"<<std::endl;

  auto w2 = ring_sub(ring_sub(ring_sub(ring_sub(
    ring_add(x_r1, r1),
    ring_mmul(y_r3, x_r1, 1, N, M)),
    ring_mmul(y_r3, r1, 1, N, M)),
    ring_mmul(r3, x_r1, 1, N, M)),
    ring_mmul(r3, r1, 1, N, M)
  );

  std::cout<<"*******************logreg finish*******************"<<std::endl;

  return w1;              
}


}  // namespace ppu::mpc::semi2k
