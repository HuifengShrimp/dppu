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


#include "ppu/mpc/fss/arithmetic.h"

#include "ppu/core/array_ref_util.h"
#include "ppu/core/trace.h"
#include "ppu/core/vectorize.h"
#include "ppu/mpc/fss/object.h"
#include "ppu/mpc/fss/utils.h"
#include "ppu/mpc/interfaces.h"
#include "ppu/mpc/prg_state.h"
#include "ppu/mpc/semi2k/type.h"
#include "ppu/mpc/util/communicator.h"
#include "ppu/mpc/util/ring_ops.h"

namespace ppu::mpc::fss {

ArrayRef LogReg::proc(KernelEvalContext* ctx, const ArrayRef& x,
                      const ArrayRef& y, const ArrayRef& w,
                      int64_t M, int64_t N) const {
  std::cout<<"******************logreg start***************************"<<std::endl;
  PPU_TRACE_OP(this, x, y, w);

  //directly input x,y,w

  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* dealer = ctx->caller()->getState<FSSState>()->dealer();

  //generate dealer correlated randomness.
  auto [r1, r2, r3, c1, c2, c3, c4, c5] = dealer->lr(field, M, N);

  //Open x-r1, w-r2, y-r3
  auto res =
      vectorize({ring_sub(x, r1), ring_sub(w, r2), ring_sub(y, r3)}, [&](const ArrayRef& s){
        return comm->allReduce(ReduceOp::ADD, s, kName);
      });
  
  auto x_r1 = std::move(res[0]);
  auto w_r2 = std::move(res[1]);
  auto y_r3 = std::move(res[2]);

  //Transpose : x_r1^T
  size_t i = 0, index;
  ArrayRef x_r1T(makeType<RingTy>(field), N * M);
  for ( i = 0 ; i < r1.elsize() ; i++){
    index = (i % N) * M + (i / N);
    x_r1T.at<int32_t>(index) = x_r1.at<int32_t>(i);
    std::cout<<x_r1.at<int32_t>(i)<<std::endl;
  }

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



}  // namespace ppu::mpc::fss
