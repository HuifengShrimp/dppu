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


#include "ppu/mpc/beaver/beaver_tfp.h"

#include <random>

#include "ppu/core/array_ref_util.h"
#include "ppu/link/link.h"
#include "ppu/mpc/beaver/prg_tensor.h"
#include "ppu/mpc/util/ring_ops.h"
#include "ppu/utils/serialize.h"

namespace ppu::mpc {
namespace {

uint128_t GetHardwareRandom128() {
  std::random_device rd;
  // call random_device four times, make sure uint128 is random in 2^128 set.
  uint64_t lhs = static_cast<uint64_t>(rd()) << 32 | rd();
  uint64_t rhs = static_cast<uint64_t>(rd()) << 32 | rd();
  return MakeUint128(lhs, rhs);
}

}  // namespace

BeaverTfp::BeaverTfp(std::shared_ptr<link::Context> lctx)
    : lctx_(lctx), seed_(GetHardwareRandom128()), counter_(0) {
  auto buf = utils::SerializeUint128(seed_);
  std::vector<Buffer> all_bufs =
      link::Gather(lctx_, buf, 0, "BEAVER_TFP:SYNC_SEEDS");

  if (lctx_->Rank() == 0) {
    // Collects seeds from all parties.
    for (size_t rank = 0; rank < lctx_->WorldSize(); ++rank) {
      PrgSeed seed = utils::DeserializeUint128(all_bufs[rank]);
      tp_.setSeed(rank, lctx_->WorldSize(), seed);
    }
  }
}

Beaver::Triple BeaverTfp::Mul(FieldType field, size_t size) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, size, seed_, &counter_, &descs[0]);
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, size, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == 0) {
    c = tp_.adjustMul(descs);
  }

  return {a, b, c};
}

Beaver::Triple BeaverTfp::Dot(FieldType field, size_t M, size_t N, size_t K) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, M * K, seed_, &counter_, &descs[0]);
  auto b = prgCreateArray(field, K * N, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, M * N, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == 0) {
    c = tp_.adjustDot(descs, M, N, K);
  }

  std::cout<<"test "<<c.at<int32_t>(0)<<std::endl;

  return {a, b, c};
}

Beaver::Triple BeaverTfp::And(FieldType field, size_t size) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, size, seed_, &counter_, &descs[0]);
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, size, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == 0) {
    c = tp_.adjustAnd(descs);
  }

  return {a, b, c};
}

Beaver::Pair BeaverTfp::Trunc(FieldType field, size_t size, size_t bits) {
  std::vector<PrgArrayDesc> descs(2);

  auto a = prgCreateArray(field, size, seed_, &counter_, &descs[0]);
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);

  if (lctx_->Rank() == 0) {
    b = tp_.adjustTrunc(descs, bits);
  }

  return {a, b};
}

ArrayRef BeaverTfp::RandBit(FieldType field, size_t size) {
  PrgArrayDesc desc{};
  auto a = prgCreateArray(field, size, seed_, &counter_, &desc);

  if (lctx_->Rank() == 0) {
    a = tp_.adjustRandBit(desc);
  }

  return a;
}

Beaver::LR_set BeaverTfp::lr(FieldType field, size_t M, size_t N){

  std::cout<<"************************lr start*********************"<<std::endl;

  std::vector<PrgArrayDesc> descs(5);

  auto r1 = prgCreateArray(field, M * N, seed_, &counter_, &descs[0]);
  auto r2 = prgCreateArray(field, 1 * N, seed_, &counter_, &descs[1]);
  auto c1 = prgCreateArray(field, 1 * M, seed_, &counter_, &descs[2]);
  
  std::cout<<"************************lr start1*********************"<<std::endl;
  //r1^T
  size_t i = 0, j = 0, index;
  ArrayRef r1T(makeType<RingTy>(field), N * M);

  std::cout<<"************************lr start11*********************"<<std::endl;
  
  for ( i = 0 ; i < r1.elsize() ; i++){
    index = (i % N) * M + (i / N);
    r1T.at<int32_t>(index) = r1.at<int32_t>(i);
  }

  std::cout<<"************************lr start12*********************"<<std::endl;
  //c1=r2r1^T(1*N*N*M)
  c1 = ring_mmul(r2,r1T,1,M,N);

  std::cout<<"************************lr start2*********************"<<std::endl;

  //c2=r2(x')T r1
  ArrayRef c2(makeType<RingTy>(field), M*N*N) ;
  for( i = 0 ; i < M * N; i++){
    for( j = 0 ; j < N; j++){
      assignment(field, c2, i*N+j, ring_mul(ring_others(field, 1, r2.at<int32_t>(j)),
      ring_others(field, 1, r1T.at<int32_t>(i))).at<int32_t>(0));
      std::cout<<"************************lr start21*********************"<<std::endl;
    }
  }

  //c3 = r2·r1T·r1
  auto c3 = prgCreateArray(field, 1 * N, seed_, &counter_, &descs[3]);
  c3 = ring_mmul(c1,r1,1,N,M);

  auto r3 = prgCreateArray(field, M * 1, seed_, &counter_, &descs[4]);

  //r3^T
  ArrayRef r3T(makeType<RingTy>(field),1*M);
  for ( i = 0 ; i < r3.elsize() ; i++){
    index = (i % 1) * M + (i/1);
    r3T.at<int32_t>(index) = r3.at<int32_t>(i);
  }

   //c4 = r3^T·r1
  auto c4 = ring_mmul(r3T,r1,1,N,M);

  //c5 = r1^Tr1
  auto c5 = ring_mmul(r1T,r1,N,N,M);

  std::cout<<"************************lr end*********************"<<std::endl;

  return {r1,r2,r3,c1,c2,c3,c4,c5};
}

}  // namespace ppu::mpc
