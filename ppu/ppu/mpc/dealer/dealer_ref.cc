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


#include "ppu/mpc/dealer/dealer_ref.h"

#include "ppu/mpc/util/ring_ops.h"

namespace ppu::mpc {

Dealer::Triple DealerRef::Mul(FieldType field, size_t size) {
  return {
      ring_zeros(field, size),
      ring_zeros(field, size),
      ring_zeros(field, size),
  };
}

Dealer::Triple DealerRef::And(FieldType field, size_t size) {
  return {
      ring_zeros(field, size),
      ring_zeros(field, size),
      ring_zeros(field, size),
  };
}

Dealer::Triple DealerRef::Dot(FieldType field, size_t M, size_t N, size_t K) {
  return {
      ring_zeros(field, M * K),
      ring_zeros(field, K * N),
      ring_zeros(field, M * N),
  };
}

Dealer::LR_set lr(FieldType field, size_t M, size_t N) {
    return{
        ring_zeros(field, M * N), //r1:M*N
        ring_zeros(field, 1 * N), //r2:1*N
        ring_zeros(field, M * 1), //r3:M*1
        ring_zeros(field, 1 * M), //c1=r2r1^T : 1*M
        ring_zeros(field, 1 * N), //c2=r2(x')^Tr1 : 1*N
        ring_zeros(field, 1 * N), //c3=r2r1^Tr1 : 1*N
        ring_zeros(field, 1 * N), //c4=r3^Tr1 : 1*N
        ring_zeros(field, N * N), //c5=r1^Tr1 : N*N
    };
}

Dealer::Pair DealerRef::Trunc(FieldType field, size_t size, size_t bits) {
  return {
      ring_zeros(field, size),
      ring_zeros(field, size),
  };
}

ArrayRef DealerRef::RandBit(FieldType field, size_t size) {
  return ring_zeros(field, size);
}

}  // namespace ppu::mpc
