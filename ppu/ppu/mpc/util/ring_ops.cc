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


#include "ppu/mpc/util/ring_ops.h"

#include <cstring>

#include "absl/types/span.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xvectorize.hpp"

#include "ppu/core/array_ref_util.h"
#include "ppu/crypto/pseudo_random_generator.h"
#include "ppu/mpc/util/linalg.h"

// TODO: ArrayRef is simple enough, consider using other SIMD libraries.
namespace ppu::mpc {
namespace {
constexpr char kName[] = "RingOps";
}  // namespace

#define PPU_ENFORCE_RING(x) \
  PPU_ENFORCE(x.eltype().isa<Ring2k>(), "expect ring type, got={}", x.eltype());

#define PPU_ENFORCE_EQ_RING(x, y)                                           \
  PPU_ENFORCE(                                                              \
      x.eltype().as<Ring2k>()->field() == y.eltype().as<Ring2k>()->field(), \
      "not all ring or ring mismatch x={}, y={}", x.eltype(), y.eltype());

ArrayRef ring_rand(FieldType field, size_t size, uint128_t prg_seed,
                   uint64_t* prg_counter) {
  constexpr SymmetricCrypto::CryptoType kCryptoType =
      SymmetricCrypto::CryptoType::AES128_ECB;
  constexpr uint128_t kAesInitialVector = 0U;

  ArrayRef res(makeType<RingTy>(field), size);
  *prg_counter = FillPseudoRandom(
      kCryptoType, prg_seed, kAesInitialVector, *prg_counter,
      absl::MakeSpan(static_cast<char*>(res.data()), res.buf()->size()));

  return res;
}

ArrayRef ring_zeros(FieldType field, size_t size) {
  // TODO(jint) zero strides.
  ArrayRef res(makeType<RingTy>(field), size);
  std::memset(res.data(), 0, res.buf()->size());
  return res;
}

ArrayRef ring_ones(FieldType field, size_t size) {
  ArrayRef res = ring_zeros(field, size);
  DISPATCH_ALL_FIELDS(field, kName, [&]() {
    auto res_xt = xt_mutable_adapt<ring2k_t>(res);
    res_xt += 1;
    return;
  });
  return res;
}

ArrayRef ring_others(FieldType field, size_t size, int32_t value) {
  ArrayRef res = ring_zeros(field, size);
  DISPATCH_ALL_FIELDS(field, kName, [&]() {
    auto res_xt = xt_mutable_adapt<ring2k_t>(res);
    res_xt += value;
    return;
  });
  return res;
}

ArrayRef assignment(FieldType field, ArrayRef &res, int32_t pos, int32_t value) {
  int32_t cnt = 0;
  DISPATCH_ALL_FIELDS(field, kName, [&]() {
    auto res_xt = xt_mutable_adapt<ring2k_t>(res);
    if( cnt == pos){
          res_xt = value;
    }
    cnt++;
    return;
  });
  return res;
}

ArrayRef ring_randbit(FieldType field, size_t size) {
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    auto res = xt::random::randint<ring2k_t>({size}) & 0x1;
    return make_array(res, makeType<RingTy>(field));
  });
}

ArrayRef ring_not(const ArrayRef& x) {
  PPU_ENFORCE_RING(x);

  auto res = x.clone();
  ring_not_(res);
  return res;
}

void ring_not_(ArrayRef& x) {
  PPU_ENFORCE_RING(x);

  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    auto x_xt = xt_mutable_adapt<ring2k_t>(x);
    x_xt = ~x_xt;
    return;
  });
}

ArrayRef ring_neg(const ArrayRef& x) {
  PPU_ENFORCE_RING(x);

  auto res = x.clone();
  ring_neg_(res);
  return res;
}

void ring_neg_(ArrayRef& x) {
  PPU_ENFORCE_RING(x);

  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    auto x_xt = xt_mutable_adapt<ring2k_t>(x);
    x_xt = -x_xt;
    return;
  });
}

ArrayRef ring_add(const ArrayRef& x, const ArrayRef& y) {
  auto res = x.clone();
  ring_add_(res, y);
  return res;
}

void ring_add_(ArrayRef& x, const ArrayRef& y) {
  PPU_ENFORCE_EQ_RING(x, y);

  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    auto x_xt = xt_mutable_adapt<ring2k_t>(x);
    auto y_xt = xt_adapt<ring2k_t>(y);
    x_xt += y_xt;
    return;
  });
}

ArrayRef ring_sub(const ArrayRef& x, const ArrayRef& y) {
  auto res = x.clone();
  ring_sub_(res, y);
  return res;
}

void ring_sub_(ArrayRef& x, const ArrayRef& y) {
  PPU_ENFORCE_EQ_RING(x, y);

  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    auto x_xt = xt_mutable_adapt<ring2k_t>(x);
    auto y_xt = xt_adapt<ring2k_t>(y);
    x_xt -= y_xt;
    return;
  });
}

ArrayRef ring_mul(const ArrayRef& x, const ArrayRef& y) {
  auto res = x.clone();
  ring_mul_(res, y);
  return res;
}

void ring_mul_(ArrayRef& x, const ArrayRef& y) {
  PPU_ENFORCE_EQ_RING(x, y);

  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    auto x_xt = xt_mutable_adapt<ring2k_t>(x);
    auto y_xt = xt_adapt<ring2k_t>(y);
    x_xt *= y_xt;
    return;
  });
}

ArrayRef ring_mmul(const ArrayRef& lhs, const ArrayRef& rhs, int64_t M,
                   int64_t N, int64_t K) {
  PPU_ENFORCE_EQ_RING(lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    ArrayRef ret(lhs.eltype(), M * N);
    const auto& lhs_strides = lhs.stride();
    const auto lhs_stride_scale = lhs.elsize() / sizeof(ring2k_t);
    const auto& rhs_strides = rhs.stride();
    const auto rhs_stride_scale = rhs.elsize() / sizeof(ring2k_t);
    const auto& ret_strides = ret.stride();
    const auto ret_stride_scale = ret.elsize() / sizeof(ring2k_t);

    linalg::matmul(
        M, N, K, static_cast<const ring2k_t*>(lhs.data()),
        lhs_stride_scale * K * lhs_strides, lhs_stride_scale * lhs_strides,
        static_cast<const ring2k_t*>(rhs.data()),
        rhs_stride_scale * N * rhs_strides, rhs_stride_scale * rhs_strides,
        static_cast<ring2k_t*>(ret.data()), ret_stride_scale * N * ret_strides,
        ret_stride_scale * ret_strides);
    return ret;
  });
}

ArrayRef ring_and(const ArrayRef& x, const ArrayRef& y) {
  auto res = x.clone();
  ring_and_(res, y);
  return res;
}

void ring_and_(ArrayRef& x, const ArrayRef& y) {
  PPU_ENFORCE_EQ_RING(x, y);

  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    auto x_xt = xt_mutable_adapt<ring2k_t>(x);
    auto y_xt = xt_adapt<ring2k_t>(y);
    x_xt &= y_xt;
    return;
  });
}

ArrayRef ring_xor(const ArrayRef& x, const ArrayRef& y) {
  auto res = x.clone();
  ring_xor_(res, y);
  return res;
}

void ring_xor_(ArrayRef& x, const ArrayRef& y) {
  PPU_ENFORCE_EQ_RING(x, y);

  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    auto x_xt = xt_mutable_adapt<ring2k_t>(x);
    auto y_xt = xt_adapt<ring2k_t>(y);
    x_xt ^= y_xt;
    return;
  });
}

ArrayRef ring_arshift(const ArrayRef& x, size_t bits) {
  PPU_ENFORCE_RING(x);

  auto res = x.clone();
  ring_arshift_(res, bits);
  return res;
}

void ring_arshift_(ArrayRef& x, size_t bits) {
  PPU_ENFORCE_RING(x);

  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    // According to K&R 2nd edition the results are implementation-dependent for
    // right shifts of signed values, but "usually" its arithmetic right shift.
    using S = std::make_signed<ring2k_t>::type;
    auto x_xt = xt_mutable_adapt<S>(x);
    x_xt = xt::right_shift(x_xt, bits);
  });
}

ArrayRef ring_rshift(const ArrayRef& x, size_t bits) {
  PPU_ENFORCE_RING(x);

  auto res = x.clone();
  ring_rshift_(res, bits);
  return res;
}

void ring_rshift_(ArrayRef& x, size_t bits) {
  PPU_ENFORCE_RING(x);

  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using U = std::make_unsigned<ring2k_t>::type;
    auto x_xt = xt_mutable_adapt<U>(x);
    x_xt = xt::right_shift(x_xt, bits);
  });
}

ArrayRef ring_lshift(const ArrayRef& x, size_t bits) {
  PPU_ENFORCE_RING(x);

  auto res = x.clone();
  ring_lshift_(res, bits);
  return res;
}

void ring_lshift_(ArrayRef& x, size_t bits) {
  PPU_ENFORCE_RING(x);

  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    auto x_xt = xt_mutable_adapt<ring2k_t>(x);
    x_xt = xt::left_shift(x_xt, bits);
  });
}

ArrayRef ring_reverse_bits(const ArrayRef& x, size_t start, size_t end) {
  PPU_ENFORCE_RING(x);

  auto res = x.clone();
  ring_reverse_bits_(res, start, end);
  return res;
}
void ring_reverse_bits_(ArrayRef& x, size_t start, size_t end) {
  PPU_ENFORCE_RING(x);

  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kName, [&]() {
    auto reverse_bits_fn = [&](ring2k_t x) -> ring2k_t {
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

    auto x_xt = xt_mutable_adapt<ring2k_t>(x);
    x_xt = xt::vectorize(reverse_bits_fn)(x_xt);
  });
}

}  // namespace ppu::mpc
