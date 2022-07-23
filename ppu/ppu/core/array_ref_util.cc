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


#include "ppu/core/array_ref_util.h"

#include <random>

#include "ppu/crypto/pseudo_random_generator.h"

namespace ppu {
namespace {

// TODO(jint) optimize me, the template expanded binary is too large.
template <typename E>
NdArrayRef encodeXexprToNdArray(E&& frm_xt, const Type& to_type,
                                size_t fxp_bits) {
  NdArrayRef arr(to_type, {frm_xt.shape().begin(), frm_xt.shape().end()});
  PPU_ENFORCE(to_type.size() * 8 > fxp_bits,
              "encode fxp_bits={} greater than type size={}", fxp_bits,
              to_type.size() * 8);

  using FrmT = typename E::value_type;
  DISPATCH_ALL_FIELDS(to_type.as<Ring2k>()->field(), "_", [&]() {
    if constexpr (std::is_floating_point_v<FrmT>) {
      size_t k = sizeof(ring2k_t) * 8;
      ring2k_t scalar = ring2k_t(1) << fxp_bits;

      // Reference: https://eprint.iacr.org/2019/599.pdf
      // The safe ring range is [-2^(k-2), 2^(k-2))
      auto ring_lower_bound = -(ring2k_t)std::pow(2, k - 2);
      auto ring_upper_bound = (ring2k_t)std::pow(2, k - 2) - 1;

      // Use xtensor function to remove nan/inf
      auto finite = xt::nan_to_num(frm_xt);

      // Start clamp to remove -inf/inf, float cannot represent consecutive
      // integers above 2^24, so we use double here to workaround this issue.
      auto u_clamp_to = static_cast<double>(ring_upper_bound / scalar);
      auto u_clamp = xt::minimum(xt::cast<double>(finite), u_clamp_to);

      auto l_clamp_to = static_cast<double>(ring_lower_bound / scalar);
      auto l_clamp = xt::maximum(u_clamp, l_clamp_to);

      xt_mutable_adapt<ring2k_t>(arr) = xt::cast<ring2k_t>(l_clamp * scalar);

    } else {
      xt_mutable_adapt<ring2k_t>(arr) = xt::cast<ring2k_t>(frm_xt);
    }
  });
  return arr;
}

template <typename E>
NdArrayRef decodeXexprToNdArray(E&& frm_xt, const Type& to_type,
                                size_t fxp_bits, DataType dtype) {
  NdArrayRef arr(to_type, {frm_xt.shape().begin(), frm_xt.shape().end()});
#define CASE(NAME, TYPE, _)                                                  \
  case NAME: {                                                               \
    if (dtype == DT_FXP) {                                                   \
      const TYPE scalar = TYPE(1ll << fxp_bits);                             \
      xt_mutable_adapt<TYPE>(arr) =                                          \
          xt::cast<TYPE>(std::forward<E>(frm_xt)) / scalar;                  \
    } else {                                                                 \
      xt_mutable_adapt<TYPE>(arr) = xt::cast<TYPE>(std::forward<E>(frm_xt)); \
    }                                                                        \
    break;                                                                   \
  }

  switch (to_type.as<PtTy>()->pt_type()) {
    FOREACH_PT_TYPES(CASE)
    default:
      PPU_THROW("decoding to type={} not supported", to_type);
  }

#undef CASE

  return arr;
}

template <typename T>
NdArrayRef make_ndarray_impl(PtBufferView bv) {
  // make a compact ndarray
  auto arr = NdArrayRef(makePtType(bv.pt_type), bv.shape);

  // assign to it.
  xt_mutable_adapt<T>(arr) =
      xt::adapt(static_cast<T const*>(bv.ptr), ppu::numel(bv.shape),
                xt::no_ownership(), bv.shape, bv.strides);

  return arr;
}

}  // namespace

std::ostream& operator<<(std::ostream& out, PtBufferView v) {
  out << fmt::format("PtBufferView<{},{}x{},{}>", v.ptr,
                     fmt::join(v.shape, "x"), v.pt_type,
                     fmt::join(v.strides, "x"));
  return out;
}

NdArrayRef make_ndarray(PtBufferView bv) {
#define CASE(NAME, CTYPE, _)             \
  case NAME: {                           \
    return make_ndarray_impl<CTYPE>(bv); \
  }

  switch (bv.pt_type) {
    FOREACH_PT_TYPES(CASE)
    default:
      PPU_THROW("should not be here, pt_type={}", bv.pt_type);
  }

#undef CASE
}

NdArrayRef encodeToRing(const NdArrayRef& arr, const Type& to_type,
                        size_t fxp_bits, DataType* dtype) {
  const Type& frm_type = arr.eltype();

  PPU_ENFORCE(frm_type.isa<PtTy>(), "source must be pt_type, got={}", frm_type);
  PPU_ENFORCE(to_type.isa<RingTy>(), "target be ring_type, got={}", to_type);

  if (dtype) {
    if (frm_type == F32 || frm_type == F64) {
      *dtype = DT_FXP;
    } else {
      *dtype = DT_INT;
    }
  }

#define CASE(NAME, CTYPE, _)                                              \
  if (frm_type.as<PtTy>()->pt_type() == NAME) {                           \
    return encodeXexprToNdArray(xt_adapt<CTYPE>(arr), to_type, fxp_bits); \
  }

  FOREACH_PT_TYPES(CASE)

#undef CASE

  PPU_THROW("shold not be here");
}

NdArrayRef decodeFromRing(const NdArrayRef& arr, const Type& to_type,
                          size_t fxp_bits, DataType dtype) {
  const Type& frm_type = arr.eltype();

  PPU_ENFORCE(frm_type.isa<RingTy>(), "source must be ring_type, got={}",
              frm_type);
  PPU_ENFORCE(to_type.isa<PtTy>(), "target be pt_type, got={}", to_type);

  return DISPATCH_ALL_FIELDS(frm_type.as<Ring2k>()->field(), "_", [&]() {
    return decodeXexprToNdArray(xt_adapt<ring2k_t>(arr), to_type, fxp_bits,
                                dtype);
  });
}

NdArrayRef zeros(const Type& type, const std::vector<int64_t>& shape) {
  PPU_ENFORCE(isIntTy(type), "only support int, got={}", type);

  NdArrayRef arr(type, shape);
  memset(arr.data(), 0, arr.buf()->size());

  return arr;
}

NdArrayRef randint(const Type& type, const std::vector<int64_t>& shape) {
  PPU_ENFORCE(isIntTy(type), "expect int, got={}", type);

  std::random_device rd;
  PseudoRandomGenerator<int64_t> prg(rd());

  NdArrayRef arr(type, shape);
  prg.Fill(absl::MakeSpan(static_cast<char*>(arr.data()), arr.buf()->size()));

  return arr;
}

NdArrayRef sum(absl::Span<NdArrayRef const> arrs) {
  PPU_ENFORCE(!arrs.empty(), "expected non empty, got size={}", arrs.size());
  NdArrayRef res = zeros(arrs[0].eltype(), arrs[0].shape());
  for (const auto& arr : arrs) {
    res = add(res, arr);
  }

  return res;
}

NdArrayRef add(const NdArrayRef& a, const NdArrayRef& b) {
  PPU_ENFORCE(a.eltype() == b.eltype(), "type mismatch, a={}, b={}", a.eltype(),
              b.eltype());
  PPU_ENFORCE(a.eltype().isa<PtTy>(), "expect plaintext type, got={}",
              a.eltype());

  const PtType pt_type = a.eltype().as<PtTy>()->pt_type();
  NdArrayRef c(a.eltype(), a.shape());
  DISPATCH_ALL_PT_TYPES(pt_type, "pt.add", [&]() {
    xt_mutable_adapt<_PtTypeT>(c) =
        xt_adapt<_PtTypeT>(a) + xt_adapt<_PtTypeT>(b);
  });

  return c;
}

NdArrayRef sub(const NdArrayRef& a, const NdArrayRef& b) {
  PPU_ENFORCE(a.eltype() == b.eltype(), "type mismatch, a={}, b={}", a.eltype(),
              b.eltype());
  PPU_ENFORCE(a.eltype().isa<PtTy>(), "expect plaintext type, got={}",
              a.eltype());

  const PtType pt_type = a.eltype().as<PtTy>()->pt_type();
  NdArrayRef c(a.eltype(), a.shape());
  DISPATCH_ALL_PT_TYPES(pt_type, "pt.sub", [&]() {
    xt_mutable_adapt<_PtTypeT>(c) =
        xt_adapt<_PtTypeT>(a) - xt_adapt<_PtTypeT>(b);
  });

  return c;
}

}  // namespace ppu
