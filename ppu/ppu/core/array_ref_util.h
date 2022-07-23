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


#pragma once

// TODO: we need a prelude for int128 before include xtensor
#include "ppu/utils/int128.h"
//

#include "absl/types/span.h"
#include "xtensor/xadapt.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xio.hpp"

#include "ppu/core/array_ref.h"
#include "ppu/core/type.h"

namespace ppu {
namespace details {

inline std::vector<int64_t> ByteToElementStrides(std::vector<int64_t> strides,
                                                 size_t elsize) {
  std::transform(strides.begin(), strides.end(), strides.begin(),
                 [&](size_t c) -> size_t { return c / elsize; });
  return strides;
}

inline std::vector<int64_t> ElementToByteStrides(std::vector<int64_t> strides,
                                                 size_t elsize) {
  std::transform(strides.begin(), strides.end(), strides.begin(),
                 [&](size_t c) -> size_t { return c * elsize; });
  return strides;
}

}  // namespace details

// This module tries to unify all xt::adapt usage, to avoid potential abuse.

// TODO(jint) add const/non-const overload?
template <typename T>
auto xt_adapt(const ArrayRef& aref) {
  PPU_ENFORCE(aref.elsize() == sizeof(T), "adapt eltype={} with size={}",
              aref.eltype(), sizeof(T));

  std::vector<int64_t> shape = {static_cast<int64_t>(aref.numel())};

  std::vector<int64_t> strides = {aref.stride()};

  return xt::adapt(static_cast<const T*>(aref.data()),
                   static_cast<size_t>(aref.numel()), xt::no_ownership(), shape,
                   strides);
}

template <typename T>
auto xt_mutable_adapt(ArrayRef& aref) {
  PPU_ENFORCE(aref.elsize() == sizeof(T), "adapt eltype={} with size={}",
              aref.eltype(), sizeof(T));

  std::vector<int64_t> shape = {aref.numel()};
  std::vector<int64_t> strides = {aref.stride()};

  return xt::adapt(static_cast<T*>(aref.data()),
                   static_cast<size_t>(aref.numel()), xt::no_ownership(), shape,
                   strides);
}

// Make a array from xtensor expression.
template <typename E,
          typename T = typename std::remove_const<typename E::value_type>::type>
ArrayRef make_array(const xt::xexpression<E>& e, const Type& eltype) {
  PPU_ENFORCE(sizeof(T) == sizeof(typename E::value_type));
  PPU_ENFORCE(sizeof(T) == eltype.size());

  auto&& ret = xt::eval(e.derived_cast());

  // TODO(jint) for matmul, we also pass as a 1D array, but the result is 2D
  // PPU_ENFORCE(ret.shape().size() == 1);

  std::vector<std::size_t> shape = {ret.size()};
  ArrayRef aref(eltype, ret.size());

  xt::adapt(static_cast<T*>(aref.data()), ret.size(), xt::no_ownership(),
            shape) = ret;
  return aref;
}

// NOTE(junfeng): layout_type of return is xt::layout_type::dynamic.
template <typename T>
auto xt_mutable_adapt(NdArrayRef& aref) {
  PPU_ENFORCE(aref.elsize() == sizeof(T), "adapt eltype={} with size={}",
              aref.eltype(), sizeof(T));

  return xt::adapt(static_cast<T*>(aref.data()), aref.numel(),
                   xt::no_ownership(), aref.shape(), aref.strides());
}

// NOTE(junfeng): layout_type of return is xt::layout_type::dynamic.
template <typename T>
auto xt_adapt(const NdArrayRef& aref) {
  PPU_ENFORCE(aref.elsize() == sizeof(T), "adapt eltype={} with size={}",
              aref.eltype(), sizeof(T));

  return xt::adapt(static_cast<const T*>(aref.data()), aref.numel(),
                   xt::no_ownership(), aref.shape(), aref.strides());
}

// A view of a plaintext buffer.
// Please do not direct use this class if possible.
struct PtBufferView {
  void const* const ptr;               // Pointer to the underlying storage
  PtType const pt_type;                // Plaintext data type.
  std::vector<int64_t> const shape;    // Shape of the tensor.
  std::vector<int64_t> const strides;  // Strides in byte.

  // We have to take a concrete buffer as a view.
  PtBufferView() = delete;

  explicit PtBufferView(void const* ptr, PtType pt_type,
                        std::vector<int64_t> shape,
                        std::vector<int64_t> strides)
      : ptr(ptr),
        pt_type(pt_type),
        shape(std::move(shape)),
        strides(std::move(strides)) {}

  // View c++ builtin scalar type as a buffer
  template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
  /* implicit */ PtBufferView(T const& s)
      : ptr(static_cast<void const*>(&s)),
        pt_type(PtTypeToEnum<T>::value),
        shape(),
        strides() {}

  // View a xt::xarray as a buffer.
  template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
  /* implicit */ PtBufferView(const xt::xarray<T>& xarr)
      : ptr(static_cast<void const*>(xarr.data())),
        pt_type(PtTypeToEnum<T>::value),
        shape(xarr.shape().begin(), xarr.shape().end()),
        strides({xarr.strides().begin(), xarr.strides().end()}) {}

  template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
  /* implicit */ PtBufferView(const std::vector<T>& xarr)
      : ptr(static_cast<void const*>(xarr.data())),
        pt_type(PtTypeToEnum<T>::value),
        shape({static_cast<int64_t>(xarr.size())}),
        strides({1}) {}
};

std::ostream& operator<<(std::ostream& out, PtBufferView v);

// Make a ndarray from a plaintext buffer.
NdArrayRef make_ndarray(PtBufferView bv);

// Make a NdArrayRef from an xt expression.
template <typename E,
          typename T = typename std::remove_const<typename E::value_type>::type,
          std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
NdArrayRef make_ndarray(const xt::xexpression<E>& e) {
  auto&& ee = xt::eval(e.derived_cast());

  const Type eltype = makePtType<T>();
  auto arr = NdArrayRef(eltype, {ee.shape().begin(), ee.shape().end()});
  xt_mutable_adapt<T>(arr) = ee;

  return arr;
}

// Encode an array to another array
//
// Given:
//   x: the source array
//   to_type: encoded storage type
//   fxp_bits: number of fractional bits for fixed point.
//
// Then:
//   let scalar = 1 << fxp_bits
//   in
//   y = cast<to_type>(x * scalar)  if type(x) is float
//    |= cast<to_type>(x)           if type(x) is integer
//   dtype = FXP if type(x) is float
//        |= INT if type(x) is integer
NdArrayRef encodeToRing(const NdArrayRef& x, const Type& to_type,
                        size_t fxp_bits, DataType* dtype = nullptr);

// Decode an array to another array
//
// Given:
//   x: the source array
//   to_type: the decoded storage type
//   fxp_bits: number of fractional bits for fixed point.
//   dtype: the source array data type.
//
// Then:
//   let scalar = 1 << fxp_bits
//   in
//   y = cast<to_type>(x) / scalar  if dtype is FXP
//    |= cast<to_type>(x)           if dtype is INT
NdArrayRef decodeFromRing(const NdArrayRef& x, const Type& to_type,
                          size_t fxp_bits, DataType dtype);

//
NdArrayRef zeros(const Type& type, const std::vector<int64_t>& shape);

//
NdArrayRef randint(const Type& type, const std::vector<int64_t>& shape);

//
NdArrayRef sum(absl::Span<NdArrayRef const> arrs);

//
NdArrayRef add(const NdArrayRef& a, const NdArrayRef& b);

//
NdArrayRef sub(const NdArrayRef& a, const NdArrayRef& b);

}  // namespace ppu
