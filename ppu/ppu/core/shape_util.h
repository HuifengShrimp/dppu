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

#include <numeric>
#include <vector>

#include "ppu/core/type_util.h"
#include "ppu/utils/exception.h"

namespace ppu {

inline int64_t numel(absl::Span<const int64_t> shape) {
  return std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1),
                         std::multiplies<>());
}

// Citation:
// https://github.com/xtensor-stack/xtensor-blas/blob/master/include/xtensor-blas/xlinalg.hpp
inline std::vector<int64_t> DeduceDotShape(absl::Span<const int64_t> lhs,
                                           absl::Span<const int64_t> rhs) {
  // One side is scalar.
  if (lhs.empty() || rhs.empty()) {
    return lhs.empty() ? std::vector<int64_t>(rhs.begin(), rhs.end())
                       : std::vector<int64_t>(lhs.begin(), lhs.end());
  }

  // Vector dot product.
  if (lhs.size() == 1 && rhs.size() == 1) {
    PPU_ENFORCE_EQ(lhs[0], rhs[0],
                   "DeduceDotShape: shape mismatch: lhs={} ,rhs={}", lhs, rhs);
    return {1};
  }

  if (lhs.size() == 2 && rhs.size() == 1) {
    // Matrix-times-vector product.
    PPU_ENFORCE_EQ(lhs[1], rhs[0],
                   "DeduceDotShape: shape mismatch: lhs={} ,rhs={}", lhs, rhs);

    return {lhs[0]};
  } else if (lhs.size() == 1 && rhs.size() == 2) {
    // Matrix-times-vector product.
    PPU_ENFORCE_EQ(lhs[0], rhs[0],
                   "DeduceDotShape: shape mismatch: lhs={} ,rhs={}", lhs, rhs);

    return {rhs[1]};
  } else if (lhs.size() == 2 && rhs.size() == 2) {
    // Matrix-product.
    PPU_ENFORCE_EQ(lhs[1], rhs[0],
                   "DeduceDotShape: shape mismatch: lhs={} ,rhs={}", lhs, rhs);
    return {lhs[0], rhs[1]};
  } else {
    // If lhs is an N-D array and rhs is an M-D array (where M>=2), it is a sum
    // product over the last axis of lhs and the second-to-last axis of rhs:
    //    dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
    auto lhs_back = lhs.back();
    size_t rhs_match_dim = 0;

    // rhs may be vector.
    if (rhs.size() > 1) {
      rhs_match_dim = rhs.size() - 2;
    }

    PPU_ENFORCE_EQ(lhs_back, rhs[rhs_match_dim],
                   "DeduceDotShape: shape mismatch: lhs={} ,rhs={}", lhs, rhs);

    int lhs_dim = static_cast<int>(lhs.size());
    int rhs_dim = static_cast<int>(rhs.size());

    int nd = lhs_dim + rhs_dim - 2;

    size_t j = 0;
    std::vector<int64_t> result(nd);

    for (int i = 0; i < lhs_dim - 1; ++i) {
      result[j++] = lhs[i];
    }

    for (int i = 0; i < rhs_dim - 2; ++i) {
      result[j++] = rhs[i];
    }

    if (rhs_dim > 1) {
      result[j++] = rhs.back();
    }

    return result;
  }
}

// This function assumes row major
inline size_t flattenIndex(absl::Span<const int64_t> multi_index,
                           absl::Span<const int64_t> shape,
                           absl::Span<const int64_t> strides = {}) {
  size_t linear_idx = 0;
  if (strides.empty()) {
    size_t scale = 1;
    for (int64_t idx = multi_index.size() - 1; idx >= 0; --idx) {
      linear_idx += multi_index[idx] * scale;
      scale *= shape[idx];
    }
  } else {
    for (int64_t idx = multi_index.size() - 1; idx >= 0; --idx) {
      linear_idx += multi_index[idx] * strides[idx];
    }
  }
  return linear_idx;
}

template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
inline bool bumpIndices(absl::Span<const T> shape, absl::Span<T> indices) {
  PPU_ENFORCE(shape.size() == indices.size());
  for (int64_t dimno = indices.size() - 1; dimno >= 0; --dimno) {
    T limit = shape[dimno];
    if (indices[dimno] + 1 < limit) {
      indices[dimno]++;
      // Whenever an index of a dimension is increased, it means that all
      // following dimensions have maxed out, so they must go to 0.
      std::fill(indices.begin() + dimno + 1, indices.end(), 0);
      return true;
    }
  }
  return false;
}

template <typename T>
inline std::vector<int64_t> makeShape(T begin, T end) {
  std::vector<int64_t> ret;
  for (auto iter = begin; iter != end; ++iter) {
    ret.emplace_back(static_cast<int64_t>(*iter));
  }
  return ret;
}

}  // namespace ppu
