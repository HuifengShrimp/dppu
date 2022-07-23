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


#include "ppu/hal/sort.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "xtensor/xsort.hpp"

#include "ppu/hal/permute_util.h"

namespace ppu::hal {
namespace {

// NOTE(junfeng): modifications to xtensor code are
// 1. add support for stable_sort
// 2. add another comparator(greater)
// 3. fix layout_type to row_major to remove unreached code.

// https://github.com/xtensor-stack/xtensor/blob/cc23f11446c0ec67fbba4582228c182d5ee1e40a/include/xtensor/xsort.hpp#L306-L326
template <class E,
          class R = typename xt::detail::linear_argsort_result_type<E>::type>
inline auto flatten_argsort_impl(const xt::xexpression<E>& e, bool is_stable,
                                 bool is_less) {
  const auto& de = e.derived_cast();

  auto cit = de.template begin<xt::layout_type::row_major>();
  using const_iterator = decltype(cit);
  auto ad = xt::xiterator_adaptor<const_iterator, const_iterator>(cit, cit,
                                                                  de.size());

  using result_type = R;
  result_type result;
  result.resize({de.size()});
  auto comp = [&ad, &is_less](std::size_t x, std::size_t y) {
    if (is_less) {
      return ad[x] < ad[y];
    } else {
      return ad[x] > ad[y];
    }
  };

  std::iota(result.begin(), result.end(), 0);

  if (is_stable) {
    std::stable_sort(result.begin(), result.end(), comp);
  } else {
    std::sort(result.begin(), result.end(), comp);
  }

  return result;
}

// https://github.com/xtensor-stack/xtensor/blob/cc23f11446c0ec67fbba4582228c182d5ee1e40a/include/xtensor/xsort.hpp#L272-L304
template <class Ed, class Ei>
inline void argsort_over_leading_axis(const Ed& data, Ei& inds, bool is_stable,
                                      bool is_less) {
  std::size_t n_iters =
      std::accumulate(data.shape().begin(), data.shape().end() - 1,
                      std::size_t(1), std::multiplies<>());
  std::ptrdiff_t data_secondary_stride = data.shape(data.dimension() - 1);
  std::ptrdiff_t inds_secondary_stride = inds.shape(inds.dimension() - 1);

  auto ptr = data.data();
  auto indices_ptr = inds.data();

  for (std::size_t i = 0; i < n_iters; ++i, ptr += data_secondary_stride,
                   indices_ptr += inds_secondary_stride) {
    auto comp = [&ptr, &is_less](std::size_t x, std::size_t y) {
      if (is_less) {
        return *(ptr + x) < *(ptr + y);
      } else {
        return *(ptr + x) > *(ptr + y);
      }
    };

    std::iota(indices_ptr, indices_ptr + inds_secondary_stride, 0);

    if (is_stable) {
      std::stable_sort(indices_ptr, indices_ptr + inds_secondary_stride, comp);
    } else {
      std::sort(indices_ptr, indices_ptr + inds_secondary_stride, comp);
    }
  }
}

// https://github.com/xtensor-stack/xtensor/blob/cc23f11446c0ec67fbba4582228c182d5ee1e40a/include/xtensor/xsort.hpp#L334-L377
template <class E>
inline auto argsort(const xt::xexpression<E>& e, size_t axis, bool is_stable,
                    bool is_less) {
  using eval_type = typename xt::detail::sort_eval_type<E>::type;
  using result_type = typename xt::detail::argsort_result_type<eval_type>::type;

  const auto& de = e.derived_cast();

  if (de.dimension() == 1) {
    return flatten_argsort_impl<E, result_type>(e, is_stable, is_less);
  }

  if (axis < de.dimension() - 1) {
    xt::dynamic_shape<std::size_t> permutation;
    xt::dynamic_shape<std::size_t> reverse_permutation;
    std::tie(permutation, reverse_permutation) = xt::detail::get_permutations(
        de.dimension(), axis, xt::layout_type::row_major);

    eval_type ev = xt::transpose(de, permutation);

    result_type res = result_type::from_shape(ev.shape());
    argsort_over_leading_axis(ev, res, is_stable, is_less);
    res = xt::transpose(res, reverse_permutation);
    return res;
  }

  result_type res = result_type::from_shape(de.shape());
  argsort_over_leading_axis(de, res, is_stable, is_less);
  return res;
}

}  // namespace

std::vector<Value> sort(HalContext* ctx, const std::vector<Value>& operands,
                        size_t dimension, bool is_stable, bool is_less) {
  PPU_ENFORCE(!operands.empty());

  if (!operands.front().is_public()) {
    PPU_THROW("not implemented.");
  }

  const auto arg = DISPATCH_ALL_FIELDS(
      ctx->GetField(), "argsort", [&]() -> xt::xarray<size_t> {
        const auto fst_operand_xt =
            xt::eval(xt_adapt<ring2k_t>(operands.front()));

        return argsort(fst_operand_xt, dimension, is_stable, is_less);
      });

  std::vector<Value> res;
  res.reserve(operands.size());

  for (const auto& operand : operands) {
    res.emplace_back(permute(ctx, operand, dimension, arg));
  }

  return res;
}

}  // namespace ppu::hal
