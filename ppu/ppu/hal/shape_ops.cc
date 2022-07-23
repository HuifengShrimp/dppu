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


#include "ppu/hal/shape_ops.h"

#include "xtensor/xeval.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xstrides.hpp"

#include "ppu/core/array_ref_util.h"
#include "ppu/core/vectorize.h"
#include "ppu/utils/exception.h"

namespace ppu::hal {

namespace {

std::vector<int64_t> DeducePadShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<size_t>& edge_padding_low,
    const std::vector<size_t>& edge_padding_high,
    const std::vector<size_t>& interior_padding) {
  std::vector<int64_t> dims;
  PPU_ENFORCE(edge_padding_low.size() == input_shape.size());
  PPU_ENFORCE(edge_padding_high.size() == input_shape.size());
  PPU_ENFORCE(interior_padding.size() == input_shape.size());
  for (size_t i = 0; i < input_shape.size(); i++) {
    dims.emplace_back(edge_padding_low[i] + edge_padding_high[i] +
                      interior_padding[i] * (input_shape.at(i) - 1) +
                      input_shape.at(i));
  }

  return dims;
}

// Adapted from:
// https://github.com/xtensor-stack/xtensor/blob/78aaac39143caa78da7c5c0734ccef957535f0c0/include/xtensor/xoperation.hpp#L877-L900
template <xt::layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class T>
inline auto AllIndices(const T& arr) {
  const auto& shape = arr.shape();
  using index_type = xt::xindex_type_t<typename T::shape_type>;
  using size_type = typename T::size_type;

  auto idx = xtl::make_sequence<index_type>(arr.dimension(), 0);
  std::vector<index_type> indices;

  size_type total_size = xt::compute_size(shape);
  for (size_type i = 0; i < total_size;
       i++, xt::detail::next_idx<L>(shape, idx)) {
    indices.push_back(idx);
  }
  return indices;
}

}  // namespace

Value transpose(HalContext* ctx, const Value& in,
                const std::vector<size_t>& permutation) {
  PPU_TRACE_OP(ctx, in);

  // TODO(jint) dont touch membuf, manipulate strides for transpose.
  return DISPATCH_ALL_ELSIZE(in.elsize(), [&]() -> Value {
    const auto& out =
        permutation.empty()
            ? xt::eval(xt::transpose(xt_adapt<element_t>(in)))
            : xt::eval(xt::transpose(xt_adapt<element_t>(in), permutation));

    // TODO(jint) double-check xt strides convention.
    auto buf = makeBuffer(out.data(), out.size() * _kSize);
    return Value(std::move(buf), in.eltype(), out.shape(), out.strides(), 0);
  });
}

Value concatenate(HalContext* ctx, absl::Span<const Value> values,
                  const size_t& axis) {
  PPU_TRACE_OP(ctx, axis);

  // Enforce all types are the same
  PPU_ENFORCE(std::all_of(
      values.begin() + 1, values.end(),
      [&](const Value& v) { return v.eltype() == values.begin()->eltype(); }));

  // Enforce axis
  PPU_ENFORCE(std::all_of(values.begin(), values.end(), [&](const Value& v) {
    return static_cast<size_t>(axis) < v.shape().size() ||
           (v.shape().empty() && axis == 0);
  }));

  // Sanity shape
  for (size_t d = 0; d < values.front().shape().size(); ++d) {
    if (d == axis) {
      continue;
    }
    PPU_ENFORCE(
        std::all_of(values.begin() + 1, values.end(), [&](const Value& v) {
          return v.shape()[d] == values.front().shape()[d];
        }));
  }

  std::vector<int64_t> result_shape = values.front().shape();
  for (auto iter = values.begin() + 1; iter != values.end(); ++iter) {
    result_shape[axis] += iter->shape()[axis];
  }

  // Preallocate output buffer
  Value result(values.front().eltype(), result_shape);

  int64_t b_dimension_offset = 0;
  for (const auto& v : values) {
    std::vector<int64_t> from_indicies(result_shape.size(), 0);
    std::vector<int64_t> to_indicies(result_shape.size(), 0);
    do {
      to_indicies = from_indicies;
      to_indicies[axis] += b_dimension_offset;
      result.CopyElementFrom(v, from_indicies, to_indicies);
    } while (bumpIndices<int64_t>(v.shape(), absl::MakeSpan(from_indicies)));
    b_dimension_offset += v.shape()[axis];
  }

  return result;
}

Value slice(HalContext* ctx, const Value& in,
            const std::vector<size_t>& start_indices,
            const std::vector<size_t>& end_indices,
            const std::vector<size_t>& strides) {
  PPU_TRACE_OP(ctx, in, start_indices, end_indices, strides);

  PPU_ENFORCE(in.shape().size() == start_indices.size());
  PPU_ENFORCE(in.shape().size() == end_indices.size());
  PPU_ENFORCE(strides.empty() || (in.shape().size() == strides.size()));

  xt::xstrided_slice_vector sv;
  for (size_t idx = 0; idx < in.shape().size(); ++idx) {
    sv.push_back(xt::range(start_indices[idx], end_indices[idx],
                           strides.empty() ? 1 : strides[idx]));
  }

  return DISPATCH_ALL_ELSIZE(in.elsize(), [&]() -> Value {
    const auto& out = xt::strided_view(xt_adapt<element_t>(in), sv);

    return Value(in.buf(), in.eltype(), out.shape(), out.strides(),
                 out.data_offset() * in.elsize());
  });
}

Value reshape(HalContext* ctx, const Value& in,
              const std::vector<int64_t>& to_shape) {
  PPU_TRACE_OP(ctx, in, to_shape);

  PPU_ENFORCE(ppu::numel(in.shape()) == ppu::numel(to_shape));

  // TODO(jint) dont touch membuf, manipulate strides for transpose.
  return DISPATCH_ALL_ELSIZE(in.elsize(), [&]() -> Value {
    const auto& out =
        xt::eval(xt::reshape_view(xt_adapt<element_t>(in), to_shape));

    auto buf = makeBuffer(out.data(), out.size() * _kSize);
    return Value(std::move(buf), in.eltype(), out.shape(), out.strides(), 0);
  });
}

Value broadcast_to(HalContext* ctx, const Value& in,
                   const std::vector<int64_t>& to_shape,
                   const std::vector<size_t>& in_dims) {
  PPU_TRACE_OP(ctx, in, to_shape);

  if (in.shape() == to_shape) {
    return in;
  }

  Value operand;
  if (!in_dims.empty() && (in.shape().size() != to_shape.size())) {
    // Needs a reshape
    std::vector<int64_t> reshape_to(to_shape.size(), 1);
    for (size_t idx = 0; idx < in_dims.size(); ++idx) {
      reshape_to[in_dims[idx]] = in.shape()[idx];
    }
    operand = hal::reshape(ctx, in, reshape_to);
  } else {
    operand = in;
  }

  return DISPATCH_ALL_ELSIZE(in.elsize(), [&]() -> Value {
    const auto& out =
        xt::eval(xt::broadcast(xt_adapt<element_t>(operand), to_shape));

    // TODO: drop this copy
    auto buf = makeBuffer(out.data(), out.size() * _kSize);
    return Value(std::move(buf), operand.eltype(), out.shape(), out.strides(),
                 0);
  });
}

Value reverse(HalContext* ctx, const Value& in,
              const std::vector<size_t>& dimensions) {
  PPU_TRACE_OP(ctx, in, dimensions);
  return DISPATCH_ALL_ELSIZE(in.elsize(), [&]() -> Value {
    xt::xarray<element_t> expr = xt_adapt<element_t>(in);

    for (const auto dim : dimensions) {
      expr = xt::flip(expr, dim);
    }

    const auto& out = xt::eval(expr);

    // TODO: drop this copy
    auto buf = makeBuffer(out.data(), out.size() * _kSize);
    return Value(std::move(buf), in.eltype(), out.shape(), out.strides(), 0);
  });
}

Value pad(HalContext* ctx, const Value& in, const Value& padding_value,
          const std::vector<size_t>& edge_padding_low,
          const std::vector<size_t>& edge_padding_high,
          const std::vector<size_t>& interior_padding) {
  PPU_ENFORCE(in.dtype() == padding_value.dtype());
  PPU_ENFORCE(in.vtype() == padding_value.vtype());

  Value broadcasted =
      broadcast_to(ctx, padding_value,
                   DeducePadShape(in.shape(), edge_padding_low,
                                  edge_padding_high, interior_padding));

  return DISPATCH_ALL_ELSIZE(in.elsize(), [&]() -> Value {
    auto ret = xt_mutable_adapt<element_t>(broadcasted);
    auto in_xt = xt::eval(xt_adapt<element_t>(in));

    for (const auto& idx : AllIndices(in_xt)) {
      std::vector<size_t> idx_hat;
      for (size_t i = 0; i < idx.size(); ++i) {
        idx_hat.emplace_back(idx[i] + edge_padding_low[i] +
                             interior_padding[i] * idx[i]);
      }

      ret[idx_hat] = in_xt[idx];
    }

    PPU_ENFORCE(broadcasted.shape() == ret.shape());

    // TODO: drop this copy
    auto buf = makeBuffer(ret.data(), ret.size() * _kSize);
    return Value(std::move(buf), broadcasted.eltype(), ret.shape(),
                 ret.strides(), 0);
  });
}

}  // namespace ppu::hal
