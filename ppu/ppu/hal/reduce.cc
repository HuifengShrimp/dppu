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


#include "ppu/hal/reduce.h"

#include "ppu/core/vectorize.h"
#include "ppu/hal/polymorphic.h"
#include "ppu/hal/shape_ops.h"
#include "ppu/utils/exception.h"

namespace ppu::hal {
namespace {

Value Concat(HalContext* ctx, absl::Span<const Value> vs, size_t dim) {
  PPU_ENFORCE(vs.size() > 0, "concat requires length greater than 0");
  const auto& shape = vs[0].shape();
  for (size_t idx = 1; idx < vs.size(); idx++) {
    PPU_ENFORCE(vs[idx].shape() == shape, "shape mis-match");
  }

  return concatenate(ctx, vs, dim);
}

std::vector<Value> Split(HalContext* ctx, const Value& in, size_t dim,
                         size_t num_splits) {
  const auto& shape = in.shape();

  PPU_ENFORCE(dim < shape.size(), "invalid dim={}", dim);
  PPU_ENFORCE(
      shape.at(dim) % num_splits == 0,
      "split assume original values are concatenated at dim={} with same shape",
      dim);

  size_t width = shape.at(dim) / num_splits;

  std::vector<Value> results;
  for (size_t idx = 0; idx < num_splits; idx++) {
    std::vector<size_t> start_indices(shape.size(), 0);
    std::vector<size_t> end_indices(in.shape().begin(), in.shape().end());
    start_indices[dim] = idx * width;
    end_indices[dim] = (idx + 1) * width;

    results.push_back(slice(ctx, in, start_indices, end_indices, {}));
  }
  return results;
}

}  // namespace

Value reduce(HalContext* ctx, const Value& in, const Value& init,
             const std::vector<size_t>& dimensions,
             const BinaryFn<Value>& binary_op) {
  Value res = in;

  for (const auto& dim : dimensions) {
    PPU_ENFORCE(dim < in.shape().size(),
                "reduce dim={} should be small than tensor rank={}", dim,
                in.shape().size());

    // split dim.
    std::vector<Value> vals;
    for (int64_t idx = 0; idx < res.shape().at(dim); idx++) {
      std::vector<size_t> start_indices(res.shape().size(), 0);
      std::vector<size_t> end_indices(res.shape().begin(), res.shape().end());
      start_indices[dim] = idx;
      end_indices[dim] = idx + 1;

      vals.push_back(slice(ctx, res, start_indices, end_indices, {}));
    }

    auto concat = [&](absl::Span<const Value> vs) {
      return Concat(ctx, vs, 0);
    };
    auto split = [&](const Value& v, size_t num_splits) {
      return Split(ctx, v, 0, num_splits);
    };
    auto vectorize_op = [&](absl::Span<const Value> lhs,
                            absl::Span<const Value> rhs) {
      return Vectorize<Value>(lhs, rhs, concat, split, binary_op);
    };

    // reduce it.
    res = VectorizedReduce<Value>(vals, vectorize_op);
  }

  std::vector<int64_t> new_shape;
  for (size_t idx = 0; idx < in.shape().size(); idx++) {
    if (std::find(dimensions.begin(), dimensions.end(), idx) ==
        dimensions.end()) {
      new_shape.push_back(in.shape().at(idx));
    }
  }

  // last step, reshape it.
  auto tail = reshape(ctx, res, new_shape);

  return binary_op(broadcast_to(ctx, init, tail.shape()), tail);
}

}  // namespace ppu::hal
