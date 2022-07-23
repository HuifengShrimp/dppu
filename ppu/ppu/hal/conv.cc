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


#include "ppu/hal/conv.h"

#include <cstddef>
#include <tuple>
#include <vector>

#include "io_ops.h"
#include "shape_ops.h"

#include "ppu/hal/polymorphic.h"
#include "ppu/hal/reduce.h"
#include "ppu/hal/shape_ops.h"

namespace ppu::hal {

Value conv2d_b01f_01io_b01f(
    HalContext* ctx, const Value& input, const Value& kernel,
    const std::vector<size_t>& window_strides,
    const std::vector<std::pair<size_t, size_t>>& padding) {
  PPU_ENFORCE(input.shape().size() == 4 && kernel.shape().size() == 4);
  PPU_ENFORCE(input.shape()[3] == kernel.shape()[2]);
  PPU_ENFORCE(window_strides.size() == 2 && padding.size() == 2);

  // 01io -> o01i
  auto transposed_kernel = transpose(ctx, kernel, {3, 0, 1, 2});

  PPU_ENFORCE(input.shape()[3] == transposed_kernel.shape()[3]);

  const auto batch = input.shape()[0];
  const auto feature = input.shape()[3];
  const auto input_h = input.shape()[1];
  const auto input_w = input.shape()[2];
  const auto out = transposed_kernel.shape()[0];
  const auto kernel_h = transposed_kernel.shape()[1];
  const auto kernel_w = transposed_kernel.shape()[2];

  // add padding
  auto padding_value = input.dtype() == DT_INT
                           ? make_value(ctx, input.vtype(), 0)
                           : make_value(ctx, input.vtype(), 0.0);
  auto padded_input =
      pad(ctx, input, padding_value, {0, padding[0].first, padding[1].first, 0},
          {0, padding[0].second, padding[1].second, 0}, {0, 0, 0, 0});

  PPU_ENFORCE((input_h + padding[0].first + padding[0].second) >=
              static_cast<size_t>(kernel_h));
  PPU_ENFORCE((input_w + padding[1].first + padding[1].second) >=
              static_cast<size_t>(kernel_w));

  const size_t out_h =
      (input_h - kernel_h + padding[0].first + padding[0].second) /
          window_strides[0] +
      1;
  const size_t out_w =
      (input_w - kernel_w + padding[1].first + padding[1].second) /
          window_strides[1] +
      1;

  DataType ret_dtype = input.dtype();

  if (kernel.dtype() != ret_dtype) {
    // If input dtype is different from kernel, it must be DT_INT + DT_FXP,
    // thus, result must be DT_FXP
    ret_dtype = DT_FXP;
  }

#define OPTIMIZED
#ifdef OPTIMIZED
  std::vector<Value> im2col_elements;

  const Value flattened_transposed_kernel =
      reshape(ctx, transposed_kernel, {out, feature * kernel_h * kernel_w});
  for (size_t i = 0;
       i <= input_h - kernel_h + padding[0].first + padding[0].second;
       i += window_strides[0]) {
    for (size_t j = 0;
         j <= input_w - kernel_w + padding[1].first + padding[1].second;
         j += window_strides[1]) {
      const auto sliced_input =
          reshape(ctx,
                  slice(ctx, padded_input, {0, i, j, 0},
                        {static_cast<size_t>(batch), i + kernel_h, j + kernel_w,
                         static_cast<size_t>(feature)},
                        {}),
                  {batch, feature * kernel_h * kernel_w});

      im2col_elements.emplace_back(sliced_input);
    }
  }

  PPU_ENFORCE(!im2col_elements.empty());

  Value im2col = concatenate(ctx, im2col_elements, 1);

  im2col = reshape(ctx, im2col,
                   {static_cast<int64_t>(batch * out_h * out_w),
                    feature * kernel_h * kernel_w});

  auto ret = matmul(ctx, im2col, transpose(ctx, flattened_transposed_kernel));

  return reshape(
      ctx, ret,
      {batch, static_cast<int64_t>(out_h), static_cast<int64_t>(out_w), out});
#else
  Value ret = makeValue(NdArrayRef(input.eltype().as<ValueTy>()->mpc_type(),
                                   {out_h * out_w * batch * out}),
                        ret_dtype);

  auto init_val = make_value(ctx, VIS_PUBLIC, 0);

  size_t idx = 0;
  for (size_t b = 0; b < batch; b++) {
    for (size_t o = 0; o < out; o++) {
      for (size_t i = 0;
           i <= input_h - kernel_h + padding[0].first + padding[0].second;
           i += window_strides[0]) {
        for (size_t j = 0;
             j <= input_w - kernel_w + padding[1].first + padding[1].second;
             j += window_strides[1]) {
          auto sliced_input =
              slice(ctx, padded_input, {b, i, j, 0},
                    {b + 1, i + kernel_h, j + kernel_w, feature}, {});

          auto sliced_kernel = slice(ctx, transposed_kernel, {o, 0, 0, 0},
                                     {o + 1, kernel_h, kernel_w, feature}, {});

          auto prod = mul(ctx, sliced_input, sliced_kernel);

          auto reduced_sum = reduce(ctx, prod, init_val, {3, 2, 1},
                                    [&ctx](const Value& a, const Value& b) {
                                      return add(ctx, a, b);
                                    });
          PPU_ENFORCE(reduced_sum.numel() == 1);
          ret.CopyElementFrom(reduced_sum, 0, idx++);
        }
      }
    }
  }
  // bf01
  ret = reshape(ctx, ret, {batch, out, out_h, out_w});

  // bf01 -> b01f
  return transpose(ctx, ret, {0, 2, 3, 1});
#endif
}

}  // namespace ppu::hal
