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


#include "ppu/numpy/initializer.h"

#include "ppu/core/buffer.h"
#include "ppu/hal/io_ops.h"

namespace ppu::numpy {

hal::Value ones(HalContext* ctx, absl::Span<const int64_t> shape,
                DataType dtype) {
  PPU_TRACE_OP(ctx, shape, dtype);

  if (dtype == DT_FXP) {
    xt::xarray<float> data = xt::ones<float>(shape);
    return hal::make_public(ctx, data).as_fxp();
  } else if (dtype == DT_INT) {
    xt::xarray<int32_t> data = xt::ones<int32_t>(shape);
    return hal::make_public(ctx, data).as_int();
  } else {
    PPU_THROW("ones, not supported type={}", dtype);
  }
}

hal::Value ones_like(HalContext* ctx, const hal::Value& x) {
  PPU_TRACE_OP(ctx, x);
  return ones(ctx, x.shape(), x.dtype());
}

hal::Value zeros(HalContext* ctx, absl::Span<const int64_t> shape,
                 DataType dtype) {
  PPU_TRACE_OP(ctx, shape, dtype);

  if (dtype == DT_FXP) {
    xt::xarray<float> data = xt::zeros<float>(shape);
    return hal::make_public(ctx, data).as_fxp();
  } else if (dtype == DT_INT) {
    xt::xarray<int32_t> data = xt::zeros<int32_t>(shape);
    return hal::make_public(ctx, data).as_int();
  } else {
    PPU_THROW("zeros, not supported type={}", dtype);
  }
}

hal::Value zeros_like(HalContext* ctx, const hal::Value& x) {
  PPU_TRACE_OP(ctx, x);
  return zeros(ctx, x.shape(), x.dtype());
}

}  // namespace ppu::numpy
