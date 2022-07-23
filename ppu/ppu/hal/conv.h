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

#include <utility>

#include "ppu/hal/context.h"
#include "ppu/hal/value.h"

namespace ppu::hal {

// NOTE(junfeng): Although the naming is lengthy but accurate. Please
// check the comments below. All the conv2D could be expressed by this method
// with proper permutaton.
// Input tensor is 4 dimensions with [batch dim, spatial dim 0, spatial dim 1,
// feature dim]. Kernel tensor is 4 dimensions with [spatial dim 0, spatial dim
// 1, input dim, output dim]. Output tensor is 4 dimensions with [batch dim,
// spatial dim 0, spatial dim 1, output dim]
Value conv2d_b01f_01io_b01f(
    HalContext* ctx, const Value& input, const Value& kernel,
    const std::vector<size_t>& window_strides = std::vector<size_t>{1, 1},
    const std::vector<std::pair<size_t, size_t>>& padding = {{0, 0}, {0, 0}});

}  // namespace ppu::hal
