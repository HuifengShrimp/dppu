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

#include "ppu/core/vectorize.h"
#include "ppu/hal/context.h"
#include "ppu/hal/value.h"

namespace ppu::hal {

/// applies a reduction function to one or more arrays in parallel.
// @param in, the input value
// @param init, the init value
// @param dimensions, unordered array of dimensions to reduce.
// @param binary_op, a computation function
Value reduce(HalContext* ctx, const Value& in, const Value& init,
             const std::vector<size_t>& dimensions,
             const BinaryFn<Value>& binary_op);

}  // namespace ppu::hal
