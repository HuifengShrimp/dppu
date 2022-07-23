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

#include "ppu/core/array_ref_util.h"
#include "ppu/hal/value.h"

namespace ppu {

class HalContext;

namespace hal {

Value scalar_const(HalContext* ctx, PtBufferView bv);

Value shaped_const(HalContext* ctx, PtBufferView bv,
                   const std::vector<int64_t>& shape);

}  // namespace hal
}  // namespace ppu
