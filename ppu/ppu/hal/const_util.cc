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


#include "ppu/hal/const_util.h"

#include "ppu/hal/io_ops.h"
#include "ppu/hal/shape_ops.h"
#include "ppu/utils/exception.h"

namespace ppu::hal {

Value scalar_const(HalContext* ctx, PtBufferView bv) {
  PPU_ENFORCE(numel(bv.shape) == 1);
  return make_public(ctx, bv);
}

Value shaped_const(HalContext* ctx, PtBufferView bv,
                   const std::vector<int64_t>& shape) {
  // This helper function can't do slice
  PPU_ENFORCE(numel(bv.shape) <= numel(shape));

  // If view shape is same as destination shape, just make public
  if (bv.shape == shape) {
    return make_public(ctx, bv);
  }

  // Same numel but shape is different, do a reshape
  if (numel(bv.shape) == numel(shape)) {
    return reshape(ctx, make_public(ctx, bv), shape);
  }

  // Other, do a broadcast, let broadcast handles the sanity check
  return broadcast_to(ctx, make_public(ctx, bv), shape);
}

}  // namespace ppu::hal
