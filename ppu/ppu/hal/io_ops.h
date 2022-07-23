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
#include "ppu/hal/context.h"
#include "ppu/hal/value.h"

namespace ppu::hal {

/// make a constant value
// @param content, the content of constant value
// @param type, the plaintext type of the value
// @param shape, shape of the constant
Value constant(HalContext* ctx, const std::shared_ptr<Buffer>& content,
               const PtType& type, const std::vector<int64_t>& shape);

Value iota(HalContext* ctx, size_t numel);

/// Make a secret from a buffer view.
//
// Note: this function will be called from all ppu device slice.
//
// if rank == kInvalidRank:
//   then all ppu slice provides exactly the same buffer.
// else:
//   the buffer from rank is used, which means this secret from `rank` in
//   colocated mode.
Value make_secret(HalContext* ctx, PtBufferView bv, Rank owner = kInvalidRank);

// Import a buffer into a public variable.
Value make_public(HalContext* ctx, PtBufferView bv);

// Export a value to a buffer.
NdArrayRef dump_public(HalContext* ctx, const Value& v);

// General make value interface.
// - if vtype != PRIVATE, rank is ignored.
Value make_value(HalContext* ctx, Visibility vtype, PtBufferView bv,
                 Rank rank = kInvalidRank);

}  // namespace ppu::hal
