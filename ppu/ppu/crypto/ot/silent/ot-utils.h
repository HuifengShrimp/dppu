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

#include "emp-tool/utils/block.h"

namespace ppu {
template <typename basetype>
void pack_ot_messages(basetype *y, const basetype *const *data,
                      const emp::block *pad, int ysize, int bsize, int bitsize,
                      int N);

template <typename basetype>
void unpack_ot_messages(basetype *data, const uint8_t *r, const basetype *recvd,
                        const emp::block *pad, int bsize, int bitsize, int N);

void pack_cot_messages(uint64_t *y, const uint64_t *corr_data, uint32_t ysize,
                       uint32_t bsize, int bitsize);

void unpack_cot_messages(uint64_t *corr_data, const uint64_t *recvd, int bsize,
                         int bitsize);

}  // namespace ppu
