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
#include "ppu/mpc/io_interface.h"

namespace ppu::mpc {

// A helper base class which handles all public related stuffs.
class RingIo : public IoInterface {
 protected:
  size_t const world_size_;

  virtual NdArrayRef reconstructSecret(
      const std::vector<NdArrayRef>& shares) const = 0;

  std::vector<NdArrayRef> randAdditiveSplits(const NdArrayRef& arr) const;

 public:
  RingIo(size_t world_size) : world_size_(world_size) {}

  std::vector<NdArrayRef> makePublic(const NdArrayRef& raw) const override;

  NdArrayRef reconstruct(const std::vector<NdArrayRef>& shares) const override;
};

}  // namespace ppu::mpc
