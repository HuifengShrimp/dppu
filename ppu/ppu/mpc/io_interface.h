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
#include <vector>

#include "ppu/core/array_ref.h"

namespace ppu::mpc {

// The basic io interface of protocols.
class IoInterface {
 public:
  virtual ~IoInterface() = default;

  // Make shares (public/secret) from plaintext.
  //
  // @param raw, a plaintext ndarray in ring2k space.
  // @return a list of ndarray, each of which could be send to one mpc engine
  //         without data loss.
  //
  // If client want to support fixedpoint, it's the client's responsibility to
  // encode it to ring first.
  virtual std::vector<NdArrayRef> makePublic(const NdArrayRef& raw) const = 0;
  virtual std::vector<NdArrayRef> makeSecret(const NdArrayRef& raw) const = 0;

  // Reconstruct shares into plaintext.
  //
  // @param shares, a list of secret shares.
  // @return a combined plaintext in ring2k space.
  virtual NdArrayRef reconstruct(
      const std::vector<NdArrayRef>& shares) const = 0;
};

}  // namespace ppu::mpc
