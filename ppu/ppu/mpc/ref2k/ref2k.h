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

#include "ppu/core/type_util.h"
#include "ppu/link/link.h"
#include "ppu/mpc/base2k/ring_io.h"
#include "ppu/mpc/object.h"

namespace ppu::mpc {

class Ref2kIo final : public RingIo {
 public:
  using RingIo::RingIo;

  std::vector<NdArrayRef> makeSecret(const NdArrayRef& raw) const override;

  NdArrayRef reconstructSecret(
      const std::vector<NdArrayRef>& shares) const override;
};

std::unique_ptr<Object> makeRef2kProtocol(
    const std::shared_ptr<link::Context>& lctx);

std::unique_ptr<Ref2kIo> makeRef2kIo(size_t npc);

}  // namespace ppu::mpc
