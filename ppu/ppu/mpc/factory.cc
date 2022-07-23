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


#include "ppu/mpc/factory.h"

#include <memory>

#include "ppu/mpc/aby3/io.h"
#include "ppu/mpc/aby3/protocol.h"
#include "ppu/mpc/cheetah/io.h"
#include "ppu/mpc/cheetah/protocol.h"
#include "ppu/mpc/fss/io.h"
#include "ppu/mpc/fss/protocol.h"
#include "ppu/mpc/ref2k/ref2k.h"
#include "ppu/mpc/semi2k/io.h"
#include "ppu/mpc/semi2k/protocol.h"
#include "ppu/utils/exception.h"

namespace ppu::mpc {

std::unique_ptr<Object> Factory::CreateCompute(
    ProtocolKind kind, const std::shared_ptr<link::Context>& lctx) {
  switch (kind) {
    case ProtocolKind::REF2K: {
      return makeRef2kProtocol(lctx);
    }
    case ProtocolKind::SEMI2K: {
      return makeSemi2kProtocol(lctx);
    }
    case ProtocolKind::ABY3: {
      return makeAby3Protocol(lctx);
    }
    case ProtocolKind::CHEETAH: {
      return makeCheetahProtocol(lctx);
    }
    case ProtocolKind::FSS: {
      return makeFSSProtocol(lctx);
    }
    default: {
      PPU_THROW("Invalid protocol kind {}", kind);
    }
  }
  return nullptr;
}

std::unique_ptr<IoInterface> Factory::CreateIO(ProtocolKind kind, size_t npc) {
  switch (kind) {
    case ProtocolKind::REF2K: {
      return makeRef2kIo(npc);
    }
    case ProtocolKind::SEMI2K: {
      return semi2k::makeSemi2kIo(npc);
    }
    case ProtocolKind::ABY3: {
      return aby3::makeAby3Io(npc);
    }
    case ProtocolKind::CHEETAH: {
      return cheetah::makeCheetahIo(npc);
    }
    case ProtocolKind::FSS: {
      return fss::makeFSSIo(npc);
    }
    default: {
      PPU_THROW("Invalid protocol kind {}", kind);
    }
  }
  return nullptr;
}

}  // namespace ppu::mpc
