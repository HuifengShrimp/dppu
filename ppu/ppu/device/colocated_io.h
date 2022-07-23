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
#include "ppu/device/processor.h"
#include "ppu/hal/value.h"

namespace ppu::device {

class ColocatedIo final {
  // plaintext variable buffer.
  Processor *const processor_;

  std::map<std::string, NdArrayRef> pending_;

public:
  explicit ColocatedIo(Processor *dev) : processor_(dev) {}

  size_t rank() const { return processor_->lctx()->Rank(); }

  size_t world_size() const { return processor_->lctx()->WorldSize(); }

  /// Set a variable to the device.
  void setVar(const std::string &name, PtBufferView bv);

  /// Get a variable from the device.
  hal::Value getVar(const std::string &name);

  /// Synchronize.
  void sync();
};

} // namespace ppu::device
