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

#include "ppu/mpc/beaver/beaver_cheetah.h"
#include "ppu/mpc/util/communicator.h"

namespace ppu::mpc {

// TODO(jint) split this into individual states.
class CheetahState : public State {
  std::unique_ptr<BeaverCheetah> beaver_;
  std::shared_ptr<CheetahPrimitives> primitives_;

 public:
  static constexpr char kName[] = "CheetahState";

  explicit CheetahState(std::shared_ptr<link::Context> lctx) {
    primitives_ = std::make_shared<CheetahPrimitives>(lctx);
    beaver_ = std::make_unique<BeaverCheetah>(lctx);
    beaver_->set_primitives(primitives_);
  }

  ~CheetahState() {}

  Beaver* beaver() { return beaver_.get(); }
  CheetahPrimitives* primitives() { return primitives_.get(); }
};

}  // namespace ppu::mpc
