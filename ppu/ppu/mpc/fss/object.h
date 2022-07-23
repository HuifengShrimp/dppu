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

#include "ppu/mpc/beaver/beaver_fss.h"
#include "ppu/mpc/util/communicator.h"
#

namespace ppu::mpc {

// TODO(jint) split this into individual states.
class FSSState : public State {
  std::unique_ptr<BeaverFss> dealer_;
  // std::shared_ptr<CheetahPrimitives> primitives_;

 public:
  static constexpr char kName[] = "FSSState";

  explicit FSSState(std::shared_ptr<link::Context> lctx) {
    // primitives_ = std::make_shared<CheetahPrimitives>(lctx);
    dealer_ = std::make_unique<BeaverFss>(lctx);
    // dealer_->set_primitives(primitives_);
  }

  // ~FSSState() {}

  BeaverFss* dealer() { return dealer_.get(); }
  // CheetahPrimitives* primitives() { return primitives_.get(); }
};

}  // namespace ppu::mpc
