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

#include <memory>
#include <random>

#include "ppu/core/encoding.h"
#include "ppu/core/trace.h"
#include "ppu/link/link.h"
#include "ppu/mpc/object.h"

#include "ppu/ppu.pb.h"

namespace ppu {

// The evaluation context for all ppu operators.
class HalContext final {
  const RuntimeConfig rt_config_;

  const std::shared_ptr<link::Context> lctx_;

  std::default_random_engine rand_engine_;

  std::unique_ptr<mpc::Object> prot_;

 public:
  explicit HalContext(RuntimeConfig config,
                      std::shared_ptr<link::Context> lctx);

  HalContext(const HalContext& other) = delete;
  HalContext& operator=(const HalContext& other) = delete;

  HalContext(HalContext&& other) = default;

  //
  const std::shared_ptr<link::Context>& lctx() const { return lctx_; }

  mpc::Object* prot() const { return prot_.get(); }

  size_t FxpBits() const { return FxpFractionalBits(rt_config_); }

  FieldType GetField() const { return rt_config_.field(); }

  //
  const RuntimeConfig& rt_config() const { return rt_config_; }

  //
  std::default_random_engine& rand_engine() { return rand_engine_; }
};

}  // namespace ppu
