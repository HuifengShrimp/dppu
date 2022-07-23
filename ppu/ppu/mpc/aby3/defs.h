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

#include <complex>

#include "ppu/link/link.h"
#include "ppu/mpc/aby3/ot.h"
#include "ppu/mpc/object.h"

namespace ppu::mpc::aby3 {

// re-use the std::complex definition for aby3 share, where:
//   real(x) is the first share piece.
//   imag(x) is the second share piece.
template <typename T>
using Share = std::complex<T>;

class Aby3State : public State {
  std::shared_ptr<link::Context> lctx_;

  OT3Party ot_;

 public:
  static constexpr char kName[] = "Aby3State";

  explicit Aby3State(std::shared_ptr<link::Context> lctx)
      : lctx_(lctx), ot_(lctx) {}

  const std::shared_ptr<link::Context>& lctx() { return lctx_; }

  OT3Party* ot() { return &ot_; }
};

}  // namespace ppu::mpc::aby3
