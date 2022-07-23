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

#include "ppu/mpc/kernel.h"
#include "ppu/mpc/semi2k/boolean.h"
#include "ppu/mpc/util/cexpr.h"

namespace ppu::mpc::cheetah {

using util::Const;
using util::K;
using util::Log;
using util::N;

typedef ppu::mpc::semi2k::ZeroB ZeroB;

typedef ppu::mpc::semi2k::B2P B2P;

typedef ppu::mpc::semi2k::P2B P2B;

typedef ppu::mpc::semi2k::AndBP AndBP;

class AndBB : public BinaryKernel {
 public:
  static constexpr char kName[] = "AndBB";

  util::CExpr latency() const override { return Const(0); }

  util::CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

typedef ppu::mpc::semi2k::XorBP XorBP;

typedef ppu::mpc::semi2k::XorBB XorBB;

typedef ppu::mpc::semi2k::LShiftB LShiftB;

typedef ppu::mpc::semi2k::RShiftB RShiftB;

typedef ppu::mpc::semi2k::ARShiftB ARShiftB;

typedef ppu::mpc::semi2k::ReverseBitsB ReverseBitsB;

}  // namespace ppu::mpc::cheetah
