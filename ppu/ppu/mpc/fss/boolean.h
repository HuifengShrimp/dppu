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
#include "ppu/mpc/util/cexpr.h"
#include "ppu/mpc/semi2k/boolean.h"

namespace ppu::mpc::fss {

using util::Const;
using util::K;
using util::Log;
using util::N;

typedef ppu::mpc::semi2k::ZeroB ZeroB;

typedef ppu::mpc::semi2k::B2P B2P;

typedef ppu::mpc::semi2k::P2B P2B;

typedef ppu::mpc::semi2k::AndBP AndBP;

typedef ppu::mpc::semi2k::AndBB AndBB;

typedef ppu::mpc::semi2k::XorBP XorBP;

typedef ppu::mpc::semi2k::XorBB XorBB;

typedef ppu::mpc::semi2k::LShiftB LShiftB;

typedef ppu::mpc::semi2k::RShiftB RShiftB;

typedef ppu::mpc::semi2k::ARShiftB ARShiftB;

typedef ppu::mpc::semi2k::ReverseBitsB ReverseBitsB;


}  // namespace ppu::mpc::fss
