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
#include "ppu/mpc/semi2k/arithmetic.h"
#include "ppu/mpc/util/cexpr.h"

namespace ppu::mpc::fss {

using util::Const;
using util::K;
using util::Log;
using util::N;

typedef ppu::mpc::semi2k::ZeroA ZeroA;

typedef ppu::mpc::semi2k::P2A P2A;

typedef ppu::mpc::semi2k::A2P A2P;

typedef ppu::mpc::semi2k::NegA NegA;

typedef ppu::mpc::semi2k::AddAP AddAP;

typedef ppu::mpc::semi2k::AddAA AddAA;

typedef ppu::mpc::semi2k::MulAP MulAP;

typedef ppu::mpc::semi2k::MulAA MulAA;

typedef ppu::mpc::semi2k::MatMulAP MatMulAP;

typedef ppu::mpc::semi2k::MatMulAA MatMulAA;


class LogReg : public LogRegKernel {
 public:
  static constexpr char kName[] = "LogReg";

  util::CExpr latency() const override { return Const(1); }

  util::CExpr comm() const override {
    // TODO(jint) express M, N, K
    return nullptr;
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x,
                      const ArrayRef& y, const ArrayRef& w,
                      int64_t M, int64_t N) const override;
};


}  // namespace ppu::mpc::fss

