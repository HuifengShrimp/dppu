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

#include "ppu/core/array_ref.h"
#include "ppu/core/array_ref_util.h"
#include "ppu/mpc/aby3/defs.h"
#include "ppu/mpc/kernel.h"

namespace ppu::mpc::aby3 {

// Referrence:
// ABY3: A Mixed Protocol Framework for Machine Learning
// P16 5.3 Share Conversions, Bit Decomposition
// https://eprint.iacr.org/2018/403.pdf
//
// Latency: 2 + log(nbits) from 2 rotate and 1 ppa.
// TODO(junfeng): Optimize anount of comm.
class A2B : public UnaryKernel {
 public:
  static constexpr char kName[] = "A2B";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

// Referrence:
// IV.E Boolean to Arithmetic Sharing (B2A), extended to 3pc settings.
// https://encrypto.de/papers/DSZ15.pdf
//
// Latency: 4 + log(nbits) - 3 rotate + 1 send/rec + 1 ppa.
// TODO(junfeng): Optimize anount of comm.
class B2A : public UnaryKernel {
 public:
  static constexpr char kName[] = "B2A";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x) const override;
};

// Referrence:
// 5.4.1 Semi-honest Security
// https://eprint.iacr.org/2018/403.pdf
//
// Latency: 2.
class B2AByOT : public UnaryWithBitsKernel {
 public:
  static constexpr char kName[] = "B2AByOT";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t valid_bits) const override;
};

class AddBB : public BinaryKernel {
 public:
  static constexpr char kName[] = "AddBB";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

}  // namespace ppu::mpc::aby3
