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

#include <utility>

#include "ppu/core/array_ref.h"
#include "ppu/mpc/kernel.h"

namespace ppu::mpc::base2k {

class RandP : public Kernel {
 public:
  static constexpr char kName[] = "RandP";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  void evaluate(KernelEvalContext* ctx) const override;
  ArrayRef proc(KernelEvalContext* ctx, FieldType field, size_t size) const;
};

class NegP : public UnaryKernel {
 public:
  static constexpr char kName[] = "NegP";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class EqzP : public UnaryKernel {
 public:
  static constexpr char kName[] = "EqzP";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class AddPP : public BinaryKernel {
 public:
  static constexpr char kName[] = "AddPP";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class MulPP : public BinaryKernel {
 public:
  static constexpr char kName[] = "MulPP";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class MatMulPP : public MatmulKernel {
 public:
  static constexpr char kName[] = "MatMulPP";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs, int64_t M, int64_t N,
                int64_t K) const override;
};

class AndPP : public BinaryKernel {
 public:
  static constexpr char kName[] = "AndPP";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class XorPP : public BinaryKernel {
 public:
  static constexpr char kName[] = "XorPP";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class LShiftP : public UnaryWithBitsKernel {
 public:
  static constexpr char kName[] = "LShiftP";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};

class RShiftP : public UnaryWithBitsKernel {
 public:
  static constexpr char kName[] = "RShiftP";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};

class ReverseBitsP : public ReverseBitsKernel {
 public:
  static constexpr char kName[] = "ReverseBitsP";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in, size_t start,
                size_t end) const override;
};

class ARShiftP : public UnaryWithBitsKernel {
 public:
  static constexpr char kName[] = "ARShiftP";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};

class MsbP : public UnaryKernel {
 public:
  static constexpr char kName[] = "MsbP";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

}  // namespace ppu::mpc::base2k
