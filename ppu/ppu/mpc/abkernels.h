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

#include "ppu/mpc/object.h"

namespace ppu::mpc {

class ABState : public State {
 public:
  static constexpr char kName[] = "ABState";

  bool lazy_ab = true;
};

class P2S : public UnaryKernel {
 public:
  static constexpr char kName[] = "P2S";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class S2P : public UnaryKernel {
 public:
  static constexpr char kName[] = "S2P";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class NegS : public UnaryKernel {
 public:
  static constexpr char kName[] = "NegS";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class AddSP : public BinaryKernel {
 public:
  static constexpr char kName[] = "AddSP";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class AddSS : public BinaryKernel {
 public:
  static constexpr char kName[] = "AddSS";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class MulSP : public BinaryKernel {
 public:
  static constexpr char kName[] = "MulSP";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class MulSS : public BinaryKernel {
 public:
  static constexpr char kName[] = "MulSS";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class MatMulSP : public MatmulKernel {
 public:
  static constexpr char kName[] = "MatMulSP";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& A, const ArrayRef& B,
                int64_t M, int64_t N, int64_t K) const override;
};

class MatMulSS : public MatmulKernel {
 public:
  static constexpr char kName[] = "MatMulSS";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& A, const ArrayRef& B,
                int64_t M, int64_t N, int64_t K) const override;
};

class LogReg : public LogRegKernel {
 public:
  static constexpr char kName[] = "LogReg";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& A, const ArrayRef& B, const ArrayRef& C,
                int64_t M, int64_t N) const override;
};

class AndSP : public BinaryKernel {
 public:
  static constexpr char kName[] = "AndSP";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class AndSS : public BinaryKernel {
 public:
  static constexpr char kName[] = "AndSS";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class XorSP : public BinaryKernel {
 public:
  static constexpr char kName[] = "XorSP";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class XorSS : public BinaryKernel {
 public:
  static constexpr char kName[] = "XorSS";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class EqzS : public UnaryKernel {
 public:
  static constexpr char kName[] = "EqzS";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class LShiftS : public UnaryWithBitsKernel {
 public:
  static constexpr char kName[] = "LShiftS";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};

class RShiftS : public UnaryWithBitsKernel {
 public:
  static constexpr char kName[] = "RShiftS";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};

class ARShiftS : public UnaryWithBitsKernel {
 public:
  static constexpr char kName[] = "ARShiftS";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};

class TruncPrS : public UnaryWithBitsKernel {
 public:
  static constexpr char kName[] = "TruncPrS";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};

class ReverseBitsS : public Kernel {
 public:
  static constexpr char kName[] = "ReverseBitsS";

  Kind kind() const override { return Kind::kDynamic; }

  void evaluate(EvalContext* ctx) const override {
    ctx->setOutput(proc(ctx, ctx->getParam<ArrayRef>(0),
                        ctx->getParam<size_t>(1), ctx->getParam<size_t>(2)));
  }
  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in, size_t start,
                size_t end) const;
};

class MsbS : public UnaryKernel {
 public:
  static constexpr char kName[] = "MsbS";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

}  // namespace ppu::mpc
