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

namespace ppu::mpc::semi2k {

using util::Const;
using util::K;
using util::Log;
using util::N;

class ZeroA : public Kernel {
 public:
  static constexpr char kName[] = "ZeroA";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(
        proc(ctx, ctx->getParam<FieldType>(0), ctx->getParam<size_t>(1)));
  }

  ArrayRef proc(KernelEvalContext* ctx, FieldType field, size_t size) const;
};

class P2A : public UnaryKernel {
 public:
  static constexpr char kName[] = "P2A";

  util::CExpr latency() const override { return Const(0); }

  util::CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class A2P : public UnaryKernel {
 public:
  static constexpr char kName[] = "A2P";

  util::CExpr latency() const override { return Const(1); }

  util::CExpr comm() const override { return K(); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class NegA : public UnaryKernel {
 public:
  static constexpr char kName[] = "NegA";

  util::CExpr latency() const override { return Const(0); }

  util::CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
class AddAP : public BinaryKernel {
 public:
  static constexpr char kName[] = "AddAP";

  util::CExpr latency() const override { return Const(0); }

  util::CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class AddAA : public BinaryKernel {
 public:
  static constexpr char kName[] = "AddAA";

  util::CExpr latency() const override { return Const(0); }

  util::CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
class MulAP : public BinaryKernel {
 public:
  static constexpr char kName[] = "MulAP";

  util::CExpr latency() const override { return Const(0); }

  util::CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class MulAA : public BinaryKernel {
 public:
  static constexpr char kName[] = "MulAA";

  util::CExpr latency() const override {
    // TODO: consider beaver
    return Const(1);
  }

  util::CExpr comm() const override { return K() * 2 * (N() - 1); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

////////////////////////////////////////////////////////////////////
// matmul family
////////////////////////////////////////////////////////////////////
class MatMulAP : public MatmulKernel {
 public:
  static constexpr char kName[] = "MatMulAP";

  util::CExpr latency() const override { return Const(0); }

  util::CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& A, const ArrayRef& B,
                int64_t M, int64_t N, int64_t K) const override;
};

class MatMulAA : public MatmulKernel {
 public:
  static constexpr char kName[] = "MatMulAA";

  util::CExpr latency() const override { return Const(1); }

  util::CExpr comm() const override {
    // TODO(jint) express M, N, K
    return nullptr;
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& A, const ArrayRef& B,
                int64_t M, int64_t N, int64_t K) const override;
};

class TruncPrA : public UnaryWithBitsKernel {
 public:
  static constexpr char kName[] = "TruncPrA";

  util::CExpr latency() const override {
    // TODO: handle case > 3PC
    return Const(0);
  }

  util::CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};

}  // namespace ppu::mpc::semi2k
