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


#include "ppu/mpc/abkernels.h"

#include "ppu/core/trace.h"

namespace ppu::mpc {
namespace {

ArrayRef _Lazy2B(Object* obj, const ArrayRef& in) {
  if (in.eltype().isa<AShare>()) {
    return obj->call("A2B", in);
  } else {
    PPU_ENFORCE(in.eltype().isa<BShare>());
    return in;
  }
}

ArrayRef _Lazy2A(Object* obj, const ArrayRef& in) {
  if (in.eltype().isa<BShare>()) {
    return obj->call("B2A", in);
  } else {
    PPU_ENFORCE(in.eltype().isa<AShare>());
    return in;
  }
}

#define _LAZY_AB ctx->caller()->getState<ABState>()->lazy_ab

#define _2A(x) _Lazy2A(ctx->caller(), x)
#define _2B(x) _Lazy2B(ctx->caller(), x)

#define _A2P(x) ctx->caller()->call("A2P", x)
#define _P2A(x) ctx->caller()->call("P2A", x)
#define _NegA(x) ctx->caller()->call("NegA", x)
#define _AddAP(lhs, rhs) ctx->caller()->call("AddAP", lhs, rhs)
#define _AddAA(lhs, rhs) ctx->caller()->call("AddAA", lhs, rhs)
#define _MulAP(lhs, rhs) ctx->caller()->call("MulAP", lhs, rhs)
#define _MulAA(lhs, rhs) ctx->caller()->call("MulAA", lhs, rhs)
#define _TruncPrA(in, bits) ctx->caller()->call("TruncPrA", in, bits)
#define _MatMulAP(A, B, M, N, K) ctx->caller()->call("MatMulAP", A, B, M, N, K)
#define _MatMulAA(A, B, M, N, K) ctx->caller()->call("MatMulAA", A, B, M, N, K)
#define _LogReg(A, B, C, M, N) ctx->caller()->call("LogReg", A, B, C, M, N)
#define _B2P(x) ctx->caller()->call("B2P", x)
#define _P2B(x) ctx->caller()->call("P2B", x)
#define _A2B(x) ctx->caller()->call("A2B", x)
#define _B2A(x) ctx->caller()->call("B2A", x)
#define _AndBP(lhs, rhs) ctx->caller()->call("AndBP", lhs, rhs)
#define _AndBB(lhs, rhs) ctx->caller()->call("AndBB", lhs, rhs)
#define _XorBP(lhs, rhs) ctx->caller()->call("XorBP", lhs, rhs)
#define _XorBB(lhs, rhs) ctx->caller()->call("XorBB", lhs, rhs)
#define _LShiftB(in, bits) ctx->caller()->call("LShiftB", in, bits)
#define _RShiftB(in, bits) ctx->caller()->call("RShiftB", in, bits)
#define _ARShiftB(in, bits) ctx->caller()->call("ARShiftB", in, bits)
#define _ReverseBitsB(in, start, end) \
  ctx->caller()->call("ReverseBitsB", in, start, end)
#define _MsbA(in) ctx->caller()->call("MsbA", in)
}  // namespace

ArrayRef P2S::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  PPU_TRACE_OP(this, in);
  return _P2A(in);
}

ArrayRef S2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  PPU_TRACE_OP(this, in);
  if (_LAZY_AB) {
    return _A2P(_2A(in));
  }
  return _A2P(in);
}

ArrayRef NegS::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  PPU_TRACE_OP(this, in);
  if (_LAZY_AB) {
    return _NegA(_2A(in));
  }
  return _NegA(in);
}

ArrayRef AddSP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);
  if (_LAZY_AB) {
    return _AddAP(_2A(lhs), rhs);
  }
  return _AddAP(lhs, rhs);
}

ArrayRef AddSS::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);
  if (_LAZY_AB) {
    return _AddAA(_2A(lhs), _2A(rhs));
  }
  return _AddAA(lhs, rhs);
}

ArrayRef MulSP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);
  if (_LAZY_AB) {
    return _MulAP(_2A(lhs), rhs);
  }
  return _MulAP(lhs, rhs);
}

ArrayRef MulSS::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);
  if (_LAZY_AB) {
    return _MulAA(_2A(lhs), _2A(rhs));
  }
  return _MulAA(lhs, rhs);
}

ArrayRef MatMulSP::proc(KernelEvalContext* ctx, const ArrayRef& A,
                        const ArrayRef& B, int64_t M, int64_t N,
                        int64_t K) const {
  PPU_TRACE_OP(this, A, B);
  if (_LAZY_AB) {
    return _MatMulAP(_2A(A), B, M, N, K);
  }
  return _MatMulAP(A, B, M, N, K);
}

ArrayRef MatMulSS::proc(KernelEvalContext* ctx, const ArrayRef& A,
                        const ArrayRef& B, int64_t M, int64_t N,
                        int64_t K) const {
  PPU_TRACE_OP(this, A, B);
  if (_LAZY_AB) {
    return _MatMulAA(_2A(A), _2A(B), M, N, K);
  }
  return _MatMulAA(A, B, M, N, K);
}

ArrayRef LogRegAll::proc(KernelEvalContext* ctx, const ArrayRef& A,
                         const ArrayRef& B, const ArrayRef& C, int64_t M,
                         int64_t N) const{
  PPU_TRACE_OP(this, A, B, C);
  return _LogReg(A, B, C, M, N);
}

ArrayRef AndSP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);
  if (_LAZY_AB) {
    return _AndBP(_2B(lhs), rhs);
  }
  return _B2A(_AndBP(_A2B(lhs), rhs));
}

ArrayRef AndSS::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);
  if (_LAZY_AB) {
    return _AndBB(_2B(lhs), _2B(rhs));
  }
  return _B2A(_AndBB(_A2B(lhs), _A2B(rhs)));
}

ArrayRef XorSP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);
  if (_LAZY_AB) {
    return _XorBP(_2B(lhs), rhs);
  }
  return _B2A(_XorBP(_A2B(lhs), rhs));
}

ArrayRef XorSS::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  PPU_TRACE_OP(this, lhs, rhs);
  if (_LAZY_AB) {
    return _XorBB(_2B(lhs), _2B(rhs));
  }
  return _B2A(_XorBB(_A2B(lhs), _A2B(rhs)));
}

ArrayRef EqzS::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  PPU_TRACE_OP(this, in);
  //
  PPU_THROW("TODO");
}

ArrayRef LShiftS::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  // TODO(jint) left shift could be done in arithmetic share space.
  PPU_TRACE_OP(this, in, bits);
  if (_LAZY_AB) {
    return _LShiftB(_2B(in), bits);
  }
  return _B2A(_LShiftB(_A2B(in), bits));
}

ArrayRef RShiftS::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  PPU_TRACE_OP(this, in, bits);
  if (_LAZY_AB) {
    return _RShiftB(_2B(in), bits);
  }
  return _B2A(_RShiftB(_A2B(in), bits));
}

ArrayRef ARShiftS::proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t bits) const {
  PPU_TRACE_OP(this, in, bits);
  if (_LAZY_AB) {
    return _ARShiftB(_2B(in), bits);
  }
  return _B2A(_ARShiftB(_A2B(in), bits));
}

ArrayRef TruncPrS::proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t bits) const {
  PPU_TRACE_OP(this, in, bits);
  if (_LAZY_AB) {
    return _TruncPrA(_2A(in), bits);
  }
  return _TruncPrA(in, bits);
}

ArrayRef ReverseBitsS::proc(KernelEvalContext* ctx, const ArrayRef& in,
                            size_t start, size_t end) const {
  PPU_TRACE_OP(this, in, start, end);
  if (_LAZY_AB) {
    return _ReverseBitsB(_2B(in), start, end);
  }
  return _B2A(_ReverseBitsB(_A2B(in), start, end));
}

ArrayRef MsbS::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  PPU_TRACE_OP(this, in);
  if (ctx->caller()->hasKernel("MsbA")) {
    if (_LAZY_AB) {
      if (in.eltype().isa<BShare>()) {
        return _RShiftB(in, in.elsize() * 8 - 1);
      } else {
        // fast path, directly apply msb in AShare, result a BShare.
        return _MsbA(in);
      }
    } else {
      // Do it in AShare domain, and convert back to AShare.
      return _B2A(_MsbA(in));
    }
  } else {
    if (_LAZY_AB) {
      return _RShiftB(_2B(in), in.elsize() * 8 - 1);
    }
    return _B2A(_RShiftB(_A2B(in), in.elsize() * 8 - 1));
  }
}

}  // namespace ppu::mpc
