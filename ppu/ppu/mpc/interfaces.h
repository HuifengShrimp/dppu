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

#define METHOD0(NAME, R0) \
  R0 NAME() { return obj_->call(#NAME); }

#define METHOD1(NAME, R0, P0) \
  R0 NAME(const P0& p0) { return obj_->call(#NAME, p0); }

#define METHOD2(NAME, R0, P0, P1) \
  R0 NAME(const P0& p0, const P1& p1) { return obj_->call(#NAME, p0, p1); }

#define METHOD3(NAME, R0, P0, P1, P2)                 \
  R0 NAME(const P0& p0, const P1& p1, const P2& p2) { \
    return obj_->call(#NAME, p0, p1, p2);             \
  }

#define METHOD4(NAME, R0, P0, P1, P2, P3)                           \
  R0 NAME(const P0& p0, const P1& p1, const P2& p2, const P3& p3) { \
    return obj_->call(#NAME, p0, p1, p2, p3);                       \
  }

#define METHOD5(NAME, R0, P0, P1, P2, P3, P4)                     \
  R0 NAME(const P0& p0, const P1& p1, const P2& p2, const P3& p3, \
          const P4& p4) {                                         \
    return obj_->call(#NAME, p0, p1, p2, p3, p4);                 \
  }

class IArithmetic : public Interface {
 public:
  explicit IArithmetic(Object* obj) : Interface(obj) {}

  METHOD2(ZeroA, ArrayRef, FieldType, size_t);

  METHOD1(A2P, ArrayRef, ArrayRef)
  METHOD1(P2A, ArrayRef, ArrayRef)

  METHOD1(NegA, ArrayRef, ArrayRef)

  METHOD2(AddAP, ArrayRef, ArrayRef, ArrayRef)
  METHOD2(AddAA, ArrayRef, ArrayRef, ArrayRef)

  METHOD2(MulAP, ArrayRef, ArrayRef, ArrayRef)
  METHOD2(MulAA, ArrayRef, ArrayRef, ArrayRef)

  METHOD2(TruncPrA, ArrayRef, ArrayRef, size_t)

  METHOD5(MatMulAP, ArrayRef, ArrayRef, ArrayRef, size_t, size_t, size_t)
  METHOD5(MatMulAA, ArrayRef, ArrayRef, ArrayRef, size_t, size_t, size_t)

  // take msb.
  METHOD1(MsbA, ArrayRef, ArrayRef)
};

class IBoolean : public Interface {
 public:
  explicit IBoolean(Object* obj) : Interface(obj) {}

  METHOD2(ZeroB, ArrayRef, FieldType, size_t);

  METHOD1(B2P, ArrayRef, ArrayRef)
  METHOD1(P2B, ArrayRef, ArrayRef)
  METHOD1(A2B, ArrayRef, ArrayRef)
  METHOD1(B2A, ArrayRef, ArrayRef)

  METHOD2(AndBP, ArrayRef, ArrayRef, ArrayRef)
  METHOD2(AndBB, ArrayRef, ArrayRef, ArrayRef)

  METHOD2(XorBP, ArrayRef, ArrayRef, ArrayRef)
  METHOD2(XorBB, ArrayRef, ArrayRef, ArrayRef)

  METHOD2(LShiftB, ArrayRef, ArrayRef, size_t)
  METHOD2(RShiftB, ArrayRef, ArrayRef, size_t)
  METHOD2(ARShiftB, ArrayRef, ArrayRef, size_t)
  METHOD3(ReverseBitsB, ArrayRef, ArrayRef, size_t, size_t)

  // TODO(jint) should we remove this from protocol?
  METHOD2(AddBB, ArrayRef, ArrayRef, ArrayRef)
};

class IRandom : public Interface {
 public:
  explicit IRandom(Object* obj) : Interface(obj) {}

  // parties random a public together.
  METHOD2(RandP, ArrayRef, FieldType, size_t)

  // parties random a secret together.
  METHOD2(RandS, ArrayRef, FieldType, size_t)
};

// Interface of computation.
class ICompute : public Interface {
 public:
  explicit ICompute(Object* obj) : Interface(obj) {}

  // convert a public to a secret.
  METHOD1(P2S, ArrayRef, ArrayRef)
  // aka, reveal, open a secret as a public.
  METHOD1(S2P, ArrayRef, ArrayRef)

  /// Unary ops
  // calculate negate of an given value.
  // Note: in pure ring 2k, data range is from [0, 2^k), but we assume all data
  // are encoded as two's complement in 2k space, we compute negate in two's
  // complement space.
  METHOD1(NegP, ArrayRef, ArrayRef)
  METHOD1(NegS, ArrayRef, ArrayRef)

  // equal to zero.
  METHOD1(EqzP, ArrayRef, ArrayRef)
  METHOD1(EqzS, ArrayRef, ArrayRef)

  /// Mixed ops
  METHOD2(LShiftP, ArrayRef, ArrayRef, size_t)
  METHOD2(LShiftS, ArrayRef, ArrayRef, size_t)

  METHOD2(RShiftP, ArrayRef, ArrayRef, size_t)
  METHOD2(RShiftS, ArrayRef, ArrayRef, size_t)

  METHOD2(ARShiftP, ArrayRef, ArrayRef, size_t)
  METHOD2(ARShiftS, ArrayRef, ArrayRef, size_t)

  METHOD2(TruncPrS, ArrayRef, ArrayRef, size_t)
  METHOD3(ReverseBitsP, ArrayRef, ArrayRef, size_t, size_t)
  METHOD3(ReverseBitsS, ArrayRef, ArrayRef, size_t, size_t)

  METHOD2(AddPP, ArrayRef, ArrayRef, ArrayRef)
  METHOD2(AddSP, ArrayRef, ArrayRef, ArrayRef)
  METHOD2(AddSS, ArrayRef, ArrayRef, ArrayRef)

  METHOD2(MulPP, ArrayRef, ArrayRef, ArrayRef)
  METHOD2(MulSP, ArrayRef, ArrayRef, ArrayRef)
  METHOD2(MulSS, ArrayRef, ArrayRef, ArrayRef)

  METHOD2(AndPP, ArrayRef, ArrayRef, ArrayRef)
  METHOD2(AndSP, ArrayRef, ArrayRef, ArrayRef)
  METHOD2(AndSS, ArrayRef, ArrayRef, ArrayRef)

  METHOD2(XorPP, ArrayRef, ArrayRef, ArrayRef)
  METHOD2(XorSP, ArrayRef, ArrayRef, ArrayRef)
  METHOD2(XorSS, ArrayRef, ArrayRef, ArrayRef)

  METHOD5(MatMulPP, ArrayRef, ArrayRef, ArrayRef, int64_t, int64_t, int64_t)
  METHOD5(MatMulSP, ArrayRef, ArrayRef, ArrayRef, int64_t, int64_t, int64_t)
  METHOD5(MatMulSS, ArrayRef, ArrayRef, ArrayRef, int64_t, int64_t, int64_t)

  METHOD4(LogReg, ArrayRef, ArrayRef, ArrayRef, int64_t, int64_t)

  METHOD1(MsbP, ArrayRef, ArrayRef)
  METHOD1(MsbS, ArrayRef, ArrayRef)
};

}  // namespace ppu::mpc
