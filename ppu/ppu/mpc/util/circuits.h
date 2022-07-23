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

#include <functional>

#include "absl/numeric/bits.h"

#include "ppu/core/vectorize.h"

namespace ppu::mpc {
namespace details {

// TODO(jint): general utility
template <class T>
struct dependent_false : std::false_type {};

}  // namespace details

// multi-bit circuit basic block
//
// with LShift/RShift, we can build any constant input, plus xor/and, we
// can build any complicate circuits.
template <typename T>
struct CircuitBasicBlock {
  // multi-bit xor. i.e. 0010 xor 1010 -> 1000
  using Xor = std::function<T(T const&, T const&)>;

  // multi-bit and. i.e. 0010 xor 1010 -> 0010
  using And = std::function<T(T const&, T const&)>;

  // (logic) left shift
  using LShift = std::function<T(T const&, size_t)>;

  // (logic) right shift
  using RShift = std::function<T(T const&, size_t)>;

  size_t num_bits;

  Xor _xor;
  And _and;
  LShift lshift;
  RShift rshift;
};

template <typename T>
CircuitBasicBlock<T> DefaultCircuitBasicBlock() {
  if constexpr (std::is_integral_v<T>) {
    CircuitBasicBlock<T> cbb;
    cbb.num_bits = sizeof(T) * 8;
    cbb._xor = [](T const& lhs, T const& rhs) -> T { return lhs ^ rhs; };
    cbb._and = [](T const& lhs, T const& rhs) -> T { return lhs & rhs; };
    cbb.lshift = [](T const& x, size_t bits) -> T { return x << bits; };
    cbb.rshift = [](T const& x, size_t bits) -> T { return x >> bits; };
    return cbb;
  } else {
    static_assert(details::dependent_false<T>::value,
                  "Not implemented for circuit basic block.");
  }
}

/// Reference:
///  PPA (Parallel Prefix Adder)
///  http://users.encs.concordia.ca/~asim/COEN_6501/Lecture_Notes/Parallel%20prefix%20adders%20presentation.pdf
///
/// Why KoggleStone:
///  - easy to implement.
///
/// Analysis:
///  AND Gates: 1 + log(k) (additional 1 for `g` generation)
template <typename T>
T KoggleStoneAdder(
    const T& lhs, const T& rhs,
    const CircuitBasicBlock<T> bb = DefaultCircuitBasicBlock<T>()) {
  // Generate p & g.
  T p = bb._xor(lhs, rhs);
  T g = bb._and(lhs, rhs);

  // Parallel Prefix Graph: Koggle Stone.
  // We write prefix element as P, G, where:
  //  (G0, P0) = (g0, p0)
  //  (Gi, Pi) = (gi, pi) o (Gi-1, Pi-1)
  // The `o` here is:
  //  (G0, P0) o (G1, P1) = (G0 ^ (P0 & G1), P0 & P1)
  //
  // We can perform AND vectorization for above two AND:
  T G0 = g;
  T P0 = p;
  for (size_t idx = 0; idx < absl::bit_width(bb.num_bits) - 1; ++idx) {
    const size_t offset = 1UL << idx;

    // G1 = G << offset
    // P1 = P << offset
    T G1 = bb.lshift(G0, offset);
    T P1 = bb.lshift(P0, offset);

    // In the Kogge-Stone graph, we need to keep the lowest |offset| P, G
    // unmodified.
    //
    //// P0 = P0 & P1
    //// G0 = G0 ^ (P0 & G1)

    if constexpr (hasSimdTrait<T>::value) {
      std::vector<T> res = vectorize({P0, P0}, {P1, G1}, bb._and);
      P0 = std::move(res[0]);
      G1 = std::move(res[1]);
    } else {
      G1 = bb._and(P0, G1);
      P0 = bb._and(P0, P1);
    }

    G0 = bb._xor(G1, G0);
  }

  // Carry = G0
  // C = Carry << 1;
  // out = C ^ P
  T C = bb.lshift(G0, 1);
  return bb._xor(p, C);
}

}  // namespace ppu::mpc
