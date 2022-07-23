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

#include <numeric>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/types/span.h"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xexpression.hpp"

#include "ppu/core/buffer.h"
#include "ppu/link/link.h"
#include "ppu/mpc/object.h"
#include "ppu/utils/exception.h"
#include "ppu/utils/int128.h"

// This module defines the protocol comm pattern used for all
// protocols.

namespace ppu::mpc {
namespace detail {

template <class E>
Buffer SerializeXtensor(const xt::xexpression<E>& x) {
  using storage_t = typename E::value_type;
  auto&& xx = xt::eval(x.derived_cast());
  return {reinterpret_cast<char const*>(xx.data()),
          static_cast<int64_t>(xx.size() * sizeof(storage_t))};
}

template <typename T, typename ST,
          std::enable_if_t<std::is_trivially_copyable_v<T>, int> = 0>
xt::xarray<T> BuildXtensor(const ST& shape, const Buffer& buf) {
  const int64_t numel =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());

  PPU_ENFORCE(buf.size() == numel * (int64_t)sizeof(T),
              "expected buf=({}x{}), got={}", numel, sizeof(T), buf.size());

  // TLDR:
  // when T = __int128, gcc will try to optimize assignment with XMM
  // registers, which expects 16 bit alignment.
  //
  // for example: [here](https://godbolt.org/z/K43Meo7Yz)
  // ```cpp
  // int proc(__int128*x, __int128*y) {
  //   for (int i = 0; i < 10; i ++) {
  //     y[i] = x[i];
  //   }
  // }
  // ```
  //
  // with `-O1` or higher, will generate
  // ```assembly
  // movdqa xmm0,XMMWORD PTR [rdi+rax*1]
  // movaps XMMWORD PTR [rsi+rax*1],xmm0
  // ```
  //
  // while MOVDQA expects 16bit aligment (for __int128), or it will raise a
  // SEGMENT fault.
  //
  // see
  // [MOVDQA](https://mudongliang.github.io/x86/html/file_module_x86_id_183.html)
  // for more details.
  //
  // Solution: we copy str to an aligned buffer.
  static_assert(alignof(int128_t) == 16);

  // C++17 guarantees alignment of allocated memory.
  auto ptr = std::make_unique<T[]>(numel);

  std::memcpy(ptr.get(), buf.data(), buf.size());

  // copy shape, make xt::adapt happy.
  std::vector<size_t> _shape(shape.begin(), shape.end());
  return xt::adapt_smart_ptr(ptr.get(), _shape, std::move(ptr));
}

}  // namespace detail

enum class ReduceOp {
  INVALID = 0,
  ADD = 1,
  XOR = 2,
};

// link::algorithms does not make assumption on data types, (it works on
// buffer), which means it's hard to write algorithms which depends on data
// arithmetics, such like reduce/allreduce.
//
// In mpc module, we have concrete data type definition, so we can fill this
// gap.
class Communicator : public State {
 public:
  static constexpr char kName[] = "Communicator";

  struct Stats {
    //
    size_t latency = 0;

    // Number of communication in bytes.
    //
    // For collective MPI algorithms only.
    // TODO(jint) add formal definition for asymmetric algorithms.
    size_t comm = 0;

    Stats operator-(const Stats& rhs) const {
      return {latency - rhs.latency, comm - rhs.comm};
    }
  };

  mutable Stats stats_;

  const std::shared_ptr<link::Context> lctx_;

  explicit Communicator(std::shared_ptr<link::Context> lctx)
      : lctx_(std::move(lctx)) {}

  Stats getStats() const { return stats_; }

  size_t getWorldSize() const { return lctx_->WorldSize(); }

  size_t getRank() const { return lctx_->Rank(); }

  // All reduce
  ArrayRef allReduce(ReduceOp op, const ArrayRef& in, std::string_view tag);
  template <typename E, typename T = typename E::value_type>
  xt::xarray<T> allReduce(ReduceOp op, const xt::xexpression<E>& in,
                          std::string_view tag);

  // Reduce
  ArrayRef reduce(ReduceOp op, const ArrayRef& in, size_t root,
                  std::string_view tag);
  template <typename E, typename T = typename E::value_type>
  xt::xarray<T> reduce(ReduceOp op, const xt::xexpression<E>& in, Rank root,
                       std::string_view tag);

  // Rotate
  // TODO: implement me.
  // ArrayRef rotate(const ArrayRef& in, std::string_view tag);
  template <typename E, typename T = typename E::value_type>
  xt::xarray<T> rotate(const xt::xexpression<E>& in, std::string_view tag);
};

template <typename E, typename T>
xt::xarray<T> Communicator::allReduce(ReduceOp op, const xt::xexpression<E>& in,
                                      std::string_view tag) {
  const std::vector<Buffer> all_buf =
      link::AllGather(lctx_, detail::SerializeXtensor(in), tag);

  const auto& in_x = in.derived_cast();

  xt::xarray<T> res = xt::zeros_like(in);
  for (const auto& buf : all_buf) {
    if (op == ReduceOp::ADD) {
      res += detail::BuildXtensor<T>(in_x.shape(), buf);
    } else if (op == ReduceOp::XOR) {
      res ^= detail::BuildXtensor<T>(in_x.shape(), buf);
    } else {
      PPU_THROW("unsupported reduce op={}", static_cast<int>(op));
    }
  }

  // TODO: count comm & latency

  return res;
}

template <typename E, typename T>
xt::xarray<T> Communicator::reduce(ReduceOp op, const xt::xexpression<E>& in,
                                   Rank root, std::string_view tag) {
  PPU_ENFORCE(root < lctx_->WorldSize());

  const auto& in_x = in.derived_cast();

  const std::vector<Buffer> all_buf =
      link::Gather(lctx_, detail::SerializeXtensor(in), root, tag);

  xt::xarray<T> res = xt::zeros_like(in);
  for (const auto& buf : all_buf) {
    if (op == ReduceOp::ADD) {
      res += detail::BuildXtensor<T>(in_x.shape(), buf);
    } else if (op == ReduceOp::XOR) {
      res ^= detail::BuildXtensor<T>(in_x.shape(), buf);
    } else {
      PPU_THROW("unsupported reduce op={}", static_cast<int>(op));
    }
  }

  // TODO: count comm & latency
  return res;
}

template <typename E, typename T>
xt::xarray<T> Communicator::rotate(const xt::xexpression<E>& in,
                                   std::string_view tag) {
  const auto& in_x = in.derived_cast();

  lctx_->SendAsync(lctx_->PrevRank(), detail::SerializeXtensor(in), tag);

  // TODO(jint) drop this copy.
  auto buf = lctx_->Recv(lctx_->NextRank(), tag);
  return detail::BuildXtensor<T>(in_x.shape(), buf);
}

}  // namespace ppu::mpc
