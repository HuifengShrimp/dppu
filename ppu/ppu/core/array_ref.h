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

#include <memory>
#include <vector>

#include "fmt/format.h"
#include "fmt/ostream.h"
#include "spdlog/spdlog.h"

#include "ppu/core/buffer.h"
#include "ppu/core/shape_util.h"
#include "ppu/core/type.h"
#include "ppu/core/vectorize.h"
#include "ppu/utils/exception.h"

namespace ppu {

// An array of objects on a memory buffer.
class ArrayRef {
  std::shared_ptr<Buffer> buf_{nullptr};

  // element type.
  Type eltype_{};

  // number of elements.
  int64_t numel_{0};

  // element stride, in number of elements.
  int64_t stride_{0};

  // start offset from the mem buffer, in bytes.
  int64_t offset_{0};

 public:
  ArrayRef() = default;

  ArrayRef(std::shared_ptr<Buffer> buf, Type eltype, int64_t numel,
           int64_t stride, int64_t offset)
      : buf_(std::move(buf)),
        eltype_(std::move(eltype)),
        numel_(numel),
        stride_(stride),
        offset_(offset) {
    // sanity check.
    PPU_ENFORCE(offset + stride * numel <= buf_->size());
  }

  ArrayRef(std::shared_ptr<Buffer> buf, Type eltype)
      : buf_(std::move(buf)), eltype_(std::move(eltype)) {
    PPU_ENFORCE(buf_->size() % eltype_.size() == 0);

    numel_ = buf_->size() / eltype_.size();
    stride_ = 1;
    offset_ = 0;
  }

  // Create a new buffer of uninitialized elements and ref to it.
  ArrayRef(Type eltype, size_t numel)
      : ArrayRef(makeBuffer(numel * eltype.size()),
                 eltype,  // eltype
                 numel,   // numel
                 1,       // stride,
                 0        // offset
        ) {}

  // Return total number of elements.
  int64_t numel() const { return numel_; }

  size_t elsize() const { return eltype_.size(); }

  int64_t stride() const { return stride_; }

  int64_t offset() const { return offset_; }

  const Type& eltype() const { return eltype_; }

  Type& eltype() { return eltype_; }

  // https://numpy.org/doc/stable/user/basics.indexing.html#slicing-and-striding
  ArrayRef slice(int64_t start, int64_t stop, int64_t stride = 1);

  std::shared_ptr<Buffer> buf() const { return buf_; }

  // Create a new buffer if current underline buffer is not compact.
  // while compact means offset > 0 or stride != elsize.
  // Or return the underline buffer.
  std::shared_ptr<Buffer> getOrCreateCompactBuf() const;

  bool isCompact() const;

  ArrayRef clone() const;

  // View this array ref as another type.
  // @param force, true if ignore the type check.
  ArrayRef as(const Type& new_ty, bool force = false) const;

  // Test two array are bitwise equal
  bool operator==(const ArrayRef& other) const;

  // Get data pointer
  void* data() {
    return reinterpret_cast<void*>(buf_->data<std::byte>() + offset_);
  }
  void const* data() const {
    return reinterpret_cast<void const*>(buf_->data<std::byte>() + offset_);
  }

  // Get element.
  template <typename T = std::byte>
  T& at(int64_t pos) {
    return *reinterpret_cast<T*>(static_cast<std::byte*>(data()) +
                                 stride_ * pos * elsize());
  }
  template <typename T = std::byte>
  const T& at(int64_t pos) const {
    return *reinterpret_cast<const T*>(static_cast<const std::byte*>(data()) +
                                       stride_ * pos * elsize());
  }
};

template <>
struct SimdTrait<ArrayRef> {
  using PackInfo = std::vector<size_t>;

  template <typename InputIt>
  static ArrayRef pack(InputIt first, InputIt last, PackInfo& pi) {
    PPU_ENFORCE(first != last);

    size_t total_numel = 0;
    const Type ty = first->eltype();
    for (auto itr = first; itr != last; ++itr) {
      PPU_ENFORCE(itr->eltype() == ty, "type mismatch {} != {}", itr->eltype(),
                  ty);
      total_numel += itr->numel();
    }
    ArrayRef result(first->eltype(), total_numel);
    size_t res_index = 0;
    for (; first != last; ++first) {
      for (int64_t index = 0; index < first->numel(); index++) {
        memcpy(&result.at(res_index + index), &first->at(index), ty.size());
      }
      pi.push_back(first->numel());
      res_index += first->numel();
    }
    return result;
  }

  template <typename OutputIt>
  static OutputIt unpack(const ArrayRef& v, OutputIt result,
                         const PackInfo& pi) {
    const int64_t total_num =
        std::accumulate(pi.begin(), pi.end(), 0, std::plus<>());

    PPU_ENFORCE(v.numel() == total_num, "split number mismatch {} != {}",
                v.numel(), total_num);

    int64_t offset = 0;
    for (const auto& sz : pi) {
      *result++ = ArrayRef(v.buf(), v.eltype(), sz, v.stride(), offset);
      offset += sz * v.elsize();
    }

    return result;
  }
};

// utilities
std::vector<int64_t> compactStrides(const std::vector<int64_t>& shape);

// N-dimensional array reference.
//
// About: 0-dimension processing.
// We use numpy/xtensor 0-dimension setting.
class NdArrayRef {
  std::shared_ptr<Buffer> buf_{nullptr};

  Type eltype_{};

  // the shape.
  std::vector<int64_t> shape_{};

  // the strides, in number of elements.
  std::vector<int64_t> strides_{};

  // start offset from the mem buffer.
  int64_t offset_{0};

 public:
  NdArrayRef() = default;

  // full constructor
  NdArrayRef(std::shared_ptr<Buffer> buf, Type eltype,
             std::vector<int64_t> shape, std::vector<int64_t> strides,
             int64_t offset)
      : buf_(std::move(buf)),
        eltype_(std::move(eltype)),
        shape_(std::move(shape)),
        strides_(std::move(strides)),
        offset_(offset) {
    PPU_ENFORCE(ppu::numel(absl::MakeSpan(shape_)) *
                    static_cast<int64_t>(eltype_.size()) <=
                buf_->size());

    // PPU_ENFORCE(
    //     isCompact(),
    //     "Only compact strides is supported for now, got=({}), elsize={}",
    //     fmt::join(strides_, "x"), elsize());
  }

  // convenient constructor to accept shape/strides from xtensor.
  template <typename ShapeT, typename StridesT>
  NdArrayRef(std::shared_ptr<Buffer> buf, Type eltype, ShapeT&& shape,
             StridesT&& strides, int64_t offset)
      : NdArrayRef(std::move(buf), std::move(eltype),
                   {shape.begin(), shape.end()},
                   {strides.begin(), strides.end()}, offset) {}

  // constructor, view buf as a compact buffer with given shape.
  NdArrayRef(std::shared_ptr<Buffer> buf, Type eltype,
             std::vector<int64_t> shape)
      : NdArrayRef(std::move(buf),         // buf
                   eltype,                 // eltype
                   shape,                  // shape
                   compactStrides(shape),  // strides
                   0                       // offset
        ) {}

  // constructor, create a new buffer of elements and ref to it.
  NdArrayRef(Type eltype, std::vector<int64_t> shape)
      : NdArrayRef(makeBuffer(ppu::numel(shape) * eltype.size()),  // buf
                   eltype,                                         // eltype
                   shape,                                          // shape
                   compactStrides(shape),                          // strides
                   0                                               // offset
        ) {}

  // copy and move constructable, using referential semantic.
  NdArrayRef(const NdArrayRef& other) = default;
  NdArrayRef(NdArrayRef&& other) = default;
  NdArrayRef& operator=(const NdArrayRef& other) = default;
  NdArrayRef& operator=(NdArrayRef&& other) = default;

  // Returns the rank of the nd array.
  size_t rank() const { return shape_.size(); }

  // Return the size of the given dimension.
  size_t dim(size_t idx) const {
    PPU_ENFORCE(idx < rank());
    return shape_[idx];
  }

  // Return total number of elements.
  int64_t numel() const;

  size_t elsize() const { return eltype_.size(); }

  std::vector<int64_t> const& strides() const { return strides_; }

  std::vector<int64_t> const& shape() const { return shape_; }

  int64_t offset() const { return offset_; }

  const Type& eltype() const { return eltype_; }

  Type& eltype() { return eltype_; }

  std::shared_ptr<Buffer> buf() const { return buf_; }

  bool isCompact() const;

  NdArrayRef clone() const;

  // View this array ref as another type.
  // @param force, true if ignore the type check.
  NdArrayRef as(const Type& new_ty, bool force = false) const;

  // Get data pointer
  void* data() { return buf_->data<std::byte>() + offset_; }
  const void* data() const { return buf_->data<std::byte>() + offset_; }

  // Get element.
  template <typename T = std::byte>
  T& at(const std::vector<int64_t>& pos) {
    auto fi = flattenIndex(absl::MakeSpan(pos), absl::MakeSpan(shape_),
                           absl::MakeSpan(strides_));
    return *reinterpret_cast<T*>(static_cast<std::byte*>(data()) +
                                 elsize() * fi);
  }

  template <typename T = std::byte>
  const T& at(const std::vector<int64_t>& pos) const {
    auto fi = flattenIndex(absl::MakeSpan(pos), absl::MakeSpan(shape_),
                           absl::MakeSpan(strides_));
    return *reinterpret_cast<const T*>(static_cast<const std::byte*>(data()) +
                                       elsize() * fi);
  }
};

inline std::ostream& operator<<(std::ostream& out, const ArrayRef& v) {
  // TODO(jint) type printer.
  out << fmt::format("ArrayRef<{}x{}>", v.numel(), v.elsize());
  return out;
}

std::ostream& operator<<(std::ostream& out, const NdArrayRef& v);

}  // namespace ppu
