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
#include <vector>

#include "fmt/format.h"
#include "fmt/ostream.h"

#include "ppu/utils/exception.h"

namespace ppu {

// A buffer is an RAII object which represent an in memory buffer.
class Buffer final {
  void* ptr_{nullptr};
  int64_t size_{0};

 public:
  // default constructor, create an empty buffer.
  Buffer() = default;
  explicit Buffer(int64_t size) : size_(size) {
    PPU_ENFORCE(size >= 0);
    // colloc init with zeros.
    ptr_ = calloc(size, 1);
  }

  Buffer(const void* ptr, int64_t size, bool take_ownership = false) {
    PPU_ENFORCE(size >= 0);
    size_ = size;
    if (take_ownership) {
      ptr_ = const_cast<void*>(ptr);
    } else {
      ptr_ = malloc(size);
      std::memcpy(ptr_, ptr, size);
    }
  }

  ~Buffer() { free(ptr_); }

  Buffer(const Buffer& other) { *this = other; }
  Buffer& operator=(const Buffer& other) {
    if (size_ < other.size_) {
      ptr_ = malloc(other.size_);
      PPU_ENFORCE(ptr_ != nullptr, "alloc memory of {} size failed",
                  other.size_);
    }
    size_ = other.size_;
    // Copy DataBuffers
    std::memcpy(ptr_, other.ptr_, size_);
    return *this;
  }

  Buffer(Buffer&& other) noexcept { *this = std::move(other); };
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      std::swap(size_, other.size_);
      std::swap(ptr_, other.ptr_);
    }
    return *this;
  }

  template <typename T = void>
  T* data() {
    return static_cast<T*>(ptr_);
  }
  template <typename T = void>
  T const* data() const {
    return reinterpret_cast<T const*>(ptr_);
  }

  // return size of the buffer, in bytes.
  int64_t size() const { return size_; }

  bool operator==(const Buffer& other) const {
    if (size_ != other.size_) {
      return false;
    }

    return (std::memcmp(ptr_, other.ptr_, size_) == 0);
  }

  void resize(int64_t new_size) {
    PPU_ENFORCE(new_size > 0);
    if (new_size == size_) {
      return;
    }
    if (new_size != 0) {
      if (ptr_) {
        ptr_ = realloc(ptr_, new_size);
      } else {
        ptr_ = malloc(new_size);
      }
    }
    size_ = new_size;
  }

  void* release() {
    void* tmp = ptr_;
    ptr_ = nullptr;
    size_ = 0;
    return tmp;
  }
};

// TODO(jint) drop this two API.
std::shared_ptr<Buffer> makeBuffer(int64_t size);
// make a buffer and copy content to it.
std::shared_ptr<Buffer> makeBuffer(void const* ptr, int64_t size);
// Convert a buffer to shared_ptr
std::shared_ptr<Buffer> makeBuffer(Buffer&& buf);

std::ostream& operator<<(std::ostream& out, const Buffer& v);

}  // namespace ppu
