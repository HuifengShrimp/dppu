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


#include "ppu/core/array_ref.h"

#include <numeric>

#include "ppu/core/buffer.h"
#include "ppu/core/shape_util.h"

namespace ppu {

std::vector<int64_t> compactStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size());
  const size_t size = shape.size();
  for (size_t dim = size; dim > 0; dim--) {
    strides[dim - 1] = dim == size ? 1 : strides[dim] * shape[dim];
  }
  // This follows the xtensor style, @jint I think both 0 or `default value`
  // should be OK.
  for (size_t dim = 0; dim < size; dim++) {
    if (shape[dim] == 1) {
      strides[dim] = 0;
    }
  }
  return strides;
}

bool ArrayRef::isCompact() const { return stride_ == 1; }

std::shared_ptr<Buffer> ArrayRef::getOrCreateCompactBuf() const {
  if (isCompact()) {
    return buf();
  }
  return clone().buf();
}

ArrayRef ArrayRef::clone() const {
  ArrayRef res(eltype(), numel());

  for (int64_t idx = 0; idx < numel(); idx++) {
    const auto* frm = &at(idx);
    auto* dst = &res.at(idx);

    std::memcpy(dst, frm, elsize());
  }

  return res;
}

ArrayRef ArrayRef::as(const Type& new_ty, bool force) const {
  if (!force) {
    PPU_ENFORCE(elsize() == new_ty.size(),
                "viewed type={} not equal to origin type={}", new_ty, eltype());
  }

  return {buf(), new_ty, numel(), stride(), offset()};
}

ArrayRef ArrayRef::slice(int64_t start, int64_t stop, int64_t stride) {
  // From
  // https://numpy.org/doc/stable/user/basics.indexing.html#slicing-and-striding
  //
  // The basic slice syntax is i:j:k where i is the starting index, j is the
  // stopping index, and k is the step (). This selects the m elements (in the
  // corresponding dimension) with index values i, i + k, …, i + (m - 1) k
  // where and q and r are the quotient and remainder obtained by dividing j -
  // i by k: j - i = q k + r, so that i + (m - 1) k < j.
  PPU_ENFORCE(start < numel_, "start={}, numel_={}", start, numel_);

  const int64_t q = (stop - start) / stride;
  const int64_t r = (stop - start) % stride;
  const int64_t m = q + static_cast<int64_t>(r != 0);

  const int64_t n_stride = stride_ * stride;
  const int64_t n_offset = offset_ + start * stride_ * elsize();

  return {buf(), eltype_, m, n_stride, n_offset};
}

bool ArrayRef::operator==(const ArrayRef& other) const {
  if (numel() != other.numel() || eltype() != other.eltype()) {
    return false;
  }

  for (int64_t idx = 0; idx < numel(); idx++) {
    const auto* a = &at(idx);
    const auto* b = &other.at(idx);

    if (memcmp(a, b, elsize()) != 0) {
      return false;
    }
  }

  return true;
}

NdArrayRef NdArrayRef::as(const Type& new_ty, bool force) const {
  if (!force) {
    PPU_ENFORCE(elsize() == new_ty.size(),
                "viewed type={} not equal to origin type={}", new_ty, eltype());
    return {buf(), new_ty, shape(), strides(), offset()};
  }
  // Force view, we need to adjust strides
  auto distance = ((strides().empty() ? 1 : strides().back()) * elsize());
  PPU_ENFORCE(distance % new_ty.size() == 0);

  std::vector<int64_t> new_strides = strides();
  std::transform(new_strides.begin(), new_strides.end(), new_strides.begin(),
                 [&](int64_t s) { return (elsize() * s) / new_ty.size(); });

  return {buf(), new_ty, shape(), new_strides, offset()};
}

int64_t NdArrayRef::numel() const {
  return ppu::numel(absl::MakeSpan(shape()));
}

bool NdArrayRef::isCompact() const {
  return compactStrides(shape()) == strides();
}

NdArrayRef NdArrayRef::clone() const {
  NdArrayRef res(eltype(), shape());

  std::vector<int64_t> indices(shape().size(), 0);

  do {
    const auto* frm = &at(indices);
    auto* dst = &res.at(indices);

    std::memcpy(dst, frm, elsize());
  } while (bumpIndices<int64_t>(shape(), absl::MakeSpan(indices)));

  return res;
}

}  // namespace ppu
