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


#include "ppu/utils/serialize.h"

#include "ppu/utils/serializable.pb.h"

namespace ppu::utils {

Buffer SerializeArrayOfBuffers(const std::vector<Buffer>& bufs) {
  ArrayOfBuffer proto;
  for (const auto& b : bufs) {
    proto.add_bufs(b.data<char>(), b.size());
  }
  auto serialized = proto.SerializeAsString();
  return {serialized.c_str(), static_cast<int64_t>(serialized.size())};
}

std::vector<Buffer> DeserializeArrayOfBuffers(const Buffer& buf) {
  ArrayOfBuffer proto;
  std::vector<Buffer> bufs;
  proto.ParseFromArray(buf.data(), buf.size());
  for (const auto& b : proto.bufs()) {
    bufs.emplace_back(b.c_str(), b.size());
  }
  return bufs;
}

Buffer SerializeInt128(int128_t v) {
  Int128Proto proto;
  auto parts = DecomposeInt128(v);
  proto.set_hi(parts.first);
  proto.set_lo(parts.second);

  auto s = proto.SerializeAsString();
  return {s.c_str(), static_cast<int64_t>(s.size())};
}

int128_t DeserializeInt128(const Buffer& buf) {
  Int128Proto proto;
  proto.ParseFromArray(buf.data(), buf.size());
  return MakeInt128(proto.hi(), proto.lo());
}

Buffer SerializeUint128(uint128_t v) {
  Uint128Proto proto;
  auto parts = DecomposeUInt128(v);
  proto.set_hi(parts.first);
  proto.set_lo(parts.second);

  auto s = proto.SerializeAsString();
  return {s.c_str(), static_cast<int64_t>(s.size())};
}

uint128_t DeserializeUint128(const Buffer& buf) {
  Uint128Proto proto;
  proto.ParseFromArray(buf.data(), buf.size());
  return MakeUint128(proto.hi(), proto.lo());
}

}  // namespace ppu::utils
