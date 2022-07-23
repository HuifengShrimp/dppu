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


#include "ppu/core/type_util.h"

#include "absl/strings/str_join.h"

namespace ppu {

//////////////////////////////////////////////////////////////
// Visibility related
//////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, const Visibility& vtype) {
  switch (vtype) {
    case VIS_PUBLIC:
      os << "P";
      break;
    case VIS_SECRET:
      os << "S";
      break;
    default:
      os << "Unknown";
  }
  return os;
}

//////////////////////////////////////////////////////////////
// Datatype related
//////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, const DataType& dtype) {
  os << DataType_Name(dtype);
  return os;
}

//////////////////////////////////////////////////////////////
// Plaintext types.
//////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, const PtType& pt_type) {
  os << PtType_Name(pt_type);
  return os;
}

size_t SizeOf(PtType ptt) {
#define CASE(Name, Type, _) \
  case (Name):              \
    return sizeof(Type);
  switch (ptt) {
    FOREACH_PT_TYPES(CASE);
    default:
      PPU_THROW("unknown size of {}", ptt);
  }
#undef CASE
}

//////////////////////////////////////////////////////////////
// ProtocolKind utils
//////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, ProtocolKind protocol) {
  os << ProtocolKind_Name(protocol);
  return os;
}

//////////////////////////////////////////////////////////////
// Field 2k types, TODO(jint) support Zq
//////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, FieldType field) {
  os << FieldType_Name(field);
  return os;
}

PtType GetStorageType(FieldType field) {
#define CASE(Name, StorageType) \
  case FieldType::Name:         \
    return StorageType;         \
    break;
  switch (field) {
    FIELD_TO_STORAGE_MAP(CASE)
    default:
      PPU_THROW("unknown storage type of {}", field);
  }
#undef CASE
}

FieldType PtTypeToField(PtType pt_type) {
#define CASE(FIELD_NAME, PT_NAME) \
  case PT_NAME:                   \
    return FieldType::FIELD_NAME;

  switch (pt_type) {
    FIELD_TO_STORAGE_MAP(CASE)
    default:
      PPU_THROW("can not convert pt_type={} to field", pt_type);
  }
#undef CASE
}

}  // namespace ppu
