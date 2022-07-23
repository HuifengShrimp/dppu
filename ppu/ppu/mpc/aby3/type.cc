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


#include "ppu/mpc/aby3/type.h"

#include <mutex>

namespace ppu::mpc::aby3 {

void registerTypes() {
  static std::once_flag flag;

  std::call_once(flag, []() {
    TypeContext::getTypeContext()->addTypes<AShrTy, BShrTy>();
  });
}

}  // namespace ppu::mpc::aby3
