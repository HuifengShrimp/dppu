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


#include "ppu/hal/debug.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

#include "ppu/hal/test_util.h"
#include "ppu/hal/type_cast.h"  // For reveal

namespace ppu::hal {

void dbg_print(HalContext* ctx, const Value& v) {
  if (v.is_public()) {
    std::stringstream ss;
    if (v.is_fxp()) {
      auto pt = test::dump_public_as<float>(ctx, v);
      ss << pt << std::endl;
    } else if (v.is_int()) {
      auto pt = test::dump_public_as<int64_t>(ctx, v);
      ss << pt << std::endl;
    } else {
      PPU_THROW("unsupport dtype={}", v.dtype());
    }
    if ((ctx->lctx() && ctx->lctx()->Rank() == 0) || ctx->lctx() == nullptr) {
      SPDLOG_INFO(ss.str());
    }
  } else if (v.is_secret()) {
    dbg_print(ctx, reveal(ctx, v));
  } else {
    PPU_THROW("unsupport vtype={}", v.vtype());
  }
}

}  // namespace ppu::hal
