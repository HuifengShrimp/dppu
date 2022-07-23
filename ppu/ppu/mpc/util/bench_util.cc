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


#include "ppu/mpc/util/bench_util.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xrandom.hpp"

#include "ppu/core/array_ref_util.h"
#include "ppu/core/type_util.h"
#include "ppu/mpc/util/communicator.h"
#include "ppu/mpc/util/simulate.h"

namespace ppu::mpc::bench {

ArrayRef RandP(FieldType field, size_t size, std::mt19937::result_type seed,
               int min, int max) {
  return DISPATCH_ALL_FIELDS(field, "test::RandP", [&]() {
    static_assert(std::is_integral_v<ring2k_t>);

    std::mt19937 engine(seed);

    // FIXME(jint)
    auto ty = makeType<Ring2kPublTy>(field);
    return make_array(xt::random::randint<ring2k_t>({size}, min, max, engine),
                      ty);
  });
}

// Evaluate a function for a given world size.
void Eval(size_t world_size,
          std::function<void(std::shared_ptr<link::Context> lctx)> proc) {
  util::simulate(world_size, proc);
}

}  // namespace ppu::mpc::bench