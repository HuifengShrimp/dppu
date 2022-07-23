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


// clang-format off
// To run the example, start two terminals:
// > bazel run //examples/cpp:simple_pphlo -- --rank=0
// > bazel run //examples/cpp:simple_pphlo -- --rank=1
// clang-format on

#include "examples/cpp/utils.h"
#include "spdlog/spdlog.h"

#include "ppu/device/colocated_io.h"

// This example demostrates the basic compute functionality of ppu vm.
void constant_add(ppu::device::Processor* proc) {
  // Write the assembly, this code simple add two numbers.
  // - `%1` is a constant public integer, with dtype int32 and value 1.
  // - `%2` is a constant public integer, with dtype int32 and value 2.
  // - `%3` is the sum of two integers.
  // - `dbg_print` print the value of `%3`
  constexpr auto code = R"PPHlo(
func @main() -> () {
    %0 = "pphlo.constant"() {value = dense<1> : tensor<i64>} : () -> tensor<!pphlo.pint>
    %1 = "pphlo.constant"() {value = dense<2> : tensor<i64>} : () -> tensor<!pphlo.pint>
    %2 = "pphlo.add"(%0, %1) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    "pphlo.dbg_print"(%2) : (tensor<!pphlo.pint>) -> ()
    return
}
)PPHlo";

  // Run it, with no input and output, (since the program does not contain IO)
  proc->run(code, {}, {});
}

// This example demostrates how to pass parameters.
void parameters(ppu::device::Processor* proc) {
  // prepare the environments of the device.
  {
    // In this example, data owner also participates the computation progress,
    // which is called "colocated mode" in ppu system.
    ppu::device::ColocatedIo io(proc);

    if (io.rank() == 0) {
      // rank-0, set a float variable 3.14 as 'x' to the device.
      float x = 3.14;
      io.setVar("x", x);
    } else {
      // rank-1, set a integer variable 2 as 'y' to the device.
      int y = 2;
      io.setVar("y", y);
    }

    // syncrhonize, after this step, all device engines share the same knowledge
    // of IO environment.
    io.sync();
  }

  // prepare the assembly
  // - `%1` reads parameter from position 0 (classical positional parameter).
  // - `%2` reads parameter from position 1.
  // - `%3` is the product of two values, it will do auto type promotion.
  // - `dbg_print` print the value of `%3`
  constexpr auto code = R"PPHlo(
func @main(%arg0: tensor<!pphlo.sint>, %arg1: tensor<!pphlo.sint>) -> () {
  %0 = "pphlo.multiply"(%arg0, %arg1) : (tensor<!pphlo.sint>, tensor<!pphlo.sint>) -> tensor<!pphlo.sint>
  "pphlo.dbg_print"(%0) : (tensor<!pphlo.sint>) -> ()
  return
}
  )PPHlo";

  // run the assembly, with
  // - "x" binding to the first parameter (position 0).
  // - "y" binding to the second parameter (position 1).
  // - there is no output bindings.
  proc->run(code, {"x", "y"}, {});
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  auto proc = MakeProcessor();

  parameters(proc.get());

  constant_add(proc.get());

  return 0;
}
