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


#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <string>

#include "google/protobuf/util/json_util.h"
#include "llvm/Support/CommandLine.h"
#include "spdlog/spdlog.h"

#include "ppu/mpc/factory.h"
#include "ppu/mpc/interfaces.h"
#include "ppu/mpc/object.h"
#include "ppu/mpc/util/communicator.h"
#include "ppu/mpc/util/simulate.h"

#include "ppu/mpc/tools/complexity.pb.h"

namespace ppu::mpc {

internal::SingleComplexityReport dumpSemi2k() {
  internal::SingleComplexityReport single_report;
  single_report.set_protocol("Semi2k");

  // the interested kernel whitelist.
  const std::vector<std::string> kWhitelist = {
      "A2B", "B2A", "A2P", "B2P", "AddBB", "MatMulAA", "MatMulAP"};

  // print header
  fmt::print("{:<15}, {:<20}, {:<20}\n", "name", "latency", "comm");

  util::simulate(2, [&](const std::shared_ptr<link::Context>& lctx) -> void {
    auto prot = Factory::CreateCompute(ProtocolKind::SEMI2K, lctx);
    if (lctx->Rank() != 0) {
      return;
    }

    for (auto name : kWhitelist) {
      auto* kernel = prot->getKernel(name);
      auto latency = kernel->latency();
      auto comm = kernel->comm();

      const std::string latency_str = latency ? latency->expr() : "Unknown";
      const std::string comm_str = comm ? comm->expr() : "Unknown";

      fmt::print("{:<15}, {:<20}, {:<20}\n", name, latency_str, comm_str);

      auto* entry = single_report.add_entries();
      entry->set_kernel(name);
      entry->set_comm(comm_str);
      entry->set_latency(latency_str);
    }

    return;
  });

  return single_report;
}

}  // namespace ppu::mpc

llvm::cl::opt<std::string> OutputFilename(
    "out", llvm::cl::desc("Specify output json filename"),
    llvm::cl::value_desc("filename"));

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // suppress all link logs.
  spdlog::set_level(spdlog::level::off);

  ppu::mpc::internal::ComplexityReport report;

  *(report.add_reports()) = ppu::mpc::dumpSemi2k();

  if (!OutputFilename.empty()) {
    std::string json;
    google::protobuf::util::JsonPrintOptions json_options;
    json_options.preserve_proto_field_names = true;

    PPU_ENFORCE(
        google::protobuf::util::MessageToJsonString(report, &json, json_options)
            .ok());

    std::ofstream out(OutputFilename.getValue());

    out << json;
  }
}
