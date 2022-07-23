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


#include "ppu/link/algorithm/trace.h"

#include <mutex>

#include "absl/strings/escaping.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/spdlog.h"

namespace ppu::link {
namespace {

static const char* kLoggerName = "logger";
static const char* kLoggerPath = "trace.log";

}  // namespace

void SetupTrace(TraceOptions opts) {
  spdlog::rotating_logger_mt(kLoggerName, kLoggerPath, opts.max_log_file_size,
                             opts.max_log_file_count);
}

void TraceLog(std::string_view event, std::string_view tag,
              std::string_view value) {
#ifdef ENABLE_LINK_TRACE
  static std::once_flag gInitTrace;
  std::call_once(gInitTrace, []() { SetupTrace(); });

  // trace this action anyway.
  SPDLOG_TRACE("[LINK] key={},tag={}", event, tag);

  // write to link file trace if enabled.
  auto logger = spdlog::get(kLoggerName);
  if (logger) {
    SPDLOG_LOGGER_INFO(logger, "[link] key={},tag={},value={}", event, tag,
                       absl::BytesToHexString(value));
  }
#endif
}

}  // namespace ppu::link
