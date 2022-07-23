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


#include "ppu/psi/cryptor/cryptor_selector.h"

#include <cstdlib>

#include "cpu_features/cpuinfo_x86.h"
#include "spdlog/spdlog.h"

#include "ppu/psi/cryptor/donna_ecc_cryptor.h"
#include "ppu/psi/cryptor/fourq_cryptor.h"
#include "ppu/psi/cryptor/ipp_ecc_cryptor.h"

namespace ppu {

namespace {

static const auto kCpuFeatures = cpu_features::GetX86Info().features;

std::unique_ptr<IEccCryptor> GetIppCryptor() {
  if (kCpuFeatures.avx512ifma) {
    SPDLOG_INFO("Using IPPCP");
    return std::make_unique<IppEccCryptor>();
  }
  return {};
}

std::unique_ptr<IEccCryptor> GetDonnaCryptor() {
  SPDLOG_INFO("Using Donna");
  return std::make_unique<DonnaEccCryptor>();
}

std::unique_ptr<IEccCryptor> GetFourQCryptor() {
  if (kCpuFeatures.avx2) {
    SPDLOG_INFO("Using FourQ");
    return std::make_unique<FourQEccCryptor>();
  }
  return {};
}

}  // namespace

std::unique_ptr<IEccCryptor> CreateEccCryptor(CurveType type) {
  std::unique_ptr<IEccCryptor> cryptor;
  switch (type) {
    case CurveType::Curve25519: {
      cryptor = GetIppCryptor();
      if (cryptor == nullptr) {
        cryptor = GetDonnaCryptor();
      }
      break;
    }
    case CurveType::CurveFourQ: {
      cryptor = GetFourQCryptor();
      PPU_ENFORCE(cryptor != nullptr, "FourQ requires AVX2 instruction");
      break;
    }
  }
  PPU_ENFORCE(cryptor != nullptr, "Cryptor should not be nullptr");
  return cryptor;
}

}  // namespace ppu
