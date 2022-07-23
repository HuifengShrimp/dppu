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



#include "ppu/crypto/asymmetric_util.h"

#include "openssl/bn.h"
#include "openssl/pem.h"
#include "openssl/rsa.h"

#include "ppu/utils/exception.h"
#include "ppu/utils/scope_guard.h"

namespace ppu::crypto {

using UniqueBio = std::unique_ptr<BIO, decltype(&BIO_free)>;
using UniqueRsa = std::unique_ptr<RSA, decltype(&RSA_free)>;

namespace {

constexpr int kRsaKeyBitSize = 2048;

}  // namespace

namespace internal {

UniquePkey CreatePriPkeyFromSm2Pem(utils::ByteContainerView pem) {
  UniqueBio pem_bio(BIO_new_mem_buf(pem.data(), pem.size()), BIO_free);
  EC_KEY* ec_key =
      PEM_read_bio_ECPrivateKey(pem_bio.get(), nullptr, nullptr, nullptr);
  PPU_ENFORCE(ec_key != nullptr, "No ec private key from pem.");
  ON_SCOPE_EXIT([&] { EC_KEY_free(ec_key); });
  EVP_PKEY* pri_key = EVP_PKEY_new();
  PPU_ENFORCE(pri_key != nullptr);
  PPU_ENFORCE_GT(EVP_PKEY_set1_EC_KEY(pri_key, ec_key), 0);
  PPU_ENFORCE_GT(EVP_PKEY_set_alias_type(pri_key, EVP_PKEY_SM2), 0);

  return UniquePkey(pri_key, ::EVP_PKEY_free);
}

UniquePkey CreatePubPkeyFromSm2Pem(utils::ByteContainerView pem) {
  UniqueBio pem_bio(BIO_new_mem_buf(pem.data(), pem.size()), BIO_free);
  EC_KEY* ec_key =
      PEM_read_bio_EC_PUBKEY(pem_bio.get(), nullptr, nullptr, nullptr);
  PPU_ENFORCE(ec_key != nullptr, "No ec public key from pem.");
  ON_SCOPE_EXIT([&] { EC_KEY_free(ec_key); });
  EVP_PKEY* pub_key = EVP_PKEY_new();
  PPU_ENFORCE(pub_key != nullptr);
  PPU_ENFORCE_GT(EVP_PKEY_set1_EC_KEY(pub_key, ec_key), 0);
  PPU_ENFORCE_GT(EVP_PKEY_set_alias_type(pub_key, EVP_PKEY_SM2), 0);

  return UniquePkey(pub_key, ::EVP_PKEY_free);
}

}  // namespace internal

std::tuple<std::string, std::string> CreateSm2KeyPair() {
  // Create sm2 curve
  EC_KEY* ec_key = EC_KEY_new();
  PPU_ENFORCE(ec_key != nullptr);
  ON_SCOPE_EXIT([&] { EC_KEY_free(ec_key); });
  EC_GROUP* ec_group = EC_GROUP_new_by_curve_name(NID_sm2);
  PPU_ENFORCE(ec_group != nullptr);
  ON_SCOPE_EXIT([&] { EC_GROUP_free(ec_group); });
  PPU_ENFORCE_GT(EC_KEY_set_group(ec_key, ec_group), 0);
  PPU_ENFORCE_GT(EC_KEY_generate_key(ec_key), 0);

  // Read private key
  BIO* pri_bio = BIO_new(BIO_s_mem());
  ON_SCOPE_EXIT([&] { BIO_free(pri_bio); });
  PPU_ENFORCE_GT(PEM_write_bio_ECPrivateKey(pri_bio, ec_key, nullptr, nullptr,
                                            0, nullptr, nullptr),
                 0);
  std::string private_key(BIO_pending(pri_bio), '\0');
  PPU_ENFORCE_GT(BIO_read(pri_bio, private_key.data(), private_key.size()), 0);

  // Read public key
  BIO* pub_bio = BIO_new(BIO_s_mem());
  ON_SCOPE_EXIT([&] { BIO_free(pub_bio); });
  PPU_ENFORCE_GT(PEM_write_bio_EC_PUBKEY(pub_bio, ec_key), 0);
  std::string public_key(BIO_pending(pub_bio), '\0');
  PPU_ENFORCE_GT(BIO_read(pub_bio, public_key.data(), public_key.size()), 0);

  return std::make_tuple(public_key, private_key);
}

std::tuple<std::string, std::string> CreateRsaKeyPair() {
  std::unique_ptr<BIGNUM, decltype(&BN_free)> exp(BN_new(), BN_free);
  PPU_ENFORCE_EQ(BN_set_word(exp.get(), RSA_F4), 1, "BN_set_word failed.");
  UniqueRsa rsa(RSA_new(), RSA_free);
  PPU_ENFORCE(
      RSA_generate_key_ex(rsa.get(), kRsaKeyBitSize, exp.get(), nullptr),
      "Generate rsa key pair failed.");

  std::string public_key;
  {
    UniqueBio bio(BIO_new(BIO_s_mem()), BIO_free);
    PPU_ENFORCE(bio, "New bio failed.");
    PPU_ENFORCE(PEM_write_bio_RSAPublicKey(bio.get(), rsa.get()),
                "Write public key failed.");
    int size = BIO_pending(bio.get());
    PPU_ENFORCE_GT(size, 0, "Bad key size.");
    public_key.resize(size);
    PPU_ENFORCE_GT(BIO_read(bio.get(), public_key.data(), size), 0,
                   "Cannot read bio.");
  }

  std::string private_key;
  {
    UniqueBio bio(BIO_new(BIO_s_mem()), BIO_free);
    PPU_ENFORCE(bio, "New bio failed.");
    PPU_ENFORCE(PEM_write_bio_RSAPrivateKey(bio.get(), rsa.get(), nullptr,
                                            nullptr, 0, 0, nullptr),
                "Write private key failed.");
    int size = BIO_pending(bio.get());
    PPU_ENFORCE_GT(size, 0, "Bad key size.");
    private_key.resize(size);
    PPU_ENFORCE_GT(BIO_read(bio.get(), private_key.data(), size), 0,
                   "Cannot read bio.");
  }

  return std::make_tuple(public_key, private_key);
}

}  // namespace ppu::crypto