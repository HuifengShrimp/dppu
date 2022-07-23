load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

SECRETFLOW_GIT = "https://github.com/secretflow"

def ppu_deps():
    _rule_python()
    _rules_foreign_cc()
    _com_github_madler_zlib()
    _com_google_protobuf()
    _com_google_googletest()
    _com_google_absl()
    _com_github_google_leveldb()
    _com_github_brpc_brpc()
    _com_github_fmtlib_fmt()
    _com_github_gabime_spdlog()
    _com_github_libtom_libtommath()
    _com_github_openssl_openssl()
    _com_github_google_benchmark()
    _com_github_gperftools_gperftools()
    _com_github_floodyberry_curve25519_donna()
    _com_github_blake3team_blake3()
    _com_github_xtensor_xtensor()
    _com_github_xtensor_xtl()
    _com_github_grpc_grpc()
    _com_github_tensorflow()
    _com_bazelbuild_bazel_rules_docker()
    _com_github_pybind11_bazel()
    _com_github_pybind11()
    _com_github_google_cpu_features()
    _com_github_intel_ipp()
    _com_github_microsoft_fourqlib()
    _com_github_google_boringssl()
    _com_github_emptoolkit_emp_tool()
    _com_github_emptoolkit_emp_ot()

    maybe(
        git_repository,
        name = "simplest_ot",
        commit = "540047499cd34ebc6e8f4f42322c088511bd7434",
        recursive_init_submodules = True,
        remote = "{}/simplest-ot.git".format(SECRETFLOW_GIT),
    )

    maybe(
        git_repository,
        name = "aio",
        commit = "2726a6b9e6d0a74f08a2ee55e36e938b1f921975",
        recursive_init_submodules = True,
        remote = "{}/aio.git".format(SECRETFLOW_GIT),
    )

def _com_github_madler_zlib():
    maybe(
        http_archive,
        name = "zlib",
        build_file = "@ppulib//bazel:zlib.BUILD",
        strip_prefix = "zlib-1.2.11",
        sha256 = "629380c90a77b964d896ed37163f5c3a34f6e6d897311f1df2a7016355c45eff",
        type = ".tar.gz",
        urls = [
            "https://github.com/madler/zlib/archive/refs/tags/v1.2.11.tar.gz",
        ],
    )

def _com_github_floodyberry_curve25519_donna():
    maybe(
        http_archive,
        name = "com_github_floodyberry_curve25519_donna",
        strip_prefix = "curve25519-donna-2fe66b65ea1acb788024f40a3373b8b3e6f4bbb2",
        sha256 = "ba57d538c241ad30ff85f49102ab2c8dd996148456ed238a8c319f263b7b149a",
        type = "tar.gz",
        build_file = "@ppulib//bazel:curve25519-donna.BUILD",
        urls = [
            "https://github.com/floodyberry/curve25519-donna/archive/2fe66b65ea1acb788024f40a3373b8b3e6f4bbb2.tar.gz",
        ],
    )

def _com_github_google_leveldb():
    maybe(
        http_archive,
        name = "com_github_google_leveldb",
        strip_prefix = "leveldb-1.23",
        sha256 = "9a37f8a6174f09bd622bc723b55881dc541cd50747cbd08831c2a82d620f6d76",
        type = "tar.gz",
        build_file = "@ppulib//bazel:leveldb.BUILD",
        patch_args = ["-p1"],
        patches = ["@ppulib//bazel:patches/leveldb.patch"],
        urls = [
            "https://github.com/google/leveldb/archive/refs/tags/1.23.tar.gz",
        ],
    )

def _com_github_brpc_brpc():
    maybe(
        http_archive,
        name = "com_github_brpc_brpc",
        sha256 = "114ab4b2a73ff60da1d578a153bece1bee4d9ded5f3a63f0e15b65e04f16cefe",
        strip_prefix = "incubator-brpc-1.0.0",
        type = "tar.gz",
        patch_args = ["-p1"],
        patches = ["@ppulib//bazel:patches/brpc.patch"],
        urls = [
            "https://github.com/apache/incubator-brpc/archive/refs/tags/1.0.0.tar.gz",
        ],
    )

def _com_github_grpc_grpc():
    maybe(
        http_archive,
        name = "com_github_grpc_grpc",
        sha256 = "8c05641b9f91cbc92f51cc4a5b3a226788d7a63f20af4ca7aaca50d92cc94a0d",
        strip_prefix = "grpc-1.44.0",
        type = "tar.gz",
        urls = [
            "https://github.com/grpc/grpc/archive/refs/tags/v1.44.0.tar.gz",
        ],
    )

def _com_google_protobuf():
    maybe(
        http_archive,
        name = "com_google_protobuf",
        sha256 = "ba0650be1b169d24908eeddbe6107f011d8df0da5b1a5a4449a913b10e578faf",
        strip_prefix = "protobuf-3.19.4",
        type = "tar.gz",
        urls = [
            "https://github.com/protocolbuffers/protobuf/releases/download/v3.19.4/protobuf-all-3.19.4.tar.gz",
        ],
    )

def _com_google_absl():
    maybe(
        http_archive,
        name = "com_google_absl",
        sha256 = "dcf71b9cba8dc0ca9940c4b316a0c796be8fab42b070bb6b7cab62b48f0e66c4",
        type = "tar.gz",
        strip_prefix = "abseil-cpp-20211102.0",
        urls = [
            "https://github.com/abseil/abseil-cpp/archive/refs/tags/20211102.0.tar.gz",
        ],
    )

def _com_github_openssl_openssl():
    maybe(
        http_archive,
        name = "com_github_openssl_openssl",
        sha256 = "dac036669576e83e8523afdb3971582f8b5d33993a2d6a5af87daa035f529b4f",
        type = "tar.gz",
        strip_prefix = "openssl-OpenSSL_1_1_1l",
        urls = [
            "https://github.com/openssl/openssl/archive/refs/tags/OpenSSL_1_1_1l.tar.gz",
        ],
        build_file = "@ppulib//bazel:openssl.BUILD",
    )

def _com_github_fmtlib_fmt():
    maybe(
        http_archive,
        name = "com_github_fmtlib_fmt",
        strip_prefix = "fmt-8.1.1",
        sha256 = "3d794d3cf67633b34b2771eb9f073bde87e846e0d395d254df7b211ef1ec7346",
        build_file = "@ppulib//bazel:fmtlib.BUILD",
        urls = [
            "https://github.com/fmtlib/fmt/archive/refs/tags/8.1.1.tar.gz",
        ],
    )

def _com_github_gabime_spdlog():
    maybe(
        http_archive,
        name = "com_github_gabime_spdlog",
        strip_prefix = "spdlog-1.9.2",
        type = "tar.gz",
        sha256 = "6fff9215f5cb81760be4cc16d033526d1080427d236e86d70bb02994f85e3d38",
        build_file = "@ppulib//bazel:spdlog.BUILD",
        urls = [
            "https://github.com/gabime/spdlog/archive/refs/tags/v1.9.2.tar.gz",
        ],
    )

def _com_google_googletest():
    maybe(
        http_archive,
        name = "com_google_googletest",
        sha256 = "b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5",
        type = "tar.gz",
        strip_prefix = "googletest-release-1.11.0",
        urls = [
            "https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz",
        ],
    )

def _com_github_libtom_libtommath():
    maybe(
        http_archive,
        name = "com_github_libtom_libtommath",
        sha256 = "f3c20ab5df600d8d89e054d096c116417197827d12732e678525667aa724e30f",
        type = "tar.gz",
        strip_prefix = "libtommath-1.2.0",
        patch_args = ["-p1"],
        patches = ["@ppulib//bazel:patches/libtommath-1.2.0.patch"],
        urls = [
            "https://github.com/libtom/libtommath/archive/v1.2.0.tar.gz",
        ],
        build_file = "@ppulib//bazel:libtommath.BUILD",
    )

def _com_github_google_benchmark():
    maybe(
        http_archive,
        name = "com_github_google_benchmark",
        type = "tar.gz",
        strip_prefix = "benchmark-1.6.1",
        sha256 = "6132883bc8c9b0df5375b16ab520fac1a85dc9e4cf5be59480448ece74b278d4",
        urls = [
            "https://github.com/google/benchmark/archive/refs/tags/v1.6.1.tar.gz",
        ],
    )

def _com_github_gperftools_gperftools():
    maybe(
        http_archive,
        name = "com_github_gperftools_gperftools",
        type = "tar.gz",
        strip_prefix = "gperftools-2.9.1",
        sha256 = "ea566e528605befb830671e359118c2da718f721c27225cbbc93858c7520fee3",
        urls = [
            "https://github.com/gperftools/gperftools/releases/download/gperftools-2.9.1/gperftools-2.9.1.tar.gz",
        ],
        build_file = "@ppulib//bazel:gperftools.BUILD",
    )

def _com_github_blake3team_blake3():
    maybe(
        http_archive,
        name = "com_github_blake3team_blake3",
        strip_prefix = "BLAKE3-1.3.0",
        sha256 = "a559309c2dad8cc8314ea779664ec5093c79de2e9be14edbf76ae2ce380222c0",
        build_file = "@ppulib//bazel:blake3.BUILD",
        urls = [
            "https://github.com/BLAKE3-team/BLAKE3/archive/refs/tags/1.3.0.tar.gz",
        ],
    )

def _com_github_xtensor_xtensor():
    maybe(
        http_archive,
        name = "com_github_xtensor_xtensor",
        sha256 = "37738aa0865350b39f048e638735c05d78b5331073b6329693e8b8f0902df713",
        strip_prefix = "xtensor-0.24.0",
        build_file = "@ppulib//bazel:xtensor.BUILD",
        type = "tar.gz",
        urls = [
            "https://github.com/xtensor-stack/xtensor/archive/refs/tags/0.24.0.tar.gz",
        ],
    )

def _com_github_xtensor_xtl():
    maybe(
        http_archive,
        name = "com_github_xtensor_xtl",
        sha256 = "f4a81e3c9ca9ddb42bd4373967d4859ecfdca1aba60b9fa6ced6c84d8b9824ff",
        strip_prefix = "xtl-0.7.3",
        build_file = "@ppulib//bazel:xtl.BUILD",
        type = "tar.gz",
        urls = [
            "https://github.com/xtensor-stack/xtl/archive/refs/tags/0.7.3.tar.gz",
        ],
    )

def _rule_python():
    maybe(
        http_archive,
        name = "rules_python",
        sha256 = "a30abdfc7126d497a7698c29c46ea9901c6392d6ed315171a6df5ce433aa4502",
        strip_prefix = "rules_python-0.6.0",
        urls = [
            "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.6.0.tar.gz",
        ],
    )

def _rules_foreign_cc():
    maybe(
        http_archive,
        name = "rules_foreign_cc",
        sha256 = "bcd0c5f46a49b85b384906daae41d277b3dc0ff27c7c752cc51e43048a58ec83",
        strip_prefix = "rules_foreign_cc-0.7.1",
        urls = [
            "https://github.com/bazelbuild/rules_foreign_cc/archive/0.7.1.tar.gz",
        ],
    )

def _com_github_tensorflow():
    TFRT_COMMIT = "c3e082762b7664bbc7ffd2c39e86464928e27c0c"
    TFRT_SHA256 = "9b7fabe6e786e6437bb7cd1a4bed8416da6f08969266e57945805017092900c6"
    maybe(
        http_archive,
        name = "tf_runtime",
        sha256 = TFRT_SHA256,
        strip_prefix = "runtime-{commit}".format(commit = TFRT_COMMIT),
        urls = [
            "http://mirror.tensorflow.org/github.com/tensorflow/runtime/archive/{commit}.tar.gz".format(commit = TFRT_COMMIT),
            "https://github.com/tensorflow/runtime/archive/{commit}.tar.gz".format(commit = TFRT_COMMIT),
        ],
    )
    LLVM_COMMIT = "55c71c9eac9bc7f956a05fa9258fad4f86565450"
    LLVM_SHA256 = "1459d328ea67802f5b7c64349ba300b5ddc4a78838d6b77a8a970fe99ed3e78c"
    maybe(
        http_archive,
        name = "llvm-raw",
        build_file_content = "#empty",
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        urls = [
		"file:///home/admin/bazel_tools/llvm-project-55c71c9eac9bc7f956a05fa9258fad4f86565450.tar.gz",

        ],
    )
    SKYLIB_VERSION = "1.0.3"
    http_archive(
        name = "bazel_skylib",
        sha256 = "1c531376ac7e5a180e0237938a2536de0c54d93f5c278634818e0efc952dd56c",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = SKYLIB_VERSION),
            "https://github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = SKYLIB_VERSION),
        ],
    )

    # We need tensorflow to handle xla->mlir hlo
    maybe(
        http_archive,
        name = "org_tensorflow",
        sha256 = "66b953ae7fba61fd78969a2e24e350b26ec116cf2e6a7eb93d02c63939c6f9f7",
        strip_prefix = "tensorflow-2.8.0",
        patch_args = ["-p1"],
        # Fix mlir package visibility
        patches = ["@ppulib//bazel:patches/tensorflow.patch"],
        type = ".tar.gz",
        urls = [
            "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.8.0.tar.gz",
        ],
    )

def _com_bazelbuild_bazel_rules_docker():
    maybe(
        http_archive,
        name = "io_bazel_rules_docker",
        sha256 = "92779d3445e7bdc79b961030b996cb0c91820ade7ffa7edca69273f404b085d5",
        strip_prefix = "rules_docker-0.20.0",
        urls = [
            "https://github.com/bazelbuild/rules_docker/releases/download/v0.20.0/rules_docker-v0.20.0.tar.gz",
        ],
    )

def _com_github_pybind11_bazel():
    maybe(
        http_archive,
        name = "pybind11_bazel",
        sha256 = "a5666d950c3344a8b0d3892a88dc6b55c8e0c78764f9294e806d69213c03f19d",
        strip_prefix = "pybind11_bazel-26973c0ff320cb4b39e45bc3e4297b82bc3a6c09",
        urls = [
            "https://github.com/pybind/pybind11_bazel/archive/26973c0ff320cb4b39e45bc3e4297b82bc3a6c09.zip",
        ],
    )

def _com_github_pybind11():
    maybe(
        http_archive,
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11.BUILD",
        sha256 = "c6160321dc98e6e1184cc791fbeadd2907bb4a0ce0e447f2ea4ff8ab56550913",
        strip_prefix = "pybind11-2.9.1",
        urls = [
            "file:///home/admin/bazel_tools/pybind11-2.9.1.tar.gz",
        ],
    )

def _com_github_google_cpu_features():
    maybe(
        http_archive,
        name = "com_github_google_cpu_features",
        strip_prefix = "cpu_features-0.6.0",
        type = "tar.gz",
        sha256 = "95a1cf6f24948031df114798a97eea2a71143bd38a4d07d9a758dda3924c1932",
        build_file = "@ppulib//bazel:cpu_features.BUILD",
        urls = [
            "https://github.com/google/cpu_features/archive/refs/tags/v0.6.0.tar.gz",
        ],
    )

def _com_github_intel_ipp():
    maybe(
        http_archive,
        name = "com_github_intel_ipp",
        sha256 = "0b277548c59e6bfe489e634d622b54be3708086fc006a441d39922c2d6d43f0d",
        strip_prefix = "ipp-crypto-ippcp_2021.5",
        build_file = "@ppulib//bazel:ipp.BUILD",
        patch_args = ["-p1"],
        patches = ["@ppulib//bazel:patches/ippcp.patch"],
        urls = [
            "https://github.com/intel/ipp-crypto/archive/refs/tags/ippcp_2021.5.tar.gz",
        ],
    )

def _com_github_microsoft_fourqlib():
    maybe(
        http_archive,
        name = "com_github_microsoft_fourqlib",
        type = "zip",
        strip_prefix = "FourQlib-ff61f680505c98c98e33387962223ce0b5e620bc",
        sha256 = "59f1ebc35735217fc8c8f02c41765560ce3c5a8abd3937b0e2f4db45c49b6e73",
        build_file = "@ppulib//bazel:microsoft_fourqlib.BUILD",
        patch_args = ["-p1"],
        patches = ["@ppulib//bazel:patches/fourq.patch"],
        urls = [
            "https://github.com/microsoft/FourQlib/archive/ff61f680505c98c98e33387962223ce0b5e620bc.zip",
        ],
    )

# boringssl is required by grpc, we manually use a higher version.
def _com_github_google_boringssl():
    maybe(
        http_archive,
        name = "boringssl",
        sha256 = "09a9ea8b7ecdc97a7e2f128fc0fa7fcc91d781832ad19293054d3547f95fb2cd",
        strip_prefix = "boringssl-5ad11497644b75feba3163135da0909943541742",
        urls = [
            "https://github.com/google/boringssl/archive/5ad11497644b75feba3163135da0909943541742.zip",
        ],
    )

def _com_github_emptoolkit_emp_tool():
    maybe(
        http_archive,
        name = "com_github_emptoolkit_emp_tool",
        sha256 = "217a2cc46f1839efe0f23f6e615fd032094fb53695925be4ca18ae6c7c3e643c",
        strip_prefix = "emp-tool-0.2.3",
        type = "tar.gz",
        patch_args = ["-p1"],
        patches = ["@ppulib//bazel:patches/emp-tool.patch"],
        urls = [
            "https://github.com/emp-toolkit/emp-tool/archive/refs/tags/0.2.3.tar.gz",
        ],
        build_file = "@ppulib//bazel:emp-tool.BUILD",
    )

def _com_github_emptoolkit_emp_ot():
    maybe(
        http_archive,
        name = "com_github_emptoolkit_emp_ot",
        sha256 = "9c1198e04e2a081386814e9bea672fa6b4513829961c4ee150634354da609a91",
        strip_prefix = "emp-ot-0.2.2",
        type = "tar.gz",
        patch_args = ["-p1"],
        patches = ["@ppulib//bazel:patches/emp-ot.patch"],
        urls = [
            "https://github.com/emp-toolkit/emp-ot/archive/refs/tags/0.2.2.tar.gz",
        ],
        build_file = "@ppulib//bazel:emp-ot.BUILD",
    )

