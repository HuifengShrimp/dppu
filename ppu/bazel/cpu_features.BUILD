load("@ppulib//bazel:ppu.bzl", "ppu_cmake_external")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

ppu_cmake_external(
    name = "cpu_features",
    cache_entries = {
        "CMAKE_INSTALL_LIBDIR": "lib",
        "CMAKE_C_FLAGS": "-fPIC",
    },
    lib_source = ":all_srcs",
    out_lib_dir = "lib",
    out_static_libs = ["libcpu_features.a"],
)
