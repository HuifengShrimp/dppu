load("@ppulib//bazel:ppu.bzl", "ppu_cmake_external")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

ppu_cmake_external(
    name = "zlib",
    cache_entries = {
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
    },
    lib_source = ":all_srcs",
    out_static_libs = ["libz.a"],
)
