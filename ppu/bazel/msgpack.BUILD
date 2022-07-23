load("@ppulib//bazel:ppu.bzl", "ppu_cmake_external")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

ppu_cmake_external(
    name = "msgpack",
    cache_entries = {
        "MSGPACK_CXX17": "ON",
        "MSGPACK_BUILD_EXAMPLES": "OFF",
        "BUILD_SHARED_LIBS": "OFF",
        "MSGPACK_BUILD_TESTS": "OFF",
    },
    lib_source = ":all_srcs",
    out_headers_only = True,
)
