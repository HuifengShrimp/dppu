load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "emp-tool",
    srcs = glob([
        "emp-tool/emp-tool.cpp",
        "emp-tool/utils/*.cpp",
        "emp-tool/utils/*.h",
        "emp-tool/utils/*.hpp",
        "emp-tool/execution/*.cpp",
        "emp-tool/execution/*.h",
        "emp-tool/circuits/files/*.cpp",
        "emp-tool/circuits/files/*.h",
        "emp-tool/circuits/*.cpp",
        "emp-tool/circuits/*.h",
        "emp-tool/circuits/*.hpp",
        "emp-tool/gc/*.cpp",
        "emp-tool/gc/*.h",
        "emp-tool/io/*.cpp",
        "emp-tool/io/*.h",
    ]),
    hdrs = [
        "emp-tool/emp-tool.h",
    ],
    copts = [
        "-Wno-ignored-attributes",
        "-Wno-ignored-qualifiers",
    ],
    includes = [""],
    deps = ["@com_github_openssl_openssl//:openssl"],
)
