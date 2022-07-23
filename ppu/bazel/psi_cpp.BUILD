load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "psi_cpp",
    srcs = [
        "include/curve25519-donna/curve25519.c",
    ],
    hdrs = glob([
        "include/cppcodec/**/*.hpp",
        "include/curve25519/**/*.hpp",
        "include/curve25519-donna/**/*.h",
    ]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
