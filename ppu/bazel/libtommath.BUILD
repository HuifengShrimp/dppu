load("@rules_foreign_cc//foreign_cc:defs.bzl", "make")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

make(
    name = "libtommath",
    env = {"PREFIX": "$INSTALLDIR"},
    lib_source = ":all_srcs",
    out_static_libs = ["libtommath.a"],
    targets = ["install"],
)
