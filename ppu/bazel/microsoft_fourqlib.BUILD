load("@rules_foreign_cc//foreign_cc:defs.bzl", "make")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

make(
    name = "fourqlib",
    args = [
        "ARCH=x64",
        "AVX2=TRUE",
    ] + select({
        "@bazel_tools//src/conditions:darwin": ["ASM=FALSE"],
        "//conditions:default": ["ASM=TRUE"],
    }),
    defines = [
        "__LINUX__",
        "_AMD64_",
    ],
    env = select({
        "@bazel_tools//src/conditions:darwin": {
            "AR": "ar",
        },
        "//conditions:default": {},
    }),
    lib_source = ":all_srcs",
    out_static_libs = ["libfourq.a"],
    targets = ["install"],
    tool_prefix = "export BUILD_TMPDIR=$BUILD_TMPDIR/FourQ_64bit_and_portable &&",
)
