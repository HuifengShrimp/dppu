load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "emp-ot",
    hdrs = glob([
        "emp-ot/*.h",
        "emp-ot/*.hpp",
        "emp-ot/ferret/*.hpp",
        "emp-ot/ferret/*.h",
    ]),
    copts = [
        "-Wno-ignored-attributes",
        "-Wno-ignored-qualifiers",
    ],
    deps = [
        "@com_github_emptoolkit_emp_tool//:emp-tool",
    ],
)
