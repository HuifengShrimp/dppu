#! /bin/sh

bazel build //ppu/dialect:pphlo_op_doc

cp `bazel info workspace`/bazel-bin/ppu/dialect/pphlo_op_doc.md .
