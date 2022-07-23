#!/bin/bash

# c++
find . -type f -name '*.h' -o -name '*.cc' | xargs clang-format -i 

# bazel
buildifier -r .

# python
find . -type f -name '*.py' -print0 | xargs -0 yapf -i
