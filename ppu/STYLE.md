# C++ coding style

* This draft mainly come from [Envoy Style](https://github.com/envoyproxy/envoy/blob/master/STYLE.md) with a few adjustments.

* The Nebula source code is formatted using clang-format. Thus all white spaces, etc. issues are taken care of automatically. The Linter (`WIP`) tests will automatically check the code format and fail. There are make targets that can both check the format (check_format) as well as fix the code format for you (fix_format). Errors in .clang-tidy are enforced while other warnings are suggestions. Note that code and comment blocks designated `clang-format off` must be closed with `clang-format on`. To run these checks locally, see [Support Tools](support/README.md). 
* Beyond code formatting, for the most part Nebula uses the [Google C++ style guidelines](https://google.github.io/styleguide/cppguide.html). The following section covers the major areas where we deviate from the Google guidelines.

# Deviations from Google C++ style guidelines

* Exceptions are allowed and encouraged where appropriate. When using exceptions, do not add additional error handing that cannot possibly happen in the case an exception is thrown.

* Do use exceptions for:
  - Constructor failure.
  - Error handling in deep call stacks, where exceptions provide material
    improvements to code complexity and readability.

* Apply caution when using exceptions on the data path for general purpose error handling. Exceptions are not caught on the data path and they should not be used for simple error handling, e.g. with shallow call stacks, where explicit error handling provides a more readable and easier to reason about implementation.

* Prefer `unique_ptr` over `shared_ptr` wherever possible. `unique_ptr` makes ownership in production code easier to reason about. 

* The Google C++ style guide points out that [non-POD static and global variables are forbidden](https://google.github.io/styleguide/cppguide.html#Static_and_Global_Variables). This _includes_ types such as `std::string`. We encourage the use of the advice in the [C++ FAQ on the static initialization fiasco](https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use) for how to best handle this.

* Header guards should use `#pragma once`.

* [WIP: Thread annotations](https://github.com/abseil/abseil-cpp/blob/master/absl/base/thread_annotations.h), such as `GUARDED_BY`, should be used for shared state guarded by locks/mutexes.

* Functions intended to be local to a cc file should be declared in an anonymous namespace, rather than using  the 'static' keyword. Note that the [Google C++ style guide](https://google.github.io/styleguide/cppguide.html#Unnamed_Namespaces_and_Static_Variables) allows either, but in Nebula we  prefer anonymous namespaces.

* Braces are required for all control statements include single line if, while, etc. statements.

# Comments style

* We are using [Google C++ comments style](https://google.github.io/styleguide/cppguide.html#Comments).
* We suggest using \`\` to strengthen important arguments or information.
* Use `//`, don't use `/* */`.

# Git commit message style

* Please see [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/)
* We strongly suggest using meanful imperative [keywords](https://github.com/joelparkerhenderson/git_commit_message#summary-keywords).

## Aside: Exception vs Status
Why exception:
* See [Zhihu exception discuss.](https://www.zhihu.com/question/22889420/answer/282031257)
* Usually, exception leads to bigger binary size. Besides, introducing exception might encounter  performance drop too. Overall, the exceptions are favored if we are not write something like  `Linux Kernel`.

## Aside: how to write exception safe code
Please see http://exceptionsafecode.com/
Slides: [Exception Safe Coding](http://exceptionsafecode.com/slides/esc.pdf).
[Video](https://youtu.be/N9bR0ztmmEQ) version.

# Error handling

A few general notes on our error handling philosophy:

* All error code returns should be checked.

* OOM events (both memory and FDs) are considered fatal crashing errors. An OOM error should never silently be ignored and should crash the process either via the C++ allocation error exception, an explicit `PPU_ENFORCE` following a third party library call, or an obvious crash on a subsequent line via null pointer dereference. This rule is again based on the philosophy that the engineering costs of properly handling these cases are not worth it. Time is better spent designing proper system controls that shed load if resource usage becomes too high, etc.

* At a very high level, our philosophy is that errors that are *likely* to happen should be gracefully handled. Examples of likely errors include any type of network error, disk IO error, bad data returned by an API call, bad data read from runtime files, etc. Errors that are *unlikely* to happen should lead to process death, under the assumption that the additional burden of defensive coding and testing is not an effective use of time for an error that should not happen given proper system setup. Examples of these types of errors include not being able to open the shared memory region, an invalid initial JSON config read from disk, system calls that should not fail assuming correct parameters (which should be validated via tests), etc. Examples of system calls that should not fail when passed valid parameters include most usages of `setsockopt()`, `getsockopt()`, the kernel returning a valid `sockaddr` after a successful call to `accept()`, `pthread_create()`, `pthread_join()`, etc.

* Tip: If the thought of adding the extra test coverage, logging, and stats to handle an error and continue seems ridiculous because *"this should never happen"*, it's a very good indication that the appropriate behavior is to terminate the process and not handle the error. When in doubt, please discuss.

* Per above it's acceptable to turn failures into crash semantics via `PPU_ENFORCE(message)` if there is no other sensible behavior, e.g. in OOM (memory/FD) scenarios. Use `PPU_ENFORCE` liberally, but do not use it for things that will crash in an obvious way in a subsequent line. E.g., do not do `PPU_ENFORCE(foo != nullptr); foo->doSomething();`. Note that there is a gray line between external environment failures and program invariant violations. For example, memory corruption due to a security issue (a bug, deliberate buffer overflow etc.) might manifest as a violation of program invariants or as a detectable condition in the external environment (e.g. some library returning a highly unexpected error code or buffer contents). 

# Hermetic and deterministic tests

Tests should be hermetic, i.e. have all dependencies explicitly captured and not depend on the local environment. In general, there should be no non-local network access. In addition:

* Port numbers should not be hardcoded. Tests should bind to port zero and then discover the bound port  when needed. This avoids flakes due to conflicting ports and allows tests to be executed concurrently by  Bazel. See [`ppu/biz/tasks/unittest/task_test_util.cpp`](ppu/biz/tasks/unittest/task_test_util.cpp) for examples of tests that do this.

Tests should be deterministic. They should not rely on randomness or details such as the current time.

# C++ tips

## [Abseil Tips](https://abseil.io/tips/)

AWESOME TIPS from the Abseil team.

A few shortcuts:

* [Flags Are Globals](https://abseil.io/tips/103)
* [Avoid Flags, Especially in Library Code](https://abseil.io/tips/45)
* [Return Policy](https://abseil.io/tips/11)
* [Callsite Readability and bool Parameters](https://abseil.io/tips/94)
* [Return Values, References, and Lifetimes](https://abseil.io/tips/101)
* [Copy Elision and Pass-by-value](https://abseil.io/tips/117)

## [CppCoreGuidelines](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md)

By Bjarne Stroustrup & Herb Sutter.
We don't use GSL but it is worthwhile to go through the guidelines.

# Dependency Injection

- All third party dependencies should be injected by bazel `http_archive`. You should at least provide an `external` download url for the package. The github link is recommended.
- If the dependency does not support `bazel` build, think about wrap a build file for it. See [fmt BUILD](bazel/fmtlib.BUILD).
- If we need to patch the dependency, think about provide a patch file. See [brpc patch](bazel/patches/brpc.patch).
- `DONT` directly download the packages into `third_party`. To speedup ACI, we suggest the downloaded packages should be copy into `third_party/archive`, which is a `submodule` of our repo. As we requires at least one `external` link available, the build should be fine without `third_party/archive`.

# UnitTest/IntegrationTest

## UnitTest
Developer must write unittest for new code. And the line coverage must be greater than 80%(maybe defined later).

CI flow will run unittest and check line coverage automatically.
- If unittest failed, the branch can't be merged to master;
- If line coverage is less than 80%, a message will be sent to dingding group. The code must be reviewed and accepted by at least two developers, then the code can be merged to master.

## IntegrationTest
After branch merged to master, the CI flow will run the intergration test automatically. if the test failed, a message will be sent to dingding group, the developer must handle the bug as soon as possible.

If you want to run the intergration test before you merge the branch to master, you can have a try:
TODO: add how to run intergration test offline

# Google style guides for other languages

* [Python](https://google.github.io/styleguide/pyguide.html)
* [Bash](https://google.github.io/styleguide/shell.xml)
* [Bazel](https://bazel.build/versions/master/docs/skylark/build-style.html)
