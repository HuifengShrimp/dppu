# PPU: Privacy-Preserving Processing Unit

PPU (Privacy-Preserving Processing Unit) aims to be a `provable`, `measurable` secure computation device, which provides computation ability while keep your private data protected.

## Project status

Currently, we mainly focus on the `provable` security. It contains a secure runtime that evaluates [XLA](https://www.tensorflow.org/xla/operation_semantics)-like tensor operations, which use [MPC](https://en.wikipedia.org/wiki/Secure_multi-party_computation) as the underline evaluation engine to protect privacy information.

## Contents

- [Documentation](TODO: doc)
- [Roadmap](TODO: doc)
- [Build and test](#Build)
- [FAQ](#FAQ)

## Build

### Prerequisite

#### Docker

```sh
## start container
docker run -d -it --name ppu-gcc11-dev-$(whoami) \
         --mount type=bind,source="$(pwd)",target=/home/admin/dev/ \
         -w /home/admin/dev \
         --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
         --cap-add=NET_ADMIN \
         registry.hub.docker.com/secretflow/ppu-gcc11-anolis-dev:latest

# attach to build container
docker exec -it ppu-gcc11-dev-$(whoami) bash
```
#### Linux

```sh
Install gcc>=11.2, cmake>=3.18, ninja, nasm>=2.15, python==3.8, bazel==4.2

python3 -m pip install -r docker/requirements.txt
```

#### macOS

```sh
# Install Xcode
https://apps.apple.com/us/app/xcode/id497799835?mt=12

# Select Xcode toolchain version
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer

# Install homebrew
https://brew.sh/

# Install dependencies
brew install bazel cmake ninja nasm

# Install python dependencies
python3 -m pip install -r docker/requirements.txt
```

### Build & UnitTest

``` sh

# build as debug
bazel build //... -c dbg

# build as release
bazel build //... -c opt

# test
bazel test //...

# [optional] build & test with ASAN
bazel build //... -c dbg --config=asan
bazel test //... --config=asan -c dbg
```

### Bazel build options

- `--define gperf=on` enable gperf
- `--define tracelog=on` enable link trace log.

### Build docs

```sh
# prerequisite
pip install -U sphinx
pip install -U recommonmark

cd docs & make html  # html docs will be in docs/_build/html
```

## FAQ

> How can I use PPU?

PPU could be treated as a programmable device, it's not designed to be used directly. Normally we use secretflow framework, which use PPU as the underline privacy-preserving computing device.

## Acknowledgement

Thanks for [Alibaba Gemini Lab](https://alibaba-gemini-lab.github.io) for the security advices.
