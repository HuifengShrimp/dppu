#! /bin/bash
#
# Copyright 2021 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

bazel build //ppu:ppu_wheel -c opt

ppu_wheel=$(<bazel-bin/ppu/ppu_wheel.name)
ppu_wheel_path="bazel-bin/ppu/${ppu_wheel//sf-ppu/sf_ppu}"

python3 -m pip install $ppu_wheel_path --force-reinstall


