
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


# This examples show how to use [flax](https://github.com/google/flax) as
# frontend to write a simple DNN, and run it on PPU.
#
# Start two node servers.
# > bazel run //examples/python/distributed:daemon -- up
#
# Run this example script.
# > bazel run //examples/python/e2e:flax_dnn_ppu

import examples.python.distributed as distr
import examples.python.utils.fe_utils as fe
import examples.python.utils.dataset_utils as dataset_utils
import examples.python.e2e.flax_dnn as flax_dnn
from examples.python.distributed import P1, P2, PPU

import argparse

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/2pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = distr.config.parse_json(file.read())
distr.init(conf)


@distr.device(P1)
def load_feature_r1():
    import examples.python.utils.dataset_utils as dataset_utils
    return dataset_utils.load_feature(0)


@distr.device(P1)
def load_label():
    import examples.python.utils.dataset_utils as dataset_utils
    return dataset_utils.load_label()


@distr.device(P2)
def load_feature_r2():
    import examples.python.utils.dataset_utils as dataset_utils
    return dataset_utils.load_feature(1)


def run_on_ppu():
    x1 = load_feature_r1()
    x2 = load_feature_r2()

    y = load_label()

    mlp = flax_dnn.SimpleMLP([30, 8, 1], n_epochs=10)
    mlp.log_model()

    @distr.device(PPU)
    @fe.jax2ppu
    def main(x1, x2, y):
        import jax
        x = jax.numpy.concatenate((x1, x2), axis=1)

        print("run flax train")
        params = mlp.fit_auto_grad(x, y)

        print("run flax test")
        mlp.set_params(params)
        return mlp.loss(x, y), mlp.predict(params, x), params

    loss, prediction, params = main(x1, x2, y)
    print(distr.get(loss))

    # print(distr.get(prediction))
    # print(distr.get(params))


if __name__ == '__main__':
    print('Run on CPU\n------\n')
    flax_dnn.run_on_cpu()
    print('Run on PPU\n------\n')
    run_on_ppu()
