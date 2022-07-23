
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


# Start two node servers.
# > bazel run //examples/python/distributed:daemon -- up
#
# Run this example script.
# > bazel run //examples/python/e2e:tf_lr_ppu

from math import dist
import jax
import tensorflow as tf

import examples.python.distributed as distr
import examples.python.utils.dataset_utils as dataset_utils
import examples.python.e2e.tf_lr as tf_lr
import examples.python.utils.fe_utils as fe
from examples.python.distributed import P1, P2, PPU
from sklearn import metrics

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

    w = distr.device(PPU)(fe.tf2ppu(
        tf.autograph.experimental.do_not_convert(
            tf_lr.LogitRegression().fit_auto_grad)))(x1, x2, y)

    print(w)
    w_ = distr.get(w)
    print(w_)

    x1, x2, y = dataset_utils.load_feature(0), dataset_utils.load_feature(
        1), dataset_utils.load_label()

    print('AUC={}'.format(
        metrics.roc_auc_score(
            y, tf_lr.predict(tf.concat((x1, x2), axis=1),
                             tf.cast(w_, tf.double)))))


if __name__ == "__main__":
    print('Run on CPU\n------\n')
    tf_lr.run_on_cpu()

    print('Run on PPU\n------\n')
    run_on_ppu()
