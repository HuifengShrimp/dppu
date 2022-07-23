
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
# > bazel run //examples/python/stats:pvalue

import jax

from jax import numpy as jnp
from scipy.stats import norm

import examples.python.distributed as distr
import examples.python.utils.dataset_utils as dataset_utils
import examples.python.utils.fe_utils as fe
from examples.python.distributed import P1, P2, PPU

import argparse

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/2pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = distr.config.parse_json(file.read())
    distr.init(conf)


def sigmoid(x):
    # return 0.5 * (jnp.tanh(x / 2) + 1)
    return 1 / (1 + jnp.exp(-x))


def predict(x, w, b):
    return sigmoid(jnp.matmul(x, w) + b)


# jax.numpy.inv is not working for now, use newton method to simulate it.
def newton_matrix_inv(x, round=25):
    assert x.shape[0] == x.shape[1]
    assert round > 0
    E = jnp.eye(x.shape[0])
    A = E * (1 / x.trace())
    for _ in range(round):
        A = jnp.matmul(A, (2 * E - jnp.matmul(x, A)))

    return A


@distr.device(P1)
def load_feature_r1():
    import examples.python.utils.dataset_utils as dataset_utils

    x = dataset_utils.standardize(dataset_utils.load_feature(0))
    # append ones col for base
    return dataset_utils.add_constant_col(x)


@distr.device(P2)
def load_feature_r2():
    import examples.python.utils.dataset_utils as dataset_utils

    return dataset_utils.standardize(dataset_utils.load_feature(1))


# from @nebula:task_stats_pvalue_test.cc
@distr.device(P1)
def load_weight_r1():
    return jnp.array([[
        -0.330207439,
        0.137967204,
        0.574166151,
        -0.332886617,
        0.142702923,
        0.113152234,
        -0.484829124,
        -0.324604116,
        -0.822246556,
        0.342770452,
        -0.036922242,
        1.177567015,
        -0.319786008,
        -0.117482498,
        0.304628396,
        0.603964821,  # base
    ]]).T


@distr.device(P2)
def load_weight_r2():
    return jnp.array([[
        -0.082062528,
        0.161846416,
        0.313125998,
        0.416683549,
        0.35770503,
        -0.323749637,
        -0.190857702,
        -0.395735305,
        -0.241595546,
        -0.030582347,
        -0.314795237,
        -0.273756992,
        -0.325732314,
        -0.215765326,
        0.134412119,
    ]]).T


def run_dist():
    x1 = load_feature_r1()
    x2 = load_feature_r2()
    w1 = load_weight_r1()
    w2 = load_weight_r2()

    @distr.device(PPU)
    @fe.jax2ppu
    def sspvalue(x1, x2, w1, w2):
        # pvalue for logit lr
        x = jax.numpy.concatenate((x1, x2), axis=1)
        w = jax.numpy.concatenate((w1, w2), axis=0)
        y_hat = sigmoid(jnp.matmul(x, w))
        a_diagonal = (y_hat * (1 - y_hat)).flatten()
        H = jnp.matmul((x.transpose() * a_diagonal), x)
        # jax.numpy.linalg.inv break ppu.binding.compile
        # use newton method instead.
        H_inv = newton_matrix_inv(H).diagonal().reshape(31, 1)
        return jnp.square(w) / H_inv

    z_square = distr.get(sspvalue(x1, x2, w1, w2))
    z = jnp.sqrt(z_square)
    pvalue = 2 * (1 - norm.cdf(z))
    print(pvalue.T)


def run_origin():
    x, _ = dataset_utils.load_full_dataset()
    x = dataset_utils.standardize(x)
    # from @nebula:task_stats_pvalue_test.cc
    b = 0.603964821
    w = jnp.array([[
        -0.330207439,
        0.137967204,
        0.574166151,
        -0.332886617,
        0.142702923,
        0.113152234,
        -0.484829124,
        -0.324604116,
        -0.822246556,
        0.342770452,
        -0.036922242,
        1.177567015,
        -0.319786008,
        -0.117482498,
        0.304628396,
        -0.082062528,
        0.161846416,
        0.313125998,
        0.416683549,
        0.35770503,
        -0.323749637,
        -0.190857702,
        -0.395735305,
        -0.241595546,
        -0.030582347,
        -0.314795237,
        -0.273756992,
        -0.325732314,
        -0.215765326,
        0.134412119,
    ]]).T

    x_ones = jnp.c_[x, jnp.ones((x.shape[0], 1))]
    y_hat = predict(x, w, b)
    a_diagonal = (y_hat * (1 - y_hat)).flatten()
    H = jnp.matmul((x_ones.transpose() * a_diagonal), x_ones)
    H_inv = jnp.linalg.inv(H).diagonal().reshape(31, 1)
    wb = jnp.append(w, b).reshape(31, 1)
    z_square = jnp.square(wb) / H_inv
    z = jnp.sqrt(z_square)
    pvalue = 2 * (1 - norm.cdf(z))
    print(pvalue.T)


if __name__ == "__main__":
    # run_origin()
    run_dist()
