
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


import jax
import jax.numpy as jnp
from jax import grad, jit, lax, random, vmap


def print_xla(f, *args, **kwargs):
    c = jax.xla_computation(f)(*args, **kwargs)
    print(c.as_hlo_text())


####################################################################
# simple
####################################################################
def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


key = random.PRNGKey(0)
x = random.normal(key, (100, ))

# print_xla(selu, x)


####################################################################
# recursive jit/grad
####################################################################
def sum_logistic(x):
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))


x_small = jnp.arange(3.)
derivative_fn = grad(sum_logistic)

# print_xla(derivative_fn, x_small)
# print_xla(grad(jit(grad(jit(grad(sum_logistic))))), 1.0)


####################################################################
# python control flow
####################################################################
@jit
def g(x):
    y = 0.
    for i in range(x.shape[0]):
        y = y + x[i]
    return y


# print(g(jnp.array([1., 2., 3.])))
# print_xla(g, x)


####################################################################
# lax control flow
####################################################################
def g(num):
    init_val = 0
    cond_fun = lambda x: x < num
    body_fun = lambda x: x + 1
    g = lax.while_loop(cond_fun, body_fun, init_val)


# print_xla(g, 10)

####################################################################
# capture
####################################################################
capture = 10


def g(x):
    return x + capture


print_xla(g, 15)
