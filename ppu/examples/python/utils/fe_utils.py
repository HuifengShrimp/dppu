
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


# This module defines helper function to convert diverse front-end to PPU iR.

from examples.python.distributed.ppu_wrapper import PpuFunction


# Convert a jax function to a PpuFunction
def jax2ppu(fn):
    class JaxPpuFunction(PpuFunction):
        def __init__(self, py_fn):
            self.py_fn = py_fn

        def jit(self, *args, **kwargs):
            import jax
            from jax.tree_util import tree_flatten

            convert_method = lambda shape, dtype: jax.numpy.zeros(shape, dtype)

            args = [self.mock_args(arg, convert_method) for arg in args]
            kwargs = {
                k: self.mock_args(v, convert_method)
                for k, v in kwargs.items()
            }

            xla, pytree = jax.xla_computation(self.py_fn,
                                              return_shape=True)(*args,
                                                                 **kwargs)

            return "JAX", (xla.as_serialized_hlo_module_proto(), pytree)

    return JaxPpuFunction(fn)


# Convert a tensorflow function to a PpuFunction
def tf2ppu(fn):
    class TensorflowPpuFunction(PpuFunction):
        def __init__(self, py_fn):
            self.py_fn = py_fn

        def jit(self, *args, **kwargs):
            import numpy as np
            import tensorflow as tf

            convert_method = lambda shape, dtype: tf.convert_to_tensor(
                np.zeros(shape, dtype))

            args = [self.mock_args(arg, convert_method) for arg in args]
            kwargs = {
                k: self.mock_args(v, convert_method)
                for k, v in kwargs.items()
            }

            fn_hat = tf.function(fn,
                                 jit_compile=True,
                                 experimental_relax_shapes=True)

            xla = fn_hat.experimental_get_compiler_ir(
                *args, **kwargs)(stage="hlo_serialized")

            cf = fn_hat.get_concrete_function(*args, **kwargs)

            return "TF", (xla, cf)

    return TensorflowPpuFunction(fn)
