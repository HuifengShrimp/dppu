## XLA Support status

List of XLA(mhlo-mlir) Ops that PPU supports:

    The list of mhlo ops is obtained from this file:
        https://github.com/tensorflow/mlir-hlo/blob/master/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td

    General limitation with PPU:
        * Dynamic shape is not supported
        * Complex number is not supported
        * PPU only supports fixed-point numbers, so no-finite is not supported

### XLA nullary ops

|  Op Name       | supported(fully/partial/no) | notes       |
| :------------: | :-------------------------: | ----------- |
| `constant`     | fully                       | Always yields a public value
| `iota`         | fully                       | Always yields a public value
| `dynamic_iota` | no                          |
| `create_token` | no                          |

Count: Total = 4, fully supported = 2

### XLA unary elementwise ops

|  Op Name       | supported(fully/partial/no) | notes       |
| :------------: | :-------------------------: | ----------- |
| `abs`          | fully                       |
| `cbrt`         | no                          |
| `ceil`         | fully                       |
| `convert`      | fully                       |
| `count_leading_zeros`| no                    |
| `cosine`       | no                          |
| `exponential`  | fully                       |
| `exponential_minus_one`| no                  |
| `floor`        | fully                       |
| `imag`         | no                          |
| `is_finite`    | no                          |
| `log`          | fully                       |
| `log_plus_one` | fully                       |
| `logistic`     | fully                       |
| `not`          | no                          | There is a bug in current not implementation
| `negate`       | fully                       |
| `popcnt`       | not                         |
| `real`         | not                         |
| `round_nearest_afz`| not                     |
| `rsqrt`        | not                         |
| `sign`         | not                         |
| `sine`         | not                         |
| `sqrt`         | fully                       |
| `tanh`         | not                         |

Count: Total = 24, fully supported = 10, partial = 0

### XLA binary elementwise ops

|  Op Name       | supported(fully/partial/no) | notes       |
| :------------: | :-------------------------: | ----------- |
| `add`          | fully                       |
| `atan2`        | no                          |
| `complex`      | no                          |
| `compare`      | fully                       |
| `divide`       | fully                       |
| `maximum`      | fully                       |
| `minimum`      | fully                       |
| `multiply`     | fully                       |
| `power`        | fully                       |
| `remainder`    | no                          |
| `shift_left`   | partial                     | rhs must be a public
| `shift_right_arithmetic` | no                |
| `shift_right_logical` | partial              | rhs must be a public
| `subtract`     | fully                       |

Count: Total = 14, fully supported = 8, partial = 2

### XLA binary logical elementwise ops

|  Op Name       | supported(fully/partial/no) | notes       |
| :------------: | :-------------------------: | ----------- |
| `and`          | fully                       |
| `or`           | fully                       |
| `xor`          | fully                       |

Count: Total = 3, fully supported = 3

### XLA communication ops

|  Op Name       | supported(fully/partial/no) | notes       |
| :------------: | :-------------------------: | ----------- |
| `infeed`       | no                          |
| `outfeed`      | no                          |
| `send`         | no                          |
| `recv`         | no                          |

Count: Total = 4, fully supported = 0

### XLA parallelism related ops

|  Op Name       | supported(fully/partial/no) | notes       |
| :------------: | :-------------------------: | ----------- |
| `replica_id`   | no                          |

Count: Total = 1, fully supported = 0

### XLA control flow ops

|  Op Name       | supported(fully/partial/no) | notes       |
| :------------: | :-------------------------: | ----------- |
| `after_all`    | no                          |
| `if`           | partial                     | condition variable must be public scalar
| `case`         | no                          |
| `while`        | partial                     | condition region must return a public scalar
| `all_gather`   | no                          |
| `all_reduce`   | no                          |
| `reduce_scatter` | no                        |
| `all_to_all`   | no                          |
| `reduce`       | fully                       | inherits limitations from reduce function

Count: Total = 9, fully supported = 1, partial = 2

### XLA tuple ops

|  Op Name       | supported(fully/partial/no) | notes       |
| :------------: | :-------------------------: | ----------- |
| `get_tuple_element` | fully                  |
| `tuple`        | fully                       |

Count: Total = 2, fully supported = 2

### XLA other ops

|  Op Name       | supported(fully/partial/no) | notes       |
| :------------: | :-------------------------: | ----------- |
| `slice`        | fully                       |
| `dynamic-slice`| no                          |
| `dynamic-update-slice`| no                   |
| `batch_norm_grad`| fully                     | Rely on XLA's batchnorm_expander pass
| `batch_norm_inference`| fully                | Rely on XLA's batchnorm_expander pass
| `batch_norm_training` | fully                | Rely on XLA's batchnorm_expander pass
| `bitcast_convert` | partial                  | Only supports convert to type of same size
| `broadcast`    | fully                       |
| `broadcast_in_dim` | no                      |
| `dynamic_broadcast_in_dim` | no              |
| `cholesky`     | fully                       | Rely on CholeskyExpander pass
| `clamp`        | fully                       |
| `concatenate`  | fully                       |
| `collective_permute` | no                    |
| `convolution`  | partial                     | Must be 2D conv with dialation == 1
| `copy`         | no                          |
| `cross-replica-sum` | no                     |
| `custom_call`  | no                          |
| `dot`          | partial                     | Up to 2D dot
| `dot_general`  | partial                     | Inherits dot limitation and rely on XLA DotDecompose pass
| `einsum`       | no                          |
| `unary_einsum` | no                          |
| `fft`          | no                          |
| `gather`       | fully                       |
| `get_dimension_size` | no                    |
| `map`          | fully                       | Rely on XLA's MapInliner pass
| `reshape`      | fully                       |
| `dynamic_reshape` | no                       |
| `scatter`      | no                          |
| `select`       | fully                       |
| `select_and_scatter` | no                    |
| `set_dimension_size` | no                    |
| `sort`         | partial                     | Inputs must be public tensors
| `reverse`      | fully                       |
| `pad`          | fully                       |
| `trace`        | no                          |
| `transpose`    | partial                     | General transpose is not supported, permutation must be a reverse of input dimension index
| `triangular_solve` | fully                   | Rely on XLA's TriangularSolverExpander pass
| `reduce_window`| partial                     | Variadic reduce window is not supported
| `return`       | fully                       |
| `torch_index_select` | no                    |
| `optimization_barrier` | no                  |

Count: Total = 42, fully supported = 16, partial = 7

### XLA RNG ops

|  Op Name       | supported(fully/partial/no) | notes       |
| :------------: | :-------------------------: | ----------- |
| `rng_uniform`  | partial                     | Bound [a, b) must all be public scalar, result is also a public tensor
| `rng_normal`   | no                          |
| `rng_bit_generator` | no                     |

Count: Total = 3, fully supported = 0, partial = 1

### XLA quantize op

|  Op Name       | supported(fully/partial/no) | notes       |
| :------------: | :-------------------------: | ----------- |
| `dequantize`   | no                          |

Count: Total = 1, fully supported = 0, partial = 0

### XLA miscellaneous ops

|  Op Name       | supported(fully/partial/no) | notes       |
| :------------: | :-------------------------: | ----------- |
| `fusion`       | no                          |
| `bitcast`      | no                          | Internal op to XLA/GPU
| `reduce_precision` | no                      |
| `real_dynamic_slice` | no                    |
| `dynamic_pad`  | no                          |
| `dynamic_gather` | no                        |
| `dynamic_conv` | no                          |
| `print`        | no                          |
| `compute_reshape_shape` | no                 |
| `cstr_reshapable` | no                       |

Count: Total = 10, fully supported = 0, partial = 0
