# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")]
)
@triton.jit
def leaky_relu_kernel(x, negative_slope):
    return tl.maximum(x, 0.0) + negative_slope * tl.minimum(x, 0.0)


def leaky_relu(A, negative_slope=0.01):
    logger.debug("GEMS_ASCEND LEAKY_RELU")
    return leaky_relu_kernel(A, negative_slope)


def leaky_relu_(A, negative_slope=0.01):
    logger.debug("GEMS_ASCEND LEAKY_RELU_")
    return leaky_relu_kernel(A, negative_slope, out0=A)


def leaky_relu_out(A, negative_slope=0.01, *, out=None):
    logger.debug("GEMS_ASCEND LEAKY_RELU_OUT")
    if out is None:
        return leaky_relu_kernel(A, negative_slope)
    return leaky_relu_kernel(A, negative_slope, out0=out)
