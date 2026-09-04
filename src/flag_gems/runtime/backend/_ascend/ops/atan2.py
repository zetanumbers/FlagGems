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

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_atan2 = tl_extra_shim.atan2

logger = logging.getLogger(__name__)


@triton.jit
def _atan2_safe(x, y):
    PI = 3.141592653589793
    HALF_PI = 1.5707963267948966
    INF = float("inf")

    result = _atan2(x, y)

    x_inf = tl.abs(x) == INF
    y_inf = tl.abs(y) == INF
    x_pos = x > 0
    y_pos = y > 0

    result = tl.where(x_inf & x_pos & ~y_inf, HALF_PI, result)
    result = tl.where(x_inf & ~x_pos & ~y_inf, -HALF_PI, result)

    result = tl.where(x_inf & x_pos & y_inf & y_pos, PI / 4, result)
    result = tl.where(x_inf & x_pos & y_inf & ~y_pos, 3 * PI / 4, result)
    result = tl.where(x_inf & ~x_pos & y_inf & y_pos, -PI / 4, result)
    result = tl.where(x_inf & ~x_pos & y_inf & ~y_pos, -3 * PI / 4, result)

    # Fix: Ascend NPU atan2 treats -0 as +0, causing sign errors near the
    # negative x-axis (off by 2*PI). Detect negative-zero via the sign bit
    # and force the IEEE 754 compliant result.
    x_bits = x.to(tl.int32, bitcast=True)
    y_bits = y.to(tl.int32, bitcast=True)
    x_is_neg_zero = (x == 0.0) & (x_bits < 0)
    y_is_neg_zero = (y == 0.0) & (y_bits < 0)
    y_is_neg = (y < 0) | y_is_neg_zero
    # atan2(-0, y<0 or y=-0) should be -PI, not +PI
    result = tl.where(x_is_neg_zero & y_is_neg, -PI, result)
    # atan2(+0, -0) should be +PI, not 0
    result = tl.where(y_is_neg_zero & (x == 0.0) & ~x_is_neg_zero, PI, result)

    result = tl.where((x != x) | (y != y), float("nan"), result)

    return result


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def atan2_kernel(x, y):
    return _atan2_safe(x.to(tl.float32), y.to(tl.float32))


def atan2(input, other):
    logger.debug("GEMS_ASCEND ATAN2")
    return atan2_kernel(input, other)


def atan2_out(input, other, out):
    logger.debug("GEMS_ASCEND ATAN2_OUT")
    return atan2_kernel(input, other, out0=out)
