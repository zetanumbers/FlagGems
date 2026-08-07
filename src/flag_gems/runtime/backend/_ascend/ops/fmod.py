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


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def fmod_func(x, y):
    dtype = x.dtype
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    quotient = x_fp32 / y_fp32
    quotient = tl.where(quotient >= 0, tl.floor(quotient), tl.ceil(quotient))
    result = x_fp32 - y_fp32 * quotient
    # Two-pass correction for NPU approximate division
    abs_y = tl.abs(y_fp32)
    result = tl.where(result >= abs_y, result - abs_y, result)
    result = tl.where(result <= -abs_y, result + abs_y, result)
    # Second pass for edge cases where first pass misses
    q2 = result / y_fp32
    q2 = tl.where(q2 >= 0, tl.floor(q2), tl.ceil(q2))
    result = result - y_fp32 * q2
    return result.to(dtype)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def fmod_func_tensor_scalar(x, y):
    dtype = x.dtype
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    quotient = x_fp32 / y_fp32
    quotient = tl.where(quotient >= 0, tl.floor(quotient), tl.ceil(quotient))
    result = x_fp32 - y_fp32 * quotient
    # Correction for NPU approximate division
    abs_y = tl.abs(y_fp32)
    # Boundary correction: if |result| >= |y|, quotient was off by 1
    result = tl.where(result >= abs_y, result - abs_y, result)
    result = tl.where(result <= -abs_y, result + abs_y, result)
    # Sign correction: fmod result must have same sign as x (or be 0)
    # If result has opposite sign, quotient was off by 1 in the other direction
    result = tl.where((x_fp32 > 0) & (result < 0), result + abs_y, result)
    result = tl.where((x_fp32 < 0) & (result > 0), result - abs_y, result)
    return result.to(dtype)


def fmod_tensor(A, B):
    logger.debug("GEMS_ASCEND FMOD_TENSOR")
    return fmod_func(A, B)


def fmod_scalar(A, B):
    logger.debug("GEMS_ASCEND FMOD_SCALAR")
    return fmod_func_tensor_scalar(A, B)


def fmod_tensor_(A, B):
    logger.debug("GEMS_ASCEND FMOD_TENSOR_")
    return fmod_func(A, B, out0=A)


def fmod_scalar_(A, B):
    logger.debug("GEMS_ASCEND FMOD_SCALAR_")
    return fmod_func_tensor_scalar(A, B, out0=A)
