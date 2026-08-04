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

import torch

from flag_gems.runtime import torch_device_fn

from .full import (
    SMALL_TENSOR_THRESHOLD,
    check_dtype,
    full_func,
    full_func_scalar,
    full_kernel
)

logger = logging.getLogger(__name__)


def full_like(
    x,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    logger.debug("GEMS_ASCEND FULL_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    fill_value = check_dtype(fill_value, dtype, device)
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    if N == 0:
        return out
    if N < SMALL_TENSOR_THRESHOLD:
        # Small tensor: use pointwise_dynamic for lower launch overhead
        if isinstance(fill_value, torch.Tensor):
            return full_func(out, fill_value, out0=out)
        else:
            return full_func_scalar(out, fill_value, out0=out)
    # Large tensor: use hand-written multi-core kernel
    if isinstance(fill_value, torch.Tensor):
        fill_value = fill_value.item()
    grid = min(40, N)
    BLOCK_SIZE = (N + grid - 1) // grid
    SUBBLOCK_SIZE = min(8192, BLOCK_SIZE)

    with torch_device_fn.device(device):
        full_kernel[grid,](out, N, fill_value, BLOCK_SIZE, SUBBLOCK_SIZE)
    return out
