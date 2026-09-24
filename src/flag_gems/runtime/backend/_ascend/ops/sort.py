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
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.topk import _get_finfo_val, _get_iinfo_val
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit()
def sort_kernel(
    in_ptr,
    out_ptr,
    out_index_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    offset = tl.program_id(0) * N + cols
    in_ptr += offset
    out_ptr += offset
    out_index_ptr += offset

    if IS_FLOAT:
        mask_val = _get_finfo_val(in_ptr.dtype.element_ty, return_max=not DESCENDING)
        in_val = tl.load(in_ptr, mask=mask, other=mask_val)
        in_val = tl.where(in_val.dtype.is_fp64(), in_val, in_val.to(tl.float32))

        # Convert float bits to monotonically sortable uint32 key
        in_bits = in_val.to(tl.uint32, bitcast=True)
        sign_mask = tl.full(in_bits.shape, 0x80000000, dtype=tl.uint32)
        sortable_key = tl.where(
            (in_bits & sign_mask) != 0, ~in_bits, in_bits ^ sign_mask
        )
    else:
        mask_val = _get_iinfo_val(in_ptr.dtype.element_ty, return_max=not DESCENDING)
        in_val = tl.load(in_ptr, mask=mask, other=mask_val).to(tl.int32)

        # For signed int32, flip MSB to get monotonically sortable uint32 key
        in_bits = in_val.to(tl.uint32, bitcast=True)
        sign_mask = tl.full(in_bits.shape, 0x80000000, dtype=tl.uint32)
        sortable_key = in_bits ^ sign_mask

    # XOR MSB so that unsigned key ordering matches signed int32 ordering
    # (required because tl.sort on Ascend uses signed comparison for int64)
    adjusted_key = (sortable_key ^ sign_mask).to(tl.int32)

    # Pack (key, index) into a single int64 so tl.sort handles both.
    # For descending, negate the index so that ties break by smaller
    # original index (matching torch.sort convention).
    if DESCENDING:
        idx_val = (BLOCK_SIZE - 1 - cols).to(tl.int32)
    else:
        idx_val = cols.to(tl.int32)

    packed = (adjusted_key.to(tl.int64) << 32) | (idx_val.to(tl.int64) & 0xFFFFFFFF)

    sorted_packed = tl.sort(packed, dim=0, descending=DESCENDING)

    # Unpack
    sorted_key_i32 = (sorted_packed >> 32).to(tl.int32)
    sorted_index_raw = (sorted_packed & 0xFFFFFFFF).to(tl.int64)

    if DESCENDING:
        sorted_index = (BLOCK_SIZE - 1 - sorted_index_raw).to(tl.int64)
    else:
        sorted_index = sorted_index_raw

    # Recover original value from sortable key
    recovered_sortable = sorted_key_i32.to(tl.uint32) ^ sign_mask
    if IS_FLOAT:
        recovered_bits = tl.where(
            (recovered_sortable & sign_mask) != 0,
            recovered_sortable ^ sign_mask,
            ~recovered_sortable,
        )
        sorted_val = recovered_bits.to(tl.float32, bitcast=True)
    else:
        recovered_bits = recovered_sortable ^ sign_mask
        sorted_val = recovered_bits.to(tl.int32, bitcast=True)

    tl.store(out_ptr, sorted_val, mask=mask)
    tl.store(out_index_ptr, sorted_index, mask=mask)


def sort(inp, dim=-1, descending=False):
    logger.debug("GEMS_ASCEND SORT")
    sort_elem_cnt = inp.shape[dim]
    if sort_elem_cnt == 1:
        return inp, torch.zeros_like(inp, dtype=torch.int64)
    elif sort_elem_cnt > 128:  # TODO: Optimize implementation for large cases.
        return torch.sort(inp, stable=False, dim=dim, descending=descending)
    block_size = triton.next_power_of_2(sort_elem_cnt)

    if dim < 0:
        dim = dim + inp.ndim
    if dim != inp.ndim - 1:
        inp = torch.movedim(inp, dim, -1).contiguous()
    else:
        inp = inp.contiguous()
    batch_size = math.prod(inp.shape) // sort_elem_cnt

    out = torch.empty_like(inp)
    out_index = torch.empty_like(inp, dtype=torch.int64)

    with torch_device_fn.device(inp.device):
        sort_kernel[batch_size,](
            inp,
            out,
            out_index,
            N=sort_elem_cnt,
            BLOCK_SIZE=block_size,
            DESCENDING=descending,
            IS_FLOAT=inp.is_floating_point(),
        )

    if dim != inp.ndim - 1:
        out = torch.movedim(out, -1, dim)
        out_index = torch.movedim(out_index, -1, dim)
    return out, out_index
