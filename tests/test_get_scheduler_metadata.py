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

import math

import pytest
import torch

import flag_gems
from flag_gems.ops.get_scheduler_metadata import (
    get_num_splits,
    get_optimal_block_mn,
    get_pack_gqa,
    get_pagedkv_tma,
    round_up_headdim,
    round_up_headdimv,
    tile_size_fwd_sm90,
    use_one_mma_wg,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.device_info import get_device_capability, get_device_info

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device

# Shape configs for QUICK_MODE
if cfg.QUICK_MODE:
    BATCH_SIZE_LIST = [1, 8]
    MAX_SEQLEN_K_LIST = [512]
else:
    BATCH_SIZE_LIST = [1, 8, 256, 512]
    MAX_SEQLEN_K_LIST = [512, 2048]


def _get_scheduler_metadata_ref(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads,
    num_heads_k,
    headdim,
    headdim_v,
    qkv_dtype,
    seqused_k,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    cu_seqlens_k_new=None,
    seqused_q=None,
    leftpad_k=None,
    page_size=None,
    max_seqlen_k_new=0,
    is_causal=False,
    window_size_left=-1,
    window_size_right=-1,
    has_softcap=False,
    num_splits=0,
    pack_gqa=None,
    sm_margin=0,
):
    """Native PyTorch reference mirroring flag_gems.get_scheduler_metadata.

    Replaces the two Triton kernels (_prepare_pass1_kernel and
    _prepare_pass2_kernel) with equivalent PyTorch tensor operations.
    """
    device_ = seqused_k.device
    dtype = torch.int32

    supported_dtypes = (torch.half, torch.bfloat16)
    assert qkv_dtype in supported_dtypes
    assert num_heads % num_heads_k == 0

    effective_is_causal = is_causal
    effective_window_left = window_size_left if window_size_left >= 0 else -1
    effective_window_right = window_size_right

    if effective_window_left >= max_seqlen_k - 1:
        effective_window_left = -1
    if effective_window_right >= max_seqlen_q - 1:
        effective_window_right = -1

    if (
        max_seqlen_q == 1
        and effective_window_left == -1
        and effective_window_right == -1
    ):
        if (headdim <= 64 or headdim > 128) or page_size is None:
            effective_is_causal = False

    if effective_is_causal:
        effective_window_right = 0

    final_is_causal = effective_window_left < 0 and effective_window_right == 0
    final_is_local = (
        effective_window_left >= 0 or effective_window_right >= 0
    ) and not final_is_causal

    major, minor = get_device_capability()
    arch = major * 10 + minor
    num_sm = get_device_info().sm_count - sm_margin

    softcap = 1.0 if has_softcap else 0.0
    element_size = qkv_dtype.itemsize
    has_page_table = page_size is not None
    d_rounded = round_up_headdim(headdim)
    dv_rounded = round_up_headdimv(headdim_v)

    pagedkv_tma = get_pagedkv_tma(
        arch=arch,
        page_size=page_size if page_size is not None else 1,
        has_page_table=has_page_table,
        leftpad_k=leftpad_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k_new=max_seqlen_k_new,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        d_rounded=d_rounded,
        dv_rounded=dv_rounded,
        is_causal=final_is_causal,
        is_local=final_is_local,
        element_size=element_size,
        softcap=has_softcap,
    )

    blockM, blockN = get_optimal_block_mn(
        device=device_,
        headdim=headdim,
        headdim_v=headdim_v,
        is_causal=final_is_causal,
        is_local=final_is_local,
        has_softcap=has_softcap,
        element_size=element_size,
        paged_kv=has_page_table,
        pagedkv_tma=pagedkv_tma,
    )

    varlen_q_flag = cu_seqlens_q is not None or seqused_q is not None
    pack_gqa = (
        pack_gqa
        if pack_gqa is not None
        else get_pack_gqa(
            arch=arch,
            has_page_table=has_page_table,
            pagedkv_tma=pagedkv_tma,
            num_splits=num_splits,
            num_heads=num_heads,
            num_heads_k=num_heads_k,
            varlen_q=varlen_q_flag,
            seqlen_q=max_seqlen_q,
            d_rounded=d_rounded,
            dv_rounded=dv_rounded,
            is_causal=final_is_causal,
            is_local=final_is_local,
            element_size=element_size,
            softcap=has_softcap,
        )
    )
    qhead_per_khead = (
        1 if not pack_gqa else (num_heads + num_heads_k - 1) // num_heads_k
    )
    num_head_k = num_heads_k if pack_gqa else num_heads

    # ---- pass1 kernel in pure PyTorch ----
    if seqused_q is not None:
        q_len = seqused_q.to(dtype)
    elif cu_seqlens_q is not None:
        q_len = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    else:
        q_len = torch.full((batch_size,), max_seqlen_q, dtype=dtype, device=device_)
    q_len = q_len * qhead_per_khead
    num_m_blocks = (q_len + blockM - 1) // blockM

    # k_len
    k_len = seqused_k.to(dtype)
    if leftpad_k is not None:
        k_len = k_len - leftpad_k.to(dtype)
    if max_seqlen_k_new > 0:
        if cu_seqlens_k_new is not None:
            k_len = k_len + (cu_seqlens_k_new[1:] - cu_seqlens_k_new[:-1])
        else:
            k_len = k_len + max_seqlen_k_new
    num_n_blocks = (k_len + blockN - 1) // blockN

    total_blocks_val = int((num_m_blocks * num_n_blocks).sum().item())

    use_dynamic_split = batch_size <= 992

    if num_splits <= 0:
        eff_num_splits = get_num_splits(
            batch_size=batch_size,
            num_heads=num_heads,
            num_heads_k=num_heads_k,
            headdim=headdim,
            headdim_v=headdim_v,
            d_rounded=d_rounded,
            dv_rounded=dv_rounded,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_k_new=max_seqlen_k_new,
            arch=arch,
            num_sm=num_sm,
            is_causal=final_is_causal,
            is_local=final_is_local,
            has_softcap=softcap,
            is_varlen=True,
            has_page_table=has_page_table,
            pack_gqa=pack_gqa,
            window_size_left=effective_window_left,
            window_size_right=effective_window_right,
            element_size=element_size,
            use_dynamic_split=use_dynamic_split,
        )
    else:
        eff_num_splits = num_splits

    eff_num_splits = min(eff_num_splits, 256, num_sm)
    pack_gqa = True if eff_num_splits > 1 else pack_gqa

    qhead_per_khead = (
        1 if not pack_gqa else (num_heads + num_heads_k - 1) // num_heads_k
    )
    num_head_k = num_heads_k if pack_gqa else num_heads

    is_varlen = True
    if arch >= 90:
        uomw = use_one_mma_wg(
            arch=arch,
            headdim=headdim,
            seqlen_q=max_seqlen_q,
            pack_gqa=pack_gqa,
            num_heads=num_heads,
            num_heads_k=num_heads_k,
        )
        blockM, blockN = tile_size_fwd_sm90(
            headdim=round_up_headdim(headdim),
            headdim_v=round_up_headdimv(headdim_v),
            is_causal=final_is_causal,
            is_local=final_is_local,
            element_size=element_size,
            v_colmajor=False,
            paged_kv_non_TMA=(has_page_table and not pagedkv_tma),
            softcap=has_softcap,
            use_one_mma_wg=uomw,
        )
    else:
        blockM, blockN = get_optimal_block_mn(
            device=device_,
            headdim=headdim,
            headdim_v=headdim_v,
            is_causal=final_is_causal,
            is_local=final_is_local,
            has_softcap=has_softcap,
            element_size=element_size,
            paged_kv=has_page_table,
            pagedkv_tma=pagedkv_tma,
            varlen_and_split=is_varlen and (eff_num_splits > 1),
            append_kv=(max_seqlen_k_new > 0),
        )

    # Recompute num_m_blocks / num_n_blocks with final blockM/blockN
    num_m_blocks = (q_len + blockM - 1) // blockM
    num_n_blocks = (k_len + blockN - 1) // blockN
    total_blocks_val = int((num_m_blocks * num_n_blocks).sum().item())

    # ---- pass2 kernel in pure PyTorch ----
    if use_dynamic_split:
        blocks_per_sm = math.ceil(total_blocks_val * 1.1 * num_head_k / num_sm)
        blocks_per_sm = max(1, blocks_per_sm)
        num_splits_dynamic = (num_n_blocks + blocks_per_sm - 1) // blocks_per_sm
        num_splits_dynamic = torch.clamp(num_splits_dynamic, min=1, max=eff_num_splits)
    else:
        num_splits_dynamic = torch.full(
            (batch_size,), eff_num_splits, dtype=dtype, device=device_
        )

    final_num_splits = eff_num_splits
    scheduler_needs_semaphore = arch >= 90 or final_num_splits > 1

    alloc_size = int(scheduler_needs_semaphore) + int(use_dynamic_split) * batch_size

    if alloc_size > 0:
        scheduler_metadata = torch.empty(alloc_size, dtype=torch.int32, device=device_)
        offset = 0
        if scheduler_needs_semaphore:
            scheduler_metadata[offset] = 0
            offset += 1
        if use_dynamic_split:
            scheduler_metadata[offset:] = num_splits_dynamic.to(torch.int32)
        return scheduler_metadata
    else:
        return torch.empty((0,), dtype=torch.int32, device=device_)


@pytest.mark.get_scheduler_metadata
@pytest.mark.skipif(cfg.TO_CPU, reason="Skipping correctness test in CPU mode.")
@pytest.mark.parametrize("batch_size", BATCH_SIZE_LIST)
@pytest.mark.parametrize("max_seqlen_k", MAX_SEQLEN_K_LIST)
@pytest.mark.parametrize("headdim", [64, 128])
@pytest.mark.parametrize("num_splits_static", [0, 4])
@pytest.mark.parametrize("seed", [42])
def test_scheduler_metadata_correctness(
    batch_size, max_seqlen_k, headdim, num_splits_static, seed
):
    device = torch_device_fn.current_device()
    utils.init_seed(seed)

    seqused_k = torch.randint(
        1, max_seqlen_k, (batch_size,), dtype=torch.int32, device=device
    )
    num_heads, num_heads_k = 32, 8
    headdim_v = headdim
    qkv_dtype = torch.float16

    ref_metadata = _get_scheduler_metadata_ref(
        batch_size=batch_size,
        max_seqlen_q=1,
        max_seqlen_k=max_seqlen_k,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        headdim=headdim,
        headdim_v=headdim_v,
        qkv_dtype=qkv_dtype,
        seqused_k=seqused_k,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        cu_seqlens_k_new=None,
        seqused_q=None,
        leftpad_k=None,
        page_size=None,
        max_seqlen_k_new=0,
        is_causal=False,
        window_size_left=-1,
        window_size_right=-1,
        has_softcap=False,
        num_splits=num_splits_static,
        pack_gqa=True,
        sm_margin=0,
    )

    with flag_gems.use_gems():
        gems_metadata = flag_gems.get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            num_heads=num_heads,
            num_heads_k=num_heads_k,
            headdim=headdim,
            headdim_v=headdim_v,
            qkv_dtype=qkv_dtype,
            seqused_k=seqused_k,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            cu_seqlens_k_new=None,
            seqused_q=None,
            leftpad_k=None,
            page_size=None,
            max_seqlen_k_new=0,
            is_causal=False,
            window_size_left=-1,
            window_size_right=-1,
            has_softcap=False,
            num_splits=num_splits_static,
            pack_gqa=True,
            sm_margin=0,
        )

    utils.gems_assert_close(gems_metadata, ref_metadata, dtype=torch.int32)
