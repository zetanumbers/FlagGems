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

import random

import pytest
import torch

import flag_gems

from . import conftest as cfg

random.seed(42)

NUM_SMS = 80


def _get_paged_mqa_logits_metadata_ref(context_lens, block_size, num_sms):
    """Native PyTorch reference for get_paged_mqa_logits_metadata.

    Mirrors the Triton kernel logic: ceil-divide each context length by
    SPLIT_KV (256), inclusive-cumsum to get per-batch segment boundaries,
    then evenly partition total segments across SMs.
    """
    SPLIT_KV = 256
    device = context_lens.device

    if context_lens.dim() == 2:
        batch_size, next_n = context_lens.shape
        ctx_lens = context_lens[:, next_n - 1]
    else:
        batch_size = context_lens.shape[0]
        ctx_lens = context_lens

    if batch_size == 0:
        return torch.zeros((num_sms + 1, 2), dtype=torch.int32, device=device)

    num_segs = (ctx_lens + SPLIT_KV - 1) // SPLIT_KV
    prefix_sum = torch.cumsum(num_segs, dim=0).to(torch.int32)
    total_segs = int(prefix_sum[-1].item())

    q, r = divmod(total_segs, num_sms)

    schedule_metadata = torch.zeros((num_sms + 1, 2), dtype=torch.int32, device=device)
    for sm_idx in range(num_sms + 1):
        seg_starts = sm_idx * q + min(sm_idx, r)

        is_le = prefix_sum <= seg_starts
        q_idx = int(is_le.sum().item())

        if q_idx > 0:
            prev_prefix = int(prefix_sum[:q_idx].max().item())
        else:
            prev_prefix = 0
        kv_split_idx = seg_starts - prev_prefix

        schedule_metadata[sm_idx, 0] = q_idx
        schedule_metadata[sm_idx, 1] = kv_split_idx

    return schedule_metadata


# Shape configs for QUICK_MODE
if cfg.QUICK_MODE:
    BATCH_NEXTN_SHAPES = [(4, 1)]
else:
    BATCH_NEXTN_SHAPES = [(4, 1), (2, 2)]


@pytest.mark.get_paged_mqa_logits_metadata
@pytest.mark.parametrize("batch_size, next_n", BATCH_NEXTN_SHAPES)
@pytest.mark.parametrize("avg_ctx_len", [1024, 2048])
def test_get_paged_mqa_logits_metadata(batch_size, next_n, avg_ctx_len):
    context_lens_2d = (
        torch.randint(
            int(0.8 * avg_ctx_len), int(1.2 * avg_ctx_len), (batch_size, next_n)
        )
        .to(flag_gems.device)
        .to(torch.int32)
    )

    ref = _get_paged_mqa_logits_metadata_ref(context_lens_2d, 64, NUM_SMS)
    res = flag_gems.get_paged_mqa_logits_metadata(context_lens_2d, 64, NUM_SMS)

    assert torch.equal(ref, res)
