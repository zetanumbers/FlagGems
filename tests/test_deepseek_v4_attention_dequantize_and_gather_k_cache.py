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

import pytest
import torch

import flag_gems
import flag_gems.testing as fg_testing
from flag_gems.fused.deepseek_v4_attention_dequantize_and_gather_k_cache import (
    dequantize_and_gather_k_cache,
)
from flag_gems.utils.device_info import get_device_capability

try:
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        dequantize_and_gather_k_cache as vllm_dequantize_and_gather_k_cache,
    )

    _HAS_VLLM_DEQUANTIZE_AND_GATHER_K_CACHE = True
except Exception:
    vllm_dequantize_and_gather_k_cache = None
    _HAS_VLLM_DEQUANTIZE_AND_GATHER_K_CACHE = False


def is_support_fp8e4nv():
    major, minor = get_device_capability()
    return major * 10 + minor >= 89


def _fill_cache(k_cache, expected_rows, block_size, nope_dim, rope_dim, scale_slots):
    token_data_size = nope_dim + rope_dim * 2
    for slot, row in expected_rows.items():
        block = slot // block_size
        pos = slot % block_size
        base = pos * token_data_size
        x = (
            torch.arange(nope_dim, device=k_cache.device, dtype=torch.float32) / 32.0
            + slot / 8.0
        ).to(torch.float8_e4m3fn)
        rope = (
            torch.arange(rope_dim, device=k_cache.device, dtype=torch.float32) / 16.0
            + slot
        ).to(torch.bfloat16)
        k_cache[block, base : base + nope_dim].copy_(x.view(torch.uint8))
        k_cache[block, base + nope_dim : base + nope_dim + rope_dim * 2].copy_(
            rope.view(torch.uint8)
        )
        scale_base = block_size * token_data_size + pos * scale_slots
        k_cache[block, scale_base : scale_base + scale_slots] = 127
        row[..., :nope_dim] = x.to(torch.float32).to(torch.bfloat16)
        row[..., nope_dim : nope_dim + rope_dim] = rope


@pytest.mark.parametrize(
    ("batch", "seq_len", "gather_len", "block_size", "nope_dim", "rope_dim"),
    [
        (1, 6, 3, 4, 64, 16),
        (2, 12, 5, 64, 448, 64),
    ],
)
@pytest.mark.dequantize_and_gather_k_cache
def test_dequantize_and_gather_k_cache_accuracy(
    batch, seq_len, gather_len, block_size, nope_dim, rope_dim
):
    device = flag_gems.device
    scale_slots = (nope_dim + 63) // 64 + (1 if nope_dim % 64 == 0 else 0)
    output_dim = nope_dim + rope_dim
    token_data_size = nope_dim + rope_dim * 2
    block_stride = block_size * token_data_size + block_size * scale_slots
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = batch * blocks_per_seq
    k_cache = torch.zeros((num_blocks, block_stride), device=device, dtype=torch.uint8)
    out = torch.empty(
        (batch, gather_len, output_dim), device=device, dtype=torch.bfloat16
    )
    expected = torch.empty_like(out)
    rows = {}
    for req in range(batch):
        start_pos = seq_len - gather_len
        for local_i in range(gather_len):
            pos = start_pos + local_i
            physical_block = req * blocks_per_seq + pos // block_size
            slot = physical_block * block_size + pos % block_size
            rows[slot] = expected[req : req + 1, local_i : local_i + 1, :]
    _fill_cache(k_cache, rows, block_size, nope_dim, rope_dim, scale_slots)

    seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    gather_lens = torch.full((batch,), gather_len, device=device, dtype=torch.int32)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(
        batch, blocks_per_seq
    )
    dequantize_and_gather_k_cache(
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        rope_dim=rope_dim,
        nope_dim=nope_dim,
        scale_slots=scale_slots,
    )

    fg_testing.assert_close(out, expected, dtype=torch.bfloat16, equal_nan=True)


def _native_dequantize_and_gather_k_cache(
    out, k_cache, seq_lens, gather_lens, block_table, block_size, offset,
    rope_dim=64, nope_dim=None, scale_slots=None,
):
    """Native PyTorch reference for dequantize_and_gather_k_cache."""
    import triton
    num_reqs = seq_lens.shape[0]
    output_dim = out.shape[-1]
    if nope_dim is None:
        nope_dim = output_dim - rope_dim
    if scale_slots is None:
        scale_slots = triton.cdiv(nope_dim, 64) + (1 if nope_dim % 64 == 0 else 0)
    token_data_size = nope_dim + rope_dim * 2
    k_cache_2d = k_cache.view(k_cache.shape[0], -1) if k_cache.ndim == 3 else k_cache
    k_cache_flat = k_cache_2d.view(-1)
    cache_block_stride = k_cache_2d.stride(0)

    for req in range(num_reqs):
        seq_len = int(seq_lens[req].item())
        gather_len = int(gather_lens[req].item()) if gather_lens is not None else seq_len
        start_pos = seq_len - gather_len
        for local_i in range(gather_len):
            pos = start_pos + local_i
            block_in_seq = pos // block_size
            pos_in_block = pos % block_size
            physical_block = int(block_table[req, block_in_seq].item())
            cache_base = physical_block * cache_block_stride
            token_data_base = cache_base + pos_in_block * token_data_size
            scale_base = cache_base + block_size * token_data_size + pos_in_block * scale_slots

            for qblock in range(triton.cdiv(nope_dim, 64)):
                qstart = qblock * 64
                qend = min(qstart + 64, nope_dim)
                for j in range(qstart, qend):
                    x_u8 = int(k_cache_flat[token_data_base + j].item())
                    x_fp32 = float(torch.tensor([x_u8], dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32).item())
                    encoded = int(k_cache_flat[scale_base + qblock].item())
                    scale = float(torch.exp2(torch.tensor(encoded, dtype=torch.float32) - 127.0).item())
                    out[req, offset + local_i, j] = torch.tensor(x_fp32 * scale, dtype=torch.bfloat16)

            # Copy rope (bf16) portion
            rope_start = token_data_base + nope_dim
            rope_bytes = k_cache_flat[rope_start : rope_start + rope_dim * 2].view(torch.bfloat16)
            out[req, offset + local_i, nope_dim : nope_dim + rope_dim] = rope_bytes


@pytest.mark.dequantize_and_gather_k_cache
def test_dequantize_and_gather_k_cache_vllm_accuracy():
    device = flag_gems.device
    batch = 2
    seq_len = 12
    gather_len = 5
    block_size = 64
    nope_dim = 448
    rope_dim = 64
    scale_slots = 8
    output_dim = nope_dim + rope_dim
    token_data_size = nope_dim + rope_dim * 2
    block_stride = block_size * token_data_size + block_size * scale_slots
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = batch * blocks_per_seq
    k_cache = torch.zeros((num_blocks, block_stride), device=device, dtype=torch.uint8)
    expected_rows = torch.empty(
        (batch, gather_len, output_dim), device=device, dtype=torch.bfloat16
    )
    rows = {}
    for req in range(batch):
        start_pos = seq_len - gather_len
        for local_i in range(gather_len):
            pos = start_pos + local_i
            physical_block = req * blocks_per_seq + pos // block_size
            slot = physical_block * block_size + pos % block_size
            rows[slot] = expected_rows[req : req + 1, local_i : local_i + 1, :]
    _fill_cache(k_cache, rows, block_size, nope_dim, rope_dim, scale_slots)

    actual = torch.empty_like(expected_rows)
    expected = torch.empty_like(expected_rows)
    seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    gather_lens = torch.full((batch,), gather_len, device=device, dtype=torch.int32)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(
        batch, blocks_per_seq
    )

    dequantize_and_gather_k_cache(
        actual,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        rope_dim=rope_dim,
        nope_dim=nope_dim,
        scale_slots=scale_slots,
    )
    if vllm_dequantize_and_gather_k_cache:
        vllm_dequantize_and_gather_k_cache(
            expected,
            k_cache,
            seq_lens,
            gather_lens,
            block_table,
            block_size,
            0,
        )
    else:
        _native_dequantize_and_gather_k_cache(
            expected,
            k_cache,
            seq_lens,
            gather_lens,
            block_table,
            block_size,
            0,
            rope_dim=rope_dim,
            nope_dim=nope_dim,
            scale_slots=scale_slots,
        )

    fg_testing.assert_close(actual, expected, dtype=torch.bfloat16, equal_nan=True)
