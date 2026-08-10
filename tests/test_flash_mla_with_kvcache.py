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

"""
Tests for flash_mla_with_kvcache Triton implementation.
On CUDA, uses the vLLM FlashMLA implementation as ground truth.
On NPU (or any device without vLLM), uses a native PyTorch reference.
"""

import math

import pytest
import torch

import flag_gems

# CUDA reference (optional)
try:
    from vllm.third_party.flashmla.flash_mla_interface import (
        flash_mla_with_kvcache as cuda_flash_mla,
    )
    from vllm.third_party.flashmla.flash_mla_interface import (
        get_mla_metadata as cuda_get_mla_metadata,
    )

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# Triton implementation under test
from flag_gems.fused.flash_mla_with_kvcache import FlashMLASchedMeta
from flag_gems.fused.flash_mla_with_kvcache import (
    flash_mla_with_kvcache as triton_flash_mla,
)
from flag_gems.fused.flash_mla_with_kvcache import (
    get_mla_metadata as triton_get_mla_metadata,
)

from . import conftest as cfg

DEVICE = flag_gems.device
ON_CUDA = "cuda" in str(DEVICE)
CUDA_AVAILABLE = torch.cuda.is_available()

# Shape configs for QUICK_MODE
if cfg.QUICK_MODE:
    V32_FP8_CONFIGS = [(2, 64, 64, 50)]
    EXTRA_KV_CONFIGS = [(64, 512, 12)]
else:
    V32_FP8_CONFIGS = [(2, 64, 64, 50), (2, 128, 128, 100)]
    EXTRA_KV_CONFIGS = [(64, 512, 12), (2, 512, 260)]
FP8_MAX = 448.0


# ============================================================================
# Native PyTorch reference implementation (used when vLLM is unavailable, e.g. NPU)
# ============================================================================


def _native_get_mla_metadata():
    return FlashMLASchedMeta(), None


def _gather_dequant_v32(k_cache, indices, page_block_size):
    """Gather and dequantize V32 FP8 KV cache (656 bytes per token).

    Layout: [0:512] FP8 NoPE, [512:528] 4 FP32 scales, [528:656] 64 BF16 RoPE.
    V = NoPE (512 dims), K = concat(NoPE, RoPE) (576 dims).
    """
    batch, seq_q, topk = indices.shape
    num_pages = k_cache.shape[0]

    flat_idx = indices.reshape(-1)
    block_ids = flat_idx // page_block_size
    rel_ids = flat_idx % page_block_size

    gathered = k_cache[block_ids, rel_ids, 0, :]  # (B*S*T, 656)
    gathered = gathered.reshape(batch, seq_q, topk, 656)

    fp8_data = gathered[..., :512].view(torch.float8_e4m3fn).float()
    scales = gathered[..., 512:528].view(torch.float32).reshape(
        batch, seq_q, topk, 4
    )
    rope = gathered[..., 528:656].view(torch.bfloat16).reshape(
        batch, seq_q, topk, 64
    ).float()

    fp8_groups = fp8_data.reshape(batch, seq_q, topk, 4, 128)
    nope = (fp8_groups * scales[..., None]).reshape(batch, seq_q, topk, 512)

    K = torch.cat([nope, rope], dim=-1)  # (B, S, T, 576)
    V = nope  # (B, S, T, 512)
    return K.to(torch.bfloat16), V.to(torch.bfloat16)


def _gather_dequant_model1(k_cache, indices, page_block_size):
    """Gather and dequantize MODEL1 FP8 KV cache (584 bytes per token, page-oriented).

    Per page: [token_data (pbs*576), scales (pbs*8)].
    Per token: 448 FP8 NoPE + 64 BF16 RoPE = 576 bytes, 8 bytes scales (7 used).
    V = concat(NoPE, RoPE) (512 dims), K = same (512 dims).
    """
    batch, seq_q, topk = indices.shape
    num_pages = k_cache.shape[0]

    # MODEL1 is page-oriented: extract token data and scales separately
    page_flat = k_cache.reshape(num_pages, -1)  # (num_pages, pbs * 584)
    token_data = page_flat[:, : page_block_size * 576].reshape(
        num_pages, page_block_size, 576
    )
    scale_data = page_flat[:, page_block_size * 576 :].reshape(
        num_pages, page_block_size, 8
    )

    flat_idx = indices.reshape(-1)
    block_ids = flat_idx // page_block_size
    rel_ids = flat_idx % page_block_size

    gathered_token = token_data[block_ids, rel_ids]  # (B*S*T, 576)
    gathered_scale = scale_data[block_ids, rel_ids]  # (B*S*T, 8)

    gathered_token = gathered_token.reshape(batch, seq_q, topk, 576)
    gathered_scale = gathered_scale.reshape(batch, seq_q, topk, 8)

    fp8_data = gathered_token[..., :448].view(torch.float8_e4m3fn).float()
    rope = gathered_token[..., 448:576].view(torch.bfloat16).reshape(
        batch, seq_q, topk, 64
    ).float()

    # E8M0 scales: scale = 2^(uint8 - 127)
    scale_u8 = gathered_scale[..., :7]
    scales = torch.pow(2.0, scale_u8.to(torch.float32) - 127.0)

    fp8_groups = fp8_data.reshape(batch, seq_q, topk, 7, 64)
    nope = (fp8_groups * scales[..., None]).reshape(batch, seq_q, topk, 448)

    K = torch.cat([nope, rope], dim=-1)  # (B, S, T, 512)
    V = K  # MODEL1: V includes both NoPE and RoPE
    return K.to(torch.bfloat16), V.to(torch.bfloat16)


def _native_sparse_decode(
    q,
    k_cache,
    indices,
    out,
    lse,
    scale,
    attn_sink,
    topk_length,
    is_fp8_kvcache,
    extra_k_cache,
    extra_indices_in_kvcache,
    extra_topk_length,
):
    """Native PyTorch sparse decode reference."""
    batch, seq_q, h_q, d_qk = q.shape
    topk = indices.shape[-1]
    page_block_size = k_cache.shape[1]

    # Gather and dequantize main KV
    if d_qk == 576:
        K, V = _gather_dequant_v32(k_cache, indices, page_block_size)
        head_dim_v = 512
    else:
        K, V = _gather_dequant_model1(k_cache, indices, page_block_size)
        head_dim_v = 512

    # Gather extra KV if provided
    if extra_k_cache is not None:
        extra_topk = extra_indices_in_kvcache.shape[-1]
        extra_pbs = extra_k_cache.shape[1]
        K_extra, V_extra = _gather_dequant_model1(
            extra_k_cache, extra_indices_in_kvcache, extra_pbs
        )
        K = torch.cat([K, K_extra], dim=2)
        V = torch.cat([V, V_extra], dim=2)
        total_topk = topk + extra_topk
    else:
        total_topk = topk

    # Build mask for topk_length
    if topk_length is not None or extra_topk_length is not None:
        mask = torch.ones(batch, seq_q, total_topk, dtype=torch.bool, device=q.device)
        arange = torch.arange(total_topk, device=q.device)
        if topk_length is not None:
            for b in range(batch):
                tl = topk_length[b].item()
                mask[b, :, tl:topk] = False
        if extra_topk_length is not None:
            for b in range(batch):
                etl = extra_topk_length[b].item()
                mask[b, :, topk + etl :] = False
    else:
        mask = None

    _compute_attention(q, K, V, out, lse, scale, mask, attn_sink)


def _native_dense_decode(
    q,
    k_cache,
    block_table,
    cache_seqlens,
    out,
    lse,
    scale,
    causal,
    head_dim_v,
):
    """Native PyTorch dense decode reference with paged attention."""
    batch, seq_q, h_q, d_qk = q.shape
    page_block_size = k_cache.shape[1]
    max_pages_per_seq = block_table.shape[1]

    for b in range(batch):
        seqlen = min(
            cache_seqlens[b].item(), max_pages_per_seq * page_block_size
        )
        num_pages_needed = (seqlen + page_block_size - 1) // page_block_size
        pages = block_table[b, :num_pages_needed]
        gathered = k_cache[pages]  # (num_pages, pbs, h_k, d_qk)
        K_b = gathered.reshape(-1, d_qk)[:seqlen]  # (seqlen, d_qk)
        V_b = K_b[:, :head_dim_v]  # (seqlen, head_dim_v)

        Q_b = q[b].float()  # (seq_q, h_q, d_qk)
        K_flat = K_b.float()  # (seqlen, d_qk)
        V_flat = V_b.float()  # (seqlen, head_dim_v)

        # QK
        qk = torch.einsum("shd,td->sht", Q_b, K_flat) * scale

        # Causal mask
        if causal and seq_q > 1:
            causal_mask = torch.triu(
                torch.ones(seq_q, seqlen, device=q.device), diagonal=1
            ).bool()
            qk = qk.masked_fill(causal_mask[:, None, :], float("-inf"))

        # LSE and softmax
        lse_val = torch.logsumexp(qk, dim=-1)  # (seq_q, h_q)
        p = torch.exp(qk - lse_val[..., None])

        # Output
        out_b = torch.einsum("sht,td->shd", p, V_flat)

        out[b] = out_b.to(out.dtype)
        lse[b] = lse_val.T  # (h_q, seq_q)


def _compute_attention(q, K, V, out, lse, scale, mask, attn_sink):
    """Compute attention with optional mask and attn_sink.

    Q: (batch, seq_q, h_q, d_qk)
    K: (batch, seq_q, n_kv, d_qk)
    V: (batch, seq_q, n_kv, head_dim_v)
    """
    batch, seq_q, h_q, d_qk = q.shape
    n_kv = K.shape[2]
    head_dim_v = V.shape[-1]

    Q = q.reshape(batch * seq_q, h_q, d_qk).float()
    K_flat = K.reshape(batch * seq_q, n_kv, d_qk).float()
    V_flat = V.reshape(batch * seq_q, n_kv, head_dim_v).float()

    qk = torch.matmul(Q, K_flat.transpose(-1, -2)) * scale  # (B*S, h_q, n_kv)

    if mask is not None:
        mask_flat = mask.reshape(batch * seq_q, n_kv)
        qk = qk.masked_fill(~mask_flat[:, None, :], float("-inf"))

    lse_val = torch.logsumexp(qk, dim=-1)  # (B*S, h_q)
    p = torch.exp(qk - lse_val[..., None])
    out_val = torch.matmul(p, V_flat)  # (B*S, h_q, head_dim_v)

    if attn_sink is not None:
        sink = attn_sink.float()  # (h_q,)
        sink_weight = torch.exp(sink[None, :] - lse_val)  # (B*S, h_q)
        out_val = out_val / (1 + sink_weight[..., None])
        lse_val = torch.log(
            torch.exp(lse_val) + torch.exp(sink[None, :])
        )

    out.copy_(out_val.reshape(batch, seq_q, h_q, head_dim_v).to(out.dtype))
    lse.copy_(lse_val.reshape(batch, h_q, seq_q))


def _native_flash_mla_with_kvcache(
    q,
    k_cache,
    block_table,
    cache_seqlens,
    head_dim_v,
    sched_meta,
    num_splits=None,
    softmax_scale=None,
    causal=False,
    is_fp8_kvcache=False,
    indices=None,
    attn_sink=None,
    extra_k_cache=None,
    extra_indices_in_kvcache=None,
    topk_length=None,
    extra_topk_length=None,
    out=None,
):
    """Native PyTorch reference for flash_mla_with_kvcache."""
    batch, seq_q, h_q, d_qk = q.shape

    if softmax_scale is None:
        softmax_scale = d_qk ** (-0.5)

    if out is None:
        out = torch.empty(
            batch, seq_q, h_q, head_dim_v, dtype=q.dtype, device=q.device
        )

    lse = torch.empty(
        batch, h_q, seq_q, dtype=torch.float32, device=q.device
    )

    if indices is not None:
        _native_sparse_decode(
            q,
            k_cache,
            indices,
            out,
            lse,
            softmax_scale,
            attn_sink,
            topk_length,
            is_fp8_kvcache,
            extra_k_cache,
            extra_indices_in_kvcache,
            extra_topk_length,
        )
    else:
        _native_dense_decode(
            q,
            k_cache,
            block_table,
            cache_seqlens,
            out,
            lse,
            softmax_scale,
            causal,
            head_dim_v,
        )

    return out, lse


# ============================================================================
# KV cache generators
# ============================================================================


def generate_v32_fp8_kv_cache(
    num_pages, page_block_size, h_k=1, d_nope=512, d_rope=64, device=DEVICE
):
    """Generate V32 FP8 KV cache: 512 FP8 + 4 FP32 scales + 64 BF16."""
    total_tokens = num_pages * page_block_size

    nope_data = (
        torch.randn(total_tokens, h_k, d_nope, dtype=torch.bfloat16, device=device)
        * 0.1
    )
    nope_flat = nope_data.reshape(-1, d_nope)
    groups = nope_flat.reshape(-1, 4, 128)
    scales = groups.float().abs().amax(dim=-1) / FP8_MAX
    scales = scales.clamp(min=1e-12)
    quantized = (groups.float() / scales[:, :, None]).clamp(-FP8_MAX, FP8_MAX)
    fp8_data = quantized.reshape(-1, d_nope).to(torch.float8_e4m3fn)

    rope_data = (
        torch.randn(total_tokens, h_k, d_rope, dtype=torch.bfloat16, device=device)
        * 0.1
    )

    kv_cache = torch.zeros(
        num_pages, page_block_size, h_k, 656, dtype=torch.uint8, device=device
    )
    kv_cache[:, :, :, :512] = fp8_data.view(torch.uint8).reshape(
        num_pages, page_block_size, h_k, 512
    )
    kv_cache[:, :, :, 512:528] = (
        scales.reshape(num_pages, page_block_size, h_k, 4)
        .to(torch.float32)
        .view(torch.uint8)
        .reshape(num_pages, page_block_size, h_k, 16)
    )
    kv_cache[:, :, :, 528:656] = (
        rope_data.reshape(num_pages, page_block_size, h_k, d_rope)
        .view(torch.uint8)
        .reshape(num_pages, page_block_size, h_k, 128)
    )
    return kv_cache


def generate_model1_fp8_kv_cache(
    num_pages, page_block_size, h_k=1, d_nope=448, d_rope=64, device=DEVICE
):
    """Generate MODEL1 page-oriented cache: page data first, then E8M0 scales."""
    assert h_k == 1, "FlashMLA sparse decode currently supports h_k == 1"
    total_tokens = num_pages * page_block_size
    token_data_bytes = d_nope + d_rope * 2
    scale_bytes = 8

    nope_data = (
        torch.randn(total_tokens, d_nope, dtype=torch.bfloat16, device=device) * 0.1
    )
    groups = nope_data.reshape(total_tokens, 7, 64).float()
    block_max = torch.clamp(groups.abs().amax(dim=-1), min=1e-4)
    exponent = torch.ceil(torch.log2(block_max / FP8_MAX))
    scales = torch.exp2(exponent)
    fp8_data = (
        torch.clamp(groups / scales[:, :, None], -FP8_MAX, FP8_MAX)
        .reshape(total_tokens, d_nope)
        .to(torch.float8_e4m3fn)
    )

    rope_data = (
        torch.randn(total_tokens, d_rope, dtype=torch.bfloat16, device=device) * 0.1
    )
    data_bytes = torch.empty(
        total_tokens, token_data_bytes, dtype=torch.uint8, device=device
    )
    data_bytes[:, :d_nope] = fp8_data.view(torch.uint8).reshape(total_tokens, d_nope)
    data_bytes[:, d_nope:] = rope_data.view(torch.uint8).reshape(total_tokens, 128)

    encoded_scales = torch.zeros(
        total_tokens, scale_bytes, dtype=torch.uint8, device=device
    )
    encoded_scales[:, :7] = torch.clamp(exponent + 127.0, 0, 255).to(torch.uint8)

    kv_cache = torch.zeros(
        num_pages, page_block_size, h_k, 584, dtype=torch.uint8, device=device
    )
    page_flat = kv_cache.view(num_pages, -1)
    page_flat[:, : page_block_size * token_data_bytes] = data_bytes.reshape(
        num_pages, page_block_size * token_data_bytes
    )
    page_flat[:, page_block_size * token_data_bytes :] = encoded_scales.reshape(
        num_pages, page_block_size * scale_bytes
    )
    return kv_cache


# ============================================================================
# Test helpers
# ============================================================================


def check_close(triton_out, ref_out, name, cos_threshold=0.99):
    diff = (triton_out.float() - ref_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        triton_out.float().flatten(), ref_out.float().flatten(), dim=0
    ).item()
    print(
        f"  {name}: max_diff={max_diff:.6f}, "
        f"mean_diff={mean_diff:.6f}, cos_sim={cos_sim:.8f}"
    )
    assert cos_sim > cos_threshold, f"{name} cosine similarity too low: {cos_sim}"


def _run_ref(q, k_cache, block_table, cache_seqlens, head_dim_v, **kwargs):
    """Run reference implementation: vLLM CUDA if available, else native PyTorch."""
    if VLLM_AVAILABLE:
        ref_meta, _ = cuda_get_mla_metadata()
        return cuda_flash_mla(
            q, k_cache, block_table, cache_seqlens, head_dim_v, ref_meta, **kwargs
        )
    else:
        ref_meta, _ = triton_get_mla_metadata()
        return _native_flash_mla_with_kvcache(
            q, k_cache, block_table, cache_seqlens, head_dim_v, ref_meta, **kwargs
        )


def _run_triton(q, k_cache, block_table, cache_seqlens, head_dim_v, **kwargs):
    triton_meta, _ = triton_get_mla_metadata()
    return triton_flash_mla(
        q, k_cache, block_table, cache_seqlens, head_dim_v, triton_meta, **kwargs
    )


# On CUDA: require vLLM (original behavior). On NPU: run with native reference.
pytestmark = pytest.mark.skipif(
    ON_CUDA and not (VLLM_AVAILABLE and CUDA_AVAILABLE),
    reason="vLLM FlashMLA and CUDA are required on CUDA backend",
)


@pytest.mark.parametrize(
    "batch,h_q,topk,num_pages",
    V32_FP8_CONFIGS,
)
def test_sparse_decode_v32_fp8(batch, h_q, topk, num_pages):
    """V32 sparse decode uses 656-byte FP8 cache and no dynamic topk."""
    print("\n=== test_sparse_decode_v32_fp8 ===")
    seq_q, d_qk = 1, 576
    head_dim_v = 512
    page_block_size = 64

    torch.manual_seed(42)
    q = torch.randn(batch, seq_q, h_q, d_qk, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = generate_v32_fp8_kv_cache(num_pages, page_block_size)
    total_tokens = num_pages * page_block_size
    indices = torch.randint(
        0, total_tokens, (batch, seq_q, topk), dtype=torch.int32, device=DEVICE
    )
    attn_sink = torch.randn(h_q, dtype=torch.float32, device=DEVICE)

    ref_out_buf = torch.empty(
        batch, seq_q, h_q, head_dim_v, dtype=torch.bfloat16, device=DEVICE
    )
    triton_out_buf = torch.empty_like(ref_out_buf)
    ref_out, ref_lse = _run_ref(
        q,
        kv_cache,
        None,
        None,
        head_dim_v,
        is_fp8_kvcache=True,
        indices=indices,
        attn_sink=attn_sink,
        out=ref_out_buf,
    )
    triton_out, triton_lse = _run_triton(
        q,
        kv_cache,
        None,
        None,
        head_dim_v,
        is_fp8_kvcache=True,
        indices=indices,
        attn_sink=attn_sink,
        out=triton_out_buf,
    )

    assert ref_out.data_ptr() == ref_out_buf.data_ptr()
    assert triton_out.data_ptr() == triton_out_buf.data_ptr()
    check_close(triton_out, ref_out, "out")
    check_close(triton_lse, ref_lse, "lse")


def test_sparse_decode_model1_topk_length_attn_sink_out():
    """MODEL1 log-like path: 584-byte cache, topk_length, attn_sink, out."""
    print("\n=== test_sparse_decode_model1_topk_length_attn_sink_out ===")
    batch, seq_q, h_q, d_qk, topk = 8, 1, 64, 512, 128
    head_dim_v = 512
    page_block_size = 64
    num_pages = 16

    torch.manual_seed(43)
    q = torch.randn(batch, seq_q, h_q, d_qk, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = generate_model1_fp8_kv_cache(num_pages, page_block_size)
    total_tokens = num_pages * page_block_size
    indices = torch.randint(
        0, total_tokens, (batch, seq_q, topk), dtype=torch.int32, device=DEVICE
    )
    topk_length = torch.randint(1, topk + 1, (batch,), dtype=torch.int32, device=DEVICE)
    attn_sink = torch.randn(h_q, dtype=torch.float32, device=DEVICE)
    ref_out_buf = torch.empty(
        batch, seq_q, h_q, head_dim_v, dtype=torch.bfloat16, device=DEVICE
    )
    triton_out_buf = torch.empty_like(ref_out_buf)

    ref_out, ref_lse = _run_ref(
        q,
        kv_cache,
        None,
        None,
        head_dim_v,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        out=ref_out_buf,
    )
    triton_out, triton_lse = _run_triton(
        q,
        kv_cache,
        None,
        None,
        head_dim_v,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        out=triton_out_buf,
    )

    check_close(triton_out, ref_out, "out", cos_threshold=0.98)
    check_close(triton_lse, ref_lse, "lse", cos_threshold=0.98)


@pytest.mark.parametrize(
    "extra_page_block_size,extra_topk,extra_num_pages",
    EXTRA_KV_CONFIGS,
)
def test_sparse_decode_model1_extra_kv(
    extra_page_block_size, extra_topk, extra_num_pages
):
    """MODEL1 attends original sparse cache plus extra sparse cache."""
    print("\n=== test_sparse_decode_model1_extra_kv ===")
    batch, seq_q, h_q, d_qk, topk = 4, 1, 64, 512, 128
    head_dim_v = 512
    page_block_size = 64
    num_pages = 12

    torch.manual_seed(44 + extra_page_block_size)
    q = torch.randn(batch, seq_q, h_q, d_qk, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = generate_model1_fp8_kv_cache(num_pages, page_block_size)
    extra_k_cache = generate_model1_fp8_kv_cache(extra_num_pages, extra_page_block_size)

    indices = torch.randint(
        0,
        num_pages * page_block_size,
        (batch, seq_q, topk),
        dtype=torch.int32,
        device=DEVICE,
    )
    extra_indices = torch.randint(
        0,
        extra_num_pages * extra_page_block_size,
        (batch, seq_q, extra_topk),
        dtype=torch.int32,
        device=DEVICE,
    )
    topk_length = torch.randint(1, topk + 1, (batch,), dtype=torch.int32, device=DEVICE)
    extra_topk_length = torch.randint(
        1, extra_topk + 1, (batch,), dtype=torch.int32, device=DEVICE
    )
    attn_sink = torch.randn(h_q, dtype=torch.float32, device=DEVICE)

    ref_out, ref_lse = _run_ref(
        q,
        kv_cache,
        None,
        None,
        head_dim_v,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_topk_length,
    )
    triton_out, triton_lse = _run_triton(
        q,
        kv_cache,
        None,
        None,
        head_dim_v,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_topk_length,
    )

    check_close(triton_out, ref_out, "out", cos_threshold=0.98)
    check_close(triton_lse, ref_lse, "lse", cos_threshold=0.98)


def test_dense_decode_seq_q1():
    """Dense decode is covered only for seq_q=1 in the Triton implementation."""
    print("\n=== test_dense_decode_seq_q1 ===")
    batch, seq_q, h_q, d_qk = 4, 1, 128, 576
    h_k = 1
    head_dim_v = 512
    page_block_size = 64
    seqlen = 256

    torch.manual_seed(45)
    q = torch.randn(batch, seq_q, h_q, d_qk, dtype=torch.bfloat16, device=DEVICE)
    max_pages_per_seq = math.ceil(seqlen / page_block_size) + 4
    total_pages = batch * max_pages_per_seq
    kv_cache = (
        torch.randn(
            total_pages, page_block_size, h_k, d_qk, dtype=torch.bfloat16, device=DEVICE
        )
        * 0.1
    )
    block_table = torch.arange(total_pages, dtype=torch.int32, device=DEVICE).view(
        batch, max_pages_per_seq
    )
    cache_seqlens = torch.full((batch,), seqlen, dtype=torch.int32, device=DEVICE)
    cache_seqlens[0] = seqlen // 2
    cache_seqlens[-1] = seqlen + page_block_size

    ref_out, ref_lse = _run_ref(
        q, kv_cache, block_table, cache_seqlens, head_dim_v, causal=True
    )
    triton_out, triton_lse = _run_triton(
        q, kv_cache, block_table, cache_seqlens, head_dim_v, causal=True
    )

    check_close(triton_out, ref_out, "out")
    check_close(triton_lse, ref_lse, "lse")


def test_error_v32_rejects_topk_length():
    q = torch.randn(1, 1, 64, 576, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = generate_v32_fp8_kv_cache(4, 64)
    indices = torch.randint(0, 256, (1, 1, 64), dtype=torch.int32, device=DEVICE)
    topk_length = torch.ones(1, dtype=torch.int32, device=DEVICE)
    meta, _ = triton_get_mla_metadata()

    with pytest.raises(AssertionError, match="dynamic topk length"):
        triton_flash_mla(
            q,
            kv_cache,
            None,
            None,
            512,
            meta,
            is_fp8_kvcache=True,
            indices=indices,
            topk_length=topk_length,
        )


def test_error_model1_extra_cache_requires_extra_indices():
    q = torch.randn(1, 1, 64, 512, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = generate_model1_fp8_kv_cache(4, 64)
    extra_k_cache = generate_model1_fp8_kv_cache(4, 64)
    indices = torch.randint(0, 256, (1, 1, 64), dtype=torch.int32, device=DEVICE)
    meta, _ = triton_get_mla_metadata()

    with pytest.raises(AssertionError, match="extra_indices_in_kvcache"):
        triton_flash_mla(
            q,
            kv_cache,
            None,
            None,
            512,
            meta,
            is_fp8_kvcache=True,
            indices=indices,
            extra_k_cache=extra_k_cache,
        )


def test_error_dense_rejects_sparse_only_args():
    q = torch.randn(1, 1, 64, 576, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = torch.randn(4, 64, 1, 576, dtype=torch.bfloat16, device=DEVICE)
    block_table = torch.arange(4, dtype=torch.int32, device=DEVICE).view(1, 4)
    cache_seqlens = torch.full((1,), 64, dtype=torch.int32, device=DEVICE)
    attn_sink = torch.randn(64, dtype=torch.float32, device=DEVICE)
    meta, _ = triton_get_mla_metadata()

    with pytest.raises(AssertionError, match="must be None when dense"):
        triton_flash_mla(
            q,
            kv_cache,
            block_table,
            cache_seqlens,
            512,
            meta,
            attn_sink=attn_sink,
        )


def test_error_sched_meta_reuse_mismatch():
    q = torch.empty(1, 1, 64, 512, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = torch.empty(4, 64, 1, 584, dtype=torch.uint8, device=DEVICE)
    meta = FlashMLASchedMeta(
        have_initialized=True,
        config=FlashMLASchedMeta.Config(
            b=2,
            s_q=1,
            h_q=64,
            page_block_size=64,
            h_k=1,
            causal=False,
            is_fp8_kvcache=True,
            topk=64,
            extra_page_block_size=None,
            extra_topk=None,
        ),
    )

    with pytest.raises(AssertionError, match="sched_meta.config.b"):
        triton_flash_mla(q, kv_cache, None, None, 512, meta)


if __name__ == "__main__":
    test_sparse_decode_v32_fp8(2, 64, 64, 50)
    test_sparse_decode_v32_fp8(2, 128, 128, 100)
    test_sparse_decode_model1_topk_length_attn_sink_out()
    test_sparse_decode_model1_extra_kv(64, 512, 12)
    test_dense_decode_seq_q1()
