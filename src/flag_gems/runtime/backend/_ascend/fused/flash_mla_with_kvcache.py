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

"""Ascend NPU implementation of MODEL1 sparse decode (head_dim_k == 512).

This is a standalone implementation that mirrors the GPU kernel in
``flag_gems.fused.flash_mla_with_kvcache`` but uses bf16 dequant to halve
UB usage, smaller tile configs to fit within the 216 KB UB budget, and a
two-pass workaround for the extra-KV path (HAVE_EXTRA=True would need
~533 KB UB, far exceeding the limit).
"""

import torch
import triton
import triton.language as tl


# MODEL1 sparse decode: BDP=256 creates large [BDP, BK] tiles.
# BK=32 overflows UB even with bf16 dequant; BK=16 halves all [BDP, BK]
# tiles and fits within the 216 KB UB budget.
_SPARSE_DECODE_MODEL1_CONFIGS = [
    triton.Config({"BK": 32, "BH": 16}, num_warps=4, num_stages=1),
    triton.Config({"BK": 16, "BH": 16}, num_warps=4, num_stages=1),
    triton.Config({"BK": 16, "BH": 8}, num_warps=2, num_stages=1),
]


@triton.autotune(
    configs=_SPARSE_DECODE_MODEL1_CONFIGS,
    key=[
        "HQ",
        "TOPK",
        "EXTRA_TOPK",
        "HAVE_ATTN_SINK",
        "HAVE_TOPK_LENGTH",
        "HAVE_EXTRA",
        "HAVE_EXTRA_TOPK_LENGTH",
    ],
)
@triton.jit
def _sparse_decode_model1_kernel(
    q,
    kv,
    indices,
    extra_kv,
    extra_indices,
    attn_sink,
    topk_length,
    extra_topk_length,
    sm_scale: tl.constexpr,
    output,
    lse,
    stride_qb,
    stride_qsq,
    stride_qh,
    stride_kv_block,
    stride_ib,
    stride_isq,
    stride_extra_kv_block,
    stride_eib,
    stride_eisq,
    stride_ob,
    stride_osq,
    stride_oh,
    stride_lseb,
    stride_lseh,
    SQ,
    HQ: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    EXTRA_PAGE_SIZE: tl.constexpr,
    NUM_BLOCKS,
    EXTRA_NUM_BLOCKS,
    TOPK: tl.constexpr,
    EXTRA_TOPK: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    HAVE_EXTRA: tl.constexpr,
    HAVE_EXTRA_TOPK_LENGTH: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    num_head_blocks: tl.constexpr = (HQ + BH - 1) // BH
    pid = tl.program_id(0)
    i_b = pid // (SQ * num_head_blocks)
    remainder = pid % (SQ * num_head_blocks)
    i_sq = remainder // num_head_blocks
    i_sq = i_sq.to(tl.int64)
    i_gbh = remainder % num_head_blocks
    gbh_base = i_gbh * BH

    NOPE: tl.constexpr = 448
    ROPE: tl.constexpr = 64
    BDP: tl.constexpr = 256
    TOKEN_DATA_BYTES: tl.constexpr = 576
    SCALE_BYTES: tl.constexpr = 8

    q_base = q + i_b * stride_qb + i_sq * stride_qsq + gbh_base * stride_qh
    t_base = indices + i_b * stride_ib + i_sq * stride_isq
    et_base = extra_indices + i_b * stride_eib + i_sq * stride_eisq
    attn_sink_ptr = attn_sink + gbh_base if HAVE_ATTN_SINK else 0
    topk_length_ptr = topk_length + i_b if HAVE_TOPK_LENGTH else 0
    extra_topk_length_ptr = extra_topk_length + i_b if HAVE_EXTRA_TOPK_LENGTH else 0
    o_base = output + i_b * stride_ob + i_sq * stride_osq + gbh_base * stride_oh
    l_base = lse + i_b * stride_lseb + gbh_base * stride_lseh + i_sq

    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, BDP)
    offs_t = tl.arange(0, BK)
    offs_rope = tl.arange(0, ROPE)

    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(q_ptr, eviction_policy="evict_first")
    q_blk1_nope = tl.load(
        q_ptr + BDP,
        mask=offs_d[None, :] < (NOPE - BDP),
        other=0.0,
        eviction_policy="evict_first",
    )
    q_rope = tl.load(
        q_base + offs_h[:, None] * stride_qh + (NOPE + offs_rope[None, :]),
        eviction_policy="evict_first",
    )

    max_log = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
    acc0 = tl.zeros([BDP, BH], dtype=tl.float32)
    acc1 = tl.zeros([BDP, BH], dtype=tl.float32)

    topk_len = tl.load(topk_length_ptr) if HAVE_TOPK_LENGTH else TOPK
    NK = tl.cdiv(topk_len, BK)
    for ck in range(NK):
        t_offs = BK * ck + offs_t
        t_msk = t_offs < topk_len
        kv_ids = tl.load(t_base + t_offs, t_msk, other=-1)
        block_ids = kv_ids // PAGE_SIZE
        rel_ids = kv_ids - block_ids * PAGE_SIZE
        valid_ids = t_msk & (kv_ids >= 0) & (block_ids < NUM_BLOCKS)
        block_ids = tl.where(valid_ids, block_ids, 0)
        rel_ids = tl.where(valid_ids, rel_ids, 0)

        token_base = (
            kv + block_ids.to(tl.int64) * stride_kv_block + rel_ids * TOKEN_DATA_BYTES
        )
        scale_base = (
            kv
            + block_ids.to(tl.int64) * stride_kv_block
            + PAGE_SIZE * TOKEN_DATA_BYTES
            + rel_ids * SCALE_BYTES
        )

        kv_fp8_0_u8 = tl.load(
            token_base[None, :] + offs_d[:, None],
            mask=valid_ids[None, :],
            other=0,
            cache_modifier=".cg",
        )
        kv_fp8_1_u8 = tl.load(
            token_base[None, :] + (BDP + offs_d[:, None]),
            mask=valid_ids[None, :] & (offs_d[:, None] < (NOPE - BDP)),
            other=0,
            cache_modifier=".cg",
        )

        scale0_u8 = tl.load(scale_base + 0, mask=valid_ids, other=127)
        scale1_u8 = tl.load(scale_base + 1, mask=valid_ids, other=127)
        scale2_u8 = tl.load(scale_base + 2, mask=valid_ids, other=127)
        scale3_u8 = tl.load(scale_base + 3, mask=valid_ids, other=127)
        scale4_u8 = tl.load(scale_base + 4, mask=valid_ids, other=127)
        scale5_u8 = tl.load(scale_base + 5, mask=valid_ids, other=127)
        scale6_u8 = tl.load(scale_base + 6, mask=valid_ids, other=127)

        scale0 = (scale0_u8.to(tl.int16) << 7).to(tl.bfloat16, bitcast=True)
        scale1 = (scale1_u8.to(tl.int16) << 7).to(tl.bfloat16, bitcast=True)
        scale2 = (scale2_u8.to(tl.int16) << 7).to(tl.bfloat16, bitcast=True)
        scale3 = (scale3_u8.to(tl.int16) << 7).to(tl.bfloat16, bitcast=True)
        scale4 = (scale4_u8.to(tl.int16) << 7).to(tl.bfloat16, bitcast=True)
        scale5 = (scale5_u8.to(tl.int16) << 7).to(tl.bfloat16, bitcast=True)
        scale6 = (scale6_u8.to(tl.int16) << 7).to(tl.bfloat16, bitcast=True)

        kv_fp8_0 = kv_fp8_0_u8.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
        scale_0 = tl.where(
            offs_d[:, None] < 64,
            scale0[None, :],
            tl.where(
                offs_d[:, None] < 128,
                scale1[None, :],
                tl.where(offs_d[:, None] < 192, scale2[None, :], scale3[None, :]),
            ),
        )
        kv_blk0 = kv_fp8_0 * scale_0
        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)

        kv_fp8_1 = kv_fp8_1_u8.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
        scale_1 = tl.where(
            offs_d[:, None] < 64,
            scale4[None, :],
            tl.where(offs_d[:, None] < 128, scale5[None, :], scale6[None, :]),
        )
        nope_tail = kv_fp8_1 * scale_1
        qk = tl.dot(q_blk1_nope, nope_tail, qk, out_dtype=tl.float32)

        rope_ptr = (token_base + NOPE).to(tl.pointer_type(tl.bfloat16))
        rope_blk = tl.load(
            rope_ptr[None, :] + offs_rope[:, None],
            mask=valid_ids[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        qk = tl.dot(q_rope, rope_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale
        qk = tl.where(valid_ids[None, :], qk, float("-inf"))

        kv_blk1 = tl.where(
            offs_d[:, None] < (NOPE - BDP),
            nope_tail,
            tl.load(
                rope_ptr[None, :] + (offs_d[:, None] - (NOPE - BDP)),
                mask=valid_ids[None, :] & (offs_d[:, None] >= (NOPE - BDP)),
                other=0.0,
                cache_modifier=".cg",
            ),
        )

        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk
        exp_qk_t = tl.trans(exp_qk.to(tl.bfloat16))
        acc0 = tl.dot(
            kv_blk0,
            exp_qk_t,
            acc0 * alpha[None, :],
            out_dtype=tl.float32,
        )
        acc1 = tl.dot(
            kv_blk1,
            exp_qk_t,
            acc1 * alpha[None, :],
            out_dtype=tl.float32,
        )
        max_log = new_max

    if HAVE_EXTRA:
        extra_topk_len = (
            tl.load(extra_topk_length_ptr) if HAVE_EXTRA_TOPK_LENGTH else EXTRA_TOPK
        )
        ENK = tl.cdiv(extra_topk_len, BK)
        for ck in range(ENK):
            t_offs = BK * ck + offs_t
            t_msk = t_offs < extra_topk_len
            kv_ids = tl.load(et_base + t_offs, t_msk, other=-1)
            block_ids = kv_ids // EXTRA_PAGE_SIZE
            rel_ids = kv_ids - block_ids * EXTRA_PAGE_SIZE
            valid_ids = t_msk & (kv_ids >= 0) & (block_ids < EXTRA_NUM_BLOCKS)
            block_ids = tl.where(valid_ids, block_ids, 0)
            rel_ids = tl.where(valid_ids, rel_ids, 0)

            token_base = (
                extra_kv
                + block_ids.to(tl.int64) * stride_extra_kv_block
                + rel_ids * TOKEN_DATA_BYTES
            )
            scale_base = (
                extra_kv
                + block_ids.to(tl.int64) * stride_extra_kv_block
                + EXTRA_PAGE_SIZE * TOKEN_DATA_BYTES
                + rel_ids * SCALE_BYTES
            )

            kv_fp8_0_u8 = tl.load(
                token_base[None, :] + offs_d[:, None],
                mask=valid_ids[None, :],
                other=0,
                cache_modifier=".cg",
            )
            kv_fp8_1_u8 = tl.load(
                token_base[None, :] + (BDP + offs_d[:, None]),
                mask=valid_ids[None, :] & (offs_d[:, None] < (NOPE - BDP)),
                other=0,
                cache_modifier=".cg",
            )

            scale0_u8 = tl.load(scale_base + 0, mask=valid_ids, other=127)
            scale1_u8 = tl.load(scale_base + 1, mask=valid_ids, other=127)
            scale2_u8 = tl.load(scale_base + 2, mask=valid_ids, other=127)
            scale3_u8 = tl.load(scale_base + 3, mask=valid_ids, other=127)
            scale4_u8 = tl.load(scale_base + 4, mask=valid_ids, other=127)
            scale5_u8 = tl.load(scale_base + 5, mask=valid_ids, other=127)
            scale6_u8 = tl.load(scale_base + 6, mask=valid_ids, other=127)

            scale0 = (scale0_u8.to(tl.int16) << 7).to(tl.bfloat16, bitcast=True)
            scale1 = (scale1_u8.to(tl.int16) << 7).to(tl.bfloat16, bitcast=True)
            scale2 = (scale2_u8.to(tl.int16) << 7).to(tl.bfloat16, bitcast=True)
            scale3 = (scale3_u8.to(tl.int16) << 7).to(tl.bfloat16, bitcast=True)
            scale4 = (scale4_u8.to(tl.int16) << 7).to(tl.bfloat16, bitcast=True)
            scale5 = (scale5_u8.to(tl.int16) << 7).to(tl.bfloat16, bitcast=True)
            scale6 = (scale6_u8.to(tl.int16) << 7).to(tl.bfloat16, bitcast=True)

            kv_fp8_0 = kv_fp8_0_u8.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
            scale_0 = tl.where(
                offs_d[:, None] < 64,
                scale0[None, :],
                tl.where(
                    offs_d[:, None] < 128,
                    scale1[None, :],
                    tl.where(offs_d[:, None] < 192, scale2[None, :], scale3[None, :]),
                ),
            )
            kv_blk0 = kv_fp8_0 * scale_0
            qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)

            kv_fp8_1 = kv_fp8_1_u8.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
            scale_1 = tl.where(
                offs_d[:, None] < 64,
                scale4[None, :],
                tl.where(offs_d[:, None] < 128, scale5[None, :], scale6[None, :]),
            )
            nope_tail = kv_fp8_1 * scale_1
            qk = tl.dot(q_blk1_nope, nope_tail, qk, out_dtype=tl.float32)

            rope_ptr = (token_base + NOPE).to(tl.pointer_type(tl.bfloat16))
            rope_blk = tl.load(
                rope_ptr[None, :] + offs_rope[:, None],
                mask=valid_ids[None, :],
                other=0.0,
                cache_modifier=".cg",
            )
            qk = tl.dot(q_rope, rope_blk, qk, out_dtype=tl.float32)
            qk *= sm_scale
            qk = tl.where(valid_ids[None, :], qk, float("-inf"))

            kv_blk1 = tl.where(
                offs_d[:, None] < (NOPE - BDP),
                nope_tail,
                tl.load(
                    rope_ptr[None, :] + (offs_d[:, None] - (NOPE - BDP)),
                    mask=valid_ids[None, :] & (offs_d[:, None] >= (NOPE - BDP)),
                    other=0.0,
                    cache_modifier=".cg",
                ),
            )

            new_max = tl.maximum(max_log, tl.max(qk, axis=1))
            exp_qk = tl.math.exp(qk - new_max[:, None])
            sum_qk = tl.sum(exp_qk, axis=1)
            alpha = tl.math.exp(max_log - new_max)
            sum_exp = sum_exp * alpha + sum_qk
            exp_qk_t = tl.trans(exp_qk.to(tl.bfloat16))
            acc0 = tl.dot(
                kv_blk0,
                exp_qk_t,
                acc0 * alpha[None, :],
                out_dtype=tl.float32,
            )
            acc1 = tl.dot(
                kv_blk1,
                exp_qk_t,
                acc1 * alpha[None, :],
                out_dtype=tl.float32,
            )
            max_log = new_max

    valid_mask = max_log != float("-inf")
    orig_lse = max_log + tl.math.log(sum_exp)
    lse_out = tl.where(valid_mask, orig_lse, float("inf"))
    tl.store(l_base + offs_h * stride_lseh, lse_out)

    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + offs_h)
        sum_exp_new_lse = tl.math.exp(orig_lse) + tl.math.exp(sink)
        factor = tl.math.exp(max_log) / sum_exp_new_lse
    else:
        factor = 1.0 / sum_exp

    acc0_t = tl.trans(acc0)
    acc1_t = tl.trans(acc1)
    out_vals0 = tl.where(valid_mask[:, None], acc0_t * factor[:, None], 0.0)
    out_vals1 = tl.where(valid_mask[:, None], acc1_t * factor[:, None], 0.0)
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_d[None, :]
    tl.store(o_ptr, out_vals0.to(tl.bfloat16))
    tl.store(o_ptr + BDP, out_vals1.to(tl.bfloat16))


def ascend_sparse_decode_model1(
    q,
    kv,
    indices,
    out,
    lse,
    attn_sink,
    topk_length,
    extra_kv,
    extra_indices,
    extra_topk_length,
    batch_size,
    seq_q,
    num_heads_q,
    topk,
    page_block_size,
    softmax_scale,
):
    """Ascend MODEL1 sparse decode dispatch.

    When extra_kv is provided, uses two-pass workaround to avoid UB
    overflow.  Otherwise, launches the kernel directly.
    """
    if extra_kv is not None:
        _sparse_decode_model1_two_pass(
            q, kv, indices, out, lse,
            attn_sink, topk_length,
            extra_kv, extra_indices, extra_topk_length,
            batch_size, seq_q, num_heads_q,
            softmax_scale,
        )
    else:
        grid = lambda META: (
            batch_size * seq_q * triton.cdiv(num_heads_q, META["BH"]),
        )
        _sparse_decode_model1_kernel[grid](
            q,
            kv,
            indices,
            kv,
            indices,
            attn_sink if attn_sink is not None else None,
            topk_length if topk_length is not None else None,
            None,
            softmax_scale,
            out,
            lse,
            q.stride(0), q.stride(1), q.stride(2),
            kv.stride(0),
            indices.stride(0), indices.stride(1),
            kv.stride(0), indices.stride(0), indices.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            lse.stride(0), lse.stride(1),
            seq_q, num_heads_q,
            page_block_size, 1,
            kv.shape[0], 0,
            topk, 0,
            attn_sink is not None,
            topk_length is not None,
            False,
            False,
        )


def _sparse_decode_model1_two_pass(
    q, kv, indices, out, lse,
    attn_sink, topk_length,
    extra_kv, extra_indices, extra_topk_length,
    batch_size, seq_q, num_heads_q,
    softmax_scale,
):
    """Two-pass MODEL1 decode: run the kernel separately for the main KV
    cache and the extra KV cache, then combine the partial results using
    the flash-attention reduction formula.
    """
    out1 = torch.empty_like(out)
    lse1 = torch.empty_like(lse)
    out2 = torch.empty_like(out)
    lse2 = torch.empty_like(lse)

    extra_topk = extra_indices.shape[-1]
    extra_pbs = extra_kv.shape[1]

    def _launch(cur_q, cur_kv, cur_indices, cur_out, cur_lse,
                cur_topk_length, cur_pbs, cur_num_blocks, cur_topk):
        g = lambda META: (
            batch_size * seq_q * triton.cdiv(num_heads_q, META["BH"]),
        )
        _sparse_decode_model1_kernel[g](
            cur_q,
            cur_kv,
            cur_indices,
            cur_kv,
            cur_indices,
            None,
            cur_topk_length,
            None,
            softmax_scale,
            cur_out,
            cur_lse,
            cur_q.stride(0), cur_q.stride(1), cur_q.stride(2),
            cur_kv.stride(0),
            cur_indices.stride(0), cur_indices.stride(1),
            cur_kv.stride(0), cur_indices.stride(0), cur_indices.stride(1),
            cur_out.stride(0), cur_out.stride(1), cur_out.stride(2),
            cur_lse.stride(0), cur_lse.stride(1),
            seq_q, num_heads_q,
            cur_pbs, cur_pbs,
            cur_num_blocks, 0,
            cur_topk, 0,
            False,
            cur_topk_length is not None,
            False,
            False,
        )

    _launch(q, kv, indices, out1, lse1,
            topk_length, kv.shape[1], kv.shape[0], indices.shape[-1])
    _launch(q, extra_kv, extra_indices, out2, lse2,
            extra_topk_length, extra_pbs, extra_kv.shape[0], extra_topk)

    neg_inf = float("-inf")
    lse1_s = torch.where(torch.isinf(lse1), neg_inf, lse1)
    lse2_s = torch.where(torch.isinf(lse2), neg_inf, lse2)
    max_lse = torch.maximum(lse1_s, lse2_s)
    w1 = torch.exp(lse1_s - max_lse)
    w2 = torch.exp(lse2_s - max_lse)
    denom = w1 + w2

    if attn_sink is not None:
        w_sink = torch.exp(attn_sink[None, :, None] - max_lse)
        denom = denom + w_sink

    w1p = w1.permute(0, 2, 1).unsqueeze(-1)
    w2p = w2.permute(0, 2, 1).unsqueeze(-1)
    denomp = denom.permute(0, 2, 1).unsqueeze(-1)

    out_raw = (out1 * w1p + out2 * w2p) / denomp
    out.copy_(torch.nan_to_num(out_raw, nan=0.0))
    valid = denom > 0
    lse.copy_(torch.where(valid, max_lse + torch.log(denom), float("inf")))
