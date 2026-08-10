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

import torch
import triton
import triton.language as tl

_is_npu = not torch.cuda.is_available()

if _is_npu:
    indexer_fwd_configs = [
        triton.Config({"num_stages": 1, "num_warps": 4}),
        triton.Config({"num_stages": 1, "num_warps": 8}),
    ]
else:
    indexer_fwd_configs = [
        triton.Config({"num_stages": 2, "num_warps": 4}),
        triton.Config({"num_stages": 4, "num_warps": 8}),
    ]


@triton.autotune(  # Decorate the kernel
    configs=indexer_fwd_configs,
    key=["Q", "K", "H", "D"],
)
@triton.jit
def triton_lighting_indexer_k_tiled(
    q_index,
    k_index,
    cu_bg_seqlens,
    cu_ed_seqlens,
    weights,
    logits,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_wh,
    stride_lm,
    stride_ln,
    Q: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    TK: tl.constexpr,
    D: tl.constexpr,
    CU: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
):
    i_sh, i_k = tl.program_id(0), tl.program_id(1)

    offs_cu = tl.arange(0, BQ) + i_sh * BQ
    mask_cu = offs_cu < CU
    bos_vec = tl.load(cu_bg_seqlens + offs_cu, mask_cu, 1000000000) + i_k * TK
    eos_vec_raw = tl.load(cu_ed_seqlens + offs_cu, mask_cu, -1000000000)
    eos_vec = tl.minimum(eos_vec_raw, bos_vec + TK)
    bos = tl.maximum(tl.min(bos_vec, 0), 0)
    eos = tl.minimum(tl.max(eos_vec, 0), K)

    offs_bq = tl.arange(0, BQ * H) + i_sh * (BQ * H)
    offs_boq = tl.arange(0, BQ) + i_sh * BQ
    offs_d = tl.arange(0, D)
    mask_bq = offs_bq < Q * H
    mask_d = offs_d < D
    mask_boq = offs_boq < Q

    q_ptr = q_index + offs_bq[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_msk = mask_bq[:, None] & mask_d[None, :]
    q_blk = tl.load(q_ptr, q_msk, 0.0).to(tl.float32)  # [BQ*H, D]

    w_ptr = weights + offs_bq * stride_wh
    w_blk = tl.load(w_ptr, mask_bq, 0.0).to(tl.float32)  # [BQ*H]

    o_base = logits + bos * stride_ln
    k_base = k_index + bos * stride_kn

    NCK: tl.constexpr = (TK + BK - 1) // BK
    for ck in range(NCK):
        offs_bk = ck * BK + tl.arange(0, BK)
        abs_offs = bos + offs_bk
        mask_bk = (abs_offs < eos) & (offs_bk < TK)
        k_ptr = k_base + offs_d[:, None] * stride_kd + offs_bk[None, :] * stride_kn
        k_msk = mask_d[:, None] & mask_bk[None, :]
        k_blk = tl.load(k_ptr, k_msk, 0.0).to(tl.float32)
        acc = tl.dot(q_blk, k_blk, out_dtype=tl.float32)  # [BQ*H, BK]
        acc = tl.maximum(acc, 0.0) * w_blk[:, None]
        # Reshape [BQ*H, BK] -> [BQ, H, BK] and sum over H -> [BQ, BK]
        out_blk = acc.reshape(BQ, H, BK).sum(1)
        out_ptr = (
            o_base + offs_boq[:, None] * stride_lm + offs_bk[None, :] * stride_ln
        )
        out_msk = (
            mask_boq[:, None]
            & mask_bk[None, :]
            & (bos_vec[:, None] <= abs_offs[None, :])
            & (eos_vec[:, None] > abs_offs[None, :])
        )
        tl.store(out_ptr, out_blk, out_msk)


def triton_lighting_indexer_k_tiled_interface(
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke
):
    Q, H, D = q.shape[0], q.shape[1], q.shape[2]
    K = kv.shape[0]
    CU = cu_seqlen_ks.shape[0]
    logits = torch.full([Q, K], float("-inf"), device=q.device, dtype=torch.float32)
    BQ = 1
    BK = 64
    TK = 2048
    NQ = triton.cdiv(Q, BQ)
    NK = triton.cdiv(K, TK)
    grid = (NQ, NK)
    triton_lighting_indexer_k_tiled[grid](
        q,
        kv,
        cu_seqlen_ks,
        cu_seqlen_ke,
        weights,
        logits,
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        weights.stride(1),
        logits.stride(0),
        logits.stride(1),
        Q,
        H,
        K,
        TK,
        D,
        CU,
        BQ,
        BK,
    )
    return logits
