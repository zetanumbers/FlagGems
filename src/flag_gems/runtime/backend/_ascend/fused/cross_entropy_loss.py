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
import triton
import triton.language as tl
from torch.nn import _reduction as _Reduction

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("cross_entropy_loss"),
    key=["C", "D"],
)
@triton.jit
def celoss_lse_kernel(
    inp_ptr,
    lse_ptr,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = ext.program_id(0)
    pid_n = ext.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tmp_max = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        mask = (offset_c[:, None] < C) & (offset_d[None, :] < D)
        inp = tl.load(inp_ptrs, mask=mask, other=0.0).to(tl.float32)
        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.where(mask, tl.exp(inp - cur_max), 0.0)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max
    final_max = tl.max(tmp_max, axis=0)
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max[None, :])
    _raw_sum = tl.sum(tmp_sum, axis=0)
    lse = tl.where(_raw_sum > 0, tl.log(_raw_sum), 0.0) + final_max
    tl.store(lse_ptr + pid_n * D + offset_d, lse, mask=offset_d < D)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("cross_entropy_loss"),
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["label_smoothing"])
def celoss_probability_kernel(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    out_ptr,
    lse_ptr,
    label_smoothing,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = ext.program_id(0)
    pid_n = ext.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offset_d < D

    lse = tl.load(lse_ptr + pid_n * D + offset_d, mask=d_mask, other=0.0)

    out = tl.zeros([BLOCK_D], dtype=tl.float32)
    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        tgt_ptrs = tgt_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        mask = (offset_c[:, None] < C) & d_mask[None, :]
        inp = tl.load(inp_ptrs, mask, other=0).to(tl.float32)
        tgt = tl.load(tgt_ptrs, mask, other=0).to(tl.float32)
        tgt = tgt * (1.0 - label_smoothing) + label_smoothing / C
        log = lse[None, :] - inp
        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask.to(tl.float32)
        else:
            w = tl.load(w_ptr + offset_c, mask=w_mask, other=0).to(tl.float32)
        out += tl.sum(tl.where(mask, log * tgt * w[:, None], 0.0), axis=0)

    out_ptrs = out_ptr + pid_n * D + offset_d
    tl.store(out_ptrs, out, mask=d_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("cross_entropy_loss"),
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["ignore_index", "label_smoothing"])
def celoss_indices_smooth_kernel(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    out_ptr,
    lse_ptr,
    ignore_index,
    label_smoothing,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = ext.program_id(0)
    pid_n = ext.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tgt_ptrs = tgt_ptr + pid_n * D + offset_d
    tgt_mask = offset_d < D
    tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)

    ignore_mask = (tgt != ignore_index) & tgt_mask

    lse = tl.load(lse_ptr + pid_n * D + offset_d, mask=tgt_mask, other=0.0)

    sum_smooth_w = tl.zeros([BLOCK_D], dtype=tl.float32)
    sum_smooth_wx = tl.zeros([BLOCK_D], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        mask = (offset_c[:, None] < C) & (offset_d[None, :] < D)
        inp = tl.load(inp_ptrs, mask=mask, other=0.0).to(tl.float32)

        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask.to(tl.float32)
        else:
            w = tl.load(w_ptr + offset_c, w_mask, other=0).to(tl.float32)

        smooth = tl.where(
            offset_c[:, None] == tgt[None, :],
            1 - label_smoothing + label_smoothing / C,
            label_smoothing / C,
        ).to(tl.float32)

        smooth_w = smooth * w[:, None]
        sum_smooth_w += tl.sum(tl.where(mask, smooth_w, 0.0), axis=0)
        sum_smooth_wx += tl.sum(tl.where(mask, smooth_w * inp, 0.0), axis=0)

    out = lse * sum_smooth_w - sum_smooth_wx
    out = tl.where(ignore_mask, out, 0)
    out_ptrs = out_ptr + pid_n * D + offset_d
    tl.store(out_ptrs, out, mask=tgt_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("cross_entropy_loss"),
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["ignore_index", "mean_num"])
def celoss_indices_bwd(
    out_grad_ptr,
    inp_ptr,
    tgt_ptr,
    wtgt_ptr,
    inp_grad_ptr,
    ignore_index,
    mean_num,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = ext.program_id(0)
    pid_n = ext.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tgt_ptrs = tgt_ptr + pid_n * D + offset_d
    tgt_mask = offset_d < D
    tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)
    out_grad_ptrs = out_grad_ptr + pid_n * D + offset_d
    out_grad = tl.load(out_grad_ptrs, mask=tgt_mask, other=0).to(tl.float32)[None, :]

    w_tgt = tl.load(wtgt_ptr + pid_n * D + offset_d, mask=tgt_mask, other=0).to(
        tl.float32
    )[None, :]

    ignore_mask = (tgt != ignore_index)[None, :]

    tmp_max = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp_mask = (offset_c[:, None] < C) & (offset_d[None, :] < D)
        inp = tl.load(inp_ptrs, mask=inp_mask, other=0.0).to(tl.float32)
        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.where(inp_mask, tl.exp(inp - cur_max), 0.0)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max
    final_max = tl.max(tmp_max, axis=0)[None, :]
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max)
    final_sum = tl.sum(tmp_sum, axis=0)[None, :]
    final_sum = tl.where(final_sum == 0, 1.0, final_sum)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp_mask = (offset_c[:, None] < C) & (offset_d[None, :] < D)
        inp = tl.load(inp_ptrs, mask=inp_mask, other=0.0).to(tl.float32)
        minus_one = offset_c[:, None] == tgt[None, :]
        inp_grad = (
            (tl.exp(inp - final_max) / final_sum - minus_one)
            * w_tgt
            * out_grad
            * mean_num
        )
        inp_grad_ptrs = (
            inp_grad_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        )
        tl.store(inp_grad_ptrs, inp_grad, mask=inp_mask & ignore_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("cross_entropy_loss"),
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["label_smoothing", "mean_num"])
def celoss_probability_bwd(
    out_grad_ptr,
    inp_ptr,
    tgt_ptr,
    w_ptr,
    inp_grad_ptr,
    label_smoothing,
    mean_num,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = ext.program_id(0)
    pid_n = ext.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    out_grad_ptrs = out_grad_ptr + pid_n * D + offset_d
    out_grad = tl.load(out_grad_ptrs, mask=offset_d < D, other=0).to(tl.float32)[
        None, :
    ]

    tmp_max = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    w_tgt_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        mask = (offset_c[:, None] < C) & (offset_d[None, :] < D)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp = tl.load(inp_ptrs, mask=mask, other=0.0).to(tl.float32)

        tgt_ptrs = tgt_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        tgt = tl.load(tgt_ptrs, mask, other=0).to(tl.float32)
        tgt = tgt * (1 - label_smoothing) + label_smoothing / C

        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask.to(tl.float32)
        else:
            w_ptrs = w_ptr + offset_c
            w = tl.load(w_ptrs, w_mask, other=0).to(tl.float32)

        w_tgt_sum += tgt * w[:, None]

        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.where(mask, tl.exp(inp - cur_max), 0.0)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max
    final_max = tl.max(tmp_max, axis=0)[None, :]
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max)
    final_sum = tl.sum(tmp_sum, axis=0)[None, :]
    final_sum = tl.where(final_sum == 0, 1.0, final_sum)
    w_tgt_sum = tl.sum(w_tgt_sum, axis=0)[None, :]

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        offset = pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp_ptrs = inp_ptr + offset
        mask = (offset_c[:, None] < C) & (offset_d[None, :] < D)
        inp = tl.load(inp_ptrs, mask, other=0).to(tl.float32)

        tgt_ptrs = tgt_ptr + offset
        tgt = tl.load(tgt_ptrs, mask, other=0).to(tl.float32)
        tgt = tgt * (1 - label_smoothing) + label_smoothing / C

        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask.to(tl.float32)
        else:
            w_ptrs = w_ptr + offset_c
            w = tl.load(w_ptrs, w_mask, other=0).to(tl.float32)

        grad = w_tgt_sum / final_sum * tl.exp(inp - final_max) - tgt * w[:, None]
        inp_grad = grad * out_grad * mean_num

        inp_grad_ptrs = inp_grad_ptr + offset
        tl.store(inp_grad_ptrs, inp_grad, mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("cross_entropy_loss"),
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["ignore_index", "label_smoothing", "mean_num"])
def celoss_indices_smooth_bwd(
    out_grad_ptr,
    inp_ptr,
    tgt_ptr,
    w_ptr,
    inp_grad_ptr,
    ignore_index,
    label_smoothing,
    mean_num,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = ext.program_id(0)
    pid_n = ext.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tgt_ptrs = tgt_ptr + pid_n * D + offset_d
    tgt_mask = offset_d < D
    tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)
    out_grad_ptrs = out_grad_ptr + pid_n * D + offset_d
    out_grad = tl.load(out_grad_ptrs, mask=tgt_mask, other=0).to(tl.float32)[None, :]

    ignore_mask = (tgt != ignore_index)[None, :]

    tmp_max = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    w_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp_mask = (offset_c[:, None] < C) & (offset_d[None, :] < D)
        inp = tl.load(inp_ptrs, mask=inp_mask, other=0.0).to(tl.float32)

        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask.to(tl.float32)
        else:
            w_ptrs = w_ptr + offset_c
            w = tl.load(w_ptrs, w_mask, other=0).to(tl.float32)

        smooth = tl.full([BLOCK_C, BLOCK_D], label_smoothing / C, dtype=tl.float32)
        smooth = tl.where(
            offset_c[:, None] == tgt[None, :],
            1 - label_smoothing + label_smoothing / C,
            smooth,
        )

        w_sum += smooth * w[:, None]

        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.where(inp_mask, tl.exp(inp - cur_max), 0.0)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max
    final_max = tl.max(tmp_max, axis=0)[None, :]
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max)
    final_sum = tl.sum(tmp_sum, axis=0)[None, :]
    final_sum = tl.where(final_sum == 0, 1.0, final_sum)
    w_sum = tl.sum(w_sum, axis=0)[None, :]

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp_mask = (offset_c[:, None] < C) & (offset_d[None, :] < D)
        inp = tl.load(inp_ptrs, mask=inp_mask, other=0.0).to(tl.float32)

        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask.to(tl.float32)
        else:
            w_ptrs = w_ptr + offset_c
            w = tl.load(w_ptrs, w_mask, other=0).to(tl.float32)

        smooth = tl.where(
            offset_c[:, None] == tgt[None, :],
            1 - label_smoothing + label_smoothing / C,
            label_smoothing / C,
        )

        grad = w_sum / final_sum * tl.exp(inp - final_max) - smooth * w[:, None]
        inp_grad = grad * out_grad * mean_num
        inp_grad = tl.where(ignore_mask, inp_grad, 0.0)
        inp_grad_ptrs = (
            inp_grad_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        )
        tl.store(inp_grad_ptrs, inp_grad, mask=inp_mask)


@libentry()
@triton.jit
def sum_and_scale(
    inp_ptr,
    out_ptr,
    N,
    scalebyw: tl.constexpr,
    BLOCK_N: tl.constexpr = 128,
    scale=1.0,
    mean_num=None,
):
    mid_sum = tl.zeros(
        [
            BLOCK_N,
        ],
        dtype=tl.float32,
    )
    if scalebyw:
        mid_wgt = tl.zeros(
            [
                BLOCK_N,
            ],
            dtype=tl.float32,
        )
        for off in range(0, N, BLOCK_N):
            offset = off + tl.arange(0, BLOCK_N)
            inp_ptrs = inp_ptr + offset
            mask = offset < N
            inp_vals = tl.load(inp_ptrs, mask=mask, other=0.0)
            mid_sum += inp_vals
            wgt_ptrs = scale + offset
            wgt_vals = tl.load(wgt_ptrs, mask=mask, other=0.0)
            mid_wgt += wgt_vals
        out_val = tl.sum(mid_sum)
        scale_val = tl.sum(mid_wgt)
        tl.store(mean_num, scale_val)
    else:
        for off in range(0, N, BLOCK_N):
            offset = off + tl.arange(0, BLOCK_N)
            inp_ptrs = inp_ptr + offset
            mask = offset < N
            inp_vals = tl.load(inp_ptrs, mask=mask, other=0.0)
            mid_sum += inp_vals
        out_val = tl.sum(mid_sum)
        scale_val = scale
    out_val /= scale_val
    tl.store(out_ptr, out_val)


class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, weight, reduction, ignore_index, label_smoothing):
        logger.debug("GEMS_ASCEND CROSS_ENTROPY_LOSS")

        shape = list(inp.shape)
        dim = inp.ndim
        N = 1 if dim == 1 else shape[0]
        C = shape[0] if dim == 1 else shape[1]
        D = inp.numel() // N // C
        axis = 0 if dim == 1 else 1
        del shape[axis]

        inp = inp.contiguous()
        tgt = target.contiguous()
        weight = weight.contiguous() if weight is not None else None
        grid = lambda meta: (triton.cdiv(D, meta["BLOCK_D"]), N)

        # Step 1: Compute LSE in a separate kernel (no gather, no weights)
        lse_buf = torch.empty(shape, dtype=torch.float32, device=inp.device)
        with torch_device_fn.device(inp.device):
            celoss_lse_kernel[grid](inp, lse_buf, C, D)

        # Step 2: Precompute w_tgt in PyTorch (avoids gather load bug on NPU)
        if tgt.ndim == dim:
            w_tgt = None
        else:
            ignore_mask = (tgt != ignore_index)
            if weight is not None:
                if D == 1:
                    w_tgt = weight[tgt]
                else:
                    w_tgt = weight[tgt.reshape(-1)].reshape(shape)
                w_tgt = torch.where(ignore_mask, w_tgt, torch.zeros_like(w_tgt))
            else:
                w_tgt = ignore_mask.to(torch.float32)
            w_tgt = w_tgt.to(torch.float32)

        # Step 3: Compute output
        out = torch.empty(shape, dtype=torch.float32, device=inp.device)

        if tgt.ndim == dim:
            inp_f = inp.float()
            tgt_f = tgt.float()
            tgt_f = tgt_f * (1.0 - label_smoothing) + label_smoothing / C
            log = lse_buf.unsqueeze(1) - inp_f
            if weight is not None:
                if D == 1:
                    out = (log * tgt_f * weight.float().unsqueeze(0)).sum(dim=1)
                else:
                    out = (log * tgt_f * weight.float().unsqueeze(0).unsqueeze(-1)).sum(dim=1)
            else:
                out = (log * tgt_f).sum(dim=1)
        elif label_smoothing == 0:
            # Non-smooth: compute entirely with precomputed values (no gather)
            if D == 1:
                inp_tgt = inp.gather(1, tgt.unsqueeze(1)).squeeze(1).float()
            else:
                idx = tgt.unsqueeze(1).expand(N, 1, D)
                inp_tgt = inp.gather(1, idx).squeeze(1).float()
            out = w_tgt * (lse_buf - inp_tgt)
        else:
            # Smooth: compute in PyTorch for precision (Triton float32 accumulation
            # loses precision for large C with label smoothing)
            inp_f = inp.float()
            tgt_clamped = tgt.clamp(min=0)
            tgt_factor = 1.0 - label_smoothing + label_smoothing / C
            if D == 1:
                smooth = torch.full(
                    (N, C), label_smoothing / C,
                    device=inp.device, dtype=torch.float32,
                )
                smooth.scatter_(1, tgt_clamped.unsqueeze(1), value=tgt_factor)
                if weight is not None:
                    smooth_w = smooth * weight.float()
                else:
                    smooth_w = smooth
                smooth_w = torch.where(ignore_mask.unsqueeze(1), smooth_w, 0.0)
            else:
                smooth = torch.full(
                    (N, C, D), label_smoothing / C,
                    device=inp.device, dtype=torch.float32,
                )
                smooth.scatter_(1, tgt_clamped.unsqueeze(1), value=tgt_factor)
                if weight is not None:
                    smooth_w = smooth * weight.float().unsqueeze(-1)
                else:
                    smooth_w = smooth
                smooth_w = torch.where(ignore_mask.unsqueeze(1), smooth_w, 0.0)
            sum_smooth_w = smooth_w.sum(dim=1)
            sum_smooth_wx = (smooth_w * inp_f).sum(dim=1)
            out = lse_buf * sum_smooth_w - sum_smooth_wx

        mean_num = 1
        if reduction == 1:  # MEAN
            if tgt.ndim == dim:
                out = out.sum() / (N * D)
                mean_num = torch.tensor(float(N * D), device=inp.device)
            else:
                wgt_sum = w_tgt.sum()
                out = out.sum() / wgt_sum
                mean_num = wgt_sum
        elif reduction == 2:  # SUM
            out = out.sum()

        if inp.requires_grad:
            ctx.save_for_backward(inp, tgt, weight)
            ctx.N = N
            ctx.C = C
            ctx.D = D
            ctx.ignore_index = ignore_index
            ctx.label_smoothing = label_smoothing
            ctx.shape = shape
            ctx.mean_num = mean_num

        return out.to(inp.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS_ASCEND CROSS_ENTROPY_LOSS_VJP")

        inp, tgt, weight = ctx.saved_tensors
        N = ctx.N
        C = ctx.C
        D = ctx.D
        ignore_index = ctx.ignore_index
        label_smoothing = ctx.label_smoothing
        mean_num = (
            1 / ctx.mean_num.item()
            if isinstance(ctx.mean_num, torch.Tensor)
            else 1 / ctx.mean_num
        )
        shape = ctx.shape

        out_grad = out_grad.broadcast_to(shape).contiguous()

        inp_grad = torch.zeros(inp.shape, dtype=inp.dtype, device=inp.device)
        grid = lambda meta: (triton.cdiv(D, meta["BLOCK_D"]), N)
        if tgt.ndim == inp.ndim:
            celoss_probability_bwd[grid](
                out_grad, inp, tgt, weight, inp_grad, label_smoothing, mean_num, C, D
            )
        elif label_smoothing == 0:
            # Precompute w_tgt to avoid gather load
            ignore_mask = (tgt != ignore_index)
            if weight is not None:
                if D == 1:
                    w_tgt = weight[tgt]
                else:
                    w_tgt = weight[tgt.reshape(-1)].reshape(shape)
                w_tgt = torch.where(ignore_mask, w_tgt, torch.zeros_like(w_tgt))
            else:
                w_tgt = ignore_mask.to(torch.float32)
            w_tgt = w_tgt.to(torch.float32).contiguous()
            celoss_indices_bwd[grid](
                out_grad, inp, tgt, w_tgt, inp_grad, ignore_index, mean_num, C, D
            )
        else:
            celoss_indices_smooth_bwd[grid](
                out_grad,
                inp,
                tgt,
                weight,
                inp_grad,
                ignore_index,
                label_smoothing,
                mean_num,
                C,
                D,
            )
        return inp_grad, None, None, None, None, None


def cross_entropy_loss(
    inp, target, weight=None, reduction="mean", ignore_index=-100, label_smoothing=0.0
):
    return CrossEntropyLoss.apply(
        inp,
        target,
        weight,
        _Reduction.get_enum(reduction),
        ignore_index,
        label_smoothing,
    )
