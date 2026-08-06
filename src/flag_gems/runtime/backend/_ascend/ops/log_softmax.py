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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend import heuristics_config_utils as _hcu
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.heuristics(_hcu.HEURISTICS_CONFIGS["softmax_non_inner"])
@triton.jit
def log_softmax_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets[None, :]
        mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
        inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        m = tl.max(inp, 0)
        z = tl.sum(tl.exp(inp - m[None, :]), 0)
        # z = tl.sum(tl.where(mask, tl.exp(inp - m[None, :]), 0.0), 0)
        out = inp - m[None, :] - tl.log(z)[None, :]
        tl.store(output_ptr + offsets, out, mask=mask)
    else:
        # 3-pass algorithm for numerical stability with large N:
        # Pass 1: find global max
        m = tl.full([TILE_K], value=float("-inf"), dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets[None, :]
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            m = tl.maximum(m, tl.max(inp, 0))

        # Pass 2: compute sum(exp(inp - m)) — no rescaling needed
        z = tl.zeros([TILE_K], dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets[None, :]
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            z += tl.sum(tl.exp(inp - m[None, :]), 0)
            # z += tl.sum(tl.where(mask, tl.exp(inp - m[None, :]), 0.0), 0)

        log_z = tl.log(z)

        # Pass 3: write output
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets[None, :]
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            out = inp - m[None, :] - log_z[None, :]
            tl.store(output_ptr + offsets, out, mask=mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("log_softmax"), key=["M", "N"])
@triton.jit
def log_softmax_kernel(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # 3-pass algorithm for numerical stability with large N:
    # Pass 1: find global max per row
    m = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.maximum(m, tl.max(inp, 1))

    # Pass 2: compute sum(exp(inp - m)) — no rescaling needed
    z = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        z += tl.sum(tl.exp(inp - m[:, None]), 1)
        # z += tl.sum(tl.where(mask, tl.exp(inp - m[:, None]), 0.0), 1)

    log_z = tl.log(z)

    # Pass 3: write output
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        o = inp - m[:, None] - log_z[:, None]
        tl.store(output_ptr + offset, o, mask=mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("log_softmax"), key=["M", "N"])
@triton.jit
def log_softmax_backward_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    scale = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        out_grad_ptrs = out_grad_ptr + offsets
        out_grad = tl.load(out_grad_ptrs, mask=mask, other=0.0).to(tl.float32)
        scale += out_grad
    scale = tl.sum(scale, 1)

    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        out_ptrs = out_ptr + offsets
        out = tl.load(out_ptrs, mask=mask, other=0.0).to(tl.float32)
        out_grad_ptrs = out_grad_ptr + offsets
        out_grad = tl.load(out_grad_ptrs, mask=mask, other=0.0).to(tl.float32)
        in_grad = out_grad - tl.exp(out) * scale[:, None]
        in_grad_ptrs = in_grad_ptr + offsets
        tl.store(in_grad_ptrs, in_grad, mask=mask)


def log_softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_ASCEND LOG_SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    inp = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype
    out = torch.empty_like(inp, dtype=dtype)
    K = inp.numel() // M // N

    with torch_device_fn.device(inp.device):
        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            log_softmax_kernel_non_inner[grid](out, inp, M, N, K)
        else:
            grid = lambda meta: (
                triton.cdiv(M, meta["BLOCK_M"]),
                K,
            )
            log_softmax_kernel[grid](out, inp, M, N, K)
    return out


def log_softmax_out(self, dim, half_to_float=False, *, out):
    logger.debug("GEMS_ASCEND LOG_SOFTMAX_OUT")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    inp = self.contiguous()
    K = inp.numel() // M // N

    with torch_device_fn.device(inp.device):
        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            log_softmax_kernel_non_inner[grid](out, inp, M, N, K)
        else:
            grid = lambda meta: (
                triton.cdiv(M, meta["BLOCK_M"]),
                K,
            )
            log_softmax_kernel[grid](out, inp, M, N, K)
    return out


def log_softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS_ASCEND LOG_SOFTMAX_VJP")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    in_grad = torch.empty_like(output, dtype=input_dtype)
    K = output.numel() // M // N

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch_device_fn.device(in_grad.device):
        log_softmax_backward_kernel[grid](
            output,
            grad_output,
            in_grad,
            M,
            N,
            K,
        )
    return in_grad
