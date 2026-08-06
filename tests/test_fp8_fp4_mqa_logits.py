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

try:
    from vllm.platforms import current_platform
    from vllm.third_party.deep_gemm.utils import per_custom_dims_cast_to_fp8
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits as vllm_fp8_fp4_mqa_logits

    VLLM_AVAILABLE = True
    SM90_AVAILABLE = current_platform.has_device_capability(90)
except ImportError:
    VLLM_AVAILABLE = False
    SM90_AVAILABLE = False

import flag_gems
from flag_gems.fused.fp8_fp4_mqa_logits import fp8_fp4_mqa_logits

from .accuracy_utils import gems_assert_close, to_reference

device = flag_gems.device

# DeepSeek V4 production config
H = 64
D = 128

# Test shapes: (M, N) covering decode and prefill workloads
DECODE_SHAPES = [(1, 1024), (1, 2048), (1, 4096), (4, 2048), (4, 4096)]
PREFILL_SHAPES = [
    (64, 4096),
    (256, 4096),
    (1024, 4096),
    (2048, 4096),
    (1024, 8192),
]



def _per_custom_dims_cast_to_fp8(x, custom_dims, use_ue8m0=False):
    """Native replacement for vLLM's per_custom_dims_cast_to_fp8.

    Quantizes x to FP8 e4m3fn.  For custom_dims=(0,) on a 2D (N, D) tensor,
    each row n gets its own scale factor computed from amax over dim 1.
    Returns (fp8_tensor, scale) where scale has shape (N,).
    """
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max

    if custom_dims == (0,):
        # Per-row scale: amax over all dims except 0, keepdim for broadcasting
        reduce_dims = tuple(d for d in range(x.dim()) if d not in custom_dims)
        scale = x.float().abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-4) / fp8_max
        x_scaled = (x.float() / scale).clamp(-fp8_max, fp8_max)
        return x_scaled.to(fp8_dtype), scale.squeeze(-1).contiguous()
    else:
        scale = x.float().abs().amax().clamp_min(1e-4) / fp8_max
        x_scaled = (x.float() / scale).clamp(-fp8_max, fp8_max)
        return x_scaled.to(fp8_dtype), scale.contiguous()


def _native_fp8_fp4_mqa_logits_ref(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke,
                                   clean_logits=True):
    """Native PyTorch reference for fp8_fp4_mqa_logits.

    Computes: logits[m, n] = sum_h(ReLU(sum_d(q[m,h,d]*k[n,d]) * k_scale[n])
                                    * weights[m, h])

    Then if clean_logits, fills positions outside [ks, ke) with -inf.
    """
    q_values, _ = q
    k_values, k_scales = kv

    M, H, D = q_values.shape
    N = k_values.shape[0]

    q_f = q_values.to(torch.float32)   # (M, H, D)
    k_f = k_values.to(torch.float32)   # (N, D)
    k_scale = k_scales.to(torch.float32)  # (N,)

    # dot[q,h,n] = sum_d q[m,h,d] * k[n,d]
    # = einsum("mhd,nd->mhn")
    dot = torch.einsum("mhd,nd->mhn", q_f, k_f)  # (M, H, N)

    # Apply k_scale and ReLU
    dot = torch.relu(dot * k_scale[None, None, :])  # (M, H, N)

    # Apply per-head weights
    w = weights.to(torch.float32)  # (M, H)
    dot = dot * w[:, :, None]  # (M, H, N)

    # Reduce over heads
    logits = dot.sum(dim=1)  # (M, N)

    if clean_logits:
        for m in range(M):
            ks = int(cu_seqlen_ks[m].item())
            ke = int(cu_seqlen_ke[m].item())
            mask = torch.zeros(N, dtype=torch.bool, device=logits.device)
            mask[ks:ke] = True
            logits[m] = torch.where(mask, logits[m], torch.full((N,), float("-inf"), device=logits.device))

    return logits



def _build_inputs(M, N, device):
    """Build FP8 quantized inputs matching vLLM DeepGEMM conventions."""
    torch.manual_seed(42)

    q_bf16 = torch.randn(M, H, D, device=device, dtype=torch.bfloat16)
    k_bf16 = torch.randn(N, D, device=device, dtype=torch.bfloat16)
    weights = torch.randn(M, H, device=device, dtype=torch.float32).abs()

    q_fp8 = q_bf16.to(torch.float8_e4m3fn)
    if VLLM_AVAILABLE:
        k_fp8, k_scale = per_custom_dims_cast_to_fp8(k_bf16, (0,), False)
    else:
        k_fp8, k_scale = _per_custom_dims_cast_to_fp8(k_bf16, (0,), False)

    ks = torch.zeros(M, dtype=torch.int32, device=device)
    ke = torch.full((M,), N, dtype=torch.int32, device=device)

    return q_fp8, k_fp8, k_scale, weights, ks, ke


@pytest.mark.fp8_fp4_mqa_logits
@pytest.mark.parametrize(
    "M, N",
    DECODE_SHAPES + PREFILL_SHAPES,
    ids=[f"{m}x{n}" for m, n in DECODE_SHAPES + PREFILL_SHAPES],
)
@pytest.mark.parametrize("clean_logits", [True, False])
def test_fp8_fp4_mqa_logits(M, N, clean_logits):
    q_fp8, k_fp8, k_scale, weights, ks, ke = _build_inputs(M, N, device)
    if VLLM_AVAILABLE:
        ref_out = vllm_fp8_fp4_mqa_logits(
            q=(q_fp8, None),
            kv=(k_fp8, k_scale),
            weights=weights,
            cu_seqlen_ks=ks,
            cu_seqlen_ke=ke,
            clean_logits=clean_logits,
        )
    else:
        ref_out = _native_fp8_fp4_mqa_logits_ref(
            q=(q_fp8, None),
            kv=(k_fp8, k_scale),
            weights=weights,
            cu_seqlen_ks=ks,
            cu_seqlen_ke=ke,
            clean_logits=clean_logits,
        )
    ref_out = to_reference(ref_out)

    with flag_gems.use_gems():
        res_out = fp8_fp4_mqa_logits(
            q=(q_fp8, None),
            kv=(k_fp8, k_scale),
            weights=weights,
            cu_seqlen_ks=ks,
            cu_seqlen_ke=ke,
            clean_logits=clean_logits,
        )

    gems_assert_close(
        res_out, ref_out, res_out.dtype, equal_nan=True, atol=5e-2, reduce_dim=1
    )
