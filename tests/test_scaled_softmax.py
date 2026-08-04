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
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    ATTN_HEADS = [2]
else:
    ATTN_HEADS = [2, 4, 8, 16, 32]

# Make sure every thread has same seed.
random.seed(time.time() // 100)


@pytest.mark.scaled_softmax_forward
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("attn_heads", ATTN_HEADS)
@pytest.mark.parametrize("query_seq_len", [64, 128])
@pytest.mark.parametrize("key_seq_len", [128, 256, 512, 1024])
@pytest.mark.parametrize("scale_factor", [0.1])
def test_scaled_softmax_forward(
    batch_size, attn_heads, query_seq_len, key_seq_len, scale_factor, dtype
):
    # try:
    #     from transformer_engine.pytorch import cpp_extensions as tex
    # except ImportError:
    #     pytest.skip("transformer_engine_torch is not available, skipping accuracy test")
    #
    # te_scaled_softmax_forward = getattr(tex, "scaled_softmax_forward", None)
    # if te_scaled_softmax_forward is None:
    #     pytest.skip("'scaled_softmax_forward' not found in TransformerEngine")

    s = torch.randn(
        (batch_size, attn_heads, query_seq_len, key_seq_len),
        dtype=dtype,
        device=flag_gems.device,
    )

    s_ref = utils.to_reference(s, upcast=True)
    p_ref = torch.softmax(s_ref * scale_factor, dim=-1)
    with flag_gems.use_gems():
        p = flag_gems.scaled_softmax_forward(s, scale_factor)

    utils.gems_assert_close(p, p_ref, dtype, equal_nan=True)


@pytest.mark.scaled_softmax_backward
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("attn_heads", ATTN_HEADS)
@pytest.mark.parametrize("query_seq_len", [64, 128])
@pytest.mark.parametrize("key_seq_len", [128, 256, 512, 1024])
@pytest.mark.parametrize("scale_factor", [0.1])
def test_scaled_softmax_backward(
    batch_size, attn_heads, query_seq_len, key_seq_len, scale_factor, dtype
):
    # try:
    #     from transformer_engine.pytorch import cpp_extensions as tex
    # except ImportError:
    #     pytest.skip("transformer_engine_torch is not available, skipping accuracy test")
    #
    # te_scaled_softmax_forward = getattr(tex, "scaled_softmax_forward", None)
    # te_scaled_softmax_backward = getattr(tex, "scaled_softmax_backward", None)
    # if te_scaled_softmax_forward is None:
    #     pytest.skip("'scaled_softmax_forward' not found in TransformerEngine")
    # if te_scaled_softmax_backward is None:
    #     pytest.skip("'scaled_softmax_backward' not found in TransformerEngine")

    out_grad = torch.randn(
        (batch_size, attn_heads, query_seq_len, key_seq_len),
        dtype=dtype,
        device=flag_gems.device,
    )
    s = torch.randn(
        (batch_size, attn_heads, query_seq_len, key_seq_len),
        dtype=dtype,
        device=flag_gems.device,
    )

    s_ref = utils.to_reference(s, upcast=True).detach().clone().requires_grad_(True)
    p_ref = torch.softmax(s_ref * scale_factor, dim=-1)
    out_grad_ref = utils.to_reference(out_grad, upcast=True)
    (in_grad_ref,) = torch.autograd.grad(
        p_ref, s_ref, grad_outputs=out_grad_ref
    )
    #te_scaled_softmax_forward(s, scale_factor)
    with flag_gems.use_gems():
        p = flag_gems.scaled_softmax_forward(s, scale_factor)
        in_grad = flag_gems.scaled_softmax_backward(out_grad, p, scale_factor)

    utils.gems_assert_close(
        in_grad, in_grad_ref, dtype, equal_nan=True, reduce_dim=s.shape[-1]
    )
