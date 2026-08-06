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
from flag_gems import topk_softmax

from . import accuracy_utils as utils

# Make sure every thread has same seed.
random.seed(time.time() // 100)


def torch_topk_softmax(
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        token_expert_indices: torch.Tensor,
        gating_output: torch.Tensor,
        renormalize: bool = False,
) -> None:
    num_tokens, num_experts = gating_output.shape
    k = topk_weights.size(-1)

    logits = gating_output.to(torch.float32)
    row_max = logits.max(dim=-1, keepdim=True).values
    exp_vals = torch.exp(logits - row_max)
    probs = exp_vals / (exp_vals.sum(dim=-1, keepdim=True) + 1e-8)

    weights = torch.empty_like(topk_weights, dtype=torch.float32)
    indices = torch.empty_like(topk_indices, dtype=torch.int64)
    token_expert = torch.empty_like(token_expert_indices, dtype=torch.int32)

    selected_sum = torch.zeros(num_tokens, dtype=torch.float32, device=probs.device)
    for ki in range(k):
        curr_max, curr_arg = probs.max(dim=-1)
        weights[:, ki] = curr_max
        indices[:, ki] = curr_arg.to(torch.int64)
        token_expert[:, ki] = (ki * num_tokens + torch.arange(num_tokens, device=probs.device, dtype=torch.int32)).to(torch.int32)
        if renormalize:
            selected_sum = selected_sum + curr_max
        probs[torch.arange(num_tokens, device=probs.device), curr_arg] = float("-inf")
    if renormalize:
        norm = selected_sum + 1e-8
        weights = weights / norm.unsqueeze(-1)
    topk_weights.copy_(weights.to(topk_weights.dtype))
    topk_indices.copy_(indices.to(topk_indices.dtype))
    token_expert_indices.copy_(token_expert.to(token_expert_indices.dtype))

def generate_test_params():
    params = [torch.int32, torch.int64]
    if utils.SkipVersion("torch", ">2.2"):
        params.append(torch.uint32)

    return params


@pytest.mark.skipif(
    flag_gems.vendor_name == "metax", reason="Issue #2857: RuntimeError"
)
@pytest.mark.topk_softmax
@pytest.mark.parametrize("index_dtype", generate_test_params())
@pytest.mark.parametrize(
    "num_tokens, num_experts, topk",
    [
        (1, 4, 2),
        (4, 8, 2),
        (8, 16, 4),
        (32, 64, 8),
        (128, 128, 16),
        (500, 255, 30),
        (512, 256, 32),
        (1024, 512, 32),
    ],
)
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_softmax(
    num_tokens, num_experts, topk, input_dtype, index_dtype, renormalize
):
    if flag_gems.vendor_name == "mthreads" and index_dtype == torch.uint32:
        # Issue #2858: torch musa does not support uint32
        index_dtype = torch.int64

    torch.manual_seed(42)
    device = flag_gems.device

    gating_output = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )

    ref_weights = torch.empty(num_tokens, topk, device=device, dtype=torch.float32)
    ref_indices = torch.empty(num_tokens, topk, device=device, dtype=index_dtype)
    ref_token_expert = torch.empty(num_tokens, topk, device=device, dtype=torch.int32)

    torch_topk_softmax(
        ref_weights,
        ref_indices,
        ref_token_expert,
        gating_output,
        renormalize,
    )

    gems_weights = torch.empty_like(ref_weights)
    gems_indices = torch.empty_like(ref_indices)
    gems_token_expert = torch.empty_like(ref_token_expert)

    topk_softmax(
        gems_weights,
        gems_indices,
        gems_token_expert,
        gating_output,
        renormalize,
    )

    assert torch.allclose(
        gems_weights, ref_weights, atol=1e-5
    ), "topk_weights mismatch"
    assert torch.equal(
        gems_indices.cpu(), ref_indices.cpu()
    ), "topk_indices mismatch (fp32)"
    assert torch.equal(
        gems_token_expert.cpu(), ref_token_expert.cpu()
    ), "token_expert_indices mismatch"

    if renormalize:
        sums = gems_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
