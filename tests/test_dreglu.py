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

from . import accuracy_utils as utils

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_OP = getattr(tex, "dreglu", None)
except ImportError:
    TE_OP = None

def _torch_dreglu(grad_output, input_tensor, quantizer=None):
    """Pure-PyTorch reference for the backward of ReGLU.

    ReGLU forward:  output = relu(a) * b   (input split along last dim)
    Backward:
        grad_a = grad_output * d_relu(a) * b
        grad_b = grad_output * relu(a)
    """
    N = input_tensor.shape[-1] // 2
    a = input_tensor[..., :N]
    b = input_tensor[..., N:]
    relu_a = torch.relu(a)
    d_relu_a = (a > 0).to(input_tensor.dtype)
    grad_a = grad_output * d_relu_a * b
    grad_b = grad_output * relu_a
    return torch.cat([grad_a, grad_b], dim=-1)


REF_OP = TE_OP if TE_OP is not None else _torch_dreglu


@pytest.mark.dreglu
@pytest.mark.parametrize("shape", utils.GLU_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dreglu(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    grad_output_shape = list(shape)
    grad_output_shape[-1] //= 2
    grad_output = torch.randn(
        tuple(grad_output_shape), dtype=dtype, device=flag_gems.device
    )

    ref_out = REF_OP(grad_output, input_tensor, None)
    ref_out = utils.to_reference(ref_out)
    with flag_gems.use_gems():
        res_out = flag_gems.dreglu(grad_output, input_tensor, None)
    utils.gems_assert_close(res_out, ref_out, dtype)
