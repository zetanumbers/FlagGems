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

    TE_OP = getattr(tex, "geglu", None)
except ImportError:
    TE_OP = None


def _geglu_pytorch(input_tensor, quantizer=None):
    H = input_tensor.shape[-1] // 2
    x = input_tensor.to(torch.float32)
    a, b = x[..., :H], x[..., H:]
    out = torch.nn.functional.gelu(a, approximate="tanh") * b
    return out.to(input_tensor.dtype)


REF_OP = TE_OP if TE_OP is not None else _geglu_pytorch


@pytest.mark.geglu
@pytest.mark.parametrize("shape", utils.GLU_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_geglu(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = REF_OP(input_tensor, None)
    ref_out = utils.to_reference(ref_out)

    with flag_gems.use_gems():
        res_out = flag_gems.geglu(input_tensor)
    utils.gems_assert_close(res_out, ref_out, dtype)
