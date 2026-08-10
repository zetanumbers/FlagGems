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

# LSTM cell backward test shapes
LSTM_SHAPES = [
    (1, 4),  # batch=1, hidden=4
    (2, 8),  # batch=2, hidden=8
    (4, 16),  # batch=4, hidden=16
    (8, 32),  # batch=8, hidden=32
    (16, 64),  # batch=16, hidden=64
]


@pytest.mark.thnn_fused_lstm_cell_backward_impl
@pytest.mark.skipif(
    flag_gems.vendor_name != "cuda",
    reason="torch.ops.aten._thnn_fused_lstm_cell only supported on CUDA",
)
@pytest.mark.parametrize("shape", LSTM_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_thnn_fused_lstm_cell_backward_impl_cuda(shape, dtype):
    """Test accuracy for _thnn_fused_lstm_cell_backward_impl."""
    batch_size, hidden_size = shape
    dev = flag_gems.device

    # Create input tensors on CUDA (forward op is CUDA-only)
    input_gates = torch.randn(batch_size, 4 * hidden_size, dtype=dtype, device=dev)
    hidden_gates = torch.randn(batch_size, 4 * hidden_size, dtype=dtype, device=dev)
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=dev)
    input_bias = torch.zeros(4 * hidden_size, dtype=dtype, device=dev)
    hidden_bias = torch.randn(4 * hidden_size, dtype=dtype, device=dev)

    # Forward pass (CUDA-only; _thnn_fused_lstm_cell has no CPU kernel)
    hx, cy, workspace = torch.ops.aten._thnn_fused_lstm_cell(
        input_gates, hidden_gates, cx, input_bias, hidden_bias
    )

    # Gradient tensors for backward
    grad_hy = torch.randn_like(hx)
    grad_cy = torch.randn_like(cy)

    # Backward pass — ATen reference (outside use_gems, on CUDA)
    ref_out = torch.ops.aten._thnn_fused_lstm_cell_backward_impl(
        grad_hy, grad_cy, cx, cy, workspace, True
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._thnn_fused_lstm_cell_backward_impl(
            grad_hy, grad_cy, cx, cy, workspace, True
        )

    # Compare outputs — ref_out order: (grad_input_gates, grad_cx, grad_biases)
    for i, (ref, res) in enumerate(zip(ref_out, res_out)):
        assert (
            res.shape == ref.shape
        ), f"Shape mismatch at output[{i}]: {res.shape} vs {ref.shape}"
        assert (
            res.dtype == ref.dtype
        ), f"Dtype mismatch at output[{i}]: {res.dtype} vs {ref.dtype}"
        utils.gems_assert_close(res.cpu(), ref.cpu(), dtype, atol=5e-2)



@pytest.mark.thnn_fused_lstm_cell_backward_impl
@pytest.mark.parametrize("shape", LSTM_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_thnn_fused_lstm_cell_backward_impl(shape, dtype):
    """Test accuracy for _thnn_fused_lstm_cell_backward_impl."""
    batch_size, hidden_size = shape
    dev = flag_gems.device
    torch.manual_seed(42)

    # Create input tensors on device
    input_gates = torch.randn(batch_size, 4 * hidden_size, dtype=dtype, device=dev)
    hidden_gates = torch.randn(batch_size, 4 * hidden_size, dtype=dtype, device=dev)
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=dev)
    input_bias = torch.zeros(4 * hidden_size, dtype=dtype, device=dev)
    hidden_bias = torch.randn(4 * hidden_size, dtype=dtype, device=dev)

    # Reference forward on CPU
    ref_inputs_gates = utils.to_reference(input_gates)
    ref_hidden_gates = utils.to_reference(hidden_gates)
    ref_cx = utils.to_reference(cx)
    ref_input_bias = utils.to_reference(input_bias)
    ref_hidden_bias = utils.to_reference(hidden_bias)

    gates = ref_inputs_gates + ref_hidden_gates + ref_input_bias + ref_hidden_bias
    i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)
    i_gate = torch.sigmoid(i_gate)
    f_gate = torch.sigmoid(f_gate)
    g_gate = torch.tanh(g_gate)
    o_gate = torch.sigmoid(o_gate)
    ref_cy = f_gate * ref_cx + i_gate * g_gate
    ref_workspace = torch.cat([i_gate, f_gate, g_gate, o_gate], dim=1)

    # Gradient tensors(same shape as hy and cy)
    grad_hy = torch.randn(batch_size, hidden_size, dtype=dtype, device=dev)
    grad_cy = torch.randn(batch_size, hidden_size, dtype=dtype, device=dev)
    ref_grad_hy = utils.to_reference(grad_hy)
    ref_grad_cy = utils.to_reference(grad_cy)

    # Reference backward on CPU
    tanh_cy = torch.tanh(ref_cy)
    d_cy = ref_grad_cy * o_gate * (1.0 - tanh_cy * tanh_cy) + ref_grad_cy
    grad_i = d_cy * g_gate * i_gate * (1.0 - i_gate)
    grad_f = d_cy * ref_cx * f_gate * (1.0 - f_gate)
    grad_g = d_cy * i_gate * (1.0 - g_gate)
    grad_o = ref_grad_hy * tanh_cy * o_gate * (1.0 - o_gate)
    ref_grad_input_gates = torch.cat([grad_i, grad_f, grad_g, grad_o], dim=1)
    ref_grad_cx = d_cy * f_gate
    ref_grad_biases = ref_grad_input_gates.sum(dim=0)

    cy_device = ref_cy.to(dtype=dtype, device=dev)
    workspace_device = ref_workspace.to(dtype=dtype, device=dev)

    with flag_gems.use_gems():
        res_grad_input_gates, res_grad_cx, res_grad_biases = (
            torch.ops.aten._thnn_fused_lstm_cell_backward_impl(
                grad_hy, grad_cy, cx, cy_device, workspace_device, True
            )
        )

    # Compare outputs — ref_out order: (grad_input_gates, grad_cx, grad_biases)
    ref_out = (ref_grad_input_gates, ref_grad_cx, ref_grad_biases)
    res_out = (res_grad_input_gates, res_grad_cx, res_grad_biases)

    for i, (ref, res) in enumerate(zip(ref_out, res_out)):
        assert (
                res.shape == ref.shape
        ), f"Shape mismatch at output[{i}]: {res.shape} vs {ref.shape}"
        assert (
                res.dtype == ref.dtype
        ), f"Dtype mismatch at output[{i}]: {res.dtype} vs {ref.dtype}"
        utils.gems_assert_close(res.cpu(), ref.cpu(), dtype, atol=5e-2)
