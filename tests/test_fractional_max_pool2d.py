import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _native_fractional_max_pool2d(
    input, kernel_size, output_size, return_indices=True, _random_samples=None
):
    if isinstance(kernel_size, int):
        kernel_h, kernel_w = kernel_size, kernel_size
    else:
        kernel_h, kernel_w = kernel_size
    if isinstance(output_size, int):
        out_h, out_w = output_size, output_size
    else:
        out_h, out_w = output_size

    N, C, in_h, in_w = input.shape
    alpha_h = (in_h - kernel_h) / (out_h - 1) if out_h > 1 else 0.0
    alpha_w = (in_w - kernel_w) / (out_w - 1) if out_w > 1 else 0.0

    sample_h = _random_samples[..., 0].to(torch.float64)
    sample_w = _random_samples[..., 1].to(torch.float64)

    h_off = torch.arange(out_h, device=input.device, dtype=torch.float64)
    h_starts = (
        (h_off.view(1, 1, -1) + sample_h.unsqueeze(2)) * alpha_h
    ).to(torch.int64) - (sample_h.unsqueeze(2) * alpha_h).to(torch.int64)
    h_starts[..., -1] = in_h - kernel_h

    w_off = torch.arange(out_w, device=input.device, dtype=torch.float64)
    w_starts = (
        (w_off.view(1, 1, -1) + sample_w.unsqueeze(2)) * alpha_w
    ).to(torch.int64) - (sample_w.unsqueeze(2) * alpha_w).to(torch.int64)
    w_starts[..., -1] = in_w - kernel_w

    h_starts_exp = h_starts.unsqueeze(3).expand(N, C, out_h, out_w)
    w_starts_exp = w_starts.unsqueeze(2).expand(N, C, out_h, out_w)

    inp_flat = input.reshape(N, C, in_h * in_w)
    max_val = torch.full(
        (N, C, out_h, out_w), float("-inf"), dtype=torch.float32, device=input.device
    )
    max_idx = torch.zeros((N, C, out_h, out_w), dtype=torch.int64, device=input.device)

    for kh in range(kernel_h):
        for kw in range(kernel_w):
            h_in = h_starts_exp + kh
            w_in = w_starts_exp + kw
            flat = (h_in * in_w + w_in).reshape(N, C, out_h * out_w)
            cur_val = torch.gather(inp_flat, 2, flat).reshape(N, C, out_h, out_w).to(torch.float32)
            cur_idx = h_in * in_w + w_in
            upd = cur_val > max_val
            max_val = torch.where(upd, cur_val, max_val)
            max_idx = torch.where(upd, cur_idx, max_idx)

    return max_val.to(input.dtype), max_idx


def _native_fractional_max_pool2d_backward(
    grad_output, input, kernel_size, output_size, indices
):
    if isinstance(kernel_size, int):
        kernel_h, kernel_w = kernel_size, kernel_size
    else:
        kernel_h, kernel_w = kernel_size
    if isinstance(output_size, int):
        out_h, out_w = output_size, output_size
    else:
        out_h, out_w = output_size

    N, C, in_h, in_w = input.shape
    grad_input = torch.zeros(
        N, C, in_h * in_w, dtype=torch.float32, device=input.device
    )
    grad_flat = grad_output.reshape(N, C, -1).to(torch.float32)
    idx_flat = indices.reshape(N, C, -1).to(torch.int64)
    grad_input.scatter_add_(2, idx_flat, grad_flat)
    return grad_input.reshape(N, C, in_h, in_w)

# Custom shapes for fractional_max_pool2d: (batch, channels, height, width), kernel_size, output_size
FRACTIONAL_MAXPOOL2D_CONFIGS = [
    # Classic case: 2x2 kernel, output size 16x16
    ((4, 3, 32, 32), 2, (16, 16)),
    # Different output sizes
    ((8, 16, 28, 28), 2, (14, 14)),
    # Test different output sizes
    ((2, 4, 20, 20), 2, (10, 10)),
    # Larger case
    ((1, 64, 56, 56), 2, (28, 28)),
    # No padding, different kernel sizes
    ((2, 8, 16, 16), 2, (8, 8)),
    # Non-square output
    ((2, 8, 16, 20), 2, (8, 10)),
]


@pytest.mark.fractional_max_pool2d
@pytest.mark.parametrize(
    "shape, kernel_size, output_size", FRACTIONAL_MAXPOOL2D_CONFIGS
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fractional_max_pool2d(shape, kernel_size, output_size, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, True)

    # Use the same random_samples for both to ensure identical pooling regions.
    # PyTorch C++ requires _random_samples dtype == input dtype.
    # Our implementation converts to double internally, so we pass float32 to ours.
    # Both paths ultimately compute intervals with the same float64 values.
    N, C = shape[0], shape[1]
    random_samples = torch.rand(N, C, 2, dtype=torch.float32, device=flag_gems.device)
    ref_random_samples = random_samples.to(device=ref_inp.device, dtype=ref_inp.dtype)
    if flag_gems.vendor_name == "ascend":
        ref_out, ref_indices = _native_fractional_max_pool2d(
            ref_inp,
            kernel_size=kernel_size,
            output_size=output_size,
            return_indices=True,
            _random_samples=ref_random_samples,
        )
    else:
        ref_out, ref_indices = torch.nn.functional.fractional_max_pool2d(
            ref_inp,
            kernel_size=kernel_size,
            output_size=output_size,
            return_indices=True,
            _random_samples=ref_random_samples,
        )

    res_out, res_indices = flag_gems.fractional_max_pool2d(
        inp,
        kernel_size=kernel_size,
        output_size=output_size,
        _random_samples=random_samples,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    # Indices should match exactly since we use the same random_samples
    torch.testing.assert_close(res_indices.to(ref_indices.device), ref_indices)


@pytest.mark.fractional_max_pool2d_backward
@pytest.mark.parametrize(
    "shape, kernel_size, output_size", FRACTIONAL_MAXPOOL2D_CONFIGS
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fractional_max_pool2d_backward(shape, kernel_size, output_size, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, upcast=True)

    # Use the same random_samples for both.
    N, C = shape[0], shape[1]
    random_samples = torch.rand(N, C, 2, dtype=torch.float32, device=flag_gems.device)
    ref_random_samples = random_samples.to(device=ref_inp.device, dtype=ref_inp.dtype)
    if flag_gems.vendor_name == "ascend":
        ref_out, ref_indices = _native_fractional_max_pool2d(
            ref_inp,
            kernel_size=kernel_size,
            output_size=output_size,
            return_indices=True,
            _random_samples=ref_random_samples,
        )
    else:
        ref_out, ref_indices = torch.nn.functional.fractional_max_pool2d(
            ref_inp,
            kernel_size=kernel_size,
            output_size=output_size,
            return_indices=True,
            _random_samples=ref_random_samples,
        )
    res_out, res_indices = flag_gems.fractional_max_pool2d(
        inp,
        kernel_size=kernel_size,
        output_size=output_size,
        _random_samples=random_samples,
    )

    out_grad = torch.randn_like(res_out, device=flag_gems.device)
    ref_grad = utils.to_reference(out_grad, upcast=True)
    if flag_gems.vendor_name == "ascend":
        ref_in_grad = _native_fractional_max_pool2d_backward(
            ref_grad,
            ref_inp,
            kernel_size=kernel_size,
            output_size=output_size,
            indices=ref_indices,
        )
        res_in_grad = flag_gems.fractional_max_pool2d_backward(
            out_grad,
            inp,
            kernel_size=kernel_size,
            output_size=output_size,
            indices=res_indices,
        ).to(dtype)
    else:
        (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
        res_in_grad = flag_gems.fractional_max_pool2d_backward(
            out_grad,
            inp,
            kernel_size=kernel_size,
            output_size=output_size,
            indices=res_indices,
        )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
