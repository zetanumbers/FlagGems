import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


# TODO: Reference GitHub issue for multi-backend support
def npu_shifted_chebyshev_polynomial_w(x, n):
    """Pure-aten spliced implementation of the shifted Chebyshev W polynomial.

    torch_npu does not adapt aten::shifted_chebyshev_polynomial_w, so calling
    torch.special.shifted_chebyshev_polynomial_w on NPU silently falls back to
    CPU. This reference stitches the recurrence together from basic pointwise
    ops (broadcast/mul/sub/where/eq) that every backend supports, so it runs
    natively on device:

        y = 2x - 1
        W_0 = 1,  W_1 = 2y + 1,  W_k = 2y * W_{k-1} - W_{k-2}
        W_n = 0 for n < 0
    """
    if not isinstance(n, torch.Tensor):
        n = torch.as_tensor(n, dtype=torch.int64, device=x.device)

    # Mimic aten TensorIterator semantics: broadcast, then promote (float+int -> float)
    xb, nb = torch.broadcast_tensors(x, n)
    common_dtype = torch.promote_types(xb.dtype, nb.dtype)
    if not common_dtype.is_floating_point:
        raise NotImplementedError("only floating-point inputs are supported")
    xc = xb.to(common_dtype)
    deg = nb.to(common_dtype)

    y = 2 * xc - 1

    # Elements with n < 0 (or not yet covered by a recurrence layer) stay 0.
    result = torch.where(deg == 0, torch.ones_like(y), torch.zeros_like(y))
    result = torch.where(deg == 1, 2 * y + 1, result)

    p_prev = torch.ones_like(y)  # W_0
    p_curr = 2 * y + 1  # W_1

    # Host-side loop bound requires one D2H sync.
    max_deg = max(int(deg.max().item()), 1)
    for k in range(2, max_deg + 1):
        p_next = 2 * y * p_curr - p_prev
        result = torch.where(deg == k, p_next, result)
        p_prev, p_curr = p_curr, p_next

    return result


@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia" and flag_gems.vendor_name != "ascend",
    reason="requires NVIDIA or NPU backend ; not supported on other backends",
)
@pytest.mark.special_shifted_chebyshev_polynomial_w
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# PyTorch reference only supports float32 for this operator
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_shifted_chebyshev_polynomial_w(shape, dtype):
    # x: the values to evaluate the polynomial at
    # n: the degree of the shifted Chebyshev polynomial
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Cover full supported range n ∈ [0, 10]
    inp2 = torch.randint(0, 11, shape, dtype=torch.int32, device=flag_gems.device)

    # Reference: spliced from basic ops, computed in float64 on device to
    # keep the reference well above fp32 ulp noise, then rounded to fp32.
    # No CPU fallback involved, no dispatch registration (no use_gems conflict).
    ref_out = npu_shifted_chebyshev_polynomial_w(
        inp1.to(torch.float64), inp2.to(torch.float64)
    ).to(torch.float32)

    with flag_gems.use_gems():
        res_out = torch.special.shifted_chebyshev_polynomial_w(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia" and flag_gems.vendor_name != "ascend",
    reason="requires NVIDIA or NPU backend ; not supported on other backends",
)
@pytest.mark.special_shifted_chebyshev_polynomial_w
def test_special_shifted_chebyshev_polynomial_w_n_out_of_range():
    # Verify that n > 10 raises an error.
    # The validation lives in the Python wrapper function, not in the Triton kernel.
    # On Nvidia with PyTorch 2.10, torch.special dispatch goes through PyTorch's
    # native CUDA kernel and never reaches our wrapper, so we test the wrapper directly.
    from flag_gems.ops.special_shifted_chebyshev_polynomial_w import (
        special_shifted_chebyshev_polynomial_w,
    )

    x = torch.randn(3, dtype=torch.float32, device=flag_gems.device)
    n = torch.tensor([0, 5, 11], dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(ValueError, match="n must be <= 10"):
        special_shifted_chebyshev_polynomial_w(x, n)

