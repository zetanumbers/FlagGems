import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def fractional_max_pool2d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    in_h,
    in_w,
    out_h,
    out_w,
    out_stride_nc,
    BLOCK_W: tl.constexpr,
):
    """One program per (N*C) plane. Iterates over output rows sequentially,
    accumulates into grad_input via tl.atomic_add within a single program
    so no cross-program races occur."""
    pid_nc = tl.program_id(0)

    in_nc_offset = pid_nc * in_h * in_w
    w_offs = tl.arange(0, BLOCK_W)
    w_mask_base = w_offs < out_w

    for h in tl.range(0, out_h):
        out_offsets = pid_nc * out_stride_nc + h * out_w + w_offs
        w_mask = w_mask_base

        grad_vals = tl.load(grad_output_ptr + out_offsets, mask=w_mask, other=0.0).to(tl.float32)
        idx_vals = tl.load(indices_ptr + out_offsets, mask=w_mask, other=0)

        in_offsets = in_nc_offset + idx_vals
        valid = w_mask & (idx_vals >= 0)

        tl.atomic_add(grad_input_ptr + in_offsets, grad_vals, mask=valid)


def fractional_max_pool2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    kernel_size,
    output_size,
    indices: torch.Tensor,
):
    logger.debug("GEMS_ASCEND FRACTIONAL_MAX_POOL2D_BACKWARD")
    if isinstance(kernel_size, int):
        kernel_h, kernel_w = kernel_size, kernel_size
    else:
        kernel_h, kernel_w = kernel_size

    if isinstance(output_size, int):
        out_h = out_w = output_size
    else:
        out_h, out_w = output_size

    in_n, in_c, in_h, in_w = input.shape

    grad_output = grad_output.contiguous().to(torch.float32)
    indices = indices.contiguous().to(torch.int64)

    grad_input = torch.zeros(in_n, in_c, in_h, in_w, dtype=torch.float32, device=input.device)

    if grad_input.numel() == 0:
        return grad_input.to(input.dtype)

    BLOCK_W = triton.next_power_of_2(out_w)

    fractional_max_pool2d_backward_kernel[(in_n * in_c,)](
        grad_output,
        indices,
        grad_input,
        in_h,
        in_w,
        out_h,
        out_w,
        out_h * out_w,
        BLOCK_W,
    )

    return grad_input.to(input.dtype)
