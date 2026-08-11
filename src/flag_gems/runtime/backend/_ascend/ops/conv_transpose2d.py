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

"""Ascend NPU dispatch for ``conv_transpose2d``.

The generic (GPU) implementation in ``flag_gems.ops.conv_transpose2d``
hard-codes CUDA device checks and uses ``tf32x3`` input precision for
``tl.dot``, neither of which is supported on Ascend NPU.

This module patches the GPU module at import time so that the shared
Triton kernels accept NPU tensors and use ``ieee`` input precision.
"""

import importlib

import torch

_gpu = importlib.import_module("flag_gems.ops.conv_transpose2d")

_GENERAL_TRITON_DTYPES = _gpu._GENERAL_TRITON_DTYPES


_original_validate = _gpu._validate_conv_transpose2d_args
_original_stride2 = _gpu._can_use_stride2_pad1_3x3_direct


def _validate_conv_transpose2d_args_npu(
    input,
    weight,
    bias,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    output_padding_h,
    output_padding_w,
    groups,
    dilation_h,
    dilation_w,
):
    if input.device.type not in ("cuda", "npu") or weight.device != input.device:
        return False
    if input.dim() != 4 or weight.dim() != 4:
        return False
    if not input.is_contiguous() or not weight.is_contiguous():
        return False
    if input.dtype not in _GENERAL_TRITON_DTYPES or weight.dtype != input.dtype:
        return False
    if (
        input.dtype is torch.bfloat16
        and input.device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        return False
    if bias is not None:
        if bias.device != input.device or bias.dtype != input.dtype:
            return False
        if bias.dim() != 1 or not bias.is_contiguous():
            return False
    if groups <= 0:
        raise RuntimeError("groups must be a positive integer")
    if stride_h <= 0 or stride_w <= 0:
        raise RuntimeError("non-positive stride is not supported")
    if dilation_h <= 0 or dilation_w <= 0:
        raise RuntimeError("dilation should be greater than zero")
    if padding_h < 0 or padding_w < 0:
        raise RuntimeError("negative padding is not supported")
    if output_padding_h < 0 or output_padding_w < 0:
        raise RuntimeError("negative output_padding is not supported")
    if output_padding_h >= stride_h and output_padding_h >= dilation_h:
        raise RuntimeError(
            "output padding must be smaller than either stride or dilation"
        )
    if output_padding_w >= stride_w and output_padding_w >= dilation_w:
        raise RuntimeError(
            "output padding must be smaller than either stride or dilation"
        )

    input_channels = input.shape[1]
    weight_input_channels = weight.shape[0]
    output_channels_per_group = weight.shape[1]
    weight_height = weight.shape[2]
    weight_width = weight.shape[3]
    if (
        input_channels <= 0
        or output_channels_per_group <= 0
        or weight_height <= 0
        or weight_width <= 0
    ):
        raise RuntimeError(
            "non-empty input channels and weight dimensions are required"
        )
    if input_channels != weight_input_channels:
        raise RuntimeError(
            "expected input channel dimension to match weight input channels"
        )
    if input_channels % groups != 0:
        raise RuntimeError("input channels must be divisible by groups")
    output_channels = output_channels_per_group * groups
    if bias is not None and bias.numel() != output_channels:
        raise RuntimeError("expected bias to have one element per output channel")

    input_height = input.shape[2]
    input_width = input.shape[3]
    output_height = (
        (input_height - 1) * stride_h
        - 2 * padding_h
        + dilation_h * (weight_height - 1)
        + output_padding_h
        + 1
    )
    output_width = (
        (input_width - 1) * stride_w
        - 2 * padding_w
        + dilation_w * (weight_width - 1)
        + output_padding_w
        + 1
    )
    if output_height <= 0 or output_width <= 0:
        raise RuntimeError("calculated output size is too small")
    return True


def _can_use_stride2_pad1_3x3_direct_npu(
    input,
    weight,
    bias,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    output_padding_h,
    output_padding_w,
    groups,
    dilation_h,
    dilation_w,
):
    return False


_gpu._validate_conv_transpose2d_args = _validate_conv_transpose2d_args_npu
_gpu._can_use_stride2_pad1_3x3_direct = _can_use_stride2_pad1_3x3_direct_npu


_KERNEL_NAMES = [
    "_conv_transpose2d_direct_kernel",
    "_conv_transpose2d_stride2_pad1_3x3_kernel",
    "_conv_transpose2d_residue_kernel",
    "_conv_transpose2d_residue_static_kernel",
    "_conv_transpose2d_scatter_no_overlap_kernel",
    "_conv_transpose2d_1x1_kernel",
]

for _name in _KERNEL_NAMES:
    _kernel = getattr(_gpu, _name, None)
    if _kernel is not None and hasattr(_kernel, "jit_function"):
        _jf = _kernel.jit_function
        _src_dict = getattr(_jf, "__dict__", None)
        if _src_dict is not None and "_src" in _src_dict and "tf32x3" in _src_dict["_src"]:
            _src_dict["_src"] = _src_dict["_src"].replace(
                'input_precision="tf32x3"', 'input_precision="ieee"'
            )


def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    return _gpu.conv_transpose2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )
