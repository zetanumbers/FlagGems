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

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.log_normal_
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log_normal_(shape, dtype):
    # Test log_normal_ with standard log-normal parameters (mean=0, std=1)
    # For log-normal distribution: if X ~ N(mean, std), then exp(X) ~ LogNormal(mean, std)
    # Expected mean = exp(mean + std^2/2) = exp(0 + 1/2) = exp(0.5) ≈ 1.6487
    # Expected variance = (exp(std^2) - 1) * exp(2*mean + std^2)
    #                   = (exp(1) - 1) * exp(1) = 1.7183 * 2.7183 ≈ 4.6708
    # Using std=1 (instead of std=2) keeps the tail thin enough that float16
    # never overflows: exp(Z*1+0) > 65504 requires Z > 11.09 (prob ≈ 0).
    mean_param = 0.0
    std_param = 1.0
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        x.log_normal_(mean_param, std_param)
    x_res = utils.to_reference(x)
    # All values should be positive (log-normal is always positive)
    assert (x_res > 0).all()
    # Test mean is approximately correct using statistical validation
    # For probability distributions, use looser tolerance due to sampling variance
    x_float = x_res.to(torch.float32)
    mean_res = torch.mean(x_float)
    expected_mean = torch.tensor(
        np.exp(mean_param + std_param**2 / 2), device=mean_res.device
    )
    # Tolerance: 15% of expected_mean to account for sampling variance
    mean_tol = 0.15 * expected_mean.item()
    utils.gems_assert_close(mean_res, expected_mean, torch.float32, atol=mean_tol)
