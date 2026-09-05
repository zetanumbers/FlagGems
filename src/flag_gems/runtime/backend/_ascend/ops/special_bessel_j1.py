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

import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@triton.jit
def _j1_impl(x):
    ax = tl.abs(x)

    # ---- small-argument path (|x| < 8): normalized NR rational polynomial ----
    # Original Numerical Recipes coefficients have terms up to ~7.2e10 which
    # cause catastrophic cancellation in fp32.  Dividing both numerator P and
    # denominator Q by Q[0] = 1.44725228442e11 makes every coefficient small
    # and fp32-safe while keeping the ratio P/Q identical.
    #   J1(x) = x * P(x^2) / Q(x^2)
    # Horner form (highest-degree coefficient first).
    y = x * x

    # P_norm  (degree-5 polynomial in y, constant term ~= 0.5)
    p = -2.083974189889571e-10
    p = p * y + 1.085124188025982e-07
    p = p * y + (-2.053969077126938e-05)
    p = p * y + 1.674876285976241e-03
    p = p * y + (-5.455205923661070e-02)
    p = p * y + 5.0e-01

    # Q_norm  (degree-5 polynomial in y, constant term = 1.0)
    q = 6.909645338032818e-12
    q = q * y + 2.604930348070488e-09
    q = q * y + 6.871464983028477e-07
    q = q * y + 1.284040449619842e-04
    q = q * y + 1.589588216764820e-02
    q = q * y + 1.0

    small_val = x * (p / q)

    # ---- large-argument path (|x| >= 8): NR asymptotic (coeffs are fp32-safe) ----
    z = 8.0 / ax
    yy = z * z
    xx = ax - 2.356194491  # 3*pi/4

    pp = 1.0 + yy * (
        0.183105e-2
        + yy
        * (
            -0.3516396496e-4
            + yy * (0.2457520174e-5 + yy * (-0.240337019e-6))
        )
    )
    qq = 0.04687499995 + yy * (
        -0.2002690873e-3
        + yy * (0.8449199096e-5 + yy * (-0.88228987e-6))
    )

    large_val = (
        tl.sqrt(0.636619772 / ax) * (tl.cos(xx) * pp - z * tl.sin(xx) * qq)
    )
    large_val = tl.where(x < 0.0, -large_val, large_val)

    return tl.where(ax < 8.0, small_val, large_val)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def j1_func(x):
    return _j1_impl(x.to(tl.float32))


def special_bessel_j1(A):
    logger.debug("GEMS_ASCEND SPECIAL_BESSEL_J1")
    return j1_func(A)
