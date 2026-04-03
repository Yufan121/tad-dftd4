# This file is part of tad-dftd4.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
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
"""
Data: Chemical hardnesses
=========================

Element-specific chemical hardnesses for the charge scaling function used
to extrapolate the C6 coefficients in DFT-D4.
"""
import torch

try:
    from tad_mctc.data import GAM as _GAM_SOURCE
except ImportError:
    _GAM_SOURCE = None

# Embedded GAM data from tad_mctc 0.5.3 for compatibility with older versions.
_GAM_DATA = [
    0.0000, 0.4726, 0.9220, 0.1745, 0.2570, 0.3395, 0.4220, 0.5044, 0.5869,
    0.6693, 0.7519, 0.1796, 0.2216, 0.2635, 0.3054, 0.3473, 0.3892, 0.4312,
    0.4731, 0.1711, 0.2028, 0.2101, 0.2174, 0.2247, 0.2320, 0.2393, 0.2467,
    0.2540, 0.2613, 0.2686, 0.2759, 0.3076, 0.3393, 0.3724, 0.4027, 0.4345,
    0.4661, 0.1559, 0.1865, 0.1936, 0.2006, 0.2077, 0.2148, 0.2218, 0.2289,
    0.2360, 0.2431, 0.2501, 0.2572, 0.2878, 0.3185, 0.3491, 0.3798, 0.4104,
    0.4411, 0.0502, 0.0676, 0.0850, 0.1025, 0.1199, 0.1373, 0.1548, 0.1722,
    0.1896, 0.2070, 0.2245, 0.2419, 0.2593, 0.2768, 0.2942, 0.3116, 0.3290,
    0.3459, 0.3639, 0.3813, 0.3988, 0.4161, 0.4336, 0.4510, 0.4685, 0.4858,
    0.1253, 0.1427, 0.1601, 0.1776, 0.1950, 0.2124, 0.0726, 0.0942, 0.0992,
    0.1042, 0.1424, 0.1639, 0.1855, 0.2237, 0.2511, 0.2503, 0.2884, 0.3100,
    0.3316, 0.3532, 0.3682, 0.3963, 0.4014, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000,
]


def GAM(device=None, dtype=None):
    """Chemical hardness parameters (Hartree)."""
    if _GAM_SOURCE is not None:
        src = _GAM_SOURCE() if callable(_GAM_SOURCE) else _GAM_SOURCE
        kw = {}
        if device is not None:
            kw["device"] = device
        if dtype is not None:
            kw["dtype"] = dtype
        return src.to(**kw) if kw else src
    kw = {"dtype": dtype or torch.float64}
    if device is not None:
        kw["device"] = device
    return torch.tensor(_GAM_DATA, **kw)


__all__ = ["GAM"]
