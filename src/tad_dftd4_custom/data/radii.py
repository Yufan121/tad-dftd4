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
Data: Radii
===========

Covalent radii and van der Waals radii.
"""
import os
import torch
import numpy as np
from tad_mctc.data import COV_D3 as _COV_D3_SOURCE

# Wrap COV_D3 for consistent callable API.
if callable(_COV_D3_SOURCE):
    COV_D3 = _COV_D3_SOURCE
else:
    def COV_D3(device=None, dtype=None):
        kw = {}
        if device is not None:
            kw["device"] = device
        if dtype is not None:
            kw["dtype"] = dtype
        return _COV_D3_SOURCE.to(**kw) if kw else _COV_D3_SOURCE


# VDW_PAIRWISE: try tad_mctc first, fall back to bundled .npy file.
try:
    from tad_mctc.data import VDW_PAIRWISE as _VDW_SOURCE
except ImportError:
    _VDW_SOURCE = None

_VDW_CACHE = None

def VDW_PAIRWISE(device=None, dtype=None):
    """Pairwise van der Waals radii (Bohr)."""
    global _VDW_CACHE

    if _VDW_SOURCE is not None:
        src = _VDW_SOURCE() if callable(_VDW_SOURCE) else _VDW_SOURCE
        kw = {}
        if device is not None:
            kw["device"] = device
        if dtype is not None:
            kw["dtype"] = dtype
        return src.to(**kw) if kw else src

    if _VDW_CACHE is None:
        path = os.path.join(os.path.dirname(__file__), "vdw_pairwise.npy")
        _VDW_CACHE = torch.from_numpy(np.load(path)).to(dtype=torch.float64)

    t = _VDW_CACHE
    kw = {}
    if device is not None:
        kw["device"] = device
    if dtype is not None:
        kw["dtype"] = dtype
    return t.to(**kw) if kw else t


__all__ = ["COV_D3", "VDW_PAIRWISE"]
