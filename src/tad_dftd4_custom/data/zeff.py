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
Data: Charges
=============

Effective charges (imported from *tad-mctc*).
"""
from tad_mctc.data.zeff import ZEFF as _ZEFF

# Wrap tensor as callable for consistent API across tad_mctc versions.
# In 0.5.3+, ZEFF is a function; in 0.4.3, it's a tensor.
if callable(_ZEFF):
    ZEFF = _ZEFF
else:
    def ZEFF(device=None, dtype=None):
        t = _ZEFF
        if device is not None:
            t = t.to(device=device)
        if dtype is not None:
            t = t.to(dtype=dtype)
        return t

__all__ = ["ZEFF"]
