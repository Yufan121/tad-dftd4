"""
Compatibility layer for tad_mctc version differences.

Provides items that exist in tad_mctc>=0.5 but not in <=0.4.3,
so the custom fork works with whichever version is installed.
"""
import torch
from collections.abc import Callable
from torch import Tensor

# --- typing aliases ---
# In tad_mctc>=0.5, these are exported from tad_mctc.typing.
# In tad_mctc<=0.4.3, they don't exist. Define locally.
try:
    from tad_mctc.typing import CNFunction, CountingFunction
except ImportError:
    CNFunction = Callable[..., Tensor]
    CountingFunction = Callable[[Tensor, Tensor], Tensor]

# --- data constants ---
try:
    from tad_mctc.data import GAM
except ImportError:
    GAM = None  # Not used in core D4 calculation path

try:
    from tad_mctc.data import VDW_PAIRWISE
except ImportError:
    VDW_PAIRWISE = None  # Not used in core D4 calculation path
