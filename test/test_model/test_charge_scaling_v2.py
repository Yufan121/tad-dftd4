# This file is part of tad-dftd4.
#
# SPDX-Identifier: Apache-2.0
"""
Tests for the v2 charge_scaling_noref function (2026-05-13).

Verifies the three physical invariants of the ratio-form scaling
ζ(r;β,p) = e^β · r^p / (r^p + e^β − 1), with r = Z_A / (Z_A + q_A):

    1. ζ(q=0) = 1                             (neutral atom not rescaled)
    2. q > 0 (cation) → ζ < 1                 (less polarizable)
    3. q < 0 (anion)  → ζ > 1                 (more polarizable)

Plus structural checks:
    * ζ is strictly monotonically decreasing in q
    * Anion limit saturates at e^β (not NaN, not divergent)
    * NN-predicted shifts (within the trained ±0.5 range) preserve all three
      invariants for every element

The v1 sigmoid form had inverted slope (cation got ζ>1) and the NN was
pinned in a regime that couldn't escape; see project_charge_scaling_bug.md
for context.
"""
from __future__ import annotations

import math

import pytest
import torch

from tad_dftd4_custom.model.d4 import D4Model

# Elements covered in the trained dataset (PBE0 May-12 baselines):
#   H, C, N, O, F, P, S, Cl, Br
ELEMENTS = [1, 6, 7, 8, 9, 15, 16, 17, 35]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _scaling_for_atoms(Z_list, q_list, *, beta_shift=None, delta_shift=None,
                       dtype=torch.float64):
    """Run charge_scaling_noref on a single-frame system of arbitrary atoms.

    Each element of Z_list / q_list defines one atom; positions are unused
    by charge_scaling_noref but the model still requires `numbers`.
    """
    numbers = torch.tensor(Z_list)
    q = torch.tensor(q_list, dtype=dtype)
    model = D4Model(numbers, dtype=dtype)
    param = {}
    if beta_shift is not None:
        param["beta"] = torch.tensor(beta_shift, dtype=dtype)
    if delta_shift is not None:
        param["delta"] = torch.tensor(delta_shift, dtype=dtype)
    return model.charge_scaling_noref(q, param)


# ---------------------------------------------------------------------------
# core invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Z", ELEMENTS)
def test_zeta_neutral_is_exactly_one(Z: int) -> None:
    """ζ(q=0) = 1 exactly, for every element."""
    zeta = _scaling_for_atoms([Z], [0.0])
    assert torch.allclose(
        zeta, torch.ones_like(zeta), atol=1e-12
    ), f"Z={Z}: ζ(q=0)={zeta.item()} != 1 (off by {abs(zeta.item()-1):.2e})"


@pytest.mark.parametrize("Z", ELEMENTS)
@pytest.mark.parametrize("q", [0.1, 0.3, 0.5])
def test_zeta_cation_less_than_one(Z: int, q: float) -> None:
    """For q > 0 (cation): ζ < 1 (less polarizable, physical sign)."""
    zeta = _scaling_for_atoms([Z], [q])
    assert zeta.item() < 1.0, (
        f"Z={Z} q={q}: ζ={zeta.item():.6f} should be < 1 (cation)"
    )


@pytest.mark.parametrize("Z", ELEMENTS)
@pytest.mark.parametrize("q", [-0.1, -0.3, -0.5])
def test_zeta_anion_greater_than_one(Z: int, q: float) -> None:
    """For q < 0 (anion): ζ > 1 (more polarizable, physical sign)."""
    zeta = _scaling_for_atoms([Z], [q])
    assert zeta.item() > 1.0, (
        f"Z={Z} q={q}: ζ={zeta.item():.6f} should be > 1 (anion)"
    )


# ---------------------------------------------------------------------------
# shape / structural checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Z", [1, 6, 8, 17])
def test_zeta_strictly_decreasing_in_q(Z: int) -> None:
    """ζ(q) is strictly monotonically decreasing across [-0.5, +0.5]."""
    q_grid = torch.linspace(-0.5, 0.5, 21, dtype=torch.float64)
    zetas = []
    for q in q_grid.tolist():
        zetas.append(_scaling_for_atoms([Z], [q]).item())
    diffs = [zetas[i + 1] - zetas[i] for i in range(len(zetas) - 1)]
    assert all(d < 0 for d in diffs), (
        f"Z={Z}: ζ is not strictly decreasing; diffs={diffs}"
    )


def test_zeta_anion_saturates_at_exp_beta() -> None:
    """As q → −Z_eff (extreme anion), ζ approaches e^β ≈ exp(3) ≈ 20.09."""
    # Use H (Z_eff=1); push q toward -1 but stay safely above the clamp
    # threshold (-Z_eff + 1e-3) so we test the analytic limit rather than
    # the clamp branch.
    zeta_extreme = _scaling_for_atoms([1], [-0.999]).item()
    target = math.exp(3.0)
    # Should be within ~5% of e^β at this extreme
    assert abs(zeta_extreme - target) / target < 0.05, (
        f"H q=-0.999: ζ={zeta_extreme:.4f}, expected ≈ exp(3)≈{target:.4f}"
    )


def test_zeta_does_not_NaN_at_clamp_boundary() -> None:
    """q = −Z_eff exactly should hit the clamp branch, NOT produce NaN/Inf."""
    # H: clamp triggers when z_A = 1 + q ≤ 1e-3, i.e. q ≤ -0.999
    for q in [-1.0, -1.5, -2.0]:
        zeta = _scaling_for_atoms([1], [q])
        assert torch.isfinite(zeta).all(), (
            f"H q={q}: ζ={zeta} is not finite (NaN/Inf leaked)"
        )


# ---------------------------------------------------------------------------
# NN-shift robustness: trained shifts must preserve physical signs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Z", ELEMENTS)
@pytest.mark.parametrize("beta_shift,delta_shift", [
    (+0.5, +0.5),    # NN range upper bound (per half_range_pt=0.5)
    (-0.5, -0.5),    # NN range lower bound
    (+0.3, -0.4),    # mixed, plausible trained value
])
def test_signs_preserved_under_nn_shifts(
    Z: int, beta_shift: float, delta_shift: float
) -> None:
    """All three invariants (neutral=1, cation<1, anion>1) must hold for
    every (β_shift, δ_shift) in the trained NN-output range."""
    # neutral
    zeta_n = _scaling_for_atoms([Z], [0.0],
                                beta_shift=[beta_shift],
                                delta_shift=[delta_shift]).item()
    assert abs(zeta_n - 1.0) < 1e-10, (
        f"Z={Z} β_sh={beta_shift} δ_sh={delta_shift}: ζ(0)={zeta_n} != 1"
    )
    # cation
    zeta_c = _scaling_for_atoms([Z], [+0.3],
                                beta_shift=[beta_shift],
                                delta_shift=[delta_shift]).item()
    assert zeta_c < 1.0, (
        f"Z={Z} β_sh={beta_shift} δ_sh={delta_shift}: "
        f"ζ(q=+0.3)={zeta_c} should be < 1"
    )
    # anion
    zeta_a = _scaling_for_atoms([Z], [-0.3],
                                beta_shift=[beta_shift],
                                delta_shift=[delta_shift]).item()
    assert zeta_a > 1.0, (
        f"Z={Z} β_sh={beta_shift} δ_sh={delta_shift}: "
        f"ζ(q=-0.3)={zeta_a} should be > 1"
    )


# ---------------------------------------------------------------------------
# autograd: confirm gradients flow through every input
# ---------------------------------------------------------------------------

def test_grad_flows_through_q_and_nn_shifts() -> None:
    """Backward through ζ.sum() should produce non-zero gradients on q,
    beta_shift, and delta_shift (the three inputs the NN drives)."""
    dtype = torch.float64
    numbers = torch.tensor([8, 1, 1])
    q = torch.tensor([-0.5, 0.25, 0.25], dtype=dtype, requires_grad=True)
    beta_shift = torch.tensor([0.05, -0.03, 0.07], dtype=dtype,
                              requires_grad=True)
    delta_shift = torch.tensor([-0.04, 0.06, -0.02], dtype=dtype,
                               requires_grad=True)
    model = D4Model(numbers, dtype=dtype)
    param = {"beta": beta_shift, "delta": delta_shift}
    zeta = model.charge_scaling_noref(q, param)
    zeta.sum().backward()
    for t, name in [(q, "q"), (beta_shift, "beta_shift"),
                    (delta_shift, "delta_shift")]:
        assert t.grad is not None, f"{name}.grad is None"
        assert t.grad.norm().item() > 1e-12, (
            f"{name}.grad norm too small: {t.grad.norm().item():.2e}"
        )
