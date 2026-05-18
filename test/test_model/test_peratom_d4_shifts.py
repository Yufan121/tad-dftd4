# This file is part of tad-dftd4.
#
# SPDX-Identifier: Apache-2.0
"""
Tests for per-atom shifts on D4 dispersion params.

NN-D4 v4 (2026-05-15) introduced the 6 per-atom shifts on
s6/s8/s10/a1/a2/s9 with *additive* semantics::

    s6_eff_AB = s6_base + 0.5 * (s6_delta_A + s6_delta_B)        # v4

NN-D4 v5 (2026-05-18) reframes the same 6 shifts as *fractional*
(multiplicative) corrections, so the upstream NN clamp
``|delta| <= 0.5`` automatically gives a ±50% physics-aware envelope
around the per-DFT baseline::

    s6_eff_AB = s6_base * (1 + 0.5 * (s6_delta_A + s6_delta_B))  # v5

This file verifies BOTH bit-identity at delta=0 (so v3-era ckpts trained
with delta=0 still reproduce vanilla) AND the v5 uniform-shift identity
``E(delta_i = c) == E(s_base -> s_base * (1+c))``. The v4 additive
identity ``E(delta_i = c) == E(s_base -> s_base + c)`` no longer holds
for s_base != 1.

For DFTs with ``s_base = 0`` (s10 in every DFT, s9 in some) the
multiplicative form makes the delta head harmless: ``0 * (1 + delta) = 0``.
This matches the physics convention that D4 terms disabled by the DFT
parameter table should stay disabled by NN.
"""
from __future__ import annotations

import math

import pytest
import torch

from tad_dftd4_custom import dftd4


# Small fixed water-trimer-ish system in atomic units (Bohr)
NUMBERS = torch.tensor([8, 1, 1, 8, 1, 1, 8, 1, 1], dtype=torch.long)
POSITIONS = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [0.95, 0.0, 0.0],
        [-0.30, 0.90, 0.0],
        [4.0, 0.0, 0.0],
        [4.95, 0.0, 0.0],
        [3.70, 0.90, 0.0],
        [0.0, 4.0, 0.0],
        [0.95, 4.0, 0.0],
        [-0.30, 4.90, 0.0],
    ],
    dtype=torch.float64,
)
CHARGE = torch.tensor(0.0, dtype=torch.float64)

# B3LYP-D4 BJ params (representative)
SCALAR_PARAM = {
    "s6": torch.tensor(1.0, dtype=torch.float64),
    "s8": torch.tensor(2.02929367, dtype=torch.float64),
    "s9": torch.tensor(1.0, dtype=torch.float64),
    "s10": torch.tensor(0.0, dtype=torch.float64),
    "a1": torch.tensor(0.40868035, dtype=torch.float64),
    "a2": torch.tensor(4.53807137, dtype=torch.float64),
    "alp": torch.tensor(16.0, dtype=torch.float64),
}


def _zero_deltas(nat: int) -> dict:
    z = torch.zeros(nat, dtype=torch.float64)
    return {
        "s6_delta": z.clone(),
        "s8_delta": z.clone(),
        "s10_delta": z.clone(),
        "a1_delta": z.clone(),
        "a2_delta": z.clone(),
        "s9_delta": z.clone(),
    }


# ---------------------------------------------------------------------------
# 1. Zero-delta strict equality with scalar path
# ---------------------------------------------------------------------------


def test_zero_delta_matches_scalar_path_strict():
    """Δ=0 across all 6 shifts must reproduce the scalar (legacy) D4 energy
    to within float64 round-off."""
    e_old = dftd4(NUMBERS, POSITIONS, CHARGE, SCALAR_PARAM).sum()
    p_new = dict(SCALAR_PARAM)
    p_new.update(_zero_deltas(NUMBERS.shape[0]))
    e_new = dftd4(NUMBERS, POSITIONS, CHARGE, p_new).sum()

    diff = (e_new - e_old).abs().item()
    assert diff < 1.0e-14, (
        f"Δ=0 path diverges from scalar path: |Δ| = {diff:.3e}"
    )


@pytest.mark.parametrize(
    "key",
    ["s6_delta", "s8_delta", "s10_delta", "a1_delta", "a2_delta", "s9_delta"],
)
def test_each_delta_present_alone_zero_value_matches_scalar(key):
    """Adding a single Δ=0 tensor for ONE param at a time must still match
    the scalar path. Catches accidental triggering of the per-pair branch
    when Δ is technically present but zero."""
    e_old = dftd4(NUMBERS, POSITIONS, CHARGE, SCALAR_PARAM).sum()
    p_new = dict(SCALAR_PARAM)
    p_new[key] = torch.zeros(NUMBERS.shape[0], dtype=torch.float64)
    e_new = dftd4(NUMBERS, POSITIONS, CHARGE, p_new).sum()
    diff = (e_new - e_old).abs().item()
    assert diff < 1.0e-14, f"{key}=0 alone diverges: |Δ| = {diff:.3e}"


# ---------------------------------------------------------------------------
# 1b. Uniform per-atom Δ ≡ multiplicatively-shifted scalar (v5 identity)
# ---------------------------------------------------------------------------
#
# v5 semantics: setting Δ_i = c for every atom must reproduce the energy of
# the equivalent scalar shift s -> s_base * (1 + c). ½ pair-average and ⅓
# triplet-average both collapse to (1 + c) when Δ is constant, and the
# baseline scaling factors out. This catches mis-normalised averaging AND
# regressions in the multiplicative reframe.


@pytest.mark.parametrize(
    "key,scalar_key,c",
    [
        ("s6_delta", "s6", 0.07),
        ("s8_delta", "s8", -0.05),
        ("s10_delta", "s10", 0.10),
        ("a1_delta", "a1", 0.03),
        ("a2_delta", "a2", -0.20),
        ("s9_delta", "s9", 0.08),
    ],
)
def test_uniform_delta_equals_scalar_shift(key, scalar_key, c):
    """A uniform per-atom shift Δ_i = c must equal the equivalent
    *multiplicative* scalar shift on the global parameter (v5 semantics).
    For ``s_base = 0`` (e.g. s10) the multiplicative reframe gives identical
    energies trivially because both paths produce zero."""
    nat = NUMBERS.shape[0]

    # Path A: uniform per-atom delta on the *_delta field
    p_a = dict(SCALAR_PARAM)
    p_a[key] = torch.full((nat,), c, dtype=torch.float64)
    e_a = dftd4(NUMBERS, POSITIONS, CHARGE, p_a).sum()

    # Path B: equivalent v5 multiplicative scalar shift on the global parameter
    p_b = dict(SCALAR_PARAM)
    p_b[scalar_key] = SCALAR_PARAM[scalar_key] * (1.0 + c)
    e_b = dftd4(NUMBERS, POSITIONS, CHARGE, p_b).sum()

    diff = (e_a - e_b).abs().item()
    scale = max(e_a.abs().item(), 1e-12)
    rel = diff / scale
    # float64 round-off floor ~ a few × 1e-15; per-pair vs scalar paths take
    # different reduction orders so absolute identity isn't possible.
    assert rel < 1.0e-12, (
        f"{key}={c} (uniform) does NOT match v5 multiplicative scalar shift "
        f"{scalar_key}*={1.0 + c}: |Δ|={diff:.3e}, rel={rel:.3e}\n"
        f"  E(uniform_delta) = {e_a.item():.10e}\n"
        f"  E(scalar_shift)  = {e_b.item():.10e}"
    )


# ---------------------------------------------------------------------------
# 2. Non-zero delta actually changes the energy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,scale",
    [
        ("s6_delta", 0.05),
        ("s8_delta", 0.05),
        ("s10_delta", 0.05),
        ("a1_delta", 0.05),
        ("a2_delta", 0.05),
        ("s9_delta", 0.05),
    ],
)
def test_nonzero_delta_moves_energy(key, scale):
    """A small but nonzero per-atom Δ must change the total energy (or be
    exactly zero when the multiplicative form makes the head dead because
    the DFT baseline disables that D4 term, i.e. ``s_base = 0``). Catches
    accidental silent drop of the new path."""
    nat = NUMBERS.shape[0]
    e_old = dftd4(NUMBERS, POSITIONS, CHARGE, SCALAR_PARAM).sum()

    p_new = dict(SCALAR_PARAM)
    # Make Δ position-dependent so the pair-average doesn't accidentally cancel
    p_new[key] = torch.linspace(-scale, scale, nat, dtype=torch.float64)
    e_new = dftd4(NUMBERS, POSITIONS, CHARGE, p_new).sum()
    diff = (e_new - e_old).abs().item()

    if key == "s10_delta":
        # s10_base = 0 for every supported DFT → multiplicative reframe makes
        # the s10_delta head physically dead (intended behavior).
        assert diff == 0.0, (
            f"{key} with s10_base=0 should leave energy unchanged under v5 "
            f"multiplicative semantics, got |Δ|={diff:.3e}"
        )
    else:
        assert diff > 1.0e-8, (
            f"{key} perturbation did not change energy: |Δ| = {diff:.3e}"
        )


# ---------------------------------------------------------------------------
# 2b. v5 envelope check: clamp |delta| <= 0.5 implies s_eff in [0.5*s, 1.5*s]
# ---------------------------------------------------------------------------
#
# Verify the physics-aware envelope claim: with the upstream NN clamp at
# |delta| <= 0.5 (custom_clamp on params_pred), the effective pair-/triplet-
# averaged scaling factor lies in [0.5, 1.5] for every pair/triplet, i.e. the
# realised s_eff value is bounded to [0.5*s_base, 1.5*s_base]. We probe this
# at the energy level: shifting delta to +0.5 (resp. -0.5) for every atom
# should give E(s_base * 1.5) and E(s_base * 0.5) respectively.


@pytest.mark.parametrize(
    "key,scalar_key",
    [
        ("s6_delta", "s6"),
        ("s8_delta", "s8"),
        ("a1_delta", "a1"),
        ("a2_delta", "a2"),
        ("s9_delta", "s9"),
    ],
)
@pytest.mark.parametrize("c", [-0.5, 0.5])
def test_v5_clamp_envelope(key, scalar_key, c):
    """Uniform Δ = ±0.5 (the NN clamp boundary) must equal E(s_base * (1±0.5))."""
    nat = NUMBERS.shape[0]

    p_a = dict(SCALAR_PARAM)
    p_a[key] = torch.full((nat,), c, dtype=torch.float64)
    e_a = dftd4(NUMBERS, POSITIONS, CHARGE, p_a).sum()

    p_b = dict(SCALAR_PARAM)
    p_b[scalar_key] = SCALAR_PARAM[scalar_key] * (1.0 + c)
    e_b = dftd4(NUMBERS, POSITIONS, CHARGE, p_b).sum()

    rel = (e_a - e_b).abs().item() / max(e_a.abs().item(), 1e-12)
    assert rel < 1.0e-12, (
        f"{key}={c} (clamp wall) does NOT match s_base*(1+{c}): rel={rel:.3e}\n"
        f"  E(uniform_delta) = {e_a.item():.10e}\n"
        f"  E(scaled_base)   = {e_b.item():.10e}"
    )


# ---------------------------------------------------------------------------
# 3. Autograd: 1st-order (FD match) and 2nd-order (Hessian symmetric)
# ---------------------------------------------------------------------------


def _build_param_with_grad(nat: int, keys):
    p = {k: v.clone() for k, v in SCALAR_PARAM.items()}
    for k in keys:
        p[k] = torch.zeros(nat, dtype=torch.float64, requires_grad=True)
    return p


@pytest.mark.parametrize(
    "key",
    ["s6_delta", "s8_delta", "a1_delta", "a2_delta", "s9_delta"],
)
def test_autograd_first_order_fd_matches(key):
    """Autograd dE/d(Δ_atom) must match a central finite-difference estimate
    on the same energy."""
    nat = NUMBERS.shape[0]

    # Analytical gradient
    p = _build_param_with_grad(nat, [key])
    e = dftd4(NUMBERS, POSITIONS, CHARGE, p).sum()
    g_auto = torch.autograd.grad(e, p[key])[0]

    # Finite-difference gradient on the first 3 atoms (cheap)
    h = 1.0e-5
    fd = torch.zeros(nat, dtype=torch.float64)
    for i in range(min(3, nat)):
        p_plus = {k: v.clone() for k, v in SCALAR_PARAM.items()}
        p_plus[key] = torch.zeros(nat, dtype=torch.float64)
        p_plus[key][i] = h
        e_plus = dftd4(NUMBERS, POSITIONS, CHARGE, p_plus).sum()

        p_minus = {k: v.clone() for k, v in SCALAR_PARAM.items()}
        p_minus[key] = torch.zeros(nat, dtype=torch.float64)
        p_minus[key][i] = -h
        e_minus = dftd4(NUMBERS, POSITIONS, CHARGE, p_minus).sum()
        fd[i] = (e_plus - e_minus).item() / (2 * h)

    # Tighter on params that produce O(1e-3) gradients; looser on s9_delta
    # which only enters via 3-body (gradients ~1e-9 here, dominated by FD
    # round-off noise).
    tol = 5.0e-5 if key == "s9_delta" else 1.0e-6
    rel_err = ((g_auto[:3] - fd[:3]).abs() / (fd[:3].abs() + 1e-12)).max().item()
    assert rel_err < tol, (
        f"autograd vs FD mismatch for {key}: rel_err = {rel_err:.3e}\n"
        f"auto = {g_auto[:3]}, fd = {fd[:3]}"
    )


def test_autograd_second_order_hessian_symmetric():
    """∂²E/∂Δa1_i∂Δa1_j must be symmetric (Hessian) — catches autograd
    misregistration of the new path. a1 is chosen because it enters
    nonlinearly via radius = a1*sqrt(R0) + a2 → 1/(d^n + radius^n); s_*
    enter linearly and so have trivial zero Hessians."""
    nat = NUMBERS.shape[0]
    p = _build_param_with_grad(nat, ["a1_delta"])
    e = dftd4(NUMBERS, POSITIONS, CHARGE, p).sum()
    g = torch.autograd.grad(e, p["a1_delta"], create_graph=True)[0]

    H = torch.zeros(nat, nat, dtype=torch.float64)
    for i in range(nat):
        gi = torch.autograd.grad(g[i], p["a1_delta"], retain_graph=True)[0]
        H[i] = gi

    asym = (H - H.T).abs().max().item()
    h_scale = H.abs().max().item()
    assert h_scale > 0, "Hessian is identically zero — a1 path may be broken"
    assert asym < 1.0e-12, f"Hessian not symmetric: max |H-H.T| = {asym:.3e}"


def test_autograd_third_order_with_positions_and_delta():
    """Third-order: ∂³E/∂R∂R∂Δs8 — confirms graph survives 3 differentiations.
    Mirrors the project_nnd4_autograd_orders.md verification convention.

    NOTE: we use a quadratic-in-force reduction (``(g_R**2).sum()``) rather
    than ``g_R.sum()``, because total-force sum vanishes by translational
    invariance and would make the second derivative identically zero.
    """
    nat = NUMBERS.shape[0]
    pos = POSITIONS.clone().detach().requires_grad_(True)
    p = dict(SCALAR_PARAM)
    delta = torch.zeros(nat, dtype=torch.float64, requires_grad=True)
    p["s8_delta"] = delta

    e = dftd4(NUMBERS, pos, CHARGE, p).sum()
    g_R = torch.autograd.grad(e, pos, create_graph=True)[0]
    # Quadratic-in-force scalar that is NOT zero by translational symmetry.
    loss = (g_R * g_R).sum()
    g_RR = torch.autograd.grad(
        loss, pos, create_graph=True, retain_graph=True
    )[0]
    g_RRD = torch.autograd.grad(g_RR.sum(), delta)[0]

    assert torch.isfinite(g_RRD).all(), "third derivative has NaN/Inf"
    assert g_RRD.abs().max().item() > 0, (
        "third derivative is identically zero — autograd graph broken"
    )


def test_autograd_positions_still_work_with_deltas():
    """Forces (dE/dR) must still be computable when delta tensors are in
    param. This catches any accidental graph-detach in the new path."""
    nat = NUMBERS.shape[0]
    pos = POSITIONS.clone().detach().requires_grad_(True)
    p = dict(SCALAR_PARAM)
    p.update(
        {
            "s6_delta": torch.zeros(nat, dtype=torch.float64),
            "s8_delta": torch.full((nat,), 0.03, dtype=torch.float64),
            "a1_delta": torch.zeros(nat, dtype=torch.float64),
            "a2_delta": torch.zeros(nat, dtype=torch.float64),
            "s9_delta": torch.full((nat,), -0.02, dtype=torch.float64),
        }
    )
    e = dftd4(NUMBERS, pos, CHARGE, p).sum()
    forces = torch.autograd.grad(e, pos)[0]
    assert torch.isfinite(forces).all(), "forces contain NaN/Inf"
    assert forces.abs().sum().item() > 0.0, "forces are all zero"
