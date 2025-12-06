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
Model: D4S
==========

This module contains the definition of the D4 dispersion model for the
evaluation of C6 coefficients.

Upon instantiation, the reference polarizabilities are calculated for the
all atoms of the molecule(s) and stored in the model class.


Example
-------
>>> import torch
>>> import tad_dftd4 as d4
>>>
>>> numbers = torch.tensor([14, 1, 1, 1, 1]) # SiH4
>>> model = d4.D4Model(numbers)
>>>
>>> # calculate Gaussian weights, optionally pass CN and partial charges
>>> gw = model.weight_references()
>>> c6 = model.get_atomic_c6(gw)
"""
from __future__ import annotations

import torch
from tad_mctc.math import einsum
from tad_mctc.typing import Literal, Tensor, overload

from .. import data
from ..damping.parameters.base import Param
from ..utils import is_exceptional, trapzd_noref
from .base import WF_DEFAULT, BaseModel

__all__ = ["D4Model"]


class D4Model(BaseModel):
    """
    The D4 dispersion model.
    """

    def _get_wf(self) -> Tensor:
        """Default weighting factor."""
        return torch.tensor(WF_DEFAULT, **self.dd)

    @overload
    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
        *,
        with_dgwdq: Literal[False] = ...,
        with_dgwdcn: Literal[False] = ...,
    ) -> Tensor: ...

    @overload
    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
        *,
        with_dgwdq: Literal[True],
        with_dgwdcn: Literal[False] = ...,
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
        *,
        with_dgwdq: Literal[False] = ...,
        with_dgwdcn: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
        *,
        with_dgwdq: Literal[True],
        with_dgwdcn: Literal[True],
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
        *,
        with_dgwdq: bool = False,
        with_dgwdcn: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """
        Calculate the weights of the reference system
        (shape: ``(..., nat, nref)``).

        Parameters
        ----------
        cn : Tensor | None, optional
            Coordination number of every atom. Defaults to `None` (0).
        q : Tensor | None, optional
            Partial charge of every atom. Defaults to `None` (0).
        with_dgwdq : bool, optional
            Whether to also calculate the derivative of the weights with
            respect to the partial charges. Defaults to `False`.
        with_dgwdcn : bool, optional
            Whether to also calculate the derivative of the weights with
            respect to the coordination numbers. Defaults to `False`.

        Returns
        -------
        Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]
            Weights for the atomic reference systems (shape:
            ``(..., nat, ref)``). If ``with_dgwdq`` is ``True``, also returns
            the derivative of the weights with respect to the partial charges.
            If ``with_dgwdcn`` is ``True``, also returns the derivative of the
            weights with respect to the coordination numbers.
        """
        if cn is None:
            cn = torch.zeros(self.numbers.shape, **self.dd)
        if q is None:
            q = torch.zeros(self.numbers.shape, **self.dd)

        if self.ref_charges == "eeq":
            # pylint: disable=import-outside-toplevel
            from ..reference.d4.charge_eeq import clsq as _refq
        elif self.ref_charges == "gfn2":
            # pylint: disable=import-outside-toplevel
            from ..reference.d4.charge_gfn2 import refq as _refq
        else:
            raise ValueError(f"Unknown reference charges: {self.ref_charges}")

        # pylint: disable=import-outside-toplevel
        from ..reference import d4 as d4ref

        refq = _refq.to(**self.dd)[self.numbers]

        zero = torch.tensor(0.0, **self.dd)
        zero_double = torch.tensor(0.0, device=self.device, dtype=torch.double)

        refc = d4ref.refc.to(self.device)[self.numbers]
        mask = refc > 0

        # Due to the exponentiation, `norm` and `expw` may become very small
        # (down to 1e-300). This causes problems for the division by `norm`,
        # since single precision, i.e. `torch.float`, only goes to around 1e-38.
        # Consequently, some values become zero although the actual result
        # should be close to one. The problem does not arise when using `torch.
        # double`. In order to avoid this error, which is also difficult to
        # detect, this part always uses `torch.double`. `params.refcovcn` is
        # saved with `torch.double`, but I still made sure...
        refcn = d4ref.refcovcn.to(device=self.device, dtype=torch.double)[
            self.numbers
        ]

        # For vectorization, we reformulate the Gaussian weighting function:
        # exp(-wf * igw * (cn - cn_ref)^2) = [exp(-(cn - cn_ref)^2)]^(wf *igw)
        # Gaussian weighting function part 1: exp(-(cn - cn_ref)^2)
        dcn = cn.unsqueeze(-1).type(torch.double) - refcn
        tmp = torch.exp(-dcn * dcn)

        # Gaussian weighting function part 2: tmp^(wf * igw)
        # (While the Fortran version just loops over the number of gaussian
        # weights `igw`, we have to use masks and explicitly implement the
        # formulas for exponentiation. Luckily, `igw` only takes on the values
        # 1 and 3.)
        def refc_pow(n: int) -> Tensor:
            return sum(
                (torch.pow(tmp, i * self.wf) for i in range(1, n + 1)),
                torch.tensor(0.0, device=tmp.device),
            )

        refc_pow_1 = torch.where(refc == 1, refc_pow(1), tmp)
        refc_pow_final = torch.where(refc == 3, refc_pow(3), refc_pow_1)

        expw = torch.where(mask, refc_pow_final, zero_double)

        # Normalize weights, but keep shape. This needs double precision.
        # Moreover, we need to mask the normalization to avoid division by zero
        # for autograd. Strangely, `storch.divide` gives erroneous results for
        # some elements (Mg, e.g. in MB16_43/03).
        norm = torch.where(
            mask,
            torch.sum(expw, dim=-1, keepdim=True),
            torch.tensor(1e-300, device=self.device, dtype=torch.double),
        )

        # back to real dtype
        gw_temp = (expw / norm).type(self.dtype)

        # maximum reference CN for each atom
        maxcn = torch.max(refcn, dim=-1, keepdim=True)[0]

        # prevent division by 0 and small values
        gw = torch.where(
            is_exceptional(gw_temp, self.dtype),
            torch.where(refcn == maxcn, torch.tensor(1.0, **self.dd), zero),
            gw_temp,
        )

        # unsqueeze for reference dimension
        zeff = data.ZEFF(self.device)[self.numbers].unsqueeze(-1)
        gam = data.GAM(**self.dd)[self.numbers].unsqueeze(-1) * self.gc
        q = q.unsqueeze(-1)

        # charge scaling
        zeta = torch.where(mask, self._zeta(gam, refq + zeff, q + zeff), zero)

        if with_dgwdq is False and with_dgwdcn is False:
            return zeta * gw

        # DERIVATIVES

        outputs = [zeta * gw]

        if with_dgwdcn is True:

            def _dpow(n: int) -> Tensor:
                return sum(
                    (
                        2 * i * self.wf * dcn * torch.pow(tmp, i * self.wf)
                        for i in range(1, n + 1)
                    ),
                    zero_double,
                )

            wf_1 = torch.where(refc == 1, _dpow(1), zero_double)
            wf_3 = torch.where(refc == 3, _dpow(3), zero_double)
            dexpw = wf_1 + wf_3

            # no mask needed here, already masked in `dexpw`
            dnorm = torch.sum(dexpw, dim=-1, keepdim=True)

            _dgw = (dexpw - expw * dnorm / norm) / norm
            dgw = torch.where(
                is_exceptional(_dgw, self.dtype), zero, _dgw.type(self.dtype)
            )

            outputs.append(zeta * dgw)

        if with_dgwdq is True:
            dzeta = torch.where(
                mask, self._dzeta(gam, refq + zeff, q + zeff), zero
            )

            outputs.append(dzeta * gw)

        return tuple(outputs)  # type: ignore

    def get_atomic_c6(
        self, 
        gw: Tensor | None = None, 
        param: Param | None = None, 
        alpha_mode: str = "reference"
    ) -> Tensor:
        """
        Calculate atomic C6 dispersion coefficients.

        This method supports two modes:
        
        **Reference mode (default, alpha_mode="reference")**:
        1. Getting weighted polarizabilities for each atom (with optional corrections)
        2. Performing Casimir-Polder integration over imaginary frequencies
        3. Adding optional per-pair C6 corrections
        
        **No-reference mode (alpha_mode="noref")**:
        1. Bypasses reference systems, CN, and charges
        2. Uses base polarizabilities (alpha_0) indexed by atomic number
        3. Adds frequency-dependent corrections (dynamic_alpha_delta_w)
        4. Performs Casimir-Polder integration

        Parameters
        ----------
        gw : Tensor | None, optional
            Weights for the atomic reference systems of shape
            `(..., nat, nref)`. Required when ``alpha_mode="reference"``.
        param : Param | None, optional
            Damping parameters. Can contain:
            - For reference mode: ``dynamic_alpha_delta``, ``c6_delta``
            - For noref mode: ``dynamic_alpha_delta_w``, ``alpha_0``, ``c6_delta``
        alpha_mode : str, optional
            Mode for C6 calculation. Either ``"reference"`` (default) or ``"noref"``.

        Returns
        -------
        Tensor
            C6 coefficients for all atom pairs of shape `(..., nat, nat)`.

        Raises
        ------
        ValueError
            If ``alpha_mode`` is invalid or required parameters are missing.
        """
        if alpha_mode == "noref":
            # No-reference mode: bypass reference systems
            if param is None:
                raise ValueError(
                    "param is required for alpha_mode='noref' to provide "
                    "dynamic_alpha_delta_w"
                )
            return self.get_dynamic_c6_noref(param)
        
        elif alpha_mode == "reference":
            # Standard reference-based mode
            if gw is None:
                raise ValueError(
                    "gw (Gaussian weights) is required for alpha_mode='reference'"
                )
            if param is None:
                raise ValueError("param is required for C6 calculation")
            
            # Get weighted polarizabilities (with optional alpha corrections)
            weighted_alpha = self.get_weighted_pols(gw, param)
            
            # Perform Casimir-Polder integration to get C6 from weighted alphas
            c6 = trapzd_noref(weighted_alpha)
            
            # Add optional per-pair C6 corrections
            c6_delta = param.get("c6_delta", None)
            if c6_delta is not None:
                if isinstance(c6_delta, (int, float)):
                    c6 = c6 + c6_delta
                else:
                    assert c6.shape == c6_delta.shape, (
                        f"c6 and c6_delta must have the same shape, "
                        f"but got {c6.shape} and {c6_delta.shape}"
                    )
                    c6 = c6 + c6_delta
            
            return c6
        
        else:
            raise ValueError(
                f"Invalid alpha_mode='{alpha_mode}'. "
                f"Must be either 'reference' or 'noref'."
            )

    def get_weighted_pols(self, gw: Tensor, param: Param | None = None) -> Tensor:
        """
        Calculate the weighted polarizabilities for each atom and frequency.

        Parameters
        ----------
        gw : Tensor
            Weights for the atomic reference systems of shape
            ``(..., nat, nref)``.
        param : Param | None, optional
            Damping parameters. If provided and contains ``dynamic_alpha_delta``,
            it will be added to the weighted polarizabilities. Defaults to ``None``.

        Returns
        -------
        Tensor
            Weighted polarizabilities of shape ``(..., nat, 23)``.
        """
        a = self._get_alpha()
        weighted_alpha = einsum("...nr,...nrw->...nw", gw, a)
        
        # Apply per-atom alpha correction if provided
        if param is not None:
            alpha_delta = param.get("dynamic_alpha_delta", None)
            if alpha_delta is not None:
                # alpha_delta should have shape (..., nat, 23) or (..., nat, 1) or (..., nat)
                if isinstance(alpha_delta, (int, float)):
                    # scalar case: apply to all atoms and frequencies
                    weighted_alpha = weighted_alpha + alpha_delta
                else:
                    # Ensure proper broadcasting
                    if alpha_delta.dim() < weighted_alpha.dim():
                        # Add frequency dimension if needed
                        while alpha_delta.dim() < weighted_alpha.dim():
                            alpha_delta = alpha_delta.unsqueeze(-1)
                    weighted_alpha = weighted_alpha + alpha_delta
        
        return weighted_alpha

    def get_dynamic_c6_noref(self, param: Param) -> Tensor:
        """
        Calculate C6 coefficients using a reference-system-free approach.
        
        This method bypasses:
        - Reference systems
        - Coordination number calculation
        - Charge calculation (EEQ)
        
        Instead, it uses base polarizabilities (alpha_0) indexed by atomic number,
        adds frequency-dependent corrections (dynamic_alpha_delta_w), and performs
        Casimir-Polder integration.

        Parameters
        ----------
        param : Param
            Parameters containing:
            - ``dynamic_alpha_delta_w``: Per-atom, per-frequency polarizability 
              corrections with shape ``(..., natom, 23)``. **Required**.
            - ``alpha_0``: Base polarizabilities per element with shape ``(max_Z, 23)``.
              Optional, defaults to zeros from reference data.

        Returns
        -------
        Tensor
            C6 coefficients for all atom pairs of shape ``(..., nat, nat)``.

        Raises
        ------
        ValueError
            If ``dynamic_alpha_delta_w`` is not provided or has incorrect shape.
        """
        # Get dynamic_alpha_delta_w (required)
        dynamic_alpha_delta_w = param.get("dynamic_alpha_delta_w", None)
        if dynamic_alpha_delta_w is None:
            raise ValueError(
                "dynamic_alpha_delta_w is required for alpha_mode='noref' "
                "but was not provided in param dict."
            )
        
        # Validate shape of dynamic_alpha_delta_w
        # Expected: (..., natom, 23) where ... are batch dimensions
        if dynamic_alpha_delta_w.shape[-1] != 23:
            raise ValueError(
                f"dynamic_alpha_delta_w must have 23 frequency points in the last "
                f"dimension, but got shape {dynamic_alpha_delta_w.shape}"
            )
        
        # Check that the number of atoms matches
        expected_nat = self.numbers.shape[-1]
        if dynamic_alpha_delta_w.shape[-2] != expected_nat:
            raise ValueError(
                f"dynamic_alpha_delta_w has {dynamic_alpha_delta_w.shape[-2]} atoms "
                f"but model has {expected_nat} atoms"
            )
        
        # Get alpha_0 (base polarizabilities per element)
        alpha_0_data = param.get("alpha_0", None)
        if alpha_0_data is None:
            # Use default zeros from reference data
            # pylint: disable=import-outside-toplevel
            from ..reference import d4 as d4ref
            alpha_0_data = d4ref.alpha_0.to(**self.dd)
        else:
            # User-provided alpha_0
            alpha_0_data = alpha_0_data.to(**self.dd)
        
        # Validate alpha_0 shape: should be (max_Z, 23)
        if alpha_0_data.dim() != 2 or alpha_0_data.shape[-1] != 23:
            raise ValueError(
                f"alpha_0 must have shape (max_Z, 23), but got {alpha_0_data.shape}"
            )
        
        # Index alpha_0 by atomic numbers to get base polarizabilities for each atom
        # alpha_base: shape (..., natom, 23)
        alpha_base = alpha_0_data[self.numbers]
        
        # Add the dynamic correction
        # alpha_total = alpha_base + dynamic_alpha_delta_w
        alpha_total = alpha_base + dynamic_alpha_delta_w
        
        # Perform Casimir-Polder integration to get C6
        c6 = trapzd_noref(alpha_total)
        
        return c6
