"""
Centralized atmosphere and floor management for relativistic hydrodynamics.

This module provides a unified interface for atmospheric parameters and floor
application strategies, following the IllinoisGRMHD/ approach.

References:
    - IllinoisGRMHD: apply_tau_floor__enforce_limits_on_primitives_and_recompute_conservs.C
    - Etienne et al. (2012): https://arxiv.org/pdf/1112.0568.pdf (Appendix A)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class AtmosphereParams:
    """
    Centralized atmospheric parameters for hydrodynamics.

    These parameters control floors, ceilings, and fallback values throughout
    the hydrodynamics evolution pipeline.

    Attributes
    ----------
    rho_floor : float
        Minimum rest mass density (atmosphere value). Default: 1e-13
    p_floor : float
        Minimum pressure floor. Default: 1e-15
    v_max : float
        Maximum allowed velocity (as fraction of c). Default: 0.999999
    W_max : float
        Maximum allowed Lorentz factor. Default: 1e3
    tau_atm_factor : float
        Safety factor for tau atmosphere: tau_atm = factor * p_floor. Default: 1.0
    conservative_floor_safety : float
        Safety factor for conservative variable floors (0.999999 in IllinoisGRMHD)

    Usage
    -----
    Create once at the top level (e.g., in TOVEvolution script):

        atm_params = AtmosphereParams(rho_floor=1e-14, p_floor=1e-16)
        matter = PerfectFluid(eos=eos, atmosphere=atm_params)

    All subsystems (cons2prim, valencia, reconstruction, riemann) will use
    the same parameters automatically.
    """

    # Primary floors
    rho_floor: float = 1e-10
    p_floor: float = 100*(rho_floor)**2

    # Velocity and Lorentz factor limits
    v_max: float = 0.999999
    W_max: float = 1.0e2

    # Conservative variable floor parameters
    tau_atm_factor: float = 1.0  # tau_atm = tau_atm_factor * p_floor
    conservative_floor_safety: float = 0.99  # Safety factor for inequalities

    def __post_init__(self):
        """Validate parameters."""
        if self.rho_floor <= 0:
            raise ValueError(f"rho_floor must be positive, got {self.rho_floor}")
        if self.p_floor <= 0:
            raise ValueError(f"p_floor must be positive, got {self.p_floor}")
        if not 0 < self.v_max < 1:
            raise ValueError(f"v_max must be in (0, 1), got {self.v_max}")
        if self.W_max <= 1:
            raise ValueError(f"W_max must be > 1, got {self.W_max}")

    @property
    def tau_atm(self):
        """Atmosphere value for tau (energy density)."""
        return self.tau_atm_factor * self.p_floor

    def to_cons2prim_params(self):
        """
        Convert atmosphere parameters to cons2prim solver format.

        Returns:
            dict: Parameters for Cons2PrimSolver
        """
        return {
            "rho_floor": self.rho_floor,
            "p_floor": self.p_floor,
            "v_max": self.v_max,
            "W_max": self.W_max,
        }


class FloorApplicator:
    """
    Applies intelligent floors to conservative and primitive variables.

    Implements the IllinoisGRMHD strategy:
    1. Apply primitive variable floors (rho, P, v)
    2. Apply conservative variable consistency floors (tau, S_i)
    3. Recompute conservatives from floored primitives if needed

    This ensures physical validity and prevents cons2prim failures.
    """

    def __init__(self, atmosphere: AtmosphereParams, eos):
        """
        Parameters
        ----------
        atmosphere : AtmosphereParams
            Atmospheric parameters
        eos : EOS
            Equation of state for pressure calculations
        """
        self.atm = atmosphere
        self.eos = eos

    def apply_primitive_floors(self, rho0, vr, p, gamma_rr):
        """
        Apply floors to primitive variables.

        Parameters
        ----------
        rho0 : array
            Rest mass density
        vr : array
            Radial velocity
        p : array
            Pressure
        gamma_rr : array
            Radial metric component

        Returns
        -------
        rho0_floor, vr_floor, p_floor : arrays
            Floored primitive variables
        """
        # Density floor
        rho0_floor = np.maximum(rho0, self.atm.rho_floor)

        # Pressure floor (can depend on EOS if needed)
        p_floor = np.maximum(p, self.atm.p_floor)

        # Velocity limit
        vr_floor = np.copy(vr)
        v2 = gamma_rr * vr**2
        violation_mask = v2 >= self.atm.v_max**2
        if np.any(violation_mask):
            vr_floor[violation_mask] = (
                np.sign(vr[violation_mask]) * self.atm.v_max /
                np.sqrt(np.maximum(gamma_rr[violation_mask], 1e-30))
            )

        return rho0_floor, vr_floor, p_floor

    def apply_conservative_floors(self, D, Sr, tau, gamma_rr, metric_psi6=None):
        """
        Apply conservative variable consistency floors following IllinoisGRMHD.

        This implements the inequalities from Etienne et al. (2012) Appendix A:
        - tau >= tau_atm
        - S^2 <= tau * (tau + 2*D)

        IMPORTANT: This method expects PHYSICAL (non-densitized) conservative variables.
        The floor thresholds (tau_atm, etc.) are calibrated for physical values.
        If your state vector stores densitized variables D̃ = e^{6φ}D, you must
        de-densitify them before calling this method.

        Parameters
        ----------
        D : array
            Conserved density (PHYSICAL, non-densitized: D = ρ₀W)
        Sr : array
            Conserved radial momentum (PHYSICAL, non-densitized: Sʳ = ρ₀hW²vʳγᵣᵣ)
        tau : array
            Conserved energy (PHYSICAL, non-densitized: τ = ρ₀hW² - p - D)
        gamma_rr : array
            Radial metric component
        metric_psi6 : array, optional
            Conformal factor psi^6 for near-BH handling

        Returns
        -------
        D_floor, Sr_floor, tau_floor : arrays
            Floored conservative variables (physical, non-densitized)
        floor_applied : bool array
            Mask indicating where floors were applied
        """
        N = len(D)
        D_floor = np.copy(D)
        Sr_floor = np.copy(Sr)
        tau_floor = np.copy(tau)
        floor_applied = np.zeros(N, dtype=bool)

        # 1. Tau floor (simplest case: no magnetic fields)
        tau_min = self.atm.tau_atm
        tau_violated = tau < tau_min
        if np.any(tau_violated):
            tau_floor[tau_violated] = tau_min
            floor_applied[tau_violated] = True

        # 2. S^2 constraint: S^2 <= safetyfactor * tau * (tau + 2*D)
        # Compute S^2 using metric
        S2 = Sr**2 / np.maximum(gamma_rr, 1e-30)

        # RHS of inequality
        RHS = self.atm.conservative_floor_safety * tau_floor * (tau_floor + 2.0 * D_floor)

        # Check violation
        S_violated = S2 > RHS
        if np.any(S_violated):
            # Rescale Sr to satisfy constraint
            rescale_factor = np.sqrt(RHS[S_violated] / np.maximum(S2[S_violated], 1e-30))
            Sr_floor[S_violated] *= rescale_factor
            floor_applied[S_violated] = True

        return D_floor, Sr_floor, tau_floor, floor_applied

    def apply_atmosphere_fallback(self, rho0, vr, p, eps, W, h, mask):
        """
        Set atmosphere values for failed points.

        Parameters
        ----------
        rho0, vr, p, eps, W, h : arrays
            Primitive variables to modify in-place
        mask : bool array
            Points to set to atmosphere
        """
        if not np.any(mask):
            return

        rho0[mask] = self.atm.rho_floor
        vr[mask] = 0.0
        p[mask] = self.atm.p_floor

        # EOS-consistent epsilon
        try:
            eps[mask] = self.eos.eps_from_rho_p(self.atm.rho_floor, self.atm.p_floor)
        except:
            eps[mask] = 1e-10

        W[mask] = 1.0

        # Enthalpy (EOS-dependent)
        if hasattr(self.eos, 'K') and hasattr(self.eos, 'gamma'):
            # Polytropic: h = 1 + ε
            h[mask] = 1.0 + eps[mask]
        else:
            # Ideal gas: h = 1 + ε + P/ρ
            h[mask] = 1.0 + eps[mask] + self.atm.p_floor / self.atm.rho_floor


def create_default_atmosphere(rho_floor: Optional[float] = None) -> AtmosphereParams:
    """
    Convenience function to create atmosphere with custom density floor.

    Parameters
    ----------
    rho_floor : float, optional
        Custom atmosphere density. If None, uses default (1e-13)

    Returns
    -------
    AtmosphereParams
        Atmosphere configuration

    Examples
    --------
    >>> atm = create_default_atmosphere(rho_floor=1e-14)
    >>> matter = PerfectFluid(eos=eos, atmosphere=atm)
    """
    if rho_floor is not None:
        return AtmosphereParams(rho_floor=rho_floor)
    else:
        return AtmosphereParams()
