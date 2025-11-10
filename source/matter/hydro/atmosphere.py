"""
Centralized atmosphere and floor management for relativistic hydrodynamics.

This module provides a unified interface for atmospheric parameters and floor
application strategies, following the   / + approach.

References:
    -   : apply_tau_floor__enforce_limits_on_primitives_and_recompute_conservs.C
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
    the hydrodynamics evolution pipeline, following   / .

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
        Safety factor for conservative variable floors (0.999999 in   )
    mb : float
        Baryon mass for ideal gas EOS (in geometric units). Default: 1.0

    Usage
    -----
    Create once at the top level (e.g., in TOVEvolution script):

        atm_params = AtmosphereParams(rho_floor=1e-14, T_floor=1e-8)
        matter = PerfectFluid(eos=eos, atmosphere=atm_params)

    All subsystems (cons2prim, valencia, reconstruction, riemann) will use
    the same parameters automatically.
    """

    # Primary floors
    rho_floor: float = 1e-10
    p_floor: float = 1e-11
    # No explicit temperature or threshold here;   floors rely on
    # p_floor and rho_floor directly.

    # Velocity and Lorentz factor limits
    v_max: float = 0.999999
    W_max: float = 1.0e3

    # Conservative variable floor parameters
    tau_atm_factor: float = 1.0  # tau_atm = tau_atm_factor * p_floor
    conservative_floor_safety: float = 0.999999  # Safety factor for inequalities

    # Baryon mass for ideal gas EOS
    mb: float = 1.0

    def __post_init__(self):
        """Validate parameters."""
        if self.rho_floor <= 0:
            raise ValueError(f"rho_floor must be positive, got {self.rho_floor}")
        if self.p_floor <= 0:
            raise ValueError(f"p_floor must be positive, got {self.p_floor}")
        # No temperature or rho_threshold checks in  -style params
        if not 0 < self.v_max < 1:
            raise ValueError(f"v_max must be in (0, 1), got {self.v_max}")
        if self.W_max <= 1:
            raise ValueError(f"W_max must be > 1, got {self.W_max}")

    @property
    def tau_atm(self):
        """Atmosphere value for tau (energy density)."""
        return self.tau_atm_factor * self.p_floor



    def to_cons2prim_params(self):
        """Export parameters in format expected by Cons2PrimSolver."""
        return {
            "rho_floor": self.rho_floor,
            "p_floor": self.p_floor,
            "v_max": self.v_max,
            "W_max": self.W_max,
        }


class FloorApplicator:
    """
    Applies intelligent floors to conservative and primitive variables.

    Implements the    strategy:
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
        Apply floors to primitive variables ( /  -style).

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
        # Atmosphere: if rho < rho_floor, apply full atmosphere
        atmosphere_mask = rho0 < self.atm.rho_floor

        rho0_floor = np.copy(rho0)
        vr_floor = np.copy(vr)
        p_floor = np.copy(p)

        # Apply atmosphere where below floor
        if np.any(atmosphere_mask):
            rho0_floor[atmosphere_mask] = self.atm.rho_floor
            vr_floor[atmosphere_mask] = 0.0  # Zero velocity in atmosphere
            # Use pressure floor directly
            p_floor[atmosphere_mask] = self.atm.p_floor

        # Also apply minimum floors regardless
        rho0_floor = np.maximum(rho0_floor, self.atm.rho_floor)
        p_floor = np.maximum(p_floor, self.atm.p_floor)

        # Velocity limit (apply everywhere)
        v2 = gamma_rr * vr_floor**2
        violation_mask = v2 >= self.atm.v_max**2
        if np.any(violation_mask):
            vr_floor[violation_mask] = (
                np.sign(vr_floor[violation_mask]) * self.atm.v_max /
                np.sqrt(np.maximum(gamma_rr[violation_mask], 1e-30))
            )

        return rho0_floor, vr_floor, p_floor

    def apply_conservative_floors(self, D, Sr, tau, gamma_rr, metric_psi6=None):
        """
        Apply conservative variable consistency floors following   .

        This implements the inequalities from Etienne et al. (2012) Appendix A:
        - tau >= tau_atm
        - S^2 <= tau * (tau + 2*D)

        Parameters
        ----------
        D : array
            Conserved density
        Sr : array
            Conserved radial momentum
        tau : array
            Conserved energy
        gamma_rr : array
            Radial metric component
        metric_psi6 : array, optional
            Conformal factor psi^6 for near-BH handling

        Returns
        -------
        D_floor, Sr_floor, tau_floor : arrays
            Floored conservative variables
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
            ratio = RHS[S_violated] / np.maximum(S2[S_violated], 1e-30)
            rescale_factor = np.sqrt(np.maximum(ratio, 0.0))
            Sr_floor[S_violated] *= rescale_factor
            floor_applied[S_violated] = True

        return D_floor, Sr_floor, tau_floor, floor_applied


def apply_floors_to_state(state_2d, grid, hydro, rho_threshold=None):
    """
      / -style floor application to both primitives and conservatives.

    This function centralizes atmosphere handling following the
    strategy (Etienne+ 2012, Appendix A):
      1) Recover primitives (cons2prim)
      2) Apply floors/limits to primitives (ρ, v, P)
      3) Apply conservative consistency floors (τ floor, S^2 ≤ safety·τ(τ+2D))
      4) Recompute conservatives from floored primitives where needed

    IMPORTANT: This function only operates on PHYSICAL cells [NUM_GHOSTS:-NUM_GHOSTS].
    Ghost cells are handled by grid.fill_boundaries() with parity conditions and should
    NOT be modified by atmosphere/floor logic.

    Notes on definitions:
      - D, S_r, τ are DENSITIZED conservatives (D̃ = e^{6φ} D, etc.) as stored in state_2d
      - γ_rr is the physical metric component used for v^2 and S^2

    Parameters
    ----------
    state_2d : ndarray
        Full state array with BSSN + hydro; modified in-place
    grid : Grid
        Grid object (for r, ghost counts, metric builders)
    hydro : PerfectFluid
        Hydrodynamics object (provides EOS, cons2prim and valencia metrics)
    rho_threshold : float, optional
        Threshold for atmosphere mask; default 10·rho_floor

    Returns
    -------
    state_2d : ndarray
        Modified state with floors applied consistently
    """
    from source.bssn.bssnvars import BSSNVars
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS
    from source.core.spacing import NUM_GHOSTS
    from source.bssn.tensoralgebra import get_bar_gamma_LL

    atm = hydro.atmosphere
    if rho_threshold is None:
        rho_threshold = 10.0 * atm.rho_floor

    # Define physical cells slice (exclude ghost cells on both ends)
    phys = slice(NUM_GHOSTS, -NUM_GHOSTS)

    # Extract conservatives (ONLY physical cells)
    D = state_2d[NUM_BSSN_VARS + 0, phys]
    Sr = state_2d[NUM_BSSN_VARS + 1, phys]
    tau = state_2d[NUM_BSSN_VARS + 2, phys]

    # Build BSSN geometry to compute γ_rr (physical)
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
    bar_gamma_LL = get_bar_gamma_LL(grid.r, bssn_vars.h_LL, hydro.background)
    phi = np.asarray(bssn_vars.phi, dtype=float)
    e4phi = np.exp(4.0 * phi)
    gamma_rr_full = e4phi * bar_gamma_LL[:, 0, 0]
    gamma_rr = gamma_rr_full[phys]  # Extract only physical cells

    # Recover primitives (cons2prim has its own floors for invalid points)
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    prim = hydro._get_primitives(bssn_vars, grid.r)
    rho0 = prim['rho0'][phys]  # Only physical cells
    vr = prim['vr'][phys]
    p = prim['p'][phys]

    # Create floor applicator
    floor_app = FloorApplicator(atmosphere=atm, eos=hydro.eos)

    # STEP 1: Apply primitive floors (ρ, v, P)
    rho0_f, vr_f, p_f = floor_app.apply_primitive_floors(rho0, vr, p, gamma_rr)

    # Force v=0 in atmosphere regions (ρ below threshold)
    atm_mask = rho0_f < rho_threshold
    if np.any(atm_mask):
        vr_f[atm_mask] = 0.0

    # STEP 2: Apply conservative floors (τ floor, S^2 constraint)
    D_f, Sr_f, tau_f, cons_mask = floor_app.apply_conservative_floors(D, Sr, tau, gamma_rr)

    # STEP 3: Identify points where primitives changed
    prim_mask = (
        (np.abs(rho0_f - rho0) > 1e-14) |
        (np.abs(vr_f - vr) > 1e-14) |
        (np.abs(p_f - p) > 1e-14)
    )

    # STEP 4: Recompute conservatives from floored primitives where needed
    needs = prim_mask | cons_mask
    if np.any(needs):
        from source.matter.hydro.cons2prim import prim_to_cons
        # Compute densitization factor at physical cells
        e6phi = np.exp(6.0 * phi[phys])
        D_new, Sr_new, tau_new = prim_to_cons(rho0_f, vr_f, p_f, gamma_rr, hydro.eos, e6phi=e6phi)
        # Apply cons floors again to ensure final consistency
        D_new, Sr_new, tau_new, _ = floor_app.apply_conservative_floors(D_new, Sr_new, tau_new, gamma_rr)

        D_f[needs] = D_new[needs]
        Sr_f[needs] = Sr_new[needs]
        tau_f[needs] = tau_new[needs]

    # Update state_2d with floored values (ONLY in physical cells)
    state_2d[NUM_BSSN_VARS + 0, phys] = D_f
    state_2d[NUM_BSSN_VARS + 1, phys] = Sr_f
    state_2d[NUM_BSSN_VARS + 2, phys] = tau_f

    # NOTE: Ghost cells are NOT touched here. They will be filled by grid.fill_boundaries()
    # with proper parity conditions after this function returns.

    return state_2d


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
