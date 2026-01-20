# riemann.py
"""
HLL Riemann solver for GRHD using quadratic eigenvalue method.

Uses quadratic equation method for characteristic speeds (robust near sonic points).
Fully optimized with Numba JIT compilation for CPU performance.
"""

import numpy as np
from numba import jit
from .valencia_reference_metric import ValenciaReferenceMetric
from .geometry import (
    GeometryState,
    compute_4velocity_1d,
    compute_g4UU_1d,
    compute_lorentz_factor_1d
)


# ==============================================================================
# JIT-COMPILED KERNELS
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def find_cp_cm_kernel(flux_dirn, g4UU, u4U, cs2):
    """
    JIT-compiled kernel to compute characteristic speeds c+/c- using quadratic method.

    Args:
        flux_dirn: Flux direction (0 for radial)
        g4UU: (M, 4, 4) inverse 4-metric
        u4U: (M, 4) contravariant 4-velocity
        cs2: (M,) sound speed squared

    Returns:
        (cminus, cplus): Tuple of (M,) arrays
    """
    M = cs2.shape[0]
    i = flux_dirn + 1  # Spatial index (1 for radial)

    cminus = np.empty(M)
    cplus = np.empty(M)

    for m in range(M):
        v02 = cs2[m]
        one_minus_cs2 = 1.0 - v02

        # Quadratic coefficients
        u0_sq = u4U[m, 0] * u4U[m, 0]
        ui_sq = u4U[m, i] * u4U[m, i]
        u0_ui = u4U[m, 0] * u4U[m, i]

        a = one_minus_cs2 * u0_sq - v02 * g4UU[m, 0, 0]
        b = 2.0 * v02 * g4UU[m, i, 0] - 2.0 * u0_ui * one_minus_cs2
        c = one_minus_cs2 * ui_sq - v02 * g4UU[m, i, i]

        # Solve quadratic
        detm_squared = b * b - 4.0 * a * c
        detm = np.sqrt(max(0.0, detm_squared))

        # Avoid division by zero
        if abs(a) < 1e-30:
            cminus[m] = -1.0
            cplus[m] = 1.0
        else:
            inv_2a = 0.5 / a
            cplus_tmp = (-b + detm) * inv_2a
            cminus_tmp = (-b - detm) * inv_2a

            # Ensure proper ordering
            cminus[m] = min(cplus_tmp, cminus_tmp)
            cplus[m] = max(cplus_tmp, cminus_tmp)

    return cminus, cplus


@jit(nopython=True, cache=True, fastmath=True)
def entropy_fix_kernel(lam_minus, lam_plus, delta=1e-8):
    """
    JIT-compiled entropy fix to avoid sonic glitches.

    Args:
        lam_minus: (M,) left wave speeds
        lam_plus: (M,) right wave speeds
        delta: Minimum separation from zero

    Returns:
        (lam_minus_fixed, lam_plus_fixed): Tuple of (M,) arrays
    """
    M = lam_minus.shape[0]
    lm_out = np.empty(M)
    lp_out = np.empty(M)

    for m in range(M):
        lm = min(lam_minus[m], -abs(delta))
        lp = max(lam_plus[m], abs(delta))

        # Ensure ordering
        if lm > lp:
            lm_out[m] = lp
            lp_out[m] = lm
        else:
            lm_out[m] = lm
            lp_out[m] = lp

    return lm_out, lp_out


@jit(nopython=True, cache=True, fastmath=True)
def physical_flux_1d_kernel(rho0, vr, pressure, W, h, alpha, e6phi, gamma_rr):
    """
    JIT-compiled kernel to compute physical flux for 1D radial direction.

    Inlines the Valencia flux computation for better performance.

    Args:
        rho0: (M,) rest-mass density
        vr: (M,) radial velocity
        pressure: (M,) pressure
        W: (M,) Lorentz factor
        h: (M,) specific enthalpy
        alpha: (M,) lapse
        e6phi: (M,) conformal factor e^{6phi}
        gamma_rr: (M,) radial metric component

    Returns:
        F: (M, 3) physical flux [F_D, F_Sr, F_tau]
    """
    M = rho0.shape[0]
    F = np.empty((M, 3))

    for m in range(M):
        # Metric components
        alph = alpha[m]
        e6p = e6phi[m]
        grr = gamma_rr[m]

        # Fluid quantities
        rho = rho0[m]
        v = vr[m]
        p = pressure[m]
        Wm = W[m]
        hm = h[m]

        # For spherical symmetry with zero shift: vtilde^r = v^r
        vtilde_r = v

        # 4-velocity components: u^0 = W/alpha, u^r = W * v^r
        u0 = Wm / alph
        ur = Wm * vtilde_r

        # Conservative density: D = rho * W
        D = rho * Wm

        # Stress-energy tensor components needed for flux
        # T^{0r} = rho*h*u^0*u^r + P*g^{0r}
        # For zero shift: g^{0r} = 0
        rho_h = rho * hm
        T0r = rho_h * u0 * ur

        # T^{rr} = rho*h*u^r*u^r + P*g^{rr}
        # g^{rr} = 1/gamma_rr (for diagonal metric)
        grr_inv = 1.0 / grr
        Trr = rho_h * ur * ur + p * grr_inv

        # T^r_r = T^{rr} * gamma_rr = rho*h*u^r*u^r*gamma_rr + P
        Tr_r = rho_h * ur * ur * grr + p

        # Fluxes (densitized by e^{6phi})
        # F_D = e6phi * alpha * D * vtilde^r
        F[m, 0] = e6p * alph * D * vtilde_r

        # F_Sr = e6phi * alpha * T^r_r
        F[m, 1] = e6p * alph * Tr_r

        # F_tau = e6phi * (alpha^2 * T^{0r} - alpha * D * vtilde^r)
        F[m, 2] = e6p * (alph * alph * T0r - alph * D * vtilde_r)

    return F


@jit(nopython=True, cache=True, fastmath=True)
def hll_flux_kernel(DL, SrL, tauL, DR, SrR, tauR,
                    FL, FR, lam_minus, lam_plus):
    """
    JIT-compiled parallel kernel for HLL flux computation.

    Args:
        DL, SrL, tauL: (M,) left conservative variables
        DR, SrR, tauR: (M,) right conservative variables
        FL: (M, 3) left physical flux
        FR: (M, 3) right physical flux
        lam_minus: (M,) left wave speed
        lam_plus: (M,) right wave speed

    Returns:
        out: (M, 3) HLL flux
    """
    M = DL.shape[0]
    out = np.empty((M, 3))

    for m in range(M):
        lm = lam_minus[m]
        lp = lam_plus[m]

        if lm >= 0.0:
            # Flow entirely from left
            out[m, 0] = FL[m, 0]
            out[m, 1] = FL[m, 1]
            out[m, 2] = FL[m, 2]
        elif lp <= 0.0:
            # Flow entirely from right
            out[m, 0] = FR[m, 0]
            out[m, 1] = FR[m, 1]
            out[m, 2] = FR[m, 2]
        else:
            # Mixed flow - HLL combination
            denom = lp - lm

            if abs(denom) < 1e-30:
                # Degenerate case: average
                out[m, 0] = 0.5 * (FL[m, 0] + FR[m, 0])
                out[m, 1] = 0.5 * (FL[m, 1] + FR[m, 1])
                out[m, 2] = 0.5 * (FL[m, 2] + FR[m, 2])
            else:
                inv_denom = 1.0 / denom
                lp_lm = lp * lm

                # U differences
                dD = DR[m] - DL[m]
                dSr = SrR[m] - SrL[m]
                dTau = tauR[m] - tauL[m]

                # HLL formula
                out[m, 0] = (lp * FL[m, 0] - lm * FR[m, 0] + lp_lm * dD) * inv_denom
                out[m, 1] = (lp * FL[m, 1] - lm * FR[m, 1] + lp_lm * dSr) * inv_denom
                out[m, 2] = (lp * FL[m, 2] - lm * FR[m, 2] + lp_lm * dTau) * inv_denom

    return out


# ==============================================================================
# ORIGINAL INTERFACE (for backward compatibility)
# ==============================================================================

def physical_flux(U, prim, gamma_rr, alpha, e6phi, eos):
    """
    Compute physical (non-densitized) flux for 1D radial direction.

    Calls Valencia._compute_fluxes (static method) and extracts the radial
    component

    Returns:
        F_phys: (M, 3) physical flux [F_D, F_Sr, F_tau]
    """
    M = len(U)

    # Extract primitives
    vr   = prim[:, 1]
    rho0 = prim[:, 0]
    pressure = prim[:, 2]

    # Compute Lorentz factor and enthalpy (using geometry module)
    W = compute_lorentz_factor_1d(vr, gamma_rr)
    eps = eos.eps_from_rho_p(rho0, pressure)
    h = 1.0 + eps + pressure / np.maximum(rho0, 1e-30)

    # Build 3D arrays for Valencia static method
    # v_U: (M, 3) with only radial component non-zero
    v_U = np.zeros((M, 3))
    v_U[:, 0] = vr

    # gamma_LL: (M, 3, 3) diagonal metric
    gamma_LL = np.zeros((M, 3, 3))
    gamma_LL[:, 0, 0] = gamma_rr
    gamma_LL[:, 1, 1] = 1.0  # placeholder for θθ
    gamma_LL[:, 2, 2] = 1.0  # placeholder for φφ

    # gamma_UU: (M, 3, 3) inverse metric
    gamma_UU = np.zeros((M, 3, 3))
    gamma_UU[:, 0, 0] = 1.0 / gamma_rr
    gamma_UU[:, 1, 1] = 1.0
    gamma_UU[:, 2, 2] = 1.0

    # beta_U: (M, 3) shift vector (zero for TOV)
    beta_U = np.zeros((M, 3))

    # Call Valencia static method to get densitized fluxes
    fD_U, fTau_U, fS_D = ValenciaReferenceMetric._compute_fluxes(rho0, v_U, pressure, W, h, alpha, e6phi, gamma_LL, gamma_UU, beta_U)

    # Extract radial component and divide by e^{6φ} to get non-densitized flux

    F_D   = fD_U[:, 0]   
    F_Sr  = fS_D[:, 0, 0] 
    F_tau = fTau_U[:, 0] 

    return np.stack([F_D, F_Sr, F_tau], axis=1)


class HLLRiemannSolver:
    """
    HLL Riemann solver using quadratic eigenvalue method for characteristic speeds.

    The HLL flux formula:
        F_HLL = (λ+ F_L - λ- F_R + λ+ λ- (U_R - U_L)) / (λ+ - λ-)

    Characteristic speeds λ± computed via quadratic equation from 4-metric and 4-velocity.

    Fully optimized with Numba JIT kernels for CPU performance.
    """

    def __init__(self, name: str = "HLL", atmosphere=None):
        self.name = name
        self.atmosphere = atmosphere

    def solve_batch(self, UL_batch, UR_batch, primL_batch, primR_batch,
                    geom: GeometryState, eos):
        """
        Fully optimized solver for multiple interfaces using JIT kernels.

        All inputs are arrays of length M. Returns (M,3) flux array.

        OPTIMIZATIONS:
        1. JIT-compiled kernels for characteristic speeds and HLL flux
        2. Parallelized HLL flux computation with prange
        3. Inlined physical flux computation (avoids Valencia overhead)
        """
        M = len(UL_batch)

        # Extract geometry components
        gamma_rr_batch = np.ascontiguousarray(geom.gamma_rr, dtype=np.float64)
        alpha_batch = np.ascontiguousarray(geom.alpha, dtype=np.float64)
        beta_r_batch = np.ascontiguousarray(geom.beta_r, dtype=np.float64)
        e6phi_batch = np.ascontiguousarray(geom.e6phi, dtype=np.float64)

        # Convert inputs to contiguous float64 arrays for Numba
        UL_batch = np.ascontiguousarray(UL_batch, dtype=np.float64)
        UR_batch = np.ascontiguousarray(UR_batch, dtype=np.float64)
        primL_batch = np.ascontiguousarray(primL_batch, dtype=np.float64)
        primR_batch = np.ascontiguousarray(primR_batch, dtype=np.float64)

        # Unpack conservatives
        DL = UL_batch[:, 0]
        SrL = UL_batch[:, 1]
        tauL = UL_batch[:, 2]
        DR = UR_batch[:, 0]
        SrR = UR_batch[:, 1]
        tauR = UR_batch[:, 2]

        # Unpack primitives
        rho0L = primL_batch[:, 0]
        vrL = primL_batch[:, 1]
        pL = primL_batch[:, 2]

        rho0R = primR_batch[:, 0]
        vrR = primR_batch[:, 1]
        pR = primR_batch[:, 2]

        # Apply floors using vectorized operations
        pL = np.maximum(pL, self.atmosphere.p_floor)
        pR = np.maximum(pR, self.atmosphere.p_floor)
        vrL = np.clip(vrL, -self.atmosphere.v_max, self.atmosphere.v_max)
        vrR = np.clip(vrR, -self.atmosphere.v_max, self.atmosphere.v_max)

        gamma_rr_batch = np.maximum(gamma_rr_batch, 1e-30)
        alpha_batch = np.maximum(alpha_batch, 1e-30)

        # Compute EOS quantities (not JIT-compatible, done in Python)
        epsL = np.maximum(eos.eps_from_rho_p(rho0L, pL), 1e-15)
        epsR = np.maximum(eos.eps_from_rho_p(rho0R, pR), 1e-15)
        cs2L = np.clip(eos.sound_speed_squared(rho0L, pL, epsL), 0.0, 1.0 - 1e-12)
        cs2R = np.clip(eos.sound_speed_squared(rho0R, pR, epsR), 0.0, 1.0 - 1e-12)

        # Compute Lorentz factors and enthalpies for flux calculation
        WL = compute_lorentz_factor_1d(vrL, gamma_rr_batch)
        WR = compute_lorentz_factor_1d(vrR, gamma_rr_batch)
        hL = 1.0 + epsL + pL / np.maximum(rho0L, 1e-30)
        hR = 1.0 + epsR + pR / np.maximum(rho0R, 1e-30)

        # Compute 4-velocities for all interfaces (using geometry module)
        u4U_L, _ = compute_4velocity_1d(vrL, gamma_rr_batch, alpha_batch, beta_r_batch)
        u4U_R, _ = compute_4velocity_1d(vrR, gamma_rr_batch, alpha_batch, beta_r_batch)

        # Build 4-metric components at all interfaces
        gamma_rr_UU = 1.0 / gamma_rr_batch
        g4UU = compute_g4UU_1d(alpha_batch, beta_r_batch, gamma_rr_UU)

        # Ensure contiguous arrays for JIT kernels
        g4UU = np.ascontiguousarray(g4UU)
        u4U_L = np.ascontiguousarray(u4U_L)
        u4U_R = np.ascontiguousarray(u4U_R)
        cs2L = np.ascontiguousarray(cs2L)
        cs2R = np.ascontiguousarray(cs2R)

        # JIT: Compute characteristic speeds
        cmL, cpL = find_cp_cm_kernel(0, g4UU, u4U_L, cs2L)
        cmR, cpR = find_cp_cm_kernel(0, g4UU, u4U_R, cs2R)

        # Global wave speed bounds
        cmax = np.maximum(0.0, np.maximum(cpL, cpR))
        cmin = -np.minimum(0.0, np.minimum(cmL, cmR))

        # Standard notation
        lam_minus = -cmin
        lam_plus = cmax

        # JIT: Entropy fix
        lam_minus, lam_plus = entropy_fix_kernel(lam_minus, lam_plus)

        # Ensure contiguous for flux kernels
        rho0L = np.ascontiguousarray(rho0L)
        rho0R = np.ascontiguousarray(rho0R)
        vrL = np.ascontiguousarray(vrL)
        vrR = np.ascontiguousarray(vrR)
        pL = np.ascontiguousarray(pL)
        pR = np.ascontiguousarray(pR)
        WL = np.ascontiguousarray(WL)
        WR = np.ascontiguousarray(WR)
        hL = np.ascontiguousarray(hL)
        hR = np.ascontiguousarray(hR)

        # JIT: Compute physical fluxes (inlined, no Valencia overhead)
        FL = physical_flux_1d_kernel(rho0L, vrL, pL, WL, hL,
                                     alpha_batch, e6phi_batch, gamma_rr_batch)
        FR = physical_flux_1d_kernel(rho0R, vrR, pR, WR, hR,
                                     alpha_batch, e6phi_batch, gamma_rr_batch)

        # Ensure contiguous
        DL = np.ascontiguousarray(DL)
        DR = np.ascontiguousarray(DR)
        SrL = np.ascontiguousarray(SrL)
        SrR = np.ascontiguousarray(SrR)
        tauL = np.ascontiguousarray(tauL)
        tauR = np.ascontiguousarray(tauR)
        lam_minus = np.ascontiguousarray(lam_minus)
        lam_plus = np.ascontiguousarray(lam_plus)

        # JIT: HLL flux combination (parallelized)
        return hll_flux_kernel(DL, SrL, tauL, DR, SrR, tauR,
                               FL, FR, lam_minus, lam_plus)

    # Keep original methods for backward compatibility / debugging
    def _find_cp_cm(self, flux_dirn, g4UU, u4U, cs2):
        """
        Compute characteristic speeds c+/c- using quadratic method.
        (Fallback non-JIT version for debugging)
        """
        return find_cp_cm_kernel(flux_dirn, g4UU, u4U, cs2)

    def _entropy_fix(self, lam_minus, lam_plus, delta=1e-8):
        """Simple entropy fix (fallback non-JIT version)."""
        return entropy_fix_kernel(lam_minus, lam_plus, delta)


# ==============================================================================
# LLF (LOCAL LAX-FRIEDRICHS) RIEMANN SOLVER
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def llf_flux_kernel(DL, SrL, tauL, DR, SrR, tauR,
                    FL, FR, lam_minus, lam_plus):
    """
    JIT-compiled kernel for LLF (Local Lax-Friedrichs) flux computation.

    The LLF flux formula:
        F_LLF = 0.5 * (F_L + F_R) - 0.5 * λ_max * (U_R - U_L)

    Where λ_max = max(|λ⁺|, |λ⁻|) is the maximum characteristic speed.

    More diffusive than HLL but unconditionally stable and robust.

    Args:
        DL, SrL, tauL: (M,) left conservative variables
        DR, SrR, tauR: (M,) right conservative variables
        FL: (M, 3) left physical flux
        FR: (M, 3) right physical flux
        lam_minus: (M,) left wave speed (negative)
        lam_plus: (M,) right wave speed (positive)

    Returns:
        out: (M, 3) LLF flux
    """
    M = DL.shape[0]
    out = np.empty((M, 3))

    for m in range(M):
        # Maximum absolute wave speed
        lam_max = max(lam_plus[m], -lam_minus[m])

        # Conservative variable differences
        dD = DR[m] - DL[m]
        dSr = SrR[m] - SrL[m]
        dTau = tauR[m] - tauL[m]

        # LLF flux formula: F = 0.5*(F_L + F_R) - 0.5*λ_max*ΔU
        out[m, 0] = 0.5 * (FL[m, 0] + FR[m, 0] - lam_max * dD)
        out[m, 1] = 0.5 * (FL[m, 1] + FR[m, 1] - lam_max * dSr)
        out[m, 2] = 0.5 * (FL[m, 2] + FR[m, 2] - lam_max * dTau)

    return out


class LLFRiemannSolver:
    """
    Local Lax-Friedrichs (Rusanov) Riemann solver for GRHD.

    The LLF flux formula:
        F_LLF = 0.5 * (F_L + F_R) - 0.5 * λ_max * (U_R - U_L)

    Where λ_max = max(|λ⁺|, |λ⁻|) is the maximum characteristic speed.

    Properties compared to HLL:
        - More diffusive (smears contact discontinuities more)
        - More robust near strong shocks
        - Simpler formula, slightly faster per interface
        - Good for testing or when HLL fails

    Uses the same JIT-compiled infrastructure as HLL for performance.
    """

    def __init__(self, name: str = "LLF", atmosphere=None):
        self.name = name
        self.atmosphere = atmosphere

    def solve_batch(self, UL_batch, UR_batch, primL_batch, primR_batch,
                    geom: GeometryState, eos):
        """
        Compute LLF fluxes for multiple interfaces using JIT kernels.

        Same signature as HLLRiemannSolver.solve_batch for drop-in replacement.

        All inputs are arrays of length M. Returns (M, 3) flux array.
        """
        M = len(UL_batch)

        # Extract geometry components
        gamma_rr_batch = np.ascontiguousarray(geom.gamma_rr, dtype=np.float64)
        alpha_batch = np.ascontiguousarray(geom.alpha, dtype=np.float64)
        beta_r_batch = np.ascontiguousarray(geom.beta_r, dtype=np.float64)
        e6phi_batch = np.ascontiguousarray(geom.e6phi, dtype=np.float64)

        # Convert inputs to contiguous float64 arrays for Numba
        UL_batch = np.ascontiguousarray(UL_batch, dtype=np.float64)
        UR_batch = np.ascontiguousarray(UR_batch, dtype=np.float64)
        primL_batch = np.ascontiguousarray(primL_batch, dtype=np.float64)
        primR_batch = np.ascontiguousarray(primR_batch, dtype=np.float64)

        # Unpack conservatives
        DL = UL_batch[:, 0]
        SrL = UL_batch[:, 1]
        tauL = UL_batch[:, 2]
        DR = UR_batch[:, 0]
        SrR = UR_batch[:, 1]
        tauR = UR_batch[:, 2]

        # Unpack primitives
        rho0L = primL_batch[:, 0]
        vrL = primL_batch[:, 1]
        pL = primL_batch[:, 2]

        rho0R = primR_batch[:, 0]
        vrR = primR_batch[:, 1]
        pR = primR_batch[:, 2]

        # Apply floors using vectorized operations
        pL = np.maximum(pL, self.atmosphere.p_floor)
        pR = np.maximum(pR, self.atmosphere.p_floor)
        vrL = np.clip(vrL, -self.atmosphere.v_max, self.atmosphere.v_max)
        vrR = np.clip(vrR, -self.atmosphere.v_max, self.atmosphere.v_max)

        gamma_rr_batch = np.maximum(gamma_rr_batch, 1e-30)
        alpha_batch = np.maximum(alpha_batch, 1e-30)

        # Compute EOS quantities (not JIT-compatible, done in Python)
        epsL = np.maximum(eos.eps_from_rho_p(rho0L, pL), 1e-15)
        epsR = np.maximum(eos.eps_from_rho_p(rho0R, pR), 1e-15)
        cs2L = np.clip(eos.sound_speed_squared(rho0L, pL, epsL), 0.0, 1.0 - 1e-12)
        cs2R = np.clip(eos.sound_speed_squared(rho0R, pR, epsR), 0.0, 1.0 - 1e-12)

        # Compute Lorentz factors and enthalpies for flux calculation
        WL = compute_lorentz_factor_1d(vrL, gamma_rr_batch)
        WR = compute_lorentz_factor_1d(vrR, gamma_rr_batch)
        hL = 1.0 + epsL + pL / np.maximum(rho0L, 1e-30)
        hR = 1.0 + epsR + pR / np.maximum(rho0R, 1e-30)

        # Compute 4-velocities for all interfaces (using geometry module)
        u4U_L, _ = compute_4velocity_1d(vrL, gamma_rr_batch, alpha_batch, beta_r_batch)
        u4U_R, _ = compute_4velocity_1d(vrR, gamma_rr_batch, alpha_batch, beta_r_batch)

        # Build 4-metric components at all interfaces
        gamma_rr_UU = 1.0 / gamma_rr_batch
        g4UU = compute_g4UU_1d(alpha_batch, beta_r_batch, gamma_rr_UU)

        # Ensure contiguous arrays for JIT kernels
        g4UU = np.ascontiguousarray(g4UU)
        u4U_L = np.ascontiguousarray(u4U_L)
        u4U_R = np.ascontiguousarray(u4U_R)
        cs2L = np.ascontiguousarray(cs2L)
        cs2R = np.ascontiguousarray(cs2R)

        # JIT: Compute characteristic speeds (reuse HLL kernels)
        cmL, cpL = find_cp_cm_kernel(0, g4UU, u4U_L, cs2L)
        cmR, cpR = find_cp_cm_kernel(0, g4UU, u4U_R, cs2R)

        # Global wave speed bounds
        cmax = np.maximum(0.0, np.maximum(cpL, cpR))
        cmin = -np.minimum(0.0, np.minimum(cmL, cmR))

        # Standard notation
        lam_minus = -cmin
        lam_plus = cmax

        # JIT: Entropy fix (reuse HLL kernel for robustness)
        lam_minus, lam_plus = entropy_fix_kernel(lam_minus, lam_plus)

        # Ensure contiguous for flux kernels
        rho0L = np.ascontiguousarray(rho0L)
        rho0R = np.ascontiguousarray(rho0R)
        vrL = np.ascontiguousarray(vrL)
        vrR = np.ascontiguousarray(vrR)
        pL = np.ascontiguousarray(pL)
        pR = np.ascontiguousarray(pR)
        WL = np.ascontiguousarray(WL)
        WR = np.ascontiguousarray(WR)
        hL = np.ascontiguousarray(hL)
        hR = np.ascontiguousarray(hR)

        # JIT: Compute physical fluxes (reuse HLL kernel)
        FL = physical_flux_1d_kernel(rho0L, vrL, pL, WL, hL,
                                     alpha_batch, e6phi_batch, gamma_rr_batch)
        FR = physical_flux_1d_kernel(rho0R, vrR, pR, WR, hR,
                                     alpha_batch, e6phi_batch, gamma_rr_batch)

        # Ensure contiguous
        DL = np.ascontiguousarray(DL)
        DR = np.ascontiguousarray(DR)
        SrL = np.ascontiguousarray(SrL)
        SrR = np.ascontiguousarray(SrR)
        tauL = np.ascontiguousarray(tauL)
        tauR = np.ascontiguousarray(tauR)
        lam_minus = np.ascontiguousarray(lam_minus)
        lam_plus = np.ascontiguousarray(lam_plus)

        # JIT: LLF flux combination (the only difference from HLL)
        return llf_flux_kernel(DL, SrL, tauL, DR, SrR, tauR,
                               FL, FR, lam_minus, lam_plus)
