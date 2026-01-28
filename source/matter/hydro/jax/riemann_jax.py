"""
JAX-based HLL and LLF Riemann solvers for GRHD.

Uses quadratic eigenvalue method for characteristic speeds.
API-compatible with riemann.py.

Usage:
    Set ENGRENAGE_BACKEND=jax before importing, or use directly:

    from source.matter.hydro.riemann_jax import HLLRiemannSolverJAX, LLFRiemannSolverJAX
    solver = HLLRiemannSolverJAX(atmosphere=atm)
    flux = solver.solve_batch(UL, UR, primL, primR, gamma_rr, alpha, beta_r, eos, e6phi)
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np


# ==============================================================================
# JAX GEOMETRY HELPERS (inlined for self-contained module)
# ==============================================================================

@jit
def compute_lorentz_factor_1d_jax(vr, gamma_rr):
    """
    Compute Lorentz factor W = 1/sqrt(1 - v^2) for 1D radial flow.

    Args:
        vr: (M,) radial velocity v^r
        gamma_rr: (M,) radial metric component γ_rr

    Returns:
        W: (M,) Lorentz factor
    """
    v2 = gamma_rr * vr * vr
    v2_safe = jnp.minimum(v2, 1.0 - 1e-12)
    return 1.0 / jnp.sqrt(1.0 - v2_safe)


@jit
def compute_4velocity_1d_jax(vr, gamma_rr, alpha, beta_r):
    """
    Compute contravariant 4-velocity for 1D radial flow.

    Args:
        vr: (M,) radial velocity v^r
        gamma_rr: (M,) radial metric component γ_rr
        alpha: (M,) lapse
        beta_r: (M,) radial shift (typically 0)

    Returns:
        u4U: (M, 4) contravariant 4-velocity [u^0, u^r, 0, 0]
    """
    W = compute_lorentz_factor_1d_jax(vr, gamma_rr)
    M = vr.shape[0]

    # u^0 = W/α
    u0 = W / alpha

    # u^r = W*(v^r - β^r/α) but for TOV β^r = 0, so u^r = W*v^r
    # More generally: u^i = W*(v^i - β^i/α)
    ur = W * vr - (beta_r / alpha) * W

    u4U = jnp.zeros((M, 4))
    u4U = u4U.at[:, 0].set(u0)
    u4U = u4U.at[:, 1].set(ur)

    return u4U


@jit
def compute_g4UU_1d_jax(alpha, beta_r, gamma_rr_UU):
    """
    Compute inverse 4-metric g^{μν} for 1D radial flow.

    Args:
        alpha: (M,) lapse
        beta_r: (M,) radial shift
        gamma_rr_UU: (M,) inverse radial metric (1/γ_rr)

    Returns:
        g4UU: (M, 4, 4) inverse 4-metric
    """
    M = alpha.shape[0]
    g4UU = jnp.zeros((M, 4, 4))

    alpha2 = alpha * alpha

    # g^{00} = -1/α²
    g4UU = g4UU.at[:, 0, 0].set(-1.0 / alpha2)

    # g^{0r} = g^{r0} = β^r/α²
    beta_over_alpha2 = beta_r / alpha2
    g4UU = g4UU.at[:, 0, 1].set(beta_over_alpha2)
    g4UU = g4UU.at[:, 1, 0].set(beta_over_alpha2)

    # g^{rr} = γ^{rr} - β^r β^r / α²
    g4UU = g4UU.at[:, 1, 1].set(gamma_rr_UU - beta_r * beta_r / alpha2)

    # g^{θθ} and g^{φφ} are 1 for unit sphere (placeholder)
    g4UU = g4UU.at[:, 2, 2].set(1.0)
    g4UU = g4UU.at[:, 3, 3].set(1.0)

    return g4UU


# ==============================================================================
# JAX RIEMANN SOLVER KERNELS
# ==============================================================================

@jit
def find_cp_cm_jax(flux_dirn, g4UU, u4U, cs2):
    """
    Compute characteristic speeds c+/c- using quadratic method (JAX version).

    Uses quadratic equation method for robust computation near sonic points.

    Args:
        flux_dirn: Flux direction (0 for radial)
        g4UU: (M, 4, 4) inverse 4-metric
        u4U: (M, 4) contravariant 4-velocity
        cs2: (M,) sound speed squared

    Returns:
        (cminus, cplus): Tuple of (M,) arrays
    """
    i = flux_dirn + 1  # Spatial index (1 for radial)

    v02 = cs2
    one_minus_cs2 = 1.0 - v02

    # Quadratic coefficients
    u0_sq = u4U[:, 0] * u4U[:, 0]
    ui_sq = u4U[:, i] * u4U[:, i]
    u0_ui = u4U[:, 0] * u4U[:, i]

    a = one_minus_cs2 * u0_sq - v02 * g4UU[:, 0, 0]
    b = 2.0 * v02 * g4UU[:, i, 0] - 2.0 * u0_ui * one_minus_cs2
    c = one_minus_cs2 * ui_sq - v02 * g4UU[:, i, i]

    # Solve quadratic
    detm_squared = b * b - 4.0 * a * c
    detm = jnp.sqrt(jnp.maximum(0.0, detm_squared))

    # Avoid division by zero
    inv_2a = 0.5 / jnp.where(jnp.abs(a) < 1e-30, 1.0, a)

    cplus_tmp = (-b + detm) * inv_2a
    cminus_tmp = (-b - detm) * inv_2a

    # Ensure proper ordering
    cminus = jnp.minimum(cplus_tmp, cminus_tmp)
    cplus = jnp.maximum(cplus_tmp, cminus_tmp)

    # Handle degenerate case
    cminus = jnp.where(jnp.abs(a) < 1e-30, -1.0, cminus)
    cplus = jnp.where(jnp.abs(a) < 1e-30, 1.0, cplus)

    return cminus, cplus


@jit
def entropy_fix_jax(lam_minus, lam_plus, delta=1e-8):
    """
    Entropy fix to avoid sonic glitches (JAX version).

    Args:
        lam_minus: (M,) left wave speeds
        lam_plus: (M,) right wave speeds
        delta: Minimum separation from zero

    Returns:
        (lam_minus_fixed, lam_plus_fixed): Tuple of (M,) arrays
    """
    lm = jnp.minimum(lam_minus, -jnp.abs(delta))
    lp = jnp.maximum(lam_plus, jnp.abs(delta))

    # Ensure ordering
    lm_out = jnp.where(lm > lp, lp, lm)
    lp_out = jnp.where(lm > lp, lm, lp)

    return lm_out, lp_out


@jit
def physical_flux_1d_jax(rho0, vr, pressure, W, h, alpha, e6phi, gamma_rr):
    """
    Compute physical flux for 1D radial direction (JAX version).

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
    # For spherical symmetry with zero shift: vtilde^r = v^r
    vtilde_r = vr

    # 4-velocity components: u^0 = W/alpha, u^r = W * v^r
    u0 = W / alpha
    ur = W * vtilde_r

    # Conservative density: D = rho * W
    D = rho0 * W

    # Stress-energy tensor components needed for flux
    # T^{0r} = rho*h*u^0*u^r + P*g^{0r}
    # For zero shift: g^{0r} = 0
    rho_h = rho0 * h
    T0r = rho_h * u0 * ur

    # T^{rr} = rho*h*u^r*u^r + P*g^{rr}
    # g^{rr} = 1/gamma_rr (for diagonal metric)
    grr_inv = 1.0 / gamma_rr
    Trr = rho_h * ur * ur + pressure * grr_inv

    # T^r_r = T^{rr} * gamma_rr = rho*h*u^r*u^r*gamma_rr + P
    Tr_r = rho_h * ur * ur * gamma_rr + pressure

    # Fluxes (densitized by e^{6phi})
    # F_D = e6phi * alpha * D * vtilde^r
    F_D = e6phi * alpha * D * vtilde_r

    # F_Sr = e6phi * alpha * T^r_r
    F_Sr = e6phi * alpha * Tr_r

    # F_tau = e6phi * (alpha^2 * T^{0r} - alpha * D * vtilde^r)
    F_tau = e6phi * (alpha * alpha * T0r - alpha * D * vtilde_r)

    return jnp.stack([F_D, F_Sr, F_tau], axis=1)


@jit
def hll_flux_jax(DL, SrL, tauL, DR, SrR, tauR, FL, FR, lam_minus, lam_plus):
    """
    HLL flux computation (JAX version).

    The HLL flux formula:
        F_HLL = (λ+ F_L - λ- F_R + λ+ λ- (U_R - U_L)) / (λ+ - λ-)

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

    lm = lam_minus
    lp = lam_plus
    denom = lp - lm

    inv_denom = 1.0 / jnp.where(jnp.abs(denom) < 1e-30, 1.0, denom)
    lp_lm = lp * lm

    # U differences
    dD = DR - DL
    dSr = SrR - SrL
    dTau = tauR - tauL

    # HLL formula
    F_hll_D = (lp * FL[:, 0] - lm * FR[:, 0] + lp_lm * dD) * inv_denom
    F_hll_Sr = (lp * FL[:, 1] - lm * FR[:, 1] + lp_lm * dSr) * inv_denom
    F_hll_tau = (lp * FL[:, 2] - lm * FR[:, 2] + lp_lm * dTau) * inv_denom

    # Handle special cases
    # Flow entirely from left (lm >= 0)
    left_flow = lm >= 0.0
    F_hll_D = jnp.where(left_flow, FL[:, 0], F_hll_D)
    F_hll_Sr = jnp.where(left_flow, FL[:, 1], F_hll_Sr)
    F_hll_tau = jnp.where(left_flow, FL[:, 2], F_hll_tau)

    # Flow entirely from right (lp <= 0)
    right_flow = lp <= 0.0
    F_hll_D = jnp.where(right_flow, FR[:, 0], F_hll_D)
    F_hll_Sr = jnp.where(right_flow, FR[:, 1], F_hll_Sr)
    F_hll_tau = jnp.where(right_flow, FR[:, 2], F_hll_tau)

    # Degenerate case (average)
    degenerate = jnp.abs(denom) < 1e-30
    F_hll_D = jnp.where(degenerate, 0.5 * (FL[:, 0] + FR[:, 0]), F_hll_D)
    F_hll_Sr = jnp.where(degenerate, 0.5 * (FL[:, 1] + FR[:, 1]), F_hll_Sr)
    F_hll_tau = jnp.where(degenerate, 0.5 * (FL[:, 2] + FR[:, 2]), F_hll_tau)

    return jnp.stack([F_hll_D, F_hll_Sr, F_hll_tau], axis=1)


@jit
def llf_flux_jax(DL, SrL, tauL, DR, SrR, tauR, FL, FR, lam_minus, lam_plus):
    """
    Local Lax-Friedrichs (LLF/Rusanov) flux computation (JAX version).

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
    # Maximum absolute wave speed
    lam_max = jnp.maximum(lam_plus, -lam_minus)

    # Conservative variable differences
    dD = DR - DL
    dSr = SrR - SrL
    dTau = tauR - tauL

    # LLF flux formula: F = 0.5*(F_L + F_R) - 0.5*λ_max*ΔU
    F_D = 0.5 * (FL[:, 0] + FR[:, 0] - lam_max * dD)
    F_Sr = 0.5 * (FL[:, 1] + FR[:, 1] - lam_max * dSr)
    F_tau = 0.5 * (FL[:, 2] + FR[:, 2] - lam_max * dTau)

    return jnp.stack([F_D, F_Sr, F_tau], axis=1)


# ==============================================================================
# JAX RIEMANN SOLVER CLASSES
# ==============================================================================

class HLLRiemannSolverJAX:
    """
    HLL Riemann solver using quadratic eigenvalue method (JAX version).

    The HLL flux formula:
        F_HLL = (λ+ F_L - λ- F_R + λ+ λ- (U_R - U_L)) / (λ+ - λ-)

    Characteristic speeds λ± computed via quadratic equation from 4-metric and 4-velocity.

    Optimized with JAX JIT compilation for GPU acceleration.
    """

    def __init__(self, name: str = "HLL_JAX", atmosphere=None):
        self.name = name
        self.atmosphere = atmosphere

    def solve_batch(self, UL_batch, UR_batch, primL_batch, primR_batch,
                    gamma_rr_batch, alpha_batch, beta_r_batch, eos, e6phi_batch):
        """
        Compute HLL fluxes for multiple interfaces using JAX.

        All inputs are arrays of length M. Returns (M, 3) flux array.

        Same signature as the Numba HLLRiemannSolver for drop-in replacement.
        """
        # Convert inputs to JAX arrays
        UL = jnp.asarray(UL_batch, dtype=jnp.float64)
        UR = jnp.asarray(UR_batch, dtype=jnp.float64)
        primL = jnp.asarray(primL_batch, dtype=jnp.float64)
        primR = jnp.asarray(primR_batch, dtype=jnp.float64)
        gamma_rr = jnp.asarray(gamma_rr_batch, dtype=jnp.float64)
        alpha = jnp.asarray(alpha_batch, dtype=jnp.float64)
        beta_r = jnp.asarray(beta_r_batch, dtype=jnp.float64)
        e6phi = jnp.asarray(e6phi_batch, dtype=jnp.float64)

        # Unpack conservatives
        DL = UL[:, 0]
        SrL = UL[:, 1]
        tauL = UL[:, 2]
        DR = UR[:, 0]
        SrR = UR[:, 1]
        tauR = UR[:, 2]

        # Unpack primitives
        rho0L = primL[:, 0]
        vrL = primL[:, 1]
        pL = primL[:, 2]

        rho0R = primR[:, 0]
        vrR = primR[:, 1]
        pR = primR[:, 2]

        # Apply floors
        pL = jnp.maximum(pL, self.atmosphere.p_floor)
        pR = jnp.maximum(pR, self.atmosphere.p_floor)
        vrL = jnp.clip(vrL, -self.atmosphere.v_max, self.atmosphere.v_max)
        vrR = jnp.clip(vrR, -self.atmosphere.v_max, self.atmosphere.v_max)

        gamma_rr = jnp.maximum(gamma_rr, 1e-30)
        alpha = jnp.maximum(alpha, 1e-30)

        # Compute EOS quantities (need to handle potential non-JAX EOS)
        # Convert to numpy for EOS calls, then back to JAX
        rho0L_np = np.asarray(rho0L)
        rho0R_np = np.asarray(rho0R)
        pL_np = np.asarray(pL)
        pR_np = np.asarray(pR)

        epsL_np = np.maximum(eos.eps_from_rho_p(rho0L_np, pL_np), 1e-15)
        epsR_np = np.maximum(eos.eps_from_rho_p(rho0R_np, pR_np), 1e-15)
        cs2L_np = np.clip(eos.sound_speed_squared(rho0L_np, pL_np, epsL_np), 0.0, 1.0 - 1e-12)
        cs2R_np = np.clip(eos.sound_speed_squared(rho0R_np, pR_np, epsR_np), 0.0, 1.0 - 1e-12)

        epsL = jnp.asarray(epsL_np)
        epsR = jnp.asarray(epsR_np)
        cs2L = jnp.asarray(cs2L_np)
        cs2R = jnp.asarray(cs2R_np)

        # Compute Lorentz factors and enthalpies
        WL = compute_lorentz_factor_1d_jax(vrL, gamma_rr)
        WR = compute_lorentz_factor_1d_jax(vrR, gamma_rr)
        hL = 1.0 + epsL + pL / jnp.maximum(rho0L, 1e-30)
        hR = 1.0 + epsR + pR / jnp.maximum(rho0R, 1e-30)

        # Compute 4-velocities
        u4U_L = compute_4velocity_1d_jax(vrL, gamma_rr, alpha, beta_r)
        u4U_R = compute_4velocity_1d_jax(vrR, gamma_rr, alpha, beta_r)

        # Build inverse 4-metric
        gamma_rr_UU = 1.0 / gamma_rr
        g4UU = compute_g4UU_1d_jax(alpha, beta_r, gamma_rr_UU)

        # Compute characteristic speeds
        cmL, cpL = find_cp_cm_jax(0, g4UU, u4U_L, cs2L)
        cmR, cpR = find_cp_cm_jax(0, g4UU, u4U_R, cs2R)

        # Global wave speed bounds
        cmax = jnp.maximum(0.0, jnp.maximum(cpL, cpR))
        cmin = -jnp.minimum(0.0, jnp.minimum(cmL, cmR))

        # Standard notation
        lam_minus = -cmin
        lam_plus = cmax

        # Entropy fix
        lam_minus, lam_plus = entropy_fix_jax(lam_minus, lam_plus)

        # Compute physical fluxes
        FL = physical_flux_1d_jax(rho0L, vrL, pL, WL, hL, alpha, e6phi, gamma_rr)
        FR = physical_flux_1d_jax(rho0R, vrR, pR, WR, hR, alpha, e6phi, gamma_rr)

        # HLL flux combination
        flux = hll_flux_jax(DL, SrL, tauL, DR, SrR, tauR, FL, FR, lam_minus, lam_plus)

        # Convert back to numpy for compatibility
        return np.asarray(flux)


class LLFRiemannSolverJAX:
    """
    Local Lax-Friedrichs (Rusanov) Riemann solver (JAX version).

    The LLF flux formula:
        F_LLF = 0.5 * (F_L + F_R) - 0.5 * λ_max * (U_R - U_L)

    Where λ_max = max(|λ⁺|, |λ⁻|) is the maximum characteristic speed.

    Properties compared to HLL:
        - More diffusive (smears contact discontinuities more)
        - More robust near strong shocks
        - Simpler formula, slightly faster per interface
        - Good for testing or when HLL fails

    Optimized with JAX JIT compilation for GPU acceleration.
    """

    def __init__(self, name: str = "LLF_JAX", atmosphere=None):
        self.name = name
        self.atmosphere = atmosphere

    def solve_batch(self, UL_batch, UR_batch, primL_batch, primR_batch,
                    gamma_rr_batch, alpha_batch, beta_r_batch, eos, e6phi_batch):
        """
        Compute LLF fluxes for multiple interfaces using JAX.

        Same signature as HLLRiemannSolverJAX.solve_batch for drop-in replacement.

        All inputs are arrays of length M. Returns (M, 3) flux array.
        """
        # Convert inputs to JAX arrays
        UL = jnp.asarray(UL_batch, dtype=jnp.float64)
        UR = jnp.asarray(UR_batch, dtype=jnp.float64)
        primL = jnp.asarray(primL_batch, dtype=jnp.float64)
        primR = jnp.asarray(primR_batch, dtype=jnp.float64)
        gamma_rr = jnp.asarray(gamma_rr_batch, dtype=jnp.float64)
        alpha = jnp.asarray(alpha_batch, dtype=jnp.float64)
        beta_r = jnp.asarray(beta_r_batch, dtype=jnp.float64)
        e6phi = jnp.asarray(e6phi_batch, dtype=jnp.float64)

        # Unpack conservatives
        DL = UL[:, 0]
        SrL = UL[:, 1]
        tauL = UL[:, 2]
        DR = UR[:, 0]
        SrR = UR[:, 1]
        tauR = UR[:, 2]

        # Unpack primitives
        rho0L = primL[:, 0]
        vrL = primL[:, 1]
        pL = primL[:, 2]

        rho0R = primR[:, 0]
        vrR = primR[:, 1]
        pR = primR[:, 2]

        # Apply floors
        pL = jnp.maximum(pL, self.atmosphere.p_floor)
        pR = jnp.maximum(pR, self.atmosphere.p_floor)
        vrL = jnp.clip(vrL, -self.atmosphere.v_max, self.atmosphere.v_max)
        vrR = jnp.clip(vrR, -self.atmosphere.v_max, self.atmosphere.v_max)

        gamma_rr = jnp.maximum(gamma_rr, 1e-30)
        alpha = jnp.maximum(alpha, 1e-30)

        # Compute EOS quantities (need to handle potential non-JAX EOS)
        rho0L_np = np.asarray(rho0L)
        rho0R_np = np.asarray(rho0R)
        pL_np = np.asarray(pL)
        pR_np = np.asarray(pR)

        epsL_np = np.maximum(eos.eps_from_rho_p(rho0L_np, pL_np), 1e-15)
        epsR_np = np.maximum(eos.eps_from_rho_p(rho0R_np, pR_np), 1e-15)
        cs2L_np = np.clip(eos.sound_speed_squared(rho0L_np, pL_np, epsL_np), 0.0, 1.0 - 1e-12)
        cs2R_np = np.clip(eos.sound_speed_squared(rho0R_np, pR_np, epsR_np), 0.0, 1.0 - 1e-12)

        epsL = jnp.asarray(epsL_np)
        epsR = jnp.asarray(epsR_np)
        cs2L = jnp.asarray(cs2L_np)
        cs2R = jnp.asarray(cs2R_np)

        # Compute Lorentz factors and enthalpies
        WL = compute_lorentz_factor_1d_jax(vrL, gamma_rr)
        WR = compute_lorentz_factor_1d_jax(vrR, gamma_rr)
        hL = 1.0 + epsL + pL / jnp.maximum(rho0L, 1e-30)
        hR = 1.0 + epsR + pR / jnp.maximum(rho0R, 1e-30)

        # Compute 4-velocities
        u4U_L = compute_4velocity_1d_jax(vrL, gamma_rr, alpha, beta_r)
        u4U_R = compute_4velocity_1d_jax(vrR, gamma_rr, alpha, beta_r)

        # Build inverse 4-metric
        gamma_rr_UU = 1.0 / gamma_rr
        g4UU = compute_g4UU_1d_jax(alpha, beta_r, gamma_rr_UU)

        # Compute characteristic speeds (reuse HLL logic)
        cmL, cpL = find_cp_cm_jax(0, g4UU, u4U_L, cs2L)
        cmR, cpR = find_cp_cm_jax(0, g4UU, u4U_R, cs2R)

        # Global wave speed bounds
        cmax = jnp.maximum(0.0, jnp.maximum(cpL, cpR))
        cmin = -jnp.minimum(0.0, jnp.minimum(cmL, cmR))

        # Standard notation
        lam_minus = -cmin
        lam_plus = cmax

        # Entropy fix (for robustness)
        lam_minus, lam_plus = entropy_fix_jax(lam_minus, lam_plus)

        # Compute physical fluxes
        FL = physical_flux_1d_jax(rho0L, vrL, pL, WL, hL, alpha, e6phi, gamma_rr)
        FR = physical_flux_1d_jax(rho0R, vrR, pR, WR, hR, alpha, e6phi, gamma_rr)

        # LLF flux combination (the only difference from HLL)
        flux = llf_flux_jax(DL, SrL, tauL, DR, SrR, tauR, FL, FR, lam_minus, lam_plus)

        # Convert back to numpy for compatibility
        return np.asarray(flux)


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def create_hll_solver_jax(atmosphere=None):
    """Create JAX HLL Riemann solver."""
    return HLLRiemannSolverJAX(atmosphere=atmosphere)


def create_llf_solver_jax(atmosphere=None):
    """Create JAX LLF Riemann solver."""
    return LLFRiemannSolverJAX(atmosphere=atmosphere)


# Aliases for compatibility
HLLRiemannSolver = HLLRiemannSolverJAX
LLFRiemannSolver = LLFRiemannSolverJAX
