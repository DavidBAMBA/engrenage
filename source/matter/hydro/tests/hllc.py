# riemann.py
"""
HLL, LLF, and HLLC Riemann solvers for GRHD using quadratic eigenvalue method.

Uses quadratic equation method for characteristic speeds (robust near sonic points).
Fully optimized with Numba JIT compilation for CPU performance.
"""

import numpy as np
from numba import jit, prange
from .valencia_reference_metric import ValenciaReferenceMetric
from .geometry import (
    GeometryState,
    compute_4velocity_1d,
    compute_g4UU_1d,
    compute_lorentz_factor_1d
)


# ==============================================================================
# JIT-COMPILED KERNELS (SHARED BY ALL SOLVERS)
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def find_cp_cm_kernel(flux_dirn, g4UU, u4U, cs2):
    """
    JIT-compiled kernel to compute characteristic speeds c+/c- using quadratic method.

    Returns:
        (cminus, cplus): Tuple of (M,) arrays
    """
    M = cs2.shape[0]
    i = flux_dirn + 1  # Spatial index (1 for radial)

    cminus = np.empty(M)
    cplus = np.empty(M)

    for m in prange(M):
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


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def physical_flux_1d_kernel(rho0, vr, pressure, W, h, alpha, e6phi, gamma_rr, beta_r):
    """
    JIT-compiled kernel to compute physical flux for 1D radial direction.

    Implements Valencia flux formulas with full shift support:
        vtilde^r = v^r - β^r/α
        F̃_D^r   = e^{6φ} α D vtilde^r
        F̃_Sr^r  = e^{6φ} α T^r_r
        F̃_τ^r   = e^{6φ} (α² T^{0r} - α D vtilde^r)


    Returns:
        F: (M, 3) physical flux [F_D, F_Sr, F_tau]
    """
    M = rho0.shape[0]
    F = np.empty((M, 3))

    for m in prange(M):
        # Metric components
        alph = alpha[m]
        e6p = e6phi[m]
        grr = gamma_rr[m]
        betar = beta_r[m]

        # Fluid quantities
        rho = rho0[m]
        v = vr[m]
        p = pressure[m]
        Wm = W[m]
        hm = h[m]

        # Valencia velocity: vtilde^r = v^r - β^r/α
        vtilde_r = v - betar / alph

        # 4-velocity components: u^0 = W/α, u^r = W * vtilde^r
        u0 = Wm / alph
        ur = Wm * vtilde_r

        # Conservative density: D = ρ W
        D = rho * Wm

        # Inverse metric components with shift
        alph_sq = alph * alph
        grr_inv = 1.0 / grr

        # g^{0r} = β^r/α²
        g4UU_0r = betar / alph_sq

        # g^{rr} = γ^{rr} - β^r β^r/α²
        g4UU_rr = grr_inv - betar * betar / alph_sq

        # Stress-energy tensor components
        rho_h = rho * hm

        # T^{0r} = ρh u^0 u^r + P g^{0r}
        T0r = rho_h * u0 * ur + p * g4UU_0r

        # v_r = γ_{rr} v^r (covariant velocity component)
        v_r = grr * v

        # T^r_r using equation (27): W² ρh v_r ṽ^r + p
        Tr_r = rho_h * Wm * Wm * v_r * vtilde_r + p

        # Fluxes (densitized by e^{6φ})
        # F̃_D = e^{6φ} α D vtilde^r
        F[m, 0] = e6p * alph * D * vtilde_r

        # F̃_Sr = e^{6φ} α T^r_r
        F[m, 1] = e6p * alph * Tr_r

        # F̃_τ = e^{6φ} (α² T^{0r} - α D vtilde^r)
        F[m, 2] = e6p * (alph_sq * T0r - alph * D * vtilde_r)

    return F


# ==============================================================================
# HLLC-SPECIFIC KERNELS
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def transform_to_local_frame_kernel(vr, rho0, pressure, eps, cs2,
                                     alpha, beta_r, gamma_rr):
    """
    Transform primitives to local orthonormal tetrad frame.
    
    Following Lam & Shibata (2025) and White, Stone & Gammie (2016):
        v^(r̂) = √γ_rr (v^r - β^r/α)
        
    NO relativistic denominator (shift is coordinate transport, not physical velocity).
    
    Args:
        vr: (M,) coordinate velocity v^r
        rho0: (M,) rest mass density
        pressure: (M,) pressure
        eps: (M,) specific internal energy
        cs2: (M,) sound speed squared
        alpha, beta_r, gamma_rr: (M,) metric components
        
    Returns:
        (v_local, W_local, h, D_local, S_local, E_local, F_D, F_S, F_E, lambda_minus, lambda_plus):
            All (M,) arrays of local frame quantities
    """
    M = vr.shape[0]
    
    v_local = np.empty(M)
    W_local = np.empty(M)
    h = np.empty(M)
    D_local = np.empty(M)
    S_local = np.empty(M)
    E_local = np.empty(M)
    F_D = np.empty(M)
    F_S = np.empty(M)
    F_E = np.empty(M)
    lambda_minus = np.empty(M)
    lambda_plus = np.empty(M)
    
    for m in prange(M):
        alph = alpha[m]
        betar = beta_r[m]
        sqrt_grr = np.sqrt(gamma_rr[m])
        
        # Transform velocity to local frame: v^(r̂) = √γ_rr (v^r - β^r/α)
        v_loc = sqrt_grr * (vr[m] - betar / alph)
        
        # Clip to prevent superluminal velocities in local frame
        v_loc = max(-0.9999, min(0.9999, v_loc))
        v_local[m] = v_loc
        
        # Lorentz factor in local frame
        W_loc = 1.0 / np.sqrt(1.0 - v_loc * v_loc)
        W_local[m] = W_loc
        
        # Specific enthalpy
        rho = rho0[m]
        p = pressure[m]
        h_val = 1.0 + eps[m] + p / max(rho, 1e-30)
        h[m] = h_val
        
        # Conservative variables in local frame
        D_loc = rho * W_loc
        S_loc = rho * h_val * W_loc * W_loc * v_loc
        E_loc = rho * h_val * W_loc * W_loc - p
        
        D_local[m] = D_loc
        S_local[m] = S_loc
        E_local[m] = E_loc
        
        # Physical fluxes in local frame (Minkowski)
        F_D[m] = D_loc * v_loc
        F_S[m] = S_loc * v_loc + p
        F_E[m] = S_loc  # = (E + p) * v in local frame
        
        # Characteristic speeds in local frame (special relativistic)
        cs = np.sqrt(cs2[m])
        denom_plus = 1.0 + v_loc * cs
        denom_minus = 1.0 - v_loc * cs
        
        # Avoid division by zero
        if abs(denom_plus) < 1e-30:
            lam_p = 1.0
        else:
            lam_p = (v_loc + cs) / denom_plus
            
        if abs(denom_minus) < 1e-30:
            lam_m = -1.0
        else:
            lam_m = (v_loc - cs) / denom_minus
            
        lambda_plus[m] = lam_p
        lambda_minus[m] = lam_m
    
    return (v_local, W_local, h, D_local, S_local, E_local, 
            F_D, F_S, F_E, lambda_minus, lambda_plus)


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def hllc_flux_kernel(D_L, S_L, E_L, D_R, S_R, E_R,
                     F_D_L, F_S_L, F_E_L, F_D_R, F_S_R, F_E_R,
                     v_loc_L, v_loc_R, p_L, p_R,
                     lam_L, lam_R, v_interface):
    """
    JIT-compiled HLLC flux computation in local tetrad frame.
    
    Implements Mignone & Bodo (2005) HLLC algorithm in local Minkowski frame,
    then region selection based on interface velocity.
    
    Args:
        D_L, S_L, E_L: (M,) left local conserved variables
        D_R, S_R, E_R: (M,) right local conserved variables
        F_D_L, F_S_L, F_E_L: (M,) left local fluxes
        F_D_R, F_S_R, F_E_R: (M,) right local fluxes
        v_loc_L, v_loc_R: (M,) local frame velocities
        p_L, p_R: (M,) pressures
        lam_L, lam_R: (M,) characteristic speeds
        v_interface: (M,) interface velocity in local frame
        
    Returns:
        F_out: (M, 3) HLLC flux in local frame [F_D, F_S, F_E]
    """
    M = D_L.shape[0]
    F_out = np.empty((M, 3))
    
    for m in prange(M):
        lam_minus = lam_L[m]
        lam_plus = lam_R[m]
        v_int = v_interface[m]
        
        # HLL averages
        denom = lam_plus - lam_minus
        if abs(denom) < 1e-30:
            # Fallback to arithmetic average
            F_out[m, 0] = 0.5 * (F_D_L[m] + F_D_R[m])
            F_out[m, 1] = 0.5 * (F_S_L[m] + F_S_R[m])
            F_out[m, 2] = 0.5 * (F_E_L[m] + F_E_R[m])
            continue
            
        inv_denom = 1.0 / denom
        
        # HLL state
        D_HLL = (lam_plus * D_R[m] - lam_minus * D_L[m] + F_D_L[m] - F_D_R[m]) * inv_denom
        S_HLL = (lam_plus * S_R[m] - lam_minus * S_L[m] + F_S_L[m] - F_S_R[m]) * inv_denom
        E_HLL = (lam_plus * E_R[m] - lam_minus * E_L[m] + F_E_L[m] - F_E_R[m]) * inv_denom
        
        F_D_HLL = (lam_plus * F_D_L[m] - lam_minus * F_D_R[m] + lam_plus * lam_minus * (D_R[m] - D_L[m])) * inv_denom
        F_S_HLL = (lam_plus * F_S_L[m] - lam_minus * F_S_R[m] + lam_plus * lam_minus * (S_R[m] - S_L[m])) * inv_denom
        F_E_HLL = (lam_plus * F_E_L[m] - lam_minus * F_E_R[m] + lam_plus * lam_minus * (E_R[m] - E_L[m])) * inv_denom
        
        # Contact wave speed (quadratic equation for pressure)
        # λ_c² E_HLL - λ_c (F_E_HLL + F_S_HLL) + S_HLL = 0
        a_coef = E_HLL
        b_coef = -(F_E_HLL + F_S_HLL)
        c_coef = S_HLL
        
        # Solve quadratic
        if abs(a_coef) < 1e-30:
            # Linear equation
            if abs(b_coef) < 1e-30:
                lam_c = 0.0
            else:
                lam_c = -c_coef / b_coef
        else:
            discr = b_coef * b_coef - 4.0 * a_coef * c_coef
            if discr < 0.0:
                discr = 0.0
            sqrt_discr = np.sqrt(discr)
            
            # Two roots - choose physically relevant one
            lam_c1 = (-b_coef + sqrt_discr) / (2.0 * a_coef)
            lam_c2 = (-b_coef - sqrt_discr) / (2.0 * a_coef)
            
            # Choose root between lam_minus and lam_plus
            if lam_minus <= lam_c1 <= lam_plus:
                lam_c = lam_c1
            elif lam_minus <= lam_c2 <= lam_plus:
                lam_c = lam_c2
            else:
                # Fallback: average of roots
                lam_c = 0.5 * (lam_c1 + lam_c2)
        
        # Contact pressure
        p_c = -lam_c * F_E_HLL - F_S_HLL
        
        # Star region states (left and right of contact)
        denom_cL = lam_minus - lam_c
        denom_cR = lam_plus - lam_c
        
        if abs(denom_cL) < 1e-30:
            D_cL = D_L[m]
            S_cL = S_L[m]
            E_cL = E_L[m]
        else:
            factor_L = (lam_minus - v_loc_L[m]) / denom_cL
            D_cL = D_L[m] * factor_L
            S_cL = (S_L[m] * (lam_minus - v_loc_L[m]) + p_c - p_L[m]) / denom_cL
            E_cL = (E_L[m] * (lam_minus - v_loc_L[m]) + p_c * lam_c - p_L[m] * v_loc_L[m]) / denom_cL
        
        if abs(denom_cR) < 1e-30:
            D_cR = D_R[m]
            S_cR = S_R[m]
            E_cR = E_R[m]
        else:
            factor_R = (lam_plus - v_loc_R[m]) / denom_cR
            D_cR = D_R[m] * factor_R
            S_cR = (S_R[m] * (lam_plus - v_loc_R[m]) + p_c - p_R[m]) / denom_cR
            E_cR = (E_R[m] * (lam_plus - v_loc_R[m]) + p_c * lam_c - p_R[m] * v_loc_R[m]) / denom_cR
        
        # Star region fluxes
        F_D_cL = F_D_L[m] + lam_minus * (D_cL - D_L[m])
        F_S_cL = F_S_L[m] + lam_minus * (S_cL - S_L[m])
        F_E_cL = F_E_L[m] + lam_minus * (E_cL - E_L[m])
        
        F_D_cR = F_D_R[m] + lam_plus * (D_cR - D_R[m])
        F_S_cR = F_S_R[m] + lam_plus * (S_cR - S_R[m])
        F_E_cR = F_E_R[m] + lam_plus * (E_cR - E_R[m])
        
        # Region selection based on interface velocity
        if lam_minus >= v_int:
            # Left state
            F_out[m, 0] = F_D_L[m]
            F_out[m, 1] = F_S_L[m]
            F_out[m, 2] = F_E_L[m]
        elif lam_minus < v_int <= lam_c:
            # Left star region
            F_out[m, 0] = F_D_cL
            F_out[m, 1] = F_S_cL
            F_out[m, 2] = F_E_cL
        elif lam_c < v_int < lam_plus:
            # Right star region
            F_out[m, 0] = F_D_cR
            F_out[m, 1] = F_S_cR
            F_out[m, 2] = F_E_cR
        else:
            # Right state
            F_out[m, 0] = F_D_R[m]
            F_out[m, 1] = F_S_R[m]
            F_out[m, 2] = F_E_R[m]
    
    return F_out


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def transform_flux_to_global_kernel(F_local, alpha, e4phi, sqrt_hatg, sqrt_bargamma_rr):
    """
    Transform fluxes from local tetrad frame to global Valencia coordinates.
    
    Transformation factor: T = α e^(4φ) √ĝ / √γ̄_rr
    
    For flat reference metric with γ̄_rr = 1:
        T = α e^(4φ) r² sin(θ)
        
    Args:
        F_local: (M, 3) fluxes in local frame [F_D, F_S, F_E]
        alpha: (M,) lapse function
        e4phi: (M,) e^(4φ) conformal factor
        sqrt_hatg: (M,) √ĝ from reference metric
        sqrt_bargamma_rr: (M,) √γ̄_rr
        
    Returns:
        F_global: (M, 3) densitized Valencia fluxes [F̃_D, F̃_Sr, F̃_τ]
    """
    M = F_local.shape[0]
    F_global = np.empty((M, 3))
    
    for m in prange(M):
        # Transformation factor
        T = alpha[m] * e4phi[m] * sqrt_hatg[m] / sqrt_bargamma_rr[m]
        
        # Transform density flux (D is scalar)
        F_global[m, 0] = T * F_local[m, 0]
        
        # Transform momentum flux (S is spatial vector component)
        F_global[m, 1] = T * F_local[m, 1]
        
        # Transform energy flux
        # τ = E - D, so F_τ = F_E - F_D
        F_global[m, 2] = T * (F_local[m, 2] - F_local[m, 0])
    
    return F_global


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def hll_flux_kernel(DL, SrL, tauL, DR, SrR, tauR,
                    FL, FR, lam_minus, lam_plus):
    """
    JIT-compiled parallel kernel for HLL flux computation.

    HLL formula: F = (λ⁺ F_L - λ⁻ F_R + λ⁺λ⁻ ΔU) / (λ⁺ - λ⁻)

    Note: entropy_fix_kernel guarantees λ⁻ < 0 < λ⁺, so no edge cases needed.

    Args:
        DL, SrL, tauL: (M,) left conservative variables
        DR, SrR, tauR: (M,) right conservative variables
        FL: (M, 3) left physical flux
        FR: (M, 3) right physical flux
        lam_minus: (M,) left wave speed (negative after entropy fix)
        lam_plus: (M,) right wave speed (positive after entropy fix)

    Returns:
        out: (M, 3) HLL flux
    """
    M = DL.shape[0]
    out = np.empty((M, 3))

    for m in prange(M):
        lm = lam_minus[m]
        lp = lam_plus[m]

        inv_denom = 1.0 / (lp - lm)
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


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def hll_solver_fused_kernel(
    # Primitives (M,)
    rho0L, vrL, pL, epsL, rho0R, vrR, pR, epsR,
    # Conservatives (M,)
    DL, SrL, tauL, DR, SrR, tauR,
    # Sound speeds (M,)
    cs2L, cs2R,
    # Geometry (M,)
    alpha, beta_r, gamma_rr, e6phi,
    # Output (M, 3)
    F_out
):
    """
    Fully fused HLL Riemann solver kernel.

    Computes everything in a single parallel loop:
    1. Lorentz factors and enthalpies
    2. 4-velocities and g4UU components
    3. Characteristic speeds (quadratic method)
    4. Physical fluxes (left and right)
    5. HLL flux combination

    This eliminates all intermediate arrays and kernel launch overhead.
    """
    M = rho0L.shape[0]
    ENTROPY_DELTA = 1e-8

    for m in prange(M):
        # ========== Geometry ==========
        alph = alpha[m]
        betar = beta_r[m]
        grr = gamma_rr[m]
        e6p = e6phi[m]

        alph_sq = alph * alph
        grr_inv = 1.0 / grr

        # g^{μν} components for radial direction
        g4UU_00 = -1.0 / alph_sq
        g4UU_0r = betar / alph_sq
        g4UU_rr = grr_inv - betar * betar / alph_sq

        # ========== LEFT STATE ==========
        rhoL = rho0L[m]
        vL = vrL[m]
        presL = pL[m]

        # Lorentz factor: W = 1/sqrt(1 - v^2 γ_rr)
        v2_grrL = vL * vL * grr
        if v2_grrL >= 1.0:
            v2_grrL = 1.0 - 1e-10
        WL = 1.0 / np.sqrt(1.0 - v2_grrL)

        # Enthalpy
        hL = 1.0 + epsL[m] + presL / max(rhoL, 1e-30)

        # Valencia velocity and 4-velocity
        vtilde_rL = vL - betar / alph
        u0L = WL / alph
        urL = WL * vtilde_rL

        # Characteristic speeds (quadratic method)
        v02L = cs2L[m]
        one_minus_cs2L = 1.0 - v02L

        u0_sqL = u0L * u0L
        ur_sqL = urL * urL
        u0_urL = u0L * urL

        aL = one_minus_cs2L * u0_sqL - v02L * g4UU_00
        bL = 2.0 * v02L * g4UU_0r - 2.0 * u0_urL * one_minus_cs2L
        cL = one_minus_cs2L * ur_sqL - v02L * g4UU_rr

        detm_sqL = bL * bL - 4.0 * aL * cL
        if detm_sqL < 0.0:
            detm_sqL = 0.0
        detmL = np.sqrt(detm_sqL)

        if abs(aL) < 1e-30:
            cmL = -1.0
            cpL = 1.0
        else:
            inv_2aL = 0.5 / aL
            cp_tmpL = (-bL + detmL) * inv_2aL
            cm_tmpL = (-bL - detmL) * inv_2aL
            cmL = min(cp_tmpL, cm_tmpL)
            cpL = max(cp_tmpL, cm_tmpL)

        # Physical flux LEFT
        D_L = rhoL * WL
        rho_hL = rhoL * hL
        T0rL = rho_hL * u0L * urL + presL * g4UU_0r

        # v_r = γ_{rr} v^r (covariant velocity component)
        v_rL = grr * vL

        # T^r_r using equation (27): W² ρh v_r ṽ^r + p
        Tr_rL = rho_hL * WL * WL * v_rL * vtilde_rL + presL

        F_D_L = e6p * alph * D_L * vtilde_rL
        F_Sr_L = e6p * alph * Tr_rL
        F_tau_L = e6p * (alph_sq * T0rL - alph * D_L * vtilde_rL)

        # ========== RIGHT STATE ==========
        rhoR = rho0R[m]
        vR = vrR[m]
        presR = pR[m]

        # Lorentz factor
        v2_grrR = vR * vR * grr
        if v2_grrR >= 1.0:
            v2_grrR = 1.0 - 1e-10
        WR = 1.0 / np.sqrt(1.0 - v2_grrR)

        # Enthalpy
        hR = 1.0 + epsR[m] + presR / max(rhoR, 1e-30)

        # Valencia velocity and 4-velocity
        vtilde_rR = vR - betar / alph
        u0R = WR / alph
        urR = WR * vtilde_rR

        # Characteristic speeds
        v02R = cs2R[m]
        one_minus_cs2R = 1.0 - v02R

        u0_sqR = u0R * u0R
        ur_sqR = urR * urR
        u0_urR = u0R * urR

        aR = one_minus_cs2R * u0_sqR - v02R * g4UU_00
        bR = 2.0 * v02R * g4UU_0r - 2.0 * u0_urR * one_minus_cs2R
        cR = one_minus_cs2R * ur_sqR - v02R * g4UU_rr

        detm_sqR = bR * bR - 4.0 * aR * cR
        if detm_sqR < 0.0:
            detm_sqR = 0.0
        detmR = np.sqrt(detm_sqR)

        if abs(aR) < 1e-30:
            cmR = -1.0
            cpR = 1.0
        else:
            inv_2aR = 0.5 / aR
            cp_tmpR = (-bR + detmR) * inv_2aR
            cm_tmpR = (-bR - detmR) * inv_2aR
            cmR = min(cp_tmpR, cm_tmpR)
            cpR = max(cp_tmpR, cm_tmpR)

        # Physical flux RIGHT
        D_R = rhoR * WR
        rho_hR = rhoR * hR
        T0rR = rho_hR * u0R * urR + presR * g4UU_0r

        # v_r = γ_{rr} v^r (covariant velocity component)
        v_rR = grr * vR

        # T^r_r using equation (27): W² ρh v_r ṽ^r + p
        Tr_rR = rho_hR * WR * WR * v_rR * vtilde_rR + presR

        F_D_R = e6p * alph * D_R * vtilde_rR
        F_Sr_R = e6p * alph * Tr_rR
        F_tau_R = e6p * (alph_sq * T0rR - alph * D_R * vtilde_rR)

        # ========== WAVE SPEEDS ==========
        cmax = max(0.0, max(cpL, cpR))
        cmin = -min(0.0, min(cmL, cmR))

        lam_minus = -cmin
        lam_plus = cmax

        # Entropy fix
        lam_minus = min(lam_minus, -ENTROPY_DELTA)
        lam_plus = max(lam_plus, ENTROPY_DELTA)

        if lam_minus > lam_plus:
            tmp = lam_minus
            lam_minus = lam_plus
            lam_plus = tmp

        # ========== HLL FLUX ==========
        inv_denom = 1.0 / (lam_plus - lam_minus)
        lp_lm = lam_plus * lam_minus

        dD = DR[m] - DL[m]
        dSr = SrR[m] - SrL[m]
        dTau = tauR[m] - tauL[m]

        F_out[m, 0] = (lam_plus * F_D_L - lam_minus * F_D_R + lp_lm * dD) * inv_denom
        F_out[m, 1] = (lam_plus * F_Sr_L - lam_minus * F_Sr_R + lp_lm * dSr) * inv_denom
        F_out[m, 2] = (lam_plus * F_tau_L - lam_minus * F_tau_R + lp_lm * dTau) * inv_denom


# ==============================================================================
# HLL RIEMANN SOLVER
# ==============================================================================

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
                                     alpha_batch, e6phi_batch, gamma_rr_batch, beta_r_batch)
        FR = physical_flux_1d_kernel(rho0R, vrR, pR, WR, hR,
                                     alpha_batch, e6phi_batch, gamma_rr_batch, beta_r_batch)

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

    def solve_batch_fused(self, UL_batch, UR_batch, primL_batch, primR_batch,
                          geom: GeometryState, eos):
        """
        Ultra-fast HLL solver using fully fused kernel.

        Computes everything in a single parallel loop, eliminating
        all intermediate arrays and kernel launch overhead.

        Approximately 2x faster than solve_batch for typical problem sizes.
        """
        M = len(UL_batch)

        # Extract geometry
        gamma_rr = np.ascontiguousarray(geom.gamma_rr, dtype=np.float64)
        alpha = np.ascontiguousarray(geom.alpha, dtype=np.float64)
        beta_r = np.ascontiguousarray(geom.beta_r, dtype=np.float64)
        e6phi = np.ascontiguousarray(geom.e6phi, dtype=np.float64)

        # Ensure contiguous float64
        UL_batch = np.ascontiguousarray(UL_batch, dtype=np.float64)
        UR_batch = np.ascontiguousarray(UR_batch, dtype=np.float64)
        primL_batch = np.ascontiguousarray(primL_batch, dtype=np.float64)
        primR_batch = np.ascontiguousarray(primR_batch, dtype=np.float64)

        # Unpack primitives
        rho0L = primL_batch[:, 0]
        vrL = primL_batch[:, 1]
        pL = primL_batch[:, 2]
        rho0R = primR_batch[:, 0]
        vrR = primR_batch[:, 1]
        pR = primR_batch[:, 2]

        # Apply floors
        pL = np.maximum(pL, self.atmosphere.p_floor)
        pR = np.maximum(pR, self.atmosphere.p_floor)
        vrL = np.clip(vrL, -self.atmosphere.v_max, self.atmosphere.v_max)
        vrR = np.clip(vrR, -self.atmosphere.v_max, self.atmosphere.v_max)

        # EOS quantities (done in Python, not JIT-compatible)
        epsL = np.maximum(eos.eps_from_rho_p(rho0L, pL), 1e-15)
        epsR = np.maximum(eos.eps_from_rho_p(rho0R, pR), 1e-15)
        cs2L = np.clip(eos.sound_speed_squared(rho0L, pL, epsL), 0.0, 1.0 - 1e-12)
        cs2R = np.clip(eos.sound_speed_squared(rho0R, pR, epsR), 0.0, 1.0 - 1e-12)

        # Ensure all inputs are contiguous
        rho0L = np.ascontiguousarray(rho0L)
        vrL = np.ascontiguousarray(vrL)
        pL = np.ascontiguousarray(pL)
        epsL = np.ascontiguousarray(epsL)
        rho0R = np.ascontiguousarray(rho0R)
        vrR = np.ascontiguousarray(vrR)
        pR = np.ascontiguousarray(pR)
        epsR = np.ascontiguousarray(epsR)
        cs2L = np.ascontiguousarray(cs2L)
        cs2R = np.ascontiguousarray(cs2R)

        DL = np.ascontiguousarray(UL_batch[:, 0])
        SrL = np.ascontiguousarray(UL_batch[:, 1])
        tauL = np.ascontiguousarray(UL_batch[:, 2])
        DR = np.ascontiguousarray(UR_batch[:, 0])
        SrR = np.ascontiguousarray(UR_batch[:, 1])
        tauR = np.ascontiguousarray(UR_batch[:, 2])

        # Output array
        F_out = np.empty((M, 3), dtype=np.float64)

        # Call fused kernel
        hll_solver_fused_kernel(
            rho0L, vrL, pL, epsL, rho0R, vrR, pR, epsR,
            DL, SrL, tauL, DR, SrR, tauR,
            cs2L, cs2R,
            alpha, beta_r, gamma_rr, e6phi,
            F_out
        )

        return F_out


# ==============================================================================
# LLF (LOCAL LAX-FRIEDRICHS) RIEMANN SOLVER
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True, parallel=True)
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

    for m in prange(M):
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
                                     alpha_batch, e6phi_batch, gamma_rr_batch, beta_r_batch)
        FR = physical_flux_1d_kernel(rho0R, vrR, pR, WR, hR,
                                     alpha_batch, e6phi_batch, gamma_rr_batch, beta_r_batch)

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


# ==============================================================================
# HLLC RIEMANN SOLVER
# ==============================================================================

class HLLCRiemannSolver:
    """
    HLLC Riemann solver using tetrad transformation for GRHD.
    
    Following Lam & Shibata (2025, arXiv:2502.03223) and 
    White, Stone & Gammie (2016, ApJS 225, 22).
    
    Algorithm:
    1. Transform primitives to local orthonormal tetrad frame
    2. Compute HLLC flux in local Minkowski spacetime
    3. Transform flux back to global Valencia coordinates
    
    Key feature: Resolves contact discontinuities (less diffusive than HLL/LLF).
    
    Fully optimized with Numba JIT kernels for CPU performance.
    """
    
    def __init__(self, name: str = "HLLC", atmosphere=None):
        self.name = name
        self.atmosphere = atmosphere
        
    def solve_batch(self, UL_batch, UR_batch, primL_batch, primR_batch,
                    geom: GeometryState, eos):
        """
        Compute HLLC fluxes for multiple interfaces using tetrad transformation.
        
        All inputs are arrays of length M. Returns (M, 3) flux array.
        
        Steps:
        1. Extract geometry and primitives
        2. Transform to local frame
        3. Compute HLLC flux in local frame
        4. Transform flux back to global frame
        """
        M = len(UL_batch)
        
        # Extract geometry components (ensure contiguous float64)
        gamma_rr = np.ascontiguousarray(geom.gamma_rr, dtype=np.float64)
        alpha = np.ascontiguousarray(geom.alpha, dtype=np.float64)
        beta_r = np.ascontiguousarray(geom.beta_r, dtype=np.float64)
        e6phi = np.ascontiguousarray(geom.e6phi, dtype=np.float64)
        
        # For flux transformation
        e4phi = np.ascontiguousarray(geom.e4phi, dtype=np.float64)
        sqrt_hatg = np.ascontiguousarray(geom.sqrtdetgam, dtype=np.float64)
        
        # Reference metric component (flat reference: γ̄_rr = 1)
        sqrt_bargamma_rr = np.ones(M, dtype=np.float64)
        
        # Convert inputs to contiguous float64 arrays
        primL_batch = np.ascontiguousarray(primL_batch, dtype=np.float64)
        primR_batch = np.ascontiguousarray(primR_batch, dtype=np.float64)
        
        # Unpack primitives
        rho0L = primL_batch[:, 0]
        vrL = primL_batch[:, 1]
        pL = primL_batch[:, 2]
        
        rho0R = primR_batch[:, 0]
        vrR = primR_batch[:, 1]
        pR = primR_batch[:, 2]
        
        # Apply floors
        pL = np.maximum(pL, self.atmosphere.p_floor)
        pR = np.maximum(pR, self.atmosphere.p_floor)
        vrL = np.clip(vrL, -self.atmosphere.v_max, self.atmosphere.v_max)
        vrR = np.clip(vrR, -self.atmosphere.v_max, self.atmosphere.v_max)
        
        gamma_rr = np.maximum(gamma_rr, 1e-30)
        alpha = np.maximum(alpha, 1e-30)
        
        # Compute EOS quantities
        epsL = np.maximum(eos.eps_from_rho_p(rho0L, pL), 1e-15)
        epsR = np.maximum(eos.eps_from_rho_p(rho0R, pR), 1e-15)
        cs2L = np.clip(eos.sound_speed_squared(rho0L, pL, epsL), 0.0, 1.0 - 1e-12)
        cs2R = np.clip(eos.sound_speed_squared(rho0R, pR, epsR), 0.0, 1.0 - 1e-12)
        
        # Ensure contiguous
        rho0L = np.ascontiguousarray(rho0L)
        vrL = np.ascontiguousarray(vrL)
        pL = np.ascontiguousarray(pL)
        epsL = np.ascontiguousarray(epsL)
        cs2L = np.ascontiguousarray(cs2L)
        
        rho0R = np.ascontiguousarray(rho0R)
        vrR = np.ascontiguousarray(vrR)
        pR = np.ascontiguousarray(pR)
        epsR = np.ascontiguousarray(epsR)
        cs2R = np.ascontiguousarray(cs2R)
        
        # Transform LEFT state to local frame
        (v_loc_L, W_loc_L, h_L, D_loc_L, S_loc_L, E_loc_L,
         F_D_L, F_S_L, F_E_L, lam_L, lam_R_L) = transform_to_local_frame_kernel(
            vrL, rho0L, pL, epsL, cs2L, alpha, beta_r, gamma_rr
        )
        
        # Transform RIGHT state to local frame
        (v_loc_R, W_loc_R, h_R, D_loc_R, S_loc_R, E_loc_R,
         F_D_R, F_S_R, F_E_R, lam_L_R, lam_R_R) = transform_to_local_frame_kernel(
            vrR, rho0R, pR, epsR, cs2R, alpha, beta_r, gamma_rr
        )
        
        # Take global wave speeds (max/min of left and right)
        lam_minus = np.minimum(lam_L, lam_L_R)
        lam_plus = np.maximum(lam_R_L, lam_R_R)
        
        # Interface velocity in local frame: v^(r̂)_interface = β^r/(α√γ_rr)
        sqrt_grr = np.sqrt(gamma_rr)
        v_interface = beta_r / (alpha * sqrt_grr)
        v_interface = np.ascontiguousarray(v_interface)
        
        # Ensure all inputs to HLLC kernel are contiguous
        D_loc_L = np.ascontiguousarray(D_loc_L)
        S_loc_L = np.ascontiguousarray(S_loc_L)
        E_loc_L = np.ascontiguousarray(E_loc_L)
        D_loc_R = np.ascontiguousarray(D_loc_R)
        S_loc_R = np.ascontiguousarray(S_loc_R)
        E_loc_R = np.ascontiguousarray(E_loc_R)
        
        F_D_L = np.ascontiguousarray(F_D_L)
        F_S_L = np.ascontiguousarray(F_S_L)
        F_E_L = np.ascontiguousarray(F_E_L)
        F_D_R = np.ascontiguousarray(F_D_R)
        F_S_R = np.ascontiguousarray(F_S_R)
        F_E_R = np.ascontiguousarray(F_E_R)
        
        v_loc_L = np.ascontiguousarray(v_loc_L)
        v_loc_R = np.ascontiguousarray(v_loc_R)
        pL = np.ascontiguousarray(pL)
        pR = np.ascontiguousarray(pR)
        lam_minus = np.ascontiguousarray(lam_minus)
        lam_plus = np.ascontiguousarray(lam_plus)
        
        # Compute HLLC flux in local frame
        F_local = hllc_flux_kernel(
            D_loc_L, S_loc_L, E_loc_L, D_loc_R, S_loc_R, E_loc_R,
            F_D_L, F_S_L, F_E_L, F_D_R, F_S_R, F_E_R,
            v_loc_L, v_loc_R, pL, pR,
            lam_minus, lam_plus, v_interface
        )
        
        # Transform flux back to global Valencia coordinates
        F_global = transform_flux_to_global_kernel(
            F_local, alpha, e4phi, sqrt_hatg, sqrt_bargamma_rr
        )
        
        return F_global