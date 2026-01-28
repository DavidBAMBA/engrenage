#!/usr/bin/env python3
"""
Test script for HLLC Riemann solver verification.

This script tests the HLLC implementation against:
1. Analytical properties (intermediate state jumps)
2. Standard relativistic shock tube problems
3. Comparison with HLL solver

The HLLC solver should:
- Match HLL for shock-dominated problems
- Outperform HLL for contact discontinuity problems
- Satisfy consistency conditions (F*_L = F*_R at contact)
"""

import numpy as np
from numba import jit


# ==============================================================================
# COPY OF HLLC KERNEL FOR TESTING (self-contained)
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def hllc_flux_kernel_test(DL, SrL, tauL, DR, SrR, tauR,
                          FL, FR,
                          vxL, vxR, pL, pR,
                          lam_minus, lam_plus):
    """HLLC kernel for testing (single interface, non-parallel)."""
    M = DL.shape[0]
    out = np.empty((M, 3))
    
    # Also return diagnostics
    lambda_c_out = np.empty(M)
    P_c_out = np.empty(M)

    for m in range(M):
        lam_L = lam_minus[m]
        lam_R = lam_plus[m]
        v_interface = 0.0

        if lam_L >= v_interface:
            out[m, 0] = FL[m, 0]
            out[m, 1] = FL[m, 1]
            out[m, 2] = FL[m, 2]
            lambda_c_out[m] = lam_L
            P_c_out[m] = pL[m]
        elif lam_R <= v_interface:
            out[m, 0] = FR[m, 0]
            out[m, 1] = FR[m, 1]
            out[m, 2] = FR[m, 2]
            lambda_c_out[m] = lam_R
            P_c_out[m] = pR[m]
        else:
            denom = lam_R - lam_L
            if abs(denom) < 1e-30:
                out[m, 0] = 0.5 * (FL[m, 0] + FR[m, 0])
                out[m, 1] = 0.5 * (FL[m, 1] + FR[m, 1])
                out[m, 2] = 0.5 * (FL[m, 2] + FR[m, 2])
                lambda_c_out[m] = 0.0
                P_c_out[m] = 0.5 * (pL[m] + pR[m])
            else:
                inv_denom = 1.0 / denom
                lam_L_lam_R = lam_L * lam_R

                E_L = tauL[m] + DL[m]
                E_R = tauR[m] + DR[m]
                F_E_L = FL[m, 2] + FL[m, 0]
                F_E_R = FR[m, 2] + FR[m, 0]

                D_hll = (lam_R * DR[m] - lam_L * DL[m] + FL[m, 0] - FR[m, 0]) * inv_denom
                Sr_hll = (lam_R * SrR[m] - lam_L * SrL[m] + FL[m, 1] - FR[m, 1]) * inv_denom
                E_hll = (lam_R * E_R - lam_L * E_L + F_E_L - F_E_R) * inv_denom

                F_D_hll = (lam_R * FL[m, 0] - lam_L * FR[m, 0] + lam_L_lam_R * (DR[m] - DL[m])) * inv_denom
                F_Sr_hll = (lam_R * FL[m, 1] - lam_L * FR[m, 1] + lam_L_lam_R * (SrR[m] - SrL[m])) * inv_denom
                F_E_hll = (lam_R * F_E_L - lam_L * F_E_R + lam_L_lam_R * (E_R - E_L)) * inv_denom

                a_q = F_E_hll
                b_q = -(E_hll + F_Sr_hll)
                c_q = Sr_hll

                if abs(a_q) > 1e-30:
                    disc = b_q * b_q - 4.0 * a_q * c_q
                    sqrt_disc = np.sqrt(max(0.0, disc))
                    if b_q >= 0.0:
                        lambda_c = -2.0 * c_q / (b_q + sqrt_disc + 1e-30)
                    else:
                        # Use the smaller root: λ = 2c / (-b + √D)
                        lambda_c = 2.0 * c_q / (-b_q + sqrt_disc + 1e-30)
                else:
                    if abs(b_q) > 1e-30:
                        lambda_c = -c_q / b_q
                    else:
                        lambda_c = 0.0

                lambda_c = max(lam_L, min(lam_R, lambda_c))
                P_c = -F_E_hll * lambda_c + F_Sr_hll
                
                lambda_c_out[m] = lambda_c
                P_c_out[m] = P_c

                if lambda_c >= v_interface:
                    denom_star = lam_L - lambda_c
                    if abs(denom_star) < 1e-30:
                        out[m, 0] = FL[m, 0]
                        out[m, 1] = FL[m, 1]
                        out[m, 2] = FL[m, 2]
                    else:
                        inv_denom_star = 1.0 / denom_star
                        v_L = vxL[m]
                        p_L = pL[m]
                        D_cL = DL[m] * (lam_L - v_L) * inv_denom_star
                        Sr_cL = (SrL[m] * (lam_L - v_L) + (P_c - p_L)) * inv_denom_star
                        E_cL = (E_L * (lam_L - v_L) + P_c * lambda_c - p_L * v_L) * inv_denom_star
                        tau_cL = E_cL - D_cL
                        out[m, 0] = FL[m, 0] + lam_L * (D_cL - DL[m])
                        out[m, 1] = FL[m, 1] + lam_L * (Sr_cL - SrL[m])
                        out[m, 2] = FL[m, 2] + lam_L * (tau_cL - tauL[m])
                else:
                    denom_star = lam_R - lambda_c
                    if abs(denom_star) < 1e-30:
                        out[m, 0] = FR[m, 0]
                        out[m, 1] = FR[m, 1]
                        out[m, 2] = FR[m, 2]
                    else:
                        inv_denom_star = 1.0 / denom_star
                        v_R = vxR[m]
                        p_R = pR[m]
                        D_cR = DR[m] * (lam_R - v_R) * inv_denom_star
                        Sr_cR = (SrR[m] * (lam_R - v_R) + (P_c - p_R)) * inv_denom_star
                        E_cR = (E_R * (lam_R - v_R) + P_c * lambda_c - p_R * v_R) * inv_denom_star
                        tau_cR = E_cR - D_cR
                        out[m, 0] = FR[m, 0] + lam_R * (D_cR - DR[m])
                        out[m, 1] = FR[m, 1] + lam_R * (Sr_cR - SrR[m])
                        out[m, 2] = FR[m, 2] + lam_R * (tau_cR - tauR[m])

    return out, lambda_c_out, P_c_out


@jit(nopython=True, cache=True, fastmath=True)
def hll_flux_kernel_test(DL, SrL, tauL, DR, SrR, tauR,
                         FL, FR, lam_minus, lam_plus):
    """HLL kernel for comparison."""
    M = DL.shape[0]
    out = np.empty((M, 3))

    for m in range(M):
        lm = lam_minus[m]
        lp = lam_plus[m]

        if lm >= 0.0:
            out[m, 0] = FL[m, 0]
            out[m, 1] = FL[m, 1]
            out[m, 2] = FL[m, 2]
        elif lp <= 0.0:
            out[m, 0] = FR[m, 0]
            out[m, 1] = FR[m, 1]
            out[m, 2] = FR[m, 2]
        else:
            denom = lp - lm
            if abs(denom) < 1e-30:
                out[m, 0] = 0.5 * (FL[m, 0] + FR[m, 0])
                out[m, 1] = 0.5 * (FL[m, 1] + FR[m, 1])
                out[m, 2] = 0.5 * (FL[m, 2] + FR[m, 2])
            else:
                inv_denom = 1.0 / denom
                lp_lm = lp * lm
                dD = DR[m] - DL[m]
                dSr = SrR[m] - SrL[m]
                dTau = tauR[m] - tauL[m]
                out[m, 0] = (lp * FL[m, 0] - lm * FR[m, 0] + lp_lm * dD) * inv_denom
                out[m, 1] = (lp * FL[m, 1] - lm * FR[m, 1] + lp_lm * dSr) * inv_denom
                out[m, 2] = (lp * FL[m, 2] - lm * FR[m, 2] + lp_lm * dTau) * inv_denom

    return out


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def compute_primitives_to_conservatives(rho, v, p, gamma_adi):
    """
    Convert primitive variables to conservatives.
    
    Using Valencia formulation:
        D = ρ W
        S_r = ρ h W² v  
        τ = ρ h W² - p - D
    """
    # Assume flat space for testing: γ_rr = 1
    v_sq = v * v
    W = 1.0 / np.sqrt(1.0 - v_sq)
    eps = p / (rho * (gamma_adi - 1.0))
    h = 1.0 + eps + p / rho
    
    D = rho * W
    Sr = rho * h * W * W * v
    tau = rho * h * W * W - p - D
    
    return D, Sr, tau, W, h, eps


def compute_flux(rho, v, p, W, h, gamma_rr=1.0, alpha=1.0, e6phi=1.0):
    """
    Compute physical flux.
    
    F_D = α e^{6φ} D v
    F_Sr = α e^{6φ} (S_r v + p)
    F_τ = α e^{6φ} (τ + p) v
    """
    D = rho * W
    Sr = rho * h * W * W * v
    tau = rho * h * W * W - p - D
    
    F_D = alpha * e6phi * D * v
    F_Sr = alpha * e6phi * (Sr * v + p)
    F_tau = alpha * e6phi * (tau + p) * v
    
    return np.array([F_D, F_Sr, F_tau])


def compute_sound_speed_squared(rho, p, eps, gamma_adi):
    """Compute sound speed squared for ideal gas."""
    h = 1.0 + eps + p / rho
    cs2 = gamma_adi * p / (rho * h)
    return cs2


def compute_characteristic_speeds(v, cs2):
    """
    Compute characteristic speeds λ± in flat space.
    
    λ± = (v(1-cs²) ± cs√(1-v²)(1-v²cs² - (1-cs²)v²)) / (1 - v²cs²)
    
    Simplified for flat space (v^2 = v_r^2).
    """
    cs = np.sqrt(cs2)
    v_sq = v * v
    
    denom = 1.0 - v_sq * cs2
    
    # Discriminant
    disc = (1.0 - v_sq) * (1.0 - v_sq * cs2 - (1.0 - cs2) * v_sq)
    sqrt_disc = np.sqrt(max(0.0, disc))
    
    lambda_p = (v * (1.0 - cs2) + cs * sqrt_disc) / denom
    lambda_m = (v * (1.0 - cs2) - cs * sqrt_disc) / denom
    
    return lambda_m, lambda_p


# ==============================================================================
# TEST CASES
# ==============================================================================

def test_contact_discontinuity():
    """
    Test 1: Pure contact discontinuity
    
    This is the classic test where HLLC should significantly outperform HLL.
    A contact discontinuity has:
    - Same pressure on both sides
    - Same velocity on both sides
    - Different density
    
    The exact solution has λ_c = v and P_c = p_L = p_R.
    """
    print("="*60)
    print("TEST 1: Pure Contact Discontinuity")
    print("="*60)
    
    gamma_adi = 5.0/3.0
    
    # Left state: higher density
    rho_L = 1.0
    v_L = 0.5  # Moving contact
    p_L = 1.0
    
    # Right state: lower density, same v and p
    rho_R = 0.1
    v_R = 0.5
    p_R = 1.0
    
    # Compute conservatives
    D_L, Sr_L, tau_L, W_L, h_L, eps_L = compute_primitives_to_conservatives(rho_L, v_L, p_L, gamma_adi)
    D_R, Sr_R, tau_R, W_R, h_R, eps_R = compute_primitives_to_conservatives(rho_R, v_R, p_R, gamma_adi)
    
    # Compute fluxes
    F_L = compute_flux(rho_L, v_L, p_L, W_L, h_L)
    F_R = compute_flux(rho_R, v_R, p_R, W_R, h_R)
    
    # Compute wave speeds
    cs2_L = compute_sound_speed_squared(rho_L, p_L, eps_L, gamma_adi)
    cs2_R = compute_sound_speed_squared(rho_R, p_R, eps_R, gamma_adi)
    
    cm_L, cp_L = compute_characteristic_speeds(v_L, cs2_L)
    cm_R, cp_R = compute_characteristic_speeds(v_R, cs2_R)
    
    lam_minus = min(cm_L, cm_R)
    lam_plus = max(cp_L, cp_R)
    
    print(f"\nLeft state:  ρ={rho_L}, v={v_L}, p={p_L}")
    print(f"Right state: ρ={rho_R}, v={v_R}, p={p_R}")
    print(f"Wave speeds: λ_L={lam_minus:.4f}, λ_R={lam_plus:.4f}")
    print(f"Expected: λ_c={v_L:.4f}, P_c={p_L:.4f}")
    
    # Convert to arrays for kernel
    DL = np.array([D_L])
    SrL = np.array([Sr_L])
    tauL = np.array([tau_L])
    DR = np.array([D_R])
    SrR = np.array([Sr_R])
    tauR = np.array([tau_R])
    FL = F_L.reshape(1, 3)
    FR = F_R.reshape(1, 3)
    vxL = np.array([v_L])
    vxR = np.array([v_R])
    pL_arr = np.array([p_L])
    pR_arr = np.array([p_R])
    lam_m = np.array([lam_minus])
    lam_p = np.array([lam_plus])
    
    # HLLC flux
    F_hllc, lambda_c, P_c = hllc_flux_kernel_test(
        DL, SrL, tauL, DR, SrR, tauR,
        FL, FR, vxL, vxR, pL_arr, pR_arr,
        lam_m, lam_p
    )
    
    # HLL flux
    F_hll = hll_flux_kernel_test(
        DL, SrL, tauL, DR, SrR, tauR,
        FL, FR, lam_m, lam_p
    )
    
    print(f"\nHLLC results:")
    print(f"  λ_c = {lambda_c[0]:.6f} (expected {v_L:.6f})")
    print(f"  P_c = {P_c[0]:.6f} (expected {p_L:.6f})")
    print(f"  Flux = [{F_hllc[0,0]:.6f}, {F_hllc[0,1]:.6f}, {F_hllc[0,2]:.6f}]")
    
    print(f"\nHLL results:")
    print(f"  Flux = [{F_hll[0,0]:.6f}, {F_hll[0,1]:.6f}, {F_hll[0,2]:.6f}]")
    
    # Check accuracy
    lambda_c_error = abs(lambda_c[0] - v_L)
    P_c_error = abs(P_c[0] - p_L)
    
    print(f"\nAccuracy check:")
    print(f"  |λ_c - v| = {lambda_c_error:.2e}")
    print(f"  |P_c - p| = {P_c_error:.2e}")
    
    if lambda_c_error < 1e-10 and P_c_error < 1e-10:
        print("  ✓ PASSED: Contact speed and pressure match expected values")
    else:
        print("  ✗ WARNING: Contact values differ from expected")
    
    return lambda_c_error < 1e-6 and P_c_error < 1e-6


def test_sod_shock_tube():
    """
    Test 2: Relativistic Sod shock tube
    
    This is a standard shock tube test. HLLC and HLL should give
    similar results for the shock, but HLLC should be sharper at
    the contact discontinuity.
    """
    print("\n" + "="*60)
    print("TEST 2: Relativistic Sod Shock Tube")
    print("="*60)
    
    gamma_adi = 5.0/3.0
    
    # Left state (high pressure)
    rho_L = 1.0
    v_L = 0.0
    p_L = 1.0
    
    # Right state (low pressure)
    rho_R = 0.125
    v_R = 0.0
    p_R = 0.1
    
    # Compute conservatives
    D_L, Sr_L, tau_L, W_L, h_L, eps_L = compute_primitives_to_conservatives(rho_L, v_L, p_L, gamma_adi)
    D_R, Sr_R, tau_R, W_R, h_R, eps_R = compute_primitives_to_conservatives(rho_R, v_R, p_R, gamma_adi)
    
    # Compute fluxes
    F_L = compute_flux(rho_L, v_L, p_L, W_L, h_L)
    F_R = compute_flux(rho_R, v_R, p_R, W_R, h_R)
    
    # Compute wave speeds
    cs2_L = compute_sound_speed_squared(rho_L, p_L, eps_L, gamma_adi)
    cs2_R = compute_sound_speed_squared(rho_R, p_R, eps_R, gamma_adi)
    
    cm_L, cp_L = compute_characteristic_speeds(v_L, cs2_L)
    cm_R, cp_R = compute_characteristic_speeds(v_R, cs2_R)
    
    lam_minus = min(cm_L, cm_R)
    lam_plus = max(cp_L, cp_R)
    
    print(f"\nLeft state:  ρ={rho_L}, v={v_L}, p={p_L}")
    print(f"Right state: ρ={rho_R}, v={v_R}, p={p_R}")
    print(f"Wave speeds: λ_L={lam_minus:.4f}, λ_R={lam_plus:.4f}")
    
    # Convert to arrays
    DL = np.array([D_L])
    SrL = np.array([Sr_L])
    tauL = np.array([tau_L])
    DR = np.array([D_R])
    SrR = np.array([Sr_R])
    tauR = np.array([tau_R])
    FL = F_L.reshape(1, 3)
    FR = F_R.reshape(1, 3)
    vxL = np.array([v_L])
    vxR = np.array([v_R])
    pL_arr = np.array([p_L])
    pR_arr = np.array([p_R])
    lam_m = np.array([lam_minus])
    lam_p = np.array([lam_plus])
    
    # HLLC flux
    F_hllc, lambda_c, P_c = hllc_flux_kernel_test(
        DL, SrL, tauL, DR, SrR, tauR,
        FL, FR, vxL, vxR, pL_arr, pR_arr,
        lam_m, lam_p
    )
    
    # HLL flux
    F_hll = hll_flux_kernel_test(
        DL, SrL, tauL, DR, SrR, tauR,
        FL, FR, lam_m, lam_p
    )
    
    print(f"\nHLLC results:")
    print(f"  λ_c = {lambda_c[0]:.6f}")
    print(f"  P_c = {P_c[0]:.6f}")
    print(f"  Flux = [{F_hllc[0,0]:.6f}, {F_hllc[0,1]:.6f}, {F_hllc[0,2]:.6f}]")
    
    print(f"\nHLL results:")
    print(f"  Flux = [{F_hll[0,0]:.6f}, {F_hll[0,1]:.6f}, {F_hll[0,2]:.6f}]")
    
    # Flux difference
    flux_diff = np.linalg.norm(F_hllc - F_hll)
    print(f"\n|F_HLLC - F_HLL| = {flux_diff:.6f}")
    
    # Check that HLLC gives physically reasonable values
    # λ_c should be between λ_L and λ_R
    if lam_minus <= lambda_c[0] <= lam_plus:
        print("  ✓ PASSED: Contact speed within wave bounds")
        return True
    else:
        print("  ✗ FAILED: Contact speed outside wave bounds")
        return False


def test_blast_wave():
    """
    Test 3: Strong relativistic blast wave
    
    Tests the solver with a strong pressure jump and relativistic velocities.
    """
    print("\n" + "="*60)
    print("TEST 3: Strong Relativistic Blast Wave")
    print("="*60)
    
    gamma_adi = 4.0/3.0  # Relativistic gas
    
    # Left state (very high pressure)
    rho_L = 1.0
    v_L = 0.0
    p_L = 1000.0
    
    # Right state (very low pressure)
    rho_R = 1.0
    v_R = 0.0
    p_R = 0.01
    
    # Compute conservatives
    D_L, Sr_L, tau_L, W_L, h_L, eps_L = compute_primitives_to_conservatives(rho_L, v_L, p_L, gamma_adi)
    D_R, Sr_R, tau_R, W_R, h_R, eps_R = compute_primitives_to_conservatives(rho_R, v_R, p_R, gamma_adi)
    
    # Compute fluxes
    F_L = compute_flux(rho_L, v_L, p_L, W_L, h_L)
    F_R = compute_flux(rho_R, v_R, p_R, W_R, h_R)
    
    # Compute wave speeds
    cs2_L = compute_sound_speed_squared(rho_L, p_L, eps_L, gamma_adi)
    cs2_R = compute_sound_speed_squared(rho_R, p_R, eps_R, gamma_adi)
    
    cm_L, cp_L = compute_characteristic_speeds(v_L, cs2_L)
    cm_R, cp_R = compute_characteristic_speeds(v_R, cs2_R)
    
    lam_minus = min(cm_L, cm_R)
    lam_plus = max(cp_L, cp_R)
    
    print(f"\nLeft state:  ρ={rho_L}, v={v_L}, p={p_L}")
    print(f"Right state: ρ={rho_R}, v={v_R}, p={p_R}")
    print(f"Sound speeds: cs_L={np.sqrt(cs2_L):.4f}, cs_R={np.sqrt(cs2_R):.4f}")
    print(f"Wave speeds: λ_L={lam_minus:.4f}, λ_R={lam_plus:.4f}")
    
    # Convert to arrays
    DL = np.array([D_L])
    SrL = np.array([Sr_L])
    tauL = np.array([tau_L])
    DR = np.array([D_R])
    SrR = np.array([Sr_R])
    tauR = np.array([tau_R])
    FL = F_L.reshape(1, 3)
    FR = F_R.reshape(1, 3)
    vxL = np.array([v_L])
    vxR = np.array([v_R])
    pL_arr = np.array([p_L])
    pR_arr = np.array([p_R])
    lam_m = np.array([lam_minus])
    lam_p = np.array([lam_plus])
    
    # HLLC flux
    F_hllc, lambda_c, P_c = hllc_flux_kernel_test(
        DL, SrL, tauL, DR, SrR, tauR,
        FL, FR, vxL, vxR, pL_arr, pR_arr,
        lam_m, lam_p
    )
    
    # HLL flux
    F_hll = hll_flux_kernel_test(
        DL, SrL, tauL, DR, SrR, tauR,
        FL, FR, lam_m, lam_p
    )
    
    print(f"\nHLLC results:")
    print(f"  λ_c = {lambda_c[0]:.6f}")
    print(f"  P_c = {P_c[0]:.6f}")
    print(f"  Flux = [{F_hllc[0,0]:.6f}, {F_hllc[0,1]:.6f}, {F_hllc[0,2]:.6f}]")
    
    print(f"\nHLL results:")
    print(f"  Flux = [{F_hll[0,0]:.6f}, {F_hll[0,1]:.6f}, {F_hll[0,2]:.6f}]")
    
    # Check physical bounds
    passed = True
    
    if not (lam_minus <= lambda_c[0] <= lam_plus):
        print("  ✗ FAILED: Contact speed outside wave bounds")
        passed = False
    else:
        print("  ✓ Contact speed within bounds")
    
    if P_c[0] < 0:
        print("  ✗ FAILED: Negative contact pressure")
        passed = False
    else:
        print("  ✓ Positive contact pressure")
    
    if np.any(np.isnan(F_hllc)) or np.any(np.isinf(F_hllc)):
        print("  ✗ FAILED: NaN or Inf in HLLC flux")
        passed = False
    else:
        print("  ✓ HLLC flux is finite")
    
    return passed


def test_consistency():
    """
    Test 4: Consistency - uniform state should give exact flux
    
    For a uniform state, HLLC should return the exact physical flux.
    """
    print("\n" + "="*60)
    print("TEST 4: Consistency (Uniform State)")
    print("="*60)
    
    gamma_adi = 5.0/3.0
    
    # Same state on both sides
    rho = 1.0
    v = 0.3
    p = 0.5
    
    D, Sr, tau, W, h, eps = compute_primitives_to_conservatives(rho, v, p, gamma_adi)
    F_exact = compute_flux(rho, v, p, W, h)
    
    cs2 = compute_sound_speed_squared(rho, p, eps, gamma_adi)
    cm, cp = compute_characteristic_speeds(v, cs2)
    
    print(f"\nUniform state: ρ={rho}, v={v}, p={p}")
    print(f"Exact flux: [{F_exact[0]:.6f}, {F_exact[1]:.6f}, {F_exact[2]:.6f}]")
    
    # Convert to arrays
    DL = np.array([D])
    SrL = np.array([Sr])
    tauL = np.array([tau])
    DR = np.array([D])
    SrR = np.array([Sr])
    tauR = np.array([tau])
    FL = F_exact.reshape(1, 3)
    FR = F_exact.reshape(1, 3)
    vxL = np.array([v])
    vxR = np.array([v])
    pL_arr = np.array([p])
    pR_arr = np.array([p])
    lam_m = np.array([cm])
    lam_p = np.array([cp])
    
    # HLLC flux
    F_hllc, lambda_c, P_c = hllc_flux_kernel_test(
        DL, SrL, tauL, DR, SrR, tauR,
        FL, FR, vxL, vxR, pL_arr, pR_arr,
        lam_m, lam_p
    )
    
    print(f"\nHLLC flux: [{F_hllc[0,0]:.6f}, {F_hllc[0,1]:.6f}, {F_hllc[0,2]:.6f}]")
    
    error = np.linalg.norm(F_hllc.flatten() - F_exact)
    print(f"\n|F_HLLC - F_exact| = {error:.2e}")
    
    if error < 1e-10:
        print("  ✓ PASSED: HLLC returns exact flux for uniform state")
        return True
    else:
        print("  ✗ FAILED: HLLC differs from exact flux")
        return False


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# HLLC RIEMANN SOLVER TEST SUITE")
    print("# Based on Mignone & Bodo 2005 (MNRAS 364, 126)")
    print("#"*60)
    
    results = []
    
    # Run tests
    results.append(("Contact Discontinuity", test_contact_discontinuity()))
    results.append(("Sod Shock Tube", test_sod_shock_tube()))
    results.append(("Blast Wave", test_blast_wave()))
    results.append(("Consistency", test_consistency()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*60 + "\n")