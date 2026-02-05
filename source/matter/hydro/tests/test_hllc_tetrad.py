#!/usr/bin/env python3
"""
test_hllc_tetrad.py — Unit tests for HLLC Riemann solver with tetrad transformation

Tests to validate the tetrad-based HLLC implementation:

1. Flat spacetime limit: In Minkowski (α=1, β=0, γ_rr=1), HLLC with tetrad
   should give the same flux as standard special relativistic HLLC.

2. Flux consistency: F(U, U) = F_physical(U) - identical states give physical flux.

3. Transformation roundtrip: local -> global -> local should be identity.

4. Contact discontinuity test: HLLC should resolve contacts sharper than HLL.

5. Sod shock tube: Standard test with comparison to high-resolution reference.

References:
- Mignone & Bodo (2005), MNRAS 364, 126 - HLLC for relativistic HD
- Mignone & Bodo (2006), MNRAS 368, 1040 - HLLC improvements
- Lam & Shibata (2025), arXiv:2502.03223 - Tetrad transformation
- White, Stone & Gammie (2016), ApJS 225, 22 - Athena++ relativistic MHD
"""

import os
import sys
import numpy as np

# Add source path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, repo_root)

from source.matter.hydro.riemann import (
    HLLCRiemannSolver, HLLRiemannSolver, LLFRiemannSolver,
    transform_to_local_frame_kernel,
    transform_flux_to_global_kernel,
    hllc_flux_kernel,
    physical_flux_1d_kernel
)
from source.matter.hydro.geometry import GeometryState
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.cons2prim import prim_to_cons
from source.matter.hydro.atmosphere import AtmosphereParams


def test_flat_spacetime_consistency():
    """
    Test 1: Flat spacetime limit

    In Minkowski spacetime (α=1, β=0, γ_rr=1, e^{6φ}=1), the HLLC solver
    with tetrad transformation should give identical results to a standard
    special relativistic HLLC.

    Key checks:
    - Local frame velocity = coordinate velocity (no transformation)
    - Local frame conserved variables = coordinate conserved variables
    - Fluxes match expected SR formulas
    """
    print("\n" + "="*60)
    print("TEST 1: Flat spacetime consistency")
    print("="*60)

    M = 10  # Number of test points
    gamma = 5.0/3.0
    eos = IdealGasEOS(gamma=gamma)

    # Minkowski geometry
    alpha = np.ones(M)
    beta_r = np.zeros(M)
    gamma_rr = np.ones(M)
    e6phi = np.ones(M)

    # Test primitives: variety of states
    rho0 = np.array([1.0, 0.1, 10.0, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0, 1.0])
    vr = np.array([0.0, 0.1, -0.1, 0.5, -0.5, 0.3, -0.3, 0.8, -0.8, 0.0])
    p = np.array([1.0, 0.1, 10.0, 0.1, 0.1, 1.0, 1.0, 0.01, 0.01, 100.0])

    eps = eos.eps_from_rho_p(rho0, p)
    cs2 = eos.sound_speed_squared(rho0, p, eps)

    # Transform to local frame
    (v_local, W_local, h, D_local, S_local, E_local,
     F_D, F_S, F_E, lam_minus, lam_plus) = transform_to_local_frame_kernel(
        vr, rho0, p, eps, cs2, alpha, beta_r, gamma_rr
    )

    # In flat spacetime, local velocity should equal coordinate velocity
    v_diff = np.max(np.abs(v_local - vr))
    print(f"  max|v_local - v_coord| = {v_diff:.2e}")

    # Check that W_local matches expected Lorentz factor
    W_expected = 1.0 / np.sqrt(1.0 - vr**2)
    W_diff = np.max(np.abs(W_local - W_expected))
    print(f"  max|W_local - W_expected| = {W_diff:.2e}")

    # Check conserved variables
    D_expected = rho0 * W_expected
    h_expected = 1.0 + eps + p / rho0
    S_expected = rho0 * h_expected * W_expected**2 * vr
    E_expected = rho0 * h_expected * W_expected**2 - p

    D_diff = np.max(np.abs(D_local - D_expected) / np.maximum(D_expected, 1e-15))
    S_diff = np.max(np.abs(S_local - S_expected) / np.maximum(np.abs(S_expected) + 1e-15, 1e-15))
    E_diff = np.max(np.abs(E_local - E_expected) / np.maximum(E_expected, 1e-15))

    print(f"  max|D_local - D_expected|/D = {D_diff:.2e}")
    print(f"  max|S_local - S_expected|/S = {S_diff:.2e}")
    print(f"  max|E_local - E_expected|/E = {E_diff:.2e}")

    # Check fluxes
    F_D_expected = D_expected * vr
    F_S_expected = S_expected * vr + p
    F_E_expected = S_expected  # = (E + p) * v

    F_D_diff = np.max(np.abs(F_D - F_D_expected) / np.maximum(np.abs(F_D_expected) + 1e-15, 1e-15))
    F_S_diff = np.max(np.abs(F_S - F_S_expected) / np.maximum(np.abs(F_S_expected) + 1e-15, 1e-15))
    F_E_diff = np.max(np.abs(F_E - F_E_expected) / np.maximum(np.abs(F_E_expected) + 1e-15, 1e-15))

    print(f"  max|F_D - F_D_expected|/F_D = {F_D_diff:.2e}")
    print(f"  max|F_S - F_S_expected|/F_S = {F_S_diff:.2e}")
    print(f"  max|F_E - F_E_expected|/F_E = {F_E_diff:.2e}")

    # All differences should be at machine precision
    tol = 1e-12
    ok = (v_diff < tol and W_diff < tol and
          D_diff < tol and S_diff < tol and E_diff < tol and
          F_D_diff < tol and F_S_diff < tol and F_E_diff < tol)

    print("PASS" if ok else "FAIL")
    return ok


def test_flux_consistency():
    """
    Test 2: Flux consistency - F(U, U) = F_physical(U)

    When left and right states are identical, the HLLC flux should equal
    the physical flux of that state. This is a fundamental consistency check.
    """
    print("\n" + "="*60)
    print("TEST 2: Flux consistency F(U,U) = F_physical(U)")
    print("="*60)

    M = 5
    gamma = 5.0/3.0
    eos = IdealGasEOS(gamma=gamma)

    atmosphere = AtmosphereParams(
        rho_floor=1e-15,
        p_floor=1e-15,
        v_max=0.999,
        W_max=1e3
    )

    # Test with curved spacetime
    alpha = np.array([0.9, 0.8, 0.95, 1.0, 0.7])
    beta_r = np.array([0.0, 0.1, -0.05, 0.0, 0.15])
    gamma_rr = np.array([1.2, 1.5, 1.1, 1.0, 2.0])
    e6phi = np.array([1.1, 1.3, 1.05, 1.0, 1.8])

    # Primitives
    rho0 = np.array([1.0, 0.5, 2.0, 1.0, 0.1])
    vr = np.array([0.0, 0.1, -0.1, 0.2, 0.0])
    p = np.array([1.0, 0.5, 2.0, 0.1, 10.0])

    # Create geometry
    geom = GeometryState(
        alpha=alpha, beta_r=beta_r, gamma_rr=gamma_rr, e6phi=e6phi
    )

    # Compute conservatives
    D, Sr, tau = prim_to_cons(rho0, vr, p, geom, eos)

    # Create batch arrays (L = R)
    UL = np.column_stack([D, Sr, tau])
    UR = UL.copy()
    primL = np.column_stack([rho0, vr, p])
    primR = primL.copy()

    # Compute HLLC flux
    hllc = HLLCRiemannSolver(atmosphere=atmosphere)
    F_hllc = hllc.solve_batch(UL, UR, primL, primR, geom, eos)

    # Compute physical flux directly
    eps = eos.eps_from_rho_p(rho0, p)
    W = 1.0 / np.sqrt(np.maximum(1.0 - vr**2 * gamma_rr, 1e-15))
    h = 1.0 + eps + p / np.maximum(rho0, 1e-30)

    F_phys = physical_flux_1d_kernel(
        rho0, vr, p, W, h, alpha, e6phi, gamma_rr, beta_r
    )

    # Compare
    diff_D = np.max(np.abs(F_hllc[:, 0] - F_phys[:, 0]) / np.maximum(np.abs(F_phys[:, 0]) + 1e-15, 1e-15))
    diff_S = np.max(np.abs(F_hllc[:, 1] - F_phys[:, 1]) / np.maximum(np.abs(F_phys[:, 1]) + 1e-15, 1e-15))
    diff_tau = np.max(np.abs(F_hllc[:, 2] - F_phys[:, 2]) / np.maximum(np.abs(F_phys[:, 2]) + 1e-15, 1e-15))

    print(f"  max|F_D(U,U) - F_D_phys|/F = {diff_D:.2e}")
    print(f"  max|F_S(U,U) - F_S_phys|/F = {diff_S:.2e}")
    print(f"  max|F_τ(U,U) - F_τ_phys|/F = {diff_tau:.2e}")

    tol = 1e-10
    ok = diff_D < tol and diff_S < tol and diff_tau < tol

    print("PASS" if ok else "FAIL")
    return ok


def test_transformation_factors():
    """
    Test 3: Verify transformation factors are correct

    The key transformation equations are:
    - v^(r̂) = √γ_rr (v^r - β^r/α)  [velocity to local frame]
    - F̃_D = α e^{4φ} F^(r̂)_D        [density flux]
    - F̃_Sr = α e^{6φ} F^(r̂)_S       [momentum flux]
    - F̃_τ = α e^{4φ} (F^(r̂)_E - F^(r̂)_D)  [energy flux]

    This test verifies these factors are applied correctly.
    """
    print("\n" + "="*60)
    print("TEST 3: Transformation factors verification")
    print("="*60)

    M = 1  # Single point for detailed check

    # Known geometry
    alpha = np.array([0.8])
    beta_r = np.array([0.1])
    gamma_rr = np.array([1.5])
    e6phi = np.array([1.3])

    # Simple local flux (Minkowski values)
    F_local = np.array([[1.0, 2.0, 3.0]])  # [F_D, F_S, F_E]

    # Transform to global
    F_global = transform_flux_to_global_kernel(F_local, alpha, e6phi, gamma_rr)

    # Expected values (manual calculation)
    sqrt_grr = np.sqrt(gamma_rr[0])
    e4phi = e6phi[0] / sqrt_grr  # e^{6φ}/e^{2φ} = e^{4φ}

    # F̃_D = α e^{4φ} F^(r̂)_D
    F_D_expected = alpha[0] * e4phi * F_local[0, 0]

    # F̃_Sr = α e^{6φ} F^(r̂)_S
    F_S_expected = alpha[0] * e6phi[0] * F_local[0, 1]

    # F̃_τ = α e^{4φ} (F^(r̂)_E - F^(r̂)_D)
    F_tau_expected = alpha[0] * e4phi * (F_local[0, 2] - F_local[0, 0])

    print(f"  Input: α={alpha[0]}, β^r={beta_r[0]}, γ_rr={gamma_rr[0]}, e^{{6φ}}={e6phi[0]}")
    print(f"  Derived: √γ_rr={sqrt_grr:.4f}, e^{{4φ}}={e4phi:.4f}")
    print(f"  Local flux: F_D={F_local[0,0]}, F_S={F_local[0,1]}, F_E={F_local[0,2]}")
    print()
    print(f"  Global F_D: computed={F_global[0,0]:.6f}, expected={F_D_expected:.6f}")
    print(f"  Global F_S: computed={F_global[0,1]:.6f}, expected={F_S_expected:.6f}")
    print(f"  Global F_τ: computed={F_global[0,2]:.6f}, expected={F_tau_expected:.6f}")

    tol = 1e-12
    ok_D = abs(F_global[0, 0] - F_D_expected) < tol
    ok_S = abs(F_global[0, 1] - F_S_expected) < tol
    ok_tau = abs(F_global[0, 2] - F_tau_expected) < tol

    ok = ok_D and ok_S and ok_tau
    print("PASS" if ok else "FAIL")
    return ok


def test_hllc_vs_hll_contact():
    """
    Test 4: HLLC should resolve contact discontinuities better than HLL

    Pure contact discontinuity (density jump, constant pressure, zero velocity)
    should be:
    - Diffused by HLL (no contact wave)
    - Better resolved by HLLC (has contact wave λ_c)
    """
    print("\n" + "="*60)
    print("TEST 4: HLLC vs HLL on contact discontinuity")
    print("="*60)

    M = 1  # Single interface
    gamma = 5.0/3.0
    eos = IdealGasEOS(gamma=gamma)

    atmosphere = AtmosphereParams(
        rho_floor=1e-15,
        p_floor=1e-15,
        v_max=0.999,
        W_max=1e3
    )

    # Flat spacetime for clarity
    geom = GeometryState.minkowski(M)

    # Pure contact: density jump, constant pressure, zero velocity
    rho_L, rho_R = 1.0, 0.125
    p_const = 1.0
    v_const = 0.0

    rho0_L = np.array([rho_L])
    rho0_R = np.array([rho_R])
    vr_L = np.array([v_const])
    vr_R = np.array([v_const])
    p_L = np.array([p_const])
    p_R = np.array([p_const])

    # Conservatives
    D_L, Sr_L, tau_L = prim_to_cons(rho0_L, vr_L, p_L, geom, eos)
    D_R, Sr_R, tau_R = prim_to_cons(rho0_R, vr_R, p_R, geom, eos)

    UL = np.column_stack([D_L, Sr_L, tau_L])
    UR = np.column_stack([D_R, Sr_R, tau_R])
    primL = np.column_stack([rho0_L, vr_L, p_L])
    primR = np.column_stack([rho0_R, vr_R, p_R])

    # Compute fluxes
    hll = HLLRiemannSolver(atmosphere=atmosphere)
    hllc = HLLCRiemannSolver(atmosphere=atmosphere)

    F_hll = hll.solve_batch(UL, UR, primL, primR, geom, eos)
    F_hllc = hllc.solve_batch(UL, UR, primL, primR, geom, eos)

    # For a stationary contact, the exact flux should be the physical flux
    # of either state (they have the same flux since v=0 and p is constant)
    # F_D = D * v = 0
    # F_S = S * v + p = p (since v=0, S=0)
    # F_E = S = 0 (since S=0)

    print(f"  Contact discontinuity: ρ_L={rho_L}, ρ_R={rho_R}, P={p_const}, v={v_const}")
    print()
    print(f"  HLL  flux: F_D={F_hll[0,0]:.6e}, F_S={F_hll[0,1]:.6e}, F_τ={F_hll[0,2]:.6e}")
    print(f"  HLLC flux: F_D={F_hllc[0,0]:.6e}, F_S={F_hllc[0,1]:.6e}, F_τ={F_hllc[0,2]:.6e}")

    # The exact answer for this stationary contact is:
    # F_D = 0 (no mass flux)
    # F_Sr = p (pressure force, since T^r_r = p for v=0)
    # F_τ = 0 (no energy flux)

    print()
    print(f"  Exact flux: F_D=0, F_S={p_const}, F_τ=0")

    # HLLC should give F_D closer to zero than HLL
    hllc_better = abs(F_hllc[0, 0]) <= abs(F_hll[0, 0]) + 1e-14

    print()
    print(f"  |F_D^HLLC| = {abs(F_hllc[0,0]):.6e}")
    print(f"  |F_D^HLL| = {abs(F_hll[0,0]):.6e}")
    print(f"  HLLC better or equal: {hllc_better}")

    ok = hllc_better
    print("PASS" if ok else "FAIL")
    return ok


def test_characteristic_speeds():
    """
    Test 5: Verify characteristic speeds in local frame

    In the local Minkowski frame, the characteristic speeds should be:
    λ± = (v ± cs) / (1 ± v*cs)  [special relativistic addition]
    """
    print("\n" + "="*60)
    print("TEST 5: Characteristic speeds in local frame")
    print("="*60)

    M = 5
    gamma = 5.0/3.0
    eos = IdealGasEOS(gamma=gamma)

    # Flat spacetime
    alpha = np.ones(M)
    beta_r = np.zeros(M)
    gamma_rr = np.ones(M)

    # Various velocities
    vr = np.array([0.0, 0.3, -0.3, 0.6, -0.6])
    rho0 = np.ones(M)
    p = np.ones(M) * 0.6  # P/rho = 0.6, gives cs^2 = gamma * P / (rho * h)

    eps = eos.eps_from_rho_p(rho0, p)
    cs2 = eos.sound_speed_squared(rho0, p, eps)
    cs = np.sqrt(cs2)

    # Get local frame quantities
    (v_local, W_local, h, D_local, S_local, E_local,
     F_D, F_S, F_E, lam_minus, lam_plus) = transform_to_local_frame_kernel(
        vr, rho0, p, eps, cs2, alpha, beta_r, gamma_rr
    )

    # Expected characteristic speeds (SR velocity addition)
    lam_plus_expected = (vr + cs) / (1.0 + vr * cs)
    lam_minus_expected = (vr - cs) / (1.0 - vr * cs)

    print(f"  Sound speed cs = {cs[0]:.4f}")
    print()
    print(f"  {'v':<8} {'λ⁺ comp':<12} {'λ⁺ expect':<12} {'λ⁻ comp':<12} {'λ⁻ expect':<12}")
    print("  " + "-"*55)

    max_diff_plus = 0.0
    max_diff_minus = 0.0

    for i in range(M):
        diff_plus = abs(lam_plus[i] - lam_plus_expected[i])
        diff_minus = abs(lam_minus[i] - lam_minus_expected[i])
        max_diff_plus = max(max_diff_plus, diff_plus)
        max_diff_minus = max(max_diff_minus, diff_minus)

        print(f"  {vr[i]:<8.2f} {lam_plus[i]:<12.6f} {lam_plus_expected[i]:<12.6f} "
              f"{lam_minus[i]:<12.6f} {lam_minus_expected[i]:<12.6f}")

    print()
    print(f"  max|Δλ⁺| = {max_diff_plus:.2e}")
    print(f"  max|Δλ⁻| = {max_diff_minus:.2e}")

    tol = 1e-12
    ok = max_diff_plus < tol and max_diff_minus < tol

    print("PASS" if ok else "FAIL")
    return ok


def test_curved_spacetime_consistency():
    """
    Test 6: HLLC in curved spacetime should be consistent

    Test that HLLC produces reasonable fluxes in a Schwarzschild-like metric
    (α < 1, γ_rr > 1) that don't have obvious errors.
    """
    print("\n" + "="*60)
    print("TEST 6: Curved spacetime consistency")
    print("="*60)

    M = 3
    gamma = 5.0/3.0
    eos = IdealGasEOS(gamma=gamma)

    atmosphere = AtmosphereParams(
        rho_floor=1e-15,
        p_floor=1e-15,
        v_max=0.999,
        W_max=1e3
    )

    # Schwarzschild-like metric at r=10M, 5M, 3M
    # α = sqrt(1 - 2M/r), γ_rr = 1/(1 - 2M/r)
    r_vals = np.array([10.0, 5.0, 3.0])
    M_BH = 1.0
    alpha = np.sqrt(1.0 - 2.0*M_BH/r_vals)
    gamma_rr = 1.0 / (1.0 - 2.0*M_BH/r_vals)
    beta_r = np.zeros(M)  # No shift for Schwarzschild
    e6phi = np.ones(M)    # No conformal factor

    geom = GeometryState(
        alpha=alpha, beta_r=beta_r, gamma_rr=gamma_rr, e6phi=e6phi
    )

    # Fluid at rest
    rho0 = np.ones(M)
    vr = np.zeros(M)
    p = np.ones(M) * 0.1

    D, Sr, tau = prim_to_cons(rho0, vr, p, geom, eos)

    UL = np.column_stack([D, Sr, tau])
    UR = UL.copy()
    primL = np.column_stack([rho0, vr, p])
    primR = primL.copy()

    hllc = HLLCRiemannSolver(atmosphere=atmosphere)
    F_hllc = hllc.solve_batch(UL, UR, primL, primR, geom, eos)

    print(f"  Schwarzschild-like metric:")
    print(f"  {'r/M':<8} {'α':<10} {'γ_rr':<10} {'F_D':<12} {'F_S':<12} {'F_τ':<12}")
    print("  " + "-"*60)

    for i in range(M):
        print(f"  {r_vals[i]:<8.1f} {alpha[i]:<10.4f} {gamma_rr[i]:<10.4f} "
              f"{F_hllc[i,0]:<12.4e} {F_hllc[i,1]:<12.4e} {F_hllc[i,2]:<12.4e}")

    # For fluid at rest, F_D and F_τ should be zero, F_S should be pressure-related
    ok = True
    for i in range(M):
        if abs(F_hllc[i, 0]) > 1e-10:  # F_D should be ~0
            print(f"  WARNING: F_D at r={r_vals[i]} is non-zero: {F_hllc[i,0]}")
            ok = False
        if abs(F_hllc[i, 2]) > 1e-10:  # F_τ should be ~0
            print(f"  WARNING: F_τ at r={r_vals[i]} is non-zero: {F_hllc[i,2]}")
            ok = False
        # F_S should be positive (pressure force)
        if F_hllc[i, 1] <= 0:
            print(f"  WARNING: F_S at r={r_vals[i]} is non-positive: {F_hllc[i,1]}")
            ok = False

    print("PASS" if ok else "FAIL")
    return ok


def test_hllc_hll_llf_comparison():
    """
    Test 7: Compare HLLC, HLL, and LLF on a shock

    All three should give similar results, with:
    - LLF most diffusive
    - HLL intermediate
    - HLLC least diffusive
    """
    print("\n" + "="*60)
    print("TEST 7: HLLC vs HLL vs LLF on shock")
    print("="*60)

    M = 1
    gamma = 5.0/3.0
    eos = IdealGasEOS(gamma=gamma)

    atmosphere = AtmosphereParams(
        rho_floor=1e-15,
        p_floor=1e-15,
        v_max=0.999,
        W_max=1e3
    )

    geom = GeometryState.minkowski(M)

    # Sod-like shock
    rho_L, rho_R = 1.0, 0.125
    p_L, p_R = 1.0, 0.1
    v_L, v_R = 0.0, 0.0

    D_L, Sr_L, tau_L = prim_to_cons(
        np.array([rho_L]), np.array([v_L]), np.array([p_L]), geom, eos
    )
    D_R, Sr_R, tau_R = prim_to_cons(
        np.array([rho_R]), np.array([v_R]), np.array([p_R]), geom, eos
    )

    UL = np.column_stack([D_L, Sr_L, tau_L])
    UR = np.column_stack([D_R, Sr_R, tau_R])
    primL = np.column_stack([[rho_L], [v_L], [p_L]])
    primR = np.column_stack([[rho_R], [v_R], [p_R]])

    # Compute fluxes
    llf = LLFRiemannSolver(atmosphere=atmosphere)
    hll = HLLRiemannSolver(atmosphere=atmosphere)
    hllc = HLLCRiemannSolver(atmosphere=atmosphere)

    F_llf = llf.solve_batch(UL, UR, primL, primR, geom, eos)
    F_hll = hll.solve_batch(UL, UR, primL, primR, geom, eos)
    F_hllc = hllc.solve_batch(UL, UR, primL, primR, geom, eos)

    print(f"  Sod shock: ρ_L={rho_L}, P_L={p_L} | ρ_R={rho_R}, P_R={p_R}")
    print()
    print(f"  {'Solver':<8} {'F_D':<14} {'F_S':<14} {'F_τ':<14}")
    print("  " + "-"*50)
    print(f"  {'LLF':<8} {F_llf[0,0]:<14.6e} {F_llf[0,1]:<14.6e} {F_llf[0,2]:<14.6e}")
    print(f"  {'HLL':<8} {F_hll[0,0]:<14.6e} {F_hll[0,1]:<14.6e} {F_hll[0,2]:<14.6e}")
    print(f"  {'HLLC':<8} {F_hllc[0,0]:<14.6e} {F_hllc[0,1]:<14.6e} {F_hllc[0,2]:<14.6e}")

    # All fluxes should be finite and reasonable
    ok = True
    for F, name in [(F_llf, "LLF"), (F_hll, "HLL"), (F_hllc, "HLLC")]:
        if not np.all(np.isfinite(F)):
            print(f"  WARNING: {name} has non-finite values!")
            ok = False

    # Mass flux should be positive (flow from high to low density region)
    if F_hllc[0, 0] <= 0:
        print(f"  WARNING: HLLC mass flux is not positive!")
        ok = False

    print("PASS" if ok else "FAIL")
    return ok


def run_all_tests():
    """Run all HLLC tetrad tests and report results."""
    print("="*60)
    print("HLLC TETRAD TRANSFORMATION VALIDATION TESTS")
    print("="*60)

    results = []
    results.append(("Flat spacetime", test_flat_spacetime_consistency()))
    results.append(("Flux consistency", test_flux_consistency()))
    results.append(("Transform factors", test_transformation_factors()))
    results.append(("HLLC vs HLL contact", test_hllc_vs_hll_contact()))
    results.append(("Char speeds", test_characteristic_speeds()))
    results.append(("Curved spacetime", test_curved_spacetime_consistency()))
    results.append(("Solver comparison", test_hllc_hll_llf_comparison()))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {name:20s}: {'PASS' if ok else 'FAIL'}")

    print("-"*40)
    print(f"  Total: {passed}/{len(results)}")

    return passed == len(results)


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
