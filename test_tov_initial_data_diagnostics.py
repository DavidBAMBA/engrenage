#!/usr/bin/env python3
"""
Comprehensive diagnostic test for TOV initial data.

Tests:
1. TOV solution satisfies TOV equations
2. ADM quantities from TOV are consistent
3. ADM → BSSN conversion preserves constraints
4. Hamiltonian constraint is satisfied
5. Momentum constraint is satisfied
6. Stress-energy tensor is correctly computed
"""

import numpy as np
import sys
sys.path.insert(0, '/home/yo/repositories/engrenage')

from source.core.grid import Grid
from source.core.spacing import LinearSpacing, SpacingExtent
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS
from source.matter.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.atmosphere import AtmosphereParams

from examples.TOV.tov_solver import TOVSolver
from examples.TOV.tov_initial_data_interpolated import create_initial_data_interpolated

from source.bssn.constraintsdiagnostic import get_constraints_diagnostic


def test_tov_equations(tov_solution, verbose=True):
    """
    Test 1: Verify TOV solution satisfies TOV differential equations.

    TOV equations:
    dP/dr = -(ρ + P)(M + 4πr³P) / [r(r - 2M)]
    dM/dr = 4πr²ρ
    """
    if verbose:
        print("\n" + "="*70)
        print("TEST 1: TOV equations satisfaction")
        print("="*70)

    r = tov_solution['r']
    P = tov_solution['P']
    M = tov_solution['M']
    rho = tov_solution['rho_baryon']
    R_star = tov_solution['R']

    # Find stellar interior (exclude surface and exterior)
    interior = (r > 0.1) & (r < R_star * 0.95)
    r_int = r[interior]

    # Compute numerical derivatives
    dP_dr_num = np.gradient(P, r)[interior]
    dM_dr_num = np.gradient(M, r)[interior]

    # Compute theoretical TOV RHS
    denom = r_int * (r_int - 2.0 * M[interior])
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)  # Avoid division by zero

    numerator = (M[interior] + 4.0 * np.pi * r_int**3 * P[interior])
    dP_dr_theory = -(rho[interior] + P[interior]) * numerator / denom

    dM_dr_theory = 4.0 * np.pi * r_int**2 * rho[interior]

    # Compare
    dP_err = np.abs(dP_dr_num - dP_dr_theory) / (np.abs(dP_dr_theory) + 1e-10)
    dM_err = np.abs(dM_dr_num - dM_dr_theory) / (np.abs(dM_dr_theory) + 1e-10)

    if verbose:
        print(f"  dP/dr relative error: max = {np.max(dP_err):.3e}, mean = {np.mean(dP_err):.3e}")
        print(f"  dM/dr relative error: max = {np.max(dM_err):.3e}, mean = {np.mean(dM_err):.3e}")

    if np.max(dP_err) < 0.01 and np.max(dM_err) < 0.01:
        if verbose:
            print("  ✓ TOV equations satisfied to 1% accuracy")
        return True
    else:
        if verbose:
            print("  ✗ TOV equations NOT satisfied")
        return False


def test_adm_from_tov(tov_solution, grid, background, verbose=True):
    """
    Test 2: Verify ADM quantities from TOV are physically consistent.

    Check:
    - Lapse α > 0
    - Metric γ_ij is positive definite
    - Schwarzschild geometry in exterior
    """
    if verbose:
        print("\n" + "="*70)
        print("TEST 2: ADM quantities from TOV")
        print("="*70)

    from examples.TOV.tov_initial_data_interpolated import compute_adm_from_tov_interpolated

    adm_vars = compute_adm_from_tov_interpolated(tov_solution, grid, AtmosphereParams(), interp_order=11)

    alpha = adm_vars['alpha']
    gamma_LL = adm_vars['gamma_LL']
    K_LL = adm_vars['K_LL']

    # Check lapse positivity
    alpha_min = np.min(alpha)
    if verbose:
        print(f"  Lapse α: min = {alpha_min:.6f}, max = {np.max(alpha):.6f}")

    # Check metric determinant positivity
    det_gamma = (gamma_LL[:, 0, 0] * gamma_LL[:, 1, 1] * gamma_LL[:, 2, 2])
    det_min = np.min(det_gamma)

    if verbose:
        print(f"  det(γ): min = {det_min:.6e}, max = {np.max(det_gamma):.6e}")

    # Check K_ij = 0 for static TOV
    K_max = np.max(np.abs(K_LL))
    if verbose:
        print(f"  |K_ij|: max = {K_max:.3e}")

    # Check exterior matches Schwarzschild
    R_star = tov_solution['R']
    M_star = tov_solution['M_star']
    exterior = np.abs(grid.r) > R_star * 1.1

    if np.any(exterior):
        r_ext = np.abs(grid.r[exterior])
        alpha_schwarzschild = np.sqrt(1.0 - 2.0 * M_star / r_ext)
        gamma_rr_schwarzschild = 1.0 / (1.0 - 2.0 * M_star / r_ext)

        alpha_err = np.max(np.abs(alpha[exterior] - alpha_schwarzschild))
        gamma_err = np.max(np.abs(gamma_LL[exterior, 0, 0] - gamma_rr_schwarzschild))

        if verbose:
            print(f"  Exterior (r > 1.1R):")
            print(f"    |α - α_Schw|: {alpha_err:.3e}")
            print(f"    |γ_rr - γ_rr,Schw|: {gamma_err:.3e}")

    success = (alpha_min > 0) and (det_min > 0) and (K_max < 1e-10)
    if verbose:
        if success:
            print("  ✓ ADM quantities are physically consistent")
        else:
            print("  ✗ ADM quantities have problems")

    return success


def test_bssn_conversion(tov_solution, grid, background, eos, atmosphere, verbose=True):
    """
    Test 3: Verify ADM → BSSN conversion preserves key properties.

    Check:
    - det(γ̄) = det(ĝ) constraint
    - φ is smooth
    - h_ij is small
    """
    if verbose:
        print("\n" + "="*70)
        print("TEST 3: ADM → BSSN conversion")
        print("="*70)

    state = create_initial_data_interpolated(
        tov_solution, grid, background, eos,
        atmosphere=atmosphere, interp_order=11
    )

    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state[:NUM_BSSN_VARS, :])

    # Check det constraint
    from source.bssn.tensoralgebra import get_bar_gamma_LL
    bar_gamma_LL = get_bar_gamma_LL(grid.r, bssn_vars.h_LL, background)

    det_bar_gamma = np.linalg.det(bar_gamma_LL)
    det_hat_gamma = background.det_hat_gamma

    det_violation = np.abs(det_bar_gamma / det_hat_gamma - 1.0)

    if verbose:
        print(f"  det(γ̄)/det(ĝ) - 1: max = {np.max(det_violation):.3e}")

    # Check φ
    phi = bssn_vars.phi
    if verbose:
        print(f"  φ: min = {np.min(phi):.6e}, max = {np.max(phi):.6e}")

    # Check h_ij magnitude
    h_LL = bssn_vars.h_LL
    h_max = np.max(np.abs(h_LL))
    if verbose:
        print(f"  |h_ij|: max = {h_max:.3e}")

    # Check K
    K = bssn_vars.K
    K_max = np.max(np.abs(K))
    if verbose:
        print(f"  |K|: max = {K_max:.3e}")

    success = (np.max(det_violation) < 1e-3) and (K_max < 1e-10)
    if verbose:
        if success:
            print("  ✓ BSSN conversion successful")
        else:
            print("  ✗ BSSN conversion has problems")

    return success, state


def test_stress_energy_tensor(state, grid, background, eos, atmosphere, verbose=True):
    """
    Test 4: Verify stress-energy tensor is correctly computed.

    Check:
    - ρ > 0 in interior
    - S_i consistent with static star (should be ~0)
    - S_{ij} positive definite
    """
    if verbose:
        print("\n" + "="*70)
        print("TEST 4: Stress-energy tensor")
        print("="*70)

    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state[:NUM_BSSN_VARS, :])

    # Create hydro object and set background
    hydro = PerfectFluid(eos=eos, spacetime_mode='dynamic', atmosphere=atmosphere)
    hydro.background = background  # Set background for geometry calculations
    hydro.set_matter_vars(state, bssn_vars, grid)

    emtensor = hydro.get_emtensor(grid.r, bssn_vars, background)

    # Check ρ
    rho = emtensor.rho
    if verbose:
        print(f"  ρ: min = {np.min(rho):.6e}, max = {np.max(rho):.6e}")

    # Check S_i (should be ~0 for static star)
    Si = emtensor.Si
    Si_max = np.max(np.abs(Si))
    if verbose:
        print(f"  |S_i|: max = {Si_max:.3e}")

    # Check S
    S = emtensor.S
    if verbose:
        print(f"  S: min = {np.min(S):.6e}, max = {np.max(S):.6e}")

    # Check S_ij diagonal elements (should be positive)
    Sij = emtensor.Sij
    Sij_diag_min = np.min([np.min(Sij[:, i, i]) for i in range(3)])
    if verbose:
        print(f"  S_ii: min = {Sij_diag_min:.6e}")

    success = (np.min(rho) >= 0) and (Si_max < 1.0) and (Sij_diag_min >= 0)
    if verbose:
        if success:
            print("  ✓ Stress-energy tensor is physical")
        else:
            print("  ✗ Stress-energy tensor has problems")

    return success


def test_hamiltonian_constraint(state, grid, background, eos, atmosphere, verbose=True):
    """
    Test 5: Analyze Hamiltonian constraint violation term by term.

    H = (2/3)K² - Ā_ij Ā^ij + e^{-4φ}[R̄ - 8γ̄^ij∂_iφ∂_jφ - 8γ̄^ij∂_i∂_jφ + 8γ̄^ijΓ̄^k_ij∂_kφ] - 16πGρ
    """
    if verbose:
        print("\n" + "="*70)
        print("TEST 5: Hamiltonian constraint (term by term)")
        print("="*70)

    # Create hydro object and set background
    hydro = PerfectFluid(eos=eos, spacetime_mode='dynamic', atmosphere=atmosphere)
    hydro.background = background  # Set background for geometry calculations

    Ham, Mom = get_constraints_diagnostic(state, 0.0, grid, background, hydro)

    # Ham has shape [num_times, N]. For single state: [1, N]
    # Center is at r=0, which is index 0
    Ham_center = Ham[0, 0]  # time=0, spatial index=0 (center)
    Ham_max = np.max(np.abs(Ham[0, :]))
    Ham_mean = np.mean(np.abs(Ham[0, :]))

    if verbose:
        print(f"  Hamiltonian constraint H:")
        print(f"    At center: {Ham_center:.6e}")
        print(f"    max|H|:    {Ham_max:.6e}")
        print(f"    mean|H|:   {Ham_mean:.6e}")

    # Now compute term by term
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state, bssn_vars, grid)

    from source.bssn.tensoralgebra import (
        get_bar_A_squared, get_bar_gamma_UU, get_bar_gamma_LL,
        get_bar_christoffel, get_tensor_connections, get_bar_ricci_tensor, get_trace
    )

    # Get derivatives like constraintsdiagnostic.py does
    d1 = grid.get_d1_metric_quantities(state)
    d2 = grid.get_d2_metric_quantities(state)

    # Term 1: (2/3)K²
    K = bssn_vars.K
    term1 = (2.0/3.0) * K**2

    # Term 2: -Ā_ij Ā^ij
    bar_A_squared = get_bar_A_squared(grid.r, bssn_vars, background)
    term2 = -bar_A_squared

    # Term 3: Ricci scalar - compute like constraintsdiagnostic.py
    bar_gamma_UU = get_bar_gamma_UU(grid.r, bssn_vars.h_LL, background)
    bar_gamma_LL = get_bar_gamma_LL(grid.r, bssn_vars.h_LL, background)
    Delta_U, Delta_ULL, Delta_LLL = get_tensor_connections(grid.r, bssn_vars.h_LL, d1.h_LL, background)
    bar_chris = get_bar_christoffel(grid.r, Delta_ULL, background)

    bar_Rij = get_bar_ricci_tensor(grid.r, bssn_vars.h_LL, d1.h_LL, d2.h_LL,
                                     bssn_vars.lambda_U, d1.lambda_U,
                                     Delta_U, Delta_ULL, Delta_LLL,
                                     bar_gamma_UU, bar_gamma_LL, background)
    bar_R = get_trace(bar_Rij, bar_gamma_UU)

    em4phi = np.exp(-4.0 * bssn_vars.phi)
    term3 = em4phi * bar_R

    # Term 4: -8γ̄^ij∂_iφ∂_jφ
    term4 = em4phi * (-8.0 * np.einsum('xij,xi,xj->x', bar_gamma_UU, d1.phi, d1.phi))

    # Term 5: -8γ̄^ij∂_i∂_jφ
    term5 = em4phi * (-8.0 * np.einsum('xij,xij->x', bar_gamma_UU, d2.phi))

    # Term 6: +8γ̄^ijΓ̄^k_ij∂_kφ
    term6 = em4phi * (8.0 * np.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, d1.phi))

    # Term 7: -16πGρ
    emtensor = hydro.get_emtensor(grid.r, bssn_vars, background)
    eight_pi_G = 8.0 * np.pi
    term7 = -2.0 * eight_pi_G * emtensor.rho

    # Sum all terms
    H_computed = term1 + term2 + term3 + term4 + term5 + term6 + term7

    # Center is at r=0, which is index 0
    center = 0
    if verbose:
        print(f"\n  Term-by-term at center (r=0):")
        print(f"    (2/3)K²:                    {term1[center]:+.6e}")
        print(f"    -Ā_ij Ā^ij:                 {term2[center]:+.6e}")
        print(f"    e^{{-4φ}} R̄:                 {term3[center]:+.6e}")
        print(f"    -8e^{{-4φ}} γ̄^ij∂_iφ∂_jφ:    {term4[center]:+.6e}")
        print(f"    -8e^{{-4φ}} γ̄^ij∂_i∂_jφ:     {term5[center]:+.6e}")
        print(f"    +8e^{{-4φ}} γ̄^ijΓ̄^k_ij∂_kφ: {term6[center]:+.6e}")
        print(f"    -16πGρ:                     {term7[center]:+.6e}")
        print(f"    --------------------------------")
        print(f"    Sum (computed H):           {H_computed[center]:+.6e}")
        print(f"    get_constraints_diagnostic: {Ham_center:+.6e}")
        print(f"    Difference:                 {abs(H_computed[center] - Ham_center):.3e}")

    # Identify dominant term
    terms = {
        '(2/3)K²': term1[center],
        '-Ā_ij Ā^ij': term2[center],
        'e^{-4φ} R̄': term3[center],
        '-8e^{-4φ} γ̄^ij∂_iφ∂_jφ': term4[center],
        '-8e^{-4φ} γ̄^ij∂_i∂_jφ': term5[center],
        '+8e^{-4φ} γ̄^ijΓ̄^k_ij∂_kφ': term6[center],
        '-16πGρ': term7[center]
    }

    dominant_term = max(terms.items(), key=lambda x: abs(x[1]))
    if verbose:
        print(f"\n  Dominant term: {dominant_term[0]} = {dominant_term[1]:.6e}")

    success = Ham_max < 1.0
    if verbose:
        if success:
            print(f"\n  ✓ Hamiltonian constraint satisfied (|H| < 1.0)")
        else:
            print(f"\n  ✗ Hamiltonian constraint VIOLATED (|H| = {Ham_max:.1f} >> 1.0)")

    return success, Ham


def main():
    print("="*70)
    print("COMPREHENSIVE TOV INITIAL DATA DIAGNOSTICS")
    print("="*70)

    # Setup
    gamma = 2.0
    K_val = 1.0
    rho_central = 0.2

    print(f"\nTOV parameters:")
    print(f"  γ = {gamma}")
    print(f"  K = {K_val}")
    print(f"  ρ_c = {rho_central}")

    # Solve TOV
    print(f"\nSolving TOV equations...")
    solver = TOVSolver(K=K_val, Gamma=gamma)
    r_probe = np.linspace(0.0, 200.0, 4096)
    tov_probe = solver.solve(rho_central, r_max=r_probe[-1])
    r_max = 2.0 * tov_probe['R']

    r_tov = np.linspace(0.0, r_max, 4096)
    tov_solution = solver.solve(rho_central, r_max=r_max)

    print(f"  M = {tov_solution['M_star']:.6f}")
    print(f"  R = {tov_solution['R']:.6f}")
    print(f"  M/R = {tov_solution['M_star']/tov_solution['R']:.6f}")

    # Setup grid
    dr = 0.04
    spacing = LinearSpacing(int(r_max/dr), r_max, SpacingExtent.HALF)
    r = spacing[0]

    eos = IdealGasEOS(gamma=gamma)
    atmosphere = AtmosphereParams()
    state_vec = StateVector(PerfectFluid(eos=eos, spacetime_mode='dynamic', atmosphere=atmosphere))
    grid = Grid(spacing, state_vec)
    background = FlatSphericalBackground(r)

    # Run tests
    results = {}

    results['TOV_equations'] = test_tov_equations(tov_solution)
    results['ADM_from_TOV'] = test_adm_from_tov(tov_solution, grid, background)
    success_bssn, state = test_bssn_conversion(tov_solution, grid, background, eos, atmosphere)
    results['BSSN_conversion'] = success_bssn
    results['stress_energy'] = test_stress_energy_tensor(state, grid, background, eos, atmosphere)
    results['hamiltonian'], Ham = test_hamiltonian_constraint(state, grid, background, eos, atmosphere)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:25s}: {status}")

    all_passed = all(results.values())

    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED - INVESTIGATION REQUIRED")
    print("="*70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
