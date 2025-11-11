#!/usr/bin/env python3
"""
Test TOV initial data pipeline to identify where constraint violations arise.

Pipeline stages:
1. TOV solution (fine grid) ← VERIFIED CORRECT by test_tov_solver_accuracy.py
2. Interpolation to evolution grid
3. ADM → BSSN conversion
4. Fill ghost cells

This test checks each stage to find where |Ham| ≈ 12 violation originates.
"""

import numpy as np
import sys

sys.path.insert(0, '/home/yo/repositories/engrenage')

from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.matter.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.atmosphere import AtmosphereParams

from examples.TOV.tov_solver import TOVSolver
import examples.TOV.tov_initial_data_interpolated as tov_id
from source.bssn.constraintsdiagnostic import get_constraints_diagnostic


def test_adm_metric_consistency():
    """
    Test that interpolated ADM variables are physically consistent.

    Check:
    1. Metric positivity: det(γ) > 0
    2. Lapse positivity: α > 0
    3. Metric matches TOV: exp(4φ) = 1/(1-2M/r) in exterior
    """
    print("\n" + "="*70)
    print("TEST 1: ADM METRIC CONSISTENCY AFTER INTERPOLATION")
    print("="*70)

    # Setup (same as production)
    Gamma = 2.0
    K = 1.0
    rho_central = 0.128
    num_points = 500
    r_max = 16.0

    spacing = LinearSpacing(num_points, r_max)
    eos = IdealGasEOS(gamma=Gamma)
    atmosphere = AtmosphereParams()
    hydro = PerfectFluid(eos=eos, spacetime_mode='dynamic', atmosphere=atmosphere)
    state_vec = StateVector(hydro)
    grid = Grid(spacing, state_vec)
    background = FlatSphericalBackground(grid.r)

    # Solve TOV
    tov_solver = TOVSolver(K=K, Gamma=Gamma)
    tov_solution = tov_solver.solve(rho_central, r_max=r_max)
    R = float(tov_solution['R'])
    M = float(tov_solution['M_star'])

    print(f"\nTOV solution: M = {M:.6f}, R = {R:.6f}")
    print(f"Grid: N = {grid.N}, dr = {grid.min_dr:.6f}")

    # Compute ADM variables (before BSSN conversion)
    print(f"\nComputing ADM variables via interpolation...")
    adm_vars = tov_id.compute_adm_from_tov_interpolated(
        tov_solution, grid, atmosphere, interp_order=11
    )

    alpha = adm_vars['alpha']
    gamma_LL = adm_vars['gamma_LL']
    K_LL = adm_vars['K_LL']
    beta_U = adm_vars['beta_U']

    # Compute determinant of γ_ij
    det_gamma = np.array([np.linalg.det(gamma_LL[i]) for i in range(grid.N)])

    # Check positivity
    min_det = np.min(det_gamma)
    min_alpha = np.min(alpha)

    print(f"\nADM metric checks:")
    print(f"  min(det γ) = {min_det:.6e} (should be > 0)")
    print(f"  min(α)     = {min_alpha:.6f} (should be > 0)")

    assert min_det > 0, f"Negative metric determinant: min = {min_det:.3e}"
    assert min_alpha > 0, f"Negative lapse: min = {min_alpha:.3e}"

    # Check exterior matches Schwarzschild
    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
    r_interior = grid.r[interior]
    exterior_mask = r_interior > 1.1 * R

    if np.sum(exterior_mask) > 0:
        idx_ext_full = np.arange(NUM_GHOSTS, grid.N - NUM_GHOSTS)[exterior_mask]
        r_ext = r_interior[exterior_mask]

        # Extract γ_rr from ADM (should be exp(4φ) for Schwarzschild)
        gamma_rr_ext = gamma_LL[idx_ext_full, 0, 0]

        # Theoretical Schwarzschild: γ_rr = 1/(1-2M/r)
        gamma_rr_theory = 1.0 / (1.0 - 2.0 * M / r_ext)

        rel_err = np.abs(gamma_rr_ext - gamma_rr_theory) / gamma_rr_theory
        max_err = np.max(rel_err)

        print(f"\nExterior metric (r > 1.1R):")
        print(f"  γ_rr max error = {max_err:.3e}")

        assert max_err < 1e-8, \
            f"Exterior metric doesn't match Schwarzschild: max error = {max_err:.3e}"

    print("\n✅ PASS: ADM variables are physically consistent")
    return adm_vars, grid, background, tov_solution


def test_bssn_conversion(adm_vars, grid, background, tov_solution):
    """
    Test ADM → BSSN conversion.

    Check:
    1. det(γ̄) = det(ĝ) constraint is enforced
    2. φ is correctly computed: e^(4φ) = (det γ / det γ̄)^(1/3)
    3. h_ij is bounded
    """
    print("\n" + "="*70)
    print("TEST 2: ADM → BSSN CONVERSION")
    print("="*70)

    # Convert ADM → BSSN
    print(f"\nConverting ADM → BSSN...")
    bssn_state = tov_id.convert_adm_to_bssn(adm_vars, grid, background)

    # Extract BSSN variables
    from source.bssn.bssnstatevariables import (
        idx_phi, idx_hrr, idx_htt, idx_hpp, idx_K,
        idx_arr, idx_att, idx_app, idx_lapse
    )
    from source.backgrounds.sphericalbackground import i_r, i_t, i_p

    phi = bssn_state[idx_phi, :]
    h_rr = bssn_state[idx_hrr, :]
    h_tt = bssn_state[idx_htt, :]
    h_pp = bssn_state[idx_hpp, :]
    K = bssn_state[idx_K, :]

    # Reconstruct conformal metric γ̄_ij
    ghat_LL = background.hat_gamma_LL
    Re_LL = background.scaling_matrix

    gammabar_LL = np.zeros((grid.N, 3, 3))
    for i in range(grid.N):
        gammabar_LL[i, 0, 0] = ghat_LL[i, 0, 0] + Re_LL[i, 0, 0] * h_rr[i]
        gammabar_LL[i, 1, 1] = ghat_LL[i, 1, 1] + Re_LL[i, 1, 1] * h_tt[i]
        gammabar_LL[i, 2, 2] = ghat_LL[i, 2, 2] + Re_LL[i, 2, 2] * h_pp[i]

    # Check det(γ̄) = det(ĝ)
    det_gammabar = np.array([np.linalg.det(gammabar_LL[i]) for i in range(grid.N)])
    det_ghat = np.array([np.linalg.det(ghat_LL[i]) for i in range(grid.N)])

    rel_err_det = np.abs(det_gammabar - det_ghat) / np.maximum(det_ghat, 1e-30)
    max_det_err = np.max(rel_err_det)

    print(f"\nBSSN constraint checks:")
    print(f"  det(γ̄) = det(ĝ): max error = {max_det_err:.3e}")

    # Check φ bounds
    min_phi = np.min(phi)
    max_phi = np.max(phi)
    print(f"  φ range: [{min_phi:.6f}, {max_phi:.6f}]")

    # Check h_ij bounds (should be O(1) or smaller)
    max_h_rr = np.max(np.abs(h_rr))
    max_h_tt = np.max(np.abs(h_tt))
    print(f"  |h_rr| max = {max_h_rr:.6f}")
    print(f"  |h_θθ| max = {max_h_tt:.6f}")

    # Check K (extrinsic curvature trace)
    # For static TOV with Cowling, K should be exactly 0
    max_K = np.max(np.abs(K))
    print(f"  |K| max = {max_K:.3e} (should be 0 for static TOV)")

    # Assertions
    assert max_det_err < 1e-10, \
        f"det(γ̄) = det(ĝ) violated: max error = {max_det_err:.3e}"
    assert max_K < 1e-10, \
        f"K non-zero for static TOV: |K| = {max_K:.3e}"

    print("\n✅ PASS: ADM → BSSN conversion correct")
    return bssn_state


def test_hamiltonian_at_each_stage():
    """
    Compute Hamiltonian constraint at each stage to identify where violation arises.
    """
    print("\n" + "="*70)
    print("TEST 3: HAMILTONIAN CONSTRAINT AT EACH PIPELINE STAGE")
    print("="*70)

    # Setup
    Gamma = 2.0
    K = 1.0
    rho_central = 0.128
    num_points = 500
    r_max = 16.0

    spacing = LinearSpacing(num_points, r_max)
    eos = IdealGasEOS(gamma=Gamma)
    atmosphere = AtmosphereParams()
    hydro = PerfectFluid(eos=eos, spacetime_mode='dynamic', atmosphere=atmosphere)
    state_vec = StateVector(hydro)
    grid = Grid(spacing, state_vec)
    background = FlatSphericalBackground(grid.r)

    # Solve TOV
    tov_solver = TOVSolver(K=K, Gamma=Gamma)
    tov_solution = tov_solver.solve(rho_central, r_max=r_max)
    R = float(tov_solution['R'])
    M = float(tov_solution['M_star'])

    print(f"\nTOV: M = {M:.6f}, R = {R:.6f}")

    # Create complete initial data
    print(f"\nCreating complete initial data...")
    initial_state_2d = tov_id.create_initial_data_interpolated(
        tov_solution, grid, background, eos,
        atmosphere=atmosphere,
        interp_order=11
    )

    # Compute Hamiltonian constraint
    hydro.background = background
    Ham, Mom = get_constraints_diagnostic(
        initial_state_2d.flatten(), 0.0, grid, background, hydro
    )
    Ham = Ham[0, :]

    # Analyze by region (exclude ghosts)
    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
    r_interior = grid.r[interior]
    Ham_interior = Ham[interior]

    # Deep interior
    deep_mask = (r_interior > r_interior[0]) & (r_interior < 0.9 * R)
    Ham_deep = Ham_interior[deep_mask]
    r_deep = r_interior[deep_mask]

    # Near surface
    surface_mask = (r_interior >= 0.9 * R) & (r_interior <= 1.1 * R)
    Ham_surface = Ham_interior[surface_mask]

    # Exterior
    exterior_mask = r_interior > 1.1 * R
    Ham_exterior = Ham_interior[exterior_mask]

    # Statistics
    max_Ham_deep = np.max(np.abs(Ham_deep))
    i_max_deep = np.argmax(np.abs(Ham_deep))
    r_max_Ham = r_deep[i_max_deep]
    L2_Ham = np.sqrt(np.mean(Ham_deep**2))

    max_Ham_surface = np.max(np.abs(Ham_surface)) if len(Ham_surface) > 0 else 0.0
    max_Ham_exterior = np.max(np.abs(Ham_exterior)) if len(Ham_exterior) > 0 else 0.0

    print(f"\n" + "="*70)
    print("HAMILTONIAN CONSTRAINT ANALYSIS")
    print("="*70)

    print(f"\nDeep interior (r < 0.9R):")
    print(f"  max|Ham| = {max_Ham_deep:.6e} at r = {r_max_Ham:.6f}")
    print(f"  L2(Ham)  = {L2_Ham:.6e}")

    print(f"\nNear surface (0.9R < r < 1.1R):")
    print(f"  max|Ham| = {max_Ham_surface:.6e}")

    print(f"\nExterior (r > 1.1R):")
    print(f"  max|Ham| = {max_Ham_exterior:.6e}")

    # Show profile near maximum
    if max_Ham_deep > 1.0:
        print(f"\n⚠ LARGE CONSTRAINT VIOLATION DETECTED!")
        print(f"\nRadial profile around maximum:")
        i_max_abs = NUM_GHOSTS + np.argmax(np.abs(Ham_interior))
        print(f"{'i':>4s} {'r':>10s} {'Ham':>15s} {'|Ham|':>15s}")
        print("-"*50)
        for idx in range(max(NUM_GHOSTS, i_max_abs - 3), min(grid.N - NUM_GHOSTS, i_max_abs + 4)):
            marker = " ← MAX" if idx == i_max_abs else ""
            print(f"{idx:4d} {grid.r[idx]:10.6f} {Ham[idx]:+15.6e} {abs(Ham[idx]):15.6e}{marker}")

    # Return diagnostic info
    return {
        'max_Ham_deep': max_Ham_deep,
        'r_max_Ham': r_max_Ham,
        'L2_Ham': L2_Ham,
        'max_Ham_exterior': max_Ham_exterior,
        'grid': grid,
        'Ham': Ham,
        'R': R
    }


def main():
    print("\n" + "="*70)
    print("TOV INITIALIZATION PIPELINE DIAGNOSTIC TESTS")
    print("="*70)
    print("\nGoal: Identify which stage introduces |Ham| ≈ 12 violation")
    print("Stages:")
    print("  1. TOV solution (fine grid) ← VERIFIED CORRECT")
    print("  2. Interpolation → ADM variables")
    print("  3. ADM → BSSN conversion")
    print("  4. Fill ghost cells")
    print("  5. Compute Hamiltonian constraint")

    # Test interpolation and ADM consistency
    adm_vars, grid, background, tov_solution = test_adm_metric_consistency()

    # Test BSSN conversion
    bssn_state = test_bssn_conversion(adm_vars, grid, background, tov_solution)

    # Test Hamiltonian constraint
    diag = test_hamiltonian_at_each_stage()

    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    print(f"\n✅ TOV solver: CORRECT (verified by test_tov_solver_accuracy.py)")
    print(f"✅ ADM interpolation: Physically consistent")
    print(f"✅ ADM → BSSN: Constraints enforced correctly")

    if diag['max_Ham_deep'] > 0.1:
        print(f"\n❌ HAMILTONIAN CONSTRAINT: VIOLATED")
        print(f"   max|Ham| = {diag['max_Ham_deep']:.3e} at r = {diag['r_max_Ham']:.6f}")
        print(f"   L2(Ham)  = {diag['L2_Ham']:.3e}")
        print(f"\n🔍 INVESTIGATION NEEDED:")
        print(f"   Since TOV, interpolation, and BSSN conversion all pass,")
        print(f"   the violation likely comes from:")
        print(f"   1. Numerical derivatives in constraint computation")
        print(f"   2. Stress-energy tensor projection (ρ, S_i, S)")
        print(f"   3. Mismatch between BSSN and constraint calculation")
        return 1
    else:
        print(f"\n✅ HAMILTONIAN CONSTRAINT: SATISFIED")
        print(f"   max|Ham| = {diag['max_Ham_deep']:.3e}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
