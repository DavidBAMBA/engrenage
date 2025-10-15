"""
Simple Hydro Test - Engrenage

Basic test to verify hydro code runs without errors on TOV initial data.

Tests:
1. TOV solver generates valid initial data
2. Initial data interpolation works
3. Cons2prim inversion is consistent
4. Single hydro RHS evaluation completes without errors
5. Boundaries and reconstruction work correctly

This is simpler than full evolution - just verifies the machinery works.

Author: Engrenage team
Date: 2025-10-14
"""

import numpy as np
import sys
import matplotlib.pyplot as plt

# Engrenage core
sys.path.insert(0, '/home/yo/repositories/engrenage')
from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground, i_r

# BSSN
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import (NUM_BSSN_VARS, idx_phi, idx_lapse)

# Hydro
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLERiemannSolver
from source.matter.hydro.atmosphere import AtmosphereParams

# TOV
from examples.tov_solver import TOVSolver
import examples.tov_initial_data_interpolated as tov_id_interp


def main():
    """Main test execution."""
    print("="*70)
    print("SIMPLE HYDRO TEST - Engrenage")
    print("="*70)
    print("Verify hydro code runs on TOV initial data")
    print("="*70)

    # ==================================================================
    # CONFIGURATION
    # ==================================================================
    r_max = 10.0
    num_points = 300
    K = 100.0
    Gamma = 2.0
    rho_central = 1.28e-3

    # Atmosphere
    ATMOSPHERE = AtmosphereParams(
        rho_floor=1.0e-11,
        p_floor=1.0e-13,
        v_max=0.9999,
        W_max=100.0,
    )

    print(f"\nGrid: N={num_points}, r_max={r_max}")
    print(f"EOS: K={K}, Gamma={Gamma}")
    print(f"Atmosphere: ρ_floor={ATMOSPHERE.rho_floor:.2e}\n")

    # ==================================================================
    # SETUP
    # ==================================================================
    print("Setting up grid and hydro...")
    spacing = LinearSpacing(num_points, r_max)
    eos = IdealGasEOS(gamma=Gamma)
    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere=ATMOSPHERE,
        reconstructor=create_reconstruction("minmod"),
        riemann_solver=HLLERiemannSolver()
    )

    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    print(f"✓ Grid created: dr_min={grid.min_dr:.6f}")

    # ==================================================================
    # TOV SOLUTION
    # ==================================================================
    print("\nSolving TOV...")
    tov_solver = TOVSolver(K=K, Gamma=Gamma, use_isotropic=False)
    tov_num_points = num_points * 10
    tov_dr = r_max / tov_num_points

    tov_solution = tov_solver.solve(rho_central, r_max=r_max, dr=tov_dr)
    print(f"✓ TOV: M={tov_solution['M_star']:.6f}, R={tov_solution['R']:.3f}, C={tov_solution['C']:.4f}")

    # ==================================================================
    # INITIAL DATA
    # ==================================================================
    print("\nCreating initial data...")
    initial_state_2d = tov_id_interp.create_initial_data_interpolated(
        tov_solution, grid, background, eos,
        atmosphere=ATMOSPHERE,
        polytrope_K=K, polytrope_Gamma=Gamma,
        use_hydrobase_tau=True,
        interp_order=11
    )
    print("✓ Initial data created")

    # ==================================================================
    # TEST 1: Cons2Prim Consistency
    # ==================================================================
    print("\n" + "="*70)
    print("TEST 1: Conservative-to-Primitive Consistency")
    print("="*70)

    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(initial_state_2d, bssn_vars, grid)
    prim = hydro._get_primitives(bssn_vars, grid.r)

    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
    success_rate = np.sum(prim['success'][interior]) / len(prim['success'][interior])

    print(f"  Cons2Prim success rate: {success_rate*100:.2f}%")
    print(f"  ρ range: [{np.min(prim['rho0'][interior]):.3e}, {np.max(prim['rho0'][interior]):.3e}]")
    print(f"  P range: [{np.min(prim['p'][interior]):.3e}, {np.max(prim['p'][interior]):.3e}]")
    print(f"  v range: [{np.min(prim['vr'][interior]):.3e}, {np.max(prim['vr'][interior]):.3e}]")

    if success_rate < 0.95:
        print(f"  ⚠ WARNING: Cons2prim success rate is low ({success_rate*100:.1f}%)")
        failed_indices = np.where(~prim['success'][interior])[0] + NUM_GHOSTS
        print(f"  Failed at indices: {failed_indices[:10]}")
        print(f"  Failed at r: {grid.r[failed_indices[:10]]}")
    else:
        print("  ✓ PASS: Cons2prim working correctly")

    # ==================================================================
    # TEST 2: Single RHS Evaluation
    # ==================================================================
    print("\n" + "="*70)
    print("TEST 2: Single Hydro RHS Evaluation")
    print("="*70)

    try:
        # Compute derivatives (needed for RHS)
        bssn_d1 = grid.get_d1_metric_quantities(initial_state_2d)

        # Compute hydro RHS
        hydro_rhs = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1, background)

        print(f"  RHS computed successfully")
        print(f"  dD/dt range: [{np.min(hydro_rhs[0, interior]):.3e}, {np.max(hydro_rhs[0, interior]):.3e}]")
        print(f"  dSr/dt range: [{np.min(hydro_rhs[1, interior]):.3e}, {np.max(hydro_rhs[1, interior]):.3e}]")
        print(f"  dτ/dt range: [{np.min(hydro_rhs[2, interior]):.3e}, {np.max(hydro_rhs[2, interior]):.3e}]")

        # Check for NaN/Inf
        if not np.all(np.isfinite(hydro_rhs)):
            print("  ✗ FAIL: RHS contains NaN or Inf")
            nan_mask = ~np.isfinite(hydro_rhs)
            nan_locs = np.where(nan_mask)
            print(f"  NaN/Inf at: {list(zip(nan_locs[0][:5], nan_locs[1][:5]))}")
        else:
            print("  ✓ PASS: RHS evaluation successful")

    except Exception as e:
        print(f"  ✗ FAIL: RHS evaluation failed with error:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()

    # ==================================================================
    # TEST 3: Check for Instabilities at Surface
    # ==================================================================
    print("\n" + "="*70)
    print("TEST 3: Surface Stability Check")
    print("="*70)

    R_star = tov_solution['R']
    # Find stellar surface on grid
    surface_idx = np.argmin(np.abs(grid.r - R_star))
    window = 20  # Check ±20 points around surface

    start_idx = max(NUM_GHOSTS, surface_idx - window)
    end_idx = min(len(grid.r) - NUM_GHOSTS, surface_idx + window)
    surface_region = slice(start_idx, end_idx)

    print(f"  Stellar radius: R={R_star:.4f}")
    print(f"  Surface index: {surface_idx} (r={grid.r[surface_idx]:.4f})")
    print(f"  Checking region: r=[{grid.r[start_idx]:.4f}, {grid.r[end_idx-1]:.4f}]")

    # Check for large gradients
    D = initial_state_2d[NUM_BSSN_VARS + 0, surface_region]
    Sr = initial_state_2d[NUM_BSSN_VARS + 1, surface_region]
    tau = initial_state_2d[NUM_BSSN_VARS + 2, surface_region]

    # Compute relative gradients
    dD = np.diff(D)
    dSr = np.diff(Sr)
    dtau = np.diff(tau)

    max_dD = np.max(np.abs(dD / (np.abs(D[:-1]) + 1e-30)))
    max_dSr = np.max(np.abs(dSr / (np.abs(Sr[:-1]) + 1e-30)))
    max_dtau = np.max(np.abs(dtau / (np.abs(tau[:-1]) + 1e-30)))

    print(f"  Max relative gradient |ΔD/D|: {max_dD:.3e}")
    print(f"  Max relative gradient |ΔSr/Sr|: {max_dSr:.3e}")
    print(f"  Max relative gradient |Δτ/τ|: {max_dtau:.3e}")

    if max_dD > 1.0 or max_dtau > 1.0:
        print(f"  ⚠ WARNING: Large gradients detected near surface")
        print(f"  This may cause numerical instabilities")
    else:
        print("  ✓ PASS: Surface gradients are reasonable")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    all_tests_passed = (
        success_rate >= 0.95 and
        np.all(np.isfinite(hydro_rhs)) and
        max_dD < 2.0 and max_dtau < 2.0
    )

    if all_tests_passed:
        print("✓ ALL TESTS PASSED")
        print("\nYour hydro code appears to be working correctly!")
        print("Next steps:")
        print("  1. Try running TOVEvolution_corrected.py")
        print("  2. If it crashes, use this test to identify the problem")
        print("  3. Check the crash location and compare with test diagnostics")
    else:
        print("⚠ SOME TESTS FAILED")
        print("\nIssues detected:")
        if success_rate < 0.95:
            print(f"  - Cons2prim failing at {(1-success_rate)*100:.1f}% of points")
        if not np.all(np.isfinite(hydro_rhs)):
            print("  - RHS contains NaN/Inf")
        if max_dD > 2.0 or max_dtau > 2.0:
            print("  - Large gradients at stellar surface")

        print("\nRecommendations:")
        print("  - Check TOV initial data interpolation")
        print("  - Verify atmosphere treatment at surface")
        print("  - Review boundary conditions")

    print("="*70)


if __name__ == "__main__":
    main()
