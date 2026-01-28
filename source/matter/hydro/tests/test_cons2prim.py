#!/usr/bin/env python3
"""
CONS2PRIM TEST SUITE
====================

Test script for cons2prim algorithm robustness:
- Correctness validation
- Failure analysis
- Edge case testing
"""

import sys
import numpy as np
import traceback

# Add source path
sys.path.insert(0, '/home/davidbamba/repositories/engrenage')

from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.cons2prim import Cons2PrimSolver, prim_to_cons
from source.matter.hydro.atmosphere import AtmosphereParams
from source.matter.hydro.geometry import GeometryState

# ============================================================================
# TEST DATA GENERATION
# ============================================================================

def create_test_data_varied(N=1000):
    """Create varied test data covering different physical regimes."""

    # Regime 1: Normal states
    rho0_normal = np.random.uniform(0.1, 2.0, N//4)
    vr_normal = np.random.uniform(-0.5, 0.5, N//4)
    p_normal = np.random.uniform(0.01, 0.5, N//4)

    # Regime 2: Ultra-relativistic
    rho0_ultra = np.random.uniform(0.5, 1.5, N//4)
    vr_ultra = np.random.uniform(-0.9, 0.9, N//4)
    p_ultra = np.random.uniform(0.1, 1.0, N//4)

    # Regime 3: Atmosphere
    rho0_atm = np.random.uniform(1e-12, 1e-10, N//4)
    vr_atm = np.random.uniform(-0.1, 0.1, N//4)
    p_atm = np.random.uniform(1e-15, 1e-13, N//4)

    # Regime 4: Extreme conditions
    rho0_extreme = np.random.uniform(1e-11, 10.0, N//4)
    vr_extreme = np.random.uniform(-0.99, 0.99, N//4)
    p_extreme = np.random.uniform(1e-13, 2.0, N//4)

    # Combine all
    rho0 = np.concatenate([rho0_normal, rho0_ultra, rho0_atm, rho0_extreme])
    vr = np.concatenate([vr_normal, vr_ultra, vr_atm, vr_extreme])
    p = np.concatenate([p_normal, p_ultra, p_atm, p_extreme])

    return rho0, vr, p

def create_edge_case_data():
    """Create specific edge cases that are known to be challenging."""

    edge_cases = [
        # (rho0, vr, p, description)
        (1.0, 0.9, 1000.5, "High pressure, high velocity"),
        (0.125, 0.0, 0.1, "Low density, rest"),
        (5.0, 0.6, 10.0, "Moderate all"),
        (10.0, 0.9, 50.0, "High density/pressure, high velocity"),
        (1.0, 0.0, 1.0, "Moderate, rest"),
        (110.001, 0.9, 1e-6, "Very high density, tiny pressure, high velocity"),
        (100.0, 0.7, 1000.0, "High density/pressure"),
        (1e-5, 0.01, 1e-6, "Very low density/pressure"),
        (1.5, 0.3, 0.8, "Moderate isotropic"),
        (2.0, 0.4, 0.2, "Moderate density, low pressure"),
        (50.0, 0.5, 500.0, "High density/pressure"),
        (0.02, 0.99, 0.02, "Low density/pressure, ultra-relativistic"),
        (1e-3, 0.5, 1e-2, "Very low density, moderate pressure"),
    ]

    return edge_cases

# ============================================================================
# CORRECTNESS TESTING
# ============================================================================

def test_correctness():
    """Test correctness of conversion with energy conservation."""
    print("=" * 60)
    print("CORRECTNESS TESTING")
    print("=" * 60)

    eos = IdealGasEOS(2.0)
    atmosphere = AtmosphereParams()
    solver = Cons2PrimSolver(eos, atmosphere, solver_method='newton')

    # Simple test cases
    test_cases = [
        ([1.0, 2.0, 0.5], [0.1, -0.2, 0.05], [0.5, 1.0, 0.2]),
        ([0.1, 10.0, 1.0], [0.0, 0.5, -0.3], [0.01, 5.0, 0.1]),
    ]

    total_errors = []

    for i, (rho0_test, vr_test, p_test) in enumerate(test_cases):
        print(f"\nTest case {i+1}:")

        rho0 = np.array(rho0_test)
        vr = np.array(vr_test)
        p = np.array(p_test)
        N = len(rho0)

        # Create Minkowski geometry
        geom = GeometryState.minkowski(N)

        # Convert to conservative
        D, Sr, tau = prim_to_cons(rho0, vr, p, geom, eos)

        # Convert back (with p_guess = original pressure, simulating cache)
        rho0_rec, vr_rec, p_rec, eps_rec, W_rec, h_rec, success, _, _, _ = solver.convert(
            D, Sr, tau, geom, p_guess=p
        )

        print(f"  Success: {success}")
        print(f"  Original rho0: {rho0}")
        print(f"  Recovered rho0: {rho0_rec}")
        print(f"  Original vr: {vr}")
        print(f"  Recovered vr: {vr_rec}")
        print(f"  Original p: {p}")
        print(f"  Recovered p: {p_rec}")

        # Check energy conservation
        for j in range(len(D)):
            if success[j]:
                lhs = tau[j] + D[j]
                rhs = rho0_rec[j] * h_rec[j] * W_rec[j]**2 - p_rec[j]
                error = abs(lhs - rhs) / max(abs(lhs), 1e-10)
                total_errors.append(error)
                print(f"  Point {j}: Energy conservation error = {error:.2e}")

    print(f"\nOverall energy conservation:")
    print(f"  Max error: {np.max(total_errors):.2e}")
    print(f"  Mean error: {np.mean(total_errors):.2e}")
    print(f"  RMS error: {np.sqrt(np.mean(np.array(total_errors)**2)):.2e}")

# ============================================================================
# FAILURE ANALYSIS
# ============================================================================

def analyze_failures():
    """Analyze what types of states cause conversion failures."""
    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS (EDGE CASES)")
    print("=" * 60)

    eos = IdealGasEOS(2.0)
    atmosphere = AtmosphereParams()
    solver = Cons2PrimSolver(eos, atmosphere, solver_method='newton')

    # Test edge cases
    edge_cases = create_edge_case_data()

    failures = []
    successes = []

    for rho0, vr, p, description in edge_cases:
        # Create Minkowski geometry for single point
        geom = GeometryState.minkowski(1)

        # Convert to conservative
        D, Sr, tau = prim_to_cons(np.array([rho0]), np.array([vr]),
                                 np.array([p]), geom, eos)

        try:
            # With p_guess = original pressure (simulating cache)
            rho0_result, vr_result, p_result, eps_result, W_result, h_result, success, _, _, _ = solver.convert(
                D, Sr, tau, geom, p_guess=np.array([p])
            )

            if success[0]:
                successes.append((rho0, vr, p, description))
                print(f"SUCCESS: {description}")
                print(f"    Input:  rho0={rho0:.3e}, vr={vr:.3f}, p={p:.3e}")
                print(f"    Output: rho0={rho0_result[0]:.3e}, vr={vr_result[0]:.3f}, p={p_result[0]:.3e}")
            else:
                failures.append((rho0, vr, p, description))
                print(f"FAILURE: {description}")
                print(f"    Input: rho0={rho0:.3e}, vr={vr:.3f}, p={p:.3e}")
        except Exception as e:
            failures.append((rho0, vr, p, f"{description} (Exception: {str(e)})"))
            print(f"ERROR: {description}")
            print(f"    Exception: {str(e)}")

    print(f"\nSUMMARY:")
    print(f"  Successful conversions: {len(successes)}/{len(edge_cases)}")
    print(f"  Failed conversions: {len(failures)}/{len(edge_cases)}")
    print(f"  Success rate: {len(successes)/len(edge_cases)*100:.1f}%")

    if failures:
        print(f"\nFAILED CASES:")
        for rho0, vr, p, desc in failures:
            print(f"  - {desc}: rho0={rho0:.3e}, vr={vr:.3f}, p={p:.3e}")

# ============================================================================
# RANDOM STRESS TEST
# ============================================================================

def test_random_states(N=1000):
    """Test conversion with random states across all regimes."""
    print("\n" + "=" * 60)
    print(f"RANDOM STRESS TEST (N={N})")
    print("=" * 60)

    eos = IdealGasEOS(2.0)
    atmosphere = AtmosphereParams()
    solver = Cons2PrimSolver(eos, atmosphere,solver_method='newton')

    # Create varied test data
    rho0, vr, p = create_test_data_varied(N)
    geom = GeometryState.minkowski(N)

    # Convert to conservative
    D, Sr, tau = prim_to_cons(rho0, vr, p, geom, eos)

    # Convert back (with p_guess = original pressure, simulating cache)
    rho0_rec, vr_rec, p_rec, eps_rec, W_rec, h_rec, success, _, _, _ = solver.convert(
        D, Sr, tau, geom, p_guess=p
    )

    success_rate = np.mean(success)
    print(f"  Success rate: {success_rate*100:.2f}%")
    print(f"  Successful: {np.sum(success)}/{N}")
    print(f"  Failed: {np.sum(~success)}/{N}")

    # Analyze errors for successful conversions
    if np.any(success):
        rho_err = np.abs(rho0[success] - rho0_rec[success]) / np.maximum(rho0[success], 1e-30)
        vr_err = np.abs(vr[success] - vr_rec[success])
        p_err = np.abs(p[success] - p_rec[success]) / np.maximum(p[success], 1e-30)

        print(f"\n  Relative errors (successful points):")
        print(f"    rho0: max={np.max(rho_err):.2e}, mean={np.mean(rho_err):.2e}")
        print(f"    vr:   max={np.max(vr_err):.2e}, mean={np.mean(vr_err):.2e}")
        print(f"    p:    max={np.max(p_err):.2e}, mean={np.mean(p_err):.2e}")

# ============================================================================
# ATMOSPHERE LIMIT TEST
# ============================================================================

def test_atmosphere_limits():
    """Test how close to atmosphere floors the algorithm can handle."""
    print("\n" + "=" * 60)
    print("ATMOSPHERE LIMITS TEST")
    print("=" * 60)

    eos = IdealGasEOS(2.0)
    atmosphere = AtmosphereParams()
    solver = Cons2PrimSolver(eos, atmosphere, solver_method='newton')

    print(f"\nAtmosphere floors:")
    print(f"  rho_floor = {atmosphere.rho_floor:.2e}")
    print(f"  p_floor   = {atmosphere.p_floor:.2e}")

    # Test density sweep (fixed moderate pressure and low velocity)
    print("\n--- DENSITY SWEEP (p=0.01, vr=0.1) ---")
    rho_values = [1e-3, 1e-5, 1e-7, 1e-9, 1e-10, 1e-11, 5e-12, 2e-12, 1e-12, 5e-13]
    p_fixed = 0.01
    vr_fixed = 0.1

    for rho in rho_values:
        geom = GeometryState.minkowski(1)
        D, Sr, tau = prim_to_cons(np.array([rho]), np.array([vr_fixed]),
                                   np.array([p_fixed]), geom, eos)

        rho_rec, vr_rec, p_rec, _, _, _, success, _, _, _ = solver.convert(
            D, Sr, tau, geom, p_guess=np.array([p_fixed])
        )

        rel_err_rho = abs(rho - rho_rec[0]) / rho if rho > 0 else 0
        status = "OK" if success[0] and rel_err_rho < 0.01 else "FLOOR" if success[0] else "FAIL"

        print(f"  rho={rho:.1e}: rec={rho_rec[0]:.2e}, err={rel_err_rho:.2e} [{status}]")

    # Test pressure sweep with diagnostics
    print("\n--- PRESSURE SWEEP WITH DIAGNOSTICS (rho=0.01, vr=0.1) ---")
    p_values = [1e-3, 1e-5, 1e-6, 1e-7, 1e-8]
    rho_fixed = 0.01
    vr_fixed = 0.1
    gamma = 2.0  # EOS gamma

    for p in p_values:
        geom = GeometryState.minkowski(1)
        D, Sr, tau = prim_to_cons(np.array([rho_fixed]), np.array([vr_fixed]),
                                   np.array([p]), geom, eos)

        # Compute expected values analytically
        v2 = vr_fixed**2
        W_expected = 1.0 / np.sqrt(1.0 - v2)
        eps_expected = p / (rho_fixed * (gamma - 1))
        h_expected = 1.0 + eps_expected + p / rho_fixed
        cs2_expected = p * gamma * (gamma - 1) / (p * gamma + rho_fixed * (gamma - 1))

        # Show diagnostic info
        print(f"\n  p={p:.1e}:")
        print(f"    D={D[0]:.3e}, Sr={Sr[0]:.3e}, tau={tau[0]:.3e}")
        print(f"    Expected: eps={eps_expected:.3e}, h={h_expected:.6f}, cs2={cs2_expected:.3e}")
        print(f"    Check h>1: h-1 = {h_expected - 1:.3e}")

        rho_rec, vr_rec, p_rec, eps_rec, W_rec, h_rec, success, _, _, _ = solver.convert(
            D, Sr, tau, geom, p_guess=np.array([p])
        )

        rel_err_p = abs(p - p_rec[0]) / p if p > 0 else 0
        status = "OK" if success[0] and rel_err_p < 0.01 else "FLOOR" if success[0] else "FAIL"

        print(f"    Recovered: p={p_rec[0]:.2e}, eps={eps_rec[0]:.3e}, h={h_rec[0]:.6f} [{status}]")

    # Test both low (approaching atmosphere)
    print("\n--- BOTH LOW (rho and p decreasing together) ---")
    factors = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]

    for factor in factors:
        rho = atmosphere.rho_floor * factor
        # For ideal gas with gamma=2: p ~ rho * eps, use p ~ rho^2 scaling
        p = atmosphere.p_floor * factor**2

        geom = GeometryState.minkowski(1)
        D, Sr, tau = prim_to_cons(np.array([rho]), np.array([0.0]),
                                   np.array([p]), geom, eos)

        rho_rec, vr_rec, p_rec, _, _, _, success, _, _, _ = solver.convert(
            D, Sr, tau, geom, p_guess=np.array([p])
        )

        rel_err_rho = abs(rho - rho_rec[0]) / rho if rho > 0 else 0
        rel_err_p = abs(p - p_rec[0]) / p if p > 0 else 0
        status = "OK" if success[0] and rel_err_rho < 0.01 and rel_err_p < 0.01 else "FLOOR" if success[0] else "FAIL"

        print(f"  {factor:.0e}x floor: rho={rho:.1e}, p={p:.1e} -> err_rho={rel_err_rho:.2e}, err_p={rel_err_p:.2e} [{status}]")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run the complete test suite."""
    print("CONS2PRIM ROBUSTNESS TEST SUITE")
    print("=" * 60)
    print()

    try:
        # Atmosphere limits test (main focus)
        test_atmosphere_limits()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTEST FAILED: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
