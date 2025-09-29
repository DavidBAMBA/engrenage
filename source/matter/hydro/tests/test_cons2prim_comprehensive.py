#!/usr/bin/env python3
"""
COMPREHENSIVE CONS2PRIM TEST SUITE
==================================

Unified test script that combines all cons2prim testing functionality:
- Performance benchmarks
- Correctness validation
- Failure analysis
- Method tracking
- Edge case testing
- Vectorized vs legacy comparison

This replaces all the individual test_*.py, debug_*.py, and analyze_*.py files.
"""

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import traceback

# Add source path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.cons2prim import (
    cons_to_prim, Cons2PrimSolver, prim_to_cons,
    _solve_pressure, _bracket_pressure, _state_from_p
)

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
# PERFORMANCE TESTING
# ============================================================================

class PerformanceTracker:
    """Track performance metrics during testing."""

    def __init__(self):
        self.times = []
        self.success_rates = []
        self.sizes = []

    def add_measurement(self, size, time_taken, success_rate):
        self.sizes.append(size)
        self.times.append(time_taken)
        self.success_rates.append(success_rate)

    def plot_performance(self, filename="cons2prim_performance.png"):
        """Plot performance results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Time per point
        time_per_point = np.array(self.times) / np.array(self.sizes) * 1e6  # μs
        ax1.loglog(self.sizes, time_per_point, 'bo-')
        ax1.set_xlabel('Number of points')
        ax1.set_ylabel('Time per point (μs)')
        ax1.set_title('Vectorized Performance')
        ax1.grid(True)

        # Success rate
        ax2.semilogx(self.sizes, self.success_rates, 'ro-')
        ax2.set_xlabel('Number of points')
        ax2.set_ylabel('Success rate')
        ax2.set_title('Conversion Success Rate')
        ax2.grid(True)
        ax2.set_ylim([0.98, 1.02])

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Performance plot saved: {filename}")

def benchmark_performance():
    """Comprehensive performance benchmark."""
    print("=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)

    eos = IdealGasEOS(2.0)
    solver = Cons2PrimSolver(eos)
    tracker = PerformanceTracker()

    sizes = [100, 500, 1000, 5000, 10000]

    for N in sizes:
        print(f"\nTesting with N = {N} points:")

        # Create test data
        rho0, vr, p = create_test_data_varied(N)
        gamma_rr = np.ones(N)

        # Convert to conservative
        D, Sr, tau = prim_to_cons(rho0, vr, p, gamma_rr, eos)
        U = {'D': D, 'Sr': Sr, 'tau': tau}
        metric = {'gamma_rr': gamma_rr}

        # Time conversion
        start_time = time.time()
        result = solver.convert(U, metric=metric)
        elapsed_time = time.time() - start_time

        success_rate = np.mean(result['success'])
        time_per_point = elapsed_time / N * 1e6  # μs

        print(f"  Time: {elapsed_time:.6f} s")
        print(f"  Success rate: {success_rate:.3f}")
        print(f"  Time per point: {time_per_point:.2f} μs")

        tracker.add_measurement(N, elapsed_time, success_rate)

        # Check results quality
        success_mask = result['success']
        if np.any(success_mask):
            print(f"  rho0 range: [{np.min(result['rho0'][success_mask]):.3e}, {np.max(result['rho0'][success_mask]):.3e}]")
            print(f"  vr range: [{np.min(result['vr'][success_mask]):.3e}, {np.max(result['vr'][success_mask]):.3e}]")
            print(f"  p range: [{np.min(result['p'][success_mask]):.3e}, {np.max(result['p'][success_mask]):.3e}]")

    tracker.plot_performance()
    return tracker

# ============================================================================
# CORRECTNESS TESTING
# ============================================================================

def test_correctness():
    """Test correctness of conversion with energy conservation."""
    print("=" * 60)
    print("CORRECTNESS TESTING")
    print("=" * 60)

    eos = IdealGasEOS(2.0)
    solver = Cons2PrimSolver(eos)

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
        gamma_rr = np.ones(len(rho0))

        # Convert to conservative
        D, Sr, tau = prim_to_cons(rho0, vr, p, gamma_rr, eos)

        # Convert back
        U = {'D': D, 'Sr': Sr, 'tau': tau}
        metric = {'gamma_rr': gamma_rr}
        result = solver.convert(U, metric=metric)

        print(f"  Success: {result['success']}")
        print(f"  Original rho0: {rho0}")
        print(f"  Recovered rho0: {result['rho0']}")
        print(f"  Original vr: {vr}")
        print(f"  Recovered vr: {result['vr']}")
        print(f"  Original p: {p}")
        print(f"  Recovered p: {result['p']}")

        # Check energy conservation
        for j in range(len(D)):
            if result['success'][j]:
                lhs = tau[j] + D[j]
                rhs = result['rho0'][j] * result['h'][j] * result['W'][j]**2 - result['p'][j]
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
    print("=" * 60)
    print("FAILURE ANALYSIS")
    print("=" * 60)

    eos = IdealGasEOS(2.0)
    solver = Cons2PrimSolver(eos)

    # Test edge cases
    edge_cases = create_edge_case_data()

    failures = []
    successes = []

    for rho0, vr, p, description in edge_cases:
        gamma_rr = 1.0

        # Convert to conservative
        D, Sr, tau = prim_to_cons(np.array([rho0]), np.array([vr]),
                                 np.array([p]), np.array([gamma_rr]), eos)

        # Try conversion
        U = {'D': D, 'Sr': Sr, 'tau': tau}
        metric = {'gamma_rr': np.array([gamma_rr])}

        try:
            result = solver.convert(U, metric=metric)

            if result['success'][0]:
                successes.append((rho0, vr, p, description))
                print(f"✓ SUCCESS: {description}")
                print(f"    Input: rho0={rho0:.3e}, vr={vr:.3f}, p={p:.3e}")
                print(f"    Output: rho0={result['rho0'][0]:.3e}, vr={result['vr'][0]:.3f}, p={result['p'][0]:.3e}")
            else:
                failures.append((rho0, vr, p, description))
                print(f"✗ FAILURE: {description}")
                print(f"    Input: rho0={rho0:.3e}, vr={vr:.3f}, p={p:.3e}")
        except Exception as e:
            failures.append((rho0, vr, p, f"{description} (Exception: {str(e)})"))
            print(f"✗ ERROR: {description}")
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
# COMPARISON TESTING
# ============================================================================

def compare_vectorized_vs_legacy():
    """Compare vectorized implementation with legacy point-by-point."""
    print("=" * 60)
    print("VECTORIZED vs LEGACY COMPARISON")
    print("=" * 60)

    eos = IdealGasEOS(2.0)

    # Create test data
    N = 1000
    rho0, vr, p = create_test_data_varied(N)
    gamma_rr = np.ones(N)
    D, Sr, tau = prim_to_cons(rho0, vr, p, gamma_rr, eos)

    # Test vectorized approach
    print("Testing vectorized solver...")
    solver = Cons2PrimSolver(eos)
    U = {'D': D, 'Sr': Sr, 'tau': tau}
    metric = {'gamma_rr': gamma_rr}

    start_time = time.time()
    result_vec = solver.convert(U, metric=metric)
    time_vec = time.time() - start_time

    # Test legacy approach (point by point)
    print("Testing legacy point-by-point...")
    start_time = time.time()

    rho0_legacy = np.zeros(N)
    vr_legacy = np.zeros(N)
    p_legacy = np.zeros(N)
    success_legacy = np.zeros(N, dtype=bool)

    for i in range(N):
        try:
            result_single = cons_to_prim(
                {'D': D[i:i+1], 'Sr': Sr[i:i+1], 'tau': tau[i:i+1]},
                eos, metric={'gamma_rr': gamma_rr[i:i+1]}
            )
            if result_single['success'][0]:
                rho0_legacy[i] = result_single['rho0'][0]
                vr_legacy[i] = result_single['vr'][0]
                p_legacy[i] = result_single['p'][0]
                success_legacy[i] = True
        except:
            pass

    time_legacy = time.time() - start_time

    # Compare results
    success_rate_vec = np.mean(result_vec['success'])
    success_rate_legacy = np.mean(success_legacy)
    speedup = time_legacy / time_vec

    print(f"\nRESULTS:")
    print(f"  Vectorized time: {time_vec:.6f} s")
    print(f"  Legacy time: {time_legacy:.6f} s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Vectorized success rate: {success_rate_vec:.3f}")
    print(f"  Legacy success rate: {success_rate_legacy:.3f}")

    # Check agreement where both succeeded
    both_success = result_vec['success'] & success_legacy
    if np.any(both_success):
        rho0_diff = np.max(np.abs(result_vec['rho0'][both_success] - rho0_legacy[both_success]))
        vr_diff = np.max(np.abs(result_vec['vr'][both_success] - vr_legacy[both_success]))
        p_diff = np.max(np.abs(result_vec['p'][both_success] - p_legacy[both_success]))

        print(f"  Max rho0 difference: {rho0_diff:.2e}")
        print(f"  Max vr difference: {vr_diff:.2e}")
        print(f"  Max p difference: {p_diff:.2e}")

# ============================================================================
# STATISTICS ANALYSIS
# ============================================================================

def analyze_solver_statistics():
    """Analyze solver internal statistics."""
    print("=" * 60)
    print("SOLVER STATISTICS ANALYSIS")
    print("=" * 60)

    eos = IdealGasEOS(2.0)
    solver = Cons2PrimSolver(eos)

    # Reset statistics
    solver.reset_statistics()

    # Run various test cases
    test_regimes = {
        'normal': (lambda N: create_test_data_varied(N)[:3]),
        'extreme': (lambda N: (
            np.random.uniform(1e-10, 100, N),
            np.random.uniform(-0.95, 0.95, N),
            np.random.uniform(1e-15, 1000, N)
        ))
    }

    for regime_name, data_func in test_regimes.items():
        print(f"\nTesting {regime_name} regime:")

        N = 1000
        rho0, vr, p = data_func(N)
        gamma_rr = np.ones(N)
        D, Sr, tau = prim_to_cons(rho0, vr, p, gamma_rr, eos)

        U = {'D': D, 'Sr': Sr, 'tau': tau}
        metric = {'gamma_rr': gamma_rr}

        start_time = time.time()
        result = solver.convert(U, metric=metric)
        elapsed = time.time() - start_time

        stats = solver.get_statistics()

        print(f"  Time: {elapsed:.4f} s")
        print(f"  Success rate: {stats['success_rate']:.3f}")
        print(f"  Newton success rate: {stats['newton_rate']:.3f}")
        print(f"  Bisection fallback rate: {stats['bisection_rate']:.3f}")
        print(f"  Total calls: {stats['total_calls']}")

        solver.reset_statistics()

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run the complete test suite."""
    print("COMPREHENSIVE CONS2PRIM TEST SUITE")
    print("=" * 60)
    print("Testing vectorized cons2prim implementation")
    print("This replaces all individual test/debug/analyze scripts")
    print()

    start_time = time.time()

    try:
        # Core functionality tests
        test_correctness()

        # Performance testing
        benchmark_performance()

        # Failure analysis
        analyze_failures()

        # Comparison testing
        compare_vectorized_vs_legacy()

        # Statistics analysis
        analyze_solver_statistics()

        total_time = time.time() - start_time

        print("=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print(f"Total test time: {total_time:.2f} seconds")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()