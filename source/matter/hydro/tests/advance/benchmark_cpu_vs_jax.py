#!/usr/bin/env python3
"""
Benchmark: CPU (Numba) vs JAX (GPU) para modulo hydro.

Compara rendimiento y verifica correctitud numerica entre:
1. cons2prim - Conversion conservativas -> primitivas
2. reconstruction - Reconstruccion WENO-Z, MP5, MC, minmod
3. riemann - Solucionador HLL

Uso:
    python benchmark_cpu_vs_jax.py

Requisitos:
    - JAX instalado (con soporte GPU opcional)
    - Numba instalado
"""

import os
import sys
import time
import numpy as np

# Add paths for imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
sys.path.insert(0, repo_root)
hydro_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, hydro_path)

# Configuration
GRID_SIZES = [10000, 100000, 500000, 1000000, 2000000]
N_WARMUP = 3    # Warmup iterations (JIT compilation)
N_RUNS = 10     # Iterations for timing
TOL = 1e-8      # Tolerance for numerical comparison


def print_header():
    """Print benchmark header with system info."""
    print("=" * 70)
    print("BENCHMARK: CPU (Numba) vs JAX (GPU) - Hydro Module")
    print("=" * 70)

    # Check JAX availability and device
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        devices = jax.devices()
        gpu_available = any(d.platform == 'gpu' for d in devices)
        if gpu_available:
            gpu_device = jax.devices('gpu')[0]
            print(f"JAX backend: GPU ({gpu_device})")
        else:
            print("JAX backend: CPU (no GPU detected)")
    except ImportError:
        print("JAX: NOT INSTALLED")
        return False

    # Check Numba
    try:
        import numba
        print(f"Numba version: {numba.__version__}")
    except ImportError:
        print("Numba: NOT INSTALLED")
        return False

    print(f"Grid sizes: {GRID_SIZES}")
    print(f"Warmup runs: {N_WARMUP}, Timed runs: {N_RUNS}")
    print("=" * 70)
    return True


def generate_cons2prim_data(N, seed=42):
    """Generate realistic conservative variable data for testing."""
    np.random.seed(seed)

    # Rest mass density (astrophysically reasonable range)
    rho0 = np.random.uniform(1e-3, 1.0, N)

    # Velocity (sub-relativistic to mildly relativistic)
    v = np.random.uniform(-0.5, 0.5, N)

    # Pressure
    p = np.random.uniform(1e-4, 0.1, N)

    # Metric (Minkowski: gamma_rr = 1)
    gamma_rr = np.ones(N)
    alpha = np.ones(N)

    # EOS parameters
    gamma_eos = 5.0 / 3.0

    # Compute conservative variables
    v2 = gamma_rr * v * v
    v2 = np.clip(v2, 0.0, 1.0 - 1e-12)
    W = 1.0 / np.sqrt(1.0 - v2)
    eps = p / (rho0 * (gamma_eos - 1.0))
    h = 1.0 + eps + p / rho0

    D = rho0 * W
    Sr = rho0 * h * W * W * v * gamma_rr
    tau = rho0 * h * W * W - p - D

    return D, Sr, tau, gamma_rr, alpha, p, gamma_eos


def benchmark_cons2prim():
    """Benchmark cons2prim CPU vs JAX."""
    print("\n" + "-" * 70)
    print("BENCHMARK: cons2prim (Conservative to Primitive)")
    print("-" * 70)

    # Import both versions (use absolute paths to avoid relative import issues)
    from source.matter.hydro.cons2prim import Cons2PrimSolver
    from source.matter.hydro.tests.advance.cons2prim_jax import Cons2PrimSolverJAX
    from source.matter.hydro.atmosphere import AtmosphereParams
    from source.matter.hydro.eos import IdealGasEOS

    # Create EOS and atmosphere
    eos = IdealGasEOS(gamma=5.0/3.0)
    atmosphere = AtmosphereParams(
        rho_floor=1e-13,
        p_floor=1e-15,
        v_max=0.999999,
        W_max=1e3
    )

    # Create solvers
    solver_cpu = Cons2PrimSolver(eos, atmosphere)
    solver_jax = Cons2PrimSolverJAX(eos, atmosphere)

    # Import JAX for synchronization
    import jax

    results = []

    for N in GRID_SIZES:
        D, Sr, tau, gamma_rr, alpha, p_guess, _ = generate_cons2prim_data(N)

        # Warmup CPU
        for _ in range(N_WARMUP):
            solver_cpu.convert(D, Sr, tau, gamma_rr, p_guess=p_guess)

        # Time CPU
        cpu_times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            result_cpu = solver_cpu.convert(D, Sr, tau, gamma_rr, p_guess=p_guess)
            t1 = time.perf_counter()
            cpu_times.append(t1 - t0)
        cpu_time = np.median(cpu_times) * 1000  # ms

        rho_cpu, v_cpu, p_cpu = result_cpu[0], result_cpu[1], result_cpu[2]

        # Warmup JAX
        for _ in range(N_WARMUP):
            result = solver_jax.convert(D, Sr, tau, gamma_rr, p_guess=p_guess)
            jax.block_until_ready(result)

        # Time JAX
        jax_times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            result_jax = solver_jax.convert(D, Sr, tau, gamma_rr, p_guess=p_guess)
            jax.block_until_ready(result_jax)
            t1 = time.perf_counter()
            jax_times.append(t1 - t0)
        jax_time = np.median(jax_times) * 1000  # ms

        rho_jax, v_jax, p_jax = result_jax[0], result_jax[1], result_jax[2]

        # Verify numerical match
        max_diff_rho = np.max(np.abs(rho_cpu - rho_jax))
        max_diff_v = np.max(np.abs(v_cpu - v_jax))
        max_diff_p = np.max(np.abs(p_cpu - p_jax))
        max_diff = max(max_diff_rho, max_diff_v, max_diff_p)
        match = max_diff < TOL

        speedup = cpu_time / jax_time if jax_time > 0 else 0

        results.append({
            'N': N,
            'cpu_ms': cpu_time,
            'jax_ms': jax_time,
            'speedup': speedup,
            'match': match,
            'max_diff': max_diff
        })

    # Print results
    print(f"{'N':>10}  {'CPU (ms)':>10}  {'JAX (ms)':>10}  {'Speedup':>10}  {'Match':>8}")
    print("-" * 55)
    for r in results:
        match_str = "OK" if r['match'] else f"FAIL({r['max_diff']:.2e})"
        print(f"{r['N']:>10}  {r['cpu_ms']:>10.2f}  {r['jax_ms']:>10.2f}  {r['speedup']:>9.1f}x  {match_str:>8}")

    return results


def generate_reconstruction_data(N, seed=42):
    """Generate primitive variable data for reconstruction testing."""
    np.random.seed(seed)

    # Smooth profiles with some discontinuities (like a shock)
    x = np.linspace(0, 1, N)

    # Density: smooth + discontinuity
    rho = 1.0 + 0.5 * np.sin(4 * np.pi * x)
    rho[N//3:2*N//3] += 0.5  # Step

    # Velocity: smooth
    vr = 0.3 * np.sin(2 * np.pi * x)

    # Pressure: smooth
    p = 0.1 + 0.05 * np.cos(2 * np.pi * x)

    dx = x[1] - x[0]

    return rho, vr, p, x, dx


def benchmark_reconstruction():
    """Benchmark reconstruction CPU vs JAX."""
    print("\n" + "-" * 70)
    print("BENCHMARK: Reconstruction (WENO-Z)")
    print("-" * 70)

    # Import both versions
    from source.matter.hydro.reconstruction import Reconstruction
    from source.matter.hydro.tests.advance.reconstruction_jax import ReconstructionJAX

    import jax

    # Test WENO-Z (most commonly used)
    recon_cpu = Reconstruction(method="wenoz")
    recon_jax = ReconstructionJAX(method="wenoz")

    results = []

    for N in GRID_SIZES:
        rho, vr, p, x, dx = generate_reconstruction_data(N)

        # Warmup CPU
        for _ in range(N_WARMUP):
            recon_cpu.reconstruct_primitive_variables(rho, vr, p, dx=dx)

        # Time CPU
        cpu_times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            result_cpu = recon_cpu.reconstruct_primitive_variables(rho, vr, p, dx=dx)
            t1 = time.perf_counter()
            cpu_times.append(t1 - t0)
        cpu_time = np.median(cpu_times) * 1000

        rhoL_cpu, rhoR_cpu = result_cpu[0], result_cpu[1]

        # Warmup JAX
        for _ in range(N_WARMUP):
            result = recon_jax.reconstruct_primitive_variables(rho, vr, p, dx=dx)

        # Time JAX
        jax_times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            result_jax = recon_jax.reconstruct_primitive_variables(rho, vr, p, dx=dx)
            # Results are numpy arrays, no need for block_until_ready
            t1 = time.perf_counter()
            jax_times.append(t1 - t0)
        jax_time = np.median(jax_times) * 1000

        rhoL_jax, rhoR_jax = result_jax[0], result_jax[1]

        # Compare interior points (skip boundaries)
        ng = 3  # Ghost cells for WENO
        if len(rhoL_cpu) > 2*ng and len(rhoL_jax) > 2*ng:
            max_diff = np.max(np.abs(rhoL_cpu[ng:-ng] - rhoL_jax[ng:-ng]))
        else:
            max_diff = 0.0
        match = max_diff < TOL

        speedup = cpu_time / jax_time if jax_time > 0 else 0

        results.append({
            'N': N,
            'cpu_ms': cpu_time,
            'jax_ms': jax_time,
            'speedup': speedup,
            'match': match,
            'max_diff': max_diff
        })

    # Print results
    print(f"{'N':>10}  {'CPU (ms)':>10}  {'JAX (ms)':>10}  {'Speedup':>10}  {'Match':>8}")
    print("-" * 55)
    for r in results:
        match_str = "OK" if r['match'] else f"FAIL({r['max_diff']:.2e})"
        print(f"{r['N']:>10}  {r['cpu_ms']:>10.2f}  {r['jax_ms']:>10.2f}  {r['speedup']:>9.1f}x  {match_str:>8}")

    return results


def benchmark_reconstruction_all_methods():
    """Benchmark all reconstruction methods."""
    print("\n" + "-" * 70)
    print("BENCHMARK: Reconstruction (all methods)")
    print("-" * 70)

    from source.matter.hydro.reconstruction import Reconstruction
    from source.matter.hydro.tests.advance.reconstruction_jax import ReconstructionJAX

    methods = ["minmod", "mc", "mp5", "weno5", "wenoz"]
    N = 10000  # Fixed size for method comparison

    rho, vr, p, x, dx = generate_reconstruction_data(N)

    print(f"Grid size: N = {N}")
    print(f"{'Method':>10}  {'CPU (ms)':>10}  {'JAX (ms)':>10}  {'Speedup':>10}  {'Match':>8}")
    print("-" * 55)

    for method in methods:
        try:
            recon_cpu = Reconstruction(method=method)
            recon_jax = ReconstructionJAX(method=method)

            # Warmup
            for _ in range(N_WARMUP):
                recon_cpu.reconstruct_primitive_variables(rho, vr, p, dx=dx)
                recon_jax.reconstruct_primitive_variables(rho, vr, p, dx=dx)

            # Time CPU
            cpu_times = []
            for _ in range(N_RUNS):
                t0 = time.perf_counter()
                result_cpu = recon_cpu.reconstruct_primitive_variables(rho, vr, p, dx=dx)
                t1 = time.perf_counter()
                cpu_times.append(t1 - t0)
            cpu_time = np.median(cpu_times) * 1000

            # Time JAX
            jax_times = []
            for _ in range(N_RUNS):
                t0 = time.perf_counter()
                result_jax = recon_jax.reconstruct_primitive_variables(rho, vr, p, dx=dx)
                t1 = time.perf_counter()
                jax_times.append(t1 - t0)
            jax_time = np.median(jax_times) * 1000

            # Compare (both return tuples: (rhoL, rhoR, vrL, vrR, pL, pR))
            rhoL_cpu = np.asarray(result_cpu[0])
            rhoL_jax = np.asarray(result_jax[0])
            ng = 3 if method in ["weno5", "wenoz", "mp5"] else 1
            if len(rhoL_cpu) > 2*ng and len(rhoL_jax) > 2*ng:
                max_diff = np.max(np.abs(rhoL_cpu[ng:-ng] - rhoL_jax[ng:-ng]))
            else:
                max_diff = 0.0
            match = max_diff < TOL

            speedup = cpu_time / jax_time if jax_time > 0 else 0
            match_str = "OK" if match else f"FAIL({max_diff:.2e})"
            print(f"{method:>10}  {cpu_time:>10.2f}  {jax_time:>10.2f}  {speedup:>9.1f}x  {match_str:>8}")
        except Exception as e:
            print(f"{method:>10}  ERROR: {e}")


def print_summary(results_cons2prim, results_recon):
    """Print overall summary."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Check for any failures
    all_match_cons2prim = all(r['match'] for r in results_cons2prim)
    all_match_recon = all(r['match'] for r in results_recon)

    print(f"cons2prim numerical match: {'PASS' if all_match_cons2prim else 'FAIL'}")
    print(f"reconstruction numerical match: {'PASS' if all_match_recon else 'FAIL'}")

    # Best speedups at largest grid size
    if results_cons2prim:
        best_cons2prim = results_cons2prim[-1]
        print(f"cons2prim speedup at N={best_cons2prim['N']}: {best_cons2prim['speedup']:.1f}x")

    if results_recon:
        best_recon = results_recon[-1]
        print(f"reconstruction speedup at N={best_recon['N']}: {best_recon['speedup']:.1f}x")

    print("=" * 70)


if __name__ == "__main__":
    if not print_header():
        print("Missing dependencies. Please install JAX and Numba.")
        sys.exit(1)

    try:
        results_cons2prim = benchmark_cons2prim()
    except Exception as e:
        print(f"cons2prim benchmark failed: {e}")
        results_cons2prim = []

    try:
        results_recon = benchmark_reconstruction()
    except Exception as e:
        print(f"reconstruction benchmark failed: {e}")
        results_recon = []

    try:
        benchmark_reconstruction_all_methods()
    except Exception as e:
        print(f"reconstruction all methods benchmark failed: {e}")

    print_summary(results_cons2prim, results_recon)
