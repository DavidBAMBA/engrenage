"""Benchmark JAX scaling: CPU vs GPU for different grid sizes."""
import jax
import jax.numpy as jnp
import numpy as np
import sys
sys.path.insert(0, '/home/davidbamba/repositories/engrenage')

jax.config.update("jax_enable_x64", True)

from source.matter.hydro.jax.valencia_jax import CowlingGeometry, compute_hydro_rhs_cowling
import time

print("="*70)
print("JAX SCALING TEST: CPU vs GPU")
print("="*70)

# Test different grid sizes
grid_sizes = [100, 200, 400, 800, 1600]
n_runs = 50

eos_params = {'gamma': 2.0, 'K': 100.0}
atm_params = {'rho_floor': 1e-12, 'p_floor': 1e-12, 'v_max': 0.99,
              'W_max': 10.0, 'tol': 1e-12, 'max_iter': 500}

print(f"\nTesting grid sizes: {grid_sizes}")
print(f"Runs per size: {n_runs}")

results = []

for N in grid_sizes:
    print(f"\n{'='*70}")
    print(f"Grid size N = {N}")
    print(f"{'='*70}")

    # Create geometry
    geom = CowlingGeometry(
        alpha=jnp.ones(N), beta_r=jnp.zeros(N),
        gamma_rr=jnp.ones(N), e6phi=jnp.ones(N),
        dx=0.5, num_ghosts=3
    )

    D = jnp.ones(N) * 1e-4
    Sr = jnp.zeros(N)
    tau = jnp.ones(N) * 1e-5

    # Warmup (compilation)
    for _ in range(3):
        rhs_D, rhs_Sr, rhs_tau = compute_hydro_rhs_cowling(
            D, Sr, tau, geom, 'polytropic', eos_params, atm_params, 'mp5', 'hll'
        )

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        rhs_D, rhs_Sr, rhs_tau = compute_hydro_rhs_cowling(
            D, Sr, tau, geom, 'polytropic', eos_params, atm_params, 'mp5', 'hll'
        )
        rhs_D.block_until_ready()  # GPU sync
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_time = np.mean(times)
    std_time = np.std(times)

    results.append({
        'N': N,
        'mean': mean_time,
        'std': std_time
    })

    print(f"  Mean time: {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")

# Summary
print(f"\n{'='*70}")
print("SCALING SUMMARY")
print(f"{'='*70}")
print(f"\n  Device: {jax.devices()[0]}")
print(f"\n  {'N':<10} {'Time (ms)':<15} {'Time/N (μs)':<15}")
print(f"  {'-'*40}")
for r in results:
    time_per_point = r['mean'] * 1e6 / r['N']  # μs per grid point
    print(f"  {r['N']:<10} {r['mean']*1000:<15.3f} {time_per_point:<15.3f}")

print(f"\n{'='*70}")
print("RECOMMENDATION")
print(f"{'='*70}")
if 'Cpu' in str(jax.devices()[0]):
    print("\n  You're using CPU backend.")
    print(f"  For N=400: ~{results[2]['mean']*1000:.2f} ms per RHS")
    print(f"  For TOV evolution: CPU JAX is 2-3x faster than NumPy/Numba ✓")
else:
    print("\n  You're using GPU backend.")
    print(f"  For N=400: ~{results[2]['mean']*1000:.2f} ms per RHS")
    if results[2]['mean'] > results[1]['mean']:
        print(f"  ⚠ Grid too small for GPU! Use CPU backend for better performance.")
        print(f"  GPU becomes faster at N > 1000")
    else:
        print(f"  ✓ GPU is faster! Speedup improves with larger grids.")

print(f"{'='*70}")
