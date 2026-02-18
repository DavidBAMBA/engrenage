#!/usr/bin/env python3
"""
Oscillaton evolution using JAX backend.

Evolves a real scalar boson star (oscillaton) with scalar field matter
using BSSN equations, fully JIT-compiled via JAX.

Usage:
    cd examples && PYTHONPATH=.. python OscillatonEvolution_jax.py

Environment variables:
    T_FINAL: evolution time (default 10)
"""

import os
import sys
import time
import numpy as np

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', '..'))

# Ensure JAX 64-bit is enabled before any JAX import
os.environ.setdefault('ENGRENAGE_BACKEND', 'jax')

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Engrenage imports (NumPy-based, used for setup and diagnostics)
from source.initialdata.oscillatoninitialconditions import get_initial_state
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.bssn.constraintsdiagnostic import get_constraints_diagnostic
from source.core.grid import Grid
from source.core.spacing import CubicSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.matter.scalarmatter import ScalarMatter
from source.bssn.bssnstatevariables import (
    NUM_BSSN_VARS, idx_phi, idx_lapse,
    BSSN_PARITY, BSSN_ASYMP_POWER, BSSN_ASYMP_OFFSET,
)

# JAX BSSN imports
from source.bssn.jax.bssngeometry import build_bssn_background, build_derivative_matrices
from source.bssn.jax.boundaries_jax import fill_bssn_boundaries_jax
from source.core.rhsevolution_jax import get_rhs_bssn_scalar_jax


def main():
    print("=" * 60)
    print("  Oscillaton Evolution — JAX Backend")
    print("=" * 60)

    # =========================================================================
    # Configuration (matching the notebook)
    # =========================================================================
    scalar_mu = 1.0
    r_max = 64.0
    min_dr = 1.0 / 4.0
    max_dr = 2.0
    T_final = float(os.environ.get("T_FINAL", 10.0))
    eta = 1.0           # Gamma driver damping
    sigma_base = 1.0    # KO dissipation base coefficient
    num_points_t = 101  # Output time resolution

    # =========================================================================
    # Grid and initial data (NumPy)
    # =========================================================================
    my_matter = ScalarMatter(scalar_mu)
    my_state_vector = StateVector(my_matter)

    params = CubicSpacing.get_parameters(r_max, min_dr, max_dr)
    spacing = CubicSpacing(**params)
    grid = Grid(spacing, my_state_vector)
    r = grid.r
    N = grid.N
    num_vars = grid.NUM_VARS
    background = FlatSphericalBackground(r)

    print(f"  Grid: N={N}, r_max={r_max}, min_dr={min_dr:.4f}")
    print(f"  T_final={T_final}, num_vars={num_vars}")
    print(f"  CubicSpacing params: {params}")

    # Initial data (returns flattened 1D array)
    initial_state_flat = get_initial_state(grid, background)
    initial_state_2d = initial_state_flat.reshape(num_vars, -1)

    # Matter variable indices
    idx_u = my_matter.idx_u
    idx_v = my_matter.idx_v

    # =========================================================================
    # Transfer to JAX
    # =========================================================================
    print("\n[JAX] Transferring data to JAX arrays...")
    state_jax = jnp.array(initial_state_2d)
    bssn_bg = build_bssn_background(grid, background)
    deriv_mats = build_derivative_matrices(grid)
    dr = jnp.array(grid.dr)

    # =========================================================================
    # Define RK4 step
    # =========================================================================
    parity_jax = jnp.array(np.concatenate([np.array(BSSN_PARITY, dtype=np.float64),
                                            np.ones(num_vars - NUM_BSSN_VARS)]))
    asymp_power_jax = jnp.array(np.concatenate([np.array(BSSN_ASYMP_POWER, dtype=np.float64),
                                                  np.zeros(num_vars - NUM_BSSN_VARS)]))
    asymp_offset_jax = jnp.array(np.concatenate([np.array(BSSN_ASYMP_OFFSET, dtype=np.float64),
                                                    np.zeros(num_vars - NUM_BSSN_VARS)]))

    @jax.jit
    def apply_bcs(state):
        return fill_bssn_boundaries_jax(state, bssn_bg.r, NUM_GHOSTS,
                                         parity_jax, asymp_power_jax, asymp_offset_jax)

    @jax.jit
    def rhs_fn(state):
        return get_rhs_bssn_scalar_jax(
            state, bssn_bg, deriv_mats, dr,
            NUM_GHOSTS, num_vars,
            sigma_base, scalar_mu, eta,
            outer_bc_type="asymptotic",  # Oscillaton: asymptotic falloff
            fix_shift=False  # Oscillaton: use Gamma-driver for shift
        )

    @jax.jit
    def rk4_step(state, dt):
        k1 = rhs_fn(state)
        k2 = rhs_fn(state + 0.5 * dt * k1)
        k3 = rhs_fn(state + 0.5 * dt * k2)
        k4 = rhs_fn(state + dt * k3)
        return apply_bcs(state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

    # =========================================================================
    # JIT warmup
    # =========================================================================
    print("[JAX] JIT compiling (warmup)...")
    t_jit_start = time.time()

    dt_evolution = 0.4 * float(grid.min_dr)
    dt_jax = jnp.float64(dt_evolution)

    _ = rk4_step(state_jax, dt_jax).block_until_ready()
    t_jit_end = time.time()
    print(f"[JAX] JIT compilation took {t_jit_end - t_jit_start:.1f}s")

    # Benchmark
    t_bench_start = time.time()
    for _ in range(10):
        _ = rk4_step(state_jax, dt_jax).block_until_ready()
    t_bench_end = time.time()
    t_per_step = (t_bench_end - t_bench_start) / 10
    print(f"[JAX] Per-step time: {t_per_step*1000:.2f}ms")

    # =========================================================================
    # Evolution loop
    # =========================================================================
    n_steps = int(np.ceil(T_final / dt_evolution))
    dt_output = T_final / num_points_t

    print(f"\n[Evolution] T_final={T_final}, dt={dt_evolution:.6f}, "
          f"n_steps={n_steps}, output_interval={dt_output:.4f}")

    output_times = []
    output_states = []
    t_current = 0.0
    next_output_time = 0.0
    state = state_jax

    # Save initial state
    output_times.append(t_current)
    output_states.append(np.array(state))

    t_evol_start = time.time()

    for step in range(1, n_steps + 1):
        dt_actual = min(dt_evolution, T_final - t_current)
        if dt_actual <= 0:
            break
        dt_actual_jax = jnp.float64(dt_actual)

        state = rk4_step(state, dt_actual_jax)
        t_current += dt_actual

        # Save output at regular intervals
        if t_current >= next_output_time + dt_output - 1e-10:
            output_times.append(t_current)
            output_states.append(np.array(state))
            next_output_time = t_current

        # Progress
        if step % max(1, n_steps // 20) == 0 or step == n_steps:
            elapsed = time.time() - t_evol_start
            frac = t_current / T_final
            eta_remain = elapsed / max(frac, 1e-10) * (1 - frac)
            print(f"  t={t_current:.2f}/{T_final} ({frac*100:.0f}%) "
                  f"elapsed={elapsed:.1f}s ETA={eta_remain:.1f}s")

    state.block_until_ready()
    t_evol_end = time.time()
    print(f"\n[Evolution] Complete in {t_evol_end - t_evol_start:.1f}s "
          f"({n_steps} steps, {(t_evol_end - t_evol_start)/n_steps*1000:.2f}ms/step)")

    # =========================================================================
    # Diagnostics
    # =========================================================================
    print("\n[Diagnostics]")

    output_times = np.array(output_times)
    solution_flat = np.array([s.reshape(-1) for s in output_states])

    # Constraints
    print("  Computing constraints...")
    Ham, Mom = get_constraints_diagnostic(solution_flat, output_times, grid, background, my_matter)
    max_ham_final = np.max(np.abs(Ham[-1, NUM_GHOSTS:-NUM_GHOSTS]))
    print(f"  Max |Ham| at t={output_times[-1]:.2f}: {max_ham_final:.6e}")

    # =========================================================================
    # Plots
    # =========================================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Oscillaton Evolution (JAX)', fontsize=14)

        # (0,0) u and v at inner point vs time
        ax = axes[0, 0]
        idx_point = NUM_GHOSTS + 1
        r_point = r[idx_point]
        u_t = solution_flat[:, idx_u * N + idx_point]
        v_t = solution_flat[:, idx_v * N + idx_point]
        ax.plot(output_times, u_t, 'b-', label=f'u(r={r_point:.2f})')
        ax.plot(output_times, v_t, 'g-', label=f'v(r={r_point:.2f})')
        ax.set_xlabel('t')
        ax.set_ylabel('value')
        ax.legend()
        ax.grid(True)
        ax.set_title('Scalar field at inner point')

        # (0,1) u profile at several times
        ax = axes[0, 1]
        n_out = len(output_times)
        for i in range(0, n_out, max(1, n_out // 8)):
            t_i = output_times[i]
            if t_i > 0.0:
                u_profile = output_states[i][idx_u]
                ax.plot(r, u_profile, label=f't={t_i:.1f}')
        ax.set_xlabel('r')
        ax.set_ylabel('u')
        ax.set_xlim(0, 40)
        ax.legend(fontsize=8)
        ax.grid(True)
        ax.set_title('Scalar field u profile')

        # (1,0) Hamiltonian constraint
        ax = axes[1, 0]
        i_r_coord = 0  # coordinate index for radial
        for i in range(0, n_out, max(1, n_out // 6)):
            t_i = output_times[i]
            ax.plot(r, Ham[i], label=f't={t_i:.1f}')
        ax.set_xlabel('r')
        ax.set_ylabel('Ham')
        ax.set_xlim(0, 20)
        ax.set_ylim(-5e-5, 5e-5)
        ax.legend(fontsize=8)
        ax.grid(True)
        ax.set_title('Hamiltonian constraint')

        # (1,1) lapse and phi at inner point
        ax = axes[1, 1]
        phi_t = solution_flat[:, idx_phi * N + idx_point]
        lapse_t = solution_flat[:, idx_lapse * N + idx_point]
        ax.plot(output_times, phi_t, label=f'phi(r={r_point:.2f})')
        ax.plot(output_times, lapse_t, label=f'lapse(r={r_point:.2f})')
        ax.set_xlabel('t')
        ax.set_ylabel('value')
        ax.legend()
        ax.grid(True)
        ax.set_title('Gauge evolution at inner point')

        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(__file__), 'oscillaton_evolution_jax.png')
        plt.savefig(plot_path, dpi=150)
        print(f"\n[Plot] Saved to {plot_path}")
        plt.close()

    except ImportError:
        print("\n[Plot] matplotlib not available, skipping plots")

    print("\n" + "=" * 60)
    print("  Oscillaton Evolution JAX — Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
