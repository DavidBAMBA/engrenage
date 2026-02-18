#!/usr/bin/env python3
"""
Oscillaton evolution using Numba/NumPy backend (converted from OscillatonEvolution.ipynb).

Evolves a real scalar boson star (oscillaton) with scalar field matter
using BSSN equations with manual RK4 (fixed step).

Usage:
    cd examples && python BH/OscillatonEvolution_numba.py
"""

import os
import sys
import time
import numpy as np

# Ensure repo root is on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(script_dir, '..', '..')
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from source.initialdata.oscillatoninitialconditions import get_initial_state
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.bssn.constraintsdiagnostic import get_constraints_diagnostic
from source.core.rhsevolution import get_rhs
from source.core.grid import Grid
from source.core.spacing import CubicSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.matter.scalarmatter import ScalarMatter
from source.bssn.bssnstatevariables import idx_phi, idx_lapse


class _DummyBar:
    """Dummy progress bar for get_rhs."""
    def update(self, n):
        pass


def main():
    print("=" * 60)
    print("  Oscillaton Evolution — Numba/NumPy Backend")
    print("=" * 60)

    # =========================================================================
    # Configuration (same as OscillatonEvolution.ipynb / _jax.py)
    # =========================================================================
    scalar_mu = 1.0
    r_max = 64.0
    min_dr = 1.0 / 4.0
    max_dr = 2.0
    T = float(os.environ.get("T_FINAL", 10.0))
    num_points_t = 101

    # =========================================================================
    # Grid and initial data
    # =========================================================================
    my_matter = ScalarMatter(scalar_mu)
    my_state_vector = StateVector(my_matter)

    params = CubicSpacing.get_parameters(r_max, min_dr, max_dr)
    spacing = CubicSpacing(**params)
    grid = Grid(spacing, my_state_vector)
    r = grid.r
    num_points = r.size
    background = FlatSphericalBackground(r)

    print(f"  Grid: N={num_points}, r_max={r_max}, min_dr={min_dr}")
    print(f"  CubicSpacing params: {params}")

    initial_state = get_initial_state(grid, background)

    # Matter variable indices
    idx_u = my_matter.idx_u
    idx_v = my_matter.idx_v

    # =========================================================================
    # Evolution with manual RK4 (fixed step, same as JAX version)
    # =========================================================================
    dt = 0.4 * min_dr
    n_steps = int(np.ceil(T / dt))
    dt_output = T / num_points_t

    print(f"\n[Evolution] T={T}, dt={dt:.6f}, n_steps={n_steps}")
    print("[Evolution] Running manual RK4...")

    dummy_bar = _DummyBar()
    time_state = [0.0, 1e10]

    def rhs(state_flat):
        return get_rhs(0.0, state_flat, grid, background, my_matter,
                       dummy_bar, time_state)

    # Warmup (first call triggers Numba JIT)
    t_warmup_start = time.time()
    _ = rhs(initial_state)
    t_warmup = time.time() - t_warmup_start
    print(f"[Evolution] Numba warmup: {t_warmup:.1f}s")

    # Storage for outputs
    output_times = [0.0]
    output_states = [initial_state.copy()]
    t_current = 0.0
    next_output_time = 0.0
    state = initial_state.copy()

    t_evol_start = time.time()

    for step in range(1, n_steps + 1):
        dt_actual = min(dt, T - t_current)
        if dt_actual <= 0:
            break

        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt_actual * k1)
        k3 = rhs(state + 0.5 * dt_actual * k2)
        k4 = rhs(state + dt_actual * k3)
        state = state + (dt_actual / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t_current += dt_actual

        # Save output at regular intervals
        if t_current >= next_output_time + dt_output - 1e-10:
            output_times.append(t_current)
            output_states.append(state.copy())
            next_output_time = t_current

        # Progress
        if step % max(1, n_steps // 20) == 0 or step == n_steps:
            elapsed = time.time() - t_evol_start
            frac = t_current / T
            eta_remain = elapsed / max(frac, 1e-10) * (1 - frac)
            print(f"  t={t_current:.2f}/{T} ({frac*100:.0f}%) "
                  f"elapsed={elapsed:.1f}s ETA={eta_remain:.1f}s")

    t_evolution = time.time() - t_evol_start
    t_per_step = t_evolution / n_steps
    print(f"\n[Evolution] Complete in {t_evolution:.1f}s "
          f"({n_steps} steps, {t_per_step*1000:.2f}ms/step)")

    # Convert to arrays for diagnostics
    t_out = np.array(output_times)
    solution = np.array(output_states)

    # =========================================================================
    # Diagnostics
    # =========================================================================
    print("\n[Diagnostics]")

    print("  Computing constraints...")
    Ham, Mom = get_constraints_diagnostic(solution, t_out, grid, background, my_matter)
    max_ham_final = np.max(np.abs(Ham[-1, NUM_GHOSTS:-NUM_GHOSTS]))
    print(f"  Max |Ham| at t={t_out[-1]:.2f}: {max_ham_final:.6e}")

    # =========================================================================
    # Plots
    # =========================================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Oscillaton Evolution — Numba (T={T}, N={num_points})", fontsize=14)

        # (0,0) u and v at inner point vs time
        ax = axes[0, 0]
        idx_point = NUM_GHOSTS + 1
        r_point = r[idx_point]
        u_t = solution[:, idx_u * num_points + idx_point]
        v_t = solution[:, idx_v * num_points + idx_point]
        ax.plot(t_out, u_t, 'b-', label=f'u(r={r_point:.2f})')
        ax.plot(t_out, v_t, 'g-', label=f'v(r={r_point:.2f})')
        ax.set_xlabel('t')
        ax.set_ylabel('value')
        ax.legend()
        ax.grid(True)
        ax.set_title('Scalar field at inner point')

        # (0,1) u profile at several times
        ax = axes[0, 1]
        n_out = len(t_out)
        for i in range(0, n_out, max(1, n_out // 8)):
            if t_out[i] > 0.0:
                u_profile = solution[i, idx_u*num_points:(idx_u+1)*num_points]
                ax.plot(r, u_profile, label=f't={t_out[i]:.1f}')
        ax.set_xlabel('r')
        ax.set_ylabel('u')
        ax.set_xlim(0, 40)
        ax.legend(fontsize=8)
        ax.grid(True)
        ax.set_title('Scalar field u profile')

        # (1,0) Hamiltonian constraint
        ax = axes[1, 0]
        for i in range(0, n_out, max(1, n_out // 6)):
            ax.plot(r, Ham[i], label=f't={t_out[i]:.1f}')
        ax.set_xlabel('r')
        ax.set_ylabel('Ham')
        ax.set_xlim(0, 20)
        ax.set_ylim(-5e-5, 5e-5)
        ax.legend(fontsize=8)
        ax.grid(True)
        ax.set_title('Hamiltonian constraint')

        # (1,1) gauge evolution
        ax = axes[1, 1]
        phi_t = solution[:, idx_phi * num_points + idx_point]
        lapse_t = solution[:, idx_lapse * num_points + idx_point]
        ax.plot(t_out, phi_t, label=f'phi(r={r_point:.2f})')
        ax.plot(t_out, lapse_t, label=f'lapse(r={r_point:.2f})')
        ax.set_xlabel('t')
        ax.set_ylabel('value')
        ax.legend()
        ax.grid(True)
        ax.set_title('Gauge evolution at inner point')

        plt.tight_layout()
        plot_path = os.path.join(script_dir, 'oscillaton_evolution_numba.png')
        plt.savefig(plot_path, dpi=150)
        print(f"\n[Plot] Saved to {plot_path}")
        plt.close()
    except ImportError:
        print("\n[Plot] matplotlib not available")

    # =========================================================================
    # Timing summary
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"  Backend:         Numba/NumPy + manual RK4")
    print(f"  Grid points:     {num_points}")
    print(f"  T_final:         {T}")
    print(f"  dt:              {dt:.6f}")
    print(f"  Steps:           {n_steps}")
    print(f"  Evolution time:  {t_evolution:.1f}s")
    print(f"  ms/step:         {t_per_step*1000:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
