#!/usr/bin/env python3
"""
Profiling script for TOVEvolution.py

This script profiles the TOV evolution to identify performance bottlenecks.
Uses cProfile for function-level profiling and memory_profiler for memory usage.
"""

import sys
import os
import cProfile
import pstats
from io import StringIO
import numpy as np

# Add repo to path
sys.path.insert(0, '/home/yo/repositories/engrenage')

from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground

from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS

from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.atmosphere import AtmosphereParams
from source.matter.hydro.cons2prim import prim_to_cons

from examples.tov_solver import TOVSolver
import examples.tov_initial_data_interpolated as tov_id


def rk4_step(state_flat, dt, grid, background, hydro, bssn_fixed, bssn_d1_fixed, atmosphere):
    """Single RK4 step for profiling."""
    state_2d = state_flat.reshape((grid.NUM_VARS, grid.N))

    def compute_rhs(state):
        bssn_vars = BSSNVars(grid.N)
        bssn_vars.set_bssn_vars(bssn_fixed)
        hydro.set_matter_vars(state, bssn_vars, grid)
        rhs_D, rhs_Sr, rhs_tau = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1_fixed, background)

        rhs = np.zeros_like(state)
        rhs[hydro.idx_D, :] = rhs_D
        rhs[hydro.idx_Sr, :] = rhs_Sr
        rhs[hydro.idx_tau, :] = rhs_tau
        return rhs.flatten()

    state_flat = state_flat.flatten()

    # RK4 stages
    k1 = compute_rhs(state_2d)
    k2 = compute_rhs((state_flat + 0.5 * dt * k1).reshape((grid.NUM_VARS, grid.N)))
    k3 = compute_rhs((state_flat + 0.5 * dt * k2).reshape((grid.NUM_VARS, grid.N)))
    k4 = compute_rhs((state_flat + dt * k3).reshape((grid.NUM_VARS, grid.N)))

    return state_flat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def profile_tov_evolution(n_steps=10, r_max=16.0, num_points=400):
    """Profile TOV evolution with reduced resolution for faster profiling."""
    print("=" * 70)
    print("TOV EVOLUTION PROFILING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Grid points: {num_points}")
    print(f"  r_max: {r_max}")
    print(f"  Time steps: {n_steps}")
    print()

    # Parameters
    K = 100.0
    Gamma = 2.0
    rho_central = 1.28e-3

    # Atmosphere
    atmosphere = AtmosphereParams(
        rho_floor=1.0e-10,
        p_floor=1.0e-11,
        v_max=0.9999,
        W_max=100.0
    )

    # Grid setup
    print("Setting up grid...")
    spacing = LinearSpacing(num_points, r_max)
    eos = IdealGasEOS(gamma=Gamma)
    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere=atmosphere,
        reconstructor=create_reconstruction("wenoz"),
        riemann_solver=HLLRiemannSolver()
    )

    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    print(f"  Grid created: N={grid.N}, dr_min={grid.min_dr:.6f}")

    # TOV initial data
    print("Solving TOV equations...")
    tov_solver = TOVSolver(K=K, Gamma=Gamma)
    tov_solution = tov_solver.solve(rho_central, r_max=r_max)
    print(f"  TOV solved: M={tov_solution['M_star']:.6f}, R={tov_solution['R']:.3f}")

    print("Creating initial data...")
    initial_state_2d = tov_id.create_initial_data_interpolated(
        tov_solution, grid, background, eos,
        atmosphere=atmosphere,
        polytrope_K=K,
        polytrope_Gamma=Gamma,
        interp_order=5
    )
    print(f"  Initial state created")

    # Prepare for evolution
    bssn_fixed = initial_state_2d[:NUM_BSSN_VARS, :].copy()
    bssn_d1_fixed = grid.get_d1_metric_quantities(initial_state_2d)

    state_flat = initial_state_2d.flatten()
    dt = 0.1 * grid.min_dr

    print(f"  dt = {dt:.6f}")
    print(f"  State size: {state_flat.nbytes / 1e6:.2f} MB")
    print()

    # Evolution loop for profiling
    print(f"Running {n_steps} evolution steps for profiling...")
    print()

    return state_flat, dt, grid, background, hydro, bssn_fixed, bssn_d1_fixed, atmosphere, n_steps


def main():
    """Main profiling execution."""
    # Setup
    state_flat, dt, grid, background, hydro, bssn_fixed, bssn_d1_fixed, atmosphere, n_steps = \
        profile_tov_evolution(n_steps=10, r_max=16.0, num_points=400)

    # Profile evolution loop
    profiler = cProfile.Profile()

    def evolution_loop():
        state = state_flat.copy()
        for step in range(n_steps):
            state = rk4_step(state, dt, grid, background, hydro, bssn_fixed, bssn_d1_fixed, atmosphere)
            if (step + 1) % 3 == 0:
                print(f"  Step {step + 1}/{n_steps} completed")
        return state

    print("Starting profiling...")
    profiler.enable()
    state_final = evolution_loop()
    profiler.disable()

    print("\n" + "=" * 70)
    print("PROFILING RESULTS")
    print("=" * 70 + "\n")

    # Print detailed stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    print(s.getvalue())

    # Print by total time
    print("\n" + "=" * 70)
    print("TOP FUNCTIONS BY TOTAL TIME")
    print("=" * 70 + "\n")

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats(30)
    print(s.getvalue())

    # Summary statistics
    print("\n" + "=" * 70)
    print("PROFILE SUMMARY")
    print("=" * 70)

    stats = pstats.Stats(profiler)
    total_time = sum(stats.stats[key][3] for key in stats.stats)
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Number of steps: {n_steps}")
    print(f"Average time per step: {total_time/n_steps:.4f} seconds")

    # Find top 5 bottlenecks
    print(f"\nTop 5 time-consuming functions:")
    stats.sort_stats('tottime')
    count = 0
    for func, (cc, nc, tt, ct, callers) in sorted(stats.stats.items(), key=lambda x: x[1][3], reverse=True):
        if count >= 5:
            break
        if tt > 0.01:  # Only show if > 10ms
            print(f"  {count+1}. {func[2]} ({func[0]}:{func[1]})")
            print(f"     Total time: {tt:.4f}s ({tt/total_time*100:.1f}%)")
            print(f"     Calls: {cc} (ncalls: {nc})")
            print()
            count += 1

    print("=" * 70)


if __name__ == "__main__":
    main()
