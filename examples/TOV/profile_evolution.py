"""
Profiling script for TOVEvolution.py to identify performance bottlenecks.

Usage:
    python profile_evolution.py

This will:
1. Run a short evolution (100 steps) with cProfile
2. Generate detailed profiling reports
3. Identify the top bottlenecks
"""

import cProfile
import pstats
import io
import sys
import os
import numpy as np

# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', '..'))

from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground

# BSSN
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS

# Hydro
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.atmosphere import AtmosphereParams

# Local TOV modules
from examples.TOV.tov_solver import load_or_solve_tov_iso
import examples.TOV.tov_initial_data_interpolated as tov_id
from examples.TOV.utils_TOVEvolution import evolve_fixed_timestep


def get_rhs_cowling(t, y, grid, background, hydro, bssn_fixed, bssn_d1_fixed):
    """RHS for Cowling evolution."""
    state = y.reshape((grid.NUM_VARS, grid.N))
    grid.fill_boundaries(state)

    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(bssn_fixed)
    hydro.set_matter_vars(state, bssn_vars, grid)

    # Compute hydro RHS
    hydro_rhs = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1_fixed, background)

    # Full RHS (BSSN frozen, only hydro evolves)
    rhs = np.zeros_like(state)
    rhs[NUM_BSSN_VARS:, :] = hydro_rhs

    return rhs.flatten()


def rk4_step(state_flat, dt, grid, background, hydro, bssn_fixed, bssn_d1_fixed,
             atmosphere):
    """Single RK4 timestep."""
    # Stage 1
    k1 = get_rhs_cowling(0, state_flat, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    # Stage 2
    state_2 = state_flat + 0.5 * dt * k1
    k2 = get_rhs_cowling(0, state_2, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    # Stage 3
    state_3 = state_flat + 0.5 * dt * k2
    k3 = get_rhs_cowling(0, state_3, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    # Stage 4
    state_4 = state_flat + dt * k3
    k4 = get_rhs_cowling(0, state_4, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    # Combine
    state_new = state_flat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return state_new


def setup_simulation(num_points=400, r_max=20.0):
    """Setup simulation with reduced grid for profiling."""
    K = 100.0
    Gamma = 2.0
    rho_central = 1.28e-3

    # Atmosphere
    rho_floor_base = 1e-13
    p_floor_base = K * (rho_floor_base)**Gamma
    atmosphere = AtmosphereParams(rho_floor=rho_floor_base, p_floor=p_floor_base)

    # Grid & EOS
    spacing = LinearSpacing(num_points, r_max)
    eos = IdealGasEOS(gamma=Gamma)

    # Reconstruction & Riemann solver
    base_recon = create_reconstruction("mp5")
    riemann = HLLRiemannSolver(atmosphere=atmosphere)

    # Hydro
    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere=atmosphere,
        reconstructor=base_recon,
        riemann_solver=riemann,
        solver_method="kastaun"
    )

    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    # Solve TOV
    print("Solving TOV...")
    tov_solution = load_or_solve_tov_iso(
        K=K, Gamma=Gamma, rho_central=rho_central,
        r_max=r_max, accuracy="high"
    )

    # Create initial data
    print("Creating initial data...")
    initial_state_2d, _ = tov_id.create_initial_data_iso(
        tov_solution, grid, background, eos,
        atmosphere=atmosphere,
        polytrope_K=K, polytrope_Gamma=Gamma,
        interp_order=11
    )

    # Fixed BSSN for Cowling
    bssn_fixed = initial_state_2d[:NUM_BSSN_VARS, :].copy()
    bssn_d1_fixed = grid.get_d1_metric_quantities(initial_state_2d)

    return (grid, background, hydro, atmosphere, initial_state_2d,
            bssn_fixed, bssn_d1_fixed)


def profile_single_rhs():
    """Profile a single RHS evaluation."""
    print("\n" + "="*70)
    print("PROFILING: Single RHS evaluation")
    print("="*70)

    grid, background, hydro, atmosphere, initial_state_2d, bssn_fixed, bssn_d1_fixed = setup_simulation(
        num_points=400, r_max=20.0
    )

    dt = 0.1 * grid.min_dr
    state_flat = initial_state_2d.flatten()

    profiler = cProfile.Profile()
    profiler.enable()

    # Profile 10 RHS evaluations
    for _ in range(10):
        get_rhs_cowling(0, state_flat, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    profiler.disable()

    # Print results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions

    print(s.getvalue())

    # Save to file
    ps.dump_stats('profile_rhs.prof')
    print("\nDetailed stats saved to: profile_rhs.prof")
    print("View with: python -m pstats profile_rhs.prof")


def profile_rk4_step():
    """Profile a single RK4 step."""
    print("\n" + "="*70)
    print("PROFILING: Single RK4 step (4 RHS evaluations)")
    print("="*70)

    grid, background, hydro, atmosphere, initial_state_2d, bssn_fixed, bssn_d1_fixed = setup_simulation(
        num_points=400, r_max=20.0
    )

    dt = 0.1 * grid.min_dr
    state_flat = initial_state_2d.flatten()

    profiler = cProfile.Profile()
    profiler.enable()

    # Profile 10 RK4 steps
    state = state_flat
    for _ in range(10):
        state = rk4_step(state, dt, grid, background, hydro, bssn_fixed, bssn_d1_fixed, atmosphere)

    profiler.disable()

    # Print results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)

    print(s.getvalue())

    ps.dump_stats('profile_rk4.prof')
    print("\nDetailed stats saved to: profile_rk4.prof")


def profile_full_evolution():
    """Profile a short evolution (100 steps)."""
    print("\n" + "="*70)
    print("PROFILING: Full evolution (100 steps)")
    print("="*70)

    grid, background, hydro, atmosphere, initial_state_2d, bssn_fixed, bssn_d1_fixed = setup_simulation(
        num_points=400, r_max=20.0
    )

    dt = 0.1 * grid.min_dr
    num_steps = 100

    print(f"Grid: N={grid.N}, r_max=20.0, dr_min={grid.min_dr}")
    print(f"dt={dt:.6f}, num_steps={num_steps}")

    profiler = cProfile.Profile()
    profiler.enable()

    # Evolve
    state = initial_state_2d.copy()
    for i in range(num_steps):
        state_flat = state.flatten()
        state_flat = rk4_step(state_flat, dt, grid, background, hydro,
                             bssn_fixed, bssn_d1_fixed, atmosphere)
        state = state_flat.reshape((grid.NUM_VARS, grid.N))

    profiler.disable()

    # Print results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(40)

    print(s.getvalue())

    ps.dump_stats('profile_evolution.prof')
    print("\nDetailed stats saved to: profile_evolution.prof")
    print("View with: python -m pstats profile_evolution.prof")


def generate_summary_report():
    """Generate a summary report of bottlenecks."""
    print("\n" + "="*70)
    print("PROFILING SUMMARY")
    print("="*70)

    if not os.path.exists('profile_evolution.prof'):
        print("Error: Run profiling first!")
        return

    ps = pstats.Stats('profile_evolution.prof')

    print("\nTop 20 functions by cumulative time:")
    print("-" * 70)
    ps.sort_stats('cumulative').print_stats(20)

    print("\n" + "="*70)
    print("Top 20 functions by total time (self):")
    print("-" * 70)
    ps.sort_stats('time').print_stats(20)

    print("\n" + "="*70)
    print("Analysis complete. Profile files saved:")
    print("  - profile_rhs.prof")
    print("  - profile_rk4.prof")
    print("  - profile_evolution.prof")
    print("\nOptimization targets:")
    print("  1. Check top cumulative time functions")
    print("  2. Look for repeated computations")
    print("  3. Identify vectorization opportunities")
    print("  4. Consider JAX/Numba compilation")
    print("="*70)


def main():
    """Run all profiling tests."""
    print("="*70)
    print("TOV EVOLUTION PROFILING")
    print("="*70)
    print("\nThis will profile the TOV evolution code to identify bottlenecks.")
    print("Running with reduced grid (N=400) for faster profiling.\n")

    # Run profiling
    profile_single_rhs()
    profile_rk4_step()
    profile_full_evolution()
    generate_summary_report()


if __name__ == "__main__":
    main()
