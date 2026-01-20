#!/usr/bin/env python3
"""
Quick profiling script - runs TOVEvolution with cProfile.
Modifies t_final to run just a few steps.
"""
import cProfile
import pstats
import io
import sys
import os

# Monkey-patch to limit evolution steps
original_t_final = None

def profile_tov():
    """Run short TOV evolution with profiling."""
    # Import and run (setup already handles imports)
    os.chdir('/home/davidbamba/repositories/engrenage/examples/TOV')
    sys.path.insert(0, '/home/davidbamba/repositories/engrenage')

    # Import the setup parts
    from examples.TOV.TOVEvolution import main

    # We can't easily modify t_final, so let's just profile setup + a few RHS evals
    import numpy as np
    from source.core.grid import Grid
    from source.core.spacing import LinearSpacing
    from source.core.statevector import StateVector
    from source.backgrounds.sphericalbackground import FlatSphericalBackground
    from source.bssn.bssnvars import BSSNVars
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS
    from source.matter.hydro.perfect_fluid import PerfectFluid
    from source.matter.hydro.eos import PolytropicEOS
    from source.matter.hydro.reconstruction import create_reconstruction
    from source.matter.hydro.riemann import LLFRiemannSolver
    from source.matter.hydro.atmosphere import AtmosphereParams
    from examples.TOV.tov_solver import load_or_solve_tov_iso
    import examples.TOV.tov_initial_data_interpolated as tov_id

    # Setup parameters (same as TOVEvolution)
    num_points = 5000  # Larger grid for parallel efficiency
    r_max = 16.0
    K = 100.0
    Gamma = 2.0
    rho_central = 1.28e-3

    ATMOSPHERE = AtmosphereParams(
        rho_floor=1.0e-10,
        p_floor=K*(1.0e-10)**Gamma,
        v_max=0.999,
        W_max=100.0,
        conservative_floor_safety=0.999
    )

    spacing = LinearSpacing(num_points, r_max)
    eos = PolytropicEOS(K=K, gamma=Gamma)
    base_recon = create_reconstruction('wenoz')

    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere=ATMOSPHERE,
        reconstructor=base_recon,
        riemann_solver=LLFRiemannSolver(atmosphere=ATMOSPHERE)
    )

    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    print("Loading TOV solution...")
    tov_solution = load_or_solve_tov_iso(K, Gamma, rho_central, r_max=r_max)

    print("Creating initial data...")
    state0, primitives = tov_id.create_initial_data_iso(
        tov_solution, grid, background, eos, ATMOSPHERE,
        polytrope_K=K, polytrope_Gamma=Gamma
    )

    # BSSN fixed data for Cowling
    bssn_fixed = state0[:NUM_BSSN_VARS, :].copy()
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(bssn_fixed)
    bssn_d1_fixed = grid.get_d1_metric_quantities(state0)

    # Define get_rhs_cowling locally
    def get_rhs_cowling(t, y):
        state = y.reshape((grid.NUM_VARS, grid.N))
        grid.fill_boundaries(state)

        bssn_vars_local = BSSNVars(grid.N)
        bssn_vars_local.set_bssn_vars(bssn_fixed)
        hydro.set_matter_vars(state, bssn_vars_local, grid)

        hydro_rhs = hydro.get_matter_rhs(grid.r, bssn_vars_local, bssn_d1_fixed, background)

        rhs = np.zeros_like(state)
        rhs[NUM_BSSN_VARS:, :] = hydro_rhs

        return rhs.flatten()

    # RK4 step
    def rk4_step(t, y, dt):
        k1 = get_rhs_cowling(t, y)
        k2 = get_rhs_cowling(t + 0.5*dt, y + 0.5*dt*k1)
        k3 = get_rhs_cowling(t + 0.5*dt, y + 0.5*dt*k2)
        k4 = get_rhs_cowling(t + dt, y + dt*k3)
        return y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Evolve
    dt = 0.1 * grid.min_dr
    y = state0.flatten()

    # Run 1000 steps for representative profiling
    n_steps = 1000
    print(f"\nProfiling {n_steps} evolution steps ({n_steps*4} RHS evaluations)...")
    print(f"Grid: N={grid.N}, dt={dt:.6f}")

    for step in range(n_steps):
        y = rk4_step(step*dt, y, dt)

    return y


if __name__ == "__main__":
    # Warmup (JIT compile)
    print("Warmup run (JIT compilation)...")
    _ = profile_tov()

    # Profiled run
    print("\n" + "="*70)
    print("PROFILED RUN")
    print("="*70)

    profiler = cProfile.Profile()
    profiler.enable()

    _ = profile_tov()

    profiler.disable()

    # Print results
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(50)
    print(s.getvalue())
