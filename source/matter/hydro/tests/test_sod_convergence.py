#!/usr/bin/env python3
"""
test_sod_convergence.py — Convergence test for spherical Sod problem

Tests convergence rates for the spherical Sod shock tube problem
using different reconstructors and resolutions.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Add source path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, repo_root)

# Engrenage core
from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground

# BSSN
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS, idx_phi, idx_K, idx_lapse

# Hydro
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.cons2prim import prim_to_cons
from source.matter.hydro.atmosphere import AtmosphereParams


def build_hydro_and_grid(n_interior=256, r_max=1.0, gamma=1.4, reconstructor="mp5",
                          spacetime_mode="fixed_minkowski"):
    """Build grid and hydro infrastructure using engrenage architecture."""
    # 1. Create spacing
    spacing = LinearSpacing(n_interior + 2 * NUM_GHOSTS, r_max)

    # 2. Create EOS
    eos = IdealGasEOS(gamma=gamma)

    # 3. Create atmosphere parameters
    atmosphere = AtmosphereParams(
        rho_floor=1e-13,
        p_floor=1e-15,
        v_max=0.999999,
        W_max=1e3
    )

    # 4. Create hydro object (PerfectFluid)
    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode=spacetime_mode,
        atmosphere=atmosphere,
        reconstructor=create_reconstruction(reconstructor),
        riemann_solver=HLLRiemannSolver()
    )
    # Force outflow boundary conditions (not parity) for 1D Minkowski test
    hydro.valencia.boundary_mode = "outflow"

    # 5. Create state vector
    state_vector = StateVector(hydro)

    # 6. Create grid
    grid = Grid(spacing, state_vector)

    # 7. Create background geometry
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    return grid, hydro, background


def fill_ghosts_primitives(rho, v, p, ng=NUM_GHOSTS):
    """Apply boundary conditions: parity at center, outflow at outer boundary."""
    N = len(rho)

    # Left boundary (center): parity conditions
    for i in range(ng):
        mir = 2 * ng - 1 - i
        rho[i] = rho[mir]   # even
        p[i] = p[mir]       # even
        v[i] = -v[mir]      # odd

    # Right boundary (outflow): zero-gradient
    last = N - ng - 1
    for k in range(1, ng + 1):
        idx = last + k
        rho[idx] = rho[last]
        p[idx] = p[last]
        v[idx] = v[last]
        

    return rho, v, p


def primitives_to_conservatives(rho0, vr, p, grid, hydro):
    """Convert primitives to conservatives using engrenage infrastructure."""
    gamma_rr = np.ones_like(rho0)  # Minkowski
    D, Sr, tau = prim_to_cons(rho0, vr, p, gamma_rr, hydro.eos)
    return D, Sr, tau


def conservatives_to_primitives(state_2d, grid, hydro, bssn_vars):
    """Convert conservatives to primitives using engrenage infrastructure."""
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    return hydro._get_primitives(bssn_vars, grid.r)


def max_signal_speed(primitives, eos, cfl_guard=1e-6):
    """Compute maximum signal speed for CFL condition."""
    rho0 = primitives['rho0']
    vr = primitives['vr']
    p = primitives['p']
    eps = eos.eps_from_rho_p(rho0, p)
    h = 1.0 + eps + p / np.maximum(rho0, 1e-300)
    cs2 = np.clip(eos.gamma * p / np.maximum(rho0 * h, 1e-300), 0.0, 1.0 - 1e-10)
    cs = np.sqrt(cs2)
    return np.max(np.abs(vr) + cs) + cfl_guard


def get_rhs_minkowski(state_flat, grid, hydro, background, bssn_fixed, bssn_d1_fixed):
    """Compute RHS for matter evolution in fixed Minkowski spacetime."""
    state_2d = state_flat.reshape((grid.NUM_VARS, grid.N))
    grid.fill_boundaries(state_2d)

    # Build BSSN vars (frozen Minkowski)
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(bssn_fixed)

    # Set matter vars and compute hydro RHS
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    hydro_rhs = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1_fixed, background)

    # Full RHS (BSSN frozen = 0, only hydro evolves)
    rhs_2d = np.zeros_like(state_2d)
    rhs_2d[NUM_BSSN_VARS:, :] = hydro_rhs

    return rhs_2d.flatten()


def rk3_step(state_2d, grid, hydro, background, bssn_fixed, bssn_d1_fixed, cfl=0.3):
    """Single RK3 (Shu-Osher) timestep for fixed Minkowski spacetime."""
    # Build BSSN vars for primitives computation
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(bssn_fixed)

    # Get primitives for timestep calculation
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    primitives = hydro._get_primitives(bssn_vars, grid.r)

    # Compute timestep
    amax = max_signal_speed(primitives, hydro.eos)
    dt = cfl * grid.min_dr / amax

    state_flat = state_2d.flatten()

    # RK3 Stage 1
    k1 = get_rhs_minkowski(state_flat, grid, hydro, background, bssn_fixed, bssn_d1_fixed)
    state1 = state_flat + dt * k1

    # RK3 Stage 2
    k2 = get_rhs_minkowski(state1, grid, hydro, background, bssn_fixed, bssn_d1_fixed)
    state2 = 0.75 * state_flat + 0.25 * (state1 + dt * k2)

    # RK3 Stage 3
    k3 = get_rhs_minkowski(state2, grid, hydro, background, bssn_fixed, bssn_d1_fixed)
    state_new_flat = (1.0/3.0) * state_flat + (2.0/3.0) * (state2 + dt * k3)

    state_new = state_new_flat.reshape((grid.NUM_VARS, grid.N))

    return state_new, dt


def run_sod_simulation(n_interior, reconstructor="mp5", gamma=1.4, Tfinal=0.2, cfl=0.1):
    """
    Run a spherical Sod simulation with given resolution.

    Args:
        n_interior: Number of interior grid points
        reconstructor: Reconstruction method
        gamma: Adiabatic index
        Tfinal: Final evolution time
        cfl: CFL factor

    Returns:
        Dictionary with results including r, rho, p, v arrays
    """
    # Build infrastructure
    grid, hydro, background = build_hydro_and_grid(
        n_interior=n_interior, r_max=1.0, gamma=gamma, reconstructor=reconstructor
    )

    # Initial conditions: discontinuity at mid-point
    r_mid = 0.5 * (grid.r[NUM_GHOSTS] + grid.r[-NUM_GHOSTS-1])
    rho0 = np.where(grid.r < r_mid, 10.0, 1.0)
    p = np.where(grid.r < r_mid, 40000.0/3.0, 1e-6)
    vr = np.zeros(grid.N)

    rho0, vr, p = fill_ghosts_primitives(rho0, vr, p)

    # Convert to conservatives
    D, Sr, tau = primitives_to_conservatives(rho0, vr, p, grid, hydro)

    # Create full state vector with Minkowski metric
    state_2d = np.zeros((grid.NUM_VARS, grid.N))
    state_2d[idx_lapse, :] = 1.0
    state_2d[idx_phi, :] = 0.0
    state_2d[idx_K, :] = 0.0
    state_2d[hydro.idx_D, :] = D
    state_2d[hydro.idx_Sr, :] = Sr
    state_2d[hydro.idx_tau, :] = tau

    # Fixed BSSN metric and derivatives
    bssn_fixed = state_2d[:NUM_BSSN_VARS, :].copy()
    bssn_d1_fixed = grid.get_d1_metric_quantities(state_2d)

    # Time evolution (no progress bar here - handled in main)
    t, steps = 0.0, 0
    max_steps = 2000000

    while t < Tfinal and steps < max_steps:
        state_2d, dt = rk3_step(state_2d, grid, hydro, background, bssn_fixed, bssn_d1_fixed, cfl=cfl)
        t += dt
        steps += 1

    # Extract final primitives
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(bssn_fixed)
    primitives = conservatives_to_primitives(state_2d, grid, hydro, bssn_vars)

    # Return results (interior only)
    ng = NUM_GHOSTS
    return {
        'r': grid.r[ng:-ng].copy(),
        'rho': primitives['rho0'][ng:-ng].copy(),
        'p': primitives['p'][ng:-ng].copy(),
        'v': primitives['vr'][ng:-ng].copy(),
        't_final': t,
        'steps': steps,
        'n_interior': n_interior,
        'reconstructor': reconstructor
    }


def compute_error_norms(solution, reference, r_ref):
    """
    Compute L1 and L2 error norms between solution and reference.

    Args:
        solution: Dictionary with r, rho, p, v arrays
        reference: Dictionary with reference solution
        r_ref: Common r grid for comparison

    Returns:
        Dictionary with L1 and L2 errors for each variable
    """
    # Interpolate both solutions to common grid
    interp_sol_rho = interp1d(solution['r'], solution['rho'], kind='linear',
                               bounds_error=False, fill_value='extrapolate')
    interp_sol_p = interp1d(solution['r'], solution['p'], kind='linear',
                             bounds_error=False, fill_value='extrapolate')
    interp_sol_v = interp1d(solution['r'], solution['v'], kind='linear',
                             bounds_error=False, fill_value='extrapolate')

    interp_ref_rho = interp1d(reference['r'], reference['rho'], kind='linear',
                               bounds_error=False, fill_value='extrapolate')
    interp_ref_p = interp1d(reference['r'], reference['p'], kind='linear',
                             bounds_error=False, fill_value='extrapolate')
    interp_ref_v = interp1d(reference['r'], reference['v'], kind='linear',
                             bounds_error=False, fill_value='extrapolate')

    # Evaluate on common grid
    rho_sol = interp_sol_rho(r_ref)
    p_sol = interp_sol_p(r_ref)
    v_sol = interp_sol_v(r_ref)

    rho_ref = interp_ref_rho(r_ref)
    p_ref = interp_ref_p(r_ref)
    v_ref = interp_ref_v(r_ref)

    # Compute errors
    dr = r_ref[1] - r_ref[0]  # Assuming uniform spacing for simplicity

    # Weight by r^2 for spherical geometry
    weight = 4.0 * np.pi * r_ref**2

    # L1 norms
    L1_rho = np.sum(weight * np.abs(rho_sol - rho_ref)) * dr
    L1_p = np.sum(weight * np.abs(p_sol - p_ref)) * dr
    L1_v = np.sum(weight * np.abs(v_sol - v_ref)) * dr

    # L2 norms
    L2_rho = np.sqrt(np.sum(weight * (rho_sol - rho_ref)**2) * dr)
    L2_p = np.sqrt(np.sum(weight * (p_sol - p_ref)**2) * dr)
    L2_v = np.sqrt(np.sum(weight * (v_sol - v_ref)**2) * dr)

    # Also compute L∞ (max) norm
    Linf_rho = np.max(np.abs(rho_sol - rho_ref))
    Linf_p = np.max(np.abs(p_sol - p_ref))
    Linf_v = np.max(np.abs(v_sol - v_ref))

    return {
        'L1': {'rho': L1_rho, 'p': L1_p, 'v': L1_v},
        'L2': {'rho': L2_rho, 'p': L2_p, 'v': L2_v},
        'Linf': {'rho': Linf_rho, 'p': Linf_p, 'v': Linf_v}
    }


def compute_convergence_rate(errors, resolutions):
    """
    Compute convergence rates from error norms.

    Args:
        errors: List of error dictionaries
        resolutions: List of resolution values

    Returns:
        Dictionary with convergence rates
    """
    rates = {}

    for norm in ['L1', 'L2', 'Linf']:
        rates[norm] = {}
        for var in ['rho', 'p', 'v']:
            # Compute convergence rate between consecutive resolutions
            conv_rates = []
            for i in range(len(errors) - 1):
                e1 = errors[i][norm][var]
                e2 = errors[i+1][norm][var]
                h1 = 1.0 / resolutions[i]
                h2 = 1.0 / resolutions[i+1]

                if e1 > 0 and e2 > 0:
                    rate = np.log(e1/e2) / np.log(h1/h2)
                    conv_rates.append(rate)

            rates[norm][var] = conv_rates

    return rates


def plot_convergence_results(resolutions, errors, rates, reconstructor):
    """Create convergence plots."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Variables to plot
    variables = ['rho', 'p', 'v']
    var_labels = ['Density ρ', 'Pressure p', 'Velocity v^r']

    # Plot error vs resolution (top row)
    for i, (var, label) in enumerate(zip(variables, var_labels)):
        ax = axes[0, i]

        # Extract errors for this variable
        L1_errors = [e['L1'][var] for e in errors]
        L2_errors = [e['L2'][var] for e in errors]
        Linf_errors = [e['Linf'][var] for e in errors]

        # Plot errors
        ax.loglog(resolutions, L1_errors, 'o-', label='L1', linewidth=2, markersize=8)
        ax.loglog(resolutions, L2_errors, 's-', label='L2', linewidth=2, markersize=8)
        ax.loglog(resolutions, Linf_errors, '^-', label='L∞', linewidth=2, markersize=8)

        # Add reference slopes
        N_ref = np.array(resolutions)
        if var == 'rho': 
            ax.loglog(N_ref, L1_errors[0] * (N_ref[0]/N_ref)**0.5, 'k:',
                     label='1st order (shock)', alpha=0.5)
        ax.loglog(N_ref, L2_errors[0] * (N_ref[0]/N_ref)**1.0, 'k--',
                 label='1st order', alpha=0.5)
        ax.loglog(N_ref, L2_errors[0] * (N_ref[0]/N_ref)**2.0, 'k-.',
                 label='2nd order', alpha=0.5)

        ax.set_xlabel('Resolution N')
        ax.set_ylabel('Error')
        ax.set_title(f'{label} - Error vs Resolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot convergence rates (bottom row)
    for i, (var, label) in enumerate(zip(variables, var_labels)):
        ax = axes[1, i]

        # Resolution pairs for x-axis
        N_pairs = [(resolutions[j], resolutions[j+1])
                   for j in range(len(resolutions)-1)]
        x_labels = [f"{n1}-{n2}" for n1, n2 in N_pairs]
        x_pos = np.arange(len(x_labels))

        # Get convergence rates
        L1_rates = rates['L1'][var]
        L2_rates = rates['L2'][var]
        Linf_rates = rates['Linf'][var]

        # Bar plot
        width = 0.25
        if L1_rates:
            ax.bar(x_pos - width, L1_rates, width, label='L1', color='tab:blue')
        if L2_rates:
            ax.bar(x_pos, L2_rates, width, label='L2', color='tab:orange')
        if Linf_rates:
            ax.bar(x_pos + width, Linf_rates, width, label='L∞', color='tab:green')

        # Add reference lines
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='1st order')
        ax.axhline(y=2.0, color='k', linestyle='-.', alpha=0.5, label='2nd order')

        ax.set_xlabel('Resolution Pair')
        ax.set_ylabel('Convergence Rate')
        ax.set_title(f'{label} - Convergence Rates')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 3])

    fig.suptitle(f'Convergence Analysis - Spherical Sod ({reconstructor.upper()})', fontsize=14)
    plt.tight_layout()
    filename = f'sod_spherical_convergence_{reconstructor}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nConvergence plot saved as: {filename}")


def plot_solution_comparison(solutions, reconstructor):
    """Plot solutions at different resolutions."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Variables to plot
    plot_data = [
        ('rho', 'Density ρ', axes[0]),
        ('p', 'Pressure p', axes[1]),
        ('v', 'Velocity v^r', axes[2])
    ]

    # Color map for different resolutions
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(solutions)))

    for var, label, ax in plot_data:
        for i, sol in enumerate(solutions):
            ax.plot(sol['r'], sol[var], '-',
                   label=f"N={sol['n_interior']}",
                   color=colors[i], linewidth=2, alpha=0.8)

        ax.set_xlabel('r')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])

    fig.suptitle(f'Spherical Sod Solutions at Different Resolutions ({reconstructor.upper()})',
                fontsize=14)
    plt.tight_layout()
    filename = f'sod_spherical_solutions_{reconstructor}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Solution comparison saved as: {filename}")


def run_single_resolution(args):
    """Wrapper function for parallel execution of simulations."""
    N, reconstructor, Tfinal, cfl = args
    return run_sod_simulation(N, reconstructor=reconstructor,
                            Tfinal=Tfinal, cfl=cfl)


def main():
    """Main convergence test."""

    print("="*60)
    print("SPHERICAL SOD CONVERGENCE TEST")
    print("="*60)

    # Test parameters
    reconstructors = ["minmod", "mp5", "wenoz"]  # Reconstruction methods to test
    resolutions = [50, 100, 200, 400, 800, 1600, 3200]  # Grid resolutions
    Tfinal = 0.2  # Final time
    cfl = 0.1     # CFL factor

    # Prepare all simulation arguments (all reconstructors x all resolutions)
    all_sim_args = []
    for reconstructor in reconstructors:
        for N in resolutions:
            all_sim_args.append((N, reconstructor, Tfinal, cfl))

    # Determine number of cores to use
    n_total_sims = len(all_sim_args)
    n_cores = min(n_total_sims, cpu_count())
    print(f"\nTotal simulations: {n_total_sims}")
    print(f"Using {n_cores} CPU cores for parallel execution")
    print(f"Available CPU cores: {cpu_count()}")
    print(f"\nRunning all simulations in parallel...\n")

    # Run ALL simulations in parallel with single progress bar
    all_solutions = []
    with Pool(processes=n_cores) as pool:
        with tqdm(total=n_total_sims, desc="Simulations",
                  bar_format='{desc}: {n}/{total} |{bar}| [{elapsed}<{remaining}]',
                  ncols=80) as pbar:
            for result in pool.imap_unordered(run_single_resolution, all_sim_args):
                all_solutions.append(result)
                pbar.update(1)
                # Show which simulation just completed
                pbar.set_postfix_str(f"Latest: {result['reconstructor'].upper()} N={result['n_interior']}")

    print("\n" + "="*60)
    print("All simulations completed!")
    print("="*60)

    # Group solutions by reconstructor
    solutions_by_reconstructor = {rec: [] for rec in reconstructors}
    for sol in all_solutions:
        solutions_by_reconstructor[sol['reconstructor']].append(sol)

    # Sort each group by resolution
    for rec in reconstructors:
        solutions_by_reconstructor[rec].sort(key=lambda x: x['n_interior'])

    # Run single reference solution (MP5 at N=12000)
    print(f"\nRunning reference solution (MP5 N=12000)...")
    with tqdm(total=1, desc="Reference", bar_format='{desc}: |{bar}| [{elapsed}]', ncols=80) as pbar:
        reference = run_sod_simulation(12000, reconstructor="mp5", Tfinal=Tfinal, cfl=cfl)
        pbar.update(1)
    print(f"Reference solution completed in {reference['steps']} steps, t_final={reference['t_final']:.4f}")

    # Process results for each reconstructor
    for reconstructor in reconstructors:
        print(f"\n{'='*60}")
        print(f"Processing {reconstructor.upper()} results")
        print('='*60)

        solutions = solutions_by_reconstructor[reconstructor]

        # Common grid for error computation (use coarsest resolution)
        r_common = solutions[0]['r']

        # Compute errors
        print("\nComputing error norms...")
        errors = []
        for sol in solutions:
            err = compute_error_norms(sol, reference, r_common)
            errors.append(err)

        # Compute convergence rates
        print("Computing convergence rates...")
        rates = compute_convergence_rate(errors, resolutions)

        # Print results
        print("\n" + "="*50)
        print(f"CONVERGENCE RESULTS - {reconstructor.upper()}")
        print("="*50)

        print("\nError Norms:")
        print("-"*50)
        for i, N in enumerate(resolutions):
            print(f"\nN = {N}:")
            for norm in ['L1', 'L2', 'Linf']:
                print(f"  {norm:4s}: ", end="")
                print(f"ρ={errors[i][norm]['rho']:.3e}, ", end="")
                print(f"p={errors[i][norm]['p']:.3e}, ", end="")
                print(f"v={errors[i][norm]['v']:.3e}")

        print("\nConvergence Rates:")
        print("-"*50)
        for norm in ['L1', 'L2', 'Linf']:
            print(f"\n{norm} convergence rates:")
            for j in range(len(resolutions)-1):
                print(f"  N={resolutions[j]}-{resolutions[j+1]}: ", end="")
                if rates[norm]['rho']:
                    print(f"ρ={rates[norm]['rho'][j]:.2f}, ", end="")
                    print(f"p={rates[norm]['p'][j]:.2f}, ", end="")
                    print(f"v={rates[norm]['v'][j]:.2f}")

        # Calculate average convergence rate (excluding first pair which may be noisy)
        if len(rates['L2']['rho']) > 1:
            avg_rate_rho = np.mean(rates['L2']['rho'][1:])
            avg_rate_p = np.mean(rates['L2']['p'][1:])
            avg_rate_v = np.mean(rates['L2']['v'][1:])
            print(f"\nAverage L2 convergence rate (high res):")
            print(f"  ρ: {avg_rate_rho:.2f}")
            print(f"  p: {avg_rate_p:.2f}")
            print(f"  v: {avg_rate_v:.2f}")

        # Create plots
        print("\nGenerating plots...")
        plot_convergence_results(resolutions, errors, rates, reconstructor)
        plot_solution_comparison(solutions, reconstructor)

    print("\n" + "="*60)
    print("CONVERGENCE TEST COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()