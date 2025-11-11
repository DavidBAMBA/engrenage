#!/usr/bin/env python3
"""
Diagnostic script to investigate spurious momentum growth at stellar surface.

This script performs detailed analysis of the TOV initial data and evolution
to identify why velocity grows monotonically at r ≈ 9.55 (stellar surface).
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Engrenage core
sys.path.insert(0, '/home/yo/repositories/engrenage')
from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.backgrounds.sphericalbackground import FlatSphericalBackground, i_r, i_t, i_p

# BSSN
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import (NUM_BSSN_VARS, idx_phi, idx_hrr, idx_htt, idx_hpp,
                                             idx_K, idx_arr, idx_att, idx_app, idx_lapse)
from source.bssn.tensoralgebra import get_bar_gamma_LL

# Hydro
from source.matter.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import PolytropicEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.atmosphere import AtmosphereParams

# TOV
from examples.TOV.tov_solver import TOVSolver
import examples.TOV.tov_initial_data_interpolated as tov_id


def setup_system():
    """Setup grid, background, EOS, and hydro system."""
    # Grid parameters (matching TOVEvolution.py)
    N = 500
    r_max = 16.0
    spacing = LinearSpacing(N, r_max)

    # EOS
    K = 100.0
    Gamma = 2.0
    eos = PolytropicEOS(K, Gamma)

    # Atmosphere
    central_rho = 1.28e-3
    atmosphere = AtmosphereParams(
        rho_floor=1e-10 * central_rho,
        p_floor=1e-10,
        v_max=0.9999
    )

    # Hydro (create first, before grid)
    reconstructor = create_reconstruction("mp5")
    riemann_solver = HLLRiemannSolver(atmosphere=atmosphere)
    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere=atmosphere,
        reconstructor=reconstructor,
        riemann_solver=riemann_solver
    )

    # Create state vector and grid
    from source.core.statevector import StateVector
    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    return grid, background, eos, atmosphere, hydro


def generate_initial_data(grid, background, eos):
    """Generate TOV initial data."""
    print("=" * 80)
    print("GENERATING TOV INITIAL DATA")
    print("=" * 80)

    # Solve TOV
    central_rho = 1.28e-3
    K = 100.0
    Gamma = 2.0
    tov = TOVSolver(K, Gamma)
    tov_sol = tov.solve(central_rho)

    M_val = tov_sol['M'][-1] if hasattr(tov_sol['M'], '__len__') else tov_sol['M']
    R_val = tov_sol['R']
    C_val = tov_sol['C']
    print(f"TOV Solution: M={M_val:.6f}, R={R_val:.3f}, C={C_val:.4f}")

    # Create initial data
    initial_state = tov_id.create_initial_data_interpolated(
        grid, background, tov_sol, eos,
        atmosphere_rho=1e-10 * central_rho,
        interpolation_order=11
    )

    return initial_state, tov_sol


def analyze_initial_data_surface(state_2d, grid, background, hydro, tov_sol, eos):
    """
    PHASE 1: Analyze initial data at stellar surface.

    Check if we start in exact hydrostatic equilibrium.
    """
    print("\n" + "=" * 80)
    print("PHASE 1: INITIAL DATA ANALYSIS AT SURFACE")
    print("=" * 80)

    # Setup BSSN vars
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])

    # Get primitives
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    prim = hydro._get_primitives(bssn_vars, grid.r)

    # Find surface
    R_star = tov_sol['R']
    interior = prim['rho0'] > 1e-6
    i_surf = np.where(interior)[0][-1] if np.any(interior) else N // 2
    r_surf = grid.r[i_surf]

    print(f"\nStellar surface:")
    print(f"  TOV R = {R_star:.6f}")
    print(f"  Grid surface at i={i_surf}, r={r_surf:.6f}")

    # Extract metric quantities at surface region
    i_start = max(NUM_GHOSTS, i_surf - 10)
    i_end = min(grid.N - NUM_GHOSTS, i_surf + 10)
    surface_slice = slice(i_start, i_end)

    r = grid.r[surface_slice]
    rho0 = prim['rho0'][surface_slice]
    P = prim['p'][surface_slice]
    vr = prim['vr'][surface_slice]

    # Extract BSSN variables
    phi = state_2d[idx_phi, surface_slice]
    hrr = state_2d[idx_hrr, surface_slice]
    lapse = state_2d[idx_lapse, surface_slice]

    # Physical metric
    ghat_rr = background.hat_gamma_LL[surface_slice, i_r, i_r]
    Re_rr = background.scaling_matrix[surface_slice, i_r, i_r]
    gammabar_rr = ghat_rr + Re_rr * hrr
    gamma_rr = np.exp(4.0 * phi) * gammabar_rr

    print(f"\n{'i':<6} {'r':<10} {'rho0':<12} {'P':<12} {'vr':<12} {'alpha':<10} {'e^(4phi)':<10} {'gamma_rr':<10}")
    print("-" * 100)
    for i in range(len(r)):
        idx = i_start + i
        print(f"{idx:<6} {r[i]:<10.6f} {rho0[i]:<12.4e} {P[i]:<12.4e} {vr[i]:<12.4e} "
              f"{lapse[i]:<10.6f} {np.exp(4*phi[i]):<10.6f} {gamma_rr[i]:<10.6f}")

    # Check pressure gradient vs metric forces
    print("\n" + "-" * 80)
    print("HYDROSTATIC EQUILIBRIUM CHECK")
    print("-" * 80)

    # Compute gradients (centered differences in interior)
    dr = np.diff(r)
    dP_dr = np.zeros_like(P)
    dalpha_dr = np.zeros_like(lapse)
    dphi_dr = np.zeros_like(phi)

    # Interior points (not at boundaries of this window)
    for i in range(1, len(r) - 1):
        dP_dr[i] = (P[i+1] - P[i-1]) / (r[i+1] - r[i-1])
        dalpha_dr[i] = (lapse[i+1] - lapse[i-1]) / (r[i+1] - r[i-1])
        dphi_dr[i] = (phi[i+1] - phi[i-1]) / (r[i+1] - r[i-1])

    # Specific enthalpy
    eps = np.array([eos.eps_from_rho_p(rho0[i], P[i]) for i in range(len(rho0))])
    h = 1.0 + eps + P / np.maximum(rho0, 1e-30)

    # Hydrostatic balance for static TOV (v=0):
    # ∂_r P = -(ρ₀ + ρ₀ ε + P) × (∂_r α / α)
    # In index form: ∂_r P + (ρ₀ h) × (∂_r α / α) = 0

    metric_force = (rho0 * h) * (dalpha_dr / np.maximum(lapse, 1e-30))
    imbalance = dP_dr + metric_force

    print(f"\n{'i':<6} {'r':<10} {'dP/dr':<14} {'metric_force':<14} {'imbalance':<14} {'imb/P':<12}")
    print("-" * 90)
    for i in range(len(r)):
        idx = i_start + i
        rel_imb = imbalance[i] / np.maximum(abs(P[i]), 1e-30) if abs(P[i]) > 1e-30 else 0.0
        print(f"{idx:<6} {r[i]:<10.6f} {dP_dr[i]:<14.6e} {metric_force[i]:<14.6e} "
              f"{imbalance[i]:<14.6e} {rel_imb:<12.2e}")

    return i_surf, r_surf


def analyze_rhs_at_surface(state_2d, grid, background, hydro, i_surf):
    """
    PHASE 2: Compute and analyze RHS at t=0 near surface.

    Decompose dS_r/dt into flux divergence, source terms, and connection terms.
    """
    print("\n" + "=" * 80)
    print("PHASE 2: RHS ANALYSIS AT t=0")
    print("=" * 80)

    # Build BSSN containers
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
    bssn_d1 = grid.get_d1_metric_quantities(state_2d)

    # Set matter vars
    hydro.set_matter_vars(state_2d, bssn_vars, grid)

    # Compute full RHS
    rhs_D, rhs_Sr, rhs_tau = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1, background)

    # Focus on surface region
    i_start = max(NUM_GHOSTS, i_surf - 10)
    i_end = min(grid.N - NUM_GHOSTS, i_surf + 10)

    print(f"\nRHS of momentum equation dS_r/dt near surface (i={i_surf}):")
    print(f"{'i':<6} {'r':<10} {'dS_r/dt':<14}")
    print("-" * 40)
    for i in range(i_start, i_end):
        print(f"{i:<6} {grid.r[i]:<10.6f} {rhs_Sr[i]:<14.6e}")

    # Find max
    max_idx = i_start + np.argmax(np.abs(rhs_Sr[i_start:i_end]))
    print(f"\nMaximum |dS_r/dt| at i={max_idx}, r={grid.r[max_idx]:.6f}: {rhs_Sr[max_idx]:.6e}")

    # TODO: Decompose into individual terms
    # This requires accessing internal hydro components
    # For now, we report the total

    return rhs_Sr


def analyze_evolution_single_step(state_2d, grid, background, hydro, atmosphere, dt, i_surf):
    """
    PHASE 6: Evolve exactly 1 timestep and track changes at surface.
    """
    print("\n" + "=" * 80)
    print("PHASE 6: SINGLE TIMESTEP EVOLUTION")
    print("=" * 80)

    # Extract initial momentum at surface
    Sr_initial = state_2d[NUM_BSSN_VARS + 1, :].copy()

    # Simple forward Euler for debugging (NOT RK4)
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
    bssn_d1 = grid.get_d1_metric_quantities(state_2d)

    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    rhs_D, rhs_Sr, rhs_tau = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1, background)

    # Update (Euler)
    state_new = state_2d.copy()
    state_new[NUM_BSSN_VARS + 0, :] += dt * rhs_D
    state_new[NUM_BSSN_VARS + 1, :] += dt * rhs_Sr
    state_new[NUM_BSSN_VARS + 2, :] += dt * rhs_tau

    # Fill boundaries
    grid.fill_boundaries(state_new)

    # Extract new momentum
    Sr_final = state_new[NUM_BSSN_VARS + 1, :]

    # Changes
    delta_Sr = Sr_final - Sr_initial

    # Surface region
    i_start = max(NUM_GHOSTS, i_surf - 10)
    i_end = min(grid.N - NUM_GHOSTS, i_surf + 10)

    print(f"\nMomentum change after dt={dt:.6e}:")
    print(f"{'i':<6} {'r':<10} {'Sr_init':<14} {'Sr_final':<14} {'ΔSr':<14} {'dS/dt':<14}")
    print("-" * 90)
    for i in range(i_start, i_end):
        print(f"{i:<6} {grid.r[i]:<10.6f} {Sr_initial[i]:<14.6e} {Sr_final[i]:<14.6e} "
              f"{delta_Sr[i]:<14.6e} {rhs_Sr[i]:<14.6e}")

    # Check if ΔSr ≈ dt * dS/dt
    print("\nConsistency check: ΔSr vs dt × dS/dt")
    for i in range(i_start, i_end):
        expected = dt * rhs_Sr[i]
        actual = delta_Sr[i]
        ratio = actual / expected if abs(expected) > 1e-30 else 0.0
        print(f"  i={i}, r={grid.r[i]:.6f}: expected={expected:.6e}, actual={actual:.6e}, ratio={ratio:.4f}")


def plot_surface_diagnostics(state_2d, grid, background, hydro, tov_sol, i_surf):
    """Generate diagnostic plots focused on surface region."""
    print("\n" + "=" * 80)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("=" * 80)

    # Setup
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
    bssn_d1 = grid.get_d1_metric_quantities(state_2d)
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    prim = hydro._get_primitives(bssn_vars, grid.r)

    # Compute RHS
    rhs_D, rhs_Sr, rhs_tau = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1, background)

    # Focus on surface region
    R_star = tov_sol['R']
    r_window = (grid.r > R_star - 1.0) & (grid.r < R_star + 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Primitives
    ax = axes[0, 0]
    ax.semilogy(grid.r[r_window], prim['rho0'][r_window], 'b-', label='ρ₀')
    ax.semilogy(grid.r[r_window], prim['p'][r_window], 'r-', label='P')
    ax.axvline(R_star, color='k', linestyle='--', alpha=0.5, label=f'R={R_star:.3f}')
    ax.set_xlabel('r')
    ax.set_ylabel('ρ₀, P')
    ax.set_title('Primitives at Surface')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Velocity
    ax = axes[0, 1]
    ax.plot(grid.r[r_window], prim['vr'][r_window], 'g-', marker='o', markersize=3)
    ax.axvline(R_star, color='k', linestyle='--', alpha=0.5)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('r')
    ax.set_ylabel('vʳ')
    ax.set_title('Radial Velocity at t=0')
    ax.grid(True, alpha=0.3)

    # Plot 3: Metric quantities
    ax = axes[1, 0]
    phi = state_2d[idx_phi, r_window]
    lapse = state_2d[idx_lapse, r_window]
    ax.plot(grid.r[r_window], lapse, 'b-', label='α (lapse)')
    ax.plot(grid.r[r_window], np.exp(4*phi), 'r-', label='e^(4φ)')
    ax.axvline(R_star, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('r')
    ax.set_ylabel('Metric quantities')
    ax.set_title('Metric at Surface')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: RHS
    ax = axes[1, 1]
    ax.plot(grid.r[r_window], rhs_Sr[r_window], 'r-', marker='o', markersize=3)
    ax.axvline(R_star, color='k', linestyle='--', alpha=0.5)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('r')
    ax.set_ylabel('dS_r/dt')
    ax.set_title('Momentum RHS at t=0')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('surface_diagnostics.png', dpi=150, bbox_inches='tight')
    print("Saved: surface_diagnostics.png")


def main():
    """Main diagnostic routine."""
    print("=" * 80)
    print("TOV SURFACE MOMENTUM DIAGNOSTIC")
    print("=" * 80)

    # Setup
    grid, background, eos, atmosphere, hydro = setup_system()

    # Generate initial data
    state_2d, tov_sol = generate_initial_data(grid, background, eos)

    # Phase 1: Analyze initial data at surface
    i_surf, r_surf = analyze_initial_data_surface(state_2d, grid, background, hydro, tov_sol, eos)

    # Phase 2: Analyze RHS at surface
    rhs_Sr = analyze_rhs_at_surface(state_2d, grid, background, hydro, i_surf)

    # Estimate dt
    from source.matter.hydro.riemann import HLLRiemannSolver
    dt_max = hydro.riemann_solver.estimate_dt(
        grid.r,
        state_2d[idx_lapse, :],
        np.ones(grid.N),  # gamma_rr (approximate)
        np.zeros(grid.N),  # beta_r
        cfl=0.1
    )
    dt = 0.5 * dt_max  # Conservative
    print(f"\nUsing dt = {dt:.6e}")

    # Phase 6: Single step evolution
    analyze_evolution_single_step(state_2d, grid, background, hydro, atmosphere, dt, i_surf)

    # Generate plots
    plot_surface_diagnostics(state_2d, grid, background, hydro, tov_sol, i_surf)

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
