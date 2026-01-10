#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hydro-without-Hydro Test

Evolves BSSN with static matter sources from TOV solution.
The matter (hydro) variables are held fixed at the TOV equilibrium values,
while BSSN variables evolve. This tests that the spacetime remains stable
when sourced by a static perfect fluid.

Uses the same infrastructure as TOVEvolution.py for consistency.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d

# Locate repo root
def locate_repo_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (cand / 'source').is_dir():
            return cand
    return start

THIS = Path(__file__).resolve()
REPO = locate_repo_root(THIS.parent)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Core imports
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.grid import Grid
from source.core.statevector import StateVector
from source.core.rhsevolution import get_rhs
from source.backgrounds.sphericalbackground import FlatSphericalBackground

# BSSN
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import (
    NUM_BSSN_VARS,
    idx_phi, idx_hrr, idx_htt, idx_hpp,
    idx_K, idx_arr, idx_att, idx_app,
    idx_lambdar, idx_shiftr, idx_br, idx_lapse
)

# Hydro
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.atmosphere import AtmosphereParams

# TOV modules (same as TOVEvolution.py)
from examples.TOV.tov_solver import load_or_solve_tov_iso
import examples.TOV.tov_initial_data_interpolated as tov_id


class _DummyProgressBar:
    """Dummy progress bar for RHS calls."""
    def update(self, n):
        pass


def get_rhs_hwh(t, state_flat, grid, background, hydro, tov_solution, atmosphere):
    """
    RHS for Hydro-without-Hydro evolution.

    BSSN variables evolve normally, but hydro variables are reset to TOV
    equilibrium values after each RHS computation.
    """
    state = state_flat.reshape((grid.NUM_VARS, grid.N))

    # Reset matter to TOV equilibrium before computing RHS
    state = _reset_matter_to_tov(state, grid, hydro, tov_solution, atmosphere)

    # Compute full RHS
    dummy_progress = _DummyProgressBar()
    time_state = [0.0, 1.0]
    rhs_flat = get_rhs(t, state.flatten(), grid, background, hydro, dummy_progress, time_state)
    rhs = rhs_flat.reshape((grid.NUM_VARS, grid.N))

    # Zero out hydro RHS (matter stays frozen at TOV)
    rhs[NUM_BSSN_VARS:, :] = 0.0

    return rhs.flatten()


def _reset_matter_to_tov(state, grid, hydro, tov_solution, atmosphere):
    """Reset hydro variables to TOV equilibrium values."""
    r = grid.r

    # Interpolate TOV solution
    rho_interp = interp1d(tov_solution.r, tov_solution.rho_baryon,
                          kind='cubic', bounds_error=False,
                          fill_value=atmosphere.rho_floor)
    p_interp = interp1d(tov_solution.r, tov_solution.P,
                        kind='cubic', bounds_error=False,
                        fill_value=atmosphere.p_floor)

    rho0 = np.maximum(rho_interp(r), atmosphere.rho_floor)
    pressure = np.maximum(p_interp(r), atmosphere.p_floor)
    velocity = np.zeros_like(r)  # Static equilibrium

    # Convert to conservatives (v=0 -> W=1)
    eps = hydro.eos.eps_from_rho_p(rho0, pressure)
    h = 1.0 + eps + pressure / np.maximum(rho0, 1e-300)
    W = 1.0  # Static
    D = rho0 * W
    Sr = np.zeros_like(r)  # No momentum
    tau = rho0 * h * W**2 - pressure - D

    # Update state
    state[hydro.idx_D, :] = D
    state[hydro.idx_Sr, :] = Sr
    state[hydro.idx_tau, :] = tau

    return state


def rk4_step_hwh(state_flat, dt, grid, background, hydro, tov_solution, atmosphere):
    """
    Single RK4 timestep for Hydro-without-Hydro evolution.

    BSSN evolves, hydro stays frozen at TOV equilibrium.
    """
    # Stage 1
    k1 = get_rhs_hwh(0, state_flat, grid, background, hydro, tov_solution, atmosphere)

    # Stage 2
    state_2 = state_flat + 0.5 * dt * k1
    k2 = get_rhs_hwh(0, state_2, grid, background, hydro, tov_solution, atmosphere)

    # Stage 3
    state_3 = state_flat + 0.5 * dt * k2
    k3 = get_rhs_hwh(0, state_3, grid, background, hydro, tov_solution, atmosphere)

    # Stage 4
    state_4 = state_flat + dt * k3
    k4 = get_rhs_hwh(0, state_4, grid, background, hydro, tov_solution, atmosphere)

    # Combine
    state_new = state_flat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Ensure matter stays at TOV
    snew = state_new.reshape((grid.NUM_VARS, grid.N))
    snew = _reset_matter_to_tov(snew, grid, hydro, tov_solution, atmosphere)
    grid.fill_boundaries(snew)

    return snew.flatten()


def run_hwh_test(
    K=100.0,
    Gamma=2.0,
    rho_central=1.28e-3,
    r_max=16.0,
    num_points=100,
    t_final=50.0,
    cfl=0.1,
    progress=True,
    save_interval=50
):
    """
    Run Hydro-without-Hydro test.

    Parameters
    ----------
    K : float
        Polytropic constant
    Gamma : float
        Adiabatic index
    rho_central : float
        Central density
    r_max : float
        Outer boundary
    num_points : int
        Number of grid points
    t_final : float
        Final evolution time
    cfl : float
        CFL factor
    progress : bool
        Print progress
    save_interval : int
        Save snapshots every N steps

    Returns
    -------
    dict
        Results including time series and snapshots
    """
    print("="*70)
    print("HYDRO-WITHOUT-HYDRO TEST")
    print("="*70)
    print(f"  K = {K}, Gamma = {Gamma}")
    print(f"  rho_central = {rho_central:.2e}")
    print(f"  Grid: N = {num_points}, r_max = {r_max}")
    print(f"  Evolution: t_final = {t_final}, CFL = {cfl}")
    print("="*70)

    # Setup grid and EOS
    spacing = LinearSpacing(num_points, r_max)
    eos = IdealGasEOS(gamma=Gamma)

    # Atmosphere
    atmosphere = AtmosphereParams(
        rho_floor=1.0e-12,
        p_floor=K * (1.0e-12)**Gamma,
        v_max=0.999,
        W_max=100.0
    )

    # Hydro setup
    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere=atmosphere,
        reconstructor=create_reconstruction("wenoz"),
        riemann_solver=HLLRiemannSolver(atmosphere=atmosphere)
    )

    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    print(f"\nGrid created: N = {grid.N}, dr_min = {grid.min_dr:.4f}")

    # Solve TOV
    print("\nSolving TOV equations...")
    tov_solution = load_or_solve_tov_iso(
        K=K, Gamma=Gamma, rho_central=rho_central,
        r_max=r_max, accuracy="high"
    )
    print(f"  M_star = {tov_solution.M_star:.6f}")
    print(f"  R_iso = {tov_solution.R_iso:.3f}")
    print(f"  C = {tov_solution.C:.4f}")

    # Create initial data
    print("\nCreating initial data...")
    initial_state, prim_tuple = tov_id.create_initial_data_iso(
        tov_solution, grid, background, eos,
        atmosphere=atmosphere,
        polytrope_K=K, polytrope_Gamma=Gamma,
        interp_order=11
    )

    # Timestep
    dt = cfl * grid.min_dr
    num_steps = int(t_final / dt)
    print(f"\nTimestep: dt = {dt:.6f}")
    print(f"Total steps: {num_steps}")

    # Storage for time series
    center = NUM_GHOSTS
    times = [0.0]
    lapse_c = [initial_state[idx_lapse, center]]
    phi_c = [initial_state[idx_phi, center]]
    K_c = [initial_state[idx_K, center]]
    hrr_c = [initial_state[idx_hrr, center]]

    # Storage for snapshots
    times_detailed = [0.0]
    states_detailed = [initial_state.copy()]

    # Evolution
    print("\n" + "="*70)
    print("Starting HWH evolution...")
    print("="*70)

    state_flat = initial_state.flatten()
    t = 0.0

    for step in range(num_steps):
        state_flat = rk4_step_hwh(state_flat, dt, grid, background, hydro,
                                   tov_solution, atmosphere)
        t += dt

        # Extract current state
        state = state_flat.reshape((grid.NUM_VARS, grid.N))

        # Save time series every 10 steps
        if (step + 1) % 10 == 0:
            times.append(t)
            lapse_c.append(state[idx_lapse, center])
            phi_c.append(state[idx_phi, center])
            K_c.append(state[idx_K, center])
            hrr_c.append(state[idx_hrr, center])

        # Save snapshots
        if (step + 1) % save_interval == 0:
            times_detailed.append(t)
            states_detailed.append(state.copy())

        # Progress
        if progress and (step + 1) % 100 == 0:
            print(f"  step {step+1:5d}  t = {t:.3f}  "
                  f"alpha_c = {state[idx_lapse, center]:.6f}  "
                  f"phi_c = {state[idx_phi, center]:+.3e}  "
                  f"K_c = {state[idx_K, center]:+.3e}")

    # Final state
    final_state = state_flat.reshape((grid.NUM_VARS, grid.N))

    # Summary
    print("\n" + "="*70)
    print("HWH EVOLUTION COMPLETE")
    print("="*70)
    print(f"  Final time: t = {t:.3f}")
    print(f"  Steps: {num_steps}")
    print(f"\n  Central values:")
    print(f"    alpha: {lapse_c[0]:.6f} -> {lapse_c[-1]:.6f}  (delta = {lapse_c[-1]-lapse_c[0]:+.3e})")
    print(f"    phi:   {phi_c[0]:+.3e} -> {phi_c[-1]:+.3e}  (delta = {phi_c[-1]-phi_c[0]:+.3e})")
    print(f"    K:     {K_c[0]:+.3e} -> {K_c[-1]:+.3e}  (delta = {K_c[-1]-K_c[0]:+.3e})")
    print("="*70)

    return dict(
        time=t,
        steps=num_steps,
        r=grid.r,
        state=final_state,
        initial_state=initial_state,
        # Time series at center
        times=np.array(times),
        lapse_center=np.array(lapse_c),
        phi_center=np.array(phi_c),
        K_center=np.array(K_c),
        hrr_center=np.array(hrr_c),
        # Snapshots
        times_detailed=np.array(times_detailed),
        states_detailed=states_detailed,
        # TOV data
        tov=tov_solution,
        # Parameters
        K=K,
        Gamma=Gamma,
        rho_central=rho_central
    )


def plot_hwh_results(result, save_plots=True, plot_dir="plots"):
    """Plot HWH test results."""
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)

    times = result['times']
    r = result['r']
    tov = result['tov']

    # Plot 1: Time evolution at center
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # phi vs time
    axes[0].plot(times, result['phi_center'], 'b-', lw=2)
    axes[0].set_ylabel(r'$\phi$(center)')
    axes[0].set_title('HWH: Evolution at Center', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # K vs time (key diagnostic - should stay ~0)
    axes[1].plot(times, result['K_center'], 'g-', lw=2)
    axes[1].axhline(0, color='k', ls='--', alpha=0.5, label='K = 0')
    axes[1].set_ylabel('K(center)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # lapse vs time
    axes[2].plot(times, result['lapse_center'], 'r-', lw=2)
    axes[2].axhline(result['lapse_center'][0], color='r', ls='--', alpha=0.5, label='Initial')
    axes[2].set_xlabel('t')
    axes[2].set_ylabel(r'$\alpha$(center)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{plot_dir}/hwh_time_evolution.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Plot 2: Spatial profiles at different times
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    states = result['states_detailed']
    times_det = result['times_detailed']
    n_snap = min(4, len(states))
    indices = np.linspace(0, len(states)-1, n_snap, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_snap))

    for i, idx in enumerate(indices):
        state = states[idx]
        t = times_det[idx]

        axes[0, 0].plot(r, state[idx_lapse, :], color=colors[i], lw=1.5, label=f't={t:.1f}')
        axes[0, 1].plot(r, state[idx_phi, :], color=colors[i], lw=1.5, label=f't={t:.1f}')
        axes[1, 0].plot(r, state[idx_K, :], color=colors[i], lw=1.5, label=f't={t:.1f}')
        axes[1, 1].plot(r, state[idx_hrr, :], color=colors[i], lw=1.5, label=f't={t:.1f}')

    # Add TOV lapse
    alpha_tov = interp1d(tov.r, tov.alpha, kind='cubic', bounds_error=False,
                         fill_value=(tov.alpha[0], tov.alpha[-1]))(r)
    axes[0, 0].plot(r, alpha_tov, 'k--', lw=2, alpha=0.7, label='TOV')

    axes[0, 0].set_xlabel('r'); axes[0, 0].set_ylabel(r'$\alpha$')
    axes[0, 0].set_title('Lapse'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('r'); axes[0, 1].set_ylabel(r'$\phi$')
    axes[0, 1].set_title('Conformal Factor'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('r'); axes[1, 0].set_ylabel('K')
    axes[1, 0].set_title('Extrinsic Curvature K'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('r'); axes[1, 1].set_ylabel(r'$h_{rr}$')
    axes[1, 1].set_title('Metric Deviation'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('HWH: Spatial Profiles', fontsize=14)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{plot_dir}/hwh_spatial_profiles.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Plot 3: Drift analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    phi_drift = result['phi_center'] - result['phi_center'][0]
    K_drift = result['K_center'] - result['K_center'][0]
    lapse_drift = result['lapse_center'] - result['lapse_center'][0]
    hrr_drift = result['hrr_center'] - result['hrr_center'][0]

    axes[0, 0].plot(times, phi_drift, 'b-', lw=1.5)
    axes[0, 0].set_xlabel('t'); axes[0, 0].set_ylabel(r'$\Delta\phi$')
    axes[0, 0].set_title('Conformal Factor Drift'); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(times, K_drift, 'g-', lw=1.5)
    axes[0, 1].set_xlabel('t'); axes[0, 1].set_ylabel(r'$\Delta K$')
    axes[0, 1].set_title('K Drift (should be ~0)'); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(times, lapse_drift, 'r-', lw=1.5)
    axes[1, 0].set_xlabel('t'); axes[1, 0].set_ylabel(r'$\Delta\alpha$')
    axes[1, 0].set_title('Lapse Drift'); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(times, hrr_drift, 'm-', lw=1.5)
    axes[1, 1].set_xlabel('t'); axes[1, 1].set_ylabel(r'$\Delta h_{rr}$')
    axes[1, 1].set_title('Metric Drift'); axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('HWH: Drift Analysis', fontsize=14)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{plot_dir}/hwh_drift_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()

    return result


if __name__ == "__main__":
    # Run HWH test with same parameters as TOVEvolution reference
    result = run_hwh_test(
        K=100.0,
        Gamma=2.0,
        rho_central=0.2,
        r_max=8.0,
        num_points=300,
        t_final=100.0,
        cfl=0.1,
        save_interval=50,
        progress=True
    )

    # Generate plots
    plot_hwh_results(result, save_plots=True, plot_dir="plots")
