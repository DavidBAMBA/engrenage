"""
Legacy debugging plotting utilities for TOV evolution.

These functions are kept for debugging purposes. They may be useful
for inspecting the initial stages of simulation or troubleshooting.
"""

import numpy as np
import os
import matplotlib.pyplot as plt


def get_plots_dir():
    """Get plots directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(os.path.dirname(script_dir), 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir


def plot_first_step(state_t0, state_t1, grid, hydro, tov_solution=None, suffix=""):
    """Plot t=0 vs t=1×dt to inspect the first update."""
    from source.bssn.bssnvars import BSSNVars
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS

    bssn_0 = BSSNVars(grid.N)
    bssn_0.set_bssn_vars(state_t0[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t0, bssn_0, grid)
    rho0_0, vr_0, p_0, eps_0, W_0, h_0, _ = hydro._get_primitives(bssn_0, grid.r)

    bssn_1 = BSSNVars(grid.N)
    bssn_1.set_bssn_vars(state_t1[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t1, bssn_1, grid)
    rho0_1, vr_1, p_1, eps_1, W_1, h_1, _ = hydro._get_primitives(bssn_1, grid.r)

    r = grid.r

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].plot(r, rho0_0, 'b-', label='t=0', linewidth=1.5)
    axes[0, 0].plot(r, rho0_1, 'r--', label='t=dt', linewidth=1.5)
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel('rho0')
    axes[0, 0].set_title('Baryon Density')
    axes[0, 0].legend()

    axes[0, 1].plot(r, p_0, 'b-', label='t=0', linewidth=1.5)
    axes[0, 1].plot(r, p_1, 'r--', label='t=dt', linewidth=1.5)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure')
    axes[0, 1].legend()

    axes[0, 2].plot(r, vr_0, 'b-', label='t=0', linewidth=1.5)
    axes[0, 2].plot(r, vr_1, 'r--', label='t=dt', linewidth=1.5)
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel('v^r')
    axes[0, 2].set_title('Radial Velocity')
    axes[0, 2].legend()

    D_0 = state_t0[hydro.idx_D, :]
    D_1 = state_t1[hydro.idx_D, :]
    Sr_0 = state_t0[hydro.idx_Sr, :]
    Sr_1 = state_t1[hydro.idx_Sr, :]
    tau_0 = state_t0[hydro.idx_tau, :]
    tau_1 = state_t1[hydro.idx_tau, :]

    axes[1, 0].plot(r, D_0, 'b-', label='t=0', linewidth=1.5)
    axes[1, 0].plot(r, D_1, 'r--', label='t=dt', linewidth=1.5)
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel('D')
    axes[1, 0].set_title('Conserved D')
    axes[1, 0].legend()

    axes[1, 1].plot(r, Sr_0, 'b-', label='t=0', linewidth=1.5)
    axes[1, 1].plot(r, Sr_1, 'r--', label='t=dt', linewidth=1.5)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel('Sr')
    axes[1, 1].set_title('Conserved Sr')
    axes[1, 1].legend()

    axes[1, 2].plot(r, tau_0, 'b-', label='t=0', linewidth=1.5)
    axes[1, 2].plot(r, tau_1, 'r--', label='t=dt', linewidth=1.5)
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel('tau')
    axes[1, 2].set_title('Conserved tau')
    axes[1, 2].legend()

    plt.tight_layout()
    out_path = os.path.join(get_plots_dir(), f'tov_first_step{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_center_zoom(state_t0, state_t1, grid, hydro, window=0.5, suffix=""):
    """Plot zoom near origin comparing t=0 vs t=dt."""
    from source.bssn.bssnvars import BSSNVars
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS

    bssn_0 = BSSNVars(grid.N)
    bssn_0.set_bssn_vars(state_t0[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t0, bssn_0, grid)
    rho0_0, vr_0, p_0, eps_0, W_0, h_0, _ = hydro._get_primitives(bssn_0, grid.r)

    bssn_1 = BSSNVars(grid.N)
    bssn_1.set_bssn_vars(state_t1[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t1, bssn_1, grid)
    rho0_1, vr_1, p_1, eps_1, W_1, h_1, _ = hydro._get_primitives(bssn_1, grid.r)

    r = grid.r
    mask = r <= window

    D_0 = state_t0[hydro.idx_D, :]
    D_1 = state_t1[hydro.idx_D, :]
    Sr_0 = state_t0[hydro.idx_Sr, :]
    Sr_1 = state_t1[hydro.idx_Sr, :]
    tau_0 = state_t0[hydro.idx_tau, :]
    tau_1 = state_t1[hydro.idx_tau, :]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].plot(r[mask], rho0_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[0, 0].plot(r[mask], rho0_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel(r'$\rho_0$')
    axes[0, 0].set_title('Baryon Density (center)')
    axes[0, 0].legend()

    axes[0, 1].plot(r[mask], p_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[0, 1].plot(r[mask], p_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure (center)')
    axes[0, 1].legend()

    axes[0, 2].plot(r[mask], vr_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[0, 2].plot(r[mask], vr_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel(r'$v^r$')
    axes[0, 2].set_title('Radial Velocity (center)')
    axes[0, 2].legend()

    axes[1, 0].plot(r[mask], D_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[1, 0].plot(r[mask], D_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel('D')
    axes[1, 0].set_title('Conserved D (center)')
    axes[1, 0].legend()

    axes[1, 1].plot(r[mask], Sr_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[1, 1].plot(r[mask], Sr_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel(r'$S_r$')
    axes[1, 1].set_title('Conserved Sr (center)')
    axes[1, 1].legend()

    axes[1, 2].plot(r[mask], tau_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[1, 2].plot(r[mask], tau_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel(r'$\tau$')
    axes[1, 2].set_title('Conserved tau (center)')
    axes[1, 2].legend()

    plt.suptitle(f'Center Zoom: r ∈ [0, {window}]', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(get_plots_dir(), f'tov_center_zoom{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_surface_zoom(tov_solution, state_t0, state_t1, grid, hydro, primitives_t0=None, window=0.5, suffix=""):
    """Plot zoom near stellar surface."""
    from source.bssn.bssnvars import BSSNVars
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS

    if hasattr(tov_solution, 'R'):
        R_star = tov_solution.R
    else:
        R_star = tov_solution['R']

    bssn_0 = BSSNVars(grid.N)
    bssn_0.set_bssn_vars(state_t0[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t0, bssn_0, grid)
    rho0_0, vr_0, p_0, eps_0, W_0, h_0, _ = hydro._get_primitives(bssn_0, grid.r)

    bssn_1 = BSSNVars(grid.N)
    bssn_1.set_bssn_vars(state_t1[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t1, bssn_1, grid)
    rho0_1, vr_1, p_1, eps_1, W_1, h_1, _ = hydro._get_primitives(bssn_1, grid.r)

    r = grid.r

    r_min = R_star - window
    r_max = R_star + window
    mask = (r >= r_min) & (r <= r_max)

    D_0 = state_t0[hydro.idx_D, :]
    D_1 = state_t1[hydro.idx_D, :]
    Sr_0 = state_t0[hydro.idx_Sr, :]
    Sr_1 = state_t1[hydro.idx_Sr, :]
    tau_0 = state_t0[hydro.idx_tau, :]
    tau_1 = state_t1[hydro.idx_tau, :]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].plot(r[mask], rho0_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[0, 0].plot(r[mask], rho0_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[0, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel(r'$\rho_0$')
    axes[0, 0].set_title('Baryon Density (surface)')
    axes[0, 0].legend()

    axes[0, 1].plot(r[mask], p_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[0, 1].plot(r[mask], p_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[0, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Pressure (surface)')
    axes[0, 1].legend()

    axes[0, 2].plot(r[mask], vr_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[0, 2].plot(r[mask], vr_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[0, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel(r'$v^r$')
    axes[0, 2].set_title('Radial Velocity (surface)')
    axes[0, 2].legend()

    axes[1, 0].plot(r[mask], D_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[1, 0].plot(r[mask], D_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[1, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel('D')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Conserved D (surface)')
    axes[1, 0].legend()

    axes[1, 1].plot(r[mask], Sr_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[1, 1].plot(r[mask], Sr_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[1, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel(r'$S_r$')
    axes[1, 1].set_title('Conserved Sr (surface)')
    axes[1, 1].legend()

    axes[1, 2].plot(r[mask], tau_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[1, 2].plot(r[mask], tau_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[1, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel(r'$\tau$')
    axes[1, 2].set_yscale('log')
    axes[1, 2].set_title('Conserved tau (surface)')
    axes[1, 2].legend()

    plt.suptitle(f'Surface Zoom: r ∈ [{r_min:.2f}, {r_max:.2f}], R*={R_star:.2f}', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(get_plots_dir(), f'tov_surface_zoom{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_mass_and_central_density(times, Mb_series, rho_c_series, suffix=""):
    """Plot baryon mass and central density evolution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    Mb_0 = Mb_series[0]
    delta_Mb = np.array(Mb_series) - Mb_0

    axes[0].semilogy(times, np.abs(delta_Mb) + 1e-20, 'b-', linewidth=1.5)
    axes[0].set_xlabel('t')
    axes[0].set_ylabel(r'$M_b - M_{b,0}$')
    axes[0].set_title('Baryon Mass Deviation')
    axes[0].set_yscale('log')

    rho_c_0 = rho_c_series[0]
    delta_rho_c = (np.array(rho_c_series) - rho_c_0) / rho_c_0

    axes[1].plot(times, delta_rho_c, 'r-', linewidth=1.5)
    axes[1].set_xlabel('t')
    axes[1].set_ylabel(r'$(\rho_c - \rho_{c,0})/\rho_{c,0}$')
    axes[1].set_title('Central Density Relative Change')
    axes[1].set_yscale('log')

    plt.tight_layout()
    out_path = os.path.join(get_plots_dir(), f'tov_mass_central_density{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")
