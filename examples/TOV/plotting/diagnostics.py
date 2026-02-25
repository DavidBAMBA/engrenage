"""
Diagnostic plotting utilities for TOV evolution.

Main visualization functions for simulation results.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import h5py


def get_plots_dir():
    """Get plots directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(os.path.dirname(script_dir), 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir


def plot_tov_diagnostics(tov_solution, r_max, suffix=""):
    """Plot TOV solution diagnostics."""
    if hasattr(tov_solution, 'r'):
        r = tov_solution.r
        R_star = tov_solution.R
        M_star = tov_solution.M_star
        rho_baryon = tov_solution.rho_baryon
        P = tov_solution.P
        M = tov_solution.M
        alpha = tov_solution.alpha
        exp4phi = tov_solution.exp4phi
    else:
        r = tov_solution['r']
        R_star = tov_solution['R']
        M_star = tov_solution['M_star']
        rho_baryon = tov_solution['rho_baryon']
        P = tov_solution['P']
        M = tov_solution['M']
        alpha = tov_solution['alpha']
        exp4phi = tov_solution['exp4phi']

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    axes[0, 0].plot(r, rho_baryon, color='navy')
    axes[0, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5, label=f'R={R_star:.2f}')
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel('rho_0')
    axes[0, 0].set_title('Baryon Density')
    axes[0, 0].set_xlim(0, r_max)
    axes[0, 0].legend()

    axes[0, 1].plot(r, P, color='darkgreen')
    axes[0, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure')
    axes[0, 1].set_xlim(0, r_max)

    axes[0, 2].plot(r, M, color='maroon')
    axes[0, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 2].axhline(M_star, color='gray', linestyle='--', alpha=0.3, label=f'M={M_star:.3f}')
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel('M(r)')
    axes[0, 2].set_title('Enclosed Mass')
    axes[0, 2].set_xlim(0, r_max)
    axes[0, 2].legend()

    axes[1, 0].plot(r, alpha, color='purple')
    axes[1, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel('alpha')
    axes[1, 0].set_title('Lapse Function')
    axes[1, 0].set_xlim(0, r_max)

    phi = 0.25 * np.log(exp4phi)
    axes[1, 1].plot(r, phi, color='teal')
    axes[1, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel('phi')
    axes[1, 1].set_title('Conformal Factor phi')
    axes[1, 1].set_xlim(0, r_max)

    a_metric = np.sqrt(exp4phi)
    axes[1, 2].plot(r, a_metric, color='orange')
    axes[1, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel('a(r)')
    axes[1, 2].set_title('Metric a(r)')
    axes[1, 2].set_xlim(0, r_max)

    plt.tight_layout()
    out_path = os.path.join(get_plots_dir(), f'tov_solution{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_evolution(states, times, grid, hydro, rho_ref, p_ref,
                  Mb_series=None, rho_c_series=None,
                  times_series=None, suffix="", R_star=None):
    """Plot evolution at multiple checkpoints."""
    from source.bssn.bssnvars import BSSNVars
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS
    from source.core.spacing import NUM_GHOSTS

    n_states = len(states)
    r = grid.r
    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

    primitives = []
    for state in states:
        bssn = BSSNVars(grid.N)
        bssn.set_bssn_vars(state[:NUM_BSSN_VARS, :])
        hydro.set_matter_vars(state, bssn, grid)
        rho0, vr, p, eps, W, h, _ = hydro._get_primitives(bssn, grid.r)
        primitives.append((rho0, vr, p, eps, W, h))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, n_states))
    for i, (prim, t) in enumerate(zip(primitives, times)):
        ax.plot(r[interior], prim[0][interior], color=colors[i],
                label=f't={t:.2f}', linewidth=1.5)
    if R_star is not None:
        ax.axvline(R_star, color='gray', linestyle=':', alpha=0.7, label=f'R={R_star:.2f}')
    ax.set_xlabel('r')
    ax.set_ylabel(r'$\rho_0$')
    ax.set_title('Baryon Density Evolution')
    ax.set_yscale('log')
    ax.legend(loc='upper right')

    ax = axes[0, 1]
    for i, (prim, t) in enumerate(zip(primitives, times)):
        ax.plot(r[interior], prim[2][interior], color=colors[i],
                label=f't={t:.2f}', linewidth=1.5)
    if R_star is not None:
        ax.axvline(R_star, color='gray', linestyle=':', alpha=0.7, label=f'R={R_star:.2f}')
    ax.set_xlabel('r')
    ax.set_ylabel('P')
    ax.set_title('Pressure Evolution')
    ax.set_yscale('log')
    ax.legend(loc='upper right')

    ax = axes[1, 0]
    if Mb_series is not None and len(Mb_series) > 1:
        Mb_arr = np.asarray(Mb_series)
        if times_series is not None:
            t_ser = np.asarray(times_series)
        else:
            t_ser = np.linspace(0, times[-1], len(Mb_arr))
        Mb_0 = Mb_arr[0]
        delta_Mb = (Mb_arr - Mb_0) / Mb_0
        ax.plot(t_ser, delta_Mb, 'b-', linewidth=1.5)
        ax.set_ylabel(r'$\Delta M_b / M_{b,0}$')
    else:
        ax.text(0.5, 0.5, 'No time series data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
    ax.set_xlabel('t')
    ax.set_title('Baryon Mass Conservation')

    ax = axes[1, 1]
    if rho_c_series is not None and len(rho_c_series) > 1:
        rho_c_arr = np.asarray(rho_c_series)
        if times_series is not None:
            t_ser = np.asarray(times_series)
        else:
            t_ser = np.linspace(0, times[-1], len(rho_c_arr))
        rho_c_0 = rho_c_arr[0]
        delta_rho_c = (rho_c_arr - rho_c_0) / rho_c_0
        ax.plot(t_ser, delta_rho_c, 'r-', linewidth=1.5)
        ax.set_ylabel(r'$(\rho_c - \rho_{c,0})/\rho_{c,0}$')
    else:
        ax.text(0.5, 0.5, 'No time series data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
    ax.set_xlabel('t')
    ax.set_title('Central Density Relative Change')

    plt.suptitle(f'TOV Evolution', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(get_plots_dir(), f'tov_evolution{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_bssn_evolution(state_t0, state_tfinal, grid, t_0=0.0, t_final=1.0, suffix=""):
    """Plot BSSN variables at t=0 vs t=final."""
    from source.bssn.bssnstatevariables import (
        idx_phi, idx_hrr, idx_K, idx_shiftr, idx_br, idx_lapse
    )

    r = grid.r

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].plot(r, state_t0[idx_phi, :], 'b-', label='t=0', linewidth=1.5)
    axes[0, 0].plot(r, state_tfinal[idx_phi, :], 'r--', label=f't={t_final}', linewidth=1.5)
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel(r'$\phi$')
    axes[0, 0].set_title('Conformal Factor φ')
    axes[0, 0].legend()

    axes[0, 1].plot(r, state_t0[idx_hrr, :], 'b-', label='t=0', linewidth=1.5)
    axes[0, 1].plot(r, state_tfinal[idx_hrr, :], 'r--', label=f't={t_final}', linewidth=1.5)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel(r'$h_{rr}$')
    axes[0, 1].set_title('Metric Deviation h_rr')
    axes[0, 1].legend()

    axes[0, 2].plot(r, state_t0[idx_lapse, :], 'b-', label='t=0', linewidth=1.5)
    axes[0, 2].plot(r, state_tfinal[idx_lapse, :], 'r--', label=f't={t_final}', linewidth=1.5)
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel(r'$\alpha$')
    axes[0, 2].set_title('Lapse alpha')
    axes[0, 2].legend()

    axes[1, 0].plot(r, state_t0[idx_shiftr, :], 'b-', label='t=0', linewidth=1.5)
    axes[1, 0].plot(r, state_tfinal[idx_shiftr, :], 'r--', label=f't={t_final}', linewidth=1.5)
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel(r'$\beta^r$')
    axes[1, 0].set_title('Shift βʳ')
    axes[1, 0].legend()

    axes[1, 1].plot(r, state_t0[idx_br, :], 'b-', label='t=0', linewidth=1.5)
    axes[1, 1].plot(r, state_tfinal[idx_br, :], 'r--', label=f't={t_final}', linewidth=1.5)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel(r'$B^r$')
    axes[1, 1].set_title('Shift Auxiliary Bʳ')
    axes[1, 1].legend()

    axes[1, 2].plot(r, state_t0[idx_K, :], 'b-', label='t=0', linewidth=1.5)
    axes[1, 2].plot(r, state_tfinal[idx_K, :], 'r--', label=f't={t_final}', linewidth=1.5)
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel('K')
    axes[1, 2].set_title('Extrinsic Curvature K')
    axes[1, 2].legend()

    plt.suptitle('BSSN Variables: Cowling Check (should be identical)', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(get_plots_dir(), f'tov_bssn_evolution{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_constraints_evolution(output_dir, suffix=""):
    """Plot constraint violations evolution from saved HDF5 data."""
    evolution_file = os.path.join(output_dir, f'tov_evolution{suffix}.h5')

    if not os.path.exists(evolution_file):
        print(f"\nWarning: Evolution file not found: {evolution_file}")
        print(f"   Cannot plot constraints. Make sure to run evolution with data saving enabled.")
        return

    with h5py.File(evolution_file, 'r') as f:
        if 'max_Ham' not in f:
            print(f"\nWarning: No constraint data found in {evolution_file}")
            return

        times = np.array(f['time'])
        max_Ham = np.array(f['max_Ham'])
        l2_Ham = np.array(f['l2_Ham'])
        max_Mom_r = np.array(f['max_Mom_r'])
        l2_Mom_r = np.array(f['l2_Mom_r'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].semilogy(times, np.maximum(np.abs(max_Ham), 1e-20), 'r-', linewidth=1.5)
    axes[0, 0].set_xlabel('t [M]')
    axes[0, 0].set_ylabel(r'max$|$H$|$')
    axes[0, 0].set_title('Hamiltonian Constraint: Maximum Violation')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(times, np.maximum(l2_Ham, 1e-20), 'b-', linewidth=1.5)
    axes[0, 1].set_xlabel('t [M]')
    axes[0, 1].set_ylabel(r'L$_2$(H)')
    axes[0, 1].set_title('Hamiltonian Constraint: L2 Norm')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].semilogy(times, np.maximum(np.abs(max_Mom_r), 1e-20), 'r-', linewidth=1.5)
    axes[1, 0].set_xlabel('t [M]')
    axes[1, 0].set_ylabel(r'max$|$M$_r|$')
    axes[1, 0].set_title('Momentum Constraint (radial): Maximum Violation')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(times, np.maximum(l2_Mom_r, 1e-20), 'b-', linewidth=1.5)
    axes[1, 1].set_xlabel('t [M]')
    axes[1, 1].set_ylabel(r'L$_2$(M$_r$)')
    axes[1, 1].set_title('Momentum Constraint (radial): L2 Norm')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'BSSN Constraint Violations Evolution', fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(get_plots_dir(), f'constraints_evolution{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_tov_vs_initial_data_zoom(tov_solution, initial_state_2d, grid, primitives, window=0.5, suffix=""):
    """Zoom near the stellar surface to compare TOV solution vs interpolated initial data."""
    from scipy.interpolate import interp1d
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS
    from source.core.spacing import NUM_GHOSTS

    if hasattr(tov_solution, 'R_iso'):
        R = float(tov_solution.R_iso)
        r_tov = tov_solution.r_iso
        rho_tov = tov_solution.rho_baryon
        P_tov = tov_solution.P
    else:
        R = float(tov_solution['R_iso'])
        r_tov = tov_solution['r_iso']
        rho_tov = tov_solution['rho_baryon']
        P_tov = tov_solution['P']

    r = grid.r
    mask = (r >= R - window) & (r <= R + window)
    if not np.any(mask):
        return

    rho0_init, vr_init, p_init, eps_init = primitives

    D_init = initial_state_2d[NUM_BSSN_VARS + 0, :]
    Sr_init = initial_state_2d[NUM_BSSN_VARS + 1, :]
    tau_init = initial_state_2d[NUM_BSSN_VARS + 2, :]

    rho_tov_interp = interp1d(r_tov, rho_tov, kind='cubic', fill_value=0.0, bounds_error=False)
    P_tov_interp = interp1d(r_tov, P_tov, kind='cubic', fill_value=0.0, bounds_error=False)

    rho_tov_zoom = rho_tov_interp(r)
    P_tov_zoom = P_tov_interp(r)

    rZ = r[mask]
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))

    ax[0, 0].semilogy(rZ, np.maximum(rho_tov_zoom[mask], 1e-20), 'k-', linewidth=2, label='TOV (analytic)')
    ax[0, 0].semilogy(rZ, np.maximum(rho0_init[mask], 1e-20), 'b--', linewidth=1.5, label='Initial Data')
    ax[0, 0].axvline(R, color='gray', ls=':', linewidth=1.5, label=f'R={R:.3f}')
    ax[0, 0].set_title('rho_0 (zoom near surface)', fontsize=11)
    ax[0, 0].legend(fontsize=9)
    ax[0, 0].set_ylabel('rho_0', fontsize=10)

    ax[0, 1].semilogy(rZ, np.maximum(P_tov_zoom[mask], 1e-20), 'k-', linewidth=2)
    ax[0, 1].semilogy(rZ, np.maximum(p_init[mask], 1e-20), 'b--', linewidth=1.5)
    ax[0, 1].axvline(R, color='gray', ls=':', linewidth=1.5)
    ax[0, 1].set_title('P (zoom near surface)', fontsize=11)
    ax[0, 1].set_ylabel('P', fontsize=10)

    ax[0, 2].plot(rZ, vr_init[mask], 'b-', linewidth=1.5, label='Initial Data')
    ax[0, 2].axhline(0, color='k', ls='--', linewidth=2, label='TOV (v=0)')
    ax[0, 2].axvline(R, color='gray', ls=':', linewidth=1.5)
    ax[0, 2].set_title('v^r (zoom near surface)', fontsize=11)
    ax[0, 2].legend(fontsize=9)
    ax[0, 2].set_ylabel('v^r', fontsize=10)

    ax[1, 0].semilogy(rZ, np.maximum(D_init[mask], 1e-22), 'b-', linewidth=1.5)
    ax[1, 0].axvline(R, color='gray', ls=':', linewidth=1.5)
    ax[1, 0].set_title('D (conserved density)', fontsize=11)
    ax[1, 0].set_ylabel('D', fontsize=10)

    ax[1, 1].plot(rZ, Sr_init[mask], 'b-', linewidth=1.5)
    ax[1, 1].axhline(0, color='k', ls='--', linewidth=1, label='Expected (S^r=0)')
    ax[1, 1].axvline(R, color='gray', ls=':', linewidth=1.5)
    ax[1, 1].set_title('S_r (conserved momentum)', fontsize=11)
    ax[1, 1].legend(fontsize=9)
    ax[1, 1].set_ylabel('S_r', fontsize=10)

    ax[1, 2].semilogy(rZ, np.maximum(np.abs(tau_init[mask]), 1e-22), 'b-', linewidth=1.5)
    ax[1, 2].axvline(R, color='gray', ls=':', linewidth=1.5)
    ax[1, 2].set_title('tau (conserved energy)', fontsize=11)
    ax[1, 2].set_ylabel('|tau|', fontsize=10)

    for a in ax.ravel():
        a.set_xlabel('r [M]', fontsize=10)

    plt.suptitle(f'TOV Solution vs Initial Data: Surface Zoom [R-{window}, R+{window}]',
                 fontsize=13, y=0.995)
    plt.tight_layout()
    out_path = os.path.join(get_plots_dir(), f'tov_vs_initial_zoom{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")
