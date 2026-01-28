#!/usr/bin/env python3
"""
Convergence plot for TOV star evolution - Central density relative change
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

# Resolutions - UPDATE THESE TO CHANGE RESOLUTIONS
N_low = 500
N_med = 1000
N_high = 2000

# Data paths - constructed from resolution values
FOLDERS = {
    f'N={N_low}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_last_domain/tov_star_rhoc1p28em03_N{N_low}_K100_G2_cow_mp5',
    f'N={N_med}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_last_domain/tov_star_rhoc1p28em03_N{N_med}_K100_G2_cow_mp5',
    f'N={N_high}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_last_domain/tov_star_rhoc1p28em03_N{N_high}_K100_G2_cow_mp5',
}

# Resolution labels (keys to FOLDERS dictionary)
low_res = f'N={N_low}'
med_res = f'N={N_med}'
high_res = f'N={N_high}'

COLORS = ['#1f77b4', "#ff7f0e", '#2ca02c', "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#17becf"]


# ============================================================
# Original utilities (unchanged)
# ============================================================

def load_timeseries(folder_path):
    h5_file = os.path.join(folder_path, 'tov_evolution_cow.h5')
    if os.path.exists(h5_file):
        with h5py.File(h5_file, 'r') as f:
            times = f['time'][:]
            rho_central = f['rho_central'][:]
            l1_rho = f['l1_rho_error'][:] if 'l1_rho_error' in f else None
            l2_rho = f['l2_rho_error'][:] if 'l2_rho_error' in f else None
            return times, rho_central, l1_rho, l2_rho
    return None, None, None, None


def running_average(x, window):
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')


# ============================================================
# NEW: snapshot loader (needed ONLY for convergence order)
# ============================================================

def load_snapshots(folder_path):
    """
    Reads snapshots saved by engrenage.
    time is stored as an ATTRIBUTE of each snapshot group.
    """
    h5_file = os.path.join(folder_path, 'tov_snapshots_cow.h5')

    times = []
    rho_list = []

    with h5py.File(h5_file, 'r') as f:
        snaps = f['snapshots']
        for key in sorted(snaps.keys()):
            g = snaps[key]
            times.append(g.attrs['time'])
            rho_list.append(g['primitives/rho0'][:])

    return np.array(times), rho_list


def l1_error(a, b):
    """
    Compute L1 error between two arrays, interpolating if sizes differ.
    The finer array is interpolated to the coarser grid.
    """
    if len(a) == len(b):
        return np.mean(np.abs(a - b))

    # Interpolate the finer array to the coarser grid
    if len(a) < len(b):
        coarse, fine = a, b
    else:
        coarse, fine = b, a

    # Create normalized grid coordinates [0, 1]
    x_coarse = np.linspace(0, 1, len(coarse))
    x_fine = np.linspace(0, 1, len(fine))

    # Interpolate fine solution to coarse grid
    fine_interp = np.interp(x_coarse, x_fine, fine)

    return np.mean(np.abs(coarse - fine_interp))


# ============================================================
# MAIN
# ============================================================

def main():

    t_max = 2000.0

    # =========================================================
    # Create single figure with 2x2 subplots
    # =========================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2 = axes[0]  # Top row: Central density, L1 error
    ax3, ax4 = axes[1]  # Bottom row: L2 error, Convergence order

    # =========================================================
    # Load timeseries data for all resolutions
    # =========================================================
    l1_data = {}
    l2_data = {}

    for (label, folder_path), color in zip(FOLDERS.items(), COLORS):
        t, rho_c, l1_rho, l2_rho = load_timeseries(folder_path)
        if t is None:
            continue

        mask = t <= t_max
        t = t[mask]
        rho_c = rho_c[mask]

        if l1_rho is not None:
            l1_data[label] = (t, l1_rho[mask], color)

        if l2_rho is not None:
            l2_data[label] = (t, l2_rho[mask], color)

        # Plot 1: Central density relative change
        rho_c_0 = rho_c[0]
        delta_rho_rel = (rho_c - rho_c_0) / rho_c_0
        ax1.plot(t, delta_rho_rel, label=label, color=color, linewidth=0.8)

    # Configure ax1
    ax1.set_xlabel(r'$t$ [M$_\odot$]')
    ax1.set_ylabel(r'$(\rho_c-\rho_{c,0})/\rho_{c,0}$')
    ax1.set_title('(a) Central Density Relative Change')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Plot 2: L1 error
    if l1_data:
        for label, (t, l1_rho, color) in l1_data.items():
            ax2.plot(t, l1_rho * 1e5, label=label, color=color, linewidth=0.8)
        ax2.set_xlabel(r'$t$ [M$_\odot$]')
        ax2.set_ylabel(r'$L_1(\rho)$ error $[\times 10^{-5}]$')
        ax2.set_title(r'(b) $L_1$ Norm of Density Error')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

    # Plot 3: L2 error
    if l2_data:
        for label, (t, l2_rho, color) in l2_data.items():
            ax3.plot(t, l2_rho * 1e5, label=label, color=color, linewidth=0.8)
        ax3.set_xlabel(r'$t$ [M$_\odot$]')
        ax3.set_ylabel(r'$L_2(\rho)$ error $[\times 10^{-5}]$')
        ax3.set_title(r'(c) $L_2$ Norm of Density Error')
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)

    # =========================================================
    # Plot 4: Convergence order
    # =========================================================
    t1, rho1 = load_snapshots(FOLDERS[low_res])
    t2, rho2 = load_snapshots(FOLDERS[med_res])
    t3, rho3 = load_snapshots(FOLDERS[high_res])

    # Assume snapshots are synchronized
    mask = t1 <= t_max
    t_snap = t1[mask]

    E12 = []
    E23 = []

    for i in range(len(t_snap)):
        E12.append(l1_error(rho1[i], rho2[i]))
        E23.append(l1_error(rho2[i], rho3[i]))

    E12 = np.array(E12)
    E23 = np.array(E23)

    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.log(E12 / E23) / np.log(2.0)

    p_avg = running_average(p, window=50)

    valid = t_snap > 100
    ax4.plot(t_snap[valid], p[valid], color='k', alpha=0.3, lw=0.6,
             label='instantaneous')
    ax4.plot(t_snap[valid], p_avg[valid], color='k', lw=2.5,
             label='running average')

    ax4.axhline(2, ls='--', color='gray', label='2nd order')
    ax4.axhline(3, ls=':', color='gray', label='3rd order')
    ax4.axhline(5, ls='-.', color='gray', label='5th order')

    ax4.set_xlim(0, t_max)
    ax4.set_ylim(-1, 8)
    ax4.set_xlabel(r'$t$ [M$_\odot$]')
    ax4.set_ylabel(r'Convergence order $p(t)$')
    ax4.set_title(rf'(d) Convergence Order (triplet {low_res}–{med_res}–{high_res})')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    # Save in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'convergence_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # =========================================================
    # Print convergence analysis
    # =========================================================
    print("\n" + "="*60)
    print(f"CONVERGENCE ORDER ANALYSIS (triplet {low_res}, {med_res}, {high_res})")
    print("="*60)

    # Filter valid data (t > 100 to skip initial transients)
    valid_mask = t_snap > 100
    p_valid = p[valid_mask]
    t_valid = t_snap[valid_mask]
    E12_valid = E12[valid_mask]
    E23_valid = E23[valid_mask]

    # Remove NaN and Inf values
    finite_mask = np.isfinite(p_valid)
    p_finite = p_valid[finite_mask]

    print(f"\nTime range analyzed: t = [{t_valid[0]:.1f}, {t_valid[-1]:.1f}] M_sun")
    print(f"Number of snapshots: {len(p_finite)} (valid, finite)")

    print(f"\n--- L1 Error Statistics ---")
    print(f"E({low_res}, {med_res}):   mean = {np.mean(E12_valid):.3e}, max = {np.max(E12_valid):.3e}")
    print(f"E({med_res}, {high_res}):  mean = {np.mean(E23_valid):.3e}, max = {np.max(E23_valid):.3e}")
    print(f"Error ratio E12/E23: {np.mean(E12_valid)/np.mean(E23_valid):.2f} (expected ~4 for 2nd order)")

    print(f"\n--- Convergence Order p(t) ---")
    print(f"Mean:   {np.mean(p_finite):.2f}")
    print(f"Median: {np.median(p_finite):.2f}")
    print(f"Std:    {np.std(p_finite):.2f}")
    print(f"Min:    {np.min(p_finite):.2f}")
    print(f"Max:    {np.max(p_finite):.2f}")

    # Percentiles
    p25, p50, p75 = np.percentile(p_finite, [25, 50, 75])
    print(f"\nPercentiles: 25%={p25:.2f}, 50%={p50:.2f}, 75%={p75:.2f}")

    # Time intervals analysis
    intervals = [(100, 1000), (1000, 3000), (3000, 5000), (5000, 7000)]
    print(f"\n--- Convergence by time interval ---")
    for t_start, t_end in intervals:
        mask_int = (t_valid >= t_start) & (t_valid < t_end) & np.isfinite(p_valid)
        if np.sum(mask_int) > 0:
            p_int = p_valid[mask_int]
            print(f"  t=[{t_start:4d}, {t_end:4d}]: p = {np.mean(p_int):.2f} ± {np.std(p_int):.2f}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
