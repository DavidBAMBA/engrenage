#!/usr/bin/env python3
"""
Convergence plot for TOV star evolution - Central density relative change
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import argparse
from scipy.interpolate import interp1d

# Resolutions - UPDATE THESE TO CHANGE RESOLUTIONS
N_very_low = 100
N_low = 200
N_med = 400
N_high = 800
#N_very_high = 800

# Data paths - constructed from resolution values
FOLDERS = {
    f'N={N_very_low}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_refact_rmax20.0_jax/tov_star_rhoc1p28em03_N{N_very_low}_K100_G2_dyn_mp5',
    f'N={N_low}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_refact_rmax20.0_jax/tov_star_rhoc1p28em03_N{N_low}_K100_G2_dyn_mp5',
    f'N={N_med}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_refact_rmax20.0_jax/tov_star_rhoc1p28em03_N{N_med}_K100_G2_dyn_mp5',
    f'N={N_high}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_refact_rmax20.0_jax/tov_star_rhoc1p28em03_N{N_high}_K100_G2_dyn_mp5',
}
# Resolution labels for convergence order calculation (uses first 3 resolutions)
low_res = f'N={N_low}'
med_res = f'N={N_med}'
high_res = f'N={N_high}'

COLORS = ['#1f77b4', "#ff7f0e", '#2ca02c', "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#17becf"]


# ============================================================
# Original utilities (unchanged)
# ============================================================

def load_timeseries(folder_path):
    h5_file = os.path.join(folder_path, 'tov_evolution_dyn_jax.h5')
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
    h5_file = os.path.join(folder_path, 'tov_snapshots_dyn_jax.h5')

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
    The finer array is interpolated to the coarser grid using cubic interpolation.
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

    # Interpolate fine solution to coarse grid using cubic interpolation (3rd order)
    interp_func = interp1d(x_fine, fine, kind='cubic', bounds_error=False, fill_value='extrapolate')
    fine_interp = interp_func(x_coarse)

    return np.mean(np.abs(coarse - fine_interp))


def find_nearest_snapshots(t_target, t_source, rho_list):
    """
    Find nearest neighbor snapshots in time (no temporal interpolation).

    Args:
        t_target: Target time array
        t_source: Source time array
        rho_list: List of density fields at source times

    Returns:
        List of density fields at nearest source times
    """
    rho_nearest = []

    for t_tgt in t_target:
        # Find nearest time index
        idx = np.argmin(np.abs(t_source - t_tgt))
        rho_nearest.append(rho_list[idx].copy())

    return rho_nearest


# ============================================================
# MAIN
# ============================================================

def extract_resolution_from_dirname(dirname):
    """Extract resolution number from directory name."""
    import re
    match = re.search(r'[Nn]r?[=_]?(\d+)', dirname)
    if match:
        return int(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Convergence analysis (requires exactly 3 resolutions)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python plot_convergence.py                       # Use default folders
  python plot_convergence.py --data-dirs D1 D2 D3  # Exactly 3 directories
'''
    )
    parser.add_argument('--data-dirs', nargs='+', default=None,
                        help='List of data directories (exactly 3 required). Default: use FOLDERS')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for plots. Default: script_dir/plots')
    parser.add_argument('--t-max', type=float, default=2000.0,
                        help='Maximum time to plot. Default: 2000.0')
    args = parser.parse_args()

    t_max = args.t_max
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine output directory
    if args.output_dir:
        plots_dir = args.output_dir
    else:
        plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Determine data folders
    if args.data_dirs:
        # Validate exactly 3 directories
        if len(args.data_dirs) != 3:
            print(f"Error: This script requires exactly 3 directories, got {len(args.data_dirs)}")
            return

        # Use command-line provided directories
        folders_dict = {}
        for folder_path in args.data_dirs:
            if os.path.exists(folder_path):
                folder_name = os.path.basename(folder_path)
                res_num = extract_resolution_from_dirname(folder_name)
                label = f'N={res_num}' if res_num else folder_name
                folders_dict[label] = folder_path
            else:
                print(f"Warning: Folder not found: {folder_path}")

        if len(folders_dict) != 3:
            print("Error: Need exactly 3 valid directories")
            return

        # Sort by resolution to assign low/med/high
        sorted_items = sorted(folders_dict.items(), key=lambda x: extract_resolution_from_dirname(x[0]) or 0)
        low_res, med_res, high_res = sorted_items[0][0], sorted_items[1][0], sorted_items[2][0]
        folders_dict = dict(sorted_items)
    else:
        # Use default FOLDERS (backward compatibility)
        folders_dict = FOLDERS
        low_res = f'N={N_very_low}'
        med_res = f'N={N_low}'
        high_res = f'N={N_med}'

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

    for (label, folder_path), color in zip(folders_dict.items(), COLORS):
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
            ax2.plot(t, l1_rho, label=label, color=color, linewidth=0.8)
        ax2.set_xlabel(r'$t$ [M$_\odot$]')
        ax2.set_ylabel(r'$L_1(\rho)$ error')
        ax2.set_title(r'(b) $L_1$ Norm of Density Error')
        ax2.set_yscale('log')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

    # Plot 3: L2 error
    if l2_data:
        for label, (t, l2_rho, color) in l2_data.items():
            ax3.plot(t, l2_rho, label=label, color=color, linewidth=0.8)
        ax3.set_xlabel(r'$t$ [M$_\odot$]')
        ax3.set_ylabel(r'$L_2(\rho)$ error')
        ax3.set_title(r'(c) $L_2$ Norm of Density Error')
        ax3.set_yscale('log')
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)

    # =========================================================
    # Plot 4: Convergence order
    # =========================================================
    t1, rho1 = load_snapshots(folders_dict[low_res])
    t2, rho2 = load_snapshots(folders_dict[med_res])
    t3, rho3 = load_snapshots(folders_dict[high_res])

    # Use coarsest time grid as reference (has fewest snapshots)
    mask = t1 <= t_max
    t_snap = t1[mask]
    rho1 = [rho1[i] for i in range(len(t1)) if t1[i] <= t_max]

    print(f"\nSynchronizing snapshots:")
    print(f"  {low_res}: {len(t1)} snapshots, dt ≈ {np.mean(np.diff(t1)):.3f}")
    print(f"  {med_res}: {len(t2)} snapshots, dt ≈ {np.mean(np.diff(t2)):.3f}")
    print(f"  {high_res}: {len(t3)} snapshots, dt ≈ {np.mean(np.diff(t3)):.3f}")
    print(f"  Using {low_res} times as reference ({len(t_snap)} points)")

    # Find nearest neighbor snapshots (no temporal interpolation)
    print(f"  Finding nearest {med_res} snapshots to reference times...")
    rho2_interp = find_nearest_snapshots(t_snap, t2, rho2)
    print(f"  Finding nearest {high_res} snapshots to reference times...")
    rho3_interp = find_nearest_snapshots(t_snap, t3, rho3)

    # Compute errors at synchronized times
    E12 = []
    E23 = []

    for i in range(len(t_snap)):
        E12.append(l1_error(rho1[i], rho2_interp[i]))
        E23.append(l1_error(rho2_interp[i], rho3_interp[i]))

    E12 = np.array(E12)
    E23 = np.array(E23)

    print(f"  Computed L1 errors: E12 mean={np.mean(E12):.3e}, E23 mean={np.mean(E23):.3e}")

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
    # Save plot
    output_path = os.path.join(plots_dir, 'convergence_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()
    #plt.close('all')

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
