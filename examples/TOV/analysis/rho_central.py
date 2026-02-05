#!/usr/bin/env python3
"""
Plot central density relative change for TOV star evolution
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import argparse
from scipy.signal import savgol_filter

# Resolutions - UPDATE THESE TO CHANGE RESOLUTIONS
N_very_low = 1000
N_low = 2000
N_med = 4000
N_high = 8000
# Data paths - constructed from resolution values
FOLDERS = {
    f'N={N_very_low}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_rmax100_TEST_long_domain_16/tov_star_rhoc1p28em03_N{N_very_low}_K100_G2_cow_mp5',
    f'N={N_low}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_rmax100_TEST_long_domain_16/tov_star_rhoc1p28em03_N{N_low}_K100_G2_cow_mp5',
    f'N={N_med}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_rmax100_TEST_long_domain_16/tov_star_rhoc1p28em03_N{N_med}_K100_G2_cow_mp5',
    f'N={N_high}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_rmax100_TEST_long_domain_16/tov_star_rhoc1p28em03_N{N_high}_K100_G2_cow_mp5',
}
COLORS = ['#1f77b4', "#ff7f0e", '#2ca02c', "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def load_timeseries(folder_path):
    h5_file = os.path.join(folder_path, 'tov_evolution_cow.h5')
    if os.path.exists(h5_file):
        with h5py.File(h5_file, 'r') as f:
            times = f['time'][:]
            rho_central = f['rho_central'][:]
            baryon_mass = f['baryon_mass'][:] if 'baryon_mass' in f else None
            return times, rho_central, baryon_mass
    return None, None, None


def load_snapshots(folder_path):
    """
    Load snapshots with density profiles.
    Returns times, radial grid r, and list of density fields rho0(r).
    """
    h5_file = os.path.join(folder_path, 'tov_snapshots_cow.h5')
    if not os.path.exists(h5_file):
        return None, None, None

    times = []
    rho_list = []

    with h5py.File(h5_file, 'r') as f:
        # Load radial grid
        r = f['grid/r'][:]

        snaps = f['snapshots']
        for key in sorted(snaps.keys()):
            g = snaps[key]
            times.append(g.attrs['time'])
            rho_list.append(g['primitives/rho0'][:])

    return np.array(times), r, rho_list


def compute_L1_norm_discrete(rho, r=None):
    """
    Compute discrete L1 norm: Σ |ρᵢ|
    Simple sum of absolute values (no integration, no weights).

    Args:
        rho: density array
        r: radial grid (if provided, filters to physical domain r >= 0)

    Returns:
        L1 norm over physical domain only
    """
    if r is not None:
        # Filter to physical domain (exclude ghost cells where r < 0)
        mask = r >= 0
        rho = rho[mask]
    return np.sum(np.abs(rho))


def extract_resolution_from_dirname(dirname):
    """Extract resolution number from directory name."""
    import re
    match = re.search(r'[Nn]r?[=_]?(\d+)', dirname)
    if match:
        return int(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Plot central density relative change for TOV star evolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python rho_central.py                             # Use default folders
  python rho_central.py --data-dirs DIR1 DIR2 DIR3  # Custom directories
  python rho_central.py --output-dir OUT_DIR        # Custom output
'''
    )
    parser.add_argument('--data-dirs', nargs='+', default=None,
                        help='List of data directories to analyze. Default: use FOLDERS')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for plots. Default: script_dir/plots')
    parser.add_argument('--t-max', type=float, default=1000.0,
                        help='Maximum time to plot. Default: 1000.0')
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

        if not folders_dict:
            print("No valid folders found. Check the paths provided via --data-dirs")
            return
    else:
        # Use default FOLDERS (backward compatibility)
        folders_dict = FOLDERS

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    for (label, folder_path), color in zip(folders_dict.items(), COLORS):
        t, rho_c, M_b = load_timeseries(folder_path)
        if t is None:
            continue

        mask = t <= t_max
        t = t[mask]
        rho_c = rho_c[mask]

        # Plot 1: Central density relative change (raw)
        rho_c_0 = rho_c[0]
        delta_rho_rel = (rho_c - rho_c_0) / rho_c_0
        ax1.plot(t, delta_rho_rel, label=label, color=color, linewidth=0.8)

        # Plot 2: Central density relative change (smoothed)
        # Apply Savitzky-Golay filter to reduce noise
        window = min(101, len(delta_rho_rel) // 10 * 2 + 1)  # Must be odd
        if window >= 5:
            delta_rho_smooth = savgol_filter(delta_rho_rel, window, 3)
            ax2.plot(t, delta_rho_smooth, label=label, color=color, linewidth=0.8)

        # Plot 3: Mass conservation
        if M_b is not None:
            M_b = M_b[mask]
            M_b_0 = M_b[0]

            # Plot 3: log(abs(M_b - M_b_0))
            log_abs_delta_M = np.log10(np.abs(M_b / M_b_0 -1.0) + 1e-20)
            ax3.plot(t, log_abs_delta_M, label=label, color=color, linewidth=0.8)

    # =========================================================
    # Plot 4: L1 norm of rest-mass density evolution
    # Plot ALL available resolutions
    # =========================================================
    resolutions_to_plot = []
    sorted_folders = sorted(folders_dict.items(), key=lambda x: extract_resolution_from_dirname(x[0]) or 0)

    for i, (label, folder_path) in enumerate(sorted_folders):
        color = COLORS[i % len(COLORS)]
        resolutions_to_plot.append((label, folder_path, label, 1, color))

    for label, folder_path, legend_label, scale_factor, color in resolutions_to_plot:
        t_snap, r, rho_list = load_snapshots(folder_path)
        if t_snap is None:
            print(f"Warning: No snapshots found for {label}")
            continue

        # Filter by time
        mask = t_snap <= t_max
        t_snap_filtered = t_snap[mask]
        rho_list_filtered = [rho_list[i] for i in range(len(mask)) if mask[i]]

        # Compute L1 norm of (rho(t) - rho(0)) over physical domain only (r >= 0)
        rho_0 = rho_list_filtered[0]
        L1_norms = []
        for rho_t in rho_list_filtered:
            delta_rho = rho_t - rho_0
            L1_norm = compute_L1_norm_discrete(delta_rho, r=r)
            L1_norms.append(L1_norm * scale_factor)  # Apply scaling

        ax4.plot(t_snap_filtered, L1_norms, label=legend_label, color=color, linewidth=0.8)

    # Configure ax1
    ax1.set_xlabel(r'$t$ [M$_\odot$]')
    ax1.set_ylabel(r'$(\rho_c-\rho_{c,0})/\rho_{c,0}$')
    ax1.set_title('(a) Central Density Relative Change (raw)')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Configure ax2
    ax2.set_xlabel(r'$t$ [M$_\odot$]')
    ax2.set_ylabel(r'$(\rho_c-\rho_{c,0})/\rho_{c,0}$')
    ax2.set_title('(b) Central Density Relative Change (smoothed)')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Configure ax3
    ax3.set_xlabel(r'$t$ [M$_\odot$]')
    ax3.set_ylabel(r'$\log_{10}|M_b / M_{b,0} - 1.0|$')
    ax3.set_title('(c) Baryon Mass Absolute Error (log scale)')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # Configure ax4
    ax4.set_xlabel(r'Time [M$_\odot$]', fontsize=12)
    ax4.set_ylabel(r'$||\rho(t)-\rho(0)||_1$', fontsize=12)
    ax4.set_title('(d) L1 Norm of Rest-Mass Density Evolution')
    ax4.legend(fontsize=9, loc='upper left')
    ax4.grid(alpha=0.3)
    ax4.set_xlim(0, t_max)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(plots_dir, 'rho_central.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()
    plt.close('all')


if __name__ == "__main__":
    main()
