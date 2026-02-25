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
from scipy.interpolate import interp1d

# Resolutions - UPDATE THESE TO CHANGE RESOLUTIONS
N_very_low = 1000
N_low = 2000
N_med = 4000
N_high = 8000
# Data paths - constructed from resolution values
FOLDERS = {
    f'N={N_very_low}': f'/home/davidbamba/repositories/engrenage/examples/TOV/data/tov_evolution_data_refact_rmax100.0_jax/tov_star_rhoc1p28em03_N500_K100_G2_dyn_mp5',
    f'N={N_low}':      f'/home/davidbamba/repositories/engrenage/examples/TOV/data/tov_evolution_data_refact_rmax100.0_jax/tov_star_rhoc1p28em03_N1000_K100_G2_dyn_mp5',
    f'N={N_med}':      f'/home/davidbamba/repositories/engrenage/examples/TOV/data/tov_evolution_data_refact_rmax100.0_jax/tov_star_rhoc1p28em03_N2000_K100_G2_dyn_mp5',
      #f'N={N_low}': f'/home/davidbamba/repositories/engrenage/examples/TOV/data/tov_evolution_data_rmax100.0_jax_reconstructor/tov_star_rhoc1p28em03_N2000_K100_G2_cow_wz',
    #f'N={N_med}': f'/home/davidbamba/repositories/engrenage/examples/TOV/data/tov_evolution_data_rmax100.0_jax_reconstructor/tov_star_rhoc1p28em03_N4000_K100_G2_cow_wz',
    #f'N={N_high}': f'/home/davidbamba/repositories/engrenage/examples/TOV/data/tov_evolution_data_rmax100.0_jax_reconstructor/tov_star_rhoc1p28em03_N8000_K100_G2_cow_wz',
}
COLORS = ['#1f77b4', "#ff7f0e", '#2ca02c', "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def load_timeseries(folder_path):
    """Load timeseries from HDF5 or NPZ files.

    Auto-detects Cowling or dynamic mode files (with or without _jax suffix).
    Priority: timeseries.npz > dyn_jax > dyn > cow_jax > cow
    """
    # Try NPZ first (most general format)
    npz_file = os.path.join(folder_path, 'timeseries.npz')
    if os.path.exists(npz_file):
        npz = np.load(npz_file)
        times = npz['times']
        rho_central = npz['rho_central']
        baryon_mass = npz['baryon_mass'] if 'baryon_mass' in npz else None
        return times, rho_central, baryon_mass

    # Try all possible HDF5 file names (priority order)
    possible_names = [
        'tov_evolution_dyn_jax.h5',
        'tov_evolution_dyn.h5',
        'tov_evolution_cow_jax.h5',
        'tov_evolution_cow.h5',
    ]

    for filename in possible_names:
        h5_file = os.path.join(folder_path, filename)
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

    Auto-detects Cowling or dynamic mode files (with or without _jax suffix).
    """
    # Try all possible snapshot file names (priority order)
    possible_names = [
        'tov_snapshots_dyn_jax.h5',
        'tov_snapshots_dyn.h5',
        'tov_snapshots_cow_jax.h5',
        'tov_snapshots_cow.h5',
    ]

    h5_file = None
    for filename in possible_names:
        filepath = os.path.join(folder_path, filename)
        if os.path.exists(filepath):
            h5_file = filepath
            break

    if h5_file is None:
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


def subsample_to_dt(t, *arrays, dt=1.0):
    """
    Subsample data to regular time intervals.

    Args:
        t: time array
        *arrays: data arrays to subsample
        dt: time interval (default 1.0)

    Returns:
        t_sub, array1_sub, array2_sub, ...
    """
    # Create regular time grid
    t_regular = np.arange(t[0], t[-1], dt)

    # Interpolate each array to regular grid
    results = [t_regular]
    for arr in arrays:
        interp_func = interp1d(t, arr, kind='linear', bounds_error=False, fill_value='extrapolate')
        results.append(interp_func(t_regular))

    return results


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
    parser.add_argument('--t-max', type=float, default=2000.0,
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

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    ax1, ax2 = axes[0]  # Central density absolute
    ax3, ax4 = axes[1]  # Central density relative change
    ax5, ax6 = axes[2]  # Mass conservation and L1 norm

    # Time interval for subsampling
    dt_plot = 1.0  # Plot every dt=1 M_sun

    for (label, folder_path), color in zip(folders_dict.items(), COLORS):
        # Strip whitespace from folder path
        folder_path = folder_path.strip()

        # Debug: check if folder exists
        if not os.path.exists(folder_path):
            print(f"Warning: Folder does not exist: {folder_path}")
            continue

        # Debug: list files in folder
        files = os.listdir(folder_path)
        print(f"Loading {label} from {folder_path}")
        print(f"  Available files: {[f for f in files if f.endswith(('.h5', '.npz'))]}")

        t, rho_c, M_b = load_timeseries(folder_path)
        if t is None:
            print(f"  Warning: No timeseries data found for {label}")
            continue

        print(f"  Loaded {len(t)} timesteps (t: {t[0]:.1f} to {t[-1]:.1f})")

        mask = t <= t_max
        t = t[mask]
        rho_c = rho_c[mask]

        # Compute relative change
        rho_c_0 = rho_c[0]
        delta_rho_rel = (rho_c - rho_c_0) / rho_c_0

        # Subsample to dt=1 for plotting
        t_sub, rho_c_sub, delta_rho_sub = subsample_to_dt(t, rho_c, delta_rho_rel, dt=dt_plot)

        # Window for smoothing (based on subsampled data)
        window_large = min(501, len(t_sub) // 5 * 2 + 1)
        window_large = max(5, window_large)

        # Plot 1: Central density absolute (subsampled)
        ax1.plot(t_sub, rho_c_sub, label=label, color=color, linewidth=0.8)

        # Plot 2: Central density absolute (smoothed)
        if len(rho_c_sub) >= window_large:
            rho_c_smooth = savgol_filter(rho_c_sub, window_large, 3)
            ax2.plot(t_sub, rho_c_smooth, label=label, color=color, linewidth=1.5)

        # Plot 3: Central density relative change (subsampled)
        ax3.plot(t_sub, delta_rho_sub, label=label, color=color, linewidth=0.8)

        # Plot 4: Central density relative change (smoothed)
        if len(delta_rho_sub) >= window_large:
            delta_rho_smooth = savgol_filter(delta_rho_sub, window_large, 3)
            ax4.plot(t_sub, delta_rho_smooth, label=label, color=color, linewidth=1.5)

        # Plot 5: Mass conservation
        if M_b is not None:
            M_b = M_b[mask]
            M_b_0 = M_b[0]
            log_abs_delta_M = np.log10(np.abs(M_b / M_b_0 - 1.0) + 1e-20)

            # Subsample mass conservation
            t_sub_m, log_delta_M_sub = subsample_to_dt(t, log_abs_delta_M, dt=dt_plot)
            ax5.plot(t_sub_m, log_delta_M_sub, label=label, color=color, linewidth=0.8)

    # =========================================================
    # Plot 6: L1 norm of rest-mass density evolution
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

        ax6.plot(t_snap_filtered, L1_norms, label=legend_label, color=color, linewidth=0.8)

    # Configure ax1 - Central density absolute (raw)
    ax1.set_xlabel(r'$t$ [M$_\odot$]')
    ax1.set_ylabel(r'$\rho_c$')
    ax1.set_title('(a) Central Density (raw)')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # Configure ax2 - Central density absolute (smoothed)
    ax2.set_xlabel(r'$t$ [M$_\odot$]')
    ax2.set_ylabel(r'$\rho_c$')
    ax2.set_title('(b) Central Density (smoothed)')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # Configure ax3 - Relative change (raw)
    ax3.set_xlabel(r'$t$ [M$_\odot$]')
    ax3.set_ylabel(r'$(\rho_c-\rho_{c,0})/\rho_{c,0}$')
    ax3.set_title('(c) Central Density Relative Change (raw)')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # Configure ax4 - Relative change (smoothed)
    ax4.set_xlabel(r'$t$ [M$_\odot$]')
    ax4.set_ylabel(r'$(\rho_c-\rho_{c,0})/\rho_{c,0}$')
    ax4.set_title('(d) Central Density Relative Change (smoothed)')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    # Configure ax5 - Mass conservation
    ax5.set_xlabel(r'$t$ [M$_\odot$]')
    ax5.set_ylabel(r'$\log_{10}|M_b / M_{b,0} - 1.0|$')
    ax5.set_title('(e) Baryon Mass Absolute Error (log scale)')
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

    # Configure ax6 - L1 norm
    ax6.set_xlabel(r'$t$ [M$_\odot$]')
    ax6.set_ylabel(r'$||\rho(t)-\rho(0)||_1$')
    ax6.set_title('(f) L1 Norm of Rest-Mass Density Evolution')
    ax6.legend(fontsize=8, loc='upper left')
    ax6.grid(alpha=0.3)
    ax6.set_xlim(0, t_max)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(plots_dir, 'rho_central.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()
    plt.close('all')


if __name__ == "__main__":
    main()
