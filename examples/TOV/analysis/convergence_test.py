#!/usr/bin/env python3
"""
Convergence analysis using the Q-factor method.

Based on the method from the paper (Appendix B):
Q = (Δr_l^n - Δr_m^n) / (Δr_m^n - Δr_h^n)

where Δr_l, Δr_m, Δr_h are the grid spacings for low, medium, high resolutions
and n is the convergence order.

For resolutions N=200, 400, 800 with dr=40/N (uniform refinement factor 2):
- Δr_l = 0.2   (N=200)
- Δr_m = 0.1   (N=400)
- Δr_h = 0.05  (N=800)

Expected Q values for different orders (uniform refinement factor 2):
- 2nd order: Q = 4
- 3rd order: Q = 8
- 5th order: Q = 32
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import argparse
from scipy.signal import savgol_filter

# Resolutions - UPDATE THESE TO CHANGE RESOLUTIONS
N_low = 2000
N_med = 4000
N_high = 8000

# Data paths - constructed from resolution values
FOLDERS = {
    f'N={N_low}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_rmax100_TEST_long_domain/tov_star_rhoc1p28em03_N{N_low}_K100_G2_cow_mp5',
    f'N={N_med}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_rmax100_TEST_long_domain/tov_star_rhoc1p28em03_N{N_med}_K100_G2_cow_mp5',
    f'N={N_high}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_rmax100_TEST_long_domain/tov_star_rhoc1p28em03_N{N_high}_K100_G2_cow_mp5',
}

# Resolution labels (keys to FOLDERS dictionary)
low_res = f'N={N_low}'
med_res = f'N={N_med}'
high_res = f'N={N_high}'

# Grid spacings (dr = 20/N)
R_MAX = 100.0
RESOLUTIONS = {
    low_res: {'N': N_low, 'dr': R_MAX / N_low},
    med_res: {'N': N_med, 'dr': R_MAX / N_med},
    high_res: {'N': N_high, 'dr': R_MAX / N_high},
}

COLORS = {
    low_res: '#2ca02c',   # green (coarsest)
    med_res: '#1f77b4',   # blue  (medium)
    high_res: '#d62728',   # red   (finest)
}

def get_color(label, all_labels):
    """Get color for a resolution label, using default palette if not in COLORS."""
    if label in COLORS:
        return COLORS[label]
    # Generate color dynamically: green for low, blue for mid, red for high
    labels_sorted = sorted(all_labels, key=lambda x: int(x.split('=')[1]) if '=' in x else 0)
    idx = labels_sorted.index(label)
    color_map = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e', '#9467bd']
    return color_map[idx % len(color_map)]


def load_evolution_data(folder_path, suffix="_cow"):
    """
    Load evolution data (time series) from HDF5 file.
    
    Returns:
        times: array of time values
        rho_central: array of central density values
    """
    evolution_file = os.path.join(folder_path, f'tov_evolution{suffix}.h5')
    
    with h5py.File(evolution_file, 'r') as f:
        times = f['time'][:]
        rho_central = f['rho_central'][:]
    
    return times, rho_central


def compute_expected_Q(dr_l, dr_m, dr_h, order):
    """
    Compute the expected Q factor for a given convergence order.
    
    Q = (Δr_l^n - Δr_m^n) / (Δr_m^n - Δr_h^n)
    
    Args:
        dr_l: grid spacing for low resolution
        dr_m: grid spacing for medium resolution
        dr_h: grid spacing for high resolution
        order: convergence order n
    
    Returns:
        Expected Q value
    """
    numerator = dr_l**order - dr_m**order
    denominator = dr_m**order - dr_h**order
    return numerator / denominator


def find_nearest_time_indices(t_target, t_source):
    """
    Find indices in t_source that are closest to each point in t_target.

    Args:
        t_target: target time array
        t_source: source time array

    Returns:
        Array of indices into t_source
    """
    indices = np.zeros(len(t_target), dtype=int)
    for i, t in enumerate(t_target):
        indices[i] = np.argmin(np.abs(t_source - t))
    return indices


def smooth_data(y, window=51, polyorder=3):
    """Apply Savitzky-Golay smoothing."""
    if len(y) < window:
        return y
    return savgol_filter(y, window, polyorder)


def compute_Q_windowed(t, rho_l, rho_m, rho_h, window_size=100, overlap=0.5):
    """
    Compute Q-factor using L2 norm over sliding windows.

    This method is more robust for oscillating signals than point-by-point Q.
    It averages over temporal windows to reduce sensitivity to phase errors.

    Args:
        t: time array
        rho_l, rho_m, rho_h: density arrays for low, medium, high resolutions
        window_size: number of points per window
        overlap: fraction of overlap between consecutive windows (0 to 1)

    Returns:
        t_windows: center times of each window
        Q_windows: Q-factor computed for each window
    """
    if len(t) < window_size:
        print(f"Warning: Data too short for window_size={window_size}, using single window")
        window_size = len(t)

    step = max(1, int(window_size * (1 - overlap)))

    t_windows = []
    Q_windows = []

    for i in range(0, len(t) - window_size + 1, step):
        window = slice(i, i + window_size)

        # Compute L2 norms of differences over the window
        diff_lm = rho_l[window] - rho_m[window]
        diff_mh = rho_m[window] - rho_h[window]

        E_lm = np.sqrt(np.mean(diff_lm**2))
        E_mh = np.sqrt(np.mean(diff_mh**2))

        # Compute Q for this window
        if E_mh > 1e-15:
            Q = E_lm / E_mh
            # Filter extreme outliers
            if 0 < Q < 50:
                t_center = (t[i] + t[i + window_size - 1]) / 2
                t_windows.append(t_center)
                Q_windows.append(Q)

    return np.array(t_windows), np.array(Q_windows)


def extract_resolution_from_dirname(dirname):
    """Extract resolution number from directory name."""
    import re
    match = re.search(r'[Nn]r?[=_]?(\d+)', dirname)
    if match:
        return int(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Convergence Q-factor test (requires exactly 3 resolutions)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python convergence_test.py                       # Use default folders
  python convergence_test.py --data-dirs D1 D2 D3  # Exactly 3 directories
  python convergence_test.py --r-max 100           # Different domain size
'''
    )
    parser.add_argument('--data-dirs', nargs='+', default=None,
                        help='List of data directories (exactly 3 required). Default: use FOLDERS')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for plots. Default: script_dir/plots')
    parser.add_argument('--t-max', type=float, default=2000.0,
                        help='Maximum time to plot. Default: 2000.0')
    parser.add_argument('--r-max', type=float, default=40.0,
                        help='Domain size for computing dr. Default: 40.0')
    parser.add_argument('--window-size', type=int, default=100,
                        help='Window size for L2 windowed Q-factor. Default: 100')
    args = parser.parse_args()

    t_max = args.t_max
    r_max = args.r_max
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
        resolutions_dict = {}
        for folder_path in args.data_dirs:
            if os.path.exists(folder_path):
                folder_name = os.path.basename(folder_path)
                res_num = extract_resolution_from_dirname(folder_name)
                if res_num is None:
                    print(f"Warning: Could not extract resolution from {folder_name}")
                    continue
                label = f'N={res_num}'
                folders_dict[label] = folder_path
                resolutions_dict[label] = {'N': res_num, 'dr': r_max / res_num}
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
        resolutions_dict = RESOLUTIONS
        low_res = f'N={N_low}'
        med_res = f'N={N_med}'
        high_res = f'N={N_high}'

    # =========================================================
    # Load evolution data for all resolutions
    # =========================================================
    print("Loading evolution data...")

    data = {}
    for label, folder_path in folders_dict.items():
        print(f"  Loading {label}...")
        times, rho_central = load_evolution_data(folder_path)

        # Apply time mask
        mask = times <= t_max
        data[label] = {
            't': times[mask],
            'rho_c': rho_central[mask],
            'rho_c_0': rho_central[0],  # Initial central density
            'dr': resolutions_dict[label]['dr'],
            'N': resolutions_dict[label]['N'],
        }
        print(f"    {len(times[mask])} points, ρ_c(0) = {rho_central[0]:.6e}")
    
    # =========================================================
    # Get grid spacings
    # =========================================================
    dr_l = data[low_res]['dr']   # low resolution, coarsest
    dr_m = data[med_res]['dr']  # medium resolution
    dr_h = data[high_res]['dr']    # high resolution, finest

    print(f"\nGrid spacings:")
    print(f"  Δr_l ({low_res}) = {dr_l}")
    print(f"  Δr_m ({med_res}) = {dr_m}")
    print(f"  Δr_h ({high_res}) = {dr_h}")
    
    # =========================================================
    # Compute expected Q values for different orders
    # =========================================================
    print(f"\nExpected Q values:")
    for order in [2, 3, 4, 5]:
        Q_expected = compute_expected_Q(dr_l, dr_m, dr_h, order)
        print(f"  Order {order}: Q = {Q_expected:.2f}")
    
    # =========================================================
    # Find nearest time points (no interpolation)
    # =========================================================
    print("\nFinding nearest time points for comparison...")

    # Use the coarsest time grid (low_res) as reference
    t_common = data[low_res]['t']

    # Find nearest time points in medium and high resolution data
    rho_l = data[low_res]['rho_c']  # Already on common grid
    indices_m = find_nearest_time_indices(t_common, data[med_res]['t'])
    indices_h = find_nearest_time_indices(t_common, data[high_res]['t'])
    rho_m = data[med_res]['rho_c'][indices_m]
    rho_h = data[high_res]['rho_c'][indices_h]
    
    # =========================================================
    # Compute Q(t) = (ρ_l - ρ_m) / (ρ_m - ρ_h)
    # =========================================================
    print("\nComputing Q(t)...")

    # Method 1: Point-by-point Q (noisy for oscillating signals)
    # Compute differences
    diff_lm = rho_l - rho_m  # Low - Medium
    diff_mh = rho_m - rho_h  # Medium - High

    # Compute Q(t) avoiding division by very small numbers
    epsilon = 1e-20
    Q_t = diff_lm / (diff_mh + epsilon * np.sign(diff_mh + 1e-30))

    # Filter out extreme values and invalid data
    valid_mask = (t_common >= 0) & np.isfinite(Q_t) & (np.abs(Q_t) < 50)
    t_valid = t_common[valid_mask]
    Q_valid = Q_t[valid_mask]

    # Method 2: Windowed Q using L2 norm (robust for oscillating signals)
    print("Computing windowed Q-factor (L2 norm over temporal windows)...")
    window_size = min(args.window_size, len(t_common) // 10)  # Adaptive window size
    t_windows, Q_windows = compute_Q_windowed(t_common, rho_l, rho_m, rho_h,
                                               window_size=window_size, overlap=0.5)
    print(f"  Window size: {window_size} points (~{window_size * np.mean(np.diff(t_common)):.1f} M_sun)")
    print(f"  Computed {len(Q_windows)} windowed Q values")
    
    # =========================================================
    # Create figure like in the paper (2 panels)
    # =========================================================
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1, ax2 = axes
    
    # ---------------------------------------------------------
    # Top panel: Central density evolution (normalized)
    # ρ_c(t) / ρ_c(0)
    # ---------------------------------------------------------
    for label in [high_res, med_res, low_res]:  # Plot finest first (bottom of legend)
        d = data[label]
        rho_normalized = d['rho_c'] / d['rho_c_0']
        N = d['N']
        dr = d['dr']
        ax1.plot(d['t'], rho_normalized,
                label=f'Δr = {dr}',
                color=get_color(label, data.keys()),
                linewidth=1.0)
    
    ax1.set_ylabel(r'$\rho_c(t) / \rho_c(0)$', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim(0.995, 1.005)  # Wider range to see general behavior
    ax1.ticklabel_format(axis='y', style='plain', useOffset=False)
    
    # ---------------------------------------------------------
    # Bottom panel: Q(t) convergence factor
    # ---------------------------------------------------------
    # Compute expected Q for your scheme order (adjust as needed)
    # The paper uses 3rd order scheme, adjust this for your scheme
    scheme_order = 5  # MP5 is 5th order in smooth regions
    Q_expected = compute_expected_Q(dr_l, dr_m, dr_h, scheme_order)

    # Plot Q(t) point-by-point (noisy)
    ax2.plot(t_valid, Q_valid, 'k-', alpha=0.2, linewidth=0.5, label='Q(t) point-by-point')

    # Plot windowed Q (robust method) - MAIN RESULT
    if len(Q_windows) > 0:
        ax2.plot(t_windows, Q_windows, 'b-', linewidth=2.5, marker='o', markersize=3,
                label='Q(t) L2-windowed', zorder=10)

        # Compute statistics for windowed Q
        late_mask_win = t_windows > 20
        if np.sum(late_mask_win) > 0:
            Q_mean_win = np.mean(Q_windows[late_mask_win])
            Q_std_win = np.std(Q_windows[late_mask_win])
            print(f"\nWindowed Q statistics (t > 200):")
            print(f"  Mean: {Q_mean_win:.2f} ± {Q_std_win:.2f}")

    # Plot smoothed point-by-point Q (for comparison)
    if len(Q_valid) > 51:
        Q_smooth = smooth_data(Q_valid, window=51, polyorder=3)
        ax2.plot(t_valid, Q_smooth, 'orange', linewidth=1.5, alpha=0.6,
                label='Q(t) point-smoothed', linestyle='--')

    # Plot expected Q values as horizontal lines
    for order, (ls, color) in zip([2, 3], [('--', 'gray'), (':', 'red')]):
        Q_exp = compute_expected_Q(dr_l, dr_m, dr_h, order)
        ax2.axhline(Q_exp, ls=ls, color=color, linewidth=1.5,
                   label=f'{order}nd order (Q={Q_exp:.0f})' if order == 2 else f'{order}rd order (Q={Q_exp:.0f})')
    
    ax2.set_xlabel(r'$t$ [$M_\odot$]', fontsize=12)
    ax2.set_ylabel(r'$Q(t)$', fontsize=12)
    ax2.set_xlim(0, t_max)
    ax2.set_ylim(0, 20)  # Adjusted for uniform refinement factor 2 (2nd order Q=4, 3rd order Q=8)
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()

    # Save figure
    output_file = os.path.join(plots_dir, 'convergence_Q_factor.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")
    plt.show()
    #plt.close('all')

    # =========================================================
    # Print convergence analysis summary
    # =========================================================
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS SUMMARY (Q-factor method)")
    print("="*60)

    print(f"\nGrid configuration:")
    print(f"  r_max = {r_max}")
    print(f"  Resolutions: {low_res}, {med_res}, {high_res}")
    print(f"  Δr = [{dr_l}, {dr_m}, {dr_h}]")
    print(f"  dt = 0.1 * Δr = [{0.1*dr_l}, {0.1*dr_m}, {0.1*dr_h}]")
    
    print(f"\nExpected Q values by order:")
    print(f"  2nd order: Q = {compute_expected_Q(dr_l, dr_m, dr_h, 2):.2f}")
    print(f"  3rd order: Q = {compute_expected_Q(dr_l, dr_m, dr_h, 3):.2f}")

    # Statistics for point-by-point Q
    if len(Q_valid) > 0:
        # Statistics after transient (t > 200)
        late_mask = t_valid > 200
        if np.sum(late_mask) > 0:
            Q_late = Q_valid[late_mask]
            print(f"\nPoint-by-point Q statistics (t > 200 M_sun) [NOISY]:")
            print(f"  Mean Q:   {np.mean(Q_late):.2f}")
            print(f"  Median Q: {np.median(Q_late):.2f}")
            print(f"  Std Q:    {np.std(Q_late):.2f}")

    # Statistics for windowed Q (more reliable)
    if len(Q_windows) > 0:
        late_mask_win = t_windows > 200
        if np.sum(late_mask_win) > 0:
            Q_late_win = Q_windows[late_mask_win]
            print(f"\nWindowed Q statistics (t > 200 M_sun) [ROBUST]:")
            print(f"  Mean Q:   {np.mean(Q_late_win):.2f}")
            print(f"  Median Q: {np.median(Q_late_win):.2f}")
            print(f"  Std Q:    {np.std(Q_late_win):.2f}")

            # Estimate effective convergence order
            # Q = 2^n for ratio 2 refinement
            # So n = log2(Q)
            Q_mean = np.mean(Q_late_win)
            if Q_mean > 0:
                n_effective = np.log2(Q_mean)
                print(f"\n  *** Effective convergence order: n ≈ {n_effective:.2f} ***")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()