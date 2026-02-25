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

Spatial and Temporal alignment:
  - All resolutions aligned to common physical radius using cubic spatial interpolation
  - Common radius: r[3] from finest resolution
  - Common time grid: coarsest resolution as reference
  - Cubic spline interpolation used for both spatial and temporal alignment
  - Ensures all resolutions evaluated at exactly the same physical position and times
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import argparse
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# Resolutions - UPDATE THESE TO CHANGE RESOLUTIONS
N_low = 100
N_med = 200
N_high = 400

# Data paths - constructed from resolution values
FOLDERS = {
    f'N={N_low}': f'/home/davidbamba/repositories/engrenage/examples/TOV/data/tov_evolution_data_refact_rmax20.0_jax/tov_star_rhoc1p28em03_N{N_low}_K100_G2_dyn_mp5',
    f'N={N_med}': f'/home/davidbamba/repositories/engrenage/examples/TOV/data/tov_evolution_data_refact_rmax20.0_jax/tov_star_rhoc1p28em03_N{N_med}_K100_G2_dyn_mp5',
    f'N={N_high}': f'/home/davidbamba/repositories/engrenage/examples/TOV/data/tov_evolution_data_refact_rmax20.0_jax/tov_star_rhoc1p28em03_N{N_high}_K100_G2_dyn_mp5',
}

# Resolution labels (keys to FOLDERS dictionary)
low_res = f'N={N_low}'
med_res = f'N={N_med}'
high_res = f'N={N_high}'

# Grid spacings (dr = 20/N)
R_MAX = 20.0
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


def load_snapshots(folder_path, suffix="_dyn_jax"):
    """
    Load snapshots with density profiles and times.

    Returns:
        times: array of snapshot times
        r: radial grid
        rho_list: list of density profiles rho0(r) at each time
    """
    h5_file = os.path.join(folder_path, f'tov_snapshots{suffix}.h5')

    times = []
    rho_list = []

    with h5py.File(h5_file, 'r') as f:
        # Load grid
        r = f['grid/r'][:]

        # Get sorted snapshot keys
        snap_keys = sorted([k for k in f['snapshots'].keys()])

        for key in snap_keys:
            snap = f[f'snapshots/{key}']
            times.append(snap.attrs['time'])
            rho_list.append(snap['primitives/rho0'][:])

    return np.array(times), r, rho_list


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
  python convergence_test.py                                    # Use default folders
  python convergence_test.py --data-dirs D1 D2 D3              # Exactly 3 directories
  python convergence_test.py --t-min 50 --t-max 500            # Analyze from t=50 to t=500
  python convergence_test.py --r-max 100 --window-size 50      # Different domain and window
'''
    )
    parser.add_argument('--data-dirs', nargs='+', default=None,
                        help='List of data directories (exactly 3 required). Default: use FOLDERS')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for plots. Default: script_dir/plots')
    parser.add_argument('--t-min', type=float, default=0.0,
                        help='Minimum time to analyze. Default: 0.0')
    parser.add_argument('--t-max', type=float, default=2000.0,
                        help='Maximum time to plot. Default: 2000.0')
    parser.add_argument('--r-max', type=float, default=40.0,
                        help='Domain size for computing dr. Default: 40.0')
    parser.add_argument('--window-size', type=int, default=100,
                        help='Window size for L2 windowed Q-factor. Default: 100')
    args = parser.parse_args()

    t_min = args.t_min
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
    # Load snapshots for all resolutions
    # =========================================================
    print("Loading snapshots...")

    data = {}
    for label, folder_path in folders_dict.items():
        print(f"  Loading {label}...")
        times, r, rho_list = load_snapshots(folder_path)

        # Apply time mask
        mask = (times >= t_min) & (times <= t_max)
        data[label] = {
            't': times[mask],
            'r': r,
            'rho': [rho_list[i] for i in range(len(times)) if mask[i]],
            'dr': resolutions_dict[label]['dr'],
            'N': resolutions_dict[label]['N'],
        }
        print(f"    {len(times[mask])} snapshots, N={len(r)} points, r range: [{r[0]:.6f}, {r[-1]:.6f}]")
    
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
    # Interpolate central density: spatial then temporal
    # =========================================================
    print("\nInterpolating central density (spatial + temporal cubic interpolation)...")

    # Step 1: Define common radius for spatial interpolation
    # Use r[3] from the finest resolution as reference
    r_common = data[high_res]['r'][3]
    print(f"  Using common radius from finest resolution: r_common = r[3] = {r_common:.6f}")
    print(f"  Grid positions r[3] for comparison:")
    for label in [low_res, med_res, high_res]:
        print(f"    {label}: r[3] = {data[label]['r'][3]:.6f}")

    # Helper function: interpolate density to fixed physical position
    def get_rhoc_at_r(r_grid, rho_profile, r_target):
        """Interpolate density profile to target radius using cubic interpolation."""
        interp_func = interp1d(r_grid, rho_profile, kind='cubic',
                               bounds_error=False, fill_value='extrapolate')
        return interp_func(r_target)

    # Step 2: Extract central density at common radius for each snapshot
    print(f"  Extracting ρ(r={r_common:.6f}) for each snapshot (spatial cubic interpolation)...")
    rhoc_l_native = np.array([get_rhoc_at_r(data[low_res]['r'], rho, r_common)
                              for rho in data[low_res]['rho']])
    rhoc_m_native = np.array([get_rhoc_at_r(data[med_res]['r'], rho, r_common)
                              for rho in data[med_res]['rho']])
    rhoc_h_native = np.array([get_rhoc_at_r(data[high_res]['r'], rho, r_common)
                              for rho in data[high_res]['rho']])

    # Step 3: Interpolate central density to common times (temporal)
    def interpolate_rhoc_in_time(times_source, rhoc_source, times_target):
        """Interpolate central density to target times using cubic interpolation."""
        # Remove duplicate times
        unique_indices = np.unique(times_source, return_index=True)[1]
        unique_indices = np.sort(unique_indices)
        times_unique = times_source[unique_indices]
        rhoc_unique = rhoc_source[unique_indices]

        interp_func = interp1d(times_unique, rhoc_unique, kind='cubic',
                               bounds_error=False, fill_value='extrapolate')
        return interp_func(times_target)

    # Use the coarsest time grid (low_res) as reference
    t_common = data[low_res]['t']

    print(f"  Interpolating ρ_c to common time grid (temporal cubic interpolation)...")
    rho_l = interpolate_rhoc_in_time(data[low_res]['t'], rhoc_l_native, t_common)
    rho_m = interpolate_rhoc_in_time(data[med_res]['t'], rhoc_m_native, t_common)
    rho_h = interpolate_rhoc_in_time(data[high_res]['t'], rhoc_h_native, t_common)

    print(f"  Central density at r={r_common:.6f} interpolated for {len(t_common)} time points")
    
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
    # Calculate normalized density for all resolutions
    rho_l_normalized = rho_l / rho_l[0]
    rho_m_normalized = rho_m / rho_m[0]
    rho_h_normalized = rho_h / rho_h[0]

    # Plot all three resolutions
    ax1.plot(t_common, rho_h_normalized, label=f'Δr = {data[high_res]["dr"]}',
            color='#d62728', linewidth=1.0)  # red (finest)
    ax1.plot(t_common, rho_m_normalized, label=f'Δr = {data[med_res]["dr"]}',
            color='#1f77b4', linewidth=1.0)  # blue (medium)
    ax1.plot(t_common, rho_l_normalized, label=f'Δr = {data[low_res]["dr"]}',
            color='#2ca02c', linewidth=1.0)  # green (coarse)

    # Set y-limits based on low resolution oscillations (noisiest)
    rho_low_min = np.min(rho_l_normalized)
    rho_low_max = np.max(rho_l_normalized)
    margin = (rho_low_max - rho_low_min) * 0.1  # 10% margin on each side
    ax1.set_ylim(rho_low_min - margin, rho_low_max + margin)

    ax1.set_ylabel(r'$\rho_c(t) / \rho_c(0)$', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
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
    print(f"  Time range: t = [{t_min}, {t_max}]")
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