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
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

# Resolutions - UPDATE THESE TO CHANGE RESOLUTIONS
N_low = 400
N_med = 1000
N_high = 400

# Data paths - constructed from resolution values
FOLDERS = {
    f'N={N_low}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_last_politro/tov_star_rhoc1p28em03_N{N_low}_K100_G2_cow_mp5',
    f'N={N_med}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_last_domain/tov_star_rhoc1p28em03_N{N_med}_K100_G2_cow_mp5',
    f'N={N_high}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_last_politro/tov_star_rhoc1p28em03_N{N_high}_K100_G2_cow_mp5',
}

# Resolution labels (keys to FOLDERS dictionary)
low_res = f'N={N_low}'
med_res = f'N={N_med}'
high_res = f'N={N_high}'

# Grid spacings (dr = 40/N)
R_MAX = 40.0
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


def interpolate_to_common_times_spline(t_common, t_data, f_data):
    """
    Interpolate data to common time points using cubic spline.

    Uses scipy's CubicSpline for accurate interpolation as recommended
    in the paper (cubic spline has convergence order 4).

    Args:
        t_common: target time array
        t_data: source time array
        f_data: source data array

    Returns:
        Interpolated data at t_common
    """
    # Ensure time array is strictly increasing by removing duplicates
    # Keep the last occurrence of each duplicate time value
    _, unique_indices = np.unique(t_data[::-1], return_index=True)
    unique_indices = len(t_data) - 1 - unique_indices  # Reverse indices back
    unique_indices = np.sort(unique_indices)

    t_clean = t_data[unique_indices]
    f_clean = f_data[unique_indices]

    # Create cubic spline interpolator
    cs = CubicSpline(t_clean, f_clean)
    return cs(t_common)


def smooth_data(y, window=51, polyorder=3):
    """Apply Savitzky-Golay smoothing."""
    if len(y) < window:
        return y
    return savgol_filter(y, window, polyorder)


def main():
    t_max = 2000.0
    
    # =========================================================
    # Load evolution data for all resolutions
    # =========================================================
    print("Loading evolution data...")
    
    data = {}
    for label, folder_path in FOLDERS.items():
        print(f"  Loading {label}...")
        times, rho_central = load_evolution_data(folder_path)
        
        # Apply time mask
        mask = times <= t_max
        data[label] = {
            't': times[mask],
            'rho_c': rho_central[mask],
            'rho_c_0': rho_central[0],  # Initial central density
            'dr': RESOLUTIONS[label]['dr'],
            'N': RESOLUTIONS[label]['N'],
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
    # Interpolate to common time grid using cubic spline
    # =========================================================
    print("\nInterpolating to common time grid...")

    # Use the coarsest time grid (low_res) as reference
    # This is conservative - we don't create artificial data points
    t_common = data[low_res]['t']

    # Interpolate medium and high resolution data to common times
    rho_l = data[low_res]['rho_c']  # Already on common grid
    rho_m = interpolate_to_common_times_spline(t_common, data[med_res]['t'], data[med_res]['rho_c'])
    rho_h = interpolate_to_common_times_spline(t_common, data[high_res]['t'], data[high_res]['rho_c'])
    
    # =========================================================
    # Compute Q(t) = (ρ_l - ρ_m) / (ρ_m - ρ_h)
    # =========================================================
    print("\nComputing Q(t)...")
    
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
                color=COLORS[label],
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
    
    # Plot Q(t)
    ax2.plot(t_valid, Q_valid, 'k-', alpha=0.4, linewidth=0.5, label='Q(t)')
    
    # Plot smoothed Q(t)
    if len(Q_valid) > 51:
        Q_smooth = smooth_data(Q_valid, window=51, polyorder=3)
        ax2.plot(t_valid, Q_smooth, 'b-', linewidth=1.5, label='Q(t) smoothed')
    
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    output_file = os.path.join(plots_dir, 'convergence_Q_factor.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")
    plt.show()
    
    # =========================================================
    # Print convergence analysis summary
    # =========================================================
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS SUMMARY (Q-factor method)")
    print("="*60)
    
    print(f"\nGrid configuration:")
    print(f"  r_max = {R_MAX}")
    print(f"  Resolutions: {low_res}, {med_res}, {high_res}")
    print(f"  Δr = [{dr_l}, {dr_m}, {dr_h}]")
    print(f"  dt = 0.1 * Δr = [{0.1*dr_l}, {0.1*dr_m}, {0.1*dr_h}]")
    
    print(f"\nExpected Q values by order:")
    print(f"  2nd order: Q = {compute_expected_Q(dr_l, dr_m, dr_h, 2):.2f}")
    print(f"  3rd order: Q = {compute_expected_Q(dr_l, dr_m, dr_h, 3):.2f}")

    if len(Q_valid) > 0:
        # Statistics after transient (t > 200)
        late_mask = t_valid > 200
        if np.sum(late_mask) > 0:
            Q_late = Q_valid[late_mask]
            print(f"\nMeasured Q statistics (t > 200 M_sun):")
            print(f"  Mean Q:   {np.mean(Q_late):.2f}")
            print(f"  Median Q: {np.median(Q_late):.2f}")
            print(f"  Std Q:    {np.std(Q_late):.2f}")
            
            # Estimate effective convergence order
            # Q = 2^n for ratio 2 refinement
            # So n = log2(Q)
            Q_mean = np.mean(Q_late)
            if Q_mean > 0:
                n_effective = np.log2(Q_mean)
                print(f"\n  Effective convergence order: n ≈ {n_effective:.2f}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()