#!/usr/bin/env python3
"""
Convergence Test for TOV Star Evolution

Point-wise convergence test following the methodology of arXiv:2509.15303.
Computes Q(t) = (u_low - u_med) / (u_med - u_high) at each time point.

For resolution ratios 1:2:4 (N=200, 400, 800) and convergence order n,
the expected Q value is:
    Q_expected = (dr_l^n - dr_m^n) / (dr_m^n - dr_h^n)

With dr_l=2, dr_m=1, dr_h=0.5 (relative to medium):
    n=2: Q = (4-1)/(1-0.25) = 3/0.75 = 4
    n=3: Q = (8-1)/(1-0.125) = 7/0.875 = 8
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from scipy.interpolate import interp1d


def load_evolution_data(folder_path):
    """Load evolution data from folder (npz or h5 file)."""
    # Try npz first
    npz_file = os.path.join(folder_path, 'timeseries.npz')
    if os.path.exists(npz_file):
        npz = np.load(npz_file)
        return {'time': npz['times'], 'rho_central': npz['rho_central']}

    # Try h5 file
    h5_file = os.path.join(folder_path, 'tov_evolution_cow.h5')
    if os.path.exists(h5_file):
        with h5py.File(h5_file, 'r') as f:
            return {key: f[key][:] for key in f.keys()}

    raise FileNotFoundError(f"No data found in {folder_path}")


def compute_Q_expected(dr_l, dr_m, dr_h, order):
    """
    Compute expected convergence factor Q for given grid spacings and order.

    Q = (dr_l^n - dr_m^n) / (dr_m^n - dr_h^n)
    """
    return (dr_l**order - dr_m**order) / (dr_m**order - dr_h**order)


def plot_convergence(data, t_common, output_path=None):
    """
    Generate convergence analysis plots following arXiv:2509.15303 methodology.

    Two-panel plot:
    - Top: Central density evolution (normalized)
    - Bottom: Point-wise convergence factor Q(t)
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('TOV Star Convergence Test (N=100, 200, 400, 800)', fontsize=14, fontweight='bold')

    colors = {'N=100': 'blue', 'N=200': 'orange', 'N=400': 'green', 'N=800': 'red'}

    # Extract densities for Q calculation (use 3 consecutive resolutions)
    rho_100 = data['N=100']['rho_central']
    rho_200 = data['N=200']['rho_central']
    rho_400 = data['N=400']['rho_central']
    rho_800 = data['N=800']['rho_central']

    # --- Top panel: Normalized central density evolution ---
    ax = axes[0]
    for name in ['N=100', 'N=200', 'N=400', 'N=800']:
        rho_c = data[name]['rho_central']
        rho_c_0 = data[name]['rho_central'][0]
        delta_rho_rel = (rho_c - rho_c_0) / rho_c_0
        ax.plot(t_common, delta_rho_rel, label=name,
                color=colors[name], linewidth=0.8)

    ax.set_xlabel(r'$t/M_\odot$')
    ax.set_ylabel(r'$\varepsilon_c(t)/\varepsilon_c(0)$')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- Bottom panel: Point-wise convergence factor Q(t) ---
    ax = axes[1]

    # Compute Q(t) for N100-N200-N400
    eps = 1e-20
    diff_100_200 = rho_100 - rho_200
    diff_200_400 = rho_200 - rho_400
    Q_t_1 = diff_100_200 / (diff_200_400 + eps * np.sign(diff_200_400 + eps))

    # Compute Q(t) for N200-N400-N800
    diff_200_400 = rho_200 - rho_400
    diff_400_800 = rho_400 - rho_800
    Q_t_2 = diff_200_400 / (diff_400_800 + eps * np.sign(diff_400_800 + eps))

    # Smooth Q(t) with moving average for cleaner visualization
    window = 15
    Q_smooth_1 = np.convolve(Q_t_1, np.ones(window)/window, mode='same')
    Q_smooth_2 = np.convolve(Q_t_2, np.ones(window)/window, mode='same')

    ax.plot(t_common, Q_smooth_1, 'b-', linewidth=1.2, label=r'$Q$ (N100-200-400)')
    ax.plot(t_common, Q_smooth_2, 'r-', linewidth=1.2, label=r'$Q$ (N200-400-800)')

    # Expected Q values for different orders (resolution ratio 1:2:4)
    # dr_l = 2*dr_m, dr_h = 0.5*dr_m (relative to medium)
    dr_l, dr_m, dr_h = 2.0, 1.0, 0.5
    Q_2nd = compute_Q_expected(dr_l, dr_m, dr_h, 2)  # = 4
    Q_3rd = compute_Q_expected(dr_l, dr_m, dr_h, 3)  # = 8

    ax.axhline(Q_2nd, color='red', linestyle='--', linewidth=2,
               label=f'2nd order (Q={Q_2nd:.1f})')
    ax.axhline(Q_3rd, color='purple', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'3rd order (Q={Q_3rd:.1f})')

    ax.set_xlabel(r'$t/M_\odot$')
    ax.set_ylabel(r'$Q(t)$')
    ax.set_ylim(0, 12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    # Skip first 10% for transient
    t_start_idx = len(t_common) // 10
    Q_mean_1 = np.mean(Q_smooth_1[t_start_idx:])
    Q_std_1 = np.std(Q_smooth_1[t_start_idx:])
    Q_mean_2 = np.mean(Q_smooth_2[t_start_idx:])
    Q_std_2 = np.std(Q_smooth_2[t_start_idx:])

    textstr = f'Q(100-200-400) = {Q_mean_1:.2f} ± {Q_std_1:.2f}\nQ(200-400-800) = {Q_mean_2:.2f} ± {Q_std_2:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, Q_mean_2, Q_std_2


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Define paths for each resolution
    data_paths = {
    'N=100': '/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data2/tov_star_rhoc1p28em03_N100_K100_G2_cow',
    'N=200': '/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data2/tov_star_rhoc1p28em03_N200_K100_G2_cow',
    'N=400': '/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data/tov_star_rhoc1p28em03_N400_K100_G2_cow_wz',
    'N=800': '/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data2/tov_star_rhoc1p28em03_N800_K100_G2_cow',
}
    # Load data for each resolution
    raw_data = {}

    print("Loading data...")
    for name, h5_path in data_paths.items():
        raw_data[name] = load_evolution_data(h5_path)
        t = raw_data[name]['time']
        print(f"  {name}: t=[{t[0]:.1f}, {t[-1]:.1f}], dt_mean={np.mean(np.diff(t)):.4f}, points={len(t)}")

    # Find common time range
    t_min = 0.0
    t_max = min([raw_data[name]['time'][-1] for name in raw_data])
    t_max = min(t_max, 5000.0)
    print(f"\nAnalysis time range: [{t_min:.1f}, {t_max:.1f}] M_sun")

    # Interpolate to common times using cubic spline (order 4, sufficient for 2nd/3rd order schemes)
    n_points = 2000
    t_common = np.linspace(t_min, t_max, n_points)
    data = {}
    for name, raw in raw_data.items():
        data[name] = {'time': t_common}
        t_raw = raw['time']
        for key in raw.keys():
            if key != 'time' and key != 'step':
                f = interp1d(t_raw, raw[key], kind='cubic', fill_value='extrapolate')
                data[name][key] = f(t_common)

    # Print convergence analysis header
    print("\n" + "="*70)
    print("POINT-WISE CONVERGENCE TEST (arXiv:2509.15303 methodology)")
    print("="*70)
    print("\nQ(t) = (u_low - u_med) / (u_med - u_high)")
    print("\nExpected Q values for resolution ratio 1:2:4:")
    print("  2nd order scheme: Q = 4.0")
    print("  3rd order scheme: Q = 8.0")

    # Generate plots
    output_path = os.path.join(plot_dir, 'convergence_test2.png')
    fig, Q_mean, Q_std = plot_convergence(data, t_common, output_path)

    print(f"\nRESULTS:")
    print(f"  Mean Q = {Q_mean:.2f} ± {Q_std:.2f}")
    if Q_mean > 0:
        # Estimate order from Q using log
        # Q = (2^n - 1) / (1 - 2^(-n)) for resolution ratio 2
        # Simplified: for ratio 2, Q ≈ 2^n for large n
        order_est = np.log2(Q_mean) if Q_mean > 1 else 0
        print(f"  Estimated order ≈ {order_est:.2f}")

    plt.show()


if __name__ == "__main__":
    main()
