#!/usr/bin/env python3
"""
Convergence analysis for TOV star evolution - Baryon Mass
Calculates baryon mass from snapshots and convergence order.

Baryon mass: M_b = 4π ∫ D * r² * √γ_rr dr
where D is the conserved rest-mass density and γ_rr is the radial metric component.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from scipy.signal import savgol_filter

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


def load_snapshots_with_primitives(folder_path):
    """
    Load snapshots with primitives (rho0, W) and conformal factor (phi).
    Returns times, r grid, and lists of rho0, W, phi.
    """
    h5_file = os.path.join(folder_path, 'tov_snapshots_cow.h5')

    times = []
    rho0_list = []
    W_list = []
    phi_list = []

    with h5py.File(h5_file, 'r') as f:
        # Load grid
        r = f['grid/r'][:]

        snaps = f['snapshots']
        for key in sorted(snaps.keys()):
            g = snaps[key]
            times.append(g.attrs['time'])
            rho0_list.append(g['primitives/rho0'][:])
            W_list.append(g['primitives/W'][:])
            phi_list.append(g['bssn/phi'][:])

    return np.array(times), r, rho0_list, W_list, phi_list


def compute_baryon_mass(r, rho0, W, phi):
    """
    Compute baryon mass: M_b = 4π ∫ ρ₀ W ψ⁶ r² dr

    This matches the formula in utils_TOVEvolution.py:
    - ρ₀ = rest-mass density (primitive)
    - W = Lorentz factor
    - ψ = e^φ = conformal factor
    - ψ⁶ = e^{6φ} = conformal 3-metric determinant
    """
    from scipy.integrate import simpson

    # Only integrate where r > 0 (avoid singularity at origin)
    mask = r > 0
    r_pos = r[mask]
    rho0_pos = rho0[mask]
    W_pos = W[mask]
    phi_pos = phi[mask]

    # Compute ψ⁶ = e^{6φ}
    psi6 = np.exp(6.0 * phi_pos)

    # Integrand: ρ₀ W ψ⁶ r²
    integrand = rho0_pos * W_pos * psi6 * r_pos**2

    # Integrate using Simpson's rule
    M_b = 4.0 * np.pi * simpson(integrand, x=r_pos)

    return M_b


def running_average(x, window):
    """Compute running average with given window size."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')


def smooth(y, window=20, polyorder=4):
    """Apply light Savitzky-Golay smoothing to data."""
    if len(y) < window:
        return y
    return savgol_filter(y, window, polyorder)


def interpolate_to_common_times(t_ref, M_ref, t_other, M_other):
    """Interpolate M_other to the time points of t_ref."""
    return np.interp(t_ref, t_other, M_other)


def main():
    t_max = 2000.0

    # =========================================================
    # Load all data and compute baryon mass for each snapshot
    # =========================================================
    data = {}

    for label, folder_path in FOLDERS.items():
        print(f"Processing {label}...")
        times, r, rho0_list, W_list, phi_list = load_snapshots_with_primitives(folder_path)

        # Compute baryon mass for each snapshot
        M_b_list = []
        for i in range(len(times)):
            M_b = compute_baryon_mass(r, rho0_list[i], W_list[i], phi_list[i])
            M_b_list.append(M_b)

        M_b_arr = np.array(M_b_list)

        # Apply time mask
        mask = times <= t_max
        data[label] = {
            't': times[mask],
            'M_b': M_b_arr[mask],
            'M_b_0': M_b_arr[0]  # Initial baryon mass
        }
        print(f"  {label}: {len(times[mask])} snapshots, M_b(0) = {M_b_arr[0]:.6f}")

    # =========================================================
    # Compute L1 errors relative to initial mass (mass conservation)
    # =========================================================
    print("\n--- Computing mass conservation errors ---")

    l1_mass_error = {}
    for label, d in data.items():
        M_b = d['M_b']
        M_b_0 = d['M_b_0']
        # Relative mass error at each time
        rel_error = np.abs(M_b - M_b_0) / M_b_0
        l1_mass_error[label] = rel_error
        print(f"  {label}: max |ΔM_b/M_b_0| = {np.max(rel_error):.2e}")

    # =========================================================
    # Compute convergence order
    # =========================================================
    print("\n--- Computing convergence order ---")

    t_1 = data[low_res]['t']
    t_2 = data[med_res]['t']
    t_3 = data[high_res]['t']

    M_1 = data[low_res]['M_b']
    M_2 = data[med_res]['M_b']
    M_3 = data[high_res]['M_b']

    # Interpolate to common time grid (low_res, coarsest)
    M_2_interp = interpolate_to_common_times(t_1, M_1, t_2, M_2)
    M_3_interp = interpolate_to_common_times(t_1, M_1, t_3, M_3)

    # Compute pointwise errors
    E12 = np.abs(M_1 - M_2_interp)
    E23 = np.abs(M_2_interp - M_3_interp)

    # Convergence order: p = log(E12/E23) / log(2)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.log(E12 / E23) / np.log(2.0)

    p_avg = running_average(p, window=50)

    # =========================================================
    # Create figure with 2x2 subplots
    # =========================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    # Plot 1: Baryon mass evolution
    for (label, d), color in zip(data.items(), COLORS):
        ax1.plot(d['t'], smooth(d['M_b']), label=label, color=color, linewidth=0.8)

    ax1.set_xlabel(r'$t$ [M$_\odot$]')
    ax1.set_ylabel(r'$M_b$')
    ax1.set_title(r'(a) Baryon Mass Evolution')
    ax1.legend(fontsize=8)
    #ax1.grid(alpha=0.3)

    # Plot 2: Relative baryon mass change (mass conservation)
    for (label, d), color in zip(data.items(), COLORS):
        M_b = d['M_b']
        M_b_0 = d['M_b_0']
        rel_change = (M_b - M_b_0) / M_b_0
        ax2.plot(d['t'], smooth(rel_change), label=label, color=color, linewidth=0.8)

    ax2.set_xlabel(r'$t$ [M$_\odot$]')
    ax2.set_ylabel(r'$(M_b - M_{b,0}) / M_{b,0}$')
    ax2.set_title(r'(b) Relative Baryon Mass Change (Conservation)')
    ax2.legend(fontsize=8)
    #ax2.grid(alpha=0.3)

    # Plot 3: L1 error of mass conservation (log scale)
    for (label, err), color in zip(l1_mass_error.items(), COLORS):
        t = data[label]['t']
        ax3.semilogy(t, smooth(err), label=label, color=color, linewidth=0.8)

    ax3.set_xlabel(r'$t$ [M$_\odot$]')
    ax3.set_ylabel(r'$|M_b - M_{b,0}| / M_{b,0}$')
    ax3.set_title(r'(c) Mass Conservation Error')
    ax3.legend(fontsize=8)
    #ax3.grid(alpha=0.3)

    # Plot 4: Convergence order
    valid = t_1 > 100
    ax4.plot(t_1[valid], p[valid], color='k', alpha=0.3, lw=0.6, label='instantaneous')
    ax4.plot(t_1[valid], p_avg[valid], color='k', lw=2.5, label='running average')

    ax4.axhline(2, ls='--', color='gray', label='2nd order')
    ax4.axhline(3, ls=':', color='gray', label='3rd order')
    ax4.axhline(5, ls='-.', color='gray', label='5th order')

    ax4.set_xlim(0, t_max)
    ax4.set_ylim(-1, 8)
    ax4.set_xlabel(r'$t$ [M$_\odot$]')
    ax4.set_ylabel(r'Convergence order $p(t)$')
    ax4.set_title(rf'(d) Convergence Order of $M_b$ (triplet {low_res}-{med_res}-{high_res})')
    ax4.legend(fontsize=8)
    #ax4.grid(alpha=0.3)

    plt.tight_layout()

    # Save figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'baryon_mass_convergence.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # =========================================================
    # Print convergence analysis
    # =========================================================
    print("\n" + "="*60)
    print(f"BARYON MASS CONVERGENCE ANALYSIS (triplet {low_res}, {med_res}, {high_res})")
    print("="*60)

    valid_mask = t_1 > 100
    p_valid = p[valid_mask]
    t_valid = t_1[valid_mask]
    E12_valid = E12[valid_mask]
    E23_valid = E23[valid_mask]

    finite_mask = np.isfinite(p_valid)
    p_finite = p_valid[finite_mask]

    print(f"\nTime range analyzed: t = [{t_valid[0]:.1f}, {t_valid[-1]:.1f}] M_sun")
    print(f"Number of snapshots: {len(p_finite)} (valid, finite)")

    print(f"\n--- Initial Baryon Mass ---")
    for label, d in data.items():
        print(f"  {label}: M_b(0) = {d['M_b_0']:.6f}")

    print(f"\n--- Pointwise Error Statistics ---")
    print(f"E({low_res}, {med_res}):   mean = {np.mean(E12_valid):.3e}, max = {np.max(E12_valid):.3e}")
    print(f"E({med_res}, {high_res}):   mean = {np.mean(E23_valid):.3e}, max = {np.max(E23_valid):.3e}")
    ratio = np.mean(E12_valid) / np.mean(E23_valid) if np.mean(E23_valid) > 0 else np.nan
    print(f"Error ratio E12/E23: {ratio:.2f} (expected ~4 for 2nd order)")

    print(f"\n--- Convergence Order p(t) ---")
    if len(p_finite) > 0:
        print(f"Mean:   {np.mean(p_finite):.2f}")
        print(f"Median: {np.median(p_finite):.2f}")
        print(f"Std:    {np.std(p_finite):.2f}")
        print(f"Min:    {np.min(p_finite):.2f}")
        print(f"Max:    {np.max(p_finite):.2f}")

        p25, p50, p75 = np.percentile(p_finite, [25, 50, 75])
        print(f"\nPercentiles: 25%={p25:.2f}, 50%={p50:.2f}, 75%={p75:.2f}")

    intervals = [(100, 1000)  ]#, (1000, 3000), (3000, 5000), (5000, 7000)]
    print(f"\n--- Convergence by time interval ---")
    for t_start, t_end in intervals:
        mask_int = (t_valid >= t_start) & (t_valid < t_end) & np.isfinite(p_valid)
        if np.sum(mask_int) > 0:
            p_int = p_valid[mask_int]
            print(f"  t=[{t_start:4d}, {t_end:4d}]: p = {np.mean(p_int):.2f} +/- {np.std(p_int):.2f}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
