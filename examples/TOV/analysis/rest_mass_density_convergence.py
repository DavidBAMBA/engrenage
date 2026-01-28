#!/usr/bin/env python3
"""
Convergence analysis for TOV star evolution - Rest-Mass Density L1 Norm

Calculates convergence order from L1-norm of rest-mass density differences
between resolutions, similar to Figure 16 in arXiv:1612.06251.

Uses 3 resolutions: N=200, N=400, N=800

Computes using three methods:

  1. Spherical L1 (consecutive): E = 4π ∫ |ρ₁ - ρ₂| dr
     - Compares N=200 vs N=400, N=400 vs N=800
     - p = log(E₁₂/E₂₃) / log(2)

  2. Discrete L1 (consecutive): E = (1/N) Σ |ρ₁ - ρ₂|
     - Compares N=200 vs N=400, N=400 vs N=800
     - p = log(E₁₂/E₂₃) / log(2)

  3. Paper method (vs highest resolution): Δf = (1/N) Σ |f - f̄|
     - Compares N=200 vs N=800, N=400 vs N=800
     - p = log(E₁/E₂) / log(2)
     - As in Appendix B of arXiv:1612.06251

All methods compute interior only (r ≤ R_star).
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from scipy.integrate import simpson
from scipy.signal import savgol_filter

# Resolutions - UPDATE THESE TO CHANGE RESOLUTIONS
N_low = 500
N_med = 1000
N_high = 2000

# Reference resolution for Figure 4 convergence plot
N_ref = 2000

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

# Reference resolution path for Figure 4
ref_res_label = f'N={N_ref}'
ref_res_path = f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_last_domain/tov_star_rhoc1p28em03_N{N_ref}_K100_G2_cow_mp5'

COLORS = ['#1f77b4', "#ff7f0e", '#2ca02c', "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def load_snapshots(folder_path):
    """
    Load snapshots with density profiles and times.

    Returns:
        times: array of snapshot times
        r: radial grid
        rho_list: list of density profiles rho0(r) at each time
    """
    h5_file = os.path.join(folder_path, 'tov_snapshots_cow.h5')

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


def compute_L1_error(r_coarse, rho_coarse, r_fine, rho_fine, r_max=None):
    """
    Compute L1 norm of density difference: E = 4π ∫ |ρ₁ - ρ₂| dr

    Interpolates fine resolution to coarse grid.

    Args:
        r_coarse: coarse grid radii
        rho_coarse: density on coarse grid
        r_fine: fine grid radii
        rho_fine: density on fine grid
        r_max: optional upper limit for integration (e.g., R_star for interior only)

    Returns:
        L1 error (scalar)
    """
    # Interpolate fine to coarse grid
    rho_fine_interp = np.interp(r_coarse, r_fine, rho_fine)

    # Only integrate where r > 0 (avoid singularity)
    mask = r_coarse > 0
    if r_max is not None:
        mask = mask & (r_coarse <= r_max)

    r_pos = r_coarse[mask]
    diff = np.abs(rho_coarse[mask] - rho_fine_interp[mask])

    # L1 norm with spherical volume element
    integrand = diff 

    return 4.0 * np.pi * simpson(integrand, x=r_pos)


def compute_L1_error_discrete(r_coarse, rho_coarse, r_fine, rho_fine, r_max=None):
    """
    Compute discrete L1 norm: E = (1/N) Σ |ρ₁ - ρ₂|

    Interpolates fine resolution to coarse grid.

    Args:
        r_coarse: coarse grid radii
        rho_coarse: density on coarse grid
        r_fine: fine grid radii
        rho_fine: density on fine grid
        r_max: optional upper limit (e.g., R_star for interior only)

    Returns:
        L1 error (scalar)
    """
    # Interpolate fine to coarse grid
    rho_fine_interp = np.interp(r_coarse, r_fine, rho_fine)

    # Only consider where r > 0
    mask = r_coarse > 0
    if r_max is not None:
        mask = mask & (r_coarse <= r_max)

    diff = np.abs(rho_coarse[mask] - rho_fine_interp[mask])

    # Discrete L1 norm (mean absolute difference)
    return np.mean(diff)


def find_stellar_radius(r, rho, rho_atm=1e-10):
    """
    Find stellar radius where density drops to atmosphere level.

    Args:
        r: radial grid
        rho: density profile
        rho_atm: atmosphere density threshold

    Returns:
        R_star: stellar radius
    """
    idx = np.where(rho > rho_atm)[0]
    if len(idx) > 0:
        return r[idx[-1]]
    return r.max()


def running_average(x, window):
    """Compute running average with given window size."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')


def smooth(y, window=15, polyorder=3):
    """Apply Savitzky-Golay smoothing to data."""
    if len(y) < window:
        return y
    return savgol_filter(y, window, polyorder)


def find_common_times(t1, t2, t3, rel_tol=0.05):
    """
    Find times that are approximately common to all three resolutions.
    Uses relative tolerance (5% by default) to match times.

    Returns indices for each array.
    """
    # Use coarsest time sampling as reference
    t_ref = t1  # N=200 typically has fewest snapshots per unit time

    idx1 = []
    idx2 = []
    idx3 = []

    for i, t in enumerate(t_ref):
        if t < 1e-10:  # Skip t=0
            idx1.append(i)
            idx2.append(0)
            idx3.append(0)
            continue

        # Find closest time in t2
        j = np.argmin(np.abs(t2 - t))
        if np.abs(t2[j] - t) / t > rel_tol:
            continue

        # Find closest time in t3
        k = np.argmin(np.abs(t3 - t))
        if np.abs(t3[k] - t) / t > rel_tol:
            continue

        idx1.append(i)
        idx2.append(j)
        idx3.append(k)

    return np.array(idx1), np.array(idx2), np.array(idx3)


def main():
    t_max = 1000.0

    # =========================================================
    # Load all data
    # =========================================================
    print("Loading snapshots...")

    data = {}
    for label, folder_path in FOLDERS.items():
        print(f"  {label}...")
        times, r, rho_list = load_snapshots(folder_path)
        data[label] = {
            't': times,
            'r': r,
            'rho': rho_list,
            'N': len(r)
        }
        print(f"    {len(times)} snapshots, N={len(r)} points")

    # =========================================================
    # Find common times across all resolutions
    # =========================================================
    print("\nFinding common times...")

    t1, t2, t3 = data[low_res]['t'], data[med_res]['t'], data[high_res]['t']
    idx1, idx2, idx3 = find_common_times(t1, t2, t3)

    # Apply time mask
    time_mask = t1[idx1] <= t_max
    idx1 = idx1[time_mask]
    idx2 = idx2[time_mask]
    idx3 = idx3[time_mask]

    common_times = t1[idx1]
    print(f"  Found {len(common_times)} common time points up to t={t_max}")

    # =========================================================
    # Find stellar radius from initial profile
    # =========================================================
    r1 = data[low_res]['r']
    r2 = data[med_res]['r']
    r3 = data[high_res]['r']

    rho1_list = data[low_res]['rho']
    rho2_list = data[med_res]['rho']
    rho3_list = data[high_res]['rho']

    # Use highest resolution initial profile to find R_star
    R_star = find_stellar_radius(r3, rho3_list[0])
    print(f"\nStellar radius: R_star = {R_star:.3f}")
    print(f"Domain extent:  r_max  = {r1.max():.3f}")

    # =========================================================
    # Load analytical TOV solution
    # =========================================================
    print("\nLoading analytical TOV solution...")
    tov_cache_dir = "/home/davidbamba/repositories/engrenage/examples/TOV/tov_iso_cache/TOVSOL_ISO_K=100.0_G=2.0_rho=1.280000e-03"
    try:
        r_tov_analytic = np.load(os.path.join(tov_cache_dir, "r_iso.npy"))
        rho_tov_analytic = np.load(os.path.join(tov_cache_dir, "rho_baryon.npy"))
        print(f"  Loaded analytical solution from cache")
        print(f"  Points: {len(r_tov_analytic)}, max radius: {r_tov_analytic.max():.3f}")
        has_analytical = True
    except Exception as e:
        print(f"  Warning: Could not load analytical solution ({e})")
        has_analytical = False

    # =========================================================
    # Compute L1 errors at each common time
    # =========================================================
    print("\nComputing L1 errors...")

    # Full domain errors (consecutive resolutions)
    E12 = []  # Error between N=100 and N=200
    E23 = []  # Error between N=200 and N=400

    # Interior only errors (r < R_star)
    E12_int = []
    E23_int = []

    # Discrete L1 errors (interior only)
    E12_disc = []
    E23_disc = []

    # Paper method: compare against highest resolution (N=400)
    E1_paper = []  # N=100 vs N=400
    E2_paper = []  # N=200 vs N=400

    for i, (i1, i2, i3) in enumerate(zip(idx1, idx2, idx3)):
        rho1 = rho1_list[i1]
        rho2 = rho2_list[i2]
        rho3 = rho3_list[i3]

        # Full domain (spherical L1)
        e12 = compute_L1_error(r1, rho1, r2, rho2)
        e23 = compute_L1_error(r2, rho2, r3, rho3)
        E12.append(e12)
        E23.append(e23)

        # Interior only (spherical L1)
        e12_int = compute_L1_error(r1, rho1, r2, rho2, r_max=R_star)
        e23_int = compute_L1_error(r2, rho2, r3, rho3, r_max=R_star)
        E12_int.append(e12_int)
        E23_int.append(e23_int)

        # Discrete L1 (interior only)
        e12_disc = compute_L1_error_discrete(r1, rho1, r2, rho2, r_max=R_star)
        e23_disc = compute_L1_error_discrete(r2, rho2, r3, rho3, r_max=R_star)
        E12_disc.append(e12_disc)
        E23_disc.append(e23_disc)

        # Paper method: discrete L1 vs highest resolution N=400 (interior only)
        e1_paper = compute_L1_error_discrete(r1, rho1, r3, rho3, r_max=R_star)
        e2_paper = compute_L1_error_discrete(r2, rho2, r3, rho3, r_max=R_star)
        E1_paper.append(e1_paper)
        E2_paper.append(e2_paper)

    E12 = np.array(E12)
    E23 = np.array(E23)
    E12_int = np.array(E12_int)
    E23_int = np.array(E23_int)
    E12_disc = np.array(E12_disc)
    E23_disc = np.array(E23_disc)
    E1_paper = np.array(E1_paper)
    E2_paper = np.array(E2_paper)

    # =========================================================
    # Compute convergence order
    # =========================================================
    print("\nComputing convergence order...")

    with np.errstate(divide='ignore', invalid='ignore'):
        # Full domain (spherical L1)
        p = np.log(E12 / E23) / np.log(2.0)
        # Interior only (spherical L1)
        p_int = np.log(E12_int / E23_int) / np.log(2.0)
        # Discrete L1 (interior only)
        p_disc = np.log(E12_disc / E23_disc) / np.log(2.0)
        # Paper method: p = log(E1/E2) / log(2) where E1, E2 are vs highest resolution
        p_paper = np.log(E1_paper / E2_paper) / np.log(2.0)

    p_int_avg = running_average(p_int, window=30)
    p_disc_avg = running_average(p_disc, window=30)
    p_paper_avg = running_average(p_paper, window=30)

    # =========================================================
    # Figure 1: Spherical L1 norm analysis
    # =========================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))

    valid = common_times > 50  # Skip initial transient

    # Plot 1: Central density evolution
    ax = axes1[0, 0]
    for (label, d), color in zip(data.items(), COLORS):
        rho_central = [rho[3] for rho in d['rho']]
        mask = d['t'] <= t_max
        ax.plot(d['t'][mask], rho_central[:np.sum(mask)],
                label=label, color=color, linewidth=0.8)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'$\rho_c$ (central density)')
    ax.set_title(r'(a) Central Density Evolution')
    ax.legend(fontsize=8)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Plot 2: Spherical L1 errors (interior)
    ax = axes1[0, 1]
    ax.semilogy(common_times, smooth(E12_int), label=rf'$E_{{12}}$ ({low_res} vs {med_res})',
                color=COLORS[0], linewidth=1.2)
    ax.semilogy(common_times, smooth(E23_int), label=rf'$E_{{23}}$ ({med_res} vs {high_res})',
                color=COLORS[1], linewidth=1.2)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'$L_1$ error')
    ax.set_title(rf'(b) Spherical $L_1$: $4\pi \int |\Delta\rho| dr$ ($r \leq {R_star:.1f}$)')
    ax.legend(fontsize=8)

    # Plot 3: Error ratio (spherical)
    ax = axes1[1, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_int = E12_int / E23_int
    ax.plot(common_times, smooth(ratio_int), color='k', linewidth=1.2)
    ax.axhline(4, ls='--', color='gray', label='Ratio=4 (2nd order)')
    ax.axhline(8, ls=':', color='gray', label='Ratio=8 (3rd order)')
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'$E_{12} / E_{23}$')
    ax.set_title(r'(c) Error Ratio')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 12)

    # Plot 4: Convergence order (spherical)
    ax = axes1[1, 1]
    ax.plot(common_times[valid], p_int[valid], color='k', alpha=0.3, lw=0.6,
            label='instantaneous')
    ax.plot(common_times[valid], p_int_avg[valid], color='k', lw=2.5,
            label='running average')
    ax.axhline(2, ls='--', color='gray', label='2nd order')
    ax.axhline(3, ls=':', color='gray', label='3rd order')
    ax.axhline(5, ls='-.', color='gray', label='5th order')
    ax.set_xlim(0, t_max)
    ax.set_ylim(-1, 8)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'Convergence order $p(t)$')
    ax.set_title(r'(d) Convergence Order $p(t)$')
    ax.legend(fontsize=8, loc='upper right')

    fig1.suptitle(r'Spherical $L_1$ Norm: $E = 4\pi \int |\rho_1 - \rho_2| r^2 dr$', fontsize=12, y=1.02)
    fig1.tight_layout()

    # =========================================================
    # Figure 2: Discrete L1 norm analysis
    # =========================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Central density evolution
    ax = axes2[0, 0]
    for (label, d), color in zip(data.items(), COLORS):
        rho_central = [rho[3] for rho in d['rho']]
        mask = d['t'] <= t_max
        ax.plot(d['t'][mask], rho_central[:np.sum(mask)],
                label=label, color=color, linewidth=0.8)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'$\rho_c$ (central density)')
    ax.set_title(r'(a) Central Density Evolution')
    ax.legend(fontsize=8)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Plot 2: Discrete L1 errors (interior)
    ax = axes2[0, 1]
    ax.semilogy(common_times, smooth(E12_disc), label=rf'$E_{{12}}$ ({low_res} vs {med_res})',
                color=COLORS[0], linewidth=1.2)
    ax.semilogy(common_times, smooth(E23_disc), label=rf'$E_{{23}}$ ({med_res} vs {high_res})',
                color=COLORS[1], linewidth=1.2)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'$L_1$ error')
    ax.set_title(rf'(b) Discrete $L_1$: $(1/N) \sum |\Delta\rho|$ ($r \leq {R_star:.1f}$)')
    ax.legend(fontsize=8)

    # Plot 3: Error ratio (discrete)
    ax = axes2[1, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_disc = E12_disc / E23_disc
    ax.plot(common_times, smooth(ratio_disc), color='k', linewidth=1.2)
    ax.axhline(4, ls='--', color='gray', label='Ratio=4 (2nd order)')
    ax.axhline(8, ls=':', color='gray', label='Ratio=8 (3rd order)')
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'$E_{12} / E_{23}$')
    ax.set_title(r'(c) Error Ratio')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 12)

    # Plot 4: Convergence order (discrete)
    ax = axes2[1, 1]
    ax.plot(common_times[valid], p_disc[valid], color='k', alpha=0.3, lw=0.6,
            label='instantaneous')
    ax.plot(common_times[valid], p_disc_avg[valid], color='k', lw=2.5,
            label='running average')
    ax.axhline(2, ls='--', color='gray', label='2nd order')
    ax.axhline(3, ls=':', color='gray', label='3rd order')
    ax.axhline(5, ls='-.', color='gray', label='5th order')
    ax.set_xlim(0, t_max)
    ax.set_ylim(-1, 8)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'Convergence order $p(t)$')
    ax.set_title(r'(d) Convergence Order $p(t)$')
    ax.legend(fontsize=8, loc='upper right')

    fig2.suptitle(r'Discrete $L_1$ Norm: $E = (1/N) \sum |\rho_1 - \rho_2|$', fontsize=12, y=1.02)
    fig2.tight_layout()

    # =========================================================
    # Figure 3: Paper method (compare vs highest resolution)
    # =========================================================
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Central density evolution
    ax = axes3[0, 0]
    for (label, d), color in zip(data.items(), COLORS):
        rho_central = [rho[3] for rho in d['rho']]
        mask = d['t'] <= t_max
        ax.plot(d['t'][mask], rho_central[:np.sum(mask)],
                label=label, color=color, linewidth=0.8)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'$\rho_c$ (central density)')
    ax.set_title(r'(a) Central Density Evolution')
    ax.legend(fontsize=8)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Plot 2: Paper method L1 errors
    ax = axes3[0, 1]
    ax.semilogy(common_times, smooth(E1_paper), label=rf'$E_1$ ({low_res} vs {high_res})',
                color=COLORS[0], linewidth=1.2)
    ax.semilogy(common_times, smooth(E2_paper), label=rf'$E_2$ ({med_res} vs {high_res})',
                color=COLORS[1], linewidth=1.2)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'$L_1$ error')
    ax.set_title(rf'(b) Error vs Highest Resolution ($r \leq {R_star:.1f}$)')
    ax.legend(fontsize=8)

    # Plot 3: Error ratio (paper method)
    ax = axes3[1, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_paper = E1_paper / E2_paper
    ax.plot(common_times, smooth(ratio_paper), color='k', linewidth=1.2)
    ax.axhline(2, ls='--', color='gray', label='Ratio=2 (1st order)')
    ax.axhline(4, ls=':', color='gray', label='Ratio=4 (2nd order)')
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'$E_1 / E_2$')
    ax.set_title(r'(c) Error Ratio')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 8)

    # Plot 4: Convergence order (paper method)
    ax = axes3[1, 1]
    ax.plot(common_times[valid], p_paper[valid], color='k', alpha=0.3, lw=0.6,
            label='instantaneous')
    ax.plot(common_times[valid], p_paper_avg[valid], color='k', lw=2.5,
            label='running average')
    ax.axhline(1, ls='--', color='gray', label='1st order')
    ax.axhline(2, ls=':', color='gray', label='2nd order')
    ax.axhline(3, ls='-.', color='gray', label='3rd order')
    ax.set_xlim(0, t_max)
    ax.set_ylim(-1, 6)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'Convergence order $p(t)$')
    ax.set_title(r'(d) Convergence Order $p(t)$')
    ax.legend(fontsize=8, loc='upper right')

    fig3.suptitle(rf'Paper Method: $\Delta f = (1/N) \sum |f_i - \bar{{f}}_i|$ where $\bar{{f}}$ = {high_res}', fontsize=12, y=1.02)
    fig3.tight_layout()

    # =========================================================
    # Figure 4: Convergence plot log(Δρ) vs log(N)
    # Load reference resolution N=1600 for this plot
    # =========================================================
    print(f"\nLoading reference resolution {ref_res_label} for convergence plot...")
    t_ref, r_ref, rho_ref_list = load_snapshots(ref_res_path)

    # Find common times with reference resolution
    def find_matching_times(t_target, t_ref, rel_tol=0.05):
        """Find indices in t_ref that match times in t_target."""
        idx_target = []
        idx_ref = []
        for i, t in enumerate(t_target):
            j = np.argmin(np.abs(t_ref - t))
            if np.abs(t_ref[j] - t) / (t + 1e-10) < rel_tol:
                idx_target.append(i)
                idx_ref.append(j)
        return np.array(idx_target), np.array(idx_ref)

    # Match common_times with reference times
    idx_common, idx_ref = find_matching_times(common_times, t_ref)
    print(f"  Found {len(idx_common)} matching times with reference resolution")

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
    ax4_left, ax4_right = axes4

    # Resolutions and their log values (now includes all 4)
    N_values = np.array([N_low, N_med, N_high, N_ref])
    log_N = np.log10(N_values)

    # Select representative times for the plot
    target_times = [1000, 1500, 2000]
    markers = ['o', 's', '^']

    for target_t, marker, color in zip(target_times, markers, COLORS):
        # Find closest time index in common_times
        idx_t = np.argmin(np.abs(common_times[idx_common] - target_t))
        actual_t = common_times[idx_common][idx_t]

        # Get the actual index in the original arrays
        common_idx = idx_common[idx_t]
        ref_idx = idx_ref[idx_t]

        # Get indices for each resolution at this time
        i1, i2, i3 = idx1[common_idx], idx2[common_idx], idx3[common_idx]

        # Get density profiles at this time
        rho1 = rho1_list[i1]
        rho2 = rho2_list[i2]
        rho3 = rho3_list[i3]
        rho_ref = rho_ref_list[ref_idx]

        # Compute errors vs reference resolution (N=1600)
        e1 = compute_L1_error_discrete(r1, rho1, r_ref, rho_ref, r_max=R_star)
        e2 = compute_L1_error_discrete(r2, rho2, r_ref, rho_ref, r_max=R_star)
        e3 = compute_L1_error_discrete(r3, rho3, r_ref, rho_ref, r_max=R_star)

        # Use all three resolutions for fitting
        errors = np.array([e1, e2, e3])
        log_errors = np.log10(errors)
        log_N_fit = log_N[:3]

        # Linear fit to get convergence order
        slope, intercept = np.polyfit(log_N_fit, log_errors, 1)
        order = -slope  # Negative because error decreases with N

        # Plot data points
        ax4_left.scatter(log_N_fit, log_errors, s=80, marker=marker, color=color,
                   label=f't={actual_t:.0f}, order = {order:.2f}', zorder=5)

        # Plot fit line (extended to N_ref)
        log_N_line = np.linspace(log_N[0] - 0.1, log_N[-1] + 0.1, 50)
        log_error_line = slope * log_N_line + intercept
        ax4_left.plot(log_N_line, log_error_line, '-', color=color, alpha=0.5, lw=1.5)

    ax4_left.set_xlabel(r'$\log_{10}(N)$', fontsize=12)
    ax4_left.set_ylabel(r'$\log_{10}(\Delta \rho)$', fontsize=12)
    ax4_left.set_title(rf'(a) Convergence: $\Delta\rho$ vs Resolution (ref: {ref_res_label})', fontsize=11)
    ax4_left.legend(fontsize=8, loc='upper right')
    ax4_left.grid(True, alpha=0.3)

    # =========================================================
    # Right subplot: Initial vs Final density profiles
    # =========================================================
    print("\nPlotting initial and final density profiles...")

    # Get initial profiles (first snapshot, index 0)
    rho_initial_1 = rho1_list[0]
    rho_initial_2 = rho2_list[0]
    rho_initial_3 = rho3_list[0]
    rho_initial_ref = rho_ref_list[0]
    initial_time = common_times[0]

    # Get final profiles (last snapshot up to t_max)
    # Use the last index from the common times
    final_idx_1 = idx1[-1]
    final_idx_2 = idx2[-1]
    final_idx_3 = idx3[-1]

    # Find corresponding index in reference data
    final_time = common_times[-1]
    final_idx_ref = idx_ref[-1]

    # Get final density profiles
    rho_final_1 = rho1_list[final_idx_1]
    rho_final_2 = rho2_list[final_idx_2]
    rho_final_3 = rho3_list[final_idx_3]
    rho_final_ref = rho_ref_list[final_idx_ref]

    # Plot initial profiles (dashed)
    ax4_right.semilogy(r1, rho_initial_1, label=f'{low_res} (initial)', color=COLORS[0],
                       linewidth=1.5, linestyle='--', alpha=0.7)
    ax4_right.semilogy(r2, rho_initial_2, label=f'{med_res} (initial)', color=COLORS[1],
                       linewidth=1.5, linestyle='--', alpha=0.7)
    ax4_right.semilogy(r3, rho_initial_3, label=f'{high_res} (initial)', color=COLORS[2],
                       linewidth=1.5, linestyle='--', alpha=0.7)
    ax4_right.semilogy(r_ref, rho_initial_ref, label=f'{ref_res_label} (initial)', color='k',
                       linewidth=1.5, linestyle='--', alpha=0.7)

    # Plot final profiles (solid)
    ax4_right.semilogy(r1, rho_final_1, label=f'{low_res} (final)', color=COLORS[0], linewidth=1.8)
    ax4_right.semilogy(r2, rho_final_2, label=f'{med_res} (final)', color=COLORS[1], linewidth=1.8)
    ax4_right.semilogy(r3, rho_final_3, label=f'{high_res} (final)', color=COLORS[2], linewidth=1.8)
    ax4_right.semilogy(r_ref, rho_final_ref, label=f'{ref_res_label} (final)', color='k', linewidth=1.8)

    # Plot analytical TOV solution
    if has_analytical:
        ax4_right.semilogy(r_tov_analytic, rho_tov_analytic, label='Analytical TOV',
                          color='red', linewidth=2.5, linestyle='-.', alpha=0.9, zorder=10)

    # Mark stellar radius
    ax4_right.axvline(R_star, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
                      label=f'$R_*={R_star:.1f}$')

    ax4_right.set_xlabel(r'$r$ [M$_\odot$]', fontsize=12)
    ax4_right.set_ylabel(r'$\rho_0$ (rest-mass density)', fontsize=12)
    ax4_right.set_title(rf'(b) Initial vs Final Density Profiles ($t_0={initial_time:.0f}$, $t_f={final_time:.0f}$ M$_\odot$)', fontsize=11)
    ax4_right.legend(fontsize=7, loc='upper right', ncol=2)
    ax4_right.grid(True, alpha=0.3)
    ax4_right.set_xlim(0, R_star * 1.5)

    print(f"  Initial time: t = {initial_time:.1f} M_sun")
    print(f"  Final time:   t = {final_time:.1f} M_sun")

    # =========================================================
    # Compute and print differences between initial and final profiles
    # =========================================================
    print("\nComputing differences between initial and final profiles...")

    # Calculate differences for interior region (r <= R_star)
    def compute_profile_difference(r, rho_initial, rho_final, r_max):
        """Compute difference metrics between initial and final profiles."""
        mask = (r > 0) & (r <= r_max)
        r_int = r[mask]
        diff = np.abs(rho_final[mask] - rho_initial[mask])

        # Discrete L1 norm
        L1_diff = np.mean(diff)
        # Maximum difference
        max_diff = np.max(diff)
        # Central density difference (using index 3 to avoid r=0 issues)
        central_diff = rho_final[3] - rho_initial[3]

        return L1_diff, max_diff, central_diff

    diff_1_L1, diff_1_max, diff_1_central = compute_profile_difference(r1, rho_initial_1, rho_final_1, R_star)
    diff_2_L1, diff_2_max, diff_2_central = compute_profile_difference(r2, rho_initial_2, rho_final_2, R_star)
    diff_3_L1, diff_3_max, diff_3_central = compute_profile_difference(r3, rho_initial_3, rho_final_3, R_star)
    diff_ref_L1, diff_ref_max, diff_ref_central = compute_profile_difference(r_ref, rho_initial_ref, rho_final_ref, R_star)

    print(f"\n{'='*70}")
    print(f"PROFILE DIFFERENCES: Initial (t={initial_time:.1f}) vs Final (t={final_time:.1f})")
    print(f"{'='*70}")
    print(f"\n  Resolution: {low_res}")
    print(f"    L1 difference (interior):  {diff_1_L1:.6e}")
    print(f"    Max difference (interior): {diff_1_max:.6e}")
    print(f"    Central density change:    {diff_1_central:.6e}")

    print(f"\n  Resolution: {med_res}")
    print(f"    L1 difference (interior):  {diff_2_L1:.6e}")
    print(f"    Max difference (interior): {diff_2_max:.6e}")
    print(f"    Central density change:    {diff_2_central:.6e}")

    print(f"\n  Resolution: {high_res}")
    print(f"    L1 difference (interior):  {diff_3_L1:.6e}")
    print(f"    Max difference (interior): {diff_3_max:.6e}")
    print(f"    Central density change:    {diff_3_central:.6e}")

    print(f"\n  Resolution: {ref_res_label} (reference)")
    print(f"    L1 difference (interior):  {diff_ref_L1:.6e}")
    print(f"    Max difference (interior): {diff_ref_max:.6e}")
    print(f"    Central density change:    {diff_ref_central:.6e}")
    print(f"\n{'='*70}\n")

    # =========================================================
    # Compute differences vs analytical solution
    # =========================================================
    if has_analytical:
        print(f"\n{'='*70}")
        print(f"DIFFERENCES vs ANALYTICAL TOV SOLUTION (at t={final_time:.1f})")
        print(f"{'='*70}")

        # Compute error vs analytical for each resolution
        def compute_error_vs_analytical(r, rho, r_analytic, rho_analytic, r_max):
            """Compute error between numerical solution and analytical."""
            # Interpolate analytical to numerical grid
            rho_analytic_interp = np.interp(r, r_analytic, rho_analytic)
            mask = (r > 0) & (r <= r_max)
            diff = np.abs(rho[mask] - rho_analytic_interp[mask])
            L1_error = np.mean(diff)
            max_error = np.max(diff)
            central_error = rho[3] - rho_analytic_interp[3]
            return L1_error, max_error, central_error

        err_1_L1, err_1_max, err_1_central = compute_error_vs_analytical(
            r1, rho_final_1, r_tov_analytic, rho_tov_analytic, R_star)
        err_2_L1, err_2_max, err_2_central = compute_error_vs_analytical(
            r2, rho_final_2, r_tov_analytic, rho_tov_analytic, R_star)
        err_3_L1, err_3_max, err_3_central = compute_error_vs_analytical(
            r3, rho_final_3, r_tov_analytic, rho_tov_analytic, R_star)
        err_ref_L1, err_ref_max, err_ref_central = compute_error_vs_analytical(
            r_ref, rho_final_ref, r_tov_analytic, rho_tov_analytic, R_star)

        print(f"\n  Resolution: {low_res}")
        print(f"    L1 error (interior):       {err_1_L1:.6e}")
        print(f"    Max error (interior):      {err_1_max:.6e}")
        print(f"    Central density error:     {err_1_central:.6e}")

        print(f"\n  Resolution: {med_res}")
        print(f"    L1 error (interior):       {err_2_L1:.6e}")
        print(f"    Max error (interior):      {err_2_max:.6e}")
        print(f"    Central density error:     {err_2_central:.6e}")

        print(f"\n  Resolution: {high_res}")
        print(f"    L1 error (interior):       {err_3_L1:.6e}")
        print(f"    Max error (interior):      {err_3_max:.6e}")
        print(f"    Central density error:     {err_3_central:.6e}")

        print(f"\n  Resolution: {ref_res_label} (reference)")
        print(f"    L1 error (interior):       {err_ref_L1:.6e}")
        print(f"    Max error (interior):      {err_ref_max:.6e}")
        print(f"    Central density error:     {err_ref_central:.6e}")
        print(f"\n{'='*70}\n")

    plt.tight_layout()

    # Save figures
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    fig1.savefig(os.path.join(plots_dir, 'convergence_spherical_L1.png'),
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(plots_dir, 'convergence_discrete_L1.png'),
                 dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(plots_dir, 'convergence_paper_method.png'),
                 dpi=150, bbox_inches='tight')
    fig4.savefig(os.path.join(plots_dir, 'convergence_log_log.png'),
                 dpi=150, bbox_inches='tight')
    plt.show()

    # =========================================================
    # Print statistics
    # =========================================================
    print("\n" + "="*70)
    print("REST-MASS DENSITY CONVERGENCE ANALYSIS")
    print("(L1-norm method, like arXiv:1612.06251 Figure 16)")
    print("="*70)

    valid_mask = common_times > 50
    t_valid = common_times[valid_mask]

    print(f"\nTime range analyzed: t = [{t_valid[0]:.1f}, {t_valid[-1]:.1f}] M_sun")
    print(f"Stellar radius: R_star = {R_star:.3f}")
    print(f"Domain extent:  r_max  = {r1.max():.3f}")

    # Helper function to print stats for a given dataset
    def print_convergence_stats(p_data, E12_data, E23_data, label):
        p_valid_data = p_data[valid_mask]
        E12_valid_data = E12_data[valid_mask]
        E23_valid_data = E23_data[valid_mask]

        finite_mask = np.isfinite(p_valid_data)
        p_finite = p_valid_data[finite_mask]

        print(f"\n{'─'*70}")
        print(f"  {label}")
        print(f"{'─'*70}")

        print(f"\n  L1 Error Statistics:")
        print(f"    E({low_res}, {med_res}): mean = {np.mean(E12_valid_data):.3e}, max = {np.max(E12_valid_data):.3e}")
        print(f"    E({med_res}, {high_res}): mean = {np.mean(E23_valid_data):.3e}, max = {np.max(E23_valid_data):.3e}")
        ratio_mean = np.mean(E12_valid_data) / np.mean(E23_valid_data) if np.mean(E23_valid_data) > 0 else np.nan
        print(f"    Mean error ratio E12/E23: {ratio_mean:.2f}")
        print(f"      (expected ~4 for 2nd order, ~8 for 3rd order, ~32 for 5th order)")

        print(f"\n  Convergence Order p(t):")
        if len(p_finite) > 0:
            print(f"    Mean:   {np.mean(p_finite):.2f}")
            print(f"    Median: {np.median(p_finite):.2f}")
            print(f"    Std:    {np.std(p_finite):.2f}")

            p25, p50, p75 = np.percentile(p_finite, [25, 50, 75])
            print(f"    Percentiles: 25%={p25:.2f}, 50%={p50:.2f}, 75%={p75:.2f}")

        # Convergence by time interval
        intervals = [(50, 250), (250, 500), (500, 750), (750, 1000)]
        print(f"\n  Convergence by Time Interval:")
        for t_start, t_end in intervals:
            mask_interval = (t_valid >= t_start) & (t_valid < t_end) & np.isfinite(p_valid_data)
            if np.sum(mask_interval) > 0:
                p_interval = p_valid_data[mask_interval]
                print(f"    t=[{t_start:4d}, {t_end:4d}]: p = {np.mean(p_interval):.2f} +/- {np.std(p_interval):.2f}")

    # Print stats for all methods
    print_convergence_stats(p, E12, E23, "SPHERICAL L1 - FULL DOMAIN (r ≤ 20)")
    print_convergence_stats(p_int, E12_int, E23_int, f"SPHERICAL L1 - INTERIOR (r ≤ {R_star:.1f})")
    print_convergence_stats(p_disc, E12_disc, E23_disc, f"DISCRETE L1 - INTERIOR (r ≤ {R_star:.1f})")
    print_convergence_stats(p_paper, E1_paper, E2_paper, f"PAPER METHOD - vs {high_res} (r ≤ {R_star:.1f})")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
