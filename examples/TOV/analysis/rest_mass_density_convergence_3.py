#!/usr/bin/env python3
"""
Convergence analysis for TOV star evolution - Rest-Mass Density L1 Norm

Calculates convergence order from L1-norm of rest-mass density differences
between resolutions, similar to Figure 16 in arXiv:1612.06251.

Uses 3 resolutions: N1, N2, N3

Temporal alignment:
  - All resolutions aligned to common time grid (N1 as reference)
  - Nearest snapshot in time used for comparison (no interpolation)
  - Each resolution uses its closest available snapshot

Computes using two methods:

  1. Discrete L1 (consecutive pairs): E = (1/N) Σ |ρ₁ - ρ₂|
     - Two pairs: (N1, N2), (N2, N3)
     - One convergence order: p12 = log(E₁₂/E₂₃)/log(2)

  2. Paper method (vs highest resolution): Δf = (1/N) Σ |f - f̄|
     - Two comparisons vs N3: N1, N2
     - One convergence order: p12 = log(E₁/E₂)/log(2)
     - As in Appendix B of arXiv:1612.06251

All methods compute interior only (r ≤ R_star).
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import json
import argparse
from scipy.integrate import simpson
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d

# QNM analysis constants (from plot_qnm_analysis.py)
M_SUN_SECONDS = 4.926e-6
FREQ_CONVERSION = 1.0 / (M_SUN_SECONDS * 1e3)  # Convert from 1/M_sun to kHz

# Theoretical QNM frequencies in kHz (Font et al. 2002, Cowling approximation)
FREQUENCIES_COWLING_KHZ = {
    'F':  2.696, 'H1': 4.534, 'H2': 6.346, 'H3': 8.161,
}

# Frequencies for dynamic evolution (BSSN+hydro)
FREQUENCIES_DYNAMIC_KHZ = {
    'F':  1.450, 'H1': 3.958, 'H2': 5.935, 'H3': 7.812,
    'H4': 9.72,
}

# Select frequency dictionary based on data type (dyn_jax -> dynamic)
FREQUENCIES_KHZ = FREQUENCIES_DYNAMIC_KHZ

# QNM FFT settings (matching plot_qnm_analysis.py)
QNM_DELTA_T = 1.0   # Subsample to uniform dt=1 for FFT
QNM_T_START = 10.0   # Discard initial transient

# Resolutions - UPDATE THESE TO CHANGE RESOLUTIONS
N1 = 100
N2 = 200
N3 = 400
N4 = 800

# Base data directory
#DATA_DIR = '/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_rmax100.0_TEST_long_domain_long_time'

# Data paths - constructed from resolution values
FOLDERS = {
    f'N={N1}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_refact_rmax20.0_jax/tov_star_rhoc1p28em03_N{N1}_K100_G2_dyn_mp5',
    f'N={N2}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_refact_rmax20.0_jax/tov_star_rhoc1p28em03_N{N2}_K100_G2_dyn_mp5',
    f'N={N3}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_refact_rmax20.0_jax/tov_star_rhoc1p28em03_N{N3}_K100_G2_dyn_mp5',
}

# Resolution labels (keys to FOLDERS dictionary)
res1 = f'N={N1}'
res2 = f'N={N2}'
res3 = f'N={N3}'  # Highest resolution for comparison

COLORS = ['#1f77b4', "#ff7f0e", '#2ca02c', "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def load_metadata(folder_path):
    """
    Load simulation metadata from JSON file.

    Returns:
        Dictionary with metadata including stellar radius R_star
    """
    json_file = os.path.join(folder_path, 'tov_metadata_dyn_jax.json')

    try:
        with open(json_file, 'r') as f:
            metadata = json.load(f)
        return metadata
    except FileNotFoundError:
        print(f"Warning: Metadata file not found at {json_file}")
        return None


def load_snapshots(folder_path):
    """
    Load snapshots with density profiles and times.

    Returns:
        times: array of snapshot times
        r: radial grid
        rho_list: list of density profiles rho0(r) at each time
    """
    h5_file = os.path.join(folder_path, 'tov_snapshots_dyn_jax.h5')

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
    # Interpolate fine to coarse grid (cubic interpolation)
    interp_func = interp1d(r_fine, rho_fine, kind='cubic', bounds_error=False, fill_value='extrapolate')
    rho_fine_interp = interp_func(r_coarse)

    # Only consider where r > 0
    mask = r_coarse > 0
    if r_max is not None:
        mask = mask & (r_coarse <= r_max)

    diff = np.abs(rho_coarse[mask] - rho_fine_interp[mask])

    # Discrete L1 norm (mean absolute difference)
    return np.mean(diff)


def find_stellar_radius(r, rho, rho_atm=1e-16):
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
    """Compute running average with given window size.
    Uses reflect-padding at edges to avoid zero-padding artifacts."""
    if window <= 1 or len(x) == 0:
        return x
    window = min(window, len(x))
    if window <= 1:
        return x
    pad = window // 2
    # Reflect signal at boundaries to avoid edge artifacts
    x_padded = np.pad(x, pad, mode='reflect')
    kernel = np.ones(window) / window
    result = np.convolve(x_padded, kernel, mode='same')
    return result[pad:pad + len(x)]


def smooth(y, window=15, polyorder=3):
    """Apply Savitzky-Golay smoothing to data."""
    if len(y) < window:
        return y
    return savgol_filter(y, window, polyorder)


def compute_qnm_power_spectrum(t, signal_data, t_start=None, window='hann'):
    """
    Compute power spectrum for QNM analysis (matching plot_qnm_analysis.py).

    Args:
        t: time array
        signal_data: signal array (e.g., central density)
        t_start: discard data before this time (avoids initial transient)
        window: window function ('hann' or None)

    Returns:
        freq_khz: frequency array in kHz
        power: power spectrum
    """
    if t_start is not None:
        mask = t > t_start
        t_sel, sig_sel = t[mask], signal_data[mask]
    else:
        t_sel, sig_sel = t, signal_data

    # Use relative change (rho - rho_0) / rho_0 as in plot_qnm_analysis.py
    sig_rel = (sig_sel - sig_sel[0]) / sig_sel[0]
    sig_detrend = sig_rel - np.mean(sig_rel)

    n = len(sig_detrend)
    win = np.hanning(n) if window == 'hann' else np.ones(n)
    sig_windowed = sig_detrend * win

    dt = np.mean(np.diff(t_sel))
    freq = np.fft.rfftfreq(n, dt)
    fft_vals = np.fft.rfft(sig_windowed)
    freq_khz = freq * FREQ_CONVERSION
    power = np.abs(fft_vals)**2 / np.sum(win**2)

    return freq_khz, power


def find_qnm_peaks(freq_khz, power, min_freq=1.0, max_freq=14.0, max_peaks=10):
    """
    Find the most significant peaks in QNM frequency range.
    Matches find_all_peaks() from plot_qnm_analysis.py.

    Args:
        freq_khz: frequency array in kHz
        power: power spectrum
        min_freq: minimum frequency to search (kHz)
        max_freq: maximum frequency to search (kHz)
        max_peaks: maximum number of peaks to return

    Returns:
        peak_freqs: array of peak frequencies
        peak_powers: array of peak powers
    """
    mask = (freq_khz >= min_freq) & (freq_khz <= max_freq)
    freq_sel = freq_khz[mask]
    power_sel = power[mask]

    if len(power_sel) == 0:
        return np.array([]), np.array([])

    # Use log power for better peak detection
    log_power = np.log10(power_sel + 1e-50)
    noise_floor = np.median(log_power)

    # Find peaks - relaxed criteria to detect all visible peaks
    peaks, props = find_peaks(
        log_power,
        height=noise_floor + 0.3,  # ~2x above noise (relaxed)
        prominence=0.08,           # Low prominence threshold
        distance=2                 # Minimum separation
    )

    if len(peaks) == 0:
        return np.array([]), np.array([])

    # Sort by power (descending) and keep only top N peaks
    sorted_idx = np.argsort(power_sel[peaks])[::-1]
    peaks = peaks[sorted_idx[:max_peaks]]

    # Re-sort by frequency for display
    freq_order = np.argsort(freq_sel[peaks])
    peaks = peaks[freq_order]

    return freq_sel[peaks], power_sel[peaks]


def subsample_to_delta_t(t, signal_data, delta_t):
    """Efficiently subsample data to uniform time intervals using searchsorted.
    Matches plot_qnm_analysis.py."""
    target_times = np.arange(0, t.max() + delta_t, delta_t)
    indices = np.searchsorted(t, target_times)
    indices = np.clip(indices, 0, len(t) - 1)
    for i, (idx, target) in enumerate(zip(indices, target_times)):
        if idx > 0 and abs(t[idx - 1] - target) < abs(t[idx] - target):
            indices[i] = idx - 1
    _, unique_idx = np.unique(indices, return_index=True)
    indices = indices[np.sort(unique_idx)]
    return t[indices], signal_data[indices]


def load_timeseries_for_qnm(folder_path, delta_t=1.0):
    """Load rho_central from timeseries.npz and subsample, matching plot_qnm_analysis.py."""
    npz_file = os.path.join(folder_path, 'timeseries.npz')
    npz = np.load(npz_file)
    t = npz['times']
    signal_data = npz['rho_central']
    if delta_t is not None:
        t, signal_data = subsample_to_delta_t(t, signal_data, delta_t)
    return t, signal_data


def find_nearest_time_snapshots(t_target, t_source, rho_list_source):
    """
    Find snapshots at nearest times (no interpolation).

    For each target time, finds the snapshot at the closest source time.

    Args:
        t_target: target times (1D array)
        t_source: source times (1D array)
        rho_list_source: list of density profiles at source times

    Returns:
        List of density profiles at nearest times
    """
    rho_nearest = []

    for t in t_target:
        # Find index of nearest time
        idx = np.argmin(np.abs(t_source - t))
        rho_nearest.append(rho_list_source[idx])

    return rho_nearest


def extract_resolution_from_dirname(dirname):
    """Extract resolution number from directory name."""
    import re
    match = re.search(r'[Nn]r?[=_]?(\d+)', dirname)
    if match:
        return int(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Rest-mass density convergence analysis (requires exactly 3 resolutions)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python rest_mass_density_convergence_3.py                           # Use default folders
  python rest_mass_density_convergence_3.py --data-dirs D1 D2 D3      # Exactly 3 directories
  python rest_mass_density_convergence_3.py --tov-cache CACHE_PATH    # Custom TOV cache
'''
    )
    parser.add_argument('--data-dirs', nargs='+', default=None,
                        help='List of data directories (exactly 3 required). Default: use FOLDERS')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for plots. Default: script_dir/plots')
    parser.add_argument('--t-min', type=float, default=0.0,
                        help='Minimum time to analyze. Default: 0.0')
    parser.add_argument('--t-max', type=float, default=4000.0,
                        help='Maximum time to plot. Default: 6000.0')
    parser.add_argument('--tov-cache', default=None,
                        help='Path to TOV cache directory. Required if using --data-dirs.')
    args = parser.parse_args()

    t_min = 10#args.t_min
    t_max = args.t_max
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine output directory
    if args.output_dir:
        plots_dir = args.output_dir
    else:
        plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Determine data folders and TOV cache
    if args.data_dirs:
        # Validate exactly 3 directories
        if len(args.data_dirs) != 3:
            print(f"Error: This script requires exactly 3 directories, got {len(args.data_dirs)}")
            return

        if not args.tov_cache:
            print("Error: --tov-cache is required when using --data-dirs")
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

        # Sort by resolution to assign res1/res2/res3
        sorted_items = sorted(folders_dict.items(), key=lambda x: extract_resolution_from_dirname(x[0]) or 0)
        res1, res2, res3 = sorted_items[0][0], sorted_items[1][0], sorted_items[2][0]
        folders_dict = dict(sorted_items)
        tov_cache_dir = args.tov_cache

        # Extract resolution numbers for use in plots and print statements
        N1 = extract_resolution_from_dirname(res1) or 100
        N2 = extract_resolution_from_dirname(res2) or 200
        N3 = extract_resolution_from_dirname(res3) or 400
    else:
        # Use default FOLDERS (backward compatibility)
        folders_dict = FOLDERS
        # Extract res labels from FOLDERS keys
        sorted_items = sorted(folders_dict.items(), key=lambda x: extract_resolution_from_dirname(x[0]) or 0)
        res1, res2, res3 = sorted_items[0][0], sorted_items[1][0], sorted_items[2][0]
        tov_cache_dir = "/home/davidbamba/repositories/engrenage/examples/TOV/tov_iso_cache/TOVSOL_ISO_K=100.0_G=2.0_rho=1.280000e-03"

        # Extract resolution numbers for use in plots and print statements
        N1 = extract_resolution_from_dirname(res1) or 100
        N2 = extract_resolution_from_dirname(res2) or 200
        N3 = extract_resolution_from_dirname(res3) or 400

    # =========================================================
    # Load all data
    # =========================================================
    print("Loading snapshots...")

    data = {}
    for label, folder_path in folders_dict.items():
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
    # Align densities to common time grid (nearest snapshots)
    # =========================================================
    print("\nAligning densities to common time grid (nearest snapshots)...")

    # Use coarsest resolution as reference time grid
    t_ref = data[res1]['t']
    time_mask = (t_ref >= t_min) & (t_ref <= t_max)
    common_times = t_ref[time_mask]

    print(f"  Using {res1} as reference: {len(common_times)} time points in range t=[{t_min}, {t_max}]")
    print(f"  Finding nearest snapshots for {res2} and {res3}...")

    # Get source data
    t1, t2, t3 = data[res1]['t'], data[res2]['t'], data[res3]['t']
    r1 = data[res1]['r']
    r2 = data[res2]['r']
    r3 = data[res3]['r']

    rho1_list_full = data[res1]['rho']
    rho2_list_full = data[res2]['rho']
    rho3_list_full = data[res3]['rho']

    # Lowest resolution: just slice to t_max (no interpolation needed - it's the reference)
    rho1_list = [rho1_list_full[i] for i in range(len(common_times))]

    # Other resolutions: find nearest time snapshots (no interpolation)
    rho2_list = find_nearest_time_snapshots(common_times, t2, rho2_list_full)
    rho3_list = find_nearest_time_snapshots(common_times, t3, rho3_list_full)

    print(f"  Found nearest snapshots!")

    # =========================================================
    # Get stellar radius from metadata
    # =========================================================
    # Load metadata from highest resolution simulation
    metadata = load_metadata(folders_dict[res3])

    if metadata and 'tov_solution' in metadata and 'R' in metadata['tov_solution']:
        R_star = metadata['tov_solution']['R']
        R_star = float(R_star)*0.95
        print(f"\nStellar radius from metadata: R_star = {R_star:.6f}")
        print(f"  (M_star = {metadata['tov_solution']['M_star']:.6f} M_sun)")
        print(f"  (Compactness C = {metadata['tov_solution']['C']:.6f})")
    else:
        # Fallback: compute from density profile if metadata not available
        print("\nWarning: Metadata not found, computing R_star from density profile...")
        R_star = find_stellar_radius(r3, rho3_list[0])
        print(f"Stellar radius (computed): R_star = {R_star:.3f}")

    R_star = R_star
    print(f"Domain extent:  r_max  = {r1.max():.3f}")

    # =========================================================
    # Load analytical TOV solution
    # =========================================================
    print("\nLoading analytical TOV solution...")
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

    # Discrete L1 errors (interior only)
    E12_disc = []
    E23_disc = []

    # Paper method: compare against highest resolution (N3)
    E1_paper = []  # N1 vs N3
    E2_paper = []  # N2 vs N3

    for i in range(len(common_times)):
        rho1 = rho1_list[i]
        rho2 = rho2_list[i]
        rho3 = rho3_list[i]

        # Discrete L1 (interior only)
        e12_disc = compute_L1_error_discrete(r1, rho1, r2, rho2, r_max=R_star)
        e23_disc = compute_L1_error_discrete(r2, rho2, r3, rho3, r_max=R_star)
        E12_disc.append(e12_disc)
        E23_disc.append(e23_disc)

        # Paper method: discrete L1 vs highest resolution N3 (interior only)
        e1_paper = compute_L1_error_discrete(r1, rho1, r3, rho3, r_max=R_star)
        e2_paper = compute_L1_error_discrete(r2, rho2, r3, rho3, r_max=R_star)
        E1_paper.append(e1_paper)
        E2_paper.append(e2_paper)

    E12_disc = np.array(E12_disc)
    E23_disc = np.array(E23_disc)
    E1_paper = np.array(E1_paper)
    E2_paper = np.array(E2_paper)

    # =========================================================
    # Compute convergence order
    # =========================================================
    print("\nComputing convergence order...")

    with np.errstate(divide='ignore', invalid='ignore'):

        # Discrete L1 (interior only)
        p12_disc = np.log(E12_disc / E23_disc) / np.log(2.0)

        # Paper method: p = log(E1/E2) / log(N2/N1)
        p12_paper = np.log(E1_paper / E2_paper) / np.log(2.0)

    # Compute running averages
    p12_disc_avg = running_average(p12_disc, window=100)
    p12_paper_avg = running_average(p12_paper, window=100)

    valid = common_times > 50  # Skip initial transient

    # =========================================================
    # Figure 1: Discrete L1 norm analysis
    # =========================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))

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

    # Plot 2: Discrete L1 errors (interior)
    ax = axes1[0, 1]
    ax.semilogy(common_times, smooth(E12_disc), label=rf'$E_{{12}}$ ({res1} vs {res2})',
                color=COLORS[0], linewidth=1.2)
    ax.semilogy(common_times, smooth(E23_disc), label=rf'$E_{{23}}$ ({res2} vs {res3})',
                color=COLORS[1], linewidth=1.2)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'$L_1$ error')
    ax.set_title(rf'(b) Discrete $L_1$: $(1/N) \sum |\Delta\rho|$')
    ax.legend(fontsize=8)

    # Plot 3: Error ratio (discrete)
    ax = axes1[1, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio12_disc = E12_disc / E23_disc
    ax.plot(common_times, smooth(ratio12_disc), label='$E_{12}/E_{23}$',
            color=COLORS[0], linewidth=1.2)
    ax.axhline(4, ls='--', color='gray', label='Ratio=4 (2nd order)')
    ax.axhline(8, ls=':', color='gray', label='Ratio=8 (3rd order)')
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'Error Ratio')
    ax.set_title(r'(c) Error Ratios')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 12)

    # Plot 4: Convergence order (discrete)
    ax = axes1[1, 1]
    ax.plot(common_times[valid], p12_disc[valid], color=COLORS[0], alpha=0.3, lw=0.6)
    ax.plot(common_times[valid], p12_disc_avg[valid], color=COLORS[0], lw=2.0,
            label=f'p(N{N1},N{N2})')
    ax.axhline(2, ls='--', color='gray', label='2nd order')
    ax.axhline(3, ls=':', color='gray', label='3rd order')
    ax.axhline(5, ls='-.', color='gray', label='5th order')
    ax.set_xlim(0, t_max)
    ax.set_ylim(-1, 8)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'Convergence order $p(t)$')
    ax.set_title(r'(d) Convergence Order $p(t)$')
    ax.legend(fontsize=8, loc='upper right')

    fig1.suptitle(r'Discrete $L_1$ Norm: $E = (1/N) \sum |\rho_1 - \rho_2|$', fontsize=12, y=1.02)
    fig1.tight_layout()

    # =========================================================
    # Figure 2: Paper method (compare vs highest resolution)
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

    # Plot 2: Paper method L1 errors
    ax = axes2[0, 1]
    ax.semilogy(common_times, smooth(E1_paper), label=rf'$E_1$ ({res1} vs {res3})',
                color=COLORS[0], linewidth=1.2)
    ax.semilogy(common_times, smooth(E2_paper), label=rf'$E_2$ ({res2} vs {res3})',
                color=COLORS[1], linewidth=1.2)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'$L_1$ error')
    ax.set_title(rf'(b) Error vs Highest Resolution ($r \leq {R_star:.1f}$)')
    ax.legend(fontsize=8)

    # Plot 3: Error ratio (paper method)
    ax = axes2[1, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio12_paper = E1_paper / E2_paper
    ax.plot(common_times, smooth(ratio12_paper), label='$E_1/E_2$',
            color=COLORS[0], linewidth=1.2)
    ax.axhline(2, ls='--', color='gray', label='Ratio=2 (1st order)')
    ax.axhline(4, ls=':', color='gray', label='Ratio=4 (2nd order)')
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'Error Ratio')
    ax.set_title(r'(c) Error Ratios')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 8)

    # Plot 4: Convergence order (paper method)
    ax = axes2[1, 1]
    ax.plot(common_times[valid], p12_paper[valid], color=COLORS[0], alpha=0.3, lw=0.6)
    ax.plot(common_times[valid], p12_paper_avg[valid], color=COLORS[0], lw=2.0,
            label=f'p(N{N1},N{N2})')
    ax.axhline(1, ls='--', color='gray', label='1st order')
    ax.axhline(2, ls=':', color='gray', label='2nd order')
    ax.axhline(3, ls='-.', color='gray', label='3rd order')
    ax.set_xlim(0, t_max)
    ax.set_ylim(-1, 6)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'Convergence order $p(t)$')
    ax.set_title(r'(d) Convergence Order $p(t)$')
    ax.legend(fontsize=8, loc='upper right')

    fig2.suptitle(rf'Paper Method: $\Delta f = (1/N) \sum |f_i - \bar{{f}}_i|$ where $\bar{{f}}$ = {res3}', fontsize=12, y=1.02)
    fig2.tight_layout()

    # =========================================================
    # Figure 3: Convergence plot log(Δρ) vs log(N)
    # Using all 3 resolutions for log-log convergence plot
    # =========================================================
    print(f"\nPreparing Figure 3: convergence log-log plot...")

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    ax3_left, ax3_right = axes3

    # Resolutions and their log values (all 3)
    N_values = np.array([N1, N2, N3])
    log_N = np.log10(N_values)

    # Select representative times for the plot
    target_times = [1000, 1500, 2000]
    markers = ['o', 's', '^']

    for target_t, marker, color in zip(target_times, markers, COLORS):
        # Find closest time index in common_times
        idx_t = np.argmin(np.abs(common_times - target_t))
        actual_t = common_times[idx_t]

        # Get density profiles at this time (all at nearest snapshots)
        rho1 = rho1_list[idx_t]
        rho2 = rho2_list[idx_t]
        rho3 = rho3_list[idx_t]  # Highest resolution as reference

        # Compute errors vs reference resolution (N3)
        e1 = compute_L1_error_discrete(r1, rho1, r3, rho3, r_max=R_star)
        e2 = compute_L1_error_discrete(r2, rho2, r3, rho3, r_max=R_star)

        # Use the two lower resolutions for fitting (vs N3)
        errors = np.array([e1, e2])
        log_errors = np.log10(errors)
        log_N_fit = log_N[:2]

        # Linear fit to get convergence order
        slope, intercept = np.polyfit(log_N_fit, log_errors, 1)
        order = -slope  # Negative because error decreases with N

        # Plot data points
        ax3_left.scatter(log_N_fit, log_errors, s=80, marker=marker, color=color,
                   label=f't={actual_t:.0f}, order = {order:.2f}', zorder=5)

        # Plot fit line
        log_N_line = np.linspace(log_N[0] - 0.1, log_N[1] + 0.1, 50)
        log_error_line = slope * log_N_line + intercept
        ax3_left.plot(log_N_line, log_error_line, '-', color=color, alpha=0.5, lw=1.5)

    ax3_left.set_xlabel(r'$\log_{10}(N)$', fontsize=12)
    ax3_left.set_ylabel(r'$\log_{10}(\Delta \rho)$', fontsize=12)
    ax3_left.set_title(rf'(a) Convergence: $\Delta\rho$ vs Resolution (ref: {res3})', fontsize=11)
    ax3_left.legend(fontsize=8, loc='upper right')
    ax3_left.grid(True, alpha=0.3)

    # =========================================================
    # Right subplot: Initial vs Final density profiles
    # =========================================================
    print("\nPlotting initial and final density profiles...")

    # Get initial profiles (first snapshot, index 0)
    rho_initial_1 = rho1_list[0]
    rho_initial_2 = rho2_list[0]
    rho_initial_3 = rho3_list[0]
    initial_time = common_times[0]

    # Get final profiles (last snapshot up to t_max)
    # All resolutions are now aligned to nearest snapshots
    final_time = common_times[-1]

    # Get final density profiles
    rho_final_1 = rho1_list[-1]
    rho_final_2 = rho2_list[-1]
    rho_final_3 = rho3_list[-1]

    # Plot initial profiles (dashed)
    ax3_right.semilogy(r1, rho_initial_1, label=f'{res1} (initial)', color=COLORS[0],
                       linewidth=1.5, linestyle='--', alpha=0.7)
    ax3_right.semilogy(r2, rho_initial_2, label=f'{res2} (initial)', color=COLORS[1],
                       linewidth=1.5, linestyle='--', alpha=0.7)
    ax3_right.semilogy(r3, rho_initial_3, label=f'{res3} (initial)', color=COLORS[2],
                       linewidth=1.5, linestyle='--', alpha=0.7)

    # Plot final profiles (solid)
    ax3_right.semilogy(r1, rho_final_1, label=f'{res1} (final)', color=COLORS[0], linewidth=1.8)
    ax3_right.semilogy(r2, rho_final_2, label=f'{res2} (final)', color=COLORS[1], linewidth=1.8)
    ax3_right.semilogy(r3, rho_final_3, label=f'{res3} (final)', color=COLORS[2], linewidth=1.8)

    # Plot analytical TOV solution
    if has_analytical:
        ax3_right.semilogy(r_tov_analytic, rho_tov_analytic, label='Analytical TOV',
                          color='red', linewidth=2.5, linestyle='-.', alpha=0.9, zorder=10)

    # Mark stellar radius
    ax3_right.axvline(R_star, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
                      label=f'$R_*={R_star:.1f}$')

    ax3_right.set_xlabel(r'$r$ [M$_\odot$]', fontsize=12)
    ax3_right.set_ylabel(r'$\rho_0$ (rest-mass density)', fontsize=12)
    ax3_right.set_title(rf'(b) Initial vs Final Density Profiles ($t_0={initial_time:.0f}$, $t_f={final_time:.0f}$ M$_\odot$)', fontsize=11)
    ax3_right.legend(fontsize=7, loc='upper right', ncol=2)
    ax3_right.grid(True, alpha=0.3)
    ax3_right.set_xlim(0, R_star * 1.5)

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

    print(f"\n{'='*70}")
    print(f"PROFILE DIFFERENCES: Initial (t={initial_time:.1f}) vs Final (t={final_time:.1f})")
    print(f"{'='*70}")
    print(f"\n  Resolution: {res1}")
    print(f"    L1 difference (interior):  {diff_1_L1:.6e}")
    print(f"    Max difference (interior): {diff_1_max:.6e}")
    print(f"    Central density change:    {diff_1_central:.6e}")

    print(f"\n  Resolution: {res2}")
    print(f"    L1 difference (interior):  {diff_2_L1:.6e}")
    print(f"    Max difference (interior): {diff_2_max:.6e}")
    print(f"    Central density change:    {diff_2_central:.6e}")

    print(f"\n  Resolution: {res3} (reference)")
    print(f"    L1 difference (interior):  {diff_3_L1:.6e}")
    print(f"    Max difference (interior): {diff_3_max:.6e}")
    print(f"    Central density change:    {diff_3_central:.6e}")
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

        print(f"\n  Resolution: {res1}")
        print(f"    L1 error (interior):       {err_1_L1:.6e}")
        print(f"    Max error (interior):      {err_1_max:.6e}")
        print(f"    Central density error:     {err_1_central:.6e}")

        print(f"\n  Resolution: {res2}")
        print(f"    L1 error (interior):       {err_2_L1:.6e}")
        print(f"    Max error (interior):      {err_2_max:.6e}")
        print(f"    Central density error:     {err_2_central:.6e}")

        print(f"\n  Resolution: {res3} (reference)")
        print(f"    L1 error (interior):       {err_3_L1:.6e}")
        print(f"    Max error (interior):      {err_3_max:.6e}")
        print(f"    Central density error:     {err_3_central:.6e}")
        print(f"\n{'='*70}\n")

    plt.tight_layout()

    # =========================================================
    # CENTRAL DENSITY CONVERGENCE ANALYSIS
    # =========================================================
    print("\n" + "="*70)
    print("CENTRAL DENSITY CONVERGENCE ANALYSIS")
    print("="*70)

    # Define a fixed physical position for central density measurement
    r_fixed = 0.007  # Fixed physical position for all resolutions

    print(f"  Using fixed physical position: r_fixed = {r_fixed}")
    print(f"  Grid positions r[3] for comparison:")
    print(f"    {res1}: r[3] = {r1[3]:.4f}")
    print(f"    {res2}: r[3] = {r2[3]:.4f}")
    print(f"    {res3}: r[3] = {r3[3]:.4f}")

    # Function to interpolate density profile to fixed physical position
    def get_rhoc_at_fixed_r(r_grid, rho_profile, r_fixed):
        """Interpolate density profile to fixed physical position using cubic interpolation."""
        interp_func = interp1d(r_grid, rho_profile, kind='cubic',
                               bounds_error=False, fill_value='extrapolate')
        return interp_func(r_fixed)

    # Extract central density at fixed position for each snapshot (native times)
    print(f"  Extracting ρ(r={r_fixed}) for each snapshot (spatial cubic interpolation)...")
    rhoc_1_native = np.array([get_rhoc_at_fixed_r(r1, rho, r_fixed) for rho in rho1_list_full])
    rhoc_2_native = np.array([get_rhoc_at_fixed_r(r2, rho, r_fixed) for rho in rho2_list_full])
    rhoc_3_native = np.array([get_rhoc_at_fixed_r(r3, rho, r_fixed) for rho in rho3_list_full])

    # Interpolate central density to common times (temporal cubic interpolation)
    def interpolate_rhoc_in_time(times_source, rhoc_source, times_target):
        """Interpolate central density to target times using cubic interpolation."""
        interp_func = interp1d(times_source, rhoc_source, kind='cubic',
                               bounds_error=False, fill_value='extrapolate')
        return interp_func(times_target)

    print(f"  Interpolating ρ_c to common time grid (temporal cubic interpolation)...")
    rhoc_1 = interpolate_rhoc_in_time(t1, rhoc_1_native, common_times)
    rhoc_2 = interpolate_rhoc_in_time(t2, rhoc_2_native, common_times)
    rhoc_3 = interpolate_rhoc_in_time(t3, rhoc_3_native, common_times)

    print(f"  Central density at r={r_fixed} interpolated for {len(common_times)} time points")
    print(f"  ρ(r={r_fixed}, t=0): {res1}={rhoc_1[0]:.6e}, {res2}={rhoc_2[0]:.6e}, {res3}={rhoc_3[0]:.6e}")

    # Compute central density errors (consecutive pairs)
    E12_rhoc = np.abs(rhoc_1 - rhoc_2)
    E23_rhoc = np.abs(rhoc_2 - rhoc_3)

    # Paper method: compare against highest resolution (N3)
    E1_rhoc_paper = np.abs(rhoc_1 - rhoc_3)
    E2_rhoc_paper = np.abs(rhoc_2 - rhoc_3)

    # Compute convergence orders for central density
    with np.errstate(divide='ignore', invalid='ignore'):
        # Consecutive pairs method
        p12_rhoc = np.log(E12_rhoc / E23_rhoc) / np.log(2.0)

        # Paper method
        p12_rhoc_paper = np.log(E1_rhoc_paper / E2_rhoc_paper) / np.log(2.0)

    # Running averages
    p12_rhoc_avg = running_average(p12_rhoc, window=100)
    p12_rhoc_paper_avg = running_average(p12_rhoc_paper, window=100)

    # =========================================================
    # Figure 4: Central density evolution and errors (consecutive)
    # =========================================================
    fig4, axes4 = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 4a: Central density evolution (all resolutions)
    ax = axes4[0, 0]
    ax.plot(common_times, rhoc_1, label=res1, color=COLORS[0], linewidth=1.0)
    ax.plot(common_times, rhoc_2, label=res2, color=COLORS[1], linewidth=1.0)
    ax.plot(common_times, rhoc_3, label=res3, color=COLORS[2], linewidth=1.0)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'$\rho_c$ (central density)')
    ax.set_title(r'(a) Central Density Evolution')
    ax.legend(fontsize=8)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # Plot 4b: Central density errors (consecutive pairs)
    ax = axes4[0, 1]
    ax.semilogy(common_times, smooth(E12_rhoc), label=rf'$|\rho_c^{{{res1}}} - \rho_c^{{{res2}}}|$',
                color=COLORS[0], linewidth=1.2)
    ax.semilogy(common_times, smooth(E23_rhoc), label=rf'$|\rho_c^{{{res2}}} - \rho_c^{{{res3}}}|$',
                color=COLORS[1], linewidth=1.2)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'$|\Delta\rho_c|$')
    ax.set_title(r'(b) Central Density Errors (consecutive pairs)')
    ax.legend(fontsize=8)

    # Plot 4c: Error ratios (consecutive)
    ax = axes4[1, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio12_rhoc = E12_rhoc / E23_rhoc
    ax.plot(common_times, smooth(ratio12_rhoc), label='$E_{12}/E_{23}$',
            color=COLORS[0], linewidth=1.2)
    ax.axhline(4, ls='--', color='gray', label='Ratio=4 (2nd order)')
    ax.axhline(8, ls=':', color='gray', label='Ratio=8 (3rd order)')
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'Error Ratio')
    ax.set_title(r'(c) Error Ratios')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 12)

    # Plot 4d: Convergence order (consecutive)
    ax = axes4[1, 1]
    ax.plot(common_times[valid], p12_rhoc[valid], color=COLORS[0], alpha=0.3, lw=0.6)
    ax.plot(common_times[valid], p12_rhoc_avg[valid], color=COLORS[0], lw=2.0,
            label=f'p(N{N1},N{N2})')
    ax.axhline(2, ls='--', color='gray', label='2nd order')
    ax.axhline(3, ls=':', color='gray', label='3rd order')
    ax.axhline(5, ls='-.', color='gray', label='5th order')
    ax.set_xlim(0, t_max)
    ax.set_ylim(-1, 8)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'Convergence order $p(t)$')
    ax.set_title(r'(d) Convergence Order $p(t)$')
    ax.legend(fontsize=7, loc='upper right')

    fig4.suptitle(r'Central Density Convergence: $|\rho_c^{(i)} - \rho_c^{(j)}|$ (consecutive pairs)', fontsize=12, y=1.02)
    fig4.tight_layout()

    # =========================================================
    # Figure 5: Central density - Paper method (vs highest resolution)
    # =========================================================
    fig5, axes5 = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 5a: Central density evolution
    ax = axes5[0, 0]
    ax.plot(common_times, rhoc_1, label=res1, color=COLORS[0], linewidth=1.0)
    ax.plot(common_times, rhoc_2, label=res2, color=COLORS[1], linewidth=1.0)
    ax.plot(common_times, rhoc_3, label=res3, color=COLORS[2], linewidth=1.0)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'$\rho_c$ (central density)')
    ax.set_title(r'(a) Central Density Evolution')
    ax.legend(fontsize=8)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # Plot 5b: Paper method errors (vs highest resolution)
    ax = axes5[0, 1]
    ax.semilogy(common_times, smooth(E1_rhoc_paper), label=rf'$|\rho_c^{{{res1}}} - \rho_c^{{{res3}}}|$',
                color=COLORS[0], linewidth=1.2)
    ax.semilogy(common_times, smooth(E2_rhoc_paper), label=rf'$|\rho_c^{{{res2}}} - \rho_c^{{{res3}}}|$',
                color=COLORS[1], linewidth=1.2)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'$|\Delta\rho_c|$')
    ax.set_title(rf'(b) Error vs Highest Resolution ({res3})')
    ax.legend(fontsize=8)

    # Plot 5c: Error ratios (paper method)
    ax = axes5[1, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio12_rhoc_paper = E1_rhoc_paper / E2_rhoc_paper
    ax.plot(common_times, smooth(ratio12_rhoc_paper), label='$E_1/E_2$',
            color=COLORS[0], linewidth=1.2)
    ax.axhline(2, ls='--', color='gray', label='Ratio=2 (1st order)')
    ax.axhline(4, ls=':', color='gray', label='Ratio=4 (2nd order)')
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'Error Ratio')
    ax.set_title(r'(c) Error Ratios')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 8)

    # Plot 5d: Convergence order (paper method)
    ax = axes5[1, 1]
    ax.plot(common_times[valid], p12_rhoc_paper[valid], color=COLORS[0], alpha=0.3, lw=0.6)
    ax.plot(common_times[valid], p12_rhoc_paper_avg[valid], color=COLORS[0], lw=2.0,
            label=f'p(N{N1},N{N2})')
    ax.axhline(1, ls='--', color='gray', label='1st order')
    ax.axhline(2, ls=':', color='gray', label='2nd order')
    ax.axhline(3, ls='-.', color='gray', label='3rd order')
    ax.set_xlim(0, t_max)
    ax.set_ylim(-1, 6)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'Convergence order $p(t)$')
    ax.set_title(r'(d) Convergence Order $p(t)$')
    ax.legend(fontsize=8, loc='upper right')

    fig5.suptitle(rf'Central Density - Paper Method: $|\rho_c - \bar{{\rho}}_c|$ where $\bar{{\rho}}_c$ = {res3}', fontsize=12, y=1.02)
    fig5.tight_layout()

    # =========================================================
    # Figure 6: Central density log-log convergence plot
    # =========================================================
    fig6, axes6 = plt.subplots(1, 2, figsize=(14, 6))
    ax6_left, ax6_right = axes6

    # Select representative times for the plot
    target_times_rhoc = [100, 500, 1000]
    markers_rhoc = ['o', 's', '^']

    for target_t, marker, color in zip(target_times_rhoc, markers_rhoc, COLORS):
        # Find closest time index
        idx_t = np.argmin(np.abs(common_times - target_t))
        actual_t = common_times[idx_t]

        # Get central density at this time
        rhoc_at_t = np.array([rhoc_1[idx_t], rhoc_2[idx_t]])
        rhoc_ref = rhoc_3[idx_t]  # Reference (highest resolution)

        # Compute errors vs reference
        errors_rhoc = np.abs(rhoc_at_t - rhoc_ref)
        log_errors_rhoc = np.log10(errors_rhoc + 1e-20)  # Avoid log(0)
        log_N_fit = log_N[:2]

        # Linear fit to get convergence order
        # Filter out any -inf values
        valid_fit = np.isfinite(log_errors_rhoc)
        if np.sum(valid_fit) >= 2:
            slope_rhoc, intercept_rhoc = np.polyfit(log_N_fit[valid_fit], log_errors_rhoc[valid_fit], 1)
            order_rhoc = -slope_rhoc

            # Plot data points
            ax6_left.scatter(log_N_fit[valid_fit], log_errors_rhoc[valid_fit], s=80, marker=marker, color=color,
                             label=f't={actual_t:.0f}, order = {order_rhoc:.2f}', zorder=5)

            # Plot fit line
            log_N_line = np.linspace(log_N[0] - 0.1, log_N[1] + 0.1, 50)
            log_error_line = slope_rhoc * log_N_line + intercept_rhoc
            ax6_left.plot(log_N_line, log_error_line, '-', color=color, alpha=0.5, lw=1.5)

    ax6_left.set_xlabel(r'$\log_{10}(N)$', fontsize=12)
    ax6_left.set_ylabel(r'$\log_{10}(|\Delta \rho_c|)$', fontsize=12)
    ax6_left.set_title(rf'(a) Central Density Convergence (ref: {res3})', fontsize=11)
    ax6_left.legend(fontsize=8, loc='upper right')
    ax6_left.grid(True, alpha=0.3)

    # Right subplot: Central density time evolution comparison
    ax6_right.plot(common_times, rhoc_1, label=res1, color=COLORS[0], linewidth=1.2)
    ax6_right.plot(common_times, rhoc_2, label=res2, color=COLORS[1], linewidth=1.2)
    ax6_right.plot(common_times, rhoc_3, label=res3, color=COLORS[2], linewidth=1.2)

    ax6_right.set_xlabel(r'$t$ [M$_\odot$]', fontsize=12)
    ax6_right.set_ylabel(r'$\rho_c$ (central density)', fontsize=12)
    ax6_right.set_title(r'(b) Central Density Evolution', fontsize=11)
    ax6_right.legend(fontsize=8, loc='upper right')
    ax6_right.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    ax6_right.grid(True, alpha=0.3)

    fig6.suptitle(r'Central Density Convergence Analysis', fontsize=12, y=1.02)
    fig6.tight_layout()

    # =========================================================
    # Figure 7: Central density relative error and oscillation analysis
    # =========================================================
    fig7, axes7 = plt.subplots(2, 2, figsize=(12, 10))

    # Compute relative errors vs initial value (each resolution vs its own initial)
    rel_err_1 = np.abs(rhoc_1 - rhoc_1[0]) / rhoc_1[0]
    rel_err_2 = np.abs(rhoc_2 - rhoc_2[0]) / rhoc_2[0]
    rel_err_3 = np.abs(rhoc_3 - rhoc_3[0]) / rhoc_3[0]

    # Plot 7a: Relative error vs initial
    ax = axes7[0, 0]
    ax.semilogy(common_times, smooth(rel_err_1), label=res1, color=COLORS[0], linewidth=1.2)
    ax.semilogy(common_times, smooth(rel_err_2), label=res2, color=COLORS[1], linewidth=1.2)
    ax.semilogy(common_times, smooth(rel_err_3), label=res3, color=COLORS[2], linewidth=1.2)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'$|\rho_c - \rho_c^{0}|/\rho_c^{0}$')
    ax.set_title(r'(a) Relative Error vs Initial $\rho_c$')
    ax.legend(fontsize=8)

    # Plot 7b: Central density deviation from initial
    ax = axes7[0, 1]
    delta_rhoc_1 = (rhoc_1 - rhoc_1[0]) / rhoc_1[0]
    delta_rhoc_2 = (rhoc_2 - rhoc_2[0]) / rhoc_2[0]
    delta_rhoc_3 = (rhoc_3 - rhoc_3[0]) / rhoc_3[0]
    ax.plot(common_times, delta_rhoc_1, label=res1, color=COLORS[0], linewidth=1.0)
    ax.plot(common_times, delta_rhoc_2, label=res2, color=COLORS[1], linewidth=1.0)
    ax.plot(common_times, delta_rhoc_3, label=res3, color=COLORS[2], linewidth=1.0)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'$(\rho_c - \rho_c^0)/\rho_c^0$')
    ax.set_title(r'(b) Relative Deviation from Initial')
    ax.legend(fontsize=8)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # Plot 7c: Oscillation amplitude (max - min in sliding window)
    ax = axes7[1, 0]
    window_size = min(50, len(common_times) // 10)
    if window_size > 2:
        from scipy.ndimage import maximum_filter1d, minimum_filter1d
        osc_amp_1 = maximum_filter1d(rhoc_1, window_size) - minimum_filter1d(rhoc_1, window_size)
        osc_amp_2 = maximum_filter1d(rhoc_2, window_size) - minimum_filter1d(rhoc_2, window_size)
        osc_amp_3 = maximum_filter1d(rhoc_3, window_size) - minimum_filter1d(rhoc_3, window_size)
        ax.semilogy(common_times, osc_amp_1, label=res1, color=COLORS[0], linewidth=1.0)
        ax.semilogy(common_times, osc_amp_2, label=res2, color=COLORS[1], linewidth=1.0)
        ax.semilogy(common_times, osc_amp_3, label=res3, color=COLORS[2], linewidth=1.0)
    ax.set_xlabel(r'$t$ [M$_\odot$]')
    ax.set_ylabel(r'$\Delta\rho_c$ (oscillation amplitude)')
    ax.set_title(rf'(c) Oscillation Amplitude (window={window_size})')
    ax.legend(fontsize=8)

    # Plot 7d: QNM Power spectrum of central density oscillations (in kHz)
    # Loads rho_central from timeseries.npz (high cadence), matching plot_qnm_analysis.py exactly
    ax = axes7[1, 1]

    # Load native high-cadence timeseries data for each resolution
    print(f"\n  Loading timeseries.npz for QNM analysis (delta_t={QNM_DELTA_T})...")
    qnm_data = {}
    for label, folder_path in folders_dict.items():
        t_qnm, rho_qnm = load_timeseries_for_qnm(folder_path, delta_t=QNM_DELTA_T)
        qnm_data[label] = (t_qnm, rho_qnm)
        print(f"    {label}: {len(t_qnm)} points, dt={np.mean(np.diff(t_qnm)):.2f}")

    # Compute power spectra using relative change and t_start cutoff
    # (exactly as in plot_qnm_analysis.py: analyze_and_plot -> compute_power_spectrum)
    t_qnm_1, rho_qnm_1 = qnm_data[res1]
    t_qnm_2, rho_qnm_2 = qnm_data[res2]
    t_qnm_3, rho_qnm_3 = qnm_data[res3]

    freq_khz_1, power_1 = compute_qnm_power_spectrum(t_qnm_1, rho_qnm_1, t_start=QNM_T_START)
    freq_khz_2, power_2 = compute_qnm_power_spectrum(t_qnm_2, rho_qnm_2, t_start=QNM_T_START)
    freq_khz_3, power_3 = compute_qnm_power_spectrum(t_qnm_3, rho_qnm_3, t_start=QNM_T_START)

    # Plot power spectra for all 3 resolutions
    ax.semilogy(freq_khz_1, power_1, label=res1, color=COLORS[0], linewidth=1.0, alpha=0.8)
    ax.semilogy(freq_khz_2, power_2, label=res2, color=COLORS[1], linewidth=1.0, alpha=0.8)
    ax.semilogy(freq_khz_3, power_3, label=res3, color=COLORS[2], linewidth=1.2)

    # Detect peaks on highest resolution spectrum
    peak_freqs, peak_powers = find_qnm_peaks(freq_khz_3, power_3)

    # Plot detected peaks (blue dots with frequency labels, like plot_qnm_analysis.py)
    for f, p in zip(peak_freqs, peak_powers):
        if f < 14:
            ax.plot(f, p, 'bo', markersize=6, markeredgecolor='darkblue',
                    markeredgewidth=1.2, markerfacecolor='dodgerblue', zorder=5)
            ax.annotate(f'{f:.3f}', xy=(f, p), xytext=(0, 8),
                        textcoords='offset points', ha='center', fontsize=7,
                        fontweight='bold', color='blue')

    # Plot theoretical QNM frequencies (dynamic mode)
    for mode, f_theo in FREQUENCIES_KHZ.items():
        if f_theo < 14:
            ax.axvline(f_theo, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.text(f_theo, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1e-2,
                    mode, ha='center', va='bottom', fontsize=9, color='gray', alpha=0.7)

    ax.set_xlabel(r'Frequency [kHz]')
    ax.set_ylabel(r'Power')
    ax.set_title(r'(d) QNM Power Spectrum (dynamic modes)')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, 12)
    ax.grid(True, linestyle=':', alpha=0.3)

    # Set y-limits based on valid power range (like plot_qnm_analysis.py)
    valid_freq = (freq_khz_3 > 0.5) & (freq_khz_3 < 14)
    if np.any(valid_freq):
        pv = power_3[valid_freq]
        ax.set_ylim(np.min(pv[pv > 0]) * 0.1, np.max(pv) * 10)

    # Print detected peaks vs theoretical table
    print(f"\n{'='*70}")
    print(f"QNM PEAK DETECTION (highest resolution: {res3})")
    print(f"{'='*70}")
    print(f"  t_start = {QNM_T_START:.1f} M_sun, delta_t = {QNM_DELTA_T}")
    print(f"\n  {'Mode':<8} {'Theo [kHz]':<12} {'Obtained [kHz]':<15} {'Error [%]':<12}")
    print(f"  {'-'*50}")
    for mode, f_theo in FREQUENCIES_KHZ.items():
        if len(peak_freqs) == 0:
            print(f"  {mode:<8} {f_theo:<12.3f} {'N/A':<15} {'N/A':<12}")
        else:
            distances = np.abs(peak_freqs - f_theo)
            closest_idx = np.argmin(distances)
            closest_freq = peak_freqs[closest_idx]
            tolerance = 0.15 * f_theo
            if distances[closest_idx] <= tolerance:
                yerror = 100.0 * (closest_freq - f_theo) / f_theo
                print(f"  {mode:<8} {f_theo:<12.3f} {closest_freq:<15.3f} {yerror:<12.2f}")
            else:
                print(f"  {mode:<8} {f_theo:<12.3f} {'N/A':<15} {'N/A':<12}")
    print(f"{'='*70}\n")

    fig7.suptitle(r'Central Density: Relative Errors and Oscillation Analysis', fontsize=12, y=1.02)
    fig7.tight_layout()

    # Save figures
    fig1_path = os.path.join(plots_dir, 'convergence_discrete_L1.png')
    fig2_path = os.path.join(plots_dir, 'convergence_paper_method.png')
    fig3_path = os.path.join(plots_dir, 'convergence_log_log.png')
    fig4_path = os.path.join(plots_dir, 'convergence_rhoc_consecutive.png')
    fig5_path = os.path.join(plots_dir, 'convergence_rhoc_paper_method.png')
    fig6_path = os.path.join(plots_dir, 'convergence_rhoc_log_log.png')
    fig7_path = os.path.join(plots_dir, 'convergence_rhoc_oscillations.png')

    fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
    fig4.savefig(fig4_path, dpi=150, bbox_inches='tight')
    fig5.savefig(fig5_path, dpi=150, bbox_inches='tight')
    fig6.savefig(fig6_path, dpi=150, bbox_inches='tight')
    fig7.savefig(fig7_path, dpi=150, bbox_inches='tight')

    print(f"\nSaved: {fig1_path}")
    print(f"Saved: {fig2_path}")
    print(f"Saved: {fig3_path}")
    print(f"Saved: {fig4_path}")
    print(f"Saved: {fig5_path}")
    print(f"Saved: {fig6_path}")
    print(f"Saved: {fig7_path}")

    plt.show()
    #plt.close('all')

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

    # Helper function to print stats for a given dataset with 3 resolutions
    def print_convergence_stats(p12_data, E12_data, E23_data, label):
        p12_valid_data = p12_data[valid_mask]
        E12_valid_data = E12_data[valid_mask]
        E23_valid_data = E23_data[valid_mask]

        finite_mask_12 = np.isfinite(p12_valid_data)
        p12_finite = p12_valid_data[finite_mask_12]

        print(f"\n{'─'*70}")
        print(f"  {label}")
        print(f"{'─'*70}")

        print(f"\n  L1 Error Statistics:")
        print(f"    E({res1}, {res2}): mean = {np.mean(E12_valid_data):.3e}, max = {np.max(E12_valid_data):.3e}")
        print(f"    E({res2}, {res3}): mean = {np.mean(E23_valid_data):.3e}, max = {np.max(E23_valid_data):.3e}")
        ratio12_23 = np.mean(E12_valid_data) / np.mean(E23_valid_data) if np.mean(E23_valid_data) > 0 else np.nan
        print(f"    Mean error ratio E12/E23: {ratio12_23:.2f}")
        print(f"      (expected ~4 for 2nd order, ~8 for 3rd order, ~32 for 5th order)")

        print(f"\n  Convergence Order p(t):")
        print(f"    p(N{N1},N{N2}): Mean = {np.mean(p12_finite):.2f}, Median = {np.median(p12_finite):.2f}, Std = {np.std(p12_finite):.2f}")

        # Convergence by time interval
        intervals = [(50, 250), (250, 500), (500, 750), (750, 1000)]
        print(f"\n  Convergence by Time Interval:")
        for t_start, t_end in intervals:
            mask_interval_12 = (t_valid >= t_start) & (t_valid < t_end) & np.isfinite(p12_valid_data)
            if np.sum(mask_interval_12) > 0:
                p12_interval = p12_valid_data[mask_interval_12]
                print(f"    t=[{t_start:4d}, {t_end:4d}]: p12 = {np.mean(p12_interval):.2f}±{np.std(p12_interval):.2f}")

    # Print stats for all methods
    print_convergence_stats(p12_disc, E12_disc, E23_disc,
                           f"DISCRETE L1 - INTERIOR (r ≤ {R_star:.1f})")
    print_convergence_stats(p12_paper, E1_paper, E2_paper,
                           f"PAPER METHOD - vs {res3} (r ≤ {R_star:.1f})")

    # Central density statistics
    print_convergence_stats(p12_rhoc, E12_rhoc, E23_rhoc,
                           "CENTRAL DENSITY - CONSECUTIVE PAIRS")
    print_convergence_stats(p12_rhoc_paper, E1_rhoc_paper, E2_rhoc_paper,
                           f"CENTRAL DENSITY - PAPER METHOD vs {res3}")

    # Print central density summary
    print(f"\n{'─'*70}")
    print(f"  CENTRAL DENSITY SUMMARY")
    print(f"{'─'*70}")
    print(f"\n  Initial values:")
    print(f"    {res1}: ρ_c(0) = {rhoc_1[0]:.6e}")
    print(f"    {res2}: ρ_c(0) = {rhoc_2[0]:.6e}")
    print(f"    {res3}: ρ_c(0) = {rhoc_3[0]:.6e}")
    print(f"\n  Final values (t={common_times[-1]:.1f}):")
    print(f"    {res1}: ρ_c(t_f) = {rhoc_1[-1]:.6e}, Δρ_c/ρ_c = {(rhoc_1[-1]-rhoc_1[0])/rhoc_1[0]:.3e}")
    print(f"    {res2}: ρ_c(t_f) = {rhoc_2[-1]:.6e}, Δρ_c/ρ_c = {(rhoc_2[-1]-rhoc_2[0])/rhoc_2[0]:.3e}")
    print(f"    {res3}: ρ_c(t_f) = {rhoc_3[-1]:.6e}, Δρ_c/ρ_c = {(rhoc_3[-1]-rhoc_3[0])/rhoc_3[0]:.3e}")
    print(f"\n  Oscillation amplitude (max - min):")
    print(f"    {res1}: {np.max(rhoc_1) - np.min(rhoc_1):.6e}")
    print(f"    {res2}: {np.max(rhoc_2) - np.min(rhoc_2):.6e}")
    print(f"    {res3}: {np.max(rhoc_3) - np.min(rhoc_3):.6e}")

    print("\n" + "="*70 + "\n")

    # =========================================================
    # FINAL SUMMARY: Convergence orders for both methods
    # =========================================================
    print("\n" + "="*70)
    print("SUMMARY: CONVERGENCE ORDERS")
    print("="*70)

    # Compute mean convergence orders (excluding initial transient and NaN/Inf)
    def compute_mean_order(p_array, valid_mask):
        p_valid = p_array[valid_mask]
        finite_mask = np.isfinite(p_valid)
        if np.sum(finite_mask) > 0:
            return np.mean(p_valid[finite_mask]), np.std(p_valid[finite_mask])
        return np.nan, np.nan

    # Profile (L1 norm) convergence orders
    p12_disc_mean, p12_disc_std = compute_mean_order(p12_disc, valid_mask)
    p12_paper_mean, p12_paper_std = compute_mean_order(p12_paper, valid_mask)

    # Central density convergence orders
    p12_rhoc_mean, p12_rhoc_std = compute_mean_order(p12_rhoc, valid_mask)
    p12_rhoc_paper_mean, p12_rhoc_paper_std = compute_mean_order(p12_rhoc_paper, valid_mask)

    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                    DENSITY PROFILE (L1 norm)                        │
├─────────────────────────────────────────────────────────────────────┤
│  Method                        │  p(N{N1},N{N2})                    │
├────────────────────────────────┼─────────────────────────────────────┤
│  Richardson (consecutive)      │  {p12_disc_mean:5.2f} ± {p12_disc_std:4.2f}                    │
│  Paper method (vs {res3})     │  {p12_paper_mean:5.2f} ± {p12_paper_std:4.2f}                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    CENTRAL DENSITY (ρ_c)                            │
├─────────────────────────────────────────────────────────────────────┤
│  Method                        │  p(N{N1},N{N2})                    │
├────────────────────────────────┼─────────────────────────────────────┤
│  Richardson (consecutive)      │  {p12_rhoc_mean:5.2f} ± {p12_rhoc_std:4.2f}                    │
│  Paper method (vs {res3})     │  {p12_rhoc_paper_mean:5.2f} ± {p12_rhoc_paper_std:4.2f}                    │
└─────────────────────────────────────────────────────────────────────┘

Note: Expected order for 2nd order scheme: p ≈ 2
      Expected order for 5th order scheme (MP5): p ≈ 5
      Time range: t = [{t_valid[0]:.1f}, {t_valid[-1]:.1f}] M_sun
""")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
