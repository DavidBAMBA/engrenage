#!/usr/bin/env python3
"""
Quasi-Normal Mode (QNM) Analysis for TOV Star Evolution - Version 3

Detects ACTUAL peaks in the spectrum, then compares with theoretical values.
Does NOT force peaks to be near theoretical frequencies.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import h5py
import os
import argparse

# Unit conversions
M_SUN_SECONDS = 4.926e-6
FREQ_CONVERSION = 1.0 / (M_SUN_SECONDS * 1e3)

# Scaling factor: Valencia uses α in characteristic speeds, Font et al. does not
# Measured ratio f_sim/f_Font ≈ 0.889 → scale by 1/0.889 to compare
VALENCIA_TO_FONT_SCALE = 1.0 / 0.889  # ≈ 1.125

# Theoretical frequencies (Font et al. 2002)
FREQUENCIES_COWLING_KHZ = {
    'F':  2.696, 'H1': 4.534, 'H2': 6.346, 'H3': 8.161,
    'H4': 9.971, 'H5': 11.806, 'H6': 13.605,
}

def subsample_to_delta_t(t, rho_c, delta_t):
    """Efficiently subsample data to integer time intervals using searchsorted."""
    target_times = np.arange(0, t.max() + delta_t, delta_t)
    # Use searchsorted for O(log n) lookup per target
    indices = np.searchsorted(t, target_times)
    # Clamp to valid range
    indices = np.clip(indices, 0, len(t) - 1)
    # For each index, check if previous index is closer
    for i, (idx, target) in enumerate(zip(indices, target_times)):
        if idx > 0 and abs(t[idx - 1] - target) < abs(t[idx] - target):
            indices[i] = idx - 1
    # Remove duplicates while preserving order
    _, unique_idx = np.unique(indices, return_index=True)
    indices = indices[np.sort(unique_idx)]
    return t[indices], rho_c[indices]


def load_timeseries_npz(npz_file, delta_t=None):
    """Load timeseries data, optionally subsampling to integer time intervals.

    Args:
        npz_file: Path to the npz file
        delta_t: If specified, subsample to this time interval (e.g., delta_t=1 for t=1,2,3,...)
    """
    npz = np.load(npz_file)
    t = npz['times']
    rho_c = npz['rho_central']

    if delta_t is not None:
        t, rho_c = subsample_to_delta_t(t, rho_c, delta_t)
        print(f"  Subsampled to delta_t={delta_t}: {len(t)} points (t: {t[0]:.1f} to {t[-1]:.1f})")

    return {'t': t, 'rho_c': rho_c}

def load_evolution_data(h5_file, delta_t=None):
    """Load evolution data from HDF5 file, optionally subsampling."""
    with h5py.File(h5_file, 'r') as f:
        t = f['time'][:]
        rho_c = f['rho_central'][:]

    if delta_t is not None:
        t, rho_c = subsample_to_delta_t(t, rho_c, delta_t)
        print(f"  Subsampled to delta_t={delta_t}: {len(t)} points (t: {t[0]:.1f} to {t[-1]:.1f})")

    return {'t': t, 'rho_c': rho_c}

def compute_power_spectrum(t, signal_data, t_start=None, window='hann'):
    if t_start is not None:
        mask = t > t_start
        t_sel, sig_sel = t[mask], signal_data[mask]
    else:
        t_sel, sig_sel = t, signal_data
    
    sig_detrend = sig_sel - np.mean(sig_sel)
    n = len(sig_detrend)
    win = np.hanning(n) if window == 'hann' else np.ones(n)
    sig_windowed = sig_detrend * win
    
    dt = np.mean(np.diff(t_sel))
    freq = np.fft.rfftfreq(n, dt)
    fft_vals = np.fft.rfft(sig_windowed)
    freq_khz = freq * FREQ_CONVERSION
    power = np.abs(fft_vals)**2 / np.sum(win**2)
    
    return freq_khz, power

def find_all_peaks(freq_khz, power, min_freq=1.5, max_freq=14.0, max_peaks=8):
    """Find the most significant peaks."""
    mask = (freq_khz >= min_freq) & (freq_khz <= max_freq)
    freq_sel = freq_khz[mask]
    power_sel = power[mask]
    
    if len(power_sel) == 0:
        return np.array([]), np.array([])
    
    # Use log power for better peak detection
    log_power = np.log10(power_sel + 1e-50)
    noise_floor = np.median(log_power)
    
    # Find peaks - balanced criteria
    peaks, props = signal.find_peaks(
        log_power,
        height=noise_floor + 2.0,  # ~100x above noise
        prominence=0.7,            # Moderate prominence
        distance=8                 # ~1 kHz minimum separation
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

def plot_qnm_v3(t, rho_c, freq_khz, power, peak_freqs, peak_powers,
                theoretical_freqs, output_path=None, title=None, scale_factor=1.0):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')

    # Left: density evolution
    ax1 = axes[0]
    rho_c_0 = rho_c[0]
    delta_rho_rel = (rho_c - rho_c_0) / rho_c_0
    ax1.plot(t, delta_rho_rel, color='darkred', linewidth=0.8)
    ax1.set_xlabel(r'$t$ [M$_\odot$]', fontsize=12)
    ax1.set_ylabel(r'$(\rho_c - \rho_{c,0})/\rho_{c,0}$', fontsize=12)
    ax1.set_title('Central Density Relative Change', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Inset
    ax_inset = inset_axes(ax1, width="40%", height="35%", loc='upper right')
    mask_zoom = t <= min(1000, t[-1]/10)
    if np.sum(mask_zoom) > 10:
        ax_inset.plot(t[mask_zoom], rho_c[mask_zoom]/rho_c_0, 'k-', linewidth=0.6)
        ax_inset.set_xlabel('Time', fontsize=8)
        ax_inset.set_ylabel(r'$\rho_c/\rho_{c,0}$', fontsize=8)
        ax_inset.tick_params(labelsize=7)

    # Right: spectrum
    ax2 = axes[1]

    # Scale the frequency axis by the factor
    freq_khz_scaled = freq_khz * scale_factor
    peak_freqs_scaled = peak_freqs * scale_factor

    ax2.semilogy(freq_khz_scaled, power, color='#8B0000', linewidth=1.2)

    # Theoretical lines (gray, dashed)
    for mode, f_theo in theoretical_freqs.items():
        if f_theo < 14:
            ax2.axvline(f_theo, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax2.text(f_theo, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 1e-2,
                    mode, ha='center', va='bottom', fontsize=9, color='gray', alpha=0.7)

    # Detected peaks (blue dots with SCALED frequency labels)
    for f_scaled, p in zip(peak_freqs_scaled, peak_powers):
        if f_scaled < 14:
            ax2.plot(f_scaled, p, 'bo', markersize=8, markeredgecolor='darkblue',
                    markeredgewidth=1.5, markerfacecolor='dodgerblue', zorder=5)
            ax2.annotate(f'{f_scaled:.2f}', xy=(f_scaled, p), xytext=(0, 8),
                        textcoords='offset points', ha='center', fontsize=8,
                        fontweight='bold', color='blue')

    ax2.set_xlabel('Frequency [kHz] (scaled)' if scale_factor != 1.0 else 'Frequency [kHz]', fontsize=12)
    ax2.set_ylabel('Power', fontsize=12)
    ax2.set_xlim(0, 14)
    scale_text = f' (×{scale_factor:.3f})' if scale_factor != 1.0 else ''
    ax2.set_title(f'Power Spectrum{scale_text}\n', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.3)
    
    # Y limits
    valid = (freq_khz > 0.5) & (freq_khz < 14)
    if np.any(valid):
        pv = power[valid]
        ax2.set_ylim(np.min(pv[pv > 0]) * 0.1, np.max(pv) * 10)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    return fig

def load_data_from_folder(folder_path, delta_t=None):
    """Load data from a specific folder.

    Args:
        folder_path: Path to the data folder
        delta_t: If specified, subsample to this time interval (e.g., delta_t=1 for t=1,2,3,...)
    """
    for f in os.listdir(folder_path):
        filepath = os.path.join(folder_path, f)
        if f == 'timeseries.npz':
            return load_timeseries_npz(filepath, delta_t=delta_t)
        elif f.endswith('.h5') and 'evolution' in f:
            return load_evolution_data(filepath, delta_t=delta_t)
    return None

def analyze_and_plot(data, folder_name, plot_dir):
    """Analyze data and generate plot for a single dataset."""
    t, rho_c = data['t'], data['rho_c']
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {folder_name}")
    print(f"{'='*70}")
    print(f"Time: {t[0]:.1f} to {t[-1]:.1f} M_sun ({t[-1]*M_SUN_SECONDS*1e3:.2f} ms)")
    print(f"Points: {len(t)}, dt = {np.mean(np.diff(t)):.4f}")

    delta_rho = (rho_c - rho_c[0]) / rho_c[0]
    freq_khz, power = compute_power_spectrum(t, delta_rho, t_start=100)
    peak_freqs, peak_powers = find_all_peaks(freq_khz, power)

    print(f"\nDETECTED PEAKS (with Valencia→Font scaling = {VALENCIA_TO_FONT_SCALE:.3f}):")
    theo = list(FREQUENCIES_COWLING_KHZ.values())
    theo_names = list(FREQUENCIES_COWLING_KHZ.keys())

    print(f"\n{'#':<4} {'Raw [kHz]':<12} {'Scaled [kHz]':<14} {'Mode':<8} {'Theo [kHz]':<12} {'Ratio':<10}")
    print("-"*70)
    for i, (f, p) in enumerate(sorted(zip(peak_freqs, peak_powers), key=lambda x: x[0])):
        f_scaled = f * VALENCIA_TO_FONT_SCALE
        closest_idx = np.argmin(np.abs(np.array(theo) - f_scaled))
        ratio = f_scaled / theo[closest_idx]
        print(f"{i+1:<4} {f:<12.3f} {f_scaled:<14.3f} {theo_names[closest_idx]:<8} {theo[closest_idx]:<12.3f} {ratio:<10.3f}")

    if len(peak_freqs) >= 2:
        scaled_freqs = np.array(sorted(peak_freqs)) * VALENCIA_TO_FONT_SCALE
        ratios = [f/theo[i] for i, f in enumerate(scaled_freqs[:min(4, len(theo))])]
        print(f"\nAfter scaling: Average ratio = {np.mean(ratios):.3f} (difference from Font: {(np.mean(ratios)-1)*100:.1f}%)")

    output_path = os.path.join(plot_dir, f'qnm_{folder_name}.png')
    plot_qnm_v3(t, rho_c, freq_khz, power, peak_freqs, peak_powers,
               FREQUENCIES_COWLING_KHZ, output_path, title=folder_name,
               scale_factor=VALENCIA_TO_FONT_SCALE)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='QNM Analysis for TOV Star Evolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python plot_qnm_analysis.py                          # Use default folder, all data points
  python plot_qnm_analysis.py --delta-t 1             # Subsample to t=0,1,2,3,...
  python plot_qnm_analysis.py --data-dir tov_evolution_data  # Use different data folder
  python plot_qnm_analysis.py --delta-t 1 --data-dir tov_evolution_data
'''
    )
    parser.add_argument('--delta-t', type=float, default=None,
                        help='Time interval for subsampling (e.g., 1 for t=0,1,2,...). Default: None (use all data)')
    parser.add_argument('--data-dir', type=str, default='tov_evolution_data4',
                        help='Data directory name (relative to script). Default: tov_evolution_data2')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, args.data_dir)
    plot_dir = os.path.join(script_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    # Find all data folders
    data_folders = []
    for item in sorted(os.listdir(data_dir)):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            data_folders.append((item, item_path))

    if not data_folders:
        print(f"No data folders found in {data_dir}")
        return

    print(f"Data directory: {args.data_dir}")
    print(f"Found {len(data_folders)} data folder(s):")
    for name, _ in data_folders:
        print(f"  - {name}")

    delta_t = args.delta_t
    if delta_t is not None:
        print(f"\nUsing delta_t = {delta_t} (subsampled data)")
    else:
        print(f"\nUsing all data points (no subsampling)")

    generated_plots = []
    for folder_name, folder_path in data_folders:
        data = load_data_from_folder(folder_path, delta_t=delta_t)
        if data is not None:
            output_path = analyze_and_plot(data, folder_name, plot_dir)
            generated_plots.append(output_path)
        else:
            print(f"\nNo data found in: {folder_name}")

    print(f"\n{'='*70}")
    print(f"SUMMARY: Generated {len(generated_plots)} plot(s)")
    print(f"{'='*70}")
    for p in generated_plots:
        print(f"  - {p}")

    plt.show()


if __name__ == "__main__":
    main()