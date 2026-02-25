#!/usr/bin/env python3
"""
Quasi-Normal Mode (QNM) Analysis for TOV Star Evolution

Detects actual peaks in the spectrum and compares with theoretical values.
Reports mode, theoretical frequency, obtained frequency, and percentage error.
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

# Mode selection: "cow" = Cowling approximation, "dyn" = dynamic (full BSSN+hydro)
MODE = "dyn"

# Analysis variable selection: 1 = rho_central (density), 2 = v_central (velocity)
ANALYSIS_VAR = 1

# Time to start analysis (in M_sun units) - data before this is discarded
T_START = 10.0

# =============================================================================
# FOLDERS TO ANALYZE - Specify the exact folders you want to analyze
# =============================================================================
FOLDERS_TO_ANALYZE = [
    '/home/davidbamba/repositories/engrenage/examples/TOV/data/tov_evolution_data_refact_rmax50.0_jax/tov_star_rhoc1p28em03_N500_K100_G2_dyn_mp5',
]
# Theoretical frequencies (Font et al. 2002)
FREQUENCIES_COWLING_KHZ = {
    'F':  2.696, 'H1': 4.534, 'H2': 6.346, 'H3': 8.161,
    'H4': 9.971, 'H5': 11.806, 'H6': 13.605,
}

# Frequencies for dynamic evolution (BSSN+hydro) - Present 3D code
FREQUENCIES_DYNAMIC_KHZ = {
    'F':  1.450, 'H1': 3.958, 'H2': 5.935, 'H3': 7.812,
    'h4': 9.72,
}

# Select frequency dictionary based on MODE
FREQUENCIES_KHZ = FREQUENCIES_COWLING_KHZ if MODE == "cow" else FREQUENCIES_DYNAMIC_KHZ

def subsample_to_delta_t(t, signal_data, delta_t):
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
    return t[indices], signal_data[indices]


def load_timeseries_npz(npz_file, delta_t=None, var=1):
    """Load timeseries data, optionally subsampling to integer time intervals.

    Args:
        npz_file: Path to the npz file
        delta_t: If specified, subsample to this time interval (e.g., delta_t=1 for t=1,2,3,...)
        var: 1 = rho_central, 2 = v_central
    """
    npz = np.load(npz_file)
    t = npz['times']

    if var == 2:
        if 'v_central' in npz.files:
            signal_data = npz['v_central']
            var_name = 'v_central'
        else:
            print("  Warning: v_central not found in npz, falling back to rho_central")
            signal_data = npz['rho_central']
            var_name = 'rho_central'
    else:
        signal_data = npz['rho_central']
        var_name = 'rho_central'

    if delta_t is not None:
        t, signal_data = subsample_to_delta_t(t, signal_data, delta_t)
        print(f"  Subsampled to delta_t={delta_t}: {len(t)} points (t: {t[0]:.1f} to {t[-1]:.1f})")

    return {'t': t, 'signal': signal_data, 'var_name': var_name}

def load_evolution_data(h5_file, delta_t=None, var=1):
    """Load evolution data from HDF5 file, optionally subsampling.

    Args:
        h5_file: Path to the HDF5 file
        delta_t: If specified, subsample to this time interval
        var: 1 = rho_central, 2 = v_central
    """
    with h5py.File(h5_file, 'r') as f:
        t = f['time'][:]
        if var == 2:
            if 'v_central' in f.keys():
                signal_data = f['v_central'][:]
                var_name = 'v_central'
            else:
                print("  Warning: v_central not found in h5, falling back to rho_central")
                signal_data = f['rho_central'][:]
                var_name = 'rho_central'
        else:
            signal_data = f['rho_central'][:]
            var_name = 'rho_central'

    if delta_t is not None:
        t, signal_data = subsample_to_delta_t(t, signal_data, delta_t)
        print(f"  Subsampled to delta_t={delta_t}: {len(t)} points (t: {t[0]:.1f} to {t[-1]:.1f})")

    return {'t': t, 'signal': signal_data, 'var_name': var_name}

def load_v_central_from_snapshots(snapshot_file, delta_t=None):
    """Load central velocity from snapshots HDF5 file.

    This allows using v_central even from old data that doesn't have it in timeseries.
    """
    NUM_GHOSTS = 3  # Ghost points at boundary

    times = []
    v_central = []

    with h5py.File(snapshot_file, 'r') as f:
        if 'snapshots' not in f:
            print(f"  Warning: No snapshots group in {snapshot_file}")
            return None

        snap_names = sorted(f['snapshots'].keys())
        print(f"  Loading v_central from {len(snap_names)} snapshots...")

        for snap_name in snap_names:
            snap = f['snapshots'][snap_name]
            t = snap.attrs['time']
            if 'primitives' in snap and 'vr' in snap['primitives']:
                vr = snap['primitives/vr'][:]
                times.append(t)
                v_central.append(vr[NUM_GHOSTS])  # Central point

    if len(times) == 0:
        return None

    t = np.array(times)
    signal_data = np.array(v_central)

    if delta_t is not None:
        t, signal_data = subsample_to_delta_t(t, signal_data, delta_t)
        print(f"  Subsampled to delta_t={delta_t}: {len(t)} points (t: {t[0]:.1f} to {t[-1]:.1f})")

    return {'t': t, 'signal': signal_data, 'var_name': 'v_central'}


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

def find_all_peaks(freq_khz, power, min_freq=1.0, max_freq=14.0, max_peaks=10):
    """Find the most significant peaks."""
    mask = (freq_khz >= min_freq) & (freq_khz <= max_freq)
    freq_sel = freq_khz[mask]
    power_sel = power[mask]

    if len(power_sel) == 0:
        return np.array([]), np.array([])

    # Use log power for better peak detection
    log_power = np.log10(power_sel + 1e-50)
    noise_floor = np.median(log_power)

    # Find peaks - very relaxed criteria to detect all visible peaks
    peaks, props = signal.find_peaks(
        log_power,
        height=noise_floor + 0.3,  # ~2x above noise (very relaxed)
        prominence=0.08,           # Very low prominence threshold
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

def plot_qnm_v3(t, signal_data, freq_khz, power, peak_freqs, peak_powers,
                theoretical_freqs, output_path=None, title=None, var_name='rho_central'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    # Convert time to milliseconds for main plot
    t_ms = t * M_SUN_SECONDS * 1e3

    # Left: signal evolution
    ax1 = axes[0]
    if var_name == 'v_central':
        # Velocity: plot directly or subtract mean
        signal_0 = np.mean(signal_data)
        delta_signal = signal_data - signal_0
        ax1.plot(t_ms, delta_signal, color='darkblue', linewidth=0.8)
        ax1.set_xlabel(r'$t$ [ms]', fontsize=12)
        ax1.set_ylabel(r'$v_c - \bar{v}_c$', fontsize=12)
        ax1.set_title('Central Radial Velocity', fontsize=14)
    else:
        # Density: relative change
        signal_0 = signal_data[0]
        delta_signal = (signal_data - signal_0) / signal_0
        ax1.plot(t_ms, delta_signal, color='darkred', linewidth=0.8)
        ax1.set_xlabel(r'$t$ [ms]', fontsize=12)
        ax1.set_ylabel(r'$(\rho_c - \rho_{c,0})/\rho_{c,0}$', fontsize=12)
        ax1.set_title('Central Density Relative Change', fontsize=14)

    ax1.grid(True, alpha=0.3)

    # Right: spectrum
    ax2 = axes[1]

    ax2.semilogy(freq_khz, power, color='#8B0000', linewidth=1.2)

    # Theoretical lines (gray, dashed)
    for mode, f_theo in theoretical_freqs.items():
        if f_theo < 14:
            ax2.axvline(f_theo, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax2.text(f_theo, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 1e-2,
                    mode, ha='center', va='bottom', fontsize=9, color='gray', alpha=0.7)

    # Detected peaks (blue dots with frequency labels)
    for f, p in zip(peak_freqs, peak_powers):
        if f < 14:
            ax2.plot(f, p, 'bo', markersize=8, markeredgecolor='darkblue',
                    markeredgewidth=1.5, markerfacecolor='dodgerblue', zorder=5)
            ax2.annotate(f'{f:.3f}', xy=(f, p), xytext=(0, 8),
                        textcoords='offset points', ha='center', fontsize=8,
                        fontweight='bold', color='blue')

    ax2.set_xlabel('Frequency [kHz]', fontsize=12)
    ax2.set_ylabel('Power', fontsize=12)
    ax2.set_xlim(0, 9)
    ax2.set_title('Power Spectrum\n', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.3)
    
    # Y limits
    valid = (freq_khz > 0.5) & (freq_khz < 14)
    if np.any(valid):
        pv = power[valid]
        ax2.set_ylim(np.min(pv[pv > 0]) * 0.1, np.max(pv) * 10)
    
    plt.subplots_adjust(top=0.88, bottom=0.15, left=0.08, right=0.9, wspace=0.25)
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    return fig

def load_data_from_folder(folder_path, delta_t=None, var=1):
    """Load data from a specific folder.

    Args:
        folder_path: Path to the data folder
        delta_t: If specified, subsample to this time interval (e.g., delta_t=1 for t=1,2,3,...)
        var: 1 = rho_central, 2 = v_central
    """
    files = os.listdir(folder_path)

    # For v_central, try snapshots first (works with old data)
    if var == 2:
        for f in files:
            if f.endswith('.h5') and 'snapshot' in f:
                filepath = os.path.join(folder_path, f)
                data = load_v_central_from_snapshots(filepath, delta_t=delta_t)
                if data is not None:
                    return data

    # Try standard files
    for f in files:
        filepath = os.path.join(folder_path, f)
        if f == 'timeseries.npz':
            data = load_timeseries_npz(filepath, delta_t=delta_t, var=var)
            # If we got the variable we wanted, return it
            if data['var_name'] == ('v_central' if var == 2 else 'rho_central'):
                return data
            elif var == 1:  # For rho_central, always return
                return data
        elif f.endswith('.h5') and 'evolution' in f:
            data = load_evolution_data(filepath, delta_t=delta_t, var=var)
            if data['var_name'] == ('v_central' if var == 2 else 'rho_central'):
                return data
            elif var == 1:
                return data

    return None

def analyze_and_plot(data, folder_name, plot_dir):
    """Analyze data and generate plot for a single dataset."""
    t = data['t']
    signal_data = data['signal']
    var_name = data.get('var_name', 'rho_central')

    mode_name = "Cowling" if MODE == "cow" else "Dynamic"
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {folder_name} (variable: {var_name}, mode: {mode_name})")
    print(f"{'='*70}")
    print(f"Time: {t[0]:.1f} to {t[-1]:.1f} M_sun ({t[-1]*M_SUN_SECONDS*1e3:.2f} ms)")
    print(f"Points: {len(t)}, dt = {np.mean(np.diff(t)):.4f}")

    # For velocity, use signal directly; for density, use relative change
    if var_name == 'v_central':
        # Velocity oscillates around 0, use directly
        delta_signal = signal_data - np.mean(signal_data)
    else:
        # Density: use relative change
        delta_signal = (signal_data - signal_data[0]) / signal_data[0]

    freq_khz, power = compute_power_spectrum(t, delta_signal, t_start=T_START)
    peak_freqs, peak_powers = find_all_peaks(freq_khz, power)

    print(f"\nDETECTED PEAKS:")

    # Match each theoretical mode with the closest detected peak (if any)
    print(f"\n{'Mode':<8} {'Theo [kHz]':<12} {'Obtained [kHz]':<15} {'yerror [%]':<12}")
    print("-"*50)

    for mode, f_theo in FREQUENCIES_KHZ.items():
        if len(peak_freqs) == 0:
            # No peaks detected
            print(f"{mode:<8} {f_theo:<12.3f} {'N/A':<15} {'N/A':<12}")
        else:
            # Find closest detected peak to this theoretical frequency
            distances = np.abs(peak_freqs - f_theo)
            closest_idx = np.argmin(distances)
            closest_freq = peak_freqs[closest_idx]

            # Only match if the peak is within a reasonable range (e.g., ±15% of theoretical)
            tolerance = 0.15 * f_theo
            if distances[closest_idx] <= tolerance:
                yerror = 100.0 * (closest_freq - f_theo) / f_theo
                print(f"{mode:<8} {f_theo:<12.3f} {closest_freq:<15.3f} {yerror:<12.2f}")
            else:
                print(f"{mode:<8} {f_theo:<12.3f} {'N/A':<15} {'N/A':<12}")

    output_path = os.path.join(plot_dir, f'qnm_{folder_name}_{var_name}.png')
    plot_qnm_v3(t, signal_data, freq_khz, power, peak_freqs, peak_powers,
               FREQUENCIES_KHZ, output_path, title=f"{folder_name} ({var_name})",
               var_name=var_name)

    # Return analysis results for comparison table
    return {
        'output_path': output_path,
        'peak_freqs': peak_freqs,
        'peak_powers': peak_powers,
        'folder_name': folder_name
    }


def extract_resolution_from_name(folder_name):
    """Extract resolution number from folder name (e.g., 'N100', 'Nr200', 'N=400')."""
    import re
    match = re.search(r'[Nn]r?[=_]?(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None


def create_comparison_table(results_list, plot_dir, var_name='rho_central'):
    """
    Create a comparison table for the first 3 peaks across different resolutions.

    Args:
        results_list: List of dictionaries with 'folder_name', 'peak_freqs', 'peak_powers'
        plot_dir: Directory to save the plot
        var_name: Variable name for filename

    Returns:
        Path to the saved plot
    """
    # Sort results by resolution
    sorted_results = []
    for result in results_list:
        res = extract_resolution_from_name(result['folder_name'])
        if res is not None:
            sorted_results.append((res, result))

    if len(sorted_results) < 2:
        print("\nSkipping comparison table (need at least 2 resolutions)")
        return None

    sorted_results.sort(key=lambda x: x[0])  # Sort by resolution number

    # Match top 3 peaks for each resolution to theoretical modes
    mode_list = list(FREQUENCIES_KHZ.keys())
    theo_freqs = list(FREQUENCIES_KHZ.values())

    # Build comparison data: dict[mode] -> list of (resolution, freq)
    comparison = {mode: {} for mode in mode_list}

    for res, result in sorted_results:
        peak_freqs = result['peak_freqs']
        peak_powers = result['peak_powers']

        if len(peak_freqs) == 0:
            continue

        # Get top 3 peaks by power
        if len(peak_freqs) > 3:
            # Sort by power to get top 3
            power_idx = np.argsort(peak_powers)[::-1][:3]
            top3_freqs = peak_freqs[power_idx]
        else:
            top3_freqs = peak_freqs

        # Match each peak to closest theoretical mode
        for peak_freq in top3_freqs:
            # Find closest theoretical frequency
            distances = [abs(peak_freq - f_theo) for f_theo in theo_freqs]
            min_idx = np.argmin(distances)
            closest_mode = mode_list[min_idx]
            closest_theo = theo_freqs[min_idx]

            # Only match if within ±15% tolerance
            tolerance = 0.15 * closest_theo
            if distances[min_idx] <= tolerance:
                comparison[closest_mode][res] = peak_freq

    # Filter modes that have at least one detection
    active_modes = [mode for mode in mode_list if len(comparison[mode]) > 0]

    if len(active_modes) == 0:
        print("\nNo matched modes found for comparison table")
        return None

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, len(active_modes) * 0.6 + 2))
    ax.axis('off')

    # Build table data
    resolutions = [res for res, _ in sorted_results]

    # Header
    header = ['Mode', 'Theo [kHz]'] + [f'Nr={res}' for res in resolutions] + ['yerror [%]']

    # Rows
    table_data = []
    for mode in active_modes:
        f_theo = FREQUENCIES_KHZ[mode]
        row = [mode, f'{f_theo:.3f}']

        # Add obtained frequencies for each resolution
        last_freq = None
        for res in resolutions:
            if res in comparison[mode]:
                freq = comparison[mode][res]
                row.append(f'{freq:.3f}')
                last_freq = freq
            else:
                row.append('N/A')

        # Calculate yerror from last available frequency
        if last_freq is not None:
            yerror = 100.0 * (last_freq - f_theo) / f_theo
            row.append(f'{yerror:.2f}')
        else:
            row.append('N/A')

        table_data.append(row)

    # Create table
    table = ax.table(cellText=table_data, colLabels=header,
                     cellLoc='center', loc='center',
                     colWidths=[0.08, 0.12] + [0.12] * len(resolutions) + [0.12])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header - bold only, no colors
    for i in range(len(header)):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold')

    # Style rows - no colors, just highlight N/A
    for i in range(len(table_data)):
        for j in range(len(header)):
            cell = table[(i + 1, j)]
            # Highlight N/A cells
            if table_data[i][j] == 'N/A':
                cell.set_text_props(style='italic', color='gray')

    plt.title('QNM Frequency Comparison Across Resolutions',
              fontsize=14, fontweight='bold', pad=20)

    output_path = os.path.join(plot_dir, f'qnm_comparison_table_{var_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    #plt.close(fig)
    print(f"\nSaved comparison table: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='QNM Analysis for TOV Star Evolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python plot_qnm_analysis.py                # Use all data points (default folders)
  python plot_qnm_analysis.py --delta-t 1    # Subsample to t=0,1,2,3,...
  python plot_qnm_analysis.py --data-dirs DIR1 DIR2 --output-dir OUT  # Custom paths
'''
    )
    parser.add_argument('--delta-t', type=float, default=1.0,
                        help='Time interval for subsampling (e.g., 1 for t=0,1,2,...). Default: 1.0')
    parser.add_argument('--data-dirs', nargs='+', default=None,
                        help='List of data directories to analyze. Default: use FOLDERS_TO_ANALYZE')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for plots. Default: script_dir/plots')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine output directory
    if args.output_dir:
        plot_dir = args.output_dir
    else:
        plot_dir = os.path.join(script_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Determine data folders
    data_folders = []
    if args.data_dirs:
        # Use command-line provided directories
        for folder_path in args.data_dirs:
            if os.path.exists(folder_path):
                folder_name = os.path.basename(folder_path)
                data_folders.append((folder_name, folder_path))
            else:
                print(f"Warning: Folder not found: {folder_path}")
    else:
        # Use folders from FOLDERS_TO_ANALYZE list (backward compatibility)
        for folder_rel_path in FOLDERS_TO_ANALYZE:
            folder_path = os.path.join(script_dir, folder_rel_path)
            if os.path.exists(folder_path):
                folder_name = os.path.basename(folder_path)
                data_folders.append((folder_name, folder_path))
            else:
                print(f"Warning: Folder not found: {folder_path}")

    if not data_folders:
        print("No valid folders found.")
        if args.data_dirs:
            print("Check the paths provided via --data-dirs")
        else:
            print("Edit FOLDERS_TO_ANALYZE at the top of this script or use --data-dirs")
        return

    print(f"Analyzing {len(data_folders)} folder(s):")
    for name, path in data_folders:
        print(f"  - {name} ({path})")

    delta_t = args.delta_t
    if delta_t is not None:
        print(f"\nUsing delta_t = {delta_t} (subsampled data)")
    else:
        print(f"\nUsing all data points (no subsampling)")

    mode_name = "Cowling" if MODE == "cow" else "Dynamic (BSSN+hydro)"
    print(f"Mode: {mode_name} (MODE = '{MODE}')")
    print(f"Using theoretical frequencies: {list(FREQUENCIES_KHZ.keys())}")

    var_name = "rho_central" if ANALYSIS_VAR == 1 else "v_central"
    print(f"Analysis variable: {var_name} (ANALYSIS_VAR = {ANALYSIS_VAR})")

    results_list = []
    for folder_name, folder_path in data_folders:
        data = load_data_from_folder(folder_path, delta_t=delta_t, var=ANALYSIS_VAR)
        if data is not None:
            result = analyze_and_plot(data, folder_name, plot_dir)
            results_list.append(result)
        else:
            print(f"\nNo data found in: {folder_name}")

    # Create comparison table if multiple resolutions are available
    comparison_table_path = None
    if len(results_list) >= 2:
        comparison_table_path = create_comparison_table(results_list, plot_dir, var_name)

    print(f"\n{'='*70}")
    print(f"SUMMARY: Generated {len(results_list)} plot(s)")
    print(f"{'='*70}")
    for result in results_list:
        print(f"  - {result['output_path']}")

    if comparison_table_path:
        print(f"\nComparison table:")
        print(f"  - {comparison_table_path}")
    plt.show()
    #plt.close('all')


if __name__ == "__main__":
    main()