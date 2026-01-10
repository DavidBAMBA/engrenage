#!/usr/bin/env python3
"""
Quasi-Normal Mode (QNM) Analysis for TOV Star Evolution - Version 2

Improved peak detection that searches near theoretical frequencies.

References:
- Font et al. 2002 (gr-qc/0110047) - Table I & II
- GRoovy paper (arXiv:2412.03659) - Figure 4

TOV star parameters: K=100, Gamma=2, rho_c = 1.28e-3, M/R = 0.15
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import maximum_filter1d
import h5py
import os

# =============================================================================
# Unit conversions
# =============================================================================
M_SUN_SECONDS = 4.926e-6  # 1 M_sun in seconds
FREQ_CONVERSION = 1.0 / (M_SUN_SECONDS * 1e3)  # Convert 1/M_sun to kHz

# =============================================================================
# Theoretical QNM frequencies from Font et al. 2002 (gr-qc/0110047)
# For TOV star with K=100, Gamma=2, rho_c = 1.28e-3, M/R = 0.15
# =============================================================================

# Table I: Cowling approximation (fixed spacetime)
FREQUENCIES_COWLING_KHZ = {
    'F':  2.696,
    'H1': 4.534,
    'H2': 6.346,
    'H3': 8.161,
    'H4': 9.971,
    'H5': 11.806,
    'H6': 13.605,
}

# Table II: Full GR (coupled evolution)
FREQUENCIES_FULL_GR_KHZ = {
    'F':  1.450,
    'H1': 3.958,
    'H2': 5.935,
    'H3': 7.812,
}


def load_evolution_data(h5_file):
    """Load time series data from HDF5 file."""
    with h5py.File(h5_file, 'r') as f:
        data = {
            't': f['time'][:],
            'rho_c': f['rho_central'][:],
        }
        for key in ['p_central', 'max_velocity', 'max_rho_error']:
            if key in f:
                data[key] = f[key][:]
    return data


def compute_power_spectrum(t, signal_data, t_start=None, window='hann'):
    """
    Compute Power Spectral Density of the signal.
    """
    if t_start is not None:
        mask = t > t_start
        t_sel = t[mask]
        sig_sel = signal_data[mask]
    else:
        t_sel = t
        sig_sel = signal_data

    # Remove mean
    sig_detrend = sig_sel - np.mean(sig_sel)

    # Apply window function
    n = len(sig_detrend)
    if window == 'hann':
        win = np.hanning(n)
    elif window == 'blackman':
        win = np.blackman(n)
    else:
        win = np.ones(n)

    sig_windowed = sig_detrend * win

    # Compute FFT
    dt = np.mean(np.diff(t_sel))
    freq = np.fft.rfftfreq(n, dt)
    fft_vals = np.fft.rfft(sig_windowed)

    # Convert frequency to kHz
    freq_khz = freq * FREQ_CONVERSION

    # Power spectrum
    power = np.abs(fft_vals)**2
    power = power / np.sum(win**2)

    return freq_khz, power


def find_peak_near_frequency(freq_khz, power, target_freq, tolerance=0.5):
    """
    Find the peak closest to a target frequency within a tolerance.
    
    Parameters
    ----------
    freq_khz : array
        Frequency array in kHz
    power : array
        Power spectrum
    target_freq : float
        Target frequency in kHz
    tolerance : float
        Search window half-width in kHz
        
    Returns
    -------
    peak_freq : float or None
        Detected peak frequency, or None if not found
    peak_power : float or None
        Power at the peak
    """
    # Define search window
    mask = (freq_khz >= target_freq - tolerance) & (freq_khz <= target_freq + tolerance)
    
    if not np.any(mask):
        return None, None
    
    freq_window = freq_khz[mask]
    power_window = power[mask]
    
    # Find maximum within window
    max_idx = np.argmax(power_window)
    
    # Verify it's a local maximum (not at boundary)
    if max_idx == 0 or max_idx == len(power_window) - 1:
        # Check if it's still significantly above neighbors
        if power_window[max_idx] > 2 * np.median(power_window):
            return freq_window[max_idx], power_window[max_idx]
        return None, None
    
    # Check it's a true peak (higher than neighbors)
    if (power_window[max_idx] > power_window[max_idx-1] and 
        power_window[max_idx] > power_window[max_idx+1]):
        return freq_window[max_idx], power_window[max_idx]
    
    # If not a clean peak, still return max if significantly above baseline
    if power_window[max_idx] > 5 * np.median(power_window):
        return freq_window[max_idx], power_window[max_idx]
    
    return None, None


def detect_qnm_frequencies(freq_khz, power, theoretical_freqs, tolerance=0.5):
    """
    Detect QNM frequencies by searching near theoretical values.
    
    Parameters
    ----------
    freq_khz : array
        Frequency array in kHz
    power : array
        Power spectrum
    theoretical_freqs : dict
        Dictionary of theoretical frequencies {'F': f0, 'H1': f1, ...}
    tolerance : float
        Search window half-width in kHz
        
    Returns
    -------
    detected : dict
        Dictionary with detected frequencies for each mode
    """
    detected = {}
    
    for mode, f_theo in theoretical_freqs.items():
        f_det, p_det = find_peak_near_frequency(freq_khz, power, f_theo, tolerance)
        detected[mode] = {
            'theoretical': f_theo,
            'detected': f_det,
            'power': p_det,
            'diff_percent': ((f_det - f_theo) / f_theo * 100) if f_det else None
        }
    
    return detected


def print_frequency_table(detected, title="QNM Frequency Analysis"):
    """Print a formatted table of detected vs theoretical frequencies."""
    print("\n" + "="*75)
    print(title)
    print("="*75)
    print(f"\n{'Mode':<8} {'Theoretical [kHz]':<20} {'Detected [kHz]':<20} {'Diff [%]':<12}")
    print("-"*75)
    
    for mode, data in detected.items():
        f_theo = data['theoretical']
        f_det = data['detected']
        diff = data['diff_percent']
        
        if f_det is not None:
            print(f"{mode:<8} {f_theo:<20.3f} {f_det:<20.3f} {diff:<12.2f}")
        else:
            print(f"{mode:<8} {f_theo:<20.3f} {'N/A':<20} {'N/A':<12}")
    
    print("="*75)


def plot_qnm_analysis(t, rho_c, freq_khz, power, detected,
                      output_path=None, max_freq=14.0):
    """
    Create publication-quality plot with 2 subplots:
    1. Central density evolution with zoom inset
    2. Power spectrum with QNM frequencies
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # =========================================================================
    # Left plot: Central density evolution with zoom inset
    # =========================================================================
    ax1 = axes[0]

    rho_c_0 = rho_c[0]
    delta_rho_rel = (rho_c - rho_c_0) / rho_c_0

    ax1.plot(t, delta_rho_rel, color='red', linewidth=0.8)
    ax1.set_xlabel(r'$t$', fontsize=12)
    ax1.set_ylabel(r'$(\rho_c - \rho_{c,0})/\rho_{c,0}$', fontsize=12)
    ax1.set_title('Central Density Relative Change', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Zoom inset for first 1000 M
    ax_inset = inset_axes(ax1, width="45%", height="40%", loc='upper right',
                          bbox_to_anchor=(0.0, 0.0, 0.95, 0.95),
                          bbox_transform=ax1.transAxes)

    # Mask for zoom region
    mask_zoom = t <= 1000
    t_zoom = t[mask_zoom]
    delta_zoom = delta_rho_rel[mask_zoom]

    # Plot rho_c/rho_c0 in inset (like image 1)
    rho_ratio = rho_c[mask_zoom] / rho_c_0
    ax_inset.plot(t_zoom, rho_ratio, 'k-', linewidth=0.6)
    ax_inset.set_xlabel(r'Time', fontsize=9)
    ax_inset.set_ylabel(r'$\rho_c/\rho_{c,0}$', fontsize=9)
    ax_inset.tick_params(labelsize=8)
    ax_inset.set_xlim(0, 1000)

    # =========================================================================
    # Right plot: Power spectrum
    # =========================================================================
    ax2 = axes[1]

    ax2.semilogy(freq_khz, power, color='#8B0000', linewidth=1.5, label='Simulation')

    # Add theoretical frequencies and detected peaks
    for mode, data in detected.items():
        f_theo = data['theoretical']
        f_det = data['detected']
        p_det = data['power']

        if f_theo < max_freq:
            ax2.axvline(f_theo, color='gray', linestyle='--', linewidth=1, alpha=0.7)

            ymax = ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else np.max(power) * 2
            ax2.text(f_theo, ymax * 0.4, mode, ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

            if f_det is not None and p_det is not None:
                ax2.plot(f_det, p_det, 'bo', markersize=6, alpha=0.7)

    ax2.set_xlabel('Frequency [kHz]', fontsize=12)
    ax2.set_ylabel('Power', fontsize=12)
    ax2.set_xlim(0, max_freq)
    ax2.set_title('Power Spectrum - QNM Frequencies', fontsize=14)
    ax2.grid(True, which='major', linestyle=':', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, axes


def analyze_qnm(data, theoretical_freqs, t_start=None, tolerance=0.5, 
                output_dir=None, suffix=""):
    """
    Complete QNM analysis pipeline.
    
    Parameters
    ----------
    data : dict
        Dictionary with 't' and 'rho_c' arrays
    theoretical_freqs : dict
        Theoretical frequencies to compare
    t_start : float, optional
        Start time to exclude transient
    tolerance : float
        Search window for peak detection (kHz)
    output_dir : str, optional
        Output directory for plots
    suffix : str
        Suffix for output filename
    """
    t = data['t']
    rho_c = data['rho_c']
    rho_c_0 = rho_c[0]
    
    # Compute relative perturbation
    delta_rho = (rho_c - rho_c_0) / rho_c_0
    
    # Compute power spectrum
    freq_khz, power = compute_power_spectrum(t, delta_rho, t_start=t_start, window='hann')
    
    # Detect peaks near theoretical frequencies
    detected = detect_qnm_frequencies(freq_khz, power, theoretical_freqs, tolerance)
    
    # Print results
    print(f"\nSimulation parameters:")
    print(f"  Total time: {t[-1]:.1f} M_sun = {t[-1] * M_SUN_SECONDS * 1e3:.2f} ms")
    print(f"  Initial rho_c: {rho_c_0:.6e}")
    print(f"  Max |delta_rho/rho|: {np.max(np.abs(delta_rho))*100:.3f}%")
    print(f"  Peak detection tolerance: ±{tolerance} kHz")
    
    print_frequency_table(detected)
    
    # Create plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f'qnm_analysis{suffix}.png')
        plot_qnm_analysis(t, rho_c, freq_khz, power, detected, out_path)

    return detected, freq_khz, power


# =============================================================================
# Test with synthetic data
# =============================================================================
def generate_test_data(t_max=15000, dt=0.5, frequencies_khz=None, noise_level=0.0005):
    """Generate synthetic TOV oscillation data for testing."""
    if frequencies_khz is None:
        frequencies_khz = FREQUENCIES_COWLING_KHZ

    t = np.arange(0, t_max, dt)
    rho_c_0 = 1.28e-3
    
    delta_rho = np.zeros_like(t)
    
    for i, (mode, f_khz) in enumerate(frequencies_khz.items()):
        f_code = f_khz / FREQ_CONVERSION
        omega = 2 * np.pi * f_code
        amplitude = 0.005 / (i + 1)**1.5
        tau = t_max / (1.5 * (i + 1))
        delta_rho += amplitude * np.exp(-t / tau) * np.cos(omega * t)
    
    delta_rho += noise_level * np.random.randn(len(t))
    rho_c = rho_c_0 * (1 + delta_rho)
    
    return {'t': t, 'rho_c': rho_c}


def main():
    """Main function."""

    # Use paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "tov_evolution_data")
    plot_dir = os.path.join(script_dir, "plots")

    use_synthetic = True

    # Search for evolution data in star-specific subfolders (new structure)
    if os.path.exists(data_dir):
        # Find star folders
        star_folders = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('tov_star_')]

        if star_folders:
            # Use most recent star folder
            star_folder = max(star_folders, key=os.path.getmtime)
            h5_path = os.path.join(star_folder, 'tov_evolution.h5')

            if os.path.exists(h5_path):
                print(f"Loading data from: {os.path.relpath(h5_path, data_dir)}")
                data = load_evolution_data(h5_path)
                use_synthetic = False

    if use_synthetic:
        print("Generating synthetic test data...")
        data = generate_test_data()
    
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Time range: {data['t'][0]:.2f} to {data['t'][-1]:.2f} M_sun")
    print(f"Number of points: {len(data['t'])}")
    
    # Choose approximation (Cowling or full GR)
    is_cowling = True
    theoretical_freqs = FREQUENCIES_COWLING_KHZ if is_cowling else FREQUENCIES_FULL_GR_KHZ
    approx_name = "Cowling" if is_cowling else "Full GR"
    print(f"\nUsing {approx_name} approximation frequencies")
    
    # Run analysis with improved peak detection
    detected, freq_khz, power = analyze_qnm(
        data, 
        theoretical_freqs, 
        t_start=100,
        tolerance=0.4,  # Search ±0.4 kHz around theoretical
        output_dir=plot_dir,
        suffix=f"_{approx_name.lower().replace(' ', '_')}"
    )
    
    plt.show()


if __name__ == "__main__":
    main()