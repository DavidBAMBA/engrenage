#!/usr/bin/env python3
"""
Quasi-Normal Mode (QNM) Analysis for TOV Star Evolution

Analyzes the central density oscillations to extract QNM frequencies
and compares with theoretical values from Cowling approximation.

Reference: Shum et al. 2025 (arXiv:2509.15303v1)
- TOV star with K=100, Gamma=2, rho_c ~ 1.28e-3
- Cowling approximation (fixed spacetime)
- Theoretical f-mode: ~2.69 kHz, H1: ~4.55 kHz, H2: ~6.36 kHz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
import h5py
import os

# =============================================================================
# Unit conversions
# =============================================================================
# In geometric units with G = c = 1 and M_sun as mass unit:
# 1 M_sun = G M_sun / c^3 = 4.926e-6 s in time
# Frequency conversion: f[kHz] = f[1/M_sun] * c^3 / (2*pi*G*M_sun)
#                              = f[1/M_sun] * 1/(2*pi * 4.926e-6 s)
#                              = f[1/M_sun] * 32.26 kHz

M_SUN_SECONDS = 4.926e-6  # 1 M_sun in seconds
FREQ_CONVERSION = 1.0 / (2 * np.pi * M_SUN_SECONDS * 1e3)  # Convert to kHz

# Theoretical QNM frequencies from Shum et al. 2025 (Cowling approximation)
# For TOV star with K=100, Gamma=2, rho_c = 0.00128, M = 1.4 M_sun
THEORETICAL_FREQUENCIES_KHZ = {
    'F': 2.69,    # Fundamental mode
    'H1': 4.55,   # First overtone
    'H2': 6.36,   # Second overtone
}


def load_evolution_data(h5_file):
    """Load time series data from HDF5 file."""
    with h5py.File(h5_file, 'r') as f:
        data = {
            't': f['time'][:],
            'rho_c': f['rho_central'][:],
        }
        # Load additional fields if available
        for key in ['p_central', 'max_velocity', 'max_rho_error']:
            if key in f:
                data[key] = f[key][:]
    return data


def compute_psd(t, signal_data, t_start=100, window='blackman'):
    """
    Compute Power Spectral Density of the signal.

    Parameters
    ----------
    t : array
        Time array in code units (M_sun)
    signal_data : array
        Signal to analyze (e.g., delta_rho/rho)
    t_start : float
        Start time to exclude initial transient
    window : str
        Window function for FFT

    Returns
    -------
    freq_khz : array
        Frequencies in kHz
    psd : array
        Power spectral density
    """
    # Select data after transient
    mask = t > t_start
    t_sel = t[mask]
    sig_sel = signal_data[mask]

    # Remove mean/trend
    sig_detrend = sig_sel - np.mean(sig_sel)

    # Apply window
    if window == 'blackman':
        win = np.blackman(len(sig_detrend))
    elif window == 'hann':
        win = np.hann(len(sig_detrend))
    else:
        win = np.ones(len(sig_detrend))

    sig_windowed = sig_detrend * win

    # Compute FFT
    dt = np.mean(np.diff(t_sel))
    n = len(sig_windowed)
    freq = np.fft.rfftfreq(n, dt)  # Frequency in 1/M_sun
    fft_vals = np.abs(np.fft.rfft(sig_windowed))

    # Convert to kHz
    freq_khz = freq * FREQ_CONVERSION

    # Power spectral density
    psd = fft_vals**2

    return freq_khz, psd, fft_vals


def find_peaks_qnm(freq_khz, psd, min_freq=1.0, max_freq=10.0,
                   prominence_factor=0.1, n_peaks=5):
    """
    Find QNM peaks in the power spectrum.

    Parameters
    ----------
    freq_khz : array
        Frequencies in kHz
    psd : array
        Power spectral density
    min_freq, max_freq : float
        Frequency range to search (kHz)
    prominence_factor : float
        Minimum prominence as fraction of max
    n_peaks : int
        Maximum number of peaks to return

    Returns
    -------
    peak_freqs : array
        Peak frequencies in kHz
    peak_amplitudes : array
        Peak amplitudes
    """
    # Restrict to frequency range
    mask = (freq_khz > min_freq) & (freq_khz < max_freq)
    freq_sel = freq_khz[mask]
    psd_sel = psd[mask]

    # Find peaks
    peaks, properties = signal.find_peaks(
        psd_sel,
        height=np.max(psd_sel) * prominence_factor,
        distance=10,
        prominence=np.max(psd_sel) * 0.05
    )

    # Sort by amplitude
    if len(peaks) > 0:
        sorted_idx = np.argsort(psd_sel[peaks])[::-1]
        peaks = peaks[sorted_idx[:n_peaks]]

        peak_freqs = freq_sel[peaks]
        peak_amplitudes = psd_sel[peaks]
    else:
        peak_freqs = np.array([])
        peak_amplitudes = np.array([])

    return peak_freqs, peak_amplitudes


def damped_sinusoid(t, A, tau, omega, phi, C):
    """Damped sinusoidal function for fitting."""
    return A * np.exp(-t / tau) * np.cos(omega * t + phi) + C


def fit_decay_rate(t, signal_data, f_mode_khz, t_start=500, t_end=None):
    """
    Fit decay rate of the fundamental mode.

    Parameters
    ----------
    t : array
        Time in code units
    signal_data : array
        Signal (delta_rho/rho)
    f_mode_khz : float
        Expected f-mode frequency in kHz
    t_start, t_end : float
        Time window for fitting

    Returns
    -------
    tau : float
        Decay time in code units
    omega : float
        Angular frequency in 1/M_sun
    fit_params : dict
        All fitted parameters
    """
    if t_end is None:
        t_end = t[-1]

    mask = (t > t_start) & (t < t_end)
    t_fit = t[mask]
    sig_fit = signal_data[mask]

    # Apply bandpass filter around f-mode
    dt = np.mean(np.diff(t_fit))
    fs = 1.0 / dt
    f_mode_code = f_mode_khz / FREQ_CONVERSION  # Convert to code units

    # Butterworth bandpass filter
    f_low = 0.7 * f_mode_code
    f_high = 1.3 * f_mode_code
    nyq = 0.5 * fs

    if f_high < nyq:
        b, a = signal.butter(4, [f_low/nyq, f_high/nyq], btype='band')
        sig_filtered = signal.filtfilt(b, a, sig_fit)
    else:
        sig_filtered = sig_fit

    # Initial guess
    omega_guess = 2 * np.pi * f_mode_code
    A_guess = np.max(np.abs(sig_filtered))
    tau_guess = (t_fit[-1] - t_fit[0]) / 2

    try:
        popt, pcov = curve_fit(
            damped_sinusoid,
            t_fit - t_fit[0],
            sig_filtered,
            p0=[A_guess, tau_guess, omega_guess, 0, 0],
            bounds=([0, 100, 0.5*omega_guess, -np.pi, -np.inf],
                    [np.inf, np.inf, 1.5*omega_guess, np.pi, np.inf]),
            maxfev=10000
        )

        fit_params = {
            'A': popt[0],
            'tau': popt[1],
            'omega': popt[2],
            'phi': popt[3],
            'C': popt[4],
            'tau_seconds': popt[1] * M_SUN_SECONDS,
            'decay_rate_per_s': 1.0 / (popt[1] * M_SUN_SECONDS),
        }

        return popt[1], popt[2], fit_params

    except Exception as e:
        print(f"Fitting failed: {e}")
        return None, None, None


def plot_qnm_analysis(data, output_dir, suffix=""):
    """
    Create comprehensive QNM analysis plots.

    Parameters
    ----------
    data : dict
        Dictionary with 't' and 'rho_c' arrays
    output_dir : str
        Output directory for plots
    suffix : str
        Suffix for output filename
    """
    t = data['t']
    rho_c = data['rho_c']
    rho_c_0 = rho_c[0]

    # Relative perturbation
    delta_rho = (rho_c - rho_c_0) / rho_c_0

    # Compute PSD
    freq_khz, psd, fft_amp = compute_psd(t, delta_rho, t_start=100)

    # Find peaks
    peak_freqs, peak_amps = find_peaks_qnm(freq_khz, psd)

    # Create figure
    fig = plt.figure(figsize=(16, 12))

    # =========================================================================
    # Panel 1: Central density vs time
    # =========================================================================
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(t, rho_c, 'b-', linewidth=0.5, alpha=0.8)
    ax1.axhline(rho_c_0, color='r', linestyle='--', alpha=0.5,
                label=f'$\\rho_{{c,0}}$ = {rho_c_0:.4e}')
    ax1.set_xlabel('t [M$_\\odot$]', fontsize=12)
    ax1.set_ylabel('$\\rho_c$ [M$_\\odot^{-2}$]', fontsize=12)
    ax1.set_title('Central Density Evolution', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 2: Relative perturbation
    # =========================================================================
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t, delta_rho * 100, 'b-', linewidth=0.5, alpha=0.8)
    ax2.set_xlabel('t [M$_\\odot$]', fontsize=12)
    ax2.set_ylabel('$\\Delta\\rho_c / \\rho_{c,0}$ [%]', fontsize=12)
    ax2.set_title('Relative Density Perturbation', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 3: Power Spectral Density with theoretical lines
    # =========================================================================
    ax3 = fig.add_subplot(2, 2, 3)

    # Plot PSD
    ax3.semilogy(freq_khz, psd, 'b-', linewidth=1, alpha=0.8, label='Simulation')

    # Add theoretical frequencies as vertical dashed lines
    colors = {'F': 'red', 'H1': 'green', 'H2': 'orange'}
    for mode, f_theo in THEORETICAL_FREQUENCIES_KHZ.items():
        ax3.axvline(f_theo, color=colors[mode], linestyle='--', linewidth=2,
                    label=f'{mode} (theo) = {f_theo:.2f} kHz', alpha=0.8)

    # Mark detected peaks
    for i, (f_peak, amp_peak) in enumerate(zip(peak_freqs[:3], peak_amps[:3])):
        ax3.plot(f_peak, amp_peak, 'ko', markersize=8)
        ax3.annotate(f'{f_peak:.2f} kHz',
                     xy=(f_peak, amp_peak),
                     xytext=(5, 10), textcoords='offset points',
                     fontsize=9, fontweight='bold')

    ax3.set_xlabel('Frequency [kHz]', fontsize=12)
    ax3.set_ylabel('PSD', fontsize=12)
    ax3.set_title('Power Spectral Density - QNM Spectrum', fontsize=14)
    ax3.set_xlim(0, 10)
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 4: Zoom on oscillations
    # =========================================================================
    ax4 = fig.add_subplot(2, 2, 4)

    # Select a time window to show oscillations clearly
    t_zoom_start = len(t) // 4
    t_zoom_end = t_zoom_start + 500 if t_zoom_start + 500 < len(t) else len(t) // 2

    ax4.plot(t[t_zoom_start:t_zoom_end],
             delta_rho[t_zoom_start:t_zoom_end] * 100,
             'b-', linewidth=1)

    # Calculate expected period from f-mode
    f_mode = THEORETICAL_FREQUENCIES_KHZ['F']
    T_mode = 1.0 / (f_mode / FREQ_CONVERSION)  # Period in code units

    ax4.set_xlabel('t [M$_\\odot$]', fontsize=12)
    ax4.set_ylabel('$\\Delta\\rho_c / \\rho_{c,0}$ [%]', fontsize=12)
    ax4.set_title(f'Zoom: Oscillations (Expected T$_F$ = {T_mode:.1f} M$_\\odot$)',
                  fontsize=14)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    out_path = os.path.join(output_dir, f'tov_qnm_analysis{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")

    # =========================================================================
    # Create detailed frequency comparison table
    # =========================================================================
    print("\n" + "="*70)
    print("QNM FREQUENCY ANALYSIS")
    print("="*70)
    print(f"\nSimulation parameters:")
    print(f"  Total time: {t[-1]:.1f} M_sun = {t[-1] * M_SUN_SECONDS * 1e3:.2f} ms")
    print(f"  Initial rho_c: {rho_c_0:.6e}")
    print(f"  Final rho_c: {rho_c[-1]:.6e}")
    print(f"  Max |delta_rho/rho|: {np.max(np.abs(delta_rho))*100:.2f}%")

    print(f"\n{'Mode':<8} {'Theoretical [kHz]':<20} {'Detected [kHz]':<20} {'Difference [%]':<15}")
    print("-"*70)

    mode_names = ['F', 'H1', 'H2']
    for i, mode in enumerate(mode_names):
        f_theo = THEORETICAL_FREQUENCIES_KHZ[mode]
        if i < len(peak_freqs):
            f_det = peak_freqs[i]
            diff = (f_det - f_theo) / f_theo * 100
            print(f"{mode:<8} {f_theo:<20.2f} {f_det:<20.2f} {diff:<15.1f}")
        else:
            print(f"{mode:<8} {f_theo:<20.2f} {'N/A':<20} {'N/A':<15}")

    print("="*70)

    return fig, peak_freqs, peak_amps


def main():
    """Main function to run QNM analysis."""
    # Find the most recent/largest evolution file
    data_dir = "/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data"
    plot_dir = "/home/davidbamba/repositories/engrenage/examples/TOV/plots"

    # List h5 files and find the largest one (most data)
    h5_files = [f for f in os.listdir(data_dir)
                if f.startswith('tov_evolution') and f.endswith('.h5')]

    if not h5_files:
        print("No evolution data files found!")
        return

    # Get file sizes and find largest
    file_sizes = {f: os.path.getsize(os.path.join(data_dir, f)) for f in h5_files}
    largest_file = max(file_sizes, key=file_sizes.get)
    h5_path = os.path.join(data_dir, largest_file)

    print(f"Loading data from: {largest_file}")
    print(f"File size: {file_sizes[largest_file] / 1024:.1f} KB")

    # Load data
    data = load_evolution_data(h5_path)

    print(f"Time range: {data['t'][0]:.2f} to {data['t'][-1]:.2f} M_sun")
    print(f"Number of points: {len(data['t'])}")

    # Run analysis
    fig, peak_freqs, peak_amps = plot_qnm_analysis(data, plot_dir, suffix="_iso")

    plt.show()


if __name__ == "__main__":
    main()
