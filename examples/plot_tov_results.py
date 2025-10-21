#!/usr/bin/env python3
"""
Plot TOV evolution results from HDF5 data files.

This script reads the data saved by TOVEvolutionLong.py and creates plots
similar to those in the paper 1309.7808v2.pdf (Figure 2).
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
import os
import glob
from scipy.signal import find_peaks

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 10),
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})


def load_simulation_data(data_dir):
    """
    Load simulation data from HDF5 files.

    Args:
        data_dir: Directory containing the simulation output files

    Returns:
        Dictionary with evolution data, snapshots, and metadata
    """
    # Find the most recent files
    snapshot_files = glob.glob(os.path.join(data_dir, 'tov_snapshots_*.h5'))
    evolution_files = glob.glob(os.path.join(data_dir, 'tov_evolution_*.h5'))
    metadata_files = glob.glob(os.path.join(data_dir, 'tov_metadata_*.json'))

    if not snapshot_files or not evolution_files or not metadata_files:
        raise FileNotFoundError(f"Could not find simulation data in {data_dir}")

    # Use the most recent files
    snapshot_file = sorted(snapshot_files)[-1]
    evolution_file = sorted(evolution_files)[-1]
    metadata_file = sorted(metadata_files)[-1]

    print(f"Loading data from:")
    print(f"  Snapshots: {os.path.basename(snapshot_file)}")
    print(f"  Evolution: {os.path.basename(evolution_file)}")
    print(f"  Metadata:  {os.path.basename(metadata_file)}")

    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Load evolution data
    evolution_data = {}
    with h5py.File(evolution_file, 'r') as f:
        for key in f.keys():
            evolution_data[key] = f[key][:]

    # Load grid data and selected snapshots
    snapshots = {}
    with h5py.File(snapshot_file, 'r') as f:
        # Load grid
        grid_r = f['grid/r'][:]

        # Get list of available snapshots
        snapshot_groups = [g for g in f['snapshots'].keys()]
        snapshot_steps = [int(g.split('_')[1]) for g in snapshot_groups]

        # Load a few representative snapshots
        selected_steps = [0]  # Initial
        if 1000 in snapshot_steps:
            selected_steps.append(1000)
        if 10000 in snapshot_steps:
            selected_steps.append(10000)
        if 100000 in snapshot_steps:
            selected_steps.append(100000)
        # Add the last available snapshot
        if snapshot_steps:
            selected_steps.append(max(snapshot_steps))

        for step in selected_steps:
            if f'step_{step:08d}' in f['snapshots']:
                group = f[f'snapshots/step_{step:08d}']
                snapshots[step] = {
                    'time': group.attrs['time'],
                    'step': group.attrs['step']
                }
                if 'primitives' in group:
                    snapshots[step]['rho0'] = group['primitives/rho0'][:]
                    snapshots[step]['p'] = group['primitives/p'][:]
                    snapshots[step]['vr'] = group['primitives/vr'][:]

    return {
        'evolution': evolution_data,
        'snapshots': snapshots,
        'metadata': metadata,
        'grid': grid_r
    }


def plot_central_density_evolution(data, output_file=None):
    """
    Plot the evolution of normalized central density.

    Similar to Figure 2 upper panel in the paper.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Get evolution data
    # Time is in geometric units (M=1), convert to ms: 1 M_sun = 4.93e-3 ms
    time_ms = data['evolution']['time'] * 4.93e-3
    rho_c = data['evolution']['rho_central']
    rho_c0 = rho_c[0]

    # Upper panel: Normalized central density
    ax1.plot(time_ms, rho_c / rho_c0, 'b-', label='Simulation')
    ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)

    # Add inset for zoom
    if len(time_ms) > 1000:
        axins = ax1.inset_axes([0.5, 0.6, 0.45, 0.35])
        # Show first 10ms in detail
        mask = time_ms < 10
        if np.any(mask):
            axins.plot(time_ms[mask], rho_c[mask] / rho_c0, 'b-')
            axins.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
            axins.set_xlabel('time [ms]')
            axins.grid(True, alpha=0.3)

    ax1.set_ylabel(r'$\rho_c(t)/\rho_c(0)$')
    ax1.set_xlabel('time [ms]')
    ax1.set_title('Central Density Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Lower panel: Relative central density change
    rel_change = np.abs(rho_c - rho_c0) / rho_c0
    ax2.semilogy(time_ms, rel_change, 'r-', label=r'$|\rho_c(t) - \rho_c(0)|/\rho_c(0)$')
    ax2.set_ylabel(r'$|\Delta\rho_c|/\rho_c(0)$')
    ax2.set_xlabel('time [ms]')
    ax2.set_title('Relative Central Density Change')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved central density plot to {output_file}")

    return fig


def plot_l1_norm_evolution(data, output_file=None):
    """
    Plot the evolution of L1 norm of density error.

    Similar to Figure 2 middle panel in the paper.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Get evolution data
    # Time is in geometric units (M=1), convert to ms: 1 M_sun = 4.93e-3 ms
    time_ms = data['evolution']['time'] * 4.93e-3
    l1_error = data['evolution']['l1_rho_error']
    l2_error = data['evolution']['l2_rho_error']

    # Upper panel: L1 norm
    ax1.plot(time_ms, l1_error, 'b-', label=r'$||\rho(t) - \rho(0)||_1$')
    ax1.set_ylabel(r'$||\rho(t) - \rho(0)||_1$')
    ax1.set_xlabel('time [ms]')
    ax1.set_title('L1 Norm of Density Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Lower panel: L2 norm
    ax2.plot(time_ms, l2_error, 'r-', label=r'$||\rho(t) - \rho(0)||_2$')
    ax2.set_ylabel(r'$||\rho(t) - \rho(0)||_2$')
    ax2.set_xlabel('time [ms]')
    ax2.set_title('L2 Norm of Density Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved L1 norm plot to {output_file}")

    return fig


def plot_convergence_analysis(data_dirs, resolutions, output_file=None):
    """
    Plot convergence analysis similar to Figure 2 lower panel.

    Args:
        data_dirs: List of directories with different resolution runs
        resolutions: List of resolution labels (e.g., ['Δr=0.2', 'Δr=0.1', 'Δr=0.05'])
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    dr_values = []
    l1_at_fixed_time = []

    target_time = 5.0  # ms - time to measure convergence

    for data_dir, resolution in zip(data_dirs, resolutions):
        try:
            data = load_simulation_data(data_dir)
            # Time is in geometric units (M=1), convert to ms: 1 M_sun = 4.93e-3 ms
            time_ms = data['evolution']['time'] * 4.93e-3
            l1_error = data['evolution']['l1_rho_error']

            # Find value at target time
            idx = np.argmin(np.abs(time_ms - target_time))
            if np.abs(time_ms[idx] - target_time) < 1.0:  # Within 1 ms
                l1_at_fixed_time.append(l1_error[idx])
                # Extract dr from resolution label
                dr = float(resolution.split('=')[1])
                dr_values.append(dr)
        except:
            print(f"Could not load data for {resolution}")

    if len(dr_values) >= 2:
        # Fit power law
        log_dr = np.log(dr_values)
        log_l1 = np.log(l1_at_fixed_time)
        coeffs = np.polyfit(log_dr, log_l1, 1)
        convergence_order = coeffs[0]

        # Plot
        ax.loglog(dr_values, l1_at_fixed_time, 'bo-', label='Data', markersize=8)

        # Add fit line
        dr_fit = np.logspace(np.log10(min(dr_values)*0.8),
                            np.log10(max(dr_values)*1.2), 50)
        l1_fit = np.exp(coeffs[1]) * dr_fit**coeffs[0]
        ax.loglog(dr_fit, l1_fit, 'r--',
                 label=f'Slope={convergence_order:.2f}')

        ax.set_xlabel('Δr')
        ax.set_ylabel(f'$||\rho(t={target_time}ms) - \rho(0)||_1$')
        ax.set_title('Convergence Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

        print(f"Convergence order: {convergence_order:.2f}")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved convergence plot to {output_file}")

    return fig


def plot_density_snapshots(data, output_file=None):
    """
    Plot density profiles at different times.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    grid_r = data['grid']

    # Plot snapshots at different times
    colors = plt.cm.viridis(np.linspace(0, 1, len(data['snapshots'])))

    for i, (step, snapshot) in enumerate(sorted(data['snapshots'].items())):
        if 'rho0' in snapshot:
            # Time is in geometric units (M=1), convert to ms: 1 M_sun = 4.93e-3 ms
            time_ms = snapshot['time'] * 4.93e-3
            label = f't = {time_ms:.1f} ms (step {step})'
            ax.semilogy(grid_r, snapshot['rho0'], color=colors[i],
                       label=label, alpha=0.8)

    ax.set_xlabel('r')
    ax.set_ylabel(r'$\rho_0$')
    ax.set_title('Density Profile Evolution')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0, 20)  # Focus on star region

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved density snapshots to {output_file}")

    return fig


def plot_oscillation_spectrum(data, output_file=None):
    """
    Plot power spectrum to identify oscillation modes.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Get central density time series
    time = data['evolution']['time']
    rho_c = data['evolution']['rho_central']

    # Remove trend
    rho_c_detrended = rho_c - np.mean(rho_c)

    # Compute FFT
    # Time is in geometric units, convert to seconds: 1 M_sun = 4.93e-6 s
    dt = np.median(np.diff(time)) * 4.93e-6
    frequencies = np.fft.fftfreq(len(time), dt)
    fft = np.fft.fft(rho_c_detrended)
    power = np.abs(fft)**2

    # Only plot positive frequencies
    pos_mask = frequencies > 0
    freq_khz = frequencies[pos_mask] * 1000  # Convert to kHz

    # Limit to reasonable frequency range
    freq_mask = (freq_khz > 0.1) & (freq_khz < 10)

    if np.any(freq_mask):
        ax.semilogy(freq_khz[freq_mask], power[pos_mask][freq_mask], 'b-')

        # Find peaks
        peaks, _ = find_peaks(np.log10(power[pos_mask][freq_mask]), height=0)
        if len(peaks) > 0:
            peak_freqs = freq_khz[freq_mask][peaks]
            peak_powers = power[pos_mask][freq_mask][peaks]

            # Mark the strongest peaks
            strongest = np.argsort(peak_powers)[-3:]  # Top 3 peaks
            for idx in strongest:
                ax.axvline(peak_freqs[idx], color='r', linestyle='--', alpha=0.5)
                ax.text(peak_freqs[idx], peak_powers[idx]*1.5,
                       f'{peak_freqs[idx]:.2f} kHz',
                       rotation=90, ha='right', va='bottom')

    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Oscillation Spectrum of Central Density')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved spectrum plot to {output_file}")

    return fig


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Plot TOV evolution results')
    parser.add_argument('data_dir', nargs='?', default='tov_evolution_data2',
                       help='Directory containing simulation data')
    parser.add_argument('--output-dir', default='tov_plots',
                       help='Directory for output plots')
    parser.add_argument('--convergence', nargs='+',
                       help='List of directories for convergence study')
    parser.add_argument('--resolutions', nargs='+',
                       default=['Δr=0.2', 'Δr=0.1', 'Δr=0.05'],
                       help='Resolution labels for convergence study')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data_dir}...")
    try:
        data = load_simulation_data(args.data_dir)
        print(f"Loaded {len(data['evolution']['time'])} evolution points")
        print(f"Loaded {len(data['snapshots'])} snapshots")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run TOVEvolutionLong.py first to generate data.")
        return

    # Create plots
    print("\nGenerating plots...")

    # 1. Central density evolution
    plot_central_density_evolution(
        data,
        os.path.join(args.output_dir, 'central_density_evolution.png')
    )

    # 2. L1 norm evolution
    plot_l1_norm_evolution(
        data,
        os.path.join(args.output_dir, 'l1_norm_evolution.png')
    )

    # 3. Density snapshots
    plot_density_snapshots(
        data,
        os.path.join(args.output_dir, 'density_snapshots.png')
    )

    # 4. Oscillation spectrum
    plot_oscillation_spectrum(
        data,
        os.path.join(args.output_dir, 'oscillation_spectrum.png')
    )

    # 5. Convergence analysis (if multiple directories provided)
    if args.convergence and len(args.convergence) >= 2:
        print("\nPerforming convergence analysis...")
        plot_convergence_analysis(
            args.convergence,
            args.resolutions[:len(args.convergence)],
            os.path.join(args.output_dir, 'convergence_analysis.png')
        )

    print(f"\nAll plots saved to {args.output_dir}/")

    # Show plots
    plt.show()


if __name__ == "__main__":
    main()