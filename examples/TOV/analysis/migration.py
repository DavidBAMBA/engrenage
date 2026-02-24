#!/usr/bin/env python3
"""
Migration Test: Unstable to Stable Branch

Plots the evolution of a TOV star initialized in an unstable configuration
(central density above the maximum-mass model) as it migrates to a stable
configuration with approximately the same rest mass.

This test validates the code's ability to handle highly dynamical, non-adiabatic
evolution with large amplitude oscillations.

Reference: Font et al. 2002, Fig. 9
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import h5py
import os
import sys
import argparse

# Unit conversions
M_SUN_SECONDS = 4.926e-6

def load_evolution_data(h5_file):
    """Load evolution data from HDF5 file."""
    with h5py.File(h5_file, 'r') as f:
        t = f['time'][:]
        rho_central = f['rho_central'][:]
        if 'mass' in f.keys():
            mass = f['mass'][:]
        else:
            mass = None
    return {'t': t, 'rho_central': rho_central, 'mass': mass}


def load_timeseries_npz(npz_file):
    """Load timeseries data from NPZ file."""
    npz = np.load(npz_file)
    t = npz['times']
    rho_central = npz['rho_central']
    mass = npz.get('mass', None)
    return {'t': t, 'rho_central': rho_central, 'mass': mass}


def load_data_from_folder(folder_path):
    """Load data from a specific folder."""
    files = os.listdir(folder_path)

    for f in files:
        filepath = os.path.join(folder_path, f)
        if f == 'timeseries.npz':
            return load_timeseries_npz(filepath)
        elif f.endswith('.h5') and 'evolution' in f:
            return load_evolution_data(filepath)

    return None


def load_bssn_snapshots(folder_path):
    """
    Load BSSN snapshots with lapse, K, conformal factor, h_rr, shiftr, constraints, and hydro primitives.
    Returns times, radial grid r, and lists of lapse, K, phi, h_rr, shiftr, Ham, Mom_r, rho, p fields.

    Only works for dynamic evolution data.
    """
    # Try all possible snapshot file names (dynamic only)
    possible_names = [
        'tov_snapshots_dyn_jax.h5',
        'tov_snapshots_dyn.h5',
    ]

    h5_file = None
    for filename in possible_names:
        filepath = os.path.join(folder_path, filename)
        if os.path.exists(filepath):
            h5_file = filepath
            break

    if h5_file is None:
        return None, None, None, None, None, None, None, None, None, None, None

    times = []
    lapse_list = []
    K_list = []
    phi_list = []
    h_rr_list = []
    shiftr_list = []
    Ham_list = []
    Mom_r_list = []
    rho_list = []
    p_list = []

    with h5py.File(h5_file, 'r') as f:
        # Load radial grid
        r = f['grid/r'][:]

        snaps = f['snapshots']
        for key in sorted(snaps.keys()):
            g = snaps[key]
            times.append(g.attrs['time'])

            # Try to load BSSN fields (different possible locations)
            if 'bssn/lapse' in g:
                lapse_list.append(g['bssn/lapse'][:])
                K_list.append(g['bssn/K'][:])
                phi_list.append(g['bssn/phi'][:])
                # Load hrr if available (note: saved as 'hrr' not 'h_rr')
                if 'bssn/hrr' in g:
                    h_rr_list.append(g['bssn/hrr'][:])
                else:
                    h_rr_list.append(None)
                # Load shiftr if available
                if 'bssn/shiftr' in g:
                    shiftr_list.append(g['bssn/shiftr'][:])
                else:
                    shiftr_list.append(None)
            elif 'lapse' in g:
                lapse_list.append(g['lapse'][:])
                K_list.append(g['K'][:])
                phi_list.append(g['phi'][:])
                # Load hrr if available
                if 'hrr' in g:
                    h_rr_list.append(g['hrr'][:])
                else:
                    h_rr_list.append(None)
                # Load shiftr if available
                if 'shiftr' in g:
                    shiftr_list.append(g['shiftr'][:])
                else:
                    shiftr_list.append(None)
            else:
                # BSSN fields not found, return None
                return None, None, None, None, None, None, None, None, None, None, None

            # Load constraints if available
            if 'constraints/Ham' in g:
                Ham_list.append(g['constraints/Ham'][:])
                Mom_r_list.append(g['constraints/Mom_r'][:])
            else:
                # Constraints not found, append None
                Ham_list.append(None)
                Mom_r_list.append(None)

            # Load hydro primitives if available
            if 'primitives/rho0' in g:
                rho_list.append(g['primitives/rho0'][:])
                p_list.append(g['primitives/p'][:])
            else:
                # Primitives not found, append None
                rho_list.append(None)
                p_list.append(None)

    return np.array(times), r, lapse_list, K_list, phi_list, h_rr_list, shiftr_list, Ham_list, Mom_r_list, rho_list, p_list


def plot_hydro_profiles(t_snap, r, rho_list, p_list, output_path=None, title=None):
    """
    Plot density and pressure profiles at multiple time snapshots.

    Args:
        t_snap: Array of snapshot times
        r: Radial grid
        rho_list: List of density profiles
        p_list: List of pressure profiles
        output_path: Path to save the plot
        title: Title for the plot

    Returns:
        fig: Matplotlib figure
    """
    # Select 5 snapshots: evenly distributed
    n_snapshots = 5
    indices = np.linspace(0, len(t_snap) - 1, n_snapshots, dtype=int)

    # Colors for 5 time steps: blue -> cyan -> green -> orange -> red
    time_colors = ['#1f77b4', '#17becf', '#2ca02c', '#ff7f0e', '#d62728']
    time_labels = ['t₀', 't₁', 't₂', 't₃', 't₄']

    # Create figure with 1 row x 2 columns
    fig, (ax_rho, ax_p) = plt.subplots(1, 2, figsize=(14, 6))

    # Filter to physical domain (r >= 0)
    r_mask = r >= 0
    r_phys = r[r_mask]

    print(f"\n  Plotting hydro profiles at 5 time steps:")
    for idx, color, time_label in zip(indices, time_colors, time_labels):
        t_val = t_snap[idx]
        t_val_ms = t_val * M_SUN_SECONDS * 1e3

        print(f"    {time_label}: t={t_val:.1f} M_sun ({t_val_ms:.2f} ms)")

        # Density profile (use log scale)
        rho = rho_list[idx][r_mask]
        ax_rho.plot(r_phys, rho, color=color, linewidth=1.5,
                       label=f'{time_label}: {t_val_ms:.2f} ms', alpha=0.85)
        ax_rho.set_xlim(0.0, 10)
        # Pressure profile (use log scale)
        p = p_list[idx][r_mask]
        ax_p.plot(r_phys, p, color=color, linewidth=1.5,
                     label=f'{time_label}: {t_val_ms:.2f} ms', alpha=0.85)
        ax_p.set_xlim(0.0, 10)

    # Configure density plot
    ax_rho.set_xlabel(r'$r$ [code units]', fontsize=12)
    ax_rho.set_ylabel(r'Rest-mass density $\rho_0$', fontsize=12)
    ax_rho.set_title('(a) Density Profile', fontsize=13, fontweight='bold')
    ax_rho.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax_rho.grid(alpha=0.3)

    # Configure pressure plot
    ax_p.set_xlabel(r'$r$ [code units]', fontsize=12)
    ax_p.set_ylabel(r'Pressure $p$', fontsize=12)
    ax_p.set_title('(b) Pressure Profile', fontsize=13, fontweight='bold')
    ax_p.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax_p.grid(alpha=0.3)

    if title:
        fig.suptitle(f'{title} - Hydro Profiles', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved hydro profiles: {output_path}")

    plt.show()

    return fig


def make_bssn_animation(folder_path, output_path=None, fps=10, dpi=100):
    """
    Create animation showing the evolution of lapse, density, K, and conformal factor.

    Layout:
        (a) Lapse       | (b) Density profile (linear)
        (c) K           | (d) Conformal factor phi

    Args:
        folder_path: Path to data folder containing snapshots
        output_path: Path to save the video (MP4 format)
        fps: Frames per second for the video
        dpi: Resolution of the video

    Returns:
        Animation object
    """
    # Load BSSN snapshots
    print(f"\n  Loading BSSN snapshots for animation...")
    t_snap, r, lapse_list, K_list, phi_list, h_rr_list, shiftr_list, Ham_list, Mom_r_list, rho_list, p_list = load_bssn_snapshots(folder_path)

    if t_snap is None:
        print("  Error: No BSSN snapshots found")
        return None

    print(f"  Loaded {len(t_snap)} snapshots")
    print(f"  Time range: {t_snap[0]:.1f} to {t_snap[-1]:.1f} M_sun")
    print(f"  Time range: {t_snap[0]*M_SUN_SECONDS*1e3:.2f} to {t_snap[-1]*M_SUN_SECONDS*1e3:.2f} ms")

    rho_available = rho_list[0] is not None
    if not rho_available:
        print("  Warning: Density profiles not available in snapshots")

    # Filter to physical domain (r >= 0)
    r_mask = r >= 0
    r_phys = r[r_mask]

    # x-axis limit
    x_max = 10.0

    # Get initial values for reference
    lapse_init = lapse_list[0][r_mask]
    phi_init = phi_list[0][r_mask]
    K_init = K_list[0][r_mask]
    rho_init = rho_list[0][r_mask] if rho_available else None

    # Determine y-axis limits (with some padding), restricted to r <= x_max
    r_plot_mask = r_phys <= x_max

    lapse_min = min([np.min(lapse[r_mask][r_plot_mask]) for lapse in lapse_list])
    lapse_max = max([np.max(lapse[r_mask][r_plot_mask]) for lapse in lapse_list])
    lapse_margin = 0.1 * (lapse_max - lapse_min)

    phi_min = min([np.min(phi[r_mask][r_plot_mask]) for phi in phi_list])
    phi_max = max([np.max(phi[r_mask][r_plot_mask]) for phi in phi_list])
    phi_margin = 0.1 * (phi_max - phi_min) if phi_max > phi_min else 0.1

    K_min = min([np.min(K[r_mask][r_plot_mask]) for K in K_list])
    K_max = max([np.max(K[r_mask][r_plot_mask]) for K in K_list])
    K_margin = 0.1 * (K_max - K_min) if K_max > K_min else 1e-6

    if rho_available:
        rho_min = min([np.min(rho[r_mask][r_plot_mask]) for rho in rho_list if rho is not None])
        rho_max = max([np.max(rho[r_mask][r_plot_mask]) for rho in rho_list if rho is not None])
        rho_margin = 0.1 * (rho_max - rho_min) if rho_max > rho_min else 1e-6

    # Create figure: (a) Lapse | (b) Density, (c) K | (d) Phi
    fig, ((ax_lapse, ax_rho), (ax_K, ax_phi)) = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Lapse
    line_lapse, = ax_lapse.plot([], [], 'b-', linewidth=2, label='Current')
    ax_lapse.plot(r_phys, lapse_init, 'k--', linewidth=1, alpha=0.5, label='Initial')
    ax_lapse.axhline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # (b) Density (linear scale)
    if rho_available:
        line_rho, = ax_rho.plot([], [], 'r-', linewidth=2, label='Current')
        ax_rho.plot(r_phys, rho_init, 'k--', linewidth=1, alpha=0.5, label='Initial')
    else:
        line_rho = None
        ax_rho.text(0.5, 0.5, 'Density not available', ha='center', va='center',
                    transform=ax_rho.transAxes, fontsize=14)

    # (c) K
    line_K, = ax_K.plot([], [], 'g-', linewidth=2, label='Current')
    ax_K.plot(r_phys, K_init, 'k--', linewidth=1, alpha=0.5, label='Initial')
    ax_K.axhline(0.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # (d) Phi
    line_phi, = ax_phi.plot([], [], 'm-', linewidth=2, label='Current')
    ax_phi.plot(r_phys, phi_init, 'k--', linewidth=1, alpha=0.5, label='Initial')
    ax_phi.axhline(0.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Configure lapse plot
    ax_lapse.set_xlim(0, x_max)
    ax_lapse.set_ylim(lapse_min - lapse_margin, lapse_max + lapse_margin)
    ax_lapse.set_xlabel(r'$r$ [code units]', fontsize=11)
    ax_lapse.set_ylabel(r'Lapse $\alpha$', fontsize=11)
    ax_lapse.set_title('(a) Lapse Function', fontsize=12, fontweight='bold')
    ax_lapse.legend(fontsize=9, loc='best')
    ax_lapse.grid(alpha=0.3)

    # Configure density plot
    if rho_available:
        ax_rho.set_xlim(0, x_max)
        ax_rho.set_ylim(rho_min - rho_margin, rho_max + rho_margin)
    ax_rho.set_xlabel(r'$r$ [code units]', fontsize=11)
    ax_rho.set_ylabel(r'Rest-mass density $\rho_0$', fontsize=11)
    ax_rho.set_title('(b) Density Profile', fontsize=12, fontweight='bold')
    ax_rho.legend(fontsize=9, loc='best')
    ax_rho.grid(alpha=0.3)

    # Configure K plot
    ax_K.set_xlim(0, x_max)
    ax_K.set_ylim(K_min - K_margin, K_max + K_margin)
    ax_K.set_xlabel(r'$r$ [code units]', fontsize=11)
    ax_K.set_ylabel(r'Trace of Extrinsic Curvature $K$', fontsize=11)
    ax_K.set_title('(c) Extrinsic Curvature', fontsize=12, fontweight='bold')
    ax_K.legend(fontsize=9, loc='best')
    ax_K.grid(alpha=0.3)

    # Configure phi plot
    ax_phi.set_xlim(0, x_max)
    ax_phi.set_ylim(phi_min - phi_margin, phi_max + phi_margin)
    ax_phi.set_xlabel(r'$r$ [code units]', fontsize=11)
    ax_phi.set_ylabel(r'Conformal Factor $\phi$', fontsize=11)
    ax_phi.set_title('(d) Conformal Factor', fontsize=12, fontweight='bold')
    ax_phi.legend(fontsize=9, loc='best')
    ax_phi.grid(alpha=0.3)

    # Time text
    time_text = fig.text(0.5, 0.96, '', ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # Animation update function
    def update(frame):
        t_val = t_snap[frame]
        t_val_ms = t_val * M_SUN_SECONDS * 1e3

        # Update lapse
        lapse = lapse_list[frame][r_mask]
        line_lapse.set_data(r_phys, lapse)

        # Update density
        if line_rho is not None and rho_list[frame] is not None:
            rho = rho_list[frame][r_mask]
            line_rho.set_data(r_phys, rho)

        # Update K
        K = K_list[frame][r_mask]
        line_K.set_data(r_phys, K)

        # Update phi
        phi = phi_list[frame][r_mask]
        line_phi.set_data(r_phys, phi)

        # Update time text
        time_text.set_text(f'Time: {t_val:.1f} M$_\\odot$ ({t_val_ms:.2f} ms) — Frame {frame+1}/{len(t_snap)}')

        artists = [line_lapse, line_K, line_phi, time_text]
        if line_rho is not None:
            artists.insert(1, line_rho)
        return tuple(artists)

    # Create animation
    print(f"\n  Creating animation with {len(t_snap)} frames at {fps} fps...")
    anim = animation.FuncAnimation(fig, update, frames=len(t_snap),
                                  interval=1000/fps, blit=True, repeat=True)

    # Save animation
    if output_path:
        print(f"  Saving animation to: {output_path}")
        print(f"  This may take a while...")

        # Use FFmpeg writer
        writer = animation.FFMpegWriter(fps=fps, bitrate=5000,
                                       metadata={'artist': 'Engrenage TOV Evolution'})

        try:
            anim.save(output_path, writer=writer, dpi=dpi)
            print(f"  Successfully saved: {output_path}")
        except Exception as e:
            print(f"  Error saving animation: {e}")
            print(f"  Note: FFmpeg must be installed to save videos")
            print(f"  Try: sudo apt-get install ffmpeg")

    return anim


def plot_migration(data, output_path=None, title=None, rho_max_stable=None, folder_path=None):
    """Plot the migration of central density from unstable to stable branch.

    Args:
        data: Dictionary with 't', 'rho_central', and optionally 'mass'
        output_path: Path to save the plot
        title: Title for the plot
        rho_max_stable: Central density of maximum-mass stable model (for reference line)
        folder_path: Path to data folder (for loading BSSN snapshots)
    """
    t = data['t']
    rho_c = data['rho_central']
    mass = data.get('mass', None)

    # Convert time to milliseconds
    t_ms = t * M_SUN_SECONDS * 1e3

    # Normalize density to initial value
    rho_c_0 = rho_c[0]
    rho_normalized = rho_c / rho_c_0

    # Try to load BSSN snapshots
    bssn_data_available = False
    constraints_available = False
    hydro_profiles_available = False
    if folder_path is not None:
        t_snap, r, lapse_list, K_list, phi_list, h_rr_list, shiftr_list, Ham_list, Mom_r_list, rho_list, p_list = load_bssn_snapshots(folder_path)
        if t_snap is not None:
            bssn_data_available = True
            print(f"  Loaded {len(t_snap)} BSSN snapshots")
            # Check if constraints are available
            if Ham_list[0] is not None:
                constraints_available = True
                print(f"  Constraints data available")
            # Check if hydro primitives are available
            if rho_list[0] is not None:
                hydro_profiles_available = True
                print(f"  Hydro primitives data available")

    # Create figure with appropriate layout
    if bssn_data_available:
        # 3 rows x 2 columns: Density/BSSN fields/Constraints
        fig = plt.figure(figsize=(14, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.12, wspace=0.25)

        # Layout:
        # Row 0: Density, Lapse
        # Row 1: K, Phi
        # Row 2: Hamiltonian constraint, Momentum constraint
        ax1 = fig.add_subplot(gs[0, 0])      # Density
        ax_lapse = fig.add_subplot(gs[0, 1])  # Lapse
        ax_K = fig.add_subplot(gs[1, 0])      # K
        ax_phi = fig.add_subplot(gs[1, 1])    # Phi
        ax_Ham = fig.add_subplot(gs[2, 0])    # Hamiltonian constraint
        ax_Mom = fig.add_subplot(gs[2, 1])    # Momentum constraint
        ax2 = None  # No mass plot in BSSN layout
    else:
        # Original layout: 1 row with 2 columns
        fig, axes = plt.subplots(1, 2 if mass is not None else 1, figsize=(14 if mass is not None else 8, 5))
        if mass is not None:
            ax1, ax2 = axes
        else:
            ax1 = axes
            ax2 = None

    # Left: Normalized central density evolution
    ax1.plot(t_ms, rho_normalized, 'k-', linewidth=1.5, label=r'$\rho_c/\rho_{c,0}$')

    # Add reference line for maximum-mass stable model if provided
    if rho_max_stable is not None:
        rho_stable_normalized = rho_max_stable / rho_c_0
        ax1.axhline(rho_stable_normalized, color='red', linestyle='--', linewidth=1.5,
                   label=f'Max-mass stable ({rho_stable_normalized:.3f})')

    ax1.set_xlabel(r'$t$ [ms]', fontsize=11)
    ax1.set_ylabel(r'$\rho_c/\rho_{c,0}$', fontsize=11)
    ax1.set_title('(a) Central Density Evolution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # Add text with initial and final density
    textstr = f'Initial: $\\rho_{{c,0}} = {rho_c_0:.4e}$\n'
    textstr += f'Final: $\\rho_c = {rho_c[-1]:.4e}$\n'
    textstr += f'Ratio: $\\rho_c/\\rho_{{c,0}} = {rho_normalized[-1]:.3f}$'
    ax1.text(0.98, 0.97, textstr, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Right: Mass evolution (if available)
    if mass is not None:
        M_0 = mass[0]
        ax2.plot(t_ms, mass, 'b-', linewidth=1.5)
        ax2.set_xlabel(r'$t$ [ms]', fontsize=11)
        ax2.set_ylabel(r'$M$ [$M_\odot$]', fontsize=11)
        ax2.set_title('(b) Gravitational Mass', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add text with initial and final mass
        textstr = f'Initial: $M_0 = {M_0:.4f}$ $M_\\odot$\n'
        textstr += f'Final: $M = {mass[-1]:.4f}$ $M_\\odot$\n'
        textstr += f'$\\Delta M/M_0 = {100*(mass[-1]-M_0)/M_0:.2f}\\%$'
        ax2.text(0.98, 0.97, textstr, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Plot BSSN fields if available
    if bssn_data_available:
        # Select 5 snapshots: evenly distributed
        n_snapshots = 5
        indices = np.linspace(0, len(t_snap) - 1, n_snapshots, dtype=int)

        # Colors for 5 time steps: blue -> cyan -> green -> orange -> red
        time_colors = ['#1f77b4', '#17becf', '#2ca02c', '#ff7f0e', '#d62728']
        time_labels = ['t₀', 't₁', 't₂', 't₃', 't₄']

        print(f"  Plotting BSSN fields at 5 time steps:")
        for i, idx in enumerate(indices):
            print(f"    {time_labels[i]}: t={t_snap[idx]:.1f} M_sun ({t_snap[idx]*M_SUN_SECONDS*1e3:.2f} ms)")

        # Filter to physical domain (r >= 0)
        r_mask = r >= 0
        r_phys = r[r_mask]

        for i, (idx, color, time_label) in enumerate(zip(indices, time_colors, time_labels)):
            t_val = t_snap[idx]
            t_val_ms = t_val * M_SUN_SECONDS * 1e3

            # Lapse
            lapse = lapse_list[idx][r_mask]
            ax_lapse.plot(r_phys, lapse, color=color, linewidth=1.5,
                         label=f'{time_label}: {t_val_ms:.2f} ms', alpha=0.85)

            # K (trace of extrinsic curvature)
            K = K_list[idx][r_mask]
            ax_K.plot(r_phys, K, color=color, linewidth=1.5,
                     label=f'{time_label}: {t_val_ms:.2f} ms', alpha=0.85)

            # Conformal factor phi
            phi = phi_list[idx][r_mask]
            ax_phi.plot(r_phys, phi, color=color, linewidth=1.5,
                       label=f'{time_label}: {t_val_ms:.2f} ms', alpha=0.85)

        # Configure lapse plot
        ax_lapse.set_xlabel(r'$r$ [code units]', fontsize=11)
        ax_lapse.set_ylabel(r'Lapse $\alpha$', fontsize=11)
        ax_lapse.set_title('(b) Lapse Function', fontsize=12, fontweight='bold')
        ax_lapse.legend(fontsize=8, loc='lower right', framealpha=0.9)
        ax_lapse.grid(alpha=0.3)
        ax_lapse.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        # Configure K plot
        ax_K.set_xlabel(r'$r$ [code units]', fontsize=11)
        ax_K.set_ylabel(r'$K$ (Trace)', fontsize=11)
        ax_K.set_title('(c) Extrinsic Curvature', fontsize=12, fontweight='bold')
        ax_K.legend(fontsize=8, loc='best', framealpha=0.9)
        ax_K.grid(alpha=0.3)
        ax_K.axhline(0.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        # Configure phi plot
        ax_phi.set_xlabel(r'$r$ [code units]', fontsize=11)
        ax_phi.set_ylabel(r'$\phi$', fontsize=11)
        ax_phi.set_title('(d) Conformal Factor', fontsize=12, fontweight='bold')
        ax_phi.legend(fontsize=8, loc='upper right', framealpha=0.9)
        ax_phi.grid(alpha=0.3)
        ax_phi.axhline(0.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        # Plot constraint violations if available
        if constraints_available:
            print(f"  Plotting constraint violations at 5 time steps:")

            for i, (idx, color, time_label) in enumerate(zip(indices, time_colors, time_labels)):
                t_val = t_snap[idx]
                t_val_ms = t_val * M_SUN_SECONDS * 1e3

                # Hamiltonian constraint (take log10 of absolute value)
                Ham = Ham_list[idx][r_mask]
                log_Ham = np.log10(np.abs(Ham) + 1e-20)
                ax_Ham.plot(r_phys, log_Ham, color=color, linewidth=1.5,
                           label=f'{time_label}: {t_val_ms:.2f} ms', alpha=0.85)

                # Momentum constraint (take log10 of absolute value)
                Mom_r = Mom_r_list[idx][r_mask]
                log_Mom = np.log10(np.abs(Mom_r) + 1e-20)
                ax_Mom.plot(r_phys, log_Mom, color=color, linewidth=1.5,
                           label=f'{time_label}: {t_val_ms:.2f} ms', alpha=0.85)

            # Configure Hamiltonian constraint plot
            ax_Ham.set_xlabel(r'$r$ [code units]', fontsize=11)
            ax_Ham.set_ylabel(r'$\log_{10}|\mathcal{H}|$', fontsize=11)
            ax_Ham.set_title('(e) Hamiltonian Constraint Violation', fontsize=12, fontweight='bold')
            ax_Ham.legend(fontsize=8, loc='best', framealpha=0.9)
            ax_Ham.grid(alpha=0.3)

            # Configure Momentum constraint plot
            ax_Mom.set_xlabel(r'$r$ [code units]', fontsize=11)
            ax_Mom.set_ylabel(r'$\log_{10}|\mathcal{M}_r|$', fontsize=11)
            ax_Mom.set_title('(f) Momentum Constraint Violation', fontsize=12, fontweight='bold')
            ax_Mom.legend(fontsize=8, loc='best', framealpha=0.9)
            ax_Mom.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.0)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()

    # Plot hydro profiles (density and pressure) in a separate figure if available
    if hydro_profiles_available:
        fig_hydro = plot_hydro_profiles(
            t_snap, r, rho_list, p_list,
            output_path=output_path.replace('.png', '_hydro.png') if output_path else None,
            title=title
        )
        return fig, fig_hydro

    return fig


def plot_phase_diagram(data, output_path=None, title=None):
    """Plot phase diagram: M vs rho_c showing the migration path.

    Args:
        data: Dictionary with 'rho_central' and 'mass'
        output_path: Path to save the plot
        title: Title for the plot
    """
    rho_c = data['rho_central']
    mass = data.get('mass', None)

    if mass is None:
        print("Warning: Mass data not available, cannot create phase diagram")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot migration path with color gradient (time evolution)
    scatter = ax.scatter(rho_c, mass, c=np.arange(len(rho_c)),
                        cmap='viridis', s=10, alpha=0.7)

    # Mark initial and final points
    ax.plot(rho_c[0], mass[0], 'r*', markersize=20,
           markeredgecolor='darkred', markeredgewidth=1.5, label='Initial', zorder=5)
    ax.plot(rho_c[-1], mass[-1], 'go', markersize=12,
           markeredgecolor='darkgreen', markeredgewidth=1.5, label='Final', zorder=5)

    # Add arrow showing direction
    mid_idx = len(rho_c) // 4
    ax.annotate('', xy=(rho_c[mid_idx+50], mass[mid_idx+50]),
               xytext=(rho_c[mid_idx], mass[mid_idx]),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    ax.set_xlabel(r'$\rho_c$ [code units]', fontsize=13)
    ax.set_ylabel(r'$M$ [$M_\odot$]', fontsize=13)
    ax.set_title('Phase Diagram: Migration Path', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time step', fontsize=11)

    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold', y=0.96)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()

    return fig


def analyze_oscillation_period(data):
    """Analyze the oscillation period from density peaks."""
    t = data['t']
    rho_c = data['rho_central']

    # Convert time to milliseconds
    t_ms = t * M_SUN_SECONDS * 1e3

    # Find peaks in density
    from scipy import signal
    peaks, _ = signal.find_peaks(rho_c, prominence=0.05*rho_c[0])

    if len(peaks) < 2:
        print("Not enough peaks to determine oscillation period")
        return None

    # Calculate periods between consecutive peaks
    peak_times = t_ms[peaks]
    periods = np.diff(peak_times)

    print(f"\nOscillation Analysis:")
    print(f"  Number of peaks detected: {len(peaks)}")
    print(f"  Peak times [ms]: {peak_times}")
    print(f"  Periods [ms]: {periods}")
    print(f"  Mean period: {np.mean(periods):.3f} ms")
    print(f"  Std period: {np.std(periods):.3f} ms")

    return {
        'peak_times': peak_times,
        'periods': periods,
        'mean_period': np.mean(periods),
        'std_period': np.std(periods)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Migration Test: Plot unstable to stable branch evolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python migration.py --data-dir path/to/data
  python migration.py --data-dir path/to/data --rho-max-stable 5.5e-3
  python migration.py --data-dir path/to/data --output-dir plots --analyze-period
  python migration.py --data-dir path/to/data --make-video --fps 15
'''
    )
    parser.add_argument('--data-dir', required=True,
                       help='Path to the evolution data folder')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for plots. Default: same as data-dir')
    parser.add_argument('--rho-max-stable', type=float, default=None,
                       help='Central density of maximum-mass stable model (for reference line)')
    parser.add_argument('--analyze-period', action='store_true',
                       help='Analyze oscillation periods from density peaks')
    parser.add_argument('--phase-diagram', action='store_true',
                       help='Generate phase diagram (M vs rho_c)')
    parser.add_argument('--make-video', action='store_true',
                       help='Generate animation video of lapse and conformal factor evolution')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for video (default: 10)')
    parser.add_argument('--video-dpi', type=int, default=100,
                       help='DPI resolution for video (default: 100)')
    args = parser.parse_args()

    data_dir = args.data_dir

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = data_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from: {data_dir}")
    data = load_data_from_folder(data_dir)

    if data is None:
        print("Error: No evolution data found in the specified folder")
        sys.exit(1)

    folder_name = os.path.basename(data_dir)

    print(f"\nData summary:")
    print(f"  Time range: {data['t'][0]:.2f} to {data['t'][-1]:.2f} M_sun")
    print(f"  Time range: {data['t'][0]*M_SUN_SECONDS*1e3:.3f} to {data['t'][-1]*M_SUN_SECONDS*1e3:.3f} ms")
    print(f"  Number of points: {len(data['t'])}")
    print(f"  Initial central density: {data['rho_central'][0]:.4e}")
    print(f"  Final central density: {data['rho_central'][-1]:.4e}")
    print(f"  Ratio (final/initial): {data['rho_central'][-1]/data['rho_central'][0]:.3f}")

    if data['mass'] is not None:
        print(f"  Initial mass: {data['mass'][0]:.4f} M_sun")
        print(f"  Final mass: {data['mass'][-1]:.4f} M_sun")
        print(f"  Mass conservation: {100*(data['mass'][-1]-data['mass'][0])/data['mass'][0]:.2f}%")

    # Plot migration
    output_path = os.path.join(output_dir, f'migration_{folder_name}.png')
    plot_migration(data, output_path=output_path,
                  title=f"Migration Test: {folder_name}",
                  rho_max_stable=args.rho_max_stable,
                  folder_path=data_dir)

    # Plot phase diagram if requested
    if args.phase_diagram and data['mass'] is not None:
        output_path_phase = os.path.join(output_dir, f'migration_phase_{folder_name}.png')
        plot_phase_diagram(data, output_path=output_path_phase,
                          title=f"Phase Diagram: {folder_name}")

    # Analyze oscillation period if requested
    if args.analyze_period:
        analyze_oscillation_period(data)

    # Generate animation video if requested
    if args.make_video:
        print(f"\n{'='*60}")
        print(f"GENERATING BSSN ANIMATION VIDEO")
        print(f"{'='*60}")
        video_path = os.path.join(output_dir, f'bssn_evolution_{folder_name}.mp4')
        anim = make_bssn_animation(data_dir, output_path=video_path,
                                   fps=args.fps, dpi=args.video_dpi)
        if anim is not None:
            print(f"\n  Video generation complete!")
            # Optionally show the animation (comment out if running on headless server)
            # plt.show()

    print(f"\nDone!")


if __name__ == "__main__":
    main()
