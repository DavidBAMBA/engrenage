#!/usr/bin/env python3
"""
Collapse Test: Gravitational Collapse to Black Hole

Plots the evolution of a TOV star perturbed to collapse, following
Font et al. 2002, Fig. 11.

Generates:
  1. Static summary plot (central density, BSSN fields, constraints)
  2. Apparent horizon diagnostics (expansion, AH radius, BH mass)
  3. Animation video with 3 vertically-stacked panels:
     (top)    Lapse α
     (middle) Physical metric component g_rr = e^{4φ}(1 + h_rr)
     (bottom) Normalized density ρ/ρ_{c,0}

Reference: Font et al. 2002, Phys. Rev. D 65, 084024, Fig. 11
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
import json
import os
import sys
import argparse

# Add repository root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, repo_root)

# Reuse data-loading utilities from migration.py
from migration import (
    M_SUN_SECONDS,
    load_data_from_folder,
    load_bssn_snapshots,
    plot_migration,
)


# =============================================================================
# State reconstruction from HDF5 snapshots
# =============================================================================

def load_full_states_from_snapshots(folder_path):
    """
    Load full state vectors (15, N) from HDF5 snapshots for AH finder.

    Reconstructs the complete state array from individually saved BSSN
    and conservative variables.

    Returns:
        times: np.array of snapshot times (n_snapshots,)
        r: radial grid array (N,)
        states_flat: np.array of shape (n_snapshots, 15*N)
        N: number of grid points
    """
    possible_names = ['tov_snapshots_dyn_jax.h5', 'tov_snapshots_dyn.h5']

    h5_file = None
    for filename in possible_names:
        filepath = os.path.join(folder_path, filename)
        if os.path.exists(filepath):
            h5_file = filepath
            break

    if h5_file is None:
        return None, None, None, None

    bssn_order = ['phi', 'hrr', 'htt', 'hpp', 'K', 'arr', 'att', 'app',
                  'lambdar', 'shiftr', 'br', 'lapse']

    times = []
    states = []

    with h5py.File(h5_file, 'r') as f:
        r = f['grid/r'][:]
        N = int(f['grid/N'][()])

        snaps = f['snapshots']
        for key in sorted(snaps.keys()):
            g = snaps[key]
            times.append(float(g.attrs['time']))

            state = np.zeros((15, N), dtype=np.float64)

            # BSSN variables (idx 0-11)
            for idx, var_name in enumerate(bssn_order):
                if f'bssn/{var_name}' in g:
                    state[idx, :] = g[f'bssn/{var_name}'][:]
                elif var_name in g:
                    state[idx, :] = g[var_name][:]

            # Conservative hydro variables (idx 12-14)
            if 'conservatives/D' in g:
                state[12, :] = g['conservatives/D'][:]
                state[13, :] = g['conservatives/Sr'][:]
                state[14, :] = g['conservatives/tau'][:]

            states.append(state.reshape(-1))

    return np.array(times), r, np.array(states), N


def setup_grid_and_hydro(folder_path):
    """
    Reconstruct grid, background, and hydro objects from saved metadata.

    Reads tov_metadata_dyn_jax.json (or _dyn.json) for parameters.

    Returns:
        grid: Grid object
        background: FlatSphericalBackground
        hydro: PerfectFluid
        metadata: dict with full metadata
    """
    from source.core.grid import Grid
    from source.core.spacing import LinearSpacing
    from source.core.statevector import StateVector
    from source.backgrounds.sphericalbackground import FlatSphericalBackground
    from source.matter.hydro.perfect_fluid import PerfectFluid
    from source.matter.hydro.eos import IdealGasEOS
    from source.matter.hydro.reconstruction import create_reconstruction
    from source.matter.hydro.riemann import HLLRiemannSolver
    from source.matter.hydro.atmosphere import AtmosphereParams

    # Find metadata file
    possible_names = ['tov_metadata_dyn_jax.json', 'tov_metadata_dyn.json',
                      'tov_metadata_cow_jax.json', 'tov_metadata_cow.json']
    metadata = None
    for name in possible_names:
        path = os.path.join(folder_path, name)
        if os.path.exists(path):
            with open(path) as f:
                metadata = json.load(f)
            break

    if metadata is None:
        raise FileNotFoundError(f"No metadata file found in {folder_path}")

    # Extract parameters
    config = metadata.get('configuration', {})
    tov_info = metadata.get('tov_solution', {})
    atm_info = metadata.get('atmosphere', {})
    num_methods = metadata.get('numerical_methods', {})

    num_points = config.get('num_points', metadata.get('simulation', {}).get('grid_N', 2000))
    r_max = config.get('r_max', metadata.get('simulation', {}).get('grid_r_max', 100.0))
    K = tov_info.get('K', 100.0)
    Gamma = tov_info.get('Gamma', 2.0)
    rho_floor = atm_info.get('rho_floor', 1e-16)
    p_floor = atm_info.get('p_floor', K * rho_floor**Gamma)
    reconstructor = num_methods.get('reconstructor', 'mp5')
    solver_method = num_methods.get('solver_method', 'newton')

    spacing = LinearSpacing(num_points, r_max)
    eos = IdealGasEOS(gamma=Gamma)
    atmosphere = AtmosphereParams(rho_floor=rho_floor, p_floor=p_floor)
    recon = create_reconstruction(reconstructor)
    riemann = HLLRiemannSolver(atmosphere=atmosphere)

    hydro = PerfectFluid(
        eos=eos, spacetime_mode="dynamic", atmosphere=atmosphere,
        reconstructor=recon, riemann_solver=riemann, solver_method=solver_method
    )

    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    return grid, background, hydro, metadata


# =============================================================================
# Apparent Horizon diagnostics
# =============================================================================

def plot_ah_diagnostics(folder_path, output_path=None, title=None):
    """
    Compute and plot apparent horizon diagnostics from saved snapshots.

    4-panel figure:
      (0,0) Expansion ω(r) at selected times
      (0,1) AH radius vs time
      (1,0) BH mass vs time (with TOV mass reference)
      (1,1) ω(r) zoom near AH formation time
    """
    from source.bssn.ahfinder import get_horizon_diagnostics

    # Setup grid and hydro from metadata
    print("  Setting up grid and hydro from metadata...")
    grid, background, hydro, metadata = setup_grid_and_hydro(folder_path)

    # Load full states
    print("  Loading full state vectors from snapshots...")
    times, r, states_flat, _ = load_full_states_from_snapshots(folder_path)

    if times is None:
        print("  Error: No snapshot data found")
        return None

    print(f"  Computing AH diagnostics for {len(times)} snapshots...")
    omega, ah_radius, bh_mass = get_horizon_diagnostics(
        states_flat, times, grid, background, hydro)

    # Report results
    ah_mask = ah_radius > 0
    if not np.any(ah_mask):
        print("  No apparent horizon found in any snapshot")
        # Plot omega evolution to show it never crosses zero
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        n_profiles = min(8, len(times))
        indices = np.linspace(0, len(times)-1, n_profiles, dtype=int)
        for i in indices:
            t_ms = times[i] * M_SUN_SECONDS * 1e3
            ax.plot(r, omega[i], label=f't={t_ms:.3f} ms')
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax.set_xlabel('r [code units]', fontsize=12)
        ax.set_ylabel(r'Expansion $\omega$', fontsize=12)
        ax.set_xlim(0, 15)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(title or 'Expansion (no AH found)')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {output_path}")
        plt.close()
        return None

    first_ah_idx = np.argmax(ah_mask)
    t_ah_form = times[first_ah_idx]
    t_ah_ms = t_ah_form * M_SUN_SECONDS * 1e3
    print(f"  AH first appears at t={t_ah_form:.4e} ({t_ah_ms:.3f} ms)")
    print(f"  Final AH radius: {ah_radius[ah_mask][-1]:.4f}")
    print(f"  Final BH mass:   {bh_mass[ah_mask][-1]:.4f}")

    # --- 4-panel figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0): Expansion profiles at selected times
    ax = axes[0, 0]
    n_profiles = min(8, len(times))
    indices = np.linspace(0, len(times)-1, n_profiles, dtype=int)
    for i in indices:
        t_ms = times[i] * M_SUN_SECONDS * 1e3
        style = '-' if ah_radius[i] > 0 else '--'
        ax.plot(r, omega[i], style, label=f't={t_ms:.3f} ms', linewidth=1.2)
    ax.axhline(0, color='k', linestyle=':', linewidth=0.5)
    ax.set_xlabel('r [code units]', fontsize=11)
    ax.set_ylabel(r'Expansion $\omega$', fontsize=11)
    ax.set_xlim(0, 15)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title(r'(a) Expansion $\omega(r)$', fontsize=12, fontweight='bold')

    # (0,1): AH radius vs time
    ax = axes[0, 1]
    t_ah_ms_arr = times[ah_mask] * M_SUN_SECONDS * 1e3
    ax.plot(t_ah_ms_arr, ah_radius[ah_mask], 'b-o', markersize=3)
    ax.set_xlabel('t [ms]', fontsize=11)
    ax.set_ylabel('AH radius [code units]', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title('(b) Apparent Horizon Radius', fontsize=12, fontweight='bold')

    # (1,0): BH mass vs time
    ax = axes[1, 0]
    ax.plot(t_ah_ms_arr, bh_mass[ah_mask], 'r-o', markersize=3)
    tov_info = metadata.get('tov_solution', {})
    M_star = tov_info.get('M_star', None)
    if M_star is not None:
        ax.axhline(M_star, color='gray', linestyle='--',
                   label=f'$M_{{TOV}}={M_star:.4f}$')
        ax.legend(fontsize=10)
    ax.set_xlabel('t [ms]', fontsize=11)
    ax.set_ylabel('BH mass [code units]', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title('(c) Black Hole Mass', fontsize=12, fontweight='bold')

    # (1,1): omega zoom near AH formation
    ax = axes[1, 1]
    i_start = max(0, first_ah_idx - 2)
    i_end = min(len(times), first_ah_idx + 3)
    for i in range(i_start, i_end):
        t_ms = times[i] * M_SUN_SECONDS * 1e3
        style = '-' if ah_radius[i] > 0 else '--'
        lw = 2.0 if i == first_ah_idx else 1.2
        ax.plot(r, omega[i], style, linewidth=lw,
                label=f't={t_ms:.3f} ms' + (' (AH forms)' if i == first_ah_idx else ''))
    ax.axhline(0, color='k', linestyle=':', linewidth=0.5)
    ax.set_xlabel('r [code units]', fontsize=11)
    ax.set_ylabel(r'Expansion $\omega$', fontsize=11)
    ax.set_xlim(0, 10)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title(r'(d) $\omega(r)$ near AH Formation', fontsize=12, fontweight='bold')

    fig.suptitle(title or f'AH Diagnostics: {os.path.basename(folder_path)}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()

    return {'times': times, 'ah_radius': ah_radius, 'bh_mass': bh_mass, 'omega': omega}


def plot_horizon_mass(ah_data, metadata, output_path=None, title=None):
    """
    Standalone plot of Horizon Mass vs time (Font et al. 2002, Fig. 12).

    Single-panel plot showing BH mass growth after AH formation,
    with TOV mass reference line.

    Args:
        ah_data: dict returned by plot_ah_diagnostics() with keys
                 'times', 'ah_radius', 'bh_mass'
        metadata: dict with TOV solution info (for M_star reference)
        output_path: path to save the figure
        title: optional title
    """
    if ah_data is None:
        print("  No AH data available for horizon mass plot")
        return

    times = ah_data['times']
    ah_radius = ah_data['ah_radius']
    bh_mass = ah_data['bh_mass']

    ah_mask = ah_radius > 0
    if not np.any(ah_mask):
        print("  No apparent horizon found — skipping horizon mass plot")
        return

    t_ms = times[ah_mask] * M_SUN_SECONDS * 1e3
    mass = bh_mass[ah_mask]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(t_ms, mass, 'k-', linewidth=2)

    # TOV mass reference
    tov_info = metadata.get('tov_solution', {})
    M_star = tov_info.get('M_star', None)
    if M_star is not None:
        ax.axhline(M_star, color='gray', linestyle='--', linewidth=1.5,
                   label=f'$M_{{TOV}} = {M_star:.4f}$')
        ax.legend(fontsize=12, loc='lower right')

    ax.set_xlabel(r'$t$ [ms]', fontsize=14)
    ax.set_ylabel('Horizon Mass', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# Collapse animation (Font et al. 2002, Fig. 11)
# =============================================================================

def make_collapse_animation(folder_path, output_path=None, fps=10, dpi=100):
    """
    Create animation of gravitational collapse (Font et al. 2002, Fig. 11).

    Three vertically-stacked panels:
        (top)    Lapse α
        (middle) g_rr = e^{4φ}(1 + h_rr)   (physical radial metric)
        (bottom) ρ / ρ_{c,0}                 (normalized rest-mass density)

    Args:
        folder_path: Path to data folder containing snapshots
        output_path: Path to save the video (MP4 format)
        fps: Frames per second for the video
        dpi: Resolution of the video

    Returns:
        Animation object
    """
    print(f"\n  Loading BSSN snapshots for collapse animation...")
    t_snap, r, lapse_list, K_list, phi_list, h_rr_list, shiftr_list, \
        Ham_list, Mom_r_list, rho_list, p_list = load_bssn_snapshots(folder_path)

    if t_snap is None:
        print("  Error: No BSSN snapshots found")
        return None

    print(f"  Loaded {len(t_snap)} snapshots")
    print(f"  Time range: {t_snap[0]:.1f} to {t_snap[-1]:.1f} M_sun")
    print(f"  Time range: {t_snap[0]*M_SUN_SECONDS*1e3:.2f} to {t_snap[-1]*M_SUN_SECONDS*1e3:.2f} ms")

    rho_available = rho_list[0] is not None
    h_rr_available = h_rr_list[0] is not None

    if not rho_available:
        print("  Error: Density profiles not available in snapshots")
        return None
    if not h_rr_available:
        print("  Warning: h_rr not available, using g_rr = e^{4phi} (h_rr=0)")

    # Filter to physical domain (r >= 0)
    r_mask = r >= 0
    r_phys = r[r_mask]
    x_max = 10.0
    r_plot_mask = r_phys <= x_max

    # Initial central density for normalization
    rho_c0 = rho_list[0][r_mask][r_plot_mask][0]  # ρ at r~0
    print(f"  Initial central density: {rho_c0:.4e}")

    # Precompute g_rr = e^{4φ}(1 + h_rr) for all snapshots
    grr_list = []
    for i in range(len(t_snap)):
        phi = phi_list[i][r_mask]
        e4phi = np.exp(4.0 * phi)
        if h_rr_available and h_rr_list[i] is not None:
            h_rr = h_rr_list[i][r_mask]
            grr = e4phi * (1.0 + h_rr)
        else:
            grr = e4phi
        grr_list.append(grr)

    # Reference profiles (initial)
    lapse_init = lapse_list[0][r_mask]
    grr_init = grr_list[0]
    rho_init = rho_list[0][r_mask]
    rho_norm_init = rho_init / rho_c0

    # Y-axis limits (computed over r <= x_max)
    lapse_vals = [lapse[r_mask][r_plot_mask] for lapse in lapse_list]
    lapse_min = min(np.min(v) for v in lapse_vals)
    lapse_max = max(np.max(v) for v in lapse_vals)
    lapse_pad = 0.1 * max(lapse_max - lapse_min, 0.01)

    grr_vals = [g[r_plot_mask] for g in grr_list]
    grr_min = min(np.min(v) for v in grr_vals)
    grr_max = max(np.max(v) for v in grr_vals)
    grr_pad = 0.1 * max(grr_max - grr_min, 0.1)

    rho_norm_vals = [rho[r_mask][r_plot_mask] / rho_c0 for rho in rho_list if rho is not None]
    rho_norm_min = min(np.min(v) for v in rho_norm_vals)
    rho_norm_max = max(np.max(v) for v in rho_norm_vals)
    rho_norm_pad = 0.1 * max(rho_norm_max - rho_norm_min, 0.1)

    # --- Figure: 3 rows, 1 column (Font et al. style) ---
    fig, (ax_lapse, ax_grr, ax_rho) = plt.subplots(
        3, 1, figsize=(8, 12), sharex=True
    )

    # (top) Lapse
    line_lapse, = ax_lapse.plot([], [], 'k-', linewidth=2, label='Current')
    ax_lapse.plot(r_phys, lapse_init, 'k--', linewidth=1.5, alpha=0.4, label='Initial')

    ax_lapse.set_xlim(0, x_max)
    ax_lapse.set_ylim(lapse_min - lapse_pad, lapse_max + lapse_pad)
    ax_lapse.set_ylabel(r'$\alpha$', fontsize=14)
    ax_lapse.legend(fontsize=9, loc='upper right')
    ax_lapse.grid(alpha=0.3)

    # (middle) g_rr
    line_grr, = ax_grr.plot([], [], 'k-', linewidth=2, label='Current')
    ax_grr.plot(r_phys, grr_init, 'k--', linewidth=1.5, alpha=0.4, label='Initial')

    ax_grr.set_xlim(0, x_max)
    ax_grr.set_ylim(grr_min - grr_pad, grr_max + grr_pad)
    ax_grr.set_ylabel(r'$g_{rr}$', fontsize=14)
    ax_grr.legend(fontsize=9, loc='upper right')
    ax_grr.grid(alpha=0.3)

    # (bottom) ρ/ρ_{c,0}
    line_rho, = ax_rho.plot([], [], 'k-', linewidth=2, label='Current')
    ax_rho.plot(r_phys, rho_norm_init, 'k--', linewidth=1.5, alpha=0.4, label='Initial')

    ax_rho.set_xlim(0, x_max)
    ax_rho.set_ylim(rho_norm_min - rho_norm_pad, rho_norm_max + rho_norm_pad)
    ax_rho.set_xlabel(r'$r$ [code units]', fontsize=13)
    ax_rho.set_ylabel(r'$\rho / \rho_{c,0}$', fontsize=14)
    ax_rho.legend(fontsize=9, loc='upper right')
    ax_rho.grid(alpha=0.3)

    # Time text
    time_text = fig.text(0.5, 0.96, '', ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # Animation update
    def update(frame):
        t_val = t_snap[frame]
        t_val_ms = t_val * M_SUN_SECONDS * 1e3

        lapse = lapse_list[frame][r_mask]
        line_lapse.set_data(r_phys, lapse)

        grr = grr_list[frame]
        line_grr.set_data(r_phys, grr)

        rho_norm = rho_list[frame][r_mask] / rho_c0
        line_rho.set_data(r_phys, rho_norm)

        time_text.set_text(
            f'Time: {t_val:.1f} M$_\\odot$ ({t_val_ms:.3f} ms) '
            f'\u2014 Frame {frame+1}/{len(t_snap)}'
        )

        return line_lapse, line_grr, line_rho, time_text

    # Create animation
    print(f"\n  Creating animation with {len(t_snap)} frames at {fps} fps...")
    anim = animation.FuncAnimation(
        fig, update, frames=len(t_snap),
        interval=1000/fps, blit=True, repeat=True
    )

    # Save
    if output_path:
        print(f"  Saving animation to: {output_path}")
        print(f"  This may take a while...")
        writer = animation.FFMpegWriter(
            fps=fps, bitrate=5000,
            metadata={'artist': 'Engrenage TOV Collapse'}
        )
        try:
            anim.save(output_path, writer=writer, dpi=dpi)
            print(f"  Successfully saved: {output_path}")
        except Exception as e:
            print(f"  Error saving animation: {e}")
            print(f"  Note: FFmpeg must be installed to save videos")
            print(f"  Try: sudo apt-get install ffmpeg")

    return anim


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collapse Test: Gravitational collapse to black hole (Font et al. 2002)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python collapse.py --data-dir path/to/data
  python collapse.py --data-dir path/to/data --make-video --fps 15
  python collapse.py --data-dir path/to/data --make-video --output-dir plots
'''
    )
    parser.add_argument('--data-dir', required=True,
                        help='Path to the evolution data folder')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for plots. Default: same as data-dir')
    parser.add_argument('--rho-max-stable', type=float, default=None,
                        help='Central density of maximum-mass stable model (for reference line)')
    parser.add_argument('--make-video', action='store_true',
                        help='Generate collapse animation video')
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second for video (default: 5)')
    parser.add_argument('--video-dpi', type=int, default=100,
                        help='DPI resolution for video (default: 100)')
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    output_dir = args.output_dir if args.output_dir else data_dir
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

    # 1. Static summary plot (same as migration.py)
    output_path = os.path.join(output_dir, f'collapse_{folder_name}.png')
    plot_migration(
        data, output_path=output_path,
        title=f"Collapse Test: {folder_name}",
        rho_max_stable=args.rho_max_stable,
        folder_path=data_dir
    )

    # 2. Apparent Horizon diagnostics
    print(f"\n{'='*60}")
    print(f"APPARENT HORIZON DIAGNOSTICS")
    print(f"{'='*60}")
    ah_plot_path = os.path.join(output_dir, f'ah_diagnostics_{folder_name}.png')
    ah_data = None
    try:
        ah_data = plot_ah_diagnostics(
            data_dir, output_path=ah_plot_path,
            title=f"AH Diagnostics: {folder_name}"
        )
    except Exception as e:
        print(f"  AH diagnostics failed: {e}")
        import traceback
        traceback.print_exc()

    # 2b. Standalone Horizon Mass plot (Font et al. 2002, Fig. 12)
    if ah_data is not None:
        _, _, _, metadata = setup_grid_and_hydro(data_dir)
        hmass_path = os.path.join(output_dir, f'horizon_mass_{folder_name}.png')
        plot_horizon_mass(
            ah_data, metadata, output_path=hmass_path,
            title=f"Horizon Mass: {folder_name}"
        )

    # 3. Collapse animation video
    if args.make_video:
        print(f"\n{'='*60}")
        print(f"GENERATING COLLAPSE ANIMATION VIDEO")
        print(f"{'='*60}")
        video_path = os.path.join(output_dir, f'collapse_{folder_name}.mp4')
        anim = make_collapse_animation(
            data_dir, output_path=video_path,
            fps=args.fps, dpi=args.video_dpi
        )
        if anim is not None:
            print(f"\n  Video generation complete!")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
