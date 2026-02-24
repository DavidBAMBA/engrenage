#!/usr/bin/env python3
"""
Plot final density and pressure profiles for different resolutions.
Compares the surface behavior across resolutions.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
import os
import argparse

# Data folders for each resolution
FOLDERS = {
    'N=100': '/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_rmax20_TEST_kas_ideal_long_domain_lasttest/tov_star_rhoc1p28em03_N100_K100_G2_cow_mp5',
    'N=200': '/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_rmax20_TEST_kas_ideal_long_domain_lasttest/tov_star_rhoc1p28em03_N200_K100_G2_cow_mp5',
    'N=400': '/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_rmax20_TEST_kas_ideal_long_domain_lasttest/tov_star_rhoc1p28em03_N400_K100_G2_cow_mp5',
    'N=800': '/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_rmax20_TEST_kas_ideal_long_domain_lasttest/tov_star_rhoc1p28em03_N800_K100_G2_cow_mp5',
}

# Colors for each resolution (will be populated dynamically if needed)
COLORS = {
    'N=200': 'C0',
    'N=400': 'C1',
    'N=800': 'C2',
    'N=1600': 'C3',
}

def get_color(label, all_labels):
    """Get color for a resolution label, using default palette if not in COLORS."""
    if label in COLORS:
        return COLORS[label]
    # Generate color dynamically using matplotlib's color cycle
    idx = sorted(all_labels).index(label)
    return f'C{idx}'


def load_final_snapshot(folder):
    """Load the final snapshot from a simulation folder."""
    snapshot_file = os.path.join(folder, 'tov_snapshots_dyn_jax.h5')

    if not os.path.exists(snapshot_file):
        print(f"  Warning: {snapshot_file} not found")
        return None

    try:
        with h5py.File(snapshot_file, 'r') as f:
            # Get grid
            r = f['grid']['r'][:]
            N = int(f['grid']['N'][()])

            snaps = f['snapshots']

            # Try to list keys, but handle corrupted files by probing directly
            try:
                snap_keys = list(snaps.keys())
                snap_keys_sorted = sorted(snap_keys, key=lambda x: int(x.split('_')[1]))
                initial_key = snap_keys_sorted[0]
                final_key = snap_keys_sorted[-1]
            except RuntimeError:
                # File may be corrupted - try probing specific steps
                print(f"  Warning: Cannot list snapshots, probing directly...")
                initial_key = 'step_00000000'

                # Find the latest available snapshot
                final_key = None
                for step in [3000000, 2500000, 2400000, 2300000, 2200000, 2000000,
                             1500000, 1000000, 500000, 100000]:
                    key = f'step_{step:08d}'
                    try:
                        _ = snaps[key]['primitives']['rho0'][0]
                        final_key = key
                        break
                    except:
                        continue

                if final_key is None:
                    print(f"  Error: Could not find any valid snapshot")
                    return None

            # Extract step number for time estimation
            final_step = int(final_key.split('_')[1])

            # Read initial data
            init_snap = snaps[initial_key]['primitives']
            rho0_init = init_snap['rho0'][:]
            p_init = init_snap['p'][:]
            vr_init = init_snap['vr'][:]

            # Read final data
            final_snap = snaps[final_key]['primitives']
            rho0_final = final_snap['rho0'][:]
            p_final = final_snap['p'][:]
            vr_final = final_snap['vr'][:]

            # Get time from evolution file if available
            evol_file = os.path.join(folder, 'tov_evolution_cow.h5')
            t_final = None
            if os.path.exists(evol_file):
                with h5py.File(evol_file, 'r') as ef:
                    t = ef['time'][:]
                    t_final = t[-1]

            return {
                'r': r,
                'N': N,
                'rho0_init': rho0_init,
                'p_init': p_init,
                'vr_init': vr_init,
                'rho0_final': rho0_final,
                'p_final': p_final,
                'vr_final': vr_final,
                'final_step': final_step,
                't_final': t_final,
            }

    except Exception as e:
        print(f"  Error loading {folder}: {e}")
        return None


def extract_resolution_from_dirname(dirname):
    """Extract resolution number from directory name."""
    import re
    match = re.search(r'[Nn]r?[=_]?(\d+)', dirname)
    if match:
        return int(match.group(1))
    return None


def main():
    """Create comparison plots."""
    parser = argparse.ArgumentParser(
        description='Plot final density and pressure profiles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python plot_final_profiles.py                    # Use default folders
  python plot_final_profiles.py --data-dirs D1 D2  # Custom directories
  python plot_final_profiles.py --r-star 8.5       # Custom stellar radius
'''
    )
    parser.add_argument('--data-dirs', nargs='+', default=None,
                        help='List of data directories to analyze. Default: use FOLDERS')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for plots. Default: script_dir/plots')
    parser.add_argument('--r-star', type=float, default=8.125,
                        help='Stellar radius for reference line. Default: 8.125')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine output directory
    if args.output_dir:
        plots_dir = args.output_dir
    else:
        plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Determine data folders
    if args.data_dirs:
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

        if not folders_dict:
            print("No valid folders found. Check the paths provided via --data-dirs")
            return
    else:
        # Use default FOLDERS (backward compatibility)
        folders_dict = FOLDERS

    print("Loading data from all resolutions...")
    print("=" * 60)

    data = {}
    for name, folder in sorted(folders_dict.items(), key=lambda x: int(x[0].split('=')[1]) if '=' in x[0] else 0):
        print(f"\n{name}:")
        result = load_final_snapshot(folder)
        if result is not None:
            data[name] = result
            print(f"  N = {result['N']}")
            print(f"  Final step: {result['final_step']}")
            if result['t_final'] is not None:
                print(f"  t_final = {result['t_final']:.2f}")
            print(f"  rho_c (init)  = {result['rho0_init'][0]:.6e}")
            print(f"  rho_c (final) = {result['rho0_final'][0]:.6e}")

    if len(data) == 0:
        print("\nNo data loaded!")
        return

    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Stellar radius
    R_star = args.r_star

    # =========================================================================
    # Top left: Density - Linear scale
    # =========================================================================
    ax = axes[0, 0]
    for name in sorted(data.keys(), key=lambda x: int(x.split('=')[1]) if '=' in x else 0 if '=' in x else 0):
        d = data[name]
        r = d['r']
        # Only plot interior (r >= 0)
        mask = r >= 0
        ax.plot(r[mask], d['rho0_final'][mask], '-', color=get_color(name, data.keys()),
                label=f"{name} (t={d['t_final']:.0f})" if d['t_final'] else name,
                alpha=0.8, linewidth=1.5)

    ax.axvline(R_star, color='gray', linestyle='--', alpha=0.5, label=f'R_star={R_star:.2f}')
    ax.set_xlabel('r')
    ax.set_ylabel(r'$\rho_0$')
    ax.set_title('Final Density Profile (Linear)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 12)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Top right: Density - Log scale
    # =========================================================================
    ax = axes[0, 1]
    for name in sorted(data.keys(), key=lambda x: int(x.split('=')[1]) if '=' in x else 0):
        d = data[name]
        r = d['r']
        mask = r >= 0
        ax.semilogy(r[mask], d['rho0_final'][mask], '-', color=get_color(name, data.keys()),
                    label=name, alpha=0.8, linewidth=1.5)

    ax.axvline(R_star, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('r')
    ax.set_ylabel(r'$\rho_0$')
    ax.set_title('Final Density Profile (Log)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 12)
    ax.set_ylim(1e-12, 1e-2)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Bottom left: Pressure - Linear scale
    # =========================================================================
    ax = axes[1, 0]
    for name in sorted(data.keys(), key=lambda x: int(x.split('=')[1]) if '=' in x else 0):
        d = data[name]
        r = d['r']
        mask = r >= 0
        ax.plot(r[mask], d['p_final'][mask], '-', color=get_color(name, data.keys()),
                label=name, alpha=0.8, linewidth=1.5)

    ax.axvline(R_star, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('r')
    ax.set_ylabel('P')
    ax.set_title('Final Pressure Profile (Linear)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 12)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Bottom right: Pressure - Log scale
    # =========================================================================
    ax = axes[1, 1]
    for name in sorted(data.keys(), key=lambda x: int(x.split('=')[1]) if '=' in x else 0):
        d = data[name]
        r = d['r']
        mask = r >= 0
        ax.semilogy(r[mask], d['p_final'][mask], '-', color=get_color(name, data.keys()),
                    label=name, alpha=0.8, linewidth=1.5)

    ax.axvline(R_star, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('r')
    ax.set_ylabel('P')
    ax.set_title('Final Pressure Profile (Log)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 12)
    ax.set_ylim(1e-20, 1e-3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(plots_dir, 'final_profiles_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n\nSaved: {output_path}")

    # =========================================================================
    # Second figure: Zoom on surface region
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    # Surface zoom range
    r_min, r_max = 7.0, 9.0

    # Top left: Density zoom - Linear
    ax = axes2[0, 0]
    for name in sorted(data.keys(), key=lambda x: int(x.split('=')[1]) if '=' in x else 0):
        d = data[name]
        r = d['r']
        mask = (r >= r_min) & (r <= r_max)
        ax.plot(r[mask], d['rho0_final'][mask], 'o-', color=get_color(name, data.keys()),
                label=name, alpha=0.8, linewidth=1, markersize=3)

    ax.axvline(R_star, color='gray', linestyle='--', alpha=0.5, label=f'R_star')
    ax.set_xlabel('r')
    ax.set_ylabel(r'$\rho_0$')
    ax.set_title(f'Density near Surface (r={r_min}-{r_max})')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Top right: Density zoom - Log
    ax = axes2[0, 1]
    for name in sorted(data.keys(), key=lambda x: int(x.split('=')[1]) if '=' in x else 0):
        d = data[name]
        r = d['r']
        mask = (r >= r_min) & (r <= r_max)
        ax.semilogy(r[mask], d['rho0_final'][mask], 'o-', color=get_color(name, data.keys()),
                    label=name, alpha=0.8, linewidth=1, markersize=3)

    ax.axvline(R_star, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('r')
    ax.set_ylabel(r'$\rho_0$')
    ax.set_title(f'Density near Surface - Log (r={r_min}-{r_max})')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom left: Velocity profile
    ax = axes2[1, 0]
    for name in sorted(data.keys(), key=lambda x: int(x.split('=')[1]) if '=' in x else 0):
        d = data[name]
        r = d['r']
        mask = (r >= 0) & (r <= 10)
        ax.plot(r[mask], d['vr_final'][mask], '-', color=get_color(name, data.keys()),
                label=name, alpha=0.8, linewidth=1.5)

    ax.axvline(R_star, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('r')
    ax.set_ylabel(r'$v^r$')
    ax.set_title('Final Velocity Profile')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom right: Relative change from initial
    ax = axes2[1, 1]
    for name in sorted(data.keys(), key=lambda x: int(x.split('=')[1]) if '=' in x else 0):
        d = data[name]
        r = d['r']

        # Relative change in density (only where rho > floor)
        rho_init = d['rho0_init']
        rho_final = d['rho0_final']

        # Avoid division by zero
        rel_change = np.zeros_like(rho_init)
        valid = rho_init > 1e-9
        rel_change[valid] = (rho_final[valid] - rho_init[valid]) / rho_init[valid]

        mask = (r >= 0) & (r <= 10) & valid
        ax.plot(r[mask], rel_change[mask], '-', color=get_color(name, data.keys()),
                label=name, alpha=0.8, linewidth=1.5)

    ax.axvline(R_star, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('r')
    ax.set_ylabel(r'$(\rho_{final} - \rho_{init}) / \rho_{init}$')
    ax.set_title('Relative Density Change')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save second figure
    output_path2 = os.path.join(plots_dir, 'surface_zoom_comparison.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path2}")

    plt.close('all')


if __name__ == "__main__":
    main()
