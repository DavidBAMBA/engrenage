#!/usr/bin/env python3
"""
Plot TOV evolution every 10K iterations from saved snapshots.
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from matplotlib import cm

# Configuration
SNAPSHOT_FILE = "tov_evolution_data2/tov_snapshots_20251024_191815.h5"
PLOT_INTERVAL = 10000  # Plot every 10K iterations
NUM_GHOSTS = 3
OUTPUT_DIR = "tov_plots"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_snapshots_at_interval(h5file, interval):
    """Load snapshots at specified interval."""
    snapshots = {}
    snapshot_keys = sorted([k for k in h5file['snapshots'].keys() if k.startswith('step_')])

    # Get max step
    max_step = int(snapshot_keys[-1].split('_')[1])

    # Select snapshots at interval
    selected_steps = list(range(0, max_step + 1, interval))

    for step in selected_steps:
        key = f'step_{step:08d}'
        if key in h5file['snapshots']:
            grp = h5file['snapshots'][key]
            snapshots[step] = {
                'primitives': {k: grp['primitives'][k][:] for k in grp['primitives'].keys()},
                'conservatives': {k: grp['conservatives'][k][:] for k in grp['conservatives'].keys()},
                't': grp.attrs['time'],
                'step': grp.attrs['step']
            }
            print(f"Loaded {key}: t={snapshots[step]['t']:.6e}, step={snapshots[step]['step']}")

    return snapshots

def get_primitives_from_snapshot(snapshot):
    """Extract primitive variables from snapshot."""
    # Primitives are already stored in the snapshot
    primitives = snapshot['primitives']
    conservatives = snapshot['conservatives']

    return {
        'rho0': primitives['rho0'],
        'vr': primitives['vr'],
        'p': primitives['p'],
        'W': primitives['W'],
        'eps': primitives['eps'],
        'D': conservatives['D'],
        'Sr': conservatives['Sr'],
        'tau': conservatives['tau']
    }

def plot_evolution_10k(snapshots, grid_r):
    """Plot evolution with snapshots every 10K iterations."""

    # Sort snapshots by step number
    sorted_steps = sorted(snapshots.keys())
    n_snapshots = len(sorted_steps)

    print(f"\nGenerating plots for {n_snapshots} snapshots...")

    # Create colormap
    colors = cm.viridis(np.linspace(0, 1, n_snapshots))

    # Extract primitives for all snapshots
    prims = {}
    for step in sorted_steps:
        prims[step] = get_primitives_from_snapshot(snapshots[step])

    # Interior points (exclude ghosts)
    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
    r_int = grid_r[interior]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Baryon Density Evolution
    ax = axes[0, 0]
    for i, step in enumerate(sorted_steps):
        t = snapshots[step]['t']
        rho = prims[step]['rho0'][interior]
        label = f'step={step:,} (t={t:.4e})'
        ax.plot(r_int, rho, color=colors[i], linewidth=1.5, label=label, alpha=0.8)

    ax.set_xlabel('r', fontsize=12)
    ax.set_ylabel(r'$\rho_0$', fontsize=12)
    ax.set_title('Baryon Density Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best', ncol=2)

    # Plot 2: Pressure Evolution
    ax = axes[0, 1]
    for i, step in enumerate(sorted_steps):
        t = snapshots[step]['t']
        p = np.maximum(prims[step]['p'][interior], 1e-20)
        label = f'step={step:,}'
        ax.semilogy(r_int, p, color=colors[i], linewidth=1.5, label=label, alpha=0.8)

    ax.set_xlabel('r', fontsize=12)
    ax.set_ylabel('P', fontsize=12)
    ax.set_title('Pressure Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=8, loc='best', ncol=2)

    # Plot 3: Radial Velocity Evolution
    ax = axes[1, 0]
    for i, step in enumerate(sorted_steps):
        t = snapshots[step]['t']
        vr = prims[step]['vr'][interior]
        label = f'step={step:,}'
        ax.plot(r_int, vr, color=colors[i], linewidth=1.5, label=label, alpha=0.8)

    ax.set_xlabel('r', fontsize=12)
    ax.set_ylabel(r'$v^r$', fontsize=12)
    ax.set_title('Radial Velocity Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best', ncol=2)

    # Plot 4: Conserved Density D Evolution
    ax = axes[1, 1]
    for i, step in enumerate(sorted_steps):
        t = snapshots[step]['t']
        D = prims[step]['D'][interior]
        label = f'step={step:,}'
        ax.plot(r_int, D, color=colors[i], linewidth=1.5, label=label, alpha=0.8)

    ax.set_xlabel('r', fontsize=12)
    ax.set_ylabel('D', fontsize=12)
    ax.set_title('Conserved Density Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best', ncol=2)

    # Overall title
    t_final = snapshots[sorted_steps[-1]]['t']
    step_final = sorted_steps[-1]
    plt.suptitle(f'TOV Evolution Every {PLOT_INTERVAL:,} Steps\n'
                 f't=0 → t={t_final:.6e} (steps: 0 → {step_final:,})',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(OUTPUT_DIR, 'tov_evolution_10k.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    plt.show()

    return fig

def plot_central_values(snapshots):
    """Plot evolution of central values over time."""
    sorted_steps = sorted(snapshots.keys())

    times = []
    rho_c = []
    p_c = []

    for step in sorted_steps:
        t = snapshots[step]['t']
        prim = get_primitives_from_snapshot(snapshots[step])

        # Central values (at innermost interior point)
        idx_center = NUM_GHOSTS

        times.append(t)
        rho_c.append(prim['rho0'][idx_center])
        p_c.append(prim['p'][idx_center])

    times = np.array(times)
    rho_c = np.array(rho_c)
    p_c = np.array(p_c)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Central density vs time
    ax = axes[0]
    ax.plot(times, rho_c, 'b-', linewidth=2, marker='o', markersize=6)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(r'$\rho_c$', fontsize=12)
    ax.set_title('Central Density Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 2: Fractional change in central density
    ax = axes[1]
    delta_rho_c = (rho_c - rho_c[0]) / (rho_c[0] + 1e-30)
    ax.plot(times, delta_rho_c, 'r-', linewidth=2, marker='s', markersize=6)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(r'$\Delta\rho_c / \rho_c(0)$', fontsize=12)
    ax.set_title('Central Density Fractional Change', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(OUTPUT_DIR, 'tov_central_values_10k.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    plt.show()

    return fig

def main():
    """Main execution."""
    print("="*70)
    print("TOV Evolution Analysis - Every 10K Iterations")
    print("="*70)
    print(f"Snapshot file: {SNAPSHOT_FILE}")
    print(f"Plot interval: {PLOT_INTERVAL} steps")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)

    # Open HDF5 file
    if not os.path.exists(SNAPSHOT_FILE):
        print(f"\nERROR: Snapshot file not found: {SNAPSHOT_FILE}")
        print("Please check the file path and try again.")
        return

    with h5py.File(SNAPSHOT_FILE, 'r') as f:
        # Load grid
        grid_r = f['grid']['r'][:]
        print(f"\nGrid points: {len(grid_r)}")
        print(f"r range: [{grid_r[0]:.4f}, {grid_r[-1]:.4f}]")

        # Load snapshots at 10K intervals
        print(f"\nLoading snapshots every {PLOT_INTERVAL} steps...")
        snapshots = load_snapshots_at_interval(f, PLOT_INTERVAL)

        print(f"\nLoaded {len(snapshots)} snapshots")

    # Generate plots
    print("\n" + "="*70)
    print("Generating plots...")
    print("="*70)

    fig1 = plot_evolution_10k(snapshots, grid_r)
    fig2 = plot_central_values(snapshots)

    print("\n" + "="*70)
    print("Done!")
    print("="*70)

if __name__ == "__main__":
    main()
