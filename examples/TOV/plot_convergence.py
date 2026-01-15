#!/usr/bin/env python3
"""
Convergence plot for TOV star evolution - Central density relative change
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

# Data paths (exact paths)
FOLDERS = {
    'N=1600': '/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data2/tov_star_rhoc1p28em03_N1600_K100_G2_cow',
    'N=200': '/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data2/tov_star_rhoc1p28em03_N200_K100_G2_cow',
    'N=400': '/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data/tov_star_rhoc1p28em03_N400_K100_G2_cow_wz',
    'N=800': '/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data2/tov_star_rhoc1p28em03_N800_K100_G2_cow',
}

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def load_timeseries(folder_path):
    # Try npz first
    npz_file = os.path.join(folder_path, 'timeseries.npz')
    if os.path.exists(npz_file):
        data = np.load(npz_file)
        return data['times'], data['rho_central']

    # Try h5 file
    h5_file = os.path.join(folder_path, 'tov_evolution_cow.h5')
    if os.path.exists(h5_file):
        with h5py.File(h5_file, 'r') as f:
            return f['time'][:], f['rho_central'][:]

    return None, None

def main():
    fig, ax = plt.subplots(figsize=(10, 6))

    t_max = 7000.0

    for (label, folder_path), color in zip(FOLDERS.items(), COLORS):
        t, rho_c = load_timeseries(folder_path)

        if t is None:
            print(f"No data found for {label}")
            continue

        # Limit to t <= t_max
        mask = t <= t_max
        t = t[mask]
        rho_c = rho_c[mask]

        # Compute relative change
        rho_c_0 = rho_c[0]
        delta_rho_rel = (rho_c - rho_c_0) / rho_c_0

        ax.plot(t, delta_rho_rel, label=label, color=color, linewidth=0.8)
        print(f"{label}: {len(t)} points, t=[{t[0]:.1f}, {t[-1]:.1f}]")

    ax.set_xlabel(r'$t$ [M$_\odot$]', fontsize=12)
    ax.set_ylabel(r'$(\rho_c - \rho_{c,0})/\rho_{c,0}$', fontsize=12)
    ax.set_title('Central Density Relative Change - Convergence Test', fontsize=14)
    ax.set_xlim(0, t_max)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = os.path.join(os.path.dirname(__file__), 'plots', 'convergence_test.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    plt.show()

if __name__ == "__main__":
    main()
