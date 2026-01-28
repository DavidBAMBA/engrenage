#!/usr/bin/env python3
"""
Plot central density relative change for TOV star evolution
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

# Data paths
# Data paths - UPDATE THESE TO YOUR ACTUAL PATHS
FOLDERS = {
    'N=800': '/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_last_politro/tov_star_rhoc1p28em03_N800_K100_G2_cow_mp5',
    'N=2000': '/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_last_domain/tov_star_rhoc1p28em03_N2000_K100_G2_cow_mp5',
    #'N=800': '/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_last_politro/tov_star_rhoc1p28em03_N800_K100_G2_cow_mp5',
    #'N=1600': '/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_last_politro/tov_star_rhoc1p28em03_N1600_K100_G2_cow_mp5',
}

COLORS = ['#1f77b4', "#ff7f0e", '#2ca02c', "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def load_timeseries(folder_path):
    h5_file = os.path.join(folder_path, 'tov_evolution_cow.h5')
    if os.path.exists(h5_file):
        with h5py.File(h5_file, 'r') as f:
            times = f['time'][:]
            rho_central = f['rho_central'][:]
            baryon_mass = f['baryon_mass'][:] if 'baryon_mass' in f else None
            return times, rho_central, baryon_mass
    return None, None, None


def main():
    t_max = 2000.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    for (label, folder_path), color in zip(FOLDERS.items(), COLORS):
        t, rho_c, M_b = load_timeseries(folder_path)
        if t is None:
            continue

        mask = t <= t_max
        t = t[mask]
        rho_c = rho_c[mask]

        # Plot 1: Central density relative change
        rho_c_0 = rho_c[0]
        delta_rho_rel = (rho_c - rho_c_0) / rho_c_0
        ax1.plot(t, delta_rho_rel, label=label, color=color, linewidth=0.8)

        # Plot 2: Baryon mass relative change (log scale)
        if M_b is not None:
            M_b = M_b[mask]
            M_b_0 = M_b[0]
            delta_M_b_rel = np.abs((M_b - M_b_0) / M_b_0)
            ax2.plot(t, delta_M_b_rel, label=label, color=color, linewidth=0.8)

            # Plot 3: log(abs(M_b - M_b_0))
            log_abs_delta_M = np.log10(np.abs(M_b - M_b_0) + 1e-20)
            ax3.plot(t, log_abs_delta_M, label=label, color=color, linewidth=0.8)

            # Plot 4: Mass conservation error (relative, semilogy)
            rel_error = np.abs(M_b - M_b_0) / M_b_0
            ax4.semilogy(t, rel_error, label=label, color=color, linewidth=0.8)

    # Configure ax1
    ax1.set_xlabel(r'$t$ [M$_\odot$]')
    ax1.set_ylabel(r'$(\rho_c-\rho_{c,0})/\rho_{c,0}$')
    ax1.set_title('(a) Central Density Relative Change')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Configure ax2
    ax2.set_xlabel(r'$t$ [M$_\odot$]')
    ax2.set_ylabel(r'$|(M_b-M_{b,0})/M_{b,0}|$')
    ax2.set_title('(b) Baryon Mass Relative Change (log scale)')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Configure ax3
    ax3.set_xlabel(r'$t$ [M$_\odot$]')
    ax3.set_ylabel(r'$\log_{10}|M_b - M_{b,0}|$')
    ax3.set_title('(c) Baryon Mass Absolute Error (log scale)')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # Configure ax4
    ax4.set_xlabel(r'$t$ [M$_\odot$]')
    ax4.set_ylabel(r'$|M_b - M_{b,0}| / M_{b,0}$')
    ax4.set_title('(d) Mass Conservation Error (relative)')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    # Save in plots directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'rho_central.png'), dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
