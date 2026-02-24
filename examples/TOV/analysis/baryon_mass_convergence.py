#!/usr/bin/env python3
"""
Convergence analysis for TOV star evolution - Baryon Mass
Calculates baryon mass from snapshots and convergence order.

Baryon mass: M_b = 4π ∫ D * r² * √γ_rr dr
where D is the conserved rest-mass density and γ_rr is the radial metric component.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import argparse
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# Unit conversions
M_SUN_SECONDS = 4.926e-6
FREQ_CONVERSION = 1.0 / (M_SUN_SECONDS * 1e3)

# Resolutions - UPDATE THESE TO CHANGE RESOLUTIONS
N_low = 1000
N_med = 2000
N_high = 4000

# TOV cache directory for analytical solution
TOV_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'tov_iso_cache',
                             'TOVSOL_ISO_K=100.0_G=2.0_rho=1.280000e-03')

# Data paths - constructed from resolution values
FOLDERS = {
    f'N={N_low}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_refact_rmax100.0_jax/tov_star_rhoc1p28em03_N1000_K100_G2_dyn_mp5',
    f'N={N_med}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_refact_rmax100.0_jax/tov_star_rhoc1p28em03_N2000_K100_G2_dyn_mp5',
     f'N={N_high}': f'/home/davidbamba/repositories/engrenage/examples/TOV/tov_evolution_data_refact_rmax100.0_jax/tov_star_rhoc1p28em03_N4000_K100_G2_dyn_mp5',
}
# Resolution labels (keys to FOLDERS dictionary)
low_res = f'N={N_low}'
med_res = f'N={N_med}'
high_res = f'N={N_high}'

COLORS = ['#1f77b4', "#ff7f0e", '#2ca02c', "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def load_snapshots_with_primitives(folder_path):
    """
    Load snapshots with primitives (rho0, W) and conformal factor (phi).
    Returns times, r grid, and lists of rho0, W, phi.
    """
    h5_file = os.path.join(folder_path, 'tov_snapshots_dyn_jax.h5')

    times = []
    rho0_list = []
    W_list = []
    phi_list = []

    with h5py.File(h5_file, 'r') as f:
        # Load grid
        r = f['grid/r'][:]

        snaps = f['snapshots']
        for key in sorted(snaps.keys()):
            g = snaps[key]
            times.append(g.attrs['time'])
            rho0_list.append(g['primitives/rho0'][:])
            W_list.append(g['primitives/W'][:])
            phi_list.append(g['bssn/phi'][:])

    return np.array(times), r, rho0_list, W_list, phi_list


def compute_baryon_mass(r, rho0, W, phi):
    """
    Compute baryon mass: M_b = 4π ∫ ρ₀ W ψ⁶ r² dr

    This matches the formula in utils_TOVEvolution.py:
    - ρ₀ = rest-mass density (primitive)
    - W = Lorentz factor
    - ψ = e^φ = conformal factor
    - ψ⁶ = e^{6φ} = conformal 3-metric determinant
    """
    from scipy.integrate import simpson

    # Only integrate where r > 0 (avoid singularity at origin)
    mask = r > 0
    r_pos = r[mask]
    rho0_pos = rho0[mask]
    W_pos = W[mask]
    phi_pos = phi[mask]

    # Compute ψ⁶ = e^{6φ}
    psi6 = np.exp(6.0 * phi_pos)

    # Integrand: ρ₀ W ψ⁶ r²
    integrand = rho0_pos * W_pos * psi6 * r_pos**2

    # Integrate using Simpson's rule
    M_b = 4.0 * np.pi * simpson(integrand, x=r_pos)

    return M_b


def compute_theoretical_baryon_mass(tov_cache_dir):
    """
    Compute theoretical baryon mass from the TOV ODE solution (very fine grid).
    M_b = 4π ∫ ρ₀ ψ⁶ r² dr  (W=1 for static star)
    """
    from scipy.integrate import simpson

    r_tov = np.load(os.path.join(tov_cache_dir, "r_iso.npy"))
    rho_tov = np.load(os.path.join(tov_cache_dir, "rho_baryon.npy"))
    exp4phi_tov = np.load(os.path.join(tov_cache_dir, "exp4phi.npy"))

    # ψ⁶ = (e^{4φ})^{3/2}
    psi6 = exp4phi_tov ** 1.5

    mask = r_tov > 0
    integrand = rho_tov[mask] * psi6[mask] * r_tov[mask]**2
    return 4.0 * np.pi * simpson(integrand, x=r_tov[mask])


def compute_gravitational_mass(r, phi, r_extract_frac=0.7):
    """
    Extract ADM gravitational mass from asymptotic conformal factor.

    In isotropic coords: ψ = e^φ → 1 + M/(2r) at large r
    So: M = 2r(e^φ - 1) at extraction radius.

    Averages over a radial window around r_extract for robustness.
    """
    r_extract = r_extract_frac * r.max()
    # Average over ±10% around extraction radius
    mask = (r > 0.9 * r_extract) & (r < 1.1 * r_extract)
    if np.sum(mask) < 3:
        idx = np.argmin(np.abs(r - r_extract))
        return 2.0 * r[idx] * (np.exp(phi[idx]) - 1.0)
    M_values = 2.0 * r[mask] * (np.exp(phi[mask]) - 1.0)
    return np.mean(M_values)


def load_tov_gravitational_mass(tov_cache_dir):
    """Load theoretical gravitational mass from TOV cache scalars."""
    scalars = np.load(os.path.join(tov_cache_dir, "scalars.npy"))
    return scalars[2]  # M_star


def compute_L1_error_discrete(r_coarse, rho_coarse, r_fine, rho_fine, r_max=None):
    """
    Compute discrete L1 norm: E = (1/N) Σ |ρ₁ - ρ₂|
    Interpolates fine resolution to coarse grid.
    """
    interp_func = interp1d(r_fine, rho_fine, kind='cubic', bounds_error=False, fill_value='extrapolate')
    rho_fine_interp = interp_func(r_coarse)

    mask = r_coarse > 0
    if r_max is not None:
        mask = mask & (r_coarse <= r_max)

    diff = np.abs(rho_coarse[mask] - rho_fine_interp[mask])
    return np.mean(diff)


def find_stellar_radius(r, rho, rho_atm=1e-16):
    """Find stellar radius where density drops to atmosphere level."""
    idx = np.where(rho > rho_atm)[0]
    if len(idx) > 0:
        return r[idx[-1]]
    return r.max()


def running_average(x, window):
    """Compute running average with given window size."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')


def smooth(y, window=10, polyorder=3):
    """Apply light Savitzky-Golay smoothing to data."""
    if len(y) < window:
        return y
    return savgol_filter(y, window, polyorder)


def interpolate_to_common_times(t_ref, M_ref, t_other, M_other):
    """Interpolate M_other to the time points of t_ref (cubic)."""
    f = interp1d(t_other, M_other, kind='cubic', bounds_error=False, fill_value='extrapolate')
    return f(t_ref)


def extract_resolution_from_dirname(dirname):
    """Extract resolution number from directory name."""
    import re
    match = re.search(r'[Nn]r?[=_]?(\d+)', dirname)
    if match:
        return int(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Baryon mass convergence analysis (requires exactly 3 resolutions)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python baryon_mass_convergence.py                       # Use default folders
  python baryon_mass_convergence.py --data-dirs D1 D2 D3  # Exactly 3 directories
'''
    )
    parser.add_argument('--data-dirs', nargs='+', default=None,
                        help='List of data directories (exactly 3 required). Default: use FOLDERS')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for plots. Default: script_dir/plots')
    parser.add_argument('--t-max', type=float, default=2000.0,
                        help='Maximum time to plot. Default: 2000.0')
    parser.add_argument('--tov-cache', default=None,
                        help='Path to TOV cache directory. Default: auto-detect')
    args = parser.parse_args()

    t_max = args.t_max
    tov_cache_dir = args.tov_cache if args.tov_cache else TOV_CACHE_DIR
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine output directory
    if args.output_dir:
        plots_dir = args.output_dir
    else:
        plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Determine data folders
    if args.data_dirs:
        # Validate exactly 3 directories
        if len(args.data_dirs) != 3:
            print(f"Error: This script requires exactly 3 directories, got {len(args.data_dirs)}")
            return

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

        if len(folders_dict) != 3:
            print("Error: Need exactly 3 valid directories")
            return

        # Sort by resolution to assign low/med/high
        sorted_items = sorted(folders_dict.items(), key=lambda x: extract_resolution_from_dirname(x[0]) or 0)
        low_res, med_res, high_res = sorted_items[0][0], sorted_items[1][0], sorted_items[2][0]
        folders_dict = dict(sorted_items)
    else:
        # Use default FOLDERS (backward compatibility)
        folders_dict = FOLDERS
        low_res = f'N={N_low}'
        med_res = f'N={N_med}'
        high_res = f'N={N_high}'

    # =========================================================
    # Load all data and compute baryon mass for each snapshot
    # =========================================================
    data = {}

    for label, folder_path in folders_dict.items():
        print(f"Processing {label}...")
        times, r, rho0_list, W_list, phi_list = load_snapshots_with_primitives(folder_path)

        # Compute baryon mass and gravitational mass for each snapshot
        M_b_list = []
        M_grav_list = []
        for i in range(len(times)):
            M_b_list.append(compute_baryon_mass(r, rho0_list[i], W_list[i], phi_list[i]))
            M_grav_list.append(compute_gravitational_mass(r, phi_list[i]))

        M_b_arr = np.array(M_b_list)
        M_grav_arr = np.array(M_grav_list)

        # Apply time mask
        mask = times <= t_max
        rho0_masked = [rho0_list[i] for i in range(len(times)) if mask[i]]
        data[label] = {
            't': times[mask],
            'r': r,
            'rho0': rho0_masked,
            'M_b': M_b_arr[mask],
            'M_b_0': M_b_arr[0],
            'M_grav': M_grav_arr[mask],
            'M_grav_0': M_grav_arr[0],
        }
        print(f"  {label}: {len(times[mask])} snapshots, M_b(0) = {M_b_arr[0]:.6f}, M_grav(0) = {M_grav_arr[0]:.6f}")

    # =========================================================
    # Compute theoretical masses from TOV solution
    # =========================================================
    print("\n--- Computing theoretical masses ---")
    M_b_exact = compute_theoretical_baryon_mass(tov_cache_dir)
    M_grav_exact = load_tov_gravitational_mass(tov_cache_dir)
    print(f"  M_b   (TOV analytical) = {M_b_exact:.6f}")
    print(f"  M_grav (TOV analytical) = {M_grav_exact:.6f}")
    for label, d in data.items():
        print(f"  {label}: M_b(0) = {d['M_b_0']:.6f}, err = {abs(d['M_b_0'] - M_b_exact)/M_b_exact:.2e}"
              f"  |  M_grav(0) = {d['M_grav_0']:.6f}, err = {abs(d['M_grav_0'] - M_grav_exact)/M_grav_exact:.2e}")

    # =========================================================
    # Compute errors relative to theoretical mass
    # =========================================================
    print("\n--- Computing mass conservation errors (vs theoretical) ---")

    l1_mass_error = {}
    for label, d in data.items():
        M_b = d['M_b']
        # Relative mass error vs theoretical value
        rel_error = np.abs(M_b - M_b_exact) / M_b_exact
        l1_mass_error[label] = rel_error
        print(f"  {label}: max |ΔM_b/M_b_exact| = {np.max(rel_error):.2e}")

    # =========================================================
    # Compute convergence order (baryon mass)
    # =========================================================
    print("\n--- Computing convergence order (M_b) ---")

    t_1 = data[low_res]['t']
    t_2 = data[med_res]['t']
    t_3 = data[high_res]['t']

    M_1 = data[low_res]['M_b']
    M_2 = data[med_res]['M_b']
    M_3 = data[high_res]['M_b']

    M_2_interp = interpolate_to_common_times(t_1, M_1, t_2, M_2)
    M_3_interp = interpolate_to_common_times(t_1, M_1, t_3, M_3)

    E12 = np.abs(M_1 - M_2_interp)
    E23 = np.abs(M_2_interp - M_3_interp)

    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.log(E12 / E23) / np.log(2.0)
    p_avg = running_average(p, window=10)

    # =========================================================
    # Compute convergence order (gravitational mass)
    # =========================================================
    print("--- Computing convergence order (M_grav) ---")

    Mg_1 = data[low_res]['M_grav']
    Mg_2 = data[med_res]['M_grav']
    Mg_3 = data[high_res]['M_grav']

    Mg_2_interp = interpolate_to_common_times(t_1, Mg_1, t_2, Mg_2)
    Mg_3_interp = interpolate_to_common_times(t_1, Mg_1, t_3, Mg_3)

    Eg12 = np.abs(Mg_1 - Mg_2_interp)
    Eg23 = np.abs(Mg_2_interp - Mg_3_interp)

    with np.errstate(divide='ignore', invalid='ignore'):
        pg = np.log(Eg12 / Eg23) / np.log(2.0)
    pg_avg = running_average(pg, window=3)

    # =========================================================
    # Compute L1 density convergence (rest-mass density)
    # =========================================================
    print("--- Computing convergence order (rest-mass density L1) ---")

    r_1 = data[low_res]['r']
    r_2 = data[med_res]['r']
    r_3 = data[high_res]['r']

    rho_1_list = data[low_res]['rho0']
    rho_2_list = data[med_res]['rho0']
    rho_3_list = data[high_res]['rho0']

    # Find stellar radius from highest resolution initial profile
    R_star = find_stellar_radius(r_3, rho_3_list[0]) * 0.95

    # Interpolate density profiles in time (cubic) to common time grid t_1
    def interpolate_profiles_in_time(t_source, rho_list_source, r_source, t_target):
        """Cubic interpolation of density profiles to target times."""
        rho_arr = np.array(rho_list_source)  # (n_times, n_r)
        f = interp1d(t_source, rho_arr, axis=0, kind='cubic',
                     bounds_error=False, fill_value='extrapolate')
        return f(t_target)  # (n_target, n_r)

    rho_2_at_t1 = interpolate_profiles_in_time(data[med_res]['t'], rho_2_list, r_2, t_1)
    rho_3_at_t1 = interpolate_profiles_in_time(data[high_res]['t'], rho_3_list, r_3, t_1)

    E12_rho = []
    E23_rho = []
    for i in range(len(t_1)):
        e12 = compute_L1_error_discrete(r_1, rho_1_list[i], r_2, rho_2_at_t1[i], r_max=R_star)
        e23 = compute_L1_error_discrete(r_2, rho_2_at_t1[i], r_3, rho_3_at_t1[i], r_max=R_star)
        E12_rho.append(e12)
        E23_rho.append(e23)

    E12_rho = np.array(E12_rho)
    E23_rho = np.array(E23_rho)

    with np.errstate(divide='ignore', invalid='ignore'):
        p_rho = np.log(E12_rho / E23_rho) / np.log(2.0)
    p_rho_avg = running_average(p_rho, window=3)

    # =========================================================
    # Create figure with 3x2 subplots
    # =========================================================
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    ax5, ax6 = axes[2]

    # Convert time to milliseconds
    t_ms_factor = M_SUN_SECONDS * 1e3  # M_sun -> ms
    t_max_ms = t_max * t_ms_factor

    valid = t_1 > 10

    # Panel (a): Gravitational mass conservation |ΔM|/M(0) (log scale)
    for (label, d), color in zip(data.items(), COLORS):
        M_grav = d['M_grav']
        M_grav_0 = d['M_grav_0']
        ax1.semilogy(d['t'] * t_ms_factor, smooth(np.abs(M_grav - M_grav_0) / M_grav_0),
                     label=label, color=color, linewidth=0.8)
    ax1.set_xlabel(r'$t$ [ms]')
    ax1.set_ylabel(r'$|\Delta M| / M(0)$')
    ax1.set_title(r'(a) Gravitational Mass Conservation')
    ax1.legend(fontsize=8)

    # Panel (b): Convergence order of M_grav
    ax2.plot(t_1[valid] * t_ms_factor, pg[valid], color='k', alpha=0.3, lw=0.6, label='instantaneous')
    ax2.plot(t_1[valid] * t_ms_factor, pg_avg[valid], color='k', lw=2.5, label='running average')
    ax2.axhline(2, ls='--', color='gray', label='2nd order')
    ax2.axhline(3, ls=':', color='gray', label='3rd order')
    ax2.axhline(5, ls='-.', color='gray', label='5th order')
    ax2.set_xlim(0, t_max_ms)
    ax2.set_ylim(-1, 8)
    ax2.set_xlabel(r'$t$ [ms]')
    ax2.set_ylabel(r'Convergence order $p(t)$')
    ax2.set_title(rf'(b) Convergence Order of $M_{{grav}}$')
    ax2.legend(fontsize=8)

    # Panel (c): Baryon mass conservation ΔM_b/M_b(0) (linear scale)
    for (label, d), color in zip(data.items(), COLORS):
        M_b = d['M_b']
        M_b_0 = d['M_b_0']
        ax3.plot(d['t'] * t_ms_factor, smooth((M_b - M_b_0) / M_b_0),
                 label=label, color=color, linewidth=0.8)
    ax3.set_xlabel(r'$t$ [ms]')
    ax3.set_ylabel(r'$\Delta M_b / M_b(0)$')
    ax3.set_title(r'(c) Baryon Mass Conservation')
    ax3.legend(fontsize=8)

    # Panel (d): Convergence order of M_b
    ax4.plot(t_1[valid] * t_ms_factor, p[valid], color='k', alpha=0.3, lw=0.6, label='instantaneous')
    ax4.plot(t_1[valid] * t_ms_factor, p_avg[valid], color='k', lw=2.5, label='running average')
    ax4.axhline(2, ls='--', color='gray', label='2nd order')
    ax4.axhline(3, ls=':', color='gray', label='3rd order')
    ax4.axhline(5, ls='-.', color='gray', label='5th order')
    ax4.set_xlim(0, t_max_ms)
    ax4.set_ylim(-1, 8)
    ax4.set_xlabel(r'$t$ [ms]')
    ax4.set_ylabel(r'Convergence order $p(t)$')
    ax4.set_title(rf'(d) Convergence Order of $M_b$')
    ax4.legend(fontsize=8)

    # Panel (e): Rest-mass density L1 errors
    ax5.semilogy(t_1 * t_ms_factor, smooth(E12_rho),
                 label=rf'$E_{{12}}$ ({low_res} vs {med_res})', color=COLORS[0], linewidth=1.2)
    ax5.semilogy(t_1 * t_ms_factor, smooth(E23_rho),
                 label=rf'$E_{{23}}$ ({med_res} vs {high_res})', color=COLORS[1], linewidth=1.2)
    ax5.set_xlabel(r'$t$ [ms]')
    ax5.set_ylabel(r'$L_1$ error')
    ax5.set_title(rf'(e) Rest-Mass Density $L_1$: $(1/N) \sum |\Delta\rho_0|$')
    ax5.legend(fontsize=8)

    # Panel (f): Convergence order of rest-mass density
    ax6.plot(t_1[valid] * t_ms_factor, p_rho[valid], color='k', alpha=0.3, lw=0.6, label='instantaneous')
    ax6.plot(t_1[valid] * t_ms_factor, p_rho_avg[valid], color='k', lw=2.5, label='running average')
    ax6.axhline(2, ls='--', color='gray', label='2nd order')
    ax6.axhline(3, ls=':', color='gray', label='3rd order')
    ax6.axhline(5, ls='-.', color='gray', label='5th order')
    ax6.set_xlim(0, t_max_ms)
    ax6.set_ylim(-1, 8)
    ax6.set_xlabel(r'$t$ [ms]')
    ax6.set_ylabel(r'Convergence order $p(t)$')
    ax6.set_title(rf'(f) Convergence Order of $\rho_0$')
    ax6.legend(fontsize=8)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(plots_dir, 'baryon_mass_convergence.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()
    #plt.close('all')

    # =========================================================
    # Print convergence summary
    # =========================================================
    valid_mask = t_1 > 500

    def mean_order(p_arr):
        pv = p_arr[valid_mask]
        fm = np.isfinite(pv)
        if np.sum(fm) == 0:
            return np.nan, np.nan
        return np.mean(pv[fm]), np.std(pv[fm])

    pg_mean, pg_std = mean_order(pg)
    p_mean, p_std = mean_order(p)
    prho_mean, prho_std = mean_order(p_rho)

    t_valid = t_1[valid_mask]
    print("\n" + "="*60)
    print(f"CONVERGENCE SUMMARY  ({low_res}, {med_res}, {high_res})")
    print(f"Time range: t = [{t_valid[0]:.1f}, {t_valid[-1]:.1f}] M_sun")
    print("="*60)
    print(f"\n  {'Quantity':<30} {'Mean order':>12} {'Std':>8}")
    print(f"  {'-'*50}")
    print(f"  {'M_grav (gravitational mass)':<30} {pg_mean:>10.2f}   {pg_std:>6.2f}")
    print(f"  {'M_b (baryon mass)':<30} {p_mean:>10.2f}   {p_std:>6.2f}")
    print(f"  {'rho_0 (rest-mass density L1)':<30} {prho_mean:>10.2f}   {prho_std:>6.2f}")

    print(f"\n--- Masses ---")
    print(f"  M_b    (TOV exact) = {M_b_exact:.6f}")
    print(f"  M_grav (TOV exact) = {M_grav_exact:.6f}")
    for label, d in data.items():
        print(f"  {label}: M_b(0)={d['M_b_0']:.6f}  M_grav(0)={d['M_grav_0']:.6f}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
