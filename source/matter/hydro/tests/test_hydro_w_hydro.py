#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hydro-without-Hydro Test: BSSN evolution with static TOV matter sources.
Based on Baumgarte, Hughes, and Shapiro (https://arxiv.org/abs/gr-qc/9902024)
"""

import os, sys, time
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
# ---------- localizar repo root para importar `source.*` ----------
def locate_repo_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (cand / 'source').is_dir():
            return cand
    return start

THIS = Path(__file__).resolve()
REPO = locate_repo_root(THIS.parent)
SRC  = REPO / 'source'
if str(REPO) not in sys.path: sys.path.append(str(REPO))
if str(SRC)  not in sys.path: sys.path.append(str(SRC))

# ---------- imports Engrenage ----------
from source.core.spacing import LinearSpacing, SpacingExtent, NUM_GHOSTS
from source.core.grid import Grid
from source.core.statevector import StateVector
from source.core.rhsevolution import get_rhs
from source.backgrounds.sphericalbackground import FlatSphericalBackground, i_r, i_t, i_p
from source.matter.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import PolytropicEOS, IdealGasEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.bssn.bssnstatevariables import (
    NUM_BSSN_VARS,
    idx_phi, idx_hrr, idx_htt, idx_hpp,
    idx_K, idx_arr, idx_att, idx_app,
    idx_lambdar, idx_shiftr, idx_br, idx_lapse
)

from scipy.interpolate import interp1d

# Import constraint diagnostic
from source.bssn.constraintsdiagnostic import get_constraints_diagnostic
from source.bssn.bssnvars import BSSNVars

# High-fidelity TOV initial data
from examples.TOV.tov_initial_data_interpolated import create_initial_data_interpolated
from examples.TOV.tov_solver import TOVSolver

# -------------------- Progress monitor (dummy) --------------------
class DummyProgress:
    def update(self, *args, **kwargs):
        pass

# -------------------- RK4 Time Evolution --------------------
def rk4_step_hwh(state, grid: Grid, background, matter, cfl=0.2):
    """
    RK4 time integration for Hydro-without-Hydro test.
    Evolves BSSN while keeping T^μν fixed (matter sources from TOV).

    This mirrors a reference implementation where:
    - BSSN equations are evolved with fixed T^μν source terms
    - Matter variables are NOT evolved (RHS = 0)
    - CFL default = 0.5
    """
    dummy_progress = DummyProgress()
    time_state = [0.0, 0.1]  # [last_t, deltat] format

    # Determine timestep from CFL condition
    r = grid.r
    dr = r[1] - r[0] if len(r) > 1 else 0.1
    dt = cfl * dr

    # RK4 stages (classical 4th-order Runge-Kutta)
    # k1 = f(y_n)
    rhs_full = get_rhs(0.0, state.flatten(), grid, background, matter, dummy_progress, time_state)
    k1 = rhs_full.reshape(grid.NUM_VARS, -1)

    # Zero out matter RHS (Hydro WITHOUT Hydro)
    k1[matter.idx_D,   :] = 0.0
    k1[matter.idx_Sr,  :] = 0.0
    k1[matter.idx_tau, :] = 0.0

    # k2 = f(y_n + dt/2 * k1)
    s_tmp = state + 0.5 * dt * k1
    grid.fill_boundaries(s_tmp)

    rhs_full = get_rhs(0.0, s_tmp.flatten(), grid, background, matter, dummy_progress, time_state)
    k2 = rhs_full.reshape(grid.NUM_VARS, -1)

    k2[matter.idx_D,   :] = 0.0
    k2[matter.idx_Sr,  :] = 0.0
    k2[matter.idx_tau, :] = 0.0

    # k3 = f(y_n + dt/2 * k2)
    s_tmp = state + 0.5 * dt * k2
    grid.fill_boundaries(s_tmp)

    rhs_full = get_rhs(0.0, s_tmp.flatten(), grid, background, matter, dummy_progress, time_state)
    k3 = rhs_full.reshape(grid.NUM_VARS, -1)

    k3[matter.idx_D,   :] = 0.0
    k3[matter.idx_Sr,  :] = 0.0
    k3[matter.idx_tau, :] = 0.0

    # k4 = f(y_n + dt * k3)
    s_tmp = state + dt * k3
    grid.fill_boundaries(s_tmp)

    rhs_full = get_rhs(0.0, s_tmp.flatten(), grid, background, matter, dummy_progress, time_state)
    k4 = rhs_full.reshape(grid.NUM_VARS, -1)

    k4[matter.idx_D,   :] = 0.0
    k4[matter.idx_Sr,  :] = 0.0
    k4[matter.idx_tau, :] = 0.0

    # Combine: y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    sn = state + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    grid.fill_boundaries(sn)

    return dt, sn


# -------------------- Main Evolution Function --------------------
def run_hwh_test(
    # Reference default parameters (Baumgarte paper)
    gamma=2.0,                # Polytropic index (n=1 → γ=2)
    K=1.0,                    # Polytropic constant
    rho_central=0.2,          # Central density (Baumgarte paper value)
    cfl=0.1,                  # CFL factor
    # Grid parameters
    r_max=None,              # Will be set to 2*R_iso automatically
    dr=0.01,                 # Spatial resolution
    atmosphere=1e-10,        # Atmosphere threshold
    # Evolution parameters
    t_final_factor=100.8,      # t_final = t_final_factor * TOV_Mass
    progress=True,           # Print progress
    save_interval=10         # Save frequency for detailed data
):
    """
    Run Hydro-without-Hydro test using reference parameters.

    This test evolves the BSSN equations with static TOV matter sources
    to verify numerical stability, as described in Baumgarte, Hughes & Shapiro.

    Parameters  defaults:
    - gamma = 2.0 (polytropic EOS)
    - K = 1.0 (polytropic constant)
    - rho_central = 0.129285 (central baryon density)
    - CFL = 0.1
    - t_final = 1.8 * TOV_Mass

    Returns:
        dict: Contains evolution data for analysis and plotting
    """
    # High-fidelity TOV solution
    solver = TOVSolver(K=K, Gamma=gamma)
    if r_max is None:
        # Probe solution to determine stellar radius (densely sampled)
        r_probe = np.linspace(0.0, 200.0, 4096)
        tov_probe = solver.solve(rho_central, r_max=r_probe[-1])
        r_max = 2.0 * tov_probe['R']  # Domain = 2 * stellar radius (Schwarzschild)
    else:
        # Still generate a probe for reporting consistency
        r_probe = np.linspace(0.0, r_max, 4096)
        tov_probe = solver.solve(rho_central, r_max=r_max)

    # Setup grid
    spacing = LinearSpacing(int(r_max/dr), r_max, SpacingExtent.HALF)
    r = spacing[0]

    # Setup hydro with reference parameters
    eos = IdealGasEOS(gamma=gamma)

    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode='dynamic',
        atmosphere=atmosphere,
        reconstructor=create_reconstruction("mp5"),
        riemann_solver=HLLRiemannSolver()
    )
    state_vec = StateVector(hydro)
    grid = Grid(spacing, state_vec)
    background = FlatSphericalBackground(r)

    # Set background in hydro object (required for get_emtensor)
    hydro.background = background

    # Solve TOV covering our domain (Schwarzschild coordinate)
    r_domain_max = float(np.max(np.abs(r)))
    tov_solution = solver.solve(rho_central, r_max=r_domain_max)
    M_star = tov_solution['M_star']
    R_star = tov_solution['R']

    # Report TOV properties 
    if progress:
        print("Generated TOV star with:")
        print(f"  M        = {M_star:.15e}")
        print(f"  R_star   = {R_star:.15e}")
        print(f"  M/R_star = {M_star/R_star:.15e}\n")

    tov_data = dict(tov_solution)

    # Build initial data (Schwarzschild-based interpolation + ADM→BSSN conversion)
    state = create_initial_data_interpolated(
        tov_solution,
        grid,
        background,
        hydro.eos,
        atmosphere=hydro.atmosphere,
        interp_order=11
    )

    # Extract primitive data for diagnostics
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state, bssn_vars, grid)
    primitives = hydro._get_primitives(bssn_vars, grid.r, grid)
    rho0 = primitives['rho0'].copy()
    p = primitives['p'].copy()
    v = primitives['vr'].copy()

    # CRITICAL: Fix stress-energy tensor sources for hydro-without-hydro test
    # Compute T^{μν} projections (ρ, S_i, S_{ij}, S) from initial data and freeze them.
    # This prevents matter sources from changing as the metric evolves during BSSN evolution.
    # See Baumgarte et al. (1999) gr-qc/9902024: "We only evolve the gravitational fields,
    # holding the matter sources to their OV [TOV] values."
    emtensor_initial = hydro.get_emtensor(grid.r, bssn_vars, background)
    hydro.set_fixed_emtensor_sources(
        rho=emtensor_initial.rho,
        Si=emtensor_initial.Si,
        Sij=emtensor_initial.Sij,
        S=emtensor_initial.S
    )

    # Set final time based on TOV mass
    t_final = t_final_factor * M_star

    # Initialize data storage
    t = 0.0
    steps = 0
    center = NUM_GHOSTS

    # Time series at center
    lapse_c = [state[idx_lapse, center]]
    phi_c   = [state[idx_phi, center]]
    K_c     = [state[idx_K, center]]
    hrr_c   = [state[idx_hrr, center]]
    times   = [0.0]

    # Full state snapshots
    times_detailed = [0.0]
    states_detailed = [state.copy()]

    # Calculate initial Hamiltonian constraint
    Ham_init, Mom_init = get_constraints_diagnostic(state, 0.0, grid, background, hydro)
    Ham_center = [Ham_init[0, center]]  # Hamiltonian constraint at center
    Ham_snapshots = [Ham_init[0, :].copy()]  # Full Hamiltonian profile

    if progress:
        print("Starting Hydro-without-Hydro evolution...")
        print(f"Grid: Nr={len(r)-2*NUM_GHOSTS}, r_max={r_max:.2f}")
        print(f"Time: t_final={t_final:.4f}, CFL={cfl}")
        print(f"TOV: M={M_star:.4f}, R={R_star:.4f}\n")

    # Main evolution loop
    while t < t_final and steps < 200000:
        # RK4 time step
        dt, state = rk4_step_hwh(state, grid, background, hydro, cfl=cfl)
        t += dt
        steps += 1

        # Save time series data
        if steps % 10 == 0:
            lapse_c.append(state[idx_lapse, center])
            phi_c.append(state[idx_phi, center])
            K_c.append(state[idx_K, center])
            hrr_c.append(state[idx_hrr, center])
            times.append(t)

            # Calculate Hamiltonian constraint
            Ham, Mom = get_constraints_diagnostic(state, t, grid, background, hydro)
            Ham_center.append(Ham[0, center])

        # Save full snapshots
        if steps % save_interval == 0:
            times_detailed.append(t)
            states_detailed.append(state.copy())

            # Save Hamiltonian constraint snapshot
            Ham_full, _ = get_constraints_diagnostic(state, t, grid, background, hydro)
            Ham_snapshots.append(Ham_full[0, :].copy())

        # Progress output
        if progress and steps % 100 == 0:
            print(f"  t/M={t/M_star:.3f}  dt/M={dt/M_star:.2e}  "
                  f"α_c={lapse_c[-1]:.6f}  φ_c={phi_c[-1]:+.3e}  K_c={K_c[-1]:+.3e}")

    if progress:
        print(f"\nEvolution completed:")
        print(f"  Steps: {steps}")
        print(f"  Final time: t={t:.4f} (t/M={t/M_star:.3f})")
        print(f"  Center values:")
        print(f"    α: {lapse_c[0]:.6f} → {lapse_c[-1]:.6f} (Δα={lapse_c[-1]-lapse_c[0]:+.3e})")
        print(f"    φ: {phi_c[0]:+.3e} → {phi_c[-1]:+.3e} (Δφ={phi_c[-1]-phi_c[0]:+.3e})")
        print(f"    K: {K_c[0]:+.3e} → {K_c[-1]:+.3e} (ΔK={K_c[-1]-K_c[0]:+.3e})")

    return dict(
        # Evolution results
        time=t, steps=steps, r=r, state=state,
        # Time series at center
        lapse_center=np.array(lapse_c),
        phi_center=np.array(phi_c),
        K_center=np.array(K_c),
        hrr_center=np.array(hrr_c),
        Ham_center=np.array(Ham_center),  # Hamiltonian constraint at center
        times=np.array(times),
        # Full snapshots
        times_detailed=np.array(times_detailed),
        states_detailed=states_detailed,
        Ham_snapshots=Ham_snapshots,  # Hamiltonian constraint profiles
        # Initial data
        tov=tov_data, rho0=rho0, pressure0=p, velocity0=v,
        # Parameters
        gamma=gamma, K_eos=K, rho_central=rho_central,
        TOV_Mass=M_star, TOV_Radius=R_star
    )

# -------------------- Helper for 2D plots from 1D spherical data --------------------
def create_2D_plot_from_1D_spherical(r_data, values_data, extent=2.0, resolution=100):
    """
    Create 2D plot data from 1D spherically symmetric data.

    Since Engrenage uses 1D spherical symmetry ,

    Parameters:
    -----------
    r_data : array
        Radial coordinates of 1D data
    values_data : array
        Values at each radial point
    extent : float
        Plot extent in units of M (from -extent to +extent)
    resolution : int
        Number of points in each direction

    Returns:
    --------
    X, Y, Z : arrays
        2D meshgrids for plotting
    """
    from scipy.interpolate import interp1d

    # Create 2D grid
    x = np.linspace(-extent, extent, resolution)
    y = np.linspace(-extent, extent, resolution)
    X, Y = np.meshgrid(x, y)

    # Calculate radius at each 2D point
    R = np.sqrt(X**2 + Y**2)

    # Interpolate 1D data to 2D
    # Use linear interpolation, extrapolate with nearest value
    interp_func = interp1d(r_data, values_data, kind='linear',
                          bounds_error=False, fill_value=(values_data[0], values_data[-1]))

    # Apply interpolation
    Z = interp_func(R)

    return X, Y, Z


# -------------------- Plotting Functions --------------------
def plot_hwh_results(result, save_plots=True, plot_dir="hwh_plots"):
    """
    Generate plots.

    Plots:
    1. Time evolution at center (φ, K, α vs t)
    2. Spatial profiles at different times
    3. Drift/error analysis

    These match Fig. 1 from Baumgarte, Hughes & Shapiro paper.
    """
    import os
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)

    # Extract data
    times = result['times']
    times_detailed = result['times_detailed']
    states_detailed = result['states_detailed']
    r = result['r']
    tov = result['tov']
    TOV_Mass = result['TOV_Mass']

    # Normalize time by TOV mass
    times_M = times / TOV_Mass
    times_detailed_M = times_detailed / TOV_Mass

    # ==================== PLOT 1: Time Evolution ====================
    print("\nGenerating time evolution plots...")
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))

    # Conformal factor φ vs time
    axes[0].plot(times_M, result['phi_center'], 'b-', linewidth=2)
    axes[0].set_ylabel('φ (conformal factor)', fontsize=12)
    axes[0].set_title('Hydro-without-Hydro Test: Central Evolution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, times_M[-1])

    # Extrinsic curvature K vs time (KEY FIGURE!)
    axes[1].plot(times_M, result['K_center'], 'g-', linewidth=2)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('K (extrinsic curvature)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, times_M[-1])
    # Add text showing maximum deviation
    K_max = np.max(np.abs(result['K_center']))
    axes[1].text(0.95, 0.95, f'max|K| = {K_max:.2e}',
                transform=axes[1].transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Lapse α vs time
    axes[2].plot(times_M, result['lapse_center'], 'r-', linewidth=2)
    axes[2].axhline(y=result['lapse_center'][0], color='r', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('α (lapse)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, times_M[-1])

    # Hamiltonian constraint vs time (NEW!)
    if 'Ham_center' in result:
        axes[3].semilogy(times_M, np.abs(result['Ham_center']) + 1e-16, 'purple', linewidth=2)
        axes[3].set_ylabel('|H| (Ham. constraint)', fontsize=12)
        axes[3].grid(True, alpha=0.3, which='both')
        axes[3].set_xlim(0, times_M[-1])
        Ham_max = np.max(np.abs(result['Ham_center']))
        axes[3].text(0.95, 0.95, f'max|H| = {Ham_max:.2e}',
                    transform=axes[3].transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[-1].set_xlabel('t/M', fontsize=12)

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{plot_dir}/hwh_time_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ==================== PLOT 2: Spatial Profiles ====================
    print("Generating spatial profile plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Select snapshots to plot
    n_snapshots = min(5, len(states_detailed))
    indices = np.linspace(0, len(states_detailed)-1, n_snapshots, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_snapshots))

    for i, idx in enumerate(indices):
        state_snap = states_detailed[idx]
        time_snap = times_detailed_M[idx]
        label = f't/M = {time_snap:.2f}'

        # Lapse α(r)
        axes[0,0].plot(r/TOV_Mass, state_snap[idx_lapse, :],
                      color=colors[i], linewidth=2, label=label)

        # Conformal factor φ(r)
        axes[0,1].plot(r/TOV_Mass, state_snap[idx_phi, :],
                      color=colors[i], linewidth=2, label=label)

        # Extrinsic curvature K(r)
        axes[1,0].plot(r/TOV_Mass, state_snap[idx_K, :],
                      color=colors[i], linewidth=2, label=label)

        # Metric component h_rr(r)
        axes[1,1].plot(r/TOV_Mass, state_snap[idx_hrr, :],
                      color=colors[i], linewidth=2, label=label)

    # Add initial TOV lapse
    if 'r' in tov and 'alpha' in tov:
        alpha_interp = interp1d(tov['r'], tov['alpha'], kind='linear',
                               bounds_error=False,
                               fill_value=(tov['alpha'][0], tov['alpha'][-1]))
        axes[0,0].plot(r/TOV_Mass, alpha_interp(r), 'k--',
                      linewidth=1.5, alpha=0.7, label='TOV initial')

    # Format axes
    axes[0,0].set_xlabel('r/M'); axes[0,0].set_ylabel('α'); axes[0,0].set_title('Lapse Function')
    axes[0,1].set_xlabel('r/M'); axes[0,1].set_ylabel('φ'); axes[0,1].set_title('Conformal Factor')
    axes[1,0].set_xlabel('r/M'); axes[1,0].set_ylabel('K'); axes[1,0].set_title('Extrinsic Curvature')
    axes[1,1].set_xlabel('r/M'); axes[1,1].set_ylabel('h_rr'); axes[1,1].set_title('Metric Component h_rr')

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{plot_dir}/hwh_spatial_profiles.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ==================== PLOT 3: Stability Analysis ====================
    print("Generating stability analysis plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Calculate drifts
    phi_drift = result['phi_center'] - result['phi_center'][0]
    K_drift = result['K_center']  # K should stay near 0
    lapse_drift = result['lapse_center'] - result['lapse_center'][0]

    # φ drift
    axes[0,0].semilogy(times_M, np.abs(phi_drift) + 1e-16, 'b-', linewidth=2)
    axes[0,0].set_xlabel('t/M'); axes[0,0].set_ylabel('|Δφ|')
    axes[0,0].set_title('Conformal Factor Drift')
    axes[0,0].grid(True, alpha=0.3, which='both')

    # K evolution (should remain near zero)
    axes[0,1].plot(times_M, K_drift, 'g-', linewidth=2)
    axes[0,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0,1].set_xlabel('t/M'); axes[0,1].set_ylabel('K')
    axes[0,1].set_title('Extrinsic Curvature (should ≈ 0)')
    axes[0,1].grid(True, alpha=0.3)

    # α drift
    axes[1,0].semilogy(times_M, np.abs(lapse_drift) + 1e-16, 'r-', linewidth=2)
    axes[1,0].set_xlabel('t/M'); axes[1,0].set_ylabel('|Δα|')
    axes[1,0].set_title('Lapse Drift')
    axes[1,0].grid(True, alpha=0.3, which='both')

    # L∞ norm of φ over time
    l_inf_norms = []
    for state_snap in states_detailed:
        norm = np.max(np.abs(state_snap[idx_phi, :]))
        l_inf_norms.append(norm)

    axes[1,1].semilogy(times_detailed_M, np.array(l_inf_norms) + 1e-16, 'm-', linewidth=2)
    axes[1,1].set_xlabel('t/M'); axes[1,1].set_ylabel('||φ||∞')
    axes[1,1].set_title('L∞ Norm of Conformal Factor')
    axes[1,1].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{plot_dir}/hwh_stability_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ==================== PLOT 4: 2D Hamiltonian Constraint===================
    if 'Ham_snapshots' in result:
        print("Generating 2D Hamiltonian constraint plots...")

        # Create figure with subplots for different times
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        # Select snapshots to plot
        n_snapshots = min(4, len(result['Ham_snapshots']))
        indices = np.linspace(0, len(result['Ham_snapshots'])-1, n_snapshots, dtype=int)

        for i, idx in enumerate(indices):
            if i >= 4: break  # Only plot first 4

            Ham_profile = result['Ham_snapshots'][idx]
            time_snap = result['times_detailed'][idx] if idx < len(result['times_detailed']) else 0

            # Create 2D data from 1D spherical profile
            # Note: the reference notebook uses y/M and z/M, we interpolate from our 1D solution
            extent_M = 2.0 * result['TOV_Radius'] / result['TOV_Mass']  # Plot extent in units of M

            X, Y, Ham_2D = create_2D_plot_from_1D_spherical(
                result['r'] / result['TOV_Mass'],  # Convert r to units of M
                np.log10(np.abs(Ham_profile) + 1e-16),  # log10 of |H|
                extent=extent_M,
                resolution=100
            )

            # Plot
            im = axes[i].contourf(X, Y, Ham_2D, levels=20, cmap='viridis')
            axes[i].set_title(f't/M = {time_snap/result["TOV_Mass"]:.2f}')
            axes[i].set_xlabel('y/M')
            axes[i].set_ylabel('z/M')
            axes[i].set_aspect('equal')

            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[i])
            cbar.set_label(r'$\log_{10}|H|$', rotation=270, labelpad=15)

            # Add circle showing stellar radius
            circle = Circle((0, 0), result['TOV_Radius']/result['TOV_Mass'],
                           color='red', fill=False, linestyle='--', linewidth=1)
            axes[i].add_patch(circle)

        plt.suptitle('2D Hamiltonian Constraint Violation (log₁₀|H|)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_plots:
            plt.savefig(f"{plot_dir}/hwh_hamiltonian_2D.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nPlots saved to {plot_dir}/")
    return result


def plot_baumgarte_paper_figures(result, save_plots=True, plot_dir="hwh_plots"):
    """
    Replicate figures from Baumgarte, Hughes & Shapiro (1999) paper.

    Figure 1: φ and K at center vs t/M
    Shows long-term evolution settling into numerical equilibrium.
    """
    import os
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)

    times = result['times']
    TOV_Mass = result['TOV_Mass']
    times_M = times / TOV_Mass

    # ==================== FIGURE 1: φ and K Evolution (Paper Figure 1) ====================
    print("\nGenerating Baumgarte paper Figure 1: φ and K at center...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Top panel: φ at center
    ax1.plot(times_M, result['phi_center'], 'k-', linewidth=1.5)
    ax1.set_ylabel(r'$\phi$', fontsize=14)
    ax1.set_title('Evolution of conformal exponent and trace of extrinsic curvature',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, times_M[-1])

    # Inset showing early evolution (first 100 time units)
    if times_M[-1] > 100:
        left, bottom, width, height = [0.55, 0.62, 0.35, 0.25]
        ax1_inset = fig.add_axes([left, bottom, width, height])
        mask_early = times_M <= 100
        ax1_inset.plot(times_M[mask_early], result['phi_center'][mask_early], 'k-', linewidth=1)
        ax1_inset.set_xlim(0, 100)
        ax1_inset.set_xlabel('t/M', fontsize=10)
        ax1_inset.grid(True, alpha=0.3)
        ax1_inset.tick_params(labelsize=9)

    # Bottom panel: K at center
    ax2.plot(times_M, result['K_center'], 'k-', linewidth=1.5)
    ax2.set_ylabel('K', fontsize=14)
    ax2.set_xlabel('t/M', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, times_M[-1])
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Inset showing early evolution
    if times_M[-1] > 100:
        left, bottom, width, height = [0.55, 0.18, 0.35, 0.25]
        ax2_inset = fig.add_axes([left, bottom, width, height])
        ax2_inset.plot(times_M[mask_early], result['K_center'][mask_early], 'k-', linewidth=1)
        ax2_inset.set_xlim(0, 100)
        ax2_inset.set_xlabel('t/M', fontsize=10)
        ax2_inset.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2_inset.grid(True, alpha=0.3)
        ax2_inset.tick_params(labelsize=9)

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{plot_dir}/baumgarte_fig1_phi_K_evolution.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: {plot_dir}/baumgarte_fig1_phi_K_evolution.png")
    plt.close()


def plot_baumgarte_fig2_convergence(results_list, resolutions, save_plots=True, plot_dir="hwh_plots"):
    """
    Replicate Baumgarte paper Figure 2: Convergence test for K at center.

    Shows that the code converges to second order until surface effects
    spoil convergence at later times (t > 1).

    Args:
        results_list: List of results dicts from different resolutions
        resolutions: List of grid resolutions used (e.g., [16, 32, 64])
    """
    import os
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)

    print("\nGenerating Baumgarte paper Figure 2: Convergence test...")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Note: Analytic solution is K = 0
    # We plot scaled errors to show second-order convergence

    # Plot each resolution with appropriate scaling
    # For second-order convergence, error should scale as h^2
    # So when we multiply coarse grid by 4, 16, etc., scaled errors should overlap

    n_res = len(resolutions)
    colors = ['k-', 'k--', 'k:']
    labels = []

    for i, (res, Nr) in enumerate(zip(results_list, resolutions)):
        times = res['times'] / res['TOV_Mass']
        K_center = res['K_center']

        # Calculate scaling factor relative to finest grid
        if i == n_res - 1:
            # Finest grid - no scaling
            scaling = 1.0
            label = f"16×({Nr})³"
        elif i == n_res - 2:
            # Middle resolution
            scaling = 4.0
            label = f"4×({Nr})³"
        else:
            # Coarsest resolution
            scaling = 1.0
            label = f"({Nr})³"

        ax.plot(times, K_center * scaling, colors[i], linewidth=1.5, label=label)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('t', fontsize=14)
    ax.set_ylabel('K', fontsize=14)
    ax.set_title('Convergence test for K at the center', fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3)

    # Add annotation
    ax.text(0.5, 0.05, 'Note that the analytic solution is K = 0',
            transform=ax.transAxes, ha='left', fontsize=10, style='italic')

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{plot_dir}/baumgarte_fig2_convergence.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: {plot_dir}/baumgarte_fig2_convergence.png")
    plt.close()


def plot_baumgarte_fig3_outer_boundary(results_list, ob_locations, save_plots=True, plot_dir="hwh_plots"):
    """
    Replicate Baumgarte paper Figure 3: K at center for different outer boundary locations.

    Shows how outer boundary errors propagate inward and identifies
    three sources of error:
    1. Local finite-difference error (t ~ 0.4)
    2. Surface effects (t ~ 2)
    3. Outer boundary (t ~ 4 for OB=2)

    Args:
        results_list: List of results dicts with different outer boundaries
        ob_locations: List of outer boundary locations (e.g., [1, 2, 4])
    """
    import os
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)

    print("\nGenerating Baumgarte paper Figure 3: Outer boundary analysis...")

    fig, ax = plt.subplots(figsize=(10, 6))

    linestyles = ['k--', 'k-.', 'k-']
    max_errors = []

    for i, (res, ob) in enumerate(zip(results_list, ob_locations)):
        times = res['times'] / res['TOV_Mass']
        K_center = res['K_center']

        label = f"OB at {ob}"
        ax.plot(times, K_center, linestyles[i], linewidth=1.5, label=label)

        # Find maximum error (where each OB starts affecting the solution)
        # This happens when the domain of dependence reaches the center
        if i < len(results_list) - 1:
            # Find where this curve starts to deviate significantly from the next one
            next_K = results_list[i + 1]['K_center']
            min_len = min(len(K_center), len(next_K))
            diff = np.abs(K_center[:min_len] - next_K[:min_len])
            if len(diff) > 0:
                max_idx = np.argmax(diff)
                max_errors.append((times[max_idx], K_center[max_idx]))

    # Plot dots marking maximum errors
    for t, K in max_errors:
        ax.plot(t, K, 'ko', markersize=6)

    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('t', fontsize=14)
    ax.set_ylabel('K', fontsize=14)
    ax.set_title('Evolution of K at center for different outer boundary locations', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 8)

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{plot_dir}/baumgarte_fig3_outer_boundary.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: {plot_dir}/baumgarte_fig3_outer_boundary.png")
    plt.close()


def run_convergence_test(resolutions=[72, 94], cfl=0.1, plot=True):
    """
    Run convergence 

    Tests that Hamiltonian constraint violation converges to zero
    with increasing resolution at the expected order.

    Parameters:
    -----------
    resolutions : list
        List of radial resolutions to test [Nr1, Nr2]
    cfl : float
        CFL factor for evolution
    plot : bool
        Whether to generate convergence plot

    Returns:
    --------
    dict: Convergence test results
    """
    print("\n" + "="*60)
    print("CONVERGENCE TEST ")
    print("="*60)

    results = []

    for Nr in resolutions:
        print(f"\nRunning resolution Nr = {Nr}...")

        # Calculate dr from resolution
        # Domain size is approximately 2 * R_star
        R_approx = 1.6  # Approximate for our TOV star
        dr = R_approx / Nr

        # Run simulation
        result = run_hwh_test(dr=dr, cfl=cfl, progress=False,
                             save_interval=1000)  # Less frequent saves for convergence test

        results.append(result)

        print(f"  Completed: t_final/M = {result['time']/result['TOV_Mass']:.3f}")
        print(f"  Max |K| = {np.max(np.abs(result['K_center'])):.2e}")
        print(f"  Final φ drift = {result['phi_center'][-1] - result['phi_center'][0]:.2e}")

    if plot and len(results) >= 2:
        print("\nGenerating convergence plot...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot K evolution for different resolutions
        for i, (Nr, res) in enumerate(zip(resolutions, results)):
            times_M = res['times'] / res['TOV_Mass']
            ax.plot(times_M, res['K_center'], linewidth=2,
                   label=f'Nr = {Nr}')

        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('t/M', fontsize=12)
        ax.set_ylabel('K (extrinsic curvature)', fontsize=12)
        ax.set_title('Convergence Test: K should remain ≈ 0', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Calculate convergence factor
        if len(results) == 2:
            # Compare final K values
            K1_final = results[0]['K_center'][-1]
            K2_final = results[1]['K_center'][-1]
            ratio = resolutions[0] / resolutions[1]
            expected_factor = ratio**4  # 4th order convergence

            print(f"\nConvergence Analysis:")
            print(f"  K_final(Nr={resolutions[0]}) = {K1_final:.2e}")
            print(f"  K_final(Nr={resolutions[1]}) = {K2_final:.2e}")
            print(f"  Expected ratio (4th order): {expected_factor:.2f}")
            print(f"  Actual ratio: {K1_final/K2_final:.2f}")

        plt.tight_layout()
        plt.savefig("hwh_convergence_test.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Also generate Hamiltonian profile convergence at final time
        try:
            import numpy as _np
            import matplotlib.pyplot as _plt
            # Use final stored profiles
            r1 = results[0]['r'] / results[0]['TOV_Mass']
            r2 = results[1]['r'] / results[1]['TOV_Mass']
            H1 = results[0]['Ham_snapshots'][-1]
            H2 = results[1]['Ham_snapshots'][-1]
            eps = 1e-30
            logH1 = _np.log10(_np.abs(H1) + eps)
            logH2 = _np.log10(_np.abs(H2) + eps)
            Nr1, Nr2 = resolutions[0], resolutions[1]
            if Nr1 > Nr2:
                r_hi, logH_hi, label_hi = r1, logH1, f"Nr={Nr1}"
                r_lo, logH_lo, label_lo, ratio = r2, logH2, f"Nr={Nr2}", Nr2 / Nr1
            else:
                r_hi, logH_hi, label_hi = r2, logH2, f"Nr={Nr2}"
                r_lo, logH_lo, label_lo, ratio = r1, logH1, f"Nr={Nr1}", Nr1 / Nr2
            logH_lo_scaled = logH_lo + 4.0 * _np.log10(max(ratio, 1e-16))
            fig2, ax2 = _plt.subplots(figsize=(8, 6))
            ax2.plot(r_hi, logH_hi, 'k-', label=label_hi)
            ax2.plot(r_lo, logH_lo_scaled, 'k--', label=f"{label_lo}, mult by ({ratio:.2f})^4")
            ax2.set_title(r"4$^{\mathrm{th}}$-order Convergence, at t/M=1.8")
            ax2.set_xlabel('y/M')
            ax2.set_ylabel(r'$\log_{10}$ (Ham. Constraint Violation)')
            ax2.grid(True, alpha=0.3)
            lg = ax2.legend(loc='lower right', shadow=True, fontsize='x-large')
            try:
                lg.get_frame().set_facecolor('orange')
            except Exception:
                pass
            _plt.tight_layout(); _plt.savefig("hwh_convergence_hamiltonian_profile.png", dpi=150, bbox_inches='tight'); _plt.close(fig2)
        except Exception as e:
            print(f"[warn] Could not generate Hamiltonian convergence profile: {e}")

    return results

# ==================== Main Execution  ====================
if __name__ == "__main__":
    print("="*60)
    print("HYDRO-WITHOUT-HYDRO TEST")
    print("Replicating notebook")
    print("="*60)

    #
    print("\nOption 1: Running single simulation...")
    result = run_hwh_test(
        gamma=2.0,            # Polytropic index (n=1)
        K=1.0,                # Polytropic constant
        rho_central=0.2,      # Central density (Baumgarte paper)
        cfl=0.1,              # CFL factor
        dr=0.01,              # Spatial resolution
        progress=True         # Show progress
    )

    # Generate plots
    plot_hwh_results(result, save_plots=True)

    # Generate Baumgarte paper Figure 1
    plot_baumgarte_paper_figures(result, save_plots=True)

    # Option 2: Convergence test
    print("\nOption 2: Running convergence test...")
    convergence_results = run_convergence_test(
        resolutions=[72, 94],
        cfl=0.1,
        plot=True
    )

    # Print summary
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)
    print("\nKey Results:")
    print(f"  TOV Mass: M = {result['TOV_Mass']:.6f}")
    print(f"  TOV Radius: R = {result['TOV_Radius']:.6f}")
    print(f"  Final time: t/M = {result['time']/result['TOV_Mass']:.3f}")
    print(f"  Max |K|: {np.max(np.abs(result['K_center'])):.2e}")
    print(f"  φ drift: {result['phi_center'][-1] - result['phi_center'][0]:.2e}")
    print(f"  α drift: {result['lapse_center'][-1] - result['lapse_center'][0]:.2e}")
    if 'Ham_center' in result:
        print(f"  Max |H|: {np.max(np.abs(result['Ham_center'])):.2e}")
        print(f"  Initial |H|: {np.abs(result['Ham_center'][0]):.2e}")
        print(f"  Final |H|: {np.abs(result['Ham_center'][-1]):.2e}")
    print("\nExpected behavior:")
    print("  - K should remain ≈ 0 (TOV is static solution)")
    print("  - φ and α should have minimal drift")
    print("  - Errors should converge at 4th order with resolution")
    print("\nSee generated plots in hwh_plots/ directory")
