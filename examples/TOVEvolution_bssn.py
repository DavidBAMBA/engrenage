import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.integrate import ode, trapezoid
from scipy.interpolate import interp1d

# Engrenage core
sys.path.insert(0, '/home/yo/repositories/engrenage')
from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground, i_r, i_t, i_p

# BSSN
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import (NUM_BSSN_VARS, idx_phi, idx_hrr, idx_htt, idx_hpp,
                                             idx_K, idx_arr, idx_att, idx_app, idx_lapse)
from source.bssn.tensoralgebra import get_bar_gamma_LL
from source.bssn.bssnrhs import get_bssn_rhs

# Hydro
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import PolytropicEOS, IdealGasEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLERiemannSolver
from source.matter.hydro.cons2prim import prim_to_cons
from source.matter.hydro.atmosphere import AtmosphereParams  
from source.bssn.tensoralgebra import get_bar_gamma_LL

from examples.tov_solver import TOVSolver
import examples.tov_initial_data_interpolated as tov_id

# Directory for saving plots
PLOT_DIR = "tov-bssn-plots"


def diagnose_t0_residuals(state_2d, grid, background, hydro):
    """Compute and print t=0 RHS residuals (especially dS_r/dt) to locate imbalance.

    This helps identify discrete hydrostatic imbalance (typically strongest near the surface).
    """
    # Build BSSN containers (Cowling)
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
    bssn_d1 = grid.get_d1_metric_quantities(state_2d)

    # Matter vars
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    rhs_D, rhs_Sr, rhs_tau = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1, background)

    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
    r_int = grid.r[interior]

    # Report maxima
    i_max = NUM_GHOSTS + int(np.argmax(np.abs(rhs_Sr[interior])))
    print("\nInitial RHS diagnostics (t=0):")
    print(f"  max |dS_r/dt| at r={grid.r[i_max]:.6f} (i={i_max}) -> {rhs_Sr[i_max]:.3e}")
    print(f"  max |dD/dt|   = {np.max(np.abs(rhs_D[interior])):.3e}")
    print(f"  max |dτ/dt|   = {np.max(np.abs(rhs_tau[interior])):.3e}")

    # Coarse surface estimate from primitives
    prim = hydro._get_primitives(bssn_vars, grid.r, grid=grid)
    mask_interior = prim['rho0'] > 1e-6
    if np.any(mask_interior):
        i_surf = np.where(mask_interior)[0][-1]
        print(f"  estimated stellar surface near r={grid.r[i_surf]:.6f} (i={i_surf})")
        window = slice(max(NUM_GHOSTS, i_surf-5), min(grid.N-NUM_GHOSTS, i_surf+6))
        print("  dS_r/dt in 11-pt window around surface:")
        for ii in range(window.start, window.stop):
            print(f"    i={ii:5d}, r={grid.r[ ii ]:8.5f}, dS_r/dt={rhs_Sr[ii]: .3e}")
    else:
        print("  WARNING: could not locate interior points above threshold.")


def plot_first_step(state_t0, state_t1, grid, hydro, tov_solution=None):
    """Plot only t=0 vs t=1×dt to inspect the first update."""
    bssn_0 = BSSNVars(grid.N)
    bssn_0.set_bssn_vars(state_t0[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t0, bssn_0, grid)
    prim_0 = hydro._get_primitives(bssn_0, grid.r, grid=grid)

    bssn_1 = BSSNVars(grid.N)
    bssn_1.set_bssn_vars(state_t1[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t1, bssn_1, grid)
    prim_1 = hydro._get_primitives(bssn_1, grid.r, grid=grid)

    r_int = grid.r[NUM_GHOSTS:-NUM_GHOSTS]

    # Compute mass using integral M(r) = 4π ∫_0^r ρ r^2 dr
    from scipy.integrate import cumulative_trapezoid
    rho_0 = prim_0['rho0']
    rho_1 = prim_1['rho0']
    # Full grid mass calculation
    M_0_full = 4.0 * np.pi * cumulative_trapezoid(rho_0 * grid.r**2, grid.r, initial=0.0)
    M_1_full = 4.0 * np.pi * cumulative_trapezoid(rho_1 * grid.r**2, grid.r, initial=0.0)

    # ========================================================================
    # ANALYSIS: Print values at key locations
    # ========================================================================
    print("\n" + "="*80)
    print("FIRST STEP ANALYSIS: Changes from t=0 to t=1×dt")
    print("="*80)

    # Find indices for key locations
    idx_center = NUM_GHOSTS  # First interior point (center)
    idx_left = NUM_GHOSTS + 10  # Near left boundary (interior)

    # Find index closest to stellar radius (if TOV solution provided)
    if tov_solution is not None:
        R_star = tov_solution['R']
        idx_star = np.argmin(np.abs(grid.r - R_star))
        print(f"\nStellar radius R = {R_star:.4f} at grid index {idx_star} (r = {grid.r[idx_star]:.4f})")
    else:
        # Estimate stellar surface as where density drops below threshold
        rho_threshold = 1e-6
        interior_mask = rho_0 > rho_threshold
        if np.any(interior_mask):
            idx_star = np.where(interior_mask)[0][-1]
            R_star = grid.r[idx_star]
            print(f"\nEstimated stellar surface at r = {R_star:.4f} (ρ < {rho_threshold})")
        else:
            idx_star = len(grid.r) // 2
            R_star = grid.r[idx_star]
            print(f"\nUsing midpoint as reference: r = {R_star:.4f}")

    idx_right = -NUM_GHOSTS - 1  # Last interior point (far boundary)

    def analyze_point(idx, label):
        """Analyze all quantities at a given index."""
        r = grid.r[idx]
        print(f"\n{'-'*80}")
        print(f"{label}: r = {r:.6f} (index {idx})")
        print(f"{'-'*80}")

        # Hydro primitives
        rho0_0, rho0_1 = rho_0[idx], rho_1[idx]
        p0, p1 = prim_0['p'][idx], prim_1['p'][idx]
        vr0, vr1 = prim_0['vr'][idx], prim_1['vr'][idx]

        # BSSN variables
        phi0, phi1 = state_t0[idx_phi, idx], state_t1[idx_phi, idx]
        alpha0, alpha1 = state_t0[idx_lapse, idx], state_t1[idx_lapse, idx]
        hrr0, hrr1 = state_t0[idx_hrr, idx], state_t1[idx_hrr, idx]
        K0, K1 = state_t0[idx_K, idx], state_t1[idx_K, idx]
        arr0, arr1 = state_t0[idx_arr, idx], state_t1[idx_arr, idx]

        # Mass
        M0, M1 = M_0_full[idx], M_1_full[idx]

        # Conservative variables
        D0 = state_t0[NUM_BSSN_VARS + 0, idx]
        D1 = state_t1[NUM_BSSN_VARS + 0, idx]
        Sr0 = state_t0[NUM_BSSN_VARS + 1, idx]
        Sr1 = state_t1[NUM_BSSN_VARS + 1, idx]
        tau0 = state_t0[NUM_BSSN_VARS + 2, idx]
        tau1 = state_t1[NUM_BSSN_VARS + 2, idx]

        def rel_change(v0, v1):
            """Relative change with safe division."""
            if abs(v0) < 1e-30:
                return f"{v1:.6e}" if abs(v1) > 1e-30 else "0"
            return f"{(v1-v0)/v0:.6e}"

        def abs_change(v0, v1):
            """Absolute change."""
            return f"{v1-v0:.6e}"

        print(f"\nPRIMITIVE VARIABLES:")
        print(f"  ρ₀:     {rho0_0:.8e} → {rho0_1:.8e}   Δrel = {rel_change(rho0_0, rho0_1)}")
        print(f"  P:      {p0:.8e} → {p1:.8e}   Δrel = {rel_change(p0, p1)}")
        print(f"  vʳ:     {vr0:.8e} → {vr1:.8e}   Δabs = {abs_change(vr0, vr1)}")

        print(f"\nCONSERVATIVE VARIABLES:")
        print(f"  D:      {D0:.8e} → {D1:.8e}   Δrel = {rel_change(D0, D1)}")
        print(f"  Sʳ:     {Sr0:.8e} → {Sr1:.8e}   Δabs = {abs_change(Sr0, Sr1)}")
        print(f"  τ:      {tau0:.8e} → {tau1:.8e}   Δrel = {rel_change(tau0, tau1)}")

        print(f"\nBSSN VARIABLES:")
        print(f"  φ:      {phi0:.8e} → {phi1:.8e}   Δrel = {rel_change(phi0, phi1)}")
        print(f"  α:      {alpha0:.8e} → {alpha1:.8e}   Δrel = {rel_change(alpha0, alpha1)}")
        print(f"  hʳʳ:    {hrr0:.8e} → {hrr1:.8e}   Δrel = {rel_change(hrr0, hrr1)}")
        print(f"  K:      {K0:.8e} → {K1:.8e}   Δabs = {abs_change(K0, K1)}")
        print(f"  aʳʳ:    {arr0:.8e} → {arr1:.8e}   Δabs = {abs_change(arr0, arr1)}")

        print(f"\nDERIVED QUANTITIES:")
        print(f"  M(r):   {M0:.8e} → {M1:.8e}   Δrel = {rel_change(M0, M1)}")

        # Compute specific internal energy and enthalpy
        eps0 = hydro.eos.eps_from_rho_p(rho0_0, p0)
        eps1 = hydro.eos.eps_from_rho_p(rho0_1, p1)
        h0 = 1.0 + eps0 + p0/max(rho0_0, 1e-30)
        h1 = 1.0 + eps1 + p1/max(rho0_1, 1e-30)
        print(f"  ε:      {eps0:.8e} → {eps1:.8e}   Δrel = {rel_change(eps0, eps1)}")
        print(f"  h:      {h0:.8e} → {h1:.8e}   Δrel = {rel_change(h0, h1)}")

    # Analyze key points
    analyze_point(idx_center, "CENTER (r≈0)")
    analyze_point(idx_left, "LEFT INTERIOR")
    analyze_point(idx_star, "STELLAR SURFACE")
    analyze_point(idx_right, "FAR BOUNDARY")

    # Global statistics
    print(f"\n{'='*80}")
    print("GLOBAL STATISTICS:")
    print(f"{'='*80}")

    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

    def safe_rel_error(arr0, arr1):
        """Compute relative error safely."""
        mask = np.abs(arr0) > 1e-30
        rel_err = np.zeros_like(arr0)
        rel_err[mask] = np.abs((arr1[mask] - arr0[mask]) / arr0[mask])
        return rel_err

    rho_rel_err = safe_rel_error(rho_0[interior], rho_1[interior])
    p_rel_err = safe_rel_error(prim_0['p'][interior], prim_1['p'][interior])
    phi_rel_err = safe_rel_error(state_t0[idx_phi, interior], state_t1[idx_phi, interior])
    alpha_rel_err = safe_rel_error(state_t0[idx_lapse, interior], state_t1[idx_lapse, interior])

    print(f"\nMax relative changes (interior points):")
    print(f"  ρ₀:     {np.max(rho_rel_err):.6e}")
    print(f"  P:      {np.max(p_rel_err):.6e}")
    print(f"  φ:      {np.max(phi_rel_err):.6e}")
    print(f"  α:      {np.max(alpha_rel_err):.6e}")
    print(f"  |vʳ|:   {np.max(np.abs(prim_1['vr'][interior])):.6e}")

    print(f"\nLocations of maximum changes:")
    idx_max_rho = NUM_GHOSTS + np.argmax(rho_rel_err)
    idx_max_p = NUM_GHOSTS + np.argmax(p_rel_err)
    idx_max_v = NUM_GHOSTS + np.argmax(np.abs(prim_1['vr'][interior]))
    print(f"  Max Δρ/ρ at r = {grid.r[idx_max_rho]:.6f} (index {idx_max_rho})")
    print(f"  Max ΔP/P at r = {grid.r[idx_max_p]:.6f} (index {idx_max_p})")
    print(f"  Max |vʳ| at r = {grid.r[idx_max_v]:.6f} (index {idx_max_v})")

    print("="*80 + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Density
    axes[0, 0].semilogy(r_int, prim_0['rho0'][NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label='t=0')
    axes[0, 0].semilogy(r_int, prim_1['rho0'][NUM_GHOSTS:-NUM_GHOSTS], 'r--', linewidth=1.7, label='t=1×dt')
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel(r'$\rho_0$')
    axes[0, 0].set_title('Density: first step')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Pressure
    axes[0, 1].semilogy(r_int, np.maximum(prim_0['p'][NUM_GHOSTS:-NUM_GHOSTS], 1e-20), 'b-', linewidth=2, label='t=0')
    axes[0, 1].semilogy(r_int, np.maximum(prim_1['p'][NUM_GHOSTS:-NUM_GHOSTS], 1e-20), 'r--', linewidth=1.7, label='t=1×dt')
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure: first step')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Velocity
    axes[0, 2].plot(r_int, prim_0['vr'][NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label='t=0')
    axes[0, 2].plot(r_int, prim_1['vr'][NUM_GHOSTS:-NUM_GHOSTS], 'r--', linewidth=1.7, label='t=1×dt')
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel(r'$v^r$')
    axes[0, 2].set_title('Velocity: first step')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Lapse
    axes[1, 0].plot(r_int, state_t0[idx_lapse, NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label='t=0')
    axes[1, 0].plot(r_int, state_t1[idx_lapse, NUM_GHOSTS:-NUM_GHOSTS], 'r--', linewidth=1.7, label='t=1×dt')
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel(r'$\alpha$')
    axes[1, 0].set_title('Lapse: first step')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Mass
    axes[1, 1].plot(r_int, M_0_full[NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label='t=0')
    axes[1, 1].plot(r_int, M_1_full[NUM_GHOSTS:-NUM_GHOSTS], 'r--', linewidth=1.7, label='t=1×dt')
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel('M(r)')
    axes[1, 1].set_title('Enclosed Mass: first step')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Phi
    axes[1, 2].plot(r_int, state_t0[idx_phi, NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label='t=0')
    axes[1, 2].plot(r_int, state_t1[idx_phi, NUM_GHOSTS:-NUM_GHOSTS], 'r--', linewidth=1.7, label='t=1×dt')
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel(r'$\phi$')
    axes[1, 2].set_title('Conformal Factor: first step')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'tov_first_step.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_surface_zoom(tov_solution, state_t0, state_t1, grid, hydro, window=0.5):
    """Zoom near the stellar surface R to compare t=0 vs t=1×dt.

    Plots overlays for (ρ0, P, v^r, D, S_r, τ) in a window [R−window, R+window].
    """
    R = float(tov_solution['R'])
    r = grid.r
    mask = (r >= R - window) & (r <= R + window)
    if not np.any(mask):
        return

    bssn0 = BSSNVars(grid.N)
    bssn0.set_bssn_vars(state_t0[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t0, bssn0, grid)
    prim0 = hydro._get_primitives(bssn0, r, grid=grid)

    bssn1 = BSSNVars(grid.N)
    bssn1.set_bssn_vars(state_t1[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t1, bssn1, grid)
    prim1 = hydro._get_primitives(bssn1, r, grid=grid)

    D0, Sr0, tau0 = state_t0[NUM_BSSN_VARS + 0, :], state_t0[NUM_BSSN_VARS + 1, :], state_t0[NUM_BSSN_VARS + 2, :]
    D1, Sr1, tau1 = state_t1[NUM_BSSN_VARS + 0, :], state_t1[NUM_BSSN_VARS + 1, :], state_t1[NUM_BSSN_VARS + 2, :]

    rZ = r[mask]
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    # Row 1: rho0, P, vr
    ax[0, 0].semilogy(rZ, np.maximum(prim0['rho0'][mask], 1e-20), 'b-', label='t=0')
    ax[0, 0].semilogy(rZ, np.maximum(prim1['rho0'][mask], 1e-20), 'r--', label='t=1×dt')
    ax[0, 0].axvline(R, color='gray', ls=':'); ax[0, 0].set_title('ρ0 (zoom)'); ax[0, 0].legend(); ax[0, 0].grid(True, alpha=0.3)

    ax[0, 1].semilogy(rZ, np.maximum(prim0['p'][mask], 1e-20), 'b-')
    ax[0, 1].semilogy(rZ, np.maximum(prim1['p'][mask], 1e-20), 'r--')
    ax[0, 1].axvline(R, color='gray', ls=':'); ax[0, 1].set_title('P (zoom)'); ax[0, 1].grid(True, alpha=0.3)

    ax[0, 2].plot(rZ, prim0['vr'][mask], 'b-')
    ax[0, 2].plot(rZ, prim1['vr'][mask], 'r--')
    ax[0, 2].axvline(R, color='gray', ls=':'); ax[0, 2].set_title('v^r (zoom)'); ax[0, 2].grid(True, alpha=0.3)

    # Row 2: D, Sr, tau
    ax[1, 0].semilogy(rZ, np.maximum(D0[mask], 1e-22), 'b-')
    ax[1, 0].semilogy(rZ, np.maximum(D1[mask], 1e-22), 'r--')
    ax[1, 0].axvline(R, color='gray', ls=':'); ax[1, 0].set_title('D (zoom)'); ax[1, 0].grid(True, alpha=0.3)

    ax[1, 1].plot(rZ, Sr0[mask], 'b-')
    ax[1, 1].plot(rZ, Sr1[mask], 'r--')
    ax[1, 1].axvline(R, color='gray', ls=':'); ax[1, 1].set_title('S_r (zoom)'); ax[1, 1].grid(True, alpha=0.3)

    ax[1, 2].semilogy(rZ, np.maximum(np.abs(tau0[mask]), 1e-22), 'b-')
    ax[1, 2].semilogy(rZ, np.maximum(np.abs(tau1[mask]), 1e-22), 'r--')
    ax[1, 2].axvline(R, color='gray', ls=':'); ax[1, 2].set_title('τ (zoom)'); ax[1, 2].grid(True, alpha=0.3)

    for a in ax.ravel():
        a.set_xlabel('r')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'tov_surface_zoom.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def get_rhs_full(t, y, grid, background, hydro):
    """RHS for full BSSN + hydro evolution (no Cowling approximation)."""
    state = y.reshape((grid.NUM_VARS, grid.N))
    grid.fill_boundaries(state)

    # Build BSSN containers from current state
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state[:NUM_BSSN_VARS, :])

    # Compute derivatives for BSSN RHS
    bssn_d1 = grid.get_d1_metric_quantities(state)
    bssn_d2 = grid.get_d2_metric_quantities(state)

    # Set matter variables
    hydro.set_matter_vars(state, bssn_vars, grid)

    # Get matter RHS
    hydro_rhs = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1, background)

    # Get energy-momentum tensor for BSSN evolution
    emtensor = hydro.get_emtensor(grid.r, bssn_vars, background)

    # Get BSSN RHS
    bssn_rhs = BSSNVars(grid.N)
    get_bssn_rhs(bssn_rhs, grid.r, bssn_vars, bssn_d1, bssn_d2, background, emtensor)

    # Combine into full RHS
    rhs = np.zeros_like(state)
    rhs[:NUM_BSSN_VARS, :] = bssn_rhs.set_bssn_state_vars()
    rhs[NUM_BSSN_VARS:, :] = hydro_rhs

    return rhs.flatten()


def _apply_atmosphere_reset(state_2d, grid, hydro, atmosphere, rho_threshold=None):
    """
    Apply atmosphere floors using IllinoisGRMHD strategy.

    Strategy (following IllinoisGRMHD/NRPy+):
    1. Recover primitives from conservatives (cons2prim)
    2. Apply floors to PRIMITIVES (ρ, v, P)
    3. Apply conservative variable consistency checks (tau floor, S^2 constraint)
    4. Recompute conservatives from floored primitives if needed

    This ensures thermodynamic consistency: floored conservatives correspond
    to a valid physical state.

    Args:
        state_2d: State vector (2D array)
        grid: Grid object
        hydro: Hydrodynamics object
        atmosphere: AtmosphereParams object
        rho_threshold: Threshold below which to apply atmosphere (default: 10 * rho_floor)

    Returns:
        state_2d with floors applied
    """
    from source.matter.hydro.atmosphere import FloorApplicator

    if rho_threshold is None:
        rho_threshold = 10.0 * atmosphere.rho_floor

    # Extract conservatives
    D = state_2d[NUM_BSSN_VARS + 0, :]
    Sr = state_2d[NUM_BSSN_VARS + 1, :]
    tau = state_2d[NUM_BSSN_VARS + 2, :]

    # Build BSSN geometry (needed for cons2prim and floor application)
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])

    # Get metric for floor application
    bar_gamma_LL = get_bar_gamma_LL(grid.r, bssn_vars.h_LL, hydro.background)
    phi = np.asarray(bssn_vars.phi, dtype=float)
    e4phi = np.exp(4.0 * phi)
    gamma_rr = e4phi * bar_gamma_LL[:, 0, 0]  # Physical γ_rr

    # Recover primitives (cons2prim already has some floor logic built in)
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    prim = hydro._get_primitives(bssn_vars, grid.r, grid=grid)

    rho0 = prim['rho0']
    vr = prim['vr']
    p = prim['p']

    # Create floor applicator
    floor_app = FloorApplicator(atmosphere, hydro.eos)

    # STEP 1: Apply primitive floors (ρ, v, P)
    rho0_floor, vr_floor, p_floor = floor_app.apply_primitive_floors(rho0, vr, p, gamma_rr)

    # STEP 2: Identify atmosphere regions (where there is NO real fluid)
    # In atmosphere: density is at floor and there should be NO motion
    atm_mask = D < rho_threshold

    # CRITICAL: In atmosphere, velocity MUST be zero (no fluid to move!)
    if np.any(atm_mask):
        vr_floor[atm_mask] = 0.0  # Force v=0 in atmosphere

    # STEP 3: Apply conservative consistency floors (tau, S_i constraints)
    D_floor, Sr_floor, tau_floor, cons_floor_applied = floor_app.apply_conservative_floors(
        D, Sr, tau, gamma_rr
    )

    # STEP 4: Recompute conservatives from floored primitives where needed
    # Identify points where primitive floors were applied
    prim_floor_applied = (
        (np.abs(rho0_floor - rho0) > 1e-14) |
        (np.abs(vr_floor - vr) > 1e-14) |
        (np.abs(p_floor - p) > 1e-14)
    )

    # Combine all masks to identify points that need floor application
    needs_floor = prim_floor_applied | atm_mask | cons_floor_applied

    if np.any(needs_floor):
        # Recompute conservatives from floored primitives ONLY at points that need it
        # In atmosphere: vr_floor=0, so this gives Sr=0 (no momentum in vacuum)
        D_new, Sr_new, tau_new = prim_to_cons(rho0_floor, vr_floor, p_floor, gamma_rr, hydro.eos)

        # Apply conservative floors again to ensure consistency
        D_new, Sr_new, tau_new, _ = floor_app.apply_conservative_floors(
            D_new, Sr_new, tau_new, gamma_rr
        )

        # Update state ONLY at points that need floors (selective update)
        state_2d[NUM_BSSN_VARS + 0, needs_floor] = D_new[needs_floor]
        state_2d[NUM_BSSN_VARS + 1, needs_floor] = Sr_new[needs_floor]
        state_2d[NUM_BSSN_VARS + 2, needs_floor] = tau_new[needs_floor]

    return state_2d


def rk4_step(state_flat, dt, grid, background, hydro, atmosphere):
    """
    Single RK4 (classical 4th order Runge-Kutta) timestep with atmosphere reset.
    Full BSSN + hydro evolution (no Cowling approximation).

    Args:
        atmosphere: AtmosphereParams object
    """
    # Stage 1
    k1 = get_rhs_full(0, state_flat, grid, background, hydro)

    # Stage 2
    state_2 = state_flat + 0.5 * dt * k1
    k2 = get_rhs_full(0, state_2, grid, background, hydro)

    # Stage 3
    state_3 = state_flat + 0.5 * dt * k2
    k3 = get_rhs_full(0, state_3, grid, background, hydro)

    # Stage 4
    state_4 = state_flat + dt * k3
    k4 = get_rhs_full(0, state_4, grid, background, hydro)

    # Combine
    state_new = state_flat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    snew = state_new.reshape((grid.NUM_VARS, grid.N))

    # Apply atmosphere reset after full step
    snew_reset = _apply_atmosphere_reset(snew, grid, hydro, atmosphere)

    return snew_reset.flatten()


def evolve_fixed_timestep(state_initial, dt, num_steps, grid, background, hydro,
                          atmosphere, method='rk4', t_start=0.0,
                          reference_state=None, step_offset=0):
    """Evolve with fixed timestep using RK4 (full BSSN + hydro).

    Args:
        atmosphere: AtmosphereParams object
        t_start: Starting time for this evolution segment (default: 0.0)
        reference_state: Reference state for error calculation (default: state_initial)
                        Use this to maintain consistent error measurement across multiple segments
        step_offset: Offset for step numbering in output (default: 0)
    """
    state_flat = state_initial.flatten()

    def primitives_from_state(state_flattened):
        s2d = state_flattened.reshape((grid.NUM_VARS, grid.N))
        bssn_vars = BSSNVars(grid.N)
        bssn_vars.set_bssn_vars(s2d[:NUM_BSSN_VARS, :])
        hydro.set_matter_vars(s2d, bssn_vars, grid)
        return hydro._get_primitives(bssn_vars, grid.r, grid=grid), s2d

    # Diagnostics at start
    prim_prev, s_prev = primitives_from_state(state_flat)

    # Store reference state for error calculation (use provided reference or current initial)
    if reference_state is None:
        reference_state = state_initial
    prim_initial, s_initial = primitives_from_state(reference_state.flatten())

    print("\n===== Evolution diagnostics (per step) =====")
    print("Columns: step | t | ρ_central | max_Δρ/ρ | max_vʳ | max_D | max_Sʳ | max_τ | c2p_fails")
    print("-" * 100)

    for step in range(num_steps):
        # Advance one RK4 step
        state_flat_next = rk4_step(state_flat, dt, grid, background, hydro, atmosphere)

        # Compute primitives BEFORE and AFTER to measure change
        prim_next, s_next = primitives_from_state(state_flat_next)

        # Interior slice (exclude ghosts)
        interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

        rho_prev = prim_prev['rho0'][interior]
        rho_next = prim_next['rho0'][interior]
        rho_init = prim_initial['rho0'][interior]
        p_prev = prim_prev['p'][interior]
        p_next = prim_next['p'][interior]
        v_prev = prim_prev['vr'][interior]
        v_next = prim_next['vr'][interior]

        D_prev = s_prev[NUM_BSSN_VARS + 0, interior]
        Sr_prev = s_prev[NUM_BSSN_VARS + 1, interior]
        tau_prev = s_prev[NUM_BSSN_VARS + 2, interior]
        D_next = s_next[NUM_BSSN_VARS + 0, interior]
        Sr_next = s_next[NUM_BSSN_VARS + 1, interior]
        tau_next = s_next[NUM_BSSN_VARS + 2, interior]

        # Compute more informative stats
        rho_central = float(prim_next['rho0'][NUM_GHOSTS])  # Central density

        # Maximum relative density error vs initial state
        rel_rho_err = np.abs(rho_next - rho_init) / (np.abs(rho_init) + 1e-20)
        max_rel_rho_err = float(np.max(rel_rho_err))

        # Maximum velocity
        max_abs_v = float(np.max(np.abs(v_next)))

        # Maximum conserved variables (more useful than minimum)
        max_D = float(np.max(D_next))
        max_Sr = float(np.max(np.abs(Sr_next)))
        max_tau = float(np.max(np.abs(tau_next)))

        # Cons2prim failures
        c2p_fail_count = int(np.sum(~prim_next['success']))

        t_curr = t_start + (step + 1) * dt
        step_num = step_offset + step + 1
        print(f"step {step_num:4d}  t={t_curr:.6e}:  ρ_c={rho_central:.6e}  max_Δρ/ρ={max_rel_rho_err:.3e}  "
              f"max_vʳ={max_abs_v:.3e}  max_D={max_D:.3e}  max_Sʳ={max_Sr:.3e}  "
              f"max_τ={max_tau:.3e}  c2p_fail={c2p_fail_count}")

        # Detect first signs of instability / non-physical values
        issues = []
        if not np.all(np.isfinite(rho_next)) or not np.all(np.isfinite(p_next)):
            issues.append("NaN/Inf in primitives")
        if np.any(rho_next < 0):
            issues.append("negative rho0")
        if np.any(p_next < 0):
            issues.append("negative pressure")
        if np.any(np.abs(v_next) >= 1.0):
            issues.append("superluminal v")
        if np.any(D_next < 0):
            issues.append("negative D")
        if np.any((tau_next + D_next) < 0):
            issues.append("tau + D < 0")

        # If problems detected, print focused context (location and local values)
        if issues:
            print("  -> Detected issues:", ", ".join(issues))
            # Locate worst offenders
            try:
                idx_v = NUM_GHOSTS + int(np.argmax(np.abs(prim_next['vr'][interior])))
            except Exception:
                idx_v = NUM_GHOSTS
            try:
                idx_rho_min = NUM_GHOSTS + int(np.argmin(prim_next['rho0'][interior]))
            except Exception:
                idx_rho_min = NUM_GHOSTS
            try:
                idx_tauD_min = NUM_GHOSTS + int(np.argmin((s_next[NUM_BSSN_VARS+2, interior] + s_next[NUM_BSSN_VARS+0, interior])))
            except Exception:
                idx_tauD_min = NUM_GHOSTS

            idxs = sorted(set([idx_v, idx_rho_min, idx_tauD_min]))
            for ii in idxs:
                rloc = grid.r[ii]
                print(f"     at r={rloc:.6f} (i={ii}): "
                      f"rho0={prim_next['rho0'][ii]:.6e}, P={prim_next['p'][ii]:.6e}, vr={prim_next['vr'][ii]:.6e}, "
                      f"D={s_next[NUM_BSSN_VARS+0, ii]:.6e}, Sr={s_next[NUM_BSSN_VARS+1, ii]:.6e}, tau={s_next[NUM_BSSN_VARS+2, ii]:.6e}")

            # Also report relative changes at those points (prev -> next)
            def rel(a, b):
                return (b - a) / (np.abs(a) + 1e-30)
            for ii in idxs:
                print(f"       Δrel:  Δrho0={rel(prim_prev['rho0'][ii], prim_next['rho0'][ii]):.3e}, "
                      f"ΔP={rel(prim_prev['p'][ii], prim_next['p'][ii]):.3e}, "
                      f"Δvr={prim_next['vr'][ii]-prim_prev['vr'][ii]:.3e}, "
                      f"ΔD={rel(s_prev[NUM_BSSN_VARS+0, ii], s_next[NUM_BSSN_VARS+0, ii]):.3e}, "
                      f"ΔSr={s_next[NUM_BSSN_VARS+1, ii]-s_prev[NUM_BSSN_VARS+1, ii]:.3e}, "
                      f"Δtau={rel(s_prev[NUM_BSSN_VARS+2, ii], s_next[NUM_BSSN_VARS+2, ii]):.3e}")

            # Stop early so we can inspect before blow-up cascades
            print("  -> Halting evolution early due to detected instability.")
            state_flat = state_flat_next  # return the last state
            actual_steps = step + 1
            actual_time = t_start + actual_steps * dt
            return state_flat.reshape((grid.NUM_VARS, grid.N)), actual_steps, actual_time

        # Prepare next step
        state_flat = state_flat_next
        prim_prev, s_prev = prim_next, s_next

        # Light progress marker
        if (step + 1) % 20 == 0:
            print(f"  Step {step_num}/{step_offset + num_steps} OK  (t={t_curr:.6e})")

    actual_time = t_start + num_steps * dt
    return state_flat.reshape((grid.NUM_VARS, grid.N)), num_steps, actual_time


def evolve_adaptive(state_initial, t_final, grid, background, hydro,
                   method='RK45', rtol=1e-6, atol=1e-8):
    """Evolve with adaptive timestep using scipy.integrate.solve_ivp (full BSSN + hydro)."""
    from scipy.integrate import solve_ivp

    # Wrapper for RHS compatible with solve_ivp
    def rhs_wrapper(t, y):
        return get_rhs_full(t, y, grid, background, hydro)

    state_flat = state_initial.flatten()

    print(f"  Using solve_ivp with method={method}, rtol={rtol}, atol={atol}")

    solution = solve_ivp(
        rhs_wrapper,
        t_span=(0, t_final),
        y0=state_flat,
        method=method,  # 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
        rtol=rtol,
        atol=atol,
        dense_output=False
    )

    print(f"  solve_ivp: {solution.nfev} function evaluations, status={solution.status}")

    return solution.y[:, -1].reshape((grid.NUM_VARS, grid.N))


def plot_tov_diagnostics(tov_solution, r_max):
    """Plot TOV solution diagnostics (without conformal factor)."""
    r = tov_solution['r']
    R_star = tov_solution['R']
    M_star = tov_solution['M_star']

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Density
    axes[0, 0].semilogy(r, tov_solution['rho_baryon'], color='navy')
    axes[0, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5, label=f'R={R_star:.2f}')
    axes[0, 0].set_xlabel(r"$r$")
    axes[0, 0].set_ylabel(r"$\rho_0$")
    axes[0, 0].set_title('Baryon Density')
    axes[0, 0].set_xlim(0, r_max)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Pressure
    axes[0, 1].semilogy(r, tov_solution['P'], color='darkgreen')
    axes[0, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel(r"$r$")
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure')
    axes[0, 1].set_xlim(0, r_max)
    axes[0, 1].grid(True, alpha=0.3)

    # Enclosed Mass
    axes[0, 2].plot(r, tov_solution['M'], color='maroon')
    axes[0, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 2].axhline(M_star, color='gray', linestyle='--', alpha=0.3, label=f'M={M_star:.3f}')
    axes[0, 2].set_xlabel(r"$r$")
    axes[0, 2].set_ylabel('M(r)')
    axes[0, 2].set_title('Enclosed Mass')
    axes[0, 2].set_xlim(0, r_max)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Lapse alpha(r)
    axes[1, 0].plot(r, tov_solution['alpha'], color='purple')
    axes[1, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel(r"$r$")
    axes[1, 0].set_ylabel(r'$\alpha$')
    axes[1, 0].set_title('Lapse Function')
    axes[1, 0].set_xlim(0, r_max)
    axes[1, 0].grid(True, alpha=0.3)

    # Phi(r)
    phi = 0.25 * np.log(tov_solution['exp4phi'])
    axes[1, 1].plot(r, phi, color='teal')
    axes[1, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel(r"$r$")
    axes[1, 1].set_ylabel(r'$\phi$')
    axes[1, 1].set_title(r'Conformal Factor $\phi$')
    axes[1, 1].set_xlim(0, r_max)
    axes[1, 1].grid(True, alpha=0.3)

    # a(r) metric function: a = exp(2*phi) = sqrt(exp4phi)
    a_metric = np.sqrt(tov_solution['exp4phi'])
    axes[1, 2].plot(r, a_metric, color='orange')
    axes[1, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 2].set_xlabel(r"$r$")
    axes[1, 2].set_ylabel(r'$a(r)$')
    axes[1, 2].set_title('Metric $a(r)$')
    axes[1, 2].set_xlim(0, r_max)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'tov_solution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_bssn_evolution(state_t0, state_tfinal, grid, t_0=0.0, t_final=1.0):
    """Plot BSSN variables at initial and final time to show their evolution.

    In full BSSN evolution, BSSN variables evolve dynamically.
    This plot shows how the spacetime geometry changes during evolution.

    Note on spherical symmetry:
    - In spherical symmetry, h_θθ should equal h_φφ (isotropy)
    - Similarly, A_θθ should equal A_φφ
    - We plot all components to verify this consistency
    - Only h_rr and A_rr are truly independent radial variables

    Args:
        state_t0: Initial state
        state_tfinal: Final state
        grid: Grid object
        t_0: Initial time (for labeling)
        t_final: Final time (for labeling)
    """
    r_int = grid.r[NUM_GHOSTS:-NUM_GHOSTS]

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    # Row 1: Conformal factor φ, Lapse α, Trace of extrinsic curvature K
    # φ
    axes[0, 0].plot(r_int, state_t0[idx_phi, NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label=f't={t_0:.6e}')
    axes[0, 0].plot(r_int, state_tfinal[idx_phi, NUM_GHOSTS:-NUM_GHOSTS], 'r--', linewidth=1.7, label=f't={t_final:.6e}')
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel(r'$\phi$')
    axes[0, 0].set_title('Conformal Factor')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # α
    axes[0, 1].plot(r_int, state_t0[idx_lapse, NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label=f't={t_0:.6e}')
    axes[0, 1].plot(r_int, state_tfinal[idx_lapse, NUM_GHOSTS:-NUM_GHOSTS], 'r--', linewidth=1.7, label=f't={t_final:.6e}')
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel(r'$\alpha$')
    axes[0, 1].set_title('Lapse Function')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # K
    axes[0, 2].plot(r_int, state_t0[idx_K, NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label=f't={t_0:.6e}')
    axes[0, 2].plot(r_int, state_tfinal[idx_K, NUM_GHOSTS:-NUM_GHOSTS], 'r--', linewidth=1.7, label=f't={t_final:.6e}')
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel(r'$K$')
    axes[0, 2].set_title('Trace of Extrinsic Curvature')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Conformal metric components h_rr, h_θθ, h_φφ
    # h_rr
    axes[1, 0].plot(r_int, state_t0[idx_hrr, NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label=f't={t_0:.6e}')
    axes[1, 0].plot(r_int, state_tfinal[idx_hrr, NUM_GHOSTS:-NUM_GHOSTS], 'r--', linewidth=1.7, label=f't={t_final:.6e}')
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel(r'$h_{rr}$')
    axes[1, 0].set_title('Conformal Metric $h_{rr}$')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # h_θθ
    axes[1, 1].plot(r_int, state_t0[idx_htt, NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label=f't={t_0:.6e}')
    axes[1, 1].plot(r_int, state_tfinal[idx_htt, NUM_GHOSTS:-NUM_GHOSTS], 'r--', linewidth=1.7, label=f't={t_final:.6e}')
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel(r'$h_{\theta\theta}$')
    axes[1, 1].set_title(r'Conformal Metric $h_{\theta\theta}$')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # h_φφ
    axes[1, 2].plot(r_int, state_t0[idx_hpp, NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label=f't={t_0:.6e}')
    axes[1, 2].plot(r_int, state_tfinal[idx_hpp, NUM_GHOSTS:-NUM_GHOSTS], 'r--', linewidth=1.7, label=f't={t_final:.6e}')
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel(r'$h_{\phi\phi}$')
    axes[1, 2].set_title(r'Conformal Metric $h_{\phi\phi}$')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # Row 3: Traceless extrinsic curvature components A_rr, A_θθ, A_φφ
    # A_rr
    axes[2, 0].plot(r_int, state_t0[idx_arr, NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label=f't={t_0:.6e}')
    axes[2, 0].plot(r_int, state_tfinal[idx_arr, NUM_GHOSTS:-NUM_GHOSTS], 'r--', linewidth=1.7, label=f't={t_final:.6e}')
    axes[2, 0].set_xlabel('r')
    axes[2, 0].set_ylabel(r'$A_{rr}$')
    axes[2, 0].set_title('Traceless Extrinsic Curvature $A_{rr}$')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # A_θθ
    axes[2, 1].plot(r_int, state_t0[idx_att, NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label=f't={t_0:.6e}')
    axes[2, 1].plot(r_int, state_tfinal[idx_att, NUM_GHOSTS:-NUM_GHOSTS], 'r--', linewidth=1.7, label=f't={t_final:.6e}')
    axes[2, 1].set_xlabel('r')
    axes[2, 1].set_ylabel(r'$A_{\theta\theta}$')
    axes[2, 1].set_title(r'Traceless Extrinsic Curvature $A_{\theta\theta}$')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    # A_φφ
    axes[2, 2].plot(r_int, state_t0[idx_app, NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label=f't={t_0:.6e}')
    axes[2, 2].plot(r_int, state_tfinal[idx_app, NUM_GHOSTS:-NUM_GHOSTS], 'r--', linewidth=1.7, label=f't={t_final:.6e}')
    axes[2, 2].set_xlabel('r')
    axes[2, 2].set_ylabel(r'$A_{\phi\phi}$')
    axes[2, 2].set_title(r'Traceless Extrinsic Curvature $A_{\phi\phi}$')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    plt.suptitle(f'BSSN Variables Evolution (Full Dynamics)\nt={t_0:.6e} → t={t_final:.6e}',
                 fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'tov_bssn_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Compute and print maximum changes
    print("\n" + "="*70)
    print("BSSN VARIABLES EVOLUTION STATISTICS")
    print("="*70)
    print("Maximum absolute changes during evolution:\n")

    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

    def max_change(var_idx, var_name):
        change = np.abs(state_tfinal[var_idx, interior] - state_t0[var_idx, interior])
        max_val = np.max(np.abs(state_t0[var_idx, interior]))
        max_abs_change = np.max(change)
        max_rel_change = max_abs_change / (max_val + 1e-30)
        print(f"  {var_name:10s}: max |Δ| = {max_abs_change:.6e}  max |Δ/val| = {max_rel_change:.6e}")
        return max_abs_change

    max_change(idx_phi, 'φ')
    max_change(idx_lapse, 'α')
    max_change(idx_K, 'K')
    max_change(idx_hrr, 'h_rr')
    max_change(idx_htt, 'h_θθ')
    max_change(idx_hpp, 'h_φφ')
    max_change(idx_arr, 'A_rr')
    max_change(idx_att, 'A_θθ')
    max_change(idx_app, 'A_φφ')

    # Verify spherical symmetry: h_θθ should equal h_φφ, A_θθ should equal A_φφ
    print("\nSpherical symmetry verification (should be ~0):")
    diff_h = np.abs(state_t0[idx_htt, interior] - state_t0[idx_hpp, interior])
    diff_A = np.abs(state_t0[idx_att, interior] - state_t0[idx_app, interior])
    print(f"  max |h_θθ - h_φφ| at t={t_0:.6e}: {np.max(diff_h):.6e}")
    print(f"  max |A_θθ - A_φφ| at t={t_0:.6e}: {np.max(diff_A):.6e}")

    diff_h_final = np.abs(state_tfinal[idx_htt, interior] - state_tfinal[idx_hpp, interior])
    diff_A_final = np.abs(state_tfinal[idx_att, interior] - state_tfinal[idx_app, interior])
    print(f"  max |h_θθ - h_φφ| at t={t_final:.6e}: {np.max(diff_h_final):.6e}")
    print(f"  max |A_θθ - A_φφ| at t={t_final:.6e}: {np.max(diff_A_final):.6e}")

    print("="*70 + "\n")


def plot_evolution(state_t0, state_t1, state_t100, state_t10000, grid, hydro,
                   t_1, t_100, t_10000, label_100='t_100', label_10000='t_final'):
    """Plot evolution snapshots with actual times reached.

    Args:
        label_100: Label for third snapshot (may be same as t_10000 if not enough steps)
        label_10000: Label for fourth snapshot
    """
    bssn_0 = BSSNVars(grid.N)
    bssn_0.set_bssn_vars(state_t0[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t0, bssn_0, grid)
    prim_0 = hydro._get_primitives(bssn_0, grid.r, grid=grid)

    bssn_1 = BSSNVars(grid.N)
    bssn_1.set_bssn_vars(state_t1[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t1, bssn_1, grid)
    prim_1 = hydro._get_primitives(bssn_1, grid.r, grid=grid)

    bssn_100 = BSSNVars(grid.N)
    bssn_100.set_bssn_vars(state_t100[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t100, bssn_100, grid)
    prim_100 = hydro._get_primitives(bssn_100, grid.r, grid=grid)

    bssn_10000 = BSSNVars(grid.N)
    bssn_10000.set_bssn_vars(state_t10000[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t10000, bssn_10000, grid)
    prim_10000 = hydro._get_primitives(bssn_10000, grid.r, grid=grid)

    r_int = grid.r[NUM_GHOSTS:-NUM_GHOSTS]

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    # Density
    axes[0, 0].plot(r_int, prim_0['rho0'][NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label='t=0')
    axes[0, 0].plot(r_int, prim_1['rho0'][NUM_GHOSTS:-NUM_GHOSTS], 'orange', linestyle='--', linewidth=1.5, label=f't={t_1:.6e}')
    if t_100 != t_10000:  # Only plot if different from final
        axes[0, 0].plot(r_int, prim_100['rho0'][NUM_GHOSTS:-NUM_GHOSTS], 'green', linestyle='-.', linewidth=1.5, label=f'{label_100}={t_100:.6e}')
    axes[0, 0].plot(r_int, prim_10000['rho0'][NUM_GHOSTS:-NUM_GHOSTS], 'red', linestyle=':', linewidth=1.5, label=f'{label_10000}={t_10000:.6e}')
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel(r'$\rho_0$')
    axes[0, 0].set_title('Baryon Density Evolution')
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0.0, 0.002)
    axes[0, 0].grid(True, alpha=0.3)

    # Pressure
    axes[0, 1].plot(r_int, np.maximum(prim_0['p'][NUM_GHOSTS:-NUM_GHOSTS], 1e-20), 'b-', linewidth=2, label='t=0')
    axes[0, 1].plot(r_int, np.maximum(prim_1['p'][NUM_GHOSTS:-NUM_GHOSTS], 1e-20), 'orange', linestyle='--', linewidth=1.5, label=f't={t_1:.6e}')
    if t_100 != t_10000:
        axes[0, 1].plot(r_int, np.maximum(prim_100['p'][NUM_GHOSTS:-NUM_GHOSTS], 1e-20), 'green', linestyle='-.', linewidth=1.5, label=f'{label_100}={t_100:.6e}')
    axes[0, 1].plot(r_int, np.maximum(prim_10000['p'][NUM_GHOSTS:-NUM_GHOSTS], 1e-20), 'red', linestyle=':', linewidth=1.5, label=f'{label_10000}={t_10000:.6e}')
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Velocity
    axes[1, 0].plot(r_int, prim_0['vr'][NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label='t=0')
    axes[1, 0].plot(r_int, prim_1['vr'][NUM_GHOSTS:-NUM_GHOSTS], 'orange', linestyle='--', linewidth=1.5, label=f't={t_1:.6e}')
    if t_100 != t_10000:
        axes[1, 0].plot(r_int, prim_100['vr'][NUM_GHOSTS:-NUM_GHOSTS], 'green', linestyle='-.', linewidth=1.5, label=f'{label_100}={t_100:.6e}')
    axes[1, 0].plot(r_int, prim_10000['vr'][NUM_GHOSTS:-NUM_GHOSTS], 'red', linestyle=':', linewidth=1.5, label=f'{label_10000}={t_10000:.6e}')
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel(r'$v^r$')
    axes[1, 0].set_title('Radial Velocity Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Relative density error
    delta_rho_1 = np.abs(prim_1['rho0'][NUM_GHOSTS:-NUM_GHOSTS] - prim_0['rho0'][NUM_GHOSTS:-NUM_GHOSTS]) / (np.abs(prim_0['rho0'][NUM_GHOSTS:-NUM_GHOSTS]) + 1e-20)
    delta_rho_100 = np.abs(prim_100['rho0'][NUM_GHOSTS:-NUM_GHOSTS] - prim_0['rho0'][NUM_GHOSTS:-NUM_GHOSTS]) / (np.abs(prim_0['rho0'][NUM_GHOSTS:-NUM_GHOSTS]) + 1e-20)
    delta_rho_10000 = np.abs(prim_10000['rho0'][NUM_GHOSTS:-NUM_GHOSTS] - prim_0['rho0'][NUM_GHOSTS:-NUM_GHOSTS]) / (np.abs(prim_0['rho0'][NUM_GHOSTS:-NUM_GHOSTS]) + 1e-20)

    axes[1, 1].semilogy(r_int, delta_rho_1, 'orange', linestyle='--', linewidth=1.5, label=f't={t_1:.6e}')
    if t_100 != t_10000:
        axes[1, 1].semilogy(r_int, delta_rho_100, 'green', linestyle='-.', linewidth=1.5, label=f'{label_100}={t_100:.6e}')
    axes[1, 1].semilogy(r_int, delta_rho_10000, 'red', linestyle=':', linewidth=1.5, label=f'{label_10000}={t_10000:.6e}')
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel(r'$|\Delta\rho|/\rho$')
    axes[1, 1].set_title('Relative Density Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Lapse function (alpha)
    axes[2, 0].plot(r_int, state_t0[idx_lapse, NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label='t=0')
    axes[2, 0].plot(r_int, state_t1[idx_lapse, NUM_GHOSTS:-NUM_GHOSTS], 'orange', linestyle='--', linewidth=1.5, label=f't={t_1:.6e}')
    if t_100 != t_10000:
        axes[2, 0].plot(r_int, state_t100[idx_lapse, NUM_GHOSTS:-NUM_GHOSTS], 'green', linestyle='-.', linewidth=1.5, label=f'{label_100}={t_100:.6e}')
    axes[2, 0].plot(r_int, state_t10000[idx_lapse, NUM_GHOSTS:-NUM_GHOSTS], 'red', linestyle=':', linewidth=1.5, label=f'{label_10000}={t_10000:.6e}')
    axes[2, 0].set_xlabel('r')
    axes[2, 0].set_ylabel(r'$\alpha$')
    axes[2, 0].set_title('Lapse Function Evolution')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Conformal factor (phi)
    axes[2, 1].plot(r_int, state_t0[idx_phi, NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label='t=0')
    axes[2, 1].plot(r_int, state_t1[idx_phi, NUM_GHOSTS:-NUM_GHOSTS], 'orange', linestyle='--', linewidth=1.5, label=f't={t_1:.6e}')
    if t_100 != t_10000:
        axes[2, 1].plot(r_int, state_t100[idx_phi, NUM_GHOSTS:-NUM_GHOSTS], 'green', linestyle='-.', linewidth=1.5, label=f'{label_100}={t_100:.6e}')
    axes[2, 1].plot(r_int, state_t10000[idx_phi, NUM_GHOSTS:-NUM_GHOSTS], 'red', linestyle=':', linewidth=1.5, label=f'{label_10000}={t_10000:.6e}')
    axes[2, 1].set_xlabel('r')
    axes[2, 1].set_ylabel(r'$\phi$')
    axes[2, 1].set_title('Conformal Factor Evolution')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Evolution: t=0 → t={t_1:.6e} → {label_100}={t_100:.6e} → {label_10000}={t_10000:.6e}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'tov_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    """Main execution."""
    # Create output directory for plots
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"Plots will be saved to: {PLOT_DIR}/")

    print("="*70)
    print("TOV Star Evolution - Full BSSN + Hydro")
    print("="*70)

    # ==================================================================
    # CONFIGURATION
    # ==================================================================
    r_max = 16.0
    num_points = 5000
    K = 100.0
    Gamma = 2.0
    rho_central = 1.280e-3

    # ==================================================================
    # ATMOSPHERE CONFIGURATION (Centralized floor management)
    # ==================================================================
    # Define atmosphere parameters ONCE - all subsystems will use these
    ATMOSPHERE = AtmosphereParams(
        rho_floor=1.0e-10,      # Rest mass density floor
        p_floor=1.0e-11,        # Pressure floor
        v_max=0.9999,           # Maximum velocity
        W_max=100.0,            # Maximum Lorentz factor
        tau_atm_factor=1.0,     # tau_atm = tau_atm_factor * p_floor
        conservative_floor_safety=0.999999  # Safety factor for S^2 constraint
    )

    print("=" * 70)
    print("ATMOSPHERE CONFIGURATION")
    print("=" * 70)
    print(f"  rho_floor = {ATMOSPHERE.rho_floor:.2e}")
    print(f"  p_floor   = {ATMOSPHERE.p_floor:.2e}")
    print(f"  tau_atm   = {ATMOSPHERE.tau_atm:.2e}")
    print(f"  v_max     = {ATMOSPHERE.v_max}")
    print()

    # Time integration method
    # 'fixed': RK4 with fixed timestep (fast, stable)
    # 'adaptive': solve_ivp with adaptive timestep (slower, more accurate)
    integration_method = 'fixed'  # 'fixed' or 'adaptive'

    # ==================================================================
    # SETUP
    # ==================================================================
    spacing = LinearSpacing(num_points, r_max)
    # Evolve with ideal-gas EOS using the same Gamma as the TOV polytrope
    eos = IdealGasEOS(gamma=Gamma)
    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere=ATMOSPHERE,  
        reconstructor=create_reconstruction("wenoz"),
        riemann_solver=HLLERiemannSolver()
    )

    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    print(f"Grid: N={grid.N}, r_max={r_max}, dr_min={grid.min_dr}")
    print(f"EOS: K={K}, Gamma={Gamma}\n")

    # ==================================================================
    # SOLVE TOV DIRECTLY ON EVOLUTION GRID (for discrete equilibrium)
    # ==================================================================
    print("Solving TOV equations directly on evolution grid...")
    # Strategy: Solve TOV at exact grid points to minimize interpolation error
    tov_solver = TOVSolver(K=K, Gamma=Gamma)

    print(f"  TOV grid: using evolution grid ({grid.N} points)")

    # Solve TOV directly at evolution grid points
    tov_solution = tov_solver.solve(rho_central, r_grid=grid.r, r_max=r_max)

    print(f"TOV Solution: M={tov_solution['M_star']:.6f}, R={tov_solution['R']:.3f}, C={tov_solution['C']:.4f}\n")

    plot_tov_diagnostics(tov_solution, r_max)

    # ==================================================================
    # INITIAL DATA (HIGH-ORDER INTERPOLATION, NRPy+ STRATEGY)
    # ==================================================================
    print("Creating initial data from TOV solution...")
    # Strategy (following NRPy+ tutorial):
    # 1. Interpolate ρ, P ONLY up to stellar radius R
    # 2. Outside R: use atmosphere values directly (no interpolation)
    # 3. Interpolate geometry (α, exp4φ) everywhere
    # 4. Stencil NEVER crosses the stellar surface (avoids Gibbs phenomenon)
    initial_state_2d = tov_id.create_initial_data_interpolated(
        tov_solution, grid, background, eos,
        atmosphere=ATMOSPHERE,
        polytrope_K=K, polytrope_Gamma=Gamma,
        interp_order=11  # High order Lagrange interpolation
    )

    # Diagnostics: check discrete hydrostatic balance at t=0
    diagnose_t0_residuals(initial_state_2d, grid, background, hydro)

    # Initial-data diagnostics
    tov_id.plot_initial_comparison(tov_solution, initial_state_2d, grid, hydro, output_dir=PLOT_DIR)

    # ==================================================================
    # EVOLUTION
    # ==================================================================
    if integration_method == 'fixed':
        dt = 0.15 * grid.min_dr  # CFL condition
        num_steps_total = 10000
        print(f"\nEvolving with fixed dt={dt:.6f} (CFL=0.15) for {num_steps_total} steps using RK4")

        # Single step for comparison
        state_t1 = rk4_step(initial_state_2d.flatten(), dt, grid, background, hydro,
                           ATMOSPHERE).reshape((grid.NUM_VARS, grid.N))
        t_1 = dt  
        
        # Plot only the first step changes
        plot_first_step(initial_state_2d, state_t1, grid, hydro, tov_solution)
        plot_surface_zoom(tov_solution, initial_state_2d, state_t1, grid, hydro, window=0.1)

        # Define intermediate checkpoint (around 10% of total, or 100 if total is large)
        if num_steps_total <= 100:
            checkpoint_mid = max(10, num_steps_total // 10)
        else:
            checkpoint_mid = 100
        
        # Evolve incrementally to intermediate checkpoint
        print(f"\nEvolving to step {checkpoint_mid}...")
        state_t100, steps_100, t_100 = evolve_fixed_timestep(initial_state_2d, dt, checkpoint_mid, grid, background,
                                                              hydro, ATMOSPHERE, method='rk4',
                                                              reference_state=initial_state_2d)
        print(f"  -> Reached step {steps_100}, t={t_100:.6e}")

        # Continue evolution to num_steps_total (or until it breaks)
        if num_steps_total > checkpoint_mid and steps_100 == checkpoint_mid:
            remaining_steps = num_steps_total - checkpoint_mid
            print(f"\nContinuing evolution from step {checkpoint_mid} to {num_steps_total} ({remaining_steps} more steps)...")
            state_tfinal, steps_more, t_final = evolve_fixed_timestep(state_t100, dt, remaining_steps, grid, background,
                                                                      hydro, ATMOSPHERE, method='rk4',
                                                                      t_start=t_100, reference_state=initial_state_2d,
                                                                      step_offset=checkpoint_mid)
            # t_final already includes t_start from evolve_fixed_timestep
            steps_final = checkpoint_mid + steps_more
            print(f"  -> Reached step {steps_final}, t={t_final:.6e}")
        else:
            # Evolution stopped early at checkpoint, or checkpoint == total
            state_tfinal = state_t100
            t_final = t_100
            steps_final = steps_100
        
        # For plotting: use intermediate state and final state (wherever it stopped)
        state_t10000 = state_tfinal
        t_10000 = t_final
        num_steps = steps_final

    elif integration_method == 'adaptive':
        # NOTE: Adaptive methods are slower but can be more accurate
        # Available methods: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
        t_100 = 1.00  # Time for intermediate snapshot
        t_10000 = 10.00  # Final time
        print(f"\nEvolving to t={t_10000} using solve_ivp (adaptive)")

        # For single step comparison, use small fixed dt
        dt = 0.5 * grid.min_dr
        t_1 = dt
        # Define an effective number of steps for labeling/plots
        # so code paths that expect 'num_steps' still work.
        num_steps = max(1, int(round(t_10000 / dt)))
        state_t1 = rk4_step(initial_state_2d.flatten(), dt, grid, background, hydro,
                           ATMOSPHERE).reshape((grid.NUM_VARS, grid.N))

        # Adaptive evolution to t=1.0
        state_t100 = evolve_adaptive(initial_state_2d, t_100, grid, background, hydro,
                                    method='RK45', rtol=1e-5, atol=1e-7)

        # Adaptive evolution to t=10.0
        state_t10000 = evolve_adaptive(initial_state_2d, t_10000, grid, background, hydro,
                                      method='RK45', rtol=1e-5, atol=1e-7)

    # ==================================================================
    # DIAGNOSTICS
    # ==================================================================
    # Determine labels for plotting based on actual evolution
    if integration_method == 'fixed':
        # For fixed timestep: use step numbers for clarity
        if t_100 == t_10000:
            # Same state (evolution stopped early or only ran to checkpoint)
            label_100 = f't(step={steps_final})'
            label_10000 = f't(step={steps_final})'
        else:
            label_100 = f't(step={steps_100})'
            label_10000 = f't(step={steps_final})'
    else:
        # For adaptive timestep: use time labels
        label_100 = 't_mid'
        label_10000 = 't_final'

    plot_evolution(initial_state_2d, state_t1, state_t100, state_t10000, grid, hydro,
                   t_1, t_100, t_10000, label_100=label_100, label_10000=label_10000)

    # Plot BSSN variables evolution to verify Cowling approximation
    plot_bssn_evolution(initial_state_2d, state_t10000, grid, t_0=0.0, t_final=t_10000)

    # Print detailed statistics
    bssn_0 = BSSNVars(grid.N)
    bssn_0.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(initial_state_2d, bssn_0, grid)
    prim_0 = hydro._get_primitives(bssn_0, grid.r, grid=grid)

    bssn_1 = BSSNVars(grid.N)
    bssn_1.set_bssn_vars(state_t1[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t1, bssn_1, grid)
    prim_1 = hydro._get_primitives(bssn_1, grid.r, grid=grid)

    bssn_100 = BSSNVars(grid.N)
    bssn_100.set_bssn_vars(state_t100[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t100, bssn_100, grid)
    prim_100 = hydro._get_primitives(bssn_100, grid.r, grid=grid)

    bssn_10000 = BSSNVars(grid.N)
    bssn_10000.set_bssn_vars(state_t10000[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t10000, bssn_10000, grid)
    prim_10000 = hydro._get_primitives(bssn_10000, grid.r, grid=grid)

    # Calculate actual final time (already stored in t_10000)
    t_final_actual = t_10000

    # Interior points only (exclude ghosts)
    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

    # Compute errors
    delta_rho_1 = np.abs(prim_1['rho0'][interior] - prim_0['rho0'][interior]) / (np.abs(prim_0['rho0'][interior]) + 1e-20)
    delta_rho_100 = np.abs(prim_100['rho0'][interior] - prim_0['rho0'][interior]) / (np.abs(prim_0['rho0'][interior]) + 1e-20)
    delta_rho_10000 = np.abs(prim_10000['rho0'][interior] - prim_0['rho0'][interior]) / (np.abs(prim_0['rho0'][interior]) + 1e-20)

    delta_P_1 = np.abs(prim_1['p'][interior] - prim_0['p'][interior]) / (np.abs(prim_0['p'][interior]) + 1e-20)
    delta_P_100 = np.abs(prim_100['p'][interior] - prim_0['p'][interior]) / (np.abs(prim_0['p'][interior]) + 1e-20)
    delta_P_10000 = np.abs(prim_10000['p'][interior] - prim_0['p'][interior]) / (np.abs(prim_0['p'][interior]) + 1e-20)

    # Error growth factor
    max_err_rho_1 = np.max(delta_rho_1)
    max_err_rho_100 = np.max(delta_rho_100)
    max_err_rho_10000 = np.max(delta_rho_10000)
    growth_rho = max_err_rho_10000 / max_err_rho_1 if max_err_rho_1 > 1e-15 else 0

    max_err_P_1 = np.max(delta_P_1)
    max_err_P_100 = np.max(delta_P_100)
    max_err_P_10000 = np.max(delta_P_10000)
    growth_P = max_err_P_10000 / max_err_P_1 if max_err_P_1 > 1e-15 else 0

    print(f"\n{'='*70}")
    print(f"EVOLUTION DIAGNOSTICS (t=0 → t={t_1:.6e} → t={t_100:.6e} → t={t_10000:.6e})")
    print(f"{'='*70}")

    print(f"\n1. VELOCITY EVOLUTION:")
    print(f"   Max |v^r| at t=0:             {np.max(np.abs(prim_0['vr'])):.3e}")
    print(f"   Max |v^r| at t={t_1:.6e}:   {np.max(np.abs(prim_1['vr'])):.3e}")
    if t_100 != t_10000:
        print(f"   Max |v^r| at t={t_100:.6e}:   {np.max(np.abs(prim_100['vr'])):.3e}")
    print(f"   Max |v^r| at t={t_10000:.6e}: {np.max(np.abs(prim_10000['vr'])):.3e}")

    print(f"\n2. CENTRAL DENSITY:")
    print(f"   ρ_c at t=0:                 {prim_0['rho0'][NUM_GHOSTS]:.6e}")
    print(f"   ρ_c at t={t_1:.6e}:   {prim_1['rho0'][NUM_GHOSTS]:.6e}")
    if t_100 != t_10000:
        print(f"   ρ_c at t={t_100:.6e}:   {prim_100['rho0'][NUM_GHOSTS]:.6e}")
    print(f"   ρ_c at t={t_10000:.6e}: {prim_10000['rho0'][NUM_GHOSTS]:.6e}")
    print(f"   Δρ_c/ρ_c (t={t_1:.6e}):    {abs(prim_1['rho0'][NUM_GHOSTS] - prim_0['rho0'][NUM_GHOSTS])/prim_0['rho0'][NUM_GHOSTS]:.3e}")
    if t_100 != t_10000:
        print(f"   Δρ_c/ρ_c (t={t_100:.6e}):  {abs(prim_100['rho0'][NUM_GHOSTS] - prim_0['rho0'][NUM_GHOSTS])/prim_0['rho0'][NUM_GHOSTS]:.3e}")
    print(f"   Δρ_c/ρ_c (t={t_10000:.6e}):{abs(prim_10000['rho0'][NUM_GHOSTS] - prim_0['rho0'][NUM_GHOSTS])/prim_0['rho0'][NUM_GHOSTS]:.3e}")

    print(f"\n3. DENSITY ERROR (max over domain):")
    print(f"   Max |Δρ|/ρ at t={t_1:.6e}:    {max_err_rho_1:.3e}")
    if t_100 != t_10000:
        print(f"   Max |Δρ|/ρ at t={t_100:.6e}:  {max_err_rho_100:.3e}")
    print(f"   Max |Δρ|/ρ at t={t_10000:.6e}: {max_err_rho_10000:.3e}")
    print(f"   Growth factor (final/1):        {growth_rho:.1f}x")

    print(f"\n4. PRESSURE ERROR (max over domain):")
    print(f"   Max |ΔP|/P at t={t_1:.6e}:    {max_err_P_1:.3e}")
    if t_100 != t_10000:
        print(f"   Max |ΔP|/P at t={t_100:.6e}:  {max_err_P_100:.3e}")
    print(f"   Max |ΔP|/P at t={t_10000:.6e}: {max_err_P_10000:.3e}")
    print(f"   Growth factor (final/1):        {growth_P:.1f}x")

    print(f"\n5. CONS2PRIM STATUS:")
    print(f"   Success at t=0:               {np.sum(prim_0['success'])}/{grid.N}")
    print(f"   Success at t={t_1:.6e}:   {np.sum(prim_1['success'])}/{grid.N}")
    if t_100 != t_10000:
        print(f"   Success at t={t_100:.6e}:   {np.sum(prim_100['success'])}/{grid.N}")
    print(f"   Success at t={t_10000:.6e}: {np.sum(prim_10000['success'])}/{grid.N}")

    if not np.all(prim_10000['success']):
        failed_idx = np.where(~prim_10000['success'])[0]
        print(f"   Failed points: {failed_idx[:5]} (first 5)")
        print(f"   Failed radii:  {grid.r[failed_idx[:5]]}")

    print("\n" + "="*70)
    print(f"Evolution complete. Plots saved to {PLOT_DIR}/:")
    print(f"  1. {PLOT_DIR}/tov_solution.png                - TOV solution (ρ, P, M, α)")
    print(f"  2. {PLOT_DIR}/tov_initial_data_comparison.png - TOV vs Initial data at t=0")
    if t_100 != t_10000:
        print(f"  3. {PLOT_DIR}/tov_evolution.png               - Hydro evolution: t=0 → t={t_1:.6e} → t={t_100:.6e} → t={t_10000:.6e}")
    else:
        print(f"  3. {PLOT_DIR}/tov_evolution.png               - Hydro evolution: t=0 → t={t_1:.6e} → t={t_10000:.6e}")
    print(f"  4. {PLOT_DIR}/tov_bssn_evolution.png          - BSSN variables: t=0 → t={t_10000:.6e} (Full evolution)")
    print("="*70)


if __name__ == "__main__":
    main()
