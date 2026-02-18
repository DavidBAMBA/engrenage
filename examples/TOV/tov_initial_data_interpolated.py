"""
TOV Initial Data in Isotropic Coordinates for BSSN Evolution.

This module creates initial data for BSSN evolution from a TOV solution
computed in isotropic coordinates.

Key simplification in isotropic coordinates:
- The spatial metric is conformally flat: γ_ij = e^{4φ} ĝ_ij
- Therefore γ̄_ij = ĝ_ij exactly (conformal metric = reference metric)
- This means h_ij = 0 for all components!
- All geometric information is contained in φ alone

The grid coordinate r represents r_iso (isotropic radial coordinate).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from source.core.spacing import NUM_GHOSTS
from source.bssn.bssnstatevariables import (
    NUM_BSSN_VARS,
    idx_phi, idx_hrr, idx_htt, idx_hpp,
    idx_K, idx_arr, idx_att, idx_app, idx_lapse,
    idx_lambdar, idx_shiftr, idx_br,
)
from source.backgrounds.sphericalbackground import i_r, i_t, i_p


def interpolate_tov_iso_avoiding_surface(r_target, tov_solution, field_name,
                                          interp_order=11, atmosphere_value=None):
    """
    Interpolate TOV solution field using r_iso as coordinate.
    Ensures stencil doesn't cross surface to prevent Gibbs phenomenon.

    Args:
        r_target: Radius where to interpolate (in isotropic coordinates)
        tov_solution: TOVSolutionIso object
        field_name: Name of field to interpolate ('rho_baryon', 'P', 'exp4phi', 'alpha')
        interp_order: Interpolation order (default 11)
        atmosphere_value: Value to return if outside star for matter quantities

    Returns:
        Interpolated value at r_target
    """
    r_iso = tov_solution.r_iso
    field_tov = getattr(tov_solution, field_name)
    R_iso = tov_solution.R_iso
    M_star = tov_solution.M_star

    # Handle negative radii (spherical symmetry)
    r_use = abs(r_target)

    # If outside star, use exact Schwarzschild solution in isotropic coords
    if r_use > R_iso:
        if field_name == 'exp4phi':
            # Schwarzschild in isotropic: exp4phi = (1 + M/(2r_iso))^4
            return (1.0 + M_star / (2.0 * max(r_use, 0.1)))**4
        elif field_name == 'alpha':
            # Schwarzschild in isotropic: alpha = (1 - M/(2r_iso)) / (1 + M/(2r_iso))
            return (1.0 - M_star / (2.0 * max(r_use, 0.1))) / (1.0 + M_star / (2.0 * max(r_use, 0.1)))
        elif field_name in ['rho_baryon', 'P']:
            return atmosphere_value if atmosphere_value is not None else 0.0
        else:
            return atmosphere_value if atmosphere_value is not None else 0.0

    # Clamp to data range
    # For very small r, the interpolation can be unstable, but the first point
    # is now correctly computed via Taylor expansion in the solver
    skip_origin = 0  # No longer need to skip first point
    r_use = max(r_use, r_iso[skip_origin])
    r_use = min(r_use, r_iso[-1])

    # Find stellar surface index
    surf_idx = np.argmin(np.abs(r_iso - R_iso))

    # Find target index
    idx = np.searchsorted(r_iso, r_use)
    idx = max(skip_origin, min(idx, len(r_iso) - 1))

    # Determine stencil bounds
    half_stencil = interp_order // 2
    idx_min = max(skip_origin, idx - half_stencil)
    idx_max = min(len(r_iso), idx_min + interp_order)

    # Don't allow stencil to cross stellar surface
    if r_use < R_iso:
        idx_max = min(idx_max, surf_idx + 1)
        idx_min = max(0, idx_max - interp_order)
    else:
        idx_min = max(idx_min, surf_idx)
        idx_max = min(len(r_iso), idx_min + interp_order)

    # Extract stencil
    r_stencil = r_iso[idx_min:idx_max]
    field_stencil = field_tov[idx_min:idx_max]

    # Lagrange polynomial interpolation
    result = 0.0
    for i in range(len(r_stencil)):
        L_i = 1.0
        for j in range(len(r_stencil)):
            if i != j:
                denom = r_stencil[i] - r_stencil[j]
                if abs(denom) > 1e-30:
                    L_i *= (r_use - r_stencil[j]) / denom
        result += field_stencil[i] * L_i

    return result


def compute_lambda_from_metric_fd(state_2d, grid, background):
    """
    Compute lambda^i using finite differences of the metric quantities.

    In BSSN:
        λ̄^i = γ̄^jk (Γ̄^i_jk - Γ̂^i_jk)

    where Γ̄^i_jk are computed from finite differences of γ̄_ij.

    For isotropic coordinates with h_ij = 0:
        γ̄_ij = ĝ_ij (pointwise)

    However, ∂_k γ̄_ij ≠ 0 on a numerical grid, so lambda^i ≠ 0.

    Args:
        state_2d: State vector (NUM_VARS, N) to modify in-place
        grid: Grid object with FD operators
        background: Background metric object

    Modifies:
        state_2d[idx_lambdar, :] in place
    """
    from source.bssn.bssnvars import BSSNVars
    from source.bssn.tensoralgebra import get_tensor_connections
    from source.backgrounds.sphericalbackground import i_r

    N = grid.N
    bssn_vars = BSSNVars(N)
    bssn_vars.set_bssn_vars(state_2d)

    # Get finite difference derivatives of metric quantities
    # This computes ∂_r h_ij, ∂_θ h_ij, ∂_φ h_ij using FD stencils
    d1 = grid.get_d1_metric_quantities(state_2d)

    # Get connections: Delta^i = γ̄^jk (Γ̄^i_jk - Γ̂^i_jk)
    # This function already exists in tensoralgebra.py and handles all the
    # Christoffel symbol calculations correctly
    Delta_U, Delta_ULL, Delta_LLL = get_tensor_connections(
        grid.r, bssn_vars.h_LL, d1.h_LL, background
    )

    # lambda^r is the radial component of Delta^i
    # In spherical symmetry, only the radial component is non-zero
    # Delta_U has shape (N, 3) where second index is [r, theta, phi]
    state_2d[idx_lambdar, :] = Delta_U[:, i_r]


def create_initial_data_iso(tov_solution, grid, background, eos,
                             atmosphere=None,
                             polytrope_K=None, polytrope_Gamma=None,
                             interp_order=11):
    """
    Create BSSN + hydro initial data from TOV solution in isotropic coordinates.

    Key simplification: In isotropic coordinates, the spatial metric is conformally flat:
        γ_ij = e^{4φ} ĝ_ij
    Therefore:
        γ̄_ij = ĝ_ij  (conformal metric equals reference metric)
        h_ij = 0      (no metric deviation)
        λ^r = 0       (connection function is zero)

    Args:
        tov_solution: TOVSolutionIso object
        grid: Engrenage evolution grid
        background: FlatSphericalBackground
        eos: Equation of state
        atmosphere: AtmosphereParams object
        polytrope_K, polytrope_Gamma: EOS parameters
        interp_order: Lagrange interpolation order

    Returns:
        state_2d: (NUM_VARS, N) initial data array with BSSN + hydro
        primitives: Tuple (rho0, vr, p, eps)
    """
    atmosphere_rho = atmosphere.rho_floor
    p_atm = atmosphere.p_floor

    if polytrope_K is not None and polytrope_Gamma is not None:
        p_atm = max(p_atm, polytrope_K * (atmosphere_rho ** polytrope_Gamma))

    R_iso = tov_solution.R_iso
    M_star = tov_solution.M_star
    N = grid.N

    print(f"\nCreating TOV initial data (ISOTROPIC coordinates):")
    print(f"  Stellar radius R_iso = {R_iso:.6f}")
    print(f"  Stellar radius R_schw = {tov_solution.R_schw:.6f}")
    print(f"  Total mass M = {M_star:.6f}")
    print(f"  Interpolation order: {interp_order}")
    print(f"  Key simplification: h_ij = 0 (conformally flat)")

    # Initialize arrays
    exp4phi_arr = np.ones(N)
    alpha_arr = np.ones(N)
    rho_arr = np.zeros(N)
    P_arr = np.zeros(N)

    # Interpolate TOV solution to grid
    print("  Interpolating TOV solution to evolution grid...")

    # Get the central value of exp4phi from TOV solution
    # (the solver now correctly computes this using Taylor expansion)
    exp4phi_tov = tov_solution.exp4phi
    exp4phi_central = exp4phi_tov[0]  # Use value from solver directly

    for i, r in enumerate(grid.r):
        r_abs = abs(r)

        # Special handling for origin (r=0)
        if r_abs < 1e-8:
            # At origin, use extrapolated central value (NOT exp4phi=1.0)
            exp4phi_arr[i] = exp4phi_central
            alpha_arr[i] = tov_solution.alpha[0]
            rho_arr[i] = tov_solution.rho_baryon[0]
            P_arr[i] = tov_solution.P[0]
            continue

        # Minimum radius for interpolation (avoid extremely small r)
        r_abs = max(r_abs, 1e-6)

        is_interior = r_abs < R_iso

        if is_interior:
            exp4phi_arr[i] = interpolate_tov_iso_avoiding_surface(
                r_abs, tov_solution, 'exp4phi', interp_order, 1.0)
            alpha_arr[i] = interpolate_tov_iso_avoiding_surface(
                r_abs, tov_solution, 'alpha', interp_order, 1.0)
            rho_arr[i] = interpolate_tov_iso_avoiding_surface(
                r_abs, tov_solution, 'rho_baryon', interp_order, atmosphere_rho)
            P_arr[i] = interpolate_tov_iso_avoiding_surface(
                r_abs, tov_solution, 'P', interp_order, p_atm)

            # Enforce floors
            # Note: exp4phi should be >= 1.0 (it equals 1 at origin)
            if exp4phi_arr[i] < 0.9:
                print(f"  WARNING: exp4phi[{i}] = {exp4phi_arr[i]:.3e} at r={grid.r[i]:.3e} (should be >= 1.0)")
                exp4phi_arr[i] = 1.0  # Reset to minimum physical value
            alpha_arr[i] = max(alpha_arr[i], 1e-10)
            rho_arr[i] = max(rho_arr[i], atmosphere_rho)
            P_arr[i] = max(P_arr[i], p_atm)
        else:
            # Exterior: exact Schwarzschild in isotropic coordinates
            factor = 1.0 + M_star / (2.0 * r_abs)
            exp4phi_arr[i] = factor**4
            alpha_arr[i] = (1.0 - M_star / (2.0 * r_abs)) / factor
            rho_arr[i] = atmosphere_rho
            P_arr[i] = p_atm

    # Create BSSN state vector
    state_2d = np.zeros((grid.NUM_VARS, N))

    # BSSN variables (SIMPLE in isotropic coordinates!)
    # φ_BSSN = (1/4) ln(e^{4φ})
    phi_bssn = 0.25 * np.log(np.maximum(exp4phi_arr, 1e-30))
    state_2d[idx_phi, :] = phi_bssn

    # Diagnostic: Check values at origin (find point closest to r=0)
    idx_origin = np.argmin(np.abs(grid.r))
    print(f"  Values near origin (r={grid.r[idx_origin]:.6f}, i={idx_origin}):")
    print(f"    exp4phi = {exp4phi_arr[idx_origin]:.10e}  (should be ~1.0)")
    print(f"    phi_bssn = {phi_bssn[idx_origin]:.10e}  (should be ~0.0)")
    print(f"    alpha = {alpha_arr[idx_origin]:.10e}")
    print(f"    rho = {rho_arr[idx_origin]:.10e}")

    # h_ij = 0 because γ̄_ij = ĝ_ij in isotropic coordinates
    state_2d[idx_hrr, :] = 0.0
    state_2d[idx_htt, :] = 0.0
    state_2d[idx_hpp, :] = 0.0

    # Extrinsic curvature: K = 0, A_ij = 0 for static star
    state_2d[idx_K, :] = 0.0
    state_2d[idx_arr, :] = 0.0
    state_2d[idx_att, :] = 0.0
    state_2d[idx_app, :] = 0.0

    # Connection function: λ^r must be computed using finite differences
    # Set to zero temporarily, will be computed after filling ghost zones
    state_2d[idx_lambdar, :] = 0.0

    # Gauge variables
    state_2d[idx_lapse, :] = alpha_arr
    state_2d[idx_shiftr, :] = 0.0  # Static: no shift
    state_2d[idx_br, :] = 0.0

    print(f"  BSSN metric variables set:")
    print(f"    φ:  min={np.min(phi_bssn):.6f}, max={np.max(phi_bssn):.6f}")
    print(f"    h_ij = 0 (conformally flat)")
    print(f"    λ^r will be computed using FD after ghost fill")
    print(f"    alpha:  min={np.min(alpha_arr):.6f}, max={np.max(alpha_arr):.6f}")

    # Hydro variables
    print("  Setting hydro variables...")

    # Specific internal energy
    if polytrope_Gamma is not None:
        eps_arr = P_arr / np.maximum((polytrope_Gamma - 1.0) * rho_arr, 1e-30)
    else:
        eps_arr = np.zeros(N)
        for i in range(N):
            eps_arr[i] = eos.epsilon_from_P_rho(P_arr[i], rho_arr[i])

    # Static TOV: zero velocity
    vr_arr = np.zeros(N)

    # Densitized conserved variables
    e6phi = np.exp(6.0 * phi_bssn)

    # D̃ = e^{6φ} ρ₀ W (W=1 for static)
    D_arr = e6phi * rho_arr

    # S̃^r = 0 for static star
    Sr_arr = np.zeros(N)

    # τ̃ = e^{6φ} (ρ₀hW² - p - ρ₀W) (W=1 for static)
    h_arr = 1.0 + eps_arr + P_arr / np.maximum(rho_arr, 1e-30)
    tau_arr = e6phi * (rho_arr * h_arr - P_arr - rho_arr)

    # Store hydro conservatives
    state_2d[NUM_BSSN_VARS + 0, :] = D_arr
    state_2d[NUM_BSSN_VARS + 1, :] = Sr_arr
    state_2d[NUM_BSSN_VARS + 2, :] = tau_arr

    # Ensure exterior is exactly atmosphere
    r_grid = np.abs(grid.r)
    exterior_mask = r_grid > R_iso
    n_exterior = np.sum(exterior_mask)
    if n_exterior > 0:
        state_2d[NUM_BSSN_VARS + 0, exterior_mask] = e6phi[exterior_mask] * atmosphere_rho
        state_2d[NUM_BSSN_VARS + 1, exterior_mask] = 0.0
        state_2d[NUM_BSSN_VARS + 2, exterior_mask] = e6phi[exterior_mask] * atmosphere.tau_atm
        print(f"  Set {n_exterior} exterior points to atmosphere")

    # Fill ghost zones
    grid.fill_boundaries(state_2d)

    # Compute lambda^r via finite differences for consistency with evolution.
    # Following NRPy+ approach: even though h_ij = 0 in isotropic coords,
    # we compute lambda^r numerically to ensure the same FD operators are used
    # in initial data and evolution equations.
    # Result: lambda^r ~ O(Δr^4) - small but nonzero, consistent with FD truncation.
    print("  Computing lambda^r via finite differences (4th-order)...")
    compute_lambda_from_metric_fd(state_2d, grid, background)
    lambda_min = np.min(state_2d[idx_lambdar, NUM_GHOSTS:-NUM_GHOSTS])
    lambda_max = np.max(state_2d[idx_lambdar, NUM_GHOSTS:-NUM_GHOSTS])
    print(f"    lambda^r: min={lambda_min:.6e}, max={lambda_max:.6e}")
    print(f"    (Expected: ~O(10^-10) from 4th-order FD truncation)")

    # Re-fill boundaries after lambda computation
    grid.fill_boundaries(state_2d)

    # Package primitives
    primitives = (rho_arr, vr_arr, P_arr, eps_arr)

    print("  Initial data created successfully!\n")
    return state_2d, primitives


def plot_initial_comparison(tov_solution, initial_state_2d, grid, primitives, output_dir=".", suffix=""):
    """
    Plot initial data vs TOV comparison.

    Args:
        tov_solution: TOVSolutionIso object
        initial_state_2d: BSSN state array
        grid: Grid object
        primitives: Tuple (rho0, vr, p, eps)
        output_dir: Output directory
        suffix: Suffix for output filename (e.g., "_iso" for isotropic coordinates)
    """
    rho0_all, vr_all, p_all, eps_all = primitives

    r_tov = tov_solution.r_iso
    r_grid = grid.r[NUM_GHOSTS:-NUM_GHOSTS]
    rho_grid = rho0_all[NUM_GHOSTS:-NUM_GHOSTS]
    P_grid = p_all[NUM_GHOSTS:-NUM_GHOSTS]
    v_grid = vr_all[NUM_GHOSTS:-NUM_GHOSTS]
    alpha_grid = initial_state_2d[idx_lapse, NUM_GHOSTS:-NUM_GHOSTS]
    phi_grid = initial_state_2d[idx_phi, NUM_GHOSTS:-NUM_GHOSTS]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Density
    axes[0, 0].semilogy(r_tov, np.maximum(tov_solution.rho_baryon, 1e-20), 'b-', linewidth=2, label='TOV')
    axes[0, 0].semilogy(r_grid, np.maximum(rho_grid, 1e-20), 'r--', linewidth=1.5, alpha=0.7, label='Initial')
    axes[0, 0].axvline(tov_solution.R_iso, color='gray', linestyle=':', alpha=0.5, label=f"R_iso={tov_solution.R_iso:.2f}")
    axes[0, 0].set_xlabel(r'$r_{iso}$')
    axes[0, 0].set_ylabel(r'$\rho_0$')
    axes[0, 0].set_title('Baryon Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Pressure
    axes[0, 1].semilogy(r_tov, np.maximum(tov_solution.P, 1e-20), 'b-', linewidth=2, label='TOV')
    axes[0, 1].semilogy(r_grid, np.maximum(P_grid, 1e-20), 'r--', linewidth=1.5, alpha=0.7, label='Initial')
    axes[0, 1].axvline(tov_solution.R_iso, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel(r'$r_{iso}$')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Velocity
    axes[0, 2].plot(r_grid, v_grid, 'r-', linewidth=2, label='Initial')
    axes[0, 2].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 2].axvline(tov_solution.R_iso, color='gray', linestyle=':', alpha=0.5)
    axes[0, 2].set_xlabel(r'$r_{iso}$')
    axes[0, 2].set_ylabel(r'$v^r$')
    axes[0, 2].set_title('Radial Velocity (should be 0)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Lapse
    axes[1, 0].plot(r_tov, tov_solution.alpha, 'b-', linewidth=2, label='TOV')
    axes[1, 0].plot(r_grid, alpha_grid, 'r--', linewidth=1.5, alpha=0.7, label='Initial')
    axes[1, 0].axvline(tov_solution.R_iso, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel(r'$r_{iso}$')
    axes[1, 0].set_ylabel(r'$\alpha$')
    axes[1, 0].set_title('Lapse')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Conformal factor φ
    phi_tov = 0.25 * np.log(np.maximum(tov_solution.exp4phi, 1e-30))
    axes[1, 1].plot(r_tov, phi_tov, 'b-', linewidth=2, label='TOV')
    axes[1, 1].plot(r_grid, phi_grid, 'r--', linewidth=1.5, alpha=0.7, label='Initial')
    axes[1, 1].axvline(tov_solution.R_iso, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel(r'$r_{iso}$')
    axes[1, 1].set_ylabel(r'$\phi$')
    axes[1, 1].set_title(r'Conformal Factor $\phi$ (BSSN)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # h_rr should be zero
    h_rr_grid = initial_state_2d[idx_hrr, NUM_GHOSTS:-NUM_GHOSTS]
    axes[1, 2].plot(r_grid, h_rr_grid, 'r-', linewidth=2, label=r'$h_{rr}$')
    axes[1, 2].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 2].axvline(tov_solution.R_iso, color='gray', linestyle=':', alpha=0.5)
    axes[1, 2].set_xlabel(r'$r_{iso}$')
    axes[1, 2].set_ylabel(r'$h_{rr}$')
    axes[1, 2].set_title(r'$h_{rr}$ (should be 0 in isotropic)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f'TOV Initial Data (Isotropic): M={tov_solution.M_star:.4f}, R_iso={tov_solution.R_iso:.3f}',
                 fontsize=12, y=1.00)
    plt.tight_layout()

    filepath = os.path.join(output_dir, f'tov_initial_data_comparison{suffix}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_hamiltonian_constraint_iso(tov_solution, initial_state_2d, grid, background, hydro,
                                     polytrope_K, polytrope_Gamma, rho_central, output_dir=".",
                                     show=False):
    """
    Compute and plot Hamiltonian constraint for TOV initial data in isotropic coords.
    """
    from source.bssn.constraintsdiagnostic import get_constraints_diagnostic

    print("\n" + "="*80)
    print("HAMILTONIAN CONSTRAINT (Isotropic Coordinates)")
    print("="*80)

    print("\nComputing Hamiltonian constraint...")
    Ham, Mom = get_constraints_diagnostic(
        initial_state_2d, t=0.0, grid=grid,
        background=background, matter=hydro
    )

    Ham_profile = Ham[0, :] if Ham.ndim == 2 else Ham
    R_iso = tov_solution.R_iso

    # Exclude ghost zones
    N_total = len(grid.r)
    physical_mask = np.zeros(N_total, dtype=bool)
    physical_mask[NUM_GHOSTS:N_total-NUM_GHOSTS] = True

    mask_interior = (np.abs(grid.r) < R_iso) & physical_mask
    mask_exterior = (np.abs(grid.r) >= R_iso) & physical_mask

    Ham_interior = Ham_profile[mask_interior]
    Ham_exterior = Ham_profile[mask_exterior]
    Ham_physical = Ham_profile[physical_mask]

    print(f"\nInterior (r_iso < R_iso = {R_iso:.4f}):")
    print(f"  Mean |Ham| = {np.mean(np.abs(Ham_interior)):.6e}")
    print(f"  Max |Ham|  = {np.max(np.abs(Ham_interior)):.6e}")
    print(f"  RMS Ham    = {np.sqrt(np.mean(Ham_interior**2)):.6e}")

    print(f"\nExterior:")
    print(f"  Mean |Ham| = {np.mean(np.abs(Ham_exterior)):.6e}")
    print(f"  Max |Ham|  = {np.max(np.abs(Ham_exterior)):.6e}")

    print(f"\nGlobal:")
    print(f"  Mean |Ham| = {np.mean(np.abs(Ham_physical)):.6e}")
    print(f"  Max |Ham|  = {np.max(np.abs(Ham_physical)):.6e}")

    # Compute relative error (similar to NRPy+ tutorial)
    # Use the maximum absolute value as reference
    Ham_max = np.max(np.abs(Ham_physical))

    # Avoid division by zero
    if Ham_max < 1e-30:
        Ham_max = 1.0

    # Compute relative error and log10
    relative_error = np.abs(Ham_profile) / Ham_max
    log10_relative_error = np.log10(relative_error + 1e-20)  # Add small value to avoid log(0)

    # Normalize x-axis by stellar mass
    M_star = tov_solution.M_star
    r_normalized = grid.r / M_star

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: log10(Relative Error) vs r/M (like NRPy+ tutorial)
    axes[0].plot(r_normalized, log10_relative_error, 'b-', linewidth=1.5)
    axes[0].axvline(R_iso / M_star, color='red', linestyle='--', alpha=0.5,
                    label=f'R_iso/M={R_iso/M_star:.2f}')
    axes[0].set_xlabel(r'$r_{iso}/M$')
    axes[0].set_ylabel(r'$\log_{10}$(Relative Error)')
    axes[0].set_title(r'Hamiltonian Constraint: $\log_{10}$(|Ham|/|Ham|$_{max}$)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Absolute value in log scale
    axes[1].semilogy(grid.r, np.abs(Ham_profile) + 1e-20, 'b-', linewidth=1.5)
    axes[1].axvline(R_iso, color='red', linestyle='--', alpha=0.5, label=f'R_iso={R_iso:.2f}')
    axes[1].set_xlabel(r'$r_{iso}$')
    axes[1].set_ylabel('|Ham|')
    axes[1].set_title('Hamiltonian Constraint (absolute value)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'TOV Isotropic: Ham Constraint (K={polytrope_K}, Γ={polytrope_Gamma}, ρ_c={rho_central:.3e})',
                 fontsize=12, y=0.995)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'tov_hamiltonian_constraint_iso.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filepath}")

    #if show:
    #plt.show()
    #else:
    plt.close(fig)

    print("\n" + "="*80)
    print("Note: In isotropic coords with h_ij=0, constraint violations")
    print("      should come primarily from interpolation and FD errors.")
    print("="*80 + "\n")


# Backward compatibility
create_initial_data = create_initial_data_iso


if __name__ == "__main__":
    import argparse
    from source.core.grid import Grid
    from source.core.spacing import LinearSpacing
    from source.core.statevector import StateVector
    from source.matter.hydro.perfect_fluid import PerfectFluid
    from source.matter.hydro.eos import IdealGasEOS
    from source.matter.hydro.atmosphere import AtmosphereParams
    from source.backgrounds.sphericalbackground import FlatSphericalBackground
    from examples.TOV.tov_solver import TOVSolverIso, plot_tov_iso_diagnostics

    p = argparse.ArgumentParser(description='Generate TOV initial data in isotropic coordinates')
    p.add_argument('--r_max', type=float, default=20.0)
    p.add_argument('--num_points', type=int, default=100)
    p.add_argument('--K', type=float, default=100.0)
    p.add_argument('--Gamma', type=float, default=2.0)
    p.add_argument('--rho_central', type=float, default=1.28e-3)
    p.add_argument('--atmosphere_rho', type=float, default=1.0e-16)
    p.add_argument('--interp_order', type=int, default=12)
    p.add_argument('--save_npz', type=str, default=None)
    args = p.parse_args()

    print('='*70)
    print('TOV Initial Data – ISOTROPIC COORDINATES')
    print('='*70)
    print(f"Grid: N={args.num_points}, r_max={args.r_max}")
    print(f"EOS: K={args.K}, Gamma={args.Gamma}")
    print(f"rho_central={args.rho_central}, rho_atm={args.atmosphere_rho}")
    print(f"Interpolation order: {args.interp_order}\n")

    # Grid & background
    spacing = LinearSpacing(args.num_points, args.r_max)
    ATMOSPHERE = AtmosphereParams(rho_floor=args.atmosphere_rho)
    dummy_hydro = PerfectFluid(eos=IdealGasEOS(gamma=args.Gamma), spacetime_mode="dynamic",
                               atmosphere=ATMOSPHERE)
    state_vector = StateVector(dummy_hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)

    # Solve TOV in isotropic coordinates
    solver = TOVSolverIso(K=args.K, Gamma=args.Gamma)
    tov_solution = solver.solve(args.rho_central, r_max_iso=args.r_max, accuracy='high')

    print(f"TOV Solution (Isotropic):")
    print(f"  M       = {tov_solution.M_star:.6f}")
    print(f"  R_schw  = {tov_solution.R_schw:.6f}")
    print(f"  R_iso   = {tov_solution.R_iso:.6f}")
    print(f"  C       = {tov_solution.C:.4f}\n")

    # Plot TOV diagnostics
    fig = plot_tov_iso_diagnostics(tov_solution, args.r_max)
    plt.savefig('tov_solution_iso.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Build initial data
    hydro = PerfectFluid(eos=IdealGasEOS(gamma=args.Gamma), spacetime_mode="dynamic",
                         atmosphere=ATMOSPHERE)
    hydro.background = background

    initial_state_2d, primitives = create_initial_data_iso(
        tov_solution, grid, background, hydro.eos,
        atmosphere=hydro.atmosphere,
        polytrope_K=args.K, polytrope_Gamma=args.Gamma,
        interp_order=args.interp_order
    )

    # Diagnostics
    plot_initial_comparison(tov_solution, initial_state_2d, grid, primitives)
    plot_hamiltonian_constraint_iso(tov_solution, initial_state_2d, grid, background, hydro,
                                     args.K, args.Gamma, args.rho_central)

    if args.save_npz:
        np.savez_compressed(args.save_npz,
                            state=initial_state_2d,
                            r=grid.r,
                            r_iso=tov_solution.r_iso,
                            r_schw=tov_solution.r_schw,
                            tov_rho=tov_solution.rho_baryon,
                            tov_P=tov_solution.P,
                            tov_alpha=tov_solution.alpha,
                            tov_exp4phi=tov_solution.exp4phi)
        print(f"\nSaved initial data to {args.save_npz}")

    print('\nSaved diagnostic figures:')
    print('  - tov_solution_iso.png')
    print('  - tov_initial_data_comparison_iso.png')
    print('  - tov_hamiltonian_constraint_iso.png')
