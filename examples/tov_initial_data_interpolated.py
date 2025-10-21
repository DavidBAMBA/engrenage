"""
TOV Initial Data with High-Order Interpolation + Rigorous ADM→BSSN Conversion.

This module combines the best of both worlds:
1. High-order Lagrange interpolation with Gibbs phenomenon protection 
2. Rigorous ADM→BSSN conversion enforcing det(γ̄) = det(ĝ) constraint

Key features:
- TOV solved on fine, independent grid → interpolated to evolution grid
- Interpolation stencil NEVER crosses stellar surface (avoids Gibbs phenomenon)
- Proper ADM→BSSN conversion: TOV → ADM → BSSN (not direct TOV→BSSN)
- Enforces BSSN constraint: det(γ̄) = det(ĝ)
- Correctly computes φ from determinants (not ad-hoc formulas)

"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from source.core.spacing import NUM_GHOSTS
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import (
    NUM_BSSN_VARS,
    idx_phi, idx_hrr, idx_htt, idx_hpp,
    idx_K, idx_arr, idx_att, idx_app, idx_lapse,
)
from source.backgrounds.sphericalbackground import i_r, i_t, i_p
# Note: We no longer need these imports since we follow NRPy's approach
# from source.bssn.tensoralgebra import get_bar_gamma_LL
# from source.matter.hydro.cons2prim import prim_to_cons


def interpolate_tov_avoiding_surface(r_target, tov_solution, field_name,
                                      interp_order=11, atmosphere_value=None):
    """
    Interpolate TOV solution field, ensuring stencil doesn't cross surface.
    This prevents Gibbs phenomenon at stellar surface discontinuity.

    Args:
        r_target: Radius where to interpolate
        tov_solution: Dict with TOV solution (must have 'r', 'R', 'M_star', and field_name)
        field_name: Name of field to interpolate ('rho_baryon', 'P', 'exp4phi', 'alpha', etc.)
        interp_order: Interpolation order 11
        atmosphere_value: Value to return if r_target is outside star (deprecated, will use Schwarzschild)

    Returns:
        Interpolated value at r_target

    Implementation details:
    - Uses Lagrange polynomial interpolation
    - Finds stellar surface index (Rbar_idx)
    - If r_target < R_star: stencil stays INSIDE star (idx_max ≤ surf_idx+1)
    - If r_target ≥ R_star:
        * For metric quantities (exp4phi, alpha): uses Schwarzschild solution
        * For matter quantities (rho_baryon, P): uses atmosphere_value or zero
    - This prevents interpolation across discontinuity → no Gibbs phenomenon
    """
    r_tov = tov_solution['r']
    field_tov = tov_solution[field_name]
    R_star = tov_solution['R']
    M_star = tov_solution['M_star']

    # Handle negative radii (spherical symmetry: f(r) = f(-r))
    r_use = abs(r_target)

    # If outside star, use exact Schwarzschild solution for metric quantities
    # or atmosphere values for matter quantities
    if r_use > R_star:
        if field_name == 'exp4phi':
            # Schwarzschild: exp4phi = (1 - 2M/r)^(-1)
            one_minus = max(1.0 - 2.0 * M_star / max(r_use, 0.1), 1e-10)
            return 1.0 / one_minus
        elif field_name == 'alpha':
            # Schwarzschild: alpha = sqrt(1 - 2M/r)
            one_minus = max(1.0 - 2.0 * M_star / max(r_use, 0.1), 1e-10)
            return np.sqrt(one_minus)
        elif field_name in ['rho_baryon', 'P', 'rho', 'epsilon']:
            # Matter quantities: use atmosphere or zero
            if atmosphere_value is not None:
                return atmosphere_value
            else:
                return 0.0
        else:
            # Unknown field: fallback to atmosphere_value or zero
            if atmosphere_value is not None:
                return atmosphere_value
            else:
                return 0.0

    # Clamp to data range
    r_use = max(r_use, r_tov[0])
    r_use = min(r_use, r_tov[-1])

    # Find stellar surface index in TOV data
    surf_idx = np.argmin(np.abs(r_tov - R_star))

    # Find target index using bisection (more robust than searchsorted for irregular grids)
    idx = np.searchsorted(r_tov, r_use)
    idx = max(0, min(idx, len(r_tov)-1))

    # Determine stencil bounds centered around idx
    half_stencil = interp_order // 2
    idx_min = max(0, idx - half_stencil)
    idx_max = min(len(r_tov), idx_min + interp_order)

    # CRITICAL: Don't allow stencil to cross stellar surface
    # This is the key difference from standard interpolation!
    if r_use < R_star:
        # Inside star: stencil must stay inside (all points < R_star)
        idx_max = min(idx_max, surf_idx + 1)
        idx_min = max(0, idx_max - interp_order)
    else:
        # Outside star: stencil must stay outside (all points ≥ R_star)
        idx_min = max(idx_min, surf_idx)
        idx_max = min(len(r_tov), idx_min + interp_order)

    # Extract stencil
    r_stencil = r_tov[idx_min:idx_max]
    field_stencil = field_tov[idx_min:idx_max]

    # Lagrange polynomial interpolation
    # L_i(r) = Π_{j≠i} (r - r_j) / (r_i - r_j)
    # f(r) ≈ Σ_i f_i L_i(r)
    result = 0.0
    for i in range(len(r_stencil)):
        L_i = 1.0
        for j in range(len(r_stencil)):
            if i != j:
                L_i *= (r_use - r_stencil[j]) / (r_stencil[i] - r_stencil[j])
        result += field_stencil[i] * L_i

    return result


def compute_adm_from_tov_interpolated(tov_solution, grid, atmosphere, interp_order=11):
    """
    Extract ADM variables from TOV solution using high-order interpolation.

    TOV solution is always in Schwarzschild coordinates:
        ds² = -α² dt² + γ_rr dr² + r² dΩ²
        α = exp(ν/2)
        γ_rr = (1 - 2M/r)^(-1) = exp4phi
        γ_θθ = r²
        γ_φφ = r² sin²θ

    Args:
        tov_solution: TOV solution dict (on fine, independent grid)
        grid: Engrenage evolution grid
        atmosphere: AtmosphereParams object
        interp_order: Lagrange interpolation order (default 11)

    Returns:
        adm_vars: Dict with keys 'alpha', 'beta_U', 'gamma_LL', 'K_LL'
                  All arrays have shape (N,) or (N,3) or (N,3,3)
    """
    N = grid.N
    r_grid = grid.r
    R_star = tov_solution['R']
    M_star = tov_solution['M_star']

    # Initialize ADM variables
    alpha_grid = np.ones(N)
    beta_U = np.zeros((N, 3))  # β^i = 0 (static star)
    gamma_LL = np.zeros((N, 3, 3))
    K_LL = np.zeros((N, 3, 3))  # K_ij = 0 (static star)

    print(f"  Interpolating TOV solution to evolution grid (order {interp_order})...")
    print(f"    Stellar radius: R = {R_star:.6f}")
    print(f"    Evolution grid: N = {N}, r ∈ [{r_grid[0]:.3f}, {r_grid[-1]:.3f}]")

    # Interpolate each grid point
    for i, r in enumerate(r_grid):
        r_abs = abs(r)
        is_interior = r_abs < R_star

        if is_interior:
            # INTERIOR: Interpolate TOV quantities with surface protection
            exp4phi = interpolate_tov_avoiding_surface(
                r_abs, tov_solution, 'exp4phi', interp_order, 1.0)
            alpha = interpolate_tov_avoiding_surface(
                r_abs, tov_solution, 'alpha', interp_order, 1.0)

            # Enforce positivity
            exp4phi = max(exp4phi, 1e-10)
            alpha = max(alpha, 1e-10)
        else:
            # EXTERIOR: Use Schwarzschild vacuum solution
            one_minus = max(1.0 - 2.0 * M_star / max(r_abs, 0.1), 1e-10)
            alpha = np.sqrt(one_minus)
            exp4phi = 1.0 / one_minus

        alpha_grid[i] = alpha

        # Build physical 3-metric γ_ij in Schwarzschild coordinates
        # ds² = exp4phi dr² + r² dΩ²
        gamma_LL[i, i_r, i_r] = exp4phi
        gamma_LL[i, i_t, i_t] = r_abs ** 2
        gamma_LL[i, i_p, i_p] = r_abs ** 2

    return {
        'alpha': alpha_grid,
        'beta_U': beta_U,
        'gamma_LL': gamma_LL,
        'K_LL': K_LL,
    }


def convert_adm_to_bssn(adm_vars, grid, background):
    """
    Convert ADM variables to BSSN variables .


    Key steps
    1. Compute det(γ) and det(ĝ)
    2. Compute γ̄_ij enforcing det(γ̄) = det(ĝ) constraint
    3. Compute φ = (1/12) ln(det(γ) / det(γ̄))
    4. Compute h_ij = (γ̄_ij - ĝ_ij) / Re_ij
    5. Compute K = γ^ij K_ij (trace)
    6. Compute Ā_ij = conformal traceless extrinsic curvature
    7. Compute a_ij = Ā_ij / Re_ij

    Note: We do NOT compute λ̄^i here (requires derivatives). For static TOV,
          λ̄^i = 0 is a good initial value; boundary conditions will fix it.

    Args:
        adm_vars: Dict with 'alpha', 'beta_U', 'gamma_LL', 'K_LL'
        grid: Engrenage grid
        background: FlatSphericalBackground

    Returns:
        bssn_state: (NUM_VARS, N) array with BSSN variables
    """
    N = grid.N
    alpha = adm_vars['alpha']
    beta_U = adm_vars['beta_U']
    gamma_LL = adm_vars['gamma_LL']
    K_LL = adm_vars['K_LL']

    # Get background metric ĝ_ij
    ghat_LL = background.hat_gamma_LL  # (N, 3, 3)
    ReDD = background.scaling_matrix   # (N, 3, 3)

    print("  Converting ADM → BSSN (enforcing det(γ̄) = det(ĝ) constraint)...")

    # Step 1: Compute determinants
    det_ghat = np.zeros(N)
    det_gamma = np.zeros(N)
    gamma_UU = np.zeros((N, 3, 3))

    for i in range(N):
        det_ghat[i] = np.linalg.det(ghat_LL[i])
        det_gamma[i] = np.linalg.det(gamma_LL[i])
        gamma_UU[i] = np.linalg.inv(gamma_LL[i])

    # Step 2: Compute conformal metric enforcing det(γ̄) = det(ĝ)
    # γ̄_ij = (det(ĝ) / det(γ))^(1/3) × γ_ij
    # This is the KEY constraint that makes BSSN well-posed!
    gammabar_LL = np.zeros((N, 3, 3))
    for i in range(N):
        conformal_factor_3d = (det_ghat[i] / max(det_gamma[i], 1e-300)) ** (1.0/3.0)
        gammabar_LL[i] = conformal_factor_3d * gamma_LL[i]

    # Sanity check: verify det(γ̄) ≈ det(ĝ)
    det_gammabar = np.array([np.linalg.det(gammabar_LL[i]) for i in range(N)])
    max_det_violation = np.max(np.abs(det_gammabar - det_ghat) / np.maximum(det_ghat, 1e-30))
    print(f"    det(γ̄) = det(ĝ) constraint violation: max = {max_det_violation:.3e}")

    # Step 3: Compute conformal factor φ = (1/12) ln(det(γ) / det(γ̄))
    # Note: det(γ̄) = det(ĝ) by construction, so φ = (1/12) ln(det(γ) / det(ĝ))
    phi = np.zeros(N)
    for i in range(N):
        ratio = det_gamma[i] / max(det_gammabar[i], 1e-300)
        phi[i] = (1.0/12.0) * np.log(max(ratio, 1e-300))

    # Step 4: Compute rescaled conformal metric deviation h_ij
    # h_ij = (γ̄_ij - ĝ_ij) / Re_ij
    h_LL = np.zeros((N, 3, 3))
    for i in range(N):
        for j in range(3):
            for k in range(3):
                if ReDD[i, j, k] != 0:
                    h_LL[i, j, k] = (gammabar_LL[i, j, k] - ghat_LL[i, j, k]) / ReDD[i, j, k]

    # Step 5: Compute trace of extrinsic curvature K = γ^ij K_ij
    K = np.zeros(N)
    for i in range(N):
        K[i] = np.einsum('ij,ij->', gamma_UU[i], K_LL[i])

    # Step 6: Compute conformal traceless extrinsic curvature
    # Ā_ij = (det(ĝ)/det(γ))^(1/3) × [K_ij - (1/3) γ_ij K]
    Abar_LL = np.zeros((N, 3, 3))
    for i in range(N):
        conformal_factor_3d = (det_ghat[i] / max(det_gamma[i], 1e-300)) ** (1.0/3.0)
        for j in range(3):
            for k in range(3):
                A_LL_jk = K_LL[i, j, k] - (1.0/3.0) * gamma_LL[i, j, k] * K[i]
                Abar_LL[i, j, k] = conformal_factor_3d * A_LL_jk

    # Step 7: Compute rescaled conformal traceless extrinsic curvature a_ij
    # a_ij = Ā_ij / Re_ij
    a_LL = np.zeros((N, 3, 3))
    for i in range(N):
        for j in range(3):
            for k in range(3):
                if ReDD[i, j, k] != 0:
                    a_LL[i, j, k] = Abar_LL[i, j, k] / ReDD[i, j, k]

    # Pack into BSSN state vector
    state_2d = np.zeros((grid.NUM_VARS, N))

    # Conformal metric variables
    state_2d[idx_phi, :] = phi
    state_2d[idx_hrr, :] = h_LL[:, i_r, i_r]
    state_2d[idx_htt, :] = h_LL[:, i_t, i_t]
    state_2d[idx_hpp, :] = h_LL[:, i_p, i_p]

    # Extrinsic curvature variables
    state_2d[idx_K, :] = K
    state_2d[idx_arr, :] = a_LL[:, i_r, i_r]
    state_2d[idx_att, :] = a_LL[:, i_t, i_t]
    state_2d[idx_app, :] = a_LL[:, i_p, i_p]

    # Gauge variables
    state_2d[idx_lapse, :] = alpha

    print("    ADM → BSSN conversion complete")
    return state_2d


def create_initial_data_interpolated(tov_solution, grid, background, eos,
                                      atmosphere=None,
                                      polytrope_K=None, polytrope_Gamma=None,
                                      interp_order=11):
    """
    Create BSSN + hydro initial data from TOV solution.

    This combines:
    1. High-order Lagrange interpolation with Gibbs protection
    2. Rigorous ADM→BSSN conversion enforcing constraints

    Workflow:
        TOV (fine grid) → [interpolate] → ADM (evolution grid) → BSSN

    TOV solution is always in Schwarzschild coordinates.

    Args:
        tov_solution: TOV solution dict on fine grid (independent of evolution grid)
        grid: Engrenage evolution grid
        background: FlatSphericalBackground
        eos: Equation of state
        atmosphere: AtmosphereParams object
        polytrope_K, polytrope_Gamma: EOS parameters for polytropic EOS
        interp_order: Lagrange interpolation order (default 11)

    Returns:
        state_2d: (NUM_VARS, N) initial data array with BSSN + hydro
    """
    from source.matter.hydro.atmosphere import AtmosphereParams

    # Handle atmosphere
    if atmosphere is None:
        atmosphere = AtmosphereParams()
    elif not isinstance(atmosphere, AtmosphereParams):
        raise TypeError("atmosphere must be AtmosphereParams")

    atmosphere_rho = atmosphere.rho_floor
    p_atm = atmosphere.p_floor

    # Set atmosphere pressure consistent with EOS
    if polytrope_K is not None and polytrope_Gamma is not None:
        p_atm = max(p_atm, polytrope_K * (atmosphere_rho ** polytrope_Gamma))

    R_star = tov_solution['R']

    print(f"\nCreating TOV initial data with high-order interpolation + ADM→BSSN:")
    print(f"  Coordinate system: Schwarzschild")
    print(f"  Interpolation order: {interp_order}")

    # Step 1: TOV → ADM (with high-order interpolation + Gibbs protection)
    adm_vars = compute_adm_from_tov_interpolated(
        tov_solution, grid, atmosphere, interp_order)

    # Step 2: ADM → BSSN (rigorous conversion enforcing constraints)
    bssn_state = convert_adm_to_bssn(adm_vars, grid, background)

    # Step 3: Set hydro primitives and convert to conservatives
    print("  Setting hydro variables (primitives → conservatives)...")

    r_grid = np.abs(grid.r)
    N = grid.N

    # Interpolate hydro quantities with surface protection
    rho_arr = np.zeros(N)
    P_arr = np.zeros(N)

    for i in range(N):
        r_abs = abs(grid.r[i])
        is_interior = r_abs < R_star

        if is_interior:
            rho_arr[i] = interpolate_tov_avoiding_surface(
                r_abs, tov_solution, 'rho_baryon', interp_order, atmosphere_rho)
            P_arr[i] = interpolate_tov_avoiding_surface(
                r_abs, tov_solution, 'P', interp_order, p_atm)

            # Enforce floors
            rho_arr[i] = max(rho_arr[i], atmosphere_rho)
            P_arr[i] = max(P_arr[i], p_atm)
        else:
            # Exterior: EXACT atmosphere values (no interpolation)
            rho_arr[i] = atmosphere_rho
            P_arr[i] = p_atm

    # Convert primitives → conservatives following NRPy approach
    # For TOV star with v=0: T^00 = ρ_total = ρ₀(1 + ε)
    # where ε = P/[(Γ-1)ρ₀] for polytropic EOS

    # Compute specific internal energy
    if polytrope_Gamma is not None:
        # For polytropic EOS: ε = P / [(Γ-1) ρ₀]
        eps_arr = P_arr / np.maximum((polytrope_Gamma - 1.0) * rho_arr, 1e-30)
    else:
        # Use EOS to compute epsilon from pressure and density
        eps_arr = np.zeros(N)
        for i in range(N):
            eps_arr[i] = eos.epsilon_from_P_rho(P_arr[i], rho_arr[i])

    # For static TOV (v=0), the conservative variables simplify:
    # D = ρ₀ (conserved baryon density)
    # S^r = 0 (no momentum for static star)
    # τ = ρ₀ ε (conserved energy minus rest mass)
    # This follows from T^μν for perfect fluid with v=0
    D_arr = rho_arr
    Sr_arr = np.zeros_like(rho_arr)
    tau_arr = rho_arr * eps_arr

    # Store hydro conservatives
    bssn_state[NUM_BSSN_VARS + 0, :] = D_arr
    bssn_state[NUM_BSSN_VARS + 1, :] = Sr_arr
    bssn_state[NUM_BSSN_VARS + 2, :] = tau_arr

    # Ensure exterior is EXACTLY atmosphere (prevent any interpolation artifacts)
    exterior_mask = r_grid > R_star
    n_exterior = np.sum(exterior_mask)
    if n_exterior > 0:
        bssn_state[NUM_BSSN_VARS + 0, exterior_mask] = atmosphere_rho
        bssn_state[NUM_BSSN_VARS + 1, exterior_mask] = 0.0
        bssn_state[NUM_BSSN_VARS + 2, exterior_mask] = atmosphere.tau_atm
        print(f"  Set {n_exterior} exterior points (r > {R_star:.6f}) to atmosphere")

    # Fill ghost zones
    grid.fill_boundaries(bssn_state)

    print("  Initial data created successfully!\n")
    return bssn_state


# ============================================================================
# Diagnostic/Validation Functions 
# ============================================================================

def plot_initial_comparison(tov_solution, initial_state_2d, grid, hydro, output_dir="."):
    """Plot initial data vs TOV comparison (ρ0, P, v^r, α)."""
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(initial_state_2d, bssn_vars, grid)
    prim = hydro._get_primitives(bssn_vars, grid.r, grid=grid)

    r_tov = tov_solution['r']
    r_grid = grid.r[NUM_GHOSTS:-NUM_GHOSTS]
    rho_grid = prim['rho0'][NUM_GHOSTS:-NUM_GHOSTS]
    P_grid = prim['p'][NUM_GHOSTS:-NUM_GHOSTS]
    v_grid = prim['vr'][NUM_GHOSTS:-NUM_GHOSTS]
    alpha_grid = initial_state_2d[idx_lapse, NUM_GHOSTS:-NUM_GHOSTS]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].semilogy(r_tov, tov_solution['rho_baryon'], 'b-', linewidth=2, label='TOV')
    axes[0, 0].semilogy(r_grid, np.maximum(rho_grid, 1e-20), 'r--', linewidth=1.5, alpha=0.7, label='Initial (t=0)')
    axes[0, 0].axvline(tov_solution['R'], color='gray', linestyle=':', alpha=0.5, label=f"R={tov_solution['R']:.2f}")
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel(r'$\rho_0$')
    axes[0, 0].set_title('Baryon Density')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(r_tov, tov_solution['P'], 'b-', linewidth=2, label='TOV')
    axes[0, 1].semilogy(r_grid, np.maximum(P_grid, 1e-20), 'r--', linewidth=1.5, alpha=0.7, label='Initial (t=0)')
    axes[0, 1].axvline(tov_solution['R'], color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('r'); axes[0, 1].set_ylabel('P'); axes[0, 1].set_title('Pressure')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(r_grid, v_grid, 'r-', linewidth=2, label='Initial (t=0)')
    axes[1, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(tov_solution['R'], color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('r'); axes[1, 0].set_ylabel(r'$v^r$'); axes[1, 0].set_title('Radial Velocity')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(r_tov, tov_solution['alpha'], 'b-', linewidth=2, label='TOV')
    axes[1, 1].plot(r_grid, alpha_grid, 'r--', linewidth=1.5, alpha=0.7, label='Initial (t=0)')
    axes[1, 1].axvline(tov_solution['R'], color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('r'); axes[1, 1].set_ylabel(r'$\alpha$'); axes[1, 1].set_title('Lapse')
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('TOV vs Initial Data (t=0)', fontsize=14, y=1.00)
    plt.tight_layout()
    import os
    filepath = os.path.join(output_dir, 'tov_initial_data_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")



# Backward compatibility alias
create_initial_data = create_initial_data_interpolated


if __name__ == "__main__":
    import argparse
    from source.core.grid import Grid
    from source.core.spacing import LinearSpacing
    from source.core.statevector import StateVector
    from source.matter.hydro.perfect_fluid import PerfectFluid
    from source.matter.hydro.eos import IdealGasEOS
    from source.backgrounds.sphericalbackground import FlatSphericalBackground
    from examples.tov_solver import TOVSolver, plot_tov_diagnostics

    p = argparse.ArgumentParser(description='Generate and validate TOV initial data in Engrenage.')
    p.add_argument('--r_max', type=float, default=20.0)
    p.add_argument('--num_points', type=int, default=3000)
    p.add_argument('--K', type=float, default=100.0)
    p.add_argument('--Gamma', type=float, default=2.0)
    p.add_argument('--rho_central', type=float, default=1.28e-3)
    p.add_argument('--atmosphere_rho', type=float, default=1.0e-12)
    p.add_argument('--interp_order', type=int, default=11)
    p.add_argument('--save_npz', type=str, default='')
    args = p.parse_args()

    print('='*70)
    print('TOV Initial Data – Engrenage (High-Order Interpolation + ADM→BSSN)')
    print('='*70)
    print(f"Grid: N={args.num_points}, r_max={args.r_max}")
    print(f"EOS (TOV): K={args.K}, Gamma={args.Gamma}")
    print(f"rho_central={args.rho_central}, rho_atm={args.atmosphere_rho}")
    print(f"Interpolation order: {args.interp_order}\n")

    # Grid & background
    spacing = LinearSpacing(args.num_points, args.r_max)
    dummy_hydro = PerfectFluid(eos=IdealGasEOS(gamma=args.Gamma), spacetime_mode="dynamic",
                               atmosphere=args.atmosphere_rho)
    state_vector = StateVector(dummy_hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)

    # Solve TOV on fine grid (independent of evolution grid)
    solver = TOVSolver(K=args.K, Gamma=args.Gamma)
    # Use 2x denser grid for TOV solution
    r_tov_fine = np.linspace(0, args.r_max, 2 * args.num_points)
    tov_solution = solver.solve(args.rho_central, r_grid=r_tov_fine, r_max=args.r_max)
    print(f"TOV: M={tov_solution['M_star']:.6f}, R={tov_solution['R']:.3f}, C={tov_solution['C']:.4f}\n")

    plot_tov_diagnostics(tov_solution, args.r_max)

    # Build initial data
    hydro = PerfectFluid(eos=IdealGasEOS(gamma=args.Gamma), spacetime_mode="dynamic",
                         atmosphere=args.atmosphere_rho)
    hydro.background = background
    initial_state_2d = create_initial_data_interpolated(
        tov_solution, grid, background, hydro.eos,
        atmosphere=hydro.atmosphere,
        polytrope_K=args.K, polytrope_Gamma=args.Gamma,
        interp_order=args.interp_order
    )

    # Diagnostics/plots
    plot_initial_comparison(tov_solution, initial_state_2d, grid, hydro)

    if args.save_npz:
        np.savez_compressed(args.save_npz,
                            state=initial_state_2d,
                            r=grid.r,
                            tov_r=tov_solution['r'],
                            tov_rho=tov_solution['rho_baryon'],
                            tov_P=tov_solution['P'],
                            tov_M=tov_solution['M'],
                            tov_alpha=tov_solution['alpha'],
                            tov_exp4phi=tov_solution['exp4phi'])
        print(f"\nSaved initial data NPZ to {args.save_npz}")

    print('\nSaved diagnostic figures:')
    print('  - tov_solution.png')
    print('  - tov_initial_data_comparison.png')
