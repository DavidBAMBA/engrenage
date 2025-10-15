"""
TOV Initial Data utilities for Engrenage.

- Build BSSN + hydro initial data on a given Engrenage grid from a TOV solution
- Generate diagnostics/plots to corroborate correctness of the initial data

Usage (import):
  from examples.tov_initial_data import (
      create_initial_data,
      plot_initial_comparison,
      plot_initial_geometry_comparison,
      plot_hydrostatic_equilibrium_residual,
      plot_cons2prim_consistency,
  )
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Ensure repo root is on sys.path so `source` imports work when run standalone
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
from source.bssn.tensoralgebra import get_bar_gamma_LL
from source.matter.hydro.cons2prim import prim_to_cons


def create_initial_data(tov_solution, grid, background, eos, atmosphere_rho=None,
                        polytrope_K=None, polytrope_Gamma=None,
                        use_hydrobase_tau=True,
                        p_floor: float = 1e-15,
                        atmosphere=None):
    """Create BSSN + hydro initial data from TOV solution on the Engrenage grid.

    - Maps the full TOV solution (including exterior) to the positive-r part of the grid.
    - Builds BSSN variables enforcing det(γ̄)=det(ĝ), sets K_ij=0, and copies α from TOV.
    - Sets hydro primitives from TOV (ρ0,P,v=0) and converts to conservatives using physical γ_rr.

    Args:
        atmosphere_rho: (deprecated) Use atmosphere instead
        atmosphere: AtmosphereParams object (preferred)
        p_floor: Pressure floor (deprecated if atmosphere provided)

    Atmosphere pressure is set consistently with the TOV EOS if (K,Γ) are provided.
    """
    # Handle backward compatibility
    if atmosphere is not None:
        # Use centralized atmosphere
        from source.matter.hydro.atmosphere import AtmosphereParams
        if not isinstance(atmosphere, AtmosphereParams):
            raise TypeError("atmosphere must be AtmosphereParams")
        atmosphere_rho = atmosphere.rho_floor
        p_floor = atmosphere.p_floor
    elif atmosphere_rho is None:
        # No atmosphere specified, use default
        atmosphere_rho = 1e-13
    r_tov = tov_solution['r']

    # Atmosphere pressure consistent with TOV EOS if provided
    if polytrope_K is not None and polytrope_Gamma is not None:
        p_atm = polytrope_K * (atmosphere_rho ** polytrope_Gamma)
    elif hasattr(eos, 'K') and hasattr(eos, 'gamma'):
        p_atm = eos.K * (atmosphere_rho ** eos.gamma)
    else:
        p_atm = p_floor
    # Ensure pressure floor consistency with cons2prim to avoid inconsistencies in inversion
    p_atm = max(p_atm, p_floor)

    # Check if TOV is on same grid (no interpolation needed!)
    r_grid_positive = grid.r[grid.r > 0]
    same_grid = len(r_tov) <= len(r_grid_positive) and np.allclose(r_tov, r_grid_positive[:len(r_tov)])

    if same_grid:
        # Use full TOV (interior + exterior continuation)
        rho_tov_vals = tov_solution['rho_baryon']
        P_tov_vals = tov_solution['P']
        nu_tov_vals = tov_solution['nu']
        M_tov_vals = tov_solution['M']

        n_tov = len(r_tov)
        rho_grid = np.zeros(grid.N)
        P_grid = np.zeros(grid.N)
        nu_grid = np.zeros(grid.N)
        M_grid = np.zeros(grid.N)
        exp4phi_grid = np.ones(grid.N)
        alpha_grid = np.ones(grid.N)

        # Positive radii match TOV radii in order
        positive_mask = grid.r > 0
        positive_indices = np.where(positive_mask)[0]

        rho_grid[positive_indices[:n_tov]] = rho_tov_vals
        P_grid[positive_indices[:n_tov]] = P_tov_vals
        nu_grid[positive_indices[:n_tov]] = nu_tov_vals
        M_grid[positive_indices[:n_tov]] = M_tov_vals

        if 'exp4phi' in tov_solution:
            exp4phi_grid[positive_indices[:n_tov]] = tov_solution['exp4phi']
        if 'alpha' in tov_solution:
            alpha_grid[positive_indices[:n_tov]] = tov_solution['alpha']

        # Beyond provided TOV points: fill exterior using Schwarzschild values
        if n_tov < positive_indices.size:
            M_star = M_tov_vals[-1]
            for idx in positive_indices[n_tov:]:
                r = grid.r[idx]
                one_minus = 1.0 - 2.0 * M_star / r
                if one_minus > 0.0:
                    alpha_grid[idx] = np.sqrt(one_minus)
                    exp4phi_grid[idx] = 1.0 / one_minus
                    nu_grid[idx] = np.log(one_minus)
                M_grid[idx] = M_star
            rho_grid[positive_indices[n_tov:]] = atmosphere_rho
            P_grid[positive_indices[n_tov:]] = p_atm

        # Negative radii: mirror central values for metric/lapse (for ghost fill)
        if 'exp4phi' in tov_solution:
            exp4phi_grid[~positive_mask] = tov_solution['exp4phi'][0]
        if 'alpha' in tov_solution:
            alpha_grid[~positive_mask] = tov_solution['alpha'][0]

    else:
        # Fallback: interpolate
        rho_tov_interp = interp1d(r_tov, tov_solution['rho_baryon'], kind='cubic',
                                  bounds_error=False, fill_value=(tov_solution['rho_baryon'][0], atmosphere_rho))
        P_tov_interp = interp1d(r_tov, tov_solution['P'], kind='cubic',
                                bounds_error=False, fill_value=(tov_solution['P'][0], p_atm))
        nu_tov_interp = interp1d(r_tov, tov_solution['nu'], kind='cubic',
                                 bounds_error=False, fill_value=(tov_solution['nu'][0], 0.0))
        M_tov_interp = interp1d(r_tov, tov_solution['M'], kind='cubic',
                                bounds_error=False, fill_value=(0.0, tov_solution['M'][-1]))

        rho_grid = rho_tov_interp(grid.r)
        P_grid = P_tov_interp(grid.r)
        nu_grid = nu_tov_interp(grid.r)
        M_grid = M_tov_interp(grid.r)

        exp4phi_interp = interp1d(r_tov, tov_solution['exp4phi'], kind='cubic',
                                  bounds_error=False, fill_value=(1.0, tov_solution['exp4phi'][-1]))
        alpha_interp = interp1d(r_tov, tov_solution['alpha'], kind='cubic',
                                bounds_error=False, fill_value=(1.0, tov_solution['alpha'][-1]))
        exp4phi_grid = exp4phi_interp(grid.r)
        alpha_grid = alpha_interp(grid.r)

    # Build BSSN state
    state_2d = np.zeros((grid.NUM_VARS, grid.N))
    for i, r in enumerate(grid.r):
        exp4phi_tov = exp4phi_grid[i]
        gamma_phys_rr = exp4phi_tov
        gamma_phys_thth = r ** 2
        gamma_phys_phph = r ** 2

        phi = np.log(max(exp4phi_tov, 1e-300)) / 12.0
        exp_minus_4phi = np.exp(-4.0 * phi)
        gammabar_rr = exp_minus_4phi * gamma_phys_rr
        gammabar_thth = exp_minus_4phi * gamma_phys_thth
        gammabar_phph = exp_minus_4phi * gamma_phys_phph

        ghatDD = background.hat_gamma_LL
        ReDD = background.scaling_matrix
        h_rr = (gammabar_rr - ghatDD[i, i_r, i_r]) / ReDD[i, i_r, i_r] if ReDD[i, i_r, i_r] != 0 else 0.0
        h_tt = (gammabar_thth - ghatDD[i, i_t, i_t]) / ReDD[i, i_t, i_t] if ReDD[i, i_t, i_t] != 0 else 0.0
        h_pp = (gammabar_phph - ghatDD[i, i_p, i_p]) / ReDD[i, i_p, i_p] if ReDD[i, i_p, i_p] != 0 else 0.0

        state_2d[idx_phi, i] = phi
        state_2d[idx_hrr, i] = h_rr
        state_2d[idx_htt, i] = h_tt
        state_2d[idx_hpp, i] = h_pp
        state_2d[idx_K, i] = 0.0
        state_2d[idx_arr, i] = 0.0
        state_2d[idx_att, i] = 0.0
        state_2d[idx_app, i] = 0.0
        state_2d[idx_lapse, i] = alpha_grid[i]

    # Hydro: set primitives and convert to conservatives
    # Option A (default): compute tau from HydroBase prescription (cold polytrope):
    #   eps_cold = P / [(Gamma-1) rho]  -> tau = rho * eps_cold  (for v=0)
    # If polytrope parameters are not provided, fall back to eos.eps_from_rho_p
    rho_arr = np.maximum(rho_grid, atmosphere_rho)
    P_arr = np.maximum(P_grid, p_atm)
    if use_hydrobase_tau:
        if polytrope_Gamma is not None:
            eps_arr = P_arr / np.maximum((polytrope_Gamma - 1.0) * rho_arr, 1e-30)
        else:
            eps_arr = eos.eps_from_rho_p(rho_arr, P_arr)
        D_arr = rho_arr  # v=0 => W=1
        Sr_arr = np.zeros_like(rho_arr)
        tau_arr = rho_arr * eps_arr  # v=0 => tau = rho * eps
        state_2d[NUM_BSSN_VARS + 0, :] = D_arr
        state_2d[NUM_BSSN_VARS + 1, :] = Sr_arr
        state_2d[NUM_BSSN_VARS + 2, :] = tau_arr
    else:
        for i in range(grid.N):
            rho = rho_arr[i]
            P = P_arr[i]
            gamma_rr_phys = exp4phi_grid[i]
            D, Sr, tau = prim_to_cons(rho, 0.0, P, gamma_rr_phys, eos)
            state_2d[NUM_BSSN_VARS + 0, i] = D
            state_2d[NUM_BSSN_VARS + 1, i] = Sr
            state_2d[NUM_BSSN_VARS + 2, i] = tau

    # CRITICAL: The TOV solution produces a sharp discontinuity at the stellar surface.
    # We must set all exterior points (including transition zone) to atmosphere values
    # to avoid Gibbs phenomenon with high-order reconstructors like MP5.
    #
    # Strategy: Find the stellar surface (where TOV rho drops below reasonable threshold),
    # then set everything beyond that to atmosphere values.
    if atmosphere is not None:
        from source.matter.hydro.atmosphere import AtmosphereParams
        if isinstance(atmosphere, AtmosphereParams):
            D = state_2d[NUM_BSSN_VARS + 0, :]

            # Find stellar surface: use a threshold that's much higher than floor
            # to capture the full transition region where TOV solution goes to zero
            surface_threshold = 1.0e-6  # Physical threshold for "star interior"

            interior_mask = D >= surface_threshold
            if np.any(interior_mask):
                # Find last truly interior point
                interior_indices = np.where(interior_mask)[0]
                last_interior_idx = interior_indices[-1]

                # Set all points beyond this to atmosphere
                exterior_mask = np.arange(len(D)) > last_interior_idx
                if np.any(exterior_mask):
                    state_2d[NUM_BSSN_VARS + 0, exterior_mask] = atmosphere.rho_floor
                    state_2d[NUM_BSSN_VARS + 1, exterior_mask] = 0.0
                    state_2d[NUM_BSSN_VARS + 2, exterior_mask] = atmosphere.tau_atm

                    R_surf = grid.r[last_interior_idx]
                    n_exterior = np.sum(exterior_mask)
                    print(f'  Initial data: Set {n_exterior} exterior points (r > {R_surf:.6f}) to atmosphere')

    # Fill ghost zones
    grid.fill_boundaries(state_2d)
    return state_2d


def plot_initial_comparison(tov_solution, initial_state_2d, grid, hydro):
    """Plot initial data vs TOV comparison (ρ0, P, v^r, α)."""
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(initial_state_2d, bssn_vars, grid)
    prim = hydro._get_primitives(bssn_vars, grid.r)

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
    plt.savefig('tov_initial_data_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_initial_geometry_comparison(tov_solution, initial_state_2d, grid):
    """Compare Engrenage γ_rr y α con referencias TOV/Schwarzschild.

    Nota: α = sqrt(1-2M/r) solo aplica en el exterior. En el interior se compara con α de TOV.
    """
    # Rebuild physical γ_rr from BSSN: γ_rr = e^{4φ} γ̄_rr
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])

    phi = initial_state_2d[idx_phi, :]
    e4phi = np.exp(4.0 * phi)
    # Build bar_gamma using the flat spherical background
    from source.backgrounds.sphericalbackground import FlatSphericalBackground
    bg = FlatSphericalBackground(grid.r)
    bar_gamma_LL = get_bar_gamma_LL(grid.r, bssn_vars.h_LL, bg)

    gamma_rr_eng = e4phi * bar_gamma_LL[:, i_r, i_r]
    alpha_eng = initial_state_2d[idx_lapse, :]
    alpha_tov_in = tov_solution['alpha']

    r = grid.r
    # Interpolate M(r) from TOV to full grid (set M=0 at r<=0)
    r_tov = tov_solution['r']
    M_tov = tov_solution['M']
    M_interp = interp1d(r_tov, M_tov, kind='linear', bounds_error=False,
                        fill_value=(M_tov[0], M_tov[-1]))
    M_of_r = M_interp(np.maximum(r, r_tov[0]))
    M_of_r[r <= 0.0] = 0.0
    # Interpolate TOV alpha to the full grid for consistent comparisons
    alpha_tov_interp = interp1d(r_tov, alpha_tov_in, kind='linear', bounds_error=False,
                                fill_value=(alpha_tov_in[0], alpha_tov_in[-1]))
    alpha_tov_full = alpha_tov_interp(np.maximum(r, r_tov[0]))
    alpha_tov_full[r <= 0.0] = alpha_tov_in[0]

    # Expectación vacuum (exterior): γ_rr = 1/(1-2M/r), α = sqrt(1-2M/r)
    one_minus = np.maximum(1.0 - 2.0 * M_of_r / np.maximum(r, 1e-12), 1e-12)
    gamma_rr_exp = 1.0 / one_minus
    alpha_schw = np.sqrt(one_minus)

    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
    r_int = r[interior]
    rel_err_gamma = np.abs(gamma_rr_eng[interior] - gamma_rr_exp[interior]) / np.maximum(gamma_rr_exp[interior], 1e-30)
    # Referencia: TOV en interior, Schwarzschild en exterior
    R_star = tov_solution['R']
    mask_ext = r_int >= R_star
    ref_alpha = alpha_tov_full[NUM_GHOSTS:-NUM_GHOSTS].copy()
    ref_alpha[mask_ext] = alpha_schw[NUM_GHOSTS:-NUM_GHOSTS][mask_ext]
    rel_err_alpha = np.abs(alpha_eng[interior] - ref_alpha) / np.maximum(ref_alpha, 1e-30)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(r_int, gamma_rr_eng[interior], label='Engrenage γ_rr')
    axes[0, 0].plot(r_int, gamma_rr_exp[interior], '--', label='TOV/Schwarzschild γ_rr')
    axes[0, 0].set_title('γ_rr comparison'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(r_int, alpha_eng[interior], label='Engrenage α')
    axes[0, 1].plot(r_int, alpha_tov_full[NUM_GHOSTS:-NUM_GHOSTS], ':', label='TOV α (interior)')
    axes[0, 1].plot(r_int[mask_ext], alpha_schw[NUM_GHOSTS:-NUM_GHOSTS][mask_ext], '--', label='Vacuum α (exterior)')
    axes[0, 1].set_title('α comparison'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].semilogy(r_int, np.maximum(rel_err_gamma, 1e-20))
    axes[1, 0].set_title('Rel. error γ_rr'); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(r_int, np.maximum(rel_err_alpha, 1e-20))
    axes[1, 1].set_title('Rel. error α'); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tov_geometry_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_hydrostatic_equilibrium_residual(state_t0, grid, hydro, background):
    """Check hydrostatic equilibrium: dP/dr + (ρ + ρε + P) (dα/dr)/α ≈ 0 at t=0."""
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_t0[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t0, bssn_vars, grid)
    prim = hydro._get_primitives(bssn_vars, grid.r)

    rho = prim['rho0']
    p = prim['p']
    alpha = state_t0[idx_lapse, :]

    # Derivatives using Engrenage derivative matrices (first derivative)
    D1 = grid.derivs.drn_matrix[1]
    dr_p = (p @ D1.T) / grid.dr
    dr_alpha = (alpha @ D1.T) / grid.dr
    eps = hydro.eos.eps_from_rho_p(rho, p)
    residual = dr_p + (rho + rho * eps + p) * dr_alpha / np.maximum(alpha, 1e-30)

    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
    r_int = grid.r[interior]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(r_int, dr_p[interior]); axes[0, 0].set_title('dP/dr'); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(r_int, dr_alpha[interior]); axes[0, 1].set_title('dα/dr'); axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(r_int, ((rho + rho * eps + p) * dr_alpha / np.maximum(alpha, 1e-30))[interior]); axes[1, 0].set_title('(ρ+ρε+P)dα/dr/α'); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(r_int, residual[interior]); axes[1, 1].set_title('Residual'); axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('hydrostatic_equilibrium.png', dpi=150, bbox_inches='tight'); plt.close(fig)

    # Print quick stats
    max_abs = float(np.max(np.abs(residual[interior])))
    mean = float(np.mean(residual[interior]))
    std = float(np.std(residual[interior]))
    print(f"Hydrostatic residual: max |res|={max_abs:.3e}, mean={mean:.3e}, std={std:.3e}")


def plot_cons2prim_consistency(state_t0, grid, hydro, background):
    """Round-trip consistency: prim->cons->prim errors at t=0."""
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_t0[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t0, bssn_vars, grid)
    prim = hydro._get_primitives(bssn_vars, grid.r)

    rho0 = prim['rho0']; vr = prim['vr']; p = prim['p']
    # Metric γ_rr from BSSN
    phi = state_t0[idx_phi, :]
    e4phi = np.exp(4.0 * phi)
    from source.backgrounds.sphericalbackground import FlatSphericalBackground
    bg = FlatSphericalBackground(grid.r)
    bar_gamma_LL = get_bar_gamma_LL(grid.r, bssn_vars.h_LL, bg)
    gamma_rr = e4phi * bar_gamma_LL[:, i_r, i_r]

    D_state = state_t0[NUM_BSSN_VARS + 0, :]
    Sr_state = state_t0[NUM_BSSN_VARS + 1, :]
    tau_state = state_t0[NUM_BSSN_VARS + 2, :]

    D_rec = np.zeros_like(D_state); Sr_rec = np.zeros_like(Sr_state); tau_rec = np.zeros_like(tau_state)
    for i in range(grid.N):
        D_rec[i], Sr_rec[i], tau_rec[i] = prim_to_cons(rho0[i], vr[i], p[i], gamma_rr[i], hydro.eos)

    eps = 1e-30
    dD = D_rec - D_state
    dSr = Sr_rec - Sr_state
    dtau = tau_rec - tau_state
    rel_err_D = np.abs(dD) / np.maximum(np.abs(D_state), eps)
    rel_err_Sr = np.abs(dSr) / np.maximum(np.abs(Sr_state), eps)
    rel_err_tau = np.abs(dtau) / np.maximum(np.abs(tau_state), eps)

    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
    r_int = grid.r[interior]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].semilogy(r_int, np.maximum(rel_err_D[interior], 1e-20)); axes[0].set_title('|ΔD|/D'); axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(r_int, np.maximum(rel_err_Sr[interior], 1e-20)); axes[1].set_title('|ΔSr|/Sr'); axes[1].grid(True, alpha=0.3)
    axes[2].semilogy(r_int, np.maximum(rel_err_tau[interior], 1e-20)); axes[2].set_title('|Δτ|/τ'); axes[2].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('cons2prim_consistency.png', dpi=150, bbox_inches='tight'); plt.close(fig)

    # Detail: interior-only relative error for tau and absolute error across domain
    mask_interior = prim['rho0'][NUM_GHOSTS:-NUM_GHOSTS] > 10.0 * hydro.atmosphere.rho_floor
    r_int = grid.r[interior]
    rel_tau_int = rel_err_tau[interior][mask_interior]
    r_tau_int = r_int[mask_interior]
    abs_tau = np.abs(dtau)[interior]

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))
    if r_tau_int.size > 0:
        ax2[0].semilogy(r_tau_int, np.maximum(rel_tau_int, 1e-20))
    ax2[0].set_title('|Δτ|/τ (interior only)'); ax2[0].set_xlabel('r'); ax2[0].grid(True, alpha=0.3)
    ax2[1].semilogy(r_int, np.maximum(abs_tau, 1e-30))
    ax2[1].set_title('|Δτ| (absolute)'); ax2[1].set_xlabel('r'); ax2[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('cons2prim_consistency_detail.png', dpi=150, bbox_inches='tight'); plt.close(fig2)


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
    p.add_argument('--save_npz', type=str, default='')
    args = p.parse_args()

    print('='*70)
    print('TOV Initial Data – Engrenage')
    print('='*70)
    print(f"Grid: N={args.num_points}, r_max={args.r_max}")
    print(f"EOS (TOV): K={args.K}, Gamma={args.Gamma}")
    print(f"rho_central={args.rho_central}, rho_atm={args.atmosphere_rho}\n")

    # Grid & background
    spacing = LinearSpacing(args.num_points, args.r_max)
    # Minimal state vector for constructing Grid (needs a matter provider)
    dummy_hydro = PerfectFluid(eos=IdealGasEOS(gamma=args.Gamma), spacetime_mode="dynamic",
                               atmosphere=args.atmosphere_rho)  # Float is auto-converted
    state_vector = StateVector(dummy_hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)

    # Solve TOV on positive radii of evolution grid
    solver = TOVSolver(K=args.K, Gamma=args.Gamma, use_isotropic=False)
    from source.core.spacing import NUM_GHOSTS
    r_positive = grid.r[NUM_GHOSTS:]
    tov_solution = solver.solve(args.rho_central, r_grid=r_positive, r_max=args.r_max)
    print(f"TOV: M={tov_solution['M_star']:.6f}, R={tov_solution['R']:.3f}, C={tov_solution['C']:.4f}\n")

    # Plots: TOV diagnostics
    plot_tov_diagnostics(tov_solution, args.r_max)

    # Build initial data
    hydro = PerfectFluid(eos=IdealGasEOS(gamma=args.Gamma), spacetime_mode="dynamic",
                         atmosphere=args.atmosphere_rho)  # Float is auto-converted
    hydro.background = background
    initial_state_2d = create_initial_data(
        tov_solution, grid, background, hydro.eos,
        atmosphere=hydro.atmosphere,  # Use centralized atmosphere
        polytrope_K=args.K, polytrope_Gamma=args.Gamma
    )

    # Diagnostics/plots
    plot_initial_comparison(tov_solution, initial_state_2d, grid, hydro)
    plot_initial_geometry_comparison(tov_solution, initial_state_2d, grid)
    plot_hydrostatic_equilibrium_residual(initial_state_2d, grid, hydro, background)
    plot_cons2prim_consistency(initial_state_2d, grid, hydro, background)

    # Optional save
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
        print(f"Saved initial data NPZ to {args.save_npz}")

    print('\nSaved figures:')
    print('  - tov_solution.png')
    print('  - tov_initial_data_comparison.png')
    print('  - tov_geometry_comparison.png')
    print('  - hydrostatic_equilibrium.png')
    print('  - cons2prim_consistency.png')
