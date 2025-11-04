"""
TOV Initial Data in isotropic coordinates (NRPy-style) + rigorous ADM→BSSN conversion.

This module expects the output from TOVSolverISO (isotropic radii + profiles)
and builds consistent ADM variables in the isotropic radial coordinate r.
Then it enforces det(γ̄)=det(ĝ) and converts to BSSN state variables.
Hydro primitives are taken from the interior solution and set to atmosphere outside.
"""

import numpy as np
from typing import Dict, Any

from source.core.spacing import NUM_GHOSTS
from source.bssn.bssnstatevariables import (
    NUM_BSSN_VARS,
    idx_phi, idx_hrr, idx_htt, idx_hpp,
    idx_K, idx_arr, idx_att, idx_app, idx_lapse,
)
from source.backgrounds.sphericalbackground import i_r, i_t, i_p

# Reuse robust ADM→BSSN converter from the existing module
from examples.tov_initial_data_interpolated import convert_adm_to_bssn


def _interp_surface_protected(r_query, r_int, f_int, r_ext, f_ext):
    """
    Interpolate f(r) piecewise: use interior data for r <= r_surface,
    and exact exterior for r >= r_surface.
    Input arrays must be sorted by radius.
    """
    r_query = np.asarray(r_query)
    fout = np.zeros_like(r_query, dtype=float)
    r_surf = r_int[-1]
    # interior
    mask_in = r_query <= r_surf
    if np.any(mask_in):
        fout[mask_in] = np.interp(r_query[mask_in], r_int, f_int)
    # exterior
    mask_out = ~mask_in
    if np.any(mask_out):
        fout[mask_out] = np.interp(r_query[mask_out], r_ext, f_ext)
    return fout


def create_initial_data_interpolated_iso(tov_iso_solution: Dict[str, Any], grid, background, eos,
                                         atmosphere=None,
                                         polytrope_K=None, polytrope_Gamma=None,
                                         interp_order: int = 11) -> np.ndarray:
    """
    Build BSSN + hydro state in isotropic coordinates from TOVSolverISO output.

    Arguments:
        tov_iso_solution: dict from TOVSolverISO.solve()
        grid: Engrenage Grid (r is isotropic radius)
        background: FlatSphericalBackground
        eos: EOS instance (IdealGas or Polytropic)
        atmosphere: AtmosphereParams (optional)
        polytrope_K, polytrope_Gamma: If given, used for epsilon derivation

    Returns:
        state_2d: (NUM_VARS, N) array ready for evolution
    """
    if atmosphere is None:
        from source.matter.hydro.atmosphere import AtmosphereParams
        atmosphere = AtmosphereParams()

    r = grid.r
    N = grid.N

    # Unpack isotropic profiles (sorted by construction)
    r_int = np.asarray(tov_iso_solution['r_iso_int'])
    a_int = np.asarray(tov_iso_solution['alpha_int'])
    e4_int = np.asarray(tov_iso_solution['exp4phi_int'])
    rho_b_int = np.asarray(tov_iso_solution['rho_baryon_int'])
    P_int = np.asarray(tov_iso_solution['P_int'])

    r_ext = np.asarray(tov_iso_solution['r_iso_ext'])
    a_ext = np.asarray(tov_iso_solution['alpha_ext'])
    e4_ext = np.asarray(tov_iso_solution['exp4phi_ext'])

    # Interpolate lapse and conformal factor in isotropic radius
    alpha = _interp_surface_protected(r, r_int, a_int, r_ext, a_ext)
    exp4phi = _interp_surface_protected(r, r_int, e4_int, r_ext, e4_ext)

    # Build physical ADM spatial metric in isotropic coords: γ_rr=ψ⁴, γ_θθ=ψ⁴ r²
    gamma_LL = np.zeros((N, 3, 3))
    gamma_LL[:, i_r, i_r] = exp4phi
    gamma_LL[:, i_t, i_t] = exp4phi * np.abs(r) ** 2
    gamma_LL[:, i_p, i_p] = gamma_LL[:, i_t, i_t]

    # Extrinsic curvature K_ij=0
    K_LL = np.zeros_like(gamma_LL)

    adm_vars = {
        'alpha': alpha,
        'beta_U': np.zeros((N, 3)),
        'gamma_LL': gamma_LL,
        'K_LL': K_LL,
    }

    # Convert ADM→BSSN with constraint enforcement
    state_2d = convert_adm_to_bssn(adm_vars, grid, background)

    # Hydro primitives: interpolate interior ρ0 & P; exterior atmosphere
    rho0 = np.interp(np.minimum(np.abs(r), r_int[-1]), r_int, rho_b_int)
    P = np.interp(np.minimum(np.abs(r), r_int[-1]), r_int, P_int)

    # Atmosphere outside
    mask_ext = np.abs(r) > r_int[-1]
    rho0[mask_ext] = atmosphere.rho_floor
    P[mask_ext] = max(atmosphere.p_floor, P[mask_ext].min(initial=atmosphere.p_floor))

    # v=0 (static TOV)
    vr = np.zeros(N)

    # eps per EOS
    if (polytrope_K is not None) and (polytrope_Gamma is not None):
        eps = P / np.maximum((polytrope_Gamma - 1.0) * rho0, 1e-30)
    else:
        try:
            eps = eos.eps_from_rho_p(rho0, P)
        except Exception:
            # Fallback small internal energy
            eps = P / np.maximum(1.0 * rho0, 1e-30)

    # Conservative variables for static fluid (W=1)
    D = rho0.copy()
    Sr = np.zeros_like(rho0)
    tau = rho0 * eps

    state_2d[NUM_BSSN_VARS + 0, :] = D
    state_2d[NUM_BSSN_VARS + 1, :] = Sr
    state_2d[NUM_BSSN_VARS + 2, :] = tau

    grid.fill_boundaries(state_2d)
    return state_2d
