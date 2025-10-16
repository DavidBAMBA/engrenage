"""
TOV Initial Data with High-Order Interpolation (NRPy+ strategy).

Key differences from tov_initial_data.py:
1. TOV is solved on a fine, independent grid (not evolution grid)
2. High-order Lagrange interpolation to evolution grid
3. Interpolation stencil NEVER crosses stellar surface (avoids Gibbs phenomenon)

This follows the NRPy+ approach documented in:
  nrpytutorial/TOVID_Ccodes/TOV_interpolate_1D.c
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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
from source.bssn.tensoralgebra import get_bar_gamma_LL
from source.matter.hydro.cons2prim import prim_to_cons


def lagrange_interpolate_1d(x_target, x_data, y_data, order=11):
    """
    High-order Lagrange polynomial interpolation.

    Args:
        x_target: Point where to interpolate
        x_data: Array of data points (must be sorted)
        y_data: Array of function values at x_data
        order: Interpolation order (default 11, like NRPy+)

    Returns:
        Interpolated value at x_target
    """
    n = len(x_data)
    if n < order:
        order = n

    # Find nearest index
    idx = np.searchsorted(x_data, x_target)
    idx = max(0, min(idx, n-1))

    # Center stencil around idx
    half_stencil = order // 2
    idx_min = max(0, idx - half_stencil)
    idx_max = min(n, idx_min + order)
    idx_min = max(0, idx_max - order)

    # Extract stencil
    x_stencil = x_data[idx_min:idx_max]
    y_stencil = y_data[idx_min:idx_max]

    # Lagrange interpolation
    result = 0.0
    for i in range(len(x_stencil)):
        # Compute Lagrange basis polynomial L_i(x_target)
        L_i = 1.0
        for j in range(len(x_stencil)):
            if i != j:
                L_i *= (x_target - x_stencil[j]) / (x_stencil[i] - x_stencil[j])
        result += y_stencil[i] * L_i

    return result


def interpolate_tov_avoiding_surface(r_target, tov_solution, field_name,
                                      interp_order=11, atmosphere_value=None):
    """
    Interpolate TOV solution field, ensuring stencil doesn't cross surface.

    This is the KEY innovation from NRPy+ to avoid Gibbs phenomenon.

    Args:
        r_target: Radius where to interpolate
        tov_solution: Dict with TOV solution
        field_name: Name of field to interpolate ('rho_baryon', 'P', etc.)
        interp_order: Interpolation order
        atmosphere_value: Value to return if r_target is in atmosphere

    Returns:
        Interpolated value
    """
    r_tov = tov_solution['r']
    field_tov = tov_solution[field_name]
    R_star = tov_solution['R']

    # Handle negative radii (symmetry)
    r_use = abs(r_target)

    # If outside star, return atmosphere value
    if r_use > R_star:
        if atmosphere_value is not None:
            return atmosphere_value
        # Otherwise interpolate in exterior
        r_use = max(r_use, r_tov[0])  # Clamp to data range
        r_use = min(r_use, r_tov[-1])

    # Find stellar surface index in TOV data
    surf_idx = np.argmin(np.abs(r_tov - R_star))

    # Find target index
    idx = np.searchsorted(r_tov, r_use)
    idx = max(0, min(idx, len(r_tov)-1))

    # Determine stencil bounds, ensuring it doesn't cross surface
    half_stencil = interp_order // 2
    idx_min = max(0, idx - half_stencil)
    idx_max = min(len(r_tov), idx_min + interp_order)

    # CRITICAL: Don't allow stencil to cross surface
    if r_use < R_star:
        # Inside star: stencil must stay inside
        idx_max = min(idx_max, surf_idx + 1)
        idx_min = max(0, idx_max - interp_order)
    else:
        # Outside star: stencil must stay outside
        idx_min = max(idx_min, surf_idx)
        idx_max = min(len(r_tov), idx_min + interp_order)

    # Extract stencil
    r_stencil = r_tov[idx_min:idx_max]
    field_stencil = field_tov[idx_min:idx_max]

    # Lagrange interpolation
    result = 0.0
    for i in range(len(r_stencil)):
        L_i = 1.0
        for j in range(len(r_stencil)):
            if i != j:
                L_i *= (r_use - r_stencil[j]) / (r_stencil[i] - r_stencil[j])
        result += field_stencil[i] * L_i

    return result


def create_initial_data_interpolated(tov_solution, grid, background, eos,
                                      atmosphere=None,
                                      polytrope_K=None, polytrope_Gamma=None,
                                      use_hydrobase_tau=True,
                                      interp_order=11,
                                      exterior_buffer_cells=0):
    """
    Create BSSN + hydro initial data using high-order interpolation (NRPy+ strategy).

    Args:
        tov_solution: TOV solution on fine grid (independent of evolution grid)
        grid: Evolution grid
        background: Background metric
        eos: Equation of state
        atmosphere: AtmosphereParams object
        polytrope_K, polytrope_Gamma: EOS parameters
        use_hydrobase_tau: Use HydroBase tau prescription
        interp_order: Lagrange interpolation order (default 11)

    Returns:
        2D state array with BSSN + hydro initial data
    """
    from source.matter.hydro.atmosphere import AtmosphereParams

    # Handle atmosphere
    if atmosphere is None:
        atmosphere = AtmosphereParams()
    elif not isinstance(atmosphere, AtmosphereParams):
        raise TypeError("atmosphere must be AtmosphereParams")

    atmosphere_rho = atmosphere.rho_floor
    p_atm = atmosphere.p_floor

    # Atmosphere pressure consistent with EOS
    if polytrope_K is not None and polytrope_Gamma is not None:
        p_atm = max(p_atm, polytrope_K * (atmosphere_rho ** polytrope_Gamma))

    R_star = tov_solution['R']

    print(f"  Interpolating TOV to evolution grid (order {interp_order})...")
    print(f"  Stellar radius: R = {R_star:.6f}")

    # Initialize arrays on evolution grid
    state_2d = np.zeros((grid.NUM_VARS, grid.N))

    # Interpolate for each grid point
    for i, r in enumerate(grid.r):
        r_abs = abs(r)

        # Determine if inside or outside star
        is_interior = r_abs < R_star

        if is_interior:
            # Interpolate TOV quantities
            rho = interpolate_tov_avoiding_surface(r_abs, tov_solution, 'rho_baryon',
                                                   interp_order, atmosphere_rho)
            P = interpolate_tov_avoiding_surface(r_abs, tov_solution, 'P',
                                                interp_order, p_atm)
            exp4phi = interpolate_tov_avoiding_surface(r_abs, tov_solution, 'exp4phi',
                                                       interp_order, 1.0)
            alpha = interpolate_tov_avoiding_surface(r_abs, tov_solution, 'alpha',
                                                     interp_order, 1.0)

            # Enforce floors
            rho = max(rho, atmosphere_rho)
            P = max(P, p_atm)
        else:
            # Exterior: use Schwarzschild values
            M_star = tov_solution['M_star']
            one_minus = max(1.0 - 2.0 * M_star / max(r_abs, 0.1), 1e-10)

            rho = atmosphere_rho
            P = p_atm
            exp4phi = 1.0 / one_minus
            alpha = np.sqrt(one_minus)

        # Build BSSN variables
        gamma_phys_rr = exp4phi
        gamma_phys_thth = r_abs ** 2
        gamma_phys_phph = r_abs ** 2

        phi = np.log(max(exp4phi, 1e-300)) / 12.0
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
        state_2d[idx_lapse, i] = alpha

        # Hydro: prim -> cons
        if use_hydrobase_tau and polytrope_Gamma is not None:
            eps = P / max((polytrope_Gamma - 1.0) * rho, 1e-30)
            D = rho
            Sr = 0.0
            tau = rho * eps
        else:
            D, Sr, tau = prim_to_cons(rho, 0.0, P, gamma_phys_rr, eos)

        state_2d[NUM_BSSN_VARS + 0, i] = D
        state_2d[NUM_BSSN_VARS + 1, i] = Sr
        state_2d[NUM_BSSN_VARS + 2, i] = tau

    # Fill ghost zones
    grid.fill_boundaries(state_2d)

    print(f"  Initial data created via interpolation")

    # Optional: apply an atmosphere buffer just outside the stellar surface.
    # This helps reduce discrete hydrostatic imbalance at the sharp surface interface.
    if exterior_buffer_cells and exterior_buffer_cells > 0:
        R = float(R_star)
        r = grid.r
        # Apply buffer only on +r side (physical radial direction)
        pos_idx = np.where(r >= 0.0)[0]
        if pos_idx.size > 0:
            # Find last interior index on +r side
            interior_pos = pos_idx[np.where(np.abs(r[pos_idx]) < R)[0]]
            if interior_pos.size > 0:
                last_int = int(interior_pos[-1])
                start = last_int + 1
                end = min(grid.N, start + int(exterior_buffer_cells))
                if end > start:
                    mask = np.zeros(grid.N, dtype=bool)
                    mask[start:end] = True

                    # Set atmosphere conservative vars consistently
                    state_2d[NUM_BSSN_VARS + 0, mask] = atmosphere_rho
                    state_2d[NUM_BSSN_VARS + 1, mask] = 0.0
                    # Prefer AtmosphereParams tau if available
                    try:
                        from source.matter.hydro.atmosphere import AtmosphereParams
                        if isinstance(atmosphere, AtmosphereParams):
                            state_2d[NUM_BSSN_VARS + 2, mask] = atmosphere.tau_atm
                        else:
                            # Fallback: hydrobase-style tau with polytrope if available
                            if use_hydrobase_tau and polytrope_Gamma is not None and polytrope_K is not None:
                                p_buf = max(p_atm, 0.0)
                                eps_buf = p_buf / max((polytrope_Gamma - 1.0) * atmosphere_rho, 1e-30)
                                state_2d[NUM_BSSN_VARS + 2, mask] = atmosphere_rho * eps_buf
                            else:
                                state_2d[NUM_BSSN_VARS + 2, mask] = p_atm
                    except Exception:
                        state_2d[NUM_BSSN_VARS + 2, mask] = p_atm

                    print(f"  Applied atmosphere buffer: {end - start} cells beyond R on +r side")

    return state_2d
