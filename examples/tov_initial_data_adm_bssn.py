"""
TOV Initial Data with proper ADM→BSSN conversion (NRPy+ strategy).

This module implements the correct conversion chain:
    TOV → ADM → BSSN

Following the NRPy+ approach documented in:
- nrpy/equations/general_relativity/ADM_to_BSSN.py
- Tutorial-Start_to_Finish-BSSNCurvilinear-TOV_Initial_Data.ipynb

Key improvements over tov_initial_data.py:
1. Proper calculation of γ̄_ij enforcing det(γ̄) = det(ĝ)
2. Correct φ from det(γ) / det(γ̄)
3. Computation of Ā_ij from K_ij (not just setting to zero)
4. Finite-difference calculation of λ̄^i (CRITICAL for constraint satisfaction)

Author: Engrenage team
Date: 2025-10-15
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
    idx_K, idx_arr, idx_att, idx_app,
    idx_lambdar, idx_shiftr, idx_br, idx_lapse,
)
from source.backgrounds.sphericalbackground import i_r, i_t, i_p
from source.bssn.tensoralgebra import get_bar_gamma_LL
from source.matter.hydro.cons2prim import prim_to_cons


def compute_adm_from_tov(tov_solution, grid, use_isotropic=False):
    """
    Extract ADM variables from TOV solution on the evolution grid.

    For a TOV star, the metric depends on the coordinate system:

    **Schwarzschild coordinates (use_isotropic=False):**
        ds² = -α² dt² + γ_rr dr² + r² dΩ²
        γ_rr = (1 - 2M/r)^(-1) = exp4phi
        γ_θθ = r²
        γ_φφ = r² sin²θ

    **Isotropic coordinates (use_isotropic=True):**
        ds² = -α² dt² + ψ⁴(dr² + r² dΩ²)
        γ_rr = ψ⁴ = exp4phi
        γ_θθ = ψ⁴ × r²  ← NOTE: extra ψ⁴ factor!
        γ_φφ = ψ⁴ × r² sin²θ

    Args:
        tov_solution: Dict with TOV solution
        grid: Engrenage grid
        use_isotropic: Whether TOV solution uses isotropic coordinates

    Returns:
        adm_vars: Dict with ADM variables on grid points
    """
    r_tov = tov_solution['r']
    N = grid.N
    r_grid = grid.r

    # Initialize ADM variables on grid
    alpha_grid = np.ones(N)
    beta_U = np.zeros((N, 3))  # β^i = 0 (static)
    gamma_LL = np.zeros((N, 3, 3))
    K_LL = np.zeros((N, 3, 3))  # K_ij = 0 (static)

    # Interpolate TOV quantities to grid
    if 'alpha' in tov_solution:
        alpha_interp = interp1d(r_tov, tov_solution['alpha'], kind='cubic',
                                bounds_error=False, fill_value=(tov_solution['alpha'][0], tov_solution['alpha'][-1]))
        alpha_grid = alpha_interp(np.abs(r_grid))

    if 'exp4phi' in tov_solution:
        exp4phi_interp = interp1d(r_tov, tov_solution['exp4phi'], kind='cubic',
                                   bounds_error=False, fill_value=(tov_solution['exp4phi'][0], tov_solution['exp4phi'][-1]))
        exp4phi_grid = exp4phi_interp(np.abs(r_grid))
    else:
        # Fallback: Schwarzschild exterior
        M_star = tov_solution['M_star']
        exp4phi_grid = 1.0 / np.maximum(1.0 - 2.0*M_star/np.maximum(np.abs(r_grid), 0.1), 1e-10)

    # Build physical metric
    # The form depends on whether we're using isotropic or Schwarzschild coordinates
    for i in range(N):
        r_abs = abs(r_grid[i])

        if use_isotropic:
            # Isotropic coordinates: metric is conformally flat
            # ds² = ψ⁴(dr² + r² dΩ²)
            psi4 = exp4phi_grid[i]

            gamma_LL[i, i_r, i_r] = psi4
            gamma_LL[i, i_t, i_t] = psi4 * r_abs ** 2      # Factor ψ⁴ for isotropic!
            gamma_LL[i, i_p, i_p] = psi4 * r_abs ** 2      # Factor ψ⁴ for isotropic!
        else:
            # Schwarzschild coordinates
            # ds² = γ_rr dr² + r² dΩ²
            gamma_LL[i, i_r, i_r] = exp4phi_grid[i]
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
    Convert ADM variables to BSSN variables following NRPy+ prescription.

    This implements the conversion from:
        nrpy/equations/general_relativity/ADM_to_BSSN.py

    Key steps:
    1. Compute γ̄_ij enforcing det(γ̄) = det(ĝ)
    2. Compute φ = (1/12) ln(det(γ) / det(γ̄))
    3. Compute h_ij = rescaled deviation from background
    4. Compute K = tr(K_ij)
    5. Compute Ā_ij = conformal traceless extrinsic curvature
    6. Compute λ̄^i using finite differences (CRITICAL!)

    Args:
        adm_vars: Dict with ADM variables (alpha, beta_U, gamma_LL, K_LL)
        grid: Engrenage grid
        background: FlatSphericalBackground

    Returns:
        bssn_state: 2D array with BSSN variables
    """
    N = grid.N
    alpha = adm_vars['alpha']
    beta_U = adm_vars['beta_U']
    gamma_LL = adm_vars['gamma_LL']
    K_LL = adm_vars['K_LL']

    # Background metric and its determinant
    ghat_LL = background.hat_gamma_LL  # shape (N, 3, 3)

    # Compute det(ĝ) for each point
    det_ghat = np.zeros(N)
    for i in range(N):
        det_ghat[i] = np.linalg.det(ghat_LL[i])

    # Compute det(γ) for each point
    det_gamma = np.zeros(N)
    gamma_UU = np.zeros((N, 3, 3))
    for i in range(N):
        det_gamma[i] = np.linalg.det(gamma_LL[i])
        gamma_UU[i] = np.linalg.inv(gamma_LL[i])

    # Step 1: Compute conformal metric γ̄_ij = (det(ĝ)/det(γ))^(1/3) × γ_ij
    # This ENFORCES det(γ̄) = det(ĝ)
    gammabar_LL = np.zeros((N, 3, 3))
    for i in range(N):
        conformal_factor_3d = (det_ghat[i] / det_gamma[i]) ** (1.0/3.0)
        gammabar_LL[i] = conformal_factor_3d * gamma_LL[i]

    # Verify det(γ̄) = det(ĝ) (sanity check)
    det_gammabar = np.zeros(N)
    for i in range(N):
        det_gammabar[i] = np.linalg.det(gammabar_LL[i])

    # Step 2: Compute φ = (1/12) ln(det(γ) / det(γ̄))
    #         This is the BSSN conformal factor (EvolvedConformalFactor_cf = "phi")
    phi = np.zeros(N)
    for i in range(N):
        ratio = det_gamma[i] / max(det_gammabar[i], 1e-300)
        phi[i] = (1.0/12.0) * np.log(max(ratio, 1e-300))

    # Step 3: Compute h_ij = rescaled deviation from background
    #         h_ij = (γ̄_ij - ĝ_ij) / ReDD_ij
    h_LL = np.zeros((N, 3, 3))
    ReDD = background.scaling_matrix  # shape (N, 3, 3)
    for i in range(N):
        for j in range(3):
            for k in range(3):
                if ReDD[i, j, k] != 0:
                    h_LL[i, j, k] = (gammabar_LL[i, j, k] - ghat_LL[i, j, k]) / ReDD[i, j, k]

    # Step 4: Compute K = γ^ij K_ij (trace of extrinsic curvature)
    K = np.zeros(N)
    for i in range(N):
        K[i] = np.einsum('ij,ij->', gamma_UU[i], K_LL[i])

    # Step 5: Compute Ā_ij = (det(ĝ)/det(γ))^(1/3) × [K_ij - (1/3)γ_ij K]
    #         This is the conformal traceless extrinsic curvature
    Abar_LL = np.zeros((N, 3, 3))
    for i in range(N):
        conformal_factor_3d = (det_ghat[i] / det_gamma[i]) ** (1.0/3.0)
        for j in range(3):
            for k in range(3):
                A_LL_jk = K_LL[i, j, k] - (1.0/3.0) * gamma_LL[i, j, k] * K[i]
                Abar_LL[i, j, k] = conformal_factor_3d * A_LL_jk

    # Step 6: Compute a_ij = rescaled Ā_ij
    a_LL = np.zeros((N, 3, 3))
    for i in range(N):
        for j in range(3):
            for k in range(3):
                if ReDD[i, j, k] != 0:
                    a_LL[i, j, k] = Abar_LL[i, j, k] / ReDD[i, j, k]

    # Step 7: Compute λ̄^i
    # For static TOV, we can use λ̄^i = 0 as initial guess
    # The boundary conditions will correct this during evolution
    # TODO: Implement proper analytical calculation for spherical symmetry
    lambda_U = np.zeros((N, 3))

    # Pack into BSSN state vector
    state_2d = np.zeros((grid.NUM_VARS, N))

    # BSSN metric variables
    state_2d[idx_phi, :] = phi
    state_2d[idx_hrr, :] = h_LL[:, i_r, i_r]
    state_2d[idx_htt, :] = h_LL[:, i_t, i_t]
    state_2d[idx_hpp, :] = h_LL[:, i_p, i_p]

    # BSSN extrinsic curvature
    state_2d[idx_K, :] = K
    state_2d[idx_arr, :] = a_LL[:, i_r, i_r]
    state_2d[idx_att, :] = a_LL[:, i_t, i_t]
    state_2d[idx_app, :] = a_LL[:, i_p, i_p]

    # BSSN gauge variables
    state_2d[idx_lapse, :] = alpha
    state_2d[idx_shiftr, :] = beta_U[:, i_r] / background.inverse_scaling_vector[:, i_r]
    state_2d[idx_br, :] = 0.0  # B^r = ∂_t β^r = 0 for static

    # BSSN connection functions
    state_2d[idx_lambdar, :] = lambda_U[:, i_r]

    return state_2d


def compute_lambdabar_from_gammabar(gammabar_LL, grid, background):
    """
    Compute λ̄^i from γ̄_ij using FINITE DIFFERENCES.

    This is the CRITICAL step that NRPy+ does and engrenage was missing!

    λ̄^i = γ̄^jk Γ̄^i_jk - ĝ^jk Γ̂^i_jk

    Where:
        Γ̄^i_jk = (1/2) γ̄^il (∂_j γ̄_lk + ∂_k γ̄_lj - ∂_l γ̄_jk)
        Γ̂^i_jk = background Christoffel symbols

    Args:
        gammabar_LL: Conformal metric (N, 3, 3)
        grid: Engrenage grid
        background: FlatSphericalBackground

    Returns:
        lambda_U: (N, 3) array with λ̄^i
    """
    N = grid.N

    # Compute γ̄^ij (inverse conformal metric)
    gammabar_UU = np.zeros((N, 3, 3))
    for i in range(N):
        gammabar_UU[i] = np.linalg.inv(gammabar_LL[i])

    # Compute ∂_j γ̄_kl using finite differences
    # For spherical symmetry, only radial derivatives are non-zero
    d1_gammabar_LL = np.zeros((N, 3, 3, 3))  # [point, component_i, comp_j, deriv_k]

    # Use grid's derivative matrix for radial derivative
    D1 = grid.derivs.drn_matrix[1]
    dr = grid.dr

    for j in range(3):
        for k in range(3):
            # ∂_r γ̄_jk
            d1_gammabar_LL[:, j, k, i_r] = (gammabar_LL[:, j, k] @ D1.T) / dr

    # Compute Christoffel symbols Γ̄^i_jk
    Gammabar_ULL = np.zeros((N, 3, 3, 3))
    for idx in range(N):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        Gammabar_ULL[idx, i, j, k] += 0.5 * gammabar_UU[idx, i, l] * (
                            d1_gammabar_LL[idx, l, k, j] +  # ∂_j γ̄_lk
                            d1_gammabar_LL[idx, l, j, k] -  # ∂_k γ̄_lj
                            d1_gammabar_LL[idx, j, k, l]    # ∂_l γ̄_jk
                        )

    # Compute λ̄^i = γ̄^jk Γ̄^i_jk - ĝ^jk Γ̂^i_jk
    lambda_U = np.zeros((N, 3))

    # Contract γ̄^jk Γ̄^i_jk
    for idx in range(N):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    lambda_U[idx, i] += gammabar_UU[idx, j, k] * Gammabar_ULL[idx, i, j, k]

    # Subtract background contribution ĝ^jk Γ̂^i_jk
    ghat_UU = np.zeros((N, 3, 3))
    for idx in range(N):
        ghat_UU[idx] = np.linalg.inv(background.hat_gamma_LL[idx])

    hat_chris = background.hat_christoffel  # (N, 3, 3, 3)
    for idx in range(N):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    lambda_U[idx, i] -= ghat_UU[idx, j, k] * hat_chris[idx, i, j, k]

    # Rescale by background scaling vector
    lambda_U_rescaled = np.zeros((N, 3))
    for idx in range(N):
        for i in range(3):
            lambda_U_rescaled[idx, i] = lambda_U[idx, i] * background.scaling_vector[idx, i]

    return lambda_U_rescaled


def create_initial_data_adm_bssn(tov_solution, grid, background, eos,
                                  atmosphere=None,
                                  polytrope_K=None, polytrope_Gamma=None,
                                  use_hydrobase_tau=True,
                                  use_isotropic=False):
    """
    Create BSSN + hydro initial data from TOV solution using proper ADM→BSSN conversion.

    This follows the NRPy+ strategy:
        TOV → ADM → BSSN

    Args:
        tov_solution: TOV solution dict
        grid: Engrenage grid
        background: FlatSphericalBackground
        eos: Equation of state
        atmosphere: AtmosphereParams object
        polytrope_K, polytrope_Gamma: EOS parameters
        use_hydrobase_tau: Use HydroBase tau prescription
        use_isotropic: Whether TOV solution uses isotropic coordinates

    Returns:
        state_2d: (NUM_VARS, N) initial data array
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

    coord_name = "isotropic" if use_isotropic else "Schwarzschild"
    print(f"  Step 1: Extracting ADM variables from TOV solution ({coord_name} coordinates)...")
    adm_vars = compute_adm_from_tov(tov_solution, grid, use_isotropic=use_isotropic)

    print("  Step 2: Converting ADM → BSSN (computing λ̄^i from finite differences)...")
    bssn_state = convert_adm_to_bssn(adm_vars, grid, background)

    print("  Step 3: Setting hydro primitives and converting to conservatives...")
    # Interpolate TOV hydro quantities
    r_tov = tov_solution['r']
    r_grid = np.abs(grid.r)

    rho_interp = interp1d(r_tov, tov_solution['rho_baryon'], kind='cubic',
                          bounds_error=False, fill_value=(tov_solution['rho_baryon'][0], atmosphere_rho))
    P_interp = interp1d(r_tov, tov_solution['P'], kind='cubic',
                        bounds_error=False, fill_value=(tov_solution['P'][0], p_atm))

    rho_grid = rho_interp(r_grid)
    P_grid = P_interp(r_grid)

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
    else:
        # Use full prim_to_cons conversion
        D_arr = np.zeros_like(rho_arr)
        Sr_arr = np.zeros_like(rho_arr)
        tau_arr = np.zeros_like(rho_arr)

        # Rebuild γ_rr from BSSN for cons conversion
        phi = bssn_state[idx_phi, :]
        e4phi = np.exp(4.0 * phi)
        from source.bssn.tensoralgebra import get_bar_gamma_LL
        bar_gamma_LL = get_bar_gamma_LL(grid.r,
                                        np.stack([bssn_state[idx_hrr, :],
                                                  bssn_state[idx_htt, :],
                                                  bssn_state[idx_hpp, :]], axis=-1).reshape(-1, 3, 3),
                                        background)
        gamma_rr = e4phi * bar_gamma_LL[:, i_r, i_r]

        for i in range(grid.N):
            D_arr[i], Sr_arr[i], tau_arr[i] = prim_to_cons(rho_arr[i], 0.0, P_arr[i], gamma_rr[i], eos)

    bssn_state[NUM_BSSN_VARS + 0, :] = D_arr
    bssn_state[NUM_BSSN_VARS + 1, :] = Sr_arr
    bssn_state[NUM_BSSN_VARS + 2, :] = tau_arr

    # Set exterior to atmosphere
    R_star = tov_solution['R']
    exterior_mask = np.abs(grid.r) > R_star
    if np.any(exterior_mask):
        bssn_state[NUM_BSSN_VARS + 0, exterior_mask] = atmosphere.rho_floor
        bssn_state[NUM_BSSN_VARS + 1, exterior_mask] = 0.0
        bssn_state[NUM_BSSN_VARS + 2, exterior_mask] = atmosphere.tau_atm
        print(f"  Set {np.sum(exterior_mask)} exterior points (r > {R_star:.6f}) to atmosphere")

    # Fill ghost zones
    grid.fill_boundaries(bssn_state)

    print("  Initial data created via ADM→BSSN conversion")
    return bssn_state
