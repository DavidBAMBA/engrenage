"""
JAX-native Valencia RHS for relativistic hydrodynamics.

Provides the complete hydro RHS pipeline as modular, JIT-compilable functions
mirroring the structure of valencia_reference_metric.py:

  extract_geometry()          -> Build HydroGeometry from BSSN variables
  compute_interface_fluxes()  -> Reconstruction + Riemann solver
  compute_flux_derivative()   -> Finite-volume divergence
  compute_connection_terms()  -> Reference-metric connection terms
  compute_source_terms()      -> Physical source terms from T^{mu nu}
  compute_hydro_rhs()         -> Full RHS orchestrator
  get_hydro_emtensor()        -> Stress-energy tensor for BSSN coupling

All functions are pure-functional and JIT-compilable for GPU execution.
"""

import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import NamedTuple
import numpy as np

from source.matter.hydro.jax.cons2prim_jax import (
    _solve_newton_batch_jax,
    _solve_newton_batch_polytropic_jax,
    _solve_kastaun_batch_jax,
)
from source.matter.hydro.jax.reconstruction_jax import (
    wenoz_reconstruct_3vars_jax,
    weno5_reconstruct_3vars_jax,
    mp5_reconstruct_3vars_jax,
    mc_reconstruct_3vars_jax,
    minmod_reconstruct_3vars_jax,
)
from source.matter.hydro.jax.riemann_jax import (
    physical_flux_1d_jax,
    hll_flux_jax,
    llf_flux_jax,
    find_cp_cm_jax,
    entropy_fix_jax,
    compute_lorentz_factor_1d_jax,
    compute_4velocity_1d_jax,
    compute_g4UU_1d_jax,
)
from source.matter.hydro.jax.eos_jax import (
    eps_from_rho_p_ideal,
    eps_from_rho_p_polytropic,
    sound_speed_squared_ideal,
    sound_speed_squared_polytropic,
    prim_to_cons_jax,
)
from source.bssn.jax.tensoralgebra_jax import (
    EMTensor, get_bar_gamma_LL,
)
from source.matter.hydro.jax.interpolation_jax import (
    lagrange_interpolate_to_faces_4th_order,
    lagrange_interpolate_vector_to_faces,
    lagrange_interpolate_tensor_to_faces,
)


# =============================================================================
# Geometry container
# =============================================================================

class HydroGeometry(NamedTuple):
    """
    Geometry data needed by the hydro RHS, extracted from BSSN variables.

    In Cowling mode this is built once and reused. In dynamic mode it is
    rebuilt every RHS evaluation from the current BSSN state.

    NamedTuples are automatically JAX pytrees (no registration needed).
    """
    alpha: jnp.ndarray       # (N,) lapse
    beta_r: jnp.ndarray      # (N,) radial shift component
    gamma_rr: jnp.ndarray   # (N,) radial metric component
    e6phi: jnp.ndarray      # (N,) conformal factor e^{6phi}
    e4phi: jnp.ndarray      # (N,) conformal factor e^{4phi}
    beta_U: jnp.ndarray     # (N,3) full shift vector (physical)
    gamma_LL: jnp.ndarray   # (N,3,3) full physical covariant metric
    gamma_UU: jnp.ndarray   # (N,3,3) full physical contravariant metric
    alpha_f: jnp.ndarray    # (N-1,) lapse at cell faces
    beta_U_f: jnp.ndarray   # (N-1,3) shift vector at cell faces
    gamma_LL_f: jnp.ndarray # (N-1,3,3) metric at cell faces
    e6phi_f: jnp.ndarray   # (N-1,) e^{6phi} at cell faces
    sqrt_g_hat_f: jnp.ndarray  # (N-1,) sqrt(g_hat) at cell faces


# =============================================================================
# Geometry extraction
# =============================================================================

def extract_geometry(phi, h_LL, lapse, shift_U, background):
    """
    Extract physical geometry from BSSN variables.

    Matches ValenciaReferenceMetric._extract_geometry().

    Args:
        phi: (N,) conformal factor
        h_LL: (N,3,3) rescaled metric perturbation
        lapse: (N,) lapse function
        shift_U: (N,3) rescaled shift vector
        background: BSSNBackground pytree

    Returns:
        HydroGeometry with all geometry needed for the hydro RHS.
    """
    r = background.r

    e4phi = jnp.exp(4.0 * phi)
    e6phi = jnp.exp(6.0 * phi)

    beta_U = background.inverse_scaling_vector * shift_U
    beta_r = beta_U[:, 0]

    bar_gamma_LL = get_bar_gamma_LL(r, h_LL, background)
    gamma_LL = e4phi[:, None, None] * bar_gamma_LL
    gamma_UU = jnp.linalg.inv(gamma_LL)
    gamma_rr = gamma_LL[:, 0, 0]

    sqrt_g_hat_cell = jnp.sqrt(jnp.abs(background.det_hat_gamma) + 1e-30)

    alpha_f = lagrange_interpolate_to_faces_4th_order(lapse)
    e6phi_f = lagrange_interpolate_to_faces_4th_order(e6phi)
    beta_U_f = lagrange_interpolate_vector_to_faces(beta_U)
    gamma_LL_f = lagrange_interpolate_tensor_to_faces(gamma_LL)
    sqrt_g_hat_f = lagrange_interpolate_to_faces_4th_order(sqrt_g_hat_cell)

    return HydroGeometry(
        alpha=lapse,
        beta_r=beta_r,
        gamma_rr=gamma_rr,
        e6phi=e6phi,
        e4phi=e4phi,
        beta_U=beta_U,
        gamma_LL=gamma_LL,
        gamma_UU=gamma_UU,
        alpha_f=alpha_f,
        beta_U_f=beta_U_f,
        gamma_LL_f=gamma_LL_f,
        e6phi_f=e6phi_f,
        sqrt_g_hat_f=sqrt_g_hat_f,
    )


# =============================================================================
# Cons2Prim (JAX-native, no class overhead)
# =============================================================================

@partial(jit, static_argnums=(4, 5, 6, 7, 8, 9))
def cons2prim_batch_ideal(D, Sr, tau, gamma_rr,
                          eos_gamma, rho_floor, p_floor, W_max, tol, max_iter):
    """
    Full cons2prim for ideal gas: atmosphere detection + Newton solve + floors.
    Single JIT-compiled function, no Python overhead.
    """
    gm1 = eos_gamma - 1.0
    N = D.shape[0]

    # Atmosphere detection (branchless)
    atm_mask = D < 100.0 * rho_floor

    # Initial pressure guess
    p_guess = jnp.maximum(p_floor, 0.1 * (tau + D))

    # Solve with vmap Newton
    alpha_dummy = jnp.ones(N)  # not used in ideal gas kernel
    rho0_s, vr_s, p_s, eps_s, W_s, h_s, conv_s = _solve_newton_batch_jax(
        D, Sr, tau, gamma_rr, alpha_dummy, p_guess,
        eos_gamma, p_floor, W_max, tol, max_iter
    )

    # Atmosphere values
    eps_atm = p_floor / (rho_floor * gm1)
    h_atm = 1.0 + eps_atm + p_floor / rho_floor

    # Apply atmosphere for low-D points and failed points
    failed_mask = ~conv_s | atm_mask
    rho0 = jnp.where(failed_mask, rho_floor, rho0_s)
    vr = jnp.where(failed_mask, 0.0, vr_s)
    p = jnp.where(failed_mask, p_floor, p_s)
    eps = jnp.where(failed_mask, eps_atm, eps_s)
    W = jnp.where(failed_mask, 1.0, W_s)
    h = jnp.where(failed_mask, h_atm, h_s)

    # Final floors
    rho0 = jnp.maximum(rho0, rho_floor)
    p = jnp.maximum(p, p_floor)

    return rho0, vr, p, eps, W, h


@partial(jit, static_argnums=(4, 5, 6, 7, 8, 9, 10))
def cons2prim_batch_polytropic(D, Sr, tau, gamma_rr,
                               eos_K, eos_gamma, rho_floor, p_floor, W_max, tol, max_iter):
    """
    Full cons2prim for polytropic EOS: atmosphere detection + Newton solve + floors.
    """
    gm1 = eos_gamma - 1.0
    N = D.shape[0]

    atm_mask = D < 100.0 * rho_floor

    # Initial density guess (polytropic uses rho as unknown)
    rho_guess = jnp.maximum(D, rho_floor)
    alpha_dummy = jnp.ones(N)

    rho0_s, vr_s, p_s, eps_s, W_s, h_s, conv_s = _solve_newton_batch_polytropic_jax(
        D, Sr, tau, gamma_rr, alpha_dummy, rho_guess,
        eos_K, eos_gamma, rho_floor, W_max, tol, max_iter
    )

    # Atmosphere values
    eps_atm = eos_K * rho_floor**gm1 / gm1
    p_atm = eos_K * rho_floor**eos_gamma
    h_atm = 1.0 + eos_gamma * eos_K * rho_floor**gm1 / gm1

    failed_mask = ~conv_s | atm_mask
    rho0 = jnp.where(failed_mask, rho_floor, rho0_s)
    vr = jnp.where(failed_mask, 0.0, vr_s)
    p = jnp.where(failed_mask, p_atm, p_s)
    eps = jnp.where(failed_mask, eps_atm, eps_s)
    W = jnp.where(failed_mask, 1.0, W_s)
    h = jnp.where(failed_mask, h_atm, h_s)

    rho0 = jnp.maximum(rho0, rho_floor)
    p = jnp.maximum(p, p_floor)

    return rho0, vr, p, eps, W, h


@partial(jit, static_argnums=(4, 5, 6, 7, 8, 9, 10))
def cons2prim_batch_kastaun_ideal(D, Sr, tau, gamma_rr,
                                   eos_gamma, rho_floor, p_floor, v_max, W_max, tol, max_iter):
    """
    Full cons2prim for ideal gas using Kastaun et al. (2021) bracketed solver.
    Atmosphere detection + Kastaun solve + floors.
    """
    gm1 = eos_gamma - 1.0

    # Atmosphere detection (branchless)
    atm_mask = D < 100.0 * rho_floor

    # Solve with vmap Kastaun
    rho0_s, vr_s, p_s, eps_s, W_s, h_s, conv_s = _solve_kastaun_batch_jax(
        D, Sr, tau, gamma_rr,
        eos_gamma, rho_floor, p_floor, v_max, W_max, tol, max_iter
    )

    # Atmosphere values
    eps_atm = p_floor / (rho_floor * gm1)
    h_atm = 1.0 + eps_atm + p_floor / rho_floor

    # Apply atmosphere for low-D points and failed points
    failed_mask = ~conv_s | atm_mask
    rho0 = jnp.where(failed_mask, rho_floor, rho0_s)
    vr = jnp.where(failed_mask, 0.0, vr_s)
    p = jnp.where(failed_mask, p_floor, p_s)
    eps = jnp.where(failed_mask, eps_atm, eps_s)
    W = jnp.where(failed_mask, 1.0, W_s)
    h = jnp.where(failed_mask, h_atm, h_s)

    # Final floors
    rho0 = jnp.maximum(rho0, rho_floor)
    p = jnp.maximum(p, p_floor)

    return rho0, vr, p, eps, W, h


# =============================================================================
# Reconstruction wrapper (dispatches to correct method)
# =============================================================================

@partial(jit, static_argnums=(3, 4, 5, 6, 7))
def reconstruct_primitives(rho0, vr, p, method, dx, rho_floor, p_floor, v_max, gamma_rr_f):
    """
    Reconstruct primitives at interfaces and apply floors.

    Args:
        rho0, vr, p: (N,) cell-center primitive arrays
        method: reconstruction method string
        dx: grid spacing
        rho_floor, p_floor, v_max: atmosphere parameters
        gamma_rr_f: (N-1,) face metric for velocity limiting

    Returns:
        rhoL, rhoR, vrL, vrR, pL, pR: (N-1,) face values
    """
    N = len(rho0)

    # Choose reconstruction kernel
    if method in ("wenoz", "weno-z"):
        rL_int, rR_int, vL_int, vR_int, pL_int, pR_int = wenoz_reconstruct_3vars_jax(rho0, vr, p)
        offset = 3  # WENO needs 2 ghost cells on each side -> interior starts at 3
    elif method == "weno5":
        rL_int, rR_int, vL_int, vR_int, pL_int, pR_int = weno5_reconstruct_3vars_jax(rho0, vr, p)
        offset = 3
    elif method == "mp5":
        rL_int, rR_int, vL_int, vR_int, pL_int, pR_int = mp5_reconstruct_3vars_jax(rho0, vr, p)
        offset = 3
    elif method == "mc":
        rL_int, rR_int, vL_int, vR_int, pL_int, pR_int = mc_reconstruct_3vars_jax(rho0, vr, p, dx)
        offset = 2
    else:  # minmod
        rL_int, rR_int, vL_int, vR_int, pL_int, pR_int = minmod_reconstruct_3vars_jax(rho0, vr, p, dx)
        offset = 2

    # Place into full arrays (N+1 faces)
    rhoL = jnp.zeros(N + 1)
    rhoR = jnp.zeros(N + 1)
    vrL = jnp.zeros(N + 1)
    vrR = jnp.zeros(N + 1)
    pL_arr = jnp.zeros(N + 1)
    pR_arr = jnp.zeros(N + 1)

    n_int = len(rL_int)

    if offset == 3:
        # WENO/MP5: interior starts at face 3
        rhoL = rhoL.at[offset:offset+n_int].set(rL_int)
        rhoR = rhoR.at[offset-1:offset-1+n_int].set(rR_int)
        vrL = vrL.at[offset:offset+n_int].set(vL_int)
        vrR = vrR.at[offset-1:offset-1+n_int].set(vR_int)
        pL_arr = pL_arr.at[offset:offset+n_int].set(pL_int)
        pR_arr = pR_arr.at[offset-1:offset-1+n_int].set(pR_int)

        # Fill near-boundary
        rhoL = rhoL.at[1].set(rho0[0]).at[2].set(rho0[1])
        rhoR = rhoR.at[1].set(rho0[0]).at[N-2].set(rho0[-2]).at[N-1].set(rho0[-1])
        vrL = vrL.at[1].set(vr[0]).at[2].set(vr[1])
        vrR = vrR.at[1].set(vr[0]).at[N-2].set(vr[-2]).at[N-1].set(vr[-1])
        pL_arr = pL_arr.at[1].set(p[0]).at[2].set(p[1])
        pR_arr = pR_arr.at[1].set(p[0]).at[N-2].set(p[-2]).at[N-1].set(p[-1])
    else:
        # Minmod/MC: interior starts at face 2
        rhoL = rhoL.at[2:N].set(rL_int)
        rhoR = rhoR.at[1:N-1].set(rR_int)
        vrL = vrL.at[2:N].set(vL_int)
        vrR = vrR.at[1:N-1].set(vR_int)
        pL_arr = pL_arr.at[2:N].set(pL_int)
        pR_arr = pR_arr.at[1:N-1].set(pR_int)

        rhoL = rhoL.at[1].set(rho0[0])
        rhoR = rhoR.at[N-1].set(rho0[-1])
        vrL = vrL.at[1].set(vr[0])
        vrR = vrR.at[N-1].set(vr[-1])
        pL_arr = pL_arr.at[1].set(p[0])
        pR_arr = pR_arr.at[N-1].set(p[-1])

    # Outflow boundaries
    rhoL = rhoL.at[0].set(rho0[0]).at[N].set(rho0[-1])
    rhoR = rhoR.at[0].set(rho0[0]).at[N].set(rho0[-1])
    vrL = vrL.at[0].set(vr[0]).at[N].set(vr[-1])
    vrR = vrR.at[0].set(vr[0]).at[N].set(vr[-1])
    pL_arr = pL_arr.at[0].set(p[0]).at[N].set(p[-1])
    pR_arr = pR_arr.at[0].set(p[0]).at[N].set(p[-1])

    # Reflecting boundary at r=0
    vrL = vrL.at[0].set(0.0)
    vrR = vrR.at[0].set(0.0)

    # Extract interior faces (1 to N-1 for N cells -> N-1 interior faces)
    rhoL = rhoL[1:-1]
    rhoR = rhoR[1:-1]
    vrL = vrL[1:-1]
    vrR = vrR[1:-1]
    pL_arr = pL_arr[1:-1]
    pR_arr = pR_arr[1:-1]

    # Apply floors
    rhoL = jnp.maximum(rhoL, rho_floor)
    rhoR = jnp.maximum(rhoR, rho_floor)
    pL_arr = jnp.maximum(pL_arr, p_floor)
    pR_arr = jnp.maximum(pR_arr, p_floor)

    # Velocity limiting
    v_limit = v_max / jnp.sqrt(jnp.maximum(gamma_rr_f, 1.0))
    vrL = jnp.clip(vrL, -v_limit, v_limit)
    vrR = jnp.clip(vrR, -v_limit, v_limit)

    # Surface velocity damping (atmosphere-adjacent cells)
    surface_threshold = 100.0 * rho_floor
    vrL = jnp.where(rhoL < surface_threshold, 0.0, vrL)
    vrR = jnp.where(rhoR < surface_threshold, 0.0, vrR)

    return rhoL, rhoR, vrL, vrR, pL_arr, pR_arr


# =============================================================================
# Riemann solver (JAX-native, functional interface)
# =============================================================================

@jit
def compute_hll_flux(rhoL, vrL, pL, rhoR, vrR, pR,
                     DL, SrL, tauL, DR, SrR, tauR,
                     gamma_rr, alpha, beta_r, e6phi,
                     epsL, epsR, cs2L, cs2R):
    """
    Compute HLL flux at all interfaces. Single JIT-compiled function.
    """
    # Lorentz factors and enthalpies
    WL = compute_lorentz_factor_1d_jax(vrL, gamma_rr)
    WR = compute_lorentz_factor_1d_jax(vrR, gamma_rr)
    hL = 1.0 + epsL + pL / jnp.maximum(rhoL, 1e-30)
    hR = 1.0 + epsR + pR / jnp.maximum(rhoR, 1e-30)

    # 4-velocities
    u4UL = compute_4velocity_1d_jax(vrL, gamma_rr, alpha, beta_r)
    u4UR = compute_4velocity_1d_jax(vrR, gamma_rr, alpha, beta_r)

    # Inverse 4-metric
    g4UU = compute_g4UU_1d_jax(alpha, beta_r, 1.0 / gamma_rr)

    # Characteristic speeds
    cmL, cpL = find_cp_cm_jax(0, g4UU, u4UL, cs2L)
    cmR, cpR = find_cp_cm_jax(0, g4UU, u4UR, cs2R)

    lam_minus = -jnp.minimum(0.0, jnp.minimum(cmL, cmR))
    lam_plus = jnp.maximum(0.0, jnp.maximum(cpL, cpR))
    lam_minus, lam_plus = entropy_fix_jax(-lam_minus, lam_plus)

    # Physical fluxes (with full shift support)
    FL = physical_flux_1d_jax(rhoL, vrL, pL, WL, hL, alpha, e6phi, gamma_rr, beta_r)
    FR = physical_flux_1d_jax(rhoR, vrR, pR, WR, hR, alpha, e6phi, gamma_rr, beta_r)

    # HLL combination
    return hll_flux_jax(DL, SrL, tauL, DR, SrR, tauR, FL, FR, lam_minus, lam_plus)


# =============================================================================
# Interface fluxes (reconstruction + Riemann)
# =============================================================================

def compute_interface_fluxes(rho0, vr, pressure, geom, dx,
                             eos_type, eos_gamma, eos_K,
                             rho_floor, p_floor, v_max, W_max,
                             recon_method):
    """
    Reconstruction + Riemann solver at cell interfaces.

    Matches ValenciaReferenceMetric._compute_interface_fluxes().

    Args:
        rho0, vr, pressure: (N,) cell-center primitive arrays
        geom: HydroGeometry
        dx: grid spacing (float)
        eos_type, eos_gamma, eos_K: EOS parameters
        rho_floor, p_floor, v_max, W_max: atmosphere parameters
        recon_method: reconstruction method string

    Returns:
        F_D, F_Sr, F_tau: (N-1,) numerical fluxes at interfaces
    """
    alpha_f = geom.alpha_f
    beta_r_f = geom.beta_U_f[:, 0]
    gamma_rr_f = geom.gamma_LL_f[:, 0, 0]
    e6phi_f = geom.e6phi_f

    # Reconstruct primitives at interfaces
    rhoL, rhoR, vrL, vrR, pL, pR = reconstruct_primitives(
        rho0, vr, pressure, recon_method, dx, rho_floor, p_floor, v_max, gamma_rr_f
    )

    # EOS at faces
    if eos_type == 'ideal_gas':
        epsL = eps_from_rho_p_ideal(rhoL, pL, eos_gamma)
        epsR = eps_from_rho_p_ideal(rhoR, pR, eos_gamma)
        cs2L = sound_speed_squared_ideal(rhoL, pL, epsL, eos_gamma)
        cs2R = sound_speed_squared_ideal(rhoR, pR, epsR, eos_gamma)
    else:
        epsL = eps_from_rho_p_polytropic(rhoL, pL, eos_K, eos_gamma)
        epsR = eps_from_rho_p_polytropic(rhoR, pR, eos_K, eos_gamma)
        cs2L = sound_speed_squared_polytropic(rhoL, eos_K, eos_gamma)
        cs2R = sound_speed_squared_polytropic(rhoR, eos_K, eos_gamma)

    # Prim to cons at faces (for Riemann solver)
    DL, SrL_f, tauL = prim_to_cons_jax(rhoL, vrL, pL, gamma_rr_f, e6phi_f,
                                        eos_type, eos_gamma, eos_K)
    DR, SrR_f, tauR = prim_to_cons_jax(rhoR, vrR, pR, gamma_rr_f, e6phi_f,
                                        eos_type, eos_gamma, eos_K)

    # Riemann solver
    flux = compute_hll_flux(
        rhoL, vrL, pL, rhoR, vrR, pR,
        DL, SrL_f, tauL, DR, SrR_f, tauR,
        gamma_rr_f, alpha_f, beta_r_f, e6phi_f,
        epsL, epsR, cs2L, cs2R
    )

    F_D = flux[:, 0]
    F_Sr = flux[:, 1]
    F_tau = flux[:, 2]

    return F_D, F_Sr, F_tau


# =============================================================================
# Flux divergence
# =============================================================================

def compute_flux_derivative(F_D, F_Sr, F_tau, N, num_ghosts, inv_dx):
    """
    Finite-volume flux divergence for interior cells.

    Matches ValenciaReferenceMetric._compute_flux_derivative().

    Returns:
        div_D, div_Sr, div_tau: (N,) arrays (zero in ghost cells)
    """
    i_s = num_ghosts
    i_e = N - num_ghosts

    div_D = jnp.zeros(N)
    div_Sr = jnp.zeros(N)
    div_tau = jnp.zeros(N)

    div_D = div_D.at[i_s:i_e].set(-(F_D[i_s:i_e] - F_D[i_s-1:i_e-1]) * inv_dx)
    div_Sr = div_Sr.at[i_s:i_e].set(-(F_Sr[i_s:i_e] - F_Sr[i_s-1:i_e-1]) * inv_dx)
    div_tau = div_tau.at[i_s:i_e].set(-(F_tau[i_s:i_e] - F_tau[i_s-1:i_e-1]) * inv_dx)

    return div_D, div_Sr, div_tau


# =============================================================================
# Source terms
# =============================================================================

@jit
def compute_source_terms(rho0, vr, pressure, W, h, geom,
                         K_LL, dalpha_dx, hatD_beta_U, hatD_gamma_LL):
    """
    Compute source terms for Valencia equations.

    Matches ValenciaReferenceMetric._compute_source_terms().

    Args:
        rho0, vr, pressure, W, h: (N,) primitive variables
        geom: HydroGeometry
        K_LL: (N,3,3) extrinsic curvature
        dalpha_dx: (N,3) lapse derivatives
        hatD_beta_U: (N,3,3) covariant derivative of shift
        hatD_gamma_LL: (N,3,3,3) covariant derivative of metric

    Returns:
        src_Sr: (N,) radial momentum source
        src_tau: (N,) energy source
    """
    N = rho0.shape[0]
    alpha = geom.alpha
    beta_U = geom.beta_U
    gamma_LL = geom.gamma_LL
    gamma_UU = geom.gamma_UU
    e6phi = geom.e6phi

    # Build full 3D velocity (radial only)
    v_U = jnp.zeros((N, 3))
    v_U = v_U.at[:, 0].set(vr)

    # 4-velocity
    u0 = W / alpha
    VUtilde = v_U - beta_U / alpha[:, None]
    uiU = W[:, None] * VUtilde

    # Contravariant 4-metric components
    alpha_sq = alpha**2
    g4UU_00 = -1.0 / alpha_sq
    g4UU_0i = beta_U / alpha_sq[:, None]
    g4UU_ij = gamma_UU - jnp.einsum('xi,xj->xij', beta_U, beta_U) / alpha_sq[:, None, None]

    # Stress-energy tensor T^{mu nu}
    rho_h = rho0 * h
    TUU_00 = rho_h * u0 * u0 + pressure * g4UU_00
    TUU_0i = (rho_h * u0)[:, None] * uiU + pressure[:, None] * g4UU_0i
    TUU_ij = rho_h[:, None, None] * jnp.einsum('xi,xj->xij', uiU, uiU) + pressure[:, None, None] * g4UU_ij

    # Covariant 4-metric
    beta_lower = jnp.einsum('xij,xj->xi', gamma_LL, beta_U)
    beta_sq = jnp.einsum('xi,xi->x', beta_U, beta_lower)
    g4DD_00 = -alpha_sq + beta_sq
    g4DD_0i = beta_lower
    g4DD_ij = gamma_LL

    # Mixed tensor T^0_j
    TUD_0j = TUU_00[:, None] * g4DD_0i + jnp.einsum('xi,xij->xj', TUU_0i, g4DD_ij)

    # ---- Energy source ----
    tensor_block = (
        TUU_00[:, None, None] * jnp.einsum('xi,xj->xij', beta_U, beta_U)
        + 2.0 * jnp.einsum('xi,xj->xij', TUU_0i, beta_U)
        + TUU_ij
    )
    term1_tau = jnp.einsum('xij,xij->x', K_LL, tensor_block)
    term2_tau = -(
        jnp.einsum('x,xi,xi->x', TUU_00, beta_U, dalpha_dx)
        + jnp.einsum('xi,xi->x', TUU_0i, dalpha_dx)
    )
    src_tau = alpha * e6phi * (term1_tau + term2_tau)

    # ---- Momentum source ----
    first_term = -TUU_00[:, None] * alpha[:, None] * dalpha_dx
    second_term = jnp.einsum('xj,xij->xi', TUD_0j, hatD_beta_U)

    tensor_block_mom = (
        TUU_00[:, None, None] * jnp.einsum('xj,xk->xjk', beta_U, beta_U)
        + 2.0 * jnp.einsum('xj,xk->xjk', TUU_0i, beta_U)
        + TUU_ij
    )
    third_term = 0.5 * jnp.einsum('xjk,xijk->xi', tensor_block_mom, hatD_gamma_LL)

    src_S_vec = alpha[:, None] * e6phi[:, None] * (first_term + second_term + third_term)
    src_Sr = src_S_vec[:, 0]  # Radial component

    return src_Sr, src_tau


# =============================================================================
# Connection terms
# =============================================================================

@jit
def compute_connection_terms(rho0, vr, pressure, W, h, geom,
                             hat_christoffel):
    """
    Compute connection terms from reference metric Christoffels.

    Matches ValenciaReferenceMetric._compute_connection_terms().

    conn_D   = -Gamma^k_{kj} F_tilde^j_D
    conn_tau = -Gamma^k_{kj} F_tilde^j_tau
    conn_Sr  = -Gamma^k_{kj} F_tilde^j_r + Gamma^l_{jr} F_tilde^j_l

    Args:
        rho0, vr, pressure, W, h: (N,) primitive variables
        geom: HydroGeometry
        hat_christoffel: (N,3,3,3) reference Christoffel symbols

    Returns:
        conn_D, conn_Sr, conn_tau: (N,) arrays
    """
    N = rho0.shape[0]
    alpha = geom.alpha
    beta_U = geom.beta_U
    gamma_LL = geom.gamma_LL
    gamma_UU = geom.gamma_UU
    e6phi = geom.e6phi

    # Build 3D velocity
    v_U = jnp.zeros((N, 3))
    v_U = v_U.at[:, 0].set(vr)

    # Compute densitized flux vectors
    u0U = W / alpha
    VUtilde = v_U - beta_U / alpha[:, None]
    uiU = W[:, None] * VUtilde

    alpha_sq = alpha[:, None]**2
    g4UU_0i = beta_U / alpha_sq
    g4UU_ij = gamma_UU - jnp.einsum('xi,xj->xij', beta_U, beta_U) / alpha_sq[:, None]

    rho_h = rho0 * h
    TUU_0i = (rho_h * u0U)[:, None] * uiU + pressure[:, None] * g4UU_0i
    TUU_ij = rho_h[:, None, None] * jnp.einsum('xi,xj->xij', uiU, uiU) + pressure[:, None, None] * g4UU_ij

    # T^i_j = T^{ik} g_{kj}
    TUD_ij = jnp.einsum('xik,xkj->xij', TUU_ij, gamma_LL)

    D = rho0 * W

    # Flux vectors
    fD_U = (e6phi * alpha[:] * D)[:, None] * VUtilde
    fTau_U = (alpha[:]**2 * e6phi)[:, None] * TUU_0i - (alpha[:] * e6phi)[:, None] * D[:, None] * VUtilde
    fS_UD = (alpha[:] * e6phi)[:, None, None] * TUD_ij

    # Christoffel trace
    Gamma_trace = jnp.einsum('xkkj->xj', hat_christoffel)

    conn_D = -jnp.einsum('xj,xj->x', Gamma_trace, fD_U)
    conn_tau = -jnp.einsum('xj,xj->x', Gamma_trace, fTau_U)
    conn_S = (-jnp.einsum('xl,xli->xi', Gamma_trace, fS_UD)
              + jnp.einsum('xlji,xjl->xi', hat_christoffel, fS_UD))
    conn_Sr = conn_S[:, 0]

    return conn_D, conn_Sr, conn_tau


# =============================================================================
# Stress-energy tensor for BSSN coupling
# =============================================================================

def get_hydro_emtensor(rho0, vr, pressure, W, h, geom):
    """
    Compute the stress-energy tensor from hydro primitives for BSSN coupling.

    Follows the same convention as perfect_fluid.py:get_emtensor() and
    scalarmatter_jax.py:get_scalar_emtensor(). All quantities use the
    PHYSICAL (not conformal) metric.

    The BSSN RHS (bssnrhs_jax.py) handles the conformal decomposition
    internally via the e^{-4phi} prefactors.

    Args:
        rho0, vr, pressure, W, h: (N,) primitive variables
        geom: HydroGeometry (with physical gamma_LL, gamma_UU)

    Returns:
        EMTensor(rho, Si, Sij, S) namedtuple
    """
    N = rho0.shape[0]
    gamma_LL = geom.gamma_LL
    gamma_UU = geom.gamma_UU

    # 3-velocity vector (radial only in spherical symmetry)
    v_U = jnp.zeros((N, 3))
    v_U = v_U.at[:, 0].set(vr)

    # Lower velocity with physical metric: v_i = gamma_ij v^j
    v_D = jnp.einsum('xij,xj->xi', gamma_LL, v_U)

    # Energy density: rho = rho0 * h * W^2 - p
    pref = rho0 * h * W * W
    rho_em = pref - pressure

    # Momentum density: S_i = rho0 * h * W^2 * v_i
    Si = pref[:, None] * v_D

    # Stress tensor: S_ij = rho0 * h * W^2 * v_i * v_j + p * gamma_ij
    Sij = pref[:, None, None] * jnp.einsum('xi,xj->xij', v_D, v_D) + pressure[:, None, None] * gamma_LL

    # Trace: S = gamma^{ij} S_{ij}
    S = jnp.einsum('xij,xij->x', gamma_UU, Sij)

    return EMTensor(rho=rho_em, Si=Si, Sij=Sij, S=S)


# =============================================================================
# Main RHS orchestrator
# =============================================================================

@partial(jit, static_argnums=(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18))
def compute_hydro_rhs(D, Sr, tau, geom,
                      dx, num_ghosts,
                      eos_type, eos_gamma, eos_K,
                      rho_floor, p_floor, v_max, W_max, tol, max_iter,
                      recon_method, solver_method,
                      use_connections, use_sources,
                      source_data, connection_data):
    """
    Full hydro RHS orchestrator. JIT-compiled.

    Matches ValenciaReferenceMetric.compute_rhs().

    Args:
        D, Sr, tau: (N,) densitized conservative variables
        geom: HydroGeometry pytree (dynamic, traced by JAX)
        dx: float (static) grid spacing
        num_ghosts: int (static) number of ghost cells
        eos_type: str (static) "ideal_gas" or "polytropic"
        eos_gamma, eos_K: float (static) EOS parameters
        rho_floor, p_floor, v_max, W_max, tol, max_iter: (static) atmosphere/solver params
        recon_method: str (static) reconstruction method
        solver_method: str (static) cons2prim solver
        use_connections: bool (static) include connection terms
        use_sources: bool (static) include source terms
        source_data: tuple (K_LL, dalpha_dx, hatD_beta_U, hatD_gamma_LL) or None (dynamic)
        connection_data: tuple (hat_christoffel,) or None (dynamic)

    Returns:
        (rhs_D, rhs_Sr, rhs_tau): (N,) time derivatives
        (rho0, vr, p, eps, W, h): (N,) primitives (for EMTensor computation)
    """
    N = D.shape[0]
    inv_dx = 1.0 / dx

    # 1. De-densitize conservative variables
    D_phys = D / geom.e6phi
    Sr_phys = Sr / geom.e6phi
    tau_phys = tau / geom.e6phi

    # 2. Cons2Prim
    if eos_type == 'ideal_gas' and solver_method == 'kastaun':
        rho0, vr, p, eps, W, h = cons2prim_batch_kastaun_ideal(
            D_phys, Sr_phys, tau_phys, geom.gamma_rr,
            eos_gamma, rho_floor, p_floor, v_max, W_max, tol, max_iter
        )
    elif eos_type == 'ideal_gas':
        rho0, vr, p, eps, W, h = cons2prim_batch_ideal(
            D_phys, Sr_phys, tau_phys, geom.gamma_rr,
            eos_gamma, rho_floor, p_floor, W_max, tol, max_iter
        )
    else:  # polytropic
        rho0, vr, p, eps, W, h = cons2prim_batch_polytropic(
            D_phys, Sr_phys, tau_phys, geom.gamma_rr,
            eos_K, eos_gamma, rho_floor, p_floor, W_max, tol, max_iter
        )

    # 3. Interface fluxes (reconstruction + Riemann)
    F_D, F_Sr, F_tau = compute_interface_fluxes(
        rho0, vr, p, geom, dx,
        eos_type, eos_gamma, eos_K,
        rho_floor, p_floor, v_max, W_max,
        recon_method
    )

    # 4. Flux divergence
    rhs_D, rhs_Sr, rhs_tau = compute_flux_derivative(F_D, F_Sr, F_tau, N, num_ghosts, inv_dx)

    i_s = num_ghosts
    i_e = N - num_ghosts

    # 5. Connection terms
    if use_connections:
        hat_christoffel = connection_data[0]
        conn_D, conn_Sr, conn_tau = compute_connection_terms(
            rho0, vr, p, W, h, geom, hat_christoffel
        )
        rhs_D = rhs_D.at[i_s:i_e].add(conn_D[i_s:i_e])
        rhs_Sr = rhs_Sr.at[i_s:i_e].add(conn_Sr[i_s:i_e])
        rhs_tau = rhs_tau.at[i_s:i_e].add(conn_tau[i_s:i_e])

    # 6. Source terms
    if use_sources:
        K_LL, dalpha_dx, hatD_beta_U, hatD_gamma_LL = source_data
        src_Sr, src_tau = compute_source_terms(
            rho0, vr, p, W, h, geom,
            K_LL, dalpha_dx, hatD_beta_U, hatD_gamma_LL
        )
        rhs_Sr = rhs_Sr.at[i_s:i_e].add(src_Sr[i_s:i_e])
        rhs_tau = rhs_tau.at[i_s:i_e].add(src_tau[i_s:i_e])

    return (rhs_D, rhs_Sr, rhs_tau), (rho0, vr, p, eps, W, h)


# =============================================================================
# Cowling wrapper (backward-compatible public interface)
# =============================================================================

def compute_hydro_rhs_cowling(D, Sr, tau, geom, dx, num_ghosts,
                              eos_type, eos_params, atm_params,
                              recon_method, solver_method="newton",
                              source_data=None, connection_data=None):
    """
    Compute the full hydro RHS for Cowling evolution.

    Thin Python wrapper that unpacks dict parameters and calls the
    JIT-compiled implementation.

    Args:
        D, Sr, tau: (N,) conservative variables (densitized)
        geom: HydroGeometry with pre-computed static geometry
        dx: float, grid spacing
        num_ghosts: int, number of ghost cells
        eos_type: "ideal_gas" or "polytropic"
        eos_params: dict with EOS parameters (gamma, K)
        atm_params: dict with atmosphere parameters (rho_floor, p_floor, v_max, W_max)
        recon_method: "wenoz", "weno5", "mp5", "mc", "minmod"
        solver_method: "newton" or "kastaun"
        source_data: tuple (K_LL, dalpha_dx, hatD_beta_U, hatD_gamma_LL) or None
        connection_data: tuple (hat_christoffel,) or None

    Returns:
        rhs_D, rhs_Sr, rhs_tau: (N,) time derivatives of conservative variables
    """
    use_sources = source_data is not None
    use_connections = connection_data is not None

    (rhs_D, rhs_Sr, rhs_tau), _ = compute_hydro_rhs(
        D, Sr, tau, geom,
        dx, num_ghosts,
        eos_type,
        eos_params['gamma'],
        eos_params.get('K', 0.0),
        atm_params['rho_floor'],
        atm_params['p_floor'],
        atm_params['v_max'],
        atm_params['W_max'],
        atm_params.get('tol', 1e-12),
        atm_params.get('max_iter', 500),
        recon_method,
        solver_method,
        use_connections,
        use_sources,
        source_data,
        connection_data,
    )
    return rhs_D, rhs_Sr, rhs_tau
