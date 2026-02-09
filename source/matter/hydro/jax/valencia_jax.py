"""
JAX-native Valencia RHS for relativistic hydrodynamics (Cowling approximation).

This module provides the complete hydro RHS pipeline in a single JIT-compilable
function. For Cowling evolution, geometry is static and pre-computed once.

Pipeline: cons2prim -> reconstruction -> riemann -> flux divergence + connection + sources

All functions are pure-functional and JIT-compilable for GPU execution.
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
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


# =============================================================================
# Static geometry for Cowling evolution (pre-computed once)
# =============================================================================

class CowlingGeometry:
    """
    Pre-computed static geometry for Cowling (frozen spacetime) evolution.

    All arrays are JAX arrays placed on the target device.
    Computed once from BSSN variables, then reused every timestep.

    Registered as a JAX pytree so it can be passed to @jit functions.
    Arrays are dynamic (traced), scalars/flags are static (aux_data).
    """
    def __init__(self, alpha, beta_r, gamma_rr, e6phi, dx, num_ghosts,
                 # Source term geometry (needed for non-Minkowski spacetimes)
                 K_LL=None, dalpha_dx=None,
                 hatD_beta_U=None, hatD_gamma_LL=None,
                 hat_christoffel=None,
                 beta_U=None, gamma_LL=None, gamma_UU=None,
                 e4phi=None, hat_gamma_LL=None):
        """
        Args:
            alpha: (N,) lapse
            beta_r: (N,) radial shift
            gamma_rr: (N,) radial metric component
            e6phi: (N,) conformal factor e^{6phi}
            dx: float, computational grid spacing
            num_ghosts: int, number of ghost cells
            K_LL: (N,3,3) extrinsic curvature (optional, for source terms)
            dalpha_dx: (N,3) lapse derivatives (optional)
            hatD_beta_U: (N,3,3) covariant derivative of shift (optional)
            hatD_gamma_LL: (N,3,3,3) covariant derivative of metric (optional)
            hat_christoffel: (N,3,3,3) reference Christoffel symbols (optional)
            beta_U: (N,3) full shift vector (optional)
            gamma_LL: (N,3,3) full covariant metric (optional)
            gamma_UU: (N,3,3) full contravariant metric (optional)
        """
        # Core 1D geometry (always needed)
        self.alpha = jnp.asarray(alpha)
        self.beta_r = jnp.asarray(beta_r)
        self.gamma_rr = jnp.asarray(gamma_rr)
        self.e6phi = jnp.asarray(e6phi)
        self.dx = float(dx)
        self.num_ghosts = int(num_ghosts)
        self.N = len(alpha)

        # Face-interpolated geometry
        self.alpha_f = 0.5 * (self.alpha[:-1] + self.alpha[1:])
        self.beta_r_f = 0.5 * (self.beta_r[:-1] + self.beta_r[1:])
        self.gamma_rr_f = 0.5 * (self.gamma_rr[:-1] + self.gamma_rr[1:])
        self.e6phi_f = 0.5 * (self.e6phi[:-1] + self.e6phi[1:])

        # Full 3D geometry for source terms
        self.beta_U = jnp.asarray(beta_U) if beta_U is not None else None
        self.gamma_LL = jnp.asarray(gamma_LL) if gamma_LL is not None else None
        self.gamma_UU = jnp.asarray(gamma_UU) if gamma_UU is not None else None
        self.e4phi = jnp.asarray(e4phi) if e4phi is not None else None

        # Source term pre-computed geometry
        self.K_LL = jnp.asarray(K_LL) if K_LL is not None else None
        self.dalpha_dx = jnp.asarray(dalpha_dx) if dalpha_dx is not None else None
        self.hatD_beta_U = jnp.asarray(hatD_beta_U) if hatD_beta_U is not None else None
        self.hatD_gamma_LL = jnp.asarray(hatD_gamma_LL) if hatD_gamma_LL is not None else None
        self.hat_christoffel = jnp.asarray(hat_christoffel) if hat_christoffel is not None else None

        # Flag: are source terms available?
        self.has_sources = (K_LL is not None and dalpha_dx is not None)
        self.has_connections = (hat_christoffel is not None)

    def tree_flatten(self):
        """Flatten for JAX pytree: arrays=children, scalars=aux_data."""
        children = (
            self.alpha, self.beta_r, self.gamma_rr, self.e6phi,
            self.alpha_f, self.beta_r_f, self.gamma_rr_f, self.e6phi_f,
            self.beta_U, self.gamma_LL, self.gamma_UU, self.e4phi,
            self.K_LL, self.dalpha_dx, self.hatD_beta_U, self.hatD_gamma_LL,
            self.hat_christoffel,
        )
        aux_data = (self.N, self.num_ghosts, self.dx,
                    self.has_sources, self.has_connections)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct from flattened pytree."""
        (alpha, beta_r, gamma_rr, e6phi,
         alpha_f, beta_r_f, gamma_rr_f, e6phi_f,
         beta_U, gamma_LL, gamma_UU, e4phi,
         K_LL, dalpha_dx, hatD_beta_U, hatD_gamma_LL,
         hat_christoffel) = children
        N, num_ghosts, dx, has_sources, has_connections = aux_data
        # Build object without recomputing face values
        obj = object.__new__(cls)
        obj.alpha = alpha
        obj.beta_r = beta_r
        obj.gamma_rr = gamma_rr
        obj.e6phi = e6phi
        obj.dx = dx
        obj.num_ghosts = num_ghosts
        obj.N = N
        obj.alpha_f = alpha_f
        obj.beta_r_f = beta_r_f
        obj.gamma_rr_f = gamma_rr_f
        obj.e6phi_f = e6phi_f
        obj.beta_U = beta_U
        obj.gamma_LL = gamma_LL
        obj.gamma_UU = gamma_UU
        obj.e4phi = e4phi
        obj.K_LL = K_LL
        obj.dalpha_dx = dalpha_dx
        obj.hatD_beta_U = hatD_beta_U
        obj.hatD_gamma_LL = hatD_gamma_LL
        obj.hat_christoffel = hat_christoffel
        obj.has_sources = has_sources
        obj.has_connections = has_connections
        return obj


# Register CowlingGeometry as a JAX pytree
jax.tree_util.register_pytree_node(
    CowlingGeometry,
    lambda obj: obj.tree_flatten(),
    lambda aux_data, children: CowlingGeometry.tree_unflatten(aux_data, children),
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
# Source terms (for non-Minkowski Cowling spacetimes)
# =============================================================================

@jit
def compute_source_terms(rho0, vr, pressure, W, h,
                         alpha, beta_U, gamma_LL, gamma_UU, e6phi,
                         K_LL, dalpha_dx, hatD_beta_U, hatD_gamma_LL):
    """
    Compute source terms for Valencia equations (Cowling).

    All geometry arrays are STATIC (pre-computed).
    Only rho0, vr, pressure, W, h are dynamic.

    Returns:
        src_Sr: (N,) radial momentum source
        src_tau: (N,) energy source
    """
    N = rho0.shape[0]

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
def compute_connection_terms(rho0, vr, pressure, W, h,
                             alpha, beta_U, gamma_LL, gamma_UU, e6phi,
                             hat_christoffel):
    """
    Compute connection terms from reference metric Christoffels.

    conn_D   = -Gamma^k_{kj} F_tilde^j_D
    conn_tau = -Gamma^k_{kj} F_tilde^j_tau
    conn_Sr  = -Gamma^k_{kj} F_tilde^j_r + Gamma^l_{jr} F_tilde^j_l

    Returns:
        conn_D, conn_Sr, conn_tau: (N,) arrays
    """
    N = rho0.shape[0]

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
    fD_U = (e6phi * alpha * D)[:, None] * VUtilde
    fTau_U = (alpha**2 * e6phi)[:, None] * TUU_0i - (alpha * e6phi)[:, None] * D[:, None] * VUtilde
    fS_UD = (alpha * e6phi)[:, None, None] * TUD_ij

    # Christoffel trace
    Gamma_trace = jnp.einsum('xkkj->xj', hat_christoffel)

    conn_D = -jnp.einsum('xj,xj->x', Gamma_trace, fD_U)
    conn_tau = -jnp.einsum('xj,xj->x', Gamma_trace, fTau_U)
    conn_S = (-jnp.einsum('xl,xli->xi', Gamma_trace, fS_UD)
              + jnp.einsum('xlji,xjl->xi', hat_christoffel, fS_UD))
    conn_Sr = conn_S[:, 0]

    return conn_D, conn_Sr, conn_tau


# =============================================================================
# Main RHS function (Cowling approximation)
# =============================================================================

@partial(jit, static_argnums=(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17))
def _compute_hydro_rhs_impl(D, Sr, tau, geom,
                             eos_type, eos_gamma, eos_K,
                             rho_floor, p_floor, v_max, W_max, tol, max_iter,
                             recon_method, riemann_type,
                             use_connections, use_sources,
                             solver_method):
    """
    JIT-compiled hydro RHS implementation for Cowling evolution.

    All scalar parameters are static (resolved at trace time).
    geom is a CowlingGeometry pytree (arrays traced, N/dx/flags static).
    """
    N = geom.N
    NG = geom.num_ghosts
    dx = geom.dx
    inv_dx = 1.0 / dx

    # =========================================================================
    # 1. De-densitize conservative variables (D̃ = e^{6φ} D_phys)
    # State stores DENSITIZED conservatives; cons2prim expects PHYSICAL
    # =========================================================================
    D_phys = D / geom.e6phi
    Sr_phys = Sr / geom.e6phi
    tau_phys = tau / geom.e6phi

    # =========================================================================
    # 2. Cons2Prim (on PHYSICAL conservatives)
    # =========================================================================
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

    # =========================================================================
    # 2. Reconstruction
    # =========================================================================
    rhoL, rhoR, vrL, vrR, pL, pR = reconstruct_primitives(
        rho0, vr, p, recon_method, dx, rho_floor, p_floor, v_max, geom.gamma_rr_f
    )

    # =========================================================================
    # 3. Compute EOS quantities at faces
    # =========================================================================
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

    # =========================================================================
    # 4. Prim to cons at faces (for Riemann solver)
    # =========================================================================
    DL, SrL_f, tauL = prim_to_cons_jax(rhoL, vrL, pL, geom.gamma_rr_f, geom.e6phi_f,
                                        eos_type, eos_gamma, eos_K)
    DR, SrR_f, tauR = prim_to_cons_jax(rhoR, vrR, pR, geom.gamma_rr_f, geom.e6phi_f,
                                        eos_type, eos_gamma, eos_K)

    # =========================================================================
    # 5. Riemann solver
    # =========================================================================
    flux = compute_hll_flux(
        rhoL, vrL, pL, rhoR, vrR, pR,
        DL, SrL_f, tauL, DR, SrR_f, tauR,
        geom.gamma_rr_f, geom.alpha_f, geom.beta_r_f, geom.e6phi_f,
        epsL, epsR, cs2L, cs2R
    )

    F_D = flux[:, 0]
    F_Sr = flux[:, 1]
    F_tau = flux[:, 2]

    # =========================================================================
    # 6. Flux divergence (interior cells only)
    # =========================================================================
    rhs_D = jnp.zeros(N)
    rhs_Sr = jnp.zeros(N)
    rhs_tau = jnp.zeros(N)

    i_s = NG
    i_e = N - NG

    rhs_D = rhs_D.at[i_s:i_e].set(-(F_D[i_s:i_e] - F_D[i_s-1:i_e-1]) * inv_dx)
    rhs_Sr = rhs_Sr.at[i_s:i_e].set(-(F_Sr[i_s:i_e] - F_Sr[i_s-1:i_e-1]) * inv_dx)
    rhs_tau = rhs_tau.at[i_s:i_e].set(-(F_tau[i_s:i_e] - F_tau[i_s-1:i_e-1]) * inv_dx)

    # =========================================================================
    # 7. Connection terms
    # =========================================================================
    if use_connections:
        conn_D, conn_Sr, conn_tau = compute_connection_terms(
            rho0, vr, p, W, h,
            geom.alpha, geom.beta_U, geom.gamma_LL, geom.gamma_UU, geom.e6phi,
            geom.hat_christoffel
        )
        rhs_D = rhs_D.at[i_s:i_e].add(conn_D[i_s:i_e])
        rhs_Sr = rhs_Sr.at[i_s:i_e].add(conn_Sr[i_s:i_e])
        rhs_tau = rhs_tau.at[i_s:i_e].add(conn_tau[i_s:i_e])

    # =========================================================================
    # 8. Source terms
    # =========================================================================
    if use_sources:
        src_Sr, src_tau = compute_source_terms(
            rho0, vr, p, W, h,
            geom.alpha, geom.beta_U, geom.gamma_LL, geom.gamma_UU, geom.e6phi,
            geom.K_LL, geom.dalpha_dx, geom.hatD_beta_U, geom.hatD_gamma_LL
        )
        rhs_Sr = rhs_Sr.at[i_s:i_e].add(src_Sr[i_s:i_e])
        rhs_tau = rhs_tau.at[i_s:i_e].add(src_tau[i_s:i_e])

    return rhs_D, rhs_Sr, rhs_tau


def compute_hydro_rhs_cowling(D, Sr, tau, geom, eos_type, eos_params,
                              atm_params, recon_method, riemann_type="hll",
                              solver_method="newton"):
    """
    Compute the full hydro RHS for Cowling evolution.

    Thin Python wrapper that unpacks dict parameters and calls the
    JIT-compiled implementation. The first call triggers JIT compilation;
    subsequent calls with the same static args use the cached compiled version.

    Args:
        D, Sr, tau: (N,) conservative variables (densitized)
        geom: CowlingGeometry with pre-computed static geometry
        eos_type: "ideal_gas" or "polytropic"
        eos_params: dict with EOS parameters (gamma, K)
        atm_params: dict with atmosphere parameters (rho_floor, p_floor, v_max, W_max)
        recon_method: "wenoz", "weno5", "mp5", "mc", "minmod"
        riemann_type: "hll" or "llf"
        solver_method: "newton" or "kastaun"

    Returns:
        rhs_D, rhs_Sr, rhs_tau: (N,) time derivatives of conservative variables
    """
    return _compute_hydro_rhs_impl(
        D, Sr, tau, geom,
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
        riemann_type,
        geom.has_connections,  # Pass as static argument
        geom.has_sources,      # Pass as static argument
        solver_method,
    )
