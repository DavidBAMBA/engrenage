# rhsevolution_jax.py
#
# JAX port of source/core/rhsevolution.py
# Complete BSSN + matter RHS pipeline in a single JIT-compilable function.
#
# Pipeline:
#   1. Unpack state (NUM_VARS, N) -> individual variables
#   2. Compute derivatives via matrix multiplication
#   3. Algebraic constraints (determinant)
#   4. Matter EMTensor + RHS
#   5. BSSN RHS
#   6. Gauge conditions (1+log, Gamma driver)
#   7. Advection corrections
#   8. Pack into (NUM_VARS, N) RHS
#   9. Kreiss-Oliger dissipation
#  10. Boundary conditions
#  11. Return

import jax
import jax.numpy as jnp
from functools import partial

from source.bssn.bssnstatevariables import (
    NUM_BSSN_VARS,
    idx_phi, idx_hrr, idx_htt, idx_hpp,
    idx_K, idx_arr, idx_att, idx_app,
    idx_lambdar, idx_shiftr, idx_br, idx_lapse,
    BSSN_PARITY, BSSN_ASYMP_POWER, BSSN_ASYMP_OFFSET,
)
from source.bssn.jax.tensoralgebra_jax import (
    SPACEDIM, get_bar_gamma_LL, get_bar_gamma_UU, get_det_bar_gamma,
    get_vector_advection, get_tensor_advection,
)
from source.bssn.jax.bssnrhs_jax import get_bssn_rhs_jax
from source.bssn.jax.boundaries_jax import fill_bssn_boundaries_jax
from source.matter.jax.scalarmatter_jax import get_scalar_emtensor, get_scalar_rhs

# Coordinate indices for spherical symmetry
i_r, i_t, i_p = 0, 1, 2


def _unpack_state(state, N):
    """Unpack flat state (NUM_VARS, N) into individual BSSN variables in tensor form."""
    phi = state[idx_phi]
    h_rr = state[idx_hrr]
    h_tt = state[idx_htt]
    h_pp = state[idx_hpp]
    K = state[idx_K]
    a_rr = state[idx_arr]
    a_tt = state[idx_att]
    a_pp = state[idx_app]
    lambdar = state[idx_lambdar]
    shiftr = state[idx_shiftr]
    br = state[idx_br]
    lapse = state[idx_lapse]

    # Build tensors from diagonal components (spherical symmetry)
    h_LL = jnp.zeros((N, SPACEDIM, SPACEDIM))
    h_LL = h_LL.at[:, i_r, i_r].set(h_rr)
    h_LL = h_LL.at[:, i_t, i_t].set(h_tt)
    h_LL = h_LL.at[:, i_p, i_p].set(h_pp)

    a_LL = jnp.zeros((N, SPACEDIM, SPACEDIM))
    a_LL = a_LL.at[:, i_r, i_r].set(a_rr)
    a_LL = a_LL.at[:, i_t, i_t].set(a_tt)
    a_LL = a_LL.at[:, i_p, i_p].set(a_pp)

    lambda_U = jnp.zeros((N, SPACEDIM))
    lambda_U = lambda_U.at[:, i_r].set(lambdar)

    shift_U = jnp.zeros((N, SPACEDIM))
    shift_U = shift_U.at[:, i_r].set(shiftr)

    b_U = jnp.zeros((N, SPACEDIM))
    b_U = b_U.at[:, i_r].set(br)

    return phi, h_LL, K, a_LL, lambda_U, shift_U, b_U, lapse


def _compute_derivatives(state, deriv_mats, dr, N, shift_r):
    """Compute all needed derivatives via matrix multiplication."""
    # First derivative indices (same as BSSNFirstDerivs)
    d1_indices = jnp.array([idx_phi, idx_hrr, idx_htt, idx_hpp, idx_K,
                             idx_arr, idx_att, idx_app, idx_lambdar, idx_shiftr, idx_lapse])
    # Second derivative indices (same as BSSNSecondDerivs)
    d2_indices = jnp.array([idx_phi, idx_hrr, idx_htt, idx_hpp, idx_shiftr, idx_lapse])

    # Compute raw d1 for all needed variables
    d1_raw = (state @ deriv_mats.d1_matrix.T) / dr  # (NUM_VARS, N)

    # Compute raw d2 for needed variables
    d2_raw = (state @ deriv_mats.d2_matrix.T) / (dr * dr)  # (NUM_VARS, N)

    # Advection: direction depends on sign of shift
    # BSSN upwind: shift_r >= 0 -> right/forward stencil, else left/backward
    advec_l = (state @ deriv_mats.advec_l_matrix.T) / dr
    advec_r = (state @ deriv_mats.advec_r_matrix.T) / dr
    # Branchless selection
    mask = (shift_r >= 0)[jnp.newaxis, :]  # (1, N) broadcast
    advec_raw = jnp.where(mask, advec_r, advec_l)  # (NUM_VARS, N)

    # --- Build tensor forms for d1 ---
    d1_phi = jnp.zeros((N, SPACEDIM))
    d1_phi = d1_phi.at[:, i_r].set(d1_raw[idx_phi])

    d1_K = jnp.zeros((N, SPACEDIM))
    d1_K = d1_K.at[:, i_r].set(d1_raw[idx_K])

    d1_lapse = jnp.zeros((N, SPACEDIM))
    d1_lapse = d1_lapse.at[:, i_r].set(d1_raw[idx_lapse])

    d1_h_LL = jnp.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM))
    d1_h_LL = d1_h_LL.at[:, i_r, i_r, i_r].set(d1_raw[idx_hrr])
    d1_h_LL = d1_h_LL.at[:, i_t, i_t, i_r].set(d1_raw[idx_htt])
    d1_h_LL = d1_h_LL.at[:, i_p, i_p, i_r].set(d1_raw[idx_hpp])

    d1_a_LL = jnp.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM))
    d1_a_LL = d1_a_LL.at[:, i_r, i_r, i_r].set(d1_raw[idx_arr])
    d1_a_LL = d1_a_LL.at[:, i_t, i_t, i_r].set(d1_raw[idx_att])
    d1_a_LL = d1_a_LL.at[:, i_p, i_p, i_r].set(d1_raw[idx_app])

    d1_lambda_U = jnp.zeros((N, SPACEDIM, SPACEDIM))
    d1_lambda_U = d1_lambda_U.at[:, i_r, i_r].set(d1_raw[idx_lambdar])

    d1_shift_U = jnp.zeros((N, SPACEDIM, SPACEDIM))
    d1_shift_U = d1_shift_U.at[:, i_r, i_r].set(d1_raw[idx_shiftr])

    # --- Build tensor forms for d2 ---
    d2_phi = jnp.zeros((N, SPACEDIM, SPACEDIM))
    d2_phi = d2_phi.at[:, i_r, i_r].set(d2_raw[idx_phi])

    d2_lapse = jnp.zeros((N, SPACEDIM, SPACEDIM))
    d2_lapse = d2_lapse.at[:, i_r, i_r].set(d2_raw[idx_lapse])

    d2_h_LL = jnp.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM, SPACEDIM))
    d2_h_LL = d2_h_LL.at[:, i_r, i_r, i_r, i_r].set(d2_raw[idx_hrr])
    d2_h_LL = d2_h_LL.at[:, i_t, i_t, i_r, i_r].set(d2_raw[idx_htt])
    d2_h_LL = d2_h_LL.at[:, i_p, i_p, i_r, i_r].set(d2_raw[idx_hpp])

    d2_shift_U = jnp.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM))
    d2_shift_U = d2_shift_U.at[:, i_r, i_r, i_r].set(d2_raw[idx_shiftr])

    # --- Build tensor forms for advection ---
    advec_phi = jnp.zeros((N, SPACEDIM))
    advec_phi = advec_phi.at[:, i_r].set(advec_raw[idx_phi])

    advec_K = jnp.zeros((N, SPACEDIM))
    advec_K = advec_K.at[:, i_r].set(advec_raw[idx_K])

    advec_lapse = jnp.zeros((N, SPACEDIM))
    advec_lapse = advec_lapse.at[:, i_r].set(advec_raw[idx_lapse])

    advec_h_LL = jnp.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM))
    advec_h_LL = advec_h_LL.at[:, i_r, i_r, i_r].set(advec_raw[idx_hrr])
    advec_h_LL = advec_h_LL.at[:, i_t, i_t, i_r].set(advec_raw[idx_htt])
    advec_h_LL = advec_h_LL.at[:, i_p, i_p, i_r].set(advec_raw[idx_hpp])

    advec_a_LL = jnp.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM))
    advec_a_LL = advec_a_LL.at[:, i_r, i_r, i_r].set(advec_raw[idx_arr])
    advec_a_LL = advec_a_LL.at[:, i_t, i_t, i_r].set(advec_raw[idx_att])
    advec_a_LL = advec_a_LL.at[:, i_p, i_p, i_r].set(advec_raw[idx_app])

    advec_lambda_U = jnp.zeros((N, SPACEDIM, SPACEDIM))
    advec_lambda_U = advec_lambda_U.at[:, i_r, i_r].set(advec_raw[idx_lambdar])

    # Scalar field derivatives (u = state[12], v = state[13])
    idx_u = NUM_BSSN_VARS
    idx_v = NUM_BSSN_VARS + 1

    d1_u = jnp.zeros((N, SPACEDIM))
    d1_u = d1_u.at[:, i_r].set(d1_raw[idx_u])

    d2_u = jnp.zeros((N, SPACEDIM, SPACEDIM))
    d2_u = d2_u.at[:, i_r, i_r].set(d2_raw[idx_u])

    advec_u = jnp.zeros((N, SPACEDIM))
    advec_u = advec_u.at[:, i_r].set(advec_raw[idx_u])

    advec_v = jnp.zeros((N, SPACEDIM))
    advec_v = advec_v.at[:, i_r].set(advec_raw[idx_v])

    return (d1_phi, d1_K, d1_lapse, d1_h_LL, d1_a_LL, d1_lambda_U, d1_shift_U,
            d2_phi, d2_lapse, d2_h_LL, d2_shift_U,
            advec_phi, advec_K, advec_lapse, advec_h_LL, advec_a_LL, advec_lambda_U,
            d1_u, d2_u, advec_u, advec_v)


def _enforce_det_constraint(h_LL, background):
    """
    Enforce det(bar_gamma) = det(hat_gamma) algebraic constraint.
    Rescales bar_gamma_LL so its determinant matches hat_gamma.
    Handles NaN/inf at ghost cells near r=0 where det(hat_gamma)→0.
    """
    epsilon_LL = h_LL * background.scaling_matrix
    bar_gamma_LL = epsilon_LL + background.hat_gamma_LL
    det_bar_gamma = jnp.linalg.det(bar_gamma_LL)
    det_hat_gamma = background.det_hat_gamma

    ratio = det_bar_gamma / (det_hat_gamma + 1e-300)
    # Replace NaN/inf with 1.0 (no rescaling at problematic points)
    ratio = jnp.where(jnp.isfinite(ratio), ratio, 1.0)
    ratio = jnp.clip(ratio, 1e-300, 1e300)
    rescaling_factor = jnp.power(ratio, -1.0 / 3.0)

    new_bar_gamma_LL = rescaling_factor[:, jnp.newaxis, jnp.newaxis] * bar_gamma_LL
    h_LL_new = (new_bar_gamma_LL - background.hat_gamma_LL) * background.inverse_scaling_matrix
    # Protect against NaN from ghost cells
    h_LL_new = jnp.where(jnp.isfinite(h_LL_new), h_LL_new, h_LL)
    return h_LL_new


def _pack_bssn_rhs(dphidt, dhdt, dKdt, dadt, dlambdadt, dlapse_dt, dshift_dt, db_dt, N):
    """Pack BSSN RHS from tensor form back to (NUM_BSSN_VARS, N) flat array."""
    rhs_bssn = jnp.zeros((NUM_BSSN_VARS, N))
    rhs_bssn = rhs_bssn.at[idx_phi].set(dphidt)
    rhs_bssn = rhs_bssn.at[idx_hrr].set(dhdt[:, i_r, i_r])
    rhs_bssn = rhs_bssn.at[idx_htt].set(dhdt[:, i_t, i_t])
    rhs_bssn = rhs_bssn.at[idx_hpp].set(dhdt[:, i_p, i_p])
    rhs_bssn = rhs_bssn.at[idx_K].set(dKdt)
    rhs_bssn = rhs_bssn.at[idx_arr].set(dadt[:, i_r, i_r])
    rhs_bssn = rhs_bssn.at[idx_att].set(dadt[:, i_t, i_t])
    rhs_bssn = rhs_bssn.at[idx_app].set(dadt[:, i_p, i_p])
    rhs_bssn = rhs_bssn.at[idx_lambdar].set(dlambdadt[:, i_r])
    rhs_bssn = rhs_bssn.at[idx_shiftr].set(dshift_dt[:, i_r])
    rhs_bssn = rhs_bssn.at[idx_br].set(db_dt[:, i_r])
    rhs_bssn = rhs_bssn.at[idx_lapse].set(dlapse_dt)
    return rhs_bssn


@partial(jax.jit, static_argnums=(4, 5))
def get_rhs_bssn_scalar_jax(state, bssn_bg, deriv_mats, dr,
                              num_ghosts, num_vars,
                              sigma_base, scalar_mu, eta):
    """
    Full BSSN + scalar field RHS, JIT-compiled.

    Args:
        state: (NUM_VARS, N) current state
        bssn_bg: BSSNBackground pytree
        deriv_mats: DerivativeMatrices pytree
        dr: (N,) grid spacing array
        num_ghosts: int (static) number of ghost cells
        num_vars: int (static) total number of variables
        sigma_base: float, KO dissipation base coefficient
        scalar_mu: float, scalar field mass parameter
        eta: float, Gamma driver damping parameter

    Returns:
        rhs_state: (NUM_VARS, N) right-hand side
    """
    N = state.shape[1]
    r = bssn_bg.r

    # Precompute BC arrays (used for RHS boundary conditions at the end)
    parity = jnp.array(jnp.concatenate([jnp.array(BSSN_PARITY, dtype=jnp.float64),
                                         jnp.ones(num_vars - NUM_BSSN_VARS)]))
    asymp_power = jnp.array(jnp.concatenate([jnp.array(BSSN_ASYMP_POWER, dtype=jnp.float64),
                                               jnp.zeros(num_vars - NUM_BSSN_VARS)]))
    asymp_offset = jnp.array(jnp.concatenate([jnp.array(BSSN_ASYMP_OFFSET, dtype=jnp.float64),
                                                jnp.zeros(num_vars - NUM_BSSN_VARS)]))

    # 1. Unpack state
    phi, h_LL, K, a_LL, lambda_U, shift_U, b_U, lapse = _unpack_state(state, N)
    u = state[NUM_BSSN_VARS]
    v = state[NUM_BSSN_VARS + 1]

    # 2. Enforce determinant constraint
    h_LL = _enforce_det_constraint(h_LL, bssn_bg)

    # Limit conformal factor
    phi = jnp.minimum(phi, 1.0e6)

    # 3. Compute all derivatives
    shift_r = shift_U[:, i_r]
    (d1_phi, d1_K, d1_lapse, d1_h_LL, d1_a_LL, d1_lambda_U, d1_shift_U,
     d2_phi, d2_lapse, d2_h_LL, d2_shift_U,
     advec_phi, advec_K, advec_lapse, advec_h_LL, advec_a_LL, advec_lambda_U,
     d1_u, d2_u, advec_u, advec_v) = _compute_derivatives(state, deriv_mats, dr, N, shift_r)

    # 4. Matter sources
    my_emtensor = get_scalar_emtensor(u, v, d1_u, phi, h_LL, bssn_bg, scalar_mu)
    dudt, dvdt = get_scalar_rhs(u, v, d1_u, d2_u, advec_u, advec_v,
                                 phi, d1_phi, h_LL, d1_h_LL, lapse, d1_lapse, K,
                                 shift_U, bssn_bg, scalar_mu)

    # 5. BSSN RHS
    dphidt, dhdt, dKdt, dadt, dlambdadt = get_bssn_rhs_jax(
        r, phi, h_LL, K, a_LL, lambda_U, lapse, shift_U, b_U,
        d1_phi, d1_h_LL, d1_K, d1_a_LL, d1_lambda_U, d1_lapse, d1_shift_U,
        d2_phi, d2_h_LL, d2_lapse, d2_shift_U,
        bssn_bg, my_emtensor)

    # 6. Gauge conditions
    # 1+log slicing
    dlapse_dt = -2.0 * lapse * K
    # Gamma driver
    dshift_dt = b_U
    db_dt = 0.75 * dlambdadt - eta * b_U
    # dlambdadt is (N, 3), shift_dt and b_dt need to be (N, 3) too
    dshift_dt_vec = jnp.zeros((N, SPACEDIM))
    dshift_dt_vec = dshift_dt_vec.at[:, i_r].set(b_U[:, i_r])
    db_dt_vec = jnp.zeros((N, SPACEDIM))
    db_dt_vec = db_dt_vec.at[:, i_r].set(0.75 * dlambdadt[:, i_r] - eta * b_U[:, i_r])

    # 7. Advection corrections
    # Scalars
    dphidt = dphidt + jnp.einsum('xj,xj->x', bssn_bg.inverse_scaling_vector * shift_U, advec_phi)
    dKdt = dKdt + jnp.einsum('xj,xj->x', bssn_bg.inverse_scaling_vector * shift_U, advec_K)

    # Vectors
    advec_lambda_U_corr = get_vector_advection(r, lambda_U, advec_lambda_U, shift_U, d1_shift_U, bssn_bg)
    dlambdadt = dlambdadt + advec_lambda_U_corr

    # Tensors
    advec_h_LL_corr = get_tensor_advection(r, h_LL, advec_h_LL, shift_U, d1_shift_U, bssn_bg)
    dhdt = dhdt + advec_h_LL_corr

    advec_a_LL_corr = get_tensor_advection(r, a_LL, advec_a_LL, shift_U, d1_shift_U, bssn_bg)
    dadt = dadt + advec_a_LL_corr

    # 8. Pack into flat RHS
    rhs_bssn = _pack_bssn_rhs(dphidt, dhdt, dKdt, dadt, dlambdadt,
                                dlapse_dt, dshift_dt_vec, db_dt_vec, N)
    rhs_matter = jnp.stack([dudt, dvdt])  # (2, N)
    rhs_state = jnp.concatenate([rhs_bssn, rhs_matter], axis=0)

    # 9. Kreiss-Oliger dissipation (BSSN variables only)
    sigma = sigma_base * lapse * jnp.exp(-2.0 * phi)
    ko_diss = (state[:NUM_BSSN_VARS] @ deriv_mats.ko_matrix.T) / (64.0 * dr)
    rhs_state = rhs_state.at[:NUM_BSSN_VARS].add(sigma * ko_diss)

    # 10. Boundary conditions on the RHS (reuse parity/asymp arrays from step 0)
    rhs_state = fill_bssn_boundaries_jax(rhs_state, r, num_ghosts, parity, asymp_power, asymp_offset)

    return rhs_state
