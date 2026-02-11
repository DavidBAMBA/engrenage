# bssnrhs_jax.py
#
# JAX port of source/bssn/bssnrhs.py
# Pure-functional BSSN RHS — returns all time derivatives as a tuple.
#
# References:
#   Etienne https://arxiv.org/abs/1712.07658v2
#   Baumgarte https://arxiv.org/abs/1211.6632

import jax.numpy as jnp

from source.bssn.jax.tensoralgebra_jax import (
    SPACEDIM, one_sixth, one_third, two_thirds, four_thirds, eight_pi_G,
    get_bar_gamma_LL, get_bar_gamma_UU, get_rescaled_bar_gamma_LL,
    get_rescaled_bar_gamma_UU,
    get_bar_A_LL, get_bar_A_UU, get_trace_bar_A, get_bar_A_squared,
    get_tensor_connections, get_bar_christoffel,
    get_bar_ricci_tensor, get_trace,
)


def get_bssn_rhs_jax(r, phi, h_LL, K, a_LL, lambda_U, lapse, shift_U, b_U,
                      d1_phi, d1_h_LL, d1_K, d1_a_LL, d1_lambda_U, d1_lapse, d1_shift_U,
                      d2_phi, d2_h_LL, d2_lapse, d2_shift_U,
                      background, emtensor):
    """
    Compute BSSN RHS (pure functional, no in-place mutation).

    All tensor arguments have shape (N, ...) where N is the number of grid points.
    d1 = first derivatives (last index is derivative direction)
    d2 = second derivatives (last two indices are derivative directions)

    Returns:
        (dphidt, dhdt, dKdt, dadt, dlambdadt): tuple of RHS arrays
    """
    N = r.shape[0]

    ####################################################################################################
    # Useful quantities
    em4phi = jnp.exp(-4.0 * phi)
    bar_gamma_LL = get_bar_gamma_LL(r, h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, h_LL, background)

    # Tensor connections
    Delta_U, Delta_ULL, Delta_LLL = get_tensor_connections(r, h_LL, d1_h_LL, background)
    bar_chris = get_bar_christoffel(r, Delta_ULL, background)

    # Rescaled shift
    Shift_U = background.inverse_scaling_vector * shift_U
    d1_Shift_U = (background.d1_inverse_scaling_vector * shift_U[:, :, jnp.newaxis]
                  + d1_shift_U * background.inverse_scaling_vector[:, :, jnp.newaxis])
    d2_Shift_U = (jnp.einsum('xijk,xi->xijk', background.d2_inverse_scaling_vector, shift_U)
                  + jnp.einsum('xik,xij->xijk', background.d1_inverse_scaling_vector, d1_shift_U)
                  + jnp.einsum('xij,xik->xijk', background.d1_inverse_scaling_vector, d1_shift_U)
                  + jnp.einsum('xi,xijk->xijk', background.inverse_scaling_vector, d2_shift_U))

    # Conformal divergence of shift
    bar_div_shift = jnp.einsum('xii->x', d1_Shift_U)
    bar_div_shift = bar_div_shift + jnp.einsum('xiij,xj->x', bar_chris, Shift_U)

    # Trace and squared of bar A
    trace_bar_A = get_trace_bar_A(r, h_LL, a_LL, background)
    bar_A_squared = get_bar_A_squared(r, h_LL, a_LL, background)
    bar_A_LL = get_bar_A_LL(r, a_LL, background.scaling_matrix)
    bar_A_UU = get_bar_A_UU(r, h_LL, a_LL, background.scaling_matrix, background)

    ####################################################################################################
    # 1. Conformal factor phi
    dphidt = (-one_sixth * lapse * K
              + one_sixth * bar_div_shift)

    ####################################################################################################
    # 2. Rescaled metric perturbation h_ij

    # hat_gamma_jk * hat_D_i shift^k (not hat_D_i beta_j!)
    hat_D_shift_U = (
        jnp.einsum('xjk,xki->xij', background.hat_gamma_LL, d1_Shift_U)
        + jnp.einsum('xjk,xkil,xl->xij', background.hat_gamma_LL, background.hat_christoffel, Shift_U)
    )

    # Rescale
    r_hat_D_shift_U = background.inverse_scaling_matrix * hat_D_shift_U
    r_bar_gamma_LL = get_rescaled_bar_gamma_LL(r, h_LL, background)

    scalar_factor = two_thirds * (lapse * trace_bar_A - bar_div_shift)

    dhdt = (scalar_factor[:, jnp.newaxis, jnp.newaxis] * r_bar_gamma_LL
            - 2.0 * lapse[:, jnp.newaxis, jnp.newaxis] * a_LL
            + r_hat_D_shift_U + jnp.transpose(r_hat_D_shift_U, axes=(0, 2, 1)))

    ####################################################################################################
    # 3. Trace of extrinsic curvature K

    bar_D2_lapse = (jnp.einsum('xij,xij->x', bar_gamma_UU, d2_lapse)
                    - jnp.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, d1_lapse))

    dKdt = (lapse * (one_third * K * K
                     + bar_A_squared + 0.5 * eight_pi_G * (emtensor.rho + emtensor.S))
            - em4phi * (bar_D2_lapse
                        + 2.0 * jnp.einsum('xij,xi,xj->x', bar_gamma_UU, d1_lapse, d1_phi)))

    ####################################################################################################
    # 4. Rescaled traceless extrinsic curvature a_ij

    bar_Rij = get_bar_ricci_tensor(r, h_LL, d1_h_LL, d2_h_LL, lambda_U, d1_lambda_U,
                                    Delta_U, Delta_ULL, Delta_LLL,
                                    bar_gamma_UU, bar_gamma_LL, background)

    AikAkj = jnp.einsum('xkl,xik,xlj->xij', bar_gamma_UU, bar_A_LL, bar_A_LL)

    dAdt_TF_part = (lapse[:, jnp.newaxis, jnp.newaxis] *
                    (-2.0 * d2_phi
                     + 4.0 * jnp.einsum('xi,xj->xij', d1_phi, d1_phi)
                     + 2.0 * jnp.einsum('xkij,xk->xij', bar_chris, d1_phi)
                     + bar_Rij - eight_pi_G * emtensor.Sij)
                    - d2_lapse
                    + jnp.einsum('xkij,xk->xij', bar_chris, d1_lapse)
                    + 2.0 * jnp.einsum('xi,xj->xij', d1_phi, d1_lapse)
                    + 2.0 * jnp.einsum('xj,xi->xij', d1_phi, d1_lapse))

    trace = get_trace(dAdt_TF_part, bar_gamma_UU)

    # Rescale
    dadt_TF_part = background.inverse_scaling_matrix * dAdt_TF_part
    r_AikAkj = background.inverse_scaling_matrix * AikAkj
    r_bar_gamma_LL_a = get_rescaled_bar_gamma_LL(r, h_LL, background)

    dadt = (-two_thirds * bar_div_shift[:, jnp.newaxis, jnp.newaxis] * a_LL
            + lapse[:, jnp.newaxis, jnp.newaxis] * (-2.0 * r_AikAkj
                                                     + K[:, jnp.newaxis, jnp.newaxis] * a_LL)
            + em4phi[:, jnp.newaxis, jnp.newaxis] * (dadt_TF_part
                                                      - one_third * trace[:, jnp.newaxis, jnp.newaxis] * r_bar_gamma_LL_a))

    ####################################################################################################
    # 5. Conformal connection lambda^i

    # bar_gamma^jk hat_D_j hat_D_k shift^i
    hat_D2_shift = (
        jnp.einsum('xjk,xijk->xi', bar_gamma_UU, d2_Shift_U)
        + jnp.einsum('xjk,xikl,xlj->xi', bar_gamma_UU, background.hat_christoffel, d1_Shift_U)
        + jnp.einsum('xjk,xijl,xlk->xi', bar_gamma_UU, background.hat_christoffel, d1_Shift_U)
        - jnp.einsum('xjk,xljk,xil->xi', bar_gamma_UU, background.hat_christoffel, d1_Shift_U)
        + jnp.einsum('xjk,xiklj,xl->xi', bar_gamma_UU, background.d1_hat_christoffel, Shift_U)
        + jnp.einsum('xjk,xijl,xlkm,xm->xi', bar_gamma_UU, background.hat_christoffel,
                      background.hat_christoffel, Shift_U)
        - jnp.einsum('xjk,xljk,xilm,xm->xi', bar_gamma_UU, background.hat_christoffel,
                      background.hat_christoffel, Shift_U)
    )

    # bar_D^i (bar_D_j beta^j)
    # Guard against det_hat_gamma=0 at ghost cells near r=0
    safe_det = jnp.maximum(background.det_hat_gamma, 1e-300)
    bar_D_div_shift = (
        jnp.einsum('xij,xkjk->xi', bar_gamma_UU, d2_Shift_U)
        + (0.5 / safe_det[:, jnp.newaxis]
           * jnp.einsum('xij,xkj,xk->xi', bar_gamma_UU, d1_Shift_U, background.d1_det_hat_gamma))
        + (0.5 / safe_det[:, jnp.newaxis]
           * jnp.einsum('xij,xjk,xk->xi', bar_gamma_UU, background.d2_det_hat_gamma, Shift_U))
        - (0.5 / safe_det[:, jnp.newaxis] / safe_det[:, jnp.newaxis]
           * jnp.einsum('xij,xj,xk,xk->xi', bar_gamma_UU, background.d1_det_hat_gamma,
                         background.d1_det_hat_gamma, Shift_U))
    )

    dlambdadt = (hat_D2_shift
                 + two_thirds * Delta_U * bar_div_shift[:, jnp.newaxis]
                 + one_third * bar_D_div_shift
                 - 2.0 * jnp.einsum('xij,xj->xi', bar_A_UU, d1_lapse)
                 + 12.0 * lapse[:, jnp.newaxis] * jnp.einsum('xij,xj->xi', bar_A_UU, d1_phi)
                 + 2.0 * lapse[:, jnp.newaxis] * jnp.einsum('xjk,xijk->xi', bar_A_UU, Delta_ULL)
                 - four_thirds * lapse[:, jnp.newaxis] * jnp.einsum('xij,xj->xi', bar_gamma_UU, d1_K)
                 - 2.0 * eight_pi_G * lapse[:, jnp.newaxis] * jnp.einsum('xij,xj->xi', bar_gamma_UU, emtensor.Si))

    # Rescale because we want change in lambda not Lambda
    dlambdadt = dlambdadt * background.scaling_vector

    return dphidt, dhdt, dKdt, dadt, dlambdadt
