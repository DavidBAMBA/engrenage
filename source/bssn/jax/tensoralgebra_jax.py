# tensoralgebra_jax.py
#
# JAX port of source/bssn/tensoralgebra.py
# Pure-functional tensor algebra for BSSN evolution.
# All functions use jax.numpy — no in-place mutations.

import jax.numpy as jnp
from typing import NamedTuple

SPACEDIM: int = 3

one_sixth = 1.0 / 6.0
one_third = 1.0 / 3.0
two_thirds = 2.0 / 3.0
four_thirds = 4.0 / 3.0

eight_pi_G = 8.0 * jnp.pi * 1.0

delta_ij = jnp.eye(SPACEDIM)


class EMTensor(NamedTuple):
    """Immutable energy-momentum tensor components."""
    rho: jnp.ndarray   # (N,)
    Si: jnp.ndarray    # (N, 3)
    Sij: jnp.ndarray   # (N, 3, 3)
    S: jnp.ndarray     # (N,)


def get_trace(T_LL, gamma_UU):
    return jnp.einsum('xij,xij->x', gamma_UU, T_LL)


def get_bar_A_LL(r, a_LL, scaling_matrix):
    return scaling_matrix * a_LL


def get_bar_A_UU(r, h_LL, a_LL, scaling_matrix, background):
    bar_gamma_UU = get_bar_gamma_UU(r, h_LL, background)
    bar_A_LL = get_bar_A_LL(r, a_LL, scaling_matrix)
    return bar_gamma_UU @ bar_A_LL @ bar_gamma_UU


def get_a_UU(r, h_LL, a_LL, background):
    r_bar_gamma_UU = get_rescaled_bar_gamma_UU(r, h_LL, background)
    return r_bar_gamma_UU @ a_LL @ r_bar_gamma_UU


def get_trace_bar_A(r, h_LL, a_LL, background):
    r_bar_gamma_UU = get_rescaled_bar_gamma_UU(r, h_LL, background)
    return get_trace(a_LL, r_bar_gamma_UU)


def get_bar_A_squared(r, h_LL, a_LL, background):
    r_bar_gamma_UU = get_rescaled_bar_gamma_UU(r, h_LL, background)
    a_UU = r_bar_gamma_UU @ a_LL @ r_bar_gamma_UU
    return jnp.einsum('xij,xij->x', a_LL, a_UU)


def get_det_bar_gamma(r, h_LL, background):
    epsilon_LL = h_LL * background.scaling_matrix
    return jnp.linalg.det(epsilon_LL + background.hat_gamma_LL)


def get_bar_gamma_LL(r, h_LL, background):
    epsilon_LL = h_LL * background.scaling_matrix
    return epsilon_LL + background.hat_gamma_LL


def get_rescaled_bar_gamma_LL(r, h_LL, background):
    return h_LL + background.hat_gamma_LL * background.inverse_scaling_matrix


def get_bar_gamma_UU(r, h_LL, background):
    bar_gamma_LL = get_bar_gamma_LL(r, h_LL, background)
    return jnp.linalg.inv(bar_gamma_LL)


def get_rescaled_bar_gamma_UU(r, h_LL, background):
    r_bar_gamma_LL = get_rescaled_bar_gamma_LL(r, h_LL, background)
    return jnp.linalg.inv(r_bar_gamma_LL)


def get_vector_advection(r, V_U, advec_V_U, shift_U, d1_shift_U, background):
    result = (jnp.einsum('xij,xj->xi', advec_V_U,
                         background.inverse_scaling_vector * shift_U)
              + jnp.einsum('xij,xj->xi',
                           background.scaling_vector[:, :, jnp.newaxis] * V_U[:, :, jnp.newaxis]
                           * background.d1_inverse_scaling_vector,
                           background.inverse_scaling_vector * shift_U)
              - jnp.einsum('xij,xj->xi', d1_shift_U,
                           background.inverse_scaling_vector * V_U)
              - jnp.einsum('xij,xj->xi',
                           background.scaling_vector[:, :, jnp.newaxis] * shift_U[:, :, jnp.newaxis]
                           * background.d1_inverse_scaling_vector,
                           background.inverse_scaling_vector * V_U))
    return result


def get_tensor_advection(r, A_LL, advec_A_LL, shift_U, d1_shift_U, background):
    result = (jnp.einsum('xijk,xk->xij', advec_A_LL,
                         background.inverse_scaling_vector * shift_U)
              + jnp.einsum('xijk,xk->xij',
                           background.inverse_scaling_matrix[:, :, :, jnp.newaxis] * A_LL[:, :, :, jnp.newaxis]
                           * background.d1_scaling_matrix,
                           background.inverse_scaling_vector * shift_U)
              + jnp.einsum('xik,xkj->xij', A_LL,
                           background.inverse_scaling_vector[:, jnp.newaxis, :] * d1_shift_U)
              + jnp.einsum('xjk,xki->xij', A_LL,
                           background.inverse_scaling_vector[:, jnp.newaxis, :] * d1_shift_U)
              + jnp.einsum('xik,xjk->xij',
                           A_LL * background.scaling_vector[:, jnp.newaxis, :],
                           background.inverse_scaling_vector[:, :, jnp.newaxis] * shift_U[:, jnp.newaxis, :]
                           * background.d1_inverse_scaling_vector)
              + jnp.einsum('xjk,xik->xij',
                           A_LL * background.scaling_vector[:, jnp.newaxis, :],
                           background.inverse_scaling_vector[:, :, jnp.newaxis] * shift_U[:, jnp.newaxis, :]
                           * background.d1_inverse_scaling_vector))
    return result


def get_bar_christoffel(r, Delta_ULL, background):
    return background.hat_christoffel + Delta_ULL


def get_tensor_connections(r, h_LL, d1_h_dx, background):
    bar_gamma_LL = get_bar_gamma_LL(r, h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, h_LL, background)

    hat_D_bar_gamma = get_hat_D_bar_gamma_LL(r, h_LL, d1_h_dx, background)

    Delta_ULL = (0.5 * jnp.einsum('xil,xklj->xijk', bar_gamma_UU, hat_D_bar_gamma)
                 + 0.5 * jnp.einsum('xil,xjlk->xijk', bar_gamma_UU, hat_D_bar_gamma)
                 - 0.5 * jnp.einsum('xil,xjkl->xijk', bar_gamma_UU, hat_D_bar_gamma))

    Delta_U = jnp.einsum('xjk,xijk->xi', bar_gamma_UU, Delta_ULL)
    Delta_LLL = jnp.einsum('xil,xljk->xijk', bar_gamma_LL, Delta_ULL)

    return Delta_U, Delta_ULL, Delta_LLL


def get_bar_ricci_tensor(r, h_LL, d1_h_dx, d2_h_dxdy, lambda_U, d1_lambda_dx,
                         Delta_U, Delta_ULL, Delta_LLL,
                         bar_gamma_UU, bar_gamma_LL, background):
    hat_D_bar_Lambda_U = get_hat_D_bar_Lambda_U(r, lambda_U, d1_lambda_dx, background)
    hat_D2_bar_gamma_LL = get_hat_D2_bar_gamma_LL(r, h_LL, d1_h_dx, d2_h_dxdy, background)

    bar_ricci = (- 0.5 * hat_D2_bar_gamma_LL
                 + 0.5 * jnp.einsum('xki,xkj->xij', bar_gamma_LL, hat_D_bar_Lambda_U)
                 + 0.5 * jnp.einsum('xkj,xki->xij', bar_gamma_LL, hat_D_bar_Lambda_U)
                 + 0.5 * jnp.einsum('xk,xijk->xij', Delta_U, Delta_LLL)
                 + 0.5 * jnp.einsum('xk,xjik->xij', Delta_U, Delta_LLL)
                 + jnp.einsum('xkl,xmki,xjml->xij', bar_gamma_UU, Delta_ULL, Delta_LLL)
                 + jnp.einsum('xkl,xmkj,ximl->xij', bar_gamma_UU, Delta_ULL, Delta_LLL)
                 + jnp.einsum('xkl,xmik,xmjl->xij', bar_gamma_UU, Delta_ULL, Delta_LLL))

    return bar_ricci


def get_hat_D_bar_Lambda_U(r, lambda_U, d1_lambda_dx, background):
    N = r.shape[0]
    Lambda_U = background.inverse_scaling_vector * lambda_U

    hat_D_Lambda = (d1_lambda_dx * background.inverse_scaling_vector[:, :, jnp.newaxis]
                    + background.d1_inverse_scaling_vector * lambda_U[:, :, jnp.newaxis]
                    + jnp.einsum('xijk,xk->xij', background.hat_christoffel, Lambda_U))

    return hat_D_Lambda


def get_hat_D2_bar_gamma_LL(r, h_LL, d1_h_dx, d2_h_dxdy, background):
    N = r.shape[0]

    bar_gamma_LL = get_bar_gamma_LL(r, h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, h_LL, background)
    hat_D_bar_gamma_LL_dx = get_hat_D_bar_gamma_LL(r, h_LL, d1_h_dx, background)
    hat_chris = background.hat_christoffel
    d1_hat_chris_dx = background.d1_hat_christoffel
    d1_m_dx = background.d1_scaling_matrix
    d2_m_dxdy = background.d2_scaling_matrix
    epsilon_LL = h_LL * background.scaling_matrix

    d1_epsilon_LL_dx = (d1_h_dx * background.scaling_matrix[:, :, :, jnp.newaxis]
                        + d1_m_dx * h_LL[:, :, :, jnp.newaxis])

    # dm_dxk * dh_dxl — build (N, 3, 3, 3, 3) tensor
    # Using einsum to avoid explicit loops
    dm_dxk_dh_dxl = jnp.einsum('xijk,xijl->xijkl', d1_m_dx, d1_h_dx)

    # First term in rhs of (27)
    hat_D2_bar_gamma_LL = (
        jnp.einsum('xkl,xijkl->xij', bar_gamma_UU, d2_h_dxdy) * background.scaling_matrix
        + jnp.einsum('xkl,xijkl->xij', bar_gamma_UU, d2_m_dxdy) * h_LL
        + 2.0 * jnp.einsum('xkl,xijkl->xij', bar_gamma_UU, dm_dxk_dh_dxl)
        - jnp.einsum('xkl,xmlik,xmj->xij', bar_gamma_UU, d1_hat_chris_dx, epsilon_LL)
        - jnp.einsum('xkl,xmljk,xim->xij', bar_gamma_UU, d1_hat_chris_dx, epsilon_LL)
        - jnp.einsum('xkl,xmli,xmjk->xij', bar_gamma_UU, hat_chris, d1_epsilon_LL_dx)
        - jnp.einsum('xkl,xmlj,ximk->xij', bar_gamma_UU, hat_chris, d1_epsilon_LL_dx))

    # Christoffel terms
    hat_D2_bar_gamma_LL = hat_D2_bar_gamma_LL + (
        - jnp.einsum('xkl,xijm,xmlk->xij', bar_gamma_UU, hat_D_bar_gamma_LL_dx, hat_chris)
        - jnp.einsum('xkl,xmjl,xmik->xij', bar_gamma_UU, hat_D_bar_gamma_LL_dx, hat_chris)
        - jnp.einsum('xkl,ximl,xmjk->xij', bar_gamma_UU, hat_D_bar_gamma_LL_dx, hat_chris))

    return hat_D2_bar_gamma_LL


def get_hat_D_bar_gamma_LL(r, h_LL, d1_h_dx, background):
    epsilon_LL = h_LL * background.scaling_matrix

    hat_D_epsilon = (d1_h_dx * background.scaling_matrix[:, :, :, jnp.newaxis]
                     + background.d1_scaling_matrix * h_LL[:, :, :, jnp.newaxis]
                     - jnp.einsum('xlik,xlj->xijk', background.hat_christoffel, epsilon_LL)
                     - jnp.einsum('xljk,xil->xijk', background.hat_christoffel, epsilon_LL))

    return hat_D_epsilon


def get_bar_div_shift(r, h_LL, d1_h_LL, shift_U, d1_shift_U, background):
    Delta_U, Delta_ULL, Delta_LLL = get_tensor_connections(r, h_LL, d1_h_LL, background)
    bar_chris = get_bar_christoffel(r, Delta_ULL, background)

    Shift_U = background.inverse_scaling_vector * shift_U
    d1_Shift_U = (background.d1_inverse_scaling_vector * shift_U[:, :, jnp.newaxis]
                  + d1_shift_U * background.inverse_scaling_vector[:, :, jnp.newaxis])

    bar_div_shift = jnp.einsum('xii->x', d1_Shift_U)
    bar_div_shift = bar_div_shift + jnp.einsum('xiij,xj->x', bar_chris, Shift_U)

    return bar_div_shift
