# scalarmatter_jax.py
#
# JAX port of source/matter/scalarmatter.py
# Pure-functional scalar field matter for Klein-Gordon equation.

import jax.numpy as jnp

from source.bssn.jax.tensoralgebra_jax import (
    EMTensor, get_bar_gamma_LL, get_bar_gamma_UU, get_bar_christoffel,
    get_tensor_connections,
)


def V_of_u(u, scalar_mu):
    """Scalar potential V(u) = 0.5 * mu^2 * u^2."""
    return 0.5 * scalar_mu * scalar_mu * u * u


def dVdu(u, scalar_mu):
    """Derivative of scalar potential."""
    return scalar_mu * scalar_mu * u


def get_scalar_emtensor(u, v, d1_u, phi, h_LL, background, scalar_mu):
    """
    Compute energy-momentum tensor for scalar field.

    Args:
        u: (N,) scalar field
        v: (N,) scalar field velocity (~ time derivative)
        d1_u: (N, 3) first derivatives of u
        phi: (N,) conformal factor
        h_LL: (N, 3, 3) rescaled metric perturbation
        background: BSSNBackground pytree
        scalar_mu: float, scalar field mass parameter

    Returns:
        EMTensor namedtuple
    """
    r = background.r
    em4phi = jnp.exp(-4.0 * phi)
    bar_gamma_UU = get_bar_gamma_UU(r, h_LL, background)
    bar_gamma_LL = get_bar_gamma_LL(r, h_LL, background)

    # rho = 0.5*v^2 + 0.5*e^{-4phi} bar_gamma^{ij} d_i u d_j u + V(u)
    rho = (0.5 * v * v
           + 0.5 * em4phi * jnp.einsum('xij,xi,xj->x', bar_gamma_UU, d1_u, d1_u)
           + V_of_u(u, scalar_mu))

    # S_i = -v * d_i u
    Si = -v[:, jnp.newaxis] * d1_u

    # S_ij
    Vt = -v * v + em4phi * jnp.einsum('xij,xi,xj->x', bar_gamma_UU, d1_u, d1_u)
    scalar_factor = -((0.5 * Vt + V_of_u(u, scalar_mu)) / em4phi)
    Sij = (scalar_factor[:, jnp.newaxis, jnp.newaxis] * bar_gamma_LL
           + jnp.einsum('xi,xj->xij', d1_u, d1_u))

    # S = trace of S_ij
    S = em4phi * jnp.einsum('xjk,xjk->x', bar_gamma_UU, Sij)

    return EMTensor(rho=rho, Si=Si, Sij=Sij, S=S)


def get_scalar_rhs(u, v, d1_u, d2_u, advec_u, advec_v,
                   phi, d1_phi, h_LL, d1_h_LL, lapse, d1_lapse, K,
                   shift_U, background, scalar_mu):
    """
    Compute Klein-Gordon RHS for scalar field.

    Args:
        u, v: (N,) scalar field and velocity
        d1_u: (N, 3) first derivatives of u
        d2_u: (N, 3, 3) second derivatives of u
        advec_u, advec_v: (N, 3) advective derivatives
        phi, d1_phi: (N,), (N, 3) conformal factor and derivative
        h_LL, d1_h_LL: metric perturbation and derivative
        lapse, d1_lapse: (N,), (N, 3) lapse and derivative
        K: (N,) trace of extrinsic curvature
        shift_U: (N, 3) shift vector
        background: BSSNBackground pytree
        scalar_mu: float

    Returns:
        (dudt, dvdt): tuple of (N,) arrays
    """
    r = background.r

    Delta_U, Delta_ULL, Delta_LLL = get_tensor_connections(r, h_LL, d1_h_LL, background)
    bar_chris = get_bar_christoffel(r, Delta_ULL, background)

    em4phi = jnp.exp(-4.0 * phi)
    bar_gamma_UU = get_bar_gamma_UU(r, h_LL, background)

    dudt = lapse * v
    dvdt = (lapse * K * v
            + 2.0 * lapse * em4phi * jnp.einsum('xij,xi,xj->x', bar_gamma_UU, d1_phi, d1_u)
            + lapse * em4phi * jnp.einsum('xij,xij->x', bar_gamma_UU, d2_u)
            + em4phi * jnp.einsum('xij,xi,xj->x', bar_gamma_UU, d1_lapse, d1_u)
            - lapse * em4phi * jnp.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, d1_u))

    # Mass term
    dvdt = dvdt - lapse * dVdu(u, scalar_mu)

    # Advection
    dudt = dudt + jnp.einsum('xj,xj->x', background.inverse_scaling_vector * shift_U, advec_u)
    dvdt = dvdt + jnp.einsum('xj,xj->x', background.inverse_scaling_vector * shift_U, advec_v)

    return dudt, dvdt
