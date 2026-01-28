"""
JAX-native atmosphere and floor management.

All functions are pure-functional: no in-place mutation, returns new arrays.
Fully JIT-compilable for GPU execution.
"""

import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
def apply_primitive_floors(rho0, vr, p, gamma_rr,
                           rho_floor, p_floor, v_max):
    """
    Apply floors to primitive variables (functional, no mutation).

    Args:
        rho0, vr, p: Primitive variables
        gamma_rr: Radial metric component
        rho_floor, p_floor, v_max: Floor parameters

    Returns:
        rho0_f, vr_f, p_f: Floored primitives
    """
    rho0_f = jnp.maximum(rho0, rho_floor)
    p_f = jnp.maximum(p, p_floor)

    # Velocity limit
    v2 = gamma_rr * vr * vr
    v_scale = v_max / jnp.sqrt(jnp.maximum(gamma_rr, 1e-30))
    vr_f = jnp.where(v2 >= v_max**2,
                     jnp.sign(vr) * v_scale,
                     vr)

    return rho0_f, vr_f, p_f


@jax.jit
def apply_conservative_floors(D, Sr, tau, gamma_rr,
                              tau_atm, safety_factor):
    """
    Apply conservative variable consistency floors (IllinoisGRMHD).

    1. tau >= tau_atm
    2. S^2 <= safety * tau*(tau + 2D)

    Args:
        D, Sr, tau: Conservative variables (physical, non-densitized)
        gamma_rr: Radial metric component
        tau_atm: Atmosphere tau value
        safety_factor: S^2 constraint safety (0.999999)

    Returns:
        D_f, Sr_f, tau_f: Floored conservative variables
    """
    # Tau floor
    tau_f = jnp.maximum(tau, tau_atm)

    # S^2 constraint
    S2 = Sr**2 / jnp.maximum(gamma_rr, 1e-30)
    RHS = safety_factor * tau_f * (tau_f + 2.0 * D)
    RHS = jnp.maximum(RHS, 0.0)

    rescale = jnp.sqrt(RHS / jnp.maximum(S2, 1e-30))
    Sr_f = jnp.where(S2 > RHS, Sr * rescale, Sr)

    return D, Sr_f, tau_f


@jax.jit
def apply_atmosphere_fallback(rho0, vr, p, eps, W, h, mask,
                              rho_floor, p_floor, eps_atm, h_atm):
    """
    Set atmosphere values for specified points (functional).

    Args:
        rho0, vr, p, eps, W, h: Primitive arrays
        mask: Boolean mask of points to set to atmosphere
        rho_floor, p_floor, eps_atm, h_atm: Atmosphere values

    Returns:
        rho0, vr, p, eps, W, h: Updated arrays (new copies)
    """
    rho0 = jnp.where(mask, rho_floor, rho0)
    vr = jnp.where(mask, 0.0, vr)
    p = jnp.where(mask, p_floor, p)
    eps = jnp.where(mask, eps_atm, eps)
    W = jnp.where(mask, 1.0, W)
    h = jnp.where(mask, h_atm, h)
    return rho0, vr, p, eps, W, h


@jax.jit
def detect_atmosphere(D, e6phi, rho_floor, buffer_factor=100.0):
    """
    Detect atmosphere points from conservative density.

    Uses the same criterion as the Numba version:
    D < buffer_factor * rho_floor * e6phi

    Args:
        D: Conservative density (densitized)
        e6phi: Conformal factor e^{6phi}
        rho_floor: Atmosphere density
        buffer_factor: Safety buffer (default 100)

    Returns:
        atm_mask: Boolean mask (True = atmosphere)
    """
    return D < buffer_factor * rho_floor * e6phi
