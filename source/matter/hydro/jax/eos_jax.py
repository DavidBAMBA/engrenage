"""
JAX-native equation of state implementations.

Pure-functional EOS: all functions take explicit parameters, no classes with state.
All functions are JIT-compilable and GPU-compatible.
"""

import jax
import jax.numpy as jnp
from functools import partial


# =============================================================================
# Ideal Gas EOS:  P = (gamma-1) rho0 eps
# =============================================================================

@jax.jit
def pressure_ideal(rho0, eps, gamma):
    """P = (gamma - 1) rho0 eps"""
    return (gamma - 1.0) * rho0 * eps


@jax.jit
def eps_from_rho_p_ideal(rho0, p, gamma):
    """eps = P / [(gamma - 1) rho0]"""
    return p / ((gamma - 1.0) * jnp.maximum(rho0, 1e-30))


@jax.jit
def enthalpy_ideal(rho0, p, eps, gamma):
    """h = 1 + eps + P/rho0"""
    return 1.0 + eps + p / jnp.maximum(rho0, 1e-30)


@jax.jit
def enthalpy_from_rho_p_ideal(rho0, p, gamma):
    """h from (rho0, P) only."""
    eps = eps_from_rho_p_ideal(rho0, p, gamma)
    return 1.0 + eps + p / jnp.maximum(rho0, 1e-30)


@jax.jit
def sound_speed_squared_ideal(rho0, p, eps, gamma):
    """c_s^2 = gamma P / (rho0 h), clipped to [0, 1]."""
    h = 1.0 + eps + p / jnp.maximum(rho0, 1e-30)
    cs2 = gamma * p / (jnp.maximum(rho0, 1e-30) * h)
    return jnp.clip(cs2, 0.0, 1.0)


# =============================================================================
# Polytropic EOS:  P = K rho0^gamma
# =============================================================================

@jax.jit
def pressure_polytropic(rho0, K, gamma):
    """P = K rho0^gamma"""
    return K * rho0**gamma


@jax.jit
def eps_from_rho_polytropic(rho0, K, gamma):
    """eps = K rho0^(gamma-1) / (gamma-1)"""
    return K * rho0**(gamma - 1.0) / (gamma - 1.0)


@jax.jit
def eps_from_rho_p_polytropic(rho0, p, K, gamma):
    """Interface-compatible: ignores p for barotropic EOS."""
    return eps_from_rho_polytropic(rho0, K, gamma)


@jax.jit
def enthalpy_polytropic(rho0, K, gamma):
    """h = 1 + gamma K rho0^(gamma-1) / (gamma-1)"""
    return 1.0 + gamma * K * rho0**(gamma - 1.0) / (gamma - 1.0)


@jax.jit
def sound_speed_squared_polytropic(rho0, K, gamma):
    """c_s^2 = gamma K rho0^(gamma-1) / h, clipped to [0, 1]."""
    h = enthalpy_polytropic(rho0, K, gamma)
    cs2 = gamma * K * rho0**(gamma - 1.0) / h
    return jnp.clip(cs2, 0.0, 1.0)


# =============================================================================
# Dispatch helpers: call correct EOS based on type string
# =============================================================================

def get_eos_functions(eos):
    """
    Extract JAX-compatible EOS closures from a NumPy EOS object.

    Returns a dict of pure functions that close over EOS parameters.
    All returned functions are JIT-compilable.

    Args:
        eos: NumPy EOS object (IdealGasEOS or PolytropicEOS)

    Returns:
        dict with keys: 'pressure', 'eps_from_rho_p', 'enthalpy', 'sound_speed_squared',
                        'type', and EOS parameters
    """
    if hasattr(eos, 'name') and eos.name.startswith('ideal_gas'):
        gamma = float(eos.gamma)
        return {
            'type': 'ideal_gas',
            'gamma': gamma,
            'pressure': lambda rho0, eps: pressure_ideal(rho0, eps, gamma),
            'eps_from_rho_p': lambda rho0, p: eps_from_rho_p_ideal(rho0, p, gamma),
            'enthalpy_from_rho_p': lambda rho0, p: enthalpy_from_rho_p_ideal(rho0, p, gamma),
            'sound_speed_squared': lambda rho0, p, eps: sound_speed_squared_ideal(rho0, p, eps, gamma),
        }
    elif hasattr(eos, 'name') and eos.name.startswith('polytropic'):
        K = float(eos.K)
        gamma = float(eos.gamma)
        return {
            'type': 'polytropic',
            'K': K,
            'gamma': gamma,
            'pressure': lambda rho0, eps=None: pressure_polytropic(rho0, K, gamma),
            'eps_from_rho_p': lambda rho0, p: eps_from_rho_p_polytropic(rho0, p, K, gamma),
            'enthalpy_from_rho_p': lambda rho0, p: enthalpy_polytropic(rho0, K, gamma),
            'sound_speed_squared': lambda rho0, p=None, eps=None: sound_speed_squared_polytropic(rho0, K, gamma),
        }
    else:
        raise ValueError(f"Unsupported EOS type for JAX backend: {type(eos)}")


# =============================================================================
# prim_to_cons (JAX version)
# =============================================================================

@partial(jax.jit, static_argnums=(5, 6, 7))
def prim_to_cons_jax(rho0, vr, pressure, gamma_rr, e6phi,
                     eos_type, eos_gamma, eos_K=0.0):
    """
    Convert primitive to conservative variables (JAX, densitized).

    D_tilde  = e^{6phi} rho0 W
    Sr_tilde = e^{6phi} rho0 h W^2 vr gamma_rr
    tau_tilde = e^{6phi} (rho0 h W^2 - P - rho0 W)

    Args:
        rho0, vr, pressure: Primitive variables (arrays)
        gamma_rr, e6phi: Metric components
        eos_type: 'ideal_gas' or 'polytropic' (static)
        eos_gamma: Adiabatic index (static)
        eos_K: Polytropic constant (static, only for polytropic)

    Returns:
        D, Sr, tau: Conservative variables (densitized)
    """
    v2 = gamma_rr * vr * vr
    v2 = jnp.clip(v2, 0.0, 1.0 - 1e-12)
    W = 1.0 / jnp.sqrt(1.0 - v2)
    v_i = vr * gamma_rr

    # EOS
    if eos_type == 'ideal_gas':
        eps = pressure / ((eos_gamma - 1.0) * jnp.maximum(rho0, 1e-30))
    else:  # polytropic
        eps = eos_K * rho0**(eos_gamma - 1.0) / (eos_gamma - 1.0)

    h = 1.0 + eps + pressure / jnp.maximum(rho0, 1e-30)

    D = e6phi * rho0 * W
    Sr = e6phi * rho0 * h * W * W * v_i
    tau = e6phi * (rho0 * h * W * W - pressure - rho0 * W)

    return D, Sr, tau
