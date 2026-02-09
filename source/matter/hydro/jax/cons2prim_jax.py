"""
JAX-based conservative to primitive variable converter.

This module provides a JAX implementation of the cons2prim solver that can run on
GPU. It mirrors the functionality of cons2prim.py but uses JAX for acceleration.

Usage:
    Set ENGRENAGE_BACKEND=jax before importing, or use directly:

    from source.matter.hydro.cons2prim_jax import Cons2PrimSolverJAX
    solver = Cons2PrimSolverJAX(eos, atmosphere)
    result = solver.convert(D, Sr, tau, gamma_rr)
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
import numpy as np

from source.matter.hydro.atmosphere import FloorApplicator
from source.matter.hydro.geometry import GeometryState
from source.matter.hydro.cons2prim import prim_to_cons


# ============================================================================
# JAX-OPTIMIZED NEWTON-RAPHSON KERNEL FOR IDEAL GAS EOS
# ============================================================================

@partial(jit, static_argnums=(6, 7, 8, 9, 10))
def _newton_kernel_jax(D, Sr, tau, gamma_rr, alpha, p_init,
                       eos_gamma, p_floor, W_max, tol, max_iter):
    """
    JAX Newton-Raphson solver for ideal gas EOS (single point).

    Uses jax.lax.while_loop for convergence iteration.

    Args:
        D, Sr, tau: Conservative variables (scalars)
        gamma_rr: Radial metric component
        alpha: Lapse function
        p_init: Initial pressure guess
        eos_gamma: Adiabatic index (static)
        p_floor: Pressure floor (static)
        W_max: Maximum Lorentz factor (static)
        tol: Convergence tolerance (static)
        max_iter: Maximum iterations (static)

    Returns:
        tuple: (rho0, vr, p, eps, W, h, converged)
    """
    c_ideal = eos_gamma / (eos_gamma - 1.0)
    gm1 = eos_gamma - 1.0

    p_start = jnp.maximum(p_init, p_floor)
    gamma_rr = jnp.maximum(gamma_rr, 1e-30)
    alpha_safe = jnp.maximum(alpha, 1e-30)

    # State for while loop: (iteration, p, converged)
    def cond_fn(state):
        iteration, p, converged = state
        return (iteration < max_iter) & (~converged)

    def body_fn(state):
        iteration, p, converged = state

        # Evaluate pressure residual
        Q = tau + D + p
        vr = Sr / (Q * gamma_rr)
        v2 = gamma_rr * vr * vr
        v2 = jnp.clip(v2, 0.0, 1.0 - 1e-12)
        W = 1.0 / jnp.sqrt(1.0 - v2)

        rho0 = D / jnp.maximum(W, 1e-30)
        eps = p / (rho0 * gm1 + 1e-30)
        h = 1.0 + eps + p / (rho0 + 1e-30)

        # Residual: f(p) = rho*h*W^2 - Q
        f = rho0 * h * W * W - Q

        # Check convergence
        is_converged = jnp.abs(f) <= tol * jnp.maximum(1.0, jnp.abs(p))

        # Analytic derivative df/dp
        E = tau + D
        Sr_sq = Sr * Sr
        Qp = E + p
        v2_d = Sr_sq / jnp.maximum(Qp**2 * gamma_rr, 1e-30)
        W_d = 1.0 / jnp.maximum(1.0 - v2_d, 1e-16) ** 0.5
        Wprime = -Sr_sq * W_d**3 / jnp.maximum(Qp**3 * gamma_rr, 1e-30)
        df = c_ideal * W_d**2 + (D + 2.0 * c_ideal * p * W_d) * Wprime - 1.0

        # Standard Newton update: p_{n+1} = p_n - f/f'
        p_new = p - f / df
        p_new = jnp.maximum(p_new, p_floor)

        return (iteration + 1, p_new, is_converged)

    # Run Newton iterations
    init_state = (0, p_start, False)
    final_iter, final_p, converged = lax.while_loop(cond_fn, body_fn, init_state)

    # Compute final primitives
    p = final_p
    Q = tau + D + p
    vr = Sr / (Q * gamma_rr)
    v2 = gamma_rr * vr * vr
    v2 = jnp.clip(v2, 0.0, 1.0 - 1e-16)
    W = 1.0 / jnp.sqrt(1.0 - v2)
    rho0 = D / jnp.maximum(W, 1e-30)
    eps = p / jnp.maximum(rho0 * gm1, 1e-30)
    h = 1.0 + eps + p / jnp.maximum(rho0, 1e-30)

    # Recompute convergence flag with FINAL values
    # (the loop flag may be stale if convergence happened on the last iteration)
    f_final = rho0 * h * W * W - Q
    tol_final = tol * jnp.maximum(1.0, jnp.abs(p))
    W_valid = (W <= W_max) & (W >= 1.0)
    rho0_valid = rho0 > 0.0
    converged_final = (jnp.abs(f_final) <= tol_final) & W_valid & rho0_valid

    return rho0, vr, p, eps, W, h, converged_final


# Vectorized version using vmap
@partial(jit, static_argnums=(6, 7, 8, 9, 10))
def _solve_newton_batch_jax(D, Sr, tau, gamma_rr, alpha, p_guess,
                            eos_gamma, p_floor, W_max, tol, max_iter):
    """
    JAX batch Newton-Raphson solver for ideal gas EOS.

    Vectorized over all points using vmap - runs in parallel on GPU.

    Args:
        D, Sr, tau: Conservative variable arrays (N,)
        gamma_rr: Metric component array (N,)
        alpha: Lapse function array (N,)
        p_guess: Initial pressure guess array (N,)
        eos_gamma: Adiabatic index (static)
        p_floor: Pressure floor (static)
        W_max: Maximum Lorentz factor (static)
        tol: Convergence tolerance (static)
        max_iter: Maximum iterations (static)

    Returns:
        tuple: (rho0, vr, p, eps, W, h, converged) - all arrays (N,)
    """
    # vmap over all input arrays (first 6 args), keep static args fixed
    newton_vmapped = vmap(
        lambda d, sr, t, grr, a, pg: _newton_kernel_jax(
            d, sr, t, grr, a, pg, eos_gamma, p_floor, W_max, tol, max_iter
        )
    )

    return newton_vmapped(D, Sr, tau, gamma_rr, alpha, p_guess)


# ============================================================================
# JAX-OPTIMIZED NEWTON-RAPHSON KERNEL FOR POLYTROPIC EOS
# ============================================================================

@partial(jit, static_argnums=(6, 7, 8, 9, 10, 11))
def _newton_kernel_polytropic_jax(D, Sr, tau, gamma_rr, alpha, rho_init,
                                   eos_K, eos_gamma, rho_floor, W_max, tol, max_iter):
    """
    JAX Newton-Raphson solver for polytropic EOS (single point).

    For barotropic EOS P = K ρ₀^Γ, solve for ρ₀ such that D = ρ₀ W.

    Returns:
        tuple: (rho0, vr, p, eps, W, h, converged)
    """
    gm1 = eos_gamma - 1.0
    K = eos_K

    rho_start = jnp.maximum(rho_init, rho_floor)
    gamma_rr = jnp.maximum(gamma_rr, 1e-30)
    alpha_safe = jnp.maximum(alpha, 1e-30)

    # State for while loop: (iteration, rho, converged)
    def cond_fn(state):
        iteration, rho, converged = state
        return (iteration < max_iter) & (~converged)

    def body_fn(state):
        iteration, rho, converged = state

        # Pressure and enthalpy from rho (barotropic)
        p = K * rho**eos_gamma
        h = 1.0 + eos_gamma * K * rho**gm1 / gm1

        # Velocity from conservatives
        Q = tau + D + p
        vr = Sr / (Q * gamma_rr)
        v2 = gamma_rr * vr * vr
        v2 = jnp.clip(v2, 0.0, 1.0 - 1e-12)
        W = 1.0 / jnp.sqrt(1.0 - v2)

        # Residual: f(rho) = D - rho * W
        f = D - rho * W

        # Check convergence
        is_converged = jnp.abs(f) <= tol * jnp.maximum(1.0, D)

        # Numerical derivative df/drho
        drho = 1e-8 * jnp.maximum(rho, 1e-10)
        rho2 = rho + drho
        p2 = K * rho2**eos_gamma
        Q2 = tau + D + p2
        vr2 = Sr / (Q2 * gamma_rr)
        v2_2 = gamma_rr * vr2 * vr2
        v2_2 = jnp.clip(v2_2, 0.0, 1.0 - 1e-12)
        W2 = 1.0 / jnp.sqrt(1.0 - v2_2)
        f2 = D - rho2 * W2
        df = (f2 - f) / drho

        # Standard Newton update: rho_{n+1} = rho_n - f/f'
        rho_new = rho - f / df
        rho_new = jnp.maximum(rho_new, rho_floor)

        return (iteration + 1, rho_new, is_converged)

    # Run Newton iterations
    init_state = (0, rho_start, False)
    final_iter, final_rho, converged = lax.while_loop(cond_fn, body_fn, init_state)

    # Compute final primitives
    rho = final_rho
    p = K * rho**eos_gamma
    h = 1.0 + eos_gamma * K * rho**gm1 / gm1
    Q = tau + D + p
    vr = Sr / (Q * gamma_rr)
    v2 = gamma_rr * vr * vr
    v2 = jnp.clip(v2, 0.0, 1.0 - 1e-16)
    W = 1.0 / jnp.sqrt(1.0 - v2)
    eps = K * rho**gm1 / gm1

    # Recompute convergence flag with FINAL values
    f_final = D - rho * W
    tol_final = tol * jnp.maximum(1.0, D)
    W_valid = (W <= W_max) & (W >= 1.0)
    converged_final = (jnp.abs(f_final) <= tol_final) & W_valid

    return rho, vr, p, eps, W, h, converged_final


# Vectorized version using vmap
@partial(jit, static_argnums=(6, 7, 8, 9, 10, 11))
def _solve_newton_batch_polytropic_jax(D, Sr, tau, gamma_rr, alpha, rho_guess,
                                        eos_K, eos_gamma, rho_floor, W_max, tol, max_iter):
    """
    JAX batch Newton-Raphson solver for polytropic EOS.

    Vectorized over all points using vmap - runs in parallel on GPU.

    Args:
        D, Sr, tau: Conservative variable arrays (N,)
        gamma_rr: Metric component array (N,)
        alpha: Lapse function array (N,)
        rho_guess: Initial density guess array (N,)
        eos_K: Polytropic constant (static)
        eos_gamma: Adiabatic index (static)
        rho_floor: Density floor (static)
        W_max: Maximum Lorentz factor (static)
        tol: Convergence tolerance (static)
        max_iter: Maximum iterations (static)

    Returns:
        tuple: (rho0, vr, p, eps, W, h, converged) - all arrays (N,)
    """
    newton_vmapped = vmap(
        lambda d, sr, t, grr, a, rg: _newton_kernel_polytropic_jax(
            d, sr, t, grr, a, rg, eos_K, eos_gamma, rho_floor, W_max, tol, max_iter
        )
    )

    return newton_vmapped(D, Sr, tau, gamma_rr, alpha, rho_guess)


# ============================================================================
# JAX KASTAUN et al. (2021) SOLVER - ROBUST CONS2PRIM WITH BRACKETING
# Based on: Kastaun et al., Phys. Rev. D 103, 023018 (2021)
# Simplified for pure hydrodynamics (no magnetic field)
# ============================================================================

def _kastaun_residual_jax(mu, D, r2, q, eos_gamma, v2_max):
    """
    Kastaun residual f(mu) = mu - mu_hat for ideal gas EOS (JAX, branchless).

    Not @jit-decorated: always called from inside a JIT context.

    Steps:
        1. v2 = min(mu^2 * r2, v2_max)
        2. W = 1/sqrt(1 - v2)
        3. rho = D/W
        4. e/D = q - mu*r2 + 1
        5. eps = W * e/D - 1,  P = rho*eps*(gamma-1),  h = 1 + eps*gamma
        6. nu = h/W
        7. mu_hat = 1/(nu + mu*r2)
        8. f = mu - mu_hat
    """
    gm1 = eos_gamma - 1.0

    v2 = jnp.minimum(mu * mu * r2, v2_max)
    v2 = jnp.minimum(v2, 1.0 - 1e-15)
    W = 1.0 / jnp.sqrt(1.0 - v2)

    rho = D / jnp.maximum(W, 1e-30)
    e_over_D = q - mu * r2 + 1.0
    eps = jnp.maximum(W * e_over_D - 1.0, 0.0)

    P = rho * eps * gm1
    h = jnp.maximum(1.0 + eps * eos_gamma, 1.0 + 1e-15)

    nu = h / W
    mu_hat = 1.0 / jnp.maximum(nu + mu * r2, 1e-30)

    return mu - mu_hat


@partial(jit, static_argnums=(4, 5, 6, 7, 8, 9, 10))
def _kastaun_kernel_jax(D, Sr, tau, gamma_rr,
                        eos_gamma, rho_floor, p_floor, v_max, W_max, tol, max_iter):
    """
    Kastaun et al. (2021) cons2prim kernel for ideal gas (single point, JAX).

    Bracketed root-finding on mu with False Position + Anderson-Bjorck.
    Fully branchless for JIT/GPU compatibility.

    Args:
        D, Sr, tau: Conservative variables (physical, not densitized) - scalars
        gamma_rr: Radial metric component
        eos_gamma: Adiabatic index (static)
        rho_floor, p_floor: Atmosphere floors (static)
        v_max, W_max: Velocity/Lorentz limits (static)
        tol, max_iter: Convergence parameters (static)

    Returns:
        (rho0, vr, p, eps, W, h, converged)
    """
    gm1 = eos_gamma - 1.0
    v2_max = v_max * v_max

    # Safe inputs
    D_safe = jnp.maximum(D, 1e-30)
    gamma_rr_safe = jnp.maximum(gamma_rr, 1e-30)

    # Auxiliary quantities
    q = tau / D_safe
    r2 = (Sr * Sr) / (D_safe * D_safe * gamma_rr_safe)

    # Bounds for mu
    mu_min = 1e-15
    mu_max_base = 1.0 / (1.0 + 1e-10)
    mu_v_limit = jnp.where(
        r2 > 1e-30,
        v_max / jnp.sqrt(jnp.maximum(r2, 1e-30)) * 0.999,
        mu_max_base
    )
    mu_max = jnp.minimum(mu_max_base, mu_v_limit)
    mu_max = jnp.maximum(mu_max, mu_min + 1e-15)

    # Evaluate residual at bounds
    f_min = _kastaun_residual_jax(mu_min, D_safe, r2, q, eos_gamma, v2_max)
    f_max = _kastaun_residual_jax(mu_max, D_safe, r2, q, eos_gamma, v2_max)

    has_bracket = f_min * f_max < 0.0

    # ---- Bracket search (20 bisection iterations if no bracket) ----
    def bracket_body(_, state):
        a, b, fa, fb, mu_test, found = state

        f_test = _kastaun_residual_jax(mu_test, D_safe, r2, q, eos_gamma, v2_max)

        left_ok = (fa * f_test < 0.0) & ~found
        right_ok = (f_test * fb < 0.0) & ~found & ~left_ok
        new_found = found | left_ok | right_ok

        b_new = jnp.where(left_ok, mu_test, b)
        fb_new = jnp.where(left_ok, f_test, fb)
        a_new = jnp.where(right_ok, mu_test, a)
        fa_new = jnp.where(right_ok, f_test, fa)

        # Next test point (bisect toward expected root)
        mu_next = jnp.where(fa > 0.0,
                            0.5 * (mu_test + b_new),
                            0.5 * (a_new + mu_test))
        mu_test_new = jnp.where(new_found, mu_test, mu_next)

        return (a_new, b_new, fa_new, fb_new, mu_test_new, new_found)

    bracket_init = (mu_min, mu_max, f_min, f_max,
                    0.5 * (mu_min + mu_max), has_bracket)
    a_br, b_br, fa_br, fb_br, _, bracket_found = lax.fori_loop(
        0, 20, bracket_body, bracket_init
    )

    # ---- False Position with Anderson-Bjorck acceleration ----
    def fp_cond(state):
        _, _, _, _, _, _, _, iteration, converged = state
        return (iteration < max_iter) & (~converged)

    def fp_body(state):
        a, b, fa, fb, x, xold, side, iteration, converged = state

        xold_new = x

        # Interpolation
        denom = fb - fa
        x_fp = jnp.where(jnp.abs(denom) < 1e-30,
                         0.5 * (a + b),
                         (a * fb - b * fa) / denom)
        x_new = jnp.where((x_fp <= a) | (x_fp >= b),
                          0.5 * (a + b), x_fp)

        fx = _kastaun_residual_jax(x_new, D_safe, r2, q, eos_gamma, v2_max)

        # Convergence check
        conv = ((jnp.abs(x_new - xold_new) <= tol * jnp.maximum(jnp.abs(x_new), 1e-15))
                | (jnp.abs(fx) <= tol))

        # Determine which side contains the root
        root_left = fa * fx < 0.0

        # Anderson-Bjorck: scale fa when b updated on same side twice
        m_l = 1.0 - fx / jnp.where(jnp.abs(fb) > 1e-30, fb, 1e-30)
        fa_l = jnp.where(side == -1.0,
                         jnp.where(m_l > 0.0, fa * m_l, fa * 0.5),
                         fa)

        # Anderson-Bjorck: scale fb when a updated on same side twice
        m_r = 1.0 - fx / jnp.where(jnp.abs(fa) > 1e-30, fa, 1e-30)
        fb_r = jnp.where(side == 1.0,
                         jnp.where(m_r > 0.0, fb * m_r, fb * 0.5),
                         fb)

        a_new = jnp.where(root_left, a, x_new)
        b_new = jnp.where(root_left, x_new, b)
        fa_new = jnp.where(root_left, fa_l, fx)
        fb_new = jnp.where(root_left, fx, fb_r)
        side_new = jnp.where(root_left, -1.0, 1.0)

        return (a_new, b_new, fa_new, fb_new, x_new, xold_new,
                side_new, iteration + 1, conv)

    fp_init = (a_br, b_br, fa_br, fb_br, a_br, a_br,
               0.0, 0, jnp.bool_(False))
    fp_result = lax.while_loop(fp_cond, fp_body, fp_init)
    mu_sol = fp_result[4]
    fp_converged = fp_result[8]

    # ---- Recover primitives from mu ----
    v2 = jnp.minimum(mu_sol * mu_sol * r2, v2_max)
    v2 = jnp.minimum(v2, 1.0 - 1e-15)
    W = 1.0 / jnp.sqrt(1.0 - v2)
    W = jnp.minimum(W, W_max * 1.0)

    rho = D_safe / jnp.maximum(W, 1e-30)
    rho = jnp.maximum(rho, rho_floor)

    e_over_D = q - mu_sol * r2 + 1.0
    eps = jnp.maximum(W * e_over_D - 1.0, 0.0)

    P = rho * eps * gm1
    P = jnp.maximum(P, p_floor)
    eps = jnp.where(P <= p_floor, p_floor / (rho * gm1), eps)

    h = jnp.maximum(1.0 + eps * eos_gamma, 1.0 + 1e-15)

    # Velocity (sign from Sr)
    vr = jnp.sign(Sr) * jnp.sqrt(jnp.maximum(v2 / gamma_rr_safe, 0.0))

    # Overall success
    valid_input = (D > 0.0) & jnp.isfinite(D) & jnp.isfinite(Sr) & jnp.isfinite(tau)
    success = valid_input & bracket_found & fp_converged

    # Atmosphere fallback values
    eps_atm = p_floor / (rho_floor * gm1)
    h_atm = 1.0 + eps_atm * eos_gamma

    rho_out = jnp.where(success, rho, rho_floor)
    vr_out = jnp.where(success, vr, 0.0)
    p_out = jnp.where(success, P, p_floor)
    eps_out = jnp.where(success, eps, eps_atm)
    W_out = jnp.where(success, W, 1.0)
    h_out = jnp.where(success, h, h_atm)

    return rho_out, vr_out, p_out, eps_out, W_out, h_out, success


@partial(jit, static_argnums=(4, 5, 6, 7, 8, 9, 10))
def _solve_kastaun_batch_jax(D, Sr, tau, gamma_rr,
                              eos_gamma, rho_floor, p_floor, v_max, W_max, tol, max_iter):
    """
    Batch Kastaun solver for ideal gas EOS using vmap.

    Args:
        D, Sr, tau: Conservative variable arrays (N,)
        gamma_rr: Metric component array (N,)
        eos_gamma, rho_floor, p_floor, v_max, W_max, tol, max_iter: static params

    Returns:
        tuple: (rho0, vr, p, eps, W, h, converged) - all arrays (N,)
    """
    kastaun_vmapped = vmap(
        lambda d, sr, t, grr: _kastaun_kernel_jax(
            d, sr, t, grr, eos_gamma, rho_floor, p_floor, v_max, W_max, tol, max_iter
        )
    )
    return kastaun_vmapped(D, Sr, tau, gamma_rr)


# ============================================================================
# MAIN SOLVER CLASS (JAX VERSION)
# ============================================================================

class Cons2PrimSolverJAX:
    """
    JAX-accelerated conservative to primitive variable converter.

    API-compatible with Cons2PrimSolver from cons2prim.py.
    Uses JAX for GPU acceleration when available.

    Args:
        eos: Equation of state object with eps_from_rho_p(rho0, p) method
        atmosphere: AtmosphereParams object (required)
        tol: Newton-Raphson tolerance (default: 1e-12)
        max_iter: Maximum iterations (default: 500)
    """

    def __init__(self, eos, atmosphere, tol=1e-12, max_iter=500):
        self.eos = eos
        self.atmosphere = atmosphere

        # Floor applicator (handles all floor logic)
        self.floor_applicator = FloorApplicator(self.atmosphere, eos)

        # Solver parameters from atmosphere
        self.rho_floor = atmosphere.rho_floor
        self.p_floor = atmosphere.p_floor
        self.v_max = atmosphere.v_max
        self.W_max = atmosphere.W_max
        self.tol = tol
        self.max_iter = max_iter

        # Check if we can use optimized path for ideal gas
        self._is_ideal_gas = (hasattr(eos, 'name') and
                              eos.name.startswith('ideal_gas') and
                              hasattr(eos, 'gamma'))

        if self._is_ideal_gas:
            self._eos_gamma = float(eos.gamma)

        # Check if we can use optimized path for polytropic EOS
        self._is_polytropic = (hasattr(eos, 'name') and
                               eos.name.startswith('polytropic') and
                               hasattr(eos, 'K') and hasattr(eos, 'gamma'))

        if self._is_polytropic:
            self._eos_K = float(eos.K)
            self._eos_gamma = float(eos.gamma)

        # Statistics tracking
        self.stats = {
            "total_calls": 0,
            "successful_conversions": 0,
            "newton_successes": 0,
            "atmosphere_fallbacks": 0,
            "conservative_floors_applied": 0
        }

    def convert(self, D, Sr, tau, geom, p_guess=None, apply_conservative_floors=True):
        """
        Convert conservative to primitive variables using JAX-accelerated Newton-Raphson.

        API-compatible with Cons2PrimSolver.convert().

        Args:
            D, Sr, tau: Conservative variables (physical, non-densitized)
            geom: GeometryState containing gamma_rr, e6phi, alpha, beta_r
            p_guess: Initial pressure guess (optional)
            apply_conservative_floors: Whether to apply conservative floors

        Returns:
            tuple: (rho0, vr, p, eps, W, h, success, D_out, Sr_out, tau_out)
        """
        self.stats["total_calls"] += 1

        # Extract geometry components
        gamma_rr = np.atleast_1d(np.asarray(geom.gamma_rr, dtype=np.float64))
        e6phi = np.atleast_1d(np.asarray(geom.e6phi, dtype=np.float64))
        alpha = np.atleast_1d(np.asarray(geom.alpha, dtype=np.float64))
        beta_r = np.atleast_1d(np.asarray(geom.beta_r, dtype=np.float64))

        # Convert to numpy for preprocessing
        D = np.atleast_1d(np.asarray(D, dtype=np.float64))
        Sr = np.atleast_1d(np.asarray(Sr, dtype=np.float64))
        tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        N = len(D)

        # Output conservative arrays
        D_out = D.copy()
        Sr_out = Sr.copy()
        tau_out = tau.copy()

        # Apply conservative variable floors (uses Numba, stays on CPU)
        if apply_conservative_floors:
            D, Sr, tau, floor_mask = self.floor_applicator.apply_conservative_floors(
                D, Sr, tau, gamma_rr
            )
            if np.any(floor_mask):
                self.stats["conservative_floors_applied"] += np.sum(floor_mask)

        # Allocate output arrays
        rho0, vr, p, eps, W, h = (np.zeros(N) for _ in range(6))
        success = np.zeros(N, dtype=bool)

        # ATMOSPHERE DETECTION
        atm_mask = D < 10.0 * self.rho_floor

        if np.any(atm_mask):
            rho0[atm_mask] = self.rho_floor
            vr[atm_mask] = 0.0
            p[atm_mask] = self.p_floor
            try:
                eps[atm_mask] = self.eos.eps_from_rho_p(self.rho_floor, self.p_floor)
            except:
                eps[atm_mask] = 1e-10
            W[atm_mask] = 1.0
            h[atm_mask] = 1.0 + eps[atm_mask] + self.p_floor / self.rho_floor
            success[atm_mask] = True
            self.stats["atmosphere_fallbacks"] += np.sum(atm_mask)

        # Solve non-atmosphere points with JAX
        solve_mask = ~atm_mask

        if np.any(solve_mask):
            # Prepare pressure guess
            if p_guess is not None:
                p_init = np.maximum(np.asarray(p_guess)[solve_mask], self.p_floor)
            else:
                p_init = np.maximum(self.p_floor, 0.1 * (tau[solve_mask] + D[solve_mask]))

            # Convert to JAX arrays and solve on GPU
            D_jax = jnp.array(D[solve_mask])
            Sr_jax = jnp.array(Sr[solve_mask])
            tau_jax = jnp.array(tau[solve_mask])
            gamma_rr_jax = jnp.array(gamma_rr[solve_mask])
            alpha_jax = jnp.array(alpha[solve_mask])
            p_init_jax = jnp.array(p_init)

            if self._is_ideal_gas:
                # JAX-accelerated Newton solver for ideal gas
                result = _solve_newton_batch_jax(
                    D_jax, Sr_jax, tau_jax, gamma_rr_jax, alpha_jax, p_init_jax,
                    self._eos_gamma, self.p_floor, self.W_max, self.tol, self.max_iter
                )

                # Convert back to numpy
                rho0_solved = np.asarray(result[0])
                vr_solved = np.asarray(result[1])
                p_solved = np.asarray(result[2])
                eps_solved = np.asarray(result[3])
                W_solved = np.asarray(result[4])
                h_solved = np.asarray(result[5])
                success_solved = np.asarray(result[6])

            elif self._is_polytropic:
                # JAX-accelerated Newton solver for polytropic EOS
                # For polytropic EOS, use rho as the unknown (not pressure)
                # Initial guess: rho ~ D (since D = rho * W and W ~ 1 for low velocities)
                rho_init = np.maximum(np.asarray(D[solve_mask]), self.rho_floor)
                rho_init_jax = jnp.array(rho_init)

                result = _solve_newton_batch_polytropic_jax(
                    D_jax, Sr_jax, tau_jax, gamma_rr_jax, alpha_jax, rho_init_jax,
                    self._eos_K, self._eos_gamma, self.rho_floor, self.W_max, self.tol, self.max_iter
                )

                # Convert back to numpy
                rho0_solved = np.asarray(result[0])
                vr_solved = np.asarray(result[1])
                p_solved = np.asarray(result[2])
                eps_solved = np.asarray(result[3])
                W_solved = np.asarray(result[4])
                h_solved = np.asarray(result[5])
                success_solved = np.asarray(result[6])

            else:
                # Fall back to CPU solver for general EOS
                # Import the Numba version for this case
                from .cons2prim import Cons2PrimSolver
                fallback = Cons2PrimSolver(self.eos, self.atmosphere, self.tol, self.max_iter)
                result = fallback._solve_vectorized_points(
                    D[solve_mask], Sr[solve_mask], tau[solve_mask],
                    gamma_rr[solve_mask], p_init, alpha[solve_mask]
                )
                rho0_solved, vr_solved, p_solved, eps_solved, W_solved, h_solved, success_solved = result

            # Place results back
            rho0[solve_mask] = rho0_solved
            vr[solve_mask] = vr_solved
            p[solve_mask] = p_solved
            eps[solve_mask] = eps_solved
            W[solve_mask] = W_solved
            h[solve_mask] = h_solved
            success[solve_mask] = success_solved

            self.stats["successful_conversions"] += np.sum(success_solved)
            self.stats["newton_successes"] += np.sum(success_solved)
            self.stats["atmosphere_fallbacks"] += np.sum(~success_solved)

        # Handle failed points with atmosphere (using FloorApplicator)
        failed_mask = ~success
        if np.any(failed_mask):
            self.floor_applicator.apply_atmosphere_fallback(
                rho0, vr, p, eps, W, h, failed_mask
            )

        # Enforce velocity limits
        rho0, vr, p = self.floor_applicator.apply_primitive_floors(rho0, vr, p, gamma_rr)

        # Final pressure floor with EOS-consistent recompute
        low_p_mask = p < self.p_floor
        if np.any(low_p_mask):
            p[low_p_mask] = self.p_floor
            rho_lp = np.maximum(rho0[low_p_mask], 1e-30)
            if self._is_ideal_gas:
                gm1 = self._eos_gamma - 1.0
                eps[low_p_mask] = self.p_floor / (rho_lp * gm1)
            else:
                eps[low_p_mask] = self.eos.eps_from_rho_p(rho0[low_p_mask], self.p_floor)
            h[low_p_mask] = 1.0 + eps[low_p_mask] + self.p_floor / rho_lp

        # Update conservatives for atmosphere points
        atm_points = rho0 <= 100.0 * self.rho_floor
        if np.any(atm_points):
            geom_atm = GeometryState(
                alpha=alpha[atm_points],
                beta_r=beta_r[atm_points],
                gamma_rr=gamma_rr[atm_points],
                e6phi=e6phi[atm_points]
            )
            D_atm, Sr_atm, tau_atm = prim_to_cons(
                rho0[atm_points], vr[atm_points], p[atm_points],
                geom_atm, self.eos
            )
            D_out[atm_points] = D_atm
            Sr_out[atm_points] = Sr_atm
            tau_out[atm_points] = tau_atm

        return rho0, vr, p, eps, W, h, success, D_out, Sr_out, tau_out

    def get_statistics(self):
        """Get conversion statistics."""
        total = max(self.stats["total_calls"], 1)
        return {
            **self.stats,
            "success_rate": self.stats["successful_conversions"] / total,
            "newton_rate": self.stats["newton_successes"] / total,
        }

    def reset_statistics(self):
        """Reset all statistics counters."""
        for key in self.stats:
            self.stats[key] = 0
