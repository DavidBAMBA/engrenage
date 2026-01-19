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
# Enable float64 precision (JAX defaults to float32)
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
import numpy as np

# Import atmosphere module - handle both relative and absolute imports
try:
    # Try relative import (when used as part of package)
    from ...atmosphere import FloorApplicator
except ImportError:
    try:
        # Try absolute import (when source is in path)
        from source.matter.hydro.atmosphere import FloorApplicator
    except ImportError:
        # Fallback: Add parent directories to path
        import sys
        import os
        _hydro_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if _hydro_path not in sys.path:
            sys.path.insert(0, _hydro_path)
        from atmosphere import FloorApplicator


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
    gamma_rr_safe = jnp.maximum(gamma_rr, 1e-30)
    alpha_safe = jnp.maximum(alpha, 1e-30)

    # State for while loop: (iteration, p, converged)
    def cond_fn(state):
        iteration, p, converged = state
        return (iteration < max_iter) & (~converged)

    def body_fn(state):
        iteration, p, converged = state

        # Evaluate pressure residual
        Q = tau + D + p
        vr = alpha_safe * Sr / (Q * gamma_rr_safe)
        v2 = gamma_rr_safe * vr * vr

        # Clamp v2 for safety
        v2 = jnp.clip(v2, 0.0, 1.0 - 1e-16)
        W = 1.0 / jnp.sqrt(1.0 - v2)

        # Check Lorentz factor bounds - if violated, adjust p and continue
        W_valid = (W <= W_max) & (W >= 1.0)

        rho0 = D / jnp.maximum(W, 1e-30)
        rho0_valid = rho0 > 0.0

        eps = p / (rho0 * gm1 + 1e-30)
        h = 1.0 + eps + p / (rho0 + 1e-30)

        # Residual
        f = rho0 * h * W * W - Q

        # Check convergence
        tol_val = tol * jnp.maximum(1.0, jnp.abs(p))
        is_converged = jnp.abs(f) <= tol_val

        # Analytic derivative for ideal gas
        E = tau + D
        Q2 = (E + p) ** 2
        Sr_sq = Sr * Sr
        v2_deriv = Sr_sq / jnp.maximum(Q2 * gamma_rr_safe, 1e-30)
        W_loc = 1.0 / jnp.maximum((1.0 - v2_deriv), 1e-16) ** 0.5
        W_cubed = W_loc * W_loc * W_loc
        Q_cubed = (E + p) ** 3
        Wprime = -Sr_sq * W_cubed / jnp.maximum(Q_cubed * gamma_rr_safe, 1e-30)
        W_sq = W_loc * W_loc
        df = c_ideal * W_sq + (D + 2.0 * c_ideal * p * W_loc) * Wprime - 1.0

        # Check if derivative is too small (matches Numba: skip iteration & reset)
        small_df = jnp.abs(df) < 1e-15

        # Newton update (only meaningful when df is valid)
        df_safe = jnp.where(small_df, 1.0, df)
        p_newton = p - f / df_safe

        # Handle invalid Newton update (NaN or negative)
        p_newton = jnp.where(
            (p_newton <= 0.0) | ~jnp.isfinite(p_newton),
            jnp.maximum(p_floor, 0.5 * p),
            p_newton
        )

        # Apply damped Newton step
        p_damped = 0.5 * p + 0.5 * p_newton

        # Final pressure update:
        # - If W or rho0 invalid: reset p (matches Numba continue behavior)
        # - If df too small: reset p (matches Numba continue behavior)
        # - Otherwise: use damped Newton
        p_new = jnp.where(
            W_valid & rho0_valid & ~small_df,
            p_damped,
            jnp.maximum(p_floor, p * 1.5 + 1e-14)
        )

        # Update convergence flag
        new_converged = is_converged & W_valid & rho0_valid

        return (iteration + 1, p_new, new_converged)

    # Run Newton iterations
    init_state = (0, p_start, False)
    final_iter, final_p, converged = lax.while_loop(cond_fn, body_fn, init_state)

    # Compute final primitives
    p = final_p
    Q = tau + D + p
    vr = alpha_safe * Sr / (Q * gamma_rr_safe)
    v2 = gamma_rr_safe * vr * vr
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
    gamma_rr_safe = jnp.maximum(gamma_rr, 1e-30)
    alpha_safe = jnp.maximum(alpha, 1e-30)

    # State for while loop: (iteration, rho, converged)
    def cond_fn(state):
        iteration, rho, converged = state
        return (iteration < max_iter) & (~converged)

    def body_fn(state):
        iteration, rho, converged = state

        # Pressure and enthalpy from rho (barotropic: depends only on rho)
        p = K * rho**eos_gamma
        h = 1.0 + eos_gamma * K * rho**gm1 / gm1

        # Compute Q = tau + D + p, then velocity
        Q = tau + D + p
        vr = alpha_safe * Sr / (Q * gamma_rr_safe)
        v2 = gamma_rr_safe * vr * vr

        # Clamp v2 for safety
        v2 = jnp.clip(v2, 0.0, 1.0 - 1e-16)
        W = 1.0 / jnp.sqrt(1.0 - v2)

        # Check Lorentz factor bounds
        W_valid = (W <= W_max) & (W >= 1.0)

        # Residual: f = D - rho * W
        f = D - rho * W

        # Check convergence
        tol_val = tol * jnp.maximum(1.0, D)
        is_converged = jnp.abs(f) <= tol_val

        # Numerical derivative df/drho
        drho = 1e-8 * jnp.maximum(rho, 1e-10)
        rho2 = rho + drho
        p2 = K * rho2**eos_gamma
        Q2 = tau + D + p2
        vr2 = alpha_safe * Sr / (Q2 * gamma_rr_safe)
        v2_2 = gamma_rr_safe * vr2 * vr2
        v2_2 = jnp.clip(v2_2, 0.0, 1.0 - 1e-16)
        W2 = 1.0 / jnp.sqrt(1.0 - v2_2)
        f2 = D - rho2 * W2

        df = (f2 - f) / drho

        # Check if derivative is too small
        small_df = jnp.abs(df) < 1e-15

        # Newton update
        df_safe = jnp.where(small_df, 1.0, df)
        rho_newton = rho - f / df_safe

        # Handle invalid Newton update
        rho_newton = jnp.where(
            (rho_newton <= 0.0) | ~jnp.isfinite(rho_newton),
            jnp.maximum(rho_floor, 0.5 * rho),
            rho_newton
        )

        # Apply damped Newton step
        rho_damped = 0.5 * rho + 0.5 * rho_newton

        # Final rho update
        rho_new = jnp.where(
            W_valid & ~small_df,
            rho_damped,
            jnp.maximum(rho_floor, rho * 0.5)
        )

        # Update convergence flag
        new_converged = is_converged & W_valid

        return (iteration + 1, rho_new, new_converged)

    # Run Newton iterations
    init_state = (0, rho_start, False)
    final_iter, final_rho, converged = lax.while_loop(cond_fn, body_fn, init_state)

    # Compute final primitives
    rho = final_rho
    p = K * rho**eos_gamma
    h = 1.0 + eos_gamma * K * rho**gm1 / gm1
    Q = tau + D + p
    vr = alpha_safe * Sr / (Q * gamma_rr_safe)
    v2 = gamma_rr_safe * vr * vr
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

    def convert(self, D, Sr, tau, gamma_rr, p_guess=None, apply_conservative_floors=True,
                e6phi=None, alpha=None, beta_r=None):
        """
        Convert conservative to primitive variables using JAX-accelerated Newton-Raphson.

        API-compatible with Cons2PrimSolver.convert().

        Returns:
            tuple: (rho0, vr, p, eps, W, h, success, D_out, Sr_out, tau_out)
        """
        self.stats["total_calls"] += 1

        # Convert to numpy for preprocessing (JAX arrays don't support in-place ops)
        D = np.atleast_1d(np.asarray(D, dtype=np.float64))
        Sr = np.atleast_1d(np.asarray(Sr, dtype=np.float64))
        tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        gamma_rr = np.atleast_1d(np.asarray(gamma_rr, dtype=np.float64))
        N = len(D)

        # Prepare densitization factors
        if e6phi is None:
            e6phi = np.ones(N)
        else:
            e6phi = np.atleast_1d(np.asarray(e6phi, dtype=np.float64))
        if alpha is None:
            alpha = np.ones(N)
        else:
            alpha = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
        if beta_r is None:
            beta_r = np.zeros(N)
        else:
            beta_r = np.atleast_1d(np.asarray(beta_r, dtype=np.float64))

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
        atm_points = rho0 <= 10.0 * self.rho_floor
        if np.any(atm_points):
            from .cons2prim import prim_to_cons
            D_atm, Sr_atm, tau_atm = prim_to_cons(
                rho0[atm_points], vr[atm_points], p[atm_points],
                gamma_rr[atm_points], self.eos,
                e6phi=e6phi[atm_points],
                alpha=alpha[atm_points],
                beta_r=beta_r[atm_points]
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
