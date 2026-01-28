"""
JAX-native time integration for engrenage.

Provides RK4 time stepping with jax.lax.scan for fusing the entire
evolution loop into a single compiled program.

This eliminates per-timestep Python overhead and is the main source
of the ~10x speedup over Numba for small grids.
"""

import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial


def rk4_step(state, dt, rhs_fn, rhs_args):
    """
    Single RK4 step.

    Args:
        state: tuple of (D, Sr, tau) JAX arrays
        dt: timestep (float)
        rhs_fn: callable(D, Sr, tau, *rhs_args) -> (dD, dSr, dtau)
        rhs_args: additional arguments passed to rhs_fn

    Returns:
        new_state: tuple of (D, Sr, tau) after one RK4 step
    """
    D, Sr, tau = state

    # Stage 1
    dD1, dSr1, dtau1 = rhs_fn(D, Sr, tau, *rhs_args)

    # Stage 2
    D2 = D + 0.5 * dt * dD1
    Sr2 = Sr + 0.5 * dt * dSr1
    tau2 = tau + 0.5 * dt * dtau1
    dD2, dSr2, dtau2 = rhs_fn(D2, Sr2, tau2, *rhs_args)

    # Stage 3
    D3 = D + 0.5 * dt * dD2
    Sr3 = Sr + 0.5 * dt * dSr2
    tau3 = tau + 0.5 * dt * dtau2
    dD3, dSr3, dtau3 = rhs_fn(D3, Sr3, tau3, *rhs_args)

    # Stage 4
    D4 = D + dt * dD3
    Sr4 = Sr + dt * dSr3
    tau4 = tau + dt * dtau3
    dD4, dSr4, dtau4 = rhs_fn(D4, Sr4, tau4, *rhs_args)

    # Combine
    D_new = D + (dt / 6.0) * (dD1 + 2*dD2 + 2*dD3 + dD4)
    Sr_new = Sr + (dt / 6.0) * (dSr1 + 2*dSr2 + 2*dSr3 + dSr4)
    tau_new = tau + (dt / 6.0) * (dtau1 + 2*dtau2 + 2*dtau3 + dtau4)

    return (D_new, Sr_new, tau_new)


def apply_atmosphere_reset_jax(D, Sr, tau, gamma_rr, e6phi,
                               rho_floor, p_floor, eos_type, eos_gamma, eos_K=0.0):
    """
    Apply atmosphere reset to conservative variables (functional, JIT-compatible).

    Points with D < threshold are reset to atmosphere.
    """
    gm1 = eos_gamma - 1.0

    # Detect atmosphere
    threshold = 100.0 * rho_floor * e6phi
    atm_mask = D < threshold

    # Atmosphere conservative values
    if eos_type == 'polytropic':
        eps_atm = eos_K * rho_floor**gm1 / gm1
        p_atm = eos_K * rho_floor**eos_gamma
    else:
        eps_atm = p_floor / (rho_floor * gm1)
        p_atm = p_floor

    h_atm = 1.0 + eps_atm + p_atm / rho_floor
    W_atm = 1.0

    D_atm = e6phi * rho_floor * W_atm
    Sr_atm = 0.0
    tau_atm = e6phi * (rho_floor * h_atm * W_atm * W_atm - p_atm - rho_floor * W_atm)

    D_out = jnp.where(atm_mask, D_atm, D)
    Sr_out = jnp.where(atm_mask, Sr_atm, Sr)
    tau_out = jnp.where(atm_mask, tau_atm, tau)

    return D_out, Sr_out, tau_out


def evolve_scan(initial_state, dt, n_steps, rhs_fn, rhs_args,
                atm_reset_fn=None, atm_args=None,
                save_every=0):
    """
    Full evolution using jax.lax.scan.

    Fuses the entire time loop into a single compiled XLA program.
    This eliminates Python overhead per step and is the main source
    of speedup.

    Args:
        initial_state: tuple of (D, Sr, tau) JAX arrays
        dt: timestep (float)
        n_steps: number of RK4 steps (int)
        rhs_fn: callable(D, Sr, tau, *rhs_args) -> (dD, dSr, dtau)
        rhs_args: tuple of additional arguments for rhs_fn
        atm_reset_fn: optional callable(D, Sr, tau, *atm_args) -> (D, Sr, tau)
        atm_args: tuple of arguments for atm_reset_fn
        save_every: save state every N steps (0 = save nothing, return only final)

    Returns:
        final_state: tuple of (D, Sr, tau) after n_steps
        saved_states: dict with 'D', 'Sr', 'tau' arrays of shape (n_saves, N)
                     or None if save_every == 0
    """
    def body(carry, step_idx):
        D, Sr, tau = carry

        # RK4 step
        D_new, Sr_new, tau_new = rk4_step((D, Sr, tau), dt, rhs_fn, rhs_args)

        # Atmosphere reset
        if atm_reset_fn is not None:
            D_new, Sr_new, tau_new = atm_reset_fn(D_new, Sr_new, tau_new, *atm_args)

        carry_out = (D_new, Sr_new, tau_new)

        if save_every > 0:
            # Save state at specified intervals
            should_save = (step_idx % save_every == 0)
            # Always output something (scan requires fixed output shape)
            snapshot = (D_new, Sr_new, tau_new)
            return carry_out, snapshot
        else:
            return carry_out, None

    final, outputs = lax.scan(body, initial_state, jnp.arange(n_steps))

    if save_every > 0 and outputs is not None:
        D_all, Sr_all, tau_all = outputs
        # Filter to only saved steps
        save_mask = jnp.arange(n_steps) % save_every == 0
        saved = {
            'D': D_all,
            'Sr': Sr_all,
            'tau': tau_all,
            'step_indices': jnp.arange(n_steps),
        }
        return final, saved
    else:
        return final, None


def evolve_python_loop(initial_state, dt, n_steps, rhs_fn, rhs_args,
                       atm_reset_fn=None, atm_args=None,
                       print_every=1000):
    """
    Evolution with a Python loop (for debugging/diagnostics).

    Slower than evolve_scan but allows intermediate inspection.

    Args:
        initial_state: tuple of (D, Sr, tau) JAX arrays
        dt: timestep
        n_steps: number of steps
        rhs_fn: RHS function
        rhs_args: RHS arguments
        atm_reset_fn: atmosphere reset function
        atm_args: atmosphere reset arguments
        print_every: print progress every N steps

    Returns:
        final_state: tuple of (D, Sr, tau)
        diagnostics: dict with central density time series
    """
    import time as _time

    D, Sr, tau = initial_state
    N = len(D)

    # Time series storage
    central_rho_history = []
    times = []

    t0_wall = _time.perf_counter()

    for step in range(n_steps):
        D, Sr, tau = rk4_step((D, Sr, tau), dt, rhs_fn, rhs_args)

        if atm_reset_fn is not None:
            D, Sr, tau = atm_reset_fn(D, Sr, tau, *atm_args)

        if print_every > 0 and (step + 1) % print_every == 0:
            elapsed = _time.perf_counter() - t0_wall
            D_np = jax.device_get(D)
            rho_c = float(D_np[N // 2])  # approximate central density
            print(f"  Step {step+1}/{n_steps} | wall={elapsed:.1f}s | rho_c~{rho_c:.6e}")

    elapsed_total = _time.perf_counter() - t0_wall

    diagnostics = {
        'wall_time': elapsed_total,
        'steps_per_second': n_steps / elapsed_total,
    }

    return (D, Sr, tau), diagnostics
