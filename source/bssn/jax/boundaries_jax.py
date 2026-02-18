# boundaries_jax.py
#
# JAX boundary conditions for BSSN + matter variables.
# Uses .at[].set() for JIT-compatible array updates.
# Loops over num_ghosts (=3) are unrolled at compile time by JAX.

import jax.numpy as jnp
from functools import partial
import jax

from source.bssn.bssnstatevariables import (
    NUM_BSSN_VARS, BSSN_PARITY, BSSN_ASYMP_POWER, BSSN_ASYMP_OFFSET,
)


def fill_bssn_boundaries_jax(state, r, num_ghosts, parity, asymp_power, asymp_offset,
                              outer_bc_type="zero_gradient", boundary_ref=None):
    """
    Apply boundary conditions to the full state (NUM_VARS, N).

    Inner boundary (r=0): parity conditions.
    Outer boundary: configurable (asymptotic, zero_gradient, sommerfeld, or dirichlet).

    Args:
        state: (NUM_VARS, N) full state array
        r: (N,) radial coordinate
        num_ghosts: int, number of ghost cells (3)
        parity: (NUM_VARS,) parity array (+1 or -1)
        asymp_power: (NUM_VARS,) asymptotic power law exponents
        asymp_offset: (NUM_VARS,) asymptotic offsets
        outer_bc_type: str, outer boundary type:
            - "asymptotic": power-law extrapolation (default for BH)
            - "zero_gradient": outflow condition (default for TOV)
            - "sommerfeld": radiative condition (experimental)
            - "dirichlet": fixed to pre-computed reference values (exact Schwarzschild)
        boundary_ref: (NUM_VARS, num_ghosts) pre-computed values for ghost cells.
            Required when outer_bc_type="dirichlet".

    Returns:
        state: (NUM_VARS, N) with boundary conditions applied
    """
    state = _fill_inner_boundary(state, num_ghosts, parity)
    state = _fill_outer_boundary(state, r, num_ghosts, asymp_power, asymp_offset, outer_bc_type, boundary_ref)
    return state


def _fill_inner_boundary(state, num_ghosts, parity):
    """
    Inner boundary: parity conditions under r -> -r.
    Ghost cells mirror interior cells with appropriate parity.
    """
    # state[:, ghost-1-i] = parity * state[:, ghost+i]  for i in 0..ghost-1
    # For num_ghosts=3: indices 2,1,0 mirror 3,4,5
    for i in range(num_ghosts):
        state = state.at[:, num_ghosts - 1 - i].set(
            parity * state[:, num_ghosts + i]
        )
    return state


def _fill_outer_boundary(state, r, num_ghosts, asymp_power, asymp_offset,
                          bc_type="zero_gradient", boundary_ref=None):
    """
    Outer boundary: configurable boundary condition.

    Supports four types:
    1. "asymptotic": power-law extrapolation (var ~ r^power + offset)
    2. "zero_gradient": outflow (∂_r var = 0), robust for TOV
    3. "sommerfeld": radiative condition (∂_t var + v ∂_r var = 0)
    4. "dirichlet": fixed to pre-computed reference values (exact analytical solution)

    Args:
        bc_type: str, boundary condition type
        boundary_ref: (NUM_VARS, num_ghosts) reference values for "dirichlet" BC
    """
    N = state.shape[1]
    idx_ref = N - num_ghosts - 1  # last interior point

    if bc_type == "asymptotic":
        # Original asymptotic extrapolation (for BH in vacuum)
        r_ref = r[idx_ref]
        val_ref = state[:, idx_ref]
        bssn_val_ref = val_ref[:NUM_BSSN_VARS]
        bssn_asymp_power = asymp_power[:NUM_BSSN_VARS]
        bssn_asymp_offset = asymp_offset[:NUM_BSSN_VARS]
        b_bssn = (bssn_val_ref - bssn_asymp_offset) / (r_ref ** bssn_asymp_power)
        matter_val_ref = val_ref[NUM_BSSN_VARS:]

        for i in range(num_ghosts):
            idx = N - num_ghosts + i
            bssn_vals = bssn_asymp_offset + b_bssn * r[idx] ** bssn_asymp_power
            new_col = jnp.concatenate([bssn_vals, matter_val_ref])
            state = state.at[:, idx].set(new_col)

    elif bc_type == "zero_gradient":
        # Zero-gradient (outflow): ∂_r var = 0
        # Robust for TOV stars and general quasi-static solutions
        val_ref = state[:, idx_ref]
        for i in range(num_ghosts):
            idx = N - num_ghosts + i
            state = state.at[:, idx].set(val_ref)

    elif bc_type == "sommerfeld":
        # Sommerfeld (radiative) for BSSN, outflow for matter
        # BSSN: ∂_t u + v ∂_r u = 0 → u_ghost = u_ref + dr * (∂_r u)_ref
        # Matter: outflow (zero-gradient) → u_ghost = u_ref
        dr = r[idx_ref] - r[idx_ref - 1]
        val_ref = state[:, idx_ref]
        val_ref_m1 = state[:, idx_ref - 1]

        # Estimate ∂_r var at last interior point (backward difference)
        dr_var = (val_ref - val_ref_m1) / dr

        for i in range(num_ghosts):
            idx = N - num_ghosts + i
            dr_ghost = r[idx] - r[idx_ref]

            # BSSN variables: Sommerfeld (extrapolate linearly)
            val_bssn = val_ref[:NUM_BSSN_VARS] + dr_ghost * dr_var[:NUM_BSSN_VARS]

            # Matter variables: Outflow (copy last interior value)
            val_matter = val_ref[NUM_BSSN_VARS:]

            val_ghost = jnp.concatenate([val_bssn, val_matter])
            state = state.at[:, idx].set(val_ghost)

    elif bc_type == "dirichlet":
        # Dirichlet: fix ghost cells to pre-computed exact values.
        # For TOV exterior this is exact Schwarzschild in isotropic coords.
        # boundary_ref shape: (NUM_VARS, num_ghosts)
        for i in range(num_ghosts):
            idx = N - num_ghosts + i
            state = state.at[:, idx].set(boundary_ref[:, i])

    else:
        raise ValueError(f"Unknown outer_bc_type: {bc_type}")

    return state
