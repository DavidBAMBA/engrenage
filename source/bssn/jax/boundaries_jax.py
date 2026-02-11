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


def fill_bssn_boundaries_jax(state, r, num_ghosts, parity, asymp_power, asymp_offset):
    """
    Apply boundary conditions to the full state (NUM_VARS, N).

    Inner boundary (r=0): parity conditions.
    Outer boundary: asymptotic extrapolation for BSSN vars,
                    zero-gradient (outflow) for matter vars.

    Args:
        state: (NUM_VARS, N) full state array
        r: (N,) radial coordinate
        num_ghosts: int, number of ghost cells (3)
        parity: (NUM_VARS,) parity array (+1 or -1)
        asymp_power: (NUM_VARS,) asymptotic power law exponents
        asymp_offset: (NUM_VARS,) asymptotic offsets

    Returns:
        state: (NUM_VARS, N) with boundary conditions applied
    """
    state = _fill_inner_boundary(state, num_ghosts, parity)
    state = _fill_outer_boundary(state, r, num_ghosts, asymp_power, asymp_offset)
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


def _fill_outer_boundary(state, r, num_ghosts, asymp_power, asymp_offset):
    """
    Outer boundary: asymptotic extrapolation for BSSN vars,
    zero-gradient for matter vars (indices >= NUM_BSSN_VARS).

    BSSN: var = offset + b * r^power, where b is determined from last interior point.
    Matter: copy last interior point value.
    """
    N = state.shape[1]
    idx_ref = N - num_ghosts - 1  # last interior point

    # Reference values for extrapolation
    r_ref = r[idx_ref]
    val_ref = state[:, idx_ref]

    # BSSN vars: asymptotic extrapolation
    bssn_val_ref = val_ref[:NUM_BSSN_VARS]
    bssn_asymp_power = asymp_power[:NUM_BSSN_VARS]
    bssn_asymp_offset = asymp_offset[:NUM_BSSN_VARS]
    b_bssn = (bssn_val_ref - bssn_asymp_offset) / (r_ref ** bssn_asymp_power)

    # Matter vars: zero-gradient
    matter_val_ref = val_ref[NUM_BSSN_VARS:]

    for i in range(num_ghosts):
        idx = N - num_ghosts + i
        bssn_vals = bssn_asymp_offset + b_bssn * r[idx] ** bssn_asymp_power
        new_col = jnp.concatenate([bssn_vals, matter_val_ref])
        state = state.at[:, idx].set(new_col)

    return state
