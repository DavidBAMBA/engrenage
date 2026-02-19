# bssngeometry.py
#
# JAX pytrees for pre-computed background quantities and derivative matrices.
# These are constant during evolution and transferred to JAX arrays once.

import jax
import jax.numpy as jnp
import numpy as np


class BSSNBackground:
    """
    Pre-computed background quantities from FlatSphericalBackground,
    transferred to JAX arrays. Constant during evolution.

    Registered as a JAX pytree: arrays are dynamic (traced),
    no static aux_data needed.
    """
    def __init__(self, r, hat_gamma_LL, hat_christoffel, d1_hat_christoffel,
                 scaling_vector, inverse_scaling_vector,
                 d1_scaling_vector, d1_inverse_scaling_vector,
                 d2_scaling_vector, d2_inverse_scaling_vector,
                 scaling_matrix, inverse_scaling_matrix,
                 d1_scaling_matrix, d2_scaling_matrix,
                 det_hat_gamma, d1_det_hat_gamma, d2_det_hat_gamma):
        self.r = r
        self.hat_gamma_LL = hat_gamma_LL
        self.hat_christoffel = hat_christoffel
        self.d1_hat_christoffel = d1_hat_christoffel
        self.scaling_vector = scaling_vector
        self.inverse_scaling_vector = inverse_scaling_vector
        self.d1_scaling_vector = d1_scaling_vector
        self.d1_inverse_scaling_vector = d1_inverse_scaling_vector
        self.d2_scaling_vector = d2_scaling_vector
        self.d2_inverse_scaling_vector = d2_inverse_scaling_vector
        self.scaling_matrix = scaling_matrix
        self.inverse_scaling_matrix = inverse_scaling_matrix
        self.d1_scaling_matrix = d1_scaling_matrix
        self.d2_scaling_matrix = d2_scaling_matrix
        self.det_hat_gamma = det_hat_gamma
        self.d1_det_hat_gamma = d1_det_hat_gamma
        self.d2_det_hat_gamma = d2_det_hat_gamma

    def tree_flatten(self):
        children = (
            self.r, self.hat_gamma_LL, self.hat_christoffel, self.d1_hat_christoffel,
            self.scaling_vector, self.inverse_scaling_vector,
            self.d1_scaling_vector, self.d1_inverse_scaling_vector,
            self.d2_scaling_vector, self.d2_inverse_scaling_vector,
            self.scaling_matrix, self.inverse_scaling_matrix,
            self.d1_scaling_matrix, self.d2_scaling_matrix,
            self.det_hat_gamma, self.d1_det_hat_gamma, self.d2_det_hat_gamma,
        )
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


jax.tree_util.register_pytree_node(
    BSSNBackground,
    BSSNBackground.tree_flatten,
    BSSNBackground.tree_unflatten,
)


class DerivativeMatrices:
    """
    Pre-computed derivative matrices from Grid.derivs, transferred to JAX arrays.
    Constant during evolution.

    Contains:
      - d1_matrix: (N, N) 4th order first derivative
      - d2_matrix: (N, N) 4th order second derivative
      - advec_l_matrix: (N, N) 3rd order left-biased (backward) advection
      - advec_r_matrix: (N, N) 3rd order right-biased (forward) advection
      - ko_matrix: (N, N) 6th order Kreiss-Oliger dissipation
    """
    def __init__(self, d1_matrix, d2_matrix, advec_l_matrix, advec_r_matrix, ko_matrix):
        self.d1_matrix = d1_matrix
        self.d2_matrix = d2_matrix
        self.advec_l_matrix = advec_l_matrix
        self.advec_r_matrix = advec_r_matrix
        self.ko_matrix = ko_matrix

    def tree_flatten(self):
        children = (
            self.d1_matrix, self.d2_matrix,
            self.advec_l_matrix, self.advec_r_matrix,
            self.ko_matrix,
        )
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


jax.tree_util.register_pytree_node(
    DerivativeMatrices,
    DerivativeMatrices.tree_flatten,
    DerivativeMatrices.tree_unflatten,
)


def build_bssn_background(grid, background):
    """
    Transfer all FlatSphericalBackground arrays to JAX.

    Args:
        grid: Grid object (for r coordinate)
        background: FlatSphericalBackground instance

    Returns:
        BSSNBackground pytree with all arrays as jnp arrays
    """
    return BSSNBackground(
        r=jnp.array(grid.r),
        hat_gamma_LL=jnp.array(background.hat_gamma_LL),
        hat_christoffel=jnp.array(background.hat_christoffel),
        d1_hat_christoffel=jnp.array(background.d1_hat_christoffel),
        scaling_vector=jnp.array(background.scaling_vector),
        inverse_scaling_vector=jnp.array(background.inverse_scaling_vector),
        d1_scaling_vector=jnp.array(background.d1_scaling_vector),
        d1_inverse_scaling_vector=jnp.array(background.d1_inverse_scaling_vector),
        d2_scaling_vector=jnp.array(background.d2_scaling_vector),
        d2_inverse_scaling_vector=jnp.array(background.d2_inverse_scaling_vector),
        scaling_matrix=jnp.array(background.scaling_matrix),
        inverse_scaling_matrix=jnp.array(background.inverse_scaling_matrix),
        d1_scaling_matrix=jnp.array(background.d1_scaling_matrix),
        d2_scaling_matrix=jnp.array(background.d2_scaling_matrix),
        det_hat_gamma=jnp.array(background.det_hat_gamma),
        d1_det_hat_gamma=jnp.array(background.d1_det_hat_gamma),
        d2_det_hat_gamma=jnp.array(background.d2_det_hat_gamma),
    )


def build_derivative_matrices(grid):
    """
    Transfer pre-computed derivative matrices from Grid to JAX arrays.

    Args:
        grid: Grid object with derivs attribute

    Returns:
        DerivativeMatrices pytree with all matrices as jnp arrays
    """
    return DerivativeMatrices(
        d1_matrix=jnp.array(grid.derivs.drn_matrix[1]),
        d2_matrix=jnp.array(grid.derivs.drn_matrix[2]),
        advec_l_matrix=jnp.array(grid.derivs.advec_x_matrix[0]),
        advec_r_matrix=jnp.array(grid.derivs.advec_x_matrix[1]),
        ko_matrix=jnp.array(grid.derivs.drn_matrix[6]),
    )


class DerivativeStencils:
    """
    Compact stencil coefficients extracted from banded derivative matrices.
    Replaces DerivativeMatrices for O(N*K) computation instead of O(N^2).

    Memory savings: (N, K) with K=4-7 instead of (N, N).
    For N=4000: ~800 KB vs ~640 MB (800x reduction).

    Contains:
      - d1_stencils: (N, 5) 4th-order first derivative, offsets [-2,-1,0,+1,+2]
      - d2_stencils: (N, 5) 4th-order second derivative (position-dependent)
      - advec_l_stencils: (N, 4) 3rd-order backward advection, offsets [-3,-2,-1,0]
      - advec_r_stencils: (N, 4) 3rd-order forward advection, offsets [0,+1,+2,+3]
      - ko_stencils: (N, 7) 6th-order KO dissipation, offsets [-3,...,+3]
    """
    def __init__(self, d1_stencils, d2_stencils, advec_l_stencils,
                 advec_r_stencils, ko_stencils):
        self.d1_stencils = d1_stencils
        self.d2_stencils = d2_stencils
        self.advec_l_stencils = advec_l_stencils
        self.advec_r_stencils = advec_r_stencils
        self.ko_stencils = ko_stencils

    def tree_flatten(self):
        children = (
            self.d1_stencils, self.d2_stencils,
            self.advec_l_stencils, self.advec_r_stencils,
            self.ko_stencils,
        )
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


jax.tree_util.register_pytree_node(
    DerivativeStencils,
    DerivativeStencils.tree_flatten,
    DerivativeStencils.tree_unflatten,
)


def build_derivative_stencils(grid):
    """
    Extract compact stencil coefficients from derivative matrices
    and transfer to JAX arrays.

    Args:
        grid: Grid object with derivs attribute

    Returns:
        DerivativeStencils pytree with (N, K) stencil arrays
    """
    stencils = grid.derivs.get_stencils()
    return DerivativeStencils(
        d1_stencils=jnp.array(stencils['d1']),
        d2_stencils=jnp.array(stencils['d2']),
        advec_l_stencils=jnp.array(stencils['advec_l']),
        advec_r_stencils=jnp.array(stencils['advec_r']),
        ko_stencils=jnp.array(stencils['ko']),
    )
