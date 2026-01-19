"""Flat Cartesian background for pure advection tests."""

import numpy as np

from source.bssn.tensoralgebra import SPACEDIM


class FlatCartesianBackground:
    """
    Flat Cartesian background for testing purposes.

    In Cartesian coordinates, all geometric quantities simplify:
    - Metric: ĝ_ij = δ_ij (identity)
    - Christoffel: Γ̂^i_{jk} = 0 (flat space)
    - Determinant: det(ĝ) = 1
    - Scaling factors: all = 1

    This gives pure advection without geometric source terms:
    ∂ρ/∂t + v ∂ρ/∂x = 0
    """

    def __init__(self, x):
        """
        Initialize flat Cartesian background.

        Parameters
        ----------
        x : ndarray
            Coordinate array (treated as Cartesian x-coordinate)
        """
        self.r = x  # Keep 'r' for compatibility with existing code
        self.N = len(x)
        N = self.N

        # Scaling vectors: all ones (no coordinate rescaling)
        self.scaling_vector = np.ones((N, SPACEDIM))
        self.inverse_scaling_vector = np.ones((N, SPACEDIM))
        self.d1_scaling_vector = np.zeros((N, SPACEDIM, SPACEDIM))
        self.d1_inverse_scaling_vector = np.zeros((N, SPACEDIM, SPACEDIM))
        self.d2_scaling_vector = np.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM))
        self.d2_inverse_scaling_vector = np.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM))

        # Scaling matrices: identity (s_ij = s_i * s_j = 1)
        self.scaling_matrix = np.broadcast_to(
            np.eye(SPACEDIM), (N, SPACEDIM, SPACEDIM)
        ).copy()
        self.inverse_scaling_matrix = self.scaling_matrix.copy()
        self.d1_scaling_matrix = np.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM))
        self.d2_scaling_matrix = np.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM, SPACEDIM))

        # Hat metric: identity (flat Cartesian)
        self.hat_gamma_LL = np.broadcast_to(
            np.eye(SPACEDIM), (N, SPACEDIM, SPACEDIM)
        ).copy()

        # Christoffel symbols: all zero (flat space)
        self.hat_christoffel = np.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM))
        self.d1_hat_christoffel = np.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM, SPACEDIM))

        # Determinant: constant = 1
        self.det_hat_gamma = np.ones(N)
        self.d1_det_hat_gamma = np.zeros((N, SPACEDIM))
        self.d2_det_hat_gamma = np.zeros((N, SPACEDIM, SPACEDIM))
