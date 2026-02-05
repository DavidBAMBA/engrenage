# tensoralgebra_kernels.py
"""
Numba-optimized kernels for BSSN tensor algebra operations used by hydro.
These kernels replace np.einsum and np.linalg.inv for better performance.
"""

import numpy as np
from numba import jit, prange

SPACEDIM = 3


@jit(nopython=True, cache=True, fastmath=True)
def inv_3x3_kernel(A, A_inv):
    """
    Compute inverse of N 3x3 matrices.

    Parameters
    ----------
    A : ndarray (N, 3, 3)
        Input matrices
    A_inv : ndarray (N, 3, 3)
        Output inverse matrices (modified in-place)
    """
    N = A.shape[0]
    for n in range(N):
        # Compute determinant using Sarrus rule
        a = A[n, 0, 0]
        b = A[n, 0, 1]
        c = A[n, 0, 2]
        d = A[n, 1, 0]
        e = A[n, 1, 1]
        f = A[n, 1, 2]
        g = A[n, 2, 0]
        h = A[n, 2, 1]
        i = A[n, 2, 2]

        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

        if abs(det) < 1e-30:
            det = 1e-30  # Avoid division by zero

        inv_det = 1.0 / det

        # Compute adjugate matrix divided by determinant
        A_inv[n, 0, 0] = (e * i - f * h) * inv_det
        A_inv[n, 0, 1] = (c * h - b * i) * inv_det
        A_inv[n, 0, 2] = (b * f - c * e) * inv_det
        A_inv[n, 1, 0] = (f * g - d * i) * inv_det
        A_inv[n, 1, 1] = (a * i - c * g) * inv_det
        A_inv[n, 1, 2] = (c * d - a * f) * inv_det
        A_inv[n, 2, 0] = (d * h - e * g) * inv_det
        A_inv[n, 2, 1] = (b * g - a * h) * inv_det
        A_inv[n, 2, 2] = (a * e - b * d) * inv_det


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def inv_3x3_kernel_parallel(A, A_inv):
    """
    Compute inverse of N 3x3 matrices (parallel version).

    Parameters
    ----------
    A : ndarray (N, 3, 3)
        Input matrices
    A_inv : ndarray (N, 3, 3)
        Output inverse matrices (modified in-place)
    """
    N = A.shape[0]
    for n in prange(N):
        # Compute determinant using Sarrus rule
        a = A[n, 0, 0]
        b = A[n, 0, 1]
        c = A[n, 0, 2]
        d = A[n, 1, 0]
        e = A[n, 1, 1]
        f = A[n, 1, 2]
        g = A[n, 2, 0]
        h = A[n, 2, 1]
        i = A[n, 2, 2]

        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

        if abs(det) < 1e-30:
            det = 1e-30  # Avoid division by zero

        inv_det = 1.0 / det

        # Compute adjugate matrix divided by determinant
        A_inv[n, 0, 0] = (e * i - f * h) * inv_det
        A_inv[n, 0, 1] = (c * h - b * i) * inv_det
        A_inv[n, 0, 2] = (b * f - c * e) * inv_det
        A_inv[n, 1, 0] = (f * g - d * i) * inv_det
        A_inv[n, 1, 1] = (a * i - c * g) * inv_det
        A_inv[n, 1, 2] = (c * d - a * f) * inv_det
        A_inv[n, 2, 0] = (d * h - e * g) * inv_det
        A_inv[n, 2, 1] = (b * g - a * h) * inv_det
        A_inv[n, 2, 2] = (a * e - b * d) * inv_det


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def hat_D_bar_gamma_LL_kernel(N, d1_h_dx, h_LL, scaling_matrix,
                               d1_scaling_matrix, hat_christoffel,
                               hat_D_epsilon):
    """
    Compute covariant derivative of conformal metric: hat{D}_k bar{gamma}_{ij}

    This replaces:
        hat_D_epsilon += d1_h_dx * background.scaling_matrix[:,:,:,np.newaxis]
        hat_D_epsilon += background.d1_scaling_matrix * h_LL[:,:,:,np.newaxis]
        hat_D_epsilon += - (np.einsum('xlik,xlj->xijk', hat_christoffel, epsilon_LL)
                          + np.einsum('xljk,xil->xijk', hat_christoffel, epsilon_LL))

    Parameters
    ----------
    N : int
        Number of grid points
    d1_h_dx : ndarray (N, 3, 3, 3)
        Derivative of h_LL
    h_LL : ndarray (N, 3, 3)
        Rescaled metric deviation
    scaling_matrix : ndarray (N, 3, 3)
        Scaling matrix s_{ij}
    d1_scaling_matrix : ndarray (N, 3, 3, 3)
        Derivative of scaling matrix
    hat_christoffel : ndarray (N, 3, 3, 3)
        Reference metric Christoffel symbols
    hat_D_epsilon : ndarray (N, 3, 3, 3)
        Output array (modified in-place)
    """
    for n in prange(N):
        # Compute epsilon_LL = h_LL * scaling_matrix
        epsilon_LL = np.empty((SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                epsilon_LL[i, j] = h_LL[n, i, j] * scaling_matrix[n, i, j]

        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                for k in range(SPACEDIM):
                    # First term: d1_h_dx * scaling_matrix
                    val = d1_h_dx[n, i, j, k] * scaling_matrix[n, i, j]

                    # Second term: d1_scaling_matrix * h_LL
                    val += d1_scaling_matrix[n, i, j, k] * h_LL[n, i, j]

                    # Third term: -einsum('xlik,xlj->xijk', hat_christoffel, epsilon_LL)
                    # = -sum_l hat_christoffel[l,i,k] * epsilon_LL[l,j]
                    for l in range(SPACEDIM):
                        val -= hat_christoffel[n, l, i, k] * epsilon_LL[l, j]

                    # Fourth term: -einsum('xljk,xil->xijk', hat_christoffel, epsilon_LL)
                    # = -sum_l hat_christoffel[l,j,k] * epsilon_LL[i,l]
                    for l in range(SPACEDIM):
                        val -= hat_christoffel[n, l, j, k] * epsilon_LL[i, l]

                    hat_D_epsilon[n, i, j, k] = val


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def get_bar_gamma_LL_kernel(N, h_LL, scaling_matrix, hat_gamma_LL, bar_gamma_LL):
    """
    Compute conformal metric: bar_gamma_LL = h_LL * scaling_matrix + hat_gamma_LL

    Parameters
    ----------
    N : int
        Number of grid points
    h_LL : ndarray (N, 3, 3)
        Rescaled metric deviation
    scaling_matrix : ndarray (N, 3, 3)
        Scaling matrix
    hat_gamma_LL : ndarray (N, 3, 3)
        Reference metric
    bar_gamma_LL : ndarray (N, 3, 3)
        Output array (modified in-place)
    """
    for n in prange(N):
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                bar_gamma_LL[n, i, j] = h_LL[n, i, j] * scaling_matrix[n, i, j] + hat_gamma_LL[n, i, j]


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def get_bar_A_LL_kernel(N, a_LL, scaling_matrix, bar_A_LL):
    """
    Compute conformal traceless extrinsic curvature: bar_A_LL = a_LL * scaling_matrix

    Parameters
    ----------
    N : int
        Number of grid points
    a_LL : ndarray (N, 3, 3)
        Rescaled extrinsic curvature deviation
    scaling_matrix : ndarray (N, 3, 3)
        Scaling matrix
    bar_A_LL : ndarray (N, 3, 3)
        Output array (modified in-place)
    """
    for n in prange(N):
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                bar_A_LL[n, i, j] = a_LL[n, i, j] * scaling_matrix[n, i, j]


# Convenience wrapper functions that allocate output and call kernels

def inv_3x3(A):
    """
    Compute inverse of N 3x3 matrices using Numba kernel.

    Parameters
    ----------
    A : ndarray (N, 3, 3)
        Input matrices

    Returns
    -------
    A_inv : ndarray (N, 3, 3)
        Inverse matrices
    """
    N = A.shape[0]
    A_inv = np.empty_like(A)
    if N > 100:
        inv_3x3_kernel_parallel(A, A_inv)
    else:
        inv_3x3_kernel(A, A_inv)
    return A_inv


def compute_hat_D_bar_gamma_LL(r, h_LL, d1_h_dx, background):
    """
    Compute covariant derivative of conformal metric using Numba kernel.

    Replaces get_hat_D_bar_gamma_LL from tensoralgebra.py with optimized version.

    Parameters
    ----------
    r : ndarray (N,)
        Radial coordinates
    h_LL : ndarray (N, 3, 3)
        Rescaled metric deviation
    d1_h_dx : ndarray (N, 3, 3, 3)
        Derivative of h_LL
    background : Background object
        Contains scaling_matrix, d1_scaling_matrix, hat_christoffel

    Returns
    -------
    hat_D_epsilon : ndarray (N, 3, 3, 3)
        Covariant derivative of conformal metric
    """
    N = len(r)
    hat_D_epsilon = np.empty((N, SPACEDIM, SPACEDIM, SPACEDIM))

    hat_D_bar_gamma_LL_kernel(
        N, d1_h_dx, h_LL,
        background.scaling_matrix,
        background.d1_scaling_matrix,
        background.hat_christoffel,
        hat_D_epsilon
    )

    return hat_D_epsilon
