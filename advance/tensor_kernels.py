# tensor_kernels.py
"""
Numba-optimized tensor algebra kernels for Valencia formulation.

These kernels replace np.einsum operations with explicit loops for better performance.
All kernels maintain 3D generality (SPACEDIM=3) for future 2D/3D extensions.

Performance benefits:
- Eliminates Python function call overhead (einsum has ~92,007 calls per 1000 steps)
- No intermediate array allocations
- Cache-friendly memory access patterns
- Parallelization with prange where beneficial
"""

from numba import njit, prange
import numpy as np

SPACEDIM = 3

# Note: parallel=True is used for larger grids. For small grids (N<1000),
# the thread overhead may exceed the benefit, but for typical simulations
# with N>1000, parallelization provides speedup.


# =============================================================================
# Basic tensor operations
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def outer_product_batch(a, b, out):
    """
    Batched outer product: xi,xj->xij

    Computes out[x,i,j] = a[x,i] * b[x,j] for all x, i, j.

    Args:
        a: (N, 3) array
        b: (N, 3) array
        out: (N, 3, 3) pre-allocated output array
    """
    N = a.shape[0]
    for x in prange(N):
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                out[x, i, j] = a[x, i] * b[x, j]


@njit(cache=True, fastmath=True, parallel=True)
def dot_product_batch(a, b, out):
    """
    Batched dot product: xi,xi->x

    Computes out[x] = sum_i a[x,i] * b[x,i].

    Args:
        a: (N, 3) array
        b: (N, 3) array
        out: (N,) pre-allocated output array
    """
    N = a.shape[0]
    for x in prange(N):
        s = 0.0
        for i in range(SPACEDIM):
            s += a[x, i] * b[x, i]
        out[x] = s


@njit(cache=True, fastmath=True, parallel=True)
def matvec_batch(A, v, out):
    """
    Batched matrix-vector product: xij,xj->xi

    Computes out[x,i] = sum_j A[x,i,j] * v[x,j].

    Args:
        A: (N, 3, 3) array
        v: (N, 3) array
        out: (N, 3) pre-allocated output array
    """
    N = A.shape[0]
    for x in prange(N):
        for i in range(SPACEDIM):
            s = 0.0
            for j in range(SPACEDIM):
                s += A[x, i, j] * v[x, j]
            out[x, i] = s


@njit(cache=True, fastmath=True, parallel=True)
def vecmat_batch(v, A, out):
    """
    Batched vector-matrix product: xi,xij->xj

    Computes out[x,j] = sum_i v[x,i] * A[x,i,j].

    Args:
        v: (N, 3) array
        A: (N, 3, 3) array
        out: (N, 3) pre-allocated output array
    """
    N = A.shape[0]
    for x in prange(N):
        for j in range(SPACEDIM):
            s = 0.0
            for i in range(SPACEDIM):
                s += v[x, i] * A[x, i, j]
            out[x, j] = s


@njit(cache=True, fastmath=True, parallel=True)
def matmul_batch(A, B, out):
    """
    Batched matrix-matrix product: xik,xkj->xij

    Computes out[x,i,j] = sum_k A[x,i,k] * B[x,k,j].

    Args:
        A: (N, 3, 3) array
        B: (N, 3, 3) array
        out: (N, 3, 3) pre-allocated output array
    """
    N = A.shape[0]
    for x in prange(N):
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                s = 0.0
                for k in range(SPACEDIM):
                    s += A[x, i, k] * B[x, k, j]
                out[x, i, j] = s


@njit(cache=True, fastmath=True, parallel=True)
def frobenius_batch(A, B, out):
    """
    Batched Frobenius inner product: xij,xij->x

    Computes out[x] = sum_{i,j} A[x,i,j] * B[x,i,j].

    Args:
        A: (N, 3, 3) array
        B: (N, 3, 3) array
        out: (N,) pre-allocated output array
    """
    N = A.shape[0]
    for x in prange(N):
        s = 0.0
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                s += A[x, i, j] * B[x, i, j]
        out[x] = s


# =============================================================================
# Higher-order tensor contractions
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def tensor_contract_3d_4d(T3, T4, out):
    """
    Contraction of 3D tensor with 4D tensor: xjk,xijk->xi

    Computes out[x,i] = sum_{j,k} T3[x,j,k] * T4[x,i,j,k].

    Args:
        T3: (N, 3, 3) array
        T4: (N, 3, 3, 3) array
        out: (N, 3) pre-allocated output array
    """
    N = T3.shape[0]
    for x in prange(N):
        for i in range(SPACEDIM):
            s = 0.0
            for j in range(SPACEDIM):
                for k in range(SPACEDIM):
                    s += T3[x, j, k] * T4[x, i, j, k]
            out[x, i] = s


@njit(cache=True, fastmath=True, parallel=True)
def tensor4_vec_contract(T4, v, out):
    """
    4D tensor contracted with vector: xjik,xk->xij

    Computes out[x,i,j] = sum_k T4[x,j,i,k] * v[x,k].
    Note: index ordering is xjik to match einsum 'xjik,xk->xij'.

    Args:
        T4: (N, 3, 3, 3) array with index order [x,j,i,k]
        v: (N, 3) array
        out: (N, 3, 3) pre-allocated output array
    """
    N = T4.shape[0]
    for x in prange(N):
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                s = 0.0
                for k in range(SPACEDIM):
                    s += T4[x, j, i, k] * v[x, k]
                out[x, i, j] = s


@njit(cache=True, fastmath=True, parallel=True)
def tensor4_mat_contract(T4, M, out):
    """
    4D tensor contracted with matrix: xlji,xjl->xi

    Computes out[x,i] = sum_{j,l} T4[x,l,j,i] * M[x,j,l].
    Note: index ordering matches einsum 'xlji,xjl->xi'.

    Args:
        T4: (N, 3, 3, 3) array with index order [x,l,j,i]
        M: (N, 3, 3) array
        out: (N, 3) pre-allocated output array
    """
    N = T4.shape[0]
    for x in prange(N):
        for i in range(SPACEDIM):
            s = 0.0
            for j in range(SPACEDIM):
                for l in range(SPACEDIM):
                    s += T4[x, l, j, i] * M[x, j, l]
            out[x, i] = s


# =============================================================================
# Christoffel symbol operations
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def christoffel_trace(chris, out):
    """
    Trace of Christoffel symbols: xkkj->xj

    Computes out[x,j] = sum_k chris[x,k,k,j] (trace over first two indices).

    Args:
        chris: (N, 3, 3, 3) Christoffel symbols Gamma^i_{jk}
        out: (N, 3) pre-allocated output array
    """
    N = chris.shape[0]
    for x in prange(N):
        for j in range(SPACEDIM):
            s = 0.0
            for k in range(SPACEDIM):
                s += chris[x, k, k, j]
            out[x, j] = s


# =============================================================================
# Combined operations for specific Valencia terms
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def scalar_vec_vec_dot(s, a, b, out):
    """
    Scalar times two vector dot: x,xi,xi->x

    Computes out[x] = s[x] * sum_i a[x,i] * b[x,i].

    Args:
        s: (N,) scalar array
        a: (N, 3) first vector
        b: (N, 3) second vector
        out: (N,) pre-allocated output array
    """
    N = s.shape[0]
    for x in prange(N):
        dot = 0.0
        for i in range(SPACEDIM):
            dot += a[x, i] * b[x, i]
        out[x] = s[x] * dot


@njit(cache=True, fastmath=True, parallel=True)
def compute_tau_lapse_term(TUU_00, beta_U, dalpha, TUU_0i, out):
    """
    Combined lapse derivative term for tau source: -(T^{00} β^i + T^{0i}) ∂_i α

    Computes out[x] = -(TUU_00[x] * sum_i(beta_U[x,i] * dalpha[x,i])
                       + sum_i(TUU_0i[x,i] * dalpha[x,i]))

    Args:
        TUU_00: (N,) T^{00}
        beta_U: (N, 3) shift vector
        dalpha: (N, 3) lapse derivatives
        TUU_0i: (N, 3) T^{0i}
        out: (N,) pre-allocated output array
    """
    N = TUU_00.shape[0]
    for x in prange(N):
        term1 = 0.0
        term2 = 0.0
        for i in range(SPACEDIM):
            term1 += beta_U[x, i] * dalpha[x, i]
            term2 += TUU_0i[x, i] * dalpha[x, i]
        out[x] = -(TUU_00[x] * term1 + term2)


@njit(cache=True, fastmath=True, parallel=True)
def compute_tensor_block(TUU_00, beta_U, TUU_0i, TUU_ij, out):
    """
    Compute the tensor block used in energy/momentum sources:
    T^{00} β^i β^j + 2 T^{0i} β^j + T^{ij}

    Args:
        TUU_00: (N,) T^{00}
        beta_U: (N, 3) shift vector
        TUU_0i: (N, 3) T^{0i}
        TUU_ij: (N, 3, 3) T^{ij}
        out: (N, 3, 3) pre-allocated output array
    """
    N = TUU_00.shape[0]
    for x in prange(N):
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                out[x, i, j] = (TUU_00[x] * beta_U[x, i] * beta_U[x, j]
                              + 2.0 * TUU_0i[x, i] * beta_U[x, j]
                              + TUU_ij[x, i, j])


# =============================================================================
# Wrapper functions that allocate output (for easier API use)
# =============================================================================

def outer_product_batch_alloc(a, b):
    """Allocating version of outer_product_batch."""
    out = np.empty((a.shape[0], SPACEDIM, SPACEDIM), dtype=a.dtype)
    outer_product_batch(a, b, out)
    return out


def dot_product_batch_alloc(a, b):
    """Allocating version of dot_product_batch."""
    out = np.empty(a.shape[0], dtype=a.dtype)
    dot_product_batch(a, b, out)
    return out


def matvec_batch_alloc(A, v):
    """Allocating version of matvec_batch."""
    out = np.empty((A.shape[0], SPACEDIM), dtype=A.dtype)
    matvec_batch(A, v, out)
    return out


def vecmat_batch_alloc(v, A):
    """Allocating version of vecmat_batch."""
    out = np.empty((A.shape[0], SPACEDIM), dtype=A.dtype)
    vecmat_batch(v, A, out)
    return out


def matmul_batch_alloc(A, B):
    """Allocating version of matmul_batch."""
    out = np.empty((A.shape[0], SPACEDIM, SPACEDIM), dtype=A.dtype)
    matmul_batch(A, B, out)
    return out


def frobenius_batch_alloc(A, B):
    """Allocating version of frobenius_batch."""
    out = np.empty(A.shape[0], dtype=A.dtype)
    frobenius_batch(A, B, out)
    return out
