"""
JAX-based High-Order Reconstruction Methods.

Provides WENO-Z reconstruction using JAX for GPU acceleration.
API-compatible with reconstruction.py.

Usage:
    Set ENGRENAGE_BACKEND=jax before importing, or use directly:

    from source.matter.hydro.reconstruction_jax import ReconstructionJAX
    recon = ReconstructionJAX(method="wenoz")
    u_L, u_R = recon.reconstruct(u, dx=dx)
"""

import jax
# Enable float64 precision (JAX defaults to float32)
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, vmap
import numpy as np


# ==============================================================================
# JAX RECONSTRUCTION KERNELS
# ==============================================================================

@jit
def wenoz_face_kernel_jax(um2, um1, u0, up1, up2):
    """
    WENO-Z face reconstruction kernel (single point, JAX version).

    Fifth-order WENO-Z scheme with tau enhancement.
    """
    # Smoothness indicators
    beta0 = (13.0/12.0) * (um2 - 2.0*um1 + u0)**2 + 0.25 * (um2 - 4.0*um1 + 3.0*u0)**2
    beta1 = (13.0/12.0) * (um1 - 2.0*u0 + up1)**2 + 0.25 * (um1 - up1)**2
    beta2 = (13.0/12.0) * (u0 - 2.0*up1 + up2)**2 + 0.25 * (3.0*u0 - 4.0*up1 + up2)**2

    # Tau indicator - key improvement of WENO-Z
    tau5 = jnp.abs(beta0 - beta2)

    # Ideal weights
    d0, d1, d2 = 0.1, 0.6, 0.3
    eps = 1.0e-6

    # Modified alpha weights with tau enhancement
    alpha0 = d0 * (1.0 + (tau5 / (beta0 + eps))**2)
    alpha1 = d1 * (1.0 + (tau5 / (beta1 + eps))**2)
    alpha2 = d2 * (1.0 + (tau5 / (beta2 + eps))**2)

    # Normalized weights
    sum_alpha = alpha0 + alpha1 + alpha2
    w0 = alpha0 / sum_alpha
    w1 = alpha1 / sum_alpha
    w2 = alpha2 / sum_alpha

    # Candidate polynomials
    p0 = (2.0*um2 - 7.0*um1 + 11.0*u0) / 6.0
    p1 = (-um1 + 5.0*u0 + 2.0*up1) / 6.0
    p2 = (2.0*u0 + 5.0*up1 - up2) / 6.0

    return w0*p0 + w1*p1 + w2*p2


@jit
def weno5_face_kernel_jax(um2, um1, u0, up1, up2):
    """
    WENO5 face reconstruction kernel (single point, JAX version).

    Fifth-order WENO (standard version without tau).
    """
    # Smoothness indicators
    beta0 = (13.0/12.0) * (um2 - 2.0*um1 + u0)**2 + 0.25 * (um2 - 4.0*um1 + 3.0*u0)**2
    beta1 = (13.0/12.0) * (um1 - 2.0*u0 + up1)**2 + 0.25 * (um1 - up1)**2
    beta2 = (13.0/12.0) * (u0 - 2.0*up1 + up2)**2 + 0.25 * (3.0*u0 - 4.0*up1 + up2)**2

    # Ideal weights
    d0, d1, d2 = 0.1, 0.6, 0.3
    eps = 1.0e-6

    # Alpha weights
    alpha0 = d0 / (eps + beta0)**2
    alpha1 = d1 / (eps + beta1)**2
    alpha2 = d2 / (eps + beta2)**2

    # Normalized weights
    sum_alpha = alpha0 + alpha1 + alpha2
    w0 = alpha0 / sum_alpha
    w1 = alpha1 / sum_alpha
    w2 = alpha2 / sum_alpha

    # Candidate polynomials
    p0 = (2.0*um2 - 7.0*um1 + 11.0*u0) / 6.0
    p1 = (-um1 + 5.0*u0 + 2.0*up1) / 6.0
    p2 = (2.0*u0 + 5.0*up1 - up2) / 6.0

    return w0*p0 + w1*p1 + w2*p2


@jit
def minmod_jax(a, b):
    """Two-argument minmod function (JAX version)."""
    return jnp.where(
        a * b <= 0.0,
        0.0,
        jnp.where(jnp.abs(a) < jnp.abs(b), a, b)
    )


@jit
def minmod3_jax(a, b, c):
    """
    Three-argument minmod function (JAX version).
    Returns the value with smallest absolute magnitude if all have same sign, else 0.
    """
    # All must have same sign
    same_sign = (a * b > 0.0) & (a * c > 0.0)

    abs_a = jnp.abs(a)
    abs_b = jnp.abs(b)
    abs_c = jnp.abs(c)

    # Find minimum absolute value
    min_ab = jnp.where(abs_a <= abs_b, a, b)
    abs_min_ab = jnp.where(abs_a <= abs_b, abs_a, abs_b)
    result = jnp.where(abs_min_ab <= abs_c, min_ab, c)

    return jnp.where(same_sign, result, 0.0)


@jit
def mc_jax(a, b):
    """
    MC (Monotonized Central) limiter for two slopes (JAX version).

    MC(a, b) = minmod(2a, 2b, (a+b)/2)
    """
    return minmod3_jax(2.0 * a, 2.0 * b, 0.5 * (a + b))


@jit
def minmod4_jax(w, x, y, z):
    """Four-argument minmod function (JAX version)."""
    s = 0.125 * (jnp.sign(w) + jnp.sign(x)) * jnp.abs(
        (jnp.sign(w) + jnp.sign(y)) * (jnp.sign(w) + jnp.sign(z)))
    return s * jnp.minimum(jnp.minimum(jnp.abs(w), jnp.abs(x)),
                           jnp.minimum(jnp.abs(y), jnp.abs(z)))


@jit
def mp5_face_kernel_jax(um2, um1, u0, up1, up2):
    """
    MP5 face reconstruction kernel for a single point (JAX version).

    Fifth-order monotonicity-preserving scheme.
    Good balance of accuracy and robustness.

    Returns:
        (uL, uR): Left state at i+1/2 and right state at i-1/2
    """
    # Parameters
    alpha = 4.0
    eps = 1e-10

    # Norm for switching criterion
    vnorm = jnp.sqrt(um2**2 + um1**2 + u0**2 + up1**2 + up2**2) + 1e-30

    # Left state at i+1/2 (from cell i towards right)
    uL_base = (2.0*um2 - 13.0*um1 + 47.0*u0 + 27.0*up1 - 3.0*up2) / 60.0
    vmp = u0 + minmod_jax(up1 - u0, alpha * (u0 - um1))

    # Switching criterion
    cond_L = (uL_base - u0) * (uL_base - vmp) <= eps * vnorm

    # Second derivatives for left state
    djm1 = um2 - 2.0*um1 + u0
    dj = um1 - 2.0*u0 + up1
    djp1 = u0 - 2.0*up1 + up2

    # Fourth-order differences
    dm4jmh = minmod4_jax(4.0*dj - djm1, 4.0*djm1 - dj, dj, djm1)
    dm4jph = minmod4_jax(4.0*dj - djp1, 4.0*djp1 - dj, dj, djp1)

    # Additional terms
    vul = u0 + alpha * (u0 - um1)
    vav = 0.5 * (u0 + up1)
    vmd = vav - 0.5 * dm4jph
    vlc = u0 + 0.5 * (u0 - um1) + (4.0/3.0) * dm4jmh

    # Min/max bounds
    vminl = jnp.maximum(jnp.minimum(jnp.minimum(u0, up1), vmd),
                        jnp.minimum(jnp.minimum(u0, vul), vlc))
    vmaxl = jnp.minimum(jnp.maximum(jnp.maximum(u0, up1), vmd),
                        jnp.maximum(jnp.maximum(u0, vul), vlc))

    # Apply bounds
    uL_bounded = uL_base + minmod_jax(vminl - uL_base, vmaxl - uL_base)
    uL = jnp.where(cond_L, uL_base, uL_bounded)

    # Right state at i-1/2 (symmetric, reversed stencil)
    uR_base = (2.0*up2 - 13.0*up1 + 47.0*u0 + 27.0*um1 - 3.0*um2) / 60.0
    vmp_r = u0 + minmod_jax(um1 - u0, alpha * (u0 - up1))

    cond_R = (uR_base - u0) * (uR_base - vmp_r) <= eps * vnorm

    # Second derivatives for right state (reversed)
    djm1_r = up2 - 2.0*up1 + u0
    dj_r = up1 - 2.0*u0 + um1
    djp1_r = u0 - 2.0*um1 + um2

    dm4jmh_r = minmod4_jax(4.0*dj_r - djm1_r, 4.0*djm1_r - dj_r, dj_r, djm1_r)
    dm4jph_r = minmod4_jax(4.0*dj_r - djp1_r, 4.0*djp1_r - dj_r, dj_r, djp1_r)

    vul_r = u0 + alpha * (u0 - up1)
    vav_r = 0.5 * (u0 + um1)
    vmd_r = vav_r - 0.5 * dm4jph_r
    vlc_r = u0 + 0.5 * (u0 - up1) + (4.0/3.0) * dm4jmh_r

    vminr = jnp.maximum(jnp.minimum(jnp.minimum(u0, um1), vmd_r),
                        jnp.minimum(jnp.minimum(u0, vul_r), vlc_r))
    vmaxr = jnp.minimum(jnp.maximum(jnp.maximum(u0, um1), vmd_r),
                        jnp.maximum(jnp.maximum(u0, vul_r), vlc_r))

    uR_bounded = uR_base + minmod_jax(vminr - uR_base, vmaxr - uR_base)
    uR = jnp.where(cond_R, uR_base, uR_bounded)

    return uL, uR


# ==============================================================================
# VECTORIZED RECONSTRUCTION FUNCTIONS
# ==============================================================================

@jit
def wenoz_reconstruct_jax(u):
    """
    WENO-Z reconstruction for all interior points (JAX version).

    Args:
        u: Input array (cell centers, length N)

    Returns:
        (u_L_interior, u_R_interior): Left and right states at interior interfaces
    """
    N = u.shape[0]

    # Create stencil indices for interior points (2 to N-3)
    # Each interior point i gets stencil [i-2, i-1, i, i+1, i+2]
    idx = jnp.arange(2, N-2)

    # Extract stencils using advanced indexing
    um2 = u[idx - 2]
    um1 = u[idx - 1]
    u0 = u[idx]
    up1 = u[idx + 1]
    up2 = u[idx + 2]

    # Apply kernel to all points using vmap
    u_L_interior = vmap(wenoz_face_kernel_jax)(um2, um1, u0, up1, up2)
    u_R_interior = vmap(wenoz_face_kernel_jax)(up2, up1, u0, um1, um2)

    return u_L_interior, u_R_interior


@jit
def weno5_reconstruct_jax(u):
    """
    WENO5 reconstruction for all interior points (JAX version).
    """
    N = u.shape[0]
    idx = jnp.arange(2, N-2)

    um2 = u[idx - 2]
    um1 = u[idx - 1]
    u0 = u[idx]
    up1 = u[idx + 1]
    up2 = u[idx + 2]

    u_L_interior = vmap(weno5_face_kernel_jax)(um2, um1, u0, up1, up2)
    u_R_interior = vmap(weno5_face_kernel_jax)(up2, up1, u0, um1, um2)

    return u_L_interior, u_R_interior


@jit
def minmod_reconstruct_jax(u, dx):
    """
    Minmod TVD reconstruction for all interior points (JAX version).
    """
    N = u.shape[0]
    idx = jnp.arange(1, N-1)

    # Compute slopes
    slope_L = (u[idx] - u[idx - 1]) / dx
    slope_R = (u[idx + 1] - u[idx]) / dx

    # Limited slopes (vectorized minmod)
    slopes = vmap(minmod_jax)(slope_L, slope_R)

    # Reconstruct
    u_R_interior = u[idx] - 0.5 * slopes * dx  # Right state at i-1/2
    u_L_interior = u[idx] + 0.5 * slopes * dx  # Left state at i+1/2

    return u_L_interior, u_R_interior


@jit
def mc_reconstruct_jax(u, dx):
    """
    MC (Monotonized Central) reconstruction for all interior points (JAX version).
    """
    N = u.shape[0]
    idx = jnp.arange(1, N-1)

    # Compute slopes
    slope_L = (u[idx] - u[idx - 1]) / dx
    slope_R = (u[idx + 1] - u[idx]) / dx

    # Limited slopes (vectorized MC limiter)
    slopes = vmap(mc_jax)(slope_L, slope_R)

    # Reconstruct
    u_R_interior = u[idx] - 0.5 * slopes * dx  # Right state at i-1/2
    u_L_interior = u[idx] + 0.5 * slopes * dx  # Left state at i+1/2

    return u_L_interior, u_R_interior


@jit
def mp5_reconstruct_jax(u):
    """
    MP5 reconstruction for all interior points (JAX version).
    """
    N = u.shape[0]
    idx = jnp.arange(2, N-2)

    um2 = u[idx - 2]
    um1 = u[idx - 1]
    u0 = u[idx]
    up1 = u[idx + 1]
    up2 = u[idx + 2]

    # vmap returns (uL_array, uR_array) tuple
    u_L_interior, u_R_interior = vmap(mp5_face_kernel_jax)(um2, um1, u0, up1, up2)

    return u_L_interior, u_R_interior


@jit
def wenoz_reconstruct_3vars_jax(rho, vr, p):
    """
    OPTIMIZED: Reconstruct 3 primitive variables in single pass (JAX version).

    Returns:
        (rho_L, rho_R, vr_L, vr_R, p_L, p_R) for interior points
    """
    N = rho.shape[0]
    idx = jnp.arange(2, N-2)

    # Extract stencils for all 3 variables
    def get_stencil(arr):
        return (arr[idx - 2], arr[idx - 1], arr[idx], arr[idx + 1], arr[idx + 2])

    rho_stencil = get_stencil(rho)
    vr_stencil = get_stencil(vr)
    p_stencil = get_stencil(p)

    # Apply WENO-Z kernel to each variable
    def apply_wenoz(stencil):
        um2, um1, u0, up1, up2 = stencil
        u_L = vmap(wenoz_face_kernel_jax)(um2, um1, u0, up1, up2)
        u_R = vmap(wenoz_face_kernel_jax)(up2, up1, u0, um1, um2)
        return u_L, u_R

    rho_L, rho_R = apply_wenoz(rho_stencil)
    vr_L, vr_R = apply_wenoz(vr_stencil)
    p_L, p_R = apply_wenoz(p_stencil)

    return rho_L, rho_R, vr_L, vr_R, p_L, p_R


@jit
def weno5_reconstruct_3vars_jax(rho, vr, p):
    """
    OPTIMIZED: Reconstruct 3 primitive variables in single pass (JAX version).

    Uses WENO5 (standard, without tau enhancement).

    Returns:
        (rho_L, rho_R, vr_L, vr_R, p_L, p_R) for interior points
    """
    N = rho.shape[0]
    idx = jnp.arange(2, N-2)

    # Extract stencils for all 3 variables
    def get_stencil(arr):
        return (arr[idx - 2], arr[idx - 1], arr[idx], arr[idx + 1], arr[idx + 2])

    rho_stencil = get_stencil(rho)
    vr_stencil = get_stencil(vr)
    p_stencil = get_stencil(p)

    # Apply WENO5 kernel to each variable
    def apply_weno5(stencil):
        um2, um1, u0, up1, up2 = stencil
        u_L = vmap(weno5_face_kernel_jax)(um2, um1, u0, up1, up2)
        u_R = vmap(weno5_face_kernel_jax)(up2, up1, u0, um1, um2)
        return u_L, u_R

    rho_L, rho_R = apply_weno5(rho_stencil)
    vr_L, vr_R = apply_weno5(vr_stencil)
    p_L, p_R = apply_weno5(p_stencil)

    return rho_L, rho_R, vr_L, vr_R, p_L, p_R


@jit
def minmod_reconstruct_3vars_jax(rho, vr, p, dx):
    """
    OPTIMIZED: Reconstruct 3 primitive variables in single pass (JAX version).

    Uses Minmod TVD limiter.

    Args:
        rho, vr, p: Primitive variables (length N)
        dx: Grid spacing (scalar)

    Returns:
        (rho_L, rho_R, vr_L, vr_R, p_L, p_R) for interior points
    """
    N = rho.shape[0]
    idx = jnp.arange(1, N-1)

    # Compute slopes for all 3 variables
    def compute_limited_slope(arr):
        slope_L = (arr[idx] - arr[idx - 1]) / dx
        slope_R = (arr[idx + 1] - arr[idx]) / dx
        return vmap(minmod_jax)(slope_L, slope_R)

    slope_rho = compute_limited_slope(rho)
    slope_vr = compute_limited_slope(vr)
    slope_p = compute_limited_slope(p)

    # Reconstruct interface states
    # Right state at i-1/2 (from cell i looking left)
    rho_R = rho[idx] - 0.5 * slope_rho * dx
    vr_R = vr[idx] - 0.5 * slope_vr * dx
    p_R = p[idx] - 0.5 * slope_p * dx

    # Left state at i+1/2 (from cell i looking right)
    rho_L = rho[idx] + 0.5 * slope_rho * dx
    vr_L = vr[idx] + 0.5 * slope_vr * dx
    p_L = p[idx] + 0.5 * slope_p * dx

    return rho_L, rho_R, vr_L, vr_R, p_L, p_R


@jit
def mc_reconstruct_3vars_jax(rho, vr, p, dx):
    """
    OPTIMIZED: Reconstruct 3 primitive variables in single pass (JAX version).

    Uses MC (Monotonized Central) limiter.

    Args:
        rho, vr, p: Primitive variables (length N)
        dx: Grid spacing (scalar)

    Returns:
        (rho_L, rho_R, vr_L, vr_R, p_L, p_R) for interior points
    """
    N = rho.shape[0]
    idx = jnp.arange(1, N-1)

    # Compute slopes for all 3 variables with MC limiter
    def compute_limited_slope(arr):
        slope_L = (arr[idx] - arr[idx - 1]) / dx
        slope_R = (arr[idx + 1] - arr[idx]) / dx
        return vmap(mc_jax)(slope_L, slope_R)

    slope_rho = compute_limited_slope(rho)
    slope_vr = compute_limited_slope(vr)
    slope_p = compute_limited_slope(p)

    # Reconstruct interface states
    # Right state at i-1/2 (from cell i looking left)
    rho_R = rho[idx] - 0.5 * slope_rho * dx
    vr_R = vr[idx] - 0.5 * slope_vr * dx
    p_R = p[idx] - 0.5 * slope_p * dx

    # Left state at i+1/2 (from cell i looking right)
    rho_L = rho[idx] + 0.5 * slope_rho * dx
    vr_L = vr[idx] + 0.5 * slope_vr * dx
    p_L = p[idx] + 0.5 * slope_p * dx

    return rho_L, rho_R, vr_L, vr_R, p_L, p_R


@jit
def mp5_reconstruct_3vars_jax(rho, vr, p):
    """
    OPTIMIZED: Reconstruct 3 primitive variables in single pass (JAX version).

    Uses MP5 (5th order monotonicity-preserving) scheme.

    Returns:
        (rho_L, rho_R, vr_L, vr_R, p_L, p_R) for interior points
    """
    N = rho.shape[0]
    idx = jnp.arange(2, N-2)

    # Extract stencils for all 3 variables
    def get_stencil(arr):
        return (arr[idx - 2], arr[idx - 1], arr[idx], arr[idx + 1], arr[idx + 2])

    rho_stencil = get_stencil(rho)
    vr_stencil = get_stencil(vr)
    p_stencil = get_stencil(p)

    # Apply MP5 kernel to each variable
    def apply_mp5(stencil):
        um2, um1, u0, up1, up2 = stencil
        u_L, u_R = vmap(mp5_face_kernel_jax)(um2, um1, u0, up1, up2)
        return u_L, u_R

    rho_L, rho_R = apply_mp5(rho_stencil)
    vr_L, vr_R = apply_mp5(vr_stencil)
    p_L, p_R = apply_mp5(p_stencil)

    return rho_L, rho_R, vr_L, vr_R, p_L, p_R


# ==============================================================================
# UNIFIED RECONSTRUCTION CLASS (JAX VERSION)
# ==============================================================================

class ReconstructionJAX:
    """
    JAX-accelerated reconstruction class.

    API-compatible with Reconstruction from reconstruction.py.
    Uses JAX for GPU acceleration when available.

    Available methods:
        - "minmod": 2nd order TVD (most robust)
        - "mc":     2nd order TVD (less diffusive than minmod)
        - "mp5":    5th order monotonicity-preserving
        - "weno5":  5th order WENO (standard)
        - "wenoz":  5th order WENO-Z (recommended)

    All methods are fully implemented in JAX for GPU acceleration.
    """

    def __init__(self, method: str = "wenoz"):
        """
        Initialize JAX reconstruction object.

        Args:
            method: "minmod", "mc", "mp5", "weno5", or "wenoz" (default: wenoz)
        """
        method = method.lower()
        if method not in {"minmod", "mc", "mp5", "weno5", "wenoz"}:
            raise ValueError(f"Unknown method: {method}. Use: minmod, mc, mp5, weno5, wenoz")

        self.method = method
        self.name = f"reconstruction_jax_{method}"

        # Select reconstruction kernel
        if method == "minmod":
            self._reconstruct_jax = minmod_reconstruct_jax
            self._needs_dx = True
        elif method == "mc":
            self._reconstruct_jax = mc_reconstruct_jax
            self._needs_dx = True
        elif method == "mp5":
            self._reconstruct_jax = mp5_reconstruct_jax
            self._needs_dx = False
        elif method == "weno5":
            self._reconstruct_jax = weno5_reconstruct_jax
            self._needs_dx = False
        else:  # wenoz
            self._reconstruct_jax = wenoz_reconstruct_jax
            self._needs_dx = False

    def reconstruct(self, u, dx=None, x=None, boundary_type: str = "outflow"):
        """
        Reconstruct left/right states at all interfaces.

        Args:
            u: 1D array of cell-centered values (length N)
            dx: Scalar grid spacing (required for minmod)
            x: 1D array of coordinates (unused)
            boundary_type: "outflow", "reflecting", or "periodic"

        Returns:
            (u_L, u_R): Tuple of numpy arrays
        """
        # Convert to JAX array
        u_jax = jnp.array(np.asarray(u, dtype=np.float64))
        N = u_jax.shape[0]

        # Get dx if needed
        if self._needs_dx:
            if dx is None:
                if x is not None:
                    dx = float(x[1] - x[0])
                else:
                    dx = 1.0
            u_L_int, u_R_int = self._reconstruct_jax(u_jax, dx)
            # For minmod: interior indices are 1 to N-2
            start_idx = 1
        else:
            u_L_int, u_R_int = self._reconstruct_jax(u_jax)
            # For WENO: interior indices are 2 to N-3
            start_idx = 2

        # Convert back to numpy
        u_L_int = np.asarray(u_L_int)
        u_R_int = np.asarray(u_R_int)
        u_np = np.asarray(u_jax)

        # Allocate full output arrays
        u_L = np.empty(N + 1, dtype=np.float64)
        u_R = np.empty(N + 1, dtype=np.float64)

        # Place interior values
        if self._needs_dx:  # minmod
            # u_L_int has left states at i+1/2 for i in [1, N-2]
            # u_R_int has right states at i-1/2 for i in [1, N-2]
            u_L[2:N] = u_L_int  # Left states at interfaces 2 to N-1
            u_R[1:N-1] = u_R_int  # Right states at interfaces 1 to N-2
        else:  # weno5, wenoz
            # Interior results for indices 2 to N-3
            n_interior = len(u_L_int)
            u_L[3:3+n_interior] = u_L_int  # Left states at i+1/2
            u_R[2:2+n_interior] = u_R_int  # Right states at i-1/2

        # Fill near-boundary cells (piecewise constant)
        u_L[1] = u_np[0]
        u_L[2] = u_np[1]
        u_R[1] = u_np[0]
        u_R[N-2] = u_np[-2]
        u_R[N-1] = u_np[-1]

        # Boundary conditions
        self._fill_boundaries(u_np, u_L, u_R, boundary_type)

        return u_L, u_R

    def reconstruct_primitive_variables(self, rho0, vr, pressure, dx=None, x=None,
                                        boundary_type="outflow"):
        """
        Reconstruct primitive variables (rho0, v^r, p) to interfaces.

        OPTIMIZED: All methods use single-pass 3-variable reconstruction.

        Returns:
            left:  (rho0_L, vr_L, p_L)
            right: (rho0_R, vr_R, p_R)
        """
        # Convert to JAX arrays
        rho_jax = jnp.array(np.asarray(rho0, dtype=np.float64))
        vr_jax = jnp.array(np.asarray(vr, dtype=np.float64))
        p_jax = jnp.array(np.asarray(pressure, dtype=np.float64))
        N = rho_jax.shape[0]

        # Convert to numpy for boundary handling
        rho_np = np.asarray(rho_jax)
        vr_np = np.asarray(vr_jax)
        p_np = np.asarray(p_jax)

        # Allocate full output arrays
        rL = np.empty(N + 1, dtype=np.float64)
        rR = np.empty(N + 1, dtype=np.float64)
        vL = np.empty(N + 1, dtype=np.float64)
        vR = np.empty(N + 1, dtype=np.float64)
        pL = np.empty(N + 1, dtype=np.float64)
        pR = np.empty(N + 1, dtype=np.float64)

        if self.method == "wenoz":
            # Use optimized 3-variable WENO-Z kernel
            rL_int, rR_int, vL_int, vR_int, pL_int, pR_int = wenoz_reconstruct_3vars_jax(
                rho_jax, vr_jax, p_jax
            )

            # Convert interior results to numpy
            rL_int = np.asarray(rL_int)
            rR_int = np.asarray(rR_int)
            vL_int = np.asarray(vL_int)
            vR_int = np.asarray(vR_int)
            pL_int = np.asarray(pL_int)
            pR_int = np.asarray(pR_int)

            # Place interior values (indices 2 to N-3)
            n_int = len(rL_int)
            rL[3:3+n_int] = rL_int
            rR[2:2+n_int] = rR_int
            vL[3:3+n_int] = vL_int
            vR[2:2+n_int] = vR_int
            pL[3:3+n_int] = pL_int
            pR[2:2+n_int] = pR_int

            # Fill near-boundary cells (WENO needs 2 ghost cells)
            rL[1] = rho_np[0]; rL[2] = rho_np[1]
            rR[1] = rho_np[0]; rR[N-2] = rho_np[-2]; rR[N-1] = rho_np[-1]
            vL[1] = vr_np[0]; vL[2] = vr_np[1]
            vR[1] = vr_np[0]; vR[N-2] = vr_np[-2]; vR[N-1] = vr_np[-1]
            pL[1] = p_np[0]; pL[2] = p_np[1]
            pR[1] = p_np[0]; pR[N-2] = p_np[-2]; pR[N-1] = p_np[-1]

        elif self.method == "weno5":
            # Use optimized 3-variable WENO5 kernel
            rL_int, rR_int, vL_int, vR_int, pL_int, pR_int = weno5_reconstruct_3vars_jax(
                rho_jax, vr_jax, p_jax
            )

            # Convert interior results to numpy
            rL_int = np.asarray(rL_int)
            rR_int = np.asarray(rR_int)
            vL_int = np.asarray(vL_int)
            vR_int = np.asarray(vR_int)
            pL_int = np.asarray(pL_int)
            pR_int = np.asarray(pR_int)

            # Place interior values (indices 2 to N-3)
            n_int = len(rL_int)
            rL[3:3+n_int] = rL_int
            rR[2:2+n_int] = rR_int
            vL[3:3+n_int] = vL_int
            vR[2:2+n_int] = vR_int
            pL[3:3+n_int] = pL_int
            pR[2:2+n_int] = pR_int

            # Fill near-boundary cells (WENO needs 2 ghost cells)
            rL[1] = rho_np[0]; rL[2] = rho_np[1]
            rR[1] = rho_np[0]; rR[N-2] = rho_np[-2]; rR[N-1] = rho_np[-1]
            vL[1] = vr_np[0]; vL[2] = vr_np[1]
            vR[1] = vr_np[0]; vR[N-2] = vr_np[-2]; vR[N-1] = vr_np[-1]
            pL[1] = p_np[0]; pL[2] = p_np[1]
            pR[1] = p_np[0]; pR[N-2] = p_np[-2]; pR[N-1] = p_np[-1]

        elif self.method == "mp5":
            # Use optimized 3-variable MP5 kernel
            rL_int, rR_int, vL_int, vR_int, pL_int, pR_int = mp5_reconstruct_3vars_jax(
                rho_jax, vr_jax, p_jax
            )

            # Convert interior results to numpy
            rL_int = np.asarray(rL_int)
            rR_int = np.asarray(rR_int)
            vL_int = np.asarray(vL_int)
            vR_int = np.asarray(vR_int)
            pL_int = np.asarray(pL_int)
            pR_int = np.asarray(pR_int)

            # Place interior values (indices 2 to N-3, same as WENO)
            n_int = len(rL_int)
            rL[3:3+n_int] = rL_int
            rR[2:2+n_int] = rR_int
            vL[3:3+n_int] = vL_int
            vR[2:2+n_int] = vR_int
            pL[3:3+n_int] = pL_int
            pR[2:2+n_int] = pR_int

            # Fill near-boundary cells (MP5 needs 2 ghost cells)
            rL[1] = rho_np[0]; rL[2] = rho_np[1]
            rR[1] = rho_np[0]; rR[N-2] = rho_np[-2]; rR[N-1] = rho_np[-1]
            vL[1] = vr_np[0]; vL[2] = vr_np[1]
            vR[1] = vr_np[0]; vR[N-2] = vr_np[-2]; vR[N-1] = vr_np[-1]
            pL[1] = p_np[0]; pL[2] = p_np[1]
            pR[1] = p_np[0]; pR[N-2] = p_np[-2]; pR[N-1] = p_np[-1]

        elif self.method == "mc":
            # Get dx for MC
            if dx is None:
                if x is not None:
                    dx = float(x[1] - x[0])
                else:
                    dx = 1.0

            # Use optimized 3-variable MC kernel
            rL_int, rR_int, vL_int, vR_int, pL_int, pR_int = mc_reconstruct_3vars_jax(
                rho_jax, vr_jax, p_jax, dx
            )

            # Convert interior results to numpy
            rL_int = np.asarray(rL_int)
            rR_int = np.asarray(rR_int)
            vL_int = np.asarray(vL_int)
            vR_int = np.asarray(vR_int)
            pL_int = np.asarray(pL_int)
            pR_int = np.asarray(pR_int)

            # Place interior values (minmod/mc: indices 1 to N-2)
            rL[2:N] = rL_int
            rR[1:N-1] = rR_int
            vL[2:N] = vL_int
            vR[1:N-1] = vR_int
            pL[2:N] = pL_int
            pR[1:N-1] = pR_int

            # Fill near-boundary cells (minmod/mc needs 1 ghost cell)
            rL[1] = rho_np[0]; rR[N-1] = rho_np[-1]
            vL[1] = vr_np[0]; vR[N-1] = vr_np[-1]
            pL[1] = p_np[0]; pR[N-1] = p_np[-1]

        else:  # minmod
            # Get dx for minmod
            if dx is None:
                if x is not None:
                    dx = float(x[1] - x[0])
                else:
                    dx = 1.0

            # Use optimized 3-variable minmod kernel
            rL_int, rR_int, vL_int, vR_int, pL_int, pR_int = minmod_reconstruct_3vars_jax(
                rho_jax, vr_jax, p_jax, dx
            )

            # Convert interior results to numpy
            rL_int = np.asarray(rL_int)
            rR_int = np.asarray(rR_int)
            vL_int = np.asarray(vL_int)
            vR_int = np.asarray(vR_int)
            pL_int = np.asarray(pL_int)
            pR_int = np.asarray(pR_int)

            # Place interior values (minmod: indices 1 to N-2)
            # rL_int has left states at i+1/2 for i in [1, N-2]
            # rR_int has right states at i-1/2 for i in [1, N-2]
            rL[2:N] = rL_int
            rR[1:N-1] = rR_int
            vL[2:N] = vL_int
            vR[1:N-1] = vR_int
            pL[2:N] = pL_int
            pR[1:N-1] = pR_int

            # Fill near-boundary cells (minmod needs 1 ghost cell)
            rL[1] = rho_np[0]; rR[N-1] = rho_np[-1]
            vL[1] = vr_np[0]; vR[N-1] = vr_np[-1]
            pL[1] = p_np[0]; pR[N-1] = p_np[-1]

        # Fill boundaries (outflow)
        self._fill_boundaries(rho_np, rL, rR, "outflow")
        self._fill_boundaries(vr_np, vL, vR, "outflow")
        self._fill_boundaries(p_np, pL, pR, "outflow")

        # Apply reflecting conditions at r≈0
        if boundary_type == "reflecting":
            rL[0], rR[0] = rho_np[0], rho_np[0]
            vL[0], vR[0] = 0.0, 0.0  # v^r is odd
            pL[0], pR[0] = p_np[0], p_np[0]

        return (rL, vL, pL), (rR, vR, pR)

    def apply_physical_limiters(self, left_tuple, right_tuple,
                                atmosphere, gamma_rr=None):
        """
        Apply physical floors and limits to reconstructed states.
        """
        rho0_L, vr_L, p_L = left_tuple
        rho0_R, vr_R, p_R = right_tuple

        # Density and pressure floors
        rho0_L = np.maximum(rho0_L, atmosphere.rho_floor)
        rho0_R = np.maximum(rho0_R, atmosphere.rho_floor)
        p_L = np.maximum(p_L, atmosphere.p_floor)
        p_R = np.maximum(p_R, atmosphere.p_floor)

        # Velocity limiting
        if gamma_rr is not None:
            v_limit = atmosphere.v_max / np.sqrt(np.maximum(gamma_rr, 1.0))
            vr_L = np.clip(vr_L, -v_limit, v_limit)
            vr_R = np.clip(vr_R, -v_limit, v_limit)
        else:
            vr_L = np.clip(vr_L, -atmosphere.v_max, atmosphere.v_max)
            vr_R = np.clip(vr_R, -atmosphere.v_max, atmosphere.v_max)

        return (rho0_L, vr_L, p_L), (rho0_R, vr_R, p_R)

    @staticmethod
    def _fill_boundaries(u, u_L, u_R, boundary_type: str):
        """Fill boundary interfaces."""
        N = len(u)

        if boundary_type == "outflow":
            u_L[0] = u[0]
            u_R[0] = u[0]
            u_L[N] = u[-1]
            u_R[N] = u[-1]
        elif boundary_type == "reflecting":
            u_L[0] = u[0]
            u_R[0] = u[0]
            u_L[N] = u[-1]
            u_R[N] = u[-1]
        elif boundary_type == "periodic":
            u_L[0] = u[-1]
            u_R[0] = u[0]
            u_L[N] = u[-1]
            u_R[N] = u[0]
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def create_reconstruction_jax(method: str = "wenoz"):
    """
    Create JAX reconstruction object.

    Args:
        method: "minmod", "weno5", or "wenoz" (default: wenoz)

    Returns:
        ReconstructionJAX instance
    """
    return ReconstructionJAX(method=method)


# Aliases for compatibility
Reconstruction = ReconstructionJAX
