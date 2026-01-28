"""
High-Order Reconstruction Methods - Fully Optimized with Numba JIT.

Provides five reconstruction schemes with optimal performance:
    1. Minmod   - 2nd order TVD (robust, stable)
    2. MC       - 2nd order TVD (less diffusive than Minmod)
    3. MP5      - 5th order monotonicity-preserving
    4. WENO5    - 5th order WENO (industry standard)
    5. WENO-Z   - 5th order WENO-Z (best overall, recommended)

All methods are fully vectorized and JIT-compiled with Numba.
"""

import numpy as np
from numba import jit, prange


# ==============================================================================
# MINMOD RECONSTRUCTION KERNELS (2ND ORDER TVD)
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def minmod_scalar(a, b):
    """Two-argument minmod function for scalars."""
    if a * b <= 0.0:
        return 0.0
    if abs(a) < abs(b):
        return a
    return b


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def minmod_reconstruct_3vars(rho, vr, p, dx,
                              rho_L, rho_R, vr_L, vr_R, p_L, p_R):
    """
    OPTIMIZED: Reconstruct 3 primitive variables (rho, vr, p) in ONE pass.

    This avoids 3 separate function calls and improves cache locality.
    All 6 output arrays are modified in-place.
    """
    N = rho.shape[0]

    # Parallel loop - process all 3 variables per cell
    for i in prange(1, N-1):
        # Load values for rho
        rho_im1 = rho[i-1]; rho_i = rho[i]; rho_ip1 = rho[i+1]
        # Load values for vr
        vr_im1 = vr[i-1]; vr_i = vr[i]; vr_ip1 = vr[i+1]
        # Load values for p
        p_im1 = p[i-1]; p_i = p[i]; p_ip1 = p[i+1]

        # Compute slopes for rho
        slope_L_rho = (rho_i - rho_im1) / dx
        slope_R_rho = (rho_ip1 - rho_i) / dx
        slope_rho = minmod_scalar(slope_L_rho, slope_R_rho)

        # Compute slopes for vr
        slope_L_vr = (vr_i - vr_im1) / dx
        slope_R_vr = (vr_ip1 - vr_i) / dx
        slope_vr = minmod_scalar(slope_L_vr, slope_R_vr)

        # Compute slopes for p
        slope_L_p = (p_i - p_im1) / dx
        slope_R_p = (p_ip1 - p_i) / dx
        slope_p = minmod_scalar(slope_L_p, slope_R_p)

        # Reconstruct interface states for all 3 variables
        rho_R[i] = rho_i - 0.5 * slope_rho * dx
        rho_L[i+1] = rho_i + 0.5 * slope_rho * dx

        vr_R[i] = vr_i - 0.5 * slope_vr * dx
        vr_L[i+1] = vr_i + 0.5 * slope_vr * dx

        p_R[i] = p_i - 0.5 * slope_p * dx
        p_L[i+1] = p_i + 0.5 * slope_p * dx

    # Near-boundary cells (piecewise constant)
    rho_L[1] = rho[0]; rho_R[N-1] = rho[-1]
    vr_L[1] = vr[0]; vr_R[N-1] = vr[-1]
    p_L[1] = p[0]; p_R[N-1] = p[-1]


# ==============================================================================
# MC (MONOTONIZED CENTRAL) RECONSTRUCTION KERNELS (2ND ORDER TVD)
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def minmod3_scalar(a, b, c):
    """
    Three-argument minmod function for scalars.
    Returns the value with smallest absolute magnitude if all have same sign, else 0.
    """
    # All must have same sign
    if a * b <= 0.0 or a * c <= 0.0:
        return 0.0
    
    # Return the one with smallest absolute value
    abs_a = abs(a)
    abs_b = abs(b)
    abs_c = abs(c)
    
    if abs_a <= abs_b and abs_a <= abs_c:
        return a
    elif abs_b <= abs_c:
        return b
    else:
        return c


@jit(nopython=True, cache=True, fastmath=True)
def mc_scalar(a, b):
    """
    MC (Monotonized Central) limiter for two slopes.
    
    MC(a, b) = minmod(2a, 2b, (a+b)/2)
    
    Less diffusive than minmod, more robust than others.
    Args:
        a: backward slope
        b: forward slope
    """
    return minmod3_scalar(2.0 * a, 2.0 * b, 0.5 * (a + b))


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def mc_reconstruct_3vars(rho, vr, p, dx,
                         rho_L, rho_R, vr_L, vr_R, p_L, p_R):
    """
    OPTIMIZED: Reconstruct 3 primitive variables (rho, vr, p) with MC limiter in ONE pass.

    This avoids 3 separate function calls and improves cache locality.
    All 6 output arrays are modified in-place.
    """
    N = rho.shape[0]

    # Parallel loop - process all 3 variables per cell
    for i in prange(1, N-1):
        # Load values for rho
        rho_im1 = rho[i-1]; rho_i = rho[i]; rho_ip1 = rho[i+1]
        # Load values for vr
        vr_im1 = vr[i-1]; vr_i = vr[i]; vr_ip1 = vr[i+1]
        # Load values for p
        p_im1 = p[i-1]; p_i = p[i]; p_ip1 = p[i+1]

        # Compute slopes for rho
        slope_L_rho = (rho_i - rho_im1) / dx
        slope_R_rho = (rho_ip1 - rho_i) / dx
        slope_rho = mc_scalar(slope_L_rho, slope_R_rho)

        # Compute slopes for vr
        slope_L_vr = (vr_i - vr_im1) / dx
        slope_R_vr = (vr_ip1 - vr_i) / dx
        slope_vr = mc_scalar(slope_L_vr, slope_R_vr)

        # Compute slopes for p
        slope_L_p = (p_i - p_im1) / dx
        slope_R_p = (p_ip1 - p_i) / dx
        slope_p = mc_scalar(slope_L_p, slope_R_p)

        # Reconstruct interface states for all 3 variables
        rho_R[i] = rho_i - 0.5 * slope_rho * dx
        rho_L[i+1] = rho_i + 0.5 * slope_rho * dx

        vr_R[i] = vr_i - 0.5 * slope_vr * dx
        vr_L[i+1] = vr_i + 0.5 * slope_vr * dx

        p_R[i] = p_i - 0.5 * slope_p * dx
        p_L[i+1] = p_i + 0.5 * slope_p * dx

    # Near-boundary cells (piecewise constant)
    rho_L[1] = rho[0]; rho_R[N-1] = rho[-1]
    vr_L[1] = vr[0]; vr_R[N-1] = vr[-1]
    p_L[1] = p[0]; p_R[N-1] = p[-1]


# ==============================================================================
# MP5 RECONSTRUCTION KERNELS (5TH ORDER MONOTONICITY-PRESERVING)
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def minmod4_scalar(w, x, y, z):
    """Four-argument minmod function for scalars."""
    s = 0.125 * (np.sign(w) + np.sign(x)) * abs(
        (np.sign(w) + np.sign(y)) * (np.sign(w) + np.sign(z)))
    return s * min(abs(w), abs(x), abs(y), abs(z))


@jit(nopython=True, cache=True, fastmath=True)
def mp5_face_kernel(um2, um1, u0, up1, up2):
    """
    MP5 face reconstruction kernel for a single point.

    Fifth-order monotonicity-preserving scheme.
    Good balance of accuracy and robustness.
    """
    # Parameters
    alpha = 4.0
    eps = 1e-20

    # Norm for switching criterion
    vnorm = np.sqrt(um2**2 + um1**2 + u0**2 + up1**2 + up2**2) + 1e-30

    # Left state at i+1/2 (from cell i towards right)
    uL = (2.0*um2 - 13.0*um1 + 47.0*u0 + 27.0*up1 - 3.0*up2) / 60.0
    vmp = u0 + minmod_scalar(up1 - u0, alpha * (u0 - um1))

    # Switching criterion
    cond = (uL - u0) * (uL - vmp) <= eps * vnorm

    if not cond:
        # Second derivatives
        djm1 = um2 - 2.0*um1 + u0
        dj = um1 - 2.0*u0 + up1
        djp1 = u0 - 2.0*up1 + up2

        # Fourth-order differences
        dm4jmh = minmod4_scalar(4.0*dj - djm1, 4.0*djm1 - dj, dj, djm1)
        dm4jph = minmod4_scalar(4.0*dj - djp1, 4.0*djp1 - dj, dj, djp1)

        # Additional terms
        vul = u0 + alpha * (u0 - um1)
        vav = 0.5 * (u0 + up1)
        vmd = vav - 0.5 * dm4jph
        vlc = u0 + 0.5 * (u0 - um1) + (4.0/3.0) * dm4jmh

        # Min/max bounds
        vminl = max(min(u0, up1, vmd), min(u0, vul, vlc))
        vmaxl = min(max(u0, up1, vmd), max(u0, vul, vlc))

        # Apply bounds
        uL += minmod_scalar(vminl - uL, vmaxl - uL)

    # Right state at i-1/2 (symmetric, reversed stencil)
    uR = (2.0*up2 - 13.0*up1 + 47.0*u0 + 27.0*um1 - 3.0*um2) / 60.0
    vmp_r = u0 + minmod_scalar(um1 - u0, alpha * (u0 - up1))

    cond_r = (uR - u0) * (uR - vmp_r) <= eps * vnorm

    if not cond_r:
        djm1_r = up2 - 2.0*up1 + u0
        dj_r = up1 - 2.0*u0 + um1
        djp1_r = u0 - 2.0*um1 + um2

        dm4jmh_r = minmod4_scalar(4.0*dj_r - djm1_r, 4.0*djm1_r - dj_r, dj_r, djm1_r)
        dm4jph_r = minmod4_scalar(4.0*dj_r - djp1_r, 4.0*djp1_r - dj_r, dj_r, djp1_r)

        vul_r = u0 + alpha * (u0 - up1)
        vav_r = 0.5 * (u0 + um1)
        vmd_r = vav_r - 0.5 * dm4jph_r
        vlc_r = u0 + 0.5 * (u0 - up1) + (4.0/3.0) * dm4jmh_r

        vminr = max(min(u0, um1, vmd_r), min(u0, vul_r, vlc_r))
        vmaxr = min(max(u0, um1, vmd_r), max(u0, vul_r, vlc_r))

        uR += minmod_scalar(vminr - uR, vmaxr - uR)

    return uL, uR


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def mp5_reconstruct_3vars(rho, vr, p,
                          rho_L, rho_R, vr_L, vr_R, p_L, p_R):
    """
    OPTIMIZED: Reconstruct 3 primitive variables (rho, vr, p) in ONE pass.

    This avoids 3 separate function calls and improves cache locality.
    All 6 output arrays are modified in-place.
    """
    N = rho.shape[0]

    # Parallel loop - process all 3 variables per cell
    for i in prange(2, N-2):
        # Load stencil for rho
        rho_m2 = rho[i-2]; rho_m1 = rho[i-1]; rho_0 = rho[i]
        rho_p1 = rho[i+1]; rho_p2 = rho[i+2]

        # Load stencil for vr
        vr_m2 = vr[i-2]; vr_m1 = vr[i-1]; vr_0 = vr[i]
        vr_p1 = vr[i+1]; vr_p2 = vr[i+2]

        # Load stencil for p
        p_m2 = p[i-2]; p_m1 = p[i-1]; p_0 = p[i]
        p_p1 = p[i+1]; p_p2 = p[i+2]

        # Reconstruct all 3 variables at once
        rho_uL, rho_uR = mp5_face_kernel(rho_m2, rho_m1, rho_0, rho_p1, rho_p2)
        vr_uL, vr_uR = mp5_face_kernel(vr_m2, vr_m1, vr_0, vr_p1, vr_p2)
        p_uL, p_uR = mp5_face_kernel(p_m2, p_m1, p_0, p_p1, p_p2)

        # Store results
        rho_L[i+1] = rho_uL; rho_R[i] = rho_uR
        vr_L[i+1] = vr_uL; vr_R[i] = vr_uR
        p_L[i+1] = p_uL; p_R[i] = p_uR

    # Near-boundary cells (piecewise constant)
    rho_L[1] = rho[0]; rho_L[2] = rho[1]
    rho_R[1] = rho[0]; rho_R[N-2] = rho[-2]; rho_R[N-1] = rho[-1]

    vr_L[1] = vr[0]; vr_L[2] = vr[1]
    vr_R[1] = vr[0]; vr_R[N-2] = vr[-2]; vr_R[N-1] = vr[-1]

    p_L[1] = p[0]; p_L[2] = p[1]
    p_R[1] = p[0]; p_R[N-2] = p[-2]; p_R[N-1] = p[-1]


# ==============================================================================
# WENO5 RECONSTRUCTION KERNELS (5TH ORDER WENO)
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def weno5_face_kernel(um2, um1, u0, up1, up2):
    """
    WENO5 face reconstruction kernel for a single point.

    Fifth-order WENO - industry standard for shock-capturing.
    """
    # Smoothness indicators
    beta0 = (13.0/12.0) * (um2 - 2.0*um1 + u0)**2 + 0.25 * (um2 - 4.0*um1 + 3.0*u0)**2
    beta1 = (13.0/12.0) * (um1 - 2.0*u0 + up1)**2 + 0.25 * (um1 - up1)**2
    beta2 = (13.0/12.0) * (u0 - 2.0*up1 + up2)**2 + 0.25 * (3.0*u0 - 4.0*up1 + up2)**2

    # Ideal weights
    d0, d1, d2 = 0.1, 0.6, 0.3
    eps = 1.0e-20

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


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def weno5_reconstruct_3vars(rho, vr, p,
                            rho_L, rho_R, vr_L, vr_R, p_L, p_R):
    """
    OPTIMIZED: Reconstruct 3 primitive variables (rho, vr, p) in ONE pass.

    This avoids 3 separate function calls and improves cache locality.
    All 6 output arrays are modified in-place.
    """
    N = rho.shape[0]

    # Parallel loop - process all 3 variables per cell
    for i in prange(2, N-2):
        # Load stencil for rho
        rho_m2 = rho[i-2]; rho_m1 = rho[i-1]; rho_0 = rho[i]
        rho_p1 = rho[i+1]; rho_p2 = rho[i+2]

        # Load stencil for vr
        vr_m2 = vr[i-2]; vr_m1 = vr[i-1]; vr_0 = vr[i]
        vr_p1 = vr[i+1]; vr_p2 = vr[i+2]

        # Load stencil for p
        p_m2 = p[i-2]; p_m1 = p[i-1]; p_0 = p[i]
        p_p1 = p[i+1]; p_p2 = p[i+2]

        # Reconstruct all 3 variables at once
        # Left states at i+1/2
        rho_L[i+1] = weno5_face_kernel(rho_m2, rho_m1, rho_0, rho_p1, rho_p2)
        vr_L[i+1] = weno5_face_kernel(vr_m2, vr_m1, vr_0, vr_p1, vr_p2)
        p_L[i+1] = weno5_face_kernel(p_m2, p_m1, p_0, p_p1, p_p2)

        # Right states at i-1/2 (reversed stencil)
        rho_R[i] = weno5_face_kernel(rho_p2, rho_p1, rho_0, rho_m1, rho_m2)
        vr_R[i] = weno5_face_kernel(vr_p2, vr_p1, vr_0, vr_m1, vr_m2)
        p_R[i] = weno5_face_kernel(p_p2, p_p1, p_0, p_m1, p_m2)

    # Near-boundary cells (piecewise constant)
    rho_L[1] = rho[0]; rho_L[2] = rho[1]
    rho_R[1] = rho[0]; rho_R[N-2] = rho[-2]; rho_R[N-1] = rho[-1]

    vr_L[1] = vr[0]; vr_L[2] = vr[1]
    vr_R[1] = vr[0]; vr_R[N-2] = vr[-2]; vr_R[N-1] = vr[-1]

    p_L[1] = p[0]; p_L[2] = p[1]
    p_R[1] = p[0]; p_R[N-2] = p[-2]; p_R[N-1] = p[-1]


# ==============================================================================
# WENO-Z RECONSTRUCTION KERNELS (5TH ORDER WENO-Z)
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def wenoz_face_kernel(um2, um1, u0, up1, up2):
    """
    WENO-Z face reconstruction kernel for a single point.

    Fifth-order WENO-Z - best overall performance.
    The tau indicator improves accuracy near smooth extrema.
    """
    # Smoothness indicators
    beta0 = (13.0/12.0) * (um2 - 2.0*um1 + u0)**2 + 0.25 * (um2 - 4.0*um1 + 3.0*u0)**2
    beta1 = (13.0/12.0) * (um1 - 2.0*u0 + up1)**2 + 0.25 * (um1 - up1)**2
    beta2 = (13.0/12.0) * (u0 - 2.0*up1 + up2)**2 + 0.25 * (3.0*u0 - 4.0*up1 + up2)**2

    # Tau indicator - key improvement of WENO-Z
    tau5 = abs(beta0 - beta2)

    # Ideal weights
    d0, d1, d2 = 0.1, 0.6, 0.3
    eps = 1.0e-20

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


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def wenoz_reconstruct_3vars(rho, vr, p,
                            rho_L, rho_R, vr_L, vr_R, p_L, p_R):
    """
    OPTIMIZED: Reconstruct 3 primitive variables (rho, vr, p) in ONE pass.

    This avoids 3 separate function calls and improves cache locality.
    All 6 output arrays are modified in-place.
    """
    N = rho.shape[0]

    # Parallel loop - process all 3 variables per cell
    for i in prange(2, N-2):
        # Load stencil for rho
        rho_m2 = rho[i-2]; rho_m1 = rho[i-1]; rho_0 = rho[i]
        rho_p1 = rho[i+1]; rho_p2 = rho[i+2]

        # Load stencil for vr
        vr_m2 = vr[i-2]; vr_m1 = vr[i-1]; vr_0 = vr[i]
        vr_p1 = vr[i+1]; vr_p2 = vr[i+2]

        # Load stencil for p
        p_m2 = p[i-2]; p_m1 = p[i-1]; p_0 = p[i]
        p_p1 = p[i+1]; p_p2 = p[i+2]

        # Reconstruct all 3 variables at once
        # Left states at i+1/2
        rho_L[i+1] = wenoz_face_kernel(rho_m2, rho_m1, rho_0, rho_p1, rho_p2)
        vr_L[i+1] = wenoz_face_kernel(vr_m2, vr_m1, vr_0, vr_p1, vr_p2)
        p_L[i+1] = wenoz_face_kernel(p_m2, p_m1, p_0, p_p1, p_p2)

        # Right states at i-1/2 (reversed stencil)
        rho_R[i] = wenoz_face_kernel(rho_p2, rho_p1, rho_0, rho_m1, rho_m2)
        vr_R[i] = wenoz_face_kernel(vr_p2, vr_p1, vr_0, vr_m1, vr_m2)
        p_R[i] = wenoz_face_kernel(p_p2, p_p1, p_0, p_m1, p_m2)

    # Near-boundary cells (piecewise constant)
    rho_L[1] = rho[0]; rho_L[2] = rho[1]
    rho_R[1] = rho[0]; rho_R[N-2] = rho[-2]; rho_R[N-1] = rho[-1]

    vr_L[1] = vr[0]; vr_L[2] = vr[1]
    vr_R[1] = vr[0]; vr_R[N-2] = vr[-2]; vr_R[N-1] = vr[-1]

    p_L[1] = p[0]; p_L[2] = p[1]
    p_R[1] = p[0]; p_R[N-2] = p[-2]; p_R[N-1] = p[-1]


# ==============================================================================
# UNIFIED RECONSTRUCTION CLASS
# ==============================================================================

class Reconstruction:
    """
    Unified reconstruction class supporting multiple methods.

    Available methods:
        - "minmod": 2nd order TVD (most robust)
        - "mc":     2nd order TVD (less diffusive than minmod)
        - "mp5":    5th order monotonicity-preserving
        - "weno5":  5th order WENO (standard)
        - "wenoz":  5th order WENO-Z (recommended, best overall)

    All methods are fully optimized with Numba JIT compilation.
    Uses single-pass 3-variable reconstruction for best cache locality.

    Usage:
        >>> recon = Reconstruction(method="mc")
        >>> left, right = recon.reconstruct_primitive_variables(rho, vr, p, dx=dx)
    """

    def __init__(self, method: str = "wenoz"):
        """
        Initialize reconstruction object.

        Args:
            method: "minmod", "mc", "mp5", "weno5", or "wenoz"
        """
        method = method.lower()
        if method not in {"minmod", "mc", "mp5", "weno5", "wenoz"}:
            raise ValueError(f"Unknown method: {method}. Use: minmod, mc, mp5, weno5, wenoz")

        self.method = method
        self.name = f"reconstruction_{method}"

    def reconstruct_primitive_variables(self, rho0, vr, pressure, dx=None, x=None, boundary_type="outflow"):
        """
        Reconstruct primitive variables (rho0, v^r, p) to interfaces.

        OPTIMIZED: All methods use single-pass 3-variable reconstruction
        for better cache locality and reduced function call overhead.

        Args:
            rho0: Rest-mass density
            vr: Radial velocity
            pressure: Pressure
            dx: Grid spacing
            x: Coordinate array
            boundary_type: Boundary condition type

        Returns:
            left:  (rho0_L, vr_L, p_L)
            right: (rho0_R, vr_R, p_R)
        """
        # Convert inputs to float64 arrays
        rho0 = np.asarray(rho0, dtype=np.float64)
        vr_arr = np.asarray(vr, dtype=np.float64)
        pressure = np.asarray(pressure, dtype=np.float64)
        N = rho0.size

        # Allocate all 6 output arrays
        rL = np.empty(N + 1, dtype=np.float64)
        rR = np.empty(N + 1, dtype=np.float64)
        vL = np.empty(N + 1, dtype=np.float64)
        vR = np.empty(N + 1, dtype=np.float64)
        pL = np.empty(N + 1, dtype=np.float64)
        pR = np.empty(N + 1, dtype=np.float64)

        # OPTIMIZED: Single call to method-specific 3-variable kernel
        if self.method == "wenoz":
            wenoz_reconstruct_3vars(rho0, vr_arr, pressure,
                                    rL, rR, vL, vR, pL, pR)
        elif self.method == "weno5":
            weno5_reconstruct_3vars(rho0, vr_arr, pressure,
                                    rL, rR, vL, vR, pL, pR)
        elif self.method == "mp5":
            mp5_reconstruct_3vars(rho0, vr_arr, pressure,
                                  rL, rR, vL, vR, pL, pR)
        elif self.method == "mc":
            # Get dx for MC
            if dx is None:
                if x is not None:
                    dx = x[1] - x[0]
                else:
                    dx = 1.0
            mc_reconstruct_3vars(rho0, vr_arr, pressure, dx,
                                 rL, rR, vL, vR, pL, pR)
        else:  # minmod
            # Get dx for minmod
            if dx is None:
                if x is not None:
                    dx = x[1] - x[0]
                else:
                    dx = 1.0
            minmod_reconstruct_3vars(rho0, vr_arr, pressure, dx,
                                     rL, rR, vL, vR, pL, pR)

        # Fill boundaries (outflow first, then apply parity if needed)
        self._fill_boundaries(rho0, rL, rR, "outflow")
        self._fill_boundaries(vr_arr, vL, vR, "outflow")
        self._fill_boundaries(pressure, pL, pR, "outflow")

        # Apply reflecting/parity conditions at r≈0
        if boundary_type == "reflecting":
            rL[0], rR[0] = rho0[0], rho0[0]
            vL[0], vR[0] = 0.0, 0.0  # v^r is odd
            pL[0], pR[0] = pressure[0], pressure[0]

        return (rL, vL, pL), (rR, vR, pR)

    def apply_physical_limiters(self, left_tuple, right_tuple,
                                atmosphere, gamma_rr=None):
        """
        Apply physical floors and limits to reconstructed states.

        Args:
            left_tuple: (rho0_L, vr_L, p_L)
            right_tuple: (rho0_R, vr_R, p_R)
            atmosphere: AtmosphereParams object
            gamma_rr: Metric component for velocity limiting

        Returns:
            (left_limited, right_limited)
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
        N = u.size

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

def create_reconstruction(method: str = "wenoz"):
    """
    Create reconstruction object.

    Automatically selects JAX or Numba backend based on ENGRENAGE_BACKEND env var.

    Args:
        method: "minmod", "mc", "mp5", "weno5", or "wenoz" (default: wenoz)

    Returns:
        Reconstruction instance (JAX or Numba version)
    """
    from .tests.advance.backend import BACKEND

    if 'jax' in BACKEND:
        # JAX backend (note: MP5 and MC not available in JAX)
        if method.lower() in ["mp5", "mc"]:
            print(f"[reconstruction] WARNING: {method.upper()} not available in JAX, using WENO-Z instead")
            method = "wenoz"
        from .reconstruction_jax import ReconstructionJAX
        return ReconstructionJAX(method=method)

    return Reconstruction(method=method)


# Backward compatibility aliases
def get_reconstruction_minmod():
    """Get minmod reconstruction."""
    return Reconstruction(method="minmod")

def get_reconstruction_mc():
    """Get MC reconstruction."""
    return Reconstruction(method="mc")

def get_reconstruction_mp5():
    """Get MP5 reconstruction."""
    return Reconstruction(method="mp5")

def get_reconstruction_weno5():
    """Get WENO5 reconstruction."""
    return Reconstruction(method="weno5")

def get_reconstruction_wenoz():
    """Get WENO-Z reconstruction."""
    return Reconstruction(method="wenoz")