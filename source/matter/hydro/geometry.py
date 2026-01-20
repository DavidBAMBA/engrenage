# geometry.py
"""Geometry utilities for GRHD calculations.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

SPACEDIM = 3


# =============================================================================
# GeometryState - Container for spacetime geometry
# =============================================================================

@dataclass
class GeometryState:
    """
    Container for spacetime geometry at grid points.

    Supports both 1D (radial) and 3D representations.
    All arrays have shape (N,) for scalars or (N,3) / (N,3,3) for tensors.

    Required fields (always present):
        alpha: Lapse function α
        beta_r: Radial shift component β^r
        gamma_rr: Radial metric component γ_rr
        e6phi: Conformal factor e^{6φ}

    Optional fields (for 3D):
        beta_U: Full shift vector β^i (N,3)
        gamma_LL: Full covariant metric γ_ij (N,3,3)
        gamma_UU: Full contravariant metric γ^ij (N,3,3)
    """
    # Required: 1D components (always present)
    alpha: np.ndarray
    beta_r: np.ndarray
    gamma_rr: np.ndarray
    e6phi: np.ndarray

    # Optional: 3D tensors
    beta_U: Optional[np.ndarray] = None
    gamma_LL: Optional[np.ndarray] = None
    gamma_UU: Optional[np.ndarray] = None

    @classmethod
    def minkowski(cls, N: int) -> 'GeometryState':
        """Factory for flat Minkowski spacetime."""
        return cls(
            alpha=np.ones(N),
            beta_r=np.zeros(N),
            gamma_rr=np.ones(N),
            e6phi=np.ones(N)
        )

    @classmethod
    def from_bssn_1d(cls, alpha: np.ndarray, beta_r: np.ndarray,
                     phi: np.ndarray, gamma_rr: np.ndarray) -> 'GeometryState':
        """Factory from 1D BSSN variables."""
        return cls(
            alpha=alpha,
            beta_r=beta_r,
            gamma_rr=gamma_rr,
            e6phi=np.exp(6.0 * phi)
        )

    def __len__(self) -> int:
        return len(self.alpha)

    def at_indices(self, indices: np.ndarray) -> 'GeometryState':
        """Extract geometry at specific indices (e.g., cell faces for Riemann)."""
        return GeometryState(
            alpha=self.alpha[indices],
            beta_r=self.beta_r[indices],
            gamma_rr=self.gamma_rr[indices],
            e6phi=self.e6phi[indices],
            beta_U=self.beta_U[indices] if self.beta_U is not None else None,
            gamma_LL=self.gamma_LL[indices] if self.gamma_LL is not None else None,
            gamma_UU=self.gamma_UU[indices] if self.gamma_UU is not None else None,
        )


# =============================================================================
# Lorentz Factor
# =============================================================================

def compute_lorentz_factor(v_U, gamma_LL):
    """
    Compute Lorentz factor W = 1/√(1 - v²) where v² = γ_{ij} v^i v^j.

    Args:
        v_U: (N, 3) contravariant velocity components
        gamma_LL: (N, 3, 3) covariant metric

    Returns:
        W: (N,) Lorentz factor
    """
    v_squared = np.einsum('xij,xi,xj->x', gamma_LL, v_U, v_U)
    W = 1.0 / np.sqrt(np.maximum(1.0 - v_squared, 1e-16))
    return W


def compute_lorentz_factor_1d(vr, gamma_rr):
    """
    Compute Lorentz factor for 1D radial flow.

    Args:
        vr: (M,) radial velocity v^r
        gamma_rr: (M,) metric component γ_{rr}

    Returns:
        W: (M,) Lorentz factor
    """
    v_squared = gamma_rr * vr ** 2
    W = 1.0 / np.sqrt(np.maximum(1.0 - v_squared, 1e-16))
    return W


# =============================================================================
# 4-Velocity
# =============================================================================

def compute_4velocity(v_U, W, alpha, beta_U):
    """
    Compute 4-velocity u^μ from 3-velocity.

    u^0 = W/alpha (timelike component)
    u^i = W (v^i-beta/alpha) (spatial components)
    Returns:
        u4U: (N, 4) four-velocity [u^t, u^x, u^y, u^z]
    """
    N = len(W)
    u4U = np.zeros((N, 4))
    u4U[:, 0] = W / alpha
    u4U[:, 1:] = W[:, None] * (v_U - beta_U / alpha[:, None])
    return u4U


def compute_4velocity_1d(vr, gamma_rr, alpha, beta_r):
    """
    Compute 4-velocity for 1D radial flow.

    u^0 = W/alpha (timelike component)
    u^r = W(v^r - β^r/alpha) (radial component)
    Returns:
        u4U: (M, 4) four-velocity [u^t, u^r, 0, 0]
        W: (M,) Lorentz factor
    """
    M = len(vr)
    W = compute_lorentz_factor_1d(vr, gamma_rr)

    u4U = np.zeros((M, 4))
    u4U[:, 0] = W / alpha
    u4U[:, 1] = W * (vr - beta_r / alpha)

    return u4U, W


def compute_vU_from_u4U(u4U):
    """
    Compute fluid three-velocity from four-velocity: v^i = u^i/u^0.

    Returns:
        v_U: (N, 3) array with spatial velocity components [v^x, v^y, v^z]
    """
    return u4U[:, 1:] / u4U[:, 0:1]


# =============================================================================
# Contravariant 4-Metric g^{μν}
# =============================================================================

def compute_g4UU(alpha, beta_U, gamma_UU):
    """
    Compute contravariant 4-metric g^{μν} from ADM variables.

    g^{tt} = -1/alpha²
    g^{ti} = β^i/alpha²
    g^{ij} = γ^{ij} - β^i β^j/alpha²

    Returns:
        g4UU: (N, 4, 4) contravariant 4-metric
    """
    N = len(alpha)
    g4UU = np.zeros((N, 4, 4))

    alpha_sq = alpha ** 2 + 1e-30

    # g^{tt} = -1/alpha²
    g4UU[:, 0, 0] = -1.0 / alpha_sq

    # g^{ti} = g^{it} = β^i/alpha²
    g4UU[:, 0, 1:] = beta_U / alpha_sq[:, None]
    g4UU[:, 1:, 0] = g4UU[:, 0, 1:]

    # g^{ij} = γ^{ij} - β^i β^j/alpha²
    g4UU[:, 1:, 1:] = gamma_UU - np.einsum('xi,xj->xij', beta_U, beta_U) / alpha_sq[:, None, None]

    return g4UU


def compute_g4UU_1d(alpha, beta_r, gamma_rr_UU):
    """
    Compute contravariant 4-metric for 1D radial case.

    Returns:
        g4UU: (M, 4, 4) contravariant 4-metric
    """
    M = len(alpha)
    g4UU = np.zeros((M, 4, 4))

    alpha_sq = alpha ** 2 + 1e-30

    # g^{tt} = -1/alpha²
    g4UU[:, 0, 0] = -1.0 / alpha_sq

    # g^{tr} = g^{rt} = β^r/alpha²
    g4UU[:, 0, 1] = beta_r / alpha_sq
    g4UU[:, 1, 0] = g4UU[:, 0, 1]

    # g^{rr} = γ^{rr} - (β^r)²/alpha²
    g4UU[:, 1, 1] = gamma_rr_UU - beta_r ** 2 / alpha_sq

    return g4UU


# =============================================================================
# Covariant 4-Metric g_{μν}
# =============================================================================

def compute_g4DD(alpha, beta_U, gamma_LL):
    """
    Compute covariant 4-metric g_{μν} from ADM variables.

    g_{tt} = -alpha² + β_k β^k
    g_{ti} = β_i (lowered with γ)
    g_{ij} = γ_{ij}

    Returns:
        g4DD: (N, 4, 4) covariant 4-metric
    """
    N = len(alpha)
    g4DD = np.zeros((N, 4, 4))

    # Lower shift: β_i = γ_{ij} β^j
    beta_lower = np.einsum('xij,xj->xi', gamma_LL, beta_U)
    beta_squared = np.einsum('xi,xi->x', beta_U, beta_lower)

    # g_{tt} = -alpha² + β_k β^k
    g4DD[:, 0, 0] = -alpha ** 2 + beta_squared

    # g_{ti} = g_{it} = β_i
    g4DD[:, 0, 1:] = beta_lower
    g4DD[:, 1:, 0] = beta_lower

    # g_{ij} = γ_{ij}
    g4DD[:, 1:, 1:] = gamma_LL

    return g4DD


# =============================================================================
# Metric component extraction (for 1D from 3D)
# =============================================================================

def extract_radial_metric(gamma_LL, gamma_UU, beta_U):
    """
    Extract radial components from 3D metric tensors.

    Returns:
        gamma_rr: (N,) γ_{rr}
        gamma_rr_UU: (N,) γ^{rr}
        beta_r: (N,) β^r
    """
    gamma_rr = gamma_LL[:, 0, 0]
    gamma_rr_UU = gamma_UU[:, 0, 0]
    beta_r = beta_U[:, 0]
    return gamma_rr, gamma_rr_UU, beta_r
