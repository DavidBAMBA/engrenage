"""
3+1 ADM Geometry Module for GRHD

This module provides classes for handling geometric quantities in numerical relativity,
including the 3+1 ADM decomposition, metric tensors, and related computations.

Classes:
    ADMGeometry: Base class for 3+1 ADM geometry computations
    MinkowskiGeometry: Flat spacetime specialization
    ValenciaGeometry: Specialized geometry for Valencia formulation
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from source.bssn.tensoralgebra import (
    SPACEDIM,
    get_bar_gamma_LL,
    get_bar_gamma_UU,
    get_det_bar_gamma,
)
from source.core.spacing import NUM_GHOSTS


class ADMGeometry:
    """
    Handles all 3+1 ADM geometry computations.

    The ADM formalism decomposes spacetime into space + time slices with:
    - Lapse function α (how much proper time passes)
    - Shift vector β^i (how coordinates shift between slices)
    - Spatial metric γ_ij (geometry of each slice)

    Provides conversions to 4D quantities needed for GRHD.
    """

    def __init__(self, alpha: np.ndarray, beta_U: np.ndarray,
                 gamma_LL: np.ndarray, gamma_UU: Optional[np.ndarray] = None):
        """
        Initialize ADM geometry.

        Parameters
        ----------
        alpha : np.ndarray
            Lapse function (N,)
        beta_U : np.ndarray
            Shift vector with upper index β^i (N, 3)
        gamma_LL : np.ndarray
            Spatial metric with lower indices γ_ij (N, 3, 3)
        gamma_UU : np.ndarray, optional
            Inverse spatial metric γ^ij. If None, will be computed.
        """
        self.alpha = alpha
        self.beta_U = beta_U
        self.gamma_LL = gamma_LL

        # Compute inverse metric if not provided
        if gamma_UU is None:
            self.gamma_UU = self._compute_inverse_metric(gamma_LL)
        else:
            self.gamma_UU = gamma_UU

        # Compute derived quantities
        self.sqrt_gamma = self._compute_sqrt_determinant(gamma_LL)

    @classmethod
    def from_bssn(cls, bssn_vars, background, bssn_d1=None):
        """
        Extract ADM geometry from BSSN variables.

        Parameters
        ----------
        bssn_vars : BSSNStateVariables
            BSSN evolution variables
        background : SphericalBackground
            Background metric and Christoffel symbols
        bssn_d1 : BSSNDerivatives, optional
            First derivatives of BSSN variables

        Returns
        -------
        ADMGeometry
            Geometry object with extracted ADM quantities
        """
        alpha_raw = getattr(bssn_vars, 'alpha', None)
        if alpha_raw is None:
            alpha_raw = getattr(bssn_vars, 'lapse', None)
        if alpha_raw is None:
            raise AttributeError("BSSN variables must provide 'alpha' or 'lapse'")
        alpha = np.asarray(alpha_raw, dtype=float)
        N = len(alpha)

        # Extract shift (handle different naming conventions)
        beta_U = np.zeros((N, SPACEDIM))

        beta_attr = getattr(bssn_vars, 'betaU', None)
        if beta_attr is None:
            beta_attr = getattr(bssn_vars, 'shift_U', None)

        if beta_attr is not None:
            shift_array = np.asarray(beta_attr, dtype=float)
            if shift_array.ndim >= 2:
                for i in range(min(SPACEDIM, shift_array.shape[1])):
                    beta_U[:, i] = shift_array[:, i]
            elif shift_array.ndim == 1:
                # Single component, assume radial
                beta_U[:, 0] = shift_array

        # CRITICAL: Apply inverse scaling from background (BSSN variables are rescaled)
        # This converts from BSSN's rescaled shift to physical shift β^i
        if hasattr(background, 'inverse_scaling_vector'):
            beta_U = beta_U * background.inverse_scaling_vector

        # Compute conformal factor e^φ
        phi = np.asarray(getattr(bssn_vars, 'phi', np.zeros(N)), dtype=float)
        e2phi = np.exp(2.0 * phi)
        e4phi = e2phi * e2phi

        # Get conformal metric from BSSN
        h_LL = getattr(bssn_vars, 'hDD', None)
        if h_LL is None:
            h_LL = getattr(bssn_vars, 'h_LL', None)
        if h_LL is not None:
            h_LL = np.asarray(h_LL, dtype=float)
        else:
            # Compute from BSSN variables
            h_LL = get_bar_gamma_LL(bssn_vars, background)

        # Physical metric: γ_ij = e^{4φ} γ̄_ij
        gamma_LL = np.zeros((N, SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                gamma_LL[:, i, j] = e4phi * h_LL[:, i, j]

        return cls(alpha, beta_U, gamma_LL)

    @staticmethod
    def _compute_inverse_metric(gamma_LL: np.ndarray) -> np.ndarray:
        """
        Compute inverse of spatial metric γ^ij.

        Parameters
        ----------
        gamma_LL : np.ndarray
            Spatial metric γ_ij (N, 3, 3)

        Returns
        -------
        np.ndarray
            Inverse metric γ^ij (N, 3, 3)
        """
        N = gamma_LL.shape[0]
        gamma_UU = np.zeros_like(gamma_LL)

        for i in range(N):
            gamma_UU[i] = np.linalg.inv(gamma_LL[i])

        return gamma_UU

    @staticmethod
    def _compute_sqrt_determinant(gamma_LL: np.ndarray) -> np.ndarray:
        """
        Compute sqrt of determinant of spatial metric.

        Parameters
        ----------
        gamma_LL : np.ndarray
            Spatial metric γ_ij (N, 3, 3)

        Returns
        -------
        np.ndarray
            sqrt(det(γ)) at each point (N,)
        """
        N = gamma_LL.shape[0]
        sqrt_gamma = np.zeros(N)

        for i in range(N):
            det_gamma = np.linalg.det(gamma_LL[i])
            sqrt_gamma[i] = np.sqrt(np.maximum(det_gamma, 1e-16))

        return sqrt_gamma

    def get_4metric_contravariant(self) -> np.ndarray:
        """
        Compute contravariant 4-metric g^{μν} from ADM variables.

        The 4-metric in terms of ADM variables is:
        g^{tt} = -1/α²
        g^{ti} = β^i/α²
        g^{ij} = γ^{ij} - β^i β^j/α²

        Returns
        -------
        np.ndarray
            4-metric g^{μν} (N, 4, 4)
        """
        N = len(self.alpha)
        g4UU = np.zeros((N, 4, 4))

        # Precompute 1/α²
        inv_alpha2 = 1.0 / (self.alpha**2 + 1e-16)

        # g^{tt} = -1/α²
        g4UU[:, 0, 0] = -inv_alpha2

        # g^{ti} = g^{it} = β^i/α²
        for i in range(SPACEDIM):
            g4UU[:, 0, i+1] = self.beta_U[:, i] * inv_alpha2
            g4UU[:, i+1, 0] = g4UU[:, 0, i+1]  # Symmetry

        # g^{ij} = γ^{ij} - β^i β^j/α²
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                g4UU[:, i+1, j+1] = (self.gamma_UU[:, i, j] -
                                      self.beta_U[:, i] * self.beta_U[:, j] * inv_alpha2)

        return g4UU

    def get_4metric_contravariant_scalar(self, idx: int) -> np.ndarray:
        """
        Get 4-metric g^{μν} at a single point (for compatibility with old code).

        Parameters
        ----------
        idx : int
            Grid point index

        Returns
        -------
        np.ndarray
            4-metric g^{μν} at point idx (4, 4)
        """
        g4UU = np.zeros((4, 4))

        inv_alpha2 = 1.0 / (self.alpha[idx]**2 + 1e-16)

        g4UU[0, 0] = -inv_alpha2

        for i in range(SPACEDIM):
            g4UU[0, i+1] = self.beta_U[idx, i] * inv_alpha2
            g4UU[i+1, 0] = g4UU[0, i+1]

        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                g4UU[i+1, j+1] = (self.gamma_UU[idx, i, j] -
                                  self.beta_U[idx, i] * self.beta_U[idx, j] * inv_alpha2)

        return g4UU

    def get_4metric_covariant(self) -> np.ndarray:
        """
        Compute covariant 4-metric g_{μν} from ADM variables.

        The covariant metric in terms of ADM variables is:
        g_{tt} = -α² + β_i β^i
        g_{ti} = β_i
        g_{ij} = γ_{ij}

        Returns
        -------
        np.ndarray
            4-metric g_{μν} (N, 4, 4)
        """
        N = len(self.alpha)
        g4DD = np.zeros((N, 4, 4))

        # Compute β_i = γ_{ij} β^j
        beta_D = np.einsum('nij,nj->ni', self.gamma_LL, self.beta_U)

        # Compute β_i β^i
        beta_squared = np.einsum('ni,ni->n', beta_D, self.beta_U)

        # g_{tt} = -α² + β_i β^i
        g4DD[:, 0, 0] = -self.alpha**2 + beta_squared

        # g_{ti} = g_{it} = β_i
        for i in range(SPACEDIM):
            g4DD[:, 0, i+1] = beta_D[:, i]
            g4DD[:, i+1, 0] = beta_D[:, i]

        # g_{ij} = γ_{ij}
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                g4DD[:, i+1, j+1] = self.gamma_LL[:, i, j]

        return g4DD

    def compute_4velocity(self, v_U: np.ndarray, W: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute 4-velocity u^μ from Valencia 3-velocity v^i.

        The 4-velocity is:
        u^0 = W/α (time component)
        u^i = W v^i - β^i u^0 (space components)

        where W = 1/√(1 - v_i v^i) is the Lorentz factor.

        Parameters
        ----------
        v_U : np.ndarray
            Valencia 3-velocity v^i (N, 3)
        W : np.ndarray, optional
            Lorentz factor. If None, will be computed.

        Returns
        -------
        np.ndarray
            4-velocity u^μ (N, 4)
        """
        N = len(self.alpha)
        u4U = np.zeros((N, 4))

        # Compute Lorentz factor if not provided
        if W is None:
            # v_i = γ_{ij} v^j
            v_D = np.einsum('nij,nj->ni', self.gamma_LL, v_U)
            # v² = v_i v^i
            v_squared = np.einsum('ni,ni->n', v_D, v_U)
            v_squared = np.clip(v_squared, 0, 0.999999)  # Ensure v < c
            W = 1.0 / np.sqrt(1.0 - v_squared)

        # u^0 = W/α
        u4U[:, 0] = W / self.alpha

        # u^i = W v^i (Valencia formulation)
        # Note: This is the Valencia 4-velocity, NOT coordinate 4-velocity U^i = W v^i - β^i u^0
        for i in range(SPACEDIM):
            u4U[:, i+1] = W * v_U[:, i]

        return u4U

    def compute_4velocity_scalar(self, idx: int, v_U: np.ndarray, W: Optional[float] = None) -> np.ndarray:
        """
        Compute 4-velocity at a single point (for compatibility).

        Parameters
        ----------
        idx : int
            Grid point index
        v_U : np.ndarray
            Valencia 3-velocity v^i at point (3,)
        W : float, optional
            Lorentz factor at point

        Returns
        -------
        np.ndarray
            4-velocity u^μ at point (4,)
        """
        u4U = np.zeros(4)

        if W is None:
            v_D = np.dot(self.gamma_LL[idx], v_U)
            v_squared = np.dot(v_D, v_U)
            v_squared = np.clip(v_squared, 0, 0.999999)
            W = 1.0 / np.sqrt(1.0 - v_squared)

        u4U[0] = W / self.alpha[idx]

        # Valencia 4-velocity: u^i = W v^i (NOT coordinate form u^i = W v^i - β^i u^0)
        for i in range(SPACEDIM):
            u4U[i+1] = W * v_U[i]

        return u4U

    def get_christoffel_symbols(self, background) -> np.ndarray:
        """
        Get reference Christoffel symbols Γ̂^i_{jk} from background.

        Parameters
        ----------
        background : SphericalBackground
            Background metric with Christoffel symbols

        Returns
        -------
        np.ndarray
            Christoffel symbols Γ̂^i_{jk} (N, 3, 3, 3)
        """
        return background.hat_christoffel


class MinkowskiGeometry(ADMGeometry):
    """
    Specialized geometry for flat Minkowski spacetime.

    All curvature vanishes, lapse = 1, shift = 0, γ_ij = η_ij.
    """

    def __init__(self, N: int):
        """
        Initialize flat spacetime geometry.

        Parameters
        ----------
        N : int
            Number of grid points
        """
        # Flat spacetime: α = 1, β^i = 0, γ_ij = diag(1, 1, 1)
        alpha = np.ones(N)
        beta_U = np.zeros((N, SPACEDIM))

        gamma_LL = np.zeros((N, SPACEDIM, SPACEDIM))
        gamma_UU = np.zeros((N, SPACEDIM, SPACEDIM))

        for i in range(SPACEDIM):
            gamma_LL[:, i, i] = 1.0
            gamma_UU[:, i, i] = 1.0

        super().__init__(alpha, beta_U, gamma_LL, gamma_UU)


@dataclass
class ValenciaGeometry:
    """
    Extended geometry for Valencia formulation.

    Includes additional quantities needed for the Valencia formulation:
    - Volume elements for conservative form
    - Conformal factor contributions
    - Grid spacing information
    """
    # Base ADM quantities
    alpha: np.ndarray          # Lapse function
    beta_U: np.ndarray         # Shift vector β^i
    gamma_LL: np.ndarray       # Spatial metric γ_ij
    gamma_UU: np.ndarray       # Inverse metric γ^ij
    sqrt_gamma: np.ndarray     # √det(γ)

    # Valencia-specific quantities
    e6phi: np.ndarray          # e^{6φ} for densitization
    sqrt_g_hat_cell: np.ndarray  # √ĝ reference metric determinant
    dr: float                  # Grid spacing

    @classmethod
    def from_adm_geometry(cls, adm_geom: ADMGeometry, phi: np.ndarray,
                          background, dr: float):
        """
        Create Valencia geometry from ADM geometry.

        Parameters
        ----------
        adm_geom : ADMGeometry
            Base ADM geometry
        phi : np.ndarray
            Conformal factor φ from BSSN
        background : SphericalBackground
            Background metric
        dr : float
            Grid spacing

        Returns
        -------
        ValenciaGeometry
            Extended geometry for Valencia formulation
        """
        e2phi = np.exp(2.0 * phi)
        e6phi = e2phi * e2phi * e2phi

        # Get reference metric determinant
        if hasattr(background, 'sqrt_g_hat_cell'):
            sqrt_g_hat_cell = background.sqrt_g_hat_cell
        else:
            # For spherical coordinates: √ĝ = r²
            sqrt_g_hat_cell = background.r**2

        return cls(
            alpha=adm_geom.alpha,
            beta_U=adm_geom.beta_U,
            gamma_LL=adm_geom.gamma_LL,
            gamma_UU=adm_geom.gamma_UU,
            sqrt_gamma=adm_geom.sqrt_gamma,
            e6phi=e6phi,
            sqrt_g_hat_cell=sqrt_g_hat_cell,
            dr=dr
        )


def extract_valencia_geometry(r: np.ndarray,
                              bssn_vars,
                              spacetime_mode: str,
                              background,
                              grid) -> ValenciaGeometry:
    """
    Extract geometric quantities from BSSN variables for Valencia formulation.

    This function handles three spacetime modes:
    - 'fixed_minkowski': Flat Minkowski spacetime
    - 'fixed': Fixed background (e.g., Schwarzschild)
    - 'dynamic': Dynamical BSSN evolution

    Parameters
    ----------
    r : np.ndarray
        Radial coordinate array
    bssn_vars : BSSNStateVariables
        BSSN state variables
    spacetime_mode : str
        'fixed_minkowski', 'fixed', or 'dynamic'
    background : SphericalBackground
        Background metric object
    grid : Grid
        Grid object with spacing information

    Returns
    -------
    ValenciaGeometry
        Container with all geometric quantities needed for Valencia formulation
    """
    N = len(r)

    # Handle fixed_minkowski specially
    if spacetime_mode == 'fixed_minkowski':
        # Flat Minkowski spacetime
        alpha = np.ones(N)
        beta_U = np.zeros((N, SPACEDIM))
        gamma_LL = np.zeros((N, SPACEDIM, SPACEDIM))
        gamma_UU = np.zeros((N, SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            gamma_LL[:, i, i] = 1.0
            gamma_UU[:, i, i] = 1.0
        sqrt_gamma = np.ones(N)
        e6phi = np.ones(N)
        sqrt_g_hat_cell = np.asarray(background.sqrt_g_hat_cell, dtype=float) if hasattr(background, 'sqrt_g_hat_cell') else np.ones(N)
    else:
        # Extract BSSN variables
        if hasattr(bssn_vars, 'alpha'):
            alpha = np.asarray(bssn_vars.alpha, dtype=float)
        elif hasattr(bssn_vars, 'lapse'):
            alpha = np.asarray(bssn_vars.lapse, dtype=float)
        else:
            alpha = np.ones(N)

        # Extract shift and apply inverse scaling from background
        beta_U = np.zeros((N, SPACEDIM))
        if hasattr(bssn_vars, 'betaU'):
            beta_attr = np.asarray(bssn_vars.betaU, dtype=float)
        elif hasattr(bssn_vars, 'shift_U'):
            beta_attr = np.asarray(bssn_vars.shift_U, dtype=float)
        else:
            beta_attr = None

        if beta_attr is not None:
            if beta_attr.ndim >= 2:
                for i in range(min(SPACEDIM, beta_attr.shape[1])):
                    beta_U[:, i] = beta_attr[:, i]
            elif beta_attr.ndim == 1:
                beta_U[:, 0] = beta_attr

        # CRITICAL: Apply inverse scaling from background (BSSN shift is rescaled)
        if hasattr(background, 'inverse_scaling_vector'):
            beta_U = beta_U * background.inverse_scaling_vector

        phi = np.asarray(bssn_vars.phi, dtype=float)

        # Compute derived quantities
        e6phi = np.exp(6.0 * phi)

        if spacetime_mode == 'fixed':
            # For fixed background, check if it has precomputed gamma or use identity
            if hasattr(background, 'gamma_LL'):
                gamma_LL = background.gamma_LL
                gamma_UU = background.gamma_UU
                sqrt_gamma = background.sqrt_gamma
            else:
                # Use identity metric for FlatSphericalBackground
                gamma_LL = np.zeros((N, SPACEDIM, SPACEDIM))
                gamma_UU = np.zeros((N, SPACEDIM, SPACEDIM))
                for i in range(SPACEDIM):
                    gamma_LL[:, i, i] = 1.0
                    gamma_UU[:, i, i] = 1.0
                sqrt_gamma = np.ones(N)

            # Handle sqrt_g_hat_cell
            if hasattr(background, 'sqrt_g_hat_cell'):
                sqrt_g_hat_cell = np.asarray(background.sqrt_g_hat_cell, dtype=float)
            elif hasattr(background, 'det_hat_gamma'):
                sqrt_g_hat_cell = np.sqrt(np.abs(background.det_hat_gamma) + 1e-30)
            else:
                sqrt_g_hat_cell = np.ones(N)
        else:  # dynamic
            # Get conformal metric from BSSN (correct attribute is h_LL, not hDD)
            bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
            bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)

            # Compute determinant and square root (use numpy directly, more efficient)
            det_bar_gamma = np.linalg.det(bar_gamma_LL)
            sqrt_bar_gamma = np.sqrt(np.abs(det_bar_gamma) + 1e-30)

            # Physical metric: γ_ij = e^{4φ} γ̄_ij (not e^{6φ}!)
            e4phi = np.exp(4.0 * phi)
            gamma_LL = e4phi[:, None, None] * bar_gamma_LL
            gamma_UU = e4phi[:, None, None]**(-1) * bar_gamma_UU
            sqrt_gamma = e6phi * sqrt_bar_gamma

            # Handle sqrt_g_hat_cell (defensive: compute from det_hat_gamma if attribute missing)
            if hasattr(background, 'sqrt_g_hat_cell'):
                sqrt_g_hat_cell = np.asarray(background.sqrt_g_hat_cell, dtype=float)
            elif hasattr(background, 'det_hat_gamma'):
                sqrt_g_hat_cell = np.sqrt(np.abs(background.det_hat_gamma) + 1e-30)
            else:
                sqrt_g_hat_cell = np.ones(N)

    dr = grid.dx if hasattr(grid, 'dx') else (r[1] - r[0] if len(r) > 1 else 1.0)

    return ValenciaGeometry(
        alpha=alpha,
        beta_U=beta_U,
        gamma_LL=gamma_LL,
        gamma_UU=gamma_UU,
        sqrt_gamma=sqrt_gamma,
        e6phi=e6phi,
        sqrt_g_hat_cell=sqrt_g_hat_cell,
        dr=dr
    )
