"""
Stress-Energy Tensor Module for GRHD

This module provides classes for computing all forms of the stress-energy tensor
in general relativistic hydrodynamics, including its projections to 3+1 ADM form.

Classes:
    StressEnergyTensor: Computes T^{μν} and related quantities
    EMTensor: 3+1 ADM projection for BSSN coupling
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from source.bssn.tensoralgebra import SPACEDIM
from source.matter.hydro.geometry import ADMGeometry


class StressEnergyTensor:
    """
    Computes all forms of the stress-energy tensor for perfect fluids.

    The stress-energy tensor for a perfect fluid is:
    T^{μν} = ρ₀ h u^μ u^ν + P g^{μν}

    where:
    - ρ₀ is the rest mass density
    - h is the specific enthalpy
    - u^μ is the 4-velocity
    - P is the pressure
    - g^{μν} is the spacetime metric
    """

    def __init__(self, geometry: ADMGeometry, rho0: np.ndarray, v_U: np.ndarray,
                 pressure: np.ndarray, W: np.ndarray, h: np.ndarray):
        """
        Initialize stress-energy tensor computation.

        Parameters
        ----------
        geometry : ADMGeometry
            Spacetime geometry (metric, lapse, shift)
        rho0 : np.ndarray
            Rest mass density (N,)
        v_U : np.ndarray
            Valencia 3-velocity v^i (N, 3)
        pressure : np.ndarray
            Pressure (N,)
        W : np.ndarray
            Lorentz factor (N,)
        h : np.ndarray
            Specific enthalpy (N,)
        """
        self.geometry = geometry
        self.rho0 = rho0
        self.v_U = v_U
        self.pressure = pressure
        self.W = W
        self.h = h

        # Compute 4-velocity once and cache it
        self._u4U = None
        self._g4UU = None
        self._g4DD = None

    @property
    def u4U(self) -> np.ndarray:
        """Get cached 4-velocity u^μ."""
        if self._u4U is None:
            self._u4U = self.geometry.compute_4velocity(self.v_U, self.W)
        return self._u4U

    @property
    def g4UU(self) -> np.ndarray:
        """Get cached contravariant 4-metric g^{μν}."""
        if self._g4UU is None:
            self._g4UU = self.geometry.get_4metric_contravariant()
        return self._g4UU

    @property
    def g4DD(self) -> np.ndarray:
        """Get cached covariant 4-metric g_{μν}."""
        if self._g4DD is None:
            self._g4DD = self.geometry.get_4metric_covariant()
        return self._g4DD

    def compute_T4UU(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute contravariant stress-energy tensor T^{μν}.

        Returns
        -------
        T00 : np.ndarray
            T^{00} component (N,)
        T0i : np.ndarray
            T^{0i} components (N, 3)
        Tij : np.ndarray
            T^{ij} components (N, 3, 3)
        """
        N = len(self.rho0)
        u4U = self.u4U
        g4UU = self.g4UU

        # Perfect fluid stress-energy tensor:
        # T^{μν} = ρ₀ h u^μ u^ν + P g^{μν}

        rho_h = self.rho0 * self.h

        # T^{00} = ρ₀ h u^0 u^0 + P g^{00}
        T00 = rho_h * u4U[:, 0] * u4U[:, 0] + self.pressure * g4UU[:, 0, 0]

        # T^{0i} = ρ₀ h u^0 u^i + P g^{0i}
        T0i = np.zeros((N, SPACEDIM))
        for i in range(SPACEDIM):
            T0i[:, i] = (rho_h * u4U[:, 0] * u4U[:, i+1] +
                         self.pressure * g4UU[:, 0, i+1])

        # T^{ij} = ρ₀ h u^i u^j + P g^{ij}
        Tij = np.zeros((N, SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                Tij[:, i, j] = (rho_h * u4U[:, i+1] * u4U[:, j+1] +
                                self.pressure * g4UU[:, i+1, j+1])

        return T00, T0i, Tij

    def compute_T4UD(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute mixed stress-energy tensor T^μ_ν.

        The mixed tensor is: T^μ_ν = T^{μδ} g_{δν}

        Returns
        -------
        T0_0 : np.ndarray
            T^0_0 component (N,)
        T0_j : np.ndarray
            T^0_j components (N, 3)
        Ti_j : np.ndarray
            T^i_j components (N, 3, 3)
        """
        # First get contravariant components
        T00, T0i, Tij = self.compute_T4UU()

        # Get covariant metric
        g4DD = self.g4DD

        N = len(self.rho0)

        # Lower the second index using g_{μν}
        # T^0_0 = T^{0μ} g_{μ0}
        T0_0 = T00 * g4DD[:, 0, 0]
        for i in range(SPACEDIM):
            T0_0 += T0i[:, i] * g4DD[:, i+1, 0]

        # T^0_j = T^{0μ} g_{μj}
        T0_j = np.zeros((N, SPACEDIM))
        for j in range(SPACEDIM):
            T0_j[:, j] = T00 * g4DD[:, 0, j+1]
            for i in range(SPACEDIM):
                T0_j[:, j] += T0i[:, i] * g4DD[:, i+1, j+1]

        # T^i_j = T^{iμ} g_{μj}
        Ti_j = np.zeros((N, SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                # T^i_0 g_{0j} contribution
                Ti_j[:, i, j] = T0i[:, i] * g4DD[:, 0, j+1]
                # T^{ik} g_{kj} contributions
                for k in range(SPACEDIM):
                    Ti_j[:, i, j] += Tij[:, i, k] * g4DD[:, k+1, j+1]

        return T0_0, T0_j, Ti_j

    def compute_T4DD(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute covariant stress-energy tensor T_{μν}.

        The covariant tensor is: T_{μν} = g_{μρ} T^{ρσ} g_{σν}

        Returns
        -------
        T_00 : np.ndarray
            T_{00} component (N,)
        T_0i : np.ndarray
            T_{0i} components (N, 3)
        T_ij : np.ndarray
            T_{ij} components (N, 3, 3)
        """
        # First get contravariant components
        T00, T0i, Tij = self.compute_T4UU()

        # Get covariant metric
        g4DD = self.g4DD

        N = len(self.rho0)

        # Lower both indices using g_{μν}
        # T_{00} = g_{0μ} T^{μν} g_{ν0}
        T_00 = np.zeros(N)
        T_00 += g4DD[:, 0, 0] * T00 * g4DD[:, 0, 0]
        for i in range(SPACEDIM):
            T_00 += 2.0 * g4DD[:, 0, i+1] * T0i[:, i] * g4DD[:, 0, 0]
            for j in range(SPACEDIM):
                T_00 += g4DD[:, 0, i+1] * Tij[:, i, j] * g4DD[:, j+1, 0]

        # T_{0i} = g_{0μ} T^{μν} g_{νi}
        T_0i = np.zeros((N, SPACEDIM))
        for i in range(SPACEDIM):
            T_0i[:, i] += g4DD[:, 0, 0] * T00 * g4DD[:, 0, i+1]
            for j in range(SPACEDIM):
                T_0i[:, i] += g4DD[:, 0, 0] * T0i[:, j] * g4DD[:, j+1, i+1]
                T_0i[:, i] += g4DD[:, 0, j+1] * T0i[:, j] * g4DD[:, 0, i+1]
                for k in range(SPACEDIM):
                    T_0i[:, i] += g4DD[:, 0, j+1] * Tij[:, j, k] * g4DD[:, k+1, i+1]

        # T_{ij} = g_{iμ} T^{μν} g_{νj}
        T_ij = np.zeros((N, SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                T_ij[:, i, j] += g4DD[:, i+1, 0] * T00 * g4DD[:, 0, j+1]
                for k in range(SPACEDIM):
                    T_ij[:, i, j] += g4DD[:, i+1, 0] * T0i[:, k] * g4DD[:, k+1, j+1]
                    T_ij[:, i, j] += g4DD[:, i+1, k+1] * T0i[:, k] * g4DD[:, 0, j+1]
                    for l in range(SPACEDIM):
                        T_ij[:, i, j] += g4DD[:, i+1, k+1] * Tij[:, k, l] * g4DD[:, l+1, j+1]

        return T_00, T_0i, T_ij

    def project_to_ADM(self) -> 'EMTensor':
        """
        Project stress-energy tensor to 3+1 ADM form for BSSN coupling.

        The ADM projections are:
        - ρ = n_μ n_ν T^{μν} (energy density seen by normal observers)
        - S_i = -γ_{iμ} n_ν T^{μν} (momentum density)
        - S_{ij} = γ_{iμ} γ_{jν} T^{μν} (stress tensor)

        where n_μ is the unit normal to spatial slices.

        Returns
        -------
        EMTensor
            ADM-projected stress-energy components
        """
        # For perfect fluid with Valencia variables, the projections are:
        # ρ = ρ₀ h W² - P
        # S_i = ρ₀ h W² v_i
        # S_{ij} = ρ₀ h W² v_i v_j + P γ_{ij}
        # S = γ^{ij} S_{ij} = ρ₀ h W² v² + 3P

        N = len(self.rho0)

        # Compute ρ₀ h W²
        rho_h_W2 = self.rho0 * self.h * self.W**2

        # Energy density
        rho = rho_h_W2 - self.pressure

        # Momentum density (lower index)
        # v_i = γ_{ij} v^j
        v_D = np.einsum('nij,nj->ni', self.geometry.gamma_LL, self.v_U)
        S_D = rho_h_W2[:, np.newaxis] * v_D

        # Stress tensor
        S_DD = np.zeros((N, SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                S_DD[:, i, j] = (rho_h_W2 * v_D[:, i] * v_D[:, j] +
                                 self.pressure * self.geometry.gamma_LL[:, i, j])

        # Trace S = γ^{ij} S_{ij}
        S_trace = np.einsum('nij,nij->n', self.geometry.gamma_UU, S_DD)

        return EMTensor(rho, S_D, S_DD, S_trace)


@dataclass
class EMTensor:
    """
    Energy-momentum tensor in 3+1 ADM form.

    This represents the projection of T^{μν} onto spatial hypersurfaces,
    as needed for coupling to BSSN evolution.
    """
    rho: np.ndarray      # Energy density ρ = n_μ n_ν T^{μν}
    S_D: np.ndarray      # Momentum density S_i = -γ_{iμ} n_ν T^{μν}
    S_DD: np.ndarray     # Stress tensor S_{ij} = γ_{iμ} γ_{jν} T^{μν}
    S: np.ndarray        # Trace S = γ^{ij} S_{ij}


class StressEnergyTensor4D:
    """
    Efficient container for 4D stress-energy tensor components.

    Uses __slots__ for better performance (eliminates dict overhead).
    This class is primarily for compatibility with existing test code.
    """
    __slots__ = ('T00', 'T0i', 'Tij')

    def __init__(self, N: int):
        """
        Preallocate arrays for N grid points.

        Parameters
        ----------
        N : int
            Number of grid points
        """
        self.T00 = np.zeros(N)
        self.T0i = np.zeros((N, SPACEDIM))
        self.Tij = np.zeros((N, SPACEDIM, SPACEDIM))

    @classmethod
    def from_components(cls, T00: np.ndarray, T0i: np.ndarray, Tij: np.ndarray):
        """
        Create from existing components.

        Parameters
        ----------
        T00 : np.ndarray
            T^{00} component (N,)
        T0i : np.ndarray
            T^{0i} components (N, 3)
        Tij : np.ndarray
            T^{ij} components (N, 3, 3)

        Returns
        -------
        StressEnergyTensor4D
            Container with the provided components
        """
        N = len(T00)
        obj = cls(N)
        obj.T00 = T00
        obj.T0i = T0i
        obj.Tij = Tij
        return obj