# valencia_reference_metric.py
"""
Valencia formulation with reference metric - REFACTORED VERSION.

This module has been refactored to be a thin wrapper around modular components:
- geometry.py: ADMGeometry, ValenciaGeometry classes, extract_valencia_geometry()
- stress_energy.py: StressEnergyTensor computation
- grhd_equations.py: GRHDEquations orchestrator

The original monolithic implementation is preserved in valencia_reference_metric_backup.py

The ValenciaReferenceMetric class maintains backward compatibility with perfect_fluid.py
while delegating actual computation to the modular components.
"""

import numpy as np
from typing import Tuple

from source.matter.hydro.atmosphere import AtmosphereParams
from source.matter.hydro.grhd_equations import GRHDEquations
from source.matter.hydro.geometry import extract_valencia_geometry


class ValenciaReferenceMetric:
    """
    Valencia formulation - thin wrapper around modular GRHD components.

    This class maintains backward compatibility with perfect_fluid.py while
    delegating actual computation to:
    - grhd_equations.py: GRHDEquations for RHS computation
    - geometry.py: extract_valencia_geometry() for geometry extraction

    The original monolithic implementation is preserved in valencia_reference_metric_backup.py
    """

    def __init__(self, boundary_mode="parity", *, atmosphere=None,
                 atmosphere_rho=None, p_floor=None, v_max=None):
        """
        Initialize Valencia formulation.

        Parameters
        ----------
        boundary_mode : str
            "parity" - Parity boundary conditions at inner boundary (r=0)
            "outflow" - Outflow (zero-gradient) at both boundaries
        atmosphere : AtmosphereParams, optional
            Centralized atmosphere configuration
        atmosphere_rho : float, optional
            Deprecated - use atmosphere instead
        p_floor : float, optional
            Deprecated - use atmosphere instead
        v_max : float, optional
            Deprecated - use atmosphere instead
        """
        self.boundary_mode = boundary_mode

        # Handle backward compatibility with old parameter names
        if atmosphere is None:
            # Old-style: individual parameters
            if atmosphere_rho is not None or p_floor is not None or v_max is not None:
                atmosphere = AtmosphereParams(
                    rho_floor=atmosphere_rho if atmosphere_rho is not None else 1e-13,
                    p_floor=p_floor if p_floor is not None else 1e-15,
                    v_max=v_max if v_max is not None else 0.999999
                )
            else:
                # No parameters provided - use defaults
                atmosphere = AtmosphereParams()

        self.atmosphere = atmosphere

        # Create GRHDEquations orchestrator (EOS will be set later)
        self.grhd = GRHDEquations(
            eos=None,  # Will be provided in compute_rhs
            atmosphere=self.atmosphere,
            boundary_mode=self.boundary_mode
        )

        # Geometric quantities cache (required by perfect_fluid.py)
        self.alpha = None              # Lapse function
        self.beta_U = None             # Shift vector (N, 3)
        self.gamma_LL = None           # Physical metric (N, 3, 3)
        self.gamma_UU = None           # Inverse metric (N, 3, 3)
        self.sqrt_gamma = None         # √det(γ)
        self.e6phi = None              # e^{6φ}
        self.sqrt_g_hat_cell = None    # √det(ĝ)
        self.dr = None                 # Grid spacing

    def _extract_geometry(self, r: np.ndarray, bssn_vars, spacetime_mode: str,
                         background, grid):
        """
        Extract geometry from BSSN variables and cache as attributes.

        This method is called by perfect_fluid.py to extract geometry before
        conservative-to-primitive conversion. Results are cached as instance
        attributes for later access.

        Delegates to geometry.extract_valencia_geometry() from geometry.py.

        Parameters
        ----------
        r : np.ndarray
            Radial coordinate
        bssn_vars : BSSNStateVariables
            BSSN evolution variables
        spacetime_mode : str
            'fixed_minkowski', 'fixed', or 'dynamic'
        background : SphericalBackground
            Background metric
        grid : Grid
            Grid with spacing information
        """
        # Use modular geometry extraction from geometry.py
        geom = extract_valencia_geometry(r, bssn_vars, spacetime_mode, background, grid)

        # Cache attributes (required by perfect_fluid.py)
        self.alpha = geom.alpha
        self.beta_U = geom.beta_U
        self.gamma_LL = geom.gamma_LL
        self.gamma_UU = geom.gamma_UU
        self.sqrt_gamma = geom.sqrt_gamma
        self.e6phi = geom.e6phi
        self.sqrt_g_hat_cell = geom.sqrt_g_hat_cell
        self.dr = geom.dr

    def compute_rhs(self, D: np.ndarray, S_tildeD: np.ndarray, tau: np.ndarray,
                    rho0: np.ndarray, v_U: np.ndarray, pressure: np.ndarray,
                    W: np.ndarray, h: np.ndarray,
                    r: np.ndarray, bssn_vars, bssn_d1, background,
                    spacetime_mode: str, eos, grid,
                    reconstructor, riemann_solver) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute RHS for Valencia formulation - delegates to GRHDEquations.

        This is the main public interface called by perfect_fluid.py for evolution.

        Parameters
        ----------
        D, S_tildeD, tau : np.ndarray
            Conservative variables
        rho0, v_U, pressure, W, h : np.ndarray
            Primitive variables and derived quantities
        r : np.ndarray
            Radial coordinate
        bssn_vars, bssn_d1 : BSSN variables and derivatives
        background : Background metric
        spacetime_mode : str
            'fixed_minkowski', 'fixed', or 'dynamic'
        eos : PolytropicEOS
            Equation of state
        grid : Grid
            Grid object
        reconstructor : Reconstruction
            Reconstruction method
        riemann_solver : RiemannSolver
            Riemann solver

        Returns
        -------
        rhs_D, rhs_S_tildeD, rhs_tau : np.ndarray
            Right-hand sides for conservative evolution
        """
        # Extract geometry first using modular function from geometry.py
        geometry = extract_valencia_geometry(r, bssn_vars, spacetime_mode, background, grid)

        # Update EOS on grhd if needed
        if self.grhd.eos is None or self.grhd.eos is not eos:
            self.grhd.eos = eos

        # Delegate to GRHDEquations orchestrator from grhd_equations.py
        return self.grhd.compute_rhs(
            D, S_tildeD, tau, rho0, v_U, pressure, W, h,
            geometry, bssn_vars, bssn_d1, background,
            reconstructor, riemann_solver, spacetime_mode, r
        )
