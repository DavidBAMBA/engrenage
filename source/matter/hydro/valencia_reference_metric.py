# valencia_reference_metric.py
"""
Valencia formulation with reference metric - full 3D BSSN-style implementation.

Follows the exact same tensor algebra pattern as bssnrhs.py:
- Full 3D einsum contractions for all tensor operations
- Spherical symmetry imposed by velocity (v^θ = v^φ = 0) and background metric
- Source terms following NRPy+ GRHD_equations.py structure
- Connection terms with CORRECT signs for covariant divergence

Conservative evolution equations in curved coordinates (two equivalent views):
    Densitized form:  ∂_t(U) + (1/√ĝ) ∂_j[√ĝ F^j] = S
    NRPy form used here:  ∂_t(U) + ∂_j(F̃^j) = S + connection
with F̃^j = α √γ F^j (no √ĝ factor). The explicit connection pieces are
    D, τ:   -Γ̂^k_{kj} F̃^j
    S_i:   -Γ̂^k_{kj} F̃^j_i + Γ̂^l_{ji} F̃^j_l

Where:
    U = (D, S_i, τ)  conserved variables
    F^j = physical fluxes
    S = geometric source terms (K_ij, ∂_iα, ∇̂γ_{ij} couplings)
"""

import numpy as np
from types import SimpleNamespace
from .atmosphere import AtmosphereParams
from .geometry import ADMGeometry, ValenciaGeometry, MinkowskiGeometry
from .grhd_equations import GRHDEquations
from .stress_energy import StressEnergyTensor, StressEnergyTensor4D

from source.bssn.tensoralgebra import SPACEDIM
from source.core.spacing import NUM_GHOSTS


class ValenciaReferenceMetric:
    """Valencia formulation - full 3D tensor algebra following BSSN pattern."""

    def __init__(self, boundary_mode="parity", *, atmosphere=None,
                 atmosphere_rho=None, p_floor=None, v_max=None):
        """
        Initialize Valencia formulation.

        Parameters
        -----------
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

        # Geometric quantities
        # These will be computed by _extract_geometry()
        self.alpha = None              # Lapse function
        self.beta_U = None             # Shift vector (N, 3)
        self.gamma_LL = None           # Physical metric (N, 3, 3)
        self.gamma_UU = None           # Inverse metric (N, 3, 3)
        self.sqrt_gamma = None         # √γ determinant factor
        self.e6phi = None              # e^(6φ) conformal factor
        self.sqrt_g_hat_cell = None    # √ĝ at cell centers
        self.sqrt_g_hat_face = None    # √ĝ at cell faces
        self.dr = None                 # Mesh spacing

        # Cached geometry objects for reuse
        self._adm_geometry = None
        self._valencia_geometry = None
        self._phi = None

        # Debug placeholders (populated during computations)
        self._debug_T4UU = None
        self._debug_T4UD = None
        self._debug_flux_density = None
        self._debug_flux_energy = None
        self._debug_flux_momentum = None
        self._debug_connection_density = None
        self._debug_connection_energy = None
        self._debug_connection_momentum = None
        self._debug_energy_source_Kij_term = None
        self._debug_energy_source_dalpha_term = None
        self._debug_energy_source_total = None
        self._debug_momentum_source_T00_alpha_term = None
        self._debug_momentum_source_T0j_beta_term = None
        self._debug_momentum_source_metric_term = None
        self._debug_momentum_source_total = None
        self._debug_hatD_beta_U = None

    def _extract_geometry(self, r, bssn_vars, spacetime_mode, background, grid):
        """
        Extract geometric quantities from BSSN variables and store as class attributes.

        IMPORTANT RESCALING NOTES:
        - bssn_vars.shift_U is the RESCALED shift (needs inverse_scaling_vector)
        - bssn_vars.h_LL is the RESCALED deviation (already includes scaling_matrix)
        - bar_gamma_LL = h_LL * scaling_matrix + hat_gamma_LL (includes scale factors)
        - Physical shift: beta^i = inverse_scaling_vector^i * shift_U^i

        Stores all geometric quantities as class attributes, including mesh spacing.
        """
        N = len(r)

        # Mesh spacing (computed once per timestep)
        if grid is not None and hasattr(grid, 'derivs') and hasattr(grid.derivs, 'dx'):
            self.dr = float(grid.derivs.dx)
        elif grid is not None and hasattr(grid, 'dr'):
            self.dr = float(grid.dr)
        elif grid is not None and hasattr(grid, 'spacing') and np.isscalar(grid.spacing):
            self.dr = float(grid.spacing)
        elif grid is not None and hasattr(grid, 'spacing') and hasattr(grid.spacing, '__len__'):
            # For 3D grids with multiple spacings, return radial spacing
            self.dr = float(grid.spacing[0])
        else:
            # Fallback: compute from radial coordinate array (used when grid is None)
            self.dr = float(r[1] - r[0]) if len(r) > 1 else 1.0

        if spacetime_mode == "fixed_minkowski":
            adm_geom = MinkowskiGeometry(N)
            phi = np.zeros(N, dtype=float)
        else:
            alpha = np.asarray(getattr(bssn_vars, 'alpha', getattr(bssn_vars, 'lapse')), dtype=float)
            shift_src = getattr(bssn_vars, 'betaU', getattr(bssn_vars, 'shift_U', None))
            if shift_src is not None:
                beta_U = np.asarray(shift_src, dtype=float)
            else:
                beta_U = np.zeros((N, SPACEDIM))
            if hasattr(background, 'inverse_scaling_vector'):
                beta_U = beta_U * background.inverse_scaling_vector

            phi = np.asarray(getattr(bssn_vars, 'phi', np.zeros(N)), dtype=float)
            h_dd = getattr(bssn_vars, 'hDD', None)
            h_ll = getattr(bssn_vars, 'h_LL', None)

            bssn_like = SimpleNamespace(
                alpha=alpha,
                betaU=beta_U,
                phi=phi,
                hDD=h_dd,
                h_LL=h_ll,
                betaU_needs_rescaling=getattr(bssn_vars, 'betaU_needs_rescaling', False),
            )

            adm_geom = ADMGeometry.from_bssn(bssn_like, background)

        val_geom = ValenciaGeometry.from_adm_geometry(adm_geom, phi, background, self.dr)

        self.alpha = adm_geom.alpha
        self.beta_U = adm_geom.beta_U
        self.gamma_LL = adm_geom.gamma_LL
        self.gamma_UU = adm_geom.gamma_UU
        self.sqrt_gamma = adm_geom.sqrt_gamma
        self.e6phi = val_geom.e6phi
        self.sqrt_g_hat_cell = val_geom.sqrt_g_hat_cell
        if getattr(val_geom, 'sqrt_g_hat_cell', None) is not None:
            if len(val_geom.sqrt_g_hat_cell) > 1:
                self.sqrt_g_hat_face = 0.5 * (val_geom.sqrt_g_hat_cell[:-1] + val_geom.sqrt_g_hat_cell[1:])
            else:
                self.sqrt_g_hat_face = val_geom.sqrt_g_hat_cell.copy()
        else:
            self.sqrt_g_hat_face = None

        self._adm_geometry = adm_geom
        self._valencia_geometry = val_geom
        self._phi = phi

    @property
    def adm_geometry(self) -> ADMGeometry:
        if self._adm_geometry is None:
            raise RuntimeError("ADM geometry not initialized; call _extract_geometry first")
        return self._adm_geometry

    @property
    def valencia_geometry(self) -> ValenciaGeometry:
        if self._valencia_geometry is None:
            raise RuntimeError("Valencia geometry not initialized; call _extract_geometry first")
        return self._valencia_geometry

    def compute_rhs(self, D, S_tildeD, tau, rho0, v_U, pressure, W, h,
                    r, bssn_vars, bssn_d1, background, spacetime_mode,
                    eos, grid, reconstructor, riemann_solver):
        """
        Compute RHS of Valencia equations using the modular GRHDEquations pipeline
        while preserving the legacy ValenciaReferenceMetric interface and debug fields.

        Variables:
            D: (N,) - densitized rest-mass density
            S_tildeD: (N, 3) - densitized momentum (full 3D vector)
            tau: (N,) - densitized energy
            v_U: (N, 3) - spatial velocity (full 3D vector)
            W: (N,) or None - Lorentz factor (computed if None)
            h: (N,) or None - specific enthalpy (computed if None)

        Returns:
            rhs_D: (N,) - time derivative of D
            rhs_S_tildeD: (N, 3) - time derivative of momentum vector (always 3D)
            rhs_tau: (N,) - time derivative of tau
        """
        N = len(r)

        # Input validation
        v_U = np.asarray(v_U)
        S_tildeD = np.asarray(S_tildeD)

        if v_U.shape != (N, SPACEDIM):
            raise ValueError(f"v_U must have shape ({N}, {SPACEDIM}), got {v_U.shape}")
        if S_tildeD.shape != (N, SPACEDIM):
            raise ValueError(f"S_tildeD must have shape ({N}, {SPACEDIM}), got {S_tildeD.shape}")

        # Enforce v^r = 0 at origin for spherical spacetimes
        if self.boundary_mode == "parity" and spacetime_mode != "fixed_minkowski":
            v_U[NUM_GHOSTS, 0] = 0.0

        # Extract geometry for W and h computation if needed
        self._extract_geometry(r, bssn_vars, spacetime_mode, background, grid)

        # Compute W and h if not provided
        if W is None:
            v_squared = np.einsum('xij,xi,xj->x', self.gamma_LL, v_U, v_U)
            W = 1.0 / np.sqrt(np.maximum(1.0 - v_squared, 1e-16))

        if h is None:
            eps = eos.eps_from_rho_p(rho0, pressure)
            h = 1.0 + eps + pressure / np.maximum(rho0, 1e-30)

        val_geom = self.valencia_geometry
        adm_geom = self.adm_geometry

        # Prepare face-centered geometry for Riemann solves
        alpha_face = 0.5 * (val_geom.alpha[:-1] + val_geom.alpha[1:]) if len(val_geom.alpha) > 1 else val_geom.alpha.copy()
        beta_face = 0.5 * (val_geom.beta_U[:-1] + val_geom.beta_U[1:]) if len(val_geom.beta_U) > 1 else val_geom.beta_U.copy()
        gamma_face = 0.5 * (val_geom.gamma_LL[:-1] + val_geom.gamma_LL[1:]) if len(val_geom.gamma_LL) > 1 else val_geom.gamma_LL.copy()

        phi_arr = np.asarray(getattr(bssn_vars, 'phi', np.zeros_like(val_geom.alpha)), dtype=float)
        if phi_arr.size > 1:
            phi_face = 0.5 * (phi_arr[:-1] + phi_arr[1:])
        else:
            phi_face = phi_arr.copy()
        e6phi_face = np.exp(6.0 * phi_face)

        class _FaceGeometry:
            __slots__ = ("alpha", "beta_U", "gamma_LL", "e6phi")

            def __init__(self, alpha, beta_U, gamma_LL, e6phi):
                self.alpha = alpha
                self.beta_U = beta_U
                self.gamma_LL = gamma_LL
                self.e6phi = e6phi

        face_geom = _FaceGeometry(alpha_face, beta_face, gamma_face, e6phi_face)

        grhd = GRHDEquations(eos, atmosphere=self.atmosphere, boundary_mode=self.boundary_mode)

        # Reconstruction and Riemann solve
        UL_batch, UR_batch, primL_batch, primR_batch = grhd.reconstruct_and_convert(
            rho0, v_U[:, 0], pressure, r, reconstructor, face_geom
        )
        F_D_face, F_S_face, F_tau_face = grhd.solve_riemann_and_densitize(
            UL_batch, UR_batch, primL_batch, primR_batch,
            face_geom, riemann_solver, spacetime_mode
        )

        rhs_D, rhs_S, rhs_tau = grhd.compute_divergence(F_D_face, F_S_face, F_tau_face, val_geom.dr)

        # Stress-energy tensor diagnostics (for tests)
        stress = StressEnergyTensor(adm_geom, rho0, v_U, pressure, W, h)
        T00, T0i, Tij = stress.compute_T4UU()
        self._debug_T4UU = StressEnergyTensor4D.from_components(T00, T0i, Tij)
        T0_0, T0_j, Ti_j = stress.compute_T4UD()
        self._debug_T4UD = StressEnergyTensor4D.from_components(T0_0, T0_j, Ti_j)

        # Default debug placeholders
        zero_scalar = np.zeros_like(D)
        zero_vector = np.zeros((len(D), SPACEDIM))
        zero_tensor = np.zeros((len(D), SPACEDIM, SPACEDIM))

        self._debug_flux_density = zero_vector.copy()
        self._debug_flux_energy = zero_vector.copy()
        self._debug_flux_momentum = zero_tensor.copy()
        self._debug_connection_density = zero_scalar.copy()
        self._debug_connection_energy = zero_scalar.copy()
        self._debug_connection_momentum = zero_vector.copy()

        # Connection contributions (skip in Minkowski)
        if spacetime_mode != 'fixed_minkowski':
            conn_D, conn_S, conn_tau, conn_debug = grhd.compute_connection_terms(
                rho0, v_U, pressure, W, h, val_geom, background.hat_christoffel,
                return_debug=True
            )
            rhs_D += conn_D
            rhs_S += conn_S
            rhs_tau += conn_tau

            self._debug_flux_density = conn_debug["rho_star_flux"]
            self._debug_flux_energy = conn_debug["tau_flux"]
            self._debug_flux_momentum = conn_debug["momentum_flux"]
            self._debug_connection_density = conn_D
            self._debug_connection_energy = conn_tau
            self._debug_connection_momentum = conn_S

        # Source terms (dynamic spacetime)
        if spacetime_mode == 'dynamic':
            src_S, src_tau, src_debug = grhd.compute_source_terms(
                rho0, v_U, pressure, W, h, val_geom,
                bssn_vars, bssn_d1, background,
                spacetime_mode, r,
                return_debug=True
            )
            rhs_S += src_S
            rhs_tau += src_tau
            self._debug_energy_source_Kij_term = src_debug["energy_Kij"]
            self._debug_energy_source_dalpha_term = src_debug["energy_dalpha"]
            self._debug_energy_source_total = src_debug["energy_total"]
            self._debug_momentum_source_T00_alpha_term = src_debug["momentum_T00_alpha"]
            self._debug_momentum_source_T0j_beta_term = src_debug["momentum_T0j_beta"]
            self._debug_momentum_source_metric_term = src_debug["momentum_metric"]
            self._debug_momentum_source_total = src_debug["momentum_total"]
            self._debug_hatD_beta_U = src_debug["hatD_beta"]
        else:
            # Ensure source debug fields exist even in non-dynamic runs
            zero_hat = np.zeros((len(D), SPACEDIM, SPACEDIM))
            self._debug_energy_source_Kij_term = zero_scalar.copy()
            self._debug_energy_source_dalpha_term = zero_scalar.copy()
            self._debug_energy_source_total = zero_scalar.copy()
            self._debug_momentum_source_T00_alpha_term = zero_vector.copy()
            self._debug_momentum_source_T0j_beta_term = zero_vector.copy()
            self._debug_momentum_source_metric_term = zero_vector.copy()
            self._debug_momentum_source_total = zero_vector.copy()
            self._debug_hatD_beta_U = zero_hat

        return rhs_D, rhs_S, rhs_tau
