import numpy as np

from source.core.grid import Grid, i_x1, i_x2, i_x3
from source.bssn.bssnstatevariables import NUM_BSSN_VARS
from source.bssn.bssnvars import BSSNVars
from source.bssn.tensoralgebra import SPACEDIM, EMTensor
from source.backgrounds.sphericalbackground import i_r

from .hydro.valencia_reference_metric import ValenciaReferenceMetric
from .hydro.cons2prim import Cons2PrimSolver
from .hydro.eos import IdealGasEOS
from .hydro.atmosphere import AtmosphereParams, create_default_atmosphere
from .hydro.geometry import ADMGeometry
from .hydro.stress_energy import StressEnergyTensor


class PerfectFluid:
    """
    Relativistic perfect fluid implementation for engrenage.

    Implements Valencia formulation for conservative evolution and provides
    stress-energy tensor for BSSN coupling.


    """

    def __init__(self, eos=None, spacetime_mode="dynamic",
                 atmosphere=None, reconstructor=None, riemann_solver=None):
        """
        Initialize relativistic perfect fluid matter.

        Args:
            eos: Equation of state (default: IdealGasEOS)
            spacetime_mode: "dynamic" or "fixed" (default: "dynamic")
            atmosphere: AtmosphereParams object or float (rho_floor). If None, uses default.
            reconstructor: Reconstruction method (optional)
            riemann_solver: Riemann solver (optional)
        """

        # engrenage BaseMatter interface requirements
        self.NUM_MATTER_VARS = 3
        self.VARIABLE_NAMES = np.array(["D_tilde", "Sr_tilde", "tau_tilde"])
        self.PARITY = np.array([1, -1, 1])  # D_tilde: even, Sr_tilde: odd, tau_tilde: even
        self.ASYMP_POWER = np.array([0, 0, 0])
        self.ASYMP_OFFSET = np.array([0, 0, 0])

        # Variable indices in state vector
        self.idx_D = NUM_BSSN_VARS
        self.idx_Sr = NUM_BSSN_VARS + 1
        self.idx_tau = NUM_BSSN_VARS + 2
        self.indices = np.array([self.idx_D, self.idx_Sr, self.idx_tau])

        # Physics
        self.eos = eos if eos is not None else IdealGasEOS()
        self.spacetime_mode = spacetime_mode

        # Atmosphere configuration
        if atmosphere is None:
            self.atmosphere = AtmosphereParams()  # Default
        elif isinstance(atmosphere, (int, float)):
            # Convenience: just specify rho_floor
            self.atmosphere = create_default_atmosphere(rho_floor=float(atmosphere))
        elif isinstance(atmosphere, AtmosphereParams):
            self.atmosphere = atmosphere
        else:
            raise TypeError(f"atmosphere must be AtmosphereParams, float, or None, got {type(atmosphere)}")

        # Create cons2prim solver with centralized atmosphere
        self.cons2prim_solver = Cons2PrimSolver(
            self.eos,
            atmosphere=self.atmosphere,
            tol=1e-10,
            max_iter=100
        )

        # Numerical methods for Valencia evolution
        self.valencia = ValenciaReferenceMetric(atmosphere=self.atmosphere )
        self.reconstructor = reconstructor
        # CRITICAL FIX: Pass atmosphere to reconstructor so floors are applied
        if self.reconstructor is not None:
            self.reconstructor.atmosphere = self.atmosphere
        self.riemann_solver = riemann_solver

        # State variables (engrenage pattern)
        self.matter_vars_set = False
        self.grid = None
        self.background = None

        # Cache for pressure guesses (improves cons2prim performance)
        self.pressure_cache = None

        # Conservative variables (densitized: Ũ = e^{6φ} U)
        self.D_tilde = None      # Densitized rest mass density: D̃ = e^{6φ} D
        self.Sr_tilde = None     # Densitized radial momentum: S̃r = e^{6φ} Sr
        self.tau_tilde = None    # Densitized energy: τ̃ = e^{6φ} tau

        # Fixed stress-energy tensor sources (for hydro-without-hydro tests)
        # When set, get_emtensor() returns these instead of computing from primitives
        self.use_fixed_emtensor = False
        self.fixed_emtensor_rho = None   # ρ = n_μ n_ν T^{μν}
        self.fixed_emtensor_Si = None    # S_i = -γ_{iμ} n_ν T^{μν}
        self.fixed_emtensor_Sij = None   # S_{ij} = γ_{iμ} γ_{jν} T^{μν}
        self.fixed_emtensor_S = None     # S = γ^{ij} S_{ij}

    def set_matter_vars(self, state_vector, bssn_vars: BSSNVars, grid: Grid):
        """Extract densitized matter variables from state vector (engrenage interface)."""
        # bssn_vars not needed here but kept for interface consistency
        _ = bssn_vars  # suppress warning

        # Extract densitized conservative variables: Ũ = e^{6φ} U
        self.D_tilde = state_vector[self.idx_D].copy()
        self.Sr_tilde = state_vector[self.idx_Sr].copy()
        self.tau_tilde = state_vector[self.idx_tau].copy()
        self.grid = grid
        self.matter_vars_set = True

    def set_fixed_emtensor_sources(self, rho, Si, Sij, S):
        """
        Set fixed stress-energy tensor sources for hydro-without-hydro tests.

        In the hydro-without-hydro test (Baumgarte et al. 1999), the matter
        sources T^{μν} are kept completely fixed while evolving only the
        gravitational field equations. This prevents the sources from changing
        as the metric evolves, which would cause spurious K growth.

        Parameters
        ----------
        rho : np.ndarray
            Energy density ρ = n_μ n_ν T^{μν} (N,)
        Si : np.ndarray
            Momentum density S_i = -γ_{iμ} n_ν T^{μν} (N, 3)
        Sij : np.ndarray
            Stress tensor S_{ij} = γ_{iμ} γ_{jν} T^{μν} (N, 3, 3)
        S : np.ndarray
            Trace S = γ^{ij} S_{ij} (N,)
        """
        self.fixed_emtensor_rho = rho.copy()
        self.fixed_emtensor_Si = Si.copy()
        self.fixed_emtensor_Sij = Sij.copy()
        self.fixed_emtensor_S = S.copy()
        self.use_fixed_emtensor = True

    def disable_fixed_emtensor_sources(self):
        """Disable fixed stress-energy sources and revert to computing from primitives."""
        self.use_fixed_emtensor = False

    def get_emtensor(self, r, bssn_vars, background):
        assert self.matter_vars_set, 'Matter vars not set'
        N = len(r)
        emtensor = EMTensor(N)

        # Hydro-without-hydro mode: Return fixed sources
        # This prevents T^{μν} from changing as the metric evolves
        if self.use_fixed_emtensor:
            emtensor.rho = self.fixed_emtensor_rho
            emtensor.Si = self.fixed_emtensor_Si
            emtensor.Sij = self.fixed_emtensor_Sij
            emtensor.S = self.fixed_emtensor_S
            return emtensor

        # Normal mode: Compute from current primitives
        # Primitive variables
        prim = self._get_primitives(bssn_vars, r)
        rho0, vr, p, W, h = prim['rho0'], prim['vr'], prim['p'], prim['W'], prim['h']

        # Build ADM geometry from BSSN variables
        from source.bssn.tensoralgebra import get_bar_gamma_LL as _get_bar_gamma_LL
        phi = np.asarray(bssn_vars.phi, dtype=float)
        e4phi = np.exp(4.0 * phi)
        bar_gamma_LL = _get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        gamma_LL = e4phi[:, None, None] * bar_gamma_LL

        # Shift and lapse
        if hasattr(bssn_vars, 'lapse') and bssn_vars.lapse is not None:
            alpha = np.asarray(bssn_vars.lapse, dtype=float)
        else:
            alpha = np.ones(N)

        beta_U = np.zeros((N, SPACEDIM))
        shift_arr = np.asarray(bssn_vars.shift_U)
        if shift_arr.ndim >= 2:
            for i in range(min(SPACEDIM, shift_arr.shape[1])):
                beta_U[:, i] = shift_arr[:, i]

        geom = ADMGeometry(alpha=alpha, beta_U=beta_U, gamma_LL=gamma_LL)

        # 3-velocity vector (spherical symmetry -> only radial component non-zero)
        v_U = np.zeros((N, SPACEDIM))
        v_U[:, 0] = vr

        # Compute stress-energy tensor and ADM projections
        st = StressEnergyTensor(geometry=geom, rho0=rho0, v_U=v_U,
                                pressure=p, W=W, h=h)
        em = st.project_to_ADM()

        # Map to BSSN EMTensor object for compatibility
        emtensor.rho = em.rho
        emtensor.Si = em.S_D
        emtensor.Sij = em.S_DD
        emtensor.S = em.S

        return emtensor

    def get_matter_rhs(self, r, bssn_vars, bssn_d1, background):
        """
        Compute matter RHS using Valencia reference-metric formulation (engrenage interface).

        Returns array [dD̃/dt, dS̃_r/dt, dτ̃/dt] for densitized state vector evolution.
        """
        assert self.matter_vars_set, 'Matter vars not set'
        self.background = background

        # Convert to primitive variables (des-densitifies internally)
        primitives = self._get_primitives(bssn_vars, r)

        # Prepare 3D inputs for Valencia (spherical symmetry: only radial components non-zero)
        N = len(r)
        v_U = np.zeros((N, SPACEDIM))
        v_U[:, 0] = primitives['vr']  # Radial velocity
        # Angular velocities remain zero for spherical symmetry

        S_tildeD = np.zeros((N, SPACEDIM))
        S_tildeD[:, 0] = self.Sr_tilde  # Densitized radial momentum
        # Angular momenta remain zero for spherical symmetry

        # Valencia evolution equations (now fully 3D)
        # NOTE: compute_rhs expects and returns densitized variables
        dDdt, dSdt, dtaudt = self.valencia.compute_rhs(
            self.D_tilde, S_tildeD, self.tau_tilde,
            primitives['rho0'], v_U, primitives['p'],
            primitives['W'], primitives['h'],
            r, bssn_vars, bssn_d1, background,
            self.spacetime_mode, self.eos, self.grid,
            self.reconstructor, self.riemann_solver
        )

        # Extract radial component for state vector (maintaining 1D storage in PerfectFluid)
        # Returns densitized RHS: dŨ/dt
        return np.array([dDdt, dSdt[:, 0], dtaudt])

    def _get_primitives(self, bssn_vars, r, grid=None):
        """Convert conservative to primitive variables using cons2prim module."""

        # Build metric for cons2prim
        # Use geometry from valencia (eliminate duplication)
        self.valencia._extract_geometry(r, bssn_vars, self.spacetime_mode, self.background, grid)
        # Extract metric components needed for cons2prim
        alpha = self.valencia.alpha
        gamma_rr = self.valencia.gamma_LL[:, 0, 0]

        # Compute densitization factor
        phi = np.asarray(bssn_vars.phi, dtype=float)
        e6phi = np.exp(6.0 * phi)

        # Pass densitized conservatives directly to cons2prim (Opción B)
        # cons2prim will de-densitify internally, solve, and re-densitify
        U_tilde = (self.D_tilde, self.Sr_tilde, self.tau_tilde)

        # Use pressure cache if available
        p_guess = self.pressure_cache

        # Call cons2prim solver with densitized variables (simplified interface)
        # Returns primitives + updated densitized conservatives (post-floor)
        result = self.cons2prim_solver.convert(
            U=U_tilde,
            gamma_rr=gamma_rr,
            alpha=alpha,
            p_guess=p_guess,
            apply_conservative_floors=True,  # Apply tau and S_i floors
            e6phi=e6phi  # Enable densitized mode (Opción B)
        )

        # Update pressure cache for next timestep
        self.pressure_cache = result['p'].copy()

        # Update densitized conservatives with post-floor values (Opción B)
        # This ensures floors are consistently applied
        if 'D_tilde' in result:
            self.D_tilde = result['D_tilde'].copy()
            self.Sr_tilde = result['Sr_tilde'].copy()
            self.tau_tilde = result['tau_tilde'].copy()

        # Return primitives only (densitized conservatives already updated)
        return result
