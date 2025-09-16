import numpy as np

from core.grid import Grid, i_x1, i_x2, i_x3
from bssn.bssnstatevariables import NUM_BSSN_VARS
from bssn.bssnvars import BSSNVars
from bssn.tensoralgebra import SPACEDIM, EMTensor, get_bar_gamma_LL, get_bar_gamma_UU
from backgrounds.sphericalbackground import i_r

from .valencia_reference_metric import ValenciaReferenceMetric
from .cons2prim import cons_to_prim
from .eos import IdealGasEOS


class PerfectFluid:
    """
    Relativistic perfect fluid implementation for engrenage.

    Follows the engrenage matter interface pattern used by ScalarMatter and NoMatter.
    Implements Valencia formulation for conservative evolution and provides
    stress-energy tensor for BSSN coupling.
    """

    def __init__(self, eos=None, spacetime_mode="dynamic",
                 atmosphere_rho=1e-13, reconstructor=None, riemann_solver=None):

        # engrenage BaseMatter interface requirements
        self.NUM_MATTER_VARS = 3
        self.VARIABLE_NAMES = np.array(["D", "S_r", "tau"])
        self.PARITY = np.array([1, -1, 1])  # D: even, S_r: odd, tau: even
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
        self.atmosphere_rho = atmosphere_rho

        # Numerical methods for Valencia evolution
        self.valencia = ValenciaReferenceMetric()
        self.reconstructor = reconstructor
        self.riemann_solver = riemann_solver

        # cons2prim configuration
        self.cons2prim_params = {
            "rho_floor": atmosphere_rho,
            "p_floor": 1e-15,
            "v_max": 0.999999,
            "tol": 1e-12,
            "max_iter": 100
        }

        # State variables (engrenage pattern)
        self.matter_vars_set = False
        self.grid = None
        self.background = None

        # Cache for pressure guesses (improves cons2prim performance)
        self.pressure_cache = None

        # Conservative variables
        self.D = None      # Rest mass density
        self.Sr = None     # Radial momentum density
        self.tau = None    # Energy density minus rest mass

    def set_matter_vars(self, state_vector, bssn_vars: BSSNVars, grid: Grid):
        """Extract matter variables from state vector (engrenage interface)."""

        self.D = state_vector[self.idx_D].copy()
        self.Sr = state_vector[self.idx_Sr].copy()
        self.tau = state_vector[self.idx_tau].copy()
        self.grid = grid
        self.matter_vars_set = True

    def get_emtensor(self, r, bssn_vars, background):
        """
        Compute stress-energy tensor for BSSN source terms (engrenage interface).

        Returns EMTensor with components:
        - rho: Energy density T^tt
        - Si: Momentum density T^ti
        - Sij: Stress tensor T^ij
        - S: Trace S = γ_ij T^ij
        """

        assert self.matter_vars_set, 'Matter vars not set'
        self.background = background

        N = len(r)
        emtensor = EMTensor(N)

        # Convert to primitive variables
        primitives = self._get_primitives(bssn_vars, r)
        rho0, vr, p = primitives['rho0'], primitives['vr'], primitives['p']
        W, h = primitives['W'], primitives['h']

        # Extract spacetime geometry
        alpha, beta_r, gamma_LL, gamma_UU = self._extract_geometry(r, bssn_vars, background)

        # Four-velocity components
        ut = W / alpha                    # u^t = W/α
        ur = W * vr                       # u^r = W v^r

        # Energy density: ρ = T^μν n_μ n_ν = ρ₀ h W² - p
        emtensor.rho = rho0 * h * W**2 - p
        emtensor.rho = np.maximum(emtensor.rho, 0.0)  # Ensure positivity

        # Momentum density: S_i = T^t_i
        emtensor.Si = np.zeros([N, SPACEDIM])
        # S_r = ρ₀ h u^t u^r + p g^tr
        gamma_UU_rr = gamma_UU[:, 0, 0]
        g_UU_tr = gamma_UU_rr * beta_r / alpha**2
        emtensor.Si[:, i_x1] = rho0 * h * ut * ur + p * g_UU_tr
        # S_θ = S_φ = 0 (spherical symmetry, no angular momentum)

        # Stress tensor: T^ij = ρ₀ h u^i u^j + p γ^ij
        emtensor.Sij = np.zeros([N, SPACEDIM, SPACEDIM])
        emtensor.Sij[:, i_x1, i_x1] = rho0 * h * ur**2 + p * gamma_UU[:, 0, 0]  # T^rr
        emtensor.Sij[:, i_x2, i_x2] = p * gamma_UU[:, 1, 1]                     # T^θθ
        emtensor.Sij[:, i_x3, i_x3] = p * gamma_UU[:, 2, 2]                     # T^φφ

        # Trace: S = γ_ij T^ij
        emtensor.S = (gamma_LL[:, 0, 0] * emtensor.Sij[:, 0, 0] +
                      gamma_LL[:, 1, 1] * emtensor.Sij[:, 1, 1] +
                      gamma_LL[:, 2, 2] * emtensor.Sij[:, 2, 2])

        return emtensor

    def get_matter_rhs(self, r, bssn_vars, bssn_d1, background):
        """
        Compute matter RHS using Valencia reference-metric formulation (engrenage interface).

        Returns array [dD/dt, dS_r/dt, dτ/dt] for state vector evolution.
        """

        assert self.matter_vars_set, 'Matter vars not set'
        self.background = background

        # Convert to primitive variables
        primitives = self._get_primitives(bssn_vars, r)

        # Valencia evolution equations
        dDdt, dSrdt, dtaudt = self.valencia.compute_rhs(
            self.D, self.Sr, self.tau,
            primitives['rho0'], primitives['vr'], primitives['p'],
            primitives['W'], primitives['h'],
            r, bssn_vars, bssn_d1, background,
            self.spacetime_mode, self.eos, self.grid,
            self.reconstructor, self.riemann_solver
        )

        return np.array([dDdt, dSrdt, dtaudt])

    def _get_primitives(self, bssn_vars, r):
        """Convert conservative to primitive variables using cons2prim module."""

        # Build metric for cons2prim
        metric = self._build_metric(bssn_vars, r)

        # Use pressure cache if available
        p_guess = self.pressure_cache

        # Call cons2prim module with pressure guess
        primitives = cons_to_prim(
            U=(self.D, self.Sr, self.tau),
            eos=self.eos,
            params=self.cons2prim_params,
            metric=metric,
            p_guess=p_guess
        )

        # Update pressure cache for next timestep
        self.pressure_cache = primitives['p'].copy()

        # Handle conversion failures
        if not np.all(primitives['success']):
            failed = np.where(~primitives['success'])[0]
            # Suprimir warnings para test rápido
            # print(f"Warning: cons2prim failed at {len(failed)} points, setting to atmosphere")

            # Set failed points to atmosphere
            primitives['rho0'][failed] = self.atmosphere_rho
            primitives['vr'][failed] = 0.0
            primitives['p'][failed] = 1e-15
            primitives['eps'][failed] = self.eos.eps_from_rho_p(self.atmosphere_rho, 1e-15)
            primitives['W'][failed] = 1.0
            primitives['h'][failed] = 1.0 + primitives['eps'][failed] + primitives['p'][failed] / primitives['rho0'][failed]

        return primitives

    def _extract_geometry(self, r, bssn_vars, background):
        """Extract spacetime geometry from BSSN variables."""

        N = len(r)

        if self.spacetime_mode == "fixed_minkowski":
            # Static Minkowski metric in spherical coordinates
            alpha = np.ones(N)
            beta_r = np.zeros(N)

            # Spatial metric γ_ij and inverse γ^ij
            gamma_LL = np.zeros([N, SPACEDIM, SPACEDIM])
            gamma_LL[:, 0, 0] = 1.0        # γ_rr = 1
            gamma_LL[:, 1, 1] = r**2       # γ_θθ = r²
            gamma_LL[:, 2, 2] = r**2       # γ_φφ = r²

            gamma_UU = np.zeros([N, SPACEDIM, SPACEDIM])
            gamma_UU[:, 0, 0] = 1.0        # γ^rr = 1
            gamma_UU[:, 1, 1] = 1.0/r**2   # γ^θθ = 1/r²
            gamma_UU[:, 2, 2] = 1.0/r**2   # γ^φφ = 1/r²

        else:
            # Dynamic spacetime from BSSN variables
            alpha = bssn_vars.lapse
            beta_r = bssn_vars.shift_U[:, i_x1] if hasattr(bssn_vars, 'shift_U') else np.zeros(N)

            # Reconstruct spatial metric: γ_ij = e^{4φ} γ̄_ij
            e4phi = np.exp(4.0 * bssn_vars.phi)
            bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)

            gamma_LL = np.zeros([N, SPACEDIM, SPACEDIM])
            for i in range(SPACEDIM):
                for j in range(SPACEDIM):
                    gamma_LL[:, i, j] = e4phi * bar_gamma_LL[:, i, j]

            # Inverse spatial metric: γ^ij = e^{-4φ} γ̄^ij
            bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)
            em4phi = np.exp(-4.0 * bssn_vars.phi)

            gamma_UU = np.zeros([N, SPACEDIM, SPACEDIM])
            for i in range(SPACEDIM):
                for j in range(SPACEDIM):
                    gamma_UU[:, i, j] = em4phi * bar_gamma_UU[:, i, j]

        return alpha, beta_r, gamma_LL, gamma_UU

    def _build_metric(self, bssn_vars, r):
        """Build metric dictionary for cons2prim module."""

        N = len(r)

        if self.spacetime_mode == "fixed_minkowski":
            return {
                "alpha": np.ones(N),
                "beta_r": np.zeros(N),
                "gamma_rr": np.ones(N)
            }
        else:
            # Dynamic spacetime
            alpha = bssn_vars.lapse
            beta_r = bssn_vars.shift_U[:, i_x1] if hasattr(bssn_vars, 'shift_U') else np.zeros(N)

            if self.background is not None:
                e4phi = np.exp(4.0 * bssn_vars.phi)
                bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, self.background)
                gamma_rr = e4phi * bar_gamma_LL[:, i_r, i_r]
            else:
                gamma_rr = np.ones(N)

            return {
                "alpha": alpha,
                "beta_r": beta_r,
                "gamma_rr": gamma_rr
            }

    def get_diagnostics(self):
        """Return basic fluid diagnostics."""

        if not self.matter_vars_set:
            return {'status': 'vars_not_set'}

        return {
            'max_density': np.max(self.D),
            'min_density': np.min(self.D),
            'total_mass': np.sum(self.D),
            'spacetime_mode': self.spacetime_mode
        }