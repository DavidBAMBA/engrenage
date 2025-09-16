import numpy as np

from core.grid import Grid, i_x1, i_x2, i_x3
from bssn.bssnstatevariables import NUM_BSSN_VARS
from bssn.bssnvars import BSSNVars
from bssn.tensoralgebra import SPACEDIM, EMTensor
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

    ✨ ARCHITECTURE (Post-refactoring):
    - Eliminates geometry extraction duplication
    - Delegates all Valencia evolution to valencia_reference_metric.py
    - Focuses solely on EMTensor computation and engrenage interface
    - Uses valencia._extract_geometry() instead of duplicating logic
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
            "tol": 1e-10,
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
        # bssn_vars not needed here but kept for interface consistency
        _ = bssn_vars  # suppress warning

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

        # Use geometry extraction from valencia (eliminate duplication)
        g = self.valencia._extract_geometry(r, bssn_vars, self.spacetime_mode, background)
        alpha, beta_r = g['alpha'], g['beta_r']

        # Construct full spatial metric tensors for EMTensor computation
        if self.spacetime_mode == "fixed_minkowski":
            gamma_rr = np.ones(N)
        else:
            gamma_rr = g['gamma_rr']

        # Build 3D metric tensors (needed for EMTensor computation)
        gamma_LL, gamma_UU = self._build_3d_metrics(r, gamma_rr)

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
        # Use geometry from valencia (eliminate duplication)
        g = self.valencia._extract_geometry(r, bssn_vars, self.spacetime_mode, self.background)
        metric = {
            "alpha": g['alpha'],
            "beta_r": g['beta_r'],
            "gamma_rr": g['gamma_rr'] if self.spacetime_mode != "fixed_minkowski" else np.ones(len(r))
        }

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

    def _build_3d_metrics(self, r, gamma_rr):
        """Build 3D metric tensors from radial component (eliminates geometry duplication)."""
        N = len(r)

        # Lower index metric γ_ij
        gamma_LL = np.zeros([N, SPACEDIM, SPACEDIM])
        gamma_LL[:, 0, 0] = gamma_rr       # γ_rr from valencia
        gamma_LL[:, 1, 1] = gamma_rr * r**2   # γ_θθ = γ_rr * r²
        gamma_LL[:, 2, 2] = gamma_rr * r**2   # γ_φφ = γ_rr * r²

        # Upper index metric γ^ij
        gamma_UU = np.zeros([N, SPACEDIM, SPACEDIM])
        gamma_UU[:, 0, 0] = 1.0 / gamma_rr    # γ^rr = 1/γ_rr
        gamma_UU[:, 1, 1] = 1.0 / (gamma_rr * r**2)  # γ^θθ = 1/(γ_rr r²)
        gamma_UU[:, 2, 2] = 1.0 / (gamma_rr * r**2)  # γ^φφ = 1/(γ_rr r²)

        return gamma_LL, gamma_UU


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