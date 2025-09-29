import numpy as np

from source.core.grid import Grid, i_x1, i_x2, i_x3
from source.bssn.bssnstatevariables import NUM_BSSN_VARS
from source.bssn.bssnvars import BSSNVars
from source.bssn.tensoralgebra import SPACEDIM, EMTensor
from source.backgrounds.sphericalbackground import i_r

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
        assert self.matter_vars_set, 'Matter vars not set'
        N = len(r)
        emtensor = EMTensor(N)

        # Primitivas
        prim = self._get_primitives(bssn_vars, r)
        rho0, vr, p, W, h = prim['rho0'], prim['vr'], prim['p'], prim['W'], prim['h']

        # Geometría coherente con BSSN
        from source.bssn.tensoralgebra import get_bar_gamma_LL
        phi = np.asarray(bssn_vars.phi, dtype=float)
        e4phi = np.exp(4.0*phi)
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)   # \bar γ_ij
        gamma_LL = e4phi[:,None,None] * bar_gamma_LL                     # γ_ij
        gamma_UU = np.linalg.inv(gamma_LL)                               # γ^{ij}

        # Energía: ρ = ρ0 h W^2 - p
        emtensor.rho = rho0*h*W*W - p

        # Velocidades espacial abajo: v_i = γ_ij v^j (v^θ=v^φ=0)
        vU = np.zeros((N, 3))
        vU[:,0] = vr
        vD = np.einsum('xij,xj->xi', gamma_LL, vU)

        # Momentum abajo y tensores de esfuerzo abajo
        emtensor.Si  = rho0*h*W*W * vD                             # S_i
        emtensor.Sij = rho0*h*W*W * np.einsum('xi,xj->xij', vD, vD) + p[:,None,None]*gamma_LL  # S_ij

        # Traza S = γ^{ij} S_{ij}
        emtensor.S   = np.einsum('xij,xij->x', gamma_UU, emtensor.Sij)
        return emtensor

    def get_matter_rhs(self, r, bssn_vars, bssn_d1, background):
        """
        Compute matter RHS using Valencia reference-metric formulation (engrenage interface).

        Returns array [dD/dt, dS_r/dt, dτ/dt] for state vector evolution.
        """
        import time
        start_matter = time.time()

        assert self.matter_vars_set, 'Matter vars not set'
        self.background = background

        # Convert to primitive variables
        start_cons2prim = time.time()
        primitives = self._get_primitives(bssn_vars, r)
        end_cons2prim = time.time()

        # Valencia evolution equations
        start_valencia = time.time()
        dDdt, dSrdt, dtaudt = self.valencia.compute_rhs(
            self.D, self.Sr, self.tau,
            primitives['rho0'], primitives['vr'], primitives['p'],
            primitives['W'], primitives['h'],
            r, bssn_vars, bssn_d1, background,
            self.spacetime_mode, self.eos, self.grid,
            self.reconstructor, self.riemann_solver
        )
        end_valencia = time.time()

        end_matter = time.time()
        # print(f"  MATTER BREAKDOWN: cons2prim={end_cons2prim-start_cons2prim:.4f}s, valencia={end_valencia-start_valencia:.4f}s, total={end_matter-start_matter:.4f}s")

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

