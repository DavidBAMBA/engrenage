import numpy as np

from source.core.grid import Grid, i_x1, i_x2, i_x3
from source.bssn.bssnstatevariables import NUM_BSSN_VARS
from source.bssn.bssnvars import BSSNVars
from source.bssn.tensoralgebra import SPACEDIM, EMTensor
from source.backgrounds.sphericalbackground import i_r

from .valencia_reference_metric import ValenciaReferenceMetric
from .eos import IdealGasEOS
from .atmosphere import AtmosphereParams

# Backend-aware import for Cons2PrimSolver
from .tests.advance.backend import BACKEND
if 'jax' in BACKEND:
    from .cons2prim_jax import Cons2PrimSolverJAX as Cons2PrimSolver
else:
    from .cons2prim import Cons2PrimSolver


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
            atmosphere: AtmosphereParams object (required for evolution)
            reconstructor: Reconstruction method (optional)
            riemann_solver: Riemann solver (optional)
        """

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

        self.atmosphere = atmosphere

        # Create cons2prim solver with centralized atmosphere
        self.cons2prim_solver = Cons2PrimSolver(
            self.eos,
            atmosphere=self.atmosphere,
            tol=1e-12,
            max_iter=200
        )

        # Numerical methods for Valencia evolution
        self.valencia = ValenciaReferenceMetric(atmosphere=self.atmosphere)
        self.reconstructor = reconstructor
        self.riemann_solver = riemann_solver

        # State variables (engrenage pattern)
        self.matter_vars_set = False
        self.grid = None
        self.background = None

        # Cache for pressure guesses in cons2prim
        self.pressure_cache = None

        # Conservative variables densitized
        self.D = None      
        self.Sr = None    
        self.tau = None    

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
        rho0, vr, p, eps, W, h, success = self._get_primitives(bssn_vars, r)

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
        pref = (rho0 * h * W * W)
        emtensor.Si  = pref[:, None] * vD  # S_i with proper broadcasting
        emtensor.Sij = pref[:, None, None] * np.einsum('xi,xj->xij', vD, vD) + p[:, None, None] * gamma_LL  # S_ij

        # Traza S = γ^{ij} S_{ij}
        emtensor.S   = np.einsum('xij,xij->x', gamma_UU, emtensor.Sij)
        return emtensor

    def get_matter_rhs(self, r, bssn_vars, bssn_d1, background):
        """
        Compute matter RHS using Valencia reference-metric formulation (engrenage interface).

        Returns array [dD/dt, dS_r/dt, dτ/dt] for state vector evolution.
        """
        assert self.matter_vars_set, 'Matter vars not set'
        self.background = background

        # Convert to primitive variables
        rho0, vr, p, eps, W, h, success = self._get_primitives(bssn_vars, r)

        # Valencia evolution equations
        dDdt, dSrdt, dtaudt = self.valencia.compute_rhs(
            self.D, self.Sr, self.tau,
            rho0, vr, p, W, h,
            r, bssn_vars, bssn_d1, background,
            self.spacetime_mode, self.eos, self.grid,
            self.reconstructor, self.riemann_solver
        )

        return np.array([dDdt, dSrdt, dtaudt])

    def _get_primitives(self, bssn_vars, r):
        """Convert conservative to primitive variables using cons2prim module.

        IMPORTANT: State vector stores DENSITIZED conservatives (D̃, S̃ʳ, τ̃) where
        D̃ = e^{6φ}D. This method de-densitizes them before calling cons2prim, which
        expects PHYSICAL conservatives.

        Returns:
            tuple: (rho0, vr, p, eps, W, h, success) - Primitive hydrodynamic variables
                   success is a boolean array indicating which points converged
        """

        # Build geometry from BSSN variables (creates self.valencia._geom)
        self.valencia._extract_geometry(r, bssn_vars, self.spacetime_mode, self.background, self.grid)

        # Get GeometryState from valencia (bundles alpha, beta_r, gamma_rr, e6phi)
        geom = self.valencia._geom

        # De-densitify conservative variables: U = Ũ / e^{6φ}
        D_phys = self.D / geom.e6phi
        Sr_phys = self.Sr / geom.e6phi
        tau_phys = self.tau / geom.e6phi

        # Use pressure cache if available
        p_guess = self.pressure_cache

        # Call cons2prim solver with PHYSICAL (non-densitized) conservatives
        # GeometryState passed so cons2prim returns properly densitized conservatives
        result = self.cons2prim_solver.convert(
            D_phys, Sr_phys, tau_phys, geom,
            p_guess=p_guess,
            apply_conservative_floors=True
        )
        rho0, vr, p, eps, W, h, success, D_new, Sr_new, tau_new = result

        # Update pressure cache for next timestep
        self.pressure_cache = p.copy()

        # Update state vector with conservatives from cons2prim
        # (already densitized, and consistent with primitives for atmosphere points)
        self.D[:] = D_new
        self.Sr[:] = Sr_new
        self.tau[:] = tau_new

        return rho0, vr, p, eps, W, h, success
