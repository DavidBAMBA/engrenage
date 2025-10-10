import numpy as np

from source.core.grid import Grid, i_x1, i_x2, i_x3
from source.bssn.bssnstatevariables import NUM_BSSN_VARS
from source.bssn.bssnvars import BSSNVars
from source.bssn.tensoralgebra import SPACEDIM, EMTensor
from source.backgrounds.sphericalbackground import i_r

from .valencia_reference_metric import ValenciaReferenceMetric
from .cons2prim import Cons2PrimSolver
from .eos import IdealGasEOS
from .atmosphere import AtmosphereParams, create_default_atmosphere


class PerfectFluid:
    """
    Relativistic perfect fluid implementation for engrenage.

    Follows the engrenage matter interface pattern used by ScalarMatter and NoMatter.
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

        # Atmosphere configuration (centralized)
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
            max_iter=200
        )

        # Numerical methods for Valencia evolution
        self.valencia = ValenciaReferenceMetric(
            boundary_mode="parity",
            atmosphere=self.atmosphere  # Pass centralized atmosphere
        )
        self.reconstructor = reconstructor
        self.riemann_solver = riemann_solver

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
        pref = (rho0 * h * W * W)
        emtensor.Si  = pref[:, None] * vD  # S_i with proper broadcasting
        emtensor.Sij = pref[:, None, None] * np.einsum('xi,xj->xij', vD, vD) \
                       + p[:, None, None] * gamma_LL  # S_ij

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
        # Map general 3D geometry to 1D radial metric expected by cons2prim
        alpha = g.get('alpha')
        # Prefer provided radial components if present; otherwise derive from full tensors
        if 'beta_r' in g:
            beta_r = g['beta_r']
        else:
            betaU = g.get('beta_U', None)
            beta_r = betaU[:, 0] if betaU is not None else np.zeros_like(alpha)

        if 'gamma_rr' in g:
            gamma_rr = g['gamma_rr']
        else:
            gammaLL = g.get('gamma_LL', None)
            if gammaLL is not None:
                gamma_rr = gammaLL[:, 0, 0]
            else:
                gamma_rr = np.ones_like(alpha)

        metric = {
            "alpha": alpha,
            "beta_r": beta_r,
            "gamma_rr": gamma_rr,
        }

        # Use pressure cache if available
        p_guess = self.pressure_cache

        # Call cons2prim solver (now with intelligent floor application)
        U = {"D": self.D, "Sr": self.Sr, "tau": self.tau}
        primitives = self.cons2prim_solver.convert(
            U=U,
            metric=metric,
            p_guess=p_guess,
            apply_conservative_floors=True  # Apply tau and S_i floors
        )

        # Update pressure cache for next timestep
        self.pressure_cache = primitives['p'].copy()

        # Note: Failed points are already handled by cons2prim_solver with atmosphere fallback
        # No need for additional manual handling here

        return primitives
