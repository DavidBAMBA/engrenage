import numpy as np

from core.grid import Grid, i_x1
from bssn.bssnstatevariables import NUM_BSSN_VARS
from bssn.bssnvars import BSSNVars
from bssn.tensoralgebra import SPACEDIM, get_bar_gamma_LL
from backgrounds.sphericalbackground import i_r

from .valencia_reference_metric import ValenciaReferenceMetric
from .stress_energy_tensor import compute_stress_energy_tensor
from .cons2prim import cons_to_prim  # ← SOLO esta función
from .eos import IdealGasEOS

class RelativisticFluid:
    """
    Relativistic fluid implementation - ULTRA SIMPLIFIED.
    Uses ONLY the centralized cons2prim module functions.
    """

    def __init__(self, eos=None, spacetime_mode="dynamic", 
                 atmosphere_rho=1e-13, reconstructor=None, riemann_solver=None):
        
        # engrenage BaseMatter interface requirements
        self.NUM_MATTER_VARS = 3
        self.VARIABLE_NAMES = np.array(["D", "S_r", "tau"])
        self.PARITY = np.array([1, -1, 1])  
        self.ASYMP_POWER = np.array([0, 0, 0])  
        self.ASYMP_OFFSET = np.array([0, 0, 0])

        # Variable indices
        self.idx_D = NUM_BSSN_VARS
        self.idx_Sr = NUM_BSSN_VARS + 1
        self.idx_tau = NUM_BSSN_VARS + 2
        self.indices = np.array([self.idx_D, self.idx_Sr, self.idx_tau])

        # Physics
        self.eos = eos if eos is not None else IdealGasEOS()
        self.spacetime_mode = spacetime_mode
        self.atmosphere_rho = atmosphere_rho

        # Numerical methods
        self.valencia = ValenciaReferenceMetric()
        self.reconstructor = reconstructor
        self.riemann_solver = riemann_solver

        # Simple cons2prim config
        self.cons2prim_params = {
            "rho_floor": atmosphere_rho,
            "p_floor": 1e-15,
            "v_max": 0.999999,
            "tol": 1e-12,
            "max_iter": 100
        }

        # State - solo lo mínimo necesario
        self.matter_vars_set = False
        self.grid = None
        self.background = None
        
        # Conservative variables
        self.D = None
        self.Sr = None  
        self.tau = None

    def set_matter_vars(self, state_vector, bssn_vars: BSSNVars, grid: Grid):
        """Extract matter variables from state vector."""
        
        self.D = state_vector[self.idx_D].copy()
        self.Sr = state_vector[self.idx_Sr].copy()
        self.tau = state_vector[self.idx_tau].copy()
        self.grid = grid
        self.matter_vars_set = True

    def get_emtensor(self, r, bssn_vars, background):
        """Compute stress-energy tensor for BSSN source terms."""
        
        assert self.matter_vars_set, 'Matter vars not set'
        self.background = background
        
        # Convert to primitives DIRECTLY using cons2prim module
        primitives = self._get_primitives(bssn_vars, r)
        
        return compute_stress_energy_tensor(
            primitives['rho0'], primitives['vr'], primitives['p'], 
            primitives['eps'], primitives['W'], primitives['h'],
            r, bssn_vars, background, self.spacetime_mode
        )

    def get_matter_rhs(self, r, bssn_vars, bssn_d1, background):
        """Compute RHS using Valencia reference-metric formulation."""
        
        assert self.matter_vars_set, 'Matter vars not set'
        self.background = background
        
        # Convert to primitives DIRECTLY using cons2prim module
        primitives = self._get_primitives(bssn_vars, r)
        
        # Valencia RHS
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
        """
        Convert conservative to primitive variables.
        ONLY uses cons2prim module - no internal implementation.
        """
        
        # Build metric for cons2prim
        metric = self._build_metric(bssn_vars, r)
        
        # Call cons2prim module function DIRECTLY
        primitives = cons_to_prim(
            U=(self.D, self.Sr, self.tau),
            eos=self.eos,
            params=self.cons2prim_params,
            metric=metric
        )
        
        # Handle failures simply
        if not np.all(primitives['success']):
            failed = np.where(~primitives['success'])[0]
            print(f"Warning: cons2prim failed at {len(failed)} points, setting to atmosphere")
            
            # Set failed points to atmosphere
            primitives['rho0'][failed] = self.atmosphere_rho
            primitives['vr'][failed] = 0.0
            primitives['p'][failed] = 1e-15
            primitives['eps'][failed] = self.eos.eps_from_rho_p(self.atmosphere_rho, 1e-15)
            primitives['W'][failed] = 1.0
            primitives['h'][failed] = 1.0 + primitives['eps'][failed] + primitives['p'][failed] / primitives['rho0'][failed]
        
        return primitives

    def _build_metric(self, bssn_vars, r):
        """Build metric dictionary for cons2prim."""
        
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
        """Return basic diagnostics."""
        
        if not self.matter_vars_set:
            return {'status': 'vars_not_set'}
        
        return {
            'max_density': np.max(self.D),
            'min_density': np.min(self.D),
            'total_mass': np.sum(self.D),
            'spacetime_mode': self.spacetime_mode
        }