import numpy as np

from core.grid import Grid, i_x1
from bssn.bssnstatevariables import NUM_BSSN_VARS
from bssn.bssnvars import BSSNVars
from bssn.tensoralgebra import SPACEDIM, get_bar_gamma_LL, get_bar_gamma_UU
from backgrounds.sphericalbackground import i_r, i_t, i_p

from .valencia_reference_metric import ValenciaReferenceMetric
from .stress_energy_tensor import compute_stress_energy_tensor
from .cons2prim import cons2prim
from .eos import IdealGasEOS

class RelativisticFluid:
    """
    Relativistic fluid implementation using Valencia reference-metric formulation.
    
    Implements BaseMatter interface for engrenage evolution system.
    Uses conservative variables {D, S_r, τ} with robust cons2prim conversion.
    
    Based on:
    - Montero et al. (2013): Reference-metric Valencia formulation
    - Banyuls et al. (1997): HRSC methods for GRHD
    """

    def __init__(self, eos=None, spacetime_mode="fixed_minkowski", 
                 atmosphere_rho=1e-13, reconstructor=None, riemann_solver=None):
        
        # engrenage BaseMatter interface requirements
        self.NUM_MATTER_VARS = 3
        self.VARIABLE_NAMES = np.array(["D", "S_r", "tau"])
        self.PARITY = np.array([1, -1, 1])  # D,τ even; S_r odd under r→-r
        self.ASYMP_POWER = np.array([0, 0, 0])  # Falloff behavior at large r
        self.ASYMP_OFFSET = np.array([0, 0, 0])

        # Variable indices in state vector
        self.idx_D = NUM_BSSN_VARS
        self.idx_Sr = NUM_BSSN_VARS + 1
        self.idx_tau = NUM_BSSN_VARS + 2
        self.indices = np.array([self.idx_D, self.idx_Sr, self.idx_tau])

        # Physics configuration
        self.eos = eos if eos is not None else IdealGasEOS()
        self.spacetime_mode = spacetime_mode
        self.atmosphere_rho = atmosphere_rho

        # Numerical methods
        self.valencia = ValenciaReferenceMetric()
        self.reconstructor = reconstructor
        self.riemann_solver = riemann_solver

        # Parameters for cons2prim
        self.cons2prim_params = {
            "rho_floor": atmosphere_rho,
            "p_floor": 1e-15,
            "v_max": 0.999999,
            "W_max": 100.0,
            "tol": 1e-12,
            "max_iter": 100
        }

        # State tracking
        self.matter_vars_set = False
        self.primitives_computed = False
        
        # Store references for use in other methods
        self.grid = None
        self.background = None

        # Conservative variables (set by set_matter_vars)
        self.D = None
        self.Sr = None  
        self.tau = None
        
        # Primitive variables (computed from conservative)
        self.rho0 = None
        self.vr = None
        self.pressure = None
        self.eps = None
        self.W = None
        self.h = None
        
        # Success flag for cons2prim
        self.cons2prim_success = None

    def set_matter_vars(self, state_vector, bssn_vars: BSSNVars, grid: Grid):
        """
        Extract matter variables from state vector.
        Required by engrenage BaseMatter interface.
        """
        
        # Extract conservative variables from state vector
        self.D = state_vector[self.idx_D].copy()
        self.Sr = state_vector[self.idx_Sr].copy()
        self.tau = state_vector[self.idx_tau].copy()

        # Store grid reference
        self.grid = grid
        
        # Reset primitive computation flag
        self.primitives_computed = False
        self.matter_vars_set = True

    def get_emtensor(self, r, bssn_vars, background):
        """
        Compute stress-energy tensor for BSSN source terms.
        Required by engrenage BaseMatter interface.
        """
        
        assert self.matter_vars_set, 'Matter vars not set'
        
        # Store background for later use
        self.background = background
        
        # Ensure primitives are computed with correct geometry
        self._ensure_primitives_computed(bssn_vars)
        
        return compute_stress_energy_tensor(
            self.rho0, self.vr, self.pressure, self.eps, self.W, self.h,
            r, bssn_vars, background, self.spacetime_mode
        )

    def get_matter_rhs(self, r, bssn_vars, bssn_d1, background):
        """
        Compute RHS for matter evolution using Valencia reference-metric formulation.
        Required by engrenage BaseMatter interface.
        """
        
        assert self.matter_vars_set, 'Matter vars not set'
        
        # Store background and ensure primitives computed
        self.background = background
        self._ensure_primitives_computed(bssn_vars)
        
        # Valencia reference-metric RHS computation
        dDdt, dSrdt, dtaudt = self.valencia.compute_rhs(
            self.D, self.Sr, self.tau,                           # Conservative variables
            self.rho0, self.vr, self.pressure, self.W, self.h,   # Primitive variables
            r, bssn_vars, bssn_d1, background,                   # Geometry
            self.spacetime_mode, self.eos, self.grid,            # Configuration
            self.reconstructor, self.riemann_solver              # Numerical methods
        )
        
        return np.array([dDdt, dSrdt, dtaudt])

    def _ensure_primitives_computed(self, bssn_vars):
        """Ensure primitive variables are computed with correct geometry."""
        
        if self.primitives_computed and self.background is not None:
            return
            
        self._conservative_to_primitive(bssn_vars)
        self.primitives_computed = True

    def _conservative_to_primitive(self, bssn_vars):
        """Convert conservative to primitive variables using centralized cons2prim."""
        
        N = self.grid.N
        r = self.grid.r
        
        # Build metric dictionary for cons2prim
        metric = self._build_metric_dict(bssn_vars, r, N)
        
        # Call centralized cons2prim function
        primitives = cons_to_prim(
            U=(self.D, self.Sr, self.tau),
            eos=self.eos,
            params=self.cons2prim_params,
            metric=metric
        )
        
        # Extract results
        self.rho0 = primitives['rho0']
        self.vr = primitives['vr']
        self.pressure = primitives['p']
        self.eps = primitives['eps']
        self.W = primitives['W']
        self.h = primitives['h']
        self.cons2prim_success = primitives['success']
        
        # Handle failed points
        failed_points = np.where(~self.cons2prim_success)[0]
        if len(failed_points) > 0:
            print(f"Warning: cons2prim failed at {len(failed_points)} points")
            self._handle_failed_points(failed_points)

    def _build_metric_dict(self, bssn_vars, r, N):
        """Build metric dictionary for cons2prim based on spacetime mode."""
        
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
                # Fallback to Minkowski if background not yet available
                gamma_rr = np.ones(N)
                
            return {
                "alpha": alpha,
                "beta_r": beta_r,
                "gamma_rr": gamma_rr
            }

    def _handle_failed_points(self, indices):
        """Handle points where cons2prim failed by setting to atmosphere."""
        
        for i in indices:
            # Set atmosphere primitives
            self.rho0[i] = self.atmosphere_rho
            self.vr[i] = 0.0
            self.pressure[i] = max(1e-15, self.eos.pressure(self.atmosphere_rho, 1e-10))
            self.eps[i] = self.eos.eps_from_rho_p(self.atmosphere_rho, self.pressure[i])
            self.W[i] = 1.0
            self.h[i] = 1.0 + self.eps[i] + self.pressure[i] / self.rho0[i]
            
            # Update conservative variables to be consistent
            # Use centralized prim_to_cons from cons2prim.py
            from .cons2prim import prim_to_cons
            gamma_rr = 1.0  # Safe default for failed points
            cons = prim_to_cons(self.rho0[i], self.vr[i], self.pressure[i], gamma_rr, self.eos)
            self.D[i], self.Sr[i], self.tau[i] = cons

    def apply_floors_and_ceilings(self, max_lorentz=100.0):
        """Apply physical floors and ceilings to prevent unphysical values."""
        
        if not self.primitives_computed:
            return
            
        modified_points = []
        
        # Density floor
        rho_floor_mask = self.rho0 < self.atmosphere_rho
        if rho_floor_mask.any():
            modified_points.extend(np.where(rho_floor_mask)[0])
        
        # Lorentz factor ceiling  
        W_ceiling_mask = self.W > max_lorentz
        if W_ceiling_mask.any():
            self.W[W_ceiling_mask] = max_lorentz
            # Recompute velocity from W
            v2_max = 1.0 - 1.0/max_lorentz**2
            self.vr[W_ceiling_mask] = np.sign(self.vr[W_ceiling_mask]) * np.sqrt(v2_max)
            modified_points.extend(np.where(W_ceiling_mask)[0])
        
        # Pressure floor
        p_floor = 1e-15
        p_floor_mask = self.pressure < p_floor
        if p_floor_mask.any():
            self.pressure[p_floor_mask] = p_floor
            self.eps[p_floor_mask] = self.eos.eps_from_rho_p(
                self.rho0[p_floor_mask], p_floor
            )
            modified_points.extend(np.where(p_floor_mask)[0])
        
        # Update conservative variables for all modified points
        if modified_points:
            from .cons2prim import prim_to_cons
            unique_points = np.unique(modified_points)
            
            for i in unique_points:
                if self.background is not None:
                    # Use proper metric
                    metric_dict = self._build_metric_dict(None, None, None)  # This needs bssn_vars
                    gamma_rr = 1.0  # Simplified for now
                else:
                    gamma_rr = 1.0
                    
                cons = prim_to_cons(self.rho0[i], self.vr[i], self.pressure[i], gamma_rr, self.eos)
                self.D[i], self.Sr[i], self.tau[i] = cons

    def get_diagnostics(self):
        """Return diagnostic quantities for monitoring evolution."""
        
        if not self.matter_vars_set:
            return {}
        
        diagnostics = {
            'max_density': np.max(self.D) if self.D is not None else 0,
            'min_density': np.min(self.D) if self.D is not None else 0,
            'total_mass': np.sum(self.D) if self.D is not None else 0,
            'primitives_computed': self.primitives_computed
        }
        
        if self.primitives_computed:
            diagnostics.update({
                'max_rest_density': np.max(self.rho0),
                'min_rest_density': np.min(self.rho0),
                'max_pressure': np.max(self.pressure),
                'min_pressure': np.min(self.pressure),
                'max_lorentz': np.max(self.W),
                'max_velocity': np.max(np.abs(self.vr)),
                'atmosphere_points': np.sum(self.rho0 <= self.atmosphere_rho * 1.1),
                'cons2prim_failures': np.sum(~self.cons2prim_success) if self.cons2prim_success is not None else 0
            })
        
        return diagnostics