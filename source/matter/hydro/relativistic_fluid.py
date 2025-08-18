import numpy as np

from core.grid import Grid, i_x1
from bssn.bssnstatevariables import NUM_BSSN_VARS
from bssn.bssnvars import BSSNVars
from bssn.tensoralgebra import SPACEDIM, get_bar_gamma_LL, get_bar_gamma_UU
from backgrounds.sphericalbackground import i_r, i_t, i_p

from .valencia_reference_metric import ValenciaReferenceMetric
from .stress_energy_tensor import compute_stress_energy_tensor
from .con2prim import ConservativeToPrimitive
from .eos import IdealGasEOS

class RelativisticFluid:
    """
    Relativistic fluid implementation using Valencia reference-metric formulation.
    
    Implements BaseMatter interface for engrenage evolution system.
    Uses conservative variables {D, S_r, τ} with robust con2prim conversion.
    
    Based on:
    - Montero et al. (2013): Reference-metric Valencia formulation
    - Bany
    """

    def __init__(self, eos=None, spacetime_mode="fixed_minkowski", 
                 atmosphere_rho=1e-13, reconstructor=None, riemann_solver=None):
        
        # engrenage BaseMatter interface requirements
        self.NUM_MATTER_VARS = 3
        self.VARIABLE_NAMES = np.array(["D", "S_r", "tau"])
        self.PARITY = np.array([1, -1, 1]) # D,τ even; S_r odd under r→-r
        self.ASYMP_POWER = np.array([0, 0, 0])# Falloff behavior at large r
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
        self.con2prim = ConservativeToPrimitive(self.eos, atmosphere_rho)
        self.reconstructor = reconstructor
        self.riemann_solver = riemann_solver

        # State tracking
        self.matter_vars_set = False
        
        # Store background and Grid reference for use in other methods
        self.grid       = None
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
        
        # Derivatives for evolution
        self.d1_D = None
        self.d1_Sr = None
        self.d1_tau = None
        self.advec_D = None
        self.advec_Sr = None
        self.advec_tau = None


    def set_matter_vars(self, state_vector, bssn_vars: BSSNVars, grid: Grid):
        """
        Extract matter variables from state vector and compute primitives.
        Required by engrenage BaseMatter interface.
        """
        
        # Extract conservative variables from state vector
        self.D = state_vector[self.idx_D].copy()
        self.Sr = state_vector[self.idx_Sr].copy()
        self.tau = state_vector[self.idx_tau].copy()

        # Store grid reference
        self.grid = grid

        # Compute derivatives needed for RHS
        self._compute_derivatives(state_vector, bssn_vars, grid)
        
        # For fixed Minkowski, we can do con2prim now
        # For dynamic spacetime, we'll defer until background is available
        if self.spacetime_mode == "fixed_minkowski":
            self._conservative_to_primitive(bssn_vars, grid)

        self.matter_vars_set = True


    def get_emtensor(self, r, bssn_vars, background):
        """
        Compute stress-energy tensor for BSSN source terms.
        Required by engrenage BaseMatter interface.
        """
        
        assert self.matter_vars_set, 'Matter vars not set'

        self.background = background


        # For dynamic spacetime, do con2prim now that we have background
        if self.spacetime_mode != "fixed_minkowski":
            self._conservative_to_primitive(bssn_vars, self.grid)
        
        return compute_stress_energy_tensor(
            self.rho0, self.vr, self.pressure, self.eps, self.W, self.h,
            r, bssn_vars, background, self.spacetime_mode
        )


    def get_matter_rhs(self, r, bssn_vars, bssn_d1, background):
        """
        Compute RHS for matter evolution using Valencia reference-metric formulation.
        Required by engrenage BaseMatter interface.
        
        Now passes the reconstructor and riemann_solver to Valencia.
        """
        
        assert self.matter_vars_set, 'Matter vars not set'
        
        # Store background for later use
        self.background = background
        
        # Valencia reference-metric RHS computation with HLLE solver
        dDdt, dSrdt, dtaudt = self.valencia.compute_rhs(
            self.D, self.Sr, self.tau,                                # Conservative variables
            self.rho0, self.vr, self.pressure, self.W, self.h,        # Primitive variables
            self.d1_D, self.d1_Sr, self.d1_tau,                       # Derivatives
            r, bssn_vars, bssn_d1, background,                        # Geometry
            self.spacetime_mode, self.eos,                            # Configuration
            self.reconstructor, self.riemann_solver                   # Pass numerical methods
        )
        
        # Add advection terms if shift is present
        """ if hasattr(bssn_vars, 'shift_U') and self.advec_D is not None:
            dDdt += np.einsum('xi,xi->x', 
                            background.inverse_scaling_vector * bssn_vars.shift_U, 
                            self.advec_D)
            dSrdt += np.einsum('xi,xi->x', 
                            background.inverse_scaling_vector * bssn_vars.shift_U, 
                            self.advec_Sr)
            dtaudt += np.einsum('xi,xi->x', 
                            background.inverse_scaling_vector * bssn_vars.shift_U, 
                            self.advec_tau) """
        
        return np.array([dDdt, dSrdt, dtaudt])
    

    def _compute_derivatives(self, state_vector, bssn_vars, grid):
        """Compute spatial derivatives needed for flux calculations."""
        
        N = grid.N
        
        # Initialize derivative arrays
        self.d1_D = np.zeros([N, SPACEDIM])
        self.d1_Sr = np.zeros([N, SPACEDIM])
        self.d1_tau = np.zeros([N, SPACEDIM])
        
        # First derivatives - Grid.get_first_derivative returns full array
        d1_state = grid.get_first_derivative(state_vector, self.indices)
        self.d1_D[:,i_x1] = d1_state[self.idx_D]
        self.d1_Sr[:,i_x1] = d1_state[self.idx_Sr]
        self.d1_tau[:,i_x1] = d1_state[self.idx_tau]
        
        # Advection derivatives for shift terms
        if hasattr(bssn_vars, 'shift_U') and np.any(bssn_vars.shift_U):
            # Create direction array for Grid.get_advection
            direction = bssn_vars.shift_U[:,i_x1] >= 0
            
            advec_state = grid.get_advection(state_vector, direction, self.indices)
            
            self.advec_D = np.zeros([N, SPACEDIM])
            self.advec_Sr = np.zeros([N, SPACEDIM])
            self.advec_tau = np.zeros([N, SPACEDIM])
            
            self.advec_D[:,i_x1] = advec_state[self.idx_D]
            self.advec_Sr[:,i_x1] = advec_state[self.idx_Sr]
            self.advec_tau[:,i_x1] = advec_state[self.idx_tau]
        else:
            # No shift, so no advection
            self.advec_D = np.zeros([N, SPACEDIM])
            self.advec_Sr = np.zeros([N, SPACEDIM])
            self.advec_tau = np.zeros([N, SPACEDIM])


    def _conservative_to_primitive(self, bssn_vars, grid):
        """Convert conservative to primitive variables using robust con2prim."""
        
        N = grid.N
        r = grid.r
        
        # Initialize primitive arrays
        self.rho0 = np.zeros(N)
        self.vr = np.zeros(N)
        self.pressure = np.zeros(N)
        self.eps = np.zeros(N)
        self.W = np.zeros(N)
        self.h = np.zeros(N)
        
        # Extract geometry
        if self.spacetime_mode == "fixed_minkowski":
            alpha = np.ones(N)
            beta_r = np.zeros(N)
            gamma_rr = np.ones(N)
        else:
            alpha = bssn_vars.lapse
            beta_r = bssn_vars.shift_U[:,i_x1] if hasattr(bssn_vars, 'shift_U') else np.zeros(N)
            
            # For dynamic spacetime, we need the background (will be set when get_emtensor is called)
            # For now, use placeholder - in practice, con2prim is called after get_emtensor
            if self.background is not None:
                e4phi = np.exp(4.0 * bssn_vars.phi)
                bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, self.background)
                gamma_rr = e4phi * bar_gamma_LL[:, i_r, i_r]
            else:
                # Fallback to Minkowski if background not yet set
                gamma_rr = np.ones(N)
        
        # Perform con2prim conversion
        success = self.con2prim.convert_all_points(
            self.D, self.Sr, self.tau,
            self.rho0, self.vr, self.pressure, self.eps, self.W, self.h,
            alpha, beta_r, gamma_rr
        )
        
        # Handle failed points
        failed_points = np.where(~success)[0]
        if len(failed_points) > 0:
            print(f"Warning: con2prim failed at {len(failed_points)} points")
            self._set_atmosphere_points(failed_points)


    def _set_atmosphere_points(self, indices):
        """Set atmosphere values at specified grid points."""
        
        for i in indices:
            self.rho0[i] = self.atmosphere_rho
            self.vr[i] = 0.0
            self.pressure[i] = self.eos.pressure(self.atmosphere_rho, 1e-10)
            self.eps[i] = 1e-10
            self.W[i] = 1.0
            self.h[i] = 1.0 + self.eps[i] + self.pressure[i] / self.rho0[i]
            
            # Reset conservative variables to be consistent
            self.D[i] = self.rho0[i] * self.W[i]
            self.Sr[i] = 0.0
            self.tau[i] = (self.rho0[i] * self.h[i] * self.W[i]**2 - 
                          self.pressure[i] - self.D[i])


    def apply_floors_and_ceilings(self, max_lorentz=100.0):
        """Apply physical floors and ceilings to prevent unphysical values."""
        
        # Density floor
        rho_floor_mask = self.rho0 < self.atmosphere_rho
        if rho_floor_mask.any():
            floor_indices = np.where(rho_floor_mask)[0]
            self._set_atmosphere_points(floor_indices)
        
        # Lorentz factor ceiling
        W_ceiling_mask = self.W > max_lorentz
        if W_ceiling_mask.any():
            self.W[W_ceiling_mask] = max_lorentz
            # Recompute velocity
            v2_max = 1.0 - 1.0/max_lorentz**2
            self.vr[W_ceiling_mask] = np.sign(self.vr[W_ceiling_mask]) * np.sqrt(v2_max)
            
            # Recompute conservative variables
            ceiling_indices = np.where(W_ceiling_mask)[0]
            for i in ceiling_indices:
                self.D[i] = self.rho0[i] * self.W[i]
                self.Sr[i] = self.rho0[i] * self.h[i] * self.W[i]**2 * self.vr[i]
                self.tau[i] = (self.rho0[i] * self.h[i] * self.W[i]**2 - 
                              self.pressure[i] - self.D[i])
        
        # Pressure floor
        p_floor = 1e-15
        p_floor_mask = self.pressure < p_floor
        if p_floor_mask.any():
            self.pressure[p_floor_mask] = p_floor
            self.eps[p_floor_mask] = self.eos.eps_from_rho_p(
                self.rho0[p_floor_mask], p_floor
            )


    def get_diagnostics(self):
        """Return diagnostic quantities for monitoring evolution."""
        
        if not self.matter_vars_set:
            return {}
        
        diagnostics = {
            'max_density': np.max(self.rho0),
            'min_density': np.min(self.rho0),
            'max_pressure': np.max(self.pressure),
            'min_pressure': np.min(self.pressure),
            'max_lorentz': np.max(self.W),
            'max_velocity': np.max(np.abs(self.vr)),
            'atmosphere_points': np.sum(self.rho0 <= self.atmosphere_rho * 1.1),
            'total_mass': np.sum(self.D),  # Approximate - needs proper integration
        }
        
        return diagnostics