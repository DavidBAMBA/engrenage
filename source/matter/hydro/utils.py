import numpy as np


import numpy as np

class HydroNumericalUtils:
    """
    Numerical utilities for relativistic hydrodynamics.
    
    Includes:
    - CFL condition calculations
    - Numerical dissipation
    - Hyperbolicity checks
    - Convergence diagnostics
    """
    
    @staticmethod
    def compute_max_wavespeeds(rho0, vr, pressure, h, W, alpha, gamma_rr, eos):
        """
        Compute characteristic speeds for CFL condition.
        
        Returns eigenvalues λ± of the flux Jacobian.
        """
        
        # Sound speed in fluid frame
        eps = eos.eps_from_rho_p(rho0, pressure)
        cs2 = eos.sound_speed_squared(rho0, pressure, eps)

        
        # Relativistic sound speed
        cs2_rel = cs2 / (1.0 + cs2)
        cs_rel = np.sqrt(cs2_rel)
        
        # Velocity squared
        vr2 = gamma_rr * vr**2
        vr2 = np.clip(vr2, 0.0, 0.999999)
        
        # Characteristic speeds
        denominator = 1.0 - vr2 * cs2_rel
        denominator = np.maximum(denominator, 1e-10)
        
        sqrt_factor = cs_rel * np.sqrt((1.0 - vr2) / denominator)
        
        lambda_plus = alpha * (vr + sqrt_factor) / denominator
        lambda_minus = alpha * (vr - sqrt_factor) / denominator
        lambda_contact = alpha * vr
        
        max_wavespeed = np.maximum(np.abs(lambda_plus), np.abs(lambda_minus))
        
        return {
            'lambda_plus': lambda_plus,
            'lambda_minus': lambda_minus,
            'lambda_contact': lambda_contact,
            'max_wavespeed': max_wavespeed
        }
    
    @staticmethod
    def apply_kreiss_oliger_dissipation(state_vars, r, sigma=0.01):
        """
        Apply Kreiss-Oliger artificial dissipation.
        
        Suppresses high-frequency numerical noise.
        """
        
        if len(r) < 3:
            return state_vars
        
        dr = r[1] - r[0]
        dissipated_vars = {}
        
        for var_name, var_data in state_vars.items():
            d2_var = np.zeros_like(var_data)
            d2_var[1:-1] = (var_data[2:] - 2*var_data[1:-1] + var_data[:-2]) / dr**2
            dissipated_vars[var_name] = var_data - sigma * dr**2 * d2_var
        
        return dissipated_vars
    
    @staticmethod
    def validate_hyperbolicity(rho0, pressure, h, vr, gamma_rr, alpha):
        """Check if system remains hyperbolic."""
        
        diagnostics = {
            'is_hyperbolic': True,
            'issues': []
        }
        
        # Sound speed check
        cs2 = pressure / (rho0 * h)
        if (cs2 > 1.0).any():
            diagnostics['is_hyperbolic'] = False
            diagnostics['issues'].append("Superluminal sound speed")
        
        # Velocity check
        v2 = gamma_rr * vr**2
        if (v2 >= 1.0).any():
            diagnostics['is_hyperbolic'] = False
            diagnostics['issues'].append("Superluminal velocity")
        
        # Discriminant check
        cs2_rel = cs2 / (1.0 + cs2)
        discriminant = (1.0 - v2) * (1.0 - v2*cs2_rel)
        if (discriminant < 0.0).any():
            diagnostics['is_hyperbolic'] = False
            diagnostics['issues'].append("Negative discriminant")
        
        return diagnostics
    
    @staticmethod
    def compute_convergence_factor(error_coarse, error_fine, refinement_factor=2):
        """Compute convergence rate from errors at two resolutions."""
        return np.log(error_coarse/error_fine) / np.log(refinement_factor)
    

class HydroDiagnostics:
    """
    Diagnostic tools for relativistic hydrodynamics simulations.
    
    Monitors:
    - Conservation laws
    - Constraint violations
    - Physical bounds
    """
    
    @staticmethod
    def check_conservation(D, Sr, tau, r):
        """Check conservation of mass, momentum, energy."""
        
        dr = r[1] - r[0] if len(r) > 1 else 1.0
        
        # Integrate conserved quantities (spherical symmetry)
        total_mass = 4.0 * np.pi * np.sum(D * r**2 * dr)
        total_momentum = 4.0 * np.pi * np.sum(Sr * r**2 * dr)
        total_energy = 4.0 * np.pi * np.sum(tau * r**2 * dr)
        
        return {
            'total_mass': total_mass,
            'total_momentum': total_momentum,
            'total_energy': total_energy
        }
    
    @staticmethod
    def check_physical_bounds(rho0, pressure, W, atmosphere_rho=1e-13):
        """Check if variables remain within physical bounds."""
        
        violations = {
            'negative_density': np.sum(rho0 < 0),
            'negative_pressure': np.sum(pressure < 0),
            'superluminal_W': np.sum(W < 1.0),
            'atmosphere_violations': np.sum(rho0 < atmosphere_rho)
        }
        
        return violations