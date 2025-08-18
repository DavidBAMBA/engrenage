# matter/hydro/__init__.py

"""
Relativistic hydrodynamics module for engrenage.

This module implements the Valencia formulation with reference metric
for robust evolution in spherical coordinates, based on:
- Montero et al. (2013): General relativistic hydrodynamics in curvilinear coordinates
- Banyuls et al. (1997): Numerical 3+1 general relativistic hydrodynamics

Key components:
- RelativisticFluid: Main matter class implementing BaseMatter interface
- Valencia reference-metric formulation for spherical coordinates
- HLLE Riemann solver with minmod reconstruction
- Conservative-primitive variable conversions
- Ideal gas equation of state
"""

from .relativistic_fluid import RelativisticFluid
from .valencia_reference_metric import ValenciaReferenceMetric
from .eos import IdealGasEOS, PolytropicEOS
from .con2prim import ConservativeToPrimitive
from .stress_energy_tensor import compute_stress_energy_tensor
from .riemann import HLLERiemannSolver
from .reconstruction import MinmodReconstruction
from .blast_wave_setup import setup_blast_wave_initial_data


# Factory function for easy fluid creation
def create_relativistic_fluid(gamma=1.4, spacetime_mode="fixed_minkowski", 
                            atmosphere_rho=1e-13, reconstruction="minmod", 
                            riemann_solver="hlle"):
    """
    Factory function to create a RelativisticFluid instance with specified configuration.
    
    Args:
        gamma: Adiabatic index for ideal gas EOS
        spacetime_mode: "fixed_minkowski" or "coupled_bssn"  
        atmosphere_rho: Atmosphere density floor
        reconstruction: "minmod" or future options
        riemann_solver: "hlle" or future options
        
    Returns:
        RelativisticFluid instance ready for engrenage evolution
    """
    
    # Create EOS
    eos = IdealGasEOS(gamma)
    
    # Create reconstruction method
    if reconstruction == "minmod":
        reconstructor = MinmodReconstruction()
    else:
        raise ValueError(f"Unknown reconstruction method: {reconstruction}")
    
    # Create Riemann solver
    if riemann_solver == "hlle":
        riemann = HLLERiemannSolver()
    else:
        raise ValueError(f"Unknown Riemann solver: {riemann_solver}")
    
    # Create main fluid object
    fluid = RelativisticFluid(
        eos=eos,
        spacetime_mode=spacetime_mode,
        atmosphere_rho=atmosphere_rho,
        reconstructor=reconstructor,
        riemann_solver=riemann
    )
    
    return fluid

# Convenience function for blast wave test setup
def create_blast_wave_fluid(gamma=1.4, p_inner=1.0, p_outer=0.1, 
                          rho_inner=1.0, rho_outer=0.125, r_discontinuity=0.5):
    """
    Create a RelativisticFluid configured for blast wave testing.
    
    Args:
        gamma: Adiabatic index
        p_inner, p_outer: Inner and outer pressures
        rho_inner, rho_outer: Inner and outer densities
        r_discontinuity: Radius of initial discontinuity
        
    Returns:
        Tuple of (fluid, initial_data_function)
    """
    
    # Create fluid in static spacetime mode
    fluid = create_relativistic_fluid(
        gamma=gamma,
        spacetime_mode="fixed_minkowski",
        atmosphere_rho=min(rho_outer * 1e-3, 1e-13)
    )
    
    # Create initial data function
    def initial_data_func(r):
        return setup_blast_wave_initial_data(
            r, p_inner, p_outer, rho_inner, rho_outer, 
            r_discontinuity, fluid.eos
        )
    
    return fluid, initial_data_func

# Available equation of state types
EOS_TYPES = {
    'ideal': IdealGasEOS,
    'polytropic': PolytropicEOS
}

# Available reconstruction methods
RECONSTRUCTION_TYPES = {
    'minmod': MinmodReconstruction
}

# Available Riemann solvers  
RIEMANN_SOLVERS = {
    'hlle': HLLERiemannSolver
}

__all__ = [
    'RelativisticFluid',
    'ValenciaReferenceMetric', 
    'IdealGasEOS',
    'PolytropicEOS',
    'ConservativeToPrimitive',
    'compute_stress_energy_tensor',
    'HLLERiemannSolver',
    'MinmodReconstruction',
    'setup_blast_wave_initial_data',
    'create_relativistic_fluid',
    'create_blast_wave_fluid',
    'EOS_TYPES',
    'RECONSTRUCTION_TYPES', 
    'RIEMANN_SOLVERS'
]