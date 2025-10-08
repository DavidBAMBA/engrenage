#!/usr/bin/env python3
"""
Example: How to use centralized atmosphere parameters in engrenage.

This demonstrates the new AtmosphereParams class for floor management,
which centralizes all floor/atmosphere configuration in one place.
"""

import numpy as np
from source.matter.hydro import PerfectFluid, AtmosphereParams
from source.matter.hydro.eos import IdealGasEOS, PolytropicEOS

# ==============================================================================
# Method 1: Simple - Just specify atmosphere density
# ==============================================================================
print("=" * 70)
print("Method 1: Simple atmosphere specification")
print("=" * 70)

eos = IdealGasEOS(gamma=2.0)

# Simplest: pass a float for rho_floor
matter1 = PerfectFluid(eos=eos, atmosphere=1e-14)
print(f"atmosphere.rho_floor = {matter1.atmosphere.rho_floor}")
print(f"atmosphere.p_floor   = {matter1.atmosphere.p_floor}")
print(f"atmosphere.v_max     = {matter1.atmosphere.v_max}")

# ==============================================================================
# Method 2: Full control with AtmosphereParams
# ==============================================================================
print("\n" + "=" * 70)
print("Method 2: Full control with AtmosphereParams")
print("=" * 70)

# For TOV evolution, you might want custom parameters
atm_params = AtmosphereParams(
    rho_floor=1e-12,           # Rest mass density floor
    p_floor=1e-14,             # Pressure floor
    v_max=0.99,                # Maximum velocity (in units of c)
    W_max=100.0,               # Maximum Lorentz factor
    tau_atm_factor=1.0,        # tau_atm = factor * p_floor
    conservative_floor_safety=0.999999  # Safety for S^2 constraint
)

matter2 = PerfectFluid(eos=eos, atmosphere=atm_params)
print(f"atmosphere.rho_floor = {matter2.atmosphere.rho_floor}")
print(f"atmosphere.tau_atm   = {matter2.atmosphere.tau_atm}")

# ==============================================================================
# Method 3: TOV-specific setup
# ==============================================================================
print("\n" + "=" * 70)
print("Method 3: TOV evolution setup")
print("=" * 70)

# For TOV, use polytropic EOS and appropriate atmosphere
K = 100.0
Gamma = 2.0
eos_tov = PolytropicEOS(K=K, gamma=Gamma)  # Note: lowercase 'gamma'

# Atmosphere should be ~6 orders of magnitude below central density
# For a typical NS with rho_c ~ 1e-3 (code units), use:
atm_tov = AtmosphereParams(
    rho_floor=1e-10,   # ~6-7 orders below typical NS density
    p_floor=K * (1e-10)**Gamma,  # Consistent with EOS
    v_max=0.9999,      # Allow near-c velocities near surface
)

matter_tov = PerfectFluid(
    eos=eos_tov,
    spacetime_mode="dynamic",
    atmosphere=atm_tov
)

print(f"TOV atmosphere:")
print(f"  rho_floor = {matter_tov.atmosphere.rho_floor:.2e}")
print(f"  p_floor   = {matter_tov.atmosphere.p_floor:.2e}")
print(f"  tau_atm   = {matter_tov.atmosphere.tau_atm:.2e}")

# ==============================================================================
# Method 4: Using in TOVEvolution script (recommended pattern)
# ==============================================================================
print("\n" + "=" * 70)
print("Method 4: Recommended usage in evolution scripts")
print("=" * 70)

print("""
# In your TOVEvolution_corrected.py or similar script:

from source.matter.hydro import PerfectFluid, AtmosphereParams
from source.matter.hydro.eos import PolytropicEOS

# Define atmosphere ONCE at the top
ATMOSPHERE = AtmosphereParams(
    rho_floor=1e-10,
    p_floor=1e-12,
    v_max=0.9999
)

# Create matter with this atmosphere
eos = PolytropicEOS(K=100.0, Gamma=2.0)
matter = PerfectFluid(
    eos=eos,
    spacetime_mode="dynamic",
    atmosphere=ATMOSPHERE
)

# Now ALL subsystems (cons2prim, valencia, reconstruction, riemann)
# automatically use the SAME atmosphere parameters!
# No need to pass rho_floor, p_floor, etc. everywhere.
""")

# ==============================================================================
# Verify floor application
# ==============================================================================
print("\n" + "=" * 70)
print("Floor application demonstration")
print("=" * 70)

# Create test data with values that violate floors
from source.matter.hydro.cons2prim import Cons2PrimSolver

solver = Cons2PrimSolver(eos, atmosphere=atm_params)

D = np.array([1.0, 1e-15, 1.0])     # Second point below floor
Sr = np.array([0.1, 0.0, 0.8])
tau = np.array([0.5, 1e-20, 0.3])   # Second point way below floor
gamma_rr = np.ones(3)

# Apply conservative floors
D_floor, Sr_floor, tau_floor, mask = solver.floor_applicator.apply_conservative_floors(
    D, Sr, tau, gamma_rr
)

print(f"Points where floors applied: {mask}")
print(f"Original tau: {tau}")
print(f"Floored tau:  {tau_floor}")
print(f"tau_atm used: {atm_params.tau_atm}")

# Run full conversion
result = solver.convert(
    U={'D': D_floor, 'Sr': Sr_floor, 'tau': tau_floor},
    metric={'gamma_rr': gamma_rr}
)

print(f"\nConversion success: {result['success']}")
print(f"Resulting rho0: {result['rho0']}")
print(f"Resulting p:    {result['p']}")

stats = solver.get_statistics()
print(f"\nFloor statistics:")
print(f"  Conservative floors applied: {stats['conservative_floors_applied']}")
print(f"  Atmosphere fallbacks: {stats['atmosphere_fallbacks']}")

print("\n" + "=" * 70)
print("✅ All examples completed successfully!")
print("=" * 70)
