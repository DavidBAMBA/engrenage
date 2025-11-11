#!/usr/bin/env python3
"""
Check hydrostatic equilibrium at a point INTERIOR to the surface,
where the solution is smooth and numerical derivatives are reliable.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/yo/repositories/engrenage')

from tov_solver import TOVSolver

# Same parameters as evolution
K, Gamma = 100.0, 2.0
central_rho = 1.28e-3

print("=" * 80)
print("TOV HYDROSTATIC EQUILIBRIUM CHECK (INTERIOR POINT)")
print("=" * 80)

# Solve TOV
tov = TOVSolver(K, Gamma)
tov_sol = tov.solve(central_rho)

R = tov_sol['R']
r_arr = tov_sol['r']

print(f"\nTOV solution:")
print(f"  Stellar radius:  R = {R:.6f}")

# Check equilibrium at several interior points near the surface
r_test_points = [R - 0.5, R - 0.2, R - 0.1, R - 0.05]

for r_test in r_test_points:
    # Find index closest to r_test
    i = np.argmin(np.abs(r_arr - r_test))

    # Extract quantities
    rho = tov_sol['rho_baryon'][i]
    P = tov_sol['P'][i]
    m = tov_sol['M'][i]
    alpha = tov_sol['alpha'][i]
    r = r_arr[i]

    # Skip if too close to atmosphere
    if rho < 1e-6:
        print(f"\nSkipping r = {r:.6f} (too close to atmosphere, ρ = {rho:.6e})")
        continue

    # Numerical derivatives (central differences)
    dP_dr = (tov_sol['P'][i+1] - tov_sol['P'][i-1]) / (r_arr[i+1] - r_arr[i-1])
    dalpha_dr = (tov_sol['alpha'][i+1] - tov_sol['alpha'][i-1]) / (r_arr[i+1] - r_arr[i-1])

    # TOV equation
    dP_dr_tov = -(rho + P) * (m + 4*np.pi*r**3*P) / (r * (r - 2*m))

    # Thermodynamics
    eps = K * rho**(Gamma-1) / (Gamma - 1)
    h = 1.0 + eps + P/rho

    # Hydrostatic balance
    pressure_force = dP_dr
    metric_force = (rho * h) * (dalpha_dr / alpha)
    imbalance = pressure_force + metric_force

    print(f"\n" + "=" * 80)
    print(f"At r = {r:.6f} (R - {R - r:.3f})")
    print("=" * 80)
    print(f"  ρ = {rho:.6e}")
    print(f"  P = {P:.6e}")
    print(f"  α = {alpha:.6f}")
    print(f"  h = {h:.6f}")

    print(f"\nTOV equation:")
    print(f"  dP/dr (numerical)   = {dP_dr:.6e}")
    print(f"  dP/dr (TOV formula) = {dP_dr_tov:.6e}")
    print(f"  Relative error      = {abs(dP_dr - dP_dr_tov)/abs(dP_dr):.6e}")

    print(f"\nHydrostatic balance (evolution form):")
    print(f"  dP/dr           = {pressure_force:.6e}")
    print(f"  (ρ h)(dα/dr/α)  = {metric_force:.6e}")
    print(f"  Imbalance       = {imbalance:.6e}")
    print(f"  Relative        = {abs(imbalance)/abs(pressure_force):.6e}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("\nIf equilibrium error is < 1e-6 in interior, then the TOV solver is accurate.")
print("The issue at surface is likely due to:")
print("  1. Discretization error when interpolating to evolution grid")
print("  2. Numerical derivatives at steep gradient")
print("  3. Atmosphere treatment at density discontinuity")
print("=" * 80)
