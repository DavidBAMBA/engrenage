#!/usr/bin/env python3
"""
Check if TOV solution satisfies hydrostatic equilibrium exactly.

The TOV equation is:
  dP/dr = -(ρ + P)(m + 4πr³P) / [r(r - 2m)]

For the evolution code, the equilibrium condition is:
  dP/dr + (ρ₀ h)(dα/dr/α) = 0

We need to verify these are consistent at the stellar surface.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/yo/repositories/engrenage')

from tov_solver import TOVSolver

# Same parameters as evolution
K, Gamma = 100.0, 2.0
central_rho = 1.28e-3

print("=" * 80)
print("TOV HYDROSTATIC EQUILIBRIUM CHECK AT SURFACE")
print("=" * 80)

# Solve TOV
tov = TOVSolver(K, Gamma)
tov_sol = tov.solve(central_rho)

print(f"\nTOV solution:")
print(f"  Central density: ρ_c = {central_rho:.6e}")
print(f"  Stellar radius:  R   = {tov_sol['R']:.6f}")
print(f"  Total mass:      M   = {tov_sol['M_star']:.6f}")

# Find surface index
R = tov_sol['R']
r_arr = tov_sol['r']
i_surf = np.argmin(np.abs(r_arr - R))

print(f"\nSurface location in TOV arrays:")
print(f"  Index: i = {i_surf}")
print(f"  Radius: r = {r_arr[i_surf]:.6f}")

# Extract quantities at surface
rho = tov_sol['rho_baryon'][i_surf]
P = tov_sol['P'][i_surf]
m = tov_sol['M'][i_surf]
alpha = tov_sol['alpha'][i_surf]

# Numerical derivatives (use central differences)
dr = r_arr[i_surf] - r_arr[i_surf-1]  # Local spacing
dP_dr = (tov_sol['P'][i_surf+1] - tov_sol['P'][i_surf-1]) / (r_arr[i_surf+1] - r_arr[i_surf-1])
dalpha_dr = (tov_sol['alpha'][i_surf+1] - tov_sol['alpha'][i_surf-1]) / (r_arr[i_surf+1] - r_arr[i_surf-1])

print(f"\nQuantities at surface:")
print(f"  ρ = {rho:.6e}")
print(f"  P = {P:.6e}")
print(f"  m = {m:.6e}")
print(f"  α = {alpha:.6f}")

print(f"\nNumerical derivatives:")
print(f"  dP/dr    = {dP_dr:.6e}")
print(f"  dα/dr    = {dalpha_dr:.6e}")
print(f"  dα/dr/α  = {dalpha_dr/alpha:.6e}")

# Check TOV equation
r = r_arr[i_surf]
dP_dr_tov = -(rho + P) * (m + 4*np.pi*r**3*P) / (r * (r - 2*m))

print(f"\nTOV equation check:")
print(f"  dP/dr (numerical)   = {dP_dr:.6e}")
print(f"  dP/dr (TOV formula) = {dP_dr_tov:.6e}")
print(f"  Difference          = {dP_dr - dP_dr_tov:.6e}")
print(f"  Relative error      = {abs(dP_dr - dP_dr_tov)/abs(dP_dr):.6e}")

# For polytropic EOS: ε = K ρ^(Γ-1) / (Γ-1)
eps = K * rho**(Gamma-1) / (Gamma - 1)
h = 1.0 + eps + P/rho
print(f"\nThermodynamics:")
print(f"  ε = {eps:.6e}")
print(f"  h = {h:.6f}")

# Check hydrostatic balance as used in evolution code
pressure_force = dP_dr
metric_force = (rho * h) * (dalpha_dr / alpha)
imbalance = pressure_force + metric_force

print(f"\nHydrostatic balance (evolution code form):")
print(f"  dP/dr           = {pressure_force:.6e}")
print(f"  (ρ h)(dα/dr/α)  = {metric_force:.6e}")
print(f"  Imbalance       = {imbalance:.6e}")
print(f"  Relative        = {imbalance/abs(pressure_force) if abs(pressure_force) > 1e-30 else 0:.6e}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()
if abs(imbalance/pressure_force) < 1e-6:
    print("✓ TOV solution satisfies equilibrium to machine precision.")
    print("  → The problem is in the discretization/interpolation to evolution grid")
else:
    print("✗ TOV solution has significant equilibrium error!")
    print("  → Need to improve TOV solver accuracy")
print()
print("=" * 80)
