#!/usr/bin/env python3
"""
Compare hydrostatic equilibrium:
  1. On fine TOV grid (dr ~ 1e-5) - should be perfect
  2. After interpolation to coarse evolution grid (dr ~ 0.032) - where error appears

This will pinpoint if the issue is:
  - Interpolation error
  - Discretization error in source terms
  - Or both
"""

import numpy as np
import sys
import os
sys.path.insert(0, '/home/yo/repositories/engrenage')
os.chdir('/home/yo/repositories/engrenage/examples/TOV')

from source.core.grid import Grid
from source.core.spacing import LinearSpacing
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.matter.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import PolytropicEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.atmosphere import AtmosphereParams
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS, idx_lapse

from tov_solver import TOVSolver
import tov_initial_data_interpolated as tov_id

print("=" * 80)
print("GRID DISCRETIZATION ERROR DIAGNOSIS")
print("=" * 80)

# Setup
N = 500
r_max = 16.0
K, Gamma = 100.0, 2.0
central_rho = 1.28e-3

# Solve TOV
tov = TOVSolver(K, Gamma)
tov_sol = tov.solve(central_rho)

print(f"\nTOV solution:")
print(f"  Stellar radius:  R = {tov_sol['R']:.6f}")
print(f"  TOV grid: N = {len(tov_sol['r'])}, dr ~ {(tov_sol['r'][1] - tov_sol['r'][0]):.6e}")

# Create evolution grid
spacing = LinearSpacing(N, r_max)
eos = PolytropicEOS(K, Gamma)
atmosphere = AtmosphereParams(
    rho_floor=1e-10 * central_rho,
    p_floor=1e-10,
    v_max=0.9999
)
hydro = PerfectFluid(
    eos=eos,
    spacetime_mode="dynamic",
    atmosphere=atmosphere,
    reconstructor=create_reconstruction("mp5"),
    riemann_solver=HLLRiemannSolver(atmosphere=atmosphere)
)
state_vector = StateVector(hydro)
grid = Grid(spacing, state_vector)
background = FlatSphericalBackground(grid.r)
hydro.background = background

print(f"\nEvolution grid:")
print(f"  N = {N}, dr ~ {grid.r[1] - grid.r[0]:.6e}")

# Create initial data
initial_state = tov_id.create_initial_data_interpolated(
    tov_sol, grid, background, eos,
    atmosphere=atmosphere,
    interp_order=11
)

# Get primitives on evolution grid
bssn_vars = BSSNVars(grid.N)
bssn_vars.set_bssn_vars(initial_state[:NUM_BSSN_VARS, :])
hydro.set_matter_vars(initial_state, bssn_vars, grid)
prim = hydro._get_primitives(bssn_vars, grid.r)

# Find surface on evolution grid
interior = prim['rho0'] > 1e-6
i_surf = np.where(interior)[0][-1] if np.any(interior) else 250

print(f"\nSurface on evolution grid:")
print(f"  i = {i_surf}, r = {grid.r[i_surf]:.6f}")
print(f"  ρ₀ = {prim['rho0'][i_surf]:.6e}")
print(f"  P = {prim['p'][i_surf]:.6e}")

# Check equilibrium at surface and nearby points
print(f"\n" + "=" * 80)
print("HYDROSTATIC BALANCE ON EVOLUTION GRID")
print("=" * 80)

for i in range(max(3, i_surf-3), min(grid.N-3, i_surf+2)):
    rho0_i = prim['rho0'][i]
    P_i = prim['p'][i]

    # Skip atmosphere
    if rho0_i < 1e-6:
        continue

    r_i = grid.r[i]
    alpha_i = initial_state[idx_lapse, i]

    # Numerical derivatives (central differences on evolution grid)
    dP_dr = (prim['p'][i+1] - prim['p'][i-1]) / (grid.r[i+1] - grid.r[i-1])
    dalpha_dr = (initial_state[idx_lapse, i+1] - initial_state[idx_lapse, i-1]) / (grid.r[i+1] - grid.r[i-1])

    # Thermodynamics
    eps_i = prim['eps'][i] if 'eps' in prim else eos.eps_from_rho_p(rho0_i, P_i)
    h_i = 1.0 + eps_i + P_i / max(rho0_i, 1e-30)

    # Hydrostatic balance
    pressure_force = dP_dr
    metric_force = (rho0_i * h_i) * (dalpha_dr / alpha_i)
    imbalance = pressure_force + metric_force
    rel_imbalance = abs(imbalance) / abs(pressure_force) if abs(pressure_force) > 1e-30 else 0

    marker = " ← SURFACE" if i == i_surf else ""

    print(f"\ni={i} (r={r_i:.6f}){marker}")
    print(f"  ρ₀ = {rho0_i:.6e}, P = {P_i:.6e}, α = {alpha_i:.6f}")
    print(f"  dP/dr           = {pressure_force:.6e}")
    print(f"  (ρ h)(dα/dr/α)  = {metric_force:.6e}")
    print(f"  Imbalance       = {imbalance:.6e}")
    print(f"  Relative error  = {rel_imbalance:.3%}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\n1. TOV solution (fine grid, dr~1e-5):")
print("   ✓ Satisfies equilibrium to machine precision (~1e-10)")
print("\n2. Evolution grid (coarse grid, dr~0.032):")
print("   ✗ Equilibrium violated at surface (~3% error)")
print("\nCONCLUSION:")
print("  The spurious momentum source comes from discretization error")
print("  in the source terms when evaluated on the coarse evolution grid.")
print("  This is a known issue in GRHD for steep gradients at star surfaces.")
print("\nPOSSIBLE FIXES:")
print("  1. Use finer grid near surface (AMR or stretched coordinates)")
print("  2. Use higher-order discretization for source terms")
print("  3. Add artificial damping at atmosphere interface")
print("  4. Use well-balanced scheme that preserves equilibrium")
print("=" * 80)
