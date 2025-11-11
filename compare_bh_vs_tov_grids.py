#!/usr/bin/env python3
"""
Compare grid setup and Hamiltonian constraint between BH and TOV initial data.
"""

import numpy as np
import sys

sys.path.insert(0, '/home/yo/repositories/engrenage')

from source.core.grid import Grid
from source.core.spacing import LinearSpacing, CubicSpacing, NUM_GHOSTS, SpacingExtent
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.matter.scalarmatter import ScalarMatter
from source.matter.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.atmosphere import AtmosphereParams
from source.initialdata.bhinitialconditions import get_initial_state as get_bh_initial_state
from source.bssn.constraintsdiagnostic import get_constraints_diagnostic

from examples.TOV.tov_solver import TOVSolver
import examples.TOV.tov_initial_data_interpolated as tov_id


print("="*70)
print("COMPARISON: BH vs TOV INITIAL DATA")
print("="*70)

# ============================================================================
# BH CASE (from BHEvolution.ipynb)
# ============================================================================

print("\n" + "="*70)
print("BH INITIAL DATA (Schwarzschild)")
print("="*70)

r_max_bh = 96.0
min_dr_bh = 1.0 / 16.0
max_dr_bh = 2.0

params_bh = CubicSpacing.get_parameters(r_max_bh, min_dr_bh, max_dr_bh)
spacing_bh = CubicSpacing(**params_bh)
print(f"\nBH Grid params: {params_bh}")

scalar_mu = 1.0
matter_bh = ScalarMatter(scalar_mu)
state_vec_bh = StateVector(matter_bh)
grid_bh = Grid(spacing_bh, state_vec_bh)
background_bh = FlatSphericalBackground(grid_bh.r)

print(f"  N total     = {grid_bh.N}")
print(f"  NUM_GHOSTS  = {NUM_GHOSTS}")
print(f"  r range     = [{grid_bh.r[0]:.6f}, {grid_bh.r[-1]:.6f}]")
print(f"  dr min      = {grid_bh.min_dr:.6f}")
print(f"  First 5 r:  {grid_bh.r[:5]}")
print(f"  Ghost cells: {grid_bh.r[:NUM_GHOSTS]}")

# Get BH initial state
initial_state_bh = get_bh_initial_state(grid_bh, background_bh)

# Compute Hamiltonian constraint
Ham_bh, Mom_bh = get_constraints_diagnostic(
    initial_state_bh, np.array([0]), grid_bh, background_bh, matter_bh
)

# Analyze (exclude ghosts)
interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
r_interior_bh = grid_bh.r[interior]
Ham_interior_bh = Ham_bh[0, interior]

max_Ham_bh = np.max(np.abs(Ham_interior_bh))
i_max_bh = np.argmax(np.abs(Ham_interior_bh))
r_max_Ham_bh = r_interior_bh[i_max_bh]
L2_Ham_bh = np.sqrt(np.mean(Ham_interior_bh**2))

print(f"\nBH Hamiltonian constraint (interior only):")
print(f"  max|Ham| = {max_Ham_bh:.6e} at r = {r_max_Ham_bh:.6f}")
print(f"  L2(Ham)  = {L2_Ham_bh:.6e}")

# Check ghost cells
Ham_ghosts_bh = np.concatenate([Ham_bh[0, :NUM_GHOSTS], Ham_bh[0, -NUM_GHOSTS:]])
max_Ham_ghosts_bh = np.max(np.abs(Ham_ghosts_bh))
print(f"  max|Ham| in GHOST cells = {max_Ham_ghosts_bh:.6e}")

# ============================================================================
# TOV CASE
# ============================================================================

print("\n" + "="*70)
print("TOV INITIAL DATA")
print("="*70)

Gamma = 2.0
K = 1.0
rho_central = 0.128
num_points_tov = 500
r_max_tov = 16.0

spacing_tov = LinearSpacing(num_points_tov, r_max_tov)
eos = IdealGasEOS(gamma=Gamma)
atmosphere = AtmosphereParams()
hydro = PerfectFluid(eos=eos, spacetime_mode='dynamic', atmosphere=atmosphere)
state_vec_tov = StateVector(hydro)
grid_tov = Grid(spacing_tov, state_vec_tov)
background_tov = FlatSphericalBackground(grid_tov.r)

print(f"\nTOV Grid:")
print(f"  N total     = {grid_tov.N}")
print(f"  NUM_GHOSTS  = {NUM_GHOSTS}")
print(f"  r range     = [{grid_tov.r[0]:.6f}, {grid_tov.r[-1]:.6f}]")
print(f"  dr min      = {grid_tov.min_dr:.6f}")
print(f"  First 5 r:  {grid_tov.r[:5]}")
print(f"  Ghost cells: {grid_tov.r[:NUM_GHOSTS]}")

# Solve TOV and create initial data
tov_solver = TOVSolver(K=K, Gamma=Gamma)
tov_solution = tov_solver.solve(rho_central, r_max=r_max_tov)
R = float(tov_solution['R'])
M = float(tov_solution['M_star'])
print(f"\nTOV solution: M = {M:.6f}, R = {R:.6f}")

initial_state_tov = tov_id.create_initial_data_interpolated(
    tov_solution, grid_tov, background_tov, eos,
    atmosphere=atmosphere, interp_order=11
)

# Compute Hamiltonian constraint
hydro.background = background_tov
Ham_tov, Mom_tov = get_constraints_diagnostic(
    initial_state_tov.flatten(), 0.0, grid_tov, background_tov, hydro
)

# Analyze (exclude ghosts)
r_interior_tov = grid_tov.r[interior]
Ham_interior_tov = Ham_tov[0, interior]

max_Ham_tov = np.max(np.abs(Ham_interior_tov))
i_max_tov = np.argmax(np.abs(Ham_interior_tov))
r_max_Ham_tov = r_interior_tov[i_max_tov]
L2_Ham_tov = np.sqrt(np.mean(Ham_interior_tov**2))

print(f"\nTOV Hamiltonian constraint (interior only):")
print(f"  max|Ham| = {max_Ham_tov:.6e} at r = {r_max_Ham_tov:.6f}")
print(f"  L2(Ham)  = {L2_Ham_tov:.6e}")

# Check ghost cells
Ham_ghosts_tov = np.concatenate([Ham_tov[0, :NUM_GHOSTS], Ham_tov[0, -NUM_GHOSTS:]])
max_Ham_ghosts_tov = np.max(np.abs(Ham_ghosts_tov))
print(f"  max|Ham| in GHOST cells = {max_Ham_ghosts_tov:.6e}")

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "="*70)
print("COMPARISON")
print("="*70)

print(f"\nGhost cell r values:")
print(f"  BH:  {grid_bh.r[:NUM_GHOSTS]}")
print(f"  TOV: {grid_tov.r[:NUM_GHOSTS]}")

print(f"\nHamiltonian constraint (interior):")
print(f"  BH:  max|Ham| = {max_Ham_bh:.6e}, L2 = {L2_Ham_bh:.6e}")
print(f"  TOV: max|Ham| = {max_Ham_tov:.6e}, L2 = {L2_Ham_tov:.6e}")

print(f"\nHamiltonian constraint (ghost cells):")
print(f"  BH:  max|Ham| = {max_Ham_ghosts_bh:.6e}")
print(f"  TOV: max|Ham| = {max_Ham_ghosts_tov:.6e}")

print(f"\n" + "="*70)
print("CONCLUSION")
print("="*70)

if max_Ham_ghosts_bh < 1e-6 and max_Ham_ghosts_tov > 1e6:
    print("\n✅ BH has no ghost cell problem (|Ham| small in ghosts)")
    print("❌ TOV has catastrophic ghost cell problem")
    print("\nWhy the difference?")
    print("  → BH initial data is ANALYTIC (exact Schwarzschild)")
    print("  → TOV initial data is INTERPOLATED from numerical solution")
    print("  → Interpolation artifacts + division by r → catastrophe")
elif max_Ham_ghosts_bh > 1e6 and max_Ham_ghosts_tov > 1e6:
    print("\n❌ BOTH have ghost cell problem!")
    print("  This suggests sphericalbackground.py division by r is the issue")
    print("  BH might work because analytic data mitigates the contamination")
else:
    print(f"\n🤔 Unexpected result - need further investigation")

print("="*70 + "\n")
