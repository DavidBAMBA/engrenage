#!/usr/bin/env python3
"""
Test creation of initial data with isotropic coordinates (no evolution)
"""

import sys
sys.path.append('/home/yo/repositories/engrenage')

import numpy as np
import time

from examples.tov_solver import TOVSolver
from examples.tov_initial_data_adm_bssn import create_initial_data_adm_bssn

from source.core.grid import Grid
from source.core.spacing import LinearSpacing
from source.core.state_vector import StateVector
from source.core.background import FlatSphericalBackground
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.atmosphere import AtmosphereParams
from source.matter.eos.ideal_gas import IdealGasEOS
from source.bssn.bssnvars import BSSNVars
from source.bssn.constraintsdiagnostic import get_constraints_diagnostic

print("Testing initial data creation with isotropic coordinates")
print("=" * 60)

# Parameters
K = 100.0
Gamma = 2.0
rho_central = 1.28e-3
num_points = 1000

print(f"Parameters: K={K}, Gamma={Gamma}, rho_c={rho_central}, N={num_points}")
print()

# Solve TOV
print("Step 1: Solving TOV in isotropic coordinates...")
start = time.time()
tov_solver = TOVSolver(K=K, Gamma=Gamma, use_isotropic=True)
tov_solution = tov_solver.solve(rho_central, r_max=20.0, dr=0.001)
elapsed = time.time() - start
print(f"  ✓ Completed in {elapsed:.2f}s")
print(f"  M={tov_solution['M_star']:.6f}, R={tov_solution['R']:.6f}")
print()

# Setup grid
R_star = tov_solution['R']
r_max = 2.0 * R_star
spacing = LinearSpacing(num_points, r_max)
eos = IdealGasEOS(gamma=Gamma)

ATMOSPHERE = AtmosphereParams(
    rho_floor=1.0e-12,
    p_floor=1.0e-14,
    v_max=0.9999,
    W_max=100.0,
)

hydro = PerfectFluid(
    eos=eos,
    spacetime_mode="dynamic",
    atmosphere=ATMOSPHERE,
    reconstructor=None,
    riemann_solver=None
)

state_vector = StateVector(hydro)
grid = Grid(spacing, state_vector)
background = FlatSphericalBackground(grid.r)
hydro.background = background

print(f"Step 2: Setting up grid N={num_points}, r_max={r_max:.6f}")
print()

# Create initial data
print("Step 3: Creating initial data (ADM→BSSN with isotropic coordinates)...")
start = time.time()

tov_solution_grid = tov_solver.solve(rho_central, r_max=r_max, dr=r_max/4000)

initial_state_2d = create_initial_data_adm_bssn(
    tov_solution_grid, grid, background, eos,
    atmosphere=ATMOSPHERE,
    polytrope_K=K, polytrope_Gamma=Gamma,
    use_hydrobase_tau=True,
    use_isotropic=True
)
elapsed = time.time() - start
print(f"  ✓ Completed in {elapsed:.2f}s")
print()

# Compute Hamiltonian constraint
print("Step 4: Computing Hamiltonian constraint...")
start = time.time()

NUM_BSSN_VARS = grid.NUM_VARS - hydro.NUM_VARS
bssn_vars = BSSNVars(grid.N)
bssn_vars.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
hydro.set_matter_vars(initial_state_2d, bssn_vars, grid)

Ham, Mom = get_constraints_diagnostic(
    initial_state_2d.flatten(), 0.0, grid, background, hydro
)
elapsed = time.time() - start

max_H = np.max(np.abs(Ham[0, :]))
log10_H = np.log10(max_H) if max_H > 0 else -np.inf

print(f"  ✓ Completed in {elapsed:.2f}s")
print(f"  max|H(t=0)| = {max_H:.6e} (log10 = {log10_H:.2f})")
print()

if log10_H < -8:
    print("✓✓✓ SUCCESS! H(t=0) ~ 10^-10 as expected!")
elif log10_H < -2:
    print("⚠ Moderate: H(t=0) ~ 10^{:.0f}, better than before but not ideal".format(log10_H))
else:
    print("✗ PROBLEM: H(t=0) ~ 10^{:.0f}, still too large".format(log10_H))
