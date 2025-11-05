#!/usr/bin/env python3
"""Quick test to check if density at origin is stable."""
import numpy as np
import sys
sys.path.insert(0, '/home/yo/repositories/engrenage')
from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.matter.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.atmosphere import AtmosphereParams
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS

# Minimal setup
N = 128
r_max = 16.0
spacing = LinearSpacing(N, r_max)
atmosphere = AtmosphereParams(rho_floor=1e-13, p_floor=1e-15)
eos = IdealGasEOS(gamma=2.0)
hydro = PerfectFluid(
    eos=eos, spacetime_mode="dynamic", atmosphere=atmosphere,
    reconstructor=create_reconstruction("minmod"),
    riemann_solver=HLLRiemannSolver(atmosphere=atmosphere)
)
state_vector = StateVector(hydro)
grid = Grid(spacing, state_vector)
background = FlatSphericalBackground(grid.r)
hydro.background = background

# Create simple equilibrium state (uniform density)
rho0 = 1e-3 * np.ones(grid.N)
P = 1e-4 * np.ones(grid.N)
vr = np.zeros(grid.N)

# Set conservatives
from source.matter.hydro.cons2prim import prim_to_cons
gamma_rr = np.ones(grid.N)
D, Sr, tau = prim_to_cons(rho0, vr, P, gamma_rr, eos)

# Create BSSN (flat spacetime)
state = np.zeros((NUM_BSSN_VARS + 3, grid.N))
# For flat spacetime: phi=0, h_ij=0, K=0, a_ij=0, lambda^r=0, shift^r=0, b^r=0, lapse=1
from source.bssn.bssnstatevariables import idx_lapse
state[idx_lapse, :] = 1.0  # Set lapse = 1 for flat spacetime

state[NUM_BSSN_VARS + 0, :] = D
state[NUM_BSSN_VARS + 1, :] = Sr
state[NUM_BSSN_VARS + 2, :] = tau

bssn_vars = BSSNVars(grid.N)
bssn_vars.set_bssn_vars(state[:NUM_BSSN_VARS, :])
bssn_d1 = grid.get_d1_metric_quantities(state)

print("Testing origin stability with W/h recalculation fix...")
print(f"Initial ρ_c = {rho0[NUM_GHOSTS]:.8e}")

# Compute RHS 10 times
for step in range(10):
    grid.fill_boundaries(state)
    hydro.set_matter_vars(state, bssn_vars, grid)
    prims = hydro._get_primitives(bssn_vars, grid.r)

    rho_c = prims['rho0'][NUM_GHOSTS]
    v_c = prims['vr'][NUM_GHOSTS]

    rhs = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1, background)
    dD_dt_c = rhs[0, NUM_GHOSTS]

    print(f"Step {step}: ρ_c={rho_c:.8e}  v^r_c={v_c:.2e}  dD/dt|_c={dD_dt_c:.2e}")

    # Small Euler step
    state[NUM_BSSN_VARS:, :] += 0.001 * rhs

prims_final = hydro._get_primitives(bssn_vars, grid.r)
rho_final = prims_final['rho0'][NUM_GHOSTS]
drho = (rho_final - rho0[NUM_GHOSTS]) / rho0[NUM_GHOSTS]

print(f"\nFinal ρ_c = {rho_final:.8e}")
print(f"Δρ/ρ = {drho:.2e}")

if abs(drho) < 1e-6:
    print("\n✓✓✓ SUCCESS! Density is stable at origin!")
else:
    print(f"\n✗ FAILURE: Density drifted by {drho:.2%}")
