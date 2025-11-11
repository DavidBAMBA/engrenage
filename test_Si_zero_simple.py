"""
Simple test: Run TOV for a few steps with Si=0 forced to see if velocity growth stops.
"""
import numpy as np
import sys
import os

# Change to examples/TOV directory
tov_dir = os.path.join(os.path.dirname(__file__), 'examples', 'TOV')
os.chdir(tov_dir)
sys.path.insert(0, tov_dir)

# Now import from the TOV directory
from tov_solver import TOVSolver
import tov_initial_data_interpolated as tov_id

# Add repo root for source imports
repo_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, repo_root)

from source.core.grid import Grid
from source.matter.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import PolytropicEOS
from source.backgrounds.sphericalbackground import FlatSphericalBackground

print("=" * 80)
print("Quick Test: TOV with S_i = 0 Forced")
print("=" * 80)
print("\nNOTE: perfect_fluid.py has been modified to force emtensor.Si[:] = 0.0")
print("This disables momentum coupling to BSSN lambda^i evolution.")

# Grid
N = 256
r_max = 20.0
grid = Grid(N=N, r_max=r_max, num_ghosts=2, mode='bssn+matter')

# EOS
K = 100.0
Gamma = 2.0
rho_central = 1.28e-3
eos = PolytropicEOS(K=K, gamma=Gamma)

# Background
background = FlatSphericalBackground(grid)

# Matter
matter = PerfectFluid(eos=eos)

# Solve TOV
print("\nSolving TOV equations...")
tov_solver = TOVSolver(K=K, Gamma=Gamma)
tov_solution = tov_solver.solve(rho_central, r_max=15.0, num_points=10000)

print(f"TOV Solution: M={tov_solution['M_star']:.6f}, R={tov_solution['R']:.4f}, C={tov_solution['C']:.5f}")

# Create initial data
print("\nCreating initial data...")
from source.matter.hydro.atmosphere import AtmosphereParams
atmosphere = AtmosphereParams(rho_floor=1e-13, p_floor=1e-15, D_floor=1e-13,
                               tau_floor=1e-15, Sr_floor=1e-16,
                               cons_tau_floor_coefficient=1.1,
                               cons_Sr_floor_coefficient=1.1)

initial_state = tov_id.create_initial_data_interpolated(
    tov_solution, grid, background, eos,
    atmosphere=atmosphere,
    interp_order=11
)

# Extract matter variables
from source.bssn.bssnvars import BSSNVars
bssn_vars = BSSNVars(initial_state, grid)
matter.set_matter_vars(initial_state, bssn_vars, grid)

# Get primitives to find surface
from source.matter.hydro.cons2prim import Cons2PrimSolver
from source.bssn.tensoralgebra import get_bar_gamma_LL

bar_gamma = get_bar_gamma_LL(grid.r, bssn_vars.h_LL, background)
em4phi = np.exp(-4.0 * bssn_vars.phi)
gamma_rr = em4phi * bar_gamma[:, 0, 0]
alpha = bssn_vars.lapse
beta_r = np.zeros(N)

cons2prim = Cons2PrimSolver(eos=eos, atmosphere=atmosphere)
metric = (alpha, beta_r, gamma_rr)
U = (matter.D, matter.Sr, matter.tau)
primitives = cons2prim.convert(U=U, metric=metric, p_guess=None, apply_conservative_floors=False)

# Find stellar surface
rho0 = primitives['rho0']
threshold = 0.01 * rho0[grid.num_ghosts]
surface_idx = None
for i in range(grid.num_ghosts, N - grid.num_ghosts):
    if rho0[i] < threshold:
        surface_idx = i
        break

if surface_idx is None:
    print("ERROR: Could not find stellar surface")
    sys.exit(1)

r_surface = grid.r[surface_idx]
print(f"\nStellar surface: i={surface_idx}, r={r_surface:.2f}")

# Initial values at surface
vr_initial = primitives['vr'][surface_idx]
Sr_initial = matter.Sr[surface_idx]

print(f"\nInitial state at surface:")
print(f"  v^r = {vr_initial:+.6e}")
print(f"  S_r = {Sr_initial:+.6e}")
print(f"  ρ₀  = {rho0[surface_idx]:.3e}")
print(f"  p   = {primitives['p'][surface_idx]:.3e}")

# Time evolution (just matter RHS, BSSN frozen for simplicity)
from source.core.derivatives import Derivatives

dt = 0.01
num_steps = 50

derivatives = Derivatives(grid, background)

print(f"\nEvolving for {num_steps} steps (dt={dt})...")

for step in range(num_steps):
    # Get BSSN derivatives
    bssn_d1 = derivatives.get_bssn_d1(initial_state, matter)

    # Compute matter RHS
    matter.matter_vars_set = True
    matter.pressure_cache = primitives['p']
    matter_rhs = matter.get_matter_rhs(grid.r, bssn_vars, bssn_d1, background)

    # Forward Euler step
    matter.D += dt * matter_rhs[0, :]
    matter.Sr += dt * matter_rhs[1, :]
    matter.tau += dt * matter_rhs[2, :]

    # Update primitives every 10 steps
    if step % 10 == 0:
        U = (matter.D, matter.Sr, matter.tau)
        primitives = cons2prim.convert(U=U, metric=metric, p_guess=primitives['p'],
                                        apply_conservative_floors=False)

# Final primitives
U = (matter.D, matter.Sr, matter.tau)
primitives = cons2prim.convert(U=U, metric=metric, p_guess=primitives['p'],
                                apply_conservative_floors=False)

vr_final = primitives['vr'][surface_idx]
Sr_final = matter.Sr[surface_idx]

print(f"\nFinal state at surface (t={num_steps*dt}):")
print(f"  v^r = {vr_final:+.6e}")
print(f"  S_r = {Sr_final:+.6e}")

# Analysis
dv = vr_final - vr_initial
dSr = Sr_final - Sr_initial

print(f"\nChanges:")
print(f"  Δv^r = {dv:+.6e}")
print(f"  ΔS_r = {dSr:+.6e}")

print("\n" + "=" * 80)
if abs(dv) < 1e-4:
    print("✓ VELOCITY IS STABLE with S_i = 0 forced!")
    print("  This confirms the BSSN-matter coupling creates a feedback loop.")
    print("  The physical coupling is correct, but TOV should have S_i = 0 exactly.")
    print("\n  → Need to investigate why S_i ≠ 0 in the simulation:")
    print("    1. Check TOV initial conditions")
    print("    2. Check cons2prim solver")
    print("    3. Check numerical errors in momentum equation")
else:
    print("⚠️  VELOCITY STILL GROWS even with S_i = 0 forced!")
    print("  The bug is NOT in the BSSN-matter coupling.")
    print("\n  → Bug must be in:")
    print("    1. GRHD momentum equations")
    print("    2. Initial conditions")
    print("    3. Boundary conditions")
