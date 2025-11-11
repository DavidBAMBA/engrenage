"""
Diagnose TOV initial conditions: Check if v^r = 0 exactly at t=0.

This script loads TOV initial data and checks:
1. Is v^r exactly zero everywhere?
2. Is S^r exactly zero everywhere?
3. Does hydrostatic equilibrium hold numerically?
"""

import numpy as np
import sys
import os

# Change to TOV directory
tov_dir = os.path.join(os.path.dirname(__file__), 'examples', 'TOV')
os.chdir(tov_dir)
sys.path.insert(0, tov_dir)

from tov_solver import TOVSolver
import tov_initial_data_interpolated as tov_id

# Add repo root
repo_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, repo_root)

from source.core.statevector import StateVector
from source.core.spacing import LinearSpacing
from source.core.grid import Grid
from source.matter.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.matter.hydro.atmosphere import AtmosphereParams

print("=" * 80)
print("TOV Initial Conditions Diagnostic")
print("=" * 80)

# Parameters
N = 512  # High resolution to see details
r_max = 20.0
K = 100.0
Gamma = 2.0
rho_central = 1.28e-3

# Grid setup
spacing = LinearSpacing(N, r_max)
eos = IdealGasEOS(gamma=Gamma)
hydro = PerfectFluid(eos=eos, spacetime_mode="dynamic")
state_vector = StateVector(hydro)
grid = Grid(spacing, state_vector)
background = FlatSphericalBackground(grid.r)

print(f"\nGrid: N={N}, r_max={r_max:.1f}, dr={spacing.dx:.4f}")

# Atmosphere
atmosphere = AtmosphereParams(
    rho_floor=1.0e-10,
    p_floor=1.0e-11,
    v_max=0.9999,
    W_max=100.0,
    tau_atm_factor=1.0,
    conservative_floor_safety=0.999999
)

# Solve TOV
print("\nSolving TOV equations...")
tov_solver = TOVSolver(K=K, Gamma=Gamma)
tov_solution = tov_solver.solve(rho_central, r_max=15.0, num_points=10000)

M_star = tov_solution['M_star']
R_star = tov_solution['R']
C = tov_solution['C']

print(f"  M_star = {M_star:.6f}")
print(f"  R_star = {R_star:.4f}")
print(f"  C = M/R = {C:.5f}")

# Create initial data
print("\nCreating initial data (interpolation order 11)...")
initial_state = tov_id.create_initial_data_interpolated(
    tov_solution, grid, background, eos,
    atmosphere=atmosphere,
    interp_order=11
)

# Extract matter variables
from source.bssn.bssnvars import BSSNVars
bssn_vars = BSSNVars(initial_state, grid)
hydro.set_matter_vars(initial_state, bssn_vars, grid)

# Get primitives
from source.matter.hydro.cons2prim import Cons2PrimSolver
from source.bssn.tensoralgebra import get_bar_gamma_LL

bar_gamma = get_bar_gamma_LL(grid.r, bssn_vars.h_LL, background)
em4phi = np.exp(-4.0 * bssn_vars.phi)
gamma_rr = em4phi * bar_gamma[:, 0, 0]
alpha = bssn_vars.lapse
beta_r = np.zeros(N)

cons2prim = Cons2PrimSolver(eos=eos, atmosphere=atmosphere)
metric = (alpha, beta_r, gamma_rr)
U = (hydro.D, hydro.Sr, hydro.tau)
primitives = cons2prim.convert(U=U, metric=metric, p_guess=None, apply_conservative_floors=False)

rho0 = primitives['rho0']
vr = primitives['vr']
p = primitives['p']
Sr = hydro.Sr

# Find stellar surface
threshold = 0.01 * rho0[grid.num_ghosts]
surface_idx = None
for i in range(grid.num_ghosts, N - grid.num_ghosts):
    if rho0[i] < threshold:
        surface_idx = i
        break

r_surface = grid.r[surface_idx]

print("\n" + "=" * 80)
print("INITIAL CONDITIONS ANALYSIS (t=0)")
print("=" * 80)

print(f"\nStellar surface: i={surface_idx}, r={r_surface:.2f}")

# Statistics
vr_max = np.max(np.abs(vr))
vr_max_idx = np.argmax(np.abs(vr))
Sr_max = np.max(np.abs(Sr))
Sr_max_idx = np.argmax(np.abs(Sr))

print(f"\nGlobal statistics:")
print(f"  max(|v^r|) = {vr_max:.6e} at i={vr_max_idx}, r={grid.r[vr_max_idx]:.2f}")
print(f"  max(|S^r|) = {Sr_max:.6e} at i={Sr_max_idx}, r={grid.r[Sr_max_idx]:.2f}")
print(f"  mean(|v^r|) = {np.mean(np.abs(vr)):.6e}")
print(f"  mean(|S^r|) = {np.mean(np.abs(Sr)):.6e}")

# Check if truly zero
vr_zero = vr_max < 1e-14
Sr_zero = Sr_max < 1e-14

if vr_zero and Sr_zero:
    print("\n✓ Initial conditions have v^r = 0 and S^r = 0 (within machine precision)")
else:
    print(f"\n⚠️  Initial conditions have NON-ZERO velocity/momentum:")
    if not vr_zero:
        print(f"    v^r_max = {vr_max:.3e} (threshold: 1e-14)")
    if not Sr_zero:
        print(f"    S^r_max = {Sr_max:.3e} (threshold: 1e-14)")

# Profile near surface
print(f"\nProfile near stellar surface (r~{r_surface:.1f}):")
print(f"{'i':>4} {'r':>8} {'ρ₀':>12} {'p':>12} {'v^r':>12} {'S^r':>12}")
print("-" * 70)

for i in range(max(grid.num_ghosts, surface_idx - 10), min(N - grid.num_ghosts, surface_idx + 10)):
    print(f"{i:4d} {grid.r[i]:8.3f} {rho0[i]:12.3e} {p[i]:12.3e} {vr[i]:+12.3e} {Sr[i]:+12.3e}")

# Check hydrostatic equilibrium numerically
print("\n" + "=" * 80)
print("HYDROSTATIC EQUILIBRIUM CHECK")
print("=" * 80)

# For TOV: ∂p/∂r = -(ρ₀ + p)(m + 4πr³p) / (r(r - 2m))
# where m(r) = integrated mass

# Compute pressure gradient numerically
from source.core.derivatives import Derivatives
derivatives = Derivatives(grid, background)

# Simple finite difference for pressure
dp_dr = np.zeros(N)
for i in range(grid.num_ghosts, N - grid.num_ghosts):
    dp_dr[i] = (p[i+1] - p[i-1]) / (2.0 * spacing.dx)

# Compute expected gradient from TOV equation
# This is approximate - just to check if roughly in equilibrium
h = 1.0 + eos.specific_internal_energy(rho0, p) + p / rho0  # Specific enthalpy

# For spherical symmetry: force balance is
# (1/ρh) ∂p/∂r + (1/α) ∂α/∂r + (1/2) ∂ln(γ_rr)/∂r = 0 (for v^r = 0)

# Compute lapse gradient
dalpha_dr = np.zeros(N)
for i in range(grid.num_ghosts, N - grid.num_ghosts):
    dalpha_dr[i] = (alpha[i+1] - alpha[i-1]) / (2.0 * spacing.dx)

# Compute metric gradient
dgamma_dr = np.zeros(N)
for i in range(grid.num_ghosts, N - grid.num_ghosts):
    dgamma_dr[i] = (gamma_rr[i+1] - gamma_rr[i-1]) / (2.0 * spacing.dx)

# Hydrostatic balance residual
residual = np.zeros(N)
for i in range(grid.num_ghosts, N - grid.num_ghosts):
    if rho0[i] > atmosphere.rho_floor * 10:  # Only inside star
        term1 = dp_dr[i] / (rho0[i] * h[i]) if rho0[i] * h[i] > 1e-20 else 0.0
        term2 = dalpha_dr[i] / alpha[i] if alpha[i] > 1e-20 else 0.0
        term3 = 0.5 * dgamma_dr[i] / gamma_rr[i] if gamma_rr[i] > 1e-20 else 0.0
        residual[i] = term1 + term2 + term3

# Find max residual inside star
residual_inside = residual[grid.num_ghosts:surface_idx]
max_residual = np.max(np.abs(residual_inside)) if len(residual_inside) > 0 else 0.0
max_residual_idx = np.argmax(np.abs(residual_inside)) + grid.num_ghosts if len(residual_inside) > 0 else 0

print(f"\nHydrostatic equilibrium residual:")
print(f"  max(|residual|) = {max_residual:.6e} at r={grid.r[max_residual_idx]:.2f}")
print(f"  mean(|residual|) inside star = {np.mean(np.abs(residual_inside)):.6e}")

if max_residual < 1e-6:
    print("\n✓ Hydrostatic equilibrium satisfied numerically (residual < 1e-6)")
else:
    print(f"\n⚠️  Hydrostatic equilibrium residual is large: {max_residual:.3e}")
    print("  This could generate initial momentum/velocity")

# Check specific locations
print("\n" + "=" * 80)
print("SPECIFIC LOCATIONS")
print("=" * 80)

locations = [
    ("Center", grid.num_ghosts),
    ("Interior", grid.num_ghosts + 50),
    ("Mid-star", surface_idx // 2),
    ("Near surface", surface_idx - 5),
    ("Surface", surface_idx),
    ("Atmosphere", surface_idx + 10)
]

print(f"\n{'Location':<15} {'r':>8} {'ρ₀':>12} {'v^r':>12} {'S^r':>12} {'residual':>12}")
print("-" * 80)
for name, idx in locations:
    if idx < N:
        print(f"{name:<15} {grid.r[idx]:8.3f} {rho0[idx]:12.3e} {vr[idx]:+12.3e} "
              f"{Sr[idx]:+12.3e} {residual[idx]:+12.3e}")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

if vr_zero and Sr_zero and max_residual < 1e-6:
    print("\n✓ Initial conditions appear CORRECT:")
    print("  - v^r = 0 (machine precision)")
    print("  - S^r = 0 (machine precision)")
    print("  - Hydrostatic equilibrium satisfied")
    print("\n→ Bug must be in TIME EVOLUTION (RHS computation or time integrator)")
    exit_code = 0
elif not vr_zero or not Sr_zero:
    print("\n⚠️  Initial conditions have NON-ZERO velocity/momentum!")
    print("  This explains why velocity grows in the simulation.")
    print("\n→ Bug is in TOV INITIAL DATA creation")
    print("  Check interpolation procedure in tov_initial_data_interpolated.py")
    exit_code = 1
elif max_residual > 1e-6:
    print("\n⚠️  Hydrostatic equilibrium NOT satisfied numerically")
    print("  This will generate velocity/momentum over time.")
    print("\n→ Bug could be in:")
    print("  1. TOV solution accuracy")
    print("  2. Interpolation procedure")
    print("  3. BSSN metric construction from TOV")
    exit_code = 1
else:
    print("\n⚠️  Unclear situation - needs further investigation")
    exit_code = 1

sys.exit(exit_code)
