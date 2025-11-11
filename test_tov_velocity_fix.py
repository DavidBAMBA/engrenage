"""
Quick test to verify TOV velocity growth is fixed after bug corrections.

Tests the radial velocity evolution at TOV stellar surface (r ≈ 9.55)
for a few timesteps to see if it remains stable or grows spuriously.
"""

import numpy as np
import sys
import os

# Add repository root to path
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from source.core.grid import Grid
from source.matter.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import PolytropicEOS
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.initialdata.tovinitialconditions import set_tov_initial_conditions_bssn
from source.bssn.bssnvars import BSSNVars

print("=" * 80)
print("Testing TOV Velocity Stability After Bug Fixes")
print("=" * 80)

# Create grid (1D radial)
N = 256
r_max = 20.0
grid = Grid(N=N, r_max=r_max, num_ghosts=2, mode='bssn+matter')

print(f"\nGrid: N={N}, r_max={r_max}")
print(f"Resolution: dr = {grid.dr:.4f}")

# Create EOS (polytropic K=100, Gamma=2)
eos = PolytropicEOS(K=100.0, gamma=2.0)

# Create matter (perfect fluid)
matter = PerfectFluid(eos=eos)

# Create background
background = FlatSphericalBackground(grid)

# Set initial conditions
print("\nSetting TOV initial conditions...")
current_state, current_time = set_tov_initial_conditions_bssn(
    grid, matter,
    central_density=1.28e-3,
    polytropic_K=100.0,
    polytropic_Gamma=2.0
)

# Extract BSSN and matter variables
bssn_vars = BSSNVars(current_state, grid)

# Extract matter variables manually
matter_vars = {
    'D': matter.D,
    'Sr': matter.Sr,
    'tau': matter.tau,
    'rho0': None,  # Will be computed
    'vr': None,
    'p': None
}

# Find stellar surface (where density drops significantly)
rho0_center = matter_vars['rho0'][grid.num_ghosts]
threshold = 0.01 * rho0_center  # 1% of central density
surface_idx = None
for i in range(grid.num_ghosts, N - grid.num_ghosts):
    if matter_vars['rho0'][i] < threshold:
        surface_idx = i
        break

if surface_idx is None:
    print("\n⚠️  Could not find stellar surface!")
    sys.exit(1)

r_surface = grid.r[surface_idx]
print(f"\nStellar surface at i={surface_idx}, r={r_surface:.2f}")
print(f"  ρ₀ = {matter_vars['rho0'][surface_idx]:.3e}")
print(f"  p  = {matter_vars['p'][surface_idx]:.3e}")
print(f"  v^r = {matter_vars['vr'][surface_idx]:.3e}")

# Extract primitives at surface
from source.matter.hydro.cons2prim import Cons2PrimSolver

cons2prim = Cons2PrimSolver(eos=eos)

# Prepare geometry
alpha = bssn_vars.lapse
beta_r = np.zeros(N)
if hasattr(bssn_vars, 'shift_U') and bssn_vars.shift_U is not None:
    shift_arr = np.asarray(bssn_vars.shift_U)
    if shift_arr.ndim >= 2:
        beta_r = shift_arr[:, 0]
    elif shift_arr.ndim == 1:
        beta_r = shift_arr

# Get metric at surface
from source.bssn.tensoralgebra import get_bar_gamma_LL
bar_gamma = get_bar_gamma_LL(grid.r, bssn_vars.h_LL, background)
em4phi = np.exp(-4.0 * bssn_vars.phi)
gamma_rr = em4phi * bar_gamma[:, 0, 0]

metric = (alpha, beta_r, gamma_rr)

# Time evolution test
print("\n" + "=" * 80)
print("TESTING VELOCITY EVOLUTION (first 10 steps)")
print("=" * 80)

dt = 0.01  # Small timestep
max_steps = 10

# Store velocity history
velocity_history = []
time_history = []

for step in range(max_steps + 1):
    # Get primitives
    U = (matter_vars['D'], matter_vars['Sr'], matter_vars['tau'])

    primitives = cons2prim.convert(
        U=U,
        metric=metric,
        p_guess=matter_vars['p'],
        apply_conservative_floors=False
    )

    vr_surface = primitives['vr'][surface_idx]
    velocity_history.append(vr_surface)
    time_history.append(current_time)

    if step % 2 == 0:
        print(f"Step {step:3d}, t={current_time:6.3f}: v^r(surface) = {vr_surface:+.6e}")

    if step == max_steps:
        break

    # Compute RHS using matter.get_matter_rhs
    # First set matter variables
    matter.D = matter_vars['D']
    matter.Sr = matter_vars['Sr']
    matter.tau = matter_vars['tau']
    matter.pressure_cache = primitives['p']
    matter.matter_vars_set = True

    # Get BSSN derivatives (finite differences)
    from source.core.derivatives import Derivatives
    derivatives = Derivatives(grid, background)

    bssn_d1 = derivatives.get_bssn_d1(current_state, matter)

    # Compute matter RHS
    matter_rhs = matter.get_matter_rhs(grid.r, bssn_vars, bssn_d1, background)

    # Simple forward Euler step
    matter_vars['D'] += dt * matter_rhs[0, :]
    matter_vars['Sr'] += dt * matter_rhs[1, :]
    matter_vars['tau'] += dt * matter_rhs[2, :]

    current_time += dt

# Analysis
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

v_initial = velocity_history[0]
v_final = velocity_history[-1]
v_change = v_final - v_initial
v_relative_change = abs(v_change / (abs(v_initial) + 1e-30))

print(f"\nInitial velocity: {v_initial:+.6e}")
print(f"Final velocity:   {v_final:+.6e}")
print(f"Absolute change:  {v_change:+.6e}")
print(f"Relative change:  {v_relative_change:.2%}")

# Check if velocity is stable
tolerance = 0.1  # 10% change is acceptable for such a short test
if v_relative_change < tolerance:
    print(f"\n✓ VELOCITY IS STABLE (change < {tolerance:.0%})")
    print("✓ Bug fixes appear to be working!")
    exit_code = 0
else:
    print(f"\n⚠️  VELOCITY STILL GROWING (change > {tolerance:.0%})")
    print("⚠️  There may be additional bugs to investigate")
    exit_code = 1

# Plot velocity evolution if matplotlib is available
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_history, velocity_history, 'o-', linewidth=2, markersize=6)
    ax.axhline(v_initial, color='r', linestyle='--', alpha=0.5, label=f'Initial: {v_initial:.3e}')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('v^r at surface', fontsize=12)
    ax.set_title(f'Radial Velocity at TOV Surface (r={r_surface:.2f})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    outfile = 'tov_velocity_stability_test.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {outfile}")

except ImportError:
    print("\n(matplotlib not available, skipping plot)")

sys.exit(exit_code)
