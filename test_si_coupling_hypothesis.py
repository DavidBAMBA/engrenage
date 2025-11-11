"""
Test if BSSN-matter momentum coupling (S_i term in lambda^i) causes TOV velocity growth.

This script runs TWO simulations:
1. NORMAL: Full BSSN-matter coupling with S_i
2. TEST: Force S_i = 0 to disable momentum coupling to lambda^i

If velocity growth stops when S_i = 0, this confirms the feedback loop hypothesis.
If velocity still grows with S_i = 0, the bug is elsewhere (GRHD equations or ICs).
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
from source.matter.hydro.cons2prim import Cons2PrimSolver
from source.core.derivatives import Derivatives

print("=" * 80)
print("Testing BSSN-Matter Coupling Hypothesis for TOV Velocity Growth")
print("=" * 80)

# Grid setup
N = 256
r_max = 20.0
grid = Grid(N=N, r_max=r_max, num_ghosts=2, mode='bssn+matter')

print(f"\nGrid: N={N}, r_max={r_max}, dr={grid.dr:.4f}")

# EOS
eos = PolytropicEOS(K=100.0, gamma=2.0)

# Background
background = FlatSphericalBackground(grid)

# Time stepping
dt = 0.01
num_steps = 50

def run_simulation(force_Si_zero=False):
    """Run TOV simulation with or without S_i coupling."""

    # Create matter
    matter = PerfectFluid(eos=eos)

    # Monkey-patch get_emtensor if we want to force Si=0
    if force_Si_zero:
        original_get_emtensor = matter.get_emtensor

        def get_emtensor_Si_zero(r, bssn_vars, background):
            emtensor = original_get_emtensor(r, bssn_vars, background)
            # FORCE Si = 0 to disable momentum coupling to BSSN
            emtensor.Si[:] = 0.0
            return emtensor

        matter.get_emtensor = get_emtensor_Si_zero

    # Set initial conditions
    current_state, current_time = set_tov_initial_conditions_bssn(
        grid, matter,
        central_density=1.28e-3,
        polytropic_K=100.0,
        polytropic_Gamma=2.0
    )

    # Extract variables
    bssn_vars = BSSNVars(current_state, grid)
    matter.set_matter_vars(current_state, bssn_vars, grid)

    # Get primitives
    from source.bssn.tensoralgebra import get_bar_gamma_LL
    bar_gamma = get_bar_gamma_LL(grid.r, bssn_vars.h_LL, background)
    em4phi = np.exp(-4.0 * bssn_vars.phi)
    gamma_rr = em4phi * bar_gamma[:, 0, 0]

    alpha = bssn_vars.lapse
    beta_r = np.zeros(N)
    if hasattr(bssn_vars, 'shift_U') and bssn_vars.shift_U is not None:
        shift_arr = np.asarray(bssn_vars.shift_U)
        if shift_arr.ndim >= 2:
            beta_r = shift_arr[:, 0]

    metric = (alpha, beta_r, gamma_rr)
    U = (matter.D, matter.Sr, matter.tau)

    cons2prim = Cons2PrimSolver(eos=eos)
    primitives = cons2prim.convert(U=U, metric=metric, p_guess=None,
                                    apply_conservative_floors=False)

    # Find stellar surface
    rho0 = primitives['rho0']
    rho0_center = rho0[grid.num_ghosts]
    threshold = 0.01 * rho0_center
    surface_idx = None
    for i in range(grid.num_ghosts, N - grid.num_ghosts):
        if rho0[i] < threshold:
            surface_idx = i
            break

    if surface_idx is None:
        print(f"  ⚠️  Could not find stellar surface!")
        return None, None

    r_surface = grid.r[surface_idx]

    # Store velocity at surface over time
    vr_history = []
    Sr_history = []
    time_history = []

    # Time evolution
    derivatives = Derivatives(grid, background)

    for step in range(num_steps + 1):
        # Get current primitives
        U = (matter.D, matter.Sr, matter.tau)
        primitives = cons2prim.convert(U=U, metric=metric, p_guess=primitives['p'],
                                        apply_conservative_floors=False)

        vr_surface = primitives['vr'][surface_idx]
        Sr_surface = matter.Sr[surface_idx]

        vr_history.append(vr_surface)
        Sr_history.append(Sr_surface)
        time_history.append(current_time)

        if step == num_steps:
            break

        # Get BSSN derivatives
        bssn_d1 = derivatives.get_bssn_d1(current_state, matter)

        # Compute matter RHS
        matter.matter_vars_set = True
        matter.pressure_cache = primitives['p']
        matter_rhs = matter.get_matter_rhs(grid.r, bssn_vars, bssn_d1, background)

        # Forward Euler step (matter only, BSSN frozen for this test)
        matter.D += dt * matter_rhs[0, :]
        matter.Sr += dt * matter_rhs[1, :]
        matter.tau += dt * matter_rhs[2, :]

        current_time += dt

    return {
        'r_surface': r_surface,
        'surface_idx': surface_idx,
        'vr_history': np.array(vr_history),
        'Sr_history': np.array(Sr_history),
        'time_history': np.array(time_history),
        'vr_initial': vr_history[0],
        'vr_final': vr_history[-1],
        'Sr_initial': Sr_history[0],
        'Sr_final': Sr_history[-1]
    }

# Run both simulations
print("\n" + "=" * 80)
print("SIMULATION 1: NORMAL (with S_i coupling to BSSN)")
print("=" * 80)
results_normal = run_simulation(force_Si_zero=False)

print("\n" + "=" * 80)
print("SIMULATION 2: TEST (S_i = 0 forced, no coupling)")
print("=" * 80)
results_test = run_simulation(force_Si_zero=True)

# Analysis
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

if results_normal is None or results_test is None:
    print("⚠️  One or both simulations failed")
    sys.exit(1)

r_surface = results_normal['r_surface']
print(f"\nStellar surface at r = {r_surface:.2f}")
print(f"Time integration: {num_steps} steps, dt = {dt}, total time = {num_steps * dt:.2f}")

print("\n" + "-" * 80)
print("NORMAL (with S_i coupling):")
print("-" * 80)
print(f"  Initial: v^r = {results_normal['vr_initial']:+.6e}, S_r = {results_normal['Sr_initial']:+.6e}")
print(f"  Final:   v^r = {results_normal['vr_final']:+.6e}, S_r = {results_normal['Sr_final']:+.6e}")
dv_normal = results_normal['vr_final'] - results_normal['vr_initial']
print(f"  Change:  Δv^r = {dv_normal:+.6e}")

print("\n" + "-" * 80)
print("TEST (S_i = 0 forced):")
print("-" * 80)
print(f"  Initial: v^r = {results_test['vr_initial']:+.6e}, S_r = {results_test['Sr_initial']:+.6e}")
print(f"  Final:   v^r = {results_test['vr_final']:+.6e}, S_r = {results_test['Sr_final']:+.6e}")
dv_test = results_test['vr_final'] - results_test['vr_initial']
print(f"  Change:  Δv^r = {dv_test:+.6e}")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

# Compare velocity growth
ratio = abs(dv_test) / (abs(dv_normal) + 1e-30)

print(f"\nVelocity change ratio: |Δv_test| / |Δv_normal| = {ratio:.3f}")

if ratio < 0.1:
    print("\n✓ HYPOTHESIS CONFIRMED!")
    print("  Forcing S_i = 0 STOPS velocity growth.")
    print("  The BSSN-matter coupling creates a feedback loop:")
    print("    S_i → λ^i evolution → Christoffel symbols → GRHD source terms → S_i grows")
    print("\n  This is NOT a bug - it's physical coupling!")
    print("  However, for a TOV star in equilibrium, S_i should be exactly zero.")
    print("\n  → Check TOV initial conditions: Is v^r = 0 exactly?")
    print("  → Check cons2prim: Is it creating spurious velocities?")
    exit_code = 0
elif ratio > 0.9:
    print("\n⚠️  HYPOTHESIS REJECTED")
    print("  Forcing S_i = 0 does NOT stop velocity growth.")
    print("  The bug is NOT in the BSSN-matter coupling.")
    print("\n  → Bug must be in:")
    print("    1. GRHD momentum equations (despite passing NRPy+ tests)")
    print("    2. TOV initial conditions (non-equilibrium)")
    print("    3. Boundary conditions")
    print("    4. Cons2prim solver")
    exit_code = 1
else:
    print("\n⚠️  PARTIAL EFFECT")
    print("  Forcing S_i = 0 reduces but does not eliminate velocity growth.")
    print("  The coupling contributes but is not the only cause.")
    exit_code = 1

# Plot if matplotlib available
try:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Velocity evolution
    ax1.plot(results_normal['time_history'], results_normal['vr_history'],
             'o-', label='Normal (with $S_i$ coupling)', linewidth=2, markersize=4)
    ax1.plot(results_test['time_history'], results_test['vr_history'],
             's-', label='Test ($S_i = 0$ forced)', linewidth=2, markersize=4)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('$v^r$ at surface', fontsize=12)
    ax1.set_title(f'Radial Velocity at TOV Surface ($r={r_surface:.2f}$)', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Momentum evolution
    ax2.plot(results_normal['time_history'], results_normal['Sr_history'],
             'o-', label='Normal (with $S_i$ coupling)', linewidth=2, markersize=4)
    ax2.plot(results_test['time_history'], results_test['Sr_history'],
             's-', label='Test ($S_i = 0$ forced)', linewidth=2, markersize=4)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('$S_r$ at surface', fontsize=12)
    ax2.set_title('Momentum Density at TOV Surface', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    outfile = 'test_Si_coupling_comparison.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {outfile}")

except ImportError:
    print("\n(matplotlib not available, skipping plot)")

sys.exit(exit_code)
