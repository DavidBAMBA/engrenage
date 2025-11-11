#!/usr/bin/env python3
"""
Investigate why the Ricci scalar R̄ is 10^58 at the center of TOV star.

This tests the Ricci tensor computation at r=0 to find the source of the
catastrophic constraint violation.
"""

import numpy as np
import sys

sys.path.insert(0, '/home/yo/repositories/engrenage')

from source.core.grid import Grid
from source.core.spacing import LinearSpacing, SpacingExtent
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.bssn.bssnstatevariables import NUM_BSSN_VARS
from source.bssn.bssnvars import BSSNVars
from source.matter.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.atmosphere import AtmosphereParams
from examples.TOV.tov_initial_data_interpolated import create_initial_data_interpolated
from examples.TOV.tov_solver import TOVSolver

from source.bssn.tensoralgebra import (
    get_bar_gamma_UU, get_bar_gamma_LL,
    get_tensor_connections, get_bar_christoffel,
    get_bar_ricci_tensor, get_trace
)

def main():
    print("="*70)
    print("INVESTIGATING RICCI SCALAR AT TOV CENTER")
    print("="*70)

    # Setup TOV
    gamma = 2.0
    K_val = 1.0
    rho_central = 0.2

    solver = TOVSolver(K=K_val, Gamma=gamma)
    r_probe = np.linspace(0.0, 200.0, 4096)
    tov_probe = solver.solve(rho_central, r_max=r_probe[-1])
    r_max = 2.0 * tov_probe['R']

    tov_solution = solver.solve(rho_central, r_max=r_max)
    print(f"\nTOV solution:")
    print(f"  M = {tov_solution['M_star']:.6f}")
    print(f"  R = {tov_solution['R']:.6f}")

    # Setup grid
    dr = 0.04
    spacing = LinearSpacing(int(r_max/dr), r_max, SpacingExtent.HALF)
    r = spacing[0]

    eos = IdealGasEOS(gamma=gamma)
    atmosphere = AtmosphereParams()
    state_vec = StateVector(PerfectFluid(eos=eos, spacetime_mode='dynamic', atmosphere=atmosphere))
    grid = Grid(spacing, state_vec)
    background = FlatSphericalBackground(r)

    # Create initial data
    state = create_initial_data_interpolated(
        tov_solution, grid, background, eos,
        atmosphere=atmosphere, interp_order=11
    )

    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state[:NUM_BSSN_VARS, :])

    # Get derivatives
    d1 = grid.get_d1_metric_quantities(state)
    d2 = grid.get_d2_metric_quantities(state)

    # Compute Ricci tensor components
    bar_gamma_UU = get_bar_gamma_UU(grid.r, bssn_vars.h_LL, background)
    bar_gamma_LL = get_bar_gamma_LL(grid.r, bssn_vars.h_LL, background)
    Delta_U, Delta_ULL, Delta_LLL = get_tensor_connections(grid.r, bssn_vars.h_LL, d1.h_LL, background)
    bar_chris = get_bar_christoffel(grid.r, Delta_ULL, background)

    bar_Rij = get_bar_ricci_tensor(
        grid.r, bssn_vars.h_LL, d1.h_LL, d2.h_LL,
        bssn_vars.lambda_U, d1.lambda_U,
        Delta_U, Delta_ULL, Delta_LLL,
        bar_gamma_UU, bar_gamma_LL, background
    )

    bar_R = get_trace(bar_Rij, bar_gamma_UU)

    print(f"\n" + "="*70)
    print("ANALYSIS AT CENTER (r=0, index=0)")
    print("="*70)

    idx = 0
    print(f"\nBSSN variables at center:")
    print(f"  φ = {bssn_vars.phi[idx]:.6e}")
    print(f"  K = {bssn_vars.K[idx]:.6e}")
    print(f"  h_rr = {bssn_vars.h_LL[idx, 0, 0]:.6e}")
    print(f"  h_θθ = {bssn_vars.h_LL[idx, 1, 1]:.6e}")
    print(f"  λ^r = {bssn_vars.lambda_U[idx, 0]:.6e}")

    print(f"\nConformal metric at center:")
    print(f"  γ̄_rr = {bar_gamma_LL[idx, 0, 0]:.6e}")
    print(f"  γ̄_θθ = {bar_gamma_LL[idx, 1, 1]:.6e}")
    print(f"  γ̄^rr = {bar_gamma_UU[idx, 0, 0]:.6e}")
    print(f"  γ̄^θθ = {bar_gamma_UU[idx, 1, 1]:.6e}")

    print(f"\nRicci tensor components at center:")
    print(f"  R̄_rr = {bar_Rij[idx, 0, 0]:.6e}")
    print(f"  R̄_θθ = {bar_Rij[idx, 1, 1]:.6e}")
    print(f"  R̄_φφ = {bar_Rij[idx, 2, 2]:.6e}")

    print(f"\nRicci scalar at center:")
    print(f"  R̄ = γ̄^ij R̄_ij = {bar_R[idx]:.6e}")

    # Check a few points along the radius
    print(f"\n" + "="*70)
    print("RADIAL PROFILE OF R̄")
    print("="*70)
    print(f"\n{'i':>4s} {'r':>10s} {'R̄':>15s} {'|R̄|':>15s}")
    print("-"*50)
    for i in [0, 1, 2, 5, 10, 20]:
        if i < len(grid.r):
            print(f"{i:4d} {grid.r[i]:10.6f} {bar_R[i]:+15.6e} {abs(bar_R[i]):15.6e}")

    # Check if problem is r=0 specific
    if abs(bar_R[0]) > abs(bar_R[1]) * 100:
        print(f"\n⚠️  SINGULARITY AT r=0: R̄[0]/R̄[1] = {abs(bar_R[0]/bar_R[1]):.2e}")
        print(f"    This suggests a coordinate singularity or interpolation issue at the center")
    else:
        print(f"\n✓ R̄ is smooth at center (R̄[0]/R̄[1] = {abs(bar_R[0]/bar_R[1]):.2e})")

    # Check derivatives
    print(f"\n" + "="*70)
    print("FIRST DERIVATIVES AT CENTER")
    print("="*70)
    print(f"  ∂_r φ = {d1.phi[idx, 0]:.6e}")
    print(f"  ∂_r h_rr = {d1.h_LL[idx, 0, 0, 0]:.6e}")
    print(f"  ∂_r λ^r = {d1.lambda_U[idx, 0, 0]:.6e}")

    print(f"\n" + "="*70)
    print("SECOND DERIVATIVES AT CENTER")
    print("="*70)
    print(f"  ∂_r ∂_r φ = {d2.phi[idx, 0, 0]:.6e}")
    print(f"  ∂_r ∂_r h_rr = {d2.h_LL[idx, 0, 0, 0, 0]:.6e}")

    # Check Christoffel symbols
    print(f"\n" + "="*70)
    print("CHRISTOFFEL SYMBOLS AT CENTER")
    print("="*70)
    print(f"  Γ̄^r_rr = {bar_chris[idx, 0, 0, 0]:.6e}")
    print(f"  Γ̄^r_θθ = {bar_chris[idx, 0, 1, 1]:.6e}")
    print(f"  Γ̄^θ_rθ = {bar_chris[idx, 1, 0, 1]:.6e}")

    # Diagnostic summary
    print(f"\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    if abs(bar_R[idx]) > 1e10:
        print(f"❌ CATASTROPHIC: R̄ = {bar_R[idx]:.3e} >> 1")
        print(f"   Likely causes:")
        print(f"   1. Division by zero or near-zero in spherical coordinates at r=0")
        print(f"   2. Interpolation artifacts at the center")
        print(f"   3. Bug in get_bar_ricci_tensor calculation")
        print(f"   4. Incorrect handling of coordinate singularity at origin")
        return 1
    else:
        print(f"✓ R̄ = {bar_R[idx]:.3e} is reasonable")
        return 0


if __name__ == "__main__":
    sys.exit(main())
