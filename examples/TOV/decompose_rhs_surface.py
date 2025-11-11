#!/usr/bin/env python3
"""
Decompose RHS at TOV surface to identify source of spurious momentum.

We know:
- Surface at i=300, r≈9.59
- dS_r/dt = 7.04e-09 at t=0
- This grows linearly → velocity increases monotonically

Goal: Identify which term causes this:
  dS_r/dt = -∂_r(F^r_r) + Source_terms + Connection_terms
"""

import numpy as np
import sys
import os

# Add engrenage to path
sys.path.insert(0, '/home/yo/repositories/engrenage')
os.chdir('/home/yo/repositories/engrenage/examples/TOV')

# Core imports
from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground, i_r

# BSSN
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS, idx_phi, idx_lapse

# Hydro
from source.matter.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import PolytropicEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.atmosphere import AtmosphereParams

# TOV
from tov_solver import TOVSolver
import tov_initial_data_interpolated as tov_id


def setup():
    """Setup grid, EOS, hydro system, and initial data."""
    # Parameters
    N = 500
    r_max = 16.0
    K, Gamma = 100.0, 2.0
    central_rho = 1.28e-3

    # Grid
    spacing = LinearSpacing(N, r_max)

    # EOS
    eos = PolytropicEOS(K, Gamma)

    # Atmosphere
    atmosphere = AtmosphereParams(
        rho_floor=1e-10 * central_rho,
        p_floor=1e-10,
        v_max=0.9999
    )

    # Hydro
    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere=atmosphere,
        reconstructor=create_reconstruction("mp5"),
        riemann_solver=HLLRiemannSolver(atmosphere=atmosphere)
    )

    # Grid
    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    # Generate TOV initial data
    print("Generating TOV initial data...")
    tov = TOVSolver(K, Gamma)
    tov_sol = tov.solve(central_rho)

    initial_state = tov_id.create_initial_data_interpolated(
        tov_sol, grid, background, eos,
        atmosphere=atmosphere,
        interp_order=11
    )

    return grid, background, hydro, initial_state, tov_sol, eos


def analyze_rhs_decomposition(grid, background, hydro, state_2d):
    """Decompose RHS into individual terms at surface."""

    print("\n" + "=" * 80)
    print("RHS DECOMPOSITION AT SURFACE")
    print("=" * 80)

    # Setup BSSN
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
    bssn_d1 = grid.get_d1_metric_quantities(state_2d)

    # Set hydro state
    hydro.set_matter_vars(state_2d, bssn_vars, grid)

    # Get full RHS
    rhs_D, rhs_Sr, rhs_tau = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1, background)

    # Find surface
    prim = hydro._get_primitives(bssn_vars, grid.r)
    interior = prim['rho0'] > 1e-6
    i_surf = np.where(interior)[0][-1] if np.any(interior) else 250

    print(f"\nSurface location: i={i_surf}, r={grid.r[i_surf]:.6f}")
    print(f"Total dS_r/dt at surface: {rhs_Sr[i_surf]:.6e}")

    # Window around surface
    i_start = max(NUM_GHOSTS, i_surf - 5)
    i_end = min(grid.N - NUM_GHOSTS, i_surf + 6)

    print(f"\nRHS in surface region:")
    print(f"{'i':<6} {'r':<10} {'rho0':<12} {'P':<12} {'dS_r/dt':<14}")
    print("-" * 70)
    for i in range(i_start, i_end):
        print(f"{i:<6} {grid.r[i]:<10.6f} {prim['rho0'][i]:<12.4e} "
              f"{prim['p'][i]:<12.4e} {rhs_Sr[i]:<14.6e}")

    # Now try to access internal components
    print("\n" + "-" * 80)
    print("ATTEMPTING TO DECOMPOSE RHS TERMS")
    print("-" * 80)

    # The RHS comes from get_matter_rhs which calls GRHDEquations internally
    # We need to replicate the calculation to see individual terms

    try:
        # Access the GRHDEquations object
        grhd = hydro.grhd_equations

        # Get primitives
        geometry_obj = grhd._build_geometry(bssn_vars, grid.r, background)

        print("\nChecking if we can access flux divergence...")

        # The challenge is that get_matter_rhs combines everything
        # Let's at least check the fluxes at interfaces

        print("\nFlux analysis at surface interfaces:")
        print(f"Interface between i={i_surf} and i={i_surf+1}")

        # Get conservatives at surface
        D_surf = state_2d[NUM_BSSN_VARS + 0, i_surf]
        Sr_surf = state_2d[NUM_BSSN_VARS + 1, i_surf]
        tau_surf = state_2d[NUM_BSSN_VARS + 2, i_surf]

        print(f"  D_tilde  = {D_surf:.6e}")
        print(f"  Sr_tilde = {Sr_surf:.6e}")
        print(f"  tau_tilde = {tau_surf:.6e}")

        # Get metric at surface
        phi_surf = state_2d[idx_phi, i_surf]
        alpha_surf = state_2d[idx_lapse, i_surf]
        e6phi_surf = np.exp(6.0 * phi_surf)

        print(f"  φ = {phi_surf:.6e}")
        print(f"  α = {alpha_surf:.6f}")
        print(f"  e^(6φ) = {e6phi_surf:.6f}")

        # De-densitize to get physical conservatives
        D_phys = D_surf / e6phi_surf
        Sr_phys = Sr_surf / e6phi_surf

        print(f"\nPhysical (non-densitized) conservatives:")
        print(f"  D = {D_phys:.6e}")
        print(f"  Sr = {Sr_phys:.6e}")

    except Exception as e:
        print(f"Could not decompose: {e}")

    return i_surf, rhs_Sr


def check_flux_balance(grid, background, hydro, state_2d, i_surf):
    """Check if fluxes are balanced at surface."""

    print("\n" + "=" * 80)
    print("FLUX BALANCE CHECK AT SURFACE")
    print("=" * 80)

    # Get primitives
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    prim = hydro._get_primitives(bssn_vars, grid.r)

    # Check gradient of pressure at surface
    i = i_surf
    if i > NUM_GHOSTS and i < grid.N - NUM_GHOSTS - 1:
        dP_dr = (prim['p'][i+1] - prim['p'][i-1]) / (grid.r[i+1] - grid.r[i-1])

        print(f"\nAt surface (i={i}, r={grid.r[i]:.6f}):")
        print(f"  P(i-1) = {prim['p'][i-1]:.6e}")
        print(f"  P(i)   = {prim['p'][i]:.6e}")
        print(f"  P(i+1) = {prim['p'][i+1]:.6e}")
        print(f"  dP/dr  = {dP_dr:.6e}")

        # For hydrostatic equilibrium, we expect:
        # dP/dr + (ρ h) * (dα/dr / α) ≈ 0

        alpha_i = state_2d[idx_lapse, i]
        dalpha_dr = (state_2d[idx_lapse, i+1] - state_2d[idx_lapse, i-1]) / (grid.r[i+1] - grid.r[i-1])

        rho0_i = prim['rho0'][i]
        P_i = prim['p'][i]
        eps_i = prim['eps'][i] if 'eps' in prim else hydro.eos.eps_from_rho_p(rho0_i, P_i)
        h_i = 1.0 + eps_i + P_i / max(rho0_i, 1e-30)

        metric_force = (rho0_i * h_i) * (dalpha_dr / alpha_i)
        imbalance = dP_dr + metric_force

        print(f"\nHydrostatic balance:")
        print(f"  ρ₀ h = {rho0_i * h_i:.6e}")
        print(f"  dα/dr / α = {dalpha_dr / alpha_i:.6e}")
        print(f"  Metric force = {metric_force:.6e}")
        print(f"  Imbalance = dP/dr + force = {imbalance:.6e}")
        print(f"  Relative imbalance = {imbalance / abs(dP_dr) if abs(dP_dr) > 1e-30 else 0:.6e}")


def main():
    print("=" * 80)
    print("SURFACE MOMENTUM RHS DECOMPOSITION")
    print("=" * 80)

    # Setup
    grid, background, hydro, state_2d, tov_sol, eos = setup()

    # Analyze RHS
    i_surf, rhs_Sr = analyze_rhs_decomposition(grid, background, hydro, state_2d)

    # Check flux balance
    check_flux_balance(grid, background, hydro, state_2d, i_surf)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nSurface location: i={i_surf}, r={grid.r[i_surf]:.6f}")
    print(f"dS_r/dt at surface: {rhs_Sr[i_surf]:.6e}")
    print("\nThis non-zero RHS accumulates linearly, causing velocity growth.")
    print("\nNext: Need to identify if this comes from:")
    print("  1. Flux divergence -∂_r(F^r_r)")
    print("  2. Source terms")
    print("  3. Connection terms (Christoffel symbols)")
    print("=" * 80)


if __name__ == "__main__":
    main()
