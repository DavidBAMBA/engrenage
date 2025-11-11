#!/usr/bin/env python3
"""
Decompose the momentum RHS at the surface into its three contributions:

    dS_r/dt = -∂_r(F^r_r) + Source_terms + Connection_terms

This will identify which term is responsible for the spurious 7.04e-09 momentum.
"""

import numpy as np
import sys
import os

sys.path.insert(0, '/home/yo/repositories/engrenage')
os.chdir('/home/yo/repositories/engrenage/examples/TOV')

from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground

from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS, idx_phi, idx_lapse

from source.matter.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import PolytropicEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.atmosphere import AtmosphereParams

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


def compute_rhs_components_at_point(i, grid, background, hydro, state_2d, bssn_vars):
    """
    Compute individual RHS components at a single grid point.

    Returns
    -------
    dict with keys:
        'flux_divergence': -∂_r(F^r_r)
        'source_terms': pressure gradient + metric force
        'connection_terms': Christoffel symbol terms
        'total_rhs': sum of all terms (should match hydro.get_matter_rhs)
    """
    # Get primitives
    prim = hydro._get_primitives(bssn_vars, grid.r)

    # Get metric quantities at cell center
    phi_i = state_2d[idx_phi, i]
    alpha_i = state_2d[idx_lapse, i]
    gamma_rr_i = np.exp(4.0 * phi_i)
    e6phi_i = np.exp(6.0 * phi_i)

    # Get conservatives at cell center
    D_tilde_i = state_2d[NUM_BSSN_VARS + 0, i]
    Sr_tilde_i = state_2d[NUM_BSSN_VARS + 1, i]
    tau_tilde_i = state_2d[NUM_BSSN_VARS + 2, i]

    # De-densitize
    D_i = D_tilde_i / e6phi_i
    Sr_i = Sr_tilde_i / e6phi_i
    tau_i = tau_tilde_i / e6phi_i

    result = {
        'i': i,
        'r': grid.r[i],
        'rho0': prim['rho0'][i],
        'vr': prim['vr'][i],
        'p': prim['p'][i],
        'alpha': alpha_i,
        'phi': phi_i,
        'gamma_rr': gamma_rr_i,
        'e6phi': e6phi_i,
        'D': D_i,
        'Sr': Sr_i,
        'tau': tau_i,
    }

    # Compute fluxes at interfaces i-1/2 and i+1/2
    # This requires reconstruction + Riemann solver
    # For now, let's get the full RHS and try to extract components

    return result


def analyze_surface_rhs(grid, background, hydro, state_2d):
    """Analyze RHS components at and near the surface."""

    print("\n" + "=" * 80)
    print("DETAILED RHS DECOMPOSITION AT SURFACE")
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

    # Analyze a few points around the surface
    print(f"\n{'='*80}")
    print("RHS COMPONENTS NEAR SURFACE")
    print(f"{'='*80}\n")

    for i in range(max(NUM_GHOSTS, i_surf-2), min(grid.N-NUM_GHOSTS, i_surf+3)):
        info = compute_rhs_components_at_point(i, grid, background, hydro, state_2d, bssn_vars)

        marker = " ← SURFACE" if i == i_surf else ""
        marker = marker + " ← ATMOSPHERE" if i == i_surf + 1 else marker

        print(f"i={i} (r={info['r']:.6f}){marker}")
        print(f"  Primitives: ρ₀={info['rho0']:.6e}, v^r={info['vr']:.6e}, P={info['p']:.6e}")
        print(f"  Metric: α={info['alpha']:.6f}, γ_rr={info['gamma_rr']:.6f}")
        print(f"  Conservatives: D={info['D']:.6e}, S_r={info['Sr']:.6e}")
        print(f"  Total RHS: dS_r/dt = {rhs_Sr[i]:.6e}")
        print()

    # Try to access internal components from GRHDEquations
    print(f"{'='*80}")
    print("ATTEMPTING TO ACCESS INTERNAL RHS COMPONENTS")
    print(f"{'='*80}\n")

    # The hydro.get_matter_rhs() calls grhd_equations internally
    # Let's try to access it
    try:
        # Check if we can access the grhd_equations object
        if hasattr(hydro, 'grhd_equations'):
            grhd = hydro.grhd_equations
            print("✓ Found grhd_equations object")

            # Check what methods are available
            print("\nAvailable methods in grhd_equations:")
            methods = [m for m in dir(grhd) if not m.startswith('_')]
            for m in methods[:10]:  # Show first 10
                print(f"  - {m}")

            # The key is to call compute_rhs with components=True or similar
            # Check the signature
            import inspect
            sig = inspect.signature(grhd.compute_rhs)
            print(f"\ncompute_rhs signature: {sig}")

        else:
            print("✗ grhd_equations not accessible")
            print("\nNeed to modify source code to expose RHS components")
            print("Alternative: Manually recompute flux divergence and source terms")

    except Exception as e:
        print(f"Error accessing internals: {e}")

    return i_surf, rhs_Sr


def main():
    print("=" * 80)
    print("DETAILED RHS DECOMPOSITION")
    print("=" * 80)

    # Setup
    grid, background, hydro, state_2d, tov_sol, eos = setup()

    # Analyze RHS
    i_surf, rhs_Sr = analyze_surface_rhs(grid, background, hydro, state_2d)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nSurface location: i={i_surf}, r={grid.r[i_surf]:.6f}")
    print(f"dS_r/dt at surface: {rhs_Sr[i_surf]:.6e}")
    print(f"dS_r/dt at atmosphere (i+1): {rhs_Sr[i_surf+1]:.6e}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\nTo fully decompose the RHS, we need to:")
    print("  1. Compute fluxes at interfaces i±1/2 explicitly")
    print("  2. Compute flux divergence: -∂_r(F^r_r) = -(F_{i+1/2} - F_{i-1/2})/dr")
    print("  3. Extract source terms from hydro code")
    print("  4. Extract connection terms from hydro code")
    print("\nThis requires either:")
    print("  a) Modifying GRHDEquations.compute_rhs() to return components")
    print("  b) Manually recomputing each term using the same formulas")
    print("=" * 80)


if __name__ == "__main__":
    main()
