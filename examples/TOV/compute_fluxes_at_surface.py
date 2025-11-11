#!/usr/bin/env python3
"""
Manually compute fluxes at interfaces near the surface to diagnose
where the spurious momentum comes from.

Key insight: If v^r = 0 everywhere, then physical fluxes should be:
  F_D = 0  (no mass flux)
  F_Sr = P  (only pressure)
  F_tau = 0 (no energy flux)

Any deviation from this indicates a problem in reconstruction or Riemann solver.
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
    N = 500
    r_max = 16.0
    K, Gamma = 100.0, 2.0
    central_rho = 1.28e-3

    spacing = LinearSpacing(N, r_max)
    eos = PolytropicEOS(K, Gamma)

    atmosphere = AtmosphereParams(
        rho_floor=1e-10 * central_rho,
        p_floor=1e-10,
        v_max=0.9999
    )

    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere=atmosphere,
        reconstructor=create_reconstruction("mp5"),
        riemann_solver=HLLRiemannSolver(atmosphere=atmosphere)
    )

    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    print("Generating TOV initial data...")
    tov = TOVSolver(K, Gamma)
    tov_sol = tov.solve(central_rho)

    initial_state = tov_id.create_initial_data_interpolated(
        tov_sol, grid, background, eos,
        atmosphere=atmosphere,
        interp_order=11
    )

    return grid, background, hydro, initial_state, eos


def compute_flux_at_interface(i_face, grid, hydro, state_2d, eos):
    """
    Compute flux at interface i_face (between cells i_face-1 and i_face).

    This manually calls reconstruction and Riemann solver to see exactly
    what fluxes are being computed.
    """
    # Get conservatives and primitives on both sides
    iL = i_face - 1
    iR = i_face

    # BSSN quantities
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])

    # Conservatives (densitized)
    D_tilde_L = state_2d[NUM_BSSN_VARS + 0, iL]
    Sr_tilde_L = state_2d[NUM_BSSN_VARS + 1, iL]
    tau_tilde_L = state_2d[NUM_BSSN_VARS + 2, iL]

    D_tilde_R = state_2d[NUM_BSSN_VARS + 0, iR]
    Sr_tilde_R = state_2d[NUM_BSSN_VARS + 1, iR]
    tau_tilde_R = state_2d[NUM_BSSN_VARS + 2, iR]

    # Metric quantities at cells
    phi_L = state_2d[idx_phi, iL]
    phi_R = state_2d[idx_phi, iR]
    alpha_L = state_2d[idx_lapse, iL]
    alpha_R = state_2d[idx_lapse, iR]

    e6phi_L = np.exp(6.0 * phi_L)
    e6phi_R = np.exp(6.0 * phi_R)
    gamma_rr_L = np.exp(4.0 * phi_L)
    gamma_rr_R = np.exp(4.0 * phi_R)

    # De-densitize conservatives
    D_L = D_tilde_L / e6phi_L
    Sr_L = Sr_tilde_L / e6phi_L
    tau_L = tau_tilde_L / e6phi_L

    D_R = D_tilde_R / e6phi_R
    Sr_R = Sr_tilde_R / e6phi_R
    tau_R = tau_tilde_R / e6phi_R

    # Primitives (assuming cons2prim was already done)
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    prim = hydro._get_primitives(bssn_vars, grid.r)

    rho0_L = prim['rho0'][iL]
    vr_L = prim['vr'][iL]
    p_L = prim['p'][iL]

    rho0_R = prim['rho0'][iR]
    vr_R = prim['vr'][iR]
    p_R = prim['p'][iR]

    # Metric at interface (average)
    phi_f = 0.5 * (phi_L + phi_R)
    alpha_f = 0.5 * (alpha_L + alpha_R)
    gamma_rr_f = np.exp(4.0 * phi_f)
    e6phi_f = np.exp(6.0 * phi_f)

    print(f"\n{'='*80}")
    print(f"INTERFACE i={i_face} (between cells {iL} and {iR})")
    print(f"{'='*80}")
    print(f"\nLeft cell (i={iL}, r={grid.r[iL]:.6f}):")
    print(f"  Primitives: ρ₀={rho0_L:.6e}, v^r={vr_L:.6e}, P={p_L:.6e}")
    print(f"  Conservatives: D={D_L:.6e}, S_r={Sr_L:.6e}, tau={tau_L:.6e}")
    print(f"  Metric: α={alpha_L:.6f}, γ_rr={gamma_rr_L:.6f}")

    print(f"\nRight cell (i={iR}, r={grid.r[iR]:.6f}):")
    print(f"  Primitives: ρ₀={rho0_R:.6e}, v^r={vr_R:.6e}, P={p_R:.6e}")
    print(f"  Conservatives: D={D_R:.6e}, S_r={Sr_R:.6e}, tau={tau_R:.6e}")
    print(f"  Metric: α={alpha_R:.6f}, γ_rr={gamma_rr_R:.6f}")

    print(f"\nInterface metrics:")
    print(f"  α_f={alpha_f:.6f}, γ_rr_f={gamma_rr_f:.6f}, e^(6φ)_f={e6phi_f:.6f}")

    # Call Riemann solver directly (no reconstruction, use cell-center values)
    riemann = hydro.riemann_solver

    UL = np.array([D_L, Sr_L, tau_L])
    UR = np.array([D_R, Sr_R, tau_R])
    primL = (rho0_L, vr_L, p_L)
    primR = (rho0_R, vr_R, p_R)

    # Physical flux (non-densitized)
    F_phys = riemann.solve(UL, UR, primL, primR, gamma_rr_f, alpha_f, 0.0, eos)

    # Densitized flux
    F_dens = alpha_f * e6phi_f * F_phys

    print(f"\nPhysical flux (non-densitized):")
    print(f"  F_D = {F_phys[0]:.6e}")
    print(f"  F_Sr = {F_phys[1]:.6e}  ← Should be ≈ P if v=0")
    print(f"  F_tau = {F_phys[2]:.6e}")

    print(f"\nDensitized flux F̃ = α e^(6φ) F^phys:")
    print(f"  F̃_D = {F_dens[0]:.6e}")
    print(f"  F̃_Sr = {F_dens[1]:.6e}")
    print(f"  F̃_tau = {F_dens[2]:.6e}")

    print(f"\nExpected F_Sr (for v=0): P_avg ≈ {0.5*(p_L + p_R):.6e}")
    print(f"Deviation: {F_phys[1] - 0.5*(p_L + p_R):.6e}")

    return {
        'i_face': i_face,
        'F_phys': F_phys,
        'F_dens': F_dens,
        'p_avg': 0.5 * (p_L + p_R),
    }


def main():
    print("=" * 80)
    print("FLUX COMPUTATION AT SURFACE INTERFACES")
    print("=" * 80)

    grid, background, hydro, state_2d, eos = setup()

    # Find surface
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    prim = hydro._get_primitives(bssn_vars, grid.r)

    interior = prim['rho0'] > 1e-6
    i_surf = np.where(interior)[0][-1] if np.any(interior) else 250

    print(f"\nSurface at i={i_surf}, r={grid.r[i_surf]:.6f}")

    # Compute fluxes at interfaces around the surface
    # Interface i_face is between cells (i_face-1) and i_face

    # i_face = 299: between cells 298 and 299 (both interior)
    # i_face = 300: between cells 299 (interior) and 300 (atmosphere) ← CRITICAL!
    # i_face = 301: between cells 300 and 301 (both atmosphere)

    flux_299 = compute_flux_at_interface(299, grid, hydro, state_2d, eos)
    flux_300 = compute_flux_at_interface(300, grid, hydro, state_2d, eos)
    flux_301 = compute_flux_at_interface(301, grid, hydro, state_2d, eos)

    # Compute flux divergence for cells 299 and 300
    dr = grid.r[1] - grid.r[0]

    print(f"\n{'='*80}")
    print("FLUX DIVERGENCE")
    print(f"{'='*80}")

    # Cell 299 (surface)
    flux_div_299 = -(flux_300['F_dens'][1] - flux_299['F_dens'][1]) / dr
    print(f"\nCell 299 (surface):")
    print(f"  F̃_Sr at i-1/2 (interface 299) = {flux_299['F_dens'][1]:.6e}")
    print(f"  F̃_Sr at i+1/2 (interface 300) = {flux_300['F_dens'][1]:.6e}")
    print(f"  Flux divergence -∂_r(F̃_Sr) = {flux_div_299:.6e}")

    # Cell 300 (atmosphere)
    flux_div_300 = -(flux_301['F_dens'][1] - flux_300['F_dens'][1]) / dr
    print(f"\nCell 300 (atmosphere):")
    print(f"  F̃_Sr at i-1/2 (interface 300) = {flux_300['F_dens'][1]:.6e}")
    print(f"  F̃_Sr at i+1/2 (interface 301) = {flux_301['F_dens'][1]:.6e}")
    print(f"  Flux divergence -∂_r(F̃_Sr) = {flux_div_300:.6e}")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("\nKey findings:")
    print(f"  Interface 300 (stellar surface) has:")
    print(f"    F_Sr^phys = {flux_300['F_phys'][1]:.6e}")
    print(f"    Expected (P_avg) = {flux_300['p_avg']:.6e}")
    print(f"    Deviation = {flux_300['F_phys'][1] - flux_300['p_avg']:.6e}")
    print(f"\n  This flux imbalance creates the spurious momentum!")
    print("=" * 80)


if __name__ == "__main__":
    main()
