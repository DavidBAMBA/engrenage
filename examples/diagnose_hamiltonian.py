"""
Diagnostic script to compute Hamiltonian constraint term by term.

This helps identify which term in H is causing the large violation.
"""

import numpy as np
import sys

sys.path.insert(0, '/home/yo/repositories/engrenage')

from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground, i_r, i_t, i_p

from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS
from source.bssn.tensoralgebra import (
    get_bar_gamma_LL, get_bar_gamma_UU, get_bar_A_squared,
    get_tensor_connections, get_bar_christoffel, get_bar_ricci_tensor,
    two_thirds, eight_pi_G
)

from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.atmosphere import AtmosphereParams

from examples.tov_solver import TOVSolver
from examples.tov_initial_data_interpolated import create_initial_data_interpolated


def compute_hamiltonian_terms(state, grid, background, hydro):
    """Compute H term by term for diagnosis."""

    N = grid.N
    r = grid.r

    # Extract BSSN variables
    bssn_vars = BSSNVars(N)
    bssn_vars.set_bssn_vars(state[:NUM_BSSN_VARS, :])

    # Set matter
    hydro.set_matter_vars(state, bssn_vars, grid)

    # Derivatives
    d1 = grid.get_d1_metric_quantities(state)
    d2 = grid.get_d2_metric_quantities(state)

    # Conformal factor
    em4phi = np.exp(-4.0 * bssn_vars.phi)

    # Metric quantities
    bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)

    # Connections
    Delta_U, Delta_ULL, Delta_LLL = get_tensor_connections(r, bssn_vars.h_LL, d1.h_LL, background)
    bar_chris = get_bar_christoffel(r, Delta_ULL, background)

    # Ricci tensor and scalar
    bar_Rij = get_bar_ricci_tensor(
        r, bssn_vars.h_LL, d1.h_LL, d2.h_LL,
        bssn_vars.lambda_U, d1.lambda_U,
        Delta_U, Delta_ULL, Delta_LLL,
        bar_gamma_UU, bar_gamma_LL, background
    )
    bar_R = np.einsum('xij,xij->x', bar_gamma_UU, bar_Rij)

    # Extrinsic curvature terms
    K_squared = bssn_vars.K ** 2
    Asquared = get_bar_A_squared(r, bssn_vars, background)

    # Derivatives of phi
    phi_dD_squared = np.einsum('xij,xi,xj->x', bar_gamma_UU, d1.phi, d1.phi)
    phi_dBarDD_contraction = np.einsum('xij,xij->x', bar_gamma_UU, d2.phi)
    phi_chris_term = np.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, d1.phi)

    # Matter source
    emtensor = hydro.get_emtensor(r, bssn_vars, background)

    # Individual terms in H
    term1 = two_thirds * K_squared
    term2 = -Asquared
    term3 = em4phi * bar_R
    term4 = -8.0 * em4phi * phi_dD_squared
    term5 = -8.0 * em4phi * phi_dBarDD_contraction
    term6 = 8.0 * em4phi * phi_chris_term
    term7 = -2.0 * eight_pi_G * emtensor.rho

    # Total
    H_total = term1 + term2 + term3 + term4 + term5 + term6 + term7

    return {
        'H': H_total,
        'term1_2K2/3': term1,
        'term2_-Asq': term2,
        'term3_barR': term3,
        'term4_-8phidD2': term4,
        'term5_-8phidBarDD': term5,
        'term6_+8phichris': term6,
        'term7_-16piGrho': term7,
        'K': bssn_vars.K,
        'phi': bssn_vars.phi,
        'rho': emtensor.rho,
    }


def main():
    print("="*70)
    print("HAMILTONIAN CONSTRAINT DIAGNOSIS")
    print("="*70)

    # Setup (same as hydro_without_hydro_test.py)
    K = 1.0
    Gamma = 2.0
    rho_central = 0.129285
    num_points = 200

    # Solve TOV
    print("\nSolving TOV...")
    r_max_TOV_solve = 20.0
    tov_num_points_solve = 4000
    tov_dr_solve = r_max_TOV_solve / tov_num_points_solve

    use_iso = False  # USE SCHWARZSCHILD COORDINATES
    tov_solver = TOVSolver(K=K, Gamma=Gamma)
    tov_solution = tov_solver.solve(rho_central, r_max=r_max_TOV_solve, dr=tov_dr_solve)

    R_star = tov_solution['R']
    M_star = tov_solution['M_star']
    r_max = 2.0 * R_star

    coord_sys = "isotropic" if use_iso else "Schwarzschild"
    print(f"TOV ({coord_sys}): M={M_star:.6f}, R={R_star:.6f}")

    # Setup grid
    spacing = LinearSpacing(num_points, r_max)
    eos = IdealGasEOS(gamma=Gamma)

    ATMOSPHERE = AtmosphereParams(rho_floor=1.0e-12, p_floor=1.0e-14)

    hydro = PerfectFluid(eos=eos, spacetime_mode="dynamic", atmosphere=ATMOSPHERE)
    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    # Create initial data
    print("\nCreating initial data...")
    initial_state = create_initial_data_interpolated(
        tov_solution, grid, background, eos,
        atmosphere=ATMOSPHERE,
        polytrope_K=K, polytrope_Gamma=Gamma,
        use_hydrobase_tau=True,
        use_isotropic=use_iso
    )

    # Compute H terms
    print("\nComputing Hamiltonian constraint terms...")
    terms = compute_hamiltonian_terms(initial_state, grid, background, hydro)

    # Analysis
    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
    r_int = grid.r[interior]

    print("\n" + "="*70)
    print("TERM-BY-TERM ANALYSIS (max absolute value in interior)")
    print("="*70)

    for key in ['term1_2K2/3', 'term2_-Asq', 'term3_barR', 'term4_-8phidD2',
                'term5_-8phidBarDD', 'term6_+8phichris', 'term7_-16piGrho']:
        val = terms[key][interior]
        max_abs = np.max(np.abs(val))
        max_idx = np.argmax(np.abs(val))
        r_max = r_int[max_idx]
        print(f"{key:20s}: max|val| = {max_abs:12.6e} at r = {r_max:.6f}")

    print("\n" + "-"*70)
    H_int = terms['H'][interior]
    H_max_abs = np.max(np.abs(H_int))
    H_max_idx = np.argmax(np.abs(H_int))
    r_H_max = r_int[H_max_idx]
    print(f"{'H_total':20s}: max|H| = {H_max_abs:12.6e} (log10 = {np.log10(H_max_abs):.2f}) at r = {r_H_max:.6f}")

    # Check K and phi values
    K_int = terms['K'][interior]
    phi_int = terms['phi'][interior]
    rho_int = terms['rho'][interior]

    print("\n" + "="*70)
    print("VARIABLE RANGES")
    print("="*70)
    print(f"K:   [{np.min(K_int):.6e}, {np.max(K_int):.6e}]")
    print(f"phi: [{np.min(phi_int):.6e}, {np.max(phi_int):.6e}]")
    print(f"rho: [{np.min(rho_int):.6e}, {np.max(rho_int):.6e}]")

    # Check initial K (should be zero for TOV)
    K_max = np.max(np.abs(K_int))
    print(f"\nInitial K should be ~0 for TOV equilibrium: max|K| = {K_max:.3e}")
    if K_max > 1e-10:
        print("  ⚠️  WARNING: Initial K is not zero! This will cause constraint violation.")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
