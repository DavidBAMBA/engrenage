#!/usr/bin/env python3
"""
Compare Schwarzschild vs Isotropic coordinate initial data.
"""

import sys
sys.path.insert(0, '/home/yo/repositories/engrenage')

import numpy as np
from examples.tov_solver import TOVSolver
from examples.tov_initial_data_adm_bssn import create_initial_data_adm_bssn, compute_adm_from_tov

from source.core.grid import Grid
from source.core.spacing import LinearSpacing
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.atmosphere import AtmosphereParams
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS

print("="*70)
print("COMPARING SCHWARZSCHILD vs ISOTROPIC COORDINATES")
print("="*70)

K = 1.0
Gamma = 2.0
rho_central = 0.129285
num_points = 200

for use_iso in [False, True]:
    coord_name = "ISOTROPIC" if use_iso else "SCHWARZSCHILD"
    print(f"\n{'='*70}")
    print(f"{coord_name} COORDINATES")
    print(f"{'='*70}")

    # Solve TOV
    tov_solver = TOVSolver(K=K, Gamma=Gamma, use_isotropic=use_iso)
    tov_solution = tov_solver.solve(rho_central, r_max=20.0, dr=0.005)

    R_star = tov_solution['R']
    M_star = tov_solution['M_star']
    r_max = 2.0 * R_star

    print(f"\nTOV Solution:")
    print(f"  M_star = {M_star:.6f}")
    print(f"  R_star = {R_star:.6f}")
    print(f"  C = M/R = {M_star/R_star:.6f}")

    # Setup grid
    spacing = LinearSpacing(num_points, r_max)
    eos = IdealGasEOS(gamma=Gamma)
    ATMOSPHERE = AtmosphereParams(rho_floor=1.0e-12, p_floor=1.0e-14)
    hydro = PerfectFluid(eos=eos, spacetime_mode="dynamic", atmosphere=ATMOSPHERE)
    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    # Resolve TOV on grid
    tov_solution_grid = tov_solver.solve(rho_central, r_max=r_max, dr=r_max/4000)

    # Extract ADM variables
    adm_vars = compute_adm_from_tov(tov_solution_grid, grid, use_isotropic=use_iso)

    # Look at a point in the interior (10% of stellar radius)
    idx_test = np.argmin(np.abs(np.abs(grid.r) - 0.1*R_star))
    r_test = np.abs(grid.r[idx_test])

    print(f"\nPhysical metric at r = {r_test:.6f}:")
    print(f"  γ_rr = {adm_vars['gamma_LL'][idx_test, 0, 0]:.6f}")
    print(f"  γ_θθ = {adm_vars['gamma_LL'][idx_test, 1, 1]:.6f}")
    print(f"  γ_φφ = {adm_vars['gamma_LL'][idx_test, 2, 2]:.6f}")

    det_gamma = np.linalg.det(adm_vars['gamma_LL'][idx_test])
    print(f"  det(γ) = {det_gamma:.6e}")

    # Theoretical values
    if use_iso:
        # Find exp4phi at this radius
        idx_tov = np.argmin(np.abs(tov_solution_grid['r'] - r_test))
        psi4 = tov_solution_grid['exp4phi'][idx_tov]
        print(f"\n  From TOV: ψ⁴ = {psi4:.6f}")
        print(f"  Expected γ_rr = ψ⁴ = {psi4:.6f}")
        print(f"  Expected γ_θθ = ψ⁴ × r² = {psi4 * r_test**2:.6f}")
        print(f"  Expected det(γ) = ψ¹² × r⁴ = {psi4**3 * r_test**4:.6e}")
    else:
        idx_tov = np.argmin(np.abs(tov_solution_grid['r'] - r_test))
        exp4phi = tov_solution_grid['exp4phi'][idx_tov]
        print(f"\n  From TOV: exp(4φ) = {exp4phi:.6f}")
        print(f"  Expected γ_rr = exp(4φ) = {exp4phi:.6f}")
        print(f"  Expected γ_θθ = r² = {r_test**2:.6f}")
        print(f"  Expected det(γ) = exp(4φ) × r⁴ = {exp4phi * r_test**4:.6e}")

    # Create full BSSN initial data
    initial_state = create_initial_data_adm_bssn(
        tov_solution_grid, grid, background, eos,
        atmosphere=ATMOSPHERE,
        polytrope_K=K, polytrope_Gamma=Gamma,
        use_hydrobase_tau=True,
        use_isotropic=use_iso
    )

    # Extract BSSN variables
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(initial_state[:NUM_BSSN_VARS, :])

    print(f"\nBSSN conformal factor φ:")
    print(f"  φ at center = {bssn_vars.phi[3]:.6e}")
    print(f"  φ at r={r_test:.6f} = {bssn_vars.phi[idx_test]:.6e}")
    print(f"  φ_min = {np.min(bssn_vars.phi):.6e}")
    print(f"  φ_max = {np.max(bssn_vars.phi):.6e}")

    # Check h_ij
    print(f"\nBSSN h_ij at r={r_test:.6f}:")
    print(f"  h_rr = {bssn_vars.h_LL[idx_test, 0, 0]:.6e}")
    print(f"  h_θθ = {bssn_vars.h_LL[idx_test, 1, 1]:.6e}")
    print(f"  h_φφ = {bssn_vars.h_LL[idx_test, 2, 2]:.6e}")

    # Compute Hamiltonian constraint
    from source.bssn.constraintsdiagnostic import get_constraints_diagnostic
    hydro.set_matter_vars(initial_state, bssn_vars, grid)
    Ham, Mom = get_constraints_diagnostic(
        initial_state.flatten(), 0.0, grid, background, hydro
    )

    max_H = np.max(np.abs(Ham[0, :]))
    log10_H = np.log10(max_H) if max_H > 0 else -np.inf

    print(f"\nHamiltonian constraint:")
    print(f"  max|H| = {max_H:.6e} (log10 = {log10_H:.2f})")

print(f"\n{'='*70}")
print("COMPARISON COMPLETE")
print(f"{'='*70}")
