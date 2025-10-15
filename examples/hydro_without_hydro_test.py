"""
Hydro Without Hydro Test - Engrenage version

Test based on NRPy+ hydro_without_hydro.py following:
Baumgarte, Hughes, and Shapiro Phys. Rev. D 60 087501 (1999)
https://arxiv.org/abs/gr-qc/9902024

Concept:
- Evolve a TOV star with STATIC fluid (no hydro evolution)
- Only evolve spacetime geometry via BSSN equations
- Fluid provides stress-energy tensor T^μν as source for BSSN
- This tests:
  1. TOV initial data correctness
  2. BSSN-hydro coupling via T^μν
  3. Geometry evolution in response to matter
  4. Constraint preservation

The fluid primitives (ρ, P, v) remain FIXED at their initial TOV values.
Only the BSSN variables (φ, h_ij, K, a_ij, α, β^i) evolve.

This is a simpler test than full GRHD evolution because:
- No Riemann solver needed
- No cons2prim inversion
- No hydro boundary conditions
- Pure test of gravitational sector

Author: Engrenage team
Date: 2025-10-14
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# Engrenage core
sys.path.insert(0, '/home/yo/repositories/engrenage')
from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground, i_r, i_t, i_p

# BSSN
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import (NUM_BSSN_VARS, idx_phi, idx_hrr, idx_htt, idx_hpp,
                                             idx_K, idx_arr, idx_att, idx_app, idx_lapse)
from source.bssn.bssnrhs import get_bssn_rhs
from source.bssn.tensoralgebra import get_bar_gamma_LL

# Hydro (for T^μν calculation only, no evolution)
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.atmosphere import AtmosphereParams
from source.matter.hydro.cons2prim import prim_to_cons

# TOV solver
from examples.tov_solver import TOVSolver
import examples.tov_initial_data_interpolated as tov_id_interp


def get_rhs_bssn_only(t, y, grid, background, hydro, rho_fixed, p_fixed, v_fixed):
    """
    RHS for BSSN evolution with FIXED fluid (hydro without hydro).

    The fluid primitives (rho, P, v) are held constant at their initial TOV values.
    Only BSSN variables evolve in response to the matter source T^μν.

    Args:
        rho_fixed: (N,) array of fixed rest-mass density
        p_fixed: (N,) array of fixed pressure
        v_fixed: (N, 3) array of fixed velocity (should be zero for TOV)
    """
    state = y.reshape((grid.NUM_VARS, grid.N))
    grid.fill_boundaries(state)

    # Extract BSSN variables
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state[:NUM_BSSN_VARS, :])

    # Set hydro matter vars (needed for T^μν calculation)
    # But use FIXED primitives, not evolved conservatives
    hydro.set_matter_vars(state, bssn_vars, grid)

    # Override with fixed primitives (this is the key: fluid doesn't evolve!)
    # We keep conservatives in state for consistency, but T^μν uses fixed primitives

    # Get BSSN RHS with matter source from FIXED fluid
    # First compute d1 quantities
    bssn_d1 = grid.get_d1_metric_quantities(state)

    # Get T^μν from fixed fluid state
    # Use the Valencia hydro object's internal methods
    N = len(grid.r)

    # Extract geometry using valencia's internal method
    g = hydro.valencia._extract_geometry(grid.r, bssn_vars, "dynamic", background)

    # Compute thermodynamic quantities from FIXED primitives
    eps = hydro.eos.eps_from_rho_p(rho_fixed, p_fixed)

    # Specific enthalpy
    try:
        h = hydro.eos.enthalpy(rho_fixed, p_fixed, eps)
    except TypeError:
        h = 1.0 + eps + p_fixed / np.maximum(rho_fixed, 1e-30)

    # Lorentz factor (should be 1 for TOV since v=0)
    v_squared = np.einsum('xij,xi,xj->x', g['gamma_LL'], v_fixed, v_fixed)
    W = 1.0 / np.sqrt(np.maximum(1.0 - v_squared, 1e-16))

    # Compute T^μν using valencia's internal method
    T4UU = hydro.valencia._compute_T4UU(rho_fixed, v_fixed, p_fixed, W, h, g)

    # Convert from dict format to full array for BSSN
    # BSSN expects full 4x4 array, Valencia returns dict
    T4UU_array = np.zeros((N, 4, 4))
    T4UU_array[:, 0, 0] = T4UU['00']
    for i in range(3):
        T4UU_array[:, 0, i+1] = T4UU['0i'][:, i]
        T4UU_array[:, i+1, 0] = T4UU['0i'][:, i]  # Symmetric
    for i in range(3):
        for j in range(3):
            T4UU_array[:, i+1, j+1] = T4UU['ij'][:, i, j]

    # Get BSSN RHS (with matter source)
    bssn_rhs = get_bssn_rhs(
        grid.r, bssn_vars, bssn_d1, background,
        gauge_mode="HarmonicSlicing",  # Match NRPy+ harmonic slicing
        shift_mode="Frozen",            # Match NRPy+ frozen shift
        enable_matter_source=True,      # Include T^μν source
        T4UU=T4UU_array
    )

    # Full RHS: BSSN evolves, hydro frozen
    rhs = np.zeros_like(state)
    rhs[:NUM_BSSN_VARS, :] = bssn_rhs
    # Hydro RHS = 0 (frozen fluid)
    rhs[NUM_BSSN_VARS:, :] = 0.0

    return rhs.flatten()


def rk4_step(state_flat, dt, grid, background, hydro, rho_fixed, p_fixed, v_fixed):
    """Single RK4 timestep for BSSN-only evolution."""
    # Stage 1
    k1 = get_rhs_bssn_only(0, state_flat, grid, background, hydro, rho_fixed, p_fixed, v_fixed)

    # Stage 2
    state_2 = state_flat + 0.5 * dt * k1
    k2 = get_rhs_bssn_only(0, state_2, grid, background, hydro, rho_fixed, p_fixed, v_fixed)

    # Stage 3
    state_3 = state_flat + 0.5 * dt * k2
    k3 = get_rhs_bssn_only(0, state_3, grid, background, hydro, rho_fixed, p_fixed, v_fixed)

    # Stage 4
    state_4 = state_flat + dt * k3
    k4 = get_rhs_bssn_only(0, state_4, grid, background, hydro, rho_fixed, p_fixed, v_fixed)

    # Combine
    state_new = state_flat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return state_new


def evolve_hydro_without_hydro(state_initial, dt, num_steps, grid, background, hydro,
                                rho_fixed, p_fixed, v_fixed):
    """
    Evolve BSSN equations with fixed fluid source.

    Args:
        rho_fixed: (N,) fixed rest-mass density
        p_fixed: (N,) fixed pressure
        v_fixed: (N, 3) fixed velocity
    """
    state_flat = state_initial.flatten()

    print("\n===== Hydro Without Hydro Evolution =====")
    print(f"Evolving {num_steps} steps with dt={dt:.6e}")
    print(f"Fluid primitives FROZEN at TOV values")
    print(f"Only BSSN geometry evolves\n")

    for step in range(num_steps):
        # RK4 step
        state_flat = rk4_step(state_flat, dt, grid, background, hydro,
                             rho_fixed, p_fixed, v_fixed)

        # Diagnostics
        if (step + 1) % 10 == 0 or step == 0:
            state_2d = state_flat.reshape((grid.NUM_VARS, grid.N))

            # Check BSSN variables
            phi = state_2d[idx_phi, :]
            alpha = state_2d[idx_lapse, :]
            K = state_2d[idx_K, :]

            interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

            print(f"Step {step+1:4d}  t={((step+1)*dt):.6e}:  "
                  f"φ[{np.min(phi[interior]):.6e}, {np.max(phi[interior]):.6e}]  "
                  f"α[{np.min(alpha[interior]):.6e}, {np.max(alpha[interior]):.6e}]  "
                  f"K[{np.min(K[interior]):.6e}, {np.max(K[interior]):.6e}]")

    return state_flat.reshape((grid.NUM_VARS, grid.N))


def plot_comparison(tov_solution, state_t0, state_tf, grid, hydro, t_final):
    """Plot initial vs final state for hydro-without-hydro test."""

    # Extract primitives (should be unchanged since fluid is frozen)
    bssn_0 = BSSNVars(grid.N)
    bssn_0.set_bssn_vars(state_t0[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t0, bssn_0, grid)
    prim_0 = hydro._get_primitives(bssn_0, grid.r)

    bssn_f = BSSNVars(grid.N)
    bssn_f.set_bssn_vars(state_tf[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_tf, bssn_f, grid)
    prim_f = hydro._get_primitives(bssn_f, grid.r)

    r_int = grid.r[NUM_GHOSTS:-NUM_GHOSTS]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: Fluid primitives (should be unchanged)
    axes[0, 0].semilogy(r_int, prim_0['rho0'][NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label='t=0')
    axes[0, 0].semilogy(r_int, prim_f['rho0'][NUM_GHOSTS:-NUM_GHOSTS], 'r--', linewidth=1.5, label=f't={t_final:.2f}')
    axes[0, 0].set_xlabel('r'); axes[0, 0].set_ylabel(r'$\rho_0$')
    axes[0, 0].set_title('Density (should be constant)')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(r_int, np.maximum(prim_0['p'][NUM_GHOSTS:-NUM_GHOSTS], 1e-20), 'b-', linewidth=2, label='t=0')
    axes[0, 1].semilogy(r_int, np.maximum(prim_f['p'][NUM_GHOSTS:-NUM_GHOSTS], 1e-20), 'r--', linewidth=1.5, label=f't={t_final:.2f}')
    axes[0, 1].set_xlabel('r'); axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure (should be constant)')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(r_int, prim_0['vr'][NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label='t=0')
    axes[0, 2].plot(r_int, prim_f['vr'][NUM_GHOSTS:-NUM_GHOSTS], 'r--', linewidth=1.5, label=f't={t_final:.2f}')
    axes[0, 2].set_xlabel('r'); axes[0, 2].set_ylabel(r'$v^r$')
    axes[0, 2].set_title('Velocity (should be zero)')
    axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

    # Row 2: BSSN variables (these evolve)
    axes[1, 0].plot(r_int, state_t0[idx_phi, NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label='t=0')
    axes[1, 0].plot(r_int, state_tf[idx_phi, NUM_GHOSTS:-NUM_GHOSTS], 'r--', linewidth=1.5, label=f't={t_final:.2f}')
    axes[1, 0].set_xlabel('r'); axes[1, 0].set_ylabel(r'$\phi$')
    axes[1, 0].set_title('Conformal Factor (evolves)')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(r_int, state_t0[idx_lapse, NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label='t=0')
    axes[1, 1].plot(r_int, state_tf[idx_lapse, NUM_GHOSTS:-NUM_GHOSTS], 'r--', linewidth=1.5, label=f't={t_final:.2f}')
    axes[1, 1].set_xlabel('r'); axes[1, 1].set_ylabel(r'$\alpha$')
    axes[1, 1].set_title('Lapse (evolves)')
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(r_int, state_t0[idx_K, NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label='t=0')
    axes[1, 2].plot(r_int, state_tf[idx_K, NUM_GHOSTS:-NUM_GHOSTS], 'r--', linewidth=1.5, label=f't={t_final:.2f}')
    axes[1, 2].set_xlabel('r'); axes[1, 2].set_ylabel('K')
    axes[1, 2].set_title('Trace Extrinsic Curvature (evolves)')
    axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hydro_without_hydro_evolution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("\nPlot saved: hydro_without_hydro_evolution.png")


def main():
    """Main test execution."""
    print("="*70)
    print("HYDRO WITHOUT HYDRO TEST - Engrenage")
    print("="*70)
    print("Following NRPy+ hydro_without_hydro.py")
    print("Baumgarte, Hughes, Shapiro PRD 60 087501 (1999)")
    print("="*70)

    # ==================================================================
    # CONFIGURATION (match NRPy+ defaults where possible)
    # ==================================================================
    r_max = 7.5             # Match NRPy+ grid_physical_size
    num_points = 200        # Coarser for testing (NRPy+ uses 72 radial)
    K = 100.0
    Gamma = 2.0
    rho_central = 1.28e-3

    # Atmosphere
    ATMOSPHERE = AtmosphereParams(
        rho_floor=1.0e-12,
        p_floor=1.0e-14,
        v_max=0.9999,
        W_max=100.0,
    )

    print(f"\nGrid: N={num_points}, r_max={r_max}")
    print(f"EOS: K={K}, Gamma={Gamma}")
    print(f"Central density: ρ_c={rho_central:.3e}")
    print(f"Atmosphere: ρ_floor={ATMOSPHERE.rho_floor:.2e}, p_floor={ATMOSPHERE.p_floor:.2e}\n")

    # ==================================================================
    # SETUP
    # ==================================================================
    spacing = LinearSpacing(num_points, r_max)
    eos = IdealGasEOS(gamma=Gamma)
    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere=ATMOSPHERE,
        reconstructor=None,  # Not needed, fluid is frozen
        riemann_solver=None  # Not needed, fluid is frozen
    )

    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    # ==================================================================
    # SOLVE TOV
    # ==================================================================
    print("Solving TOV equations...")
    tov_solver = TOVSolver(K=K, Gamma=Gamma, use_isotropic=False)

    # Fine TOV grid
    tov_num_points = num_points * 10
    tov_dr = r_max / tov_num_points

    tov_solution = tov_solver.solve(rho_central, r_max=r_max, dr=tov_dr)

    print(f"TOV: M={tov_solution['M_star']:.6f}, R={tov_solution['R']:.3f}, C={tov_solution['C']:.4f}\n")

    # ==================================================================
    # INITIAL DATA
    # ==================================================================
    print("Creating initial data...")
    initial_state_2d = tov_id_interp.create_initial_data_interpolated(
        tov_solution, grid, background, eos,
        atmosphere=ATMOSPHERE,
        polytrope_K=K, polytrope_Gamma=Gamma,
        use_hydrobase_tau=True,
        interp_order=11
    )

    # Extract and freeze fluid primitives
    bssn_initial = BSSNVars(grid.N)
    bssn_initial.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(initial_state_2d, bssn_initial, grid)
    prim_initial = hydro._get_primitives(bssn_initial, grid.r)

    # These will be held constant throughout evolution
    rho_fixed = prim_initial['rho0'].copy()
    p_fixed = prim_initial['p'].copy()
    v_fixed = np.zeros((grid.N, 3))  # TOV has zero velocity
    v_fixed[:, 0] = prim_initial['vr'].copy()

    print(f"Fluid primitives frozen at t=0:")
    print(f"  ρ_central = {rho_fixed[NUM_GHOSTS]:.6e}")
    print(f"  P_central = {p_fixed[NUM_GHOSTS]:.6e}")
    print(f"  v_max = {np.max(np.abs(v_fixed)):.6e}")

    # ==================================================================
    # EVOLUTION
    # ==================================================================
    dt = 0.1 * grid.min_dr  # CFL condition
    t_final = 1.0           # Match NRPy+ default (or less for testing)
    num_steps = int(t_final / dt)

    print(f"\nEvolving to t={t_final} with dt={dt:.6e} ({num_steps} steps)")

    state_final = evolve_hydro_without_hydro(
        initial_state_2d, dt, num_steps, grid, background, hydro,
        rho_fixed, p_fixed, v_fixed
    )

    # ==================================================================
    # DIAGNOSTICS
    # ==================================================================
    print("\n" + "="*70)
    print("FINAL DIAGNOSTICS")
    print("="*70)

    # Check that fluid didn't change
    bssn_final = BSSNVars(grid.N)
    bssn_final.set_bssn_vars(state_final[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_final, bssn_final, grid)
    prim_final = hydro._get_primitives(bssn_final, grid.r)

    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

    delta_rho = np.max(np.abs(prim_final['rho0'][interior] - rho_fixed[interior]))
    delta_p = np.max(np.abs(prim_final['p'][interior] - p_fixed[interior]))
    delta_v = np.max(np.abs(prim_final['vr'][interior] - v_fixed[interior, 0]))

    print(f"\nFluid changes (should be zero or roundoff):")
    print(f"  Max |Δρ| = {delta_rho:.3e}")
    print(f"  Max |ΔP| = {delta_p:.3e}")
    print(f"  Max |Δv| = {delta_v:.3e}")

    # Check BSSN evolution
    delta_phi = np.max(np.abs(state_final[idx_phi, interior] - initial_state_2d[idx_phi, interior]))
    delta_alpha = np.max(np.abs(state_final[idx_lapse, interior] - initial_state_2d[idx_lapse, interior]))
    delta_K = np.max(np.abs(state_final[idx_K, interior] - initial_state_2d[idx_K, interior]))

    print(f"\nBSSN changes (should be non-zero):")
    print(f"  Max |Δφ| = {delta_phi:.3e}")
    print(f"  Max |Δα| = {delta_alpha:.3e}")
    print(f"  Max |ΔK| = {delta_K:.3e}")

    # Plot
    plot_comparison(tov_solution, initial_state_2d, state_final, grid, hydro, t_final)

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)


if __name__ == "__main__":
    main()
