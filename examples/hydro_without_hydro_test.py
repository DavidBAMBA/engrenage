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

Gauge choices:
- Lapse: 1+log slicing (engrenage default): ∂_t α = -2 α K
- Shift: Frozen (β^i = 0)

Note: NRPy+ uses harmonic slicing, but we use engrenage's native 1+log.

Author: Engrenage team
Date: 2025-10-15
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
                                             idx_K, idx_arr, idx_att, idx_app,
                                             idx_lambdar, idx_shiftr, idx_br, idx_lapse)
from source.bssn.bssnrhs import get_bssn_rhs
from source.bssn.tensoralgebra import get_bar_gamma_LL, EMTensor

# Hydro (for T^μν calculation only, no evolution)
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.atmosphere import AtmosphereParams
from source.matter.hydro.cons2prim import prim_to_cons

# TOV solver
from examples.tov_solver import TOVSolver
from examples.tov_initial_data_adm_bssn import create_initial_data_adm_bssn


class FrozenFluidHelper:
    """
    Helper class that mimics PerfectFluid but uses FROZEN primitives.
    This ensures we use the exact same EMTensor calculation as perfect_fluid.py.
    """
    def __init__(self, perfect_fluid, rho_fixed, p_fixed, v_fixed):
        self.perfect_fluid = perfect_fluid
        self.rho_fixed = rho_fixed
        self.p_fixed = p_fixed
        self.v_fixed = v_fixed
        self.eos = perfect_fluid.eos
        self.background = perfect_fluid.background
        self.matter_vars_set = True  # Required flag

    def set_matter_vars(self, state, bssn_vars, grid):
        """Dummy method for compatibility with constraints diagnostic."""
        # Just delegate to ensure background is set
        self.perfect_fluid.set_matter_vars(state, bssn_vars, grid)
        self.background = self.perfect_fluid.background

    def _get_primitives(self, bssn_vars, r):
        """Return frozen primitives instead of computing from conservatives."""
        N = len(r)
        # Compute W from frozen velocity
        phi = bssn_vars.phi
        e4phi = np.exp(4.0 * phi)
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, self.perfect_fluid.background)
        gamma_LL = e4phi[:, None, None] * bar_gamma_LL

        vU = np.zeros((N, 3))
        vU[:, 0] = self.v_fixed[:, 0]
        v_squared = np.einsum('xij,xi,xj->x', gamma_LL, vU, vU)
        W = 1.0 / np.sqrt(np.maximum(1.0 - v_squared, 1e-16))

        eps = self.eos.eps_from_rho_p(self.rho_fixed, self.p_fixed)
        h = 1.0 + eps + self.p_fixed / np.maximum(self.rho_fixed, 1e-30)

        return {
            'rho0': self.rho_fixed,
            'vr': self.v_fixed[:, 0],
            'p': self.p_fixed,
            'W': W,
            'h': h
        }

    def get_emtensor(self, r, bssn_vars, background):
        """
        Use perfect_fluid's get_emtensor method with frozen primitives.
        This ensures exact consistency with perfect_fluid.py implementation.
        """
        # Temporarily override the primitives getter
        original_get_primitives = self.perfect_fluid._get_primitives
        self.perfect_fluid._get_primitives = self._get_primitives

        # Call the real get_emtensor from perfect_fluid.py
        emtensor = self.perfect_fluid.get_emtensor(r, bssn_vars, background)

        # Restore original method
        self.perfect_fluid._get_primitives = original_get_primitives

        return emtensor


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

    N = grid.N
    r = grid.r

    # Extract BSSN variables
    bssn_vars = BSSNVars(N)
    bssn_vars.set_bssn_vars(state[:NUM_BSSN_VARS, :])

    # Compute derivatives (needed for BSSN RHS)
    d1 = grid.get_d1_metric_quantities(state)
    d2 = grid.get_d2_metric_quantities(state)

    # Create empty BSSN RHS object
    bssn_rhs = BSSNVars(N)

    # Use FrozenFluidHelper to get EMTensor with exact same method as perfect_fluid.py
    # This ensures consistency with NRPy+ formula: T^μν = ρ_b h u^μ u^ν + P g^μν
    frozen_fluid = FrozenFluidHelper(hydro, rho_fixed, p_fixed, v_fixed)
    emtensor = frozen_fluid.get_emtensor(r, bssn_vars, background)

    # Compute BSSN RHS with matter source from frozen fluid
    get_bssn_rhs(bssn_rhs, r, bssn_vars, d1, d2, background, emtensor)

    # Add gauge evolution (1+log slicing)
    bssn_rhs.lapse += -2.0 * bssn_vars.lapse * bssn_vars.K
    # Shift frozen: no shift evolution terms added

    # Pack RHS into state vector format
    rhs = np.zeros_like(state)
    rhs[idx_phi, :] = bssn_rhs.phi
    rhs[idx_hrr, :] = bssn_rhs.h_LL[:, i_r, i_r]
    rhs[idx_htt, :] = bssn_rhs.h_LL[:, i_t, i_t]
    rhs[idx_hpp, :] = bssn_rhs.h_LL[:, i_p, i_p]
    rhs[idx_K, :] = bssn_rhs.K
    rhs[idx_arr, :] = bssn_rhs.a_LL[:, i_r, i_r]
    rhs[idx_att, :] = bssn_rhs.a_LL[:, i_t, i_t]
    rhs[idx_app, :] = bssn_rhs.a_LL[:, i_p, i_p]
    rhs[idx_lambdar, :] = bssn_rhs.lambda_U[:, i_r]
    rhs[idx_shiftr, :] = bssn_rhs.shift_U[:, i_r]
    rhs[idx_br, :] = bssn_rhs.b_U[:, i_r]
    rhs[idx_lapse, :] = bssn_rhs.lapse

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


def compute_hamiltonian_constraint(state, grid, background, hydro, rho_fixed, p_fixed, v_fixed):
    """Compute Hamiltonian constraint H for diagnostics."""
    from source.bssn.constraintsdiagnostic import get_constraints_diagnostic

    # Use FrozenFluidHelper with exact same EMTensor calculation
    frozen_fluid = FrozenFluidHelper(hydro, rho_fixed, p_fixed, v_fixed)

    # Compute H using engrenage's diagnostic
    Ham, _ = get_constraints_diagnostic(state, np.array([0.0]), grid, background, frozen_fluid)

    return Ham[0, :]


def plot_comparison(tov_solution, state_t0, state_tf, grid, background, hydro,
                    rho_fixed, p_fixed, v_fixed, t_final, M_star, R_star):
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

    # Compute Hamiltonian constraint
    H_initial = compute_hamiltonian_constraint(state_t0, grid, background, hydro, rho_fixed, p_fixed, v_fixed)
    H_final = compute_hamiltonian_constraint(state_tf, grid, background, hydro, rho_fixed, p_fixed, v_fixed)

    r_int = grid.r[NUM_GHOSTS:-NUM_GHOSTS]

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

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

    # Row 3: Hamiltonian constraint (KEY DIAGNOSTIC like NRPy+)
    # Plot log10|H| to match NRPy+ tutorial
    H_initial_safe = np.maximum(np.abs(H_initial[NUM_GHOSTS:-NUM_GHOSTS]), 1e-16)
    H_final_safe = np.maximum(np.abs(H_final[NUM_GHOSTS:-NUM_GHOSTS]), 1e-16)

    axes[2, 0].plot(r_int, np.log10(H_initial_safe), 'b-', linewidth=2, label='t=0')
    axes[2, 0].plot(r_int, np.log10(H_final_safe), 'r--', linewidth=1.5, label=f't={t_final:.2f}')
    axes[2, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5, label=f'R_star={R_star:.2f}')
    axes[2, 0].set_xlabel('r'); axes[2, 0].set_ylabel(r'$\log_{10}|H|$')
    axes[2, 0].set_title('Hamiltonian Constraint Violation')
    axes[2, 0].legend(); axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_ylim([-12, -2])

    # Plot |H| vs r/M (normalized) like NRPy+
    r_norm = r_int / M_star
    axes[2, 1].semilogy(r_norm, H_initial_safe, 'b-', linewidth=2, label='t=0')
    axes[2, 1].semilogy(r_norm, H_final_safe, 'r--', linewidth=1.5, label=f't={t_final:.2f}')
    axes[2, 1].axvline(R_star/M_star, color='gray', linestyle=':', alpha=0.5, label=f'R/M={R_star/M_star:.2f}')
    axes[2, 1].set_xlabel('r/M'); axes[2, 1].set_ylabel(r'$|H|$')
    axes[2, 1].set_title('Hamiltonian Constraint (normalized)')
    axes[2, 1].legend(); axes[2, 1].grid(True, alpha=0.3)

    # Max |H| in interior vs exterior
    idx_surface = np.argmin(np.abs(grid.r - R_star))
    H_max_interior = np.max(H_final_safe[:idx_surface-NUM_GHOSTS])
    H_max_exterior = np.max(H_final_safe[idx_surface-NUM_GHOSTS:])

    axes[2, 2].bar(['Interior\n(r<R)', 'Exterior\n(r>R)'],
                   [np.log10(H_max_interior), np.log10(H_max_exterior)],
                   color=['blue', 'red'], alpha=0.7)
    axes[2, 2].set_ylabel(r'$\log_{10}(\max|H|)$')
    axes[2, 2].set_title('Max Constraint Violation')
    axes[2, 2].grid(True, alpha=0.3, axis='y')
    axes[2, 2].text(0, np.log10(H_max_interior)+0.3, f'{np.log10(H_max_interior):.1f}',
                    ha='center', fontsize=10)
    axes[2, 2].text(1, np.log10(H_max_exterior)+0.3, f'{np.log10(H_max_exterior):.1f}',
                    ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('hydro_without_hydro_evolution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("\nPlot saved: hydro_without_hydro_evolution.png")
    print(f"Hamiltonian constraint at t={t_final:.3f}:")
    print(f"  max|H| (interior) = {H_max_interior:.3e} (log10 = {np.log10(H_max_interior):.1f})")
    print(f"  max|H| (exterior) = {H_max_exterior:.3e} (log10 = {np.log10(H_max_exterior):.1f})")


def main():
    """Main test execution."""
    print("="*70)
    print("HYDRO WITHOUT HYDRO TEST - Engrenage")
    print("="*70)
    print("Concept from NRPy+ hydro_without_hydro.py")
    print("Baumgarte, Hughes, Shapiro PRD 60 087501 (1999)")
    print("Gauge: 1+log slicing (engrenage), frozen shift")
    print("="*70)

    # ==================================================================
    # CONFIGURATION (match NRPy+ defaults exactly)
    # ==================================================================
    # EOS parameters - MATCH NRPy+ exactly!
    K = 1.0                 # Polytropic K (NRPy+ uses K=1, NOT 100!)
    Gamma = 2.0             # Polytropic Gamma
    rho_central = 0.129285  # Central density (NRPy+ value)

    # Grid will be set AFTER solving TOV to get stellar radius
    num_points = 200        # Coarser for testing (NRPy+ uses 72 radial)

    print(f"\nEOS: K={K}, Gamma={Gamma}")
    print(f"Central density: ρ_c={rho_central:.3e}")

    # ==================================================================
    # SOLVE TOV FIRST (to determine stellar radius)
    # ==================================================================
    print("\nSolving TOV equations...")

    # Use large temporary domain for TOV solver
    r_max_TOV_solve = 20.0  # Large enough to capture full star
    tov_num_points_solve = 4000
    tov_dr_solve = r_max_TOV_solve / tov_num_points_solve

    # Use Schwarzschild coordinates
    tov_solver = TOVSolver(K=K, Gamma=Gamma, use_isotropic=False)
    tov_solution = tov_solver.solve(rho_central, r_max=r_max_TOV_solve, dr=tov_dr_solve)

    R_star = tov_solution['R']
    M_star = tov_solution['M_star']
    C_star = tov_solution['C']

    # Set grid domain based on stellar radius (match NRPy+: domain = 2.0 * R_star)
    r_max = 2.0 * R_star

    print(f"TOV: M={M_star:.6f}, R={R_star:.6f}, C={C_star:.4f}")
    print(f"Setting domain: r_max = 2.0 * R = {r_max:.6f}")
    print(f"  → Star occupies: [0, {R_star:.3f}]")
    print(f"  → Vacuum region: [{R_star:.3f}, {r_max:.3f}]\n")

    # ==================================================================
    # SETUP GRID AND MATTER
    # ==================================================================
    spacing = LinearSpacing(num_points, r_max)
    eos = IdealGasEOS(gamma=Gamma)

    # Atmosphere
    ATMOSPHERE = AtmosphereParams(
        rho_floor=1.0e-12,
        p_floor=1.0e-14,
        v_max=0.9999,
        W_max=100.0,
    )

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

    print(f"Grid: N={num_points}, r_max={r_max:.6f}, dr_min={grid.min_dr:.6e}")
    print(f"Atmosphere: ρ_floor={ATMOSPHERE.rho_floor:.2e}, p_floor={ATMOSPHERE.p_floor:.2e}\n")

    # ==================================================================
    # INITIAL DATA
    # ==================================================================
    print("Creating initial data...")

    # Solve TOV (can be on different grid, will interpolate)
    print("  Solving TOV solution...")
    tov_solution = tov_solver.solve(rho_central, r_max=r_max, dr=r_max/4000)

    # Use new ADM→BSSN conversion (with Schwarzschild coordinates)
    initial_state_2d = create_initial_data_adm_bssn(
        tov_solution, grid, background, eos,
        atmosphere=ATMOSPHERE,
        polytrope_K=K, polytrope_Gamma=Gamma,
        use_hydrobase_tau=True,
        use_isotropic=False
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

    # DIAGNOSTIC: Check Hamiltonian constraint at t=0
    H_initial = compute_hamiltonian_constraint(initial_state_2d, grid, background, hydro,
                                                rho_fixed, p_fixed, v_fixed)
    H_initial_interior = H_initial[NUM_GHOSTS:-NUM_GHOSTS]
    H_max_initial = np.max(np.abs(H_initial_interior))
    print(f"\n  Initial Hamiltonian constraint: max|H(t=0)| = {H_max_initial:.3e} (log10 = {np.log10(H_max_initial):.1f})")
    if H_max_initial > 1e-6:
        print(f"  ⚠️  WARNING: Initial data does NOT satisfy constraints! H should be ~10^-10 or better.")

    # ==================================================================
    # EVOLUTION
    # ==================================================================
    dt = 0.1 * grid.min_dr  # CFL condition

    # Match NRPy+: t_final = 1.8 * M (ensures causality from outer boundary)
    t_final = 10.8 * M_star
    num_steps = int(t_final / dt)

    print(f"\nEvolving to t={t_final:.6f} (= 1.8*M) with dt={dt:.6e} ({num_steps} steps)")

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
    plot_comparison(tov_solution, initial_state_2d, state_final, grid, background, hydro,
                    rho_fixed, p_fixed, v_fixed, t_final, M_star, R_star)

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)


if __name__ == "__main__":
    main()
