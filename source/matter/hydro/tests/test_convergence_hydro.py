#!/usr/bin/env python3
"""
test_convergence_hydro.py — Pure advection convergence test for relativistic hydrodynamics.

Tests advection of a smooth C∞ density bump with CONSTANT velocity.
The exact analytical solution is known: ρ(x,t) = ρ₀(x - v*t)

Uses FlatCartesianBackground to eliminate all geometric source terms.
In Cartesian coordinates: ∂ρ/∂t + v ∂ρ/∂x = 0 (pure advection)

This test isolates the spatial reconstruction accuracy from:
- Riemann invariant calculations
- EOS complications
- Reference solution interpolation errors
- Geometric source terms (Christoffel symbols)

Uses RK6 Butcher (7 stages, 6th order) time integrator to avoid limiting
high-order spatial schemes.

Expected convergence:
- Minmod/MC: ~2nd order
- MP5/WENO5/WENO-Z: ~5th order
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add source path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, repo_root)

# Engrenage core
from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.cartesianbackground import FlatCartesianBackground

# BSSN
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS, idx_lapse

# Hydro
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import PolytropicEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.cons2prim import prim_to_cons
from source.matter.hydro.atmosphere import AtmosphereParams


# ============================================================================
# ADVECTION TEST PARAMETERS
# ============================================================================

# Constant advection velocity (relativistic but not too fast)
V_ADVECT = 0.5

# EOS parameters (needed for hydro infrastructure, but won't affect advection much)
K_POLY = 1.0        # Small K to keep pressure low
GAMMA = 5.0 / 3.0   # Adiabatic index

# C∞ bump function parameters
L_BUMP = 0.15       # Half-width of the bump
R_CENTER = 0.25     # Initial center (will advect to 0.25 + v*t)
BUMP_AMP = 0.01     # Small amplitude to minimize pressure gradients (linearized regime)


# ============================================================================
# SMOOTH C∞ INITIAL PROFILE
# ============================================================================

def smooth_bump(r, center=R_CENTER, L=L_BUMP, amp=BUMP_AMP):
    """
    C∞ bump function with small amplitude for linearized advection.

    rho(r) = 1 + amp * exp[-1/(1 - xi^2)]  if |xi| < 1
           = 1                              otherwise

    where xi = (r - center) / L

    Using small amplitude (amp << 1) minimizes pressure gradients,
    keeping velocity approximately constant during evolution.
    This allows testing pure advection in the linearized regime.
    """
    r = np.asarray(r)
    rho = np.ones_like(r, dtype=float)
    xi = (r - center) / L

    # Only modify points inside the bump
    mask = np.abs(xi) < 1.0

    # Avoid division by zero at xi = ±1
    xi_safe = np.where(mask, xi, 0.0)
    arg = 1.0 - xi_safe**2
    arg = np.maximum(arg, 1e-30)

    bump = np.exp(-1.0 / arg)
    rho = np.where(mask, 1.0 + amp * bump, 1.0)

    return rho


def exact_solution(r, t, v=V_ADVECT, center=R_CENTER, L=L_BUMP, amp=BUMP_AMP):
    """
    Exact analytical solution for advection: ρ(r,t) = ρ₀(r - v*t)
    """
    return smooth_bump(r, center=center + v * t, L=L, amp=amp)


def polytropic_pressure(rho, K=K_POLY, Gamma=GAMMA):
    """Polytropic pressure: p = K * rho^Gamma"""
    return K * rho**Gamma


# ============================================================================
# GRID AND STATE INITIALIZATION
# ============================================================================

def create_hydro_and_grid(n_interior=256, r_max=1.0, K=K_POLY, gamma=GAMMA,
                          reconstructor_name="minmod",
                          spacetime_mode="fixed_minkowski"):
    """Create PerfectFluid, StateVector, and Grid following engrenage patterns."""
    num_points = n_interior + 2 * NUM_GHOSTS
    spacing = LinearSpacing(num_points, r_max)
    eos = PolytropicEOS(K=K, gamma=gamma)

    atmosphere = AtmosphereParams(
        rho_floor=1e-15,
        p_floor=1e-17,
        v_max=0.999999,
        W_max=1e3,
        conservative_floor_safety=0.999
    )

    reconstructor = create_reconstruction(reconstructor_name)
    riemann_solver = HLLRiemannSolver(atmosphere=atmosphere)

    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode=spacetime_mode,
        atmosphere=atmosphere,
        reconstructor=reconstructor,
        riemann_solver=riemann_solver
    )

    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)

    background = FlatCartesianBackground(grid.r)  # Cartesian: no geometric source terms
    hydro.background = background

    return grid, hydro, background, eos


def initialize_minkowski_bssn(N):
    """Initialize BSSN variables for flat Minkowski spacetime."""
    bssn_state = np.zeros((NUM_BSSN_VARS, N))
    bssn_state[idx_lapse, :] = 1.0
    return bssn_state


def create_initial_state(grid, rho0, v, p, eos):
    """Create full state vector (BSSN + hydro) from primitive variables."""
    N = grid.N
    state_2d = np.zeros((grid.NUM_VARS, N))

    state_2d[:NUM_BSSN_VARS, :] = initialize_minkowski_bssn(N)

    gamma_rr = np.ones(N)
    D, Sr, tau = prim_to_cons(rho0, v, p, gamma_rr, eos)

    idx_D = NUM_BSSN_VARS
    idx_Sr = NUM_BSSN_VARS + 1
    idx_tau = NUM_BSSN_VARS + 2

    state_2d[idx_D, :] = D
    state_2d[idx_Sr, :] = Sr
    state_2d[idx_tau, :] = tau

    grid.fill_boundaries(state_2d)
    return state_2d


def get_rhs_minkowski(state_2d, grid, background, hydro, bssn_fixed, bssn_d1_fixed):
    """Compute RHS for fixed Minkowski evolution."""
    grid.fill_boundaries(state_2d)

    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(bssn_fixed)

    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    hydro_rhs = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1_fixed, background)

    rhs = np.zeros_like(state_2d)
    rhs[NUM_BSSN_VARS:, :] = hydro_rhs
    return rhs


def rk6_step(state_2d, dt, grid, background, hydro, bssn_fixed, bssn_d1_fixed):
    """
    Single RK6 Butcher step (7 stages, 6th order).
    """
    def rhs(state):
        grid.fill_boundaries(state)
        return get_rhs_minkowski(state, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    k1 = rhs(state_2d)
    k2 = rhs(state_2d + dt * (1/3) * k1)
    k3 = rhs(state_2d + dt * (2/3) * k2)
    k4 = rhs(state_2d + dt * (1/12 * k1 + 1/3 * k2 - 1/12 * k3))
    k5 = rhs(state_2d + dt * (-1/16 * k1 + 9/8 * k2 - 3/16 * k3 - 3/8 * k4))
    k6 = rhs(state_2d + dt * (9/8 * k2 - 3/8 * k3 - 3/4 * k4 + 1/2 * k5))
    k7 = rhs(state_2d + dt * (9/44 * k1 - 9/11 * k2 + 63/44 * k3 + 18/11 * k4 - 16/11 * k6))

    state_new = state_2d + dt * (11/120 * k1 + 27/40 * k3 + 27/40 * k4 - 4/15 * k5 - 4/15 * k6 + 11/120 * k7)
    grid.fill_boundaries(state_new)

    return state_new


def compute_dt(rho0, v, p, eos, grid, cfl=0.3):
    """Compute timestep from CFL condition."""
    eps = eos.eps_from_rho_p(rho0, p)
    h = 1.0 + eps + p / np.maximum(rho0, 1e-300)
    cs2 = np.clip(eos.gamma * p / np.maximum(rho0 * h, 1e-300), 0.0, 1.0 - 1e-10)
    cs = np.sqrt(cs2)
    amax = np.max(np.abs(v) + cs) + 1e-10
    return cfl * grid.min_dr / amax


def extract_primitives(state_2d, grid, hydro, bssn_vars):
    """Extract primitive variables from state using cons2prim."""
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    rho0, v, p, eps, W, h, success = hydro._get_primitives(bssn_vars, grid.r)
    return rho0, v, p


# ============================================================================
# ADVECTION SIMULATION
# ============================================================================

def run_advection_test(n_interior, reconstructor_name, v_advect=V_ADVECT,
                       Tfinal=0.4, r_max=1.0, cfl=0.2):
    """
    Run pure advection test with constant velocity.

    Returns:
        r_interior: radial coordinates (interior only)
        rho_final: final density (interior only)
        rho_exact: exact solution at final time
        t_final: actual final time
        steps: number of timesteps
    """
    grid, hydro, background, eos = create_hydro_and_grid(
        n_interior=n_interior, r_max=r_max, gamma=GAMMA, K=K_POLY,
        reconstructor_name=reconstructor_name,
        spacetime_mode="fixed_minkowski"
    )

    r = grid.r
    ng = NUM_GHOSTS

    # Initial conditions: C∞ density bump with CONSTANT velocity
    rho0 = smooth_bump(r, center=R_CENTER, L=L_BUMP)
    v = np.full(grid.N, v_advect)  # Constant velocity everywhere
    p = polytropic_pressure(rho0, K=K_POLY, Gamma=GAMMA)

    # Create initial state
    state_2d = create_initial_state(grid, rho0, v, p, eos)

    # Fixed BSSN
    bssn_fixed = state_2d[:NUM_BSSN_VARS, :].copy()
    bssn_d1_fixed = grid.get_d1_metric_quantities(state_2d)

    # Time evolution
    t, steps = 0.0, 0
    while t < Tfinal and steps < 100000:
        dt = compute_dt(rho0, v, p, eos, grid, cfl=cfl)

        if t + dt > Tfinal:
            dt = Tfinal - t

        state_2d = rk6_step(state_2d, dt, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

        bssn_vars = BSSNVars(grid.N)
        bssn_vars.set_bssn_vars(bssn_fixed)
        rho0, v, p = extract_primitives(state_2d, grid, hydro, bssn_vars)

        t += dt
        steps += 1

    # Compute exact solution at final time
    r_interior = r[ng:-ng]
    rho_exact = exact_solution(r_interior, t, v=v_advect, center=R_CENTER, L=L_BUMP)

    return r_interior, rho0[ng:-ng], rho_exact, t, steps


# ============================================================================
# CONVERGENCE TEST
# ============================================================================

def test_advection_convergence():
    """
    Convergence test with pure advection of smooth C∞ bump.
    Compares against EXACT analytical solution.
    """
    print("\n" + "="*70)
    print("PURE ADVECTION CONVERGENCE TEST")
    print("="*70)

    # Test parameters
    resolutions = [50, 100, 200, 400, 800]
    Tfinal = 0.4           # Bump moves from r=0.25 to r=0.45
    r_max = 1.0
    cfl = 0.2

    reconstructors = ["minmod", "mc", "mp5", "weno5", "wenoz"]

    colors = {
        "minmod": "blue",
        "mc": "green",
        "mp5": "red",
        "weno5": "purple",
        "wenoz": "orange"
    }

    print(f"\nParameters:")
    print(f"  Background: FlatCartesianBackground (no geometric source terms)")
    print(f"  Advection velocity: v = {V_ADVECT}")
    print(f"  C∞ bump: center={R_CENTER}, L={L_BUMP}, amplitude={BUMP_AMP}")
    print(f"  (Small amplitude -> linearized regime -> minimal pressure gradients)")
    print(f"  Final position: x = {R_CENTER + V_ADVECT * Tfinal:.2f}")
    print(f"  Resolutions: {resolutions}")
    print(f"  T_final: {Tfinal}")
    print(f"  CFL: {cfl}")
    print(f"\n  Comparing against EXACT analytical solution!")

    # ------------------------------------------------------------------
    # Run convergence tests
    # ------------------------------------------------------------------
    convergence_data = {recon: {} for recon in reconstructors}
    plot_results = {}

    for recon_name in reconstructors:
        print(f"\n{recon_name}:")
        for n_interior in resolutions:
            r_test, rho_test, rho_exact, t_test, steps_test = run_advection_test(
                n_interior=n_interior,
                reconstructor_name=recon_name,
                v_advect=V_ADVECT,
                Tfinal=Tfinal,
                r_max=r_max,
                cfl=cfl
            )

            # Compute errors vs EXACT solution (no interpolation needed!)
            dr = r_test[1] - r_test[0]
            L1_rho = np.sum(np.abs(rho_test - rho_exact)) * dr
            L2_rho = np.sqrt(np.sum((rho_test - rho_exact)**2) * dr)
            Linf_rho = np.max(np.abs(rho_test - rho_exact))

            convergence_data[recon_name][n_interior] = {
                "L1": L1_rho,
                "L2": L2_rho,
                "Linf": Linf_rho
            }

            print(f"  n={n_interior:4d}: L1={L1_rho:.2e}, L2={L2_rho:.2e}, Linf={Linf_rho:.2e}, steps={steps_test}")

            if n_interior == max(resolutions):
                plot_results[recon_name] = {
                    "r": r_test,
                    "rho": rho_test,
                    "rho_exact": rho_exact,
                    "t": t_test
                }

    # ------------------------------------------------------------------
    # Print convergence rates
    # ------------------------------------------------------------------
    print("\n" + "="*100)
    print("CONVERGENCE RATES (density vs EXACT solution)")
    print("="*100)

    for norm_name in ["L1", "L2", "Linf"]:
        print(f"\n--- {norm_name} norm ---")
        header = f"{'Reconstructor':<12}"
        for n in resolutions:
            header += f" {'n='+str(n):<11}"
        for i in range(1, len(resolutions)):
            header += f" {'Rate '+str(i):<8}"
        print(header)
        print("-"*100)

        for recon_name in reconstructors:
            line = f"{recon_name:<12}"
            errors = []
            for n in resolutions:
                err = convergence_data[recon_name][n][norm_name]
                errors.append(err)
                line += f" {err:<11.3e}"

            rates = []
            for i in range(1, len(errors)):
                if errors[i] > 0 and errors[i-1] > 0:
                    rate = np.log2(errors[i-1] / errors[i])
                    rates.append(rate)
                else:
                    rates.append(np.nan)

            for rate in rates:
                if np.isnan(rate):
                    line += f" {'N/A':<8}"
                else:
                    line += f" {rate:<8.2f}"

            print(line)

    print("\n" + "="*100)
    print("Expected convergence orders (using RK6 time integrator):")
    print("  - Minmod/MC: ~2 (2nd order TVD)")
    print("  - MP5/WENO5/WENO-Z: ~5 (5th order spatial)")
    print("="*100)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: Solution profiles
    ax1 = axes[0]
    res = plot_results["minmod"]
    r_init = res["r"]
    rho_init = smooth_bump(r_init, center=R_CENTER, L=L_BUMP)
    ax1.plot(r_init, rho_init, 'k:', label='Initial', linewidth=2, alpha=0.5)

    for recon_name in reconstructors:
        res = plot_results[recon_name]
        ax1.plot(res["r"], res["rho"], color=colors[recon_name],
                 label=recon_name, linewidth=1.5, alpha=0.8)

    ax1.plot(res["r"], res["rho_exact"], 'k--', label='Exact', linewidth=2, alpha=0.7)

    ax1.set_xlabel('x')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Advection (n={max(resolutions)}, t={res["t"]:.3f}, v={V_ADVECT}, Cartesian)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Middle: Error profiles
    ax2 = axes[1]
    for recon_name in reconstructors:
        res = plot_results[recon_name]
        error = res["rho"] - res["rho_exact"]
        ax2.plot(res["r"], error, color=colors[recon_name],
                 label=recon_name, linewidth=1.5, alpha=0.8)

    ax2.set_xlabel('x')
    ax2.set_ylabel('Error (ρ - ρ_exact)')
    ax2.set_title('Error Profile')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Right: Convergence plot
    ax3 = axes[2]
    for recon_name in reconstructors:
        ns = sorted(convergence_data[recon_name].keys())
        l2_errors = [convergence_data[recon_name][n]["L2"] for n in ns]
        ax3.loglog(ns, l2_errors, 'o-', color=colors[recon_name],
                   label=recon_name, linewidth=2, markersize=8)

    # Reference slopes
    n_ref = np.array(resolutions)
    l2_mid = convergence_data["minmod"][resolutions[1]]["L2"]
    ax3.loglog(n_ref, l2_mid * (resolutions[1]/n_ref)**2, 'k--', alpha=0.5, label='2nd order')
    l2_mid_5 = convergence_data["wenoz"][resolutions[1]]["L2"]
    ax3.loglog(n_ref, l2_mid_5 * (resolutions[1]/n_ref)**5, 'k:', alpha=0.5, label='5th order')

    ax3.set_xlabel('N (grid points)')
    ax3.set_ylabel('L2 Error')
    ax3.set_title('Convergence (L2 norm vs EXACT)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()

    plt.tight_layout()
    plt.savefig("test_advection_convergence.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved: test_advection_convergence.png")

    # ------------------------------------------------------------------
    # Pass/Fail check and summary
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_ok = True
    for recon_name in reconstructors:
        errors = [convergence_data[recon_name][n]["L2"] for n in sorted(resolutions)]
        if len(errors) >= 2:
            rates = []
            for i in range(1, len(errors)):
                if errors[i] > 0 and errors[i-1] > 0:
                    rates.append(np.log2(errors[i-1] / errors[i]))

            if rates:
                avg_rate = np.mean(rates[-2:])  # Use last 2 rates (asymptotic)
                expected = 5.0 if recon_name in ["mp5", "weno5", "wenoz"] else 2.0
                status = "OK" if avg_rate > expected - 0.5 else "LOW"
                print(f"  {recon_name:<8}: avg rate = {avg_rate:.2f} (expected ~{expected:.0f}) [{status}]")
                if avg_rate < 1.5:
                    all_ok = False

    print("\n" + "="*70)
    print("RESULT:", "PASS" if all_ok else "FAIL")
    print("="*70)
    return all_ok


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("="*70)
    print("ENGRENAGE HYDRO PURE ADVECTION CONVERGENCE TEST")
    print("="*70)
    print(f"Using NUM_GHOSTS = {NUM_GHOSTS}")
    print(f"Using NUM_BSSN_VARS = {NUM_BSSN_VARS}")

    ok = test_advection_convergence()

    print("\n" + "="*70)
    print("FINAL:", "PASS" if ok else "FAIL")
    print("="*70)
