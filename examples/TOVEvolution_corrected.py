"""
TOV Star Evolution using engrenage - CORRECTED VERSION

Implements:
1. TOV solver following NRPy+ equations exactly
2. Proper BSSN initial data from TOV solution
3. Conservative hydro variables
4. Cowling evolution (fixed spacetime)

Author: Claude + engrenage team
Date: 2025-10-01
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.integrate import ode, trapezoid
from scipy.interpolate import interp1d

# Engrenage core
sys.path.insert(0, '/home/yo/repositories/engrenage')
from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground, i_r

# BSSN
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS, idx_lapse
from source.bssn.tensoralgebra import get_bar_gamma_LL

# Hydro
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import PolytropicEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLERiemannSolver
from source.matter.hydro.cons2prim import prim_to_cons


class TOVSolver:
    """TOV solver in polar (Schwarzschild) coordinates."""

    def __init__(self, K, Gamma):
        self.K = K
        self.Gamma = Gamma

    def eos_pressure(self, rho_baryon):
        return self.K * rho_baryon**self.Gamma

    def eos_rho_baryon(self, P):
        if P <= 0:
            return 0.0
        return (P / self.K)**(1.0 / self.Gamma)

    def eos_rho_energy(self, P):
        if P <= 0:
            return 1e-12
        rho_b = self.eos_rho_baryon(P)
        eps = P / ((self.Gamma - 1.0) * rho_b) if rho_b > 0 else 0.0
        return rho_b * (1.0 + eps)

    def tov_rhs(self, r, y):
        """RHS of TOV equations: y = [P, Phi, M]"""
        P, Phi, M = y
        if r < 1e-10:
            return np.array([0.0, 0.0, 0.0])

        rho = self.eos_rho_energy(P)
        denom = r * (r - 2.0 * M)
        if abs(denom) < 1e-30:
            return np.array([0.0, 0.0, 0.0])

        numerator = M + 4.0 * np.pi * r**3 * P
        dP_dr = -(rho + P) * numerator / denom
        dPhi_dr = numerator / denom
        dM_dr = 4.0 * np.pi * r**2 * rho

        return np.array([dP_dr, dPhi_dr, dM_dr])

    def solve(self, rho_central, r_grid=None, r_max=20.0, dr=0.001):
        """
        Solve TOV equations from center outward.

        Args:
            rho_central: Central baryon density
            r_grid: If provided, solve at these exact radial points (recommended!)
            r_max: Maximum radius if r_grid not provided
            dr: Step size if r_grid not provided
        """
        P_c = self.eos_pressure(rho_central)
        Phi_c = 0.0
        M_c = 0.0

        solver = ode(self.tov_rhs).set_integrator('dopri5')
        solver.set_initial_value([P_c, Phi_c, M_c], 0.001)

        if r_grid is not None:
            # Solve directly at grid points (no interpolation!)
            r_arr = []
            P_arr = []
            Phi_arr = []
            M_arr = []

            for r in r_grid:
                if r < 0.001:
                    continue
                solver.integrate(r)
                if not solver.successful():
                    break
                r_arr.append(r)
                P_arr.append(max(solver.y[0], 0.0))
                Phi_arr.append(solver.y[1])
                M_arr.append(solver.y[2])
                if solver.y[0] <= 1e-10:  # Reached surface
                    break

            r_arr = np.array(r_arr)
            P_arr = np.array(P_arr)
            Phi_arr = np.array(Phi_arr)
            M_arr = np.array(M_arr)
        else:
            # Original method with fixed dr
            r_arr, P_arr, Phi_arr, M_arr = [0.001], [P_c], [Phi_c], [M_c]

            while solver.successful() and solver.y[0] > 1e-10 and solver.t < r_max:
                solver.integrate(solver.t + dr)
                r_arr.append(solver.t)
                P_arr.append(solver.y[0])
                Phi_arr.append(solver.y[1])
                M_arr.append(solver.y[2])

            r_arr = np.array(r_arr)
            P_arr = np.array(P_arr)
            Phi_arr = np.array(Phi_arr)
            M_arr = np.array(M_arr)

        rho_arr = np.array([self.eos_rho_baryon(P) for P in P_arr])

        # Surface radius and metric
        R_star = r_arr[-1]
        M_star = M_arr[-1]
        exp4phi = 1.0 / (1.0 - 2.0 * M_arr / r_arr)
        alpha = np.exp(Phi_arr) / np.sqrt(1.0 - 2.0 * M_arr / r_arr)

        return {
            'r': r_arr, 'P': P_arr, 'M': M_arr, 'Phi': Phi_arr,
            'rho_baryon': rho_arr, 'exp4phi': exp4phi, 'alpha': alpha,
            'R': R_star, 'M_star': M_star, 'C': M_star / R_star
        }


def create_initial_data(tov_solution, grid, eos, atmosphere_rho):
    """Create BSSN + hydro initial data from TOV solution."""
    r_tov = tov_solution['r']

    # Check if TOV is on same grid (no interpolation needed!)
    # TOV is solved on positive radii only, so compare with positive portion of grid
    r_grid_positive = grid.r[grid.r > 0]
    same_grid = len(r_tov) <= len(r_grid_positive) and np.allclose(r_tov, r_grid_positive[:len(r_tov)])

    if same_grid:
        # Direct copy - no interpolation errors!
        print("  TOV and evolution grids match - using direct values (no interpolation)")
        rho_tov_vals = tov_solution['rho_baryon']
        P_tov_vals = tov_solution['P']
        Phi_tov_vals = tov_solution['Phi']
        M_tov_vals = tov_solution['M']

        # Initialize arrays
        n_tov = len(r_tov)
        rho_grid = np.zeros(grid.N)
        P_grid = np.zeros(grid.N)
        Phi_grid = np.zeros(grid.N)
        M_grid = np.zeros(grid.N)

        # Find indices where grid.r > 0 (these match TOV grid)
        positive_mask = grid.r > 0
        positive_indices = np.where(positive_mask)[0]

        # Copy TOV values to corresponding positive-r grid points
        rho_grid[positive_indices[:n_tov]] = rho_tov_vals
        P_grid[positive_indices[:n_tov]] = P_tov_vals
        Phi_grid[positive_indices[:n_tov]] = Phi_tov_vals
        M_grid[positive_indices[:n_tov]] = M_tov_vals

        # Atmosphere: negative radii and beyond stellar surface
        rho_grid[~positive_mask] = atmosphere_rho  # Negative radii
        rho_grid[positive_indices[n_tov:]] = atmosphere_rho  # Beyond surface
        P_grid[positive_indices[n_tov:]] = 0.0
        Phi_grid[positive_indices[n_tov:]] = Phi_tov_vals[-1]
        M_grid[positive_indices[n_tov:]] = M_tov_vals[-1]
    else:
        # Need interpolation (less accurate)
        print("  TOV and evolution grids differ - using interpolation")
        rho_tov_interp = interp1d(r_tov, tov_solution['rho_baryon'], kind='cubic',
                                  bounds_error=False, fill_value=(tov_solution['rho_baryon'][0], 0.0))
        P_tov_interp = interp1d(r_tov, tov_solution['P'], kind='cubic',
                                bounds_error=False, fill_value=(tov_solution['P'][0], 0.0))
        Phi_tov_interp = interp1d(r_tov, tov_solution['Phi'], kind='cubic',
                                  bounds_error=False, fill_value=(tov_solution['Phi'][0], 0.0))
        M_tov_interp = interp1d(r_tov, tov_solution['M'], kind='cubic',
                                bounds_error=False, fill_value=(0.0, tov_solution['M'][-1]))

        rho_grid = rho_tov_interp(grid.r)
        P_grid = P_tov_interp(grid.r)
        Phi_grid = Phi_tov_interp(grid.r)
        M_grid = M_tov_interp(grid.r)

    # BSSN variables
    state_2d = np.zeros((grid.NUM_VARS, grid.N))

    for i, r in enumerate(grid.r):
        M = M_grid[i]
        Phi = Phi_grid[i]

        # Conformal factor: exp(4φ) = (1 - 2M/r)^(-1)
        exp4phi = 1.0 / (1.0 - 2.0 * M / r) if r > 2.0 * M else 1.0
        phi = 0.25 * np.log(exp4phi)

        # Lapse: α = exp(Φ) / sqrt(1 - 2M/r)
        alpha = np.exp(Phi) / np.sqrt(1.0 - 2.0 * M / r) if r > 2.0 * M else 1.0

        # Set BSSN vars (rest are zero for spherical symmetry)
        state_2d[0, i] = phi  # conformal factor
        state_2d[idx_lapse, i] = alpha  # lapse

    # Hydro variables
    for i, r in enumerate(grid.r):
        rho = max(rho_grid[i], atmosphere_rho)
        P = P_grid[i]
        M = M_grid[i]

        gamma_rr = 1.0 / (1.0 - 2.0 * M / r) if r > 2.0 * M else 1.0
        D, Sr, tau = prim_to_cons(rho, 0.0, P, gamma_rr, eos)

        state_2d[NUM_BSSN_VARS + 0, i] = D
        state_2d[NUM_BSSN_VARS + 1, i] = Sr
        state_2d[NUM_BSSN_VARS + 2, i] = tau

    return state_2d


def get_rhs_cowling(t, y, grid, background, hydro, bssn_fixed, bssn_d1_fixed):
    """RHS for Cowling evolution (fixed spacetime)."""
    state = y.reshape((grid.NUM_VARS, grid.N))
    grid.fill_boundaries(state)

    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(bssn_fixed)
    hydro.set_matter_vars(state, bssn_vars, grid)

    hydro_rhs = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1_fixed, background)

    # Full RHS (BSSN frozen, only hydro evolves)
    rhs = np.zeros_like(state)
    rhs[NUM_BSSN_VARS:, :] = hydro_rhs
    return rhs.flatten()


def rk4_step(state_flat, dt, grid, background, hydro, bssn_fixed, bssn_d1_fixed):
    """Single RK4 (classical 4th order Runge-Kutta) timestep."""
    # Stage 1
    k1 = get_rhs_cowling(0, state_flat, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    # Stage 2
    state_2 = state_flat + 0.5 * dt * k1
    k2 = get_rhs_cowling(0, state_2, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    # Stage 3
    state_3 = state_flat + 0.5 * dt * k2
    k3 = get_rhs_cowling(0, state_3, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    # Stage 4
    state_4 = state_flat + dt * k3
    k4 = get_rhs_cowling(0, state_4, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    # Combine
    state_new = state_flat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return state_new


def evolve_fixed_timestep(state_initial, dt, num_steps, grid, background, hydro,
                          bssn_fixed, bssn_d1_fixed, method='rk4'):
    """Evolve with fixed timestep using RK4."""
    state_flat = state_initial.flatten()

    for step in range(num_steps):
        state_flat = rk4_step(state_flat, dt, grid, background, hydro, bssn_fixed, bssn_d1_fixed)
        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}/{num_steps}")

    return state_flat.reshape((grid.NUM_VARS, grid.N))


def evolve_adaptive(state_initial, t_final, grid, background, hydro,
                   bssn_fixed, bssn_d1_fixed, method='RK45', rtol=1e-6, atol=1e-8):
    """Evolve with adaptive timestep using scipy.integrate.solve_ivp."""
    from scipy.integrate import solve_ivp

    # Wrapper for RHS compatible with solve_ivp
    def rhs_wrapper(t, y):
        return get_rhs_cowling(t, y, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    state_flat = state_initial.flatten()

    print(f"  Using solve_ivp with method={method}, rtol={rtol}, atol={atol}")

    solution = solve_ivp(
        rhs_wrapper,
        t_span=(0, t_final),
        y0=state_flat,
        method=method,  # 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
        rtol=rtol,
        atol=atol,
        dense_output=False
    )

    print(f"  solve_ivp: {solution.nfev} function evaluations, status={solution.status}")

    return solution.y[:, -1].reshape((grid.NUM_VARS, grid.N))


def plot_tov_diagnostics(tov_solution, r_max):
    """Plot TOV solution diagnostics."""
    r = tov_solution['r']
    R_star = tov_solution['R']
    M_star = tov_solution['M_star']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].semilogy(r, tov_solution['rho_baryon'], color='navy')
    axes[0, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5, label=f'R={R_star:.2f}')
    axes[0, 0].set_xlabel(r"$r$")
    axes[0, 0].set_ylabel(r"$\rho_0$")
    axes[0, 0].set_title('Baryon Density')
    axes[0, 0].set_xlim(0, r_max)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(r, tov_solution['P'], color='darkgreen')
    axes[0, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel(r"$r$")
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure')
    axes[0, 1].set_xlim(0, r_max)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(r, tov_solution['M'], color='maroon')
    axes[1, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].axhline(M_star, color='gray', linestyle='--', alpha=0.3, label=f'M={M_star:.3f}')
    axes[1, 0].set_xlabel(r"$r$")
    axes[1, 0].set_ylabel('M(r)')
    axes[1, 0].set_title('Enclosed Mass')
    axes[1, 0].set_xlim(0, r_max)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(r, tov_solution['alpha'], color='purple')
    axes[1, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel(r"$r$")
    axes[1, 1].set_ylabel(r'$\alpha$')
    axes[1, 1].set_title('Lapse Function')
    axes[1, 1].set_xlim(0, r_max)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tov_solution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_initial_comparison(tov_solution, initial_state_2d, grid, hydro):
    """Plot initial data vs TOV comparison."""
    # Get primitives from initial data
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(initial_state_2d, bssn_vars, grid)
    prim = hydro._get_primitives(bssn_vars, grid.r)

    r_tov = tov_solution['r']
    r_grid = grid.r[NUM_GHOSTS:-NUM_GHOSTS]
    rho_grid = prim['rho0'][NUM_GHOSTS:-NUM_GHOSTS]
    P_grid = prim['p'][NUM_GHOSTS:-NUM_GHOSTS]
    v_grid = prim['vr'][NUM_GHOSTS:-NUM_GHOSTS]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Density
    axes[0, 0].semilogy(r_tov, tov_solution['rho_baryon'], 'b-', linewidth=2, label='TOV')
    axes[0, 0].semilogy(r_grid, np.maximum(rho_grid, 1e-20), 'r--', linewidth=1.5, alpha=0.7, label='Initial data (t=0)')
    axes[0, 0].axvline(tov_solution['R'], color='gray', linestyle=':', alpha=0.5, label=f"R={tov_solution['R']:.2f}")
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel(r'$\rho_0$')
    axes[0, 0].set_title('Baryon Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Pressure
    axes[0, 1].semilogy(r_tov, tov_solution['P'], 'b-', linewidth=2, label='TOV')
    axes[0, 1].semilogy(r_grid, np.maximum(P_grid, 1e-20), 'r--', linewidth=1.5, alpha=0.7, label='Initial data (t=0)')
    axes[0, 1].axvline(tov_solution['R'], color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Velocity (should be zero initially)
    axes[1, 0].plot(r_grid, v_grid, 'r-', linewidth=2, label='Initial data (t=0)')
    axes[1, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(tov_solution['R'], color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel(r'$v^r$')
    axes[1, 0].set_title('Radial Velocity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Lapse
    alpha_grid = initial_state_2d[idx_lapse, NUM_GHOSTS:-NUM_GHOSTS]
    axes[1, 1].plot(r_tov, tov_solution['alpha'], 'b-', linewidth=2, label='TOV')
    axes[1, 1].plot(r_grid, alpha_grid, 'r--', linewidth=1.5, alpha=0.7, label='Initial data (t=0)')
    axes[1, 1].axvline(tov_solution['R'], color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel(r'$\alpha$')
    axes[1, 1].set_title('Lapse Function')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('TOV vs Initial Data (t=0)', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig('tov_initial_data_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_evolution(state_t0, state_t1, state_t100, grid, hydro, dt, num_steps):
    """Plot evolution snapshots: t=0, t=1*dt, t=100*dt."""
    bssn_0 = BSSNVars(grid.N)
    bssn_0.set_bssn_vars(state_t0[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t0, bssn_0, grid)
    prim_0 = hydro._get_primitives(bssn_0, grid.r)

    bssn_1 = BSSNVars(grid.N)
    bssn_1.set_bssn_vars(state_t1[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t1, bssn_1, grid)
    prim_1 = hydro._get_primitives(bssn_1, grid.r)

    bssn_100 = BSSNVars(grid.N)
    bssn_100.set_bssn_vars(state_t100[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t100, bssn_100, grid)
    prim_100 = hydro._get_primitives(bssn_100, grid.r)

    r_int = grid.r[NUM_GHOSTS:-NUM_GHOSTS]
    t_final = num_steps * dt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Density
    axes[0, 0].plot(r_int, prim_0['rho0'][NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label='t=0')
    axes[0, 0].plot(r_int, prim_1['rho0'][NUM_GHOSTS:-NUM_GHOSTS], 'orange', linestyle='--', linewidth=1.5, label=f't=1×dt')
    axes[0, 0].plot(r_int, prim_100['rho0'][NUM_GHOSTS:-NUM_GHOSTS], 'red', linestyle=':', linewidth=1.5, label=f't={num_steps}×dt')
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel(r'$\rho_0$')
    axes[0, 0].set_title('Baryon Density Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Pressure
    axes[0, 1].semilogy(r_int, np.maximum(prim_0['p'][NUM_GHOSTS:-NUM_GHOSTS], 1e-20), 'b-', linewidth=2, label='t=0')
    axes[0, 1].semilogy(r_int, np.maximum(prim_1['p'][NUM_GHOSTS:-NUM_GHOSTS], 1e-20), 'orange', linestyle='--', linewidth=1.5, label=f't=1×dt')
    axes[0, 1].semilogy(r_int, np.maximum(prim_100['p'][NUM_GHOSTS:-NUM_GHOSTS], 1e-20), 'red', linestyle=':', linewidth=1.5, label=f't={num_steps}×dt')
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Velocity
    axes[1, 0].plot(r_int, prim_0['vr'][NUM_GHOSTS:-NUM_GHOSTS], 'b-', linewidth=2, label='t=0')
    axes[1, 0].plot(r_int, prim_1['vr'][NUM_GHOSTS:-NUM_GHOSTS], 'orange', linestyle='--', linewidth=1.5, label=f't=1×dt')
    axes[1, 0].plot(r_int, prim_100['vr'][NUM_GHOSTS:-NUM_GHOSTS], 'red', linestyle=':', linewidth=1.5, label=f't={num_steps}×dt')
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel(r'$v^r$')
    axes[1, 0].set_title('Radial Velocity Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Relative density error
    delta_rho_1 = np.abs(prim_1['rho0'][NUM_GHOSTS:-NUM_GHOSTS] - prim_0['rho0'][NUM_GHOSTS:-NUM_GHOSTS]) / (np.abs(prim_0['rho0'][NUM_GHOSTS:-NUM_GHOSTS]) + 1e-20)
    delta_rho_100 = np.abs(prim_100['rho0'][NUM_GHOSTS:-NUM_GHOSTS] - prim_0['rho0'][NUM_GHOSTS:-NUM_GHOSTS]) / (np.abs(prim_0['rho0'][NUM_GHOSTS:-NUM_GHOSTS]) + 1e-20)

    axes[1, 1].semilogy(r_int, delta_rho_1, 'orange', linestyle='--', linewidth=1.5, label=f't=1×dt')
    axes[1, 1].semilogy(r_int, delta_rho_100, 'red', linestyle=':', linewidth=1.5, label=f't={num_steps}×dt')
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel(r'$|\Delta\rho|/\rho$')
    axes[1, 1].set_title('Relative Density Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Evolution: t=0 → t={dt:.4f} → t={t_final:.4f}', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig('tov_evolution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    """Main execution."""
    print("="*70)
    print("TOV Star Evolution - Cowling Approximation")
    print("="*70)

    # ==================================================================
    # CONFIGURATION
    # ==================================================================
    r_max = 11.0
    num_points = 1000  # Use 4000+ for production runs
    K = 500.0
    Gamma = 2.5  # NOTE: Gamma=2.0 is pathological (tau→0)
    rho_central = 1.28e-3
    atmosphere_rho = 1.0e-10

    # Time integration method
    # 'fixed': RK4 with fixed timestep (fast, stable)
    # 'adaptive': solve_ivp with adaptive timestep (slower, more accurate)
    integration_method = 'fixed'

    # ==================================================================
    # SETUP
    # ==================================================================
    spacing = LinearSpacing(num_points, r_max)
    eos = PolytropicEOS(K=K, gamma=Gamma)
    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere_rho=atmosphere_rho,
        reconstructor=create_reconstruction("mp5"),
        riemann_solver=HLLERiemannSolver()
    )

    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    print(f"Grid: N={grid.N}, r_max={r_max}, dr_min={grid.min_dr}")
    print(f"EOS: K={K}, Gamma={Gamma}\n")

    # ==================================================================
    # SOLVE TOV
    # ==================================================================
    print("Solving TOV equations on evolution grid (no interpolation)...")
    tov_solver = TOVSolver(K=K, Gamma=Gamma)
    r_positive = grid.r[grid.r > 0]  # Only use positive radii for TOV
    tov_solution = tov_solver.solve(rho_central, r_grid=r_positive, r_max=r_max)
    print(f"TOV Solution: M={tov_solution['M_star']:.6f}, R={tov_solution['R']:.3f}, C={tov_solution['C']:.4f}\n")

    plot_tov_diagnostics(tov_solution, r_max)

    # ==================================================================
    # INITIAL DATA
    # ==================================================================
    print("Creating initial data...")
    initial_state_2d = create_initial_data(tov_solution, grid, eos, atmosphere_rho)

    plot_initial_comparison(tov_solution, initial_state_2d, grid, hydro)

    # ==================================================================
    # EVOLUTION
    # ==================================================================
    bssn_fixed = initial_state_2d[:NUM_BSSN_VARS, :].copy()
    bssn_d1_fixed = grid.get_d1_metric_quantities(initial_state_2d)

    # Free derivative matrices after computing d1 (saves ~N² memory for Cowling)
    # For N=10000: saves ~12 GB, keeping only ~6 MB in bssn_d1_fixed
    del grid.derivs.drn_matrix
    del grid.derivs.dxn_matrix
    del grid.derivs.advec_x_matrix
    import gc
    gc.collect()

    if integration_method == 'fixed':
        dt = 0.5 * grid.min_dr  # CFL condition
        num_steps = 100
        print(f"\nEvolving with fixed dt={dt:.6f} (CFL=0.5) for {num_steps} steps using RK4")

        # Single step for comparison
        state_t1 = rk4_step(initial_state_2d.flatten(), dt, grid, background, hydro,
                           bssn_fixed, bssn_d1_fixed).reshape((grid.NUM_VARS, grid.N))

        # Multiple steps
        state_t100 = evolve_fixed_timestep(initial_state_2d, dt, num_steps, grid, background,
                                          hydro, bssn_fixed, bssn_d1_fixed, method='rk4')

    elif integration_method == 'adaptive':
        # NOTE: Adaptive methods are slower but can be more accurate
        # Available methods: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
        t_final = 0.5  # Final time
        print(f"\nEvolving to t={t_final} using solve_ivp (adaptive)")

        # For single step comparison, use small fixed dt
        dt = 0.5 * grid.min_dr
        state_t1 = rk4_step(initial_state_2d.flatten(), dt, grid, background, hydro,
                           bssn_fixed, bssn_d1_fixed).reshape((grid.NUM_VARS, grid.N))

        # Adaptive evolution
        state_t100 = evolve_adaptive(initial_state_2d, t_final, grid, background, hydro,
                                    bssn_fixed, bssn_d1_fixed, method='DOP853', rtol=1e-5, atol=1e-7)

    # ==================================================================
    # DIAGNOSTICS
    # ==================================================================
    plot_evolution(initial_state_2d, state_t1, state_t100, grid, hydro, dt, num_steps)

    # Print detailed statistics
    bssn_0 = BSSNVars(grid.N)
    bssn_0.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(initial_state_2d, bssn_0, grid)
    prim_0 = hydro._get_primitives(bssn_0, grid.r)

    bssn_1 = BSSNVars(grid.N)
    bssn_1.set_bssn_vars(state_t1[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t1, bssn_1, grid)
    prim_1 = hydro._get_primitives(bssn_1, grid.r)

    bssn_100 = BSSNVars(grid.N)
    bssn_100.set_bssn_vars(state_t100[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t100, bssn_100, grid)
    prim_100 = hydro._get_primitives(bssn_100, grid.r)

    # Calculate actual final time
    if integration_method == 'fixed':
        t_final_actual = num_steps * dt
    else:
        t_final_actual = t_final

    # Interior points only (exclude ghosts)
    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

    # Compute errors
    delta_rho_1 = np.abs(prim_1['rho0'][interior] - prim_0['rho0'][interior]) / (np.abs(prim_0['rho0'][interior]) + 1e-20)
    delta_rho_100 = np.abs(prim_100['rho0'][interior] - prim_0['rho0'][interior]) / (np.abs(prim_0['rho0'][interior]) + 1e-20)

    delta_P_1 = np.abs(prim_1['p'][interior] - prim_0['p'][interior]) / (np.abs(prim_0['p'][interior]) + 1e-20)
    delta_P_100 = np.abs(prim_100['p'][interior] - prim_0['p'][interior]) / (np.abs(prim_0['p'][interior]) + 1e-20)

    # Error growth factor
    max_err_rho_1 = np.max(delta_rho_1)
    max_err_rho_100 = np.max(delta_rho_100)
    growth_rho = max_err_rho_100 / max_err_rho_1 if max_err_rho_1 > 1e-15 else 0

    max_err_P_1 = np.max(delta_P_1)
    max_err_P_100 = np.max(delta_P_100)
    growth_P = max_err_P_100 / max_err_P_1 if max_err_P_1 > 1e-15 else 0

    print(f"\n{'='*70}")
    print(f"EVOLUTION DIAGNOSTICS (t=0 → t={dt:.4f} → t={t_final_actual:.4f})")
    print(f"{'='*70}")

    print(f"\n1. VELOCITY EVOLUTION:")
    print(f"   Max |v^r| at t=0:       {np.max(np.abs(prim_0['vr'])):.3e}")
    print(f"   Max |v^r| at t=1×dt:    {np.max(np.abs(prim_1['vr'])):.3e}")
    print(f"   Max |v^r| at t={num_steps}×dt:   {np.max(np.abs(prim_100['vr'])):.3e}")

    print(f"\n2. CENTRAL DENSITY:")
    print(f"   ρ_c at t=0:       {prim_0['rho0'][NUM_GHOSTS]:.6e}")
    print(f"   ρ_c at t=1×dt:    {prim_1['rho0'][NUM_GHOSTS]:.6e}")
    print(f"   ρ_c at t={num_steps}×dt:   {prim_100['rho0'][NUM_GHOSTS]:.6e}")
    print(f"   Δρ_c/ρ_c (1 step):   {abs(prim_1['rho0'][NUM_GHOSTS] - prim_0['rho0'][NUM_GHOSTS])/prim_0['rho0'][NUM_GHOSTS]:.3e}")
    print(f"   Δρ_c/ρ_c ({num_steps} steps): {abs(prim_100['rho0'][NUM_GHOSTS] - prim_0['rho0'][NUM_GHOSTS])/prim_0['rho0'][NUM_GHOSTS]:.3e}")

    print(f"\n3. DENSITY ERROR (max over domain):")
    print(f"   Max |Δρ|/ρ at t=1×dt:    {max_err_rho_1:.3e}")
    print(f"   Max |Δρ|/ρ at t={num_steps}×dt:   {max_err_rho_100:.3e}")
    print(f"   Growth factor (100/1):   {growth_rho:.1f}x")

    print(f"\n4. PRESSURE ERROR (max over domain):")
    print(f"   Max |ΔP|/P at t=1×dt:    {max_err_P_1:.3e}")
    print(f"   Max |ΔP|/P at t={num_steps}×dt:   {max_err_P_100:.3e}")
    print(f"   Growth factor (100/1):   {growth_P:.1f}x")

    print(f"\n5. CONS2PRIM STATUS:")
    print(f"   Success at t=0:      {np.sum(prim_0['success'])}/{grid.N}")
    print(f"   Success at t=1×dt:   {np.sum(prim_1['success'])}/{grid.N}")
    print(f"   Success at t={num_steps}×dt:  {np.sum(prim_100['success'])}/{grid.N}")

    if not np.all(prim_100['success']):
        failed_idx = np.where(~prim_100['success'])[0]
        print(f"   Failed points: {failed_idx[:5]} (first 5)")
        print(f"   Failed radii:  {grid.r[failed_idx[:5]]}")

    print("\n" + "="*70)
    print("Evolution complete. Plots saved:")
    print("  1. tov_solution.png              - TOV solution (ρ, P, M, α)")
    print("  2. tov_initial_data_comparison.png - TOV vs Initial data at t=0")
    print("  3. tov_evolution.png             - Evolution: t=0 → t=dt → t=100×dt")
    print("="*70)


if __name__ == "__main__":
    main()
