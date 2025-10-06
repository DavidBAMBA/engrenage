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
from source.backgrounds.sphericalbackground import FlatSphericalBackground, i_r, i_t, i_p

# BSSN
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import (NUM_BSSN_VARS, idx_phi, idx_hrr, idx_htt, idx_hpp,
                                             idx_K, idx_arr, idx_att, idx_app, idx_lapse)
from source.bssn.tensoralgebra import get_bar_gamma_LL

# Hydro
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import PolytropicEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLERiemannSolver
from source.matter.hydro.cons2prim import prim_to_cons


class TOVSolver:
    """TOV solver with isotropic coordinates (following NRPy+ implementation)."""

    def __init__(self, K, Gamma, use_isotropic=True):
        self.K = K
        self.Gamma = Gamma
        self.use_isotropic = use_isotropic

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

    def tov_rhs(self, r_schw, y):
        """
        RHS of TOV equations in Schwarzschild radius r_schw.

        If use_isotropic=True:
            y = [P, nu, M, r_iso]
        Else (Schwarzschild):
            y = [P, Phi, M]
        """
        if self.use_isotropic:
            P, nu, M, r_iso = y
        else:
            P, Phi, M = y

        if r_schw < 1e-10:
            if self.use_isotropic:
                return np.array([0.0, 0.0, 0.0, 1.0])  # dr_iso/dr = 1 at center
            else:
                return np.array([0.0, 0.0, 0.0])

        rho = self.eos_rho_energy(P)
        denom = r_schw * (r_schw - 2.0 * M)
        if abs(denom) < 1e-30:
            if self.use_isotropic:
                return np.array([0.0, 0.0, 0.0, 0.0])
            else:
                return np.array([0.0, 0.0, 0.0])

        numerator = M + 4.0 * np.pi * r_schw**3 * P

        # Common equations
        dP_dr = -(rho + P) * numerator / denom
        dnu_dr = numerator / denom
        dM_dr = 4.0 * np.pi * r_schw**2 * rho

        if self.use_isotropic:
            # Isotropic radius evolution: dr_iso/dr = r_iso / (r * sqrt(1 - 2M/r))
            if r_iso > 0 and r_schw > 2.0 * M:
                dr_iso_dr = r_iso / (r_schw * np.sqrt(1.0 - 2.0 * M / r_schw))
            else:
                dr_iso_dr = 1.0  # At center
            return np.array([dP_dr, dnu_dr, dM_dr, dr_iso_dr])
        else:
            return np.array([dP_dr, dnu_dr, dM_dr])

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
        nu_c = 0.0
        M_c = 0.0
        r_iso_c = 0.0

        solver = ode(self.tov_rhs).set_integrator('dopri5')
        if self.use_isotropic:
            solver.set_initial_value([P_c, nu_c, M_c, r_iso_c], 0.001)
        else:
            solver.set_initial_value([P_c, nu_c, M_c], 0.001)

        if r_grid is not None:
            # Solve directly at grid points (no interpolation!)
            r_schw_arr = []
            P_arr = []
            nu_arr = []
            M_arr = []
            r_iso_arr = [] if self.use_isotropic else None
            surface_found = False
            R_star = None
            M_star = None
            nu_star = None
            r_iso_star = None

            for r_schw in r_grid:
                if r_schw < 0.001:
                    continue

                if not surface_found:
                    solver.integrate(r_schw)
                    if not solver.successful():
                        break
                    r_schw_arr.append(r_schw)
                    P_arr.append(max(solver.y[0], 0.0))
                    nu_arr.append(solver.y[1])
                    M_arr.append(solver.y[2])
                    if self.use_isotropic:
                        r_iso_arr.append(solver.y[3])

                    if solver.y[0] <= 1e-10:  # Reached surface
                        surface_found = True
                        R_star = r_schw
                        M_star = solver.y[2]
                        nu_star = solver.y[1]
                        if self.use_isotropic:
                            r_iso_star = solver.y[3]
                else:
                    # Beyond surface: vacuum Schwarzschild solution
                    r_schw_arr.append(r_schw)
                    P_arr.append(0.0)
                    # Schwarzschild exterior: α = sqrt(1 - 2M/r)
                    # Since α = exp(nu) / sqrt(1 - 2M/r), we need:
                    # exp(nu) = (1 - 2M/r), so nu = ln(1 - 2M/r)
                    # Add offset for continuity: nu(r) = ln(1-2M/r) + [nu_star - ln(1-2M/R)]
                    if r_schw > 2.0 * M_star and R_star > 2.0 * M_star:
                        nu_ext = np.log(1.0 - 2.0*M_star/r_schw) + (nu_star - np.log(1.0 - 2.0*M_star/R_star))
                    else:
                        nu_ext = nu_star
                    nu_arr.append(nu_ext)
                    M_arr.append(M_star)

                    # Isotropic radius in exterior
                    if self.use_isotropic and r_iso_star is not None:
                        # Continue r_iso evolution in exterior
                        # dr_iso/dr = r_iso / (r * sqrt(1 - 2M/r))
                        # Integrate: r_iso = r_iso_star * exp(integral)
                        # For Schwarzschild: r_iso ~ r_iso_surface * (r/R) * sqrt((r-2M)/(R-2M))
                        if r_schw > 2.0 * M_star:
                            factor = (r_schw / R_star) * np.sqrt((r_schw - 2.0*M_star) / (R_star - 2.0*M_star))
                            r_iso_ext = r_iso_star * factor
                        else:
                            r_iso_ext = r_iso_star
                        r_iso_arr.append(r_iso_ext)

            r_schw_arr = np.array(r_schw_arr)
            P_arr = np.array(P_arr)
            nu_arr = np.array(nu_arr)
            M_arr = np.array(M_arr)
            if self.use_isotropic:
                r_iso_arr = np.array(r_iso_arr)
        else:
            # Original method with fixed dr
            r_schw_arr = [0.001]
            P_arr = [P_c]
            nu_arr = [nu_c]
            M_arr = [M_c]
            r_iso_arr = [r_iso_c] if self.use_isotropic else None

            # Integrate until surface
            while solver.successful() and solver.y[0] > 1e-10 and solver.t < r_max:
                solver.integrate(solver.t + dr)
                r_schw_arr.append(solver.t)
                P_arr.append(solver.y[0])
                nu_arr.append(solver.y[1])
                M_arr.append(solver.y[2])
                if self.use_isotropic:
                    r_iso_arr.append(solver.y[3])

            # Record surface values
            R_star = r_schw_arr[-1]
            M_star = M_arr[-1]
            nu_star = nu_arr[-1]
            r_iso_star = r_iso_arr[-1] if self.use_isotropic else None

            # Continue to r_max with vacuum solution
            r_current = R_star + dr
            while r_current <= r_max:
                r_schw_arr.append(r_current)
                P_arr.append(0.0)
                # Schwarzschild exterior
                if r_current > 2.0 * M_star and R_star > 2.0 * M_star:
                    nu_ext = np.log(1.0 - 2.0*M_star/r_current) + (nu_star - np.log(1.0 - 2.0*M_star/R_star))
                else:
                    nu_ext = nu_star
                nu_arr.append(nu_ext)
                M_arr.append(M_star)

                # Isotropic radius in exterior
                if self.use_isotropic and r_iso_star is not None:
                    if r_current > 2.0 * M_star:
                        factor = (r_current / R_star) * np.sqrt((r_current - 2.0*M_star) / (R_star - 2.0*M_star))
                        r_iso_ext = r_iso_star * factor
                    else:
                        r_iso_ext = r_iso_star
                    r_iso_arr.append(r_iso_ext)

                r_current += dr

            r_schw_arr = np.array(r_schw_arr)
            P_arr = np.array(P_arr)
            nu_arr = np.array(nu_arr)
            M_arr = np.array(M_arr)
            if self.use_isotropic:
                r_iso_arr = np.array(r_iso_arr)

        rho_arr = np.array([self.eos_rho_baryon(P) for P in P_arr])

        # Surface radius: find where P first drops below threshold
        surface_idx = np.where(P_arr <= 1e-10)[0]
        if len(surface_idx) > 0:
            R_star_schw = r_schw_arr[surface_idx[0]]
        else:
            R_star_schw = r_schw_arr[-1]  # Fallback if no surface found

        M_star = M_arr[-1]  # Total mass (same everywhere after surface)

        if self.use_isotropic:
            # Normalize r_iso following NRPy+ (line 318 in TOVola_solve.py)
            R_iso_surface = r_iso_arr[surface_idx[0]] if len(surface_idx) > 0 else r_iso_arr[-1]
            normalize = 0.5 * (np.sqrt(R_star_schw * (R_star_schw - 2.0 * M_star)) + R_star_schw - M_star) / R_iso_surface
            r_iso_normalized = r_iso_arr * normalize

            # Conformal factor in isotropic coords: exp(4φ) = (r_schw / r_iso)²
            exp4phi = (r_schw_arr / r_iso_normalized)**2

            # Normalize lapse (line 323 in TOVola_solve.py)
            nu_surface = nu_arr[surface_idx[0]] if len(surface_idx) > 0 else nu_arr[-1]
            expnu = np.exp(nu_arr - nu_surface + np.log(1.0 - 2.0 * M_star / R_star_schw))
            alpha = expnu

            # Use normalized isotropic radius as coordinate
            r_coord = r_iso_normalized
            R_star = R_iso_surface * normalize
        else:
            # Schwarzschild coordinates
            exp4phi = 1.0 / (1.0 - 2.0 * M_arr / r_schw_arr)
            alpha = np.exp(nu_arr) / np.sqrt(1.0 - 2.0 * M_arr / r_schw_arr)
            r_coord = r_schw_arr
            R_star = R_star_schw

        result = {
            'r': r_coord, 'P': P_arr, 'M': M_arr, 'nu': nu_arr,
            'rho_baryon': rho_arr, 'exp4phi': exp4phi, 'alpha': alpha,
            'R': R_star, 'M_star': M_star, 'C': M_star / R_star
        }

        if self.use_isotropic:
            result['r_schw'] = r_schw_arr
            result['r_iso'] = r_iso_normalized

        return result


def create_initial_data(tov_solution, grid, background, eos, atmosphere_rho):
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
        nu_tov_vals = tov_solution['nu']
        M_tov_vals = tov_solution['M']

        # Initialize arrays
        n_tov = len(r_tov)
        rho_grid = np.zeros(grid.N)
        P_grid = np.zeros(grid.N)
        nu_grid = np.zeros(grid.N)
        M_grid = np.zeros(grid.N)

        # Find indices where grid.r > 0 (these match TOV grid)
        positive_mask = grid.r > 0
        positive_indices = np.where(positive_mask)[0]

        # Copy TOV values to corresponding positive-r grid points
        rho_grid[positive_indices[:n_tov]] = rho_tov_vals
        P_grid[positive_indices[:n_tov]] = P_tov_vals
        nu_grid[positive_indices[:n_tov]] = nu_tov_vals
        M_grid[positive_indices[:n_tov]] = M_tov_vals

        # Atmosphere: negative radii and beyond stellar surface
        rho_grid[~positive_mask] = atmosphere_rho  # Negative radii
        rho_grid[positive_indices[n_tov:]] = atmosphere_rho  # Beyond surface
        P_grid[positive_indices[n_tov:]] = 0.0
        nu_grid[positive_indices[n_tov:]] = nu_tov_vals[-1]
        M_grid[positive_indices[n_tov:]] = M_tov_vals[-1]
    else:
        # Need interpolation (less accurate)
        print("  TOV and evolution grids differ - using interpolation")
        rho_tov_interp = interp1d(r_tov, tov_solution['rho_baryon'], kind='cubic',
                                  bounds_error=False, fill_value=(tov_solution['rho_baryon'][0], 0.0))
        P_tov_interp = interp1d(r_tov, tov_solution['P'], kind='cubic',
                                bounds_error=False, fill_value=(tov_solution['P'][0], 0.0))
        nu_tov_interp = interp1d(r_tov, tov_solution['nu'], kind='cubic',
                                  bounds_error=False, fill_value=(tov_solution['nu'][0], 0.0))
        M_tov_interp = interp1d(r_tov, tov_solution['M'], kind='cubic',
                                bounds_error=False, fill_value=(0.0, tov_solution['M'][-1]))

        rho_grid = rho_tov_interp(grid.r)
        P_grid = P_tov_interp(grid.r)
        nu_grid = nu_tov_interp(grid.r)
        M_grid = M_tov_interp(grid.r)

    # Interpolate exp4phi and alpha to grid
    exp4phi_interp = interp1d(r_tov, tov_solution['exp4phi'], kind='cubic',
                              bounds_error=False, fill_value=(1.0, tov_solution['exp4phi'][-1]))
    alpha_interp = interp1d(r_tov, tov_solution['alpha'], kind='cubic',
                            bounds_error=False, fill_value=(1.0, tov_solution['alpha'][-1]))

    exp4phi_grid = exp4phi_interp(grid.r)
    alpha_grid = alpha_interp(grid.r)

    # Get reference metric from background (ĝ_ij)
    ghatDD = background.hat_gamma_LL  # Shape: [N, 3, 3]
    ReDD = background.scaling_matrix  # Rescaling matrix s_i s_j

    # BSSN variables
    state_2d = np.zeros((grid.NUM_VARS, grid.N))

    for i, r in enumerate(grid.r):
        # Physical metric from TOV (in isotropic coordinates)
        # γ_rr = exp(4φ_TOV), γ_θθ = r² exp(4φ_TOV), γ_φφ = r² sin²θ exp(4φ_TOV)
        exp4phi_tov = exp4phi_grid[i]
        gamma_phys_rr = exp4phi_tov
        gamma_phys_thth = r**2 * exp4phi_tov
        gamma_phys_phph = r**2 * exp4phi_tov  # sin²θ = 1 in equatorial plane

        # Determinant of physical metric
        det_gamma_phys = gamma_phys_rr * gamma_phys_thth * gamma_phys_phph

        # Determinant of reference metric ĝ_ij
        det_ghat = ghatDD[i, i_r, i_r] * ghatDD[i, i_t, i_t] * ghatDD[i, i_p, i_p]

        # Conformal factor: φ = (1/12) ln(det(γ_phys) / det(ĝ))
        # Following NRPy+ convention (BSSN_in_terms_of_ADM.py line 134-135)
        if det_gamma_phys > 0 and det_ghat > 0:
            phi = (1.0/12.0) * np.log(det_gamma_phys / det_ghat)
        else:
            phi = 0.0

        # Conformal metric: γ̄_ij = e^(-4φ) γ_ij
        exp_minus_4phi = np.exp(-4.0 * phi)
        gammabar_rr = exp_minus_4phi * gamma_phys_rr
        gammabar_thth = exp_minus_4phi * gamma_phys_thth
        gammabar_phph = exp_minus_4phi * gamma_phys_phph

        # Deviation from reference: h_ij = (γ̄_ij - ĝ_ij) / ReDD_ij
        # Following engrenage convention (bssnstatevariables.py line 18-20)
        h_rr = (gammabar_rr - ghatDD[i, i_r, i_r]) / ReDD[i, i_r, i_r] if ReDD[i, i_r, i_r] != 0 else 0.0
        h_tt = (gammabar_thth - ghatDD[i, i_t, i_t]) / ReDD[i, i_t, i_t] if ReDD[i, i_t, i_t] != 0 else 0.0
        h_pp = (gammabar_phph - ghatDD[i, i_p, i_p]) / ReDD[i, i_p, i_p] if ReDD[i, i_p, i_p] != 0 else 0.0

        # TOV is static equilibrium: K_ij = 0 everywhere
        # Following NRPy+ prescription for static initial data
        trK = 0.0
        a_rr = 0.0
        a_tt = 0.0
        a_pp = 0.0

        # Lapse from TOV
        alpha = alpha_grid[i]

        # Assign BSSN variables to state vector
        state_2d[idx_phi, i] = phi
        state_2d[idx_hrr, i] = h_rr
        state_2d[idx_htt, i] = h_tt
        state_2d[idx_hpp, i] = h_pp
        state_2d[idx_K, i] = trK
        state_2d[idx_arr, i] = a_rr
        state_2d[idx_att, i] = a_tt
        state_2d[idx_app, i] = a_pp
        state_2d[idx_lapse, i] = alpha
        # lambdar, shiftr, br remain zero (will be computed from FD derivatives)

    # Hydro variables
    for i, r in enumerate(grid.r):
        rho = max(rho_grid[i], atmosphere_rho)
        P = P_grid[i]

        # Use physical metric from TOV for cons2prim (isotropic coordinates)
        gamma_rr_phys = exp4phi_grid[i]
        D, Sr, tau = prim_to_cons(rho, 0.0, P, gamma_rr_phys, eos)

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

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    axes[0, 0].plot(r, tov_solution['rho_baryon'], color='navy')
    axes[0, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5, label=f'R={R_star:.2f}')
    axes[0, 0].set_xlabel(r"$r$")
    axes[0, 0].set_ylabel(r"$\rho_0$")
    axes[0, 0].set_title('Baryon Density')
    axes[0, 0].set_xlim(0, r_max)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(r, tov_solution['P'], color='darkgreen')
    axes[0, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel(r"$r$")
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure')
    axes[0, 1].set_xlim(0, r_max)
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(r, tov_solution['M'], color='maroon')
    axes[0, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 2].axhline(M_star, color='gray', linestyle='--', alpha=0.3, label=f'M={M_star:.3f}')
    axes[0, 2].set_xlabel(r"$r$")
    axes[0, 2].set_ylabel('M(r)')
    axes[0, 2].set_title('Enclosed Mass')
    axes[0, 2].set_xlim(0, r_max)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(r, tov_solution['alpha'], color='purple')
    axes[1, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel(r"$r$")
    axes[1, 0].set_ylabel(r'$\alpha$')
    axes[1, 0].set_title('Lapse Function')
    axes[1, 0].set_xlim(0, r_max)
    axes[1, 0].grid(True, alpha=0.3)

    # Conformal factor exp(4φ)
    axes[1, 1].plot(r, tov_solution['exp4phi'], color='orange')
    axes[1, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel(r"$r$")
    axes[1, 1].set_ylabel(r'$e^{4\phi}$')
    axes[1, 1].set_title('Conformal Factor')
    axes[1, 1].set_xlim(0, r_max)
    axes[1, 1].grid(True, alpha=0.3)

    # φ itself
    phi = 0.25 * np.log(tov_solution['exp4phi'])
    axes[1, 2].plot(r, phi, color='teal')
    axes[1, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 2].set_xlabel(r"$r$")
    axes[1, 2].set_ylabel(r'$\phi$')
    axes[1, 2].set_title('Conformal Factor φ')
    axes[1, 2].set_xlim(0, r_max)
    axes[1, 2].grid(True, alpha=0.3)

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
    initial_state_2d = create_initial_data(tov_solution, grid, background, eos, atmosphere_rho)

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
