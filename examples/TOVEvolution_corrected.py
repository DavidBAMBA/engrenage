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

import csv
import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.integrate import ode, trapezoid
from scipy.interpolate import interp1d

# Engrenage core
sys.path.insert(0, '/home/yo/repositories/engrenage')
from source.core.grid import Grid
from source.core.spacing import CubicSpacing, LinearSpacing,  NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground, i_r

# BSSN
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS, idx_lapse
from source.bssn.tensoralgebra import get_bar_gamma_LL

# Hydro
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLERiemannSolver


class TOVSolver:
    """
    TOV solver following NRPy+ equations.

    Equations (from nrpy/equations/tov/TOV_equations.py):
    - dP/dr = -(ρ + P)(2M/r + 8πr²P) / [r(1 - 2M/r)]
    - dM/dr = 4πr²ρ
    - dν/dr = (2M/r + 8πr²P) / [r(1 - 2M/r)]
    - dr_iso/dr = r_iso / [r √(1 - 2M/r)]
    """

    def __init__(self, K, Gamma):
        self.K = K
        self.Gamma = Gamma

    def eos_pressure(self, rho_baryon):
        """P = K * rho_baryon^Gamma"""
        return self.K * rho_baryon**self.Gamma

    def eos_rho_baryon(self, P):
        """rho_baryon = (P/K)^(1/Gamma)"""
        if P <= 0:
            return 0.0
        return (P / self.K)**(1.0 / self.Gamma)

    def eos_rho_energy(self, P):
        """Total energy density: rho_energy = rho_baryon * (1 + eps)"""
        if P <= 0:
            return 1e-30
        rho_b = self.eos_rho_baryon(P)
        eps = P / ((self.Gamma - 1.0) * rho_b) if rho_b > 0 else 0.0
        return rho_b * (1.0 + eps)

    def tov_rhs(self, r_Schw, y):
        """
        RHS of TOV equations matching NRPy+.

        State vector y = [P, nu, M, r_iso]
        """
        P, nu, M, r_iso = y

        # Handle center
        if r_Schw == 0:
            return np.array([0.0, 0.0, 0.0, 1.0])

        rho_energy = self.eos_rho_energy(P)

        # TOV equations (NRPy+ form)
        denom = r_Schw * (1.0 - 2.0 * M / r_Schw)
        if abs(denom) < 1e-30:
            return np.array([0.0, 0.0, 0.0, 0.0])

        numerator = 2.0 * M / r_Schw + 8.0 * np.pi * r_Schw**2 * P

        dP_dr = -(rho_energy + P) * numerator / denom
        dnu_dr = numerator / denom
        dM_dr = 4.0 * np.pi * r_Schw**2 * rho_energy
        dr_iso_dr = r_iso / (r_Schw * np.sqrt(1.0 - 2.0 * M / r_Schw))

        return np.array([dP_dr, dnu_dr, dM_dr, dr_iso_dr])

    def solve(self, rho_central, rtol=1e-10, atol=1e-12):
        """
        Integrate TOV equations from center to surface.

        Returns: dict with TOV solution
        """
        # Initial conditions
        P_central = self.eos_pressure(rho_central)
        r_start = 1e-6
        rho_0 = self.eos_rho_energy(P_central)
        M_start = (4.0 * np.pi / 3.0) * rho_0 * r_start**3
        nu_start = 0.0
        r_iso_start = r_start

        y0 = np.array([P_central, nu_start, M_start, r_iso_start])

        # Setup integrator
        integrator = ode(self.tov_rhs).set_integrator('dopri5', rtol=rtol, atol=atol)
        integrator.set_initial_value(y0, r_start)

        # Storage
        r_arr = [r_start]
        P_arr = [P_central]
        nu_arr = [nu_start]
        M_arr = [M_start]
        r_iso_arr = [r_iso_start]

        dr = 1e-4
        r_current = r_start

        # Integrate until pressure drops to zero
        while integrator.successful() and r_current < 100.0:
            integrator.integrate(r_current + dr)
            P, nu, M, r_iso = integrator.y

            if P <= 1e-15 * P_central:
                break

            r_current += dr
            r_arr.append(r_current)
            P_arr.append(P)
            nu_arr.append(nu)
            M_arr.append(M)
            r_iso_arr.append(r_iso)

        # Convert to arrays
        r_Schw = np.array(r_arr)
        P = np.array(P_arr)
        nu = np.array(nu_arr)
        M = np.array(M_arr)
        r_iso = np.array(r_iso_arr)

        # Stellar surface values
        R_Schw = r_Schw[-1]
        M_star = M[-1]
        R_iso = r_iso[-1]
        nu_surface = nu[-1]

        # Apply boundary conditions (NRPy+ style)
        # Normalize r_iso
        r_iso_surface_correct = 0.5 * (np.sqrt(R_Schw * (R_Schw - 2*M_star)) + R_Schw - M_star)
        normalize = r_iso_surface_correct / R_iso
        r_iso = r_iso * normalize

        # Normalize nu (lapse)
        # At surface: alpha = sqrt(1 - 2M/R) => nu = log(alpha) = 0.5*log(1-2M/R)
        nu_surface_correct = 0.5 * np.log(1.0 - 2.0*M_star/R_Schw)
        nu = nu - nu_surface + nu_surface_correct

        # Derived quantities
        rho_baryon = np.array([self.eos_rho_baryon(p) for p in P])
        rho_energy = np.array([self.eos_rho_energy(p) for p in P])

        alpha = np.exp(nu)  # Lapse function
        exp4phi = (r_Schw / r_iso)**2  # Conformal factor

        print(f"TOV Solution: M={M_star:.6f}, R={R_Schw:.3f}, C={M_star/R_Schw:.4f}")

        return {
            'r_Schw': r_Schw,
            'r_iso': r_iso,
            'P': P,
            'rho_baryon': rho_baryon,
            'rho_energy': rho_energy,
            'M': M,
            'alpha': alpha,
            'exp4phi': exp4phi,
            'nu': nu,
            'M_star': M_star,
            'R_Schw': R_Schw,
            'R_iso': R_iso
        }


def create_tov_initial_state(grid, background, hydro, tov_solution):
    """
    Create engrenage initial state from TOV solution.

    Interpolates TOV data to grid and constructs full state vector.
    """
    r = grid.r
    N = len(r)

    # Interpolate TOV solution to grid (use isotropic radius)
    r_tov = tov_solution['r_iso']

    def safe_interp(data, fill_value=0.0):
        return interp1d(r_tov, data, kind='cubic', bounds_error=False,
                       fill_value=(data[0], fill_value), assume_sorted=True)

    rho_baryon = safe_interp(tov_solution['rho_baryon'], 1e-13)(r)
    pressure = safe_interp(tov_solution['P'], 1e-15)(r)
    alpha = safe_interp(tov_solution['alpha'], 1.0)(r)
    exp4phi = safe_interp(tov_solution['exp4phi'], 1.0)(r)

    # Ensure atmosphere
    rho_baryon = np.maximum(rho_baryon, hydro.atmosphere_rho)
    pressure = np.maximum(pressure, 1e-15)

    # Initial velocity = 0 (equilibrium)
    vr = np.zeros_like(r)

    # Lorentz factor and enthalpy
    W = np.ones_like(r)
    eps = hydro.eos.eps_from_rho_p(rho_baryon, pressure)
    h = 1.0 + eps + pressure / np.maximum(rho_baryon, 1e-30)

    # BSSN variables from TOV
    phi = 0.25 * np.log(exp4phi)  # e^{4φ} => φ = log(e^{4φ})/4

    # Conformal metric deviation h_ij = 0 (spherical)
    h_rr = np.zeros_like(r)
    h_tt = np.zeros_like(r)
    h_pp = np.zeros_like(r)

    # Extrinsic curvature K, A_ij = 0 (static)
    K = np.zeros_like(r)
    A_rr = np.zeros_like(r)
    A_tt = np.zeros_like(r)
    A_pp = np.zeros_like(r)

    # Conformal connection lambda^r = 0
    lambda_r = np.zeros_like(r)

    # Shift and auxiliary variable
    shift_r = np.zeros_like(r)
    b_r = np.zeros_like(r)

    # Lapse
    lapse = alpha

    # Conservative hydro variables
    # For static equilibrium: D = ρ₀ W, S_r = 0, τ = ρ₀ h W² - p - D
    gamma_rr = exp4phi  # In spherical: γ_rr = e^{4φ} * 1

    D = rho_baryon * W
    Sr = rho_baryon * h * W * W * vr * gamma_rr  # = 0 since vr=0
    tau = rho_baryon * h * W * W - pressure - D

    # Pack state vector
    state = np.zeros((grid.NUM_VARS, N))
    state[0, :] = phi
    state[1, :] = h_rr
    state[2, :] = h_tt
    state[3, :] = h_pp
    state[4, :] = K
    state[5, :] = A_rr
    state[6, :] = A_tt
    state[7, :] = A_pp
    state[8, :] = lambda_r
    state[9, :] = shift_r
    state[10, :] = b_r
    state[11, :] = lapse
    state[hydro.idx_D, :] = D
    state[hydro.idx_Sr, :] = Sr
    state[hydro.idx_tau, :] = tau

    return state


def get_rhs_cowling(t, y, grid, background, hydro, bssn_fixed, bssn_d1_fixed, diagnostics=None):
    """RHS for Cowling evolution with diagnostics."""
    state = y.reshape((grid.NUM_VARS, grid.N))
    grid.fill_boundaries(state)

    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(bssn_fixed)

    hydro.set_matter_vars(state, bssn_vars, grid)

    # Diagnostics
    if diagnostics is not None and diagnostics.get('print_interval', 0) > 0:
        diagnostics['call_count'] += 1
        should_print = False
        if not diagnostics.get('printed_initial', False):
            should_print = True
            diagnostics['printed_initial'] = True
        elif diagnostics['call_count'] % diagnostics['print_interval'] == 0:
            should_print = True
        if should_print:
            prim = hydro._get_primitives(bssn_vars, grid.r)

            center_idx = NUM_GHOSTS
            rho_c = prim['rho0'][center_idx]
            p_c = prim['p'][center_idx]
            v_max = np.max(np.abs(prim['vr']))
            D_c = state[hydro.idx_D, center_idx]
            Sr_c = state[hydro.idx_Sr, center_idx]
            tau_c = state[hydro.idx_tau, center_idx]
            vr_c = prim['vr'][center_idx]
            eps_c = prim.get('eps', np.zeros_like(prim['rho0']))[center_idx]
            W_c = prim.get('W', np.ones_like(prim['rho0']))[center_idx]
            h_c = prim.get('h', np.ones_like(prim['rho0']))[center_idx]

            inner = slice(NUM_GHOSTS, -NUM_GHOSTS) if grid.N > 2*NUM_GHOSTS else slice(0, grid.N)
            r_inner = grid.r[inner]
            D_inner = state[hydro.idx_D, inner]

            # Baryon masses: flat (diagnóstico) y física (isotrópica)
            M_b_flat = 4.0 * np.pi * trapezoid(D_inner * r_inner**2, r_inner)
            e6phi = np.exp(6.0 * np.asarray(bssn_vars.phi)[inner])
            M_b_phys = 4.0 * np.pi * trapezoid(D_inner * e6phi * r_inner**2, r_inner)

            # Referencias iniciales
            M_b_flat_0 = diagnostics.get('M_b_initial_flat', M_b_flat)
            if 'M_b_initial_flat' not in diagnostics:
                diagnostics['M_b_initial_flat'] = M_b_flat
            dM_b_flat = (M_b_flat - M_b_flat_0) / (M_b_flat_0 + 1e-300)

            M_b_phys_0 = diagnostics.get('M_b_initial_phys', M_b_phys)
            if 'M_b_initial_phys' not in diagnostics:
                diagnostics['M_b_initial_phys'] = M_b_phys
            dM_b_phys = (M_b_phys - M_b_phys_0) / (M_b_phys_0 + 1e-300)

            print(
                f"  t={t:6.3f}  ρ_c={rho_c:.9f}  P_c={p_c:.2e}  v_max={v_max:.2e}  "
                f"D_c={D_c:.6e}  S_r,c={Sr_c:.6e}  τ_c={tau_c:.6e}  "
                f"M_b_flat={M_b_flat:.6f}  ΔM_b_flat={dM_b_flat:.2e}  "
                f"M_b_phys={M_b_phys:.6f}  ΔM_b_phys={dM_b_phys:.2e}"
            )

            # Persist diagnostics to CSV if requested
            csv_path = diagnostics.get('csv_path')
            if csv_path is not None:
                csv_writer = diagnostics.get('csv_writer')
                if csv_writer is None:
                    csv_path.parent.mkdir(parents=True, exist_ok=True)
                    csv_file = open(csv_path, mode='w', newline='')
                    writer = csv.writer(csv_file)
                    writer.writerow([
                        't', 'rho_c', 'P_c', 'v_max', 'D_c', 'Sr_c', 'tau_c',
                        'vr_c', 'eps_c', 'W_c', 'h_c',
                        'M_b_flat', 'delta_M_b_flat', 'M_b_phys', 'delta_M_b_phys'
                    ])
                    diagnostics['csv_writer'] = writer
                    diagnostics['csv_file'] = csv_file
                else:
                    writer = csv_writer
                    csv_file = diagnostics.get('csv_file')

                writer.writerow([
                    f"{t:.15f}",
                    f"{rho_c:.15f}",
                    f"{p_c:.15f}",
                    f"{v_max:.15f}",
                    f"{D_c:.15f}",
                    f"{Sr_c:.15f}",
                    f"{tau_c:.15f}",
                    f"{vr_c:.15f}",
                    f"{eps_c:.15f}",
                    f"{W_c:.15f}",
                    f"{h_c:.15f}",
                    f"{M_b_flat:.15f}",
                    f"{dM_b_flat:.15f}",
                    f"{M_b_phys:.15f}",
                    f"{dM_b_phys:.15f}"
                ])
                csv_file.flush()

    # Compute RHS
    hydro_rhs = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1_fixed, background)

    rhs = np.zeros_like(state)
    rhs[NUM_BSSN_VARS:NUM_BSSN_VARS+3, :] = hydro_rhs

    return rhs.flatten()


def rk4_evolve(initial_state, grid, background, hydro,
               bssn_fixed, bssn_d1_fixed, diagnostics,
               T_final, min_dr, sample_times, center_idx):
    state = initial_state.copy()
    t = 0.0
    dt_max = 0.5 * min_dr
    rhs_calls = 0

    times_samples = []
    rho_samples = []

    def record_sample(current_state, current_time):
        state_2d = current_state.reshape((grid.NUM_VARS, grid.N))
        bssn_tmp = BSSNVars(grid.N)
        bssn_tmp.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
        hydro.set_matter_vars(state_2d, bssn_tmp, grid)
        prim = hydro._get_primitives(bssn_tmp, grid.r)
        times_samples.append(current_time)
        rho_samples.append(prim['rho0'][center_idx])

    record_sample(state, t)
    next_sample_idx = 1
    total_steps = 0

    while t < T_final - 1e-12:
        next_sample_time = sample_times[next_sample_idx] if next_sample_idx < len(sample_times) else T_final
        dt = min(dt_max, next_sample_time - t, T_final - t)
        if dt <= 0.0:
            next_sample_idx += 1
            continue

        k1 = get_rhs_cowling(t, state, grid, background, hydro, bssn_fixed, bssn_d1_fixed, diagnostics)
        k2 = get_rhs_cowling(t + 0.5 * dt, state + 0.5 * dt * k1, grid, background, hydro, bssn_fixed, bssn_d1_fixed, None)
        k3 = get_rhs_cowling(t + 0.5 * dt, state + 0.5 * dt * k2, grid, background, hydro, bssn_fixed, bssn_d1_fixed, None)
        k4 = get_rhs_cowling(t + dt, state + dt * k3, grid, background, hydro, bssn_fixed, bssn_d1_fixed, None)

        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        total_steps += 1
        rhs_calls += 4

        if next_sample_idx < len(sample_times) and t >= sample_times[next_sample_idx] - 1e-12:
            record_sample(state, t)
            next_sample_idx += 1

    state_final = state.reshape((grid.NUM_VARS, grid.N))
    return state_final, np.array(times_samples), np.array(rho_samples), rhs_calls, total_steps


def rk3_evolve(initial_state, grid, background, hydro,
               bssn_fixed, bssn_d1_fixed, diagnostics,
               T_final, min_dr, sample_times, center_idx):
    """SSP RK3 (TVD RK3) time integrator with fixed CFL.

    Returns:
      state_final (2D), times (1D), rho_c_series (1D), rhs_calls (int), total_steps (int)
    """
    state = initial_state.copy()
    t = 0.0
    dt_max = 0.5 * min_dr
    rhs_calls = 0

    times_samples = []
    rho_samples = []

    def record_sample(current_state, current_time):
        state_2d = current_state.reshape((grid.NUM_VARS, grid.N))
        bssn_tmp = BSSNVars(grid.N)
        bssn_tmp.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
        hydro.set_matter_vars(state_2d, bssn_tmp, grid)
        prim = hydro._get_primitives(bssn_tmp, grid.r)
        times_samples.append(current_time)
        rho_samples.append(prim['rho0'][center_idx])

    record_sample(state, t)
    next_sample_idx = 1
    total_steps = 0

    while t < T_final - 1e-12:
        next_sample_time = sample_times[next_sample_idx] if next_sample_idx < len(sample_times) else T_final
        dt = min(dt_max, next_sample_time - t, T_final - t)
        if dt <= 0.0:
            next_sample_idx += 1
            continue

        # Stage 1
        k1 = get_rhs_cowling(t, state, grid, background, hydro, bssn_fixed, bssn_d1_fixed, diagnostics)
        u1 = state + dt * k1

        # Stage 2
        k2 = get_rhs_cowling(t + dt, u1, grid, background, hydro, bssn_fixed, bssn_d1_fixed, None)
        u2 = 0.75 * state + 0.25 * (u1 + dt * k2)

        # Stage 3
        k3 = get_rhs_cowling(t + 0.5 * dt, u2, grid, background, hydro, bssn_fixed, bssn_d1_fixed, None)
        state = (1.0 / 3.0) * state + (2.0 / 3.0) * (u2 + dt * k3)

        t += dt
        total_steps += 1
        rhs_calls += 3

        if next_sample_idx < len(sample_times) and t >= sample_times[next_sample_idx] - 1e-12:
            record_sample(state, t)
            next_sample_idx += 1

    state_final = state.reshape((grid.NUM_VARS, grid.N))
    return state_final, np.array(times_samples), np.array(rho_samples), rhs_calls, total_steps



def main():
    """Main execution."""

    print("="*70)
    print("TOV Star Evolution")
    print("="*70)

    # ==================================================================
    # SETUP
    # ==================================================================

    # Grid
    r_max = 20.0
    min_dr = 0.002
    max_dr = 0.2

    params = CubicSpacing.get_parameters(r_max, min_dr, max_dr)
    spacing = CubicSpacing(**params)

    # EOS (polytropic)
    K = 100.0
    Gamma = 2.0
    eos = IdealGasEOS(gamma=Gamma)

    # Hydro
    reconstructor = create_reconstruction("mp5")
    riemann_solver = HLLERiemannSolver()

    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere_rho=1e-12,
        reconstructor=reconstructor,
        riemann_solver=riemann_solver
    )

    my_state_vector = StateVector(hydro)
    grid = Grid(spacing, my_state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    print(f"Grid: N={grid.N}, r_max={r_max}, dr_min={min_dr}")
    print(f"EOS: K={K}, Gamma={Gamma}")

    # ==================================================================
    # TOV SOLUTION
    # ==================================================================

    print("\nSolving TOV equations...")
    rho_central = 1.28e-3  # Nuclear density scale
    tov_solver = TOVSolver(K=K, Gamma=Gamma)
    tov_solution = tov_solver.solve(rho_central)

    # Visualize TOV profiles for diagnostic purposes
    r_plot = tov_solution['r_iso']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(r_plot, tov_solution['rho_baryon'], color='navy')
    axes[0, 0].set_xlabel(r"$r_{\mathrm{iso}}$")
    axes[0, 0].set_ylabel(r"$\rho_0$")
    axes[0, 0].set_title('Baryon Density')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(r_plot, tov_solution['P'], color='darkgreen')
    axes[0, 1].set_xlabel(r"$r_{\mathrm{iso}}$")
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure Profile')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(r_plot, tov_solution['M'], color='maroon')
    axes[1, 0].set_xlabel(r"$r_{\mathrm{iso}}$")
    axes[1, 0].set_ylabel('m(r)')
    axes[1, 0].set_title('Enclosed Mass')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(r_plot, tov_solution['alpha'], color='black')
    axes[1, 1].set_xlabel(r"$r_{\mathrm{iso}}$")
    axes[1, 1].set_ylabel('α')
    axes[1, 1].set_title('Lapse Function')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=True)
    plt.close(fig)

    print("\nTOV diagnostics displayed. Proceeding to build initial data and evolve...\n")

    # ==================================================================
    # INITIAL DATA
    # ==================================================================

    print("\nCreating initial data from TOV solution...")
    initial_state_2d = create_tov_initial_state(grid, background, hydro, tov_solution)
    initial_state = initial_state_2d.flatten()

    # Verify initial state
    bssn_vars_init = BSSNVars(grid.N)
    bssn_vars_init.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(initial_state_2d, bssn_vars_init, grid)
    prim0 = hydro._get_primitives(bssn_vars_init, grid.r)

    center_idx = NUM_GHOSTS
    D0_center = initial_state_2d[hydro.idx_D, center_idx]
    Sr0_center = initial_state_2d[hydro.idx_Sr, center_idx]
    tau0_center = initial_state_2d[hydro.idx_tau, center_idx]
    print(f"Initial ρ_c = {prim0['rho0'][center_idx]:.4f}")
    print(f"Initial P_c = {prim0['p'][center_idx]:.2e}")
    print(f"Initial max|v^r| = {np.max(np.abs(prim0['vr'])):.2e}")
    print(f"Initial conserved: D={D0_center:.3e}, S_r={Sr0_center:.3e}, τ={tau0_center:.3e}")

    # ==================================================================
    # EVOLUTION
    # ==================================================================

    bssn_fixed = initial_state_2d[:NUM_BSSN_VARS, :].copy()
    bssn_d1_fixed = grid.get_d1_metric_quantities(initial_state_2d)

    T_final = 1000.0
    num_outputs = 1000
    t_eval = np.linspace(0, T_final, num_outputs)

    print(f"\nEvolving for T={T_final} (Cowling approximation)")
    print("Diagnostics every 1000 RHS calls:")
    print("  " + "-"*90)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('tov_diagnostics')
    log_path = log_dir / f'tov_diagnostics_{timestamp}.csv'

    diagnostics = {
        'call_count': 0,
        'print_interval': 100,
        'csv_path': log_path,
        'csv_writer': None,
        'csv_file': None
    }

    # Select integrator: 'ivp' (solve_ivp RK45), 'rk4', or 'rk3'
    method = 'rk4'  # options: 'ivp', 'rk4', 'rk3'
    sample_times = np.linspace(0.0, T_final, num_outputs)

    if method == 'ivp':
        from scipy.integrate import solve_ivp
        solution = solve_ivp(
            get_rhs_cowling,
            [0, T_final],
            initial_state,
            args=(grid, background, hydro, bssn_fixed, bssn_d1_fixed, diagnostics),
            method='RK45',
            t_eval=sample_times,
            max_step=0.3 * min_dr
        )

        if not solution.success:
            print(f"\n  Evolution failed: {solution.message}")
            csv_file = diagnostics.get('csv_file')
            if csv_file is not None:
                csv_file.close()
            return

        state_final = solution.y[:, -1].reshape((grid.NUM_VARS, grid.N))
        times = solution.t
        rho_c_series = []
        bssn_tmp = BSSNVars(grid.N)
        for column in solution.y.T:
            state_snapshot = column.reshape((grid.NUM_VARS, grid.N))
            bssn_tmp.set_bssn_vars(state_snapshot[:NUM_BSSN_VARS, :])
            hydro.set_matter_vars(state_snapshot, bssn_tmp, grid)
            prim_snapshot = hydro._get_primitives(bssn_tmp, grid.r)
            rho_c_series.append(prim_snapshot['rho0'][center_idx])
        rho_c_series = np.array(rho_c_series)
        rhs_calls = solution.nfev
        print(f"\n✓ Evolution completed: {rhs_calls} RHS evaluations (solve_ivp)")

    elif method == 'rk4':
        state_final, times, rho_c_series, rhs_calls, total_steps = rk4_evolve(
            initial_state, grid, background, hydro,
            bssn_fixed, bssn_d1_fixed, diagnostics,
            T_final, min_dr, sample_times, center_idx
        )
        print(f"\n✓ Evolution completed: {rhs_calls} RHS evaluations ({total_steps} RK4 steps)")
    elif method == 'rk3':
        state_final, times, rho_c_series, rhs_calls, total_steps = rk3_evolve(
            initial_state, grid, background, hydro,
            bssn_fixed, bssn_d1_fixed, diagnostics,
            T_final, min_dr, sample_times, center_idx
        )
        print(f"\n✓ Evolution completed: {rhs_calls} RHS evaluations ({total_steps} RK3 steps)")
    else:
        raise ValueError(f"Unknown evolution method: {method}")

    # ==================================================================
    # FINAL DIAGNOSTICS
    # ==================================================================

    bssn_final = BSSNVars(grid.N)
    bssn_final.set_bssn_vars(state_final[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_final, bssn_final, grid)
    prim_final = hydro._get_primitives(bssn_final, grid.r)

    print(f"\nFinal state (t={T_final}):")
    print(f"  ρ_c = {prim_final['rho0'][center_idx]:.6f} (initial: {prim0['rho0'][center_idx]:.6f})")
    print(f"  P_c = {prim_final['p'][center_idx]:.2e} (initial: {prim0['p'][center_idx]:.2e})")
    print(f"  max|v^r| = {np.max(np.abs(prim_final['vr'])):.2e}")
    Df_center = state_final[hydro.idx_D, center_idx]
    Srf_center = state_final[hydro.idx_Sr, center_idx]
    tauf_center = state_final[hydro.idx_tau, center_idx]
    print(f"  Conserved center: D={Df_center:.6e} (initial {D0_center:.6e})")
    print(f"                     S_r={Srf_center:.6e} (initial {Sr0_center:.6e})")
    print(f"                     τ={tauf_center:.6e} (initial {tau0_center:.6e})")

    # Plot central density evolution over time
    if len(times) == len(rho_c_series) and len(times) > 1:
        plt.figure(figsize=(8, 4))
        plt.plot(times, rho_c_series, color='black', linewidth=2)
        plt.xlabel('t')
        plt.ylabel('ρ_c(t)')
        plt.title('Central Density Evolution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Fig. 1: time evolution of |ρ_c(t) − ρ_c(0)|
    if len(times) == len(rho_c_series) and len(times) > 1:
        delta_rhoc = np.abs(rho_c_series - rho_c_series[0])
        plt.figure(figsize=(8, 4))
        plt.semilogy(times, delta_rhoc, 'b-', linewidth=2)
        plt.xlabel('t')
        plt.ylabel('|ρ_c(t) − ρ_c(0)|')
        plt.title('Time evolution of |ρ_c(t) − ρ_c(0)|')
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Relative drift of central density scaled by 1e-3
    if len(times) == len(rho_c_series) and len(times) > 1:
        delta_rel = (rho_c_series - rho_c_series[0]) / (rho_c_series[0] + 1e-30) * 1e3
        times_ms = times * 1.0e3
        plt.figure(figsize=(8, 4))
        plt.plot(times_ms, delta_rel, color='darkred', linewidth=2)
        plt.xlabel('t [ms]')
        plt.ylabel(r'$\Delta\rho_c/\rho_{c,0} \times 10^{-3}$')
        plt.title('Central Density Drift (Relative)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    csv_file = diagnostics.get('csv_file')
    if csv_file is not None:
        csv_file.close()

    print("\n" + "="*70)
    print("Evolution complete!")
    print("="*70)


if __name__ == "__main__":
    main()
