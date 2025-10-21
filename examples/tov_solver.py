"""
Standalone TOV solver with plotting.

Solves TOV equations in Schwarzschild coordinates only.

Usage:
  python examples/tov_solver.py --K 200.0 --Gamma 2.0 --rho_central 1.28e-3 --r_max 11.0 --num_points 1000
Generates: tov_solution.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode


class TOVSolver:
    """TOV solver in Schwarzschild coordinates.

    Supports two solve modes:
    - Grid-driven: provide r_grid to sample the solution exactly at those radii.
    - Adaptive-step (NRPy+-style): no r_grid provided; an ODE integrator advances with an
      adaptively-updated target step based on local derivatives, then extends to r_max.
    """

    def __init__(self, K, Gamma):
        self.K = K
        self.Gamma = Gamma

    def eos_pressure(self, rho_baryon):
        return self.K * rho_baryon ** self.Gamma

    def eos_rho_baryon(self, P):
        if P <= 0:
            return 0.0
        return (P / self.K) ** (1.0 / self.Gamma)

    def eos_rho_energy(self, P):
        if P <= 0:
            return 1e-12
        rho_b = self.eos_rho_baryon(P)
        eps = P / ((self.Gamma - 1.0) * rho_b) if rho_b > 0 else 0.0
        return rho_b * (1.0 + eps)

    def tov_rhs(self, r_schw, y):
        """
        TOV RHS in Schwarzschild coordinates.

        Variables: y = [P, nu, M]
        - P: pressure
        - nu: metric function (related to lapse)
        - M: enclosed mass
        """
        P, nu, M = y
        rho = self.eos_rho_energy(P)

        # Near origin: use Taylor expansion
        # This ensures correct behavior as r→0 and avoids division by zero
        if r_schw < 1e-4 or M <= 0:
            # For small r, m ≈ (4π/3)r³ρ, so use Taylor-expanded equations
            # dP/dr = -(ρ+P) × [4πr(ρ/3 + P)] / [1 - 8πρr²]
            dP_dr = -(rho + P) * (4.0*np.pi/3.0*r_schw*rho + 4.0*np.pi*r_schw*P) / (1.0 - 8.0*np.pi*rho*r_schw**2)
            dnu_dr = -2.0 / (P + rho) * dP_dr  # From dP/dr relation
            dM_dr = 4.0 * np.pi * r_schw**2 * rho
            return np.array([dP_dr, dnu_dr, dM_dr])

        # Standard TOV equations for r > 1e-4
        denom = r_schw * (r_schw - 2.0 * M)
        if abs(denom) < 1e-30:
            # Failsafe: near horizon or singularity
            return np.array([0.0, 0.0, 0.0])

        numerator = M + 4.0 * np.pi * r_schw ** 3 * P

        dP_dr = -(rho + P) * numerator / denom
        dnu_dr = 2.0 * numerator / denom
        dM_dr = 4.0 * np.pi * r_schw ** 2 * rho

        return np.array([dP_dr, dnu_dr, dM_dr])

    def solve(self, rho_central, r_grid=None, r_max=20.0,
              accuracy: str = "high", integrator_type: str = "default",
              dr: float = 1.0e-4):
        """
        Solve TOV equations.

        Args:
            rho_central: Central baryon density
            r_grid: Optional radial grid to sample the solution on (grid-driven mode)
            r_max: Maximum radius
            accuracy: Adaptive-step presets ("verylow", "low", "medium", "high", "veryhigh")
            integrator_type: Force specific integrator (default uses preset)
            dr: Deprecated fixed step (kept for backward compatibility); superseded by
                adaptive-step presets if r_grid is None.

        Returns:
            dict with solution arrays
        """
        P_c = self.eos_pressure(rho_central)
        nu_c = 0.0
        M_c = 0.0

        # Choose integrator & step presets
        if accuracy == "medium":
            min_step_size = 1e-5
            max_step_size = 1e-2
            integrator = 'dop853'
        elif accuracy == "low":
            min_step_size = 1e-3
            max_step_size = 1e-1
            integrator = 'dopri5'
        elif accuracy == "verylow":
            min_step_size = 1e-1
            max_step_size = 5e-1
            integrator = 'dopri5'
        elif accuracy == "high":
            min_step_size = 1e-5
            max_step_size = 1e-5
            integrator = 'dop853'
        elif accuracy == "veryhigh":
            min_step_size = 1e-7
            max_step_size = 1e-6
            integrator = 'dop853'
        else:
            # Fallback to sensible defaults
            min_step_size = 1e-5
            max_step_size = 1e-2
            integrator = 'dop853'

        if integrator_type != "default":
            integrator = integrator_type

        solver = ode(self.tov_rhs).set_integrator(integrator)
        solver.set_initial_value([P_c, nu_c, M_c], 0.0001)

        if r_grid is not None:
            r_schw_arr = []
            P_arr = []
            nu_arr = []
            M_arr = []
            surface_found = False
            R_star = None
            M_star = None
            nu_star = None

            for r_schw in r_grid:
                # Include all grid points, even near r=0
                # For r < 0.001, use initial conditions
                if r_schw < 0.0005:
                    # Very close to origin: use initial conditions
                    r_schw_arr.append(r_schw)
                    P_arr.append(P_c)
                    nu_arr.append(nu_c)
                    M_arr.append(M_c)
                    continue

                if not surface_found:
                    # Avoid integrating to the same time; use current state instead
                    if r_schw <= solver.t + 1e-14:
                        y_now = solver.y
                    else:
                        solver.integrate(r_schw)
                        if not solver.successful():
                            break
                        y_now = solver.y

                    r_schw_arr.append(r_schw)
                    P_arr.append(max(y_now[0], 0.0))
                    nu_arr.append(y_now[1])
                    M_arr.append(y_now[2])

                    if y_now[0] <= 1e-10:
                        surface_found = True
                        R_star = r_schw
                        M_star = y_now[2]
                        nu_star = y_now[1]
                else:
                    # Exterior: vacuum solution
                    r_schw_arr.append(r_schw)
                    P_arr.append(0.0)
                    if r_schw > 2.0 * M_star and R_star > 2.0 * M_star:
                        nu_ext = np.log(1.0 - 2.0 * M_star / r_schw) + (
                            nu_star - np.log(1.0 - 2.0 * M_star / R_star)
                        )
                    else:
                        nu_ext = nu_star
                    nu_arr.append(nu_ext)
                    M_arr.append(M_star)

            r_schw_arr = np.array(r_schw_arr)
            P_arr = np.array(P_arr)
            nu_arr = np.array(nu_arr)
            M_arr = np.array(M_arr)
        else:
            r_schw_arr = [0.001]
            P_arr = [P_c]
            nu_arr = [nu_c]
            M_arr = [M_c]

            # Start with the minimum step
            dr_schw = min_step_size if min_step_size is not None else dr

            while solver.successful() and solver.y[0] > 1e-10 and solver.t < r_max:
                # Advance to next target radius; dopri5/dop853 will substep as needed
                solver.integrate(solver.t + dr_schw)

                r_now = solver.t
                P_now, nu_now, M_now = solver.y

                r_schw_arr.append(r_now)
                P_arr.append(P_now)
                nu_arr.append(nu_now)
                M_arr.append(M_now)

                # Update target step using local derivatives (limit to [min,max])
                dPdr, dnudr, dMdr = self.tov_rhs(r_now, [P_now, nu_now, M_now])
                # Use safe denominator to avoid division by ~0
                def _safe_ratio(val, der):
                    den = abs(der) if abs(der) > 1e-30 else 1e-30
                    return abs(val / den)

                # Heuristic: choose step proportional to local scale length
                est1 = _safe_ratio(P_now, dPdr)
                est2 = _safe_ratio(M_now if M_now != 0 else 1.0, dMdr)
                dr_schw = 0.1 * min(est1, est2)
                # Clip to preset bounds
                dr_schw = max(min_step_size, min(dr_schw, max_step_size))

            R_star = r_schw_arr[-1]
            M_star = M_arr[-1]
            nu_star = nu_arr[-1]

            # Extend to exterior (simple uniform extension is sufficient here)
            # Use a conservative step for extension
            dr_ext = max_step_size
            r_current = R_star + dr_ext
            while r_current <= r_max:
                r_schw_arr.append(r_current)
                P_arr.append(0.0)
                if r_current > 2.0 * M_star and R_star > 2.0 * M_star:
                    nu_ext = np.log(1.0 - 2.0 * M_star / r_current) + (
                        nu_star - np.log(1.0 - 2.0 * M_star / R_star)
                    )
                else:
                    nu_ext = nu_star
                nu_arr.append(nu_ext)
                M_arr.append(M_star)
                r_current += dr_ext

            r_schw_arr = np.array(r_schw_arr)
            P_arr = np.array(P_arr)
            nu_arr = np.array(nu_arr)
            M_arr = np.array(M_arr)

        rho_arr = np.array([self.eos_rho_baryon(P) for P in P_arr])

        # Find surface
        surface_idx = np.where(P_arr <= 1e-10)[0]
        if len(surface_idx) > 0:
            R_star_schw = r_schw_arr[surface_idx[0]]
        else:
            R_star_schw = r_schw_arr[-1]

        M_star = M_arr[-1]

        # Schwarzschild radial coordinate (only option now)
        # Normalize lapse: α→1 as r→∞, continuous at surface
        # α_int(r) = exp((ν(r) - ν(R))/2) × √(1 - 2M/R)
        # α_ext(r) = √(1 - 2M/r)
        exp4phi = 1.0 / np.maximum(1.0 - 2.0 * M_arr / r_schw_arr, 1e-10)
        nu_surface = nu_arr[surface_idx[0]] if len(surface_idx) > 0 else nu_arr[-1]
        expnu = np.exp(nu_arr - nu_surface + np.log(1.0 - 2.0 * M_star / R_star_schw))
        alpha = np.sqrt(expnu)
        r_coord = r_schw_arr
        R_star = R_star_schw

        result = {
            'r': r_coord,
            'P': P_arr,
            'M': M_arr,
            'nu': nu_arr,
            'rho_baryon': rho_arr,
            'exp4phi': exp4phi,
            'alpha': alpha,
            'R': R_star,
            'M_star': M_star,
            'C': M_star / R_star,
        }

        return result


def plot_tov_diagnostics(tov_solution, r_max):
    r = tov_solution['r']
    R_star = tov_solution['R']
    M_star = tov_solution['M_star']

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Density
    axes[0, 0].plot(r, tov_solution['rho_baryon'], color='navy')
    axes[0, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5, label=f'R={R_star:.2f}')
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel('rho_0')
    axes[0, 0].set_title('Baryon Density')
    axes[0, 0].set_xlim(0, r_max)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Pressure
    axes[0, 1].plot(r, tov_solution['P'], color='darkgreen')
    axes[0, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure')
    axes[0, 1].set_xlim(0, r_max)
    axes[0, 1].grid(True, alpha=0.3)

    # Enclosed Mass
    axes[0, 2].plot(r, tov_solution['M'], color='maroon')
    axes[0, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 2].axhline(M_star, color='gray', linestyle='--', alpha=0.3, label=f'M={M_star:.3f}')
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel('M(r)')
    axes[0, 2].set_title('Enclosed Mass')
    axes[0, 2].set_xlim(0, r_max)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Lapse alpha(r)
    axes[1, 0].plot(r, tov_solution['alpha'], color='purple')
    axes[1, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel('alpha')
    axes[1, 0].set_title('Lapse Function')
    axes[1, 0].set_xlim(0, r_max)
    axes[1, 0].grid(True, alpha=0.3)

    # Phi(r)
    phi = 0.25 * np.log(tov_solution['exp4phi'])
    axes[1, 1].plot(r, phi, color='teal')
    axes[1, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel('phi')
    axes[1, 1].set_title('Conformal Factor phi')
    axes[1, 1].set_xlim(0, r_max)
    axes[1, 1].grid(True, alpha=0.3)

    # a(r) metric function: a = exp(2*phi) = sqrt(exp4phi)
    a_metric = np.sqrt(tov_solution['exp4phi'])
    axes[1, 2].plot(r, a_metric, color='orange')
    axes[1, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel('a(r)')
    axes[1, 2].set_title('Metric a(r)')
    axes[1, 2].set_xlim(0, r_max)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tov_solution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_A_comparison(K=1e2, Gamma=2.0, rho0=1.28e-3, r_start=1e-6, r_end=50.0):
    """Replicate the A(r) comparison from TOV (1).py: reference vs manual RK4 vs solve_ivp.

    Saves figure as tov_A_comparison.png.
    """
    fluid_gamma = Gamma
    fluid_kappa = K
    TOV_rho0 = rho0
    fluid_atmos = 1e-8

    def tov_system(r, y):
        A, rho0_local = y
        eps = 1e-6
        r_safe = r if r > eps else eps
        e = fluid_kappa / (fluid_gamma - 1.0) * rho0_local ** (fluid_gamma - 1.0)
        rho_total = rho0_local * (1.0 + e)
        dA_dr = A * ((1.0 - A) / r_safe + 8.0 * np.pi * r_safe * A * rho_total)
        m = (r_safe / 2.0) * (1.0 - 1.0 / A)
        term = (rho0_local ** (1.0 - fluid_gamma) / fluid_kappa + fluid_gamma / (fluid_gamma - 1.0))
        if rho0_local < fluid_atmos:
            drho0_dr = 0.0
        else:
            drho0_dr = -rho0_local * term * (m / r_safe ** 2 + 4.0 * np.pi * r_safe * fluid_kappa * rho0_local ** fluid_gamma) \
                       / (fluid_gamma * (1.0 - 2.0 * m / r_safe))
        return [dA_dr, drho0_dr]

    # Reference solution
    from scipy.integrate import solve_ivp
    sol_ref = solve_ivp(tov_system, [r_start, r_end], [1.0, TOV_rho0], method='Radau', dense_output=True,
                        rtol=1e-12, atol=1e-14)
    r_ref = np.linspace(r_start, r_end, 5000)
    y_ref = sol_ref.sol(r_ref)
    A_ref = y_ref[0]
    rho0_ref = y_ref[1]

    # Manual RK4
    def manual_RK4(dr):
        N = int((r_end - r_start) / dr) + 1
        r_arr = np.linspace(r_start, r_end, N)
        y = np.zeros((N, 2))
        y[0, :] = [1.0, TOV_rho0]
        for i in range(N - 1):
            r_i = r_arr[i]
            y_i = y[i, :]
            h = dr if i > 0 else dr / 2.0
            k1 = np.array(tov_system(r_i, y_i))
            k2 = np.array(tov_system(r_i + h / 2.0, y_i + (h / 2.0) * k1))
            k3 = np.array(tov_system(r_i + h / 2.0, y_i + (h / 2.0) * k2))
            k4 = np.array(tov_system(r_i + h, y_i + h * k3))
            y[i + 1, :] = y_i + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return r_arr, y[:, 0], y[:, 1]

    r_manual, A_manual, rho0_manual = manual_RK4(0.00125)

    # solve_ivp with strict tolerances
    r_ivp = np.linspace(r_start, r_end, 5000)
    sol_ivp = solve_ivp(tov_system, [r_start, r_end], [1.0, TOV_rho0], method='Radau', dense_output=True,
                        rtol=1e-12, atol=1e-14)
    y_ivp = sol_ivp.sol(r_ivp)
    A_ivp = y_ivp[0]
    rho0_ivp = y_ivp[1]

    # Plot
    import matplotlib.pyplot as plt
    fig, (axA, axR) = plt.subplots(1, 2, figsize=(12, 5))

    axA.plot(r_ref, A_ref, 'k-', label='Reference A(r)')
    axA.plot(r_manual, A_manual, 'b--', label='Manual RK4 (dr=0.00125)')
    axA.plot(r_ivp, A_ivp, 'r-.', label='solve_ivp (strict tol)')
    axA.set_xlabel('r')
    axA.set_ylabel('A(r)')
    axA.set_title('Comparison of A(r)')
    axA.legend()
    axA.grid(True)

    axR.plot(r_ref, rho0_ref, 'k-', label=r'Reference $\rho_0(r)$')
    axR.plot(r_manual, rho0_manual, 'b--', label='Manual RK4 (dr=0.00125)')
    axR.plot(r_ivp, rho0_ivp, 'r-.', label='solve_ivp (strict tol)')
    axR.set_xlabel('r')
    axR.set_ylabel(r'$\rho_0(r)$')
    axR.set_title(r'Comparison of $\rho_0(r)$')
    axR.legend()
    axR.grid(True)
    plt.tight_layout()
    plt.savefig('tov_A_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _cli():
    import argparse

    p = argparse.ArgumentParser(description='Solve TOV and plot solution.')
    p.add_argument('--K', type=float, default=1.0)
    p.add_argument('--Gamma', type=float, default=2.0)
    p.add_argument('--rho_central', type=float, default=0.42)
    p.add_argument('--r_max', type=float, default=20.0)
    p.add_argument('--num_points', type=int, default=1000)
    p.add_argument('--make_A_comparison', action='store_true', help='Also plot A(r) comparison like TOV (1).py')
    args = p.parse_args()

    solver = TOVSolver(K=args.K, Gamma=args.Gamma)
    r_grid = np.linspace(0.001, args.r_max, args.num_points)
    sol = solver.solve(rho_central=args.rho_central, r_grid=r_grid, r_max=args.r_max)

    print(f"M={sol['M_star']:.6e}, R={sol['R']:.6e}, C=M/R={sol['C']:.6e}")
    plot_tov_diagnostics(sol, args.r_max)
    print('Saved tov_solution.png')

    if args.make_A_comparison:
        plot_A_comparison(K=args.K, Gamma=args.Gamma, rho0=args.rho_central, r_start=1e-6, r_end=max(args.r_max, 50.0))
        print('Saved tov_A_comparison.png')


if __name__ == '__main__':
    _cli()
