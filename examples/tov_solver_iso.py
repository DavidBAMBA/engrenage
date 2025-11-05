"""
TOV solver in isotropic coordinates ( -inspired).

This module integrates the TOV equations in Schwarzschild radius R,
while simultaneously evolving the isotropic radius r_bar via
    dr_bar/dR = r_bar / (R * sqrt(1 - 2 M(R) / R))

At the stellar surface we fix the integration constant so that r_bar is
continuous across the surface. We then provide profiles as functions of
the isotropic radius r_bar (interior), and exact vacuum formulas for the
exterior (mapping between R and r_bar).

Outputs are suitable for building ADM/BSSN data in isotropic coordinates.
"""

from typing import Optional, Dict, Any
import numpy as np
from scipy.integrate import ode


class TOVSolverISO:
    def __init__(self, K: float, Gamma: float):
        self.K = float(K)
        self.Gamma = float(Gamma)

    # EOS helpers (simple polytrope)
    def eos_pressure(self, rho_baryon: float) -> float:
        return self.K * rho_baryon ** self.Gamma

    def eos_rho_baryon(self, P: float) -> float:
        if P <= 0:
            return 0.0
        return (P / self.K) ** (1.0 / self.Gamma)

    def eos_rho_total(self, P: float) -> float:
        if P <= 0:
            return 1e-12
        rho_b = self.eos_rho_baryon(P)
        eps = P / ((self.Gamma - 1.0) * max(rho_b, 1e-30))
        return rho_b * (1.0 + eps)

    def _tov_rhs(self, R: float, y: np.ndarray):
        """
        y = [P, M, nu, rbar]
        dP/dR = -(rho + P) (M + 4π R^3 P) / [R (R - 2 M)]
        dM/dR = 4π R^2 rho
        dnu/dR = 2 (M + 4π R^3 P) / [R (R - 2 M)]
        drbar/dR = rbar / [R sqrt(1 - 2M/R)]
        """
        P, M, nu, rbar = y
        rho = self.eos_rho_total(P)

        # Near origin or degenerate, use Taylor-like expansions
        if R < 1e-6 or M <= 0.0:
            dP = -(rho + P) * (4.0 * np.pi / 3.0 * R * rho + 4.0 * np.pi * R * P) / max(1.0 - 8.0 * np.pi * rho * R * R, 1e-30)
            dnu = -2.0 / max(P + rho, 1e-30) * dP
            dM = 4.0 * np.pi * R * R * rho
            # For rbar, use limit drbar/dR ≈ 1 (rbar ≈ R near center)
            drbar = 1.0
            return np.array([dP, dM, dnu, drbar])

        denom = R * (R - 2.0 * M)
        if abs(denom) < 1e-30:
            return np.array([0.0, 0.0, 0.0, 0.0])

        num = M + 4.0 * np.pi * R ** 3 * P
        dP = -(rho + P) * num / denom
        dM = 4.0 * np.pi * R * R * rho
        dnu = 2.0 * num / denom
        # drbar/dR = rbar / (R * sqrt(1 - 2M/R))
        root = np.sqrt(max(1.0 - 2.0 * M / R, 1e-30))
        drbar = rbar / (R * root) if rbar > 0.0 else 1.0 / root
        return np.array([dP, dM, dnu, drbar])

    def solve(self, rho_central: float,
              r_max: float = 20.0,
              r_schw_start: float = 1.0e-6,
              accuracy: str = "high",
              integrator_type: str = "default") -> Dict[str, Any]:
        """
        Integrate to r_max in Schwarzschild radius, output arrays including
        isotropic radius r_iso and interior profiles.
        """
        P_c = self.eos_pressure(rho_central)
        # Initial values at small R
        y0 = np.array([P_c, 0.0, 0.0, 0.0])  # P, M, nu, rbar

        # Integrator presets
        if accuracy == "medium":
            min_step_size = 1e-5; max_step_size = 1e-2; integrator = 'dop853'
        elif accuracy == "low":
            min_step_size = 1e-3; max_step_size = 1e-1; integrator = 'dopri5'
        elif accuracy == "verylow":
            min_step_size = 1e-1; max_step_size = 5e-1; integrator = 'dopri5'
        elif accuracy == "high":
            min_step_size = 1e-5; max_step_size = 1e-5; integrator = 'dop853'
        elif accuracy == "veryhigh":
            min_step_size = 1e-7; max_step_size = 1e-6; integrator = 'dop853'
        else:
            min_step_size = 1e-5; max_step_size = 1e-2; integrator = 'dop853'
        if integrator_type != "default":
            integrator = integrator_type

        solver = ode(self._tov_rhs).set_integrator(integrator)
        solver.set_initial_value(y0, r_schw_start)

        R_list = []
        P_list = []
        M_list = []
        nu_list = []
        rbar_list = []

        R = r_schw_start
        dR = min_step_size

        # Integrate until P drops to ~0 or we reach r_max
        while solver.successful() and R <= r_max:
            y = solver.integrate(R + dR)
            R = solver.t
            P, M, nu, rbar = y
            if P < 0.0:
                P = 0.0
            R_list.append(R)
            P_list.append(P)
            M_list.append(M)
            nu_list.append(nu)
            rbar_list.append(rbar)
            if P <= 1e-10:
                break
            # Heuristic step control
            rho = self.eos_rho_total(P)
            num = M + 4.0 * np.pi * R ** 3 * P
            denom = max(R * (R - 2.0 * M), 1e-30)
            dP_dR = -(rho + P) * num / denom
            dM_dR = 4.0 * np.pi * R ** 2 * rho
            est1 = abs(P / max(dP_dR, 1e-30))
            est2 = abs(M / max(dM_dR, 1e-30)) if M > 0 else est1
            dR = max(min_step_size, min(0.1 * min(est1, est2), max_step_size))

        R_arr = np.array(R_list)
        P_arr = np.array(P_list)
        M_arr = np.array(M_list)
        nu_arr = np.array(nu_list)
        rbar_arr = np.array(rbar_list)

        if len(R_arr) == 0:
            raise RuntimeError("TOV integration failed to produce any interior points")

        # Stellar surface
        R_surf = R_arr[-1]
        M_star = M_arr[-1]

        # Fix integration constant so that rbar is continuous across surface
        # Following  : multiply by factor so that rbar matches exterior mapping at R_surf
        # Exterior mapping: rbar(R) = 0.5 * (sqrt(R(R-2M)) + R - M)
        rbar_surf_target = 0.5 * (np.sqrt(max(R_surf * (R_surf - 2.0 * M_star), 0.0)) + R_surf - M_star)
        if rbar_arr[-1] > 0:
            scale = rbar_surf_target / rbar_arr[-1]
            rbar_arr *= scale

        # Rescale nu so that alpha matches exterior at surface: alpha(R_surf) = sqrt(1 - 2M/R)
        nu_rescaled = nu_arr - nu_arr[-1] + np.log(max(1.0 - 2.0 * M_star / R_surf, 1e-30))
        alpha_int = np.sqrt(np.maximum(np.exp(nu_rescaled), 1e-300))

        # Compute exp4phi in isotropic coords: psi^4 = (R / rbar)^2
        with np.errstate(divide='ignore', invalid='ignore'):
            exp4phi_int = (R_arr / np.maximum(rbar_arr, 1e-30)) ** 2

        # Also build exterior profiles sampled on isotropic radius for convenience
        # Choose isotropic radii up to r_max_iso obtained by mapping R=r_max
        R_ext_max = max(r_max, R_surf * 1.05)
        # Build a monotonic isotropic grid for extension
        rbar_surf = rbar_arr[-1]
        rbar_ext = np.linspace(rbar_surf, 0.5 * (np.sqrt(max(R_ext_max * (R_ext_max - 2.0 * M_star), 0.0)) + R_ext_max - M_star), 256)
        # Map to R and compute exterior alpha, exp4phi
        R_ext = rbar_ext * 0.0
        with np.errstate(invalid='ignore'):
            # Invert mapping: R = rbar (1 + M/(2 rbar))^2
            R_ext = rbar_ext * (1.0 + M_star / (2.0 * np.maximum(rbar_ext, 1e-30))) ** 2
        alpha_ext = np.sqrt(np.maximum(1.0 - 2.0 * M_star / np.maximum(R_ext, 1e-30), 1e-30))
        exp4phi_ext = (R_ext / np.maximum(rbar_ext, 1e-30)) ** 2

        # Build interior arrays as functions of rbar
        order = np.argsort(rbar_arr)
        rbar_int_sorted = rbar_arr[order]
        alpha_int_sorted = alpha_int[order]
        exp4phi_int_sorted = exp4phi_int[order]
        P_int_sorted = P_arr[order]
        rho_b_int_sorted = np.array([self.eos_rho_baryon(p) for p in P_int_sorted])

        return {
            # Interior (as function of isotropic radius)
            'r_iso_int': rbar_int_sorted,
            'R_int': R_arr[order],
            'M_int': M_arr[order],
            'alpha_int': alpha_int_sorted,
            'exp4phi_int': exp4phi_int_sorted,
            'rho_baryon_int': rho_b_int_sorted,
            'P_int': P_int_sorted,
            # Exterior sampled in isotropic coords
            'r_iso_ext': rbar_ext,
            'R_ext': R_ext,
            'alpha_ext': alpha_ext,
            'exp4phi_ext': exp4phi_ext,
            # Stellar properties
            'M_star': float(M_star),
            'R_schw': float(R_surf),
            'R_iso': float(rbar_surf),
            # Joined arrays to be API-compatible with original solver
            'r': (np.concatenate([rbar_int_sorted, rbar_ext[1:]]) if rbar_ext.size > 0 else rbar_int_sorted),
            'alpha': (np.concatenate([alpha_int_sorted, alpha_ext[1:]]) if rbar_ext.size > 0 else alpha_int_sorted),
            'exp4phi': (np.concatenate([exp4phi_int_sorted, exp4phi_ext[1:]]) if rbar_ext.size > 0 else exp4phi_int_sorted),
            'rho_baryon': (np.concatenate([rho_b_int_sorted, np.zeros(max(rbar_ext.size-1, 0))]) if rbar_ext.size > 0 else rho_b_int_sorted),
            'P': (np.concatenate([P_int_sorted, np.zeros(max(rbar_ext.size-1, 0))]) if rbar_ext.size > 0 else P_int_sorted),
            'M': (np.concatenate([M_arr[order], np.full(max(rbar_ext.size-1, 0), M_star)]) if rbar_ext.size > 0 else M_arr[order]),
            'R': float(rbar_surf),
            'C': float(M_star / max(rbar_surf, 1e-30)),
        }
