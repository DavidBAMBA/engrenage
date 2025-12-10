"""
Standalone TOV solver with plotting and caching.

Solves TOV equations in Schwarzschild coordinates only.

Usage:
  python examples/tov_solver.py --K 200.0 --Gamma 2.0 --rho_central 1.28e-3 --r_max 11.0 --num_points 1000
Generates: tov_solution.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import os
import json


# Default cache directory for TOV solutions
TOV_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tov_cache")


def get_tov_cache_path(K, Gamma, rho_central, cache_dir=None):
    """Get the cache directory path for a given set of TOV parameters.

    Args:
        K: Polytropic constant
        Gamma: Polytropic exponent
        rho_central: Central density
        cache_dir: Base cache directory (default: TOV_CACHE_DIR)

    Returns:
        Path to the cache directory for these parameters
    """
    if cache_dir is None:
        cache_dir = TOV_CACHE_DIR
    folder_name = f"TOVSOL_K={K}_G={Gamma}_rho={rho_central:.6e}"
    return os.path.join(cache_dir, folder_name)


class TOVSolution:
    """Class to hold TOV solution data."""

    def __init__(self):
        # Arrays
        self.r = None              # Radial coordinate array
        self.P = None              # Pressure array
        self.M = None              # Mass function M(r) array
        self.nu = None             # Metric function nu(r) array
        self.rho_baryon = None     # Baryon density array
        self.exp4phi = None        # e^(4φ) conformal factor array
        self.alpha = None          # Lapse function α(r) array

        # Scalars
        self.R = None              # Stellar radius
        self.M_star = None         # Total mass
        self.C = None              # Compactness M/R

        # Grid info
        self.num_points = None     # Number of grid points

        # EOS parameters (needed for atmosphere consistency)
        self.K = None              # Polytropic constant K
        self.Gamma = None          # Polytropic exponent Γ
        self.rho_central = None    # Central density

    def get_surface_index(self):
        """Get index of stellar surface."""
        surface_idx = np.where(self.P <= 1e-10)[0]
        if len(surface_idx) > 0:
            return surface_idx[0]
        return len(self.r) - 1

    def save(self, cache_dir=None):
        """Save TOV solution to disk.

        Args:
            cache_dir: Directory to save to (default: uses get_tov_cache_path)

        Returns:
            Path to saved directory
        """
        if cache_dir is None:
            cache_dir = get_tov_cache_path(self.K, self.Gamma, self.rho_central)

        os.makedirs(cache_dir, exist_ok=True)

        # Save arrays as .npy files
        np.save(os.path.join(cache_dir, "r.npy"), self.r)
        np.save(os.path.join(cache_dir, "P.npy"), self.P)
        np.save(os.path.join(cache_dir, "M.npy"), self.M)
        np.save(os.path.join(cache_dir, "nu.npy"), self.nu)
        np.save(os.path.join(cache_dir, "rho_baryon.npy"), self.rho_baryon)
        np.save(os.path.join(cache_dir, "exp4phi.npy"), self.exp4phi)
        np.save(os.path.join(cache_dir, "alpha.npy"), self.alpha)

        # Save scalars as JSON
        metadata = {
            "R": float(self.R),
            "M_star": float(self.M_star),
            "C": float(self.C),
            "num_points": int(self.num_points),
            "K": float(self.K),
            "Gamma": float(self.Gamma),
            "rho_central": float(self.rho_central),
        }
        with open(os.path.join(cache_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        return cache_dir

    @classmethod
    def load(cls, cache_dir):
        """Load TOV solution from disk.

        Args:
            cache_dir: Directory containing saved solution

        Returns:
            TOVSolution object

        Raises:
            FileNotFoundError: If cache directory doesn't exist
        """
        if not os.path.exists(cache_dir):
            raise FileNotFoundError(f"TOV cache not found: {cache_dir}")

        solution = cls()

        # Load arrays
        solution.r = np.load(os.path.join(cache_dir, "r.npy"))
        solution.P = np.load(os.path.join(cache_dir, "P.npy"))
        solution.M = np.load(os.path.join(cache_dir, "M.npy"))
        solution.nu = np.load(os.path.join(cache_dir, "nu.npy"))
        solution.rho_baryon = np.load(os.path.join(cache_dir, "rho_baryon.npy"))
        solution.exp4phi = np.load(os.path.join(cache_dir, "exp4phi.npy"))
        solution.alpha = np.load(os.path.join(cache_dir, "alpha.npy"))

        # Load scalars
        with open(os.path.join(cache_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        solution.R = metadata["R"]
        solution.M_star = metadata["M_star"]
        solution.C = metadata["C"]
        solution.num_points = metadata["num_points"]
        solution.K = metadata["K"]
        solution.Gamma = metadata["Gamma"]
        solution.rho_central = metadata["rho_central"]

        return solution

    def to_dict(self):
        """Convert to dictionary format for backward compatibility."""
        return {
            'r': self.r,
            'P': self.P,
            'M': self.M,
            'nu': self.nu,
            'rho_baryon': self.rho_baryon,
            'exp4phi': self.exp4phi,
            'alpha': self.alpha,
            'R': self.R,
            'M_star': self.M_star,
            'C': self.C,
        }


def load_or_solve_tov(K, Gamma, rho_central, r_max=20.0, r_grid=None,
                      accuracy="high", cache_dir=None, force_recompute=False):
    """Load TOV solution from cache or compute if not available.

    Args:
        K: Polytropic constant
        Gamma: Polytropic exponent
        rho_central: Central density
        r_max: Maximum radius (for new computation)
        r_grid: Optional grid for grid-driven mode
        accuracy: Accuracy preset for adaptive mode
        cache_dir: Base cache directory
        force_recompute: If True, always recompute even if cache exists

    Returns:
        TOVSolution object
    """
    cache_path = get_tov_cache_path(K, Gamma, rho_central, cache_dir)

    # Try to load from cache
    if not force_recompute and os.path.exists(cache_path):
        try:
            solution = TOVSolution.load(cache_path)
            print(f"Loaded TOV solution from cache: {cache_path}")
            return solution
        except Exception as e:
            print(f"Warning: Failed to load cache ({e}), recomputing...")

    # Compute new solution
    print(f"Computing TOV solution (K={K}, Γ={Gamma}, ρ_c={rho_central:.6e})...")
    solver = TOVSolver(K=K, Gamma=Gamma)
    solution = solver.solve(rho_central, r_grid=r_grid, r_max=r_max, accuracy=accuracy)

    # Save to cache
    solution.save(cache_path)
    print(f"Saved TOV solution to: {cache_path}")

    return solution


class TOVSolver:
    """TOV solver in Schwarzschild coordinates.

    Supports two solve modes:
    - Grid-driven: provide r_grid to sample the solution exactly at those radii.
    - Adaptive-step (-style): no r_grid provided; an ODE integrator advances with an
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

        # Set up ODE solver with strict tolerances for high-order convergence
        solver = ode(self.tov_rhs).set_integrator(integrator, atol=1e-10, rtol=1e-10)
        solver.set_initial_value([P_c, nu_c, M_c], 1e-15)

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
                # For r < 1e-14, use initial conditions
                if r_schw < 1e-14:
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
            r_schw_arr = [1e-15]
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

        # Find surface using interpolation for accurate R_star
        # This gives resolution-independent surface detection
        surface_idx = np.where(P_arr <= 1e-10)[0]
        M_star = M_arr[-1]

        if len(surface_idx) > 0 and surface_idx[0] > 0:
            # Interpolate to find exact radius where P=0
            idx_before = surface_idx[0] - 1
            idx_after = surface_idx[0]
            r_before = r_schw_arr[idx_before]
            r_after = r_schw_arr[idx_after]
            P_before = P_arr[idx_before]
            P_after = P_arr[idx_after]

            # Linear interpolation to find R where P=0
            if abs(P_after - P_before) > 1e-30:
                R_star_schw = r_before + (0.0 - P_before) / (P_after - P_before) * (r_after - r_before)
            else:
                R_star_schw = r_before
        else:
            R_star_schw = r_schw_arr[-1]

        # Compute conformal factor exp(4φ)
        exp4phi = 1.0 / np.maximum(1.0 - 2.0 * M_arr / r_schw_arr, 1e-10)

        # Normalize lapse using exterior points where α = √(1 - 2M/r) exactly
        # This avoids precision issues from interpolating ν at the surface
        # Strategy: Match exp(ν/2) to Schwarzschild in the exterior
        exterior_mask = r_schw_arr > 1.1 * R_star_schw
        if np.any(exterior_mask):
            # In exterior: α_exact = √(1 - 2M/r)
            # From our integration: α = C × exp(ν/2) where C is normalization constant
            # Therefore: C = α_exact / exp(ν/2) = √(1 - 2M/r) / exp(ν/2)
            # Use median of exterior points for robustness
            alpha_exact_exterior = np.sqrt(1.0 - 2.0 * M_star / r_schw_arr[exterior_mask])
            exp_nu_half_exterior = np.exp(nu_arr[exterior_mask] / 2.0)
            normalization_constants = alpha_exact_exterior / exp_nu_half_exterior
            normalization = np.median(normalization_constants)

            # Apply normalization to all points
            alpha = normalization * np.exp(nu_arr / 2.0)
        else:
            # Fallback: not enough exterior points, use old method
            nu_ref = nu_arr[-1]
            r_ref = r_schw_arr[-1]
            expnu = np.exp(nu_arr - nu_ref + np.log(1.0 - 2.0 * M_star / r_ref))
            alpha = np.sqrt(expnu)

        r_coord = r_schw_arr
        R_star = R_star_schw

        # Create TOVSolution object
        solution = TOVSolution()
        solution.r = r_coord
        solution.P = P_arr
        solution.M = M_arr
        solution.nu = nu_arr
        solution.rho_baryon = rho_arr
        solution.exp4phi = exp4phi
        solution.alpha = alpha
        solution.R = R_star
        solution.M_star = M_star
        solution.C = M_star / R_star
        solution.num_points = len(r_coord)
        solution.K = self.K
        solution.Gamma = self.Gamma
        solution.rho_central = rho_central

        return solution


def plot_tov_diagnostics(tov_solution, r_max):
    """Plot TOV solution diagnostics.

    Args:
        tov_solution: TOVSolution object or dict
        r_max: Maximum radius for plots
    """
    # Handle both TOVSolution objects and dicts
    if isinstance(tov_solution, TOVSolution):
        r = tov_solution.r
        R_star = tov_solution.R
        M_star = tov_solution.M_star
        rho_baryon = tov_solution.rho_baryon
        P = tov_solution.P
        M = tov_solution.M
        alpha = tov_solution.alpha
        exp4phi = tov_solution.exp4phi
    else:
        r = tov_solution['r']
        R_star = tov_solution['R']
        M_star = tov_solution['M_star']
        rho_baryon = tov_solution['rho_baryon']
        P = tov_solution['P']
        M = tov_solution['M']
        alpha = tov_solution['alpha']
        exp4phi = tov_solution['exp4phi']

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Density
    axes[0, 0].plot(r, rho_baryon, color='navy')
    axes[0, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5, label=f'R={R_star:.2f}')
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel('rho_0')
    axes[0, 0].set_title('Baryon Density')
    axes[0, 0].set_xlim(0, r_max)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Pressure
    axes[0, 1].plot(r, P, color='darkgreen')
    axes[0, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure')
    axes[0, 1].set_xlim(0, r_max)
    axes[0, 1].grid(True, alpha=0.3)

    # Enclosed Mass
    axes[0, 2].plot(r, M, color='maroon')
    axes[0, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 2].axhline(M_star, color='gray', linestyle='--', alpha=0.3, label=f'M={M_star:.3f}')
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel('M(r)')
    axes[0, 2].set_title('Enclosed Mass')
    axes[0, 2].set_xlim(0, r_max)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Lapse alpha(r)
    axes[1, 0].plot(r, alpha, color='purple')
    axes[1, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel('alpha')
    axes[1, 0].set_title('Lapse Function')
    axes[1, 0].set_xlim(0, r_max)
    axes[1, 0].grid(True, alpha=0.3)

    # Phi(r)
    phi = 0.25 * np.log(exp4phi)
    axes[1, 1].plot(r, phi, color='teal')
    axes[1, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel('phi')
    axes[1, 1].set_title('Conformal Factor phi')
    axes[1, 1].set_xlim(0, r_max)
    axes[1, 1].grid(True, alpha=0.3)

    # a(r) metric function: a = exp(2*phi) = sqrt(exp4phi)
    a_metric = np.sqrt(exp4phi)
    axes[1, 2].plot(r, a_metric, color='orange')
    axes[1, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel('a(r)')
    axes[1, 2].set_title('Metric a(r)')
    axes[1, 2].set_xlim(0, r_max)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Solve TOV equations for neutron star')
    parser.add_argument('--K', type=float, default=100.0, help='Polytropic constant K')
    parser.add_argument('--Gamma', type=float, default=2.0, help='Polytropic index Γ')
    parser.add_argument('--rho_central', type=float, default=1.28e-3, help='Central density')
    parser.add_argument('--r_max', type=float, default=40.0, help='Maximum radius')
    parser.add_argument('--num_points', type=int, default=None, help='Number of grid points (default: adaptive)')
    parser.add_argument('--accuracy', type=str, default='high',
                       choices=['verylow', 'low', 'medium', 'high', 'veryhigh'],
                       help='Integration accuracy preset')
    parser.add_argument('--output', type=str, default='tov_solution.png', help='Output plot filename')

    args = parser.parse_args()

    # Create solver
    solver = TOVSolver(K=args.K, Gamma=args.Gamma)

    # Solve
    if args.num_points:
        # Grid-driven mode
        r_grid = np.linspace(1e-15, args.r_max, args.num_points)
        solution = solver.solve(args.rho_central, r_grid=r_grid)
    else:
        # Adaptive-step mode
        solution = solver.solve(args.rho_central, r_max=args.r_max, accuracy=args.accuracy)

    # Print summary
    print(f"\nTOV Solution Summary:")
    print(f"  K            = {args.K}")
    print(f"  Gamma        = {args.Gamma}")
    print(f"  rho_central  = {args.rho_central:.6e}")
    print(f"  M_star       = {solution.M_star:.6f}")
    print(f"  R_star       = {solution.R:.4f}")
    print(f"  Compactness  = {solution.C:.6f}")
    print(f"  Grid points  = {solution.num_points}")

    # Plot
    fig = plot_tov_diagnostics(solution, args.r_max)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {args.output}")
    plt.close(fig)

    # ========================================================================
    # CONVERGENCE TEST
    # ========================================================================
    print("\n" + "="*70)
    print("CONVERGENCE TEST")
    print("="*70)
    print("Testing TOV solver convergence with different resolutions...")

    # Create reference solution with very high resolution
    N_ref = 10000
    r_grid_ref = np.linspace(1e-15, args.r_max, N_ref)
    print(f"\nComputing reference solution (N={N_ref})...")
    solution_ref = solver.solve(args.rho_central, r_grid=r_grid_ref)

    # Test resolutions
    resolutions = [100, 200, 400]
    errors_P = []
    errors_M = []
    errors_alpha = []

    for N in resolutions:
        print(f"\nComputing solution with N={N}...")
        r_grid_test = np.linspace(1e-15, args.r_max, N)
        solution_test = solver.solve(args.rho_central, r_grid=r_grid_test)

        # Interpolate test solution to reference grid
        from scipy.interpolate import interp1d
        P_interp = interp1d(solution_test.r, solution_test.P, kind='cubic',
                            bounds_error=False, fill_value='extrapolate')(solution_ref.r)
        M_interp = interp1d(solution_test.r, solution_test.M, kind='cubic',
                            bounds_error=False, fill_value='extrapolate')(solution_ref.r)
        alpha_interp = interp1d(solution_test.r, solution_test.alpha, kind='cubic',
                                bounds_error=False, fill_value='extrapolate')(solution_ref.r)

        # Compute L1 errors (normalized by stellar radius for dimensional consistency)
        # Only compute error inside stellar radius
        mask_interior = solution_ref.r < solution_ref.R
        dr_ref = solution_ref.r[1] - solution_ref.r[0]

        error_P = np.sum(np.abs(P_interp[mask_interior] - solution_ref.P[mask_interior])) * dr_ref / solution_ref.R
        error_M = np.sum(np.abs(M_interp[mask_interior] - solution_ref.M[mask_interior])) * dr_ref / solution_ref.R
        error_alpha = np.sum(np.abs(alpha_interp[mask_interior] - solution_ref.alpha[mask_interior])) * dr_ref / solution_ref.R

        errors_P.append(error_P)
        errors_M.append(error_M)
        errors_alpha.append(error_alpha)

    # Print convergence results
    print("\n" + "-"*70)
    print("CONVERGENCE RESULTS (L1 errors)")
    print("-"*70)
    print(f"{'N':>6} {'Error P':>15} {'Error M':>15} {'Error α':>15}")
    print("-"*70)
    for i, N in enumerate(resolutions):
        print(f"{N:6d} {errors_P[i]:15.6e} {errors_M[i]:15.6e} {errors_alpha[i]:15.6e}")

    # Compute convergence order (p = log(e1/e2) / log(h1/h2))
    print("\n" + "-"*70)
    print("CONVERGENCE ORDER (between consecutive resolutions)")
    print("-"*70)
    print(f"{'N1 → N2':>12} {'Order P':>12} {'Order M':>12} {'Order α':>12}")
    print("-"*70)
    for i in range(len(resolutions) - 1):
        N1, N2 = resolutions[i], resolutions[i+1]
        h1, h2 = 1.0/N1, 1.0/N2

        order_P = np.log(errors_P[i] / errors_P[i+1]) / np.log(h1 / h2) if errors_P[i+1] > 0 else 0
        order_M = np.log(errors_M[i] / errors_M[i+1]) / np.log(h1 / h2) if errors_M[i+1] > 0 else 0
        order_alpha = np.log(errors_alpha[i] / errors_alpha[i+1]) / np.log(h1 / h2) if errors_alpha[i+1] > 0 else 0

        print(f"{N1:4d} → {N2:4d} {order_P:12.2f} {order_M:12.2f} {order_alpha:12.2f}")

    print("-"*70)
    print("\nNote: For ODE integrator with absolute tolerance ~1e-10,")
    print("      convergence order should approach the integrator order (typically 5-8).")
    print("="*70)
