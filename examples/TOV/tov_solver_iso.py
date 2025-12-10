"""
TOV solver in isotropic coordinates.

Solves TOV equations with coordinate transformation to isotropic coordinates.
The key difference from Schwarzschild coordinates is that we solve 4 ODEs
instead of 3, including the transformation dr_iso/dr_schw.

In isotropic coordinates, the metric is conformally flat:
    ds² = -α² dt² + e^{4φ} (dr_iso² + r_iso² dΩ²)

where e^{4φ} = (r_schw/r_iso)² is the conformal factor.

Usage:
  python examples/TOV/tov_solver_iso.py --K 100.0 --Gamma 2.0 --rho_central 1.28e-3 --r_max 20.0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import os
import json


# Default cache directory for TOV solutions
TOV_ISO_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tov_iso_cache")


def get_tov_iso_cache_path(K, Gamma, rho_central, cache_dir=None):
    """Get the cache directory path for a given set of TOV parameters.

    Args:
        K: Polytropic constant
        Gamma: Polytropic exponent
        rho_central: Central density
        cache_dir: Base cache directory (default: TOV_ISO_CACHE_DIR)

    Returns:
        Path to the cache directory for these parameters
    """
    if cache_dir is None:
        cache_dir = TOV_ISO_CACHE_DIR
    folder_name = f"TOVSOL_ISO_K={K}_G={Gamma}_rho={rho_central:.6e}"
    return os.path.join(cache_dir, folder_name)


class TOVSolutionIso:
    """Class to hold TOV solution data in isotropic coordinates."""

    def __init__(self):
        # Arrays
        self.r_schw = None         # Schwarzschild radial coordinate (integration variable)
        self.r_iso = None          # Isotropic radial coordinate (for grid)
        self.P = None              # Pressure array
        self.M = None              # Mass function M(r) array
        self.nu = None             # Metric function nu(r) array
        self.rho_baryon = None     # Baryon density array
        self.exp4phi = None        # e^(4φ) = (r_schw/r_iso)² conformal factor
        self.alpha = None          # Lapse function α(r) array

        # Scalars
        self.R_schw = None         # Stellar radius in Schwarzschild coords
        self.R_iso = None          # Stellar radius in isotropic coords
        self.M_star = None         # Total mass
        self.C = None              # Compactness M/R_schw

        # Grid info
        self.num_points = None     # Number of grid points

        # EOS parameters
        self.K = None              # Polytropic constant K
        self.Gamma = None          # Polytropic exponent Γ
        self.rho_central = None    # Central density

    # Compatibility properties for functions expecting standard TOV attributes
    @property
    def R(self):
        """Alias for R_iso for compatibility with standard TOV functions."""
        return self.R_iso

    @property
    def r(self):
        """Alias for r_iso for compatibility with standard TOV functions."""
        return self.r_iso

    def get_surface_index(self):
        """Get index of stellar surface."""
        surface_idx = np.where(self.P <= 1e-10)[0]
        if len(surface_idx) > 0:
            return surface_idx[0]
        return len(self.r_iso) - 1

    def save(self, cache_dir=None):
        """Save TOV solution to disk.

        Args:
            cache_dir: Directory to save to (default: uses get_tov_iso_cache_path)

        Returns:
            Path to saved directory
        """
        if cache_dir is None:
            cache_dir = get_tov_iso_cache_path(self.K, self.Gamma, self.rho_central)

        os.makedirs(cache_dir, exist_ok=True)

        # Save arrays as .npy files
        np.save(os.path.join(cache_dir, "r_schw.npy"), self.r_schw)
        np.save(os.path.join(cache_dir, "r_iso.npy"), self.r_iso)
        np.save(os.path.join(cache_dir, "P.npy"), self.P)
        np.save(os.path.join(cache_dir, "M.npy"), self.M)
        np.save(os.path.join(cache_dir, "nu.npy"), self.nu)
        np.save(os.path.join(cache_dir, "rho_baryon.npy"), self.rho_baryon)
        np.save(os.path.join(cache_dir, "exp4phi.npy"), self.exp4phi)
        np.save(os.path.join(cache_dir, "alpha.npy"), self.alpha)

        # Save scalars as JSON
        metadata = {
            "R_schw": float(self.R_schw),
            "R_iso": float(self.R_iso),
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
            TOVSolutionIso object

        Raises:
            FileNotFoundError: If cache directory doesn't exist
        """
        if not os.path.exists(cache_dir):
            raise FileNotFoundError(f"TOV cache not found: {cache_dir}")

        solution = cls()

        # Load arrays
        solution.r_schw = np.load(os.path.join(cache_dir, "r_schw.npy"))
        solution.r_iso = np.load(os.path.join(cache_dir, "r_iso.npy"))
        solution.P = np.load(os.path.join(cache_dir, "P.npy"))
        solution.M = np.load(os.path.join(cache_dir, "M.npy"))
        solution.nu = np.load(os.path.join(cache_dir, "nu.npy"))
        solution.rho_baryon = np.load(os.path.join(cache_dir, "rho_baryon.npy"))
        solution.exp4phi = np.load(os.path.join(cache_dir, "exp4phi.npy"))
        solution.alpha = np.load(os.path.join(cache_dir, "alpha.npy"))

        # Load scalars
        with open(os.path.join(cache_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        solution.R_schw = metadata["R_schw"]
        solution.R_iso = metadata["R_iso"]
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
            'r_schw': self.r_schw,
            'r_iso': self.r_iso,
            'P': self.P,
            'M': self.M,
            'nu': self.nu,
            'rho_baryon': self.rho_baryon,
            'exp4phi': self.exp4phi,
            'alpha': self.alpha,
            'R_schw': self.R_schw,
            'R_iso': self.R_iso,
            'M_star': self.M_star,
            'C': self.C,
        }


def load_or_solve_tov_iso(K, Gamma, rho_central, r_max=20.0,
                           accuracy="high", cache_dir=None, force_recompute=False):
    """Load TOV solution from cache or compute if not available.

    Args:
        K: Polytropic constant
        Gamma: Polytropic exponent
        rho_central: Central density
        r_max: Maximum radius (in isotropic coordinates)
        accuracy: Accuracy preset for adaptive mode
        cache_dir: Base cache directory
        force_recompute: If True, always recompute even if cache exists

    Returns:
        TOVSolutionIso object
    """
    cache_path = get_tov_iso_cache_path(K, Gamma, rho_central, cache_dir)

    # Try to load from cache
    if not force_recompute and os.path.exists(cache_path):
        try:
            solution = TOVSolutionIso.load(cache_path)
            print(f"Loaded TOV (isotropic) solution from cache: {cache_path}")
            return solution
        except Exception as e:
            print(f"Warning: Failed to load cache ({e}), recomputing...")

    # Compute new solution
    print(f"Computing TOV (isotropic) solution (K={K}, Γ={Gamma}, ρ_c={rho_central:.6e})...")
    solver = TOVSolverIso(K=K, Gamma=Gamma)
    solution = solver.solve(rho_central, r_max_iso=r_max, accuracy=accuracy)

    # Save to cache
    solution.save(cache_path)
    print(f"Saved TOV (isotropic) solution to: {cache_path}")

    return solution


class TOVSolverIso:
    """TOV solver in isotropic coordinates.

    Solves the standard TOV equations plus the coordinate transformation
    equation to get the isotropic radial coordinate r_iso.

    The 4 ODEs solved (in r_schw as independent variable):
    1. dP/dr = -(ρ+P)(m+4πr³P) / [r²(1-2m/r)]
    2. dν/dr = -2/(P+ρ) × dP/dr
    3. dM/dr = 4πr²ρ
    4. dr_iso/dr = (r_iso/r) / √(1-2m/r)  [NEW: coordinate transformation]
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
        """Total energy density ρ = ρ_b(1 + ε) where ε = P/[(Γ-1)ρ_b]."""
        if P <= 0:
            return 1e-12
        rho_b = self.eos_rho_baryon(P)
        eps = P / ((self.Gamma - 1.0) * rho_b) if rho_b > 0 else 0.0
        return rho_b * (1.0 + eps)

    def tov_rhs_iso(self, r_schw, y):
        """
        TOV RHS in Schwarzschild coordinates with isotropic coordinate evolution.

        Variables: y = [P, nu, M, r_iso]
        - P: pressure
        - nu: metric function (related to lapse)
        - M: enclosed mass
        - r_iso: isotropic radial coordinate

        Returns derivatives with respect to r_schw.
        """
        P, nu, M, r_iso = y
        rho = self.eos_rho_energy(P)

        # Near origin: use Taylor expansion to avoid numerical issues
        if r_schw < 1e-4 or M <= 0:
            # For small r, m ≈ (4π/3)r³ρ
            dP_dr = -(rho + P) * (4.0*np.pi/3.0*r_schw*rho + 4.0*np.pi*r_schw*P) / (1.0 - 8.0*np.pi*rho*r_schw**2)
            dnu_dr = -2.0 / (P + rho) * dP_dr if (P + rho) > 1e-30 else 0.0
            dM_dr = 4.0 * np.pi * r_schw**2 * rho
            # For r_iso near origin: dr_iso/dr_schw ≈ 1/√(1-8πρr²) ≈ 1
            dr_iso_dr = 1.0 / np.sqrt(1.0 - 8.0*np.pi*rho*r_schw**2) if r_schw > 1e-10 else 1.0
            return np.array([dP_dr, dnu_dr, dM_dr, dr_iso_dr])

        # Standard TOV equations for r > 1e-4
        one_minus_2m_r = 1.0 - 2.0 * M / r_schw
        if one_minus_2m_r < 1e-10:
            # Near horizon protection
            return np.array([0.0, 0.0, 0.0, 0.0])

        denom = r_schw**2 * one_minus_2m_r
        numerator = M + 4.0 * np.pi * r_schw**3 * P

        dP_dr = -(rho + P) * numerator / denom
        dnu_dr = -2.0 / (P + rho) * dP_dr if (P + rho) > 1e-30 else 0.0
        dM_dr = 4.0 * np.pi * r_schw**2 * rho

        # Coordinate transformation equation: dr_iso/dr_schw = (r_iso/r_schw) / √(1-2m/r)
        dr_iso_dr = (r_iso / r_schw) / np.sqrt(one_minus_2m_r)

        return np.array([dP_dr, dnu_dr, dM_dr, dr_iso_dr])

    def solve(self, rho_central, r_max_iso=20.0, accuracy="high"):
        """
        Solve TOV equations in isotropic coordinates.

        Args:
            rho_central: Central baryon density
            r_max_iso: Maximum radius in isotropic coordinates
            accuracy: Accuracy preset ("verylow", "low", "medium", "high", "veryhigh")

        Returns:
            TOVSolutionIso object
        """
        P_c = self.eos_pressure(rho_central)
        nu_c = 0.0
        M_c = 0.0

        # Initial r_schw and r_iso (start from small epsilon)
        r_start = 1e-10
        r_iso_start = r_start  # At origin, r_iso = r_schw

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
            min_step_size = 1e-6
            max_step_size = 1e-4
            integrator = 'dop853'
        elif accuracy == "veryhigh":
            min_step_size = 1e-7
            max_step_size = 1e-6
            integrator = 'dop853'
        else:
            min_step_size = 1e-5
            max_step_size = 1e-3
            integrator = 'dop853'

        # Set up ODE solver
        solver = ode(self.tov_rhs_iso).set_integrator(integrator, atol=1e-12, rtol=1e-12)
        solver.set_initial_value([P_c, nu_c, M_c, r_iso_start], r_start)

        # Storage arrays
        r_schw_arr = [r_start]
        r_iso_arr = [r_iso_start]
        P_arr = [P_c]
        nu_arr = [nu_c]
        M_arr = [M_c]

        # Integration loop
        dr = min_step_size
        r_max_schw = 100.0  # Large enough to contain the star

        while solver.successful() and solver.y[0] > 1e-10 * P_c and solver.t < r_max_schw:
            solver.integrate(solver.t + dr)

            r_now = solver.t
            P_now, nu_now, M_now, r_iso_now = solver.y

            r_schw_arr.append(r_now)
            r_iso_arr.append(r_iso_now)
            P_arr.append(max(P_now, 0.0))
            nu_arr.append(nu_now)
            M_arr.append(M_now)

            # Adaptive step size
            dP_dr, dnu_dr, dM_dr, dr_iso_dr = self.tov_rhs_iso(r_now, [P_now, nu_now, M_now, r_iso_now])

            def _safe_ratio(val, der):
                den = abs(der) if abs(der) > 1e-30 else 1e-30
                return abs(val / den)

            est1 = _safe_ratio(P_now, dP_dr) if P_now > 0 else max_step_size
            est2 = _safe_ratio(M_now if M_now != 0 else 1.0, dM_dr)
            dr = 0.1 * min(est1, est2)
            dr = max(min_step_size, min(dr, max_step_size))

        # Surface values
        R_schw_raw = r_schw_arr[-1]
        R_iso_raw = r_iso_arr[-1]
        M_star = M_arr[-1]
        nu_star = nu_arr[-1]

        # Convert to numpy arrays
        r_schw_arr = np.array(r_schw_arr)
        r_iso_arr = np.array(r_iso_arr)
        P_arr = np.array(P_arr)
        nu_arr = np.array(nu_arr)
        M_arr = np.array(M_arr)

        # Normalize r_iso using the exact exterior solution
        # In exterior Schwarzschild: r_iso = 0.5 * [sqrt(R(R-2M)) + R - M]
        R_iso_exact = 0.5 * (np.sqrt(R_schw_raw * (R_schw_raw - 2.0 * M_star)) + R_schw_raw - M_star)
        normalization_factor = R_iso_exact / R_iso_raw
        r_iso_arr = r_iso_arr * normalization_factor

        # Updated surface radius in isotropic coords
        R_schw = R_schw_raw
        R_iso = R_iso_exact

        # IMPORTANT: Normalize nu BEFORE extending to exterior
        # Following NRPy+ approach: rescale interior solution so that
        # nu matches the exterior Schwarzschild solution at the surface
        #
        # At surface R_schw in Schwarzschild coords: nu = log(1 - 2M/R_schw)
        # This ensures alpha -> 1 as r -> infinity

        # Find the integration endpoint (last interior point)
        surface_idx_interior = len(nu_arr) - 1

        # Schwarzschild formula for nu at the surface (in Schw coords)
        nu_surface_schw = np.log(1.0 - 2.0 * M_star / R_schw)

        # Compute offset: shift interior solution to match boundary condition
        nu_offset = nu_surface_schw - nu_arr[surface_idx_interior]

        # Apply offset to interior solution
        nu_arr = nu_arr + nu_offset

        # Now extend to exterior with properly normalized analytic solution
        # Need to extend up to r_max_iso
        r_iso_current = r_iso_arr[-1]
        dr_iso_ext = max_step_size

        while r_iso_current < r_max_iso:
            r_iso_current += dr_iso_ext

            # In exterior: r_schw = r_iso * (1 + M/(2*r_iso))^2
            r_schw_ext = r_iso_current * (1.0 + M_star / (2.0 * r_iso_current))**2

            # Schwarzschild metric functions in isotropic coords
            # nu_ext such that alpha = (1 - M/(2r_iso)) / (1 + M/(2r_iso))
            # exp(nu) = alpha^2
            # NOTE: This formula is already properly normalized (alpha -> 1 as r -> inf)
            alpha_ext = (1.0 - M_star / (2.0 * r_iso_current)) / (1.0 + M_star / (2.0 * r_iso_current))
            nu_ext = 2.0 * np.log(alpha_ext)

            r_schw_arr = np.append(r_schw_arr, r_schw_ext)
            r_iso_arr = np.append(r_iso_arr, r_iso_current)
            P_arr = np.append(P_arr, 0.0)
            nu_arr = np.append(nu_arr, nu_ext)
            M_arr = np.append(M_arr, M_star)

        # Compute derived quantities
        rho_arr = np.array([self.eos_rho_baryon(P) for P in P_arr])

        # Conformal factor: exp(4φ) = (r_schw/r_iso)^2
        # Protect against division by zero near origin
        with np.errstate(divide='ignore', invalid='ignore'):
            exp4phi_arr = (r_schw_arr / r_iso_arr)**2
        exp4phi_arr[0] = 1.0  # At origin, r_schw = r_iso, so exp4phi = 1

        # Lapse: alpha = exp(nu/2)
        alpha_arr = np.exp(nu_arr / 2.0)

        # Find surface more accurately using interpolation
        P_threshold = 1e-10
        surface_candidates = np.where(P_arr <= P_threshold)[0]
        if len(surface_candidates) > 0 and surface_candidates[0] > 0:
            idx_before = surface_candidates[0] - 1
            idx_after = surface_candidates[0]
            r_before = r_schw_arr[idx_before]
            r_after = r_schw_arr[idx_after]
            P_before = P_arr[idx_before]
            P_after = P_arr[idx_after]

            if abs(P_after - P_before) > 1e-30:
                R_schw_interp = r_before + (0.0 - P_before) / (P_after - P_before) * (r_after - r_before)
            else:
                R_schw_interp = r_before
            R_schw = R_schw_interp

        # Create solution object
        solution = TOVSolutionIso()
        solution.r_schw = r_schw_arr
        solution.r_iso = r_iso_arr
        solution.P = P_arr
        solution.M = M_arr
        solution.nu = nu_arr
        solution.rho_baryon = rho_arr
        solution.exp4phi = exp4phi_arr
        solution.alpha = alpha_arr
        solution.R_schw = R_schw
        solution.R_iso = R_iso
        solution.M_star = M_star
        solution.C = M_star / R_schw
        solution.num_points = len(r_iso_arr)
        solution.K = self.K
        solution.Gamma = self.Gamma
        solution.rho_central = rho_central

        return solution


def plot_tov_iso_diagnostics(tov_solution, r_max_iso):
    """Plot TOV solution diagnostics in isotropic coordinates.

    Args:
        tov_solution: TOVSolutionIso object
        r_max_iso: Maximum radius for plots (in isotropic coords)
    """
    r_iso = tov_solution.r_iso
    r_schw = tov_solution.r_schw
    R_iso = tov_solution.R_iso
    R_schw = tov_solution.R_schw
    M_star = tov_solution.M_star
    rho_baryon = tov_solution.rho_baryon
    P = tov_solution.P
    M = tov_solution.M
    alpha = tov_solution.alpha
    exp4phi = tov_solution.exp4phi

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Density vs r_iso
    axes[0, 0].plot(r_iso, rho_baryon, color='navy')
    axes[0, 0].axvline(R_iso, color='red', linestyle=':', alpha=0.7, label=f'R_iso={R_iso:.3f}')
    axes[0, 0].set_xlabel('r_iso')
    axes[0, 0].set_ylabel(r'$\rho_0$')
    axes[0, 0].set_title('Baryon Density')
    axes[0, 0].set_xlim(0, r_max_iso)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Pressure
    axes[0, 1].plot(r_iso, P, color='darkgreen')
    axes[0, 1].axvline(R_iso, color='red', linestyle=':', alpha=0.7)
    axes[0, 1].set_xlabel('r_iso')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure')
    axes[0, 1].set_xlim(0, r_max_iso)
    axes[0, 1].grid(True, alpha=0.3)

    # Enclosed Mass
    axes[0, 2].plot(r_iso, M, color='maroon')
    axes[0, 2].axvline(R_iso, color='red', linestyle=':', alpha=0.7)
    axes[0, 2].axhline(M_star, color='gray', linestyle='--', alpha=0.3, label=f'M={M_star:.4f}')
    axes[0, 2].set_xlabel('r_iso')
    axes[0, 2].set_ylabel('M(r)')
    axes[0, 2].set_title('Enclosed Mass')
    axes[0, 2].set_xlim(0, r_max_iso)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Lapse alpha(r_iso)
    axes[1, 0].plot(r_iso, alpha, color='purple')
    axes[1, 0].axvline(R_iso, color='red', linestyle=':', alpha=0.7)
    axes[1, 0].set_xlabel('r_iso')
    axes[1, 0].set_ylabel(r'$\alpha$')
    axes[1, 0].set_title('Lapse Function')
    axes[1, 0].set_xlim(0, r_max_iso)
    axes[1, 0].grid(True, alpha=0.3)

    # Conformal factor exp(4φ)
    axes[1, 1].plot(r_iso, exp4phi, color='teal')
    axes[1, 1].axvline(R_iso, color='red', linestyle=':', alpha=0.7)
    axes[1, 1].set_xlabel('r_iso')
    axes[1, 1].set_ylabel(r'$e^{4\phi}$')
    axes[1, 1].set_title('Conformal Factor $e^{4\\phi} = (r_{Schw}/r_{iso})^2$')
    axes[1, 1].set_xlim(0, r_max_iso)
    axes[1, 1].grid(True, alpha=0.3)

    # Coordinate transformation r_schw vs r_iso
    axes[1, 2].plot(r_iso, r_schw, color='orange', label=r'$r_{Schw}(r_{iso})$')
    axes[1, 2].plot(r_iso, r_iso, 'k--', alpha=0.3, label=r'$r_{Schw} = r_{iso}$')
    axes[1, 2].axvline(R_iso, color='red', linestyle=':', alpha=0.7)
    axes[1, 2].set_xlabel('r_iso')
    axes[1, 2].set_ylabel(r'$r_{Schw}$')
    axes[1, 2].set_title('Coordinate Transformation')
    axes[1, 2].set_xlim(0, r_max_iso)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f'TOV Solution (Isotropic Coords): M={M_star:.4f}, R_schw={R_schw:.3f}, R_iso={R_iso:.3f}, C={tov_solution.C:.4f}',
                 fontsize=12, y=1.00)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Solve TOV equations in isotropic coordinates')
    parser.add_argument('--K', type=float, default=100.0, help='Polytropic constant K')
    parser.add_argument('--Gamma', type=float, default=2.0, help='Polytropic index Γ')
    parser.add_argument('--rho_central', type=float, default=1.28e-3, help='Central density')
    parser.add_argument('--r_max', type=float, default=20.0, help='Maximum radius (isotropic)')
    parser.add_argument('--accuracy', type=str, default='high',
                        choices=['verylow', 'low', 'medium', 'high', 'veryhigh'],
                        help='Integration accuracy preset')
    parser.add_argument('--output', type=str, default='tov_solution_iso.png', help='Output plot filename')

    args = parser.parse_args()

    # Create solver
    solver = TOVSolverIso(K=args.K, Gamma=args.Gamma)

    # Solve
    solution = solver.solve(args.rho_central, r_max_iso=args.r_max, accuracy=args.accuracy)

    # Print summary
    print(f"\nTOV Solution Summary (Isotropic Coordinates):")
    print(f"  K            = {args.K}")
    print(f"  Gamma        = {args.Gamma}")
    print(f"  rho_central  = {args.rho_central:.6e}")
    print(f"  M_star       = {solution.M_star:.6f}")
    print(f"  R_schw       = {solution.R_schw:.6f}")
    print(f"  R_iso        = {solution.R_iso:.6f}")
    print(f"  Compactness  = {solution.C:.6f}")
    print(f"  Grid points  = {solution.num_points}")

    # Plot
    fig = plot_tov_iso_diagnostics(solution, args.r_max)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {args.output}")
    plt.close(fig)

    # Compare with NRPy+ values if using their parameters
    if abs(args.K - 1.0) < 0.01 and abs(args.Gamma - 2.0) < 0.01 and abs(args.rho_central - 0.129285) < 0.001:
        print("\n" + "="*60)
        print("Comparison with NRPy+ reference values (K=1, Gamma=2, rho_c=0.129285):")
        print("="*60)
        print(f"  NRPy+ M      = 0.1405   | This solver: {solution.M_star:.4f}")
        print(f"  NRPy+ R_schw = 0.9566   | This solver: {solution.R_schw:.4f}")
        print(f"  NRPy+ R_iso  = 0.8100   | This solver: {solution.R_iso:.4f}")
        print("="*60)
