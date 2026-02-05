"""
TOV solver in isotropic coordinates

Solves TOV equations with coordinate transformation to isotropic coordinates.
The key difference from Schwarzschild coordinates is that we solve 4 ODEs
instead of 3, including the transformation dr_iso/dr_schw.

In isotropic coordinates, the metric is conformally flat:
    ds² = -alpha² dt² + e^{4φ} (dr_iso² + r_iso² dΩ²)

where e^{4φ} = (r_schw/r_iso)² is the conformal factor.

EOS: Polytropic P = K * ρ_b^Γ

Usage:
  python tov_solver.py --K 100.0 --Gamma 2.0 --rho_central 1.28e-3 --r_max 20.0
"""

import numpy as np
import os

import matplotlib
if 'DISPLAY' not in os.environ or not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.integrate import ode
import warnings


# =============================================================================
# Polytropic EOS: P = K * ρ_b^Γ
# =============================================================================

class PolytropicEOS:
    """
    Polytropic equation of state: P = K * ρ_b^Γ

    For a polytrope:
        ε = P / [(Γ-1) * ρ_b]  (specific internal energy)
        ρ_total = ρ_b * (1 + ε)  (total energy density)
    """

    def __init__(self, K, Gamma):
        self.K = K
        self.Gamma = Gamma
        self._Gamma_minus_1 = Gamma - 1.0

    def P_from_rho_baryon(self, rho_b):
        """P = K * ρ_b^Γ"""
        if rho_b <= 0:
            return 0.0
        return self.K * rho_b ** self.Gamma

    def rho_baryon_from_P(self, P):
        """ρ_b = (P/K)^(1/Γ)"""
        if P <= 0:
            return 0.0
        return (P / self.K) ** (1.0 / self.Gamma)

    def rho_total_from_P(self, P):
        """ρ_total = ρ_b * (1 + ε) = ρ_b + P/(Γ-1)"""
        if P <= 0:
            return 1e-30
        rho_b = (P / self.K) ** (1.0 / self.Gamma)
        return rho_b + P / self._Gamma_minus_1


# =============================================================================
# Cache Configuration
# =============================================================================

TOV_ISO_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tov_iso_cache")


def get_tov_iso_cache_path(K, Gamma, rho_central, cache_dir=None):
    """Get the cache directory path for a given set of TOV parameters."""
    if cache_dir is None:
        cache_dir = TOV_ISO_CACHE_DIR
    folder_name = f"TOVSOL_ISO_K={K}_G={Gamma}_rho={rho_central:.6e}"
    return os.path.join(cache_dir, folder_name)


# =============================================================================
# TOV Solution Container
# =============================================================================

class TOVSolutionIso:
    """Class to hold TOV solution data in isotropic coordinates."""

    def __init__(self):
        # Arrays
        self.r_schw = None
        self.r_iso = None
        self.P = None
        self.M = None
        self.nu = None
        self.rho_baryon = None
        self.rho_total = None
        self.exp4phi = None
        self.alpha = None

        # Scalars
        self.R_schw = None
        self.R_iso = None
        self.M_star = None
        self.C = None

        # Grid info
        self.num_points = None
        self.surface_index = None

        # EOS parameters
        self.K = None
        self.Gamma = None
        self.rho_central = None

        # Diagnostics
        self.alpha_continuity_error = None
        self.integration_converged = True

    def save(self, cache_dir=None):
        """Save TOV solution to disk."""
        if cache_dir is None:
            cache_dir = get_tov_iso_cache_path(self.K, self.Gamma, self.rho_central)

        os.makedirs(cache_dir, exist_ok=True)

        np.save(os.path.join(cache_dir, "r_schw.npy"), self.r_schw)
        np.save(os.path.join(cache_dir, "r_iso.npy"), self.r_iso)
        np.save(os.path.join(cache_dir, "P.npy"), self.P)
        np.save(os.path.join(cache_dir, "M.npy"), self.M)
        np.save(os.path.join(cache_dir, "nu.npy"), self.nu)
        np.save(os.path.join(cache_dir, "rho_baryon.npy"), self.rho_baryon)
        np.save(os.path.join(cache_dir, "rho_total.npy"), self.rho_total)
        np.save(os.path.join(cache_dir, "exp4phi.npy"), self.exp4phi)
        np.save(os.path.join(cache_dir, "alpha.npy"), self.alpha)

        scalars = np.array([
            self.R_schw, self.R_iso, self.M_star, self.C,
            self.num_points, self.K, self.Gamma, self.rho_central,
            self.surface_index if self.surface_index is not None else -1,
            self.alpha_continuity_error if self.alpha_continuity_error is not None else 0.0,
            1.0 if self.integration_converged else 0.0
        ], dtype=np.float64)
        np.save(os.path.join(cache_dir, "scalars.npy"), scalars)

        return cache_dir

    @classmethod
    def load(cls, cache_dir):
        """Load TOV solution from disk."""
        if not os.path.exists(cache_dir):
            raise FileNotFoundError(f"TOV cache not found: {cache_dir}")

        solution = cls()

        solution.r_schw = np.load(os.path.join(cache_dir, "r_schw.npy"))
        solution.r_iso = np.load(os.path.join(cache_dir, "r_iso.npy"))
        solution.P = np.load(os.path.join(cache_dir, "P.npy"))
        solution.M = np.load(os.path.join(cache_dir, "M.npy"))
        solution.nu = np.load(os.path.join(cache_dir, "nu.npy"))
        solution.rho_baryon = np.load(os.path.join(cache_dir, "rho_baryon.npy"))
        solution.exp4phi = np.load(os.path.join(cache_dir, "exp4phi.npy"))
        solution.alpha = np.load(os.path.join(cache_dir, "alpha.npy"))

        rho_total_path = os.path.join(cache_dir, "rho_total.npy")
        if os.path.exists(rho_total_path):
            solution.rho_total = np.load(rho_total_path)
        else:
            solution.rho_total = None

        scalars = np.load(os.path.join(cache_dir, "scalars.npy"))
        solution.R_schw = scalars[0]
        solution.R_iso = scalars[1]
        solution.M_star = scalars[2]
        solution.C = scalars[3]
        solution.num_points = int(scalars[4])
        solution.K = scalars[5]
        solution.Gamma = scalars[6]
        solution.rho_central = scalars[7]

        if len(scalars) > 8:
            solution.surface_index = int(scalars[8]) if scalars[8] >= 0 else None
            solution.alpha_continuity_error = scalars[9]
            solution.integration_converged = scalars[10] > 0.5

        return solution


# =============================================================================
# Utility Functions
# =============================================================================

def load_or_solve_tov_iso(K, Gamma, rho_central, r_max=20.0,
                          accuracy="high", cache_dir=None, force_recompute=False):
    """Load TOV solution from cache or compute if not available."""
    cache_path = get_tov_iso_cache_path(K, Gamma, rho_central, cache_dir)

    if not force_recompute and os.path.exists(cache_path):
        try:
            solution = TOVSolutionIso.load(cache_path)
            print(f"Loaded TOV (isotropic) solution from cache: {cache_path}")
            return solution
        except Exception as e:
            print(f"Warning: Failed to load cache ({e}), recomputing...")

    print(f"Computing TOV (isotropic) solution (K={K}, Γ={Gamma}, ρ_c={rho_central:.6e})...")
    solver = TOVSolverIso(K=K, Gamma=Gamma)
    solution = solver.solve(rho_central, r_max_iso=r_max, accuracy=accuracy)

    solution.save(cache_path)
    print(f"Saved TOV (isotropic) solution to: {cache_path}")

    return solution


# =============================================================================
# Main TOV Solver Class
# =============================================================================

class TOVSolverIso:
    """
    TOV solver in isotropic coordinates.

    Solves the standard TOV equations plus the coordinate transformation
    equation to get the isotropic radial coordinate r_iso.

    The 4 ODEs solved (in r_schw as independent variable):
    1. dP/dr = -(ρ+P)(m+4πr³P) / [r²(1-2m/r)]
    2. dν/dr = -2/(P+ρ) × dP/dr
    3. dM/dr = 4πr²ρ
    4. dr_iso/dr = (r_iso/r) / √(1-2m/r)
    """

    def __init__(self, K, Gamma):
        """
        Initialize TOV solver with polytropic EOS P = K * ρ_b^Γ.

        Args:
            K: Polytropic constant
            Gamma: Polytropic exponent
        """
        self.eos = PolytropicEOS(K, Gamma)
        self.K = K
        self.Gamma = Gamma

    def tov_rhs_iso(self, r_schw, y):
        """
        TOV RHS in Schwarzschild coordinates with isotropic coordinate evolution.

        Variables: y = [P, nu, M, r_iso]
        Returns derivatives with respect to r_schw.
        """
        P, nu, M, r_iso = y

        rho = self.eos.rho_total_from_P(P)

        if rho < 1e-30:
            rho = 1e-30

        # Near origin: use Taylor expansion
        if r_schw < 1e-4 or M <= 0:
            denom = 1.0 - 8.0 * np.pi * rho * r_schw**2
            if abs(denom) < 1e-30:
                denom = 1e-30 * np.sign(denom) if denom != 0 else 1e-30

            dP_dr = -(rho + P) * (4.0*np.pi/3.0*r_schw*rho + 4.0*np.pi*r_schw*P) / denom

            if (P + rho) > 1e-30:
                dnu_dr = -2.0 / (P + rho) * dP_dr
            else:
                dnu_dr = 0.0

            dM_dr = 4.0 * np.pi * r_schw**2 * rho

            if r_schw > 1e-10:
                dr_iso_dr = 1.0 / np.sqrt(max(denom, 1e-30))
            else:
                dr_iso_dr = 1.0

            return np.array([dP_dr, dnu_dr, dM_dr, dr_iso_dr])

        # Standard TOV equations
        one_minus_2m_r = 1.0 - 2.0 * M / r_schw

        if one_minus_2m_r < 1e-10:
            warnings.warn(f"Near horizon at r={r_schw:.6e}, M={M:.6e}")
            return np.array([0.0, 0.0, 0.0, 0.0])

        denom = r_schw**2 * one_minus_2m_r
        numerator = M + 4.0 * np.pi * r_schw**3 * P

        dP_dr = -(rho + P) * numerator / denom

        if (P + rho) > 1e-30:
            dnu_dr = -2.0 / (P + rho) * dP_dr
        else:
            dnu_dr = 0.0

        dM_dr = 4.0 * np.pi * r_schw**2 * rho

        dr_iso_dr = (r_iso / r_schw) / np.sqrt(one_minus_2m_r)

        return np.array([dP_dr, dnu_dr, dM_dr, dr_iso_dr])

    def solve(self, rho_central, r_max_iso=20.0, accuracy="high", verbose=True):
        """
        Solve TOV equations in isotropic coordinates.

        Args:
            rho_central: Central baryon density
            r_max_iso: Maximum radius in isotropic coordinates
            accuracy: Accuracy preset ("verylow", "low", "medium", "high", "veryhigh")
            verbose: Print diagnostics

        Returns:
            TOVSolutionIso object
        """
        P_c = self.eos.P_from_rho_baryon(rho_central)
        nu_c = 0.0
        M_c = 0.0

        r_start = 1e-10
        r_iso_start = r_start

        accuracy_settings = {
            "verylow":  {"min_step": 1e-1,  "max_step": 5e-1, "integrator": "dopri5", "rtol": 1e-6,  "atol": 1e-6},
            "low":      {"min_step": 1e-3,  "max_step": 1e-1, "integrator": "dopri5", "rtol": 1e-8,  "atol": 1e-8},
            "medium":   {"min_step": 1e-5,  "max_step": 1e-2, "integrator": "dop853", "rtol": 1e-10, "atol": 1e-10},
            "high":     {"min_step": 1e-6,  "max_step": 1e-4, "integrator": "dop853", "rtol": 1e-12, "atol": 1e-12},
            "veryhigh": {"min_step": 1e-7,  "max_step": 1e-6, "integrator": "dop853", "rtol": 1e-14, "atol": 1e-14},
        }

        settings = accuracy_settings.get(accuracy, accuracy_settings["high"])
        min_step_size = settings["min_step"]
        max_step_size = settings["max_step"]
        integrator = settings["integrator"]
        rtol = settings["rtol"]
        atol = settings["atol"]

        solver = ode(self.tov_rhs_iso).set_integrator(integrator, atol=atol, rtol=rtol)
        solver.set_initial_value([P_c, nu_c, M_c, r_iso_start], r_start)

        r_schw_arr = [r_start]
        r_iso_arr = [r_iso_start]
        P_arr = [P_c]
        nu_arr = [nu_c]
        M_arr = [M_c]

        dr = min_step_size
        r_max_schw = 100.0

        P_threshold_relative = 1e-15
        integration_converged = True
        P_previous = P_c

        def _safe_ratio(val, der, default):
            if abs(der) < 1e-30:
                return default
            return abs(val / der)

        while solver.successful() and solver.t < r_max_schw:
            solver.integrate(solver.t + dr)

            r_now = solver.t
            P_now, nu_now, M_now, r_iso_now = solver.y

            if P_now < P_threshold_relative * P_c:
                if verbose:
                    print(f"  Stopped: P/P_c = {P_now/P_c:.2e} < {P_threshold_relative:.0e}")
                break

            if P_now > P_previous * 1.01 and len(P_arr) > 10:
                warnings.warn(f"Integration instability detected at r={r_now:.6e}: P increased")
                integration_converged = False
                break

            if P_now < 0:
                P_now = 0.0
                break

            P_previous = P_now

            r_schw_arr.append(r_now)
            r_iso_arr.append(r_iso_now)
            P_arr.append(max(P_now, 0.0))
            nu_arr.append(nu_now)
            M_arr.append(M_now)

            dP_dr, dnu_dr, dM_dr, dr_iso_dr = self.tov_rhs_iso(r_now, [P_now, nu_now, M_now, r_iso_now])

            est_P = _safe_ratio(P_now, dP_dr, max_step_size) if P_now > 0 else max_step_size
            est_M = _safe_ratio(M_now if M_now > 0 else 1.0, dM_dr, max_step_size)

            dr = 0.1 * min(est_P, est_M)
            dr = np.clip(dr, min_step_size, max_step_size)

        R_schw_raw = r_schw_arr[-1]
        R_iso_raw = r_iso_arr[-1]
        M_star = M_arr[-1]
        surface_index_interior = len(r_schw_arr) - 1

        r_schw_arr = np.array(r_schw_arr)
        r_iso_arr = np.array(r_iso_arr)
        P_arr = np.array(P_arr)
        nu_arr = np.array(nu_arr)
        M_arr = np.array(M_arr)

        # Normalize r_iso using exact exterior solution
        discriminant = R_schw_raw * (R_schw_raw - 2.0 * M_star)
        if discriminant < 0:
            warnings.warn(f"Star too compact! R_schw={R_schw_raw:.6e}, 2M={2*M_star:.6e}")
            discriminant = 0.0
            integration_converged = False

        R_iso_exact = 0.5 * (np.sqrt(discriminant) + R_schw_raw - M_star)

        if R_iso_raw > 0:
            normalization_factor = R_iso_exact / R_iso_raw
            r_iso_arr = r_iso_arr * normalization_factor

        R_schw = R_schw_raw
        R_iso = R_iso_exact

        # Normalize nu
        nu_surface_schw = np.log(1.0 - 2.0 * M_star / R_schw)
        nu_offset = nu_surface_schw - nu_arr[surface_index_interior]
        nu_arr = nu_arr + nu_offset

        alpha_interior_surface = np.exp(nu_arr[surface_index_interior] / 2.0)

        # Extend to exterior
        r_iso_current = r_iso_arr[-1]
        dr_iso_ext = max_step_size

        while r_iso_current < r_max_iso:
            r_iso_current += dr_iso_ext

            r_schw_ext = r_iso_current * (1.0 + M_star / (2.0 * r_iso_current))**2

            alpha_ext = (1.0 - M_star / (2.0 * r_iso_current)) / (1.0 + M_star / (2.0 * r_iso_current))
            nu_ext = 2.0 * np.log(alpha_ext)

            r_schw_arr = np.append(r_schw_arr, r_schw_ext)
            r_iso_arr = np.append(r_iso_arr, r_iso_current)
            P_arr = np.append(P_arr, 0.0)
            nu_arr = np.append(nu_arr, nu_ext)
            M_arr = np.append(M_arr, M_star)

        # Compute derived quantities
        rho_baryon_arr = np.array([self.eos.rho_baryon_from_P(P) for P in P_arr])
        rho_total_arr = np.array([self.eos.rho_total_from_P(P) for P in P_arr])

        with np.errstate(divide='ignore', invalid='ignore'):
            exp4phi_arr = (r_schw_arr / r_iso_arr)**2
        exp4phi_arr[0] = 1.0

        alpha_arr = np.exp(nu_arr / 2.0)

        # Validate continuity at surface
        alpha_exterior_surface = (1.0 - M_star / (2.0 * R_iso)) / (1.0 + M_star / (2.0 * R_iso))
        alpha_continuity_error = abs(alpha_interior_surface - alpha_exterior_surface)

        if alpha_continuity_error > 1e-6 and verbose:
            warnings.warn(f"alpha discontinuity at surface: interior={alpha_interior_surface:.8f}, "
                         f"exterior={alpha_exterior_surface:.8f}, error={alpha_continuity_error:.2e}")

        # Find surface more accurately
        P_surface_threshold = 1e-10
        surface_candidates = np.where(P_arr <= P_surface_threshold)[0]
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
        solution.rho_baryon = rho_baryon_arr
        solution.rho_total = rho_total_arr
        solution.exp4phi = exp4phi_arr
        solution.alpha = alpha_arr
        solution.R_schw = R_schw
        solution.R_iso = R_iso
        solution.M_star = M_star
        solution.C = M_star / R_schw if R_schw > 0 else 0.0
        solution.num_points = len(r_iso_arr)
        solution.surface_index = surface_index_interior
        solution.K = self.K
        solution.Gamma = self.Gamma
        solution.rho_central = rho_central
        solution.alpha_continuity_error = alpha_continuity_error
        solution.integration_converged = integration_converged

        if verbose:
            print(f"\nTOV Solution Summary:")
            print(f"  M_star       = {M_star:.6f}")
            print(f"  R_schw       = {R_schw:.6f}")
            print(f"  R_iso        = {R_iso:.6f}")
            print(f"  Compactness  = {solution.C:.6f}")
            print(f"  Grid points  = {solution.num_points}")
            print(f"  alpha continuity = {alpha_continuity_error:.2e}")
            print(f"  Converged    = {integration_converged}")

        return solution


# =============================================================================
# Plotting Function
# =============================================================================

def plot_tov_iso_diagnostics(tov_solution, r_max_iso=None, save_path=None):
    """
    Plot TOV solution diagnostics in isotropic coordinates.
    """
    if r_max_iso is None:
        r_max_iso = tov_solution.r_iso[-1]

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

    # Density
    axes[0, 0].plot(r_iso, rho_baryon, color='navy', linewidth=1.5)
    axes[0, 0].axvline(R_iso, color='red', linestyle=':', alpha=0.7, label=f'R_iso={R_iso:.3f}')
    axes[0, 0].set_xlabel(r'$r_{iso}$', fontsize=12)
    axes[0, 0].set_ylabel(r'$\rho_b$', fontsize=12)
    axes[0, 0].set_title('Baryon Density', fontsize=12)
    axes[0, 0].set_xlim(0, min(r_max_iso, 3*R_iso))
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Pressure
    axes[0, 1].plot(r_iso, P, color='darkgreen', linewidth=1.5)
    axes[0, 1].axvline(R_iso, color='red', linestyle=':', alpha=0.7)
    axes[0, 1].set_xlabel(r'$r_{iso}$', fontsize=12)
    axes[0, 1].set_ylabel(r'$P$', fontsize=12)
    axes[0, 1].set_title('Pressure', fontsize=12)
    axes[0, 1].set_xlim(0, min(r_max_iso, 3*R_iso))
    axes[0, 1].grid(True, alpha=0.3)

    # Mass
    axes[0, 2].plot(r_iso, M, color='maroon', linewidth=1.5)
    axes[0, 2].axvline(R_iso, color='red', linestyle=':', alpha=0.7)
    axes[0, 2].axhline(M_star, color='gray', linestyle='--', alpha=0.3, label=f'M={M_star:.4f}')
    axes[0, 2].set_xlabel(r'$r_{iso}$', fontsize=12)
    axes[0, 2].set_ylabel(r'$M(r)$', fontsize=12)
    axes[0, 2].set_title('Enclosed Mass', fontsize=12)
    axes[0, 2].set_xlim(0, r_max_iso)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Lapse
    axes[1, 0].plot(r_iso, alpha, color='purple', linewidth=1.5)
    axes[1, 0].axvline(R_iso, color='red', linestyle=':', alpha=0.7, label=f'R_iso={R_iso:.3f}')
    axes[1, 0].axhline(1.0, color='gray', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel(r'$r_{iso}$', fontsize=12)
    axes[1, 0].set_ylabel(r'$\alpha$', fontsize=12)
    axes[1, 0].set_title('Lapse Function', fontsize=12)
    axes[1, 0].set_xlim(0, r_max_iso)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Conformal factor
    axes[1, 1].plot(r_iso, exp4phi, color='teal', linewidth=1.5)
    axes[1, 1].axvline(R_iso, color='red', linestyle=':', alpha=0.7)
    axes[1, 1].axhline(1.0, color='gray', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel(r'$r_{iso}$', fontsize=12)
    axes[1, 1].set_ylabel(r'$e^{4\phi}$', fontsize=12)
    axes[1, 1].set_title(r'Conformal Factor $e^{4\phi} = (r_{Schw}/r_{iso})^2$', fontsize=12)
    axes[1, 1].set_xlim(0, r_max_iso)
    axes[1, 1].grid(True, alpha=0.3)

    # Coordinate transformation
    axes[1, 2].plot(r_iso, r_schw, color='orange', linewidth=1.5, label=r'$r_{Schw}(r_{iso})$')
    axes[1, 2].plot(r_iso, r_iso, 'k--', alpha=0.3, label=r'$r_{Schw} = r_{iso}$')
    axes[1, 2].axvline(R_iso, color='red', linestyle=':', alpha=0.7)
    axes[1, 2].set_xlabel(r'$r_{iso}$', fontsize=12)
    axes[1, 2].set_ylabel(r'$r_{Schw}$', fontsize=12)
    axes[1, 2].set_title('Coordinate Transformation', fontsize=12)
    axes[1, 2].set_xlim(0, r_max_iso)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f'TOV Solution (Isotropic): M={M_star:.4f}, R_schw={R_schw:.3f}, '
                 f'R_iso={R_iso:.3f}, C={tov_solution.C:.4f}',
                 fontsize=13, y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig


# =============================================================================
# Convergence Test Functions
# =============================================================================

def solve_tov_fixed_resolution(K, Gamma, rho_central, n_points, r_max_schw):
    """
    Solve TOV equations with fixed number of grid points using RK4.

    Uses Taylor expansion near origin to avoid singularity, then RK4
    with fixed step size for proper convergence testing.

    Args:
        K: Polytropic constant
        Gamma: Polytropic exponent
        rho_central: Central baryon density
        n_points: Number of grid points
        r_max_schw: Maximum Schwarzschild radius to integrate to

    Returns:
        Dictionary with r_schw, P, M, nu arrays and scalar values
    """
    eos = PolytropicEOS(K, Gamma)

    # Central values
    P_c = eos.P_from_rho_baryon(rho_central)
    rho_c = eos.rho_total_from_P(P_c)

    # Taylor expansion coefficients near origin
    P2 = -(2.0 * np.pi / 3.0) * (rho_c + P_c) * (rho_c + 3.0 * P_c)
    nu2 = (4.0 * np.pi / 3.0) * (rho_c + 3.0 * P_c)

    # Start from small radius using Taylor expansion
    r_taylor = min(0.05, r_max_schw * 0.1)
    r_start = r_taylor
    M_start = (4.0 * np.pi / 3.0) * rho_c * r_start**3
    P_start = P_c + P2 * r_start**2
    nu_start = nu2 * r_start**2
    r_iso_start = r_start

    # Fixed step size
    dr = (r_max_schw - r_start) / n_points

    # RK4 integrator
    def rk4_step(r, y, h):
        k1 = tov_rhs(r, y)
        k2 = tov_rhs(r + h/2, y + h/2 * k1)
        k3 = tov_rhs(r + h/2, y + h/2 * k2)
        k4 = tov_rhs(r + h, y + h * k3)
        return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    def tov_rhs(r, y):
        P, nu, M, r_iso = y

        # Metric factor
        one_minus_2m_r = 1.0 - 2.0 * M / r
        if one_minus_2m_r < 1e-10:
            return np.array([0.0, 0.0, 0.0, 0.0])

        # Exterior (vacuum): Schwarzschild solution
        if P <= 0:
            dP_dr = 0.0
            dM_dr = 0.0
            dnu_dr = 2.0 * M / (r * r * one_minus_2m_r)
        else:
            # Interior: TOV equations
            rho = eos.rho_total_from_P(P)
            dM_dr = 4.0 * np.pi * r * r * rho
            numerator = M + 4.0 * np.pi * r**3 * P
            denominator = r * r * one_minus_2m_r
            dP_dr = -(rho + P) * numerator / denominator
            dnu_dr = -2.0 / (P + rho) * dP_dr

        # Isotropic coordinate (same for interior and exterior)
        dr_iso_dr = (r_iso / r) / np.sqrt(one_minus_2m_r)

        return np.array([dP_dr, dnu_dr, dM_dr, dr_iso_dr])

    # Integration
    r_arr = [r_start]
    y = np.array([P_start, nu_start, M_start, r_iso_start])
    P_arr = [y[0]]
    nu_arr = [y[1]]
    M_arr = [y[2]]
    r_iso_arr = [y[3]]

    r = r_start
    for _ in range(n_points):
        y = rk4_step(r, y, dr)
        r += dr

        if y[0] < 0:
            y[0] = 0

        r_arr.append(r)
        P_arr.append(y[0])
        nu_arr.append(y[1])
        M_arr.append(y[2])
        r_iso_arr.append(y[3])

    # Compute lapse: alpha = exp(nu/2)
    nu_arr = np.array(nu_arr)
    alpha_arr = np.exp(nu_arr / 2.0)

    return {
        'r_schw': np.array(r_arr),
        'P': np.array(P_arr),
        'M': np.array(M_arr),
        'nu': nu_arr,
        'alpha': alpha_arr,
        'r_iso': np.array(r_iso_arr),
        'M_star': M_arr[-1],
        'alpha_end': alpha_arr[-1],
        'R_schw': r_arr[-1],
        'n_points': n_points,
        'dr': dr,
    }


def run_convergence_test(K, Gamma, rho_central, resolutions=None, save_path=None):
    """
    Run convergence test for the TOV solver.

    Computes solutions at different resolutions and calculates the
    convergence order using self-convergence (Richardson extrapolation).

    For RK4, expected order is 4.

    Args:
        K: Polytropic constant
        Gamma: Polytropic exponent
        rho_central: Central baryon density
        resolutions: List of resolutions (default: [100, 200, 400])
        save_path: Path to save convergence plot

    Returns:
        Dictionary with convergence results
    """
    if resolutions is None:
        resolutions = [100, 200, 400]

    resolutions = sorted(resolutions)

    print("="*70)
    print("TOV SOLVER CONVERGENCE TEST")
    print("="*70)
    print(f"EOS: P = {K} * rho^{Gamma}")
    print(f"Central density: rho_c = {rho_central:.6e}")
    print(f"Resolutions: {resolutions}")
    print("-"*70)

    # First find stellar radius with adaptive solver
    solver = TOVSolverIso(K, Gamma)
    ref_solution = solver.solve(rho_central, accuracy="low", verbose=False)
    R_star = ref_solution.R_schw
    r_test = 1.5 * R_star  # Test beyond stellar surface (includes exterior)

    print(f"Stellar radius R_schw = {R_star:.6f}")
    print(f"Test radius = {r_test:.6f} (150% of R_star, includes exterior)")
    print("-"*70)

    # Compute solutions at each resolution
    solutions = {}
    print("\nComputing fixed-resolution solutions...")
    for n in resolutions:
        print(f"  N = {n}...", end=" ", flush=True)
        sol = solve_tov_fixed_resolution(K, Gamma, rho_central, n, r_test)
        solutions[n] = sol
        print(f"M = {sol['M_star']:.10f}, dr = {sol['dr']:.6f}")

    # Self-convergence: compare consecutive resolutions
    print("\n" + "-"*70)
    print("SELF-CONVERGENCE ANALYSIS")
    print("-"*70)

    errors = {'M': [], 'alpha': []}

    for i in range(len(resolutions) - 1):
        n1, n2 = resolutions[i], resolutions[i+1]
        sol1, sol2 = solutions[n1], solutions[n2]

        # Compare endpoint values
        errors['M'].append(abs(sol1['M_star'] - sol2['M_star']))
        errors['alpha'].append(abs(sol1['alpha_end'] - sol2['alpha_end']))

    # Compute convergence orders
    print(f"\n{'Quantity':<12} | {'|u_N - u_2N|':<40} | {'Order':<10}")
    print("-"*70)

    orders = {}
    for key in errors:
        err_list = errors[key]
        ord_list = []

        for i in range(len(err_list) - 1):
            if err_list[i] > 1e-15 and err_list[i+1] > 1e-15 and err_list[i] > err_list[i+1]:
                # For ratio-2 refinement: order = log2(e1/e2)
                ratio = resolutions[i+2] / resolutions[i+1]
                order = np.log(err_list[i] / err_list[i+1]) / np.log(ratio)
                ord_list.append(order)
            else:
                ord_list.append(np.nan)

        orders[key] = ord_list

        err_str = ", ".join([f"{e:.2e}" for e in err_list])
        ord_str = ", ".join([f"{o:.2f}" if not np.isnan(o) else "N/A" for o in ord_list]) if ord_list else "N/A"
        print(f"{key:<12} | {err_str:<40} | {ord_str:<10}")

    print("-"*70)

    # Average order
    all_orders = [o for v in orders.values() for o in v if not np.isnan(o)]
    if all_orders:
        avg_order = np.mean(all_orders)
        print(f"\nAverage convergence order: {avg_order:.2f}")
        print(f"Expected for RK4: 4.0")

    # Plot
    if save_path or True:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: α(r) profiles
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(resolutions)))
        for i, n in enumerate(resolutions):
            sol = solutions[n]
            axes[0].plot(sol['r_schw'], sol['alpha'], '-', color=colors[i],
                        linewidth=1.5, label=f'N={n}')
        axes[0].set_xlabel(r'$r_{Schw}$', fontsize=12)
        axes[0].set_ylabel(r'$\alpha(r)$', fontsize=12)
        axes[0].set_title('Lapse Function', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Convergence
        x_pairs = [(resolutions[i] + resolutions[i+1])/2 for i in range(len(resolutions)-1)]
        axes[1].loglog(x_pairs, errors['M'], 'bo-', linewidth=2, markersize=10, label='|ΔM|')
        axes[1].loglog(x_pairs, errors['alpha'], 'rs-', linewidth=2, markersize=10, label='|Δα|')

        # Reference slopes (using alpha errors for scaling)
        x_ref = np.array([x_pairs[0], x_pairs[-1]])
        for order, ls in [(2, '--'), (4, ':')]:
            scale = errors['alpha'][0] * (x_pairs[0] / x_ref[0])**order
            axes[1].loglog(x_ref, scale * (x_ref[0] / x_ref)**order, f'k{ls}',
                          alpha=0.5, label=f'O(h^{order})')

        axes[1].set_xlabel('Resolution N', fontsize=12)
        axes[1].set_ylabel('|Difference|', fontsize=12)
        axes[1].set_title('Self-Convergence', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, which='both')

        plt.suptitle(f'TOV Convergence Test (K={K}, Γ={Gamma}, ρ_c={rho_central:.2e})', fontsize=13)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")

    print("="*70 + "\n")

    return {
        'resolutions': resolutions,
        'solutions': solutions,
        'errors': errors,
        'orders': orders,
        'average_order': avg_order if all_orders else None,
        'figure': fig if save_path or True else None,
    }


# =============================================================================
# Main Entry Point
# =============================================================================

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
    parser.add_argument('--convergence-test', action='store_true', dest='convergence_test',
                        help='Run convergence test with resolutions 100, 200, 400')
    parser.add_argument('--resolutions', type=str, default='100,200,400',
                        help='Comma-separated resolutions for convergence test')

    args = parser.parse_args()

    print("="*70)
    print("TOV Solver in Isotropic Coordinates")
    print("="*70)

    if args.convergence_test:
        # Run convergence test
        resolutions = [int(x.strip()) for x in args.resolutions.split(',')]
        convergence_path = args.output.replace('.png', '_convergence.png')
        results = run_convergence_test(args.K, args.Gamma, args.rho_central,
                                       resolutions=resolutions, save_path=convergence_path)
        plt.show()

    else:
        # Standard solve mode
        solver = TOVSolverIso(K=args.K, Gamma=args.Gamma)

        print(f"\nSolving with: P = {args.K} * rho^{args.Gamma}")
        print(f"Central density: ρ_c = {args.rho_central:.6e}")
        print(f"Accuracy: {args.accuracy}")
        print("-"*70)

        solution = solver.solve(args.rho_central, r_max_iso=args.r_max, accuracy=args.accuracy)

        print("\n" + "="*70)
        print("SOLUTION SUMMARY")
        print("="*70)
        print(f"  EOS            : P = {args.K} * rho^{args.Gamma}")
        print(f"  rho_central    : {args.rho_central:.6e}")
        print(f"  M_star         : {solution.M_star:.8f}")
        print(f"  R_schw         : {solution.R_schw:.8f}")
        print(f"  R_iso          : {solution.R_iso:.8f}")
        print(f"  Compactness    : {solution.C:.8f}")
        print(f"  Grid points    : {solution.num_points}")
        print(f"  alpha continuity: {solution.alpha_continuity_error:.2e}")
        print(f"  Converged      : {solution.integration_converged}")
        print("="*70)

        fig = plot_tov_iso_diagnostics(solution, args.r_max, save_path=args.output)
        plt.show()
