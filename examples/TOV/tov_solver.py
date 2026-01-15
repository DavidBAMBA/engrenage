"""
TOV solver in isotropic coordinates (Improved Version).

Solves TOV equations with coordinate transformation to isotropic coordinates.
The key difference from Schwarzschild coordinates is that we solve 4 ODEs
instead of 3, including the transformation dr_iso/dr_schw.

In isotropic coordinates, the metric is conformally flat:
    ds² = -alpha² dt² + e^{4φ} (dr_iso² + r_iso² dΩ²)

where e^{4φ} = (r_schw/r_iso)² is the conformal factor.

Improvements over original version:
- Decoupled EOS architecture (supports custom EOS classes)
- Improved adaptive step size control with explicit bounds
- Better stopping criterion (relative + stability check)
- Stores both rho_baryon and rho_total
- Surface continuity validation
- More robust near-origin handling

Usage:
  python tov_solver_iso_improved.py --K 100.0 --Gamma 2.0 --rho_central 1.28e-3 --r_max 20.0
"""

import numpy as np
import os

# Use non-interactive backend for headless environments
import matplotlib
if 'DISPLAY' not in os.environ or not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.integrate import ode
import warnings


# =============================================================================
# EOS Classes (Decoupled Architecture)
# =============================================================================

class EOSBase:
    """Base class for equations of state."""
    
    def P_from_rho_baryon(self, rho_b):
        """Compute pressure from baryon density."""
        raise NotImplementedError
    
    def rho_baryon_from_P(self, P):
        """Compute baryon density from pressure."""
        raise NotImplementedError
    
    def rho_total_from_P(self, P):
        """Compute total energy density ρ = ρ_b(1 + ε) from pressure."""
        raise NotImplementedError
    
    def eps_from_P(self, P):
        """Compute specific internal energy ε from pressure."""
        raise NotImplementedError


class PolytropicEOS(EOSBase):
    """
    Polytropic equation of state: P = K * ρ_b^Γ
    
    For a polytrope:
        ε = P / [(Γ-1) * ρ_b]  (specific internal energy)
        ρ_total = ρ_b * (1 + ε)  (total energy density)
    
    Parameters:
        K: Polytropic constant
        Gamma: Polytropic exponent (adiabatic index)
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
    
    def eps_from_P(self, P):
        """ε = P / [(Γ-1) * ρ_b]"""
        if P <= 0:
            return 0.0
        rho_b = self.rho_baryon_from_P(P)
        if rho_b <= 0:
            return 0.0
        return P / (self._Gamma_minus_1 * rho_b)
    
    def rho_total_from_P(self, P):
        """ρ_total = ρ_b * (1 + ε)"""
        if P <= 0:
            return 1e-30  # Small floor to avoid division issues
        rho_b = self.rho_baryon_from_P(P)
        if rho_b <= 0:
            return 1e-30
        eps = P / (self._Gamma_minus_1 * rho_b)
        return rho_b * (1.0 + eps)
    
    def __repr__(self):
        return f"PolytropicEOS(K={self.K}, Gamma={self.Gamma})"


class PiecewisePolytropicEOS(EOSBase):
    """
    Piecewise polytropic EOS for more realistic neutron star models.
    
    Each piece i has: P = K_i * ρ_b^Γ_i for ρ_{i-1} < ρ_b < ρ_i
    
    Parameters:
        rho_boundaries: Array of density boundaries [ρ_0, ρ_1, ..., ρ_n]
        Gamma_pieces: Array of Γ values for each piece [Γ_1, Γ_2, ..., Γ_n]
        K_0: K value for the first piece (others determined by continuity)
    """
    
    def __init__(self, rho_boundaries, Gamma_pieces, K_0):
        self.rho_boundaries = np.array(rho_boundaries)
        self.Gamma_pieces = np.array(Gamma_pieces)
        self.n_pieces = len(Gamma_pieces)
        
        # Compute K values for each piece (continuity condition)
        self.K_pieces = np.zeros(self.n_pieces)
        self.K_pieces[0] = K_0
        
        for i in range(1, self.n_pieces):
            rho_trans = self.rho_boundaries[i]
            P_trans = self.K_pieces[i-1] * rho_trans ** self.Gamma_pieces[i-1]
            self.K_pieces[i] = P_trans / (rho_trans ** self.Gamma_pieces[i])
    
    def _get_piece_index(self, rho_b):
        """Find which piece of the EOS applies for given density."""
        if rho_b <= 0:
            return 0
        for i in range(self.n_pieces - 1):
            if rho_b < self.rho_boundaries[i + 1]:
                return i
        return self.n_pieces - 1
    
    def P_from_rho_baryon(self, rho_b):
        if rho_b <= 0:
            return 0.0
        i = self._get_piece_index(rho_b)
        return self.K_pieces[i] * rho_b ** self.Gamma_pieces[i]
    
    def rho_baryon_from_P(self, P):
        if P <= 0:
            return 0.0
        # Need to find correct piece by iteration
        for i in range(self.n_pieces):
            rho_b = (P / self.K_pieces[i]) ** (1.0 / self.Gamma_pieces[i])
            if i == self.n_pieces - 1:
                return rho_b
            if rho_b < self.rho_boundaries[i + 1]:
                return rho_b
        return rho_b
    
    def eps_from_P(self, P):
        if P <= 0:
            return 0.0
        rho_b = self.rho_baryon_from_P(P)
        if rho_b <= 0:
            return 0.0
        i = self._get_piece_index(rho_b)
        Gamma = self.Gamma_pieces[i]
        return P / ((Gamma - 1.0) * rho_b)
    
    def rho_total_from_P(self, P):
        if P <= 0:
            return 1e-30
        rho_b = self.rho_baryon_from_P(P)
        eps = self.eps_from_P(P)
        return rho_b * (1.0 + eps)


# =============================================================================
# Cache Configuration
# =============================================================================

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


# =============================================================================
# TOV Solution Container
# =============================================================================

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
        self.rho_total = None      # Total energy density array (NEW)
        self.exp4phi = None        # e^(4φ) = (r_schw/r_iso)² conformal factor
        self.alpha = None          # Lapse function alpha(r) array

        # Scalars
        self.R_schw = None         # Stellar radius in Schwarzschild coords
        self.R_iso = None          # Stellar radius in isotropic coords
        self.M_star = None         # Total mass
        self.C = None              # Compactness M/R_schw

        # Grid info
        self.num_points = None     # Number of grid points
        self.surface_index = None  # Index of stellar surface (NEW)

        # EOS parameters
        self.K = None              # Polytropic constant K
        self.Gamma = None          # Polytropic exponent Γ
        self.rho_central = None    # Central density
        
        # Diagnostics (NEW)
        self.alpha_continuity_error = None  # Continuity error at surface
        self.integration_converged = True   # Did integration converge properly?

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
        if self.surface_index is not None:
            return self.surface_index
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
        np.save(os.path.join(cache_dir, "rho_total.npy"), self.rho_total)
        np.save(os.path.join(cache_dir, "exp4phi.npy"), self.exp4phi)
        np.save(os.path.join(cache_dir, "alpha.npy"), self.alpha)

        # Save scalars as numpy array
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
        
        # Load rho_total if exists (backward compatibility)
        rho_total_path = os.path.join(cache_dir, "rho_total.npy")
        if os.path.exists(rho_total_path):
            solution.rho_total = np.load(rho_total_path)
        else:
            solution.rho_total = None

        # Load scalars
        scalars = np.load(os.path.join(cache_dir, "scalars.npy"))
        solution.R_schw = scalars[0]
        solution.R_iso = scalars[1]
        solution.M_star = scalars[2]
        solution.C = scalars[3]
        solution.num_points = int(scalars[4])
        solution.K = scalars[5]
        solution.Gamma = scalars[6]
        solution.rho_central = scalars[7]
        
        # Load new fields if available (backward compatibility)
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


# =============================================================================
# Main TOV Solver Class
# =============================================================================

class TOVSolverIso:
    """
    TOV solver in isotropic coordinates (Improved Version).

    Solves the standard TOV equations plus the coordinate transformation
    equation to get the isotropic radial coordinate r_iso.

    The 4 ODEs solved (in r_schw as independent variable):
    1. dP/dr = -(ρ+P)(m+4πr³P) / [r²(1-2m/r)]
    2. dν/dr = -2/(P+ρ) × dP/dr
    3. dM/dr = 4πr²ρ
    4. dr_iso/dr = (r_iso/r) / √(1-2m/r)
    
    Improvements:
    - Decoupled EOS (accepts any EOSBase subclass)
    - Improved adaptive stepping with explicit bounds
    - Stability-aware stopping criterion
    - Surface continuity validation
    - Stores both rho_baryon and rho_total
    """

    def __init__(self, K=None, Gamma=None, eos=None):
        """
        Initialize TOV solver.
        
        Args:
            K: Polytropic constant (used if eos is None)
            Gamma: Polytropic exponent (used if eos is None)
            eos: EOSBase instance (takes precedence over K, Gamma)
        """
        if eos is not None:
            self.eos = eos
            # Try to extract K and Gamma for compatibility
            self.K = getattr(eos, 'K', None)
            self.Gamma = getattr(eos, 'Gamma', None)
        elif K is not None and Gamma is not None:
            self.eos = PolytropicEOS(K, Gamma)
            self.K = K
            self.Gamma = Gamma
        else:
            raise ValueError("Must provide either (K, Gamma) or eos parameter")

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
        
        # Get total energy density from EOS
        rho = self.eos.rho_total_from_P(P)
        
        # Numerical floor for safety
        if rho < 1e-30:
            rho = 1e-30

        # Near origin: use Taylor expansion to avoid numerical issues
        if r_schw < 1e-4 or M <= 0:
            # For small r, m ≈ (4π/3)r³ρ
            # dP/dr ≈ -(ρ+P)(4π/3 r ρ + 4π r P) / (1 - 8πρr²)
            denom = 1.0 - 8.0 * np.pi * rho * r_schw**2
            if abs(denom) < 1e-30:
                denom = 1e-30 * np.sign(denom) if denom != 0 else 1e-30
            
            dP_dr = -(rho + P) * (4.0*np.pi/3.0*r_schw*rho + 4.0*np.pi*r_schw*P) / denom
            
            # dν/dr = -2/(P+ρ) × dP/dr
            if (P + rho) > 1e-30:
                dnu_dr = -2.0 / (P + rho) * dP_dr
            else:
                dnu_dr = 0.0
            
            dM_dr = 4.0 * np.pi * r_schw**2 * rho
            
            # For r_iso near origin: dr_iso/dr_schw ≈ 1/√(1-8πρr²) ≈ 1
            if r_schw > 1e-10:
                dr_iso_dr = 1.0 / np.sqrt(max(denom, 1e-30))
            else:
                dr_iso_dr = 1.0
                
            return np.array([dP_dr, dnu_dr, dM_dr, dr_iso_dr])

        # Standard TOV equations for r > 1e-4
        one_minus_2m_r = 1.0 - 2.0 * M / r_schw
        
        # Near horizon protection
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

        # Coordinate transformation equation: dr_iso/dr_schw = (r_iso/r_schw) / √(1-2m/r)
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
        # Initial conditions
        P_c = self.eos.P_from_rho_baryon(rho_central)
        nu_c = 0.0
        M_c = 0.0

        # Initial r_schw and r_iso (start from small epsilon)
        r_start = 1e-10
        r_iso_start = r_start  # At origin, r_iso = r_schw

        # Choose integrator & step presets based on accuracy
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

        # Set up ODE solver
        solver = ode(self.tov_rhs_iso).set_integrator(integrator, atol=atol, rtol=rtol)
        solver.set_initial_value([P_c, nu_c, M_c, r_iso_start], r_start)

        # Storage arrays
        r_schw_arr = [r_start]
        r_iso_arr = [r_iso_start]
        P_arr = [P_c]
        nu_arr = [nu_c]
        M_arr = [M_c]

        # Integration loop with improved stopping criterion
        dr = min_step_size
        r_max_schw = 100.0  # Large enough to contain the star
        
        # Stopping thresholds
        P_threshold_relative = 1e-15  # Improved: more stringent
        integration_converged = True
        P_previous = P_c

        while solver.successful() and solver.t < r_max_schw:
            solver.integrate(solver.t + dr)

            r_now = solver.t
            P_now, nu_now, M_now, r_iso_now = solver.y

            # Check stopping conditions
            # 1. Pressure dropped below threshold
            if P_now < P_threshold_relative * P_c:
                if verbose:
                    print(f"  Stopped: P/P_c = {P_now/P_c:.2e} < {P_threshold_relative:.0e}")
                break
            
            # 2. Stability check: pressure should not increase
            if P_now > P_previous * 1.01 and len(P_arr) > 10:
                warnings.warn(f"Integration instability detected at r={r_now:.6e}: P increased")
                integration_converged = False
                break
            
            # 3. Negative pressure (unphysical)
            if P_now < 0:
                P_now = 0.0
                break

            P_previous = P_now
            
            # Store values
            r_schw_arr.append(r_now)
            r_iso_arr.append(r_iso_now)
            P_arr.append(max(P_now, 0.0))
            nu_arr.append(nu_now)
            M_arr.append(M_now)

            # Adaptive step size (improved with explicit bounds)
            dP_dr, dnu_dr, dM_dr, dr_iso_dr = self.tov_rhs_iso(r_now, [P_now, nu_now, M_now, r_iso_now])

            # Safe ratio computation
            def _safe_ratio(val, der, default):
                if abs(der) < 1e-30:
                    return default
                return abs(val / der)

            est_P = _safe_ratio(P_now, dP_dr, max_step_size) if P_now > 0 else max_step_size
            est_M = _safe_ratio(M_now if M_now > 0 else 1.0, dM_dr, max_step_size)
            
            dr = 0.1 * min(est_P, est_M)
            dr = np.clip(dr, min_step_size, max_step_size)  # Explicit bounds

        # Surface values
        R_schw_raw = r_schw_arr[-1]
        R_iso_raw = r_iso_arr[-1]
        M_star = M_arr[-1]
        surface_index_interior = len(r_schw_arr) - 1

        # Convert to numpy arrays
        r_schw_arr = np.array(r_schw_arr)
        r_iso_arr = np.array(r_iso_arr)
        P_arr = np.array(P_arr)
        nu_arr = np.array(nu_arr)
        M_arr = np.array(M_arr)

        # Normalize r_iso using the exact exterior solution
        # In exterior Schwarzschild: r_iso = 0.5 * [sqrt(R(R-2M)) + R - M]
        discriminant = R_schw_raw * (R_schw_raw - 2.0 * M_star)
        if discriminant < 0:
            warnings.warn(f"Star too compact! R_schw={R_schw_raw:.6e}, 2M={2*M_star:.6e}")
            discriminant = 0.0
            integration_converged = False
            
        R_iso_exact = 0.5 * (np.sqrt(discriminant) + R_schw_raw - M_star)
        
        if R_iso_raw > 0:
            normalization_factor = R_iso_exact / R_iso_raw
            r_iso_arr = r_iso_arr * normalization_factor
        
        # Updated surface radius in isotropic coords
        R_schw = R_schw_raw
        R_iso = R_iso_exact

        # IMPORTANT: Normalize nu BEFORE extending to exterior
        # At surface R_schw in Schwarzschild coords: nu = log(1 - 2M/R_schw)
        nu_surface_schw = np.log(1.0 - 2.0 * M_star / R_schw)
        nu_offset = nu_surface_schw - nu_arr[surface_index_interior]
        nu_arr = nu_arr + nu_offset

        # Store interior alpha for continuity check
        alpha_interior_surface = np.exp(nu_arr[surface_index_interior] / 2.0)

        # Extend to exterior with properly normalized analytic solution
        r_iso_current = r_iso_arr[-1]
        dr_iso_ext = max_step_size

        while r_iso_current < r_max_iso:
            r_iso_current += dr_iso_ext

            # In exterior: r_schw = r_iso * (1 + M/(2*r_iso))^2
            r_schw_ext = r_iso_current * (1.0 + M_star / (2.0 * r_iso_current))**2

            # Schwarzschild metric functions in isotropic coords
            # alpha = (1 - M/(2r_iso)) / (1 + M/(2r_iso))
            # exp(nu) = alpha^2
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

        # Conformal factor: exp(4φ) = (r_schw/r_iso)^2
        with np.errstate(divide='ignore', invalid='ignore'):
            exp4phi_arr = (r_schw_arr / r_iso_arr)**2
        exp4phi_arr[0] = 1.0  # At origin, r_schw = r_iso

        # Lapse: alpha = exp(nu/2)
        alpha_arr = np.exp(nu_arr / 2.0)

        # Validate continuity at surface
        alpha_exterior_surface = (1.0 - M_star / (2.0 * R_iso)) / (1.0 + M_star / (2.0 * R_iso))
        alpha_continuity_error = abs(alpha_interior_surface - alpha_exterior_surface)
        
        if alpha_continuity_error > 1e-6 and verbose:
            warnings.warn(f"alpha discontinuity at surface: interior={alpha_interior_surface:.8f}, "
                         f"exterior={alpha_exterior_surface:.8f}, error={alpha_continuity_error:.2e}")

        # Find surface more accurately using interpolation
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
# Plotting Functions
# =============================================================================

def plot_tov_iso_diagnostics(tov_solution, r_max_iso=None, save_path=None):
    """
    Plot TOV solution diagnostics in isotropic coordinates.

    Args:
        tov_solution: TOVSolutionIso object
        r_max_iso: Maximum radius for plots (in isotropic coords)
        save_path: If provided, save figure to this path
        
    Returns:
        matplotlib Figure object
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

    # Density vs r_iso
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

    # Enclosed Mass
    axes[0, 2].plot(r_iso, M, color='maroon', linewidth=1.5)
    axes[0, 2].axvline(R_iso, color='red', linestyle=':', alpha=0.7)
    axes[0, 2].axhline(M_star, color='gray', linestyle='--', alpha=0.3, label=f'M={M_star:.4f}')
    axes[0, 2].set_xlabel(r'$r_{iso}$', fontsize=12)
    axes[0, 2].set_ylabel(r'$M(r)$', fontsize=12)
    axes[0, 2].set_title('Enclosed Mass', fontsize=12)
    axes[0, 2].set_xlim(0, r_max_iso)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Lapse alpha(r_iso)
    axes[1, 0].plot(r_iso, alpha, color='purple', linewidth=1.5)
    axes[1, 0].axvline(R_iso, color='red', linestyle=':', alpha=0.7, label=f'R_iso={R_iso:.3f}')
    axes[1, 0].axhline(1.0, color='gray', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel(r'$r_{iso}$', fontsize=12)
    axes[1, 0].set_ylabel(r'$\alpha$', fontsize=12)
    axes[1, 0].set_title('Lapse Function', fontsize=12)
    axes[1, 0].set_xlim(0, r_max_iso)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Conformal factor exp(4φ)
    axes[1, 1].plot(r_iso, exp4phi, color='teal', linewidth=1.5)
    axes[1, 1].axvline(R_iso, color='red', linestyle=':', alpha=0.7)
    axes[1, 1].axhline(1.0, color='gray', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel(r'$r_{iso}$', fontsize=12)
    axes[1, 1].set_ylabel(r'$e^{4\phi}$', fontsize=12)
    axes[1, 1].set_title(r'Conformal Factor $e^{4\phi} = (r_{Schw}/r_{iso})^2$', fontsize=12)
    axes[1, 1].set_xlim(0, r_max_iso)
    axes[1, 1].grid(True, alpha=0.3)

    # Coordinate transformation r_schw vs r_iso
    axes[1, 2].plot(r_iso, r_schw, color='orange', linewidth=1.5, label=r'$r_{Schw}(r_{iso})$')
    axes[1, 2].plot(r_iso, r_iso, 'k--', alpha=0.3, label=r'$r_{Schw} = r_{iso}$')
    axes[1, 2].axvline(R_iso, color='red', linestyle=':', alpha=0.7)
    axes[1, 2].set_xlabel(r'$r_{iso}$', fontsize=12)
    axes[1, 2].set_ylabel(r'$r_{Schw}$', fontsize=12)
    axes[1, 2].set_title('Coordinate Transformation', fontsize=12)
    axes[1, 2].set_xlim(0, r_max_iso)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f'TOV Solution (Isotropic Coords): M={M_star:.4f}, R_schw={R_schw:.3f}, '
                 f'R_iso={R_iso:.3f}, C={tov_solution.C:.4f}',
                 fontsize=13, y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def plot_tov_iso_comparison(tov_solution, r_max_iso=None, save_path=None):
    """
    Plot comparison between interior numerical solution and exterior analytic solution.
    Useful for validating the matching at the stellar surface.
    
    Args:
        tov_solution: TOVSolutionIso object
        r_max_iso: Maximum radius for plots
        save_path: If provided, save figure to this path
        
    Returns:
        matplotlib Figure object
    """
    if r_max_iso is None:
        r_max_iso = min(tov_solution.r_iso[-1], 5 * tov_solution.R_iso)
    
    r_iso = tov_solution.r_iso
    R_iso = tov_solution.R_iso
    M_star = tov_solution.M_star
    alpha = tov_solution.alpha
    exp4phi = tov_solution.exp4phi
    
    # Compute analytic exterior solution for comparison
    r_ext = np.linspace(R_iso * 0.8, r_max_iso, 500)
    alpha_analytic = (1.0 - M_star/(2.0*r_ext)) / (1.0 + M_star/(2.0*r_ext))
    r_schw_analytic = r_ext * (1.0 + M_star/(2.0*r_ext))**2
    exp4phi_analytic = (r_schw_analytic / r_ext)**2
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Alpha comparison
    axes[0].plot(r_iso, alpha, 'b-', linewidth=2, label='Numerical')
    axes[0].plot(r_ext, alpha_analytic, 'r--', linewidth=1.5, label='Analytic (exterior)')
    axes[0].axvline(R_iso, color='green', linestyle=':', alpha=0.7, label=f'R_iso={R_iso:.3f}')
    axes[0].set_xlabel(r'$r_{iso}$', fontsize=12)
    axes[0].set_ylabel(r'$\alpha$', fontsize=12)
    axes[0].set_title('Lapse Function: Numerical vs Analytic', fontsize=12)
    axes[0].set_xlim(0, r_max_iso)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # exp4phi comparison
    axes[1].plot(r_iso, exp4phi, 'b-', linewidth=2, label='Numerical')
    axes[1].plot(r_ext, exp4phi_analytic, 'r--', linewidth=1.5, label='Analytic (exterior)')
    axes[1].axvline(R_iso, color='green', linestyle=':', alpha=0.7, label=f'R_iso={R_iso:.3f}')
    axes[1].set_xlabel(r'$r_{iso}$', fontsize=12)
    axes[1].set_ylabel(r'$e^{4\phi}$', fontsize=12)
    axes[1].set_title('Conformal Factor: Numerical vs Analytic', fontsize=12)
    axes[1].set_xlim(0, r_max_iso)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Surface Matching Validation (M={M_star:.4f}, C={tov_solution.C:.4f})',
                 fontsize=13)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    return fig


def plot_tov_iso_convergence(K, Gamma, rho_central, accuracies=None, r_max=20.0, save_path=None):
    """
    Plot convergence study comparing solutions at different accuracy levels.
    
    Args:
        K: Polytropic constant
        Gamma: Polytropic exponent
        rho_central: Central density
        accuracies: List of accuracy presets to compare
        r_max: Maximum radius
        save_path: If provided, save figure to this path
        
    Returns:
        matplotlib Figure object and dict of solutions
    """
    if accuracies is None:
        accuracies = ["low", "medium", "high"]
    
    solutions = {}
    for acc in accuracies:
        solver = TOVSolverIso(K=K, Gamma=Gamma)
        solutions[acc] = solver.solve(rho_central, r_max_iso=r_max, accuracy=acc, verbose=False)
    
    # Reference solution (highest accuracy)
    ref_key = accuracies[-1]
    ref = solutions[ref_key]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(accuracies)))
    
    # Mass comparison
    for i, acc in enumerate(accuracies):
        sol = solutions[acc]
        axes[0].plot(sol.r_iso, sol.M, color=colors[i], linewidth=1.5, 
                    label=f'{acc}: M={sol.M_star:.6f}')
    axes[0].set_xlabel(r'$r_{iso}$', fontsize=12)
    axes[0].set_ylabel(r'$M(r)$', fontsize=12)
    axes[0].set_title('Enclosed Mass', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 2*ref.R_iso)
    
    # Alpha comparison
    for i, acc in enumerate(accuracies):
        sol = solutions[acc]
        axes[1].plot(sol.r_iso, sol.alpha, color=colors[i], linewidth=1.5, label=acc)
    axes[1].axvline(ref.R_iso, color='red', linestyle=':', alpha=0.5)
    axes[1].set_xlabel(r'$r_{iso}$', fontsize=12)
    axes[1].set_ylabel(r'$\alpha$', fontsize=12)
    axes[1].set_title('Lapse Function', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, r_max)
    
    # Convergence in global quantities
    M_vals = [solutions[acc].M_star for acc in accuracies]
    R_vals = [solutions[acc].R_iso for acc in accuracies]
    
    ax2 = axes[2]
    ax2.bar(range(len(accuracies)), M_vals, alpha=0.7, label='M_star')
    ax2.set_xticks(range(len(accuracies)))
    ax2.set_xticklabels(accuracies)
    ax2.set_ylabel('M_star', fontsize=12)
    ax2.set_title('Global Quantities vs Accuracy', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add R_iso on secondary axis
    ax2b = ax2.twinx()
    ax2b.plot(range(len(accuracies)), R_vals, 'ro-', linewidth=2, markersize=8, label='R_iso')
    ax2b.set_ylabel('R_iso', color='red', fontsize=12)
    ax2b.tick_params(axis='y', labelcolor='red')
    
    plt.suptitle(f'Convergence Study (K={K}, Γ={Gamma}, ρ_c={rho_central:.2e})', fontsize=13)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Convergence plot saved to: {save_path}")
    
    return fig, solutions


# =============================================================================
# Convergence Test Functions
# =============================================================================

def solve_tov_fixed_resolution(K, Gamma, rho_central, n_points, r_max_schw=None, r_taylor=0.05):
    """
    Solve TOV equations with a fixed number of grid points using RK4.
    
    This uses a simple 4th-order Runge-Kutta integrator with fixed step size
    for proper convergence testing. 
    
    KEY INSIGHT: The TOV equations have a removable singularity at r=0 that
    degrades convergence to 2nd order if we integrate through it. To achieve
    4th order convergence, we use Taylor expansion up to r_taylor, then
    integrate with RK4 from there.
    
    Args:
        K: Polytropic constant
        Gamma: Polytropic exponent  
        rho_central: Central baryon density
        n_points: Number of grid points for RK4 integration
        r_max_schw: Maximum Schwarzschild radius to integrate to
        r_taylor: Radius up to which we use Taylor expansion (default: 0.05)
        
    Returns:
        TOVSolutionIso object
    """
    eos = PolytropicEOS(K, Gamma)
    
    # Central values
    P_c = eos.P_from_rho_baryon(rho_central)
    rho_c = eos.rho_total_from_P(P_c)
    
    if r_max_schw is None:
        raise ValueError("r_max_schw must be specified for fixed resolution solve")
    
    # Ensure r_taylor < r_max_schw
    r_taylor = min(r_taylor, r_max_schw * 0.1)
    
    # Taylor expansion coefficients for TOV near origin:
    #   M(r) = (4π/3) ρ_c r³ + O(r⁵)
    #   P(r) = P_c + P₂ r² + O(r⁴)  where P₂ = -(2π/3)(ρ_c + P_c)(ρ_c + 3P_c)
    #   ν(r) = ν_c + ν₂ r² + O(r⁴)  where ν₂ = (4π/3)(ρ_c + 3P_c)
    #   r_iso(r) ≈ r (to leading order)
    
    P2 = -(2.0 * np.pi / 3.0) * (rho_c + P_c) * (rho_c + 3.0 * P_c)
    nu2 = (4.0 * np.pi / 3.0) * (rho_c + 3.0 * P_c)
    
    # Initial values at r_taylor from Taylor expansion
    r_start = r_taylor
    M_start = (4.0 * np.pi / 3.0) * rho_c * r_start**3
    P_start = P_c + P2 * r_start**2
    nu_start = nu2 * r_start**2  # ν_c = 0
    r_iso_start = r_start  # Leading order
    
    # Fixed step size for RK4 from r_taylor to r_max_schw
    dr = (r_max_schw - r_start) / n_points
    
    # RK4 integration
    def rk4_step(r, y, h, rhs_func):
        k1 = rhs_func(r, y)
        k2 = rhs_func(r + h/2, y + h/2 * k1)
        k3 = rhs_func(r + h/2, y + h/2 * k2)
        k4 = rhs_func(r + h, y + h * k3)
        return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    def tov_rhs(r, y):
        return _tov_rhs_for_convergence(r, y, eos)
    
    # Storage - start with Taylor values
    r_schw_arr = [r_start]
    y = np.array([P_start, nu_start, M_start, r_iso_start])
    P_arr = [y[0]]
    nu_arr = [y[1]]
    M_arr = [y[2]]
    r_iso_arr = [y[3]]
    
    # Integrate with fixed RK4 steps from r_taylor to r_max_schw
    r = r_start
    for i in range(n_points):
        y = rk4_step(r, y, dr, tov_rhs)
        r += dr
        
        P, nu, M, r_iso = y
        
        # Ensure P stays positive
        if P < 0:
            P = 0
            y[0] = 0
        
        r_schw_arr.append(r)
        P_arr.append(P)
        nu_arr.append(nu)
        M_arr.append(M)
        r_iso_arr.append(r_iso)
    
    # Convert to arrays
    r_schw_arr = np.array(r_schw_arr)
    r_iso_arr = np.array(r_iso_arr)
    P_arr = np.array(P_arr)
    nu_arr = np.array(nu_arr)
    M_arr = np.array(M_arr)
    
    # Surface values at the integration endpoint
    R_schw = r_schw_arr[-1]
    M_star = M_arr[-1]
    R_iso = r_iso_arr[-1]
    
    # Compute derived quantities
    rho_baryon_arr = np.array([eos.rho_baryon_from_P(P) for P in P_arr])
    rho_total_arr = np.array([eos.rho_total_from_P(P) for P in P_arr])
    alpha_arr = np.exp(nu_arr / 2.0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        exp4phi_arr = np.where(r_iso_arr > 0, (r_schw_arr / r_iso_arr)**2, 1.0)
    exp4phi_arr[0] = 1.0
    
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
    solution.K = K
    solution.Gamma = Gamma
    solution.rho_central = rho_central
    solution.integration_converged = True
    
    return solution


def _tov_rhs_for_convergence(r_schw, y, eos):
    """
    TOV RHS function for convergence testing - NO BRANCHING VERSION.
    
    Uses ONLY the exact formulas without ANY if/else branching
    to achieve 4th order convergence with RK4.
    
    REQUIREMENTS for this to work:
    1. Start integration from r_start > 0 (e.g., 1e-6)
    2. Initialize M = (4π/3)ρ_c * r_start³ > 0
    3. Initialize r_iso = r_start > 0
    
    With these conditions, M > 0 and r_iso > 0 throughout integration,
    and all formulas are well-defined without branching.
    """
    P, nu, M, r_iso = y
    rho = eos.rho_total_from_P(max(P, 1e-30))
    
    # All formulas assume r > 0, M > 0, r_iso > 0
    # This is guaranteed by proper initialization
    
    # Mass equation
    dM_dr = 4.0 * np.pi * r_schw * r_schw * rho
    
    # Metric factor
    one_minus_2m_r = 1.0 - 2.0 * M / r_schw
    
    # Pressure equation: dP/dr = -(ρ+P)(M + 4πr³P) / [r²(1-2M/r)]
    numerator = M + 4.0 * np.pi * r_schw**3 * P
    denominator = r_schw * r_schw * one_minus_2m_r
    dP_dr = -(rho + P) * numerator / denominator
    
    # Metric function nu
    dnu_dr = -2.0 / (P + rho) * dP_dr
    
    # Isotropic coordinate: dr_iso/dr = (r_iso/r) / sqrt(1-2M/r)
    dr_iso_dr = (r_iso / r_schw) / np.sqrt(one_minus_2m_r)

    return np.array([dP_dr, dnu_dr, dM_dr, dr_iso_dr])


def interpolate_to_common_grid(solution, r_common):
    """
    Interpolate TOV solution to a common radial grid.
    
    Args:
        solution: TOVSolutionIso object
        r_common: Common r_schw grid to interpolate to
        
    Returns:
        Dictionary with interpolated quantities
    """
    from scipy.interpolate import interp1d
    
    # Only interpolate within the solution domain
    r_max = solution.r_schw[-1]
    mask = r_common <= r_max * 0.99  # Stay slightly inside domain
    r_interp = r_common[mask]
    
    if len(r_interp) == 0:
        return {'r_schw': np.array([]), 'mask': mask}
    
    result = {}
    
    # Interpolate each quantity
    for name, arr in [('P', solution.P), 
                      ('M', solution.M),
                      ('nu', solution.nu),
                      ('alpha', solution.alpha),
                      ('rho_baryon', solution.rho_baryon),
                      ('r_iso', solution.r_iso)]:
        if arr is not None and len(arr) == len(solution.r_schw):
            try:
                f = interp1d(solution.r_schw, arr, kind='cubic', 
                            bounds_error=False, fill_value=0.0)
                result[name] = f(r_interp)
            except Exception:
                result[name] = np.zeros_like(r_interp)
    
    result['r_schw'] = r_interp
    result['mask'] = mask
    
    return result


def compute_convergence_order(errors, resolutions):
    """
    Compute convergence order from errors at different resolutions.
    
    For errors scaling as e ~ h^p, we have:
        p = log(e1/e2) / log(h1/h2)
    
    where h = 1/N (step size proportional to 1/resolution)
    
    Args:
        errors: List of error values at each resolution
        resolutions: List of resolution values (number of points)
        
    Returns:
        List of convergence orders (one less than number of resolutions)
    """
    orders = []
    for i in range(len(errors) - 1):
        e1, e2 = errors[i], errors[i+1]
        n1, n2 = resolutions[i], resolutions[i+1]
        
        # Need both errors to be meaningful
        if e1 > 1e-14 and e2 > 1e-14 and e1 > e2:
            # h1/h2 = N2/N1 (finer grid has smaller h)
            h_ratio = n2 / n1
            e_ratio = e1 / e2
            order = np.log(e_ratio) / np.log(h_ratio)
            orders.append(order)
        elif e1 > 1e-14 and e2 > 1e-14 and e2 >= e1:
            # Error not decreasing - convergence failed
            orders.append(0.0)
        else:
            # At machine precision
            orders.append(np.nan)
    return orders



def _find_stellar_radius_fast(K, Gamma, rho_central):
    """Quickly find stellar radius using simple RK4 without exterior extension."""
    eos = PolytropicEOS(K, Gamma)
    P_c = eos.P_from_rho_baryon(rho_central)
    rho_c = eos.rho_total_from_P(P_c)

    # Taylor expansion start
    r_start = 0.01
    P2 = -(2.0 * np.pi / 3.0) * (rho_c + P_c) * (rho_c + 3.0 * P_c)
    M_start = (4.0 * np.pi / 3.0) * rho_c * r_start**3
    P_start = P_c + P2 * r_start**2

    # Simple integration to find surface
    def tov_rhs_simple(r, P, M):
        rho = eos.rho_total_from_P(max(P, 1e-30))
        if r < 1e-10:
            return 0.0, 4.0 * np.pi * r**2 * rho
        one_minus_2m_r = 1.0 - 2.0 * M / r
        if one_minus_2m_r < 1e-10:
            return 0.0, 0.0
        dP = -(rho + P) * (M + 4.0 * np.pi * r**3 * P) / (r**2 * one_minus_2m_r)
        dM = 4.0 * np.pi * r**2 * rho
        return dP, dM

    r, P, M = r_start, P_start, M_start
    dr = 0.01
    while P > 1e-15 * P_c and r < 100:
        dP, dM = tov_rhs_simple(r, P, M)
        P += dr * dP
        M += dr * dM
        r += dr
        if P < 0:
            break

    return r, M


def solve_tov_with_exterior(K, Gamma, rho_central, n_points_interior, n_points_exterior,
                             r_max_exterior=None, r_taylor=0.05):
    """
    Solve TOV equations with full exterior region for convergence testing.

    This version integrates the interior with RK4, then extends to the exterior
    using the analytical Schwarzschild solution. Returns both numerical and
    analytical values at matching points for discontinuity analysis.

    Args:
        K: Polytropic constant
        Gamma: Polytropic exponent
        rho_central: Central baryon density
        n_points_interior: Number of grid points for interior RK4 integration
        n_points_exterior: Number of grid points for exterior extension
        r_max_exterior: Maximum exterior radius (default: 2 * R_star)
        r_taylor: Radius up to which we use Taylor expansion (default: 0.05)

    Returns:
        Dictionary with solution and surface matching info
    """
    eos = PolytropicEOS(K, Gamma)

    # Fast radius finding (no exterior extension)
    R_star, M_star_approx = _find_stellar_radius_fast(K, Gamma, rho_central)

    if r_max_exterior is None:
        r_max_exterior = 2.0 * R_star

    # Central values
    P_c = eos.P_from_rho_baryon(rho_central)
    rho_c = eos.rho_total_from_P(P_c)

    # Taylor expansion coefficients
    P2 = -(2.0 * np.pi / 3.0) * (rho_c + P_c) * (rho_c + 3.0 * P_c)
    nu2 = (4.0 * np.pi / 3.0) * (rho_c + 3.0 * P_c)

    # Initial values at r_taylor
    r_taylor = min(r_taylor, R_star * 0.1)
    r_start = r_taylor
    M_start = (4.0 * np.pi / 3.0) * rho_c * r_start**3
    P_start = P_c + P2 * r_start**2
    nu_start = nu2 * r_start**2
    r_iso_start = r_start

    # Interior integration
    dr = (R_star - r_start) / n_points_interior

    def rk4_step(r, y, h, rhs_func):
        k1 = rhs_func(r, y)
        k2 = rhs_func(r + h/2, y + h/2 * k1)
        k3 = rhs_func(r + h/2, y + h/2 * k2)
        k4 = rhs_func(r + h, y + h * k3)
        return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    def tov_rhs(r, y):
        return _tov_rhs_for_convergence(r, y, eos)

    # Storage for interior
    r_schw_int = [r_start]
    y = np.array([P_start, nu_start, M_start, r_iso_start])
    P_int = [y[0]]
    nu_int = [y[1]]
    M_int = [y[2]]
    r_iso_int = [y[3]]

    # Integrate interior
    r = r_start
    surface_idx = n_points_interior
    for i in range(n_points_interior):
        y = rk4_step(r, y, dr, tov_rhs)
        r += dr
        P, nu, M, r_iso = y

        if P < 0:
            P = 0
            y[0] = 0
            surface_idx = i + 1
            break

        r_schw_int.append(r)
        P_int.append(P)
        nu_int.append(nu)
        M_int.append(M)
        r_iso_int.append(r_iso)

    # Convert to arrays
    r_schw_int = np.array(r_schw_int)
    r_iso_int = np.array(r_iso_int)
    P_int = np.array(P_int)
    nu_int = np.array(nu_int)
    M_int = np.array(M_int)

    # Surface values from numerical integration
    R_schw_num = r_schw_int[-1]
    R_iso_num = r_iso_int[-1]
    M_star = M_int[-1]
    nu_surface_num = nu_int[-1]

    # Analytical exterior solution at the surface
    # r_iso_exact = 0.5 * [sqrt(R(R-2M)) + R - M]
    discriminant = R_schw_num * (R_schw_num - 2.0 * M_star)
    if discriminant < 0:
        discriminant = 0.0
    R_iso_exact = 0.5 * (np.sqrt(discriminant) + R_schw_num - M_star)

    # Normalization factor for r_iso
    if R_iso_num > 0:
        norm_factor = R_iso_exact / R_iso_num
        r_iso_int_normalized = r_iso_int * norm_factor
    else:
        norm_factor = 1.0
        r_iso_int_normalized = r_iso_int

    R_iso_normalized = R_iso_exact

    # Normalize nu so it matches Schwarzschild at surface
    nu_surface_schw = np.log(1.0 - 2.0 * M_star / R_schw_num)
    nu_offset = nu_surface_schw - nu_surface_num
    nu_int_normalized = nu_int + nu_offset

    # Compute alpha from normalized nu
    alpha_int = np.exp(nu_int_normalized / 2.0)
    alpha_surface_num = alpha_int[-1]

    # Analytical alpha at surface
    alpha_surface_analytic = (1.0 - M_star / (2.0 * R_iso_normalized)) / \
                             (1.0 + M_star / (2.0 * R_iso_normalized))

    # Conformal factor at surface
    exp4phi_surface_num = (R_schw_num / R_iso_normalized)**2
    r_schw_analytic = R_iso_normalized * (1.0 + M_star / (2.0 * R_iso_normalized))**2
    exp4phi_surface_analytic = (r_schw_analytic / R_iso_normalized)**2

    # Extend to exterior
    r_iso_ext = []
    r_schw_ext = []
    alpha_ext_num = []  # Using numerical continuation (just analytical from M)
    alpha_ext_analytic = []
    exp4phi_ext_num = []
    exp4phi_ext_analytic = []

    dr_iso_ext = (r_max_exterior - R_iso_normalized) / n_points_exterior if n_points_exterior > 0 else 0.1
    r_iso_curr = R_iso_normalized

    for i in range(n_points_exterior):
        r_iso_curr += dr_iso_ext
        r_iso_ext.append(r_iso_curr)

        # Analytical exterior
        r_schw_a = r_iso_curr * (1.0 + M_star / (2.0 * r_iso_curr))**2
        alpha_a = (1.0 - M_star / (2.0 * r_iso_curr)) / (1.0 + M_star / (2.0 * r_iso_curr))
        exp4phi_a = (r_schw_a / r_iso_curr)**2

        r_schw_ext.append(r_schw_a)
        alpha_ext_analytic.append(alpha_a)
        exp4phi_ext_analytic.append(exp4phi_a)

        # "Numerical" exterior (same as analytical since we just use M_star)
        alpha_ext_num.append(alpha_a)
        exp4phi_ext_num.append(exp4phi_a)

    result = {
        # Interior solution
        'r_schw_int': r_schw_int,
        'r_iso_int': r_iso_int_normalized,
        'P_int': P_int,
        'M_int': M_int,
        'nu_int': nu_int_normalized,
        'alpha_int': alpha_int,

        # Surface values (numerical)
        'R_schw': R_schw_num,
        'R_iso': R_iso_normalized,
        'M_star': M_star,
        'alpha_surface_num': alpha_surface_num,
        'exp4phi_surface_num': exp4phi_surface_num,

        # Surface values (analytical)
        'alpha_surface_analytic': alpha_surface_analytic,
        'exp4phi_surface_analytic': exp4phi_surface_analytic,

        # Surface discontinuities
        'alpha_discontinuity': abs(alpha_surface_num - alpha_surface_analytic),
        'exp4phi_discontinuity': abs(exp4phi_surface_num - exp4phi_surface_analytic),

        # Normalization info
        'r_iso_norm_factor': norm_factor,
        'nu_offset': nu_offset,

        # Exterior
        'r_iso_ext': np.array(r_iso_ext),
        'r_schw_ext': np.array(r_schw_ext),
        'alpha_ext': np.array(alpha_ext_analytic),

        # Parameters
        'n_points_interior': n_points_interior,
        'n_points_exterior': n_points_exterior,
    }

    return result


def run_exterior_convergence_test(K, Gamma, rho_central, resolutions=None, save_path=None, verbose=True):
    """
    Run convergence test specifically for the exterior solution and surface matching.

    Tests:
    1. Convergence of alpha discontinuity at surface
    2. Convergence of exp(4φ) discontinuity at surface
    3. Convergence of r_iso normalization factor
    4. Interior solution convergence (for comparison)

    Args:
        K: Polytropic constant
        Gamma: Polytropic exponent
        rho_central: Central baryon density
        resolutions: List of resolutions (should have ratio 2, e.g., [100, 200, 400])
        save_path: If provided, save convergence plot to this path
        verbose: Print detailed output

    Returns:
        Dictionary containing convergence results and matplotlib Figure
    """
    if resolutions is None:
        resolutions = [100, 200, 400]

    resolutions = sorted(resolutions)

    print("\n" + "="*70)
    print("EXTERIOR & SURFACE MATCHING CONVERGENCE TEST")
    print("="*70)
    print(f"EOS: P = {K} × ρ^{Gamma}")
    print(f"Central density: ρ_c = {rho_central:.6e}")
    print(f"Resolutions: {resolutions}")
    print("-"*70)

    # Compute solutions at each resolution
    solutions = {}
    print("\nComputing solutions with interior + exterior...")
    for n in resolutions:
        print(f"  N = {n}...", end=" ", flush=True)
        sol = solve_tov_with_exterior(K, Gamma, rho_central,
                                       n_points_interior=n,
                                       n_points_exterior=100)
        solutions[n] = sol
        print(f"M = {sol['M_star']:.10f}, Δalpha = {sol['alpha_discontinuity']:.2e}, "
              f"Δψ = {sol['exp4phi_discontinuity']:.2e}")

    # Collect errors for convergence analysis
    errors = {
        'alpha_disc': [],           # alpha discontinuity at surface
        'exp4phi_disc': [],         # exp(4φ) discontinuity at surface
        'r_iso_norm': [],           # Deviation of r_iso normalization from 1
        'M_star_diff': [],          # M_star difference from finest resolution
        'R_schw_diff': [],          # R_schw difference from finest resolution
    }

    # Reference solution (finest resolution)
    ref = solutions[resolutions[-1]]

    for n in resolutions:
        sol = solutions[n]
        errors['alpha_disc'].append(sol['alpha_discontinuity'])
        errors['exp4phi_disc'].append(sol['exp4phi_discontinuity'])
        errors['r_iso_norm'].append(abs(sol['r_iso_norm_factor'] - 1.0))
        errors['M_star_diff'].append(abs(sol['M_star'] - ref['M_star']))
        errors['R_schw_diff'].append(abs(sol['R_schw'] - ref['R_schw']))

    # Self-convergence for discontinuities
    disc_diffs = {
        'alpha_disc': [],
        'exp4phi_disc': [],
    }

    for i in range(len(resolutions) - 1):
        sol1, sol2 = solutions[resolutions[i]], solutions[resolutions[i+1]]
        disc_diffs['alpha_disc'].append(abs(sol1['alpha_discontinuity'] - sol2['alpha_discontinuity']))
        disc_diffs['exp4phi_disc'].append(abs(sol1['exp4phi_discontinuity'] - sol2['exp4phi_discontinuity']))

    # Compute convergence orders
    def compute_order(vals, resolutions):
        orders = []
        for i in range(len(vals) - 1):
            if vals[i] > 1e-15 and vals[i+1] > 1e-15 and vals[i] > vals[i+1]:
                h_ratio = resolutions[i+1] / resolutions[i]
                e_ratio = vals[i] / vals[i+1]
                orders.append(np.log(e_ratio) / np.log(h_ratio))
            else:
                orders.append(np.nan)
        return orders

    orders = {}
    for key in errors:
        orders[key] = compute_order(errors[key], resolutions)

    # Print results
    print("\n" + "-"*70)
    print("SURFACE MATCHING RESULTS")
    print("-"*70)
    print(f"{'Resolution':<12} | {'Δalpha surface':<15} | {'Δψ surface':<15} | {'|norm-1|':<12}")
    print("-"*70)
    for n in resolutions:
        sol = solutions[n]
        print(f"N = {n:<6} | {sol['alpha_discontinuity']:<15.2e} | "
              f"{sol['exp4phi_discontinuity']:<15.2e} | {abs(sol['r_iso_norm_factor']-1):<12.2e}")

    print("\n" + "-"*70)
    print("CONVERGENCE ANALYSIS")
    print("-"*70)
    print(f"{'Quantity':<20} | {'Values at each N':<40} | {'Order':<15}")
    print("-"*70)

    for key in ['alpha_disc', 'exp4phi_disc', 'r_iso_norm', 'M_star_diff']:
        vals_str = ", ".join([f"{v:.2e}" for v in errors[key]])
        ord_str = ", ".join([f"{o:.2f}" if not np.isnan(o) else "N/A" for o in orders[key]]) if orders[key] else "N/A"
        print(f"{key:<20} | {vals_str:<40} | {ord_str:<15}")

    print("-"*70)

    # Key insight: Check if discontinuity is due to truncation or fundamental
    print("\n" + "-"*70)
    print("INTERPRETATION")
    print("-"*70)

    finest_sol = solutions[resolutions[-1]]
    alpha_disc = finest_sol['alpha_discontinuity']
    exp4phi_disc = finest_sol['exp4phi_discontinuity']

    if alpha_disc < 1e-10:
        print(f"  alpha discontinuity at N={resolutions[-1]}: {alpha_disc:.2e} (EXCELLENT - machine precision)")
    elif alpha_disc < 1e-6:
        print(f"  alpha discontinuity at N={resolutions[-1]}: {alpha_disc:.2e} (GOOD - converging)")
    else:
        print(f"  alpha discontinuity at N={resolutions[-1]}: {alpha_disc:.2e} (CHECK - may indicate matching issue)")

    if exp4phi_disc < 1e-10:
        print(f"  ψ discontinuity at N={resolutions[-1]}: {exp4phi_disc:.2e} (EXCELLENT - machine precision)")
    elif exp4phi_disc < 1e-6:
        print(f"  ψ discontinuity at N={resolutions[-1]}: {exp4phi_disc:.2e} (GOOD - converging)")
    else:
        print(f"  ψ discontinuity at N={resolutions[-1]}: {exp4phi_disc:.2e} (CHECK - may indicate matching issue)")

    # Note about the normalization
    norm_factor = finest_sol['r_iso_norm_factor']
    print(f"\n  r_iso normalization factor: {norm_factor:.10f}")
    print(f"  (Deviation from 1: {abs(norm_factor - 1):.2e})")
    print("  This factor corrects numerical r_iso to match analytical exterior.")

    print("-"*70)

    # Create convergence plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Surface discontinuities vs resolution
    ax = axes[0, 0]
    ax.loglog(resolutions, errors['alpha_disc'], 'o-', linewidth=2, markersize=10, label='alpha discontinuity')
    ax.loglog(resolutions, errors['exp4phi_disc'], 's-', linewidth=2, markersize=10, label='ψ discontinuity')

    # Reference slopes
    r_ref = np.array([resolutions[0], resolutions[-1]])
    for order, ls in [(2, '--'), (4, ':')]:
        if errors['alpha_disc'][0] > 0:
            scale = errors['alpha_disc'][0] * (resolutions[0] / r_ref[0])**order
            ax.loglog(r_ref, scale * (r_ref[0] / r_ref)**order, f'k{ls}', alpha=0.5, label=f'O(h^{order})')

    ax.set_xlabel('Resolution N', fontsize=12)
    ax.set_ylabel('Surface Discontinuity', fontsize=12)
    ax.set_title('Surface Matching Errors', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    # Plot 2: r_iso normalization convergence
    ax = axes[0, 1]
    ax.loglog(resolutions, errors['r_iso_norm'], 'o-', linewidth=2, markersize=10, color='green')
    ax.set_xlabel('Resolution N', fontsize=12)
    ax.set_ylabel('|norm_factor - 1|', fontsize=12)
    ax.set_title('r_iso Normalization Error', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')

    # Plot 3: Global quantities convergence
    ax = axes[0, 2]
    if errors['M_star_diff'][-1] > 0:  # Skip if all same
        ax.loglog(resolutions[:-1], errors['M_star_diff'][:-1], 'o-', linewidth=2, markersize=10, label='|ΔM_star|')
        ax.loglog(resolutions[:-1], errors['R_schw_diff'][:-1], 's-', linewidth=2, markersize=10, label='|ΔR_schw|')
        ax.legend(fontsize=9)
    ax.set_xlabel('Resolution N', fontsize=12)
    ax.set_ylabel('Difference from finest', fontsize=12)
    ax.set_title('Global Quantities Convergence', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')

    # Plot 4: alpha profile near surface
    ax = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(resolutions)))
    for i, n in enumerate(resolutions):
        sol = solutions[n]
        # Interior
        ax.plot(sol['r_iso_int'], sol['alpha_int'], '-', color=colors[i], linewidth=1.5, label=f'N={n}')

    # Analytical exterior for reference
    r_iso_plot = np.linspace(finest_sol['R_iso'], finest_sol['R_iso'] * 1.5, 100)
    alpha_analytic = (1.0 - finest_sol['M_star']/(2.0*r_iso_plot)) / (1.0 + finest_sol['M_star']/(2.0*r_iso_plot))
    ax.plot(r_iso_plot, alpha_analytic, 'k--', linewidth=2, label='Analytical ext.')
    ax.axvline(finest_sol['R_iso'], color='red', linestyle=':', alpha=0.7, label='Surface')

    ax.set_xlabel(r'$r_{iso}$', fontsize=12)
    ax.set_ylabel(r'$\alpha$', fontsize=12)
    ax.set_title('Lapse Near Surface', fontsize=12)
    ax.set_xlim(finest_sol['R_iso'] * 0.8, finest_sol['R_iso'] * 1.3)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 5: Zoom on surface discontinuity
    ax = axes[1, 1]
    for i, n in enumerate(resolutions):
        sol = solutions[n]
        ax.scatter([n], [sol['alpha_surface_num']], color=colors[i], s=100, marker='o',
                  label=f'N={n} (num)' if i == 0 else None)
        ax.scatter([n], [sol['alpha_surface_analytic']], color=colors[i], s=100, marker='x')

    # Add horizontal line for analytical value
    ax.axhline(finest_sol['alpha_surface_analytic'], color='red', linestyle='--',
              label=f'Analytical: {finest_sol["alpha_surface_analytic"]:.8f}')

    ax.set_xlabel('Resolution N', fontsize=12)
    ax.set_ylabel(r'$\alpha$ at surface', fontsize=12)
    ax.set_title('Surface alpha Values (o=num, x=analytical)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 6: Convergence order summary
    ax = axes[1, 2]
    quantities = ['alpha_disc', 'exp4phi_disc', 'r_iso_norm']
    labels = ['alpha disc', 'ψ disc', 'norm']
    final_orders = []
    for q in quantities:
        if orders[q] and len(orders[q]) > 0 and not np.isnan(orders[q][-1]):
            final_orders.append(orders[q][-1])
        else:
            final_orders.append(0)

    x_pos = np.arange(len(labels))
    ax.bar(x_pos, final_orders, color='steelblue', alpha=0.7)
    ax.axhline(y=4, color='green', linestyle='--', linewidth=2, label='4th order')
    ax.axhline(y=2, color='orange', linestyle='--', linewidth=2, label='2nd order')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Convergence Order', fontsize=12)
    ax.set_title('Surface Matching Convergence Orders', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(6, max(final_orders) * 1.2) if final_orders else 6)

    plt.suptitle(f'Exterior & Surface Matching Convergence (K={K}, Γ={Gamma}, ρ_c={rho_central:.2e})',
                 fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nConvergence plot saved to: {save_path}")

    results = {
        'resolutions': resolutions,
        'errors': errors,
        'orders': orders,
        'solutions': solutions,
        'disc_diffs': disc_diffs,
    }

    return results, fig


def run_convergence_test(K, Gamma, rho_central, resolutions=None, save_path=None, verbose=True,
                         interior_fraction=0.8):
    """
    Run a full convergence test for the TOV solver using self-convergence.

    Uses Richardson extrapolation approach: compares consecutive resolutions
    at COINCIDENT grid points (no interpolation, which would degrade convergence).

    IMPORTANT: Uses Taylor expansion near origin to avoid degrading convergence
    from the removable singularity at r=0.

    Args:
        K: Polytropic constant
        Gamma: Polytropic exponent
        rho_central: Central baryon density
        resolutions: List of resolutions (should have ratio 2 between consecutive, e.g., [100, 200, 400])
        save_path: If provided, save convergence plot to this path
        verbose: Print detailed output
        interior_fraction: Fraction of stellar radius to integrate to (default: 0.8)

    Returns:
        Dictionary containing convergence results and matplotlib Figure
    """
    if resolutions is None:
        resolutions = [100, 200, 400]
    
    resolutions = sorted(resolutions)
    
    # Verify resolutions have ratio 2
    for i in range(len(resolutions)-1):
        ratio = resolutions[i+1] / resolutions[i]
        if abs(ratio - 2.0) > 0.01:
            print(f"Warning: Resolution ratio {resolutions[i]}→{resolutions[i+1]} is {ratio:.2f}, not 2.0")
    
    print("="*70)
    print("TOV SOLVER CONVERGENCE TEST (Self-Convergence at Coincident Points)")
    print("="*70)
    print(f"EOS: P = {K} × ρ^{Gamma}")
    print(f"Central density: ρ_c = {rho_central:.6e}")
    print(f"Resolutions: {resolutions}")
    print(f"Interior fraction: {interior_fraction} (to avoid surface effects)")
    print("-"*70)
    
    # Find stellar radius
    print(f"\nFinding stellar radius...")
    solver_temp = TOVSolverIso(K=K, Gamma=Gamma)
    sol_temp = solver_temp.solve(rho_central, accuracy="low", verbose=False)
    R_star = sol_temp.R_schw
    M_star_ref = sol_temp.M_star
    print(f"  Stellar radius R_schw = {R_star:.10f}")
    print(f"  Stellar mass M = {M_star_ref:.10f}")
    
    r_interior = interior_fraction * R_star
    print(f"  Test domain: r ∈ [r_taylor, {r_interior:.6f}] ({interior_fraction*100:.0f}% of R_star)")
    
    # Compute solutions
    solutions = {}
    print("\nComputing fixed-resolution solutions (Taylor + RK4)...")
    for n in resolutions:
        print(f"  N = {n}...", end=" ")
        sol = solve_tov_fixed_resolution(K, Gamma, rho_central, n, r_max_schw=r_interior)
        solutions[n] = sol
        print(f"M(r_int) = {sol.M_star:.10f}, P(r_int) = {sol.P[-1]:.6e}")
    
    # Self-convergence at coincident points
    quantities = ['P', 'M', 'rho_baryon', 'r_iso']
    errors = {q: [] for q in quantities}
    errors['M_endpoint'] = []
    errors['P_endpoint'] = []
    
    # Compare consecutive resolutions at coincident points
    for i in range(len(resolutions) - 1):
        n1, n2 = resolutions[i], resolutions[i+1]
        sol1, sol2 = solutions[n1], solutions[n2]
        ratio = n2 // n1  # Should be 2
        
        # Endpoint differences
        errors['M_endpoint'].append(abs(sol1.M_star - sol2.M_star))
        errors['P_endpoint'].append(abs(sol1.P[-1] - sol2.P[-1]))
        
        # Field differences at coincident points (L∞ norm - max difference)
        for q in quantities:
            arr1 = getattr(sol1, q)
            arr2 = getattr(sol2, q)
            
            # Compare at coincident points
            max_diff = 0.0
            for j in range(len(arr1)):
                if j * ratio < len(arr2):
                    diff = abs(arr1[j] - arr2[j * ratio])
                    max_diff = max(max_diff, diff)
            
            # Normalize by max value
            scale = max(np.max(np.abs(arr2)), 1e-30)
            errors[q].append(max_diff / scale)
    
    # Compute orders
    def compute_self_conv_order(diffs, res_list):
        orders = []
        for i in range(len(diffs) - 1):
            if diffs[i] > 1e-15 and diffs[i+1] > 1e-15:
                r = res_list[i+2] / res_list[i+1]
                diff_ratio = diffs[i] / diffs[i+1]
                if diff_ratio > 1:
                    order = np.log(diff_ratio) / np.log(r)
                    orders.append(order)
                else:
                    orders.append(0.0)
            else:
                orders.append(np.nan)
        return orders
    
    orders = {}
    all_quantities = ['M_endpoint', 'P_endpoint'] + quantities
    
    print("\n" + "-"*70)
    print("SELF-CONVERGENCE RESULTS (at coincident grid points)")
    print("-"*70)
    print(f"Comparing consecutive resolutions (ratio 2):")
    for i in range(len(resolutions)-1):
        print(f"  N={resolutions[i]} vs N={resolutions[i+1]}")
    print("-"*70)
    print(f"{'Quantity':<15} | {'|u_N - u_2N| differences':<35} | {'Order':<15}")
    print("-"*70)
    
    for q in all_quantities:
        diff_list = errors[q]
        if len(diff_list) >= 2:
            ord_list = compute_self_conv_order(diff_list, resolutions)
        else:
            ord_list = []
        orders[q] = ord_list
        
        diff_str = ", ".join([f"{d:.2e}" for d in diff_list])
        ord_str = ", ".join([f"{o:.2f}" if not np.isnan(o) else "N/A" for o in ord_list]) if ord_list else "N/A"
        
        print(f"{q:<15} | {diff_str:<35} | {ord_str:<15}")
    
    print("-"*70)
    
    # Compute average order
    all_orders = []
    for q in all_quantities:
        all_orders.extend([o for o in orders.get(q, []) if not np.isnan(o) and o > 0.5])
    
    avg_order = None
    if all_orders:
        avg_order = np.mean(all_orders)
        std_order = np.std(all_orders)
        print(f"\nAverage convergence order: {avg_order:.2f} ± {std_order:.2f}")
        print(f"Expected for RK4: 4.0")
    
    # Create convergence plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    x_pairs = [(resolutions[i] + resolutions[i+1])/2 for i in range(len(resolutions)-1)]
    
    # Plot 1: Endpoint quantities
    ax = axes[0, 0]
    ax.loglog(x_pairs, errors['M_endpoint'], 'o-', linewidth=2, markersize=10, label='|ΔM|')
    ax.loglog(x_pairs, errors['P_endpoint'], 's-', linewidth=2, markersize=10, label='|ΔP|')
    
    x_ref = np.array([x_pairs[0], x_pairs[-1]])
    for order, ls, color in [(2, '--', 'gray'), (4, ':', 'black')]:
        scale = errors['M_endpoint'][0] * (x_pairs[0] / x_ref[0])**order
        ax.loglog(x_ref, scale * (x_ref[0] / x_ref)**order, ls, color=color, alpha=0.7, label=f'O(h^{order})')
    
    ax.set_xlabel('Resolution (N)', fontsize=12)
    ax.set_ylabel('|Difference|', fontsize=12)
    ax.set_title('Endpoint Self-Convergence', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Field quantities
    ax = axes[0, 1]
    markers = ['o', 's', '^', 'd']
    for q, m in zip(quantities, markers):
        ax.loglog(x_pairs, errors[q], f'{m}-', linewidth=2, markersize=8, label=q)
    
    for order, ls, color in [(2, '--', 'gray'), (4, ':', 'black')]:
        scale = errors['P'][0] * (x_pairs[0] / x_ref[0])**order
        ax.loglog(x_ref, scale * (x_ref[0] / x_ref)**order, ls, color=color, alpha=0.7, label=f'O(h^{order})')
    
    ax.set_xlabel('Resolution (N)', fontsize=12)
    ax.set_ylabel('Max |Difference| (relative)', fontsize=12)
    ax.set_title('Field Self-Convergence (L∞ norm)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Solution profiles
    ax = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(resolutions)))
    for i, n in enumerate(resolutions):
        sol = solutions[n]
        ax.plot(sol.r_schw, sol.M, '-', color=colors[i], linewidth=1.5+i*0.5, label=f'N={n}')
    ax.axvline(r_interior, color='red', linestyle=':', alpha=0.7, label=f'r_test')
    ax.set_xlabel(r'$r_{Schw}$', fontsize=12)
    ax.set_ylabel('M(r)', fontsize=12)
    ax.set_title('Enclosed Mass Profiles', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Convergence order bar chart
    ax = axes[1, 1]
    final_orders = []
    labels = []
    for q in all_quantities:
        if orders[q]:
            o = orders[q][-1] if not np.isnan(orders[q][-1]) else 0
            final_orders.append(o)
            labels.append(q)
    
    if final_orders:
        x_pos = np.arange(len(final_orders))
        ax.bar(x_pos, final_orders, color='steelblue', alpha=0.7)
        ax.axhline(y=4, color='green', linestyle='--', linewidth=2, label='4th order (RK4)')
        ax.axhline(y=2, color='orange', linestyle='--', linewidth=2, label='2nd order')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Convergence Order', fontsize=12)
        ax.set_title('Measured Self-Convergence Order', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(6, max(final_orders) * 1.2))
    
    plt.suptitle(f'TOV Self-Convergence (K={K}, Γ={Gamma}, ρ_c={rho_central:.2e})\n'
                 f'Taylor+RK4, coincident points, interior r ≤ {interior_fraction*100:.0f}% R_star', 
                 fontsize=13, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nConvergence plot saved to: {save_path}")
    
    results = {
        'resolutions': resolutions,
        'errors': errors,
        'orders': orders,
        'solutions': solutions,
        'r_interior': r_interior,
        'R_star': R_star,
        'average_order': avg_order
    }
    
    return results, fig

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Solve TOV equations in isotropic coordinates (Improved)')
    parser.add_argument('--K', type=float, default=100.0, help='Polytropic constant K')
    parser.add_argument('--Gamma', type=float, default=2.0, help='Polytropic index Γ')
    parser.add_argument('--rho_central', type=float, default=1.28e-3, help='Central density')
    parser.add_argument('--r_max', type=float, default=20.0, help='Maximum radius (isotropic)')
    parser.add_argument('--accuracy', type=str, default='high',
                        choices=['verylow', 'low', 'medium', 'high', 'veryhigh'],
                        help='Integration accuracy preset')
    parser.add_argument('--output', type=str, default='tov_solution_iso.png', help='Output plot filename')
    parser.add_argument('--convergence', action='store_true', help='Run accuracy preset comparison')
    parser.add_argument('--comparison', action='store_true', help='Plot interior/exterior comparison')
    parser.add_argument('--convergence-test', action='store_true', dest='convergence_test',
                        help='Run full convergence test with fixed resolutions (100, 200, 400)')
    parser.add_argument('--exterior-test', action='store_true', dest='exterior_test',
                        help='Run exterior & surface matching convergence test')
    parser.add_argument('--resolutions', type=str, default='100,200,400',
                        help='Comma-separated list of resolutions for convergence test')

    args = parser.parse_args()

    print("="*70)
    print("TOV Solver in Isotropic Coordinates")
    print("="*70)

    # Check if we can show plots interactively+
    can_show = 'DISPLAY' in os.environ and os.environ.get('DISPLAY')

    # Run convergence test if requested
    if args.convergence_test:
        resolutions = [int(x.strip()) for x in args.resolutions.split(',')]
        convergence_path = args.output.replace('.png', '_convergence_test.png')
        results, fig_conv = run_convergence_test(
            args.K, args.Gamma, args.rho_central,
            resolutions=resolutions,
            save_path=convergence_path
        )
        if can_show:
            plt.show()
        else:
            plt.close('all')

    # Run exterior convergence test if requested
    elif args.exterior_test:
        resolutions = [int(x.strip()) for x in args.resolutions.split(',')]
        exterior_path = args.output.replace('.png', '_exterior_convergence.png')
        results, fig_ext = run_exterior_convergence_test(args.K, args.Gamma, args.rho_central,resolutions=resolutions,save_path=exterior_path)
        if can_show:
            plt.show()
        else:
            plt.close('all')

    else:
        # Standard solve mode
        # Create solver with decoupled EOS
        eos = PolytropicEOS(K=args.K, Gamma=args.Gamma)
        solver = TOVSolverIso(eos=eos)

        # Solve
        print(f"\nSolving with: {eos}")
        print(f"Central density: ρ_c = {args.rho_central:.6e}")
        print(f"Accuracy: {args.accuracy}")
        print("-"*70)
        
        solution = solver.solve(args.rho_central, r_max_iso=args.r_max, accuracy=args.accuracy)

        # Print detailed summary
        print("\n" + "="*70)
        print("SOLUTION SUMMARY")
        print("="*70)
        print(f"  EOS            : P = {args.K} × ρ^{args.Gamma}")
        print(f"  ρ_central      : {args.rho_central:.6e}")
        print(f"  M_star         : {solution.M_star:.8f}")
        print(f"  R_schw         : {solution.R_schw:.8f}")
        print(f"  R_iso          : {solution.R_iso:.8f}")
        print(f"  Compactness    : {solution.C:.8f}")
        print(f"  Grid points    : {solution.num_points}")
        print(f"  alpha continuity   : {solution.alpha_continuity_error:.2e}")
        print(f"  Converged      : {solution.integration_converged}")
        print("="*70)

        # Main diagnostic plot
        fig = plot_tov_iso_diagnostics(solution, args.r_max, save_path=args.output)
        plt.close(fig)
        # Optional comparison plot
        if args.comparison:
            comparison_path = args.output.replace('.png', '_comparison.png')
            fig_comp = plot_tov_iso_comparison(solution, args.r_max, save_path=comparison_path)
        
        # Optional accuracy preset comparison
        if args.convergence:
            print("\nRunning accuracy preset comparison...")
            convergence_path = args.output.replace('.png', '_convergence.png')
            fig_conv, conv_solutions = plot_tov_iso_convergence(
                args.K, args.Gamma, args.rho_central, 
                r_max=args.r_max, save_path=convergence_path
            )
            
            print("\nAccuracy Preset Comparison:")
            print("-"*50)
            for acc, sol in conv_solutions.items():
                print(f"  {acc:10s}: M={sol.M_star:.8f}, R_iso={sol.R_iso:.8f}")

        # Compare with NRPy+ values if using their parameters
        if abs(args.K - 1.0) < 0.01 and abs(args.Gamma - 2.0) < 0.01 and abs(args.rho_central - 0.129285) < 0.001:
            print("\n" + "="*70)
            print("Comparison with NRPy+ reference values (K=1, Gamma=2, rho_c=0.129285):")
            print("="*70)
            print(f"  NRPy+ M      = 0.1405   | This solver: {solution.M_star:.4f}")
            print(f"  NRPy+ R_schw = 0.9566   | This solver: {solution.R_schw:.4f}")
            print(f"  NRPy+ R_iso  = 0.8100   | This solver: {solution.R_iso:.4f}")
            print("="*70)
