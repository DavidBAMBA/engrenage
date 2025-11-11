"""
Conservative to primitive variable conversion for relativistic hydrodynamics.

Vectorized solver using Newton-Raphson with analytic derivatives for ideal gas EOS.
Follows Valencia formulation with densitized conservatives.
"""

import numpy as np
from .atmosphere import AtmosphereParams, FloorApplicator


class Cons2PrimSolver:
    """
    Conservative to primitive variable converter with vectorized Newton-Raphson.
    
    Primitive Variable Recovery (following C implementation):
    --------------------------------------------------------
    Given conserved (D, Sr, τ) and metric γrr, solve for auxiliary variable y,
    then compute primitives:
        W  = √(y²/(y² - Sr²/γrr))     [Lorentz factor]
        h  = y/W²                      [Specific enthalpy]
        P  = y - τ                     [Pressure]
        ρ₀ = D/W                       [Rest mass density]
        ε  = P/(ρ₀(Γ-1))              [Internal energy]
        vr = Sr/y                      [Radial velocity]
    
    Args:
        eos: Equation of state with eps_from_rho_p() and enthalpy() methods
        atmosphere: AtmosphereParams for floor values (optional)
        **params: Solver parameters (tol, max_iter, etc.)
    """
    
    def __init__(self, eos, atmosphere=None, **params):
        self.eos = eos
        
        # Atmosphere/floor management
        if atmosphere is None:
            atmosphere = AtmosphereParams()
        elif isinstance(atmosphere, dict):
            atmosphere = AtmosphereParams(**atmosphere)
        self.atmosphere = atmosphere
        self.floor_applicator = FloorApplicator(self.atmosphere, eos)
        
        # Solver parameters
        self.params = {
            "rho_floor": 1e-13,
            "p_floor": 1e-15,
            "v_max": 0.999999,
            "W_max": 1.0e3,
            "tol": 1e-16,
            "max_iter": 1000,
        }
        self.params.update(params)
        self.params.update(self.atmosphere.to_cons2prim_params())
        
        # Pre-cache EOS constants for ideal gas
        if hasattr(eos, 'gamma'):
            self.gamma = float(eos.gamma)
            self.c_ideal = self.gamma / (self.gamma - 1.0)
        else:
            self.gamma = None
            self.c_ideal = None
    
    def convert(self, U, gamma_rr=None, alpha=None, p_guess=None,
                apply_conservative_floors=True, e6phi=None):
        """
        Convert conservative to primitive variables (vectorized).

        Args:
            U: Conservative variables as tuple/list (D, Sr, tau) - numpy arrays
            gamma_rr: Radial metric component (numpy array, default: 1.0)
            alpha: Lapse function (numpy array, default: 1.0)
            p_guess: Pressure guess from previous timestep (optional)
            apply_conservative_floors: Apply floors before solving (default: True)
            e6phi: Densitization factor e^{6φ} (optional)
                   If provided: de-densitize input, solve, re-densitize output

        Returns:
            dict: {'rho0', 'vr', 'p', 'eps', 'W', 'h', 'success'}
                  If e6phi provided, also: {'D_tilde', 'Sr_tilde', 'tau_tilde'}
        """
        # Extract conservative variables (assume tuple/list of numpy arrays)
        D_input, Sr_input, tau_input = U
        D_input = np.atleast_1d(np.asarray(D_input, dtype=float))
        Sr_input = np.atleast_1d(np.asarray(Sr_input, dtype=float))
        tau_input = np.atleast_1d(np.asarray(tau_input, dtype=float))
        N = len(D_input)

        # Default metric components
        if gamma_rr is None:
            gamma_rr = np.ones(N)
        else:
            gamma_rr = np.broadcast_to(np.asarray(gamma_rr, dtype=float), N)

        if alpha is None:
            alpha = np.ones(N)
        else:
            alpha = np.broadcast_to(np.asarray(alpha, dtype=float), N)
        
        # De-densitize if needed
        if e6phi is not None:
            from source.matter.hydro.grhd_equations import dedensitize_conservatives
            e6phi_arr = np.asarray(e6phi)
            D, Sr, tau = dedensitize_conservatives(D_input, Sr_input, tau_input, e6phi_arr)
        else:
            D, Sr, tau = D_input, Sr_input, tau_input
        
        # Apply conservative floors (prevents many failures)
        if apply_conservative_floors:
            D, Sr, tau, _ = self.floor_applicator.apply_conservative_floors(
                D, Sr, tau, gamma_rr
            )
        
        # Solve for primitives
        rho0, vr, p, eps, W, h, success = self._solve_all_points(
            D, Sr, tau, gamma_rr, p_guess
        )
        
        # Handle failures with atmosphere
        if np.any(~success):
            # Use atmosphere parameters directly
            rho0[~success] = self.atmosphere.rho_floor
            vr[~success] = 0.0
            p[~success] = self.atmosphere.p_floor
            eps[~success] = self.eos.eps_from_rho_p(self.atmosphere.rho_floor,
                                                     self.atmosphere.p_floor) if hasattr(self.eos, 'eps_from_rho_p') else 1e-10
            W[~success] = 1.0
            h[~success] = 1.0 + eps[~success] + p[~success] / rho0[~success]

        # Apply primitive floors (velocity limit, pressure floor)
        rho0, vr, p = self.floor_applicator.apply_primitive_floors(rho0, vr, p, gamma_rr)

        # Recompute eps and h after floor application
        if self.gamma is not None:
            eps = p / (rho0 * (self.gamma - 1.0))
            h = 1.0 + eps + p / np.maximum(rho0, 1e-30)
        else:
            eps = self.eos.eps_from_rho_p(rho0, p)
            h = self.eos.enthalpy(rho0, p, eps)
        
        # Build result
        result = {
            "rho0": rho0, "vr": vr, "p": p, "eps": eps,
            "W": W, "h": h, "success": success
        }
        
        # Re-densitize if requested
        if e6phi is not None:
            result.update(
                self._redensitize_conservatives(rho0, vr, p, eps, W, h, 
                                               alpha, e6phi_arr, gamma_rr)
            )
        
        return result
    
    def _solve_all_points(self, D, Sr, tau, gamma_rr, p_guess):
        """Vectorized Newton-Raphson solver for all points."""
        N = len(D)
        
        # Initialize
        p = np.maximum(self.params["p_floor"], 
                      p_guess if p_guess is not None else 0.1 * (tau + D))
        converged = np.zeros(N, dtype=bool)
        active = np.ones(N, dtype=bool)
        
        # Output arrays
        rho0 = np.zeros(N)
        vr = np.zeros(N)
        eps = np.zeros(N)
        W = np.ones(N)
        h = np.ones(N)
        
        # Newton-Raphson iterations
        for _ in range(self.params["max_iter"]):
            if not np.any(active):
                break
            
            # Evaluate state at current pressure
            valid, (rho0_i, vr_i, eps_i, W_i, h_i, f_i) = self._evaluate_state(
                D[active], Sr[active], tau[active], p[active], gamma_rr[active]
            )
            
            # Handle evaluation failures
            if not np.all(valid):
                idx_failed = np.where(active)[0][~valid]
                p[idx_failed] = np.maximum(self.params["p_floor"], p[idx_failed] * 1.5)
                continue
            
            # Check convergence
            tol = self.params["tol"] * np.maximum(1.0, np.abs(p[active]))
            just_converged = np.abs(f_i) <= tol

            if np.any(just_converged):
                idx_conv = np.where(active)[0][just_converged]
                converged[idx_conv] = True
                rho0[idx_conv] = rho0_i[just_converged]
                vr[idx_conv] = vr_i[just_converged]
                eps[idx_conv] = eps_i[just_converged]
                W[idx_conv] = W_i[just_converged]
                h[idx_conv] = h_i[just_converged]
                active[idx_conv] = False

                # Filter W_i and h_i to remove converged points
                W_i = W_i[~just_converged]
                h_i = h_i[~just_converged]
                f_i = f_i[~just_converged]

            if not np.any(active):
                break

            # Compute derivative and update
            df = self._compute_derivative(D[active], Sr[active], tau[active],
                                         p[active], gamma_rr[active], W_i, h_i)
            
            # Check derivative validity
            bad_df = ~np.isfinite(df) | (np.abs(df) < 1e-15)
            if np.any(bad_df):
                idx_bad = np.where(active)[0][bad_df]
                p[idx_bad] = np.maximum(self.params["p_floor"], p[idx_bad] * 1.5)
                continue
            
            # Newton update with damping
            p_new = p[active] - f_i / df
            invalid = ~np.isfinite(p_new) | (p_new <= 0)
            p_new[invalid] = np.maximum(self.params["p_floor"], 0.5 * p[active][invalid])
            p[active] = 0.5 * p[active] + 0.5 * p_new  # Damped step
        
        return rho0, vr, p, eps, W, h, converged
    
    def _evaluate_state(self, D, Sr, tau, p, gamma_rr):
        """
        Evaluate primitive state for given pressure (vectorized).
        
        Returns: (valid_mask, (rho0, vr, eps, W, h, f))
        
        Following C code logic:
            E = τ + D
            Q = E + p
            vr = Sr/(Q·γrr)
            v² = γrr·vr²
            W = 1/√(1-v²)
            ρ₀ = D/W
            ε = p/(ρ₀(Γ-1))
            h = 1 + ε + p/ρ₀
            f = ρ₀·h·W² - Q    [residual]
        """
        p = np.maximum(p, self.params["p_floor"])
        
        # Basic quantities
        E = tau + D
        Q = E + p
        g = np.maximum(gamma_rr, 1e-30)
        
        # Velocity and Lorentz factor
        vr = Sr / (Q * g)
        v2 = g * vr * vr
        v2_safe = np.clip(v2, 0.0, 1.0 - 1e-16)
        W = 1.0 / np.sqrt(1.0 - v2_safe)
        
        # Density
        rho0 = D / np.maximum(W, 1e-30)
        
        # Thermodynamics
        if self.gamma is not None:
            # Fast path for ideal gas
            eps = p / (rho0 * (self.gamma - 1.0))
            h = 1.0 + eps + p / np.maximum(rho0, 1e-30)
        else:
            # General EOS
            try:
                eps = self.eos.eps_from_rho_p(rho0, p)
                h = self.eos.enthalpy(rho0, p, eps)
            except:
                eps = np.full_like(rho0, np.inf)
                h = np.ones_like(rho0)
        
        # Residual
        f = rho0 * h * W * W - Q
        
        # Validity check
        valid = (
            (Q > 0.0) & (v2 >= 0.0) & (v2 < 1.0) &
            (W >= 1.0) & (W <= self.params["W_max"]) &
            (rho0 > 0.0) & np.isfinite(rho0) &
            (eps >= 0.0) & np.isfinite(eps) &
            (h > 1.0) & np.isfinite(h) & np.isfinite(f)
        )
        
        # Zero out invalid
        if not np.all(valid):
            rho0[~valid] = 0.0
            vr[~valid] = 0.0
            eps[~valid] = 0.0
            W[~valid] = 1.0
            h[~valid] = 1.0
            f[~valid] = np.inf
        
        return valid, (rho0, vr, eps, W, h, f)
    
    def _compute_derivative(self, D, Sr, tau, p, gamma_rr, W, h):
        """
        Compute df/dp (vectorized).
        Uses analytic formula for ideal gas, numerical otherwise.
        """
        if self.c_ideal is not None:
            # Analytic derivative for ideal gas (fast)
            E = tau + D
            Q = E + p
            g = np.maximum(gamma_rr, 1e-30)
            
            # dW/dp
            Sr_sq = Sr * Sr
            Q_cubed = Q * Q * Q
            W_cubed = W * W * W
            Wprime = -Sr_sq * W_cubed / np.maximum(Q_cubed * g, 1e-30)
            
            # df/dp
            W_sq = W * W
            df = self.c_ideal * W_sq + (D + 2.0 * self.c_ideal * p * W) * Wprime - 1.0
            
        else:
            # Numerical derivative (fallback)
            dp = np.maximum(1e-3 * np.maximum(np.abs(p), 1.0), 1e-12)
            _, (_, _, _, _, _, f1) = self._evaluate_state(D, Sr, tau, p, gamma_rr)
            _, (_, _, _, _, _, f2) = self._evaluate_state(D, Sr, tau, p + dp, gamma_rr)
            df = (f2 - f1) / dp
            df = np.where(np.isfinite(df), df, 0.0)
        
        return df
    
    def _redensitize_conservatives(self, rho0, vr, p, eps, W, h,
                                   alpha, e6phi, gamma_rr):
        """Re-compute densitized conservatives from primitives."""
        from source.matter.hydro.grhd_equations import compute_densitized_conservatives_from_primitives

        N = len(rho0)

        # Build 3D velocity (only radial component for 1D)
        v_U = np.zeros((N, 3))
        v_U[:, 0] = vr

        # Build 3D metric
        gamma_LL = np.zeros((N, 3, 3))
        gamma_LL[:, 0, 0] = gamma_rr
        gamma_LL[:, 1, 1] = gamma_rr
        gamma_LL[:, 2, 2] = gamma_rr

        if not hasattr(alpha, '__len__'):
            alpha = np.ones(N)

        # Compute densitized conservatives
        D_tilde, S_tildeD, tau_tilde = compute_densitized_conservatives_from_primitives(
            rho0, v_U, W, h, p, alpha, e6phi, gamma_LL
        )

        return {
            'D_tilde': D_tilde,
            'Sr_tilde': S_tildeD[:, 0],  # Extract radial component
            'tau_tilde': tau_tilde
        }


# ============================================================================
# PRIMITIVE TO CONSERVATIVE
# ============================================================================

def prim_to_cons(rho0, vr, pressure, gamma_rr, eos, e6phi=None, alpha=None):
    """
    Convert primitive → conservative variables.

    Following Valencia formulation:
        Non-densitized (e6phi=None):
            D  = W ρ₀
            Sr = W² ρ₀ h vr γrr
            τ  = α² T^{00} - W ρ₀

        Densitized (e6phi provided):
            D̃  = e^{6φ} W ρ₀
            S̃r = e^{6φ} W² ρ₀ h vr γrr
            τ̃  = e^{6φ} (α² T^{00} - W ρ₀)

    Args:
        rho0, vr, pressure: Primitive variables
        gamma_rr: Radial metric component
        eos: Equation of state
        e6phi: Densitization factor e^{6φ} (optional, default: None = non-densitized)
        alpha: Lapse (default: 1.0)

    Returns:
        If e6phi=None: (D, Sr, tau): Non-densitized conservatives
        If e6phi provided: (D_tilde, Sr_tilde, tau_tilde): Densitized conservatives
    """
    from source.matter.hydro.grhd_equations import compute_densitized_conservatives_from_primitives

    # Convert to arrays and broadcast
    rho0 = np.asarray(rho0)
    vr = np.asarray(vr)
    pressure = np.asarray(pressure)
    gamma_rr = np.asarray(gamma_rr)

    # Default to non-densitized (e6phi = 1.0)
    if e6phi is None:
        e6phi = np.ones_like(rho0)
    else:
        e6phi = np.asarray(e6phi)

    rho0, vr, pressure, gamma_rr, e6phi = np.broadcast_arrays(
        rho0, vr, pressure, gamma_rr, e6phi
    )
    
    # Lapse (default to 1.0 for Minkowski)
    if alpha is None:
        alpha = np.ones_like(rho0)
    else:
        alpha = np.broadcast_to(np.asarray(alpha), rho0.shape)
    
    # Lorentz factor
    v2 = gamma_rr * vr * vr
    v2 = np.clip(v2, 0.0, 1.0 - 1e-12)
    W = 1.0 / np.sqrt(1.0 - v2)
    
    # Thermodynamics
    h = eos.enthalpy_from_rho_p(rho0, pressure)
    
    # Build 3D arrays (for interface with auxiliary function)
    is_scalar = (rho0.ndim == 0)
    if is_scalar:
        v_U = np.zeros(3)
        v_U[0] = vr
        gamma_LL = np.diag([gamma_rr, gamma_rr, gamma_rr])
    else:
        N = len(rho0)
        v_U = np.zeros((N, 3))
        v_U[:, 0] = vr
        gamma_LL = np.zeros((N, 3, 3))
        gamma_LL[:, 0, 0] = gamma_rr
        gamma_LL[:, 1, 1] = gamma_rr
        gamma_LL[:, 2, 2] = gamma_rr
    
    # Compute densitized conservatives using auxiliary function
    D_tilde, S_tildeD, tau_tilde = compute_densitized_conservatives_from_primitives(
        rho0, v_U, W, h, pressure, alpha, e6phi, gamma_LL
    )
    
    # Extract radial component of momentum
    if is_scalar:
        Sr_tilde = S_tildeD[0]
    else:
        Sr_tilde = S_tildeD[:, 0]
    
    return D_tilde, Sr_tilde, tau_tilde