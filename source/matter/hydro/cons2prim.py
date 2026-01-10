# cons2prim.py

import numpy as np
from .atmosphere import FloorApplicator


# ============================================================================
# MAIN SOLVER CLASS
# ============================================================================

class Cons2PrimSolver:
    """
    Conservative to primitive variable converter.

    Simplified API that expects arrays directly. Uses AtmosphereParams
    and FloorApplicator from atmosphere.py for floor management.

    Args:
        eos: Equation of state object with eps_from_rho_p(rho0, p) method
        atmosphere: AtmosphereParams object (required)
        tol: Newton-Raphson tolerance (default: 1e-12)
        max_iter: Maximum iterations (default: 500)
    """

    def __init__(self, eos, atmosphere, tol=1e-12, max_iter=500):
        self.eos = eos
        self.atmosphere = atmosphere

        # Floor applicator (handles all floor logic)
        self.floor_applicator = FloorApplicator(self.atmosphere, eos)

        # Solver parameters from atmosphere
        self.rho_floor = atmosphere.rho_floor
        self.p_floor = atmosphere.p_floor
        self.v_max = atmosphere.v_max
        self.W_max = atmosphere.W_max
        self.tol = tol
        self.max_iter = max_iter

        # Check if we can use optimized path for ideal gas
        self._is_ideal_gas = (hasattr(eos, 'name') and
                              eos.name.startswith('ideal_gas') and
                              hasattr(eos, 'gamma'))

        if self._is_ideal_gas:
            self._eos_gamma = float(eos.gamma)

        # Statistics tracking
        self.stats = {
            "total_calls": 0,
            "successful_conversions": 0,
            "newton_successes": 0,
            "atmosphere_fallbacks": 0,
            "conservative_floors_applied": 0
        }

        # =====================================================================
        # SOLVER METHOD SELECTION (hardcoded)
        # Options: "newton"
        #   - "newton": Newton-Raphson on pressure (fast, needs good initial guess)
 # =====================================================================
        self._solver_method = "newton"  

    def convert(self, D, Sr, tau, gamma_rr, p_guess=None, apply_conservative_floors=True,
                e6phi=None, alpha=None, beta_r=None):
        """
        Convert conservative to primitive variables using vectorized Newton-Raphson.

        IMPORTANT: Expects PHYSICAL (non-densitized) conservative variables:
            D  = ρ₀ W
            Sʳ = ρ₀ h W² vʳ γᵣᵣ / alpha
            τ  = ρ₀ h - P - D

        Returns:
            tuple: (rho0, vr, p, eps, W, h, success, D_out, Sr_out, tau_out)
                   Primitive variables, success mask, and updated conservatives.
                   If e6phi not provided, D_out/Sr_out/tau_out are physical (non-densitized).
                   If e6phi provided, they are densitized.
        """
        self.stats["total_calls"] += 1

        # Ensure arrays
        D = np.atleast_1d(np.asarray(D, dtype=float))
        Sr = np.atleast_1d(np.asarray(Sr, dtype=float))
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        gamma_rr = np.atleast_1d(np.asarray(gamma_rr, dtype=float))
        N = len(D)

        # Prepare densitization factors (default to 1.0 for non-densitized output)
        if e6phi is None:
            e6phi = np.ones(N)
        else:
            e6phi = np.atleast_1d(np.asarray(e6phi, dtype=float))
        if alpha is None:
            #print("WARNING [cons2prim.convert]: alpha=None, using default alpha=1 (Minkowski)")
            alpha = np.ones(N)
        else:
            alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
        if beta_r is None:
            beta_r = np.zeros(N)
        else:
            beta_r = np.atleast_1d(np.asarray(beta_r, dtype=float))

        # Output conservative arrays (will be updated for atmosphere points)
        D_out = D.copy()
        Sr_out = Sr.copy()
        tau_out = tau.copy()

        # Apply conservative variable floors (IllinoisGRMHD strategy)
        if apply_conservative_floors:
            D, Sr, tau, floor_mask = self.floor_applicator.apply_conservative_floors(
                D, Sr, tau, gamma_rr
            )
            if np.any(floor_mask):
                self.stats["conservative_floors_applied"] += np.sum(floor_mask)

        # Allocate output arrays
        rho0, vr, p, eps, W, h = (np.zeros(N) for _ in range(6))
        success = np.zeros(N, dtype=bool)

        # ATMOSPHERE DETECTION: Low density cells get atmosphere values directly
        
        # D = ρ₀W, so D is an upper bound for ρ₀. Using 100× ensures we catch
        # all cells near the stellar surface where oscillations can occur.
        atm_mask = D < 10.0 * self.rho_floor

        if np.any(atm_mask):
            rho0[atm_mask] = self.rho_floor
            vr[atm_mask] = 0.0
            p[atm_mask] = self.p_floor
            try:
                eps[atm_mask] = self.eos.eps_from_rho_p(self.rho_floor, self.p_floor)
            except:
                eps[atm_mask] = 1e-10
            W[atm_mask] = 1.0
            h[atm_mask] = 1.0 + eps[atm_mask] + self.p_floor / self.rho_floor
            success[atm_mask] = True
            self.stats["atmosphere_fallbacks"] += np.sum(atm_mask)

        # Solve non-atmosphere points
        solve_mask = ~atm_mask

        if np.any(solve_mask):
            # Select solver method based on hardcoded setting
            result = self._solve_vectorized_points(
                    D[solve_mask], Sr[solve_mask], tau[solve_mask], gamma_rr[solve_mask],
                    p_guess[solve_mask] if p_guess is not None else None, alpha[solve_mask]
                )

            rho0_solved, vr_solved, p_solved, eps_solved, W_solved, h_solved, success_solved = result

            # Place results back
            rho0[solve_mask] = rho0_solved
            vr[solve_mask] = vr_solved
            p[solve_mask] = p_solved
            eps[solve_mask] = eps_solved
            W[solve_mask] = W_solved
            h[solve_mask] = h_solved
            success[solve_mask] = success_solved

            # Update statistics
            self.stats["successful_conversions"] += np.sum(success_solved)
            self.stats["atmosphere_fallbacks"] += np.sum(~success_solved)

        # Handle invalid and failed points with atmosphere (using FloorApplicator)
        failed_mask = ~success
        if np.any(failed_mask):
            self.floor_applicator.apply_atmosphere_fallback(
                rho0, vr, p, eps, W, h, failed_mask
            )

        # Enforce velocity limits (using FloorApplicator)
        rho0, vr, p = self.floor_applicator.apply_primitive_floors(rho0, vr, p, gamma_rr)

        # Final pressure floor with EOS-consistent recompute
        low_p_mask = p < self.p_floor
        if np.any(low_p_mask):
            p[low_p_mask] = self.p_floor
            rho_lp = np.maximum(rho0[low_p_mask], 1e-30)
            if self._is_ideal_gas:
                gm1 = self._eos_gamma - 1.0
                eps[low_p_mask] = self.p_floor / (rho_lp * gm1)
            else:
                eps[low_p_mask] = self.eos.eps_from_rho_p(rho0[low_p_mask], self.p_floor)
            h[low_p_mask] = 1.0 + eps[low_p_mask] + self.p_floor / rho_lp

        # =========================================================================
        # UPDATE CONSERVATIVE VARIABLES FOR ATMOSPHERE POINTS (Fortran Con2Prim style)
        # When primitives are reset to atmosphere, conservatives must be consistent.
        # This ensures numerical stability and prevents drift.
        # =========================================================================
        atm_points = rho0 <= 10.0 * self.rho_floor
        if np.any(atm_points):
            # Recompute conservatives from atmosphere primitives using prim_to_cons
            D_atm, Sr_atm, tau_atm = prim_to_cons(
                rho0[atm_points], vr[atm_points], p[atm_points],
                gamma_rr[atm_points], self.eos,
                e6phi=e6phi[atm_points],
                alpha=alpha[atm_points],
                beta_r=beta_r[atm_points]
            )
            D_out[atm_points] = D_atm
            Sr_out[atm_points] = Sr_atm
            tau_out[atm_points] = tau_atm

        return rho0, vr, p, eps, W, h, success, D_out, Sr_out, tau_out

    def get_statistics(self):
        """Get conversion statistics."""
        total = max(self.stats["total_calls"], 1)
        return {
            **self.stats,
            "success_rate": self.stats["successful_conversions"] / total,
            "newton_rate": self.stats["newton_successes"] / total,
        }

    def reset_statistics(self):
        """Reset all statistics counters."""
        for key in self.stats:
            self.stats[key] = 0

    def _solve_vectorized_points(self, D, Sr, tau, gamma_rr, p_guess=None, alpha=None):        
        """
        Vectorized solver for multiple points using Newton-Raphson.
        """
        N = len(D)

        # Initialize pressure guess
        if p_guess is not None:
            p = np.maximum(p_guess, self.p_floor)
        else:
            p = np.maximum(self.p_floor, 0.1 * (tau + D))

        # Track convergence
        converged = np.zeros(N, dtype=bool)
        active = np.ones(N, dtype=bool)

        # Output arrays
        rho0 = np.zeros(N)
        vr = np.zeros(N)
        eps = np.zeros(N)
        W = np.ones(N)
        h = np.ones(N)

        # Pre-cache constants for ideal gas
        _is_ideal = getattr(self.eos, "name", "").startswith("ideal_gas")
        if _is_ideal and hasattr(self.eos, 'gamma'):
            eos_gamma = float(self.eos.gamma)
            c_ideal = eos_gamma / (eos_gamma - 1.0)
        else:
            c_ideal = None

        # Newton-Raphson iterations
        for iteration in range(self.max_iter):
            if not np.any(active):
                break

            # Evaluate function for active points
            p_active = p[active]
            ok_active, states_active = self._evaluate_pressure_vectorized(
                D[active], Sr[active], tau[active], p_active, gamma_rr[active],
                alpha=alpha[active]
            )

            # Handle evaluation failures
            if not np.all(ok_active):
                failed_indices = np.where(active)[0][~ok_active]
                p[failed_indices] = np.maximum(self.p_floor,
                                             p[failed_indices] * 1.5 + 1e-14)
                continue

            rho0_active, vr_active, eps_active, W_active, h_active, f_active = states_active

            # Check convergence
            tol_active = self.tol * np.maximum(1.0, np.abs(p_active))
            converged_now = np.abs(f_active) <= tol_active

            if np.any(converged_now):
                # Mark as converged and store results
                conv_indices = np.where(active)[0][converged_now]
                converged[conv_indices] = True
                rho0[conv_indices] = rho0_active[converged_now]
                vr[conv_indices] = vr_active[converged_now]
                eps[conv_indices] = eps_active[converged_now]
                W[conv_indices] = W_active[converged_now]
                h[conv_indices] = h_active[converged_now]

                # Remove from active set
                active[conv_indices] = False

            if not np.any(active):
                break

            # Compute derivatives for remaining points
            still_active = active & ~converged
            if not np.any(still_active):
                break

            p_still = p[still_active]
            # Residual aligned with still_active
            f_still = f_active[~converged_now] if np.any(converged_now) else f_active

            if _is_ideal and c_ideal is not None:
                #print("Using ideal gas analytic derivative")    
                # === Analytic df/dp for Ideal Gas EOS (optimized with cached constants) ===
                E = tau[still_active] + D[still_active]
                Q = E + p_still
                g = np.maximum(gamma_rr[still_active], 1e-30)
                Sr_sq = Sr[still_active] ** 2  
                Q_sq = Q * Q
                v2 =  Sr_sq / np.maximum(Q_sq * g, 1e-30)
                W_loc = 1.0 / np.sqrt(np.maximum(1.0 - v2, 1e-16))
                W_loc_cubed = W_loc * W_loc * W_loc  
                Wprime = - Sr_sq * W_loc_cubed / np.maximum(Q_sq * Q * g, 1e-30)
                W_loc_squared = W_loc * W_loc
                df = c_ideal * W_loc_squared + (D[still_active] + 2.0 * c_ideal * p_still * W_loc) * Wprime - 1.0
            else:
                print("Using numerical derivative")
                # Numerical derivative
                dp = np.maximum(1e-3 * np.maximum(np.abs(p_still), 1.0), 1e-14)
                ok2, states2 = self._evaluate_pressure_vectorized(
                    D[still_active], Sr[still_active], tau[still_active],
                    p_still + dp, gamma_rr[still_active],
                    alpha=alpha[still_active]
                )
                if not np.all(ok2):
                    failed_indices = np.where(still_active)[0][~ok2]
                    p[failed_indices] = np.maximum(self.p_floor,
                                                 p[failed_indices] * 1.5 + 1e-12)
                    continue

                num = states2[5] - f_still
                num = np.where(np.isfinite(num), num, np.inf)
                dp_safe = np.where(np.abs(dp) > 0, dp, 1e-12)
                df = num / dp_safe
                df = np.where(np.isfinite(df), df, 0.0)

            # Avoid small derivatives
            small_deriv = (np.abs(df) < 1e-15)
            if np.any(small_deriv):
                bad_indices = np.where(still_active)[0][small_deriv]
                p[bad_indices] = np.maximum(self.p_floor,
                                          p[bad_indices] * 1.5 + 1e-12)
                continue

            # Update pressure with damping
            p_new = p_still - f_still / df
            invalid_update = ~np.isfinite(p_new) | (p_new <= 0)
            p_new[invalid_update] = np.maximum(self.p_floor, 0.5 * p_still[invalid_update])

            p[still_active] = 0.5 * p_still + 0.5 * p_new  # Damped Newton

        # Update Newton success count
        self.stats["newton_successes"] += np.sum(converged)

        return rho0, vr, p, eps, W, h, converged

    def _evaluate_pressure_vectorized(self, D, Sr, tau, p, gamma_rr, alpha=None):
        """
        Vectorized evaluation of pressure residual for Newton-Raphson.
        
        Returns:
            tuple: (valid, (rho0, vr, eps, W, h, f))
        """
        N = len(D)
        p = np.maximum(p, self.p_floor)

        # Lapse (default = 1.0 for Minkowski or if not provided)
        if alpha is None:
            #print("WARNING [cons2prim._evaluate_pressure_vectorized]: alpha=None, using default alpha=1 (Minkowski)")
            alpha = np.ones(N)
        else:
            alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
        
        # Pre-compute all intermediate values (vectorized)
        Q = tau + D + p
        gamma_rr_safe = np.maximum(gamma_rr, 1e-30)
        alpha_safe = np.maximum(alpha, 1e-30)
        
        # Compute velocity for all points
        # vr = alpha * Sr / (Q * γrr)
        # From: Sr = ρ h W² vr γrr / alpha
        vr = alpha_safe * Sr / (Q * gamma_rr_safe)
        v2 = gamma_rr_safe * vr * vr
        
        # Compute Lorentz factor (safe sqrt)
        v2_safe = np.clip(v2, 0.0, 1.0 - 1e-16)
        W = 1.0 / np.sqrt(1.0 - v2_safe)
        
        # Rest mass density
        rho0 = D / np.maximum(W, 1e-30)
        
        # EOS evaluation: eps and h
        rho0_safe = np.maximum(rho0, 1e-30)
        if self._is_ideal_gas:
            gm1 = self._eos_gamma - 1.0
            eps = p / (rho0_safe * gm1)
            h = 1.0 + eps + p / rho0_safe
        else:
            eps = self.eos.eps_from_rho_p(rho0, p)
            h = 1.0 + eps + p / rho0_safe
        
        # Residual for all points
        W_squared = W * W
        f = rho0 * h * W_squared - Q
        
        # For ideal gas: c_s² = p Γ (Γ-1) / [p Γ + ρ₀ (Γ-1)]
        # Physical requirement: 0 ≤ c_s² < 1
        if self._is_ideal_gas:
            gamma = self._eos_gamma
            gm1 = gamma - 1.0
            denom = p * gamma + rho0_safe * gm1
            cs2 = p * gamma * gm1 / np.maximum(denom, 1e-30)
        else:
            # For general EOS, skip sound speed check (would need EOS derivatives)
            cs2 = np.zeros_like(h) + 0.5  # Assume valid
        
        # Single combined validity check (now includes sound speed)
        valid = (
            (Q > 0.0) &
            (v2 >= 0.0) & (v2 < 1.0) &
            (W >= 1.0) & (W <= self.W_max) &
            (rho0 > 0.0) & np.isfinite(rho0) &
            np.isfinite(eps) & (eps >= 0.0) &
            np.isfinite(h) & (h > 1.0) &
            np.isfinite(f) &
            (cs2 >= 0.0) & (cs2 < 1.0)  # Sound speed physical check
        )
        
        # Set invalid points to safe defaults
        if not np.all(valid):
            rho0[~valid] = 0.0
            vr[~valid] = 0.0
            eps[~valid] = 0.0
            W[~valid] = 1.0
            h[~valid] = 1.0
            f[~valid] = np.inf
        
        return valid, (rho0, vr, eps, W, h, f)
   

# ============================================================================
# PRIMITIVE TO CONSERVATIVE CONVERSION
# ============================================================================

def prim_to_cons(rho0, vr, pressure, gamma_rr, eos, e6phi=None, alpha=None, beta_r=None):
    """
    Convert primitive to conservative variables.

    Following Valencia formulation:
        Non-densitized (e6phi=None):
            D  = W ρ₀
            Sr = W² ρ₀ h vr γrr / alpha
            τ  = W² ρ₀ h - P - D

        Densitized (e6phi provided):
            D̃  = e^{6φ} W ρ₀
            S̃r = e^{6φ} W² ρ₀ h vr γrr
            τ̃  = e^{6φ} (alpha² T^{00} - W ρ₀)

    Returns:
        tuple: (D, Sr, tau) conservative variables (densitized if e6phi provided)
    """
    # Convert inputs to arrays for unified handling
    rho0 = np.asarray(rho0)
    vr = np.asarray(vr)
    pressure = np.asarray(pressure)
    gamma_rr = np.asarray(gamma_rr)

    # Broadcast to same shape
    rho0, vr, pressure, gamma_rr = np.broadcast_arrays(rho0, vr, pressure, gamma_rr)

    # Densitization factor (default = 1.0 for non-densitized)
    if e6phi is None:
        e6phi = np.ones_like(rho0)
    else:
        e6phi = np.broadcast_to(np.asarray(e6phi), rho0.shape)

    # Lapse (default = 1.0)
    if alpha is None:
        #print("WARNING [prim_to_cons]: alpha=None, using default alpha=1 (Minkowski)")
        alpha = np.ones_like(rho0)
    else:
        alpha = np.broadcast_to(np.asarray(alpha), rho0.shape)
    
    if beta_r is None:
        beta_r = np.zeros_like(rho0)
    else:
        beta_r = np.broadcast_to(np.asarray(beta_r), rho0.shape)

    # Compute derived quantities
    v2 = gamma_rr * vr * vr
    v2 = np.clip(v2, 0.0, 1.0 - 1e-12)
    W = 1.0 / np.sqrt(1.0 - v2)
    v_i = vr * gamma_rr

    # Thermodynamic quantities from EOS
    eps = eos.eps_from_rho_p(rho0, pressure)

    # Specific enthalpy: h = 1 + ε + P/ρ₀
    h = 1.0 + eps + pressure / np.maximum(rho0, 1e-30)

    # Conservative variables with densitization
    # T^{00} = ρ₀ h W² - P

    D   = e6phi * ( rho0 * W )
    Sr  = e6phi * ( rho0 * h * W * W * v_i )
    tau = e6phi * ( rho0 * h * W * W - pressure - rho0 * W )

    return D, Sr, tau


# Backward compatibility alias
ConservativeToPrimitive = Cons2PrimSolver
