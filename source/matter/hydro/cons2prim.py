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
            "galeazzi_successes": 0,
            "atmosphere_fallbacks": 0,
            "conservative_floors_applied": 0
        }

        # =====================================================================
        # SOLVER METHOD SELECTION (hardcoded)
        # Options: "newton" or "galeazzi"
        #   - "newton": Newton-Raphson on pressure (fast, needs good initial guess)
        #   - "galeazzi": Bisection method on pressure (robust, always converges)
        #                 Inspired by bracketed methods in Galeazzi et al. and AthenaK
        #                 Uses same residual function as Newton but with guaranteed convergence
        # =====================================================================
        self._solver_method = "newton"  # Change to "newton" or "galeazzi"

    def convert(self, D, Sr, tau, gamma_rr, p_guess=None, apply_conservative_floors=True,
                e6phi=None, alpha=None, beta_r=None):
        """
        Convert conservative to primitive variables using vectorized Newton-Raphson.

        IMPORTANT: Expects PHYSICAL (non-densitized) conservative variables:
            D  = ρ₀ W
            Sʳ = ρ₀ h W² vʳ γᵣᵣ / α
            τ  = ρ₀ h - P - D

        Args:
            D: array - Conserved density (physical)
            Sr: array - Conserved radial momentum (physical)
            tau: array - Conserved energy (physical)
            gamma_rr: array - Radial metric component
            p_guess: array (optional) - Pressure guess from previous timestep
            apply_conservative_floors: bool - Apply tau/S floors before solve (default: True)
            e6phi: array (optional) - Densitization factor e^{6φ} for conservative update
            alpha: array (optional) - Lapse function for conservative update
            beta_r: array (optional) - Radial shift component for conservative update

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
            if self._solver_method == "galeazzi":
                result = self._solve_galeazzi_vectorized(
                    D[solve_mask], Sr[solve_mask], tau[solve_mask], gamma_rr[solve_mask]
                )
            else:  # "newton" (default fallback)
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
        
        Args:
            D: Conserved density
            Sr: Conserved radial momentum
            tau: Conserved energy
            p: Pressure (trial value)
            gamma_rr: Radial metric component
            alpha: Lapse function (optional, default=1.0)
        
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
        # vr = α * Sr / (Q * γrr)
        # From: Sr = ρ h W² vr γrr / α
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
   
    # =========================================================================
    # BISECTION SOLVER (bracketed root finding)
    # Uses the same residual function as Newton but with bisection root finding
    # More robust than Newton-Raphson, guaranteed to converge for physical states
    # =========================================================================

    def _evaluate_residual_for_p(self, p, D, Sr, tau, gamma_rr):
        """
        Evaluate residual f(p) = ρ₀ h W² - Q for given pressure.

        Same formulation as Newton-Raphson but returns only the residual
        and derived quantities for False Position method.

        Returns:
            tuple: (f, rho0, vr, eps, W, h, valid)
        """
        p = np.maximum(p, self.p_floor)

        # Q = τ + D + p
        Q = tau + D + p
        gamma_rr_safe = np.maximum(gamma_rr, 1e-30)

        # Velocity: vr = Sr / (Q * γᵣᵣ)
        vr = Sr / (Q * gamma_rr_safe)
        v2 = gamma_rr_safe * vr * vr

        # Lorentz factor
        v2_safe = np.clip(v2, 0.0, 1.0 - 1e-16)
        W = 1.0 / np.sqrt(1.0 - v2_safe)

        # Rest mass density
        rho0 = D / np.maximum(W, 1e-30)

        # For ideal gas
        if self._is_ideal_gas:
            gm1 = self._eos_gamma - 1.0
            eps = p / (rho0 * gm1)
            h = 1.0 + eps + p / np.maximum(rho0, 1e-30)
        else:
            eps = self.eos.eps_from_rho_p(rho0, p)
            h = 1.0 + eps + p / np.maximum(rho0, 1e-30)

        # Residual: f = ρ₀ h W² - Q
        f = rho0 * h * W * W - Q

        # Validity check
        valid = (Q > 0) & (v2 < 1.0) & (W >= 1.0) & (rho0 > 0) & np.isfinite(f)

        return f, rho0, vr, eps, W, h, valid

    def _solve_galeazzi_vectorized(self, D, Sr, tau, gamma_rr):
        """
        Vectorized solver using bisection method on pressure.

        Uses the same residual function as Newton-Raphson:
            f(p) = ρ₀ h W² - Q

        But with bracketed bisection instead of Newton iteration.
        This is more robust and guaranteed to converge for physical states.

        Args:
            D: Conserved density (physical, not densitized)
            Sr: Conserved radial momentum (physical)
            tau: Conserved energy (physical)
            gamma_rr: Radial metric component

        Returns:
            tuple: (rho0, vr, p, eps, W, h, success)
        """
        N = len(D)

        # Output arrays
        rho0 = np.zeros(N)
        vr = np.zeros(N)
        p = np.zeros(N)
        eps = np.zeros(N)
        W = np.ones(N)
        h = np.ones(N)
        success = np.zeros(N, dtype=bool)

        if not self._is_ideal_gas:
            # Fall back to Newton for non-ideal gas
            return self._solve_vectorized_points(D, Sr, tau, gamma_rr, None, None)

        gamma = self._eos_gamma
        gm1 = gamma - 1.0

        # Check for valid input
        valid_input = (D > 0) & np.isfinite(D) & np.isfinite(Sr) & np.isfinite(tau)

        if not np.any(valid_input):
            return rho0, vr, p, eps, W, h, success

        # =====================================================================
        # BRACKET ESTIMATION
        # For f(p) = ρ₀ h W² - Q, we need p_lo and p_hi such that
        # f(p_lo) and f(p_hi) have opposite signs
        #
        # Physical constraint: v² < 1 requires Q > |Sr|/√γᵣᵣ
        # So: p > |Sr|/√γᵣᵣ - τ - D for subluminal velocity
        # =====================================================================

        gamma_rr_safe = np.maximum(gamma_rr, 1e-30)

        # Minimum pressure for subluminal velocity: Q > |Sr|/√γᵣᵣ
        # p_min_physical = |Sr|/√γᵣᵣ - τ - D (with some margin)
        Sr_mag = np.abs(Sr)
        Q_min = Sr_mag / np.sqrt(gamma_rr_safe) * 1.001  # Small margin for safety
        p_min_physical = Q_min - tau - D

        # Lower bracket: max of p_floor and physical minimum
        p_lo = np.maximum(self.p_floor, p_min_physical + self.p_floor)

        # Upper bracket estimation:
        # At high p: v → 0, W → 1, ρ₀ → D, h ≈ γp/(D(γ-1))
        # f ≈ D * γp/(D(γ-1)) - (τ + D + p) = γp/(γ-1) - τ - D - p
        #   = p(γ/(γ-1) - 1) - τ - D = p/(γ-1) - τ - D
        # f = 0 → p = (τ + D)(γ-1)
        # Use a larger value to ensure we bracket the root
        p_hi = np.maximum(10.0 * np.abs(tau + D) * gm1, 10.0 * self.p_floor)
        p_hi = np.maximum(p_hi, np.abs(tau + D) + self.p_floor)
        p_hi = np.maximum(p_hi, p_lo * 100.0)  # Ensure p_hi >> p_lo

        # Evaluate at brackets
        f_lo, _, _, _, _, _, valid_lo = self._evaluate_residual_for_p(
            p_lo, D, Sr, tau, gamma_rr
        )
        f_hi, _, _, _, _, _, valid_hi = self._evaluate_residual_for_p(
            p_hi, D, Sr, tau, gamma_rr
        )

        # Check if brackets are valid (opposite signs)
        bracket_ok = valid_lo & valid_hi & (f_lo * f_hi < 0)

        # For cases where bracket fails, try to adjust
        need_adjust = valid_input & ~bracket_ok
        if np.any(need_adjust):
            # Try expanding upper bracket
            p_hi[need_adjust] *= 10.0
            f_hi[need_adjust], _, _, _, _, _, valid_hi[need_adjust] = self._evaluate_residual_for_p(
                p_hi[need_adjust], D[need_adjust], Sr[need_adjust],
                tau[need_adjust], gamma_rr[need_adjust]
            )
            bracket_ok = valid_lo & valid_hi & (f_lo * f_hi < 0)

        # Points that can be solved
        solvable = valid_input & bracket_ok

        if not np.any(solvable):
            return rho0, vr, p, eps, W, h, success

        # Extract solvable points
        idx_solve = np.where(solvable)[0]
        D_s = D[solvable]
        Sr_s = Sr[solvable]
        tau_s = tau[solvable]
        gamma_rr_s = gamma_rr[solvable]
        p_lo_s = p_lo[solvable]
        p_hi_s = p_hi[solvable]
        f_lo_s = f_lo[solvable]
        f_hi_s = f_hi[solvable]
        N_solve = len(D_s)

        # Initialize pressure guess (midpoint)
        p_s = 0.5 * (p_lo_s + p_hi_s)
        converged = np.zeros(N_solve, dtype=bool)

        # =====================================================================
        # BISECTION METHOD (guaranteed convergence)
        # More robust than False Position, O(log2(N)) convergence
        # =====================================================================
        for iteration in range(self.max_iter):
            if np.all(converged):
                break

            active = ~converged

            # Bisection: midpoint of bracket
            p_new = 0.5 * (p_lo_s[active] + p_hi_s[active])
            p_new = np.maximum(p_new, self.p_floor)

            # Evaluate at midpoint
            f_new, rho0_new, vr_new, eps_new, W_new, h_new, valid_new = self._evaluate_residual_for_p(
                p_new, D_s[active], Sr_s[active], tau_s[active], gamma_rr_s[active]
            )

            # Check convergence (either small residual or small bracket)
            tol_check = self.tol * np.maximum(1.0, np.abs(p_new))
            bracket_small = (p_hi_s[active] - p_lo_s[active]) < tol_check
            conv_now = (np.abs(f_new) < tol_check) | bracket_small
            conv_indices = np.where(active)[0][conv_now]

            if np.any(conv_now):
                converged[conv_indices] = True
                p_s[conv_indices] = p_new[conv_now]

            # Update brackets for non-converged points
            update_mask = ~conv_now & valid_new
            if not np.any(update_mask):
                continue

            active_indices = np.where(active)[0][update_mask]

            # Determine which bracket to update based on sign of f_new vs f_lo
            # f_lo and f_hi have opposite signs (guaranteed by bracketing)
            same_sign_lo = (f_new[update_mask] * f_lo_s[active_indices]) > 0

            # Update lower bracket where f_new has same sign as f_lo
            lo_update = active_indices[same_sign_lo]
            if len(lo_update) > 0:
                p_lo_s[lo_update] = p_new[update_mask][same_sign_lo]
                f_lo_s[lo_update] = f_new[update_mask][same_sign_lo]

            # Update upper bracket where f_new has same sign as f_hi
            hi_update = active_indices[~same_sign_lo]
            if len(hi_update) > 0:
                p_hi_s[hi_update] = p_new[update_mask][~same_sign_lo]
                f_hi_s[hi_update] = f_new[update_mask][~same_sign_lo]

            p_s[active_indices] = p_new[update_mask]

        # =====================================================================
        # RECOVER ALL PRIMITIVE VARIABLES FROM CONVERGED PRESSURE
        # =====================================================================
        _, rho0_s, vr_s, eps_s, W_s, h_s, _ = self._evaluate_residual_for_p(
            p_s, D_s, Sr_s, tau_s, gamma_rr_s
        )

        # Apply floors
        rho0_s = np.maximum(rho0_s, self.rho_floor)
        p_s = np.maximum(p_s, self.p_floor)

        # Store results
        rho0[idx_solve] = rho0_s
        vr[idx_solve] = vr_s
        p[idx_solve] = p_s
        eps[idx_solve] = eps_s
        W[idx_solve] = W_s
        h[idx_solve] = h_s
        success[idx_solve] = converged

        # Update statistics
        self.stats["galeazzi_successes"] += np.sum(converged)

        return rho0, vr, p, eps, W, h, success


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
            τ̃  = e^{6φ} (α² T^{00} - W ρ₀)
    Args:
        rho0: Rest mass density
        vr: Radial velocity
        pressure: Pressure
        gamma_rr: Radial metric component
        eos: Equation of state
        e6phi: Densitization factor e^{6φ} (optional, default=None for non-densitized)
        alpha: Lapse (optional, default=1.0)

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
