# cons2prim.py
"""
Conservative to primitive variable conversion for relativistic hydrodynamics.

This module provides a clean, extensible architecture for cons2prim conversion
that supports current ideal gas EOS and is ready for future EOS implementations.

Floor application follows IllinoisGRMHD strategy (see atmosphere.py).
"""

import numpy as np
from .atmosphere import AtmosphereParams, FloorApplicator

# ============================================================================
# MAIN SOLVER CLASS
# ============================================================================

class Cons2PrimSolver:
    """
    Conservative to primitive variable converter.

    This class provides efficient conversion with configurable parameters
    and built-in statistics tracking. Designed to be extended for different
    equation of state types in the future.

    Args:
        eos: Equation of state object with eps_from_rho_p(rho0, p) method
        atmosphere: AtmosphereParams object (optional, creates default if None)
        **params: Additional solver parameters (tolerance, max_iter, etc.)
    """

    def __init__(self, eos, atmosphere=None, **params):
        self.eos = eos

        # Atmosphere parameters (centralized floor management)
        if atmosphere is None:
            atmosphere = AtmosphereParams()
        elif isinstance(atmosphere, dict):
            # Backward compatibility: convert dict to AtmosphereParams
            atmosphere = AtmosphereParams(**atmosphere)
        self.atmosphere = atmosphere

        # Floor applicator
        self.floor_applicator = FloorApplicator(self.atmosphere, eos)

        # Solver-specific parameters (not floor-related)
        self.params = self._get_default_params()
        self.params.update(params)
        # Override with atmosphere parameters
        self.params.update(self.atmosphere.to_cons2prim_params())

        # Statistics tracking
        self.stats = {
            "total_calls": 0,
            "successful_conversions": 0,
            "newton_successes": 0,
            "bisection_fallbacks": 0,
            "atmosphere_fallbacks": 0,
            "conservative_floors_applied": 0
        }

    def convert(self, U, metric=None, p_guess=None, apply_conservative_floors=True):
        """
        Convert conservative to primitive variables using vectorized approach.

        Args:
            U: Conservative variables - dict {'D','Sr','tau'}, tuple (D,Sr,tau), or array
            metric: Metric components - dict or tuple (alpha, beta_r, gamma_rr) (optional)
            p_guess: Pressure guess array from previous timestep (optional)
            apply_conservative_floors: Apply tau and S_i floors before solve (default: True)

        Returns:
            dict: {'rho0', 'vr', 'p', 'eps', 'W', 'h', 'success'}
        """
        self.stats["total_calls"] += 1

        D, Sr, tau = self._parse_conservative_variables(U)
        N = len(D)
        _, _, gamma_rr = self._ensure_metric_arrays(metric, N)
        p_guess = self._validate_pressure_guess(p_guess, N)

        # Apply conservative variable floors (IllinoisGRMHD strategy)
        # This prevents many cons2prim failures by ensuring physical consistency
        if apply_conservative_floors:
            D, Sr, tau, floor_mask = self.floor_applicator.apply_conservative_floors(
                D, Sr, tau, gamma_rr
            )
            if np.any(floor_mask):
                self.stats["conservative_floors_applied"] += np.sum(floor_mask)

        # Allocate output arrays
        rho0, vr, p, eps, W, h = (np.zeros(N) for _ in range(6))
        success = np.zeros(N, dtype=bool)

        # Check input validity for all points
        valid_mask = self._is_valid_input_vectorized(D, Sr, tau)

        if np.any(valid_mask):
            # Solve for valid points using vectorized Newton-Raphson
            result = self._solve_vectorized_points(D[valid_mask], Sr[valid_mask],
                                                 tau[valid_mask], gamma_rr[valid_mask],
                                                 p_guess[valid_mask] if p_guess is not None else None)

            rho0_valid, vr_valid, p_valid, eps_valid, W_valid, h_valid, success_valid = result

            # Place results back
            rho0[valid_mask] = rho0_valid
            vr[valid_mask] = vr_valid
            p[valid_mask] = p_valid
            eps[valid_mask] = eps_valid
            W[valid_mask] = W_valid
            h[valid_mask] = h_valid
            success[valid_mask] = success_valid

            # Update statistics
            self.stats["successful_conversions"] += np.sum(success_valid)
            self.stats["atmosphere_fallbacks"] += np.sum(~success_valid)

        # Handle invalid and failed points with atmosphere
        failed_mask = ~success
        if np.any(failed_mask):
            atm = self._atmosphere_fallback()
            rho0[failed_mask] = atm[0]
            vr[failed_mask] = atm[1]
            p[failed_mask] = atm[2]
            eps[failed_mask] = atm[3]
            W[failed_mask] = atm[4]
            h[failed_mask] = atm[5]

        # Enforce velocity limits
        self._enforce_velocity_limit_vectorized(vr, gamma_rr)

        # Final primitive floors (stability near atmosphere):
        # Ensure pressure never dips below the configured floor after solve.
        # If we clamp pressure, recompute eps and h consistently via EOS.
        p_floor = float(self.params.get("p_floor", 1.0e-15))
        if p_floor > 0.0:
            low_p_mask = p < p_floor
            if np.any(low_p_mask):
                p[low_p_mask] = p_floor
                try:
                    eps[low_p_mask] = self.eos.eps_from_rho_p(rho0[low_p_mask], p[low_p_mask])
                except Exception:
                    # Fallback small internal energy if EOS call fails
                    eps[low_p_mask] = np.maximum(eps[low_p_mask], 1.0e-10)
                # Recompute enthalpy with available EOS interface
                try:
                    h[low_p_mask] = self.eos.enthalpy(rho0[low_p_mask], p[low_p_mask], eps[low_p_mask])
                except TypeError:
                    try:
                        h[low_p_mask] = self.eos.enthalpy(rho0[low_p_mask])
                    except Exception:
                        h[low_p_mask] = 1.0 + eps[low_p_mask] + p[low_p_mask] / np.maximum(rho0[low_p_mask], 1e-30)

        return {
            "rho0": rho0, "vr": vr, "p": p, "eps": eps,
            "W": W, "h": h, "success": success
        }

    def get_statistics(self):
        """Get conversion statistics."""
        total = max(self.stats["total_calls"], 1)
        return {
            **self.stats,
            "success_rate": self.stats["successful_conversions"] / total,
            "newton_rate": self.stats["newton_successes"] / total,
            "bisection_rate": self.stats["bisection_fallbacks"] / total,
        }

    def reset_statistics(self):
        """Reset all statistics counters."""
        for key in self.stats:
            self.stats[key] = 0

    def _solve_vectorized_points(self, D, Sr, tau, gamma_rr, p_guess=None):
        """
        Vectorized solver for multiple points using Newton-Raphson.
        """
        N = len(D)

        # Initialize pressure guess
        if p_guess is not None:
            p = np.maximum(p_guess, self.params["p_floor"])
        else:
            p = np.maximum(self.params["p_floor"], 0.1 * (tau + D))

        # Track convergence
        converged = np.zeros(N, dtype=bool)
        active = np.ones(N, dtype=bool)

        # Output arrays
        rho0 = np.zeros(N)
        vr = np.zeros(N)
        eps = np.zeros(N)
        W = np.ones(N)
        h = np.ones(N)

        # Ideal-gas quick check
        _is_ideal = getattr(self.eos, "name", "").startswith("ideal_gas")

        # Newton-Raphson iterations
        for iteration in range(self.params["max_iter"]):
            if not np.any(active):
                break

            # Evaluate function for active points
            p_active = p[active]
            ok_active, states_active = self._evaluate_pressure_vectorized(
                D[active], Sr[active], tau[active], p_active, gamma_rr[active]
            )

            # Handle evaluation failures
            if not np.all(ok_active):
                failed_indices = np.where(active)[0][~ok_active]
                p[failed_indices] = np.maximum(self.params["p_floor"],
                                             p[failed_indices] * 1.5 + 1e-12)
                continue

            rho0_active, vr_active, eps_active, W_active, h_active, f_active = states_active

            # Check convergence
            tol_active = self.params["tol"] * np.maximum(1.0, np.abs(p_active))
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

            if _is_ideal:
                # === Analytic df/dp for Ideal Gas EOS (2 líneas clave) ===
                E = tau[still_active] + D[still_active]
                Q = E + p_still
                g = np.maximum(gamma_rr[still_active], 1e-30)
                v2 = (Sr[still_active] ** 2) / np.maximum(Q * Q * g, 1e-30)
                W_loc = 1.0 / np.sqrt(np.maximum(1.0 - v2, 1e-16))
                Wprime = - (Sr[still_active] ** 2) * (W_loc ** 3) / np.maximum(Q ** 3 * g, 1e-30)
                c = float(self.eos.gamma) / (float(self.eos.gamma) - 1.0)
                df = c * (W_loc ** 2) + (D[still_active] + 2.0 * c * p_still * W_loc) * Wprime - 1.0
            else:
                # Numerical derivative
                dp = np.maximum(1e-3 * np.maximum(np.abs(p_still), 1.0), 1e-12)
                ok2, states2 = self._evaluate_pressure_vectorized(
                    D[still_active], Sr[still_active], tau[still_active],
                    p_still + dp, gamma_rr[still_active]
                )
                if not np.all(ok2):
                    failed_indices = np.where(still_active)[0][~ok2]
                    p[failed_indices] = np.maximum(self.params["p_floor"],
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
                p[bad_indices] = np.maximum(self.params["p_floor"],
                                          p[bad_indices] * 1.5 + 1e-12)
                continue

            # Update pressure with damping
            p_new = p_still - f_still / df
            invalid_update = ~np.isfinite(p_new) | (p_new <= 0)
            p_new[invalid_update] = np.maximum(self.params["p_floor"], 0.5 * p_still[invalid_update])

            p[still_active] = 0.5 * p_still + 0.5 * p_new  # Damped Newton

        # Handle non-converged points with bisection fallback
        not_converged = ~converged
        if np.any(not_converged):
            for i in np.where(not_converged)[0]:
                success, result = self._solve_bisection_fallback(D[i], Sr[i], tau[i], gamma_rr[i])
                if success:
                    rho0[i], vr[i], p[i], eps[i], W[i], h[i] = result
                    converged[i] = True
                    self.stats["bisection_fallbacks"] += 1

        # Update Newton success count
        newton_successes = np.sum(converged) - getattr(self, '_temp_bisection_count', 0)
        self.stats["newton_successes"] += max(0, newton_successes)

        return rho0, vr, p, eps, W, h, converged

    def _evaluate_pressure_vectorized(self, D, Sr, tau, p, gamma_rr):
        """
        Vectorized evaluation of pressure for multiple points.
        """
        return self._evaluate_pressure_vectorized_numpy(D, Sr, tau, p, gamma_rr)

    def _evaluate_pressure_vectorized_numpy(self, D, Sr, tau, p, gamma_rr):
        """
        Pure NumPy vectorized evaluation (fallback when Numba not available).
        """
        N = len(D)
        p = np.maximum(p, self.params["p_floor"])
        E = tau + D
        Q = E + p

        # Check validity
        valid = Q > 0.0

        # Initialize outputs
        rho0 = np.zeros(N)
        vr = np.zeros(N)
        eps = np.zeros(N)
        W = np.ones(N)
        h = np.ones(N)
        f = np.full(N, np.inf)

        if not np.any(valid):
            return valid, (rho0, vr, eps, W, h, f)

        # Process valid points
        Q_valid = Q[valid]
        Sr_valid = Sr[valid]
        gamma_rr_valid = np.maximum(gamma_rr[valid], 1e-30)
        D_valid = D[valid]
        p_valid = p[valid]

        # Compute velocity
        vr_valid = Sr_valid / (Q_valid * gamma_rr_valid)
        v2_valid = gamma_rr_valid * vr_valid**2

        # Velocity constraint
        v_ok = (0.0 <= v2_valid) & (v2_valid < 1.0)

        if np.any(v_ok):
            # Lorentz factor
            W_valid = 1.0 / np.sqrt(1.0 - v2_valid[v_ok])
            W_constraint = (1.0 <= W_valid) & (W_valid <= self.params["W_max"])

            if np.any(W_constraint):
                # Rest mass density
                W_good = W_valid[W_constraint]
                D_good = D_valid[v_ok][W_constraint]
                rho0_good = D_good / np.maximum(W_good, 1e-30)

                # Check rho0 validity
                rho_ok = (rho0_good > 0.0) & np.isfinite(rho0_good)

                if np.any(rho_ok):
                    rho0_final = rho0_good[rho_ok]
                    p_final = p_valid[v_ok][W_constraint][rho_ok]

                    # EOS evaluation
                    try:
                        eps_final = self.eos.eps_from_rho_p(rho0_final, p_final)
                    except:
                        eps_final = np.full_like(rho0_final, np.inf)

                    eps_ok = np.isfinite(eps_final) & (eps_final >= 0.0)

                    if np.any(eps_ok):
                        # Enthalpy
                        rho0_ultimate = rho0_final[eps_ok]
                        eps_ultimate = eps_final[eps_ok]
                        p_ultimate = p_final[eps_ok]

                        # Calculate enthalpy h using EOS interface when available
                        try:
                            # Ideal-gas style signature
                            h_ultimate = self.eos.enthalpy(rho0_ultimate, p_ultimate, eps_ultimate)
                        except TypeError:
                            try:
                                # Barotropic/polytropic signature
                                h_ultimate = self.eos.enthalpy(rho0_ultimate)
                            except Exception:
                                # Fallback (ideal-gas-like)
                                h_ultimate = 1.0 + eps_ultimate + p_ultimate / np.maximum(rho0_ultimate, 1e-30)
                        h_ok = np.isfinite(h_ultimate) & (h_ultimate > 1.0)

                        if np.any(h_ok):
                            # Build final index mapping
                            valid_indices = np.where(valid)[0]
                            v_indices = valid_indices[v_ok]
                            W_indices = v_indices[W_constraint]
                            rho_indices = W_indices[rho_ok]
                            final_indices = rho_indices[eps_ok][h_ok]

                            # Store results
                            rho0[final_indices] = rho0_ultimate[h_ok]
                            vr[final_indices] = vr_valid[v_ok][W_constraint][rho_ok][eps_ok][h_ok]
                            eps[final_indices] = eps_ultimate[h_ok]
                            W[final_indices] = W_good[rho_ok][eps_ok][h_ok]
                            h[final_indices] = h_ultimate[h_ok]

                            # Residual
                            Q_final = Q[final_indices]
                            f[final_indices] = (rho0[final_indices] * h[final_indices] *
                                              W[final_indices]**2 - Q_final)

                            # Update validity
                            temp_valid = np.zeros(N, dtype=bool)
                            temp_valid[final_indices] = True
                            valid = temp_valid

        return valid, (rho0, vr, eps, W, h, f)

    def _is_valid_input_vectorized(self, D, Sr, tau):
        """Vectorized input validation."""
        return (np.isfinite(D) & np.isfinite(Sr) & np.isfinite(tau) &
                (D >= self.params["rho_floor"]) & (tau >= -D))

    def _enforce_velocity_limit_vectorized(self, vr, gamma_rr):
        """Vectorized velocity limit enforcement."""
        v2 = gamma_rr * vr**2
        violation_mask = v2 >= self.params["v_max"]**2
        if np.any(violation_mask):
            vr[violation_mask] = (np.sign(vr[violation_mask]) * self.params["v_max"] /
                                 np.sqrt(np.maximum(gamma_rr[violation_mask], 1e-30)))

    # ------------------------------------------------------------------------
    # CORE SOLVING METHODS
    # ------------------------------------------------------------------------

    def _solve_single_point(self, D, Sr, tau, gamma_rr, p_guess=None):
        """
        Solve for primitive variables at a single grid point.

        Strategy:
        1. Try pressure guess if provided
        2. Newton-Raphson method (primary)
        3. Bisection fallback if Newton fails

        Returns:
            (success, (rho0, vr, p, eps, W, h)) or (False, None)
        """
        # Try pressure guess first
        if p_guess is not None and p_guess > self.params["p_floor"]:
            ok, primitives = self._evaluate_pressure(D, Sr, tau, p_guess, gamma_rr)
            if ok and abs(primitives[5]) <= self.params["tol"] * max(1.0, abs(p_guess)):
                return True, (primitives[0], primitives[1], p_guess, primitives[2], primitives[3], primitives[4])

        # Newton-Raphson method (primary solver)
        success, result = self._solve_newton_raphson(D, Sr, tau, gamma_rr)
        if success:
            self.stats["newton_successes"] += 1
            return True, result

        # Bisection fallback
        success, result = self._solve_bisection_fallback(D, Sr, tau, gamma_rr)
        if success:
            self.stats["bisection_fallbacks"] += 1
            return True, result

        return False, None

    def _solve_newton_raphson(self, D, Sr, tau, gamma_rr):
        """Newton-Raphson solver with numerical derivatives (analytic for IdealGas)."""
        p = max(self.params["p_floor"], 0.1 * (tau + D))

        _is_ideal = getattr(self.eos, "name", "").startswith("ideal_gas")

        for _ in range(self.params["max_iter"]):
            ok, state = self._evaluate_pressure(D, Sr, tau, p, gamma_rr)

            if not ok:
                p = max(self.params["p_floor"], p * 1.5 + 1e-12)
                continue

            rho0, vr, eps, W, h, f = state

            if abs(f) <= self.params["tol"] * max(1.0, abs(p)):
                return True, (rho0, vr, p, eps, W, h)

            if _is_ideal:
                # === Analytic df/dp for Ideal Gas EOS (2 líneas clave) ===
                E = tau + D
                Q = E + p
                g = max(gamma_rr, 1e-30)
                Wprime = - (Sr * Sr) * (W ** 3) / max(Q ** 3 * g, 1e-30)
                c = float(self.eos.gamma) / (float(self.eos.gamma) - 1.0)
                df = c * (W ** 2) + (D + 2.0 * c * p * W) * Wprime - 1.0
            else:
                # Numerical derivative
                dp = max(1e-3 * max(abs(p), 1.0), 1e-12)
                ok2, state2 = self._evaluate_pressure(D, Sr, tau, p + dp, gamma_rr)
                if not ok2:
                    p = max(self.params["p_floor"], p * 1.5 + 1e-12)
                    continue
                df = (state2[5] - f) / dp

            if not np.isfinite(df) or abs(df) < 1e-15:
                p = max(self.params["p_floor"], p * 1.5 + 1e-12)
                continue

            p_new = p - f / df
            if not np.isfinite(p_new) or p_new <= 0:
                p_new = max(self.params["p_floor"], 0.5 * p)

            p = 0.5 * p + 0.5 * p_new  # Damped Newton

        return False, None

    def _solve_bisection_fallback(self, D, Sr, tau, gamma_rr):
        """Simple bisection solver as fallback."""
        E = tau + D
        p_lo = max(self.params["p_floor"], abs(Sr) / max(gamma_rr, 1e-30) - E + 1e-15)
        p_hi = max(p_lo * 2.0, 0.1 * (E + abs(Sr)))

        # Expand bracket if needed (more aggressive to handle near-surface states)
        for _ in range(25):
            ok_lo, state_lo = self._evaluate_pressure(D, Sr, tau, p_lo, gamma_rr)
            ok_hi, state_hi = self._evaluate_pressure(D, Sr, tau, p_hi, gamma_rr)

            if ok_lo and ok_hi and state_lo[5] * state_hi[5] <= 0:
                break
            p_hi *= 10.0

        if not (ok_lo and ok_hi and state_lo[5] * state_hi[5] <= 0):
            # Secondary attempt: coarse log sweep to locate sign change
            try:
                import numpy as _np
                Pmin = max(self.params["p_floor"], 1e-20)
                Pmax = max(1e4 * (abs(E) + abs(Sr) + 1.0), 10.0)
                grid = _np.geomspace(Pmin, Pmax, num=50)
                vals = []
                oks = []
                for pg in grid:
                    ok_g, st_g = self._evaluate_pressure(D, Sr, tau, float(pg), gamma_rr)
                    vals.append(st_g[5])
                    oks.append(ok_g)
                vals = _np.array(vals)
                oks = _np.array(oks, dtype=bool)
                # find first adjacent pair with ok and opposite sign
                found = False
                for i in range(len(grid)-1):
                    if oks[i] and oks[i+1] and _np.isfinite(vals[i]) and _np.isfinite(vals[i+1]) and vals[i]*vals[i+1] <= 0:
                        p_lo, p_hi = float(grid[i]), float(grid[i+1])
                        state_lo = (0,0,0,1,1,vals[i])
                        state_hi = (0,0,0,1,1,vals[i+1])
                        found = True
                        break
                if not found:
                    return False, None
            except Exception:
                return False, None

        # Bisection iterations
        for _ in range(self.params["max_iter"]):
            c = 0.5 * (p_lo + p_hi)
            okc, state_c = self._evaluate_pressure(D, Sr, tau, c, gamma_rr)

            if okc and abs(state_c[5]) <= self.params["tol"] * max(1.0, abs(c)):
                return True, (state_c[0], state_c[1], c, state_c[2], state_c[3], state_c[4])

            if okc and state_lo[5] * state_c[5] <= 0:
                p_hi = c
                state_hi = state_c
            else:
                p_lo = c
                state_lo = state_c

        return False, None

    def _evaluate_pressure(self, D, Sr, tau, p, gamma_rr):
        """
        Evaluate primitive state for given pressure.

        Returns:
            (success, (rho0, vr, eps, W, h, f)) where f is the residual
        """
        p = max(p, self.params["p_floor"])
        E = tau + D
        Q = E + p

        if Q <= 0.0:
            return False, (0, 0, 0, 1, 1, np.inf)

        g = max(gamma_rr, 1e-30)
        vr = Sr / (Q * g)
        v2 = g * vr * vr

        if not (0.0 <= v2 < 1.0):
            return False, (0, 0, 0, 1, 1, np.inf)

        W = 1.0 / np.sqrt(1.0 - v2)
        if not (1.0 <= W <= self.params["W_max"]):
            return False, (0, 0, 0, 1, 1, np.inf)

        rho0 = D / max(W, 1e-30)
        if rho0 <= 0.0 or not np.isfinite(rho0):
            return False, (0, 0, 0, 1, 1, np.inf)

        try:
            eps = self.eos.eps_from_rho_p(rho0, p)
        except Exception:
            return False, (0, 0, 0, 1, 1, np.inf)

        if not np.isfinite(eps) or eps < 0.0:
            return False, (0, 0, 0, 1, 1, np.inf)

        # Calculate enthalpy using EOS when possible
        try:
            h = self.eos.enthalpy(rho0, p, eps)
        except TypeError:
            try:
                h = self.eos.enthalpy(rho0)
            except Exception:
                h = 1.0 + eps + p / max(rho0, 1e-30)

        if not np.isfinite(h) or h <= 1.0:
            return False, (0, 0, 0, 1, 1, np.inf)

        # Residual function
        f = rho0 * h * W * W - Q

        return True, (rho0, vr, eps, W, h, f)

    # ------------------------------------------------------------------------
    # UTILITY METHODS
    # ------------------------------------------------------------------------

    def _get_default_params(self):
        """Default solver parameters."""
        return {
            "rho_floor": 1e-13,
            "p_floor": 1e-15,
            "v_max": 0.999999,
            "W_max": 1.0e3,
            "tol": 1e-12,
            "max_iter": 500,
        }

    def _parse_conservative_variables(self, U):
        """Parse conservative variables from various input formats."""
        if isinstance(U, dict):
            D = np.atleast_1d(np.asarray(U["D"], dtype=float))
            Sr = np.atleast_1d(np.asarray(U["Sr"], dtype=float))
            tau = np.atleast_1d(np.asarray(U["tau"], dtype=float))
        elif isinstance(U, (tuple, list)) and len(U) == 3:
            D = np.atleast_1d(np.asarray(U[0], dtype=float))
            Sr = np.atleast_1d(np.asarray(U[1], dtype=float))
            tau = np.atleast_1d(np.asarray(U[2], dtype=float))
        else:
            arr = np.asarray(U, dtype=float)
            if arr.ndim == 1 and arr.size == 3:
                D, Sr, tau = arr
                D = np.atleast_1d(D)
                Sr = np.atleast_1d(Sr)
                tau = np.atleast_1d(tau)
            elif arr.ndim >= 1 and arr.shape[-1] == 3:
                D = arr[..., 0].ravel()
                Sr = arr[..., 1].ravel()
                tau = arr[..., 2].ravel()
            else:
                raise ValueError("Invalid conservative variable format")

        N = len(D)
        if not (len(Sr) == N and len(tau) == N):
            raise ValueError("Conservative variables must have same length")

        return D, Sr, tau

    def _ensure_metric_arrays(self, metric, N):
        """Ensure metric components are proper arrays."""
        if metric is None:
            return np.ones(N), np.zeros(N), np.ones(N)

        if isinstance(metric, dict):
            alpha = np.broadcast_to(metric.get("alpha", 1.0), N).astype(float)
            beta_r = np.broadcast_to(metric.get("beta_r", 0.0), N).astype(float)
            gamma_rr = np.broadcast_to(metric.get("gamma_rr", 1.0), N).astype(float)
        else:
            alpha, beta_r, gamma_rr = metric
            alpha = np.broadcast_to(alpha, N).astype(float)
            beta_r = np.broadcast_to(beta_r, N).astype(float)
            gamma_rr = np.broadcast_to(gamma_rr, N).astype(float)

        return alpha, beta_r, gamma_rr

    def _validate_pressure_guess(self, p_guess, N):
        """Validate and format pressure guess array."""
        if p_guess is None:
            return None

        p_guess = np.asarray(p_guess)
        if p_guess.shape != (N,):
            return None

        return p_guess

    def _is_valid_input(self, D, Sr, tau):
        """Check if conservative variables are physically valid."""
        return (np.isfinite(D) and np.isfinite(Sr) and np.isfinite(tau)
                and D >= self.params["rho_floor"] and tau >= -D)

    def _atmosphere_fallback(self):
        """Return atmosphere values when conversion fails."""
        rho0 = self.atmosphere.rho_floor
        vr = 0.0
        p = self.atmosphere.p_floor

        try:
            eps = self.eos.eps_from_rho_p(rho0, p)
        except Exception:
            eps = 1e-10

        W = 1.0

        # Calculate enthalpy using EOS when possible
        try:
            h = self.eos.enthalpy(rho0, p, eps)
        except TypeError:
            try:
                h = self.eos.enthalpy(rho0)
            except Exception:
                h = 1.0 + eps + p / np.maximum(rho0, 1e-30)

        return rho0, vr, p, eps, W, h

    def _enforce_velocity_limit(self, vr, i, gamma_rr):
        """Enforce maximum velocity constraint."""
        v2 = gamma_rr * (vr[i] ** 2)
        if v2 >= self.params["v_max"] ** 2:
            vr[i] = np.sign(vr[i]) * self.params["v_max"] / np.sqrt(max(gamma_rr, 1e-30))


# ============================================================================
# PRIMITIVE TO CONSERVATIVE CONVERSION
# ============================================================================

def prim_to_cons(rho0, vr, pressure, gamma_rr, eos):
    """
    Convert primitive to conservative variables.

    Args:
        rho0: Rest mass density
        vr: Radial velocity
        pressure: Pressure
        gamma_rr: Radial metric component
        eos: Equation of state

    Returns:
        tuple: (D, Sr, tau) conservative variables
    """
    # Convert inputs to arrays for unified handling
    rho0 = np.asarray(rho0)
    vr = np.asarray(vr)
    pressure = np.asarray(pressure)
    gamma_rr = np.asarray(gamma_rr)

    # Broadcast to same shape
    rho0, vr, pressure, gamma_rr = np.broadcast_arrays(rho0, vr, pressure, gamma_rr)

    # Compute derived quantities
    g = np.maximum(gamma_rr, 1e-30)
    v2 = g * vr * vr
    v2 = np.clip(v2, 0.0, 1.0 - 1e-12)
    W = 1.0 / np.sqrt(1.0 - v2)

    # Thermodynamic quantities from EOS
    eps = eos.eps_from_rho_p(rho0, pressure)

    # Specific enthalpy via EOS interface
    try:
        h = eos.enthalpy(rho0, pressure, eps)
    except TypeError:
        try:
            h = eos.enthalpy(rho0)
        except Exception:
            h = 1.0 + eps + pressure / np.maximum(rho0, 1e-30)

    # Conservative variables
    D = rho0 * W
    Sr = rho0 * h * W * W * vr * g
    tau = rho0 * h * W * W - pressure - D

    return D, Sr, tau


# ============================================================================
# BACKWARD COMPATIBILITY API
# ============================================================================

def _solve_pressure(D, Sr, tau, gamma_rr, eos, p_guess=None):
    """Legacy function for backward compatibility."""
    solver = Cons2PrimSolver(eos)
    U = {'D': np.array([D]), 'Sr': np.array([Sr]), 'tau': np.array([tau])}
    metric = {'gamma_rr': np.array([gamma_rr])}
    p_guess_array = np.array([p_guess]) if p_guess is not None else None

    result = solver.convert(U, metric=metric, p_guess=p_guess_array)

    if result['success'][0]:
        return True, (result['rho0'][0], result['vr'][0], result['p'][0],
                     result['eps'][0], result['W'][0], result['h'][0])
    else:
        return False, None

def _bracket_pressure(D, Sr, tau, gamma_rr, eos):
    """Legacy function for backward compatibility."""
    solver = Cons2PrimSolver(eos)
    ok, result = solver._solve_bisection_fallback(D, Sr, tau, gamma_rr)
    if ok:
        return True, result
    else:
        return False, None

def _state_from_p(D, Sr, tau, p, gamma_rr, eos):
    """Legacy function for backward compatibility."""
    solver = Cons2PrimSolver(eos)
    ok, state = solver._evaluate_pressure(D, Sr, tau, p, gamma_rr)
    if ok:
        return True, state[:5]  # Return rho0, vr, eps, W, h (without residual f)
    else:
        return False, None

def cons_to_prim(U, eos, params=None, metric=None, p_guess=None):
    """
    Functional interface for conservative to primitive conversion.

    This function provides backward compatibility with existing code
    while using the optimized class-based implementation internally.

    Args:
        U: Conservative variables
        eos: Equation of state
        params: Solver parameters dict (optional)
        metric: Metric components (optional)
        p_guess: Pressure guess array (optional)

    Returns:
        dict: Primitive variables and success flags
    """
    solver_params = params or {}
    solver = Cons2PrimSolver(eos, **solver_params)
    return solver.convert(U, metric=metric, p_guess=p_guess)



# Legacy class name for backward compatibility
ConservativeToPrimitive = Cons2PrimSolver
