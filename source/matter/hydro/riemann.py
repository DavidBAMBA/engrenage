# matter/hydro/riemann.py

import numpy as np


class HLLERiemannSolver:
    """
    HLLE (Harten–Lax–van Leer–Einfeldt) Riemann solver for (1D, radial) GRHD.

    This implementation is suitable for the Valencia formulation used in Engrenage.
    It returns *non-densitized* physical fluxes F^r(U) of the conservative vector
    U = (D, S_r, tau), i.e.
        F^r_D   = D * (v^r - beta^r/alpha)
        F^r_S_r = S_r * (v^r - beta^r/alpha) + p
        F^r_tau = (tau + p) * (v^r - beta^r/alpha)
    The caller (e.g., the reference-metric Valencia class) should multiply by
    α e^{6φ} √(γ̄/γ̂) if using the full reference-metric densitization.

    Wave speeds follow Banyuls et al. (1997) in 1D:
        λ± = [ α ( v^r (1 - c_s^2) ± c_s sqrt( (1 - v^2) γ^{rr} ) ) / (1 - v^2 c_s^2) ] - β^r
    with v^2 = γ_rr (v^r)^2 and γ^{rr} = 1/γ_rr.

    Notes
    -----
    * Robust to near-vacuum states via simple clipping.
    * Returns a 3-vector flux per interface call.
    * For Minkowski (α=1, β=0, γ_rr=1), λ± reduce to SR eigenvalues.
    """

    def __init__(self, name: str = "HLLE"):
        self.name = name
        self.solver_type = "approximate"

        # Stats
        self.total_calls = 0
        self.superluminal_detections = 0  # kept for compatibility (not used actively)
        self.negative_pressure_fixes = 0  # incremented if validate_input_states is used

        # Tunables
        self._eps_floor = 0.0
        self._p_floor = 1e-15
        self._v_cap = 0.999999
        self._speed_eps = 1e-12  # to avoid exact-zero denominators

    # ----------------------------------------------------------------------
    # Public APIs
    # ----------------------------------------------------------------------

    def solve(self, UL, UR, primL, primR, gamma_rr, alpha, beta_r, eos):
        """
        Solve a single-interface Riemann problem with HLLE.

        Inputs (scalars):
          UL, UR      : (D, S_r, tau) for left/right conservative states
          primL, primR: (rho0, v^r, p) for left/right primitive states
          gamma_rr    : spatial metric γ_rr at the interface
          alpha       : lapse α at the interface
          beta_r      : radial shift β_r at the interface
          eos         : EOS object with eps_from_rho_p(rho0,p) and enthalpy(rho0,p,eps)

        Returns:
          F_hlle (np.ndarray shape (3,)): non-densitized HLLE flux vector.
        """
        self.total_calls += 1

        # Unpack
        DL, SrL, tauL = UL
        DR, SrR, tauR = UR
        rho0L, vrL, pL = primL
        rho0R, vrR, pR = primR

        # Floors / safety for primitives
        pL = float(max(pL, self._p_floor))
        pR = float(max(pR, self._p_floor))
        vrL = float(np.clip(vrL, -self._v_cap, self._v_cap))
        vrR = float(np.clip(vrR, -self._v_cap, self._v_cap))
        gamma_rr = float(max(gamma_rr, 1e-30))
        alpha = float(max(alpha, 1e-30))
        beta_r = float(beta_r)

        # Specific internal energy and enthalpy
        epsL = max(eos.eps_from_rho_p(rho0L, pL), self._eps_floor)
        epsR = max(eos.eps_from_rho_p(rho0R, pR), self._eps_floor)
        # (h may be used by alternative flux formulations; kept for clarity)
        # hL = eos.enthalpy(rho0L, pL, epsL)
        # hR = eos.enthalpy(rho0R, pR, epsR)

        # Wave speed estimates (Banyuls 1D)
        lamL_min, lamL_max = self._wave_speeds_banyuls(vrL, rho0L, pL, epsL, gamma_rr, alpha, beta_r, eos)
        lamR_min, lamR_max = self._wave_speeds_banyuls(vrR, rho0R, pR, epsR, gamma_rr, alpha, beta_r, eos)

        lam_minus = min(lamL_min, lamR_min)
        lam_plus  = max(lamL_max, lamR_max)

        # Entropy fix near sonic points
        lam_minus, lam_plus = self._entropy_fix(lam_minus, lam_plus)

        # Physical (non-densitized) fluxes for L/R states
        FL = self._physical_flux(np.array([DL, SrL, tauL]), (rho0L, vrL, pL), alpha, beta_r)
        FR = self._physical_flux(np.array([DR, SrR, tauR]), (rho0R, vrR, pR), alpha, beta_r)

        # HLLE combination
        if lam_minus >= 0.0:
            return FL
        elif lam_plus <= 0.0:
            return FR
        else:
            denom = lam_plus - lam_minus
            if denom == 0.0:
                # Degenerate; average
                return 0.5 * (FL + FR)
            Udiff = np.array([DR - DL, SrR - SrL, tauR - tauL])  # UR - UL
            return (lam_plus * FL - lam_minus * FR + lam_plus * lam_minus * Udiff) / denom

    def solve_batch(self, UL_batch, UR_batch, primL_batch, primR_batch,
                    gamma_rr_batch, alpha_batch, beta_r_batch, eos):
        """
        Vectorized wrapper over interfaces.

        All inputs are arrays of the same length M. Returns (M,3) flux array.
        """
        M = len(UL_batch)
        out = np.zeros((M, 3), dtype=float)
        for i in range(M):
            out[i] = self.solve(
                UL_batch[i], UR_batch[i], primL_batch[i], primR_batch[i],
                float(gamma_rr_batch[i]), float(alpha_batch[i]), float(beta_r_batch[i]), eos
            )
        return out

    def estimate_dt(self, primitive_vars, gamma_rr, alpha, eos, dx, cfl_factor=0.5):
        """
        CFL timestep estimate based on max |λ| over a 1D array of states.

        primitive_vars: (rho0, v^r, p) arrays
        gamma_rr, alpha: arrays (or scalars) broadcastable to primitive length
        dx: scalar cell width in coordinate r
        """
        rho0, vr, p = primitive_vars
        rho0 = np.asarray(rho0, dtype=float)
        vr   = np.asarray(vr,   dtype=float)
        p    = np.asarray(p,    dtype=float)
        N = rho0.size

        gamma_rr = np.broadcast_to(gamma_rr, (N,)).astype(float)
        alpha    = np.broadcast_to(alpha,    (N,)).astype(float)

        max_speed = 0.0
        for i in range(N):
            eps = max(eos.eps_from_rho_p(rho0[i], max(p[i], self._p_floor)), self._eps_floor)
            lam_min, lam_max = self._wave_speeds_banyuls(
                float(vr[i]), float(rho0[i]), float(p[i]), eps,
                float(gamma_rr[i]), float(alpha[i]), 0.0, eos
            )
            s = max(abs(lam_min), abs(lam_max))
            if s > max_speed:
                max_speed = s

        if max_speed <= 0.0:
            return 1e10
        return float(cfl_factor) * float(dx) / max_speed

    def validate_input_states(self, UL, UR, primL, primR):
        """
        Basic physical checks on inputs (scalars).
        """
        DL, SrL, tauL = UL
        DR, SrR, tauR = UR

        if not np.isfinite(DL) or not np.isfinite(DR) or DL <= 0.0 or DR <= 0.0:
            return False, "Non-positive or non-finite D"
        if tauL < -DL or tauR < -DR:
            return False, "Energy constraint violation (tau + D < 0)"

        rho0L, vrL, pL = primL
        rho0R, vrR, pR = primR
        if rho0L <= 0.0 or rho0R <= 0.0:
            return False, "Non-positive rest-mass density"
        if pL < 0.0 or pR < 0.0:
            self.negative_pressure_fixes += 1
            return False, "Negative pressure"
        if abs(vrL) >= 1.0 or abs(vrR) >= 1.0:
            return False, "Superluminal velocity"

        return True, "OK"

    def get_solver_statistics(self):
        if self.total_calls == 0:
            return {
                "total_calls": 0,
                "superluminal_detections": 0,
                "negative_pressure_fixes": 0,
                "superluminal_rate": 0.0,
                "negative_pressure_rate": 0.0,
            }
        return {
            "total_calls": self.total_calls,
            "superluminal_detections": self.superluminal_detections,
            "negative_pressure_fixes": self.negative_pressure_fixes,
            "superluminal_rate": self.superluminal_detections / self.total_calls,
            "negative_pressure_rate": self.negative_pressure_fixes / self.total_calls,
        }

    def reset_statistics(self):
        self.total_calls = 0
        self.superluminal_detections = 0
        self.negative_pressure_fixes = 0

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------

    def _physical_flux(self, U, prim, alpha, beta_r):
        """
        Non-densitized physical flux vector for Valencia variables.
        """
        D, Sr, tau = U
        _, vr, p = prim

        vtil = vr - beta_r / alpha  # transport velocity
        fD   = D * vtil
        fSr  = Sr * vtil + p
        ftau = (tau + p) * vtil
        return np.array([fD, fSr, ftau], dtype=float)

    def _wave_speeds_banyuls(self, vr, rho0, p, eps, gamma_rr, alpha, beta_r, eos):
        """
        Characteristic speeds λ± along r (Banyuls et al., 1997) in 1D:
            λ± = [ α ( v^r (1 - c_s^2) ± c_s sqrt( (1 - v^2) γ^{rr} ) ) / (1 - v^2 c_s^2) ] - β^r
        where v^2 = γ_rr (v^r)^2, γ^{rr} = 1/γ_rr, and c_s^2 is the fluid-frame sound speed.
        """
        # Guard inputs
        gamma_rr = float(max(gamma_rr, 1e-30))
        alpha    = float(max(alpha,    1e-30))
        beta_r   = float(beta_r)

        # Sound speed squared (clip)
        cs2 = float(eos.sound_speed_squared(rho0, max(p, self._p_floor), max(eps, self._eps_floor)))
        cs2 = float(np.clip(cs2, 0.0, 1.0 - 1e-12))

        v2 = gamma_rr * vr * vr
        v2 = float(np.clip(v2, 0.0, 1.0 - 1e-12))

        inv_gamma_rr = 1.0 / gamma_rr
        one_minus_v2 = max(1.0 - v2, 1e-30)
        denom = max(1.0 - v2 * cs2, 1e-30)

        root = np.sqrt(max(one_minus_v2 * inv_gamma_rr * cs2, 0.0))
        term = vr * (1.0 - cs2)

        lam_plus  = alpha * (term + root) / denom - beta_r
        lam_minus = alpha * (term - root) / denom - beta_r

        # Enforce ordering
        if lam_minus > lam_plus:
            lam_minus, lam_plus = lam_plus, lam_minus
        return lam_minus, lam_plus

    def _entropy_fix(self, lam_minus, lam_plus, delta=1e-8):
        """
        Simple Harten-Hyman style entropy fix: enforce a minimal magnitude
        near sonic points and maintain correct ordering.
        """
        # Push the negative branch at least to -delta and the positive branch to +delta
        lam_minus = min(lam_minus, -abs(delta))
        lam_plus  = max(lam_plus,  +abs(delta))

        # Ensure ordering
        if lam_minus > lam_plus:
            lam_minus, lam_plus = lam_plus, lam_minus
        return lam_minus, lam_plus


class HLLCRiemannSolver:
    """
    Placeholder HLLC (contact-resolving) solver.
    Falls back to HLLE while keeping a compatible interface.
    """

    def __init__(self):
        self.name = "HLLC"
        self.solver_type = "contact_resolving (fallback to HLLE)"
        self._hlle = HLLERiemannSolver()

    def solve(self, UL, UR, primL, primR, gamma_rr, alpha, beta_r, eos):
        return self._hlle.solve(UL, UR, primL, primR, gamma_rr, alpha, beta_r, eos)

    def solve_batch(self, *args, **kwargs):
        return self._hlle.solve_batch(*args, **kwargs)

    def estimate_dt(self, *args, **kwargs):
        return self._hlle.estimate_dt(*args, **kwargs)

    def get_solver_statistics(self):
        return self._hlle.get_solver_statistics()

    def reset_statistics(self):
        return self._hlle.reset_statistics()


# ----------------------------------------------------------------------
# Quick self-test
# ----------------------------------------------------------------------

def _test_riemann():
    from .eos import IdealGasEOS

    eos = IdealGasEOS(gamma=1.4)
    solver = HLLERiemannSolver()

    # Left (high pressure), Right (low pressure)
    rho0L, vrL, pL = 1.0, 0.0, 1.0
    rho0R, vrR, pR = 0.125, 0.0, 0.1

    epsL = eos.eps_from_rho_p(rho0L, pL)
    epsR = eos.eps_from_rho_p(rho0R, pR)
    hL = eos.enthalpy(rho0L, pL, epsL)
    hR = eos.enthalpy(rho0R, pR, epsR)
    WL = 1.0
    WR = 1.0

    DL  = rho0L * WL
    SrL = rho0L * hL * WL * WL * vrL
    tauL = rho0L * hL * WL * WL - pL - DL

    DR  = rho0R * WR
    SrR = rho0R * hR * WR * WR * vrR
    tauR = rho0R * hR * WR * WR - pR - DR

    UL = np.array([DL, SrL, tauL])
    UR = np.array([DR, SrR, tauR])
    primL = (rho0L, vrL, pL)
    primR = (rho0R, vrR, pR)

    # Minkowski metric
    gamma_rr, alpha, beta_r = 1.0, 1.0, 0.0

    F = solver.solve(UL, UR, primL, primR, gamma_rr, alpha, beta_r, eos)

    assert np.all(np.isfinite(F)), "Non-finite HLLE flux"
    print("HLLE flux:", F)
    print("Stats:", solver.get_solver_statistics())
    print("✓ HLLE self-test passed")


if __name__ == "__main__":
    _test_riemann()
