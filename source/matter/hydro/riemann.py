# matter/hydro/riemann_hll.py
"""
HLL Riemann solver for GRHD using quadratic-eigenvalue characteristic speeds,
aligned with NRPy cmin/cmax HLL form.

Key fixes & alignment:
- Transport velocity uses Valencia form: vtil = v^r - beta^r/alpha (scalar & batch).
- HLL uses cmin/cmax (positive) as in NRPy, equivalent to lambda_± with:
    cmax = max(0, max(cp_L, cp_R)), cmin = -min(0, min(cm_L, cm_R))
- Entropy-fix batch swap corrected.
- estimate_dt now accepts beta_r (default 0) and uses it in g^{μν}.

Fluxes are non-densitized Valencia {D, S_r, tau}.
"""

import numpy as np
from .atmosphere import AtmosphereParams
from .geometry import ADMGeometry


class HLLRiemannSolver:
    def __init__(self, name: str = "HLL", atmosphere=None):
        self.name = name
        self.solver_type = "approximate"
        self.atmosphere = atmosphere if atmosphere is not None else AtmosphereParams()

        # Statistics
        self.total_calls = 0
        self.superluminal_detections = 0
        self.negative_pressure_fixes = 0

        # Numerics
        self._eps_floor = 0.0

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def solve(self, UL, UR, primL, primR, gamma_rr, alpha, beta_r, eos):
        """
        Single-interface Riemann problem (scalars).

        Inputs:
          UL, UR      : (D, S_r, tau) left/right conservatives
          primL, primR: (rho0, v^r, p) left/right primitives
          gamma_rr    : γ_rr at face
          alpha       : lapse α at face
          beta_r      : shift β^r at face
          eos         : EOS with eps_from_rho_p and sound_speed_squared
        Returns:
          np.ndarray(3,) HLL flux (non-densitized)
        """
        self.total_calls += 1

        # Unpack
        DL, SrL, tauL = UL
        DR, SrR, tauR = UR
        rho0L, vrL, pL = primL
        rho0R, vrR, pR = primR

        # Floors / clips
        v_max = self.atmosphere.v_max
        p_floor = self.atmosphere.p_floor

        pL = float(max(pL, p_floor))
        pR = float(max(pR, p_floor))
        vrL = float(np.clip(vrL, -v_max, v_max))
        vrR = float(np.clip(vrR, -v_max, v_max))
        gamma_rr = float(max(gamma_rr, 1e-30))
        alpha = float(max(alpha, 1e-30))
        beta_r = float(beta_r)

        # Sound speeds
        epsL = max(eos.eps_from_rho_p(rho0L, pL), self._eps_floor)
        epsR = max(eos.eps_from_rho_p(rho0R, pR), self._eps_floor)
        cs2L = float(np.clip(eos.sound_speed_squared(rho0L, pL, epsL), 0.0, 1.0 - 1e-12))
        cs2R = float(np.clip(eos.sound_speed_squared(rho0R, pR, epsR), 0.0, 1.0 - 1e-12))

        # 4-velocity and 4-metric
        u4U_L = self._compute_4velocity(vrL, gamma_rr)
        u4U_R = self._compute_4velocity(vrR, gamma_rr)
        g4UU = self._ADM_to_g4UU(gamma_rr, beta_r, alpha)

        # Characteristic speeds via NRPy-style quadratic
        flux_dirn = 0
        cmL, cpL = self._find_cp_cm(flux_dirn, g4UU, u4U_L, cs2L)
        cmR, cpR = self._find_cp_cm(flux_dirn, g4UU, u4U_R, cs2R)

        # NRPy mapping: positive cmin,cmax
        cmax = max(0.0, max(cpL, cpR))
        cmin = -min(0.0, min(cmL, cmR))

        # Physical fluxes (non-densitized Valencia)
        FL = self._physical_flux(np.array([DL, SrL, tauL]), (rho0L, vrL, pL), alpha, beta_r)
        FR = self._physical_flux(np.array([DR, SrR, tauR]), (rho0R, vrR, pR), alpha, beta_r)

        # HLL (cmin/cmax form; equivalent a lambda± con el mapeo anterior)
        Udiff = np.array([DR - DL, SrR - SrL, tauR - tauL])
        denom = cmax + cmin
        if denom <= 1e-30:
            return 0.5 * (FL + FR)
        return (cmin * FR + cmax * FL - (cmin * cmax) * Udiff) / denom

    def solve_batch(self, UL_batch, UR_batch, primL_batch, primR_batch,
                    gamma_rr_batch, alpha_batch, beta_r_batch, eos):
        """
        Vectorized multi-interface solver (arrays of length M). Returns (M,3).
        """
        M = len(UL_batch)

        UL_batch = np.asarray(UL_batch, dtype=float)
        UR_batch = np.asarray(UR_batch, dtype=float)
        primL_batch = np.asarray(primL_batch, dtype=float)
        primR_batch = np.asarray(primR_batch, dtype=float)
        gamma_rr_batch = np.asarray(gamma_rr_batch, dtype=float)
        alpha_batch = np.asarray(alpha_batch, dtype=float)
        beta_r_batch = np.asarray(beta_r_batch, dtype=float)

        DL, SrL, tauL = UL_batch[:, 0], UL_batch[:, 1], UL_batch[:, 2]
        DR, SrR, tauR = UR_batch[:, 0], UR_batch[:, 1], UR_batch[:, 2]

        rho0L = primL_batch[:, 0]
        vrL   = primL_batch[:, 1]
        pL    = primL_batch[:, 2]
        rho0R = primR_batch[:, 0]
        vrR   = primR_batch[:, 1]
        pR    = primR_batch[:, 2]

        # Floors/clips vectorizados
        pL = np.maximum(pL, self.atmosphere.p_floor)
        pR = np.maximum(pR, self.atmosphere.p_floor)
        vrL = np.clip(vrL, -self.atmosphere.v_max, self.atmosphere.v_max)
        vrR = np.clip(vrR, -self.atmosphere.v_max, self.atmosphere.v_max)
        gamma_rr_batch = np.maximum(gamma_rr_batch, 1e-30)
        alpha_batch = np.maximum(alpha_batch, 1e-30)

        # Sound speeds
        epsL = np.maximum(eos.eps_from_rho_p(rho0L, pL), self._eps_floor)
        epsR = np.maximum(eos.eps_from_rho_p(rho0R, pR), self._eps_floor)
        cs2L = np.clip(eos.sound_speed_squared(rho0L, pL, epsL), 0.0, 1.0 - 1e-12)
        cs2R = np.clip(eos.sound_speed_squared(rho0R, pR, epsR), 0.0, 1.0 - 1e-12)

        # 4-velocity y métrica
        u4U_L = self._compute_4velocity_batch(vrL, gamma_rr_batch)
        u4U_R = self._compute_4velocity_batch(vrR, gamma_rr_batch)
        g4UU = self._ADM_to_g4UU_batch(gamma_rr_batch, beta_r_batch, alpha_batch)

        # Características vectorizadas (NRPy)
        cmL, cpL = self._find_cp_cm_batch(0, g4UU, u4U_L, cs2L)
        cmR, cpR = self._find_cp_cm_batch(0, g4UU, u4U_R, cs2R)
        cmax = np.maximum(0.0, np.maximum(cpL, cpR))
        cmin = -np.minimum(0.0, np.minimum(cmL, cmR))

        # Flujos físicos (no densitizados)
        FL = self._physical_flux_batch(
            np.stack([DL, SrL, tauL], axis=1),
            np.stack([rho0L, vrL, pL], axis=1),
            alpha_batch, beta_r_batch
        )
        FR = self._physical_flux_batch(
            np.stack([DR, SrR, tauR], axis=1),
            np.stack([rho0R, vrR, pR], axis=1),
            alpha_batch, beta_r_batch
        )

        # HLL (cmin/cmax) vectorizado
        Udiff = np.stack([DR - DL, SrR - SrL, tauR - tauL], axis=1)
        denom = cmax + cmin
        safe_denom = np.where(np.abs(denom) < 1e-30, 1.0, denom)

        out = (cmin[:, None] * FR + cmax[:, None] * FL
               - (cmin * cmax)[:, None] * Udiff) / safe_denom[:, None]

        # Caso degenerado: promedio simple
        deg = np.abs(denom) < 1e-30
        if np.any(deg):
            out[deg] = 0.5 * (FL[deg] + FR[deg])

        return out

    def estimate_dt(self, primitive_vars, gamma_rr, alpha, eos, dx,
                    cfl_factor=0.5, beta_r=None):
        """
        CFL timestep based on max characteristic speed (includes shift if provided).

        Args:
            primitive_vars: (rho0, v^r, p) arrays
            gamma_rr, alpha: arrays or scalars
            eos: EOS
            dx: grid spacing
            cfl_factor: CFL factor
            beta_r: optional array/scalar (default 0)
        """
        rho0, vr, p = primitive_vars
        rho0 = np.asarray(rho0, dtype=float)
        vr = np.asarray(vr, dtype=float)
        p = np.asarray(p, dtype=float)
        N = rho0.size

        gamma_rr = np.broadcast_to(gamma_rr, (N,)).astype(float)
        alpha = np.broadcast_to(alpha, (N,)).astype(float)
        if beta_r is None:
            beta_r = np.zeros_like(alpha)
        else:
            beta_r = np.broadcast_to(beta_r, (N,)).astype(float)

        max_speed = 0.0
        for i in range(N):
            eps = max(eos.eps_from_rho_p(rho0[i], max(p[i], self.atmosphere.p_floor)), self._eps_floor)
            cs2 = float(np.clip(eos.sound_speed_squared(rho0[i], p[i], eps), 0.0, 1.0 - 1e-12))

            u4U = self._compute_4velocity(float(vr[i]), float(gamma_rr[i]))
            g4UU = self._ADM_to_g4UU(float(gamma_rr[i]), float(beta_r[i]), float(alpha[i]))
            cm, cp = self._find_cp_cm(0, g4UU, u4U, cs2)
            s = max(abs(cm), abs(cp))
            if s > max_speed:
                max_speed = s

        if max_speed <= 0.0:
            return 1e10
        return float(cfl_factor) * float(dx) / max_speed

    def validate_input_states(self, UL, UR, primL, primR):
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
            return {"total_calls": 0, "superluminal_detections": 0,
                    "negative_pressure_fixes": 0,
                    "superluminal_rate": 0.0, "negative_pressure_rate": 0.0}
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
    # Internals (NRPy-style characteristic speeds)
    # Now using ADMGeometry for cleaner implementation
    # ----------------------------------------------------------------------
    def _compute_4velocity(self, vr, gamma_rr):
        """
        Compute 4-velocity for scalar 1D radial flow.
        Now delegates to ADMGeometry for consistency.
        """
        # Create minimal geometry for radial flow (Minkowski + radial metric)
        N = 1
        alpha = np.ones(N)
        beta_U = np.zeros((N, 3))
        gamma_LL = np.zeros((N, 3, 3))
        gamma_LL[:, 0, 0] = gamma_rr
        gamma_LL[:, 1, 1] = 1.0
        gamma_LL[:, 2, 2] = 1.0

        geom = ADMGeometry(alpha, beta_U, gamma_LL)

        # Velocity in radial direction only
        v_U = np.zeros((N, 3))
        v_U[:, 0] = vr

        # Compute 4-velocity
        u4U_full = geom.compute_4velocity(v_U)
        return u4U_full[0]  # Return scalar version

    def _compute_4velocity_batch(self, vr, gamma_rr):
        """
        Compute 4-velocity for batch of 1D radial states.
        Now delegates to ADMGeometry for consistency.
        """
        M = len(vr)
        alpha = np.ones(M)
        beta_U = np.zeros((M, 3))
        gamma_LL = np.zeros((M, 3, 3))
        gamma_LL[:, 0, 0] = gamma_rr
        gamma_LL[:, 1, 1] = 1.0
        gamma_LL[:, 2, 2] = 1.0

        geom = ADMGeometry(alpha, beta_U, gamma_LL)

        # Velocity in radial direction only
        v_U = np.zeros((M, 3))
        v_U[:, 0] = vr

        return geom.compute_4velocity(v_U)

    def _ADM_to_g4UU(self, gamma_rr, beta_r, alpha):
        """
        Convert ADM variables to 4-metric g^{μν} (scalar version).
        Now delegates to ADMGeometry for consistency.
        """
        # Create minimal geometry
        N = 1
        alpha_arr = np.array([alpha])
        beta_U = np.zeros((N, 3))
        beta_U[:, 0] = beta_r
        gamma_LL = np.zeros((N, 3, 3))
        gamma_LL[:, 0, 0] = gamma_rr
        gamma_LL[:, 1, 1] = 1.0
        gamma_LL[:, 2, 2] = 1.0

        geom = ADMGeometry(alpha_arr, beta_U, gamma_LL)
        return geom.get_4metric_contravariant_scalar(0)

    def _ADM_to_g4UU_batch(self, gamma_rr, beta_r, alpha):
        """
        Convert ADM variables to 4-metric g^{μν} (batch version).
        Now delegates to ADMGeometry for consistency.
        """
        M = len(gamma_rr)
        beta_U = np.zeros((M, 3))
        beta_U[:, 0] = beta_r
        gamma_LL = np.zeros((M, 3, 3))
        gamma_LL[:, 0, 0] = gamma_rr
        gamma_LL[:, 1, 1] = 1.0
        gamma_LL[:, 2, 2] = 1.0

        geom = ADMGeometry(alpha, beta_U, gamma_LL)
        return geom.get_4metric_contravariant()

    def _find_cp_cm(self, flux_dirn, g4UU, u4U, cs2):
        v02 = cs2
        i = flux_dirn + 1
        a = (1.0 - v02) * (u4U[0] ** 2) - v02 * g4UU[0][0]
        b = 2.0 * v02 * g4UU[i][0] - 2.0 * u4U[i] * u4U[0] * (1.0 - v02)
        c = (1.0 - v02) * (u4U[i] ** 2) - v02 * g4UU[i][i]
        detm = np.sqrt(max(0.0, b * b - 4.0 * a * c))
        if abs(a) < 1e-30:
            return -1.0, 1.0
        cplus_tmp = 0.5 * (-b / a + detm / a)
        cminus_tmp = -0.5 * (b / a + detm / a)
        return min(cplus_tmp, cminus_tmp), max(cplus_tmp, cminus_tmp)

    def _find_cp_cm_batch(self, flux_dirn, g4UU, u4U, cs2):
        i = flux_dirn + 1
        v02 = cs2
        a = (1.0 - v02) * (u4U[:, 0] ** 2) - v02 * g4UU[:, 0, 0]
        b = 2.0 * v02 * g4UU[:, i, 0] - 2.0 * u4U[:, i] * u4U[:, 0] * (1.0 - v02)
        c = (1.0 - v02) * (u4U[:, i] ** 2) - v02 * g4UU[:, i, i]
        detm = np.sqrt(np.maximum(0.0, b * b - 4.0 * a * c))
        safe_a = np.where(np.abs(a) < 1e-30, 1e-30, a)
        cplus_tmp = 0.5 * (-b / safe_a + detm / safe_a)
        cminus_tmp = -0.5 * (b / safe_a + detm / safe_a)
        cminus = np.minimum(cplus_tmp, cminus_tmp)
        cplus = np.maximum(cplus_tmp, cminus_tmp)
        deg = np.abs(a) < 1e-30
        cminus = np.where(deg, -1.0, cminus)
        cplus  = np.where(deg,  1.0, cplus)
        return cminus, cplus

    # ----------------------------------------------------------------------
    # Flux helpers
    # ----------------------------------------------------------------------
    def _physical_flux(self, U, prim, alpha, beta_r):
        D, Sr, tau = U
        _, vr, p = prim
        vtil = vr - beta_r / alpha  # Valencia transport velocity
        fD = D * vtil
        fSr = Sr * vtil + p
        ftau = (tau + p) * vtil
        return np.array([fD, fSr, ftau], dtype=float)

    def _physical_flux_batch(self, U, prim, alpha, beta_r):
        D = U[:, 0]; Sr = U[:, 1]; tau = U[:, 2]
        vr = prim[:, 1]; p = prim[:, 2]
        vtil = vr - beta_r / alpha
        fD = D * vtil
        fSr = Sr * vtil + p
        ftau = (tau + p) * vtil
        return np.stack([fD, fSr, ftau], axis=1)

    # (Mantengo entropy-fix por compatibilidad; no es necesario si usas cmin/cmax directamente)
    def _entropy_fix(self, lam_minus, lam_plus, delta=1e-8):
        lam_minus = min(lam_minus, -abs(delta))
        lam_plus  = max(lam_plus,  abs(delta))
        # orden garantizado
        lo = min(lam_minus, lam_plus)
        hi = max(lam_minus, lam_plus)
        return lo, hi

    def _entropy_fix_batch(self, lam_minus, lam_plus, delta=1e-8):
        lam_minus = np.minimum(lam_minus, -abs(delta))
        lam_plus  = np.maximum(lam_plus,  abs(delta))
        lo = np.minimum(lam_minus, lam_plus)
        hi = np.maximum(lam_minus, lam_plus)
        return lo, hi
