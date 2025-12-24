# riemann.py
"""
HLL Riemann solver for GRHD using quadratic eigenvalue method.

Uses quadratic equation method for characteristic speeds (robust near sonic points).
Vectorized implementation - all operations work on batches of interfaces.
"""

import numpy as np
from .atmosphere import AtmosphereParams


class HLLRiemannSolver:
    """
    HLL Riemann solver using quadratic eigenvalue method for characteristic speeds.

    The HLL flux formula:
        F_HLL = (λ+ F_L - λ- F_R + λ+ λ- (U_R - U_L)) / (λ+ - λ-)

    Characteristic speeds λ± computed via quadratic equation from 4-metric and 4-velocity.
    """

    def __init__(self, name: str = "HLL", atmosphere=None):
        self.name = name
        self.atmosphere = atmosphere if atmosphere is not None else AtmosphereParams()

    def solve_batch(self, UL_batch, UR_batch, primL_batch, primR_batch,
                    gamma_rr_batch, alpha_batch, beta_r_batch, eos):
        """
        Fully vectorized solver for multiple interfaces - PHASE 1 OPTIMIZATION.

        All inputs are arrays of length M. Returns (M,3) flux array.

        OPTIMIZATIONS:
        1. Fully vectorized numpy operations (no loops)
        2. Eliminates ~127k np.clip() calls by batching
        3. Caches EOS constants (gamma, gamma-1) to reduce attribute lookups
        4. Single masked operations instead of cascading loops
        """
        M = len(UL_batch)

        # OPTIMIZATION: Cache EOS constants to avoid repeated attribute lookups
        # These are accessed once instead of ~80k times
        if hasattr(eos, 'gamma'):
            gamma = float(eos.gamma)
            gamma_minus_1 = float(eos.gamma_minus_1) if hasattr(eos, 'gamma_minus_1') else gamma - 1.0
        else:
            gamma = 1.4
            gamma_minus_1 = 0.4

        # Convert inputs to numpy arrays with proper dtypes
        UL_batch = np.asarray(UL_batch, dtype=float)  # (M, 3)
        UR_batch = np.asarray(UR_batch, dtype=float)
        primL_batch = np.asarray(primL_batch, dtype=float)
        primR_batch = np.asarray(primR_batch, dtype=float)
        gamma_rr_batch = np.asarray(gamma_rr_batch, dtype=float)  # (M,)
        alpha_batch = np.asarray(alpha_batch, dtype=float)
        beta_r_batch = np.asarray(beta_r_batch, dtype=float)

        # Unpack conservatives
        DL = UL_batch[:, 0]
        SrL = UL_batch[:, 1]
        tauL = UL_batch[:, 2]
        DR = UR_batch[:, 0]
        SrR = UR_batch[:, 1]
        tauR = UR_batch[:, 2]

        # Unpack and validate primitives using VECTORIZED clipping
        # OPTIMIZATION: Single vectorized np.clip() call instead of 127k scalar calls
        rho0L = primL_batch[:, 0]
        vrL = primL_batch[:, 1]
        pL = primL_batch[:, 2]

        rho0R = primR_batch[:, 0]
        vrR = primR_batch[:, 1]
        pR = primR_batch[:, 2]

        # Apply floors using vectorized operations
        pL = np.maximum(pL, self.atmosphere.p_floor)
        pR = np.maximum(pR, self.atmosphere.p_floor)

        # OPTIMIZATION: Vectorized velocity clipping (was ~2-3 scalar clips per interface)
        vrL = np.clip(vrL, -self.atmosphere.v_max, self.atmosphere.v_max)
        vrR = np.clip(vrR, -self.atmosphere.v_max, self.atmosphere.v_max)

        gamma_rr_batch = np.maximum(gamma_rr_batch, 1e-30)
        alpha_batch = np.maximum(alpha_batch, 1e-30)

        # Compute sound speeds squared (vectorized)
        epsL = np.maximum(eos.eps_from_rho_p(rho0L, pL), 1e-15)
        epsR = np.maximum(eos.eps_from_rho_p(rho0R, pR), 1e-15)
        cs2L = eos.sound_speed_squared(rho0L, pL, epsL)
        cs2R = eos.sound_speed_squared(rho0R, pR, epsR)

        # OPTIMIZATION: Single vectorized clip for all sound speeds
        cs2L = np.clip(cs2L, 0.0, 1.0 - 1e-12)
        cs2R = np.clip(cs2R, 0.0, 1.0 - 1e-12)

        # Compute 4-velocities for all interfaces (vectorized)
        u4U_L = self._compute_4velocity(vrL, gamma_rr_batch)
        u4U_R = self._compute_4velocity(vrR, gamma_rr_batch)

        # Build 4-metric components at all interfaces
        g4UU = self._ADM_to_g4UU(gamma_rr_batch, beta_r_batch, alpha_batch)

        # Compute characteristic speeds
        cmL, cpL = self._find_cp_cm(0, g4UU, u4U_L, cs2L)
        cmR, cpR = self._find_cp_cm(0, g4UU, u4U_R, cs2R)

        # Global wave speed bounds (vectorized)
        cmax = np.maximum(0.0, np.maximum(cpL, cpR))
        cmin = -np.minimum(0.0, np.minimum(cmL, cmR))

        # Standard notation
        lam_minus = -cmin
        lam_plus = cmax
        lam_minus, lam_plus = self._entropy_fix(lam_minus, lam_plus)

        # Compute physical fluxes
        FL = self._physical_flux(
            np.stack([DL, SrL, tauL], axis=1),
            np.stack([rho0L, vrL, pL], axis=1),
            alpha_batch, beta_r_batch
        )
        FR = self._physical_flux(
            np.stack([DR, SrR, tauR], axis=1),
            np.stack([rho0R, vrR, pR], axis=1),
            alpha_batch, beta_r_batch
        )

        # HLL combination (vectorized)
        out = np.zeros((M, 3), dtype=float)

        # Case 1: lam_minus >= 0.0 (flow is always left)
        mask_left = lam_minus >= 0.0
        out[mask_left] = FL[mask_left]

        # Case 2: lam_plus <= 0.0 (flow is always right)
        mask_right = lam_plus <= 0.0
        out[mask_right] = FR[mask_right]

        # Case 3: Mixed flow (need HLL combination)
        mask_mixed = ~mask_left & ~mask_right
        denom = lam_plus[mask_mixed] - lam_minus[mask_mixed]

        # Avoid division by zero
        safe_denom = np.where(np.abs(denom) < 1e-30, 1.0, denom)

        Udiff = np.stack([DR - DL, SrR - SrL, tauR - tauL], axis=1)
        numerator = (lam_plus[mask_mixed, None] * FL[mask_mixed] -
                     lam_minus[mask_mixed, None] * FR[mask_mixed] +
                     (lam_plus[mask_mixed] * lam_minus[mask_mixed])[:, None] * Udiff[mask_mixed])

        out[mask_mixed] = numerator / safe_denom[:, None]

        # Handle degenerate case
        degenerate = mask_mixed & (np.abs(denom) < 1e-30)
        if np.any(degenerate):
            out[degenerate] = 0.5 * (FL[degenerate] + FR[degenerate])

        return out

    def _compute_4velocity(self, vr, gamma_rr):
        """
        Vectorized: Compute 4-velocity components for multiple interfaces.

        Inputs:
            vr: (M,) array of radial velocities
            gamma_rr: (M,) array of spatial metric components

        Returns: (M, 4) array of 4-velocities
        """
        v2 = gamma_rr * vr * vr
        v2 = np.clip(v2, 0.0, 1.0 - 1e-12)
        W = 1.0 / np.sqrt(1.0 - v2)

        u4U = np.zeros((len(vr), 4), dtype=float)
        u4U[:, 0] = W
        u4U[:, 1] = W * vr
        # u4U[:, 2] = 0.0  # Already zero from initialization
        # u4U[:, 3] = 0.0
        return u4U

    def _ADM_to_g4UU(self, gamma_rr, beta_r, alpha):
        """
        Convert ADM variables to contravariant 4-metric g^{μν}.

        For 1D radial case: g^{00} = -1/α², g^{0r} = β^r/α², g^{rr} = 1/γ_rr - (β^r)²/α²

        Returns: (M, 4, 4) array of 4-metrics
        """
        M = len(gamma_rr)
        g4UU = np.zeros((M, 4, 4), dtype=float)

        alpha2 = alpha * alpha

        # Time-time component
        g4UU[:, 0, 0] = -1.0 / alpha2

        # Time-space components (only radial)
        g4UU[:, 0, 1] = beta_r / alpha2
        g4UU[:, 1, 0] = g4UU[:, 0, 1]  # Symmetry

        # Space-space components (only radial)
        g4UU[:, 1, 1] = 1.0 / gamma_rr - (beta_r * beta_r) / alpha2

        return g4UU

    def _find_cp_cm(self, flux_dirn, g4UU, u4U, cs2):
        """
        Compute characteristic speeds c+/c- using quadratic method.

        Returns: (cminus, cplus) tuple of (M,) arrays
        """
        # Quadratic equation coefficients (vectorized)
        v02 = cs2  # (M,)
        i = flux_dirn + 1  # Spatial index (1 for radial)

        # a = (1 - cs²)(u^0)² - cs² g^{00}
        a = (1.0 - v02) * (u4U[:, 0] ** 2) - v02 * g4UU[:, 0, 0]

        # b = 2 cs² g^{i0} - 2 u^i u^0 (1 - cs²)
        b = 2.0 * v02 * g4UU[:, i, 0] - 2.0 * u4U[:, i] * u4U[:, 0] * (1.0 - v02)

        # c = (1 - cs²)(u^i)² - cs² g^{ii}
        c = (1.0 - v02) * (u4U[:, i] ** 2) - v02 * g4UU[:, i, i]

        # Solve quadratic equation (vectorized)
        detm_squared = b * b - 4.0 * a * c
        detm = np.sqrt(np.maximum(0.0, detm_squared))

        # Avoid division by zero
        safe_a = np.where(np.abs(a) < 1e-30, 1e-30, a)

        # Two roots
        cplus_tmp = 0.5 * (-b / safe_a + detm / safe_a)
        cminus_tmp = -0.5 * (b / safe_a + detm / safe_a)

        # Ensure proper ordering
        cminus = np.minimum(cplus_tmp, cminus_tmp)
        cplus = np.maximum(cplus_tmp, cminus_tmp)

        # Handle degenerate case (a ≈ 0)
        degenerate = np.abs(a) < 1e-30
        cminus = np.where(degenerate, -1.0, cminus)
        cplus = np.where(degenerate, 1.0, cplus)

        return cminus, cplus

    def _physical_flux(self, U, prim, alpha, beta_r):
        """Non-densitized physical flux vector for Valencia variables."""
        D = U[:, 0]
        Sr = U[:, 1]
        tau = U[:, 2]

        vr = prim[:, 1]
        p = prim[:, 2]

        vtil = vr 
        fD = D * vtil
        fSr = Sr * vtil + p
        ftau = (tau + p) * vtil

        return np.stack([fD, fSr, ftau], axis=1)

    def _entropy_fix(self, lam_minus, lam_plus, delta=1e-8):
        """Simple entropy fix to avoid sonic glitches."""
        # Push away from zero
        lam_minus = np.minimum(lam_minus, -np.abs(delta))
        lam_plus = np.maximum(lam_plus, np.abs(delta))

        # Ensure ordering
        need_swap = lam_minus > lam_plus
        lam_minus = np.where(need_swap, lam_plus, lam_minus)
        lam_plus = np.where(need_swap, lam_minus, lam_plus)

        return lam_minus, lam_plus
    
