# matter/hydro/reconstruction.py

import numpy as np


class MinmodReconstruction:
    """
    High-resolution piecewise-linear reconstruction with TVD limiters
    (minmod / MC / superbee). Works on *non-uniform* grids.

    Interface convention (cell-centered → interfaces):
      - Input u has length N (cell centers i = 0..N-1)
      - Returns (u_L, u_R), each of length N+1 (interfaces i+1/2 = 0..N)
        * u_L[k]  := left state at interface k  (coming from cell k-1)
        * u_R[k]  := right state at interface k (coming from cell k)

      With our convention we populate interior as:
        for i in 1..N-2:
          slope_i = limiter( (u[i]-u[i-1])/dxL, (u[i+1]-u[i])/dxR )
          u_R[i]   = u[i] - 0.5*slope_i*dxL   # state at i-1/2 from cell i
          u_L[i+1] = u[i] + 0.5*slope_i*dxR   # state at i+1/2 from cell i

      Boundaries are filled by simple extrapolation (“outflow”) unless
      boundary_type dictates otherwise.

    Supports non-uniform coordinates via x array. If dx is a scalar,
    uniform spacing is assumed. If x is provided, dx is derived from x.
    """

    def __init__(self, limiter_type: str = "minmod"):
        """
        Args:
          limiter_type: "minmod", "mc" (monotized central), or "superbee"
        """
        lim = limiter_type.lower()
        if lim not in {"minmod", "mc", "superbee"}:
            raise ValueError(f"Unknown limiter type: {lim}")
        self.limiter_type = lim
        self.name = f"reconstruction_{lim}"

    # -------------------------
    # Public API
    # -------------------------

    def reconstruct(self, u, dx=None, x=None, boundary_type: str = "outflow"):
        """
        Reconstruct left/right states at all interfaces.

        Args:
          u: 1D array of cell-centered values (length N).
          dx: scalar grid spacing (uniform). Ignored if x is provided.
          x:  1D array of cell-center coordinates (length N) for non-uniform grids.
          boundary_type: "outflow", "reflecting", or "periodic".

        Returns:
          (u_L, u_R): left/right states at interfaces, each length N+1.
        """
        u = np.asarray(u, dtype=float)
        N = u.size
        if N < 2:
            # Degenerate case: just copy values
            if N == 1:
                uL = np.array([u[0], u[0]], dtype=float)
                uR = np.array([u[0], u[0]], dtype=float)
            else:
                uL = np.array([u[0], u[1]], dtype=float)
                uR = np.array([u[0], u[1]], dtype=float)
            return uL, uR

        dxL, dxR = self._one_sided_deltas(N, dx, x)

        # Allocate interfaces
        u_L = np.empty(N + 1, dtype=float)
        u_R = np.empty(N + 1, dtype=float)

        # Fill interior using limited slopes (i = 1..N-2)
        a = (u[1:-1] - u[:-2]) / dxL[1:-1]   # backward slope at i
        b = (u[2:]   - u[1:-1]) / dxR[1:-1]  # forward  slope at i
        s = self._apply_limiter(a, b)        # limited slope s_i

        # Right state at i-1/2 from cell i (i = 1..N-2) -> interfaces 1..N-2
        u_R[1:N-1] = u[1:-1] - 0.5 * s * dxL[1:-1]
        # Left state at i+1/2 from cell i (i = 1..N-2)  -> interfaces 2..N-1
        u_L[2:N]   = u[1:-1] + 0.5 * s * dxR[1:-1]

        # Near-boundary interior interfaces (ensure they are initialized)
        # Piecewise-constant at the first interior interface and last interior interface
        u_L[1]     = u[0]      # left state at interface 1 from cell 0
        u_R[N-1]   = u[-1]     # right state at interface N-1 from cell N-1

        # Boundary treatment for the domain edges (interfaces 0 and N)
        self._fill_boundaries(u, u_L, u_R, boundary_type)

        return u_L, u_R

    def reconstruct_conservative_variables(self, D, Sr, tau, dx=None, x=None, boundary_type="outflow"):
        """
        Reconstruct (D, S_r, tau) to interfaces.

        Returns:
          left:  (D_L,  Sr_L,  tau_L)
          right: (D_R,  Sr_R,  tau_R)
        """
        if boundary_type == "reflecting":
            # Build with outflow, then enforce reflecting parity at the INNER boundary (left).
            DL, DR = self.reconstruct(D, dx=dx, x=x, boundary_type="outflow")
            SL, SR = self.reconstruct(Sr, dx=dx, x=x, boundary_type="outflow")
            TL, TR = self.reconstruct(tau, dx=dx, x=x, boundary_type="outflow")
            # Parities at r≈0: D, tau even; Sr odd
            DL[0], DR[0] = D[0],  D[0]
            SL[0], SR[0] = -Sr[0], Sr[0]
            TL[0], TR[0] = tau[0], tau[0]
        else:
            DL, DR = self.reconstruct(D,   dx=dx, x=x, boundary_type=boundary_type)
            SL, SR = self.reconstruct(Sr,  dx=dx, x=x, boundary_type=boundary_type)
            TL, TR = self.reconstruct(tau, dx=dx, x=x, boundary_type=boundary_type)

        return (DL, SL, TL), (DR, SR, TR)

    def reconstruct_primitive_variables(self, rho0, vr, pressure, dx=None, x=None, boundary_type="outflow"):
        """
        Reconstruct (rho0, v^r, p) to interfaces.

        Returns:
          left:  (rho0_L, vr_L, p_L)
          right: (rho0_R, vr_R, p_R)
        """
        if boundary_type == "reflecting":
            # Build with outflow, then enforce reflecting parity at the INNER boundary (left).
            rL, rR = self.reconstruct(rho0,    dx=dx, x=x, boundary_type="outflow")
            vL, vR = self.reconstruct(vr,      dx=dx, x=x, boundary_type="outflow")
            pL, pR = self.reconstruct(pressure, dx=dx, x=x, boundary_type="outflow")
            # Parities at r≈0: rho0, p even; v^r odd
            rL[0], rR[0] = rho0[0],  rho0[0]
            vL[0], vR[0] = -vr[0],   vr[0]
            pL[0], pR[0] = pressure[0], pressure[0]
        else:
            rL, rR = self.reconstruct(rho0,    dx=dx, x=x, boundary_type=boundary_type)
            vL, vR = self.reconstruct(vr,      dx=dx, x=x, boundary_type=boundary_type)
            pL, pR = self.reconstruct(pressure, dx=dx, x=x, boundary_type=boundary_type)

        return (rL, vL, pL), (rR, vR, pR)

    def apply_physical_limiters(self, left_tuple, right_tuple,
                                atmosphere_rho=1e-13, p_floor=1e-15, v_max=0.999,
                                gamma_rr=None):
        rho0_L, vr_L, p_L = left_tuple
        rho0_R, vr_R, p_R = right_tuple

        # Límites de densidad y presión
        rho0_L = np.maximum(rho0_L, atmosphere_rho)
        rho0_R = np.maximum(rho0_R, atmosphere_rho)
        p_L = np.maximum(p_L, p_floor)
        p_R = np.maximum(p_R, p_floor)

        # Límite de velocidad considerando la métrica
        if gamma_rr is not None:
            # v² = γ_rr (v^r)² < v_max²  -> |v^r| < v_max / sqrt(γ_rr)
            v_limit = v_max / np.sqrt(np.maximum(gamma_rr, 1.0))
            vr_L = np.clip(vr_L, -v_limit, v_limit)
            vr_R = np.clip(vr_R, -v_limit, v_limit)
        else:
            vr_L = np.clip(vr_L, -v_max, v_max)
            vr_R = np.clip(vr_R, -v_max, v_max)

        return (rho0_L, vr_L, p_L), (rho0_R, vr_R, p_R)

    def check_monotonicity(self, u, u_L, u_R):
        """
        Diagnostic: count monotonicity violations (no new extrema).
        """
        u = np.asarray(u, dtype=float)
        N = u.size
        vio = 0
        for i in range(1, N - 1):
            umin = np.min([u[i - 1], u[i], u[i + 1]])
            umax = np.max([u[i - 1], u[i], u[i + 1]])
            # interface i  : right from cell i
            if u_R[i] < umin or u_R[i] > umax:
                vio += 1
            # interface i+1: left  from cell i
            if u_L[i + 1] < umin or u_L[i + 1] > umax:
                vio += 1
        return vio

    def compute_total_variation(self, u):
        """
        TV(u) = sum_i |u[i+1]-u[i]|
        """
        u = np.asarray(u, dtype=float)
        if u.size < 2:
            return 0.0
        return np.sum(np.abs(np.diff(u)))

    def reconstruct_with_boundaries(self, u, dx=None, x=None, boundary_type="outflow"):
        """
        Thin wrapper kept for API compatibility. Same as reconstruct().
        """
        return self.reconstruct(u, dx=dx, x=x, boundary_type=boundary_type)

    # -------------------------
    # Limiters (vectorized)
    # -------------------------

    def _apply_limiter(self, a, b):
        """Dispatch to the chosen limiter (vectorized over arrays)."""
        if self.limiter_type == "minmod":
            return self._minmod(a, b)
        elif self.limiter_type == "mc":
            # MC = minmod( (a+b)/2, 2a, 2b ) — do a 3-way minmod
            c1 = 0.5 * (a + b)
            c2 = 2.0 * a
            c3 = 2.0 * b
            return self._minmod3(c1, c2, c3)
        else:  # superbee
            # superbee = maxmod( minmod(2a, b), minmod(a, 2b) )
            m1 = self._minmod(2.0 * a, b)
            m2 = self._minmod(a, 2.0 * b)
            # maxmod: pick the one with larger magnitude, preserving sign if same
            pick_m1 = np.abs(m1) >= np.abs(m2)
            out = np.where(pick_m1, m1, m2)
            return out

    @staticmethod
    def _minmod(a, b):
        """Two-argument minmod, element-wise."""
        a = np.asarray(a)
        b = np.asarray(b)
        same_sign = (a * b) > 0.0
        out = np.zeros_like(a, dtype=float)
        # Where signs are same, take the one with smaller magnitude
        mag_a_le = np.abs(a) <= np.abs(b)
        out = np.where(same_sign & mag_a_le, a, out)
        out = np.where(same_sign & (~mag_a_le), b, out)
        return out

    @staticmethod
    def _minmod3(a, b, c):
        """Three-argument minmod, element-wise."""
        a = np.asarray(a)
        b = np.asarray(b)
        c = np.asarray(c)
        same_sign = (a * b > 0.0) & (a * c > 0.0)
        s = np.sign(a)
        m = np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c)))
        return np.where(same_sign, s * m, 0.0)

    # -------------------------
    # Internals
    # -------------------------

    @staticmethod
    def _one_sided_deltas(N, dx, x):
        """
        Return per-cell one-sided spacings dxL[i] = x[i] - x[i-1],
                                       dxR[i] = x[i+1] - x[i].
        For uniform grids (scalar dx), uses that value. Ends are mirrored.
        """
        if x is not None:
            x = np.asarray(x, dtype=float)
            if x.size != N:
                raise ValueError("x must have same length as u")
            dxL = np.empty(N, dtype=float)
            dxR = np.empty(N, dtype=float)
            dxL[0]  = x[1]   - x[0]
            dxR[-1] = x[-1]  - x[-2]
            dxL[1:] = x[1:]  - x[:-1]
            dxR[:-1]= x[1:]  - x[:-1]
        else:
            if dx is None:
                raise ValueError("Either dx (scalar) or x (array) must be provided")
            d = float(dx)
            dxL = np.full(N, d, dtype=float)
            dxR = np.full(N, d, dtype=float)
        # Avoid zeros
        dxL = np.where(dxL == 0.0, 1e-30, dxL)
        dxR = np.where(dxR == 0.0, 1e-30, dxR)
        return dxL, dxR

    @staticmethod
    def _fill_boundaries(u, u_L, u_R, boundary_type: str):
        """
        Populate the first/last interfaces according to boundary_type.
        Interface indices: 0 .. N

        Notes:
          • "reflecting" here copies even parity at both ends. For odd-parity fields
            (e.g., v^r or S_r at r≈0) the sign flip is enforced in the specialized
            reconstruct_* methods where the variable identity is known.
        """
        N = u.size

        if boundary_type == "outflow":
            # Zeroth-order extrapolation on both ends
            u_L[0] = u[0]
            u_R[0] = u[0]
            u_L[N] = u[-1]
            u_R[N] = u[-1]

        elif boundary_type == "reflecting":
            # Even reflection by default (odd handled per-variable upstream)
            u_L[0] = u[0]
            u_R[0] = u[0]
            u_L[N] = u[-1]
            u_R[N] = u[-1]

        elif boundary_type == "periodic":
            # Wrap-around
            u_L[0] = u[-1]
            u_R[0] = u[0]
            u_L[N] = u[-1]
            u_R[N] = u[0]

        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")


class MUSCLReconstruction:
    """
    MUSCL (κ-scheme) wrapper. For now, it delegates to MinmodReconstruction
    with MC limiter (κ≈1/3 behavior).
    """

    def __init__(self, kappa: float = 1.0 / 3.0):
        self.kappa = float(kappa)
        # Use MC limiter by default to emulate κ≈1/3 accuracy/robustness
        self._impl = MinmodReconstruction(limiter_type="mc")
        self.name = f"muscl_kappa_{self.kappa}"

    def reconstruct(self, u, dx=None, x=None, boundary_type="outflow"):
        return self._impl.reconstruct(u, dx=dx, x=x, boundary_type=boundary_type)

    def reconstruct_conservative_variables(self, D, Sr, tau, dx=None, x=None, boundary_type="outflow"):
        return self._impl.reconstruct_conservative_variables(D, Sr, tau, dx=dx, x=x, boundary_type=boundary_type)

    def reconstruct_primitive_variables(self, rho0, vr, pressure, dx=None, x=None, boundary_type="outflow"):
        return self._impl.reconstruct_primitive_variables(rho0, vr, pressure, dx=dx, x=x, boundary_type=boundary_type)

    def apply_physical_limiters(self, left_tuple, right_tuple,
                                atmosphere_rho=1e-13, p_floor=1e-15, v_max=0.999):
        return self._impl.apply_physical_limiters(left_tuple, right_tuple,
                                                  atmosphere_rho=atmosphere_rho, p_floor=p_floor, v_max=v_max)

    def check_monotonicity(self, u, u_L, u_R):
        return self._impl.check_monotonicity(u, u_L, u_R)

    def compute_total_variation(self, u):
        return self._impl.compute_total_variation(u)


