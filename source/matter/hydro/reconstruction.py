# matter/hydro/reconstruction.py

import numpy as np
from source.core.spacing import NUM_GHOSTS


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
            SL[0], SR[0] = 0.0,    0.0  # S_r = 0 at r=0 (odd parity)
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
            vL[0], vR[0] = 0.0,      0.0  # v^r = 0 at r=0 (odd parity)
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


class HighOrderReconstruction:
    """
    High-order reconstruction methods: MP5, WENO5, and WENO-Z.

    Interface convention matches MinmodReconstruction for compatibility.
    Assumes input arrays already include NUM_GHOSTS ghost cells on each side.
    """

    def __init__(self, method: str = "mp5"):
        """
        Args:
            method: "mp5", "weno5", or "wenoz"
        """
        method = method.lower()
        if method not in {"mp5", "weno5", "wenoz"}:
            raise ValueError(f"Unknown high-order method: {method}")
        self.method = method
        self.name = f"reconstruction_{method}"

    def reconstruct(self, u, dx=None, x=None, boundary_type: str = "outflow"):
        """
        Reconstruct left/right states at all interfaces.

        Args:
            u: 1D array of cell-centered values (length N, already includes ghost cells)
            dx: scalar grid spacing or None
            x: 1D array of cell-center coordinates (length N) for non-uniform grids
            boundary_type: "outflow", "reflecting", or "periodic"

        Returns:
            (u_L, u_R): left/right states at interfaces, each length N+1
        """
        u = np.asarray(u, dtype=float)
        N = u.size

        if N < 2:
            # Degenerate case: just copy values like MinmodReconstruction
            if N == 1:
                uL = np.array([u[0], u[0]], dtype=float)
                uR = np.array([u[0], u[0]], dtype=float)
            else:
                uL = np.array([u[0], u[1]], dtype=float)
                uR = np.array([u[0], u[1]], dtype=float)
            return uL, uR

        # Allocate interfaces (same as MinmodReconstruction)
        u_L = np.empty(N + 1, dtype=float)
        u_R = np.empty(N + 1, dtype=float)

        # High-order reconstruction following MinmodReconstruction pattern
        # Interior reconstruction for cells i = 1..N-2 (same range as MinmodReconstruction)
        for i in range(1, N-1):
            # Check if we can use 5-point stencil [i-2, i-1, i, i+1, i+2]
            if i >= 2 and i <= N-3:
                u_stencil = u[i-2:i+3]  # length 5

                if self.method == "mp5":
                    uL_face, uR_face = self._mp5_reconstruction(u_stencil)
                elif self.method == "weno5":
                    uL_face, uR_face = self._weno5_reconstruction(u_stencil)
                else:  # wenoz
                    uL_face, uR_face = self._wenoz_reconstruction(u_stencil)

                # Following MinmodReconstruction convention:
                # Right state at i-1/2 from cell i -> interfaces 1..N-2
                u_R[i] = uR_face
                # Left state at i+1/2 from cell i -> interfaces 2..N-1
                u_L[i+1] = uL_face
            else:
                # Near boundaries where 5-point stencil doesn't fit, use piecewise constant
                # This matches the "Near-boundary interior interfaces" in MinmodReconstruction
                u_R[i] = u[i]
                u_L[i+1] = u[i]

        # Near-boundary interior interfaces (same as MinmodReconstruction)
        u_L[1] = u[0]      # left state at interface 1 from cell 0
        u_R[N-1] = u[-1]   # right state at interface N-1 from cell N-1

        # Boundary treatment for domain edges (interfaces 0 and N)
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
            SL[0], SR[0] = 0.0,    0.0  # S_r = 0 at r=0 (odd parity)
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
            vL[0], vR[0] = 0.0,      0.0  # v^r = 0 at r=0 (odd parity)
            pL[0], pR[0] = pressure[0], pressure[0]
        else:
            rL, rR = self.reconstruct(rho0,    dx=dx, x=x, boundary_type=boundary_type)
            vL, vR = self.reconstruct(vr,      dx=dx, x=x, boundary_type=boundary_type)
            pL, pR = self.reconstruct(pressure, dx=dx, x=x, boundary_type=boundary_type)

        return (rL, vL, pL), (rR, vR, pR)

    def apply_physical_limiters(self, left_tuple, right_tuple,
                                atmosphere_rho=1e-13, p_floor=1e-15, v_max=0.999,
                                gamma_rr=None):
        """Apply physical limiters to reconstructed states."""
        rho0_L, vr_L, p_L = left_tuple
        rho0_R, vr_R, p_R = right_tuple

        # Density and pressure floors
        rho0_L = np.maximum(rho0_L, atmosphere_rho)
        rho0_R = np.maximum(rho0_R, atmosphere_rho)
        p_L = np.maximum(p_L, p_floor)
        p_R = np.maximum(p_R, p_floor)

        # Velocity limiting
        if gamma_rr is not None:
            v_limit = v_max / np.sqrt(np.maximum(gamma_rr, 1.0))
            vr_L = np.clip(vr_L, -v_limit, v_limit)
            vr_R = np.clip(vr_R, -v_limit, v_limit)
        else:
            vr_L = np.clip(vr_L, -v_max, v_max)
            vr_R = np.clip(vr_R, -v_max, v_max)

        return (rho0_L, vr_L, p_L), (rho0_R, vr_R, p_R)

    def _mp5_reconstruction(self, u_stencil):
        """
        MP5 reconstruction for a single 5-point stencil.

        For stencil [u_{i-2}, u_{i-1}, u_i, u_{i+1}, u_{i+2}]:
        - uL: left state at i+1/2 (extrapolated from cell i)
        - uR: right state at i-1/2 (extrapolated from cell i)
        """
        um2, um1, u0, up1, up2 = u_stencil

        # Parameters
        alpha = 4.0
        eps = 1e-6

        # Norm for switching criterion
        vnorm = np.sqrt(um2**2 + um1**2 + u0**2 + up1**2 + up2**2) + 1e-30

        # Left state at i+1/2 (from cell i towards right)
        uL = (2*um2 - 13*um1 + 47*u0 + 27*up1 - 3*up2) / 60.0
        vmp = u0 + self._minmod_pair_scalar(up1 - u0, alpha * (u0 - um1))

        # Switching criterion
        cond = (uL - u0) * (uL - vmp) <= eps * vnorm

        if not cond:
            # Second derivatives
            djm1 = um2 - 2*um1 + u0
            dj = um1 - 2*u0 + up1
            djp1 = u0 - 2*up1 + up2

            # Fourth-order differences
            dm4jmh = self._minmod4_scalar(4*dj - djm1, 4*djm1 - dj, dj, djm1)
            dm4jph = self._minmod4_scalar(4*dj - djp1, 4*djp1 - dj, dj, djp1)

            # Additional terms for MP5
            vul = u0 + alpha * (u0 - um1)
            vav = 0.5 * (u0 + up1)
            vmd = vav - 0.5 * dm4jph
            vlc = u0 + 0.5 * (u0 - um1) + (4.0/3.0) * dm4jmh

            # Min/max bounds
            vminl = max(min(u0, up1, vmd), min(u0, vul, vlc))
            vmaxl = min(max(u0, up1, vmd), max(u0, vul, vlc))

            # Apply bounds
            uL += self._minmod_pair_scalar(vminl - uL, vmaxl - uL)

        # Right state at i-1/2 (from cell i towards left)
        # Use symmetric formulation by reversing the stencil
        up2_r, up1_r, u0_r, um1_r, um2_r = up2, up1, u0, um1, um2

        uR = (2*up2_r - 13*up1_r + 47*u0_r + 27*um1_r - 3*um2_r) / 60.0
        vmp_r = u0_r + self._minmod_pair_scalar(um1_r - u0_r, alpha * (u0_r - up1_r))

        # Switching criterion for right state
        cond_r = (uR - u0_r) * (uR - vmp_r) <= eps * vnorm

        if not cond_r:
            # Second derivatives (reversed)
            djm1_r = up2_r - 2*up1_r + u0_r
            dj_r = up1_r - 2*u0_r + um1_r
            djp1_r = u0_r - 2*um1_r + um2_r

            # Fourth-order differences
            dm4jmh_r = self._minmod4_scalar(4*dj_r - djm1_r, 4*djm1_r - dj_r, dj_r, djm1_r)
            dm4jph_r = self._minmod4_scalar(4*dj_r - djp1_r, 4*djp1_r - dj_r, dj_r, djp1_r)

            # Additional terms for right state
            vul_r = u0_r + alpha * (u0_r - up1_r)
            vav_r = 0.5 * (u0_r + um1_r)
            vmd_r = vav_r - 0.5 * dm4jph_r
            vlc_r = u0_r + 0.5 * (u0_r - up1_r) + (4.0/3.0) * dm4jmh_r

            # Min/max bounds
            vminr = max(min(u0_r, um1_r, vmd_r), min(u0_r, vul_r, vlc_r))
            vmaxr = min(max(u0_r, um1_r, vmd_r), max(u0_r, vul_r, vlc_r))

            # Apply bounds
            uR += self._minmod_pair_scalar(vminr - uR, vmaxr - uR)

        return uL, uR

    def _weno5_reconstruction(self, u_stencil):
        """WENO5 reconstruction for a single 5-point stencil."""
        um2, um1, u0, up1, up2 = u_stencil

        # Left state at i+1/2 (standard WENO5)
        uL = self._weno5_face(um2, um1, u0, up1, up2)

        # Right state at i-1/2 (reverse stencil)
        uR = self._weno5_face(up2, up1, u0, um1, um2)

        return uL, uR

    def _weno5_face(self, um2, um1, u0, up1, up2):
        """Single WENO5 face reconstruction."""
        # Smoothness indicators
        beta0 = (13.0/12.0) * (um2 - 2*um1 + u0)**2 + 0.25 * (um2 - 4*um1 + 3*u0)**2
        beta1 = (13.0/12.0) * (um1 - 2*u0 + up1)**2 + 0.25 * (um1 - up1)**2
        beta2 = (13.0/12.0) * (u0 - 2*up1 + up2)**2 + 0.25 * (3*u0 - 4*up1 + up2)**2

        # Ideal weights
        d0, d1, d2 = 0.1, 0.6, 0.3
        eps = 1e-6

        # Alpha weights
        alpha0 = d0 / (eps + beta0)**2
        alpha1 = d1 / (eps + beta1)**2
        alpha2 = d2 / (eps + beta2)**2

        # Normalized weights
        sum_alpha = alpha0 + alpha1 + alpha2
        w0 = alpha0 / sum_alpha
        w1 = alpha1 / sum_alpha
        w2 = alpha2 / sum_alpha

        # Candidate polynomials
        p0 = (2*um2 - 7*um1 + 11*u0) / 6.0
        p1 = (-um1 + 5*u0 + 2*up1) / 6.0
        p2 = (2*u0 + 5*up1 - up2) / 6.0

        return w0*p0 + w1*p1 + w2*p2

    def _wenoz_reconstruction(self, u_stencil):
        """WENO-Z reconstruction for a single 5-point stencil."""
        um2, um1, u0, up1, up2 = u_stencil

        # Left state at i+1/2 (standard WENO-Z)
        uL = self._wenoz_face(um2, um1, u0, up1, up2)

        # Right state at i-1/2 (reverse stencil)
        uR = self._wenoz_face(up2, up1, u0, um1, um2)

        return uL, uR

    def _wenoz_face(self, um2, um1, u0, up1, up2):
        """Single WENO-Z face reconstruction."""
        # Smoothness indicators
        beta0 = (13.0/12.0) * (um2 - 2*um1 + u0)**2 + 0.25 * (um2 - 4*um1 + 3*u0)**2
        beta1 = (13.0/12.0) * (um1 - 2*u0 + up1)**2 + 0.25 * (um1 - up1)**2
        beta2 = (13.0/12.0) * (u0 - 2*up1 + up2)**2 + 0.25 * (3*u0 - 4*up1 + up2)**2

        # Tau indicator (key improvement of WENO-Z)
        tau5 = abs(beta0 - beta2)

        # Ideal weights
        d0, d1, d2 = 0.1, 0.6, 0.3
        eps = 1e-6

        # Modified alpha weights with tau
        alpha0 = d0 * (1.0 + (tau5 / (beta0 + eps))**2)
        alpha1 = d1 * (1.0 + (tau5 / (beta1 + eps))**2)
        alpha2 = d2 * (1.0 + (tau5 / (beta2 + eps))**2)

        # Normalized weights
        sum_alpha = alpha0 + alpha1 + alpha2
        w0 = alpha0 / sum_alpha
        w1 = alpha1 / sum_alpha
        w2 = alpha2 / sum_alpha

        # Candidate polynomials
        p0 = (2*um2 - 7*um1 + 11*u0) / 6.0
        p1 = (-um1 + 5*u0 + 2*up1) / 6.0
        p2 = (2*u0 + 5*up1 - up2) / 6.0

        return w0*p0 + w1*p1 + w2*p2

    @staticmethod
    def _minmod_pair_scalar(a, b):
        """Two-argument minmod function for scalars."""
        if a * b <= 0:
            return 0.0
        return a if abs(a) < abs(b) else b

    @staticmethod
    def _minmod4_scalar(w, x, y, z):
        """Four-argument minmod function for scalars."""
        s = 0.125 * (np.sign(w) + np.sign(x)) * abs(
            (np.sign(w) + np.sign(y)) * (np.sign(w) + np.sign(z)))
        return s * min(abs(w), abs(x), abs(y), abs(z))

    @staticmethod
    def _fill_boundaries(u, u_L, u_R, boundary_type: str):
        """Fill boundary interfaces for high-order methods."""
        N = u.size

        if boundary_type == "outflow":
            # Zeroth-order extrapolation
            u_L[0] = u[0]
            u_R[0] = u[0]
            u_L[N] = u[-1]
            u_R[N] = u[-1]
        elif boundary_type == "reflecting":
            # Even reflection by default
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


def create_reconstruction(method: str = "minmod"):
    """
    Factory function to create reconstruction objects.

    Args:
        method: Reconstruction method name
               Low-order: "minmod", "mc", "superbee"
               High-order: "mp5", "weno5", "wenoz"

    Returns:
        Reconstruction object with appropriate interface
    """
    method = method.lower()

    # Low-order TVD methods
    if method in {"minmod", "mc", "superbee"}:
        return MinmodReconstruction(limiter_type=method)

    # High-order methods
    elif method in {"mp5", "weno5", "wenoz"}:
        return HighOrderReconstruction(method=method)

    else:
        raise ValueError(f"Unknown reconstruction method: {method}")


# Convenience aliases for backward compatibility
def get_reconstruction_minmod():
    """Get minmod reconstruction (backward compatibility)."""
    return MinmodReconstruction(limiter_type="minmod")

def get_reconstruction_mc():
    """Get MC reconstruction (backward compatibility)."""
    return MinmodReconstruction(limiter_type="mc")

def get_reconstruction_superbee():
    """Get superbee reconstruction (backward compatibility)."""
    return MinmodReconstruction(limiter_type="superbee")



