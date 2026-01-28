# cons2prim.py

import numpy as np
from numba import jit, prange
from .atmosphere import FloorApplicator
from .geometry import GeometryState


# ============================================================================
# NUMBA-OPTIMIZED NEWTON-RAPHSON KERNEL FOR IDEAL GAS EOS
# ============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def _newton_kernel_ideal_gas(D, Sr, tau, gamma_rr, p_init,
                              eos_gamma, p_floor, W_max, tol, max_iter):
    """
    NUMBA-optimized Newton-Raphson solver for ideal gas EOS.

    Solves for pressure p such that f(p) = ρ₀ h W² - Q = 0.

    Returns:
        (rho0, vr, p, eps, W, h, converged) - all scalars
    """
    c_ideal = eos_gamma / (eos_gamma - 1.0)
    gm1 = eos_gamma - 1.0

    p = max(p_init, p_floor)
    gamma_rr = max(gamma_rr, 1e-30)

    for iteration in range(max_iter):
        # Evaluate pressure residual
        Q = tau + D + p
        vr =  Sr / (Q * gamma_rr)
        v2 = gamma_rr * vr * vr

        W = 1.0 / (1.0 - v2)**0.5

        # Check Lorentz factor bounds
        if W > W_max or W < 1.0:
            p = max(p_floor, p * 1.5 + 1e-14)
            continue

        rho0 = D / max(W, 1e-30)
        if rho0 <= 0.0:
            p = max(p_floor, p * 1.5 + 1e-14)
            continue

        eps = p / (rho0 * gm1)
        h = 1.0 + eps + p / rho0

        # Residual
        f = rho0 * h * W * W - Q

        # Check convergence
        tol_val = tol * max(1.0, abs(p))
        if abs(f) <= tol_val:
            return rho0, vr, p, eps, W, h, True

        # Analytic derivative for ideal gas
        E = tau + D
        Q2 = (E + p)**2
        Sr2 = Sr * Sr
        v2 = Sr2 / max(Q2 * gamma_rr, 1e-30)
        W = 1.0 / max((1.0 - v2), 1e-16) ** 0.5
        W2 = W * W
        W3 = W2 * W
        Q3 = (E + p)**3
        dWdp = - Sr2 * W3 / max(Q3 * gamma_rr, 1e-30)
        df = c_ideal * W2 + (D + 2.0 * c_ideal * p * W) * dWdp - 1.0

        # Avoid small derivatives
        if abs(df) < 1e-15:
            p = max(p_floor, p * 1.5 + 1e-12)
            continue

        # Newton update with damping
        p_new = p - f / df
        if p_new <= 0.0 or p_new != p_new:  # check for NaN
            p_new = max(p_floor, 0.5 * p)

        p =p_new

    # Did not converge - return best guess with converged=False
    Q = tau + D + p
    vr = Sr / (Q * gamma_rr)
    v2 = gamma_rr * vr * vr
    if v2 >= 1.0:
        v2 = 1.0 - 1e-16
    W = 1.0 / (1.0 - v2) ** 0.5
    rho0 = D / max(W, 1e-30)
    eps = p / max(rho0 * gm1, 1e-30)
    h = 1.0 + eps + p / max(rho0, 1e-30)

    return rho0, vr, p, eps, W, h, False


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def _solve_newton_batch_ideal_gas(D, Sr, tau, gamma_rr, alpha, p_guess,
                                   eos_gamma, p_floor, W_max, tol, max_iter):
    """
    NUMBA-optimized batch Newton-Raphson for ideal gas EOS.

    Parallelized over all points - each point solved independently.
    """
    N = len(D)

    # Output arrays
    rho0 = np.zeros(N)
    vr = np.zeros(N)
    p_out = np.zeros(N)
    eps = np.zeros(N)
    W = np.ones(N)
    h = np.ones(N)
    converged = np.zeros(N, dtype=np.bool_)

    # Parallel loop over all points
    for i in prange(N):
        rho0[i], vr[i], p_out[i], eps[i], W[i], h[i], converged[i] = \
            _newton_kernel_ideal_gas(
                D[i], Sr[i], tau[i], gamma_rr[i], p_guess[i],
                eos_gamma, p_floor, W_max, tol, max_iter
            )

    return rho0, vr, p_out, eps, W, h, converged


# ============================================================================
# NUMBA-OPTIMIZED NEWTON-RAPHSON KERNEL FOR POLYTROPIC EOS
# ============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def _newton_kernel_polytropic(D, Sr, tau, gamma_rr, rho_init,
                               eos_K, eos_gamma, rho_floor, W_max, tol, max_iter):
    """
    NUMBA-optimized Newton-Raphson solver for polytropic EOS.

    For barotropic EOS P = K ρ₀^Γ, solve for ρ₀ such that D = ρ₀ W.

    Returns:
        (rho0, vr, p, eps, W, h, converged) - all scalars
    """
    gm1 = eos_gamma - 1.0
    K = eos_K

    rho = max(rho_init, rho_floor)
    gamma_rr = max(gamma_rr, 1e-30)

    for iteration in range(max_iter):
        # Pressure and enthalpy from rho (barotropic: depends only on rho)
        p = K * rho**eos_gamma
        h = 1.0 + eos_gamma * K * rho**gm1 / gm1

        # Compute Q = tau + D + p, then velocity
        Q = tau + D + p
        vr =  Sr / (Q * gamma_rr)
        v2 = gamma_rr * vr * vr

        # Clamp v2 for safety
        if v2 >= 1.0:
            v2 = 1.0 - 1e-16
        if v2 < 0.0:
            v2 = 0.0

        W = 1.0 / (1.0 - v2)**0.5

        # Check Lorentz factor bounds
        if W > W_max or W < 1.0:
            rho = max(rho_floor, rho * 0.5)
            continue

        # Residual: f = D - rho * W
        f = D - rho * W

        # Check convergence
        tol_val = tol * max(1.0, D)
        if abs(f) <= tol_val:
            eps = K * rho**gm1 / gm1
            return rho, vr, p, eps, W, h, True

        # Numerical derivative df/drho
        drho = 1e-8 * max(rho, 1e-10)
        rho2 = rho + drho
        p2 = K * rho2**eos_gamma
        Q2 = tau + D + p2
        vr2 = Sr / (Q2 * gamma_rr)
        v2_2 = gamma_rr * vr2 * vr2
        if v2_2 >= 1.0:
            v2_2 = 1.0 - 1e-16
        if v2_2 < 0.0:
            v2_2 = 0.0
        W2 = 1.0 / (1.0 - v2_2)**0.5
        f2 = D - rho2 * W2

        df = (f2 - f) / drho

        # Avoid small derivatives
        if abs(df) < 1e-15:
            rho = max(rho_floor, rho * 1.5 + 1e-12)
            continue

        # Newton update with damping
        rho_new = rho - f / df
        if rho_new <= 0.0 or rho_new != rho_new:  # check for NaN
            rho_new = max(rho_floor, 0.5 * rho)

        rho = 0.5 * rho + 0.5 * rho_new

    # Did not converge - return best guess with converged=False
    p = K * rho**eos_gamma
    h = 1.0 + eos_gamma * K * rho**gm1 / gm1
    Q = tau + D + p
    vr = Sr / (Q * gamma_rr)
    v2 = gamma_rr * vr * vr
    if v2 >= 1.0:
        v2 = 1.0 - 1e-16
    W = 1.0 / (1.0 - v2)**0.5
    eps = K * rho**gm1 / gm1

    return rho, vr, p, eps, W, h, False


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def _solve_newton_batch_polytropic(D, Sr, tau, gamma_rr, alpha, rho_guess,
                                    eos_K, eos_gamma, rho_floor, W_max, tol, max_iter):
    """
    NUMBA-optimized batch Newton-Raphson for polytropic EOS.

    Parallelized over all points - each point solved independently.
    """
    N = len(D)

    # Output arrays
    rho0 = np.zeros(N)
    vr = np.zeros(N)
    p_out = np.zeros(N)
    eps = np.zeros(N)
    W = np.ones(N)
    h = np.ones(N)
    converged = np.zeros(N, dtype=np.bool_)

    # Parallel loop over all points
    for i in prange(N):
        rho0[i], vr[i], p_out[i], eps[i], W[i], h[i], converged[i] = \
            _newton_kernel_polytropic(
                D[i], Sr[i], tau[i], gamma_rr[i], rho_guess[i],
                eos_K, eos_gamma, rho_floor, W_max, tol, max_iter
            )

    return rho0, vr, p_out, eps, W, h, converged


# ============================================================================
# KASTAUN et al. (2021) SOLVER - ROBUST CONS2PRIM WITH BRACKETING
# Based on: Kastaun et al., Phys. Rev. D 103, 023018 (2021)
# Simplified for pure hydrodynamics (no magnetic field)
# ============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def _kastaun_residual_ideal_gas(mu, D, r2, q, eos_gamma, v2_max):
    """
    Compute the Kastaun residual f(μ) = μ - μ̂ for ideal gas EOS.

    Following Kastaun et al. (2021):
    1. v̂² = min(μ²r², v²_max)
    2. Ŵ = 1/√(1 - v̂²)
    3. ρ̂ = D/Ŵ
    4. e/D = q - μr² + 1  (specific total energy ratio)
    5. For ideal gas: ε = (e/ρ - 1), P = ρε(γ-1), h = 1 + εγ
    6. ν̂ = ĥ/Ŵ
    7. μ̂ = 1/(ν̂ + μr²)

    Args:
        mu: Current guess for μ
        D: Relativistic density D = ρ₀W
        r2: r² = S²/D² = (Sr)²/(D² γᵣᵣ)
        q: q = τ/D
        eos_gamma: Adiabatic index γ
        v2_max: Maximum allowed v²

    Returns:
        f: Residual value
        rho: Rest-mass density
        P: Pressure
        h: Specific enthalpy
        valid: Whether the state is physically valid
    """
    gm1 = eos_gamma - 1.0

    # Step 1: Estimate velocity squared (with limit)
    v2 = mu * mu * r2
    if v2 > v2_max:
        v2 = v2_max

    # Step 2: Lorentz factor
    if v2 >= 1.0:
        return 0.0, 0.0, 0.0, 1.0, False

    W = 1.0 / (1.0 - v2) ** 0.5

    # Step 3: Rest-mass density
    rho = D / W
    if rho <= 0.0:
        return 0.0, 0.0, 0.0, 1.0, False

    # Step 4: Total energy ratio e/D = q - μr² + 1
    # For hydro: e = D*(q - μr² + 1) = τ + D - D*μr²
    e_over_D = q - mu * r2 + 1.0

    # Step 5: For ideal gas EOS
    # e = ρ(1 + ε), so ε = e/ρ - 1 = (D*e_over_D)/(D/W) - 1 = W*e_over_D - 1
    eps = W * e_over_D - 1.0

    if eps < 0.0:
        # Negative internal energy - unphysical, but try to continue
        eps = 0.0

    # Pressure: P = ρ ε (γ-1)
    P = rho * eps * gm1

    # Enthalpy: h = 1 + ε + P/ρ = 1 + ε + ε(γ-1) = 1 + εγ
    h = 1.0 + eps * eos_gamma

    # Validity checks
    if h <= 1.0:
        h = 1.0 + 1e-15

    # Step 6: ν = h/W
    nu = h / W

    # Step 7: μ̂ = 1/(ν + μr²)
    denom = nu + mu * r2
    if denom <= 1e-30:
        return 0.0, rho, P, h, False

    mu_hat = 1.0 / denom

    # Residual: f(μ) = μ - μ̂
    f = mu - mu_hat

    return f, rho, P, h, True


@jit(nopython=True, cache=True, fastmath=True)
def _false_position_anderson_bjorck(a, b, fa, fb, tol, max_iter,
                                     D, r2, q, eos_gamma, v2_max):
    """
    False Position method with Anderson-Bjorck acceleration.

    Guaranteed to converge if a valid bracket [a,b] exists with fa*fb < 0.
    Uses Illinois-style acceleration to avoid slow convergence.

    Returns:
        mu: Solution
        converged: Whether convergence was achieved
        iterations: Number of iterations used
    """
    # Track which side was updated last (for acceleration)
    side = 0  # 0: initial, 1: left updated, -1: right updated

    x = a
    xold = a

    for iteration in range(max_iter):
        xold = x

        # False position interpolation
        denom = fb - fa
        if abs(denom) < 1e-30:
            # Degenerate case: use bisection
            x = 0.5 * (a + b)
        else:
            x = (a * fb - b * fa) / denom

        # Ensure x is within bounds (safety)
        if x <= a or x >= b:
            x = 0.5 * (a + b)

        # Evaluate at new point
        fx, rho, P, h, valid = _kastaun_residual_ideal_gas(x, D, r2, q, eos_gamma, v2_max)

        if not valid:
            # Fallback to bisection
            x = 0.5 * (a + b)
            fx, rho, P, h, valid = _kastaun_residual_ideal_gas(x, D, r2, q, eos_gamma, v2_max)
            if not valid:
                return x, False, iteration

        # Check convergence
        if abs(x - xold) <= tol * max(abs(x), 1e-15):
            return x, True, iteration

        if abs(fx) <= tol:
            return x, True, iteration

        # Update bracket
        if fa * fx < 0:
            # Root is in [a, x]
            if side == -1:
                # Same side twice: apply Anderson-Bjorck acceleration
                m = 1.0 - fx / fb
                if m > 0:
                    fa = fa * m
                else:
                    fa = fa * 0.5
            b = x
            fb = fx
            side = -1
        else:
            # Root is in [x, b]
            if side == 1:
                # Same side twice: apply Anderson-Bjorck acceleration
                m = 1.0 - fx / fa
                if m > 0:
                    fb = fb * m
                else:
                    fb = fb * 0.5
            a = x
            fa = fx
            side = 1

    return x, False, max_iter


@jit(nopython=True, cache=True, fastmath=True)
def _kastaun_kernel_ideal_gas(D, Sr, tau, gamma_rr,
                               eos_gamma, rho_floor, p_floor,
                               v_max, W_max, tol, max_iter):
    """
    Kastaun et al. (2021) cons2prim kernel for ideal gas EOS.

    Solves for μ = 1/(ν + μr²) where ν = h/W using bracketed root-finding.
    Guaranteed convergence within physical bounds.

    Args:
        D, Sr, tau: Conservative variables (physical, not densitized)
        gamma_rr: Radial metric component γᵣᵣ
        eos_gamma: Adiabatic index
        rho_floor, p_floor: Atmosphere values
        v_max, W_max: Velocity and Lorentz factor limits
        tol, max_iter: Convergence parameters

    Returns:
        (rho0, vr, p, eps, W, h, converged)
    """
    gm1 = eos_gamma - 1.0
    v2_max = v_max * v_max

    # ===== PRE-VALIDATION =====
    if D <= 0.0 or D != D:  # D <= 0 or NaN
        return rho_floor, 0.0, p_floor, p_floor/(rho_floor*gm1), 1.0, 1.0 + p_floor/rho_floor*eos_gamma/gm1, False

    if tau != tau or Sr != Sr:  # NaN check
        return rho_floor, 0.0, p_floor, p_floor/(rho_floor*gm1), 1.0, 1.0 + p_floor/rho_floor*eos_gamma/gm1, False

    gamma_rr = max(gamma_rr, 1e-30)

    # ===== COMPUTE AUXILIARY QUANTITIES =====
    q = tau / D
    r2 = (Sr * Sr) / (D * D * gamma_rr)  # r² = S²/D²

    # ===== ESTABLISH BOUNDS FOR μ =====
    # μ_min = 0 corresponds to v = 0, W = 1
    # μ_max = 1/h_min where h_min is minimum enthalpy
    # For ideal gas with cold limit: h_min → 1

    mu_min = 0.0
    h_min = 1.0 + 1e-10  # Slightly above 1 for numerical safety
    mu_max = 1.0 / h_min  # ≈ 1

    # Tighten upper bound based on maximum velocity
    # v² = μ²r², so μ_max = v_max / sqrt(r²) if r² > 0
    if r2 > 1e-30:
        mu_v_limit = v_max / (r2 ** 0.5)
        if mu_v_limit < mu_max:
            mu_max = mu_v_limit * 0.999  # Slightly inside limit

    # ===== EVALUATE RESIDUAL AT BOUNDS =====
    f_min, rho_min_eval, P_min_eval, h_min_eval, valid_min = \
        _kastaun_residual_ideal_gas(mu_min + 1e-15, D, r2, q, eos_gamma, v2_max)

    f_max, rho_max_eval, P_max_eval, h_max_eval, valid_max = \
        _kastaun_residual_ideal_gas(mu_max, D, r2, q, eos_gamma, v2_max)

    # ===== CHECK FOR VALID BRACKET =====
    if not valid_min or not valid_max:
        # State is outside physical range
        return rho_floor, 0.0, p_floor, p_floor/(rho_floor*gm1), 1.0, 1.0 + p_floor/rho_floor*eos_gamma/gm1, False

    if f_min * f_max > 0:
        # No bracket exists - try to find one with bisection search
        # This can happen for extreme states near atmosphere
        found_bracket = False
        mu_test = 0.5 * (mu_min + mu_max)

        for _ in range(20):
            f_test, rho_test, P_test, h_test, valid_test = \
                _kastaun_residual_ideal_gas(mu_test, D, r2, q, eos_gamma, v2_max)

            if valid_test:
                if f_min * f_test < 0:
                    mu_max = mu_test
                    f_max = f_test
                    found_bracket = True
                    break
                elif f_test * f_max < 0:
                    mu_min = mu_test
                    f_min = f_test
                    found_bracket = True
                    break

            # Bisect toward where we expect the root
            if f_min > 0:
                mu_test = 0.5 * (mu_test + mu_max)
            else:
                mu_test = 0.5 * (mu_min + mu_test)

        if not found_bracket:
            # Give up and return atmosphere
            return rho_floor, 0.0, p_floor, p_floor/(rho_floor*gm1), 1.0, 1.0 + p_floor/rho_floor*eos_gamma/gm1, False

    # ===== SOLVE WITH FALSE POSITION =====
    mu_sol, converged, iters = _false_position_anderson_bjorck(
        mu_min, mu_max, f_min, f_max, tol, max_iter,
        D, r2, q, eos_gamma, v2_max
    )

    if not converged:
        return rho_floor, 0.0, p_floor, p_floor/(rho_floor*gm1), 1.0, 1.0 + p_floor/rho_floor*eos_gamma/gm1, False

    # ===== RECOVER PRIMITIVES FROM μ =====
    v2 = mu_sol * mu_sol * r2
    if v2 > v2_max:
        v2 = v2_max
    if v2 >= 1.0:
        v2 = 1.0 - 1e-15

    W = 1.0 / (1.0 - v2) ** 0.5
    if W > W_max:
        W = W_max
        v2 = 1.0 - 1.0 / (W * W)

    rho = D / W
    if rho < rho_floor:
        rho = rho_floor

    # Recover thermodynamic quantities using the same method as residual
    # e/D = q - μr² + 1
    e_over_D = q - mu_sol * r2 + 1.0

    # ε = W*e_over_D - 1
    eps = W * e_over_D - 1.0
    if eps < 0.0:
        eps = p_floor / (rho * gm1)

    # P = ρ ε (γ-1)
    P = rho * eps * gm1
    if P < p_floor:
        P = p_floor
        eps = P / (rho * gm1)

    # h = 1 + ε γ
    h = 1.0 + eps * eos_gamma
    if h < 1.0:
        h = 1.0 + 1e-10

    # Radial velocity: v² = γᵣᵣ vʳ² => vʳ = ±sqrt(v²/γᵣᵣ)
    # Sign comes from Sr
    if Sr >= 0:
        vr = (v2 / gamma_rr) ** 0.5
    else:
        vr = -(v2 / gamma_rr) ** 0.5

    return rho, vr, P, eps, W, h, True


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def _solve_kastaun_batch_ideal_gas(D, Sr, tau, gamma_rr, alpha,
                                    eos_gamma, rho_floor, p_floor,
                                    v_max, W_max, tol, max_iter):
    """
    Batch Kastaun solver for ideal gas EOS.
    Parallelized over all points using prange.
    """
    N = len(D)

    # Output arrays
    rho0 = np.zeros(N)
    vr = np.zeros(N)
    p_out = np.zeros(N)
    eps = np.zeros(N)
    W = np.ones(N)
    h = np.ones(N)
    converged = np.zeros(N, dtype=np.bool_)

    # Parallel loop over all points
    for i in prange(N):
        rho0[i], vr[i], p_out[i], eps[i], W[i], h[i], converged[i] = \
            _kastaun_kernel_ideal_gas(
                D[i], Sr[i], tau[i], gamma_rr[i],
                eos_gamma, rho_floor, p_floor, v_max, W_max, tol, max_iter
            )

    return rho0, vr, p_out, eps, W, h, converged


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
        tol: Solver tolerance (default: 1e-15)
        max_iter: Maximum iterations (default: 100)
        solver_method: "newton" or "kastaun" (default: "newton")
            - "newton": Newton-Raphson on pressure (fast, needs good initial guess)
            - "kastaun": Kastaun et al. (2021) bracketed solver (robust, guaranteed convergence)
    """

    def __init__(self, eos, atmosphere, tol=1e-15, max_iter=100, solver_method="newton"):
        self.eos = eos
        self.atmosphere = atmosphere

        # Validate solver method
        valid_methods = ["newton", "kastaun"]
        if solver_method not in valid_methods:
            raise ValueError(f"solver_method must be one of {valid_methods}, got '{solver_method}'")
        self._solver_method = solver_method

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

        # Check if we can use optimized path for polytropic EOS
        self._is_polytropic = (hasattr(eos, 'name') and
                               eos.name.startswith('polytropic') and
                               hasattr(eos, 'K') and hasattr(eos, 'gamma'))

        if self._is_polytropic:
            self._eos_K = float(eos.K)
            self._eos_gamma = float(eos.gamma)

        # Statistics tracking
        self.stats = {
            "total_calls": 0,
            "successful_conversions": 0,
            "newton_successes": 0,
            "kastaun_successes": 0,
            "atmosphere_fallbacks": 0,
            "conservative_floors_applied": 0
        }  

    def convert(self, D, Sr, tau, geom: GeometryState, p_guess=None, apply_conservative_floors=True):
        """
        Convert conservative to primitive variables using vectorized Newton-Raphson.

        IMPORTANT: Expects PHYSICAL (non-densitized) conservative variables:
            D  = ρ₀ W
            Sʳ = ρ₀ h W² vʳ γᵣᵣ
            τ  = ρ₀ h - P - D

        Args:
            D, Sr, tau: Conservative variables
            geom: GeometryState containing gamma_rr, e6phi, alpha, beta_r
            p_guess: Initial pressure guess (optional)
            apply_conservative_floors: Whether to apply conservative floors

        Returns:
            tuple: (rho0, vr, p, eps, W, h, success, D_out, Sr_out, tau_out)
                   Primitive variables, success mask, and updated conservatives.
        """
        self.stats["total_calls"] += 1

        # Extract geometry components
        gamma_rr = np.atleast_1d(np.asarray(geom.gamma_rr, dtype=float))
        e6phi = np.atleast_1d(np.asarray(geom.e6phi, dtype=float))
        alpha = np.atleast_1d(np.asarray(geom.alpha, dtype=float))
        beta_r = np.atleast_1d(np.asarray(geom.beta_r, dtype=float))

        # Ensure arrays
        D = np.atleast_1d(np.asarray(D, dtype=float))
        Sr = np.atleast_1d(np.asarray(Sr, dtype=float))
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        N = len(D)

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
        # IMPORTANT: Since D is densitized (D̃ = e^{6φ} D_phys), we must densitize
        # the threshold too to be consistent with evolving conformal factor φ
        # NOTE: Buffer factor increased from 10→100 to better handle WENO5
        # oscillations at the stellar surface (GRHayL-style buffer zone)
        atm_mask = D < 100.0 * self.rho_floor * e6phi

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
        atm_points = rho0 <= 100.0 * self.rho_floor
        if np.any(atm_points):
            # Recompute conservatives from atmosphere primitives using prim_to_cons
            geom_atm = GeometryState(
                alpha=alpha[atm_points],
                beta_r=beta_r[atm_points],
                gamma_rr=gamma_rr[atm_points],
                e6phi=e6phi[atm_points]
            )
            D_atm, Sr_atm, tau_atm = prim_to_cons(
                rho0[atm_points], vr[atm_points], p[atm_points],
                geom_atm, self.eos
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
            "kastaun_rate": self.stats["kastaun_successes"] / total,
        }

    def reset_statistics(self):
        """Reset all statistics counters."""
        for key in self.stats:
            self.stats[key] = 0

    def _solve_vectorized_points(self, D, Sr, tau, gamma_rr, p_guess=None, alpha=None):
        """
        Vectorized solver dispatcher.

        Routes to the appropriate solver based on self._solver_method:
        - "newton": Newton-Raphson on pressure
        - "kastaun": Kastaun et al. (2021) bracketed solver
        """
        # =====================================================================
        # KASTAUN SOLVER (robust, guaranteed convergence)
        # =====================================================================
        if self._solver_method == "kastaun":
            if self._is_ideal_gas:
                rho0, vr, p, eps, W, h, converged = _solve_kastaun_batch_ideal_gas(
                    D, Sr, tau, gamma_rr, alpha,
                    self._eos_gamma, self.rho_floor, self.p_floor,
                    self.v_max, self.W_max, self.tol, self.max_iter
                )
                self.stats["kastaun_successes"] += np.sum(converged)
                return rho0, vr, p, eps, W, h, converged
            else:
                # Fallback to Newton for non-ideal-gas EOS
                # (Kastaun only implemented for ideal gas currently)
                pass

        # =====================================================================
        # NEWTON SOLVER (fast, default)
        # =====================================================================
        N = len(D)

        # Initialize pressure guess
        if p_guess is not None:
            p_init = np.maximum(p_guess, self.p_floor)
        else:
            p_init = np.maximum(self.p_floor, 0.1 * (tau + D))

        # =====================================================================
        # FAST PATH: Use Numba kernel for ideal gas (most common case)
        # =====================================================================
        if self._is_ideal_gas:
            rho0, vr, p, eps, W, h, converged = _solve_newton_batch_ideal_gas(
                D, Sr, tau, gamma_rr, alpha, p_init,
                self._eos_gamma, self.p_floor, self.W_max, self.tol, self.max_iter
            )
            self.stats["newton_successes"] += np.sum(converged)
            return rho0, vr, p, eps, W, h, converged

        # =====================================================================
        # FAST PATH: Use Numba kernel for polytropic EOS
        # =====================================================================
        if self._is_polytropic:
            # For polytropic EOS, use rho as the unknown (not pressure)
            # Initial guess: rho ~ D (since D = rho * W and W ~ 1 for low velocities)
            rho_init = np.maximum(D, self.rho_floor)
            rho0, vr, p, eps, W, h, converged = _solve_newton_batch_polytropic(
                D, Sr, tau, gamma_rr, alpha, rho_init,
                self._eos_K, self._eos_gamma, self.rho_floor, self.W_max, self.tol, self.max_iter
            )
            self.stats["newton_successes"] += np.sum(converged)
            return rho0, vr, p, eps, W, h, converged

        # =====================================================================
        # SLOW PATH: General EOS (fallback with np.any() reduction)
        # =====================================================================
        p = p_init.copy()
        converged = np.zeros(N, dtype=bool)
        active = np.ones(N, dtype=bool)

        # Output arrays
        rho0 = np.zeros(N)
        vr = np.zeros(N)
        eps = np.zeros(N)
        W = np.ones(N)
        h = np.ones(N)

        # Newton-Raphson iterations (reduced np.any calls)
        for iteration in range(self.max_iter):
            # Count active points once per iteration
            n_active = np.sum(active)
            if n_active == 0:
                break

            # Evaluate function for active points
            p_active = p[active]
            ok_active, states_active = self._evaluate_pressure_vectorized(
                D[active], Sr[active], tau[active], p_active, gamma_rr[active],
                alpha=alpha[active]
            )

            # Handle evaluation failures (use sum instead of any)
            n_failed = np.sum(~ok_active)
            if n_failed > 0:
                failed_indices = np.where(active)[0][~ok_active]
                p[failed_indices] = np.maximum(self.p_floor,
                                             p[failed_indices] * 1.5 + 1e-14)
                continue

            rho0_active, vr_active, eps_active, W_active, h_active, f_active = states_active

            # Check convergence
            tol_active = self.tol * np.maximum(1.0, np.abs(p_active))
            converged_now = np.abs(f_active) <= tol_active
            n_converged = np.sum(converged_now)

            if n_converged > 0:
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

                if n_converged == n_active:
                    break

            # Compute derivatives for remaining points
            still_active = active & ~converged
            n_still = np.sum(still_active)
            if n_still == 0:
                break

            p_still = p[still_active]
            # Residual aligned with still_active
            f_still = f_active[~converged_now] if n_converged > 0 else f_active

            # Numerical derivative for general EOS
            dp = np.maximum(1e-3 * np.maximum(np.abs(p_still), 1.0), 1e-14)
            ok2, states2 = self._evaluate_pressure_vectorized(
                D[still_active], Sr[still_active], tau[still_active],
                p_still + dp, gamma_rr[still_active],
                alpha=alpha[still_active]
            )
            n_ok2 = np.sum(ok2)
            if n_ok2 < n_still:
                failed_indices = np.where(still_active)[0][~ok2]
                p[failed_indices] = np.maximum(self.p_floor,
                                             p[failed_indices] * 1.5 + 1e-12)
                continue

            num = states2[5] - f_still
            num = np.where(np.isfinite(num), num, np.inf)
            dp_safe = np.where(np.abs(dp) > 0, dp, 1e-12)
            df = num / dp_safe
            df = np.where(np.isfinite(df), df, 0.0)

            # Avoid small derivatives (use sum instead of any)
            small_deriv = (np.abs(df) < 1e-15)
            n_small = np.sum(small_deriv)
            if n_small > 0:
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
        p = np.maximum(p, self.p_floor)
        
        # Pre-compute all intermediate values (vectorized)
        Q = tau + D + p
        gamma_rr = np.maximum(gamma_rr, 1e-30)
        
        # Compute velocity for all points
        # vr =  Sr / (Q * γrr)
        # From: Sr = ρ h W² vr γrr => vr = Sr / (Q γrr)
        vr = Sr / (Q * gamma_rr)
        v2 = gamma_rr * vr * vr
        
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

def prim_to_cons(rho0, vr, pressure, geom: GeometryState, eos):
    """
    Convert primitive to conservative variables.

    Following Valencia formulation:
        D̃  = e^{6φ} W ρ₀
        S̃r = e^{6φ} W² ρ₀ h vr γrr
        τ̃  = e^{6φ} (W² ρ₀ h - P - W ρ₀)

    Args:
        rho0: Rest mass density
        vr: Radial velocity v^r
        pressure: Pressure
        geom: GeometryState containing gamma_rr, e6phi (only these are used)
        eos: Equation of state

    Returns:
        tuple: (D, Sr, tau) conservative variables (densitized)
    """
    # Extract geometry components
    gamma_rr = geom.gamma_rr
    e6phi = geom.e6phi

    # Convert inputs to arrays for unified handling
    rho0 = np.asarray(rho0)
    vr = np.asarray(vr)
    pressure = np.asarray(pressure)
    gamma_rr = np.asarray(gamma_rr)
    e6phi = np.asarray(e6phi)

    # Broadcast to same shape
    rho0, vr, pressure, gamma_rr, e6phi = np.broadcast_arrays(
        rho0, vr, pressure, gamma_rr, e6phi
    )

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
