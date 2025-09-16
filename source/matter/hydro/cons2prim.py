# cons2prim.py

import numpy as np

# Optional SciPy for robust Brent root-finding
try:
    from scipy.optimize import brentq
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# Numba for JIT compilation
try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    # Fallback decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    _HAVE_NUMBA = False


def _default_params():
    return {
        "rho_floor": 1e-13,
        "p_floor": 1e-15,
        "v_max": 0.999999,       # cap on |v| in units of c
        "W_max": 1.0e3,          # cap on Lorentz factor
        "tol": 1e-12,            # relative tolerance for root solve
        "max_iter": 500,         # iterations per point
        "bracket_expand": 10.0,  # multiplicative expansion of p_high while bracketing
        "bracket_steps": 20,     # max expansions
        "newton_pert": 1e-3,     # relative perturbation for numeric df/dp
    }


def _ensure_metric(metric, N):
    """Return metric arrays (alpha, beta_r, gamma_rr). Default: Minkowski."""
    if metric is None:
        alpha = np.ones(N)
        beta_r = np.zeros(N)
        gamma_rr = np.ones(N)
        return alpha, beta_r, gamma_rr

    # Accept dict or tuple-like
    if isinstance(metric, dict):
        alpha = np.array(metric.get("alpha", 1.0), copy=False)
        beta_r = np.array(metric.get("beta_r", 0.0), copy=False)
        gamma_rr = np.array(metric.get("gamma_rr", 1.0), copy=False)
    else:
        alpha, beta_r, gamma_rr = metric

    # Broadcast scalars
    alpha = np.broadcast_to(alpha, (N,)).astype(float, copy=False)
    beta_r = np.broadcast_to(beta_r, (N,)).astype(float, copy=False)
    gamma_rr = np.broadcast_to(gamma_rr, (N,)).astype(float, copy=False)
    return alpha, beta_r, gamma_rr


def _parse_U(U):
    """
    Accepts:
      - dict with keys {'D','Sr','tau'}
      - tuple/list (D, Sr, tau)
      - ndarray shape (..., 3) meaning [...,0]=D, [...,1]=Sr, [...,2]=tau
    Returns (D, Sr, tau) as 1-D arrays of length N.
    """
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
            D = np.atleast_1d(np.array([D], dtype=float))
            Sr = np.atleast_1d(np.array([Sr], dtype=float))
            tau = np.atleast_1d(np.array([tau], dtype=float))
        elif arr.ndim >= 1 and arr.shape[-1] == 3:
            D = arr[..., 0].ravel()
            Sr = arr[..., 1].ravel()
            tau = arr[..., 2].ravel()
        else:
            raise ValueError("U must be dict with keys {'D','Sr','tau'}, a (D,Sr,tau) tuple, or an array with last dim=3")
    # Sanity check equal lengths
    N = len(D)
    if not (len(Sr) == N and len(tau) == N):
        raise ValueError("D, Sr, tau must have same length")
    return D, Sr, tau


@njit
def _state_from_p_numba(D, Sr, tau, p, gamma_rr, p_floor, W_max, gamma_eos):
    """
    Numba-optimized version of _state_from_p for ideal gas EOS.
    Returns (ok, rho0, vr, eps, W, h, f).
    """
    p = max(p, p_floor)
    E = tau + D  # = rho h W^2 - p
    Q = E + p    # = rho h W^2

    # Enforce Q>0
    if Q <= 0.0:
        return False, 0.0, 0.0, 0.0, 1.0, 1.0, np.inf

    g = max(gamma_rr, 1e-30)

    # v^r from momentum: S_r = Q * v_r, and v_r = gamma_rr * v^r  ⇒  v^r = S_r / (Q * gamma_rr)
    vr = Sr / (Q * g)

    # v^2 and W
    v2 = g * vr * vr
    if not (0.0 <= v2 < 1.0):
        return False, 0.0, 0.0, 0.0, 1.0, 1.0, np.inf

    W = 1.0 / np.sqrt(1.0 - v2)
    if not (1.0 <= W <= W_max):
        return False, 0.0, 0.0, 0.0, 1.0, 1.0, np.inf

    # rho0
    rho0 = D / max(W, 1e-30)
    if rho0 <= 0.0 or not np.isfinite(rho0):
        return False, 0.0, 0.0, 0.0, 1.0, 1.0, np.inf

    # EOS: epsilon (ideal gas)
    eps = p / ((gamma_eos - 1.0) * rho0)
    if not np.isfinite(eps) or eps < 0.0:
        return False, 0.0, 0.0, 0.0, 1.0, 1.0, np.inf

    h = 1.0 + eps + p / max(rho0, 1e-30)
    if not np.isfinite(h) or h <= 1.0:
        return False, 0.0, 0.0, 0.0, 1.0, 1.0, np.inf

    # Residual
    f = rho0 * h * W * W - Q
    return True, rho0, vr, eps, W, h, f


def _state_from_p(D, Sr, tau, p, gamma_rr, eos, params):
    """
    Compute primitive state given a pressure guess p.
    Returns (ok, rho0, vr, eps, W, h, f) where f(p) = rho0*h*W^2 - (E+p).
    """
    # Try numba version for ideal gas EOS
    if _HAVE_NUMBA and hasattr(eos, "gamma"):
        return _state_from_p_numba(D, Sr, tau, p, gamma_rr,
                                   params["p_floor"], params["W_max"], eos.gamma)

    # Fallback to original implementation
    p = float(p)
    p = max(p, params["p_floor"])
    E = tau + D  # = rho h W^2 - p
    Q = E + p    # = rho h W^2

    # Enforce Q>0
    if Q <= 0.0:
        return False, 0, 0, 0, 1, 1, np.inf

    g = float(max(gamma_rr, 1e-30))

    # v^r from momentum: S_r = Q * v_r, and v_r = gamma_rr * v^r  ⇒  v^r = S_r / (Q * gamma_rr)
    vr = Sr / (Q * g)

    # v^2 and W
    v2 = g * vr * vr
    if not (0.0 <= v2 < 1.0):
        return False, 0, 0, 0, 1, 1, np.inf

    W = 1.0 / np.sqrt(1.0 - v2)
    if not (1.0 <= W <= params["W_max"]):
        return False, 0, 0, 0, 1, 1, np.inf

    # rho0
    rho0 = D / max(W, 1e-30)
    if rho0 <= 0.0 or not np.isfinite(rho0):
        return False, 0, 0, 0, 1, 1, np.inf

    # EOS: epsilon and enthalpy
    try:
        eps = eos.eps_from_rho_p(rho0, p)
    except Exception:
        # Fallback: ideal-gas-like with gamma if available
        if hasattr(eos, "gamma"):
            gamma = float(eos.gamma)
            eps = p / ((gamma - 1.0) * rho0)
        else:
            return False, 0, 0, 0, 1, 1, np.inf
    if not np.isfinite(eps) or eps < 0.0:
        return False, 0, 0, 0, 1, 1, np.inf

    h = 1.0 + eps + p / max(rho0, 1e-30)
    if not np.isfinite(h) or h <= 1.0:
        return False, 0, 0, 0, 1, 1, np.inf

    # Residual
    f = rho0 * h * W * W - Q
    return True, rho0, vr, eps, W, h, f


def _deriv_df_dp(D, Sr, tau, p, gamma_rr, eos, params, rho0, vr, eps, W, h):
    """
    Numeric derivative df/dp via one-sided perturbation (robust, no EOS jacobian needed).
    """
    dp = max(params["newton_pert"] * max(abs(p), 1.0), 1e-12)
    p2 = p + dp
    ok2, *_rest2, f2 = _state_from_p(D, Sr, tau, p2, gamma_rr, eos, params)
    ok1, *_rest1, f1 = _state_from_p(D, Sr, tau, p,  gamma_rr, eos, params)
    if not (ok1 and ok2):
        # fallback to small symmetric difference if possible
        p0 = max(p - dp, params["p_floor"])
        ok0, *_rest0, f0 = _state_from_p(D, Sr, tau, p0, gamma_rr, eos, params)
        if not (ok0 and ok2):
            return None
        return (f2 - f0) / (p2 - p0)
    return (f2 - f1) / dp


def _bracket_pressure(D, Sr, tau, gamma_rr, eos, params):
    """
    Find [p_lo, p_hi] such that f(p_lo)*f(p_hi) <= 0 (ideally < 0).
    Use the v^2<1 constraint to get a safe lower bound, then expand high bound.
    """
    E = tau + D
    Sabs = abs(Sr)
    g = max(gamma_rr, 1e-30)

    # Lower bound: ensure Q=E+p > |S|/sqrt(g) to keep v^2<1
    p_lo = max(params["p_floor"], (Sabs / np.sqrt(g)) - E + 1e-15)

    # Start with a moderate upper bound relative to E and |S|
    scale = max(1.0, E + Sabs)
    p_hi = max(p_lo * 2.0, 0.1 * scale)

    # Evaluate f at bounds
    ok_lo, *_rest_lo, f_lo = _state_from_p(D, Sr, tau, p_lo, g, eos, params)
    ok_hi, *_rest_hi, f_hi = _state_from_p(D, Sr, tau, p_hi, g, eos, params)

    # Expand p_hi until sign changes or we hit limit
    steps = 0
    while steps < params["bracket_steps"] and (not ok_lo or not ok_hi or f_lo * f_hi > 0.0):
        # If either state invalid or no sign change, adjust bound upward
        p_hi = max(p_hi * params["bracket_expand"], p_hi + 1.0)
        ok_hi, *_rest_hi, f_hi = _state_from_p(D, Sr, tau, p_hi, g, eos, params)
        steps += 1

        # Try to slightly decrease p_lo as well if invalid
        if not ok_lo and p_lo > params["p_floor"]:
            p_lo = max(params["p_floor"], 0.5 * p_lo)
            ok_lo, *_rest_lo, f_lo = _state_from_p(D, Sr, tau, p_lo, g, eos, params)

    if ok_lo and ok_hi and f_lo * f_hi <= 0.0:
        return True, p_lo, p_hi, f_lo, f_hi
    return False, p_lo, p_hi, f_lo, f_hi


def _bracket_pressure_with_guess(D, Sr, tau, gamma_rr, eos, params, p_guess):
    """
    Try to bracket around a provided guess pressure.
    Returns (success, p_lo, p_hi, f_lo, f_hi).
    """
    # Use guess as starting point and bracket around it
    ok_guess, *_rest_guess, f_guess = _state_from_p(D, Sr, tau, p_guess, gamma_rr, eos, params)
    if not ok_guess:
        # Fallback to standard bracketing
        return _bracket_pressure(D, Sr, tau, gamma_rr, eos, params)

    # Try to find bracket around guess
    factor = 1.5
    p_lo, p_hi = p_guess / factor, p_guess * factor

    # Ensure p_lo is valid
    p_lo = max(p_lo, params["p_floor"])
    ok_lo, *_rest_lo, f_lo = _state_from_p(D, Sr, tau, p_lo, gamma_rr, eos, params)

    # Try to find p_hi
    for _ in range(10):
        ok_hi, *_rest_hi, f_hi = _state_from_p(D, Sr, tau, p_hi, gamma_rr, eos, params)
        if ok_hi:
            break
        p_hi *= factor

    if ok_lo and ok_hi and f_lo * f_hi <= 0.0:
        return True, p_lo, p_hi, f_lo, f_hi
    elif ok_guess:
        # Use guess as one endpoint and try to bracket around it
        if f_guess > 0:
            return True, max(params["p_floor"], p_guess/3), p_guess, -abs(f_guess)/2, f_guess
        else:
            return True, p_guess, p_guess*3, f_guess, abs(f_guess)/2
    else:
        # Fallback to standard
        return _bracket_pressure(D, Sr, tau, gamma_rr, eos, params)


def _solve_pressure(D, Sr, tau, gamma_rr, eos, params, p_guess=None):
    """
    Solve f(p)=0 with fallbacks: Brent → bisection → Newton → floors.
    Returns (ok, rho0, vr, p, eps, W, h).
    """
    # Try to bracket, using intelligent guess if provided
    if p_guess is not None and p_guess > params["p_floor"]:
        # Try guess first
        ok_guess, rho0_g, vr_g, eps_g, W_g, h_g, f_g = _state_from_p(D, Sr, tau, p_guess, gamma_rr, eos, params)
        if ok_guess and abs(f_g) <= params["tol"] * max(1.0, abs(p_guess)):
            return True, rho0_g, vr_g, p_guess, eps_g, W_g, h_g

        # Use guess to improve bracketing
        ok_br, p_lo, p_hi, f_lo, f_hi = _bracket_pressure_with_guess(D, Sr, tau, gamma_rr, eos, params, p_guess)
    else:
        # Standard bracketing
        ok_br, p_lo, p_hi, f_lo, f_hi = _bracket_pressure(D, Sr, tau, gamma_rr, eos, params)

    # Attempt Brent (SciPy)
    """ if ok_br and _HAVE_SCIPY:
        try:
            root = brentq(lambda p: _state_from_p(D, Sr, tau, p, gamma_rr, eos, params)[-1],
                          p_lo, p_hi, xtol=max(1e-14, params["tol"]), rtol=params["tol"], maxiter=params["max_iter"])
            ok, rho0, vr, eps, W, h, f = _state_from_p(D, Sr, tau, root, gamma_rr, eos, params)
            if ok:
                return True, rho0, vr, root, eps, W, h
        except Exception:
            pass """

    # Manual bisection
    if ok_br:
        a, b = p_lo, p_hi
        fa, fb = f_lo, f_hi
        for _ in range(params["max_iter"]):
            c = 0.5 * (a + b)
            okc, rho0_c, vr_c, eps_c, W_c, h_c, fc = _state_from_p(D, Sr, tau, c, gamma_rr, eos, params)
            if not okc:
                a, fa = c, fc
                continue
            if abs(fc) <= params["tol"] * max(1.0, abs(c)):
                return True, rho0_c, vr_c, c, eps_c, W_c, h_c
            if fa * fc <= 0.0:
                b, fb = c, fc
            else:
                a, fa = c, fc

    # Newton (secant-ish) starting near pressure scale
    p = max(params["p_floor"], 0.1 * (tau + D))
    for _ in range(params["max_iter"]):
        ok, rho0, vr, eps, W, h, f = _state_from_p(D, Sr, tau, p, gamma_rr, eos, params)
        if not ok:
            p = max(params["p_floor"], p * 1.5 + 1e-12)
            continue
        if abs(f) <= params["tol"] * max(1.0, abs(p)):
            return True, rho0, vr, p, eps, W, h
        df = _deriv_df_dp(D, Sr, tau, p, gamma_rr, eos, params, rho0, vr, eps, W, h)
        if df is None or df == 0.0 or not np.isfinite(df):
            p = max(params["p_floor"], p * 1.5 + 1e-12)
            continue
        p_new = p - f / df
        if not np.isfinite(p_new) or p_new <= 0.0:
            p_new = max(params["p_floor"], 0.5 * p)
        p = 0.5 * p + 0.5 * p_new

    # Floors (atmosphere)
    rho0 = params["rho_floor"]
    vr = 0.0
    p = params["p_floor"]
    try:
        eps = eos.eps_from_rho_p(rho0, p)
    except Exception:
        eps = 1e-10
    W = 1.0
    h = 1.0 + eps + p / rho0
    return False, rho0, vr, p, eps, W, h


def cons_to_prim(U, eos, params=None, metric=None, p_guess=None):
    """
      primitives = cons_to_prim(U, eos, params=None, metric=None, p_guess=None)

    Inputs:
      - U: dict with arrays {'D','Sr','tau'} or tuple/list (D,Sr,tau) or array with last dim=3
      - eos: an EOS object exposing at least eps_from_rho_p(rho0, p)
      - params: dict of solver parameters (floors, tolerances, max_iter, etc.). Missing entries use defaults.
      - metric: dict or tuple (alpha, beta_r, gamma_rr). Default: Minkowski.
      - p_guess: array of pressure guesses from previous timestep (optional)

    Returns dict of arrays:
      {'rho0','vr','p','eps','W','h','success'}
    """
    if params is None:
        params = {}
    cfg = _default_params()
    cfg.update(params or {})

    D, Sr, tau = _parse_U(U)
    N = len(D)
    alpha, beta_r, gamma_rr = _ensure_metric(metric, N)

    # Ensure p_guess has the right shape
    if p_guess is not None:
        p_guess = np.asarray(p_guess)
        if p_guess.shape != (N,):
            p_guess = None

    # Allocate outputs
    rho0, vr, p, eps, W, h = (np.zeros(N) for _ in range(6))
    success = np.zeros(N, dtype=bool)

    # Loop over points
    for i in range(N):
        if (not np.isfinite(D[i]) or not np.isfinite(Sr[i]) or not np.isfinite(tau[i])
            or D[i] < cfg["rho_floor"] or (tau[i] < -D[i])):
            ok = False
        else:
            # Use pressure guess if available and valid
            guess_i = None
            if p_guess is not None and np.isfinite(p_guess[i]) and p_guess[i] > cfg["p_floor"]:
                guess_i = p_guess[i]

            ok, rho0_i, vr_i, p_i, eps_i, W_i, h_i = _solve_pressure(
                D[i], Sr[i], tau[i], gamma_rr[i], eos, cfg, guess_i
            )

        if not ok:
            rho0_i, vr_i, p_i = cfg["rho_floor"], 0.0, cfg["p_floor"]
            try:
                eps_i = eos.eps_from_rho_p(rho0_i, p_i)
            except Exception:
                eps_i = 1e-10
            W_i, h_i = 1.0, 1.0 + eps_i + p_i / rho0_i
        else:
            # Enforce |v| < v_max
            v2 = gamma_rr[i] * (vr_i ** 2)
            vmax2 = cfg["v_max"] ** 2
            if v2 >= vmax2:
                vr_i = np.sign(vr_i) * cfg["v_max"] / np.sqrt(max(gamma_rr[i], 1e-30))

        rho0[i], vr[i], p[i], eps[i], W[i], h[i], success[i] = rho0_i, vr_i, p_i, eps_i, W_i, h_i, ok

    return {"rho0": rho0, "vr": vr, "p": p, "eps": eps, "W": W, "h": h, "success": success}


def prim_to_cons(rho0, vr, pressure, gamma_rr, eos):
    """
    Convert primitive (rho0, v^r, p) to conservative (D, S_r, tau) in 1D with metric γ_rr.
    Returns scalars (D, S_r, tau).
    """
    g = float(max(gamma_rr, 1e-30))
    v2 = g * float(vr) * float(vr)
    v2 = float(np.clip(v2, 0.0, 1.0 - 1e-12))
    W  = 1.0 / np.sqrt(1.0 - v2)

    # EOS: eps, enthalpy
    eps = eos.eps_from_rho_p(float(rho0), float(pressure))
    h   = 1.0 + eps + float(pressure) / max(float(rho0), 1e-30)

    D   = float(rho0) * W
    Sr  = float(rho0) * h * W * W * float(vr) * g  # S_r = ρ h W^2 v_r ; v_r = γ_rr v^r
    tau = float(rho0) * h * W * W - float(pressure) - D
    return D, Sr, tau



class ConservativeToPrimitive:
    """
    Thin OO wrapper that mirrors the previous class API.
    """

    def __init__(self, eos, atmosphere_rho=1e-13, max_iterations=50, tolerance=1e-12):
        self.eos = eos
        self.params = _default_params()
        self.params["rho_floor"] = atmosphere_rho
        self.params["p_floor"] = min(self.params["p_floor"], atmosphere_rho * 1e-2)
        self.params["max_iter"] = max_iterations
        self.params["tol"] = tolerance
        self.total_calls = 0
        self.failed_conversions = 0
        self.average_iterations = 0.0

    def convert_all_points(self, D, Sr, tau,
                           rho0_out, vr_out, p_out, eps_out, W_out, h_out,
                           alpha, beta_r, gamma_rr):
        # Delega a la nueva API funcional
        out = cons_to_prim((D, Sr, tau), self.eos,
                           params=self.params,
                           metric=(alpha, beta_r, gamma_rr))
        rho0_out[:] = out["rho0"]
        vr_out[:] = out["vr"]
        p_out[:] = out["p"]
        eps_out[:] = out["eps"]
        W_out[:] = out["W"]
        h_out[:] = out["h"]
        self.total_calls += len(D)
        self.failed_conversions += np.sum(~out["success"])
        return out["success"]

    def primitive_to_conservative(self, rho0, vr, pressure, eps, W, h,
                                  alpha, beta_r, gamma_rr):
        # Mantenemos este método dentro de la clase para compatibilidad
        g = gamma_rr
        v2 = g * vr**2
        Wc = 1.0 / np.sqrt(1.0 - np.clip(v2, 0.0, 0.9999999))
        D = rho0 * Wc
        Sr = rho0 * h * Wc**2 * vr * g  # S_r = rho h W^2 v_r
        tau = rho0 * h * Wc**2 - pressure - D
        return D, Sr, tau

    def get_statistics(self):
        tot = max(self.total_calls, 1)
        return {
            "total_calls": self.total_calls,
            "failed_conversions": self.failed_conversions,
            "failure_rate": float(self.failed_conversions) / tot,
            "average_iterations": self.average_iterations,
        }

    def reset_statistics(self):
        self.total_calls = 0
        self.failed_conversions = 0
        self.average_iterations = 0.0
