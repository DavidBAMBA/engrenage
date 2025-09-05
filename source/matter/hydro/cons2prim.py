# cons2prim.py - OPTIMIZED & FACTORIZED
import numpy as np

# Default parameters as constant
DEFAULT_PARAMS = {
    "rho_floor": 1e-13, "p_floor": 1e-15, "v_max": 0.999999, "W_max": 1e3,
    "tol": 1e-12, "max_iter": 500, "bracket_expand": 10.0, "bracket_steps": 20, "newton_pert": 1e-3
}

def cons_to_prim(U, eos, params=None, metric=None):
    """
    Convert conservative (D, S_r, tau) to primitive (rho0, v^r, p) variables.
    
    Args:
        U: (D, Sr, tau) tuple/list/dict
        eos: EOS object with eps_from_rho_p(rho0, p)
        params: solver parameters dict (optional)
        metric: (alpha, beta_r, gamma_rr) tuple/dict (optional, default Minkowski)
    
    Returns:
        dict: {'rho0', 'vr', 'p', 'eps', 'W', 'h', 'success'}
    """
    # Parse inputs efficiently
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    D, Sr, tau = _parse_conservative(U)
    alpha, beta_r, gamma_rr = _parse_metric(metric, len(D))
    
    N = len(D)
    result = {key: np.zeros(N) for key in ['rho0', 'vr', 'p', 'eps', 'W', 'h']}
    result['success'] = np.zeros(N, dtype=bool)
    
    # Solve each point
    for i in range(N):
        if _is_valid_input(D[i], Sr[i], tau[i], cfg):
            ok, state = _solve_point(D[i], Sr[i], tau[i], gamma_rr[i], eos, cfg)
        else:
            ok, state = False, None
            
        if ok:
            result['rho0'][i], result['vr'][i], result['p'][i] = state[:3]
            result['eps'][i], result['W'][i], result['h'][i] = state[3:]
            result['success'][i] = True
        else:
            # Fallback to atmosphere
            result['rho0'][i], result['vr'][i], result['p'][i] = cfg['rho_floor'], 0.0, cfg['p_floor']
            result['eps'][i] = eos.eps_from_rho_p(result['rho0'][i], result['p'][i])
            result['W'][i], result['h'][i] = 1.0, 1.0 + result['eps'][i] + result['p'][i]/result['rho0'][i]
    
    return result

def prim_to_cons(rho0, vr, pressure, gamma_rr, eos):
    """Convert primitive to conservative variables."""
    g = max(float(gamma_rr), 1e-30)
    v2 = np.clip(g * float(vr)**2, 0.0, 1.0 - 1e-12)
    W = 1.0 / np.sqrt(1.0 - v2)
    
    eps = eos.eps_from_rho_p(float(rho0), float(pressure))
    h = 1.0 + eps + float(pressure) / max(float(rho0), 1e-30)
    
    D = float(rho0) * W
    Sr = float(rho0) * h * W**2 * float(vr) * g
    tau = float(rho0) * h * W**2 - float(pressure) - D
    return D, Sr, tau

# ==================== INTERNAL HELPERS ====================

def _parse_conservative(U):
    """Parse conservative variables from various input formats."""
    if isinstance(U, dict):
        return [np.atleast_1d(np.asarray(U[k], dtype=float)) for k in ['D', 'Sr', 'tau']]
    elif isinstance(U, (tuple, list)) and len(U) == 3:
        return [np.atleast_1d(np.asarray(u, dtype=float)) for u in U]
    else:
        arr = np.asarray(U, dtype=float)
        if arr.shape[-1] == 3:
            D, Sr, tau = arr[..., 0].ravel(), arr[..., 1].ravel(), arr[..., 2].ravel()
            return D, Sr, tau
    raise ValueError("U must be dict {'D','Sr','tau'}, tuple (D,Sr,tau), or array with last dim=3")

def _parse_metric(metric, N):
    """Parse metric from various input formats, default to Minkowski."""
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

def _is_valid_input(D, Sr, tau, cfg):
    """Check if conservative variables are physically valid."""
    return (np.isfinite([D, Sr, tau]).all() and D >= cfg['rho_floor'] and tau >= -D)

def _solve_point(D, Sr, tau, gamma_rr, eos, cfg):
    """Solve for primitives at a single point using robust root finding."""
    
    # Try bracketing and bisection first
    p_lo, p_hi, f_lo, f_hi = _bracket_pressure(D, Sr, tau, gamma_rr, eos, cfg)
    
    if p_lo is not None and f_lo * f_hi <= 0:
        # Bisection method
        for _ in range(cfg['max_iter']):
            p_mid = 0.5 * (p_lo + p_hi)
            ok, rho0, vr, eps, W, h, f_mid = _evaluate_state(D, Sr, tau, p_mid, gamma_rr, eos, cfg)
            
            if not ok:
                p_lo, f_lo = p_mid, f_mid
                continue
            if abs(f_mid) <= cfg['tol'] * max(1.0, p_mid):
                return True, (rho0, vr, p_mid, eps, W, h)
            if f_lo * f_mid <= 0:
                p_hi, f_hi = p_mid, f_mid
            else:
                p_lo, f_lo = p_mid, f_mid
    
    # Fallback: Newton-like iterations
    p = max(cfg['p_floor'], 0.1 * (tau + D))
    for _ in range(cfg['max_iter'] // 4):  # Fewer iterations for fallback
        ok, rho0, vr, eps, W, h, f = _evaluate_state(D, Sr, tau, p, gamma_rr, eos, cfg)
        if not ok:
            p = max(cfg['p_floor'], p * 1.5)
            continue
        if abs(f) <= cfg['tol'] * max(1.0, p):
            return True, (rho0, vr, p, eps, W, h)
        
        # Numeric derivative
        dp = max(cfg['newton_pert'] * max(abs(p), 1.0), 1e-12)
        _, _, _, _, _, _, f2 = _evaluate_state(D, Sr, tau, p + dp, gamma_rr, eos, cfg)
        df_dp = (f2 - f) / dp
        
        if abs(df_dp) > 1e-15:
            p_new = p - f / df_dp
            p = 0.5 * p + 0.5 * max(cfg['p_floor'], p_new) if p_new > 0 else p * 1.1
        else:
            p *= 1.1
    
    return False, None

def _bracket_pressure(D, Sr, tau, gamma_rr, eos, cfg):
    """Find pressure bracket [p_lo, p_hi] where f changes sign."""
    E = tau + D
    g = max(gamma_rr, 1e-30)
    
    # Lower bound from v^2 < 1 constraint  
    p_lo = max(cfg['p_floor'], abs(Sr) / np.sqrt(g) - E + 1e-15)
    p_hi = max(p_lo * 2, 0.1 * max(1.0, E + abs(Sr)))
    
    # Evaluate bounds
    _, _, _, _, _, _, f_lo = _evaluate_state(D, Sr, tau, p_lo, g, eos, cfg)
    _, _, _, _, _, _, f_hi = _evaluate_state(D, Sr, tau, p_hi, g, eos, cfg)
    
    # Expand upper bound if needed
    for _ in range(cfg['bracket_steps']):
        if f_lo * f_hi <= 0:
            break
        p_hi *= cfg['bracket_expand']
        _, _, _, _, _, _, f_hi = _evaluate_state(D, Sr, tau, p_hi, g, eos, cfg)
    
    if f_lo * f_hi <= 0:
        return p_lo, p_hi, f_lo, f_hi
    return None, None, None, None

def _evaluate_state(D, Sr, tau, p, gamma_rr, eos, cfg):
    """Evaluate primitive state for given pressure and return residual."""
    p = max(float(p), cfg['p_floor'])
    E = tau + D
    Q = E + p
    
    if Q <= 0:
        return False, 0, 0, 0, 1, 1, np.inf
    
    g = max(float(gamma_rr), 1e-30)
    vr = Sr / (Q * g)
    v2 = g * vr**2
    
    if not (0 <= v2 < 1):
        return False, 0, 0, 0, 1, 1, np.inf
    
    W = 1.0 / np.sqrt(1.0 - v2)
    if not (1 <= W <= cfg['W_max']):
        return False, 0, 0, 0, 1, 1, np.inf
    
    rho0 = D / max(W, 1e-30)
    if rho0 <= 0 or not np.isfinite(rho0):
        return False, 0, 0, 0, 1, 1, np.inf
    
    try:
        eps = eos.eps_from_rho_p(rho0, p)
    except:
        # Fallback for ideal gas
        gamma = getattr(eos, 'gamma', 1.4)
        eps = p / ((gamma - 1.0) * rho0)
    
    if not (np.isfinite(eps) and eps >= 0):
        return False, 0, 0, 0, 1, 1, np.inf
    
    h = 1.0 + eps + p / max(rho0, 1e-30)
    if not (np.isfinite(h) and h > 1):
        return False, 0, 0, 0, 1, 1, np.inf
    
    # Residual: rho0 * h * W^2 - Q = 0
    f = rho0 * h * W**2 - Q
    return True, rho0, vr, eps, W, h, f

# ==================== LEGACY CLASS (simplified) ====================

class ConservativeToPrimitive:
    """Legacy OO interface for backward compatibility."""
    
    def __init__(self, eos, atmosphere_rho=1e-13, max_iterations=50, tolerance=1e-12):
        self.eos = eos
        self.params = {**DEFAULT_PARAMS, 
                      "rho_floor": atmosphere_rho, "max_iter": max_iterations, "tol": tolerance}
        self.total_calls = 0
        self.failed_conversions = 0
    
    def convert_all_points(self, D, Sr, tau, rho0_out, vr_out, p_out, eps_out, W_out, h_out,
                          alpha, beta_r, gamma_rr):
        """Legacy interface that modifies output arrays in-place."""
        result = cons_to_prim((D, Sr, tau), self.eos, self.params, (alpha, beta_r, gamma_rr))
        
        rho0_out[:] = result['rho0']
        vr_out[:] = result['vr'] 
        p_out[:] = result['p']
        eps_out[:] = result['eps']
        W_out[:] = result['W']
        h_out[:] = result['h']
        
        self.total_calls += len(D)
        self.failed_conversions += np.sum(~result['success'])
        return result['success']
    
    def get_statistics(self):
        return {
            "total_calls": self.total_calls,
            "failed_conversions": self.failed_conversions, 
            "failure_rate": self.failed_conversions / max(self.total_calls, 1)
        }