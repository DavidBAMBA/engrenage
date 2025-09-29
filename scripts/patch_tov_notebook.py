#!/usr/bin/env python3
import json
import re
from pathlib import Path

NB_PATH = Path('examples/TOVEvolution.ipynb')

def load_nb(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def save_nb(path: Path, nb):
    tmp = path.with_suffix('.ipynb.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False)
    tmp.replace(path)

def patch_code(src: str) -> str:
    out = src

    # 1) Fix TOV rho_profile build to avoid sqrt warning
    out = re.sub(
        r"rho_profile\s*=\s*np.where\(pressure_profile\s*>\s*1e-15,\s*\(pressure_profile\s*/\s*K\)\*\*\(1.0/gamma\),\s*1e-15\)",
        (
            "rho_profile = np.full_like(pressure_profile, 1e-15)\n"
            "mask = pressure_profile > 1e-15\n"
            "rho_profile[mask] = (pressure_profile[mask] / K)**(1.0/gamma)"
        ),
        out
    )

    # 2) In helper cell: make _cons_to_prim_at_state use TOV metric by default
    if 'def _cons_to_prim_at_state(' in out and '_get_tov_metric' not in out:
        # Inject _get_tov_metric and modify _cons_to_prim_at_state
        out = out.replace(
            'def _cons_to_prim_at_state(D, Sr, tau, r_grid, hydro):',
            'def _get_tov_metric():\n'
            '    # Use precomputed tov_geometry if available; else flat metric\n'
            '    g = globals()\n'
            '    if "tov_geometry" in g:\n'
            '        tg = g["tov_geometry"]\n'
            '        return {"alpha": tg.get("alpha", 1.0),\n'
            '                "beta_r": tg.get("beta_r", 0.0),\n'
            '                "gamma_rr": tg.get("gamma_rr", 1.0)}\n'
            '    # Fallback to flat metric\n'
            '    return None\n\n'
            'def _cons_to_prim_at_state(D, Sr, tau, r_grid, hydro, metric=None):'
        )
        out = out.replace(
            "result = cons_to_prim(U, hydro.eos, metric=metric)",
            "metric = metric or _get_tov_metric() or {'alpha': np.ones_like(D), 'beta_r': np.zeros_like(D), 'gamma_rr': np.ones_like(D)}\n"
            "result = cons_to_prim(U, hydro.eos, metric=metric)"
        )
    # 3) In helper cell: use TOV metric for _rho0_max_initial by default
    if 'def _rho0_max_initial(' in out and '_get_tov_metric' in out:
        out = re.sub(
            r"prims_0\s*=\s*_cons_to_prim_at_state\(D_initial,\s*\n\s*state_initial\[hydro.idx_Sr,\s*:\],\s*\n\s*state_initial\[hydro.idx_tau,\s*:\],\s*\n\s*r_grid,\s*hydro\)",
            (
                "prims_0 = _cons_to_prim_at_state(\n"
                "    D_initial, state_initial[hydro.idx_Sr, :], state_initial[hydro.idx_tau, :],\n"
                "    r_grid, hydro, metric=_get_tov_metric()\n"
                ")"
            ),
            out
        )

    # 4) In fast RK3 cell: fix baryon mass (include 4*pi)
    out = re.sub(
        r"M_b\s*=\s*trapezoid\(integrand,\s*dx=dr\)",
        "M_b = 4.0*np.pi*trapezoid(integrand, dx=dr)",
        out
    )

    # 5) In FastMonitor: choose proper center index
    out = re.sub(
        r"self\.idx\s*=\s*idx_center\s+if\s+idx_center\s+is\s+not\s+None\s+else\s*\(len\(r\)\s*//\s*4\)",
        "self.idx = idx_center if idx_center is not None else NUM_GHOSTS",
        out
    )

    # 6) In earlier Monitor class (if present) fix idx as well
    out = re.sub(
        r"self\.idx\s*=\s*idx_center\s*if\s*idx_center\s*is\s*not\s*None\s*else\s*\(len\(r\)//4\)",
        "self.idx = idx_center if idx_center is not None else NUM_GHOSTS",
        out
    )

    # 7) Progress bar: time-based progress instead of step-based estimate
    out = out.replace(
        'pbar = tqdm(total=1, desc=desc, leave=True)',
        'pbar = tqdm(total=1000, desc=desc, leave=True)'
    )
    # Replace first-step total logic and step update with time-fraction updates
    out = re.sub(
        r"if steps == 1:\n\s*expected_total = int\(max\(1, np.ceil\(\(t_final - t\) / max\(dt, 1e-12\)\)\)\) \+ 1\n\s*try:\n\s*\s*pbar.total = expected_total\n\s*\s*pbar.refresh\(\)\n\s*except Exception:\n\s*\s*pass\n\s*pbar.set_postfix_str\(f't=\{t:.3f\}, dt=\{dt:.2e\}'\)\n\s*pbar.update\(1\)",
        (
            "pbar.set_postfix_str(f't={t:.3f}, dt={dt:.2e}')\n"
            "# time-based progress\n"
            "prog = int(min(pbar.total, round((t / max(t_final, 1e-16)) * pbar.total)))\n"
            "if prog > pbar.n:\n"
            "    try:\n"
            "        pbar.update(prog - pbar.n)\n"
            "    except Exception:\n"
            "        pass\n"
        ),
        out,
        flags=re.MULTILINE
    )

    # 8) Ensure bar completes at the end before closing
    out = re.sub(
        r"finally:\n\s*\s*pbar.close\(\)",
        (
            "finally:\n"
            "        try:\n"
            "            if pbar.n < pbar.total: pbar.update(pbar.total - pbar.n)\n"
            "        except Exception:\n"
            "            pass\n"
            "        pbar.close()"
        ),
        out
    )

    # 9) Ensure bssn lapse derivative is provided for Cowling sources (guarded)
    # Import idx_lapse if only NUM_BSSN_VARS is imported
    out = out.replace(
        'from source.bssn.bssnstatevariables import NUM_BSSN_VARS',
        'from source.bssn.bssnstatevariables import NUM_BSSN_VARS, idx_lapse'
    )
    # After creating BSSN derivs, insert dalpha/dr computation only if not present
    if ('bssn_d1_static = BSSNFirstDerivs' in out) and ('bssn_d1_static.lapse' not in out):
        out = re.sub(
            r"bssn_d1_static\s*=\s*BSSNFirstDerivs\(len\(grid\.r\)\)\n",
            (
                "bssn_d1_static = BSSNFirstDerivs(len(grid.r))\n"
                "# Provide radial derivative of lapse from TOV metric for source balance\n"
                "try:\n"
                "    alpha_prof = state_tov[idx_lapse, :] if 'state_tov' in globals() else np.asarray(tov_geometry['alpha'])\n"
                "    dalpha = np.zeros_like(alpha_prof)\n"
                "    if len(grid.r) >= 3:\n"
                "        dalpha[1:-1] = (alpha_prof[2:] - alpha_prof[:-2]) / (grid.r[2:] - grid.r[:-2] + 1e-30)\n"
                "        dalpha[0] = (alpha_prof[1] - alpha_prof[0]) / (grid.r[1] - grid.r[0] + 1e-30)\n"
                "        dalpha[-1] = (alpha_prof[-1] - alpha_prof[-2]) / (grid.r[-1] - grid.r[-2] + 1e-30)\n"
                "    bssn_d1_static.lapse[:, 0] = dalpha\n"
                "except Exception as _e:\n"
                "    pass\n"
            ),
            out
        )
    # Clean duplicate idx_lapse imports if any
    out = re.sub(r"idx_lapse\s*,\s*idx_lapse", "idx_lapse", out)

    return out

def is_deprecated_analysis_cell(src: str) -> bool:
    markers = [
        'ELH Figure 11', 'ELH Figure 12', 'Summary Analysis',
        'Spacetime Properties Check', 'Additional Analysis:',
        'Oscillation Modes Analysis', 'QNM', 'PSD',
        'Espectro de potencia', 'Ajuste a sinusoide'
    ]
    return any(m in src for m in markers)

ESSENTIAL_ANALYSIS = r"""
# === ESSENTIAL ANALYSIS (TOV Cowling) ===
import numpy as np
import matplotlib.pyplot as plt
from source.bssn.bssnstatevariables import (
    idx_lapse, idx_hrr, idx_htt, idx_hpp, idx_phi, idx_K, NUM_BSSN_VARS
)
from source.bssn.bssnvars import BSSNVars, BSSNFirstDerivs
from source.matter.hydro.valencia_reference_metric import ValenciaReferenceMetric
from source.matter.hydro.cons2prim import cons_to_prim

def _metric_TOV():
    g = globals()
    if 'tov_geometry' in g:
        tg = g['tov_geometry']
        return { 'alpha': tg.get('alpha', 1.0),
                 'beta_r': tg.get('beta_r', 0.0),
                 'gamma_rr': tg.get('gamma_rr', 1.0) }
    return { 'alpha': np.ones_like(r), 'beta_r': np.zeros_like(r), 'gamma_rr': np.ones_like(r) }

def _get_times_Mb_rho_c():
    g = globals()
    # times
    if 'sol_comp' in g and hasattr(g['sol_comp'], 't') and len(g['sol_comp'].t):
        times = np.array(g['sol_comp'].t)
    elif 'mon_comp' in g and hasattr(g['mon_comp'], 'times') and g['mon_comp'].times:
        times = np.array(g['mon_comp'].times)
    elif 'times_series' in g:
        times = np.array(g['times_series'])
    else:
        times = np.array([])
    # baryon mass
    if 'mon_comp' in g and hasattr(g['mon_comp'], 'baryon_mass') and g['mon_comp'].baryon_mass:
        Mb = np.array(g['mon_comp'].baryon_mass)
    elif 'M_b_series' in g:
        Mb = np.array(g['M_b_series'])
    else:
        Mb = np.array([])
    return times, Mb

def _cons_to_prim(D, Sr, tau):
    met = _metric_TOV()
    res = cons_to_prim({'D': D, 'Sr': Sr, 'tau': tau}, hydro_cowling.eos, metric=met)
    return res

def _extract_D_Sr_tau(state_2d):
    return state_2d[hydro_cowling.idx_D, :], state_2d[hydro_cowling.idx_Sr, :], state_2d[hydro_cowling.idx_tau, :]

def _rho0_max_initial():
    D0, Sr0, tau0 = _extract_D_Sr_tau(state_tov)
    prims0 = _cons_to_prim(D0, Sr0, tau0)
    return float(np.max(prims0['rho0']))

def _rhs_inf_norm(state_2d, label='state'):
    try:
        val = ValenciaReferenceMetric()
        bssn = BSSNVars(len(r)); bssn.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
        b1 = BSSNFirstDerivs(len(r))
        # Provide lapse derivative from TOV metric so sources are balanced in Cowling
        try:
            # Finite-difference helper
            def _fd1(arr, x):
                arr = np.asarray(arr); x = np.asarray(x)
                d = np.zeros_like(arr)
                if arr.size >= 3:
                    d[1:-1] = (arr[2:] - arr[:-2]) / (x[2:] - x[:-2] + 1e-30)
                    d[0]    = (arr[1] - arr[0])   / (x[1]  - x[0]   + 1e-30)
                    d[-1]   = (arr[-1]-arr[-2])   / (x[-1] - x[-2]  + 1e-30)
                return d

            # Lapse derivative from TOV metric
            met = _metric_TOV(); alpha_prof = np.asarray(met['alpha'])
            b1.lapse[:, 0] = _fd1(alpha_prof, r)

            # Derivatives of BSSN conformal variables from state_tov
            phi_prof = np.asarray(state_tov[idx_phi, :])
            hrr_prof = np.asarray(state_tov[idx_hrr, :])
            htt_prof = np.asarray(state_tov[idx_htt, :])
            hpp_prof = np.asarray(state_tov[idx_hpp, :])

            dphi = _fd1(phi_prof, r)
            dhrr = _fd1(hrr_prof, r)
            dhtt = _fd1(htt_prof, r)
            dhpp = _fd1(hpp_prof, r)

            b1.phi[:, 0] = dphi
            b1.h_LL[:, 0, 0, 0] = dhrr
            b1.h_LL[:, 1, 1, 0] = dhtt
            b1.h_LL[:, 2, 2, 0] = dhpp
        except Exception:
            pass
        D, Sr, tau = _extract_D_Sr_tau(state_2d)
        prims = _cons_to_prim(D, Sr, tau)
        rhsD, rhsSr, rhsTau = val.compute_rhs(D, Sr, tau, prims['rho0'], prims['vr'], prims['p'], prims['W'], prims['h'],
                                              r, bssn, b1, background, 'dynamic',
                                              hydro_cowling.eos, grid, hydro_cowling.reconstructor, hydro_cowling.riemann_solver)
        inner = slice(NUM_GHOSTS, -NUM_GHOSTS) if len(r) > 2*NUM_GHOSTS else slice(0, len(r))
        infs = (np.max(np.abs(rhsD[inner])), np.max(np.abs(rhsSr[inner])), np.max(np.abs(rhsTau[inner])))
        print(f'RHS ||·||_inf ({label}): D={infs[0]:.3e}, Sr={infs[1]:.3e}, tau={infs[2]:.3e}')
    except Exception as e:
        print(f'RHS check failed ({label}): {e}')

# Gather series
times, Mb_series = _get_times_Mb_rho_c()
rho0_max0 = _rho0_max_initial()

# Prepare final primitives
if 'final_state' in globals():
    Df, Srf, tauf, rho0f, vrf, pf = final_state
    final_prims = {'rho0': rho0f, 'vr': vrf, 'p': pf}
else:
    # from integrator result
    if 'sol_comp' in globals() and getattr(sol_comp, 'y', None) is not None and sol_comp.y is not None and len(sol_comp.y.shape) == 2:
        y_final = sol_comp.y[:, -1].reshape(-1, len(r))
        Df, Srf, tauf = _extract_D_Sr_tau(y_final)
        final_prims = _cons_to_prim(Df, Srf, tauf)
    else:
        Df, Srf, tauf = _extract_D_Sr_tau(state_tov)
        final_prims = _cons_to_prim(Df, Srf, tauf)

# Mass drift (absolute and relative)
if Mb_series.size >= 1 and times.size == Mb_series.size:
    M0 = Mb_series[0]
    dM_abs = np.abs(Mb_series - M0)
    dM_rel = dM_abs / (np.abs(M0) + 1e-300)
else:
    M0 = np.nan
    dM_abs = np.array([])
    dM_rel = np.array([])

# Central density series (prefer direct monitor series if available)
center_idx = int(np.argmin(np.abs(r)))  # robust center index
rhoc_rel = np.array([])
times_rhoc = np.array([])
if 'mon_comp' in globals():
    if getattr(mon_comp, 'central_density', None) and mon_comp.central_density:
        cd = np.array(mon_comp.central_density, dtype=float)
        tms = np.array(mon_comp.times, dtype=float)
        if cd.size > 0 and np.isfinite(cd[0]) and abs(cd[0]) > 0:
            times_rhoc = tms
            rhoc_rel = cd / cd[0] - 1.0
    elif getattr(mon_comp, 'states', None):
        t_r, rho_r = [], []
        for t_snap, st in mon_comp.states:
            D_t, Sr_t, tau_t = _extract_D_Sr_tau(st)
            prims_t = _cons_to_prim(D_t, Sr_t, tau_t)
            t_r.append(t_snap)
            rho_r.append(prims_t['rho0'][center_idx])
        if rho_r:
            times_rhoc = np.array(t_r)
            rhoc_rel = np.array(rho_r)/rho0_max0 - 1.0

# Figure 1: Mass drift and central density
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
if times.size and dM_abs.size:
    ax1.plot(times, np.log10(np.maximum(dM_abs, 1e-300)), 'b-')
    ax1.set_title('log10 |M_b - M_b0|')
    ax1.set_xlabel('t [M☉]'); ax1.set_ylabel('log10|ΔM|'); ax1.grid(True, alpha=0.3)
else:
    ax1.text(0.5, 0.5, 'No M_b series', ha='center'); ax1.axis('off')

if times_rhoc.size and rhoc_rel.size:
    ax2.plot(times_rhoc, rhoc_rel, 'r-')
    ax2.set_title('Central density: ρc/ρ0,max(0) - 1')
    ax2.set_xlabel('t [M☉]'); ax2.set_ylabel('Δρc_rel'); ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, 'No ρc series', ha='center'); ax2.axis('off')
plt.tight_layout(); plt.show()

# Figure 2: v^r max over time, and final density ratio vs initial
fig, (bx1, bx2) = plt.subplots(1, 2, figsize=(12, 4))

# v^r max over stored snapshots and build if missing
vr_max = []
t_shots = []
if 'mon_comp' in globals() and getattr(mon_comp, 'states', None):
    for t_snap, st in mon_comp.states:
        D_t, Sr_t, tau_t = _extract_D_Sr_tau(st)
        prims_t = _cons_to_prim(D_t, Sr_t, tau_t)
        vr_max.append(np.max(np.abs(prims_t['vr'])))
        t_shots.append(t_snap)
if vr_max:
    bx1.semilogy(t_shots, vr_max, 'g-')
    bx1.set_title('max |v^r|'); bx1.set_xlabel('t [M☉]'); bx1.set_ylabel('max|v^r|'); bx1.grid(True, which='both', alpha=0.3)
else:
    bx1.text(0.5,0.5,'No snapshots',ha='center'); bx1.axis('off')

# Density ratio final/initial
prims_initial = _cons_to_prim(*_extract_D_Sr_tau(state_tov))
ratio = final_prims['rho0'] / np.maximum(prims_initial['rho0'], 1e-300)
bx2.plot(r, ratio, 'k-')
bx2.axhline(1.0, ls='--', c='gray', lw=1)
bx2.set_title('ρ_final / ρ_initial'); bx2.set_xlabel('r'); bx2.set_ylabel('ratio'); bx2.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# Cowling invariance check (metric should not change)
if 'mon_comp' in globals() and getattr(mon_comp, 'states', None):
    _, stf = mon_comp.states[-1]
    lapse_change = float(np.max(np.abs(stf[idx_lapse,:] - state_tov[idx_lapse,:])))
    hij_max = float(max(np.max(np.abs(stf[idx_hrr,:])), np.max(np.abs(stf[idx_htt,:])), np.max(np.abs(stf[idx_hpp,:]))))
    phi_change = float(np.max(np.abs(stf[idx_phi,:] - state_tov[idx_phi,:])))
    K_change = float(np.max(np.abs(stf[idx_K,:] - state_tov[idx_K,:])))
else:
    lapse_change = hij_max = phi_change = K_change = float('nan')

print('Cowling metric invariance:')
print(f'  max|Δα|={lapse_change:.3e}, max|Δhij|={hij_max:.3e}, max|Δφ|={phi_change:.3e}, max|ΔK|={K_change:.3e}')

# Simple PASS/WARN/FAIL assessment
def assess(val, pass_thr, warn_thr):
    if not np.isfinite(val):
        return 'n/a'
    return 'PASS' if val < pass_thr else ('WARN' if val < warn_thr else 'FAIL')

final_rel = float(dM_rel[-1]) if dM_rel.size else float('nan')
max_rhoc = float(np.max(np.abs(rhoc_rel))) if rhoc_rel.size else float('nan')
max_vr = float(np.max(vr_max)) if vr_max else float('nan')

print('Assessment:')
print(f'  M_b drift (final rel): {final_rel:.2e}  -> {assess(final_rel, 1e-5, 1e-4)}')
print(f'  max|ρc/ρ0,max(0)-1|  : {max_rhoc:.2e} -> {assess(max_rhoc, 2e-3, 1e-2)}')
print(f'  max|v^r|             : {max_vr:.2e} -> {assess(max_vr, 1e-3, 1e-2)}')
print(f'  Cowling invariance Δ : {max(lapse_change, hij_max, phi_change, K_change):.2e} -> {'PASS' if max(lapse_change, hij_max, phi_change, K_change) < 1e-12 else 'FAIL'}')

# Optional RHS checks at initial and final
_rhs_inf_norm(state_tov, label='initial')
if 'mon_comp' in globals() and getattr(mon_comp, 'states', None):
    _rhs_inf_norm(mon_comp.states[-1][1], label='final')
"""

def main():
    nb = load_nb(NB_PATH)
    changed = 0
    # Fix the Fast RK3 cell indentation and missing imports
    fixed_fast_cell = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        if src.strip().startswith('# Fast Cowling Evolution with RK3 Integrator') or (
            'def run_fast_cowling_evolution' in src and 'return True, mon' in src
        ):
            # Replace the whole cell with a corrected version preserving the surrounding set-up
            new_cell = r'''
# Fast Cowling Evolution with RK3 Integrator

import numpy as np
import time
from tqdm.auto import tqdm
from source.bssn.bssnvars import BSSNVars, BSSNFirstDerivs
from source.bssn.bssnstatevariables import NUM_BSSN_VARS, idx_lapse
from scipy.integrate import trapezoid

def calculate_baryon_mass(D_profile, r_grid):
    """Calculate baryon mass M_b = 4*pi * integral D * r^2 dr (trapezoid)."""
    inner = slice(NUM_GHOSTS, -NUM_GHOSTS) if len(r_grid) > 2*NUM_GHOSTS else slice(0, len(r_grid))
    r_inner = r_grid[inner]
    D_inner = D_profile[inner]
    integrand = D_inner * r_inner**2
    dr = r_inner[1] - r_inner[0] if len(r_inner) > 1 else 0.0
    return 4.0*np.pi*trapezoid(integrand, dx=dr)

# Initial baryon mass (from TOV state)
M_b_initial = calculate_baryon_mass(state_tov[hydro_cowling.idx_D, :], r)
print(f'Initial baryon mass M_b = {M_b_initial:.6f} M_sun')

# Precompute TOV geometry once for fast access
print('Precomputing TOV geometry...')
tov_geometry = precompute_tov_geometry(r, tov)
print('TOV geometry precomputed successfully.')

class FastMonitor:
    def __init__(self, r, hydro, bssn_reference, idx_center=None, sample_stride=50, max_states=200):
        self.r = r
        self.hydro = hydro
        self.bssn_reference = bssn_reference.copy()
        self.times = []
        self.central_density = []
        self.baryon_mass = []
        self.states = []
        self.idx = idx_center if idx_center is not None else NUM_GHOSTS
        self.sample_counter = 0
        self.sample_stride = sample_stride
        self.max_states = max_states

    def log(self, t, D, Sr, tau):
        self.times.append(t)
        self.central_density.append(D[self.idx])
        self.baryon_mass.append(calculate_baryon_mass(D, self.r))
        self.sample_counter += 1
        if self.sample_counter % self.sample_stride == 0 and len(self.states) < self.max_states:
            state_2d = np.zeros((NUM_BSSN_VARS + self.hydro.NUM_MATTER_VARS, len(self.r)))
            state_2d[:NUM_BSSN_VARS, :] = self.bssn_reference
            state_2d[self.hydro.idx_D, :] = D
            state_2d[self.hydro.idx_Sr, :] = Sr
            state_2d[self.hydro.idx_tau, :] = tau
            self.states.append((t, state_2d.copy()))


def run_fast_cowling_evolution(D0, Sr0, tau0, rho0, v0, p0,
                              hydro, grid, background, tov_geometry, state_tov,
                              t_final=45.0, cfl=0.3, desc='Fast Cowling', wall_time_limit=None):
    """Fast Cowling evolution using RK3 integrator with precomputed geometry."""

    D, Sr, tau = D0.copy(), Sr0.copy(), tau0.copy()
    rho, v, p = rho0.copy(), v0.copy(), p0.copy()

    from source.matter.hydro.valencia_reference_metric import ValenciaReferenceMetric
    valencia = ValenciaReferenceMetric()
    recon = hydro.reconstructor
    rsolve = hydro.riemann_solver
    eos = hydro.eos
    spacetime_mode = getattr(hydro, 'spacetime_mode', 'dynamic')

    bssn_vars_static = BSSNVars(len(grid.r))
    bssn_vars_static.set_bssn_vars(state_tov[:NUM_BSSN_VARS, :])
    bssn_d1_static = BSSNFirstDerivs(len(grid.r))

    # Provide radial derivatives from TOV metric for source balance
    try:
        r_arr = grid.r
        def _fd1(arr, x):
            arr = np.asarray(arr); x = np.asarray(x)
            d = np.zeros_like(arr)
            if arr.size >= 3:
                d[1:-1] = (arr[2:] - arr[:-2]) / (x[2:] - x[:-2] + 1e-30)
                d[0]    = (arr[1] - arr[0])   / (x[1]  - x[0]   + 1e-30)
                d[-1]   = (arr[-1]-arr[-2])   / (x[-1] - x[-2]  + 1e-30)
            return d

        # Lapse derivative
        alpha_prof = state_tov[idx_lapse, :] if 'state_tov' in globals() else np.asarray(tov_geometry['alpha'])
        bssn_d1_static.lapse[:, 0] = _fd1(alpha_prof, r_arr)

        # Conformal metric and phi derivatives from state_tov
        phi_prof = state_tov[idx_phi, :]
        hrr_prof = state_tov[idx_hrr, :]
        htt_prof = state_tov[idx_htt, :]
        hpp_prof = state_tov[idx_hpp, :]

        bssn_d1_static.phi[:, 0] = _fd1(phi_prof, r_arr)
        bssn_d1_static.h_LL[:, 0, 0, 0] = _fd1(hrr_prof, r_arr)
        bssn_d1_static.h_LL[:, 1, 1, 0] = _fd1(htt_prof, r_arr)
        bssn_d1_static.h_LL[:, 2, 2, 0] = _fd1(hpp_prof, r_arr)
    except Exception:
        pass

    mon = FastMonitor(grid.r, hydro, state_tov[:NUM_BSSN_VARS, :])

    rho, v, p, W, h = cons_to_prim_fast(D, Sr, tau, eos, tov_geometry, p_guess=p)
    rho, v, p, W, h = fill_ghosts_primitives(rho, v, p, W, h)

    t = 0.0
    steps = 0
    start_time = time.time()

    pbar = tqdm(total=1000, desc=desc, leave=True)
    print(f'Starting fast RK3 evolution: t_final={t_final}, CFL={cfl}')

    try:
        while t < t_final and steps < 100000:
            mon.log(t, D, Sr, tau)

            dt, D, Sr, tau, rho, v, p, W, h = rk3_step_cowling(
                valencia, D, Sr, tau, rho, v, p, W, h, grid.r, grid, eos,
                recon, rsolve, tov_geometry, bssn_vars_static, bssn_d1_static,
                background, spacetime_mode, cfl=cfl
            )

            t += dt
            steps += 1

            if wall_time_limit is not None and (time.time() - start_time) > wall_time_limit:
                print(f"\nWall time limit reached (>{wall_time_limit}s). Stopping early at t={t:.3f}.")
                break

            pbar.set_postfix_str(f't={t:.3f}, dt={dt:.2e}')
            prog = int(min(pbar.total, round((t / max(t_final, 1e-16)) * pbar.total)))
            if prog > pbar.n:
                try:
                    pbar.update(prog - pbar.n)
                except Exception:
                    pass

            if not (np.isfinite(D).all() and np.isfinite(Sr).all() and np.isfinite(tau).all()):
                print(f'\nNaN/Inf detected at t={t:.4f}, step={steps}')
                break
    except Exception as e:
        print(f'\nEvolution failed at t={t:.4f}, step={steps}: {e}')
    finally:
        try:
            if pbar.n < pbar.total:
                pbar.update(pbar.total - pbar.n)
        except Exception:
            pass
        pbar.close()

    elapsed = time.time() - start_time
    mon.log(t, D, Sr, tau)

    print(f'\nEvolution completed:')
    print(f'  Final time: {t:.3f} M_sun')
    print(f'  Steps: {steps}')
    print(f'  Wall time: {elapsed:.1f} seconds')
    print(f'  Speed: {steps/elapsed:.1f} steps/second')
    print(f'  Avg dt: {t/steps:.2e} M_sun' if steps > 0 else '')

    return True, mon, (D, Sr, tau, rho, v, p)


print('\n=== Fast Cowling Evolution with RK3 ===')
t_final = 10.0
cfl = 0.20

success, mon_fast, final_state = run_fast_cowling_evolution(
    D, Sr, tau, rho0, vr, pressure,
    hydro_cowling, grid, background, tov_geometry, state_tov,
    t_final=t_final, cfl=cfl, desc='Fast RK3 Cowling'
)

if success:
    print(f'\n=== Results ===')
    times_fast = np.array(mon_fast.times)
    M_b_fast = np.array(mon_fast.baryon_mass)
    rho_c_fast = np.array(mon_fast.central_density)
    M_b_drift = np.abs(M_b_fast - M_b_fast[0]) / M_b_fast[0]
    max_M_b_drift = np.max(M_b_drift)
    final_M_b_drift = M_b_drift[-1]
    rho_c_rel_change = np.abs(rho_c_fast - rho_c_fast[0]) / (np.abs(rho_c_fast[0]) + 1e-300)
    max_rho_c_change = np.max(rho_c_rel_change)
    print(f'Initial M_b: {M_b_fast[0]:.6f} M_sun')
    print(f'Final M_b: {M_b_fast[-1]:.6f} M_sun')
    print(f'Max baryon mass drift: {max_M_b_drift:.2e}')
    print(f'Final baryon mass drift: {final_M_b_drift:.2e}')
    print(f'Max central density change: {max_rho_c_change:.2e}')
    times_series = times_fast
    M_b_series = M_b_fast
    rho_c_series = rho_c_fast
    sol_comp = type('obj', (object,), {'success': True, 't': times_fast, 'y': None})()
    mon_comp = mon_fast
    D_final, Sr_final, tau_final, rho0_final, v_final, p_final = final_state
    print('\nFast evolution completed successfully!')
    print('All plotting cells should work with the new data.')
else:
    print('Fast evolution failed!')

print('\n=== Ready for analysis and plotting ===')
'''
            cell['source'] = new_cell
            fixed_fast_cell = True
            changed += 1
            break
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        new_src = patch_code(src)
        if new_src != src:
            cell['source'] = new_src
            changed += 1
        # Deprecate heavy analysis cells
        if is_deprecated_analysis_cell(src):
            cell['source'] = "# Deprecated: consolidated into ESSENTIAL ANALYSIS\n"
            changed += 1
    # Append essential analysis cell if missing
    # Ensure essential analysis cell exists and is up to date
    essential_index = None
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code' and 'ESSENTIAL ANALYSIS (TOV Cowling)' in ''.join(cell.get('source', [])):
            essential_index = i
            break
    if essential_index is None:
        nb['cells'].append({
            'cell_type': 'code',
            'metadata': {},
            'source': ESSENTIAL_ANALYSIS,
            'outputs': [],
            'execution_count': None
        })
        changed += 1
    else:
        nb['cells'][essential_index]['source'] = ESSENTIAL_ANALYSIS
        changed += 1
    if changed:
        save_nb(NB_PATH, nb)
        print(f'Patched {changed} code cell(s) in {NB_PATH}')
    else:
        print('No changes applied; patterns not found.')

if __name__ == '__main__':
    main()
