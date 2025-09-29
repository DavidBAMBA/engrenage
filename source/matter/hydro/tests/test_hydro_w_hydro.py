#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hydro–without–Hydro: evoluciona BSSN con fuentes de materia estáticas (TOV).
# Estilo y dependencias como en test.py / TOVEvolution.py.

import os, sys, time
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
# ---------- localizar repo root para importar `source.*` ----------
def locate_repo_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (cand / 'source').is_dir():
            return cand
    return start

THIS = Path(__file__).resolve()
REPO = locate_repo_root(THIS.parent)
SRC  = REPO / 'source'
if str(REPO) not in sys.path: sys.path.append(str(REPO))
if str(SRC)  not in sys.path: sys.path.append(str(SRC))

# ---------- imports Engrenage ----------
from source.core.spacing import LinearSpacing, SpacingExtent, NUM_GHOSTS
from source.core.grid import Grid
from source.core.statevector import StateVector
from source.core.rhsevolution import get_rhs
from source.backgrounds.sphericalbackground import FlatSphericalBackground, i_r, i_t, i_p
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import PolytropicEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLERiemannSolver
from source.bssn.bssnstatevariables import (
    NUM_BSSN_VARS,
    idx_phi, idx_hrr, idx_htt, idx_hpp,
    idx_K, idx_arr, idx_att, idx_app,
    idx_lambdar, idx_shiftr, idx_br, idx_lapse
)

from scipy.integrate import odeint
from scipy.interpolate import interp1d

# -------------------- utilidades --------------------
def fill_ghosts_primitives(rho, v, p, ng=NUM_GHOSTS):
    """Paridades en r≈0: rho/p pares; v impar. Cero-gradiente en borde externo."""
    rho = np.asarray(rho).copy(); v = np.asarray(v).copy(); p = np.asarray(p).copy()
    N = len(rho)
    # centro
    for i in range(ng):
        mir = 2*ng - 1 - i
        rho[i] = rho[mir]     # par
        p[i]   = p[mir]       # par
        v[i]   = -v[mir]      # impar
    # borde externo
    last = N - ng - 1
    for k in range(1, ng+1):
        idx = last + k
        rho[idx] = rho[last]
        p[idx]   = p[last]
        v[idx]   = v[last]
    return rho, v, p

def primitives_to_conserved(rho0, v, p, eos):
    eps = eos.eps_from_rho_p(rho0, p)
    h   = 1.0 + eps + p/np.maximum(rho0, 1e-300)
    W   = 1.0/np.sqrt(np.maximum(1.0 - v*v, 1e-16))
    D   = rho0 * W
    Sr  = rho0 * h * W*W * v
    tau = rho0 * h * W*W - p - D
    return D, Sr, tau, W, h

# -------------------- solver TOV (como en tu TOVEvolution) --------------------
def solve_tov_ode(r_max, rho_central, gamma, K=1.0, n_points=1000):
    """Devuelve dict con r, rho, pressure, mass, alpha, R_star, M_star."""
    p_central = K * rho_central**gamma

    def e_from_p(P):
        if P <= 0: return 1e-15
        rho = (P / K)**(1.0/gamma)
        eps = P / ((gamma - 1.0) * rho)
        return rho * (1.0 + eps)

    def tov_rhs(y, r):
        m, p = y
        if r < 1e-12: return np.array([0.0, 0.0])
        e = e_from_p(p)
        dmdr = 4.0 * np.pi * r**2 * e
        denom = r * (r - 2.0*m)
        dpdr = 0.0 if abs(denom) < 1e-15 else -(e + p) * (m + 4.0*np.pi * r**3 * p) / denom
        return np.array([dmdr, dpdr])

    r_start = 1e-6
    m_start = (4.0*np.pi/3.0) * e_from_p(p_central) * r_start**3
    y0 = np.array([m_start, p_central])

    r_grid = np.linspace(r_start, r_max, n_points)
    y  = odeint(lambda yy, rr: tov_rhs(yy, rr), y0, r_grid, atol=1e-12, rtol=1e-10, mxstep=10000)
    m  = y[:,0]; p = y[:,1]

    # superficie donde p≈0
    i_surf = np.argmax(p <= 1e-12)
    if i_surf == 0: i_surf = len(r_grid)-1
    r = r_grid[:i_surf+1]
    p = np.maximum(p[:i_surf+1], 1e-15)
    rho = (p/K)**(1.0/gamma)
    m   = m[:i_surf+1]

    # lapse aproximado (suficiente para el test)
    alpha = np.ones_like(r)
    for i, ri in enumerate(r):
        if ri > 1e-12 and m[i] > 0.0:
            fac = 1.0 - 2.0*m[i] / ri
            alpha[i] = np.sqrt(fac) if fac > 1e-12 else (alpha[i-1] if i>0 else 1.0)

    return dict(r=r, rho=rho, pressure=p, mass=m, alpha=alpha,
                R_star=float(r[-1]), M_star=float(m[-1]))

# -------------------- Progress monitor (dummy) --------------------
class DummyProgress:
    def update(self, *args, **kwargs):
        pass



# -------------------- estado inicial BSSN + materia --------------------
def build_initial_state_hwh(grid: Grid, hydro: PerfectFluid, background, r, tov, rho0, pressure, velocity):
    """
    Métrica conformemente plana + lapse de TOV, shift=0, K=a_ij=Λ^r=0.
    Materia en equilibrio hidrostático.
    """
    state = np.zeros((grid.NUM_VARS, grid.N))

    # BSSN variables: conformally flat + TOV lapse
    state[idx_phi,   :] = 0.0
    state[idx_hrr,   :] = 0.0
    state[idx_htt,   :] = 0.0
    state[idx_hpp,   :] = 0.0
    state[idx_K,     :] = 0.0
    state[idx_arr,   :] = 0.0
    state[idx_att,   :] = 0.0
    state[idx_app,   :] = 0.0
    state[idx_lambdar,:] = 0.0
    state[idx_shiftr, :] = 0.0
    state[idx_br,     :] = 0.0

    # TOV lapse
    a_interp = interp1d(tov['r'], tov['alpha'], kind='linear', bounds_error=False,
                        fill_value=(tov['alpha'][0], tov['alpha'][-1]))
    state[idx_lapse, :] = a_interp(r)

    # Matter: convert primitives to conservatives
    eps = hydro.eos.eps_from_rho_p(rho0, pressure)
    h = 1.0 + eps + pressure/np.maximum(rho0, 1e-300)
    W = 1.0/np.sqrt(np.maximum(1.0 - velocity*velocity, 1e-16))
    D = rho0 * W
    Sr = rho0 * h * W*W * velocity
    tau = rho0 * h * W*W - pressure - D

    state[hydro.idx_D,   :] = D
    state[hydro.idx_Sr,  :] = Sr
    state[hydro.idx_tau, :] = tau

    grid.fill_boundaries(state)
    return state

# -------------------- Evolución HWH usando fuentes TOV exactas --------------------
def rk3_step_hwh(state, grid: Grid, background, matter, tov_data, cfl=0.25):
    """
    Un paso RK3 para HWH: evoluciona BSSN, inserta fuentes TOV exactas.
    """
    def update_matter_from_tov(state_arr, tov_data, grid, matter):
        """Actualiza variables de materia desde solución TOV analítica."""
        r = grid.r

        # Interpolar solución TOV al grid actual
        rho_interp = interp1d(tov_data['r'], tov_data['rho'], kind='linear',
                             bounds_error=False, fill_value=1e-15)
        p_interp = interp1d(tov_data['r'], tov_data['pressure'], kind='linear',
                           bounds_error=False, fill_value=1e-15)

        # Primitivas TOV (estáticas: v=0)
        rho0 = np.maximum(rho_interp(r), 1e-15)
        pressure = np.maximum(p_interp(r), 1e-15)
        velocity = np.zeros_like(r)

        # Aplicar condiciones de borde (paridades)
        rho0, velocity, pressure = fill_ghosts_primitives(rho0, velocity, pressure)

        # Convertir a conservadas
        eps = matter.eos.eps_from_rho_p(rho0, pressure)
        h = 1.0 + eps + pressure/np.maximum(rho0, 1e-300)
        W = 1.0/np.sqrt(np.maximum(1.0 - velocity*velocity, 1e-16))
        D = rho0 * W
        Sr = rho0 * h * W*W * velocity
        tau = rho0 * h * W*W - pressure - D

        # Actualizar estado
        state_arr[matter.idx_D,   :] = D
        state_arr[matter.idx_Sr,  :] = Sr
        state_arr[matter.idx_tau, :] = tau

        return state_arr

    dummy_progress = DummyProgress()
    time_state = [0.0, 0.1]  # [last_t, deltat] format

    # Determinar dt del CFL
    r = grid.r
    dr = r[1] - r[0]
    dt = cfl * dr

    # RK3 stages
    # Stage 1: actualizar materia desde TOV, luego computar RHS
    s1 = state.copy()
    update_matter_from_tov(s1, tov_data, grid, matter)

    rhs_full = get_rhs(0.0, s1.flatten(), grid, background, matter, dummy_progress, time_state)
    rhs = rhs_full.reshape(grid.NUM_VARS, -1)

    # Solo evolucionar BSSN (congelar materia)
    rhs[matter.idx_D,   :] = 0.0
    rhs[matter.idx_Sr,  :] = 0.0
    rhs[matter.idx_tau, :] = 0.0

    s1 = state + dt * rhs
    grid.fill_boundaries(s1)

    # Stage 2: actualizar materia desde TOV
    update_matter_from_tov(s1, tov_data, grid, matter)

    rhs_full = get_rhs(0.0, s1.flatten(), grid, background, matter, dummy_progress, time_state)
    rhs = rhs_full.reshape(grid.NUM_VARS, -1)

    rhs[matter.idx_D,   :] = 0.0
    rhs[matter.idx_Sr,  :] = 0.0
    rhs[matter.idx_tau, :] = 0.0

    s2 = 0.75*state + 0.25*(s1 + dt*rhs)
    grid.fill_boundaries(s2)

    # Stage 3: actualizar materia desde TOV
    update_matter_from_tov(s2, tov_data, grid, matter)

    rhs_full = get_rhs(0.0, s2.flatten(), grid, background, matter, dummy_progress, time_state)
    rhs = rhs_full.reshape(grid.NUM_VARS, -1)

    rhs[matter.idx_D,   :] = 0.0
    rhs[matter.idx_Sr,  :] = 0.0
    rhs[matter.idx_tau, :] = 0.0

    sn = (1.0/3.0)*state + (2.0/3.0)*(s2 + dt*rhs)
    grid.fill_boundaries(sn)

    # Asegurar que el estado final tenga materia TOV correcta
    update_matter_from_tov(sn, tov_data, grid, matter)

    return dt, sn


# Modificaciones para tu función run_hwh_test() para guardar más datos

def run_hwh_test_enhanced(
    gamma=2.0, K=100.0, rho_central=1.28e-3,
    r_max=16.0, dr=0.02, t_final=20.0, cfl=0.25,
    atmosphere=1e-16, progress=True, save_interval=10
):
    """
    Versión mejorada que guarda más datos para plotting.
    
    Parameters:
    -----------
    save_interval : int
        Cada cuántos steps guardar datos adicionales
    """
    # ... (código inicial igual) ...
    
    # grid & background
    spacing = LinearSpacing(int(r_max/dr), r_max, SpacingExtent.HALF)
    r = spacing[0]
    hydro = PerfectFluid(
        eos=PolytropicEOS(K=K, gamma=gamma),
        spacetime_mode='dynamic',
        atmosphere_rho=atmosphere,
        reconstructor=create_reconstruction("mp5"),
        riemann_solver=HLLERiemannSolver()
    )
    state_vec = StateVector(hydro)
    grid = Grid(spacing, state_vec)
    background = FlatSphericalBackground(r)

    # TOV en malla auxiliar -> interpola al grid
    tov = solve_tov_ode(r_max=r[-1], rho_central=rho_central, gamma=gamma, K=K, n_points=2048)
    rho_i = interp1d(tov['r'], tov['rho'], kind='linear', bounds_error=False, fill_value=atmosphere)
    p_i   = interp1d(tov['r'], tov['pressure'], kind='linear', bounds_error=False, fill_value=atmosphere)
    rho0  = np.maximum(rho_i(r), atmosphere)
    p     = np.maximum(p_i(r), atmosphere)
    v     = np.zeros_like(r)
    rho0, v, p = fill_ghosts_primitives(rho0, v, p)
    D, Sr, tau, W, h = primitives_to_conserved(rho0, v, p, hydro.eos)

    # estado inicial BSSN (conforme‑plano + lapse TOV)
    state = build_initial_state_hwh(grid, hydro, background, r, tov, rho0, p, v)

    # Arrays para guardar evolución temporal
    t = 0.0; steps = 0
    center = NUM_GHOSTS
    
    # Datos en el centro
    lapse_c = [state[idx_lapse, center]]
    phi_c   = [state[idx_phi,   center]]
    K_c     = [state[idx_K,     center]]
    hrr_c   = [state[idx_hrr,   center]]
    
    # Perfiles completos (guardamos cada save_interval)
    times_detailed = [0.0]
    states_detailed = [state.copy()]
    
    times = [0.0]
    
    if progress: print("Hydro-without-Hydro evolution: starting...")

    while t < t_final and steps < 200000:
        dt, state = rk3_step_hwh(state, grid, background, hydro, tov, cfl=cfl)
        t += dt; steps += 1
        
        # Guardar datos en el centro cada paso
        if steps % 10 == 0:
            lapse_c.append(state[idx_lapse, center])
            phi_c.append(state[idx_phi, center])
            K_c.append(state[idx_K, center])
            hrr_c.append(state[idx_hrr, center])
            times.append(t)
            
        # Guardar estados completos menos frecuentemente
        if steps % save_interval == 0:
            times_detailed.append(t)
            states_detailed.append(state.copy())
            
        if progress and steps % 100 == 0:
            print(f" t={t:.3f}  α_c={lapse_c[-1]:.6f}  φ_c={phi_c[-1]:+.3e}  K_c={K_c[-1]:+.3e}")

    if progress:
        print(f"Done. steps={steps}, t≈{t:.3f}")
        print(f"Center: α(0)={lapse_c[0]:.6f} → α(t)={lapse_c[-1]:.6f},  Δα={lapse_c[-1]-lapse_c[0]:+.3e}")
        print(f"Center: φ(0)={phi_c[0]:+.3e} → φ(t)={phi_c[-1]:+.3e}, Δφ={phi_c[-1]-phi_c[0]:+.3e}")
        print(f"Center: K(0)={K_c[0]:+.3e} → K(t)={K_c[-1]:+.3e}, ΔK={K_c[-1]-K_c[0]:+.3e}")

    return dict(
        time=t, steps=steps, r=r, state=state, 
        # Evolución temporal en el centro
        lapse_center=np.array(lapse_c),
        phi_center=np.array(phi_c), 
        K_center=np.array(K_c),
        hrr_center=np.array(hrr_c),
        times=np.array(times), 
        # Estados completos
        times_detailed=np.array(times_detailed),
        states_detailed=states_detailed,
        # Datos iniciales
        tov=tov, rho0=rho0, pressure0=p, velocity0=v,
        # Parámetros
        gamma=gamma, K_eos=K, rho_central=rho_central
    )

# Función de plotting mejorada para usar los nuevos datos
def plot_hwh_enhanced(result, save_plots=True, plot_dir="plots"):
    """Plot con los nuevos datos guardados."""
    import os
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
    
    # Extraer datos
    times = result['times']
    times_detailed = result['times_detailed']
    states_detailed = result['states_detailed']
    r = result['r']
    tov = result['tov']
    
    # ==================== PLOT 1: Evolución temporal (como Fig. 1 del paper) ====================
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # φ vs tiempo
    ax1.plot(times, result['phi_center'], 'b-', linewidth=2, label='φ(center)')
    ax1.set_ylabel('φ', fontsize=12)
    ax1.set_title('Hydro-without-Hydro: Evolution at Center (like Fig. 1 of paper)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # K vs tiempo (¡esta es la figura clave del paper!)
    ax2.plot(times, result['K_center'], 'g-', linewidth=2, label='K(center)')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='K = 0 (equilibrium)')
    ax2.set_ylabel('K (Trace of extrinsic curvature)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # lapse vs tiempo
    ax3.plot(times, result['lapse_center'], 'r-', linewidth=2, label='α(center)')
    ax3.axhline(y=result['lapse_center'][0], color='r', linestyle='--', alpha=0.5, label='Initial α')
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('Lapse α', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{plot_dir}/hwh_time_evolution_paper_style.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # ==================== PLOT 2: Snapshots de evolución ====================
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Seleccionar algunos tiempos para mostrar
    n_snapshots = min(4, len(states_detailed))
    indices = np.linspace(0, len(states_detailed)-1, n_snapshots, dtype=int)
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, idx in enumerate(indices):
        color = colors[i]
        state_snap = states_detailed[idx]
        time_snap = times_detailed[idx]
        label = f't = {time_snap:.2f}'
        
        # Lapse
        ax1.plot(r, state_snap[idx_lapse, :], color=color, linewidth=2, label=label)
        
        # φ  
        ax2.plot(r, state_snap[idx_phi, :], color=color, linewidth=2, label=label)
        
        # K
        ax3.plot(r, state_snap[idx_K, :], color=color, linewidth=2, label=label)
        
        # h_rr
        ax4.plot(r, state_snap[idx_hrr, :], color=color, linewidth=2, label=label)
    
    # Agregar solución TOV inicial donde sea relevante
    if 'r' in tov and 'alpha' in tov:
        alpha_interp = interp1d(tov['r'], tov['alpha'], kind='linear', 
                               bounds_error=False, fill_value=(tov['alpha'][0], tov['alpha'][-1]))
        ax1.plot(r, alpha_interp(r), 'k--', linewidth=2, alpha=0.7, label='TOV initial')
    
    ax1.set_xlabel('r'); ax1.set_ylabel('Lapse α'); ax1.set_title('Lapse Evolution'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('r'); ax2.set_ylabel('φ'); ax2.set_title('Conformal Factor Evolution'); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax3.set_xlabel('r'); ax3.set_ylabel('K'); ax3.set_title('Extrinsic Curvature Trace'); ax3.legend(); ax3.grid(True, alpha=0.3)
    ax4.set_xlabel('r'); ax4.set_ylabel('h_rr'); ax4.set_title('Conformal Metric h_rr'); ax4.legend(); ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{plot_dir}/hwh_evolution_snapshots.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # ==================== PLOT 3: Error/Drift Analysis ====================
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Drifts
    phi_drift = result['phi_center'] - result['phi_center'][0]
    K_drift = result['K_center'] - result['K_center'][0]  # Debería mantenerse cerca de 0
    lapse_drift = result['lapse_center'] - result['lapse_center'][0]
    
    ax1.plot(times, phi_drift, 'b-', linewidth=2)
    ax1.set_xlabel('Time'); ax1.set_ylabel('Δφ'); ax1.set_title('Conformal Factor Drift')
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0)); ax1.grid(True, alpha=0.3)
    
    ax2.plot(times, K_drift, 'g-', linewidth=2)
    ax2.set_xlabel('Time'); ax2.set_ylabel('ΔK'); ax2.set_title('Extrinsic Curvature Trace Drift')
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0)); ax2.grid(True, alpha=0.3)
    
    ax3.plot(times, lapse_drift, 'r-', linewidth=2)
    ax3.set_xlabel('Time'); ax3.set_ylabel('Δα'); ax3.set_title('Lapse Drift')
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0)); ax3.grid(True, alpha=0.3)
    
    # Conservación (ejemplo: norma L2 de variables)
    l2_norms = []
    for state_snap in states_detailed:
        # Calcular norma L2 de φ como medida de "conservación"
        norm = np.sqrt(np.mean(state_snap[idx_phi, :]**2))
        l2_norms.append(norm)
    
    ax4.plot(times_detailed, l2_norms, 'm-', linewidth=2)
    ax4.set_xlabel('Time'); ax4.set_ylabel('||φ||₂'); ax4.set_title('L2 Norm of φ')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{plot_dir}/hwh_error_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return result

# ==================== Ejemplo de uso completo ====================
if __name__ == "__main__":
    # Parámetros conservadores para estabilidad
    result = run_hwh_test_enhanced(
        gamma=2.0,           # Polytropic index n=1 → γ=1+1/n=2
        K=100.0,             # Polytropic constant
        rho_central=5e-4,    # Central density más baja
        r_max=8.0,           # Domain size más pequeño
        dr=0.05,             # Spatial resolution más gruesa
        t_final=2.0,         # Evolution time más corto
        cfl=0.05,            # CFL factor muy conservador
        save_interval=10,    # Save full state every 10 steps
        progress=True
    )
    
    # Generar plots estilo paper
    plot_hwh_enhanced(result, save_plots=True)

