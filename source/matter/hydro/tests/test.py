#!/usr/bin/env python3
# test.py — Suite mínima para validar Valencia FULL reference-metric en 1D esférico (Minkowski)

import os
import sys
import types
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add source path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# --- imports absolutos coherentes ---
from source.core.spacing import NUM_GHOSTS
from source.backgrounds.sphericalbackground import i_r
from source.bssn.tensoralgebra import get_bar_gamma_LL

from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.reconstruction import MinmodReconstruction, create_reconstruction
from source.matter.hydro.riemann import HLLERiemannSolver
from source.matter.hydro.cons2prim import cons_to_prim, prim_to_cons
from source.matter.hydro.valencia_reference_metric import ValenciaReferenceMetric
from source.bssn.tensoralgebra import SPACEDIM
from source.backgrounds.sphericalbackground import FlatSphericalBackground


class Grid:
    def __init__(self, dx):
        self.dr = float(dx)
        

class _DummyBSSNVars:
    #Placeholders no usados en Minkowski fijo.
    def __init__(self, N):
        import numpy as np
        self.lapse   = np.ones(N)
        self.shift_U = np.zeros((N,3,3))
        self.phi     = np.zeros(N)
        self.K       = np.zeros(N)

class _DummyBSSND1:
    def __init__(self, N):
        self.lapse   = np.zeros((N,3))
        self.shift_U = np.zeros((N,3,3))
        self.phi     = np.zeros((N,3))

def build_grid(n_interior=256, r_min=1.0e-3, r_max=1.0, ng=NUM_GHOSTS):
    #Centros de celda uniformes + ghosts extrapolados linealmente.
    Nin = int(n_interior)
    r_in = np.linspace(r_min, r_max, Nin)
    dr = (r_max - r_min) / (Nin - 1)
    # Extiende a la izquierda (ghosts) y derecha por extrapolación
    left_ghosts  = r_in[0]  - dr*np.arange(ng,0,-1)
    right_ghosts = r_in[-1] + dr*np.arange(1,ng+1)
    r_full = np.concatenate([left_ghosts, r_in, right_ghosts])
    return r_full, Grid(dr), Nin

def fill_ghosts_primitives(rho, v, p, ng=NUM_GHOSTS):
    """Paridades correctas en r≈0: rho/p pares; v impar. Outflow en borde derecho."""
    N = len(rho)
    # lado izquierdo (centro)
    for i in range(ng):
        mir = 2*ng - 1 - i
        rho[i] = rho[mir]     # par
        p[i]   = p[mir]       # par
        v[i]   = -v[mir]      # impar
    # lado derecho (outflow/zero-gradient)
    last = N - ng - 1
    for k in range(1, ng+1):
        idx = last + k
        rho[idx] = rho[last]
        p[idx]   = p[last]
        v[idx]   = v[last]
    return rho, v, p

def to_conserved(rho0, v, p, eos):
    eps = eos.eps_from_rho_p(rho0, p)
    h   = 1.0 + eps + p/np.maximum(rho0, 1e-300)
    W   = 1.0/np.sqrt(np.maximum(1.0 - v*v, 1e-16))
    D   = rho0 * W
    Sr  = rho0 * h * W*W * v
    tau = rho0 * h * W*W - p - D
    return D, Sr, tau 

""" def to_conserved2(rho0, v, p, eos):
    res =prim_to_cons(rho0, v, p, eos, np.zeros_like(rho0), np.ones_like(rho0))
    return res['D'], res['Sr'], res['tau']
 """

def to_primitives(D, Sr, tau, eos, p_guess=None):
    res = cons_to_prim(
        (D, Sr, tau), eos,
        metric=(np.ones_like(D), np.zeros_like(D), np.ones_like(D)),
        p_guess=p_guess,
    )
    return res['rho0'], res['vr'], res['p']

def max_signal_speed(rho0, v, p, eos, cfl_guard=1e-6):
    eps = eos.eps_from_rho_p(rho0, p)
    h   = 1.0 + eps + p/np.maximum(rho0, 1e-300)
    cs2 = np.clip(eos.gamma * p / np.maximum(rho0*h, 1e-300), 0.0, 1.0 - 1e-10)
    cs  = np.sqrt(cs2)
    return np.max(np.abs(v) + cs) + cfl_guard

def rk3_step(valencia, D, Sr, tau, rho0, v, p, r, grid, eos, recon, rsolve, cfl=0.5,
             spacetime_mode="fixed_minkowski"):
    """Una etapa RK3 Shu–Osher usando compute_rhs (full approach)."""
    # dt CFL
    amax = max_signal_speed(rho0, v, p, eos)
    dt = cfl * grid.dr / amax

    # Dummy BSSN (no usado en Minkowski, pero la firma lo pide)
    bssn_vars = _DummyBSSNVars(len(r))
    bssn_d1   = _DummyBSSND1(len(r))
    #background = None
    background = FlatSphericalBackground(r)


    # Stage 1
    rhsD, rhsSr, rhsTau = valencia.compute_rhs(D, Sr, tau, rho0, v, p, 
                                               W=None, h=None,
                                               r=r, bssn_vars=bssn_vars, bssn_d1=bssn_d1,
                                               background=background, spacetime_mode=spacetime_mode,
                                               eos=eos, grid=grid, reconstructor=recon, riemann_solver=rsolve)
    D1   = D   + dt*rhsD
    Sr1  = Sr  + dt*rhsSr
    tau1 = tau + dt*rhsTau
    rho1, v1, p1 = to_primitives(D1, Sr1, tau1, eos, p_guess=p)
    rho1, v1, p1 = fill_ghosts_primitives(rho1, v1, p1)

    # Stage 2
    rhsD, rhsSr, rhsTau = valencia.compute_rhs(D1, Sr1, tau1, rho1, v1, p1, 
                                               W=None, h=None,
                                               r=r, bssn_vars=bssn_vars, bssn_d1=bssn_d1,
                                               background=background, spacetime_mode=spacetime_mode,
                                               eos=eos, grid=grid, reconstructor=recon, riemann_solver=rsolve)
    D2   = 0.75*D   + 0.25*(D1  + dt*rhsD)
    Sr2  = 0.75*Sr  + 0.25*(Sr1 + dt*rhsSr)
    tau2 = 0.75*tau + 0.25*(tau1+ dt*rhsTau)
    rho2, v2, p2 = to_primitives(D2, Sr2, tau2, eos, p_guess=p1)
    rho2, v2, p2 = fill_ghosts_primitives(rho2, v2, p2)

    # Stage 3
    rhsD, rhsSr, rhsTau = valencia.compute_rhs(D2, Sr2, tau2, rho2, v2, p2, 
                                               W=None, h=None,
                                               r=r, bssn_vars=bssn_vars, bssn_d1=bssn_d1,
                                               background=background, spacetime_mode=spacetime_mode,
                                               eos=eos, grid=grid, reconstructor=recon, riemann_solver=rsolve)
    Dn   = (1.0/3.0)*D   + (2.0/3.0)*(D2   + dt*rhsD)
    Snn  = (1.0/3.0)*Sr  + (2.0/3.0)*(Sr2  + dt*rhsSr)
    taun = (1.0/3.0)*tau + (2.0/3.0)*(tau2 + dt*rhsTau)
    rhon, vn, pn = to_primitives(Dn, Snn, taun, eos, p_guess=p2)
    rhon, vn, pn = fill_ghosts_primitives(rhon, vn, pn)

    return dt, Dn, Snn, taun, rhon, vn, pn

def volume_integrals(D, tau, r, grid):
    """Masa y energía total con peso 4π r^2."""
    ng = NUM_GHOSTS
    rin = r[ng:-ng]
    Din = D[ng:-ng]
    taun = tau[ng:-ng]
    mass  = 4*np.pi * np.sum(Din   * rin*rin) * grid.dr
    energ = 4*np.pi * np.sum(taun+Din) * grid.dr
    return mass, energ

# =========================
# TESTS
# =========================
def test_uniform_state():
    print("\n" + "="*60)
    print("TEST 1: Estado uniforme (Minkowski, FULL reference-metric)")
    print("="*60)

    r, grid, Nin = build_grid(n_interior=256, r_min=1e-3, r_max=1.0)
    N = len(r)
    eos  = IdealGasEOS(gamma=1.4)
    recon= MinmodReconstruction()
    rsolve = HLLERiemannSolver()
    vr   = np.zeros(N)
    rho0 = np.ones(N) * 1.0
    p    = np.ones(N) * 0.1
    rho0, vr, p = fill_ghosts_primitives(rho0, vr, p)
    D, Sr, tau = to_conserved(rho0, vr, p, eos)

    val = ValenciaReferenceMetric()
    t, Tfinal = 0.0, 0.1
    steps = 0
    while t < Tfinal and steps < 2000:
        dt, D, Sr, tau, rho0, vr, p = rk3_step(val, D, Sr, tau, rho0, vr, p, r, grid, eos, recon, rsolve, cfl=0.5)
        t += dt; steps += 1

    # Variación relativa (interior)
    ng = NUM_GHOSTS
    vD   = np.max(np.abs(D[ng:-ng]   - 1.0))
    vSr  = np.max(np.abs(Sr[ng:-ng]  - 0.0))
    vtau = np.max(np.abs(tau[ng:-ng] - (1.0*0.0 + 0.1*0.0)))  # ~0 en uniforme
    print(f"max|ΔD|={vD:.3e}, max|Sr|={vSr:.3e}, max|τ|≈{vtau:.3e}")
    ok = (vD < 5e-8) and (vSr < 5e-8)
    print("✓ PASA" if ok else "✗ FALLA")
    return ok

def test_cons2prim_roundtrip():
    print("\n" + "="*60)
    print("TEST 2: Conversión Conservadas ↔ Primitivas (roundtrip)")
    print("="*60)
    N = 128 + 2*NUM_GHOSTS
    eos = IdealGasEOS(gamma=1.4)
    rho0 = np.random.uniform(0.3e4, 2.0e-2, N)
    v    = np.random.uniform(-0.8, 0.8, N)
    p    = np.random.uniform(0.05e4, 1.0e-1, N)
    rho0, v, p = fill_ghosts_primitives(rho0, v, p)
    D, Sr, tau = to_conserved(rho0, v, p, eos)
    r2, v2, p2 = to_primitives(D, Sr, tau, eos)
    e_rho = np.max(np.abs(r2 - rho0))
    e_v   = np.max(np.abs(v2   - v))
    e_p   = np.max(np.abs(p2   - p))
    print(f"max|Δrho|={e_rho:.2e}, max|Δv|={e_v:.2e}, max|Δp|={e_p:.2e}")
    ok = (e_rho < 1e-9) and (e_v < 1e-9) and (e_p < 1e-9)
    print("✓ PASA" if ok else "✗ FALLA")
    return ok

def test_riemann_sod():
    print("\n" + "="*60)
    print("TEST 3: Sod radial - Comparación reconstructores (Minkowski, FULL reference-metric)")
    print("="*60)

    # Configuración base
    r, grid, Nin = build_grid(n_interior=200, r_min=1e-3, r_max=1.0)
    N = len(r)
    eos = IdealGasEOS(gamma=1.4)
    rsolve = HLLERiemannSolver()

    # Discontinuidad en el punto medio del dominio interior
    r_mid = 0.5*(r[NUM_GHOSTS] + r[-NUM_GHOSTS-1])
    rho0_base = np.where(r < r_mid, 10.0, 1.0)
    p_base = np.where(r < r_mid, 40.0/3.0, 1.0e-6)
    v_base = np.zeros(N)

    # Lista de métodos de reconstrucción a probar
    methods = ["minmod", "mp5"]#, "weno5", "wenoz", "mp5_hires"]
    colors = ["lightcoral", "lightblue",]# "lightgreen", "moccasin", "black"]
    labels = ["MINMOD", "MP5"]#, "WENO5", "WENO-Z", "MP5 (Hi-Res)"]
    linestyles = ["--", "--"]#, "--", "--", "-"]  # Punteadas para baja res, continua para alta res

    # Guardar resultados para cada método
    results = {}

    for i, method in enumerate(methods):
        print(f"\nEjecutando con {labels[i]}...")

        # Para MP5 alta resolución, usar más puntos
        if method == "mp5_hires":
            r_hires, grid_hires, Nin_hires = build_grid(n_interior=200, r_min=1e-3, r_max=1.0)
            N_hires = len(r_hires)

            # Discontinuidad en el punto medio del dominio interior
            r_mid_hires = 0.5*(r_hires[NUM_GHOSTS] + r_hires[-NUM_GHOSTS-1])
            rho0 = np.where(r_hires < r_mid_hires, 10.0, 1.0)
            p = np.where(r_hires < r_mid_hires, 40.0/3.0, 1.0e-6)
            v = np.zeros(N_hires)

            # Aplicar paridades/outflow
            rho0, v, p = fill_ghosts_primitives(rho0, v, p)

            # Usar grid de alta resolución
            r_current = r_hires
            grid_current = grid_hires
            actual_method = "mp5"
        else:
            # Resetear condiciones iniciales con resolución normal
            rho0 = rho0_base.copy()
            p = p_base.copy()
            v = v_base.copy()

            # Aplicar paridades/outflow
            rho0, v, p = fill_ghosts_primitives(rho0, v, p)

            # Usar grid normal
            r_current = r
            grid_current = grid
            actual_method = method

        # Conservadas iniciales
        D, Sr, tau = to_conserved(rho0, v, p, eos)

        # Crear reconstructor específico
        recon = create_reconstruction(actual_method)
        val = ValenciaReferenceMetric()

        # Evolución temporal
        t, Tfinal = 0.0, 0.2
        steps = 0
        while t < Tfinal and steps < 5000:
            dt, D, Sr, tau, rho0, v, p = rk3_step(
                val, D, Sr, tau, rho0, v, p, r_current, grid_current, eos, recon, rsolve, cfl=0.45
            )
            t += dt
            steps += 1

        # Guardar resultados
        ng = NUM_GHOSTS
        rin = r_current[ng:-ng]
        results[method] = {
            'r': rin.copy(),
            'rho': rho0[ng:-ng].copy(),
            'p': p[ng:-ng].copy(),
            'v': v[ng:-ng].copy(),
            't': t,
            'steps': steps
        }

        # Métricas de calidad
        rho_in = rho0[ng:-ng]
        grad = np.gradient(rho_in, rin)
        variation = np.std(rho_in)/np.mean(rho_in)
        contact = np.any(np.abs(grad) > 0.5)
        print(f"  {method}: t≈{t:.3f}, pasos={steps}, variación ρ={variation:.3f}, contacto={contact}")

    # ======================
    # PLOTEO COMPARATIVO CON ZOOMS
    # ======================
    # Estados iniciales (usar grid normal para referencia)
    rho0_init = np.where(r < r_mid, 10.0, 1.0)
    p_init = np.where(r < r_mid, 40.0/3.0, 1.0e-6)
    v_init = np.zeros(N)
    rho0_init, v_init, p_init = fill_ghosts_primitives(rho0_init, v_init, p_init)

    rho0_init_in = rho0_init[ng:-ng]
    p_init_in = p_init[ng:-ng]
    v_init_in = v_init[ng:-ng]
    rin_ref = r[ng:-ng]

    avg_time = np.mean([results[m]['t'] for m in methods])

    # ===== FIGURA 1: DENSIDAD =====
    fig1, (ax1, ax1_zoom) = plt.subplots(1, 2, figsize=(14, 6))

    # Dominio completo
    ax1.plot(rin_ref, rho0_init_in, 'k:', label='ρ inicial', alpha=0.7, linewidth=2)
    for i, method in enumerate(methods):
        ax1.plot(results[method]['r'], results[method]['rho'],
                color=colors[i], label=labels[i], linewidth=2, linestyle=linestyles[i])
    ax1.set_xlabel('r')
    ax1.set_ylabel('Densidad ρ₀')
    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Densidad - Dominio Completo')

    # Zoom en shock
    ax1_zoom.plot(rin_ref, rho0_init_in, 'k:', label='ρ inicial', alpha=0.7, linewidth=2)
    for i, method in enumerate(methods):
        ax1_zoom.plot(results[method]['r'], results[method]['rho'],
                     color=colors[i], label=labels[i], linewidth=2, linestyle=linestyles[i])
    ax1_zoom.set_xlabel('r')
    ax1_zoom.set_ylabel('Densidad ρ₀')
    ax1_zoom.set_xlim(0.6, 0.7)
    ax1_zoom.set_ylim(4.5, 7.5)
    ax1_zoom.grid(True, alpha=0.3)
    ax1_zoom.legend()
    ax1_zoom.set_title('Densidad - Zoom Shock')

    fig1.suptitle(f"Sod: Densidad (t ≈ {avg_time:.3f})")
    plt.tight_layout()
    plt.savefig("test2sod_density_comparison.png", dpi=150, bbox_inches="tight")

    # ===== FIGURA 2: PRESIÓN =====
    fig2, (ax2, ax2_zoom) = plt.subplots(1, 2, figsize=(14, 6))

    # Dominio completo
    ax2.plot(rin_ref, p_init_in, 'k:', label='p inicial', alpha=0.7, linewidth=2)
    for i, method in enumerate(methods):
        ax2.plot(results[method]['r'], results[method]['p'],
                color=colors[i], label=labels[i], linewidth=2, linestyle=linestyles[i])
    ax2.set_xlabel('r')
    ax2.set_ylabel('Presión p')
    ax2.set_xlim(0, 1)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('Presión - Dominio Completo')

    # Zoom en shock
    ax2_zoom.plot(rin_ref, p_init_in, 'k:', label='p inicial', alpha=0.7, linewidth=2)
    for i, method in enumerate(methods):
        ax2_zoom.plot(results[method]['r'], results[method]['p'],
                     color=colors[i], label=labels[i], linewidth=2, linestyle=linestyles[i])
    ax2_zoom.set_xlabel('r')
    ax2_zoom.set_ylabel('Presión p')
    ax2_zoom.set_xlim(0.6, 0.7)
    ax2_zoom.set_ylim(1e-1, 1e1)
    ax2_zoom.set_yscale('log')
    ax2_zoom.grid(True, alpha=0.3)
    ax2_zoom.legend()
    ax2_zoom.set_title('Presión - Zoom Shock')

    fig2.suptitle(f"Sod: Presión (t ≈ {avg_time:.3f})")
    plt.tight_layout()
    plt.savefig("test2sod_pressure_comparison.png", dpi=150, bbox_inches="tight")

    # ===== FIGURA 3: VELOCIDAD =====
    fig3, (ax3, ax3_zoom) = plt.subplots(1, 2, figsize=(14, 6))

    # Dominio completo
    ax3.plot(rin_ref, v_init_in, 'k:', label='v inicial', alpha=0.7, linewidth=2)
    for i, method in enumerate(methods):
        ax3.plot(results[method]['r'], results[method]['v'],
                color=colors[i], label=labels[i], linewidth=2, linestyle=linestyles[i])
    ax3.set_xlabel('r')
    ax3.set_ylabel('Velocidad v^r')
    ax3.set_xlim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_title('Velocidad - Dominio Completo')

    # Zoom en shock
    ax3_zoom.plot(rin_ref, v_init_in, 'k:', label='v inicial', alpha=0.7, linewidth=2)
    for i, method in enumerate(methods):
        ax3_zoom.plot(results[method]['r'], results[method]['v'],
                     color=colors[i], label=labels[i], linewidth=2, linestyle=linestyles[i])
    ax3_zoom.set_xlabel('r')
    ax3_zoom.set_ylabel('Velocidad v^r')
    ax3_zoom.set_xlim(0.5, 0.7)
    ax3_zoom.set_ylim(0.6, 0.85)
    ax3_zoom.grid(True, alpha=0.3)
    ax3_zoom.legend()
    ax3_zoom.set_title('Velocidad - Zoom Shock')

    fig3.suptitle(f"Sod: Velocidad (t ≈ {avg_time:.3f})")
    plt.tight_layout()
    plt.savefig("test2sod_velocity_comparison.png", dpi=150, bbox_inches="tight")

    print(f"\nGráficos guardados:")
    print(f"  - test2sod_density_comparison.png")
    print(f"  - test2sod_pressure_comparison.png")
    print(f"  - test2sod_velocity_comparison.png")

    # Verificar que todos los métodos funcionaron
    all_ok = True
    for method in methods:
        rho_in = results[method]['rho']
        grad = np.gradient(rho_in, results[method]['r'])
        variation = np.std(rho_in)/np.mean(rho_in)
        contact = np.any(np.abs(grad) > 0.5)
        method_ok = (variation > 0.1) and contact
        if not method_ok:
            all_ok = False
            print(f"  {method.upper()}: ✗ FALLA")
        else:
            print(f"  {method.upper()}: ✓ PASA")

    print("✓ PASA" if all_ok else "✗ FALLA")
    return all_ok

# ========= TEST ÚNICO (unificado): BLAST WAVE ESFÉRICO =========
def test_blast_wave(case='weak',
                   n_interior=800, r_min=1e-3, r_max=1.0,
                   gamma=1.4, r0=None,
                   t_final=0.4, cfl=0.45,
                   plot=True, savefig=True):
    """
    Blast wave esférico (Minkowski, FULL reference-metric) — test unificado.
      - case='weak'  : pi=1.0,    pe=0.1,   rhoi=1.0,  rhoe=0.125
      - case='strong': pi=133.33, pe=0.125, rhoi=10.0, rhoe=1.0
    Devuelve True/False según variación y presencia de discontinuidad.
    También guarda 'blast_final_<case>.png' si plot=True y savefig=True.
    """
    # --- Imports (del propio paquete) ---

    print("\n" + "="*60)
    print(f"TEST Blast Wave Esférico ({case})")
    print("="*60)

    # --- Grid y parámetros ---
    r, grid, Nin = build_grid(n_interior=n_interior, r_min=r_min, r_max=r_max)
    N  = len(r)
    ng = NUM_GHOSTS

    # --- parámetros físicos del blast (Tabla I) ---
    if case.lower() == 'weak':
        p_in, p_out   = 1.0, 0.1
        rho_in, rho_out = 1.0, 0.125
    elif case.lower() == 'strong':
        p_in, p_out   = 133.33, 0.125
        rho_in, rho_out = 10.0, 1.0
    else:
        raise ValueError("case debe ser 'weak' o 'strong'")

    if r0 is None:
        r0 = 0.5*(r[ng] + r[-ng-1])  # membrana en el centro del dominio interior

    # --- estado inicial (primitivas) + paridades/outflow ---
    rho0 = np.where(r < r0, rho_in, rho_out).astype(float)
    p    = np.where(r < r0, p_in,  p_out ).astype(float)
    v    = np.zeros_like(r, dtype=float)
    rho0, v, p = fill_ghosts_primitives(rho0, v, p)

    # --- objetos físicos ---
    eos    = IdealGasEOS(gamma=gamma)
    recon  = MinmodReconstruction()
    rsolve = HLLERiemannSolver()
    val    = ValenciaReferenceMetric()

    # --- conservadas iniciales ---
    D, Sr, tau = to_conserved(rho0, v, p, eos)

    # --- evolución RK3 con compute_rhs (full approach) ---
    t, steps = 0.0, 0
    while t < t_final and steps < 20000:
        dt, D, Sr, tau, rho0, v, p = rk3_step(
            val, D, Sr, tau, rho0, v, p,
            r, grid, eos, recon, rsolve, cfl=cfl,
            spacetime_mode="fixed_minkowski"
        )
        t += dt; steps += 1

    # --- métricas de validación ---
    rin    = r[ng:-ng]
    rho_in = rho0[ng:-ng]
    grad   = np.gradient(rho_in, rin)
    variation = float(np.std(rho_in)/np.mean(rho_in))
    contact   = bool(np.any(np.abs(grad) > 0.5))
    print(f"[blast:{case}] t≈{t:.4f}, pasos={steps}, variación relativa ρ={variation:.3f}")

    # --- ploteo final (ρ, p, v, W) ---
    if plot:
        eps = eos.eps_from_rho_p(rho0, p)
        h   = 1.0 + eps + p/np.maximum(rho0, 1e-300)
        W   = 1.0/np.sqrt(np.maximum(1.0 - v*v, 1e-16))

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        ax[0,0].plot(rin, rho0[ng:-ng], 'r-', lw=2); ax[0,0].set_xlabel("r"); ax[0,0].set_ylabel("ρ₀"); ax[0,0].grid(True, alpha=0.3)
        ax[0,1].plot(rin, p[ng:-ng],    'r-', lw=2); ax[0,1].set_xlabel("r"); ax[0,1].set_ylabel("p"); ax[0,1].grid(True, alpha=0.3)
        ax[1,0].plot(rin, v[ng:-ng],    'r-', lw=2); ax[1,0].set_xlabel("r"); ax[1,0].set_ylabel("v^r"); ax[1,0].grid(True, alpha=0.3)
        ax[1,1].plot(rin, W[ng:-ng],    'r-', lw=2); ax[1,1].set_xlabel("r"); ax[1,1].set_ylabel("W"); ax[1,1].grid(True, alpha=0.3)
        fig.suptitle(f"Spherical blast ({case}) — Γ={gamma}, r0={r0:.3f}, t≈{t:.3f}, pasos={steps}")
        fig.tight_layout()
        if savefig:
            fname = f"blast_final_{case}2.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Gráfico guardado como '{fname}'")
        else:
            plt.show()

    ok = (variation > 0.1) and contact
    print("✓ PASA" if ok else "✗ FALLA")
    return ok


def test_conservation_short():
    print("\n" + "="*60)
    print("TEST 4: Conservación global (masa/energía)")
    print("="*60)
    r, grid, Nin = build_grid(n_interior=256, r_min=1e-3, r_max=1.0)
    N = len(r)
    eos  = IdealGasEOS(gamma=4.0/3.0)
    recon= MinmodReconstruction()
    rsolve = HLLERiemannSolver()

    rho0 = 1.0 + 0.5*np.exp(-((r-0.6)**2)/0.01)
    p    = 0.1*rho0
    v    = 0.02*np.sin(4*np.pi*(r - r[NUM_GHOSTS]))
    rho0, v, p = fill_ghosts_primitives(rho0, v, p)
    D, Sr, tau = to_conserved(rho0, v, p, eos)

    val = ValenciaReferenceMetric()
    m0, e0 = volume_integrals(D, tau, r, grid)
    steps = 0
    while steps < 200:
        dt, D, Sr, tau, rho0, v, p = rk3_step(val, D, Sr, tau, rho0, v, p, r, grid, eos, recon, rsolve, cfl=0.25)
        steps += 1
    m1, e1 = volume_integrals(D, tau, r, grid)
    dm = abs(m1-m0)/max(m0,1e-15)
    de = abs(e1-e0)/max(e0,1e-15)
    print(f"ΔM/M={dm:.3e}, ΔE/E={de:.3e}")
    ok = (dm < 5e-3) and (de < 5e-3)
    print("✓ PASA" if ok else "✗ FALLA")
    return ok

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("="*60)
    print("SUITE — Valencia FULL Reference-Metric (Minkowski fijo)")
    print("="*60)

    results = []
    results.append(("Uniforme",      test_uniform_state()))
    results.append(("cons2prim",     test_cons2prim_roundtrip()))
    results.append(("Conservación",  test_conservation_short()))
    results.append(("Sod radial",    test_riemann_sod()))
    results.append(("Blast weak",    test_blast_wave(case="weak")))
    results.append(("Blast strong",  test_blast_wave(case="strong")))

    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"{name:14s}: {'✓ PASÓ' if ok else '✗ FALLÓ'}")
    print("-"*40)
    print(f"Total: {passed}/{len(results)}")
