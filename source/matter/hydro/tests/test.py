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
from source.core.spacing import NUM_GHOSTS, LinearSpacing, SpacingExtent
from source.core.grid import Grid
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import i_r
from source.bssn.tensoralgebra import get_bar_gamma_LL

from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.reconstruction import MinmodReconstruction, create_reconstruction
from source.matter.hydro.riemann import HLLERiemannSolver
from source.matter.hydro.cons2prim import cons_to_prim, prim_to_cons
from source.matter.hydro.valencia_reference_metric import ValenciaReferenceMetric
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.bssn.tensoralgebra import SPACEDIM
from source.backgrounds.sphericalbackground import FlatSphericalBackground


class _DummyBSSNVars:
    #Placeholders no usados en Minkowski fijo.
    def __init__(self, N):
        self.lapse   = np.ones(N)
        self.shift_U = np.zeros((N,3))
        self.phi     = np.zeros(N)
        self.K       = np.zeros(N)
        self.h_LL    = np.zeros((N,3,3))
        self.a_LL    = np.zeros((N,3,3))

class _DummyBSSND1:
    def __init__(self, N):
        self.lapse   = np.zeros((N,3))
        self.shift_U = np.zeros((N,3,3))
        self.phi     = np.zeros((N,3))
        self.h_LL    = np.zeros((N,3,3,3))

def build_engrenage_grid_with_matter(n_interior=256, r_min=1.0e-3, r_max=1.0, ng=NUM_GHOSTS,
                                     eos=None, reconstructor=None, riemann_solver=None):
    """Create standard Engrenage Grid with PerfectFluid matter class."""
    Nin = int(n_interior)
    num_points = Nin + 2 * ng  # total points including ghosts
    spacing = LinearSpacing(num_points, r_max, SpacingExtent.HALF)

    matter = PerfectFluid(
        eos=eos,
        spacetime_mode="fixed_minkowski",
        reconstructor=reconstructor,
        riemann_solver=riemann_solver
    )

    # Create state vector and grid
    state_vector = StateVector(matter)
    grid = Grid(spacing, state_vector)

    # Manually adjust r coordinates to get the desired r_min at first interior point
    r_interior_start = grid.r[ng]
    shift = r_min - r_interior_start
    grid.r = grid.r + shift

    return grid, matter, Nin

def build_engrenage_grid(n_interior=256, r_min=1.0e-3, r_max=1.0, ng=NUM_GHOSTS):
    """Wrapper for compatibility - creates grid with default matter setup."""
    grid, matter, Nin = build_engrenage_grid_with_matter(n_interior, r_min, r_max, ng)
    return grid, Nin

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

def rk4_step(valencia, D, Sr, tau, rho0, v, p, r, grid, eos, recon, rsolve, cfl=0.5,
             spacetime_mode="fixed_minkowski"):
    """Una etapa RK4 clásico usando compute_rhs (full approach)."""
    # dt CFL
    amax = max_signal_speed(rho0, v, p, eos)
    dt = cfl * grid.min_dr / amax

    # Dummy BSSN (no usado en Minkowski, pero la firma lo pide)
    bssn_vars = _DummyBSSNVars(len(r))
    bssn_d1   = _DummyBSSND1(len(r))
    background = FlatSphericalBackground(r)

    # Stage 1: k1 = f(U_n)
    k1_D, k1_Sr, k1_tau = valencia.compute_rhs(D, Sr, tau, rho0, v, p,
                                                W=None, h=None,
                                                r=r, bssn_vars=bssn_vars, bssn_d1=bssn_d1,
                                                background=background, spacetime_mode=spacetime_mode,
                                                eos=eos, grid=grid, reconstructor=recon, riemann_solver=rsolve)

    D1   = D   + 0.5*dt*k1_D
    Sr1  = Sr  + 0.5*dt*k1_Sr
    tau1 = tau + 0.5*dt*k1_tau
    rho1, v1, p1 = to_primitives(D1, Sr1, tau1, eos, p_guess=p)
    rho1, v1, p1 = fill_ghosts_primitives(rho1, v1, p1)

    # Stage 2: k2 = f(U_n + 0.5*dt*k1)
    k2_D, k2_Sr, k2_tau = valencia.compute_rhs(D1, Sr1, tau1, rho1, v1, p1,
                                                W=None, h=None,
                                                r=r, bssn_vars=bssn_vars, bssn_d1=bssn_d1,
                                                background=background, spacetime_mode=spacetime_mode,
                                                eos=eos, grid=grid, reconstructor=recon, riemann_solver=rsolve)

    D2   = D   + 0.5*dt*k2_D
    Sr2  = Sr  + 0.5*dt*k2_Sr
    tau2 = tau + 0.5*dt*k2_tau
    rho2, v2, p2 = to_primitives(D2, Sr2, tau2, eos, p_guess=p1)
    rho2, v2, p2 = fill_ghosts_primitives(rho2, v2, p2)

    # Stage 3: k3 = f(U_n + 0.5*dt*k2)
    k3_D, k3_Sr, k3_tau = valencia.compute_rhs(D2, Sr2, tau2, rho2, v2, p2,
                                                W=None, h=None,
                                                r=r, bssn_vars=bssn_vars, bssn_d1=bssn_d1,
                                                background=background, spacetime_mode=spacetime_mode,
                                                eos=eos, grid=grid, reconstructor=recon, riemann_solver=rsolve)

    D3   = D   + dt*k3_D
    Sr3  = Sr  + dt*k3_Sr
    tau3 = tau + dt*k3_tau
    rho3, v3, p3 = to_primitives(D3, Sr3, tau3, eos, p_guess=p2)
    rho3, v3, p3 = fill_ghosts_primitives(rho3, v3, p3)

    # Stage 4: k4 = f(U_n + dt*k3)
    k4_D, k4_Sr, k4_tau = valencia.compute_rhs(D3, Sr3, tau3, rho3, v3, p3,
                                                W=None, h=None,
                                                r=r, bssn_vars=bssn_vars, bssn_d1=bssn_d1,
                                                background=background, spacetime_mode=spacetime_mode,
                                                eos=eos, grid=grid, reconstructor=recon, riemann_solver=rsolve)

    # Final update: U_{n+1} = U_n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    Dn   = D   + (dt/6.0)*(k1_D   + 2.0*k2_D   + 2.0*k3_D   + k4_D)
    Snn  = Sr  + (dt/6.0)*(k1_Sr  + 2.0*k2_Sr  + 2.0*k3_Sr  + k4_Sr)
    taun = tau + (dt/6.0)*(k1_tau + 2.0*k2_tau + 2.0*k3_tau + k4_tau)

    rhon, vn, pn = to_primitives(Dn, Snn, taun, eos, p_guess=p3)
    rhon, vn, pn = fill_ghosts_primitives(rhon, vn, pn)

    return dt, Dn, Snn, taun, rhon, vn, pn

def rk4_step_matter_class(matter, D, Sr, tau, r, grid, cfl=0.5, spacetime_mode="fixed_minkowski"):
    """RK4 step using PerfectFluid matter class (Engrenage pattern)."""
    from source.bssn.bssnvars import BSSNVars

    # Setup matter variables in the matter class
    state_vector = np.zeros((grid.NUM_VARS, len(r)))
    state_vector[matter.idx_D] = D
    state_vector[matter.idx_Sr] = Sr
    state_vector[matter.idx_tau] = tau

    # Dummy BSSN variables for Minkowski
    bssn_vars = _DummyBSSNVars(len(r))
    bssn_d1 = _DummyBSSND1(len(r))
    background = FlatSphericalBackground(r)

    # Set matter variables
    matter.set_matter_vars(state_vector, bssn_vars, grid)

    # Get primitives for CFL calculation
    primitives = matter._get_primitives(bssn_vars, r)
    rho0, vr, p = primitives['rho0'], primitives['vr'], primitives['p']

    # Apply boundary conditions to primitives
    rho0, vr, p = fill_ghosts_primitives(rho0, vr, p)

    # Update conservatives after BC
    D, Sr, tau = to_conserved(rho0, vr, p, matter.eos)

    # CFL condition
    amax = max_signal_speed(rho0, vr, p, matter.eos)
    dt = cfl * grid.min_dr / amax

    # RK4 Stage 1: k1 = f(U_n)
    state_vector[matter.idx_D] = D
    state_vector[matter.idx_Sr] = Sr
    state_vector[matter.idx_tau] = tau
    matter.set_matter_vars(state_vector, bssn_vars, grid)

    k1 = matter.get_matter_rhs(r, bssn_vars, bssn_d1, background)
    D1 = D + 0.5*dt * k1[0]
    Sr1 = Sr + 0.5*dt * k1[1]
    tau1 = tau + 0.5*dt * k1[2]

    rho1, vr1, p1 = to_primitives(D1, Sr1, tau1, matter.eos)
    rho1, vr1, p1 = fill_ghosts_primitives(rho1, vr1, p1)
    D1, Sr1, tau1 = to_conserved(rho1, vr1, p1, matter.eos)

    # RK4 Stage 2: k2 = f(U_n + 0.5*dt*k1)
    state_vector[matter.idx_D] = D1
    state_vector[matter.idx_Sr] = Sr1
    state_vector[matter.idx_tau] = tau1
    matter.set_matter_vars(state_vector, bssn_vars, grid)

    k2 = matter.get_matter_rhs(r, bssn_vars, bssn_d1, background)
    D2 = D + 0.5*dt * k2[0]
    Sr2 = Sr + 0.5*dt * k2[1]
    tau2 = tau + 0.5*dt * k2[2]

    rho2, vr2, p2 = to_primitives(D2, Sr2, tau2, matter.eos)
    rho2, vr2, p2 = fill_ghosts_primitives(rho2, vr2, p2)
    D2, Sr2, tau2 = to_conserved(rho2, vr2, p2, matter.eos)

    # RK4 Stage 3: k3 = f(U_n + 0.5*dt*k2)
    state_vector[matter.idx_D] = D2
    state_vector[matter.idx_Sr] = Sr2
    state_vector[matter.idx_tau] = tau2
    matter.set_matter_vars(state_vector, bssn_vars, grid)

    k3 = matter.get_matter_rhs(r, bssn_vars, bssn_d1, background)
    D3 = D + dt * k3[0]
    Sr3 = Sr + dt * k3[1]
    tau3 = tau + dt * k3[2]

    rho3, vr3, p3 = to_primitives(D3, Sr3, tau3, matter.eos)
    rho3, vr3, p3 = fill_ghosts_primitives(rho3, vr3, p3)
    D3, Sr3, tau3 = to_conserved(rho3, vr3, p3, matter.eos)

    # RK4 Stage 4: k4 = f(U_n + dt*k3)
    state_vector[matter.idx_D] = D3
    state_vector[matter.idx_Sr] = Sr3
    state_vector[matter.idx_tau] = tau3
    matter.set_matter_vars(state_vector, bssn_vars, grid)

    k4 = matter.get_matter_rhs(r, bssn_vars, bssn_d1, background)

    # Final update: U_{n+1} = U_n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    Dn = D + (dt/6.0)*(k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0])
    Srn = Sr + (dt/6.0)*(k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1])
    taun = tau + (dt/6.0)*(k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2])

    rhon, vrn, pn = to_primitives(Dn, Srn, taun, matter.eos)
    rhon, vrn, pn = fill_ghosts_primitives(rhon, vrn, pn)

    return dt, Dn, Srn, taun, rhon, vrn, pn

def compute_derivative_engrenage(var, grid, order=1):
    """Compute spatial derivative using Engrenage's derivative system."""
    if hasattr(grid, 'derivs'):
        # Use Engrenage derivatives
        return grid.derivs.drn_matrix[order] @ var
    else:
        # Fallback to numpy gradient
        if order == 1:
            return np.gradient(var, grid.min_dr)
        else:
            raise NotImplementedError(f"Order {order} derivatives not implemented for fallback")

def volume_integrals(D, tau, r, grid):
    """Masa y energía total con peso 4π r^2."""
    ng = NUM_GHOSTS
    rin = r[ng:-ng]
    Din = D[ng:-ng]
    taun = tau[ng:-ng]
    mass  = 4*np.pi * np.sum(Din   * rin*rin) * grid.min_dr
    energ = 4*np.pi * np.sum(taun+Din) * grid.min_dr
    return mass, energ

# =========================
# TESTS
# =========================
def test_uniform_state():
    print("\n" + "="*60)
    print("TEST 1: Estado uniforme (Minkowski, FULL reference-metric)")
    print("="*60)

    grid, Nin = build_engrenage_grid(n_interior=256, r_min=1e-3, r_max=1.0)
    r = grid.r
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
        dt, D, Sr, tau, rho0, vr, p = rk4_step(val, D, Sr, tau, rho0, vr, p, r, grid, eos, recon, rsolve, cfl=0.5)
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
    grid, Nin = build_engrenage_grid(n_interior=200, r_min=1e-3, r_max=1.0)
    r = grid.r
    N = len(r)
    eos = IdealGasEOS(gamma=1.4)
    rsolve = HLLERiemannSolver()

    # Discontinuidad en el punto medio del dominio interior
    r_mid = 0.5*(r[NUM_GHOSTS] + r[-NUM_GHOSTS-1])
    rho0_base = np.where(r < r_mid, 10.0, 1.0)
    p_base = np.where(r < r_mid, 40.0/3.0, 1.0e-6)
    v_base = np.zeros(N)

    # Lista de métodos de reconstrucción a probar
    methods = ["mp5"]# "mp5", "weno5","wenoz", "mp5_hires"]#, "weno5", "wenoz", "mp5_hires"]
    colors = ["lightcoral"]# "lightblue", "lightgreen", "cyan", "black"]  
    labels = ["MP5"]# od"]#, "MP5", "WENO5" , "WENOZ", "MP5 (Hi-Res)"]
    linestyles = ["--"]#, "-", "--", "--", "-"]

    # Guardar resultados para cada método
    results = {}

    for i, method in enumerate(methods):
        print(f"\nEjecutando con {labels[i]}...")

        # Para MP5 alta resolución, usar más puntos
        if method == "mp5_hires":
            grid_hires, Nin_hires = build_engrenage_grid(n_interior=2000, r_min=1e-3, r_max=1.0)
            r_hires = grid_hires.r
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
            dt, D, Sr, tau, rho0, v, p = rk4_step(
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
        # Use Engrenage derivatives for gradient calculation
        rho_full = np.zeros_like(r_current)
        rho_full[ng:-ng] = rho_in
        grad_full = compute_derivative_engrenage(rho_full, grid_current, order=1)
        grad = grad_full[ng:-ng]  # Extract interior gradient
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
# ========= TEST: BLAST WAVE ESFÉRICO — COMPARACIÓN DE RECONSTRUCTORES =========
def test_blast_compare(case='weak',
                       n_interior=200, r_min=1e-3, r_max=1.0,
                       gamma=1.4, r0=None,
                       t_final=0.40, cfl=0.45,
                       methods=("minmod", "mp5", "weno5", "mp5_hires"),
                       hires_factor=10,           # multiplicador de resolución para la referencia
                       spacetime_mode="fixed_minkowski",
                       plot=True, savefig=True):
    """
    Blast wave esférico (Minkowski, FULL reference-metric) — comparación de reconstructores.
      - case='weak'  : p_in=1.0,   p_out=0.1,   rho_in=1.0,  rho_out=0.125
      - case='strong': p_in=133.33,p_out=0.125, rho_in=10.0, rho_out=1.0

    'mp5_hires' corre con n_interior*hires_factor y sirve como baseline de alta resolución.
    Devuelve True si todos los métodos superan umbrales básicos de variación/contacto.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    print("\n" + "="*70)
    print(f"TEST Blast Esférico — Comparación de Reconstructores (case={case})")
    print("="*70)

    # --- Grid base ---
    grid, Nin = build_engrenage_grid(n_interior=n_interior, r_min=r_min, r_max=r_max)
    r  = grid.r
    N  = len(r)
    ng = NUM_GHOSTS

    # --- parámetros físicos del blast ---
    if case.lower() == 'weak':
        p_in, p_out     = 1.0, 0.1
        rho_in, rho_out = 1.0, 0.125
    elif case.lower() == 'strong':
        p_in, p_out     = 133.33, 0.125
        rho_in, rho_out = 10.0, 1.0
    else:
        raise ValueError("case debe ser 'weak' o 'strong'")

    # radio de membrana
    if r0 is None:
        r0 = 0.5*(r[ng] + r[-ng-1])

    # --- estado inicial base (para reusar en métodos de resolución normal) ---
    rho0_base = np.where(r < r0, rho_in, rho_out).astype(float)
    p_base    = np.where(r < r0, p_in,  p_out ).astype(float)
    v_base    = np.zeros_like(r, dtype=float)

    # --- objetos físicos comunes ---
    eos    = IdealGasEOS(gamma=gamma)
    rsolve = HLLERiemannSolver()
    val    = ValenciaReferenceMetric()

    # --- configuración de estilos de plot ---
    # (mantener longitudes en sincronía con 'methods')
    default_colors     = ["lightcoral", "lightblue", "lightgreen", "moccasin", "thistle", "black"]
    default_labels_map = {
        "minmod":    "Minmod",
        "mp5":       "MP5",
        "weno5":     "WENO5",
        "wenoz":     "WENO-Z",
        "mp5_hires": "MP5 (Hi-Res)",
    }
    default_linestyles = {"mp5_hires": "-", "mp5": "-", "weno5": "--", "minmod": "--", "wenoz": "--"}

    # --- result container ---
    results = {}

    # ======================
    # EJECUCIÓN POR MÉTODO
    # ======================
    for idx, method in enumerate(methods):
        label = default_labels_map.get(method, method.upper())
        print(f"\nEjecutando con {label}...")

        # ¿Alta resolución para baseline?
        if method == "mp5_hires":
            n_hi = max(4*n_interior, int(n_interior*hires_factor))
            grid_hi, Nin_hi = build_engrenage_grid(n_interior=n_hi, r_min=r_min, r_max=r_max)
            r_current  = grid_hi.r
            grid_curr  = grid_hi
            actual_mth = "mp5"

            # CI en alta resolución
            r0_hi = 0.5*(r_current[ng] + r_current[-ng-1])
            rho0 = np.where(r_current < r0_hi, rho_in, rho_out).astype(float)
            p    = np.where(r_current < r0_hi, p_in,  p_out ).astype(float)
            v    = np.zeros_like(r_current, dtype=float)

        else:
            # resolución normal
            r_current  = r
            grid_curr  = grid
            actual_mth = method

            rho0 = rho0_base.copy()
            p    = p_base.copy()
            v    = v_base.copy()

        # aplicar paridades / outflow en primitivas
        rho0, v, p = fill_ghosts_primitives(rho0, v, p)

        # convertir a conservadas
        D, Sr, tau = to_conserved(rho0, v, p, eos)

        # reconstructor específico
        recon = create_reconstruction(actual_mth)

        # evolución temporal
        t, steps = 0.0, 0
        while t < t_final and steps < 20000:
            dt, D, Sr, tau, rho0, v, p = rk4_step(
                val, D, Sr, tau, rho0, v, p,
                r_current, grid_curr, eos, recon, rsolve, cfl=cfl,
                spacetime_mode=spacetime_mode
            )
            t += dt
            steps += 1

        # guardar resultados del interior
        rin = r_current[ng:-ng]
        results[method] = {
            "r": rin.copy(),
            "rho": rho0[ng:-ng].copy(),
            "p":   p[ng:-ng].copy(),
            "v":   v[ng:-ng].copy(),
            "t":   t,
            "steps": steps,
            "label": label,
            "color": default_colors[idx % len(default_colors)],
            "ls":    default_linestyles.get(method, "--"),
        }

        # métricas rápidas
        rho_interior = rho0[ng:-ng]
        rho_full = np.zeros_like(r_current)
        rho_full[ng:-ng] = rho_interior
        grad_full = compute_derivative_engrenage(rho_full, grid_curr, order=1)
        grad = grad_full[ng:-ng]
        variation = float(np.std(rho_interior)/np.mean(rho_interior))
        contact = bool(np.any(np.abs(grad) > 0.5))
        print(f"  {label}: t≈{t:.4f}, pasos={steps}, variación ρ={variation:.3f}, contacto={contact}")

    # ======================
    # PLOTS COMPARATIVOS
    # ======================
    if plot:
        # Estados iniciales (como referencia visual, con grid base)
        rho0_init = np.where(r < r0, rho_in, rho_out).astype(float)
        p_init    = np.where(r < r0, p_in,  p_out ).astype(float)
        v_init    = np.zeros_like(r, dtype=float)
        rho0_init, v_init, p_init = fill_ghosts_primitives(rho0_init, v_init, p_init)
        rin_ref = r[ng:-ng]

        # tiempo medio de simulaciones (solo para el título)
        avg_time = np.mean([results[m]["t"] for m in methods])

        # === Figura 1: Densidad ===
        fig1, (ax1, ax1z) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.plot(rin_ref, rho0_init[ng:-ng], 'k:', label='ρ inicial', alpha=0.7, linewidth=2)
        for m in methods:
            R = results[m]
            ax1.plot(R["r"], R["rho"], color=R["color"], linestyle=R["ls"], linewidth=2, label=R["label"])
        ax1.set_xlabel('r'); ax1.set_ylabel('Densidad ρ₀'); ax1.set_xlim(r_min, r_max); ax1.grid(True, alpha=0.3); ax1.legend()
        ax1.set_title('Densidad — Dominio completo')

        # zona de choque (ajusta si tu choque se mueve distinto)
        ax1z.plot(rin_ref, rho0_init[ng:-ng], 'k:', alpha=0.7, linewidth=2)
        for m in methods:
            R = results[m]
            ax1z.plot(R["r"], R["rho"], color=R["color"], linestyle=R["ls"], linewidth=2, label=R["label"])
        ax1z.set_xlabel('r'); ax1z.set_ylabel('Densidad ρ₀'); ax1z.set_xlim(0.45, 0.8); ax1z.grid(True, alpha=0.3)
        ax1z.set_title('Densidad — Zoom shock')

        fig1.suptitle(f"Blast esférico ({case}) — ρ₀ (t ≈ {avg_time:.3f})")
        plt.tight_layout()
        if savefig:
            plt.savefig(f"blast_{case}_density_compare.png", dpi=150, bbox_inches="tight")

        # === Figura 2: Presión ===
        fig2, (ax2, ax2z) = plt.subplots(1, 2, figsize=(14, 6))
        ax2.plot(rin_ref, p_init[ng:-ng], 'k:', label='p inicial', alpha=0.7, linewidth=2)
        for m in methods:
            R = results[m]
            ax2.plot(R["r"], R["p"], color=R["color"], linestyle=R["ls"], linewidth=2, label=R["label"])
        ax2.set_xlabel('r'); ax2.set_ylabel('p'); ax2.set_xlim(r_min, r_max); ax2.set_yscale('log'); ax2.grid(True, alpha=0.3); ax2.legend()
        ax2.set_title('Presión — Dominio completo')

        ax2z.plot(rin_ref, p_init[ng:-ng], 'k:', alpha=0.7, linewidth=2)
        for m in methods:
            R = results[m]
            ax2z.plot(R["r"], R["p"], color=R["color"], linestyle=R["ls"], linewidth=2, label=R["label"])
        ax2z.set_xlabel('r'); ax2z.set_ylabel('p'); ax2z.set_xlim(0.45, 0.8); ax2z.set_yscale('log'); ax2z.grid(True, alpha=0.3)
        ax2z.set_title('Presión — Zoom shock')

        fig2.suptitle(f"Blast esférico ({case}) — p (t ≈ {avg_time:.3f})")
        plt.tight_layout()
        if savefig:
            plt.savefig(f"blast_{case}_pressure_compare.png", dpi=150, bbox_inches="tight")

        # === Figura 3: Velocidad ===
        fig3, (ax3, ax3z) = plt.subplots(1, 2, figsize=(14, 6))
        ax3.plot(rin_ref, v_init[ng:-ng], 'k:', label='v inicial', alpha=0.7, linewidth=2)
        for m in methods:
            R = results[m]
            ax3.plot(R["r"], R["v"], color=R["color"], linestyle=R["ls"], linewidth=2, label=R["label"])
        ax3.set_xlabel('r'); ax3.set_ylabel('v^r'); ax3.set_xlim(r_min, r_max); ax3.grid(True, alpha=0.3); ax3.legend()
        ax3.set_title('Velocidad — Dominio completo')

        ax3z.plot(rin_ref, v_init[ng:-ng], 'k:', alpha=0.7, linewidth=2)
        for m in methods:
            R = results[m]
            ax3z.plot(R["r"], R["v"], color=R["color"], linestyle=R["ls"], linewidth=2, label=R["label"])
        ax3z.set_xlabel('r'); ax3z.set_ylabel('v^r'); ax3z.set_xlim(0.45, 0.8); ax3z.grid(True, alpha=0.3)
        ax3z.set_title('Velocidad — Zoom shock')

        fig3.suptitle(f"Blast esférico ({case}) — v^r (t ≈ {avg_time:.3f})")
        plt.tight_layout()
        if savefig:
            plt.savefig(f"blast_{case}_velocity_compare.png", dpi=150, bbox_inches="tight")

        if savefig:
            print("\nGráficos guardados:")
            print(f"  - blast_{case}_density_compare.png")
            print(f"  - blast_{case}_pressure_compare.png")
            print(f"  - blast_{case}_velocity_compare.png")
        else:
            plt.show()

    # ======================
    # CHECKS SENCILLOS
    # ======================
    all_ok = True
    for m in methods:
        R = results[m]
        rho_in = R["rho"]
        # gradiente numérico simple en r del propio método (sin mezclar grids)
        grad = np.gradient(rho_in, R["r"])
        variation = float(np.std(rho_in)/np.mean(rho_in))
        contact = bool(np.any(np.abs(grad) > 0.5))
        ok = (variation > 0.1) and contact
        all_ok = all_ok and ok
        print(f"  {R['label']}: {'✓ PASA' if ok else '✗ FALLA'} (var={variation:.3f}, contact={contact})")

    print("✓ PASA" if all_ok else "✗ FALLA")
    return all_ok

def test_conservation_short():
    print("\n" + "="*60)
    print("TEST 4: Conservación global (masa/energía)")
    print("="*60)
    grid, Nin = build_engrenage_grid(n_interior=256, r_min=1e-3, r_max=1.0)
    r = grid.r
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
        dt, D, Sr, tau, rho0, v, p = rk4_step(val, D, Sr, tau, rho0, v, p, r, grid, eos, recon, rsolve, cfl=0.25)
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
