#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adiabatic smooth flow — QUICK CHECK (Cartesian 1D)  N=200, MP5
- EOS: polytropic  p = K rho^Gamma  (K=100, Gamma=5/3)
- IC: bump C^∞ en rho, J_- constante => onda suave a la derecha (no alcanza bordes)
- Geometría: *cartesiana* con Γ̂ = 0 (sin términos 2/r)
- BCs: outflow en ambos extremos
"""

import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- rutas (igual que tus tests) ----
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, repo_root)

# núcleo engrenage
from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector

# BSSN
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS, idx_phi, idx_K, idx_lapse

# hidro (tus módulos)
from source.matter.perfect_fluid import PerfectFluid                          # :contentReference[oaicite:7]{index=7}
from source.matter.hydro.eos import PolytropicEOS                                   # :contentReference[oaicite:8]{index=8}
from source.matter.hydro.reconstruction import create_reconstruction                # :contentReference[oaicite:9]{index=9}
from source.matter.hydro.riemann import HLLRiemannSolver                            # :contentReference[oaicite:10]{index=10}
from source.matter.hydro.cons2prim import prim_to_cons                              # :contentReference[oaicite:11]{index=11}
from source.matter.hydro.atmosphere import AtmosphereParams                         # :contentReference[oaicite:12]{index=12}

# -----------------------------
# Background cartesiano plano
# -----------------------------
class FlatCartesianBackground:
    """
    Mínimo *background* cartesiano para Valencia ref-metric:
    - det_hat_gamma = 1
    - hat_christoffel = 0
    - inverse_scaling_vector = 1 (no re-escalado)
    """
    def __init__(self, x):
        N = len(x)
        self.det_hat_gamma = np.ones(N, dtype=float)
        self.hat_christoffel = np.zeros((N, 3, 3, 3), dtype=float)
        self.inverse_scaling_vector = np.ones((N, 3), dtype=float)
        self.d1_inverse_scaling_vector = np.zeros((N, 3, 3), dtype=float)

# -----------------------------
# Utilidades del test
# -----------------------------
def build_grid_and_hydro(N=200, xmax=2.5, eos=None, recon="mp5"):
    spacing = LinearSpacing(N + 2*NUM_GHOSTS, xmax)
    eos = eos or PolytropicEOS(K=100.0, gamma=5.0/3.0)
    atmosphere = AtmosphereParams(rho_floor=1e-13, p_floor=1e-15, v_max=0.999999, W_max=1e3)
    hydro = PerfectFluid(eos=eos, spacetime_mode="fixed_minkowski",
                         atmosphere=atmosphere,
                         reconstructor=create_reconstruction(recon),
                         riemann_solver=HLLRiemannSolver())
    # Forzamos BCs "outflow" en Valencia (no paridad)
    hydro.valencia.boundary_mode = "outflow"
    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatCartesianBackground(grid.r)
    hydro.background = background
    return grid, hydro, background

def bump_density(x, center=0.6, L=0.3):
    xi = (x - center)/L
    return 1.0 + np.where(np.abs(xi) < 1.0, np.exp(-1.0/(1.0 - xi**2)), 0.0)

def cs_from_eos(eos, rho):
    # c_s^2 = Γ K ρ^{Γ-1} / h  (barotrópica), implementado en tu EOS
    p = eos.pressure(rho)
    eps = eos.eps_from_rho(rho)
    cs2 = eos.sound_speed_squared(rho, p, eps)
    return np.sqrt(np.clip(cs2, 0.0, 1.0 - 1e-12))

def A_of_cs(cs, gamma):
    q = np.sqrt(gamma - 1.0)
    cs = np.minimum(cs, q - 1e-14)
    return (1.0/q) * np.log((q + cs)/(q - cs))

def v_from_Jminus_constant(rho, eos, gamma):
    cs  = cs_from_eos(eos, rho)
    cs0 = float(cs_from_eos(eos, 1.0))
    return np.tanh(A_of_cs(cs, gamma) - A_of_cs(cs0, gamma))

def fill_ghosts_outflow_prims(rho, v, p, ng=NUM_GHOSTS):
    # outflow en ambos extremos
    left = ng; right = len(rho) - ng - 1
    for i in range(ng):
        rho[i] = rho[left];    v[i] = v[left];    p[i] = p[left]
        rho[-1-i] = rho[right]; v[-1-i] = v[right]; p[-1-i] = p[right]
    return rho, v, p

def fill_boundaries_outflow_state(state_2d, ng=NUM_GHOSTS):
    # copia el primer/último punto interior (BSSN + hydro)
    left = ng; right = state_2d.shape[1] - ng - 1
    for i in range(ng):
        state_2d[:, i]      = state_2d[:, left]
        state_2d[:, -1 - i] = state_2d[:, right]
    return state_2d

def prim_to_cons_array(rho0, vr, p, grid, hydro):
    gamma_xx = np.ones_like(rho0)      # cartesiano: γ_xx = 1
    return prim_to_cons(rho0, vr, p, gamma_xx, hydro.eos)

def conservatives_to_primitives(state_2d, grid, hydro, bssn_vars):
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    return hydro._get_primitives(bssn_vars, grid.r)

def max_signal_speed(prims, eos):
    rho, v, p = prims['rho0'], prims['vr'], prims['p']
    eps = eos.eps_from_rho_p(rho, p)
    cs = np.sqrt(np.clip(eos.sound_speed_squared(rho, p, eps), 0.0, 1.0-1e-12))
    return float(np.max(np.abs(v) + cs) + 1e-8)

# RHS (sin paridad; BC outflow)
def rhs_cartesian(state_flat, grid, hydro, background, bssn_fixed, bssn_d1_fixed):
    st2 = state_flat.reshape((grid.NUM_VARS, grid.N))
    fill_boundaries_outflow_state(st2)                      # <- BC outflow
    bssn_vars = BSSNVars(grid.N); bssn_vars.set_bssn_vars(bssn_fixed)
    hydro.set_matter_vars(st2, bssn_vars, grid)
    rhs_hydro = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1_fixed, background)
    out = np.zeros_like(st2)
    out[NUM_BSSN_VARS:, :] = rhs_hydro
    return out.flatten()

def rk4_step(state_2d, grid, hydro, background, bssn_fixed, bssn_d1_fixed, cfl=0.2):
    bssn_vars = BSSNVars(grid.N); bssn_vars.set_bssn_vars(bssn_fixed)
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    prims = hydro._get_primitives(bssn_vars, grid.r)
    dt = cfl * grid.min_dr / max_signal_speed(prims, hydro.eos)
    y0 = state_2d.flatten()
    k1 = rhs_cartesian(y0, grid, hydro, background, bssn_fixed, bssn_d1_fixed)
    k2 = rhs_cartesian(y0 + 0.5*dt*k1, grid, hydro, background, bssn_fixed, bssn_d1_fixed)
    k3 = rhs_cartesian(y0 + 0.5*dt*k2, grid, hydro, background, bssn_fixed, bssn_d1_fixed)
    k4 = rhs_cartesian(y0 + dt*k3, grid, hydro, background, bssn_fixed, bssn_d1_fixed)
    yN = y0 + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return yN.reshape((grid.NUM_VARS, grid.N)), dt

# -----------------------------
# Main
# -----------------------------
def main():
    # Parámetros del test
    N = 200; xmax = 2.5
    Gamma = 5.0/3.0; K = 100.0
    center = 0.6; L = 0.3
    tfinal = 1.6; cfl = 0.2
    sample_times = [0.0, 0.4, 0.8, 1.2, 1.6]

    eos = PolytropicEOS(K=K, gamma=Gamma)
    grid, hydro, background = build_grid_and_hydro(N, xmax, eos, recon="mp5")

    # IC: ρ bump + J_- cte
    rho0 = bump_density(grid.r, center=center, L=L)
    p = eos.pressure(rho0)
    vr = v_from_Jminus_constant(rho0, eos, Gamma)
    rho0, vr, p = fill_ghosts_outflow_prims(rho0, vr, p)

    # Conservadas y estado completo
    D, Sr, tau = prim_to_cons_array(rho0, vr, p, grid, hydro)
    state_2d = np.zeros((grid.NUM_VARS, grid.N))
    state_2d[idx_lapse,:] = 1.0; state_2d[idx_phi,:] = 0.0; state_2d[idx_K,:] = 0.0
    state_2d[hydro.idx_D,:] = D; state_2d[hydro.idx_Sr,:] = Sr; state_2d[hydro.idx_tau,:] = tau

    # BSSN fijo (Minkowski)
    bssn_fixed = state_2d[:NUM_BSSN_VARS,:].copy()
    bssn_d1_fixed = grid.get_d1_metric_quantities(state_2d)

    # Evolución y muestreo
    t = 0.0; steps = 0; next_i = 0
    ng = NUM_GHOSTS; x_in = grid.r[ng:-ng]
    times, RHOs, Vs, peaks = [], [], [], []

    # muestra t=0
    bssn_vars0 = BSSNVars(grid.N); bssn_vars0.set_bssn_vars(bssn_fixed)
    prim0 = conservatives_to_primitives(state_2d, grid, hydro, bssn_vars0)
    times.append(0.0); RHOs.append(prim0['rho0'][ng:-ng].copy()); Vs.append(prim0['vr'][ng:-ng].copy())
    peaks.append(float(x_in[np.argmax(RHOs[-1])]))
    next_i = 1

    while t < tfinal and steps < 200000:
        state_2d, dt = rk4_step(state_2d, grid, hydro, background, bssn_fixed, bssn_d1_fixed, cfl=cfl)
        t += dt; steps += 1
        while next_i < len(sample_times) and t >= sample_times[next_i] - 1e-12:
            bssn_vars = BSSNVars(grid.N); bssn_vars.set_bssn_vars(bssn_fixed)
            prims = conservatives_to_primitives(state_2d, grid, hydro, bssn_vars)
            times.append(t)
            RHOs.append(prims['rho0'][ng:-ng].copy())
            Vs.append(prims['vr'][ng:-ng].copy())
            peaks.append(float(x_in[np.argmax(RHOs[-1])]))
            next_i += 1

    print(f"[OK] t≈{t:.3f}, pasos={steps}, muestras={len(times)}")

    # ======= gráficas =======
    plt.figure(figsize=(9,5))
    for tk, rho in zip(times, RHOs):
        plt.plot(x_in, rho, '--', lw=2, label=f"t={tk:.2f}")
    plt.xlim(0, xmax); plt.xlabel("x"); plt.ylabel(r"$\rho_0$")
    plt.title("Adiabatic smooth flow — evolución de la densidad (MP5, N=200, cartesiano)")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig("rho_evolution.png", dpi=140, bbox_inches="tight")

    plt.figure(figsize=(9,5))
    for tk, vv in zip(times, Vs):
        plt.plot(x_in, vv, '--', lw=2, label=f"t={tk:.2f}")
    plt.xlim(0, xmax); plt.xlabel("x"); plt.ylabel(r"$v$")
    plt.title("Adiabatic smooth flow — evolución de la velocidad (MP5, N=200, cartesiano)")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig("v_evolution.png", dpi=140, bbox_inches="tight")

    t_arr = np.array(times); x_arr = np.array(peaks)
    if len(t_arr) >= 2:
        A = np.vstack([t_arr, np.ones_like(t_arr)]).T
        vel, x0 = np.linalg.lstsq(A, x_arr, rcond=None)[0]
        print(f"pico: x(0)≈{x_arr[0]:.3f} → x({t_arr[-1]:.2f})≈{x_arr[-1]:.3f},  vel_media≈{vel:.3f}")
    plt.figure(figsize=(6,4.5))
    plt.plot(t_arr, x_arr, "o-"); plt.xlabel("t"); plt.ylabel("posición del pico ρ₀")
    plt.title("Traslación del pulso"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("peak_vs_time.png", dpi=140, bbox_inches="tight")

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
