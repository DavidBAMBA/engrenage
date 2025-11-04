#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adiabatic smooth flow (isentropic) — Convergence test (Cartesian 1D, Minkowski)

Corrección clave respecto al quick-check previo:
- Geometría cartesiana (Γ̂ = 0) y BCs *outflow* en ambos extremos → no hay
  términos geométricos que distorsionen la onda suave.
- Se usa Valencia (reference-metric) + reconstructores + HLL + cons↔prim

EOS: politrópica  p = K rho^Gamma  (K=100, Gamma=5/3 por defecto)
IC:  ρ(x) = 1 + exp(-1/(1-ξ^2)) para |ξ|<1; v(x) de J_- constante ⇒ onda hacia la derecha
     sin tocar fronteras para el tiempo seleccionado.

Salida: CSV con L1(ρ,v,p) y órdenes, y gráficas de convergencia por método.
"""

import os, sys, argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ---------------------------------------------------------------------
# Añade la ruta del repo (igual patrón que tus tests)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, repo_root)

# Núcleo engrenage
from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector

# BSSN (congelado en Minkowski)
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS, idx_phi, idx_K, idx_lapse

# Hidro (tus módulos)
from source.matter.hydro.perfect_fluid import PerfectFluid           # ← usa Valencia ref-metric :contentReference[oaicite:6]{index=6}
from source.matter.hydro.eos import PolytropicEOS                    # ← EOS politrópica                :contentReference[oaicite:7]{index=7}
from source.matter.hydro.reconstruction import create_reconstruction # ← reconstructores (MP5/WENOZ)    :contentReference[oaicite:8]{index=8}
from source.matter.hydro.riemann import HLLRiemannSolver             # ← Riemann HLL                    :contentReference[oaicite:9]{index=9}
from source.matter.hydro.cons2prim import prim_to_cons               # ← prim→cons                      :contentReference[oaicite:10]{index=10}
from source.matter.hydro.atmosphere import AtmosphereParams          # ← floors/atmósfera centralizada  :contentReference[oaicite:11]{index=11}

# -----------------------------
# Background cartesiano plano
# -----------------------------
class FlatCartesianBackground:
    """
    Background mínimo cartesiano para Valencia ref-metric:
      det_hat_gamma = 1, Γ̂ = 0, inverse_scaling_vector = 1.
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
    # ¡Forzamos BCs outflow en Valencia!
    hydro.valencia.boundary_mode = "outflow"     # evita paridad del origen (cartesiano)
    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatCartesianBackground(grid.r)
    hydro.background = background
    return grid, hydro, background

def bump_density(x, center=0.6, L=0.3):
    xi = (x - center)/L
    return 1.0 + np.where(np.abs(xi) < 1.0, np.exp(-1.0/(1.0 - xi**2)), 0.0)

def cs_from_eos(eos, rho):
    p = eos.pressure(rho)
    eps = eos.eps_from_rho(rho)  # barotrópica
    cs2 = eos.sound_speed_squared(rho, p, eps)
    return np.sqrt(np.clip(cs2, 0.0, 1.0-1e-12))

def A_of_cs(cs, gamma):
    q = np.sqrt(gamma - 1.0)
    cs = np.minimum(cs, q - 1e-14)
    return (1.0/q) * np.log((q + cs)/(q - cs))

def v_from_Jminus_constant(rho, eos, gamma):
    cs  = cs_from_eos(eos, rho)
    cs0 = float(cs_from_eos(eos, 1.0))
    return np.tanh(A_of_cs(cs, gamma) - A_of_cs(cs0, gamma))

def fill_ghosts_outflow_prims(rho, v, p, ng=NUM_GHOSTS):
    # outflow en ambos extremos (copia del primer/último interior)
    left = ng; right = len(rho) - ng - 1
    for i in range(ng):
        rho[i] = rho[left];    v[i] = v[left];    p[i] = p[left]
        rho[-1-i] = rho[right]; v[-1-i] = v[right]; p[-1-i] = p[right]
    return rho, v, p

def fill_boundaries_outflow_state(state_2d, ng=NUM_GHOSTS):
    left = ng; right = state_2d.shape[1] - ng - 1
    for i in range(ng):
        state_2d[:, i]      = state_2d[:, left]
        state_2d[:, -1 - i] = state_2d[:, right]
    return state_2d

def prim_to_cons_array(rho0, vr, p, grid, hydro):
    gamma_xx = np.ones_like(rho0)
    return prim_to_cons(rho0, vr, p, gamma_xx, hydro.eos)

def conservatives_to_primitives(state_2d, grid, hydro, bssn_vars):
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    return hydro._get_primitives(bssn_vars, grid.r)

def max_signal_speed(prims, eos):
    rho, v, p = prims['rho0'], prims['vr'], prims['p']
    eps = eos.eps_from_rho_p(rho, p)
    cs = np.sqrt(np.clip(eos.sound_speed_squared(rho, p, eps), 0.0, 1.0-1e-12))
    return float(np.max(np.abs(v) + cs) + 1e-8)

# RHS y paso temporal (RK4)
def rhs_cartesian(state_flat, grid, hydro, background, bssn_fixed, bssn_d1_fixed):
    st2 = state_flat.reshape((grid.NUM_VARS, grid.N))
    grid.fill_boundaries(st2)  # BCs del grid
    #fill_boundaries_outflow_state(st2)  # BCs outflow para todo el estado
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
    dt = 0.1*cfl * grid.min_dr / max_signal_speed(prims, hydro.eos)
    y0 = state_2d.flatten()
    k1 = rhs_cartesian(y0, grid, hydro, background, bssn_fixed, bssn_d1_fixed)
    k2 = rhs_cartesian(y0 + 0.5*dt*k1, grid, hydro, background, bssn_fixed, bssn_d1_fixed)
    k3 = rhs_cartesian(y0 + 0.5*dt*k2, grid, hydro, background, bssn_fixed, bssn_d1_fixed)
    k4 = rhs_cartesian(y0 + dt*k3, grid, hydro, background, bssn_fixed, bssn_d1_fixed)
    yN = y0 + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return yN.reshape((grid.NUM_VARS, grid.N)), dt

# Evolución una vez (devuelve primitivos interiores al tfinal)
def evolve_solution(n_interior, r_max, eos, reconstructor, center, L, t_final, cfl, progress_callback=None):
    grid, hydro, background = build_grid_and_hydro(n_interior, r_max, eos, reconstructor)

    # IC: ρ bump + J_- cte
    rho0 = bump_density(grid.r, center=center, L=L)
    p = eos.pressure(rho0)
    vr = v_from_Jminus_constant(rho0, eos, eos.gamma)
    rho0, vr, p = fill_ghosts_outflow_prims(rho0, vr, p)

    # Conservadas
    D, Sr, tau = prim_to_cons_array(rho0, vr, p, grid, hydro)

    # Estado completo
    state_2d = np.zeros((grid.NUM_VARS, grid.N))
    state_2d[idx_lapse,:] = 1.0; state_2d[idx_phi,:] = 0.0; state_2d[idx_K,:] = 0.0
    state_2d[hydro.idx_D,:] = D; state_2d[hydro.idx_Sr,:] = Sr; state_2d[hydro.idx_tau,:] = tau

    # BSSN congelado (Minkowski)
    bssn_fixed = state_2d[:NUM_BSSN_VARS,:].copy()
    bssn_d1_fixed = grid.get_d1_metric_quantities(state_2d)

    # Evolución
    t = 0.0; steps = 0
    while t < t_final and steps < 300000:
        state_2d, dt = rk4_step(state_2d, grid, hydro, background, bssn_fixed, bssn_d1_fixed, cfl=cfl)
        t += dt; steps += 1

        # Callback de progreso
        if progress_callback is not None:
            progress_callback(t, t_final, steps)

    # Primitivos (interior)
    bssn_vars = BSSNVars(grid.N); bssn_vars.set_bssn_vars(bssn_fixed)
    prims = conservatives_to_primitives(state_2d, grid, hydro, bssn_vars)
    ng = NUM_GHOSTS
    rin = grid.r[ng:-ng].copy()
    rho = prims['rho0'][ng:-ng].copy()
    v = prims['vr'][ng:-ng].copy()
    p  = prims['p'][ng:-ng].copy()
    return rin, rho, v, p

# Métricas de error y orden
def l1_error_against_reference(r_in, u_in, r_ref, u_ref):
    u_ref_interp = np.interp(r_in, r_ref, u_ref)
    return float(np.mean(np.abs(u_in - u_ref_interp)))

def compute_orders(h_list, e_list):
    p = []
    for i in range(len(e_list)-1):
        if e_list[i+1] <= 0 or e_list[i] <= 0:
            p.append(np.nan)
        else:
            p.append(np.log(e_list[i]/e_list[i+1]) / np.log(h_list[i]/h_list[i+1]))
    return p

# Función wrapper para ejecutar una simulación individual (para paralelización)
def run_single_simulation(N, rmax, eos_params, method, center, L, tfinal, cfl, position):
    """
    Ejecuta una sola simulación y devuelve los resultados.
    eos_params es un dict con K y gamma para reconstruir el EOS.
    position: posición de la barra en la terminal
    """
    eos = PolytropicEOS(K=eos_params['K'], gamma=eos_params['gamma'])

    # Crear barra de progreso para esta simulación
    desc = f"{method.upper():6s} N={N:4d}"
    pbar = tqdm(total=100, desc=desc, position=position, leave=True,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}<{remaining}')

    # Callback para actualizar la barra (actualizar cada 1% para reducir overhead)
    last_update = [0]  # usar lista para modificar en closure
    def update_progress(t, t_final, steps):
        # Solo actualizar cada 100 pasos o cuando cambia el porcentaje
        if steps % 100 == 0 or t >= t_final:
            percent = int(100 * t / t_final)
            if percent > last_update[0]:
                pbar.update(percent - last_update[0])
                last_update[0] = percent

    r, rho, v, p = evolve_solution(N, rmax, eos, method, center, L, tfinal, cfl,
                                   progress_callback=update_progress)

    # Asegurar que llega al 100%
    pbar.update(100 - last_update[0])
    pbar.close()

    h = rmax / N
    return {
        'method': method,
        'N': N,
        'h': h,
        'r': r,
        'rho': rho,
        'v': v,
        'p': p
    }

# -----------------------------
# Programa principal
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Convergence test: adiabatic smooth flow (Cartesian)")
    ap.add_argument("--methods", nargs="+", default=["mp5", "minmod", "weno5", "wenoz"], help="Reconstrutores: minmod, ppm, mp5, weno5, wenoz")
    ap.add_argument("--res", nargs="+", type=int, default=[50, 200, 400, 800, 1600], help="N de celdas interiores")
    ap.add_argument("--ref-mult", type=int, default=8, help="Resolución de referencia = ref_mult * max(res)")
    ap.add_argument("--rmax", type=float, default=2.5, help="Extremo derecho del dominio")
    ap.add_argument("--center", type=float, default=0.6, help="Centro del bump")
    ap.add_argument("--L", type=float, default=0.3, help="Semianchura del bump")
    ap.add_argument("--gamma", type=float, default=5.0/3.0, help="Índice politrópico Γ")
    ap.add_argument("--K", type=float, default=100.0, help="Constante politrópica K")
    ap.add_argument("--tfinal", type=float, default=0.8, help="Tiempo final (antes de la cáustica)")
    ap.add_argument("--cfl", type=float, default=0.2, help="Factor CFL")
    ap.add_argument("--out-prefix", type=str, default="smoothflow_cart", help="Prefijo de ficheros de salida")
    args = ap.parse_args()

    eos = PolytropicEOS(K=args.K, gamma=args.gamma)
    eos_params = {'K': args.K, 'gamma': args.gamma}

    # --- solución de referencia (alta resolución, WENO-Z por defecto) ---
    n_ref = args.ref_mult * max(args.res)
    print(f"[REF] corriendo referencia: recon=wenoz, N={n_ref}, tf={args.tfinal}")
    r_ref, rho_ref, v_ref, p_ref = evolve_solution(
        n_ref, args.rmax, eos, "wenoz", args.center, args.L, args.tfinal, args.cfl
    )
    print(f"[REF] referencia completada\n")

    # --- Preparar todas las tareas para ejecutar en paralelo ---
    tasks = []
    for method in args.methods:
        for N in args.res:
            tasks.append((N, args.rmax, eos_params, method, args.center, args.L, args.tfinal, args.cfl))

    print(f"[INFO] Ejecutando {len(tasks)} simulaciones en paralelo...\n")

    # --- Ejecutar todas las simulaciones en paralelo ---
    results = {}  # dict: (method, N) -> resultado

    # Reservar espacio para las barras de progreso
    print("\n" * len(tasks))

    with ProcessPoolExecutor() as executor:
        # Enviar todas las tareas con posición para las barras
        future_to_info = {}
        for i, task in enumerate(tasks):
            future = executor.submit(run_single_simulation, *task, i)
            future_to_info[future] = (task[3], task[0])  # (method, N)

        # Recolectar resultados a medida que se completan
        for future in as_completed(future_to_info):
            result = future.result()
            method, N = result['method'], result['N']
            results[(method, N)] = result

    print(f"\n\n[INFO] Todas las simulaciones completadas. Calculando errores y órdenes...\n")

    # --- Procesar resultados por método ---
    all_rows = []
    for method in args.methods:
        print(f"=== {method.upper()} ===")
        h_list, e_rho_list, e_v_list, e_p_list = [], [], [], []
        for N in args.res:
            res = results[(method, N)]
            r, rho, v, p = res['r'], res['rho'], res['v'], res['p']
            h = res['h']
            h_list.append(h)
            e_rho = l1_error_against_reference(r, rho, r_ref, rho_ref)
            e_v   = l1_error_against_reference(r, v,   r_ref, v_ref)
            e_p   = l1_error_against_reference(r, p,   r_ref, p_ref)
            e_rho_list.append(e_rho); e_v_list.append(e_v); e_p_list.append(e_p)
            print(f"  N={N:4d}  L1(ρ)={e_rho:.3e}  L1(v)={e_v:.3e}  L1(p)={e_p:.3e}")

        # órdenes de convergencia (pareados)
        p_rho = compute_orders(h_list, e_rho_list)
        p_v   = compute_orders(h_list, e_v_list)
        p_p   = compute_orders(h_list, e_p_list)

        # registrar filas
        for i, N in enumerate(args.res):
            all_rows.append({
                "method": method, "N": N, "h": h_list[i],
                "L1_rho": e_rho_list[i], "L1_v": e_v_list[i], "L1_p": e_p_list[i],
                "order_rho": (p_rho[i-1] if i>0 else np.nan),
                "order_v":   (p_v[i-1]   if i>0 else np.nan),
                "order_p":   (p_p[i-1]   if i>0 else np.nan),
            })

        # Plot de convergencia (ρ)
        plt.figure(figsize=(7,5))
        plt.loglog(h_list, e_rho_list, 'o-', lw=2.0, label='L1(ρ₀)')
        if len(h_list) >= 2:
            h0, e0 = h_list[0], e_rho_list[0]
            hs = np.array(h_list)
            c2 = e0/(h0**2 + 1e-30); c5 = e0/(h0**5 + 1e-30)
            plt.loglog(hs, c2*hs**2, ':', label='O(h²) guía')
            plt.loglog(hs, c5*hs**5, ':', label='O(h⁵) guía')
        plt.gca().invert_xaxis()
        plt.xlabel('h = Δx'); plt.ylabel('L1(ρ₀)')
        plt.title(f'Convergencia — {method.upper()} (t={args.tfinal})')
        plt.grid(True, which='both', alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_conv_rho_{method}.png", dpi=140, bbox_inches="tight")
        plt.close()

        # Comparación con la referencia (ρ) en la malla más fina de este método
        Nfin = args.res[-1]
        res_fin = results[(method, Nfin)]
        r_fin, rho_fin = res_fin['r'], res_fin['rho']
        plt.figure(figsize=(9,5))
        plt.plot(r_ref, rho_ref, 'k-', lw=2.0, label='Referencia (WENO-Z, alta res.)')
        plt.plot(r_fin, rho_fin, '--', lw=2.0, label=f'{method.upper()} (N={Nfin})')
        plt.xlim(0, args.rmax); plt.xlabel('x'); plt.ylabel('ρ₀')
        plt.title(f'Adiabatic smooth flow — {method.upper()}  (t={args.tfinal})')
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_rho_{method}_N{Nfin}.png", dpi=140, bbox_inches="tight")
        plt.close()

    # Guardar CSV
    csv_name = f"{args.out_prefix}_convergence.csv"
    with open(csv_name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"\nResultados guardados en {csv_name}")
    print("Gráficos: *_conv_rho_<method>.png y *_rho_<method>_N<fin>.png")

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
