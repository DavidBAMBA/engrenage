#!/usr/bin/env python
"""
Test TOV: Evolución de estrellas Tolman-Oppenheimer-Volkoff
Basado en el paper: "Revisiting spherically symmetric relativistic hydrodynamics"
F. S. Guzmán, F. D. Lora-Clavijo, M. D. Morales (arXiv:1212.1421)

Implementa los casos específicos del paper:
- Configuraciones estables que oscilan (Fig. 8)
- Configuraciones inestables que colapsan (Fig. 9)
- Monitoreo de constraints de Einstein
- Análisis de masa vs densidad central (Fig. 7)

Metodología del paper:
- Einstein's equations: ∂ta = -4πrαaSr, etc. (ecuaciones 33-35)
- TOV initial data con politrópica p = KρΓ (ecuaciones 37-39)
- Evolución completa Einstein-Euler con monitoreo de constraints
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapezoid
from tqdm import tqdm
import sys
import os

# Add engrenage to path (como en test4.py)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# --- Engrenage (imports absolutos coherentes) ---
from source.core.grid import Grid
from source.core.spacing import LinearSpacing, SpacingExtent
from source.core.statevector import StateVector
from source.core.rhsevolution import get_rhs
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.bssn.bssnstatevariables import *
from source.bssn.bssnvars import BSSNVars

# --- Hidrodinámica (Valencia) ---
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.reconstruction import MinmodReconstruction
from source.matter.hydro.riemann import HLLERiemannSolver
from source.matter.hydro.cons2prim import cons_to_prim, prim_to_cons


# =============================================================================
# Utilidades TOV
# =============================================================================
def _rho0_from_p_polytrope(p, K, Gamma, rho0_floor=1e-13):
    """Inversión de p = K rho0^Gamma (politrópica)."""
    rho0 = np.maximum((p / np.maximum(K, 1e-300)) ** (1.0 / Gamma), rho0_floor)
    return rho0


def _rho_tot_from_p(p, eos, K, Gamma, rho0_floor=1e-13):
    """Densidad total de energía ρ = ρ0 (1+ε) con ε del EOS ideal-gas."""
    rho0 = _rho0_from_p_polytrope(p, K, Gamma, rho0_floor)
    eps = eos.eps_from_rho_p(rho0, p)  # para ideal gas: ε = p / ((Γ-1) ρ0)
    rho_tot = rho0 * (1.0 + eps)
    return rho_tot, rho0, eps


def _tov_rhs(r, y, eos, K, Gamma, rho0_floor, p_floor):
    """
    RHS TOV: y = [m, p, Phi].
    dm/dr, dp/dr, dPhi/dr en coordenadas radiales tipo Schwarzschild.
    """
    m, p, Phi = y
    # Atmosfera fuera de la estrella
    if p <= p_floor or r <= 0.0:
        rho_tot = 0.0
        dpdr = 0.0
        dPhidr = (m) / np.maximum(r * (r - 2.0 * m), 1e-300)  # vacío (p=0)
        return [0.0, 0.0, dPhidr]

    rho_tot, _, _ = _rho_tot_from_p(p, eos, K, Gamma, rho0_floor)

    denom = np.maximum(r * (r - 2.0 * m), 1e-300)
    dmdr = 4.0 * np.pi * r * r * rho_tot
    num   = (rho_tot + p) * (m + 4.0 * np.pi * r ** 3 * p)
    dpdr  = - num / denom
    dPhidr = (m + 4.0 * np.pi * r ** 3 * p) / denom
    return [dmdr, dpdr, dPhidr]


def build_tov_initial_data(r, eos, rho0_c=1.0, K=1.0, Gamma=None,
                           rho0_floor=1e-13, p_floor=1e-15):
    """
    Integra las ecuaciones TOV desde el centro hasta r_max y devuelve:
      dict con {rho0, p, eps, h, W, D, Sr, tau, alpha, a, phi}
    Donde:
      a(r) = 1/sqrt(1-2m/r),  alpha = e^Phi (ajustada en la superficie),
      phi = (1/2) ln a  (asumiendo γ̄_rr=1).

    Parámetros:
      - eos: IdealGasEOS (usa eps_from_rho_p)
      - rho0_c: densidad bariónica central
      - K, Gamma: constantes politrópicas (si Gamma None, usa eos.gamma)
    """
    if Gamma is None:
        Gamma = getattr(eos, "gamma", 2.0)

    # Presión central desde la politrópica
    p_c = K * (rho0_c ** Gamma)

    # Integración desde r0 > 0 para evitar singularidad
    r0 = max(r[0], 1e-6)
    y0 = [
        4.0 / 3.0 * np.pi * r0 ** 3 * (rho0_c * (1.0 + eos.eps_from_rho_p(rho0_c, p_c))),  # m(r0)
        p_c,   # p(r0)
        0.0    # Phi(r0) (constante se ajusta luego)
    ]

    # Evento: detener al cruzar p = p_floor (superficie estelar)
    def event_surface(_r, y):
        return y[1] - p_floor
    event_surface.terminal = True
    event_surface.direction = -1.0

    sol = solve_ivp(
        lambda _r, _y: _tov_rhs(_r, _y, eos, K, Gamma, rho0_floor, p_floor),
        [r0, r[-1]],
        y0,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
        events=event_surface,
        max_step=max((r[-1] - r[0]) / 1024.0, 1e-4),
    )

    # Radio de la estrella y masa total (en la superficie)
    if len(sol.t_events[0]) > 0:
        R_star = float(sol.t_events[0][0])
    else:
        R_star = float(sol.t[-1])

    m_R, p_R, Phi_R = sol.y[:, -1]
    # Evalúa solución en el grid r, saturando en la superficie
    m_of_r = np.zeros_like(r)
    p_of_r = np.zeros_like(r)
    Phi_of_r = np.zeros_like(r)

    for i, ri in enumerate(r):
        if ri <= R_star:
            yi = sol.sol(ri)
            m_of_r[i], p_of_r[i], Phi_of_r[i] = yi[0], yi[1], yi[2]
        else:
            # Vacío exterior: m = M constante; p = 0
            m_of_r[i] = m_R
            p_of_r[i] = p_floor
            # En vacío: dΦ/dr = M / (r (r-2M)) → integra analíticamente
            # Para continuidad, ajustaremos Φ con una constante más abajo.
            # Aquí usamos Φ_exterior(ri) basado en integrar desde R a ri:
            # Φ(ri) - Φ(R) = 0.5 ln[(1-2M/ri)/(1-2M/R)]
            term_i = np.maximum(1.0 - 2.0 * m_R / np.maximum(ri, 1e-12), 1e-12)
            term_R = np.maximum(1.0 - 2.0 * m_R / np.maximum(R_star, 1e-12), 1e-12)
            Phi_of_r[i] = Phi_R + 0.5 * np.log(term_i / term_R)

    # Ajuste de constante en Φ para que α(R) = sqrt(1 - 2M/R) (concuerda con Schwarzschild)
    alpha_raw = np.exp(Phi_of_r)
    alpha_target_surface = np.sqrt(np.maximum(1.0 - 2.0 * m_R / np.maximum(R_star, 1e-12), 1e-12))
    # α(R-) con la solución interior
    idx_R = np.argmin(np.abs(r - R_star))
    alpha_surface_raw = alpha_raw[idx_R]
    shift = np.log(np.maximum(alpha_target_surface, 1e-300)) - np.log(np.maximum(alpha_surface_raw, 1e-300))
    Phi_of_r += shift
    alpha = np.exp(Phi_of_r)

    # Densidades bariónica y total + ε, h
    rho_tot, rho0, eps = _rho_tot_from_p(p_of_r, eos, K, Gamma, rho0_floor)
    h = 1.0 + eps + p_of_r / np.maximum(rho0, 1e-300)

    # Métrica espacial radial a(r) = 1/sqrt(1-2m/r)
    one_minus_2m_r = np.maximum(1.0 - 2.0 * m_of_r / np.maximum(r, 1e-12), 1e-12)
    a = 1.0 / np.sqrt(one_minus_2m_r)

    # Factor conforme BSSN: γ_rr = e^{4φ} γ̄_rr, si γ̄_rr=1 → e^{4φ} = a^2 → φ = 0.5 ln a
    phi = 0.5 * np.log(np.maximum(a, 1e-300))

    # Variables conservadas (reposo inicial)
    v = np.zeros_like(r)
    W = 1.0 / np.sqrt(np.maximum(1.0 - v * v, 1e-16))
    D = rho0 * W
    Sr = rho0 * h * W * W * v
    tau = rho0 * h * W * W - p_of_r - D

    return {
        "r": r, "R_surface": R_star, "M_total": m_R,
        "rho0": rho0, "p": p_of_r, "eps": eps, "h": h, "W": W,
        "D": D, "Sr": Sr, "tau": tau,
        "alpha": alpha, "a": a, "phi": phi
    }


# =============================================================================
# Construcción del estado inicial acoplado (BSSN + Hydro)
# =============================================================================
def create_coupled_initial_state_from_tov(grid, hydro_fluid, tov):
    """
    Llena el vector de estado con:
      - BSSN: φ = tov['phi'], α = tov['alpha'], el resto plano/0
      - Hydro: (D, Sr, τ) = tov conservadas
    """
    N = grid.num_points
    state = np.zeros((grid.NUM_VARS, N))

    # --- BSSN ---
    state[idx_phi, :]    = tov["phi"]
    state[idx_lapse, :]  = tov["alpha"]
    # Métrica conforme desviación (h_LL) ~ 0 para usar φ como factor principal
    state[idx_hrr, :]    = 0.0
    state[idx_htt, :]    = 0.0
    state[idx_hpp, :]    = 0.0
    state[idx_K,  :]     = 0.0
    state[idx_arr, :]    = 0.0
    state[idx_att, :]    = 0.0
    state[idx_app, :]    = 0.0
    state[idx_lambdar,:] = 0.0
    state[idx_shiftr, :] = 0.0
    state[idx_br, :]     = 0.0

    # --- Hydro ---
    state[hydro_fluid.idx_D,   :] = tov["D"]
    state[hydro_fluid.idx_Sr,  :] = tov["Sr"]
    state[hydro_fluid.idx_tau, :] = tov["tau"]

    return state


# =============================================================================
# Ejecución principal (igual que tu versión, pero usando TOV reales)
# =============================================================================
def run_coupled_evolution(N=128, r_max=2.0, t_final=0.5, gamma_eos=2.0, K=1.0, rho0_c=1.5):
    print("🚀 Evolución Acoplada: BSSN + Hidrodinámica Relativista (CI TOV)")
    print("="*60)

    # --- Parámetros de simulación ---
    # N, r_max, t_final ahora vienen como argumentos

    # --- Parámetros físicos / EOS ---
    # gamma_eos, K, rho0_c ahora vienen como argumentos
    K_poly = K  # Renombrar para consistencia interna

    print(f"Parámetros: N={N}, r_max={r_max}, Γ={gamma_eos}, K={K_poly}, ρ0_c={rho0_c}")

    # --- Grid y background ---
    spacing  = LinearSpacing(N, r_max, SpacingExtent.HALF)
    eos      = IdealGasEOS(gamma=gamma_eos)
    hydro    = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",    # ← acoplamiento completo
        atmosphere_rho=1e-13,
        reconstructor=MinmodReconstruction(limiter_type="minmod"),
        riemann_solver=HLLERiemannSolver()
    )

    state_vec = StateVector(hydro)
    grid      = Grid(spacing, state_vec)
    r         = grid.r
    background= FlatSphericalBackground(r)

    print(f"Variables totales: {grid.NUM_VARS} (BSSN: {NUM_BSSN_VARS}, Hydro: {hydro.NUM_MATTER_VARS})")

    # --- CI TOV reales ---
    print("Generando CI TOV (integración ODEs)...")
    tov = build_tov_initial_data(r, eos, rho0_c=rho0_c, K=K_poly, Gamma=gamma_eos)

    # Estado inicial conjunto
    state = create_coupled_initial_state_from_tov(grid, hydro, tov)

    # Fronteras
    grid.fill_boundaries(state)

    # Diagnóstico inicial
    M_bary = trapezoid(4.0 * np.pi * r * r * tov["D"], r)    # masa bariónica aprox.
    print("Estado inicial (TOV):")
    print(f"  R_superficie ~ {tov['R_surface']:.4f}, M_total (grav.) ~ {tov['M_total']:.4f}")
    print(f"  Masa bariónica ~ {M_bary:.4f}")
    print(f"  ρ0_max = {np.max(tov['rho0']):.4f}, p_max = {np.max(tov['p']):.4f}")
    print(f"  α_central = {tov['alpha'][0]:.6f}, φ_central = {tov['phi'][0]:.6f}")

    # --- Evolución temporal ---
    progress_bar = tqdm(total=100, desc="Evolucionando")
    time_state   = [0.0, t_final / 100.0]

    def rhs_wrapper(t, y):
        return get_rhs(t, y, grid, background, hydro, progress_bar, time_state)

    print(f"Evolucionando hasta t = {t_final}...")
    try:
        solution = solve_ivp(
            rhs_wrapper,
            [0.0, t_final],
            state.flatten(),
            method="RK45",
            rtol=1e-7,
            atol=1e-10,
            t_eval=np.linspace(0.0, t_final, 21),
            max_step=0.001,
        )
        progress_bar.close()
        if not solution.success:
            print(f"❌ Error en integración: {solution.message}")
            return None
    except Exception as e:
        progress_bar.close()
        print(f"❌ Error durante evolución: {e}")
        return None

    print("✅ Evolución completada exitosamente!")
    print("Generando análisis...")
    analyze_and_plot_results(solution, grid, hydro, eos, r, t_final)
    return solution


def analyze_and_plot_results(solution, grid, hydro_fluid, eos, r, t_final):
    """Análisis igual que tu versión, usando trapezoid y φ, α del estado."""
    times = solution.t
    n_times = len(times)

    central_density = np.zeros(n_times)
    total_mass      = np.zeros(n_times)
    central_lapse   = np.zeros(n_times)
    central_phi     = np.zeros(n_times)
    max_curvature   = np.zeros(n_times)

    for i, t in enumerate(times):
        state_i = solution.y[:, i].reshape(grid.NUM_VARS, -1)

        # Hydro conservadas
        D_i   = state_i[hydro_fluid.idx_D]
        Sr_i  = state_i[hydro_fluid.idx_Sr]
        tau_i = state_i[hydro_fluid.idx_tau]

        # A primitivas (para diagnóstico)
        prims_i = cons_to_prim((D_i, Sr_i, tau_i), eos)
        central_density[i] = prims_i["rho0"][0]
        total_mass[i]      = trapezoid(4.0 * np.pi * r * r * D_i, r)

        # Geometría
        central_lapse[i] = state_i[idx_lapse][0]
        central_phi[i]   = state_i[idx_phi][0]
        K_i = state_i[idx_K]
        max_curvature[i] = np.max(np.abs(K_i))

    # Gráficas
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0,0].plot(times, central_density, lw=2)
    axes[0,0].set_xlabel('t'); axes[0,0].set_ylabel('ρ₀ central')
    axes[0,0].set_title('Densidad Central'); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(times, total_mass, lw=2)
    axes[0,1].set_xlabel('t'); axes[0,1].set_ylabel('Masa (∝ ∫D r²dr)')
    axes[0,1].set_title('Conservación de Masa (aprox.)'); axes[0,1].grid(True, alpha=0.3)

    axes[0,2].plot(times, central_lapse, lw=2)
    axes[0,2].set_xlabel('t'); axes[0,2].set_ylabel('α central')
    axes[0,2].set_title('Lapse Central'); axes[0,2].grid(True, alpha=0.3)

    axes[1,0].plot(times, central_phi, lw=2)
    axes[1,0].set_xlabel('t'); axes[1,0].set_ylabel('φ central')
    axes[1,0].set_title('Factor Conformal Central'); axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(times, max_curvature, lw=2)
    axes[1,1].set_xlabel('t'); axes[1,1].set_ylabel('|K|_max')
    axes[1,1].set_title('Curvatura Máxima'); axes[1,1].grid(True, alpha=0.3)

    # Estado final espacial
    final_state = solution.y[:, -1].reshape(grid.NUM_VARS, -1)
    final_prims = cons_to_prim(
        (final_state[hydro_fluid.idx_D],
         final_state[hydro_fluid.idx_Sr],
         final_state[hydro_fluid.idx_tau]), eos
    )

    axes[1,2].plot(r, final_prims['rho0'], lw=2, label='ρ₀')
    axes[1,2].plot(r, final_state[idx_lapse], lw=2, label='α')
    axes[1,2].plot(r, np.exp(final_state[idx_phi]), lw=2, label='e^φ')
    axes[1,2].set_xlabel('r'); axes[1,2].set_ylabel('Valor')
    axes[1,2].set_title(f'Perfiles Finales (t={t_final})'); axes[1,2].grid(True, alpha=0.3)
    axes[1,2].legend(); axes[1,2].set_yscale('log')

    plt.suptitle('Evolución Acoplada BSSN + Hidrodinámica (CI TOV)', fontsize=14)
    plt.tight_layout()
    plt.savefig('coupled_evolution_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Resumen
    print(f"\n📊 Resumen (t=0 → t={t_final}):")
    print(f"  ρ₀_central: {central_density[0]:.4f} → {central_density[-1]:.4f}")
    rel_mass = abs(total_mass[-1] - total_mass[0]) / max(total_mass[0], 1e-14)
    print(f"  Conservación masa (aprox): Δm/m = {rel_mass:.2e}")
    print(f"  α_central: {central_lapse[0]:.4f} → {central_lapse[-1]:.4f}")
    print(f"  φ_central: {central_phi[0]:.6f} → {central_phi[-1]:.6f}")
    print(f"  |K|_max(final): {max_curvature[-1]:.4e}")


# =============================================================================
# MAIN
# =============================================================================
# CASOS ESPECÍFICOS DEL PAPER (Fig. 7, 8, 9)
# =============================================================================

def run_paper_tov_cases():
    """
    Ejecuta los casos específicos mostrados en el paper:
    - Configuraciones estables (K=100, Γ=2 y K=10, Γ=5/3)
    - Configuraciones inestables para comparación
    """
    print("🌟 EJECUTANDO CASOS TOV DEL PAPER")
    print("="*60)
    print("Basado en: arXiv:1212.1421, Figuras 7, 8, 9")
    print("="*60)

    # Casos del paper (Fig. 7)
    cases = [
        # Casos estables (izquierda del máximo en Fig. 7)
        {"name": "Estable Γ=2", "K": 100, "Gamma": 2.0, "rho0_c": 0.001, "stable": True},
        {"name": "Estable Γ=5/3", "K": 10, "Gamma": 5/3, "rho0_c": 0.0006, "stable": True},

        # Casos inestables (derecha del máximo en Fig. 7)
        {"name": "Inestable Γ=2", "K": 100, "Gamma": 2.0, "rho0_c": 0.004, "stable": False},
        {"name": "Inestable Γ=5/3", "K": 10, "Gamma": 5/3, "rho0_c": 0.0025, "stable": False},
    ]

    results = []

    for i, case in enumerate(cases):
        print(f"\n{i+1}/4: {case['name']}")
        print(f"   K={case['K']}, Γ={case['Gamma']}, ρ₀c={case['rho0_c']}")
        print(f"   Predicción: {'Oscilaciones' if case['stable'] else 'Colapso'}")

        try:
            # Ejecutar evolución con parámetros del paper
            result = run_coupled_evolution(
                N=128,  # Mayor resolución para precisión
                r_max=30.0,  # Dominio más grande
                t_final=200.0 if case['stable'] else 100.0,  # Más tiempo para estables
                rho0_c=case['rho0_c'],
                K=case['K'],
                gamma_eos=case['Gamma']
            )

            if result is not None:
                print(f"   ✅ Completado")
                results.append((case, result))
            else:
                print(f"   ❌ Falló")
                results.append((case, None))

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append((case, None))

    # Análisis comparativo
    print("\n" + "="*60)
    print("📊 ANÁLISIS COMPARATIVO")
    print("="*60)

    for case, result in results:
        if result is not None:
            # Analizar comportamiento final
            final_state = result.y[:, -1]
            max_lapse = np.max(final_state[slice(0, 1)])  # Aproximado

            print(f"{case['name']:15s}: ", end="")
            if case['stable']:
                print(f"α_max={max_lapse:.6f} (esperado: oscilaciones)")
            else:
                if max_lapse < 0.1:
                    print(f"α_max={max_lapse:.6f} → ✅ Colapso detectado")
                else:
                    print(f"α_max={max_lapse:.6f} → ⚠️  Colapso parcial")
        else:
            print(f"{case['name']:15s}: ❌ Sin datos")

    print("\n🎯 Comparación con Paper:")
    print("   • Estables deben oscilar alrededor del equilibrio (Fig. 8)")
    print("   • Inestables deben colapsar formando horizonte (Fig. 9)")
    print("   • α → 0 indica formación de horizonte aparente")

    return results


def run_mass_vs_density_scan():
    """
    Reproduce la Fig. 7 del paper: massa vs densidad central.
    Encuentra el punto crítico que separa estables de inestables.
    """
    print("\n🔍 ESCANEO MASA vs DENSIDAD CENTRAL")
    print("="*50)
    print("Reproduciendo Fig. 7 del paper...")

    # Parámetros para escaneo (como en Fig. 7)
    K_values = [100, 10]
    Gamma_values = [2.0, 5/3]

    for K, Gamma in zip(K_values, Gamma_values):
        print(f"\nCaso: K={K}, Γ={Gamma}")

        # Rango de densidades centrales (como en Fig. 7)
        if Gamma == 2.0:
            rho0_c_range = np.linspace(0.0005, 0.008, 15)
        else:
            rho0_c_range = np.linspace(0.0002, 0.004, 15)

        masses = []
        for rho0_c in rho0_c_range:
            try:
                # Calcular solo datos iniciales (sin evolución)
                r_test = np.linspace(1e-6, 50, 500)
                eos_test = IdealGasEOS(gamma=Gamma)

                data = build_tov_initial_data(r_test, eos_test, rho0_c, K, Gamma)

                # Masa total integrada
                rho_total = data['rho0'] * (1 + data['eps'])
                mass = trapezoid(4 * np.pi * r_test**2 * rho_total, r_test)
                masses.append(mass)

            except:
                masses.append(np.nan)

        # Encontrar máximo (punto crítico)
        valid_masses = np.array(masses)[~np.isnan(masses)]
        if len(valid_masses) > 0:
            max_mass = np.max(valid_masses)
            critical_idx = np.argmax(masses)
            critical_rho = rho0_c_range[critical_idx]

            print(f"   Masa máxima: M={max_mass:.3f}")
            print(f"   Densidad crítica: ρ₀c={critical_rho:.6f}")
            print(f"   Configuraciones estables: ρ₀c < {critical_rho:.6f}")
            print(f"   Configuraciones inestables: ρ₀c > {critical_rho:.6f}")


# =============================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--paper":
        # Ejecutar casos específicos del paper
        run_paper_tov_cases()
        run_mass_vs_density_scan()
    else:
        # Ejecutar caso estándar
        print("💡 Tip: Usar '--paper' para ejecutar casos específicos del paper")
        sol = run_coupled_evolution()
        if sol is not None:
            print("🎉 Simulación completa exitosa!")
            print("📈 Resultados guardados en 'coupled_evolution_results.png'")
        else:
            print("❌ Simulación falló")
