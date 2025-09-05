#!/usr/bin/env python
"""
Ejemplo de evolución acoplada completa: BSSN + Hidrodinámica Relativista
Caso: Colapso gravitacional de fluido relativista
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

# Importaciones de Engrenage
from source.core.grid import Grid
from source.core.spacing import LinearSpacing, SpacingExtent  
from source.core.statevector import StateVector
from source.core.rhsevolution import get_rhs
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.bssn.bssnstatevariables import *
from source.bssn.bssnvars import BSSNVars

# Importaciones de Hidrodinámica
from source.matter.hydro.relativistic_fluid import RelativisticFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.reconstruction import MinmodReconstruction
from source.matter.hydro.riemann import HLLERiemannSolver
from source.matter.hydro.cons2prim import cons_to_prim, prim_to_cons

def setup_tolman_oppenheimer_volkoff_initial_data(r, M_central=1.0, R_star=0.8, 
                                                  rho_central=1.0, eos=None):
    """
    Condiciones iniciales aproximadas tipo TOV (Tolman-Oppenheimer-Volkoff)
    para un fluido en equilibrio hidrostático.
    
    Esto es una aproximación simple; para casos reales se resolvería 
    la ecuación TOV completa.
    """
    N = len(r)
    
    # Perfil de densidad tipo gaussiano/exponencial
    sigma = R_star / 3.0
    rho0 = rho_central * np.exp(-(r/sigma)**2)
    rho0 = np.maximum(rho0, 1e-13)  # atmosphere floor
    
    # Presión en equilibrio hidrostático (aproximación)
    # p ~ rho^Γ para politropa
    gamma = eos.gamma if hasattr(eos, 'gamma') else 2.0
    p_central = 0.1 * rho_central**(gamma)
    pressure = p_central * (rho0/rho_central)**(gamma)
    pressure = np.maximum(pressure, 1e-15)
    
    # Velocidad inicial (reposo)
    vr = np.zeros(N)
    
    # Variables derivadas
    eps = eos.eps_from_rho_p(rho0, pressure)
    h = 1.0 + eps + pressure/rho0
    W = 1.0 / np.sqrt(1.0 - vr**2)
    
    # Conservadas
    D = rho0 * W
    Sr = rho0 * h * W**2 * vr  # = 0 inicialmente
    tau = rho0 * h * W**2 - pressure - D
    
    # BSSN: métrica inicial aproximada para el fluido
    # Usamos aproximación de campo débil: φ ~ -M/(8πr)
    #total_mass = np.trapz(4*np.pi * r**2 * rho0, r)
    total_mass = np.trapezoid(4*np.pi * r**2 * rho0, r)

    phi_init = np.zeros(N)
    
    # Solo aplicar corrección gravitacional donde hay materia significativa
    matter_mask = rho0 > 10 * 1e-13
    if np.any(matter_mask):
        # Aproximación de campo débil
        r_matter = r[matter_mask]
        M_enclosed = np.cumsum(4*np.pi * r_matter**2 * rho0[matter_mask]) * (r_matter[1] - r_matter[0])
        phi_correction = -0.1 * M_enclosed / (8*np.pi * np.maximum(r_matter, 1e-10))
        phi_init[matter_mask] = phi_correction
    
    # Otras variables BSSN iniciales (aproximación métrica plana)
    h_LL_init = np.zeros((N, 3, 3))  # desviación de métrica plana
    K_init = np.zeros(N)             # curvatura extrínseca inicial
    a_LL_init = np.zeros((N, 3, 3))  # parte traceless de curvatura extrínseca
    lambda_U_init = np.zeros((N, 3)) # conexión
    shift_U_init = np.zeros((N, 3))  # shift (gauge)
    b_U_init = np.zeros((N, 3))      # derivada temporal de shift
    lapse_init = np.ones(N)          # lapse (gauge)
    
    return {
        # Hidrodinámica
        'D': D, 'Sr': Sr, 'tau': tau,
        'rho0': rho0, 'vr': vr, 'pressure': pressure,
        'eps': eps, 'h': h, 'W': W,
        # BSSN
        'phi': phi_init,
        'h_LL': h_LL_init,
        'K': K_init,
        'a_LL': a_LL_init,
        'lambda_U': lambda_U_init,
        'shift_U': shift_U_init,
        'b_U': b_U_init,
        'lapse': lapse_init
    }

def create_coupled_initial_state(grid, hydro_fluid, initial_data):
    """
    Crea el vector de estado inicial completo combinando BSSN + hidrodinámica.
    """
    N = grid.num_points
    state = np.zeros((grid.NUM_VARS, N))
    
    # BSSN variables
    state[idx_phi, :] = initial_data['phi']
    state[idx_hrr, :] = initial_data['h_LL'][:, 0, 0]  # h_rr
    state[idx_htt, :] = initial_data['h_LL'][:, 1, 1]  # h_θθ  
    state[idx_hpp, :] = initial_data['h_LL'][:, 2, 2]  # h_φφ
    state[idx_K, :] = initial_data['K']
    state[idx_arr, :] = initial_data['a_LL'][:, 0, 0]  # a_rr
    state[idx_att, :] = initial_data['a_LL'][:, 1, 1]  # a_θθ
    state[idx_app, :] = initial_data['a_LL'][:, 2, 2]  # a_φφ
    state[idx_lambdar, :] = initial_data['lambda_U'][:, 0]  # λ^r
    state[idx_shiftr, :] = initial_data['shift_U'][:, 0]    # β^r
    state[idx_br, :] = initial_data['b_U'][:, 0]            # b^r
    state[idx_lapse, :] = initial_data['lapse']             # α
    
    # Hydro variables
    state[hydro_fluid.idx_D, :] = initial_data['D']
    state[hydro_fluid.idx_Sr, :] = initial_data['Sr'] 
    state[hydro_fluid.idx_tau, :] = initial_data['tau']
    
    return state

def run_coupled_evolution():
    """
    Ejecuta una simulación completa de evolución acoplada BSSN + hidrodinámica.
    """
    print("🚀 Evolución Acoplada: BSSN + Hidrodinámica Relativista")
    print("="*60)
    
    # Parámetros de simulación
    N = 128               # resolución
    r_max = 2.0          # dominio espacial  
    t_final = 0.5        # tiempo final
    
    # Parámetros físicos del fluido
    M_central = 0.5      # masa característica
    R_star = 0.8         # radio característica de la estrella
    rho_central = 2.0    # densidad central
    gamma_eos = 2.0      # índice adiabático
    
    print(f"Parámetros: N={N}, r_max={r_max}, M={M_central}, R*={R_star}")
    
    # Setup del grid
    spacing = LinearSpacing(N, r_max, SpacingExtent.HALF)
    
    # Setup de la materia (hidrodinámica)
    eos = IdealGasEOS(gamma=gamma_eos)
    hydro_fluid = RelativisticFluid(
        eos=eos,
        spacetime_mode="dynamic",  # ← Acoplamiento completo
        atmosphere_rho=1e-13,
        reconstructor=MinmodReconstruction(limiter_type="minmod"),
        riemann_solver=HLLERiemannSolver()
    )
    
    # State vector y grid
    state_vec = StateVector(hydro_fluid)
    grid = Grid(spacing, state_vec)
    r = grid.r
    background = FlatSphericalBackground(r)
    
    print(f"Variables totales: {grid.NUM_VARS} (BSSN: {NUM_BSSN_VARS}, Hydro: {hydro_fluid.NUM_MATTER_VARS})")
    
    # Condiciones iniciales
    print("Generando condiciones iniciales...")
    initial_data = setup_tolman_oppenheimer_volkoff_initial_data(
        r, M_central=M_central, R_star=R_star, 
        rho_central=rho_central, eos=eos
    )
    
    # Estado inicial completo
    state = create_coupled_initial_state(grid, hydro_fluid, initial_data)
    
    # Aplicar boundary conditions
    grid.fill_boundaries(state)
    
    # Diagnósticos iniciales
    print("Estado inicial:")
    print(f"  Masa total (hidro): {np.sum(initial_data['D']):.4f}")
    print(f"  Densidad máxima: {np.max(initial_data['rho0']):.4f}")
    print(f"  Presión máxima: {np.max(initial_data['pressure']):.4f}")
    print(f"  φ mínimo: {np.min(initial_data['phi']):.6f}")
    
    # Setup para evolución
    progress_bar = tqdm(total=100, desc="Evolucionando")
    time_state = [0.0, t_final/100]  # para progress bar
    
    def rhs_wrapper(t, y):
        return get_rhs(t, y, grid, background, hydro_fluid, progress_bar, time_state)
    
    # Evolución temporal
    print(f"Evolucionando hasta t = {t_final}...")
    
    try:
        solution = solve_ivp(
            rhs_wrapper,
            [0, t_final],
            state.flatten(),
            method='RK45',
            rtol=1e-7,
            atol=1e-10,
            t_eval=np.linspace(0, t_final, 21),  # 21 puntos de salida
            max_step=0.001  # paso pequeño para estabilidad
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
    
    # Análisis y visualización
    print("Generando análisis...")
    analyze_and_plot_results(solution, grid, hydro_fluid, eos, r, t_final)
    
    return solution

def analyze_and_plot_results(solution, grid, hydro_fluid, eos, r, t_final):
    """
    Analiza los resultados y genera gráficos de la evolución acoplada.
    """
    times = solution.t
    n_times = len(times)
    
    # Extraer evolución de cantidades clave
    central_density = np.zeros(n_times)
    total_mass = np.zeros(n_times)
    central_lapse = np.zeros(n_times)
    central_phi = np.zeros(n_times)
    max_curvature = np.zeros(n_times)
    
    for i, t in enumerate(times):
        state_i = solution.y[:, i].reshape(grid.NUM_VARS, -1)
        
        # Hidrodinámica
        D_i = state_i[hydro_fluid.idx_D]
        Sr_i = state_i[hydro_fluid.idx_Sr]
        tau_i = state_i[hydro_fluid.idx_tau]
        
        # Convertir a primitivas para análisis
        prims_i = cons_to_prim((D_i, Sr_i, tau_i), eos)
        
        central_density[i] = prims_i['rho0'][0]  # densidad central
        total_mass[i] = np.trapz(4*np.pi * r**2 * D_i, r)
        
        # BSSN
        central_lapse[i] = state_i[idx_lapse][0]
        central_phi[i] = state_i[idx_phi][0]
        K_i = state_i[idx_K]
        max_curvature[i] = np.max(np.abs(K_i))
    
    # Plot evolución temporal
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0,0].plot(times, central_density, 'b-', lw=2)
    axes[0,0].set_xlabel('t'); axes[0,0].set_ylabel('ρ₀ central')
    axes[0,0].set_title('Densidad Central'); axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(times, total_mass, 'g-', lw=2)
    axes[0,1].set_xlabel('t'); axes[0,1].set_ylabel('Masa Total')
    axes[0,1].set_title('Conservación de Masa'); axes[0,1].grid(True, alpha=0.3)
    
    axes[0,2].plot(times, central_lapse, 'r-', lw=2)
    axes[0,2].set_xlabel('t'); axes[0,2].set_ylabel('α central')
    axes[0,2].set_title('Lapse Central'); axes[0,2].grid(True, alpha=0.3)
    
    axes[1,0].plot(times, central_phi, 'm-', lw=2)
    axes[1,0].set_xlabel('t'); axes[1,0].set_ylabel('φ central')
    axes[1,0].set_title('Factor Conformal Central'); axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(times, max_curvature, 'c-', lw=2)
    axes[1,1].set_xlabel('t'); axes[1,1].set_ylabel('|K|_max')
    axes[1,1].set_title('Curvatura Máxima'); axes[1,1].grid(True, alpha=0.3)
    
    # Estado final espacial
    final_state = solution.y[:, -1].reshape(grid.NUM_VARS, -1)
    final_prims = cons_to_prim(
        (final_state[hydro_fluid.idx_D], 
         final_state[hydro_fluid.idx_Sr], 
         final_state[hydro_fluid.idx_tau]), eos
    )
    
    axes[1,2].plot(r, final_prims['rho0'], 'b-', label='ρ₀', lw=2)
    axes[1,2].plot(r, final_state[idx_lapse], 'r-', label='α', lw=2)
    axes[1,2].plot(r, np.exp(final_state[idx_phi]), 'g-', label='e^φ', lw=2)
    axes[1,2].set_xlabel('r'); axes[1,2].set_ylabel('Valor')
    axes[1,2].set_title(f'Perfiles Finales (t={t_final})'); axes[1,2].grid(True, alpha=0.3)
    axes[1,2].legend(); axes[1,2].set_yscale('log')
    
    plt.suptitle('Evolución Acoplada BSSN + Hidrodinámica', fontsize=14)
    plt.tight_layout()
    plt.savefig('coupled_evolution_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Resumen numérico
    print(f"\n📊 Resumen de Evolución (t=0 → t={t_final}):")
    print(f"  Densidad central: {central_density[0]:.4f} → {central_density[-1]:.4f}")
    print(f"  Conservación masa: Δm/m = {abs(total_mass[-1] - total_mass[0])/total_mass[0]:.2e}")
    print(f"  Lapse central: {central_lapse[0]:.4f} → {central_lapse[-1]:.4f}")
    print(f"  φ central: {central_phi[0]:.6f} → {central_phi[-1]:.6f}")
    print(f"  Curvatura máxima final: {max_curvature[-1]:.4f}")
    
    # Verificar si el colapso está progresando
    if central_lapse[-1] < 0.8 * central_lapse[0]:
        print("  🌌 Indicios de colapso gravitacional!")
    else:
        print("  ⚖️  Evolución estable (no colapso significativo)")

if __name__ == "__main__":
    # Ejecutar simulación acoplada
    solution = run_coupled_evolution()
    
    if solution is not None:
        print("🎉 Simulación completa exitosa!")
        print("📈 Resultados guardados en 'coupled_evolution_results.png'")
    else:
        print("❌ Simulación falló")