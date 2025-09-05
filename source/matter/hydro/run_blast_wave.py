#!/usr/bin/env python
"""
Test de Blast Wave usando el framework engrenage completo.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp
from tqdm import tqdm

# Importaciones de Engrenage
from core.grid import Grid
from core.spacing import LinearSpacing, SpacingExtent
from core.statevector import StateVector
from core.rhsevolution import get_rhs
from backgrounds.sphericalbackground import FlatSphericalBackground
from bssn.bssnstatevariables import NUM_BSSN_VARS, idx_lapse
from bssn.bssnvars import BSSNVars

# Importaciones de Hidrodinámica
from matter.hydro.relativistic_fluid import RelativisticFluid
from matter.hydro.eos import IdealGasEOS
from matter.hydro.reconstruction import MinmodReconstruction
from matter.hydro.riemann import HLLERiemannSolver
from source.matter.hydro.cons2prim import cons2prim

def setup_blast_wave_initial_data(r, p_inner, p_outer, rho_inner, rho_outer, r_discontinuity, eos):
    """Condiciones iniciales para blast wave."""
    N = len(r)
    rho0 = np.where(r < r_discontinuity, rho_inner, rho_outer)
    pressure = np.where(r < r_discontinuity, p_inner, p_outer)
    vr = np.zeros(N)
    
    eps = eos.eps_from_rho_p(rho0, pressure)
    h = 1.0 + eps + pressure / rho0
    W = np.ones(N)
    
    D = rho0 * W
    Sr = np.zeros(N)
    tau = rho0 * h * W**2 - pressure - D
    
    return {'D': D, 'Sr': Sr, 'tau': tau, 
            'rho0': rho0, 'vr': vr, 'pressure': pressure}

def run_simulation():
    print("🚀 Simulación Blast Wave con engrenage...")
    
    # Parámetros
    N = 200
    r_max = 1.0
    t_final = 0.1
    
    # Física
    p_inner, p_outer = 1.0, 0.01
    rho_inner, rho_outer = 1.0, 0.125
    r_discontinuity = 0.5
    gamma_eos = 1.4
    
    # Grid
    spacing = LinearSpacing(N, r_max, SpacingExtent.HALF)
    
    # Materia
    eos = IdealGasEOS(gamma=gamma_eos)
    hydro_fluid = RelativisticFluid(
        eos=eos,
        spacetime_mode="fixed_minkowski",
        atmosphere_rho=1e-10,
        reconstructor=MinmodReconstruction(limiter_type="minmod"),
        riemann_solver=HLLERiemannSolver()
    )
    
    state_vec = StateVector(hydro_fluid)
    grid = Grid(spacing, state_vec)
    r = grid.r
    background = FlatSphericalBackground(r)
    
    # Estado inicial
    initial = setup_blast_wave_initial_data(r, p_inner, p_outer, 
                                           rho_inner, rho_outer, 
                                           r_discontinuity, eos)
    
    state = np.zeros((grid.NUM_VARS, N))
    state[idx_lapse, :] = 1.0  # Minkowski
    state[hydro_fluid.idx_D, :] = initial['D']
    state[hydro_fluid.idx_Sr, :] = initial['Sr']
    state[hydro_fluid.idx_tau, :] = initial['tau']
    
    # Wrapper para RHS
    pbar = tqdm(total=100, desc="Evolucionando", disable=True)
    
    def rhs_wrapper(t, y):
        return get_rhs(t, y, grid, background, hydro_fluid, pbar, [0, 0.01])
    
    # Integración con RK45
    print("Integrando con RK45...")
    solution = solve_ivp(
        rhs_wrapper,
        [0, t_final],
        state.flatten(),
        method='RK45',
        rtol=1e-6,
        atol=1e-9,
        t_eval=np.linspace(0, t_final, 11),
        max_step=0.0005  # Paso máximo pequeño para estabilidad
    )
    
    if not solution.success:
        print(f"❌ Error: {solution.message}")
        return
    
    # Visualización
    print("Generando gráficos...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for i, t in enumerate(solution.t[::2]):  # Cada 2 tiempos
        state_at_t = solution.y[:, i*2].reshape(grid.NUM_VARS, -1)
        
        # Convertir a primitivas
        D = state_at_t[hydro_fluid.idx_D]
        Sr = state_at_t[hydro_fluid.idx_Sr]
        tau = state_at_t[hydro_fluid.idx_tau]
        
        prims = cons2prim((D, Sr, tau), eos)
        
        color = plt.cm.viridis(i / 5)
        axes[0,0].plot(r, prims['rho0'], label=f't={t:.3f}', color=color)
        axes[0,1].plot(r, prims['p'], color=color)
        axes[1,0].plot(r, prims['vr'], color=color)
        axes[1,1].plot(r, prims['W'], color=color)
    
    axes[0,0].set_xlabel('r'); axes[0,0].set_ylabel('Densidad')
    axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].set_xlabel('r'); axes[0,1].set_ylabel('Presión')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].set_xlabel('r'); axes[1,0].set_ylabel('Velocidad')
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].set_xlabel('r'); axes[1,1].set_ylabel('Factor Lorentz')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle('Blast Wave con engrenage + RK45')
    plt.tight_layout()
    plt.savefig('blast_wave_engrenage.png', dpi=150)
    plt.show()
    
    print("✅ Simulación completada!")
    
    # Análisis final
    final_state = solution.y[:, -1].reshape(grid.NUM_VARS, -1)
    final_prims = cons2prim(
        (final_state[hydro_fluid.idx_D], 
         final_state[hydro_fluid.idx_Sr], 
         final_state[hydro_fluid.idx_tau]), eos
    )
    
    print(f"\n📊 Estado final (t={t_final}):")
    print(f"  Velocidad máxima: {np.max(np.abs(final_prims['vr'])):.4f}c")
    print(f"  Densidad máxima: {np.max(final_prims['rho0']):.4f}")
    print(f"  Factor Lorentz máximo: {np.max(final_prims['W']):.4f}")

if __name__ == "__main__":
    run_simulation()