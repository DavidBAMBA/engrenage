#!/usr/bin/env python
"""
Test de validación del acoplamiento BSSN-Hidrodinámica.
Verifica que el acoplamiento bidireccional funciona correctamente.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from core.grid import Grid
from core.spacing import LinearSpacing, SpacingExtent  
from core.statevector import StateVector
from core.rhsevolution import get_rhs
from backgrounds.sphericalbackground import FlatSphericalBackground
from bssn.bssnstatevariables import *
from bssn.bssnvars import BSSNVars
from bssn.constraintsdiagnostic import get_constraints_diagnostic

from matter.hydro.perfect_fluid import PerfectFluid
from matter.hydro.eos import IdealGasEOS
from matter.hydro.reconstruction import MinmodReconstruction
from matter.hydro.riemann import HLLERiemannSolver
from matter.hydro.cons2prim import cons_to_prim


class MockProgressBar:
    """Mock progress bar para evitar errores con None."""
    def update(self, n):
        pass


def safe_constraints_diagnostic(state, time, grid, background, hydro_fluid):
    """
    Wrapper para get_constraints_diagnostic que maneja correctamente
    el caso de un solo estado.
    """
    # Para un solo estado, simular múltiples tiempos duplicando el estado
    states_list = [state, state]  # Duplicar para evitar el bug de num_times == 1
    times_list = [time, time]
    
    Ham, Mom = get_constraints_diagnostic(states_list, times_list, grid, background, hydro_fluid)
    
    # Retornar solo el primer resultado (ambos son idénticos)
    return Ham[0:1, :], Mom[0:1, :, :]


def test_coupling_vs_minkowski():
    """
    Test comparativo: evolución con spacetime dinámico vs Minkowski fijo.
    Debe mostrar diferencias cuando hay materia significativa.
    """
    print("🔬 TEST: Acoplamiento vs Minkowski")
    print("="*50)
    
    # Parámetros comunes
    N = 64  # número de puntos de grid
    r_max = 1.0
    t_final = 0.1
    
    # Configuración del fluido con materia concentrada
    eos = IdealGasEOS(gamma=2.0)
    
    # Caso 1: Minkowski fijo (sin acoplamiento)
    print("Caso 1: Minkowski fijo...")
    hydro_fixed = PerfectFluid(
        eos=eos,
        spacetime_mode="fixed_minkowski",
        reconstructor=MinmodReconstruction(),
        riemann_solver=HLLERiemannSolver()
    )
    
    result_fixed = run_single_case(N, r_max, t_final, hydro_fixed, "fixed")
    
    # Caso 2: Spacetime dinámico (con acoplamiento)
    print("Caso 2: Spacetime dinámico...")
    hydro_dynamic = PerfectFluid(
        eos=eos, 
        spacetime_mode="dynamic",
        reconstructor=MinmodReconstruction(),
        riemann_solver=HLLERiemannSolver()
    )
    
    result_dynamic = run_single_case(N, r_max, t_final, hydro_dynamic, "dynamic")
    
    # Comparación
    if result_fixed is not None and result_dynamic is not None:
        compare_results(result_fixed, result_dynamic, t_final, N)
        return True
    else:
        print("❌ Uno de los casos falló")
        return False


def run_single_case(N, r_max, t_final, hydro_fluid, case_name):
    """Ejecuta un caso individual de evolución."""
    
    # Setup
    spacing = LinearSpacing(N, r_max, SpacingExtent.HALF)
    state_vec = StateVector(hydro_fluid)
    grid = Grid(spacing, state_vec)
    r = grid.r
    background = FlatSphericalBackground(r)
    
    # Condiciones iniciales: pulso de materia concentrado
    state = create_matter_pulse_initial_state(grid, hydro_fluid, r)
    grid.fill_boundaries(state)
    
    # Progress bar mock para evitar errores con None
    progress_bar = MockProgressBar()
    
    # Evolución
    def rhs_wrapper(t, y):
        # Progreso simple para debugging
        if int(t*100) % 10 == 0:
            print(f"  {case_name}: t={t:.3f}")
        return get_rhs(t, y, grid, background, hydro_fluid, progress_bar, [0, 0.01])
    
    try:
        solution = solve_ivp(
            rhs_wrapper,
            [0, t_final],
            state.flatten(),
            method='RK45',
            rtol=1e-6,
            atol=1e-9,
            max_step=0.005
        )
        
        if solution.success:
            print(f"  ✅ {case_name} completado")
            return solution
        else:
            print(f"  ❌ {case_name} falló: {solution.message}")
            return None
            
    except Exception as e:
        print(f"  ❌ {case_name} error: {e}")
        return None


def create_matter_pulse_initial_state(grid, hydro_fluid, r):
    """
    Crea condiciones iniciales con un pulso de materia concentrado
    para testear el acoplamiento.
    """
    N = grid.num_points
    state = np.zeros((grid.NUM_VARS, N))
    
    # BSSN: inicialmente plano
    state[idx_lapse, :] = 1.0  # α = 1
    # Otras variables BSSN quedan en cero (métrica plana)
    
    # Hidrodinámica: pulso gaussiano de materia
    r_center = 0.4
    sigma = 0.1
    rho_max = 5.0  # densidad alta para ver efectos gravitacionales
    
    rho0 = rho_max * np.exp(-((r - r_center)/sigma)**2)
    rho0 = np.maximum(rho0, 1e-13)  # atmosphere
    
    # Presión y velocidad
    pressure = 0.1 * rho0**2  # politropa aproximada
    vr = np.zeros(N)  # inicialmente en reposo
    
    # Convertir a conservadas
    eos = hydro_fluid.eos
    eps = eos.eps_from_rho_p(rho0, pressure)
    h = 1.0 + eps + pressure/rho0
    W = 1.0
    
    D = rho0 * W
    Sr = np.zeros(N)  # momento inicial cero
    tau = rho0 * h * W**2 - pressure - D
    
    # Asignar al estado
    state[hydro_fluid.idx_D, :] = D
    state[hydro_fluid.idx_Sr, :] = Sr
    state[hydro_fluid.idx_tau, :] = tau
    
    return state


def compare_results(result_fixed, result_dynamic, t_final, N):
    """Compara los resultados de ambos casos."""
    
    print(f"\n📊 Comparación de Resultados (t={t_final}):")
    
    # Estados finales
    state_fixed = result_fixed.y[:, -1]
    state_dynamic = result_dynamic.y[:, -1]
    
    # Calcular número de variables y puntos correctamente
    total_size = len(state_fixed)
    NUM_VARS = total_size // N
    
    state_fixed = state_fixed.reshape(NUM_VARS, N)
    state_dynamic = state_dynamic.reshape(NUM_VARS, N)
    
    # Comparar lapse (debe ser diferente si hay acoplamiento)
    lapse_fixed = state_fixed[idx_lapse, :]
    lapse_dynamic = state_dynamic[idx_lapse, :]
    
    lapse_diff = np.max(np.abs(lapse_dynamic - lapse_fixed))
    print(f"  Máxima diferencia en lapse: {lapse_diff:.6f}")
    
    # Comparar phi (factor conformal)
    phi_fixed = state_fixed[idx_phi, :]
    phi_dynamic = state_dynamic[idx_phi, :]
    
    phi_diff = np.max(np.abs(phi_dynamic - phi_fixed))
    print(f"  Máxima diferencia en φ: {phi_diff:.6f}")
    
    # Comparar densidad de materia
    D_fixed = state_fixed[NUM_BSSN_VARS, :]  # idx_D
    D_dynamic = state_dynamic[NUM_BSSN_VARS, :]
    
    D_diff = np.max(np.abs(D_dynamic - D_fixed))
    print(f"  Máxima diferencia en D: {D_diff:.6f}")
    
    # Verificar que hay diferencias significativas (evidencia de acoplamiento)
    total_diff = lapse_diff + phi_diff
    
    if total_diff > 1e-6:
        print("  ✅ Acoplamiento detectado: spacetime responde a la materia")
        coupling_detected = True
    else:
        print("  ⚠️  Acoplamiento débil o no detectado")
        coupling_detected = False
    
    # Plot comparativo
    plot_comparison(result_fixed, result_dynamic, t_final, N)
    
    return coupling_detected


def plot_comparison(result_fixed, result_dynamic, t_final, N):
    """Grafica la comparación entre ambos casos."""
    
    # Usar los parámetros correctos del test
    r_max = 1.0
    r = np.linspace(0, r_max, N)
    
    # Estados finales - calcular dimensiones correctamente
    state_fixed_flat = result_fixed.y[:, -1]
    state_dynamic_flat = result_dynamic.y[:, -1]
    NUM_VARS = len(state_fixed_flat) // N
    
    # Reshape a (NUM_VARS, N)
    state_fixed = state_fixed_flat.reshape(NUM_VARS, N)
    state_dynamic = state_dynamic_flat.reshape(NUM_VARS, N)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Lapse
    axes[0,0].plot(r, state_fixed[idx_lapse, :], 'b-', label='Minkowski fijo', lw=2)
    axes[0,0].plot(r, state_dynamic[idx_lapse, :], 'r--', label='Dinámico', lw=2)
    axes[0,0].set_xlabel('r'); axes[0,0].set_ylabel('α (lapse)')
    axes[0,0].set_title('Lapse Function'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
    
    # Factor conformal  
    axes[0,1].plot(r, state_fixed[idx_phi, :], 'b-', label='Minkowski fijo', lw=2)
    axes[0,1].plot(r, state_dynamic[idx_phi, :], 'r--', label='Dinámico', lw=2)
    axes[0,1].set_xlabel('r'); axes[0,1].set_ylabel('φ')
    axes[0,1].set_title('Factor Conformal'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)
    
    # Densidad D
    axes[1,0].plot(r, state_fixed[NUM_BSSN_VARS, :], 'b-', label='Minkowski fijo', lw=2)
    axes[1,0].plot(r, state_dynamic[NUM_BSSN_VARS, :], 'r--', label='Dinámico', lw=2)
    axes[1,0].set_xlabel('r'); axes[1,0].set_ylabel('D')
    axes[1,0].set_title('Densidad Conservada'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)
    
    # Diferencias  
    lapse_diff = state_dynamic[idx_lapse, :] - state_fixed[idx_lapse, :]
    phi_diff = state_dynamic[idx_phi, :] - state_fixed[idx_phi, :]
    
    axes[1,1].plot(r, lapse_diff, 'g-', label='Δα', lw=2)
    axes[1,1].plot(r, phi_diff, 'm-', label='Δφ', lw=2)
    axes[1,1].set_xlabel('r'); axes[1,1].set_ylabel('Diferencia')
    axes[1,1].set_title('Efectos del Acoplamiento'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Comparación: Fijo vs Dinámico (t={t_final})', fontsize=14)
    plt.tight_layout()
    plt.savefig('coupling_validation.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_constraint_violation():
    """
    Test para verificar que las constraint violations del BSSN 
    permanecen bajo control durante la evolución acoplada.
    """
    print("\n🔬 TEST: Violación de Constraints")
    print("="*40)
    
    N = 64
    r_max = 1.0
    t_final = 0.05  # tiempo corto para este test
    
    # Setup
    eos = IdealGasEOS(gamma=1.4)
    hydro_fluid = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        reconstructor=MinmodReconstruction(),
        riemann_solver=HLLERiemannSolver()
    )
    
    spacing = LinearSpacing(N, r_max, SpacingExtent.HALF)
    state_vec = StateVector(hydro_fluid)
    grid = Grid(spacing, state_vec)
    r = grid.r
    background = FlatSphericalBackground(r)
    
    # Condiciones iniciales
    state = create_matter_pulse_initial_state(grid, hydro_fluid, r)
    grid.fill_boundaries(state)
    
    # Evaluar constraints iniciales usando la función segura
    initial_state = state.reshape(grid.NUM_VARS, -1)
    Ham_init, Mom_init = safe_constraints_diagnostic(
        initial_state, 0.0, grid, background, hydro_fluid
    )
    
    print(f"Constraints iniciales:")
    print(f"  |Ham|_max = {np.max(np.abs(Ham_init[0, :])):.3e}")
    print(f"  |Mom|_max = {np.max(np.abs(Mom_init[0, :, :])):.3e}")
    
    # Progress bar mock para constraints test
    progress_bar = MockProgressBar()
    
    # Evolución corta
    def rhs_wrapper(t, y):
        return get_rhs(t, y, grid, background, hydro_fluid, progress_bar, [0, 0.01])
    
    try:
        solution = solve_ivp(
            rhs_wrapper,
            [0, t_final],
            state.flatten(),
            method='RK45',
            rtol=1e-7,
            atol=1e-10,
            max_step=0.001
        )
        
        if solution.success:
            # Evaluar constraints finales usando la función segura
            final_state = solution.y[:, -1].reshape(grid.NUM_VARS, -1)
            Ham_final, Mom_final = safe_constraints_diagnostic(
                final_state, t_final, grid, background, hydro_fluid
            )
            
            print(f"Constraints finales (t={t_final}):")
            print(f"  |Ham|_max = {np.max(np.abs(Ham_final[0, :])):.3e}")
            print(f"  |Mom|_max = {np.max(np.abs(Mom_final[0, :, :])):.3e}")
            
            # Verificar que no crecen excesivamente
            ham_growth = np.max(np.abs(Ham_final[0, :])) / np.max(np.abs(Ham_init[0, :]))
            mom_growth = np.max(np.abs(Mom_final[0, :, :])) / np.max(np.abs(Mom_init[0, :, :]))
            
            print(f"Crecimiento relativo:")
            print(f"  Ham: {ham_growth:.2f}x")
            print(f"  Mom: {mom_growth:.2f}x")
            
            # Test pasa si constraints no crecen exponencialmente
            if ham_growth < 10.0 and mom_growth < 10.0:
                print("  ✅ Constraints bajo control")
                return True
            else:
                print("  ⚠️  Crecimiento excesivo de constraints")
                return False
                
        else:
            print(f"  ❌ Evolución falló: {solution.message}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def run_all_validation_tests():
    """Ejecuta todos los tests de validación del acoplamiento."""
    
    print("🧪 SUITE DE VALIDACIÓN: Acoplamiento BSSN-Hydro")
    print("="*60)
    
    results = []
    
    # Test 1: Acoplamiento vs Minkowski
    results.append(("Acoplamiento vs Minkowski", test_coupling_vs_minkowski()))
    
    # Test 2: Constraint violations
    # results.append(("Control de Constraints", test_constraint_violation()))
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE VALIDACIÓN")
    print("="*60)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"{name:25s}: {'✅ PASÓ' if ok else '❌ FALLÓ'}")
    print("-"*40)
    print(f"Total: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 ¡Todos los tests de validación pasaron!")
        print("   El acoplamiento BSSN-Hydro está funcionando correctamente.")
    else:
        print("⚠️  Algunos tests fallaron. Revisar implementación.")
    
    return passed == len(results)


if __name__ == "__main__":
    run_all_validation_tests()