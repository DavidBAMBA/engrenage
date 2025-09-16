#!/usr/bin/env python
"""
Test rápido de sanidad: Verifica que el acoplamiento BSSN-Hidro funciona.
Evoluciona un pulso de materia y verifica que:
1. El código corre sin errores
2. El spacetime responde a la materia
3. No hay NaNs o infinitos
4. Los constraints no explotan
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Add source path (como en test.py)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Imports del framework
from source.core.grid import Grid
from source.core.spacing import LinearSpacing, SpacingExtent
from source.core.statevector import StateVector
from source.core.rhsevolution import get_rhs
from source.backgrounds.sphericalbackground import FlatSphericalBackground, i_r
from source.bssn.bssnstatevariables import *
from source.bssn.tensoralgebra import SPACEDIM

# Definir índices si no están importados
try:
    from source.core.grid import i_x1, i_x2, i_x3
except:
    i_x1, i_x2, i_x3 = 0, 1, 2

# Tu módulo de hidrodinámica
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.reconstruction import MinmodReconstruction
from source.matter.hydro.riemann import HLLERiemannSolver


def quick_sanity_test():
    """
    Test mínimo: evoluciona por tiempo corto y verifica funcionamiento básico.
    """
    print("\n" + "="*60)
    print("🚀 TEST RÁPIDO DE SANIDAD: Acoplamiento BSSN-Hidro")
    print("="*60)
    
    # --- Configuración para video ---
    N = 32  # Resolución para visualización
    r_max = 1.0
    t_final = 0.1  # Tiempo para evolución
    dt_save = 0.0001  # Intervalo de guardado para video
    
    print(f"📋 Config: N={N}, r_max={r_max}, t_final={t_final}")
    
    # --- Setup ---
    try:
        print("1️⃣ Inicializando grid y fluido...", end="")
        
        spacing = LinearSpacing(N, r_max, SpacingExtent.HALF)
        
        # Fluido con acoplamiento dinámico
        hydro = PerfectFluid(
            eos=IdealGasEOS(gamma=2.0),
            spacetime_mode="dynamic",  # ← CLAVE: acoplamiento activado
            reconstructor=MinmodReconstruction(),
            riemann_solver=HLLERiemannSolver()
        )
        
        state_vec = StateVector(hydro)
        grid = Grid(spacing, state_vec)
        r = grid.r
        background = FlatSphericalBackground(r)
        
        print(" ✅")
        
    except Exception as e:
        print(f" ❌\n   Error en setup: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # --- Condiciones iniciales ---
    try:
        print("2️⃣ Creando condiciones iniciales...", end="")
        
        state = np.zeros((grid.NUM_VARS, N))
        
        # BSSN: métrica plana
        state[idx_lapse, :] = 1.0
        
        # Hidro: pulso gaussiano DÉBIL (evitar colapso)
        r_center = 0.5
        sigma = 0.2  # Más ancho
        rho0 = 1.0 + 0.1 * np.exp(-((r - r_center)/sigma)**2)  # Mucho más débil

        # Convertir a variables conservadas
        pressure = 0.3 * rho0  # Mayor presión para evitar colapso
        vr = np.zeros(N)
        
        eps = hydro.eos.eps_from_rho_p(rho0, pressure)
        h = 1.0 + eps + pressure/rho0
        W = 1.0
        
        D = rho0 * W
        Sr = np.zeros(N)
        tau = rho0 * h * W**2 - pressure - D
        
        state[hydro.idx_D, :] = D
        state[hydro.idx_Sr, :] = Sr
        state[hydro.idx_tau, :] = tau
        
        grid.fill_boundaries(state)
        
        # Guardar valores iniciales para comparación
        lapse_initial = state[idx_lapse, N//2].copy()
        phi_initial = state[idx_phi, N//2].copy()
        
        print(" ✅")
        
    except Exception as e:
        print(f" ❌\n   Error en CI: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # --- Evolución con guardado temporal ---
    try:
        print("3️⃣ Evolucionando sistema y guardando datos...")

        # Storage para datos temporales
        import csv
        import os

        # Crear directorio para datos
        data_dir = "hydro_evolution_data"
        os.makedirs(data_dir, exist_ok=True)

        # Arrays para almacenar evolución
        time_points = []
        saved_states = []

        class DataSaver:
            def __init__(self):
                self.last_save_t = -dt_save
                self.save_count = 0

            def update(self, n):
                pass

            def maybe_save(self, t, y):
                if t - self.last_save_t >= dt_save or abs(t - t_final) < 1e-10:
                    # Reshape y guarda estado
                    state_2d = y.reshape(grid.NUM_VARS, -1)

                    # Guardar datos
                    time_points.append(t)
                    saved_states.append(state_2d.copy())

                    # Guardar CSV individual
                    filename = f"{data_dir}/state_{self.save_count:04d}.csv"
                    with open(filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        # Header
                        writer.writerow(['r', 'lapse', 'phi', 'D', 'Sr', 'tau'])
                        # Data
                        for i in range(N):
                            writer.writerow([
                                grid.r[i],
                                state_2d[idx_lapse, i],
                                state_2d[idx_phi, i],
                                state_2d[hydro.idx_D, i],
                                state_2d[hydro.idx_Sr, i],
                                state_2d[hydro.idx_tau, i]
                            ])

                    print(f"   t={t:.4f} (guardado #{self.save_count})")
                    self.last_save_t = t
                    self.save_count += 1

        saver = DataSaver()

        def rhs(t, y):
            saver.maybe_save(t, y)
            return get_rhs(t, y, grid, background, hydro, saver, [0, t_final])

        # Guardar estado inicial
        saver.maybe_save(0.0, state.flatten())

        solution = solve_ivp(
            rhs,
            [0, t_final],
            state.flatten(),
            method='RK45',
            rtol=1e-5,
            atol=1e-8,
            max_step=0.01  # Pasos más grandes para eficiencia
        )

        # Guardar estado final si no se guardó
        if solution.success:
            saver.maybe_save(t_final, solution.y[:, -1])

        # Guardar metadata
        with open(f"{data_dir}/metadata.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['parameter', 'value'])
            writer.writerow(['N', N])
            writer.writerow(['r_max', r_max])
            writer.writerow(['t_final', t_final])
            writer.writerow(['dt_save', dt_save])
            writer.writerow(['num_saved_states', saver.save_count])
            writer.writerow(['success', solution.success])

        print(f"\n   💾 Guardados {saver.save_count} estados en '{data_dir}/')")
        
        if not solution.success:
            print(f" ❌\n   Integración falló: {solution.message}")
            return False
            
        print(" ✅")
        
    except Exception as e:
        print(f" ❌\n   Error en evolución: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # --- Verificaciones ---
    print("\n📊 VERIFICACIONES:")
    all_good = True

    # Usar el último estado guardado
    if saved_states:
        final_state_2d = saved_states[-1]

        # 1. Check NaNs/Infs
        if np.any(np.isnan(final_state_2d)) or np.any(np.isinf(final_state_2d)):
            print("   ❌ Encontrados NaN o Inf en estado final")
            all_good = False
        else:
            print("   ✅ Sin NaN/Inf")

        # 2. Verificar respuesta gravitacional
        lapse_final = final_state_2d[idx_lapse, N//2]
        phi_final = final_state_2d[idx_phi, N//2]

        lapse_change = abs(lapse_final - lapse_initial)
        phi_change = abs(phi_final - phi_initial)

        print(f"   Δα = {lapse_change:.2e}, Δφ = {phi_change:.2e}")

        if lapse_change > 1e-8 or phi_change > 1e-8:
            print("   ✅ Spacetime responde a la materia")
        else:
            print("   ⚠️  Respuesta gravitacional muy débil")
            all_good = False

        # 3. Verificar conservación de masa
        D_initial = state[hydro.idx_D, :].sum()
        D_final = final_state_2d[hydro.idx_D, :].sum()
        mass_change = abs(D_final - D_initial) / D_initial

        print(f"   ΔM/M = {mass_change:.2e}")
        if mass_change < 0.01:
            print("   ✅ Masa aproximadamente conservada")
        else:
            print("   ⚠️  Cambio significativo en masa")
            all_good = False
    else:
        print("   ❌ No se guardaron datos para verificación")
        all_good = False
    
    # 4. Plot rápido (opcional)
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Densidad
        ax1.plot(r, state[hydro.idx_D, :], 'b-', label='Inicial', alpha=0.5)
        ax1.plot(r, final_state_2d[hydro.idx_D, :], 'r-', label='Final')
        ax1.set_xlabel('r'); ax1.set_ylabel('D')
        ax1.set_title('Densidad'); ax1.legend(); ax1.grid(True, alpha=0.3)
        
        # Lapse
        ax2.plot(r, state[idx_lapse, :], 'b-', label='Inicial', alpha=0.5)
        ax2.plot(r, final_state_2d[idx_lapse, :], 'r-', label='Final')
        ax2.set_xlabel('r'); ax2.set_ylabel('α')
        ax2.set_title('Lapse'); ax2.legend(); ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Test Rápido (t={t_final})')
        plt.tight_layout()
        plt.savefig('sanity_test.png', dpi=100)
        print("\n   📈 Gráfico guardado: sanity_test.png")
        
    except Exception as e:
        print(f"   ⚠️  No se pudo generar gráfico: {e}")
    
    # --- Resultado final ---
    print("\n" + "="*60)
    if all_good:
        print("✅ TEST PASADO: El acoplamiento funciona correctamente")
        print("   - El código corre sin errores")
        print("   - El spacetime responde a la materia")
        print("   - No hay problemas numéricos críticos")
        return True
    else:
        print("⚠️  TEST CON ADVERTENCIAS: Revisar los puntos marcados")
        return False


def minimal_smoke_test():
    """
    Test aún más mínimo: solo verifica que el código corre.
    """
    print("\n🔥 SMOKE TEST (ultra-rápido)")
    print("-"*40)
    
    try:
        # Setup mínimo
        N = 16
        spacing = LinearSpacing(N, 1.0, SpacingExtent.HALF)
        
        hydro = PerfectFluid(
            eos=IdealGasEOS(gamma=2.0),
            spacetime_mode="dynamic",
            reconstructor=MinmodReconstruction(),
            riemann_solver=HLLERiemannSolver()
        )
        
        state_vec = StateVector(hydro)
        grid = Grid(spacing, state_vec)
        
        # Estado trivial
        state = np.zeros((grid.NUM_VARS, N))
        state[idx_lapse, :] = 1.0
        state[hydro.idx_D, :] = 1.0
        
        # Un solo paso de RHS
        class DummyProgress:
            def update(self, n): pass
        
        rhs = get_rhs(0, state.flatten(), grid, 
                     FlatSphericalBackground(grid.r),
                     hydro, DummyProgress(), [0, 0.1])
        
        if np.any(np.isnan(rhs)):
            print("❌ RHS contiene NaN")
            return False
        
        print("✅ El código corre y calcula RHS sin errores")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 VALIDACIÓN RÁPIDA DEL MÓDULO HYDRO")
    print("="*60)
    
    # Primero el smoke test
    if not minimal_smoke_test():
        print("\n❌ Smoke test falló. Hay problemas básicos de integración.")
        exit(1)
    
    # Luego el test de sanidad
    if quick_sanity_test():
        print("\n🎉 ¡TODO BIEN! Tu módulo está correctamente integrado.")
        exit(0)
    else:
        print("\n⚠️  Hay algunos problemas. Revisa los mensajes anteriores.")
        exit(1)