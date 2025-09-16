#!/usr/bin/env python
"""
Profiling del acoplamiento BSSN-Hidro para identificar cuellos de botella.
"""

import os
import sys
import numpy as np
import time
import cProfile
import pstats
from io import StringIO

# Add source path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from source.core.grid import Grid
from source.core.spacing import LinearSpacing, SpacingExtent
from source.core.statevector import StateVector
from source.core.rhsevolution import get_rhs
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.bssn.bssnstatevariables import *
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.reconstruction import MinmodReconstruction
from source.matter.hydro.riemann import HLLERiemannSolver

def setup_problem():
    """Setup básico del problema."""
    N = 32
    r_max = 1.0

    spacing = LinearSpacing(N, r_max, SpacingExtent.HALF)

    hydro = PerfectFluid(
        eos=IdealGasEOS(gamma=2.0),
        spacetime_mode="dynamic",
        reconstructor=MinmodReconstruction(),
        riemann_solver=HLLERiemannSolver()
    )

    state_vec = StateVector(hydro)
    grid = Grid(spacing, state_vec)
    r = grid.r
    background = FlatSphericalBackground(r)

    # Estado inicial
    state = np.zeros((grid.NUM_VARS, N))
    state[idx_lapse, :] = 1.0

    # Pulso gaussiano
    r_center = 0.5
    sigma = 0.1
    rho0 = 1.0 + 2.0 * np.exp(-((r - r_center)/sigma)**2)
    pressure = 0.1 * rho0
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

    return grid, background, hydro, state

def profile_single_rhs():
    """Profile una sola evaluación del RHS."""
    print("🔍 PROFILING: Una evaluación del RHS")
    print("-" * 50)

    grid, background, hydro, state = setup_problem()

    class DummyProgress:
        def update(self, n): pass

    progress = DummyProgress()
    t = 0.0
    y = state.flatten()

    # Warmup
    print("⏱️  Warmup...")
    for _ in range(3):
        get_rhs(t, y, grid, background, hydro, progress, [0, 0.1])

    # Timing manual
    print("⏱️  Timing manual (10 evaluaciones)...")
    start_time = time.time()
    for i in range(10):
        rhs = get_rhs(t, y, grid, background, hydro, progress, [0, 0.1])
    end_time = time.time()

    avg_time = (end_time - start_time) / 10
    print(f"   Tiempo promedio por RHS: {avg_time:.4f} segundos")
    print(f"   Extrapolación para 1000 pasos: {avg_time * 1000 / 60:.1f} minutos")

    # Profiling detallado
    print("\n🔬 Profiling detallado...")
    pr = cProfile.Profile()
    pr.enable()

    # Múltiples evaluaciones para estadísticas
    for _ in range(5):
        get_rhs(t, y, grid, background, hydro, progress, [0, 0.1])

    pr.disable()

    # Análisis de resultados
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 funciones

    print("\n📊 TOP 20 FUNCIONES MÁS COSTOSAS:")
    print(s.getvalue())

    return avg_time

def profile_components():
    """Profile componentes individuales."""
    print("\n🧩 PROFILING: Componentes individuales")
    print("-" * 50)

    grid, background, hydro, state = setup_problem()

    class DummyProgress:
        def update(self, n): pass

    progress = DummyProgress()
    t = 0.0
    y = state.flatten()
    state_2d = y.reshape(grid.NUM_VARS, -1)

    # 1. BSSN RHS solo
    print("1️⃣ BSSN RHS (sin hidro)...")
    start = time.time()
    for _ in range(10):
        from source.bssn.bssnrhs import get_bssn_rhs
        rhs_bssn = get_bssn_rhs(state_2d, grid.derivs, background, None)
    bssn_time = (time.time() - start) / 10
    print(f"   BSSN: {bssn_time:.4f} s")

    # 2. Cons2prim
    print("2️⃣ Conversión cons2prim...")
    start = time.time()
    for _ in range(10):
        from source.bssn.bssnvars import BSSNVars
        bssn_vars = BSSNVars(state_2d, background)
        primitives = hydro._get_primitives(bssn_vars, grid.r)
    cons2prim_time = (time.time() - start) / 10
    print(f"   Cons2prim: {cons2prim_time:.4f} s")

    # 3. Stress-energy tensor
    print("3️⃣ Stress-energy tensor...")
    start = time.time()
    for _ in range(10):
        emtensor = hydro.stress_energy_tensor(state_2d, grid.r, background)
    emtensor_time = (time.time() - start) / 10
    print(f"   EM tensor: {emtensor_time:.4f} s")

    # 4. Hidro RHS
    print("4️⃣ Hidro RHS...")
    start = time.time()
    for _ in range(10):
        bssn_vars = BSSNVars(state_2d, background)
        rhs_hydro = hydro.rhs(state_2d, grid, bssn_vars, background)
    hydro_time = (time.time() - start) / 10
    print(f"   Hidro RHS: {hydro_time:.4f} s")

    total_components = bssn_time + cons2prim_time + emtensor_time + hydro_time
    print(f"\n📊 RESUMEN:")
    print(f"   BSSN:       {bssn_time:.4f} s ({bssn_time/total_components*100:.1f}%)")
    print(f"   Cons2prim:  {cons2prim_time:.4f} s ({cons2prim_time/total_components*100:.1f}%)")
    print(f"   EM tensor:  {emtensor_time:.4f} s ({emtensor_time/total_components*100:.1f}%)")
    print(f"   Hidro RHS:  {hydro_time:.4f} s ({hydro_time/total_components*100:.1f}%)")
    print(f"   TOTAL:      {total_components:.4f} s")

def check_step_size():
    """Verifica el tamaño de paso que usa el integrador."""
    print("\n⏱️  ANÁLISIS DEL PASO DE TIEMPO")
    print("-" * 50)

    grid, background, hydro, state = setup_problem()

    class StepTracker:
        def __init__(self):
            self.steps = []
            self.times = []

        def update(self, n):
            pass

    tracker = StepTracker()

    def rhs_with_tracking(t, y):
        tracker.times.append(t)
        if len(tracker.times) > 1:
            dt = tracker.times[-1] - tracker.times[-2]
            tracker.steps.append(dt)

        class DummyProgress:
            def update(self, n): pass

        return get_rhs(t, y, grid, background, hydro, DummyProgress(), [0, 0.01])

    # Correr por tiempo muy corto para ver pasos
    from scipy.integrate import solve_ivp

    print("🚀 Corriendo integración corta para analizar pasos...")
    solution = solve_ivp(
        rhs_with_tracking,
        [0, 0.001],  # Solo 0.001 unidades
        state.flatten(),
        method='RK45',
        rtol=1e-5,
        atol=1e-8,
        max_step=0.001
    )

    if tracker.steps:
        avg_step = np.mean(tracker.steps)
        min_step = np.min(tracker.steps)
        max_step = np.max(tracker.steps)

        print(f"   Pasos dados: {len(tracker.steps)}")
        print(f"   Paso promedio: {avg_step:.2e}")
        print(f"   Paso mínimo:   {min_step:.2e}")
        print(f"   Paso máximo:   {max_step:.2e}")
        print(f"   Para t=0.1 necesitaría ~{0.1/avg_step:.0f} pasos")

        # Estimar tiempo total
        grid, background, hydro, state = setup_problem()
        start = time.time()
        get_rhs(0, state.flatten(), grid, background, hydro, StepTracker(), [0, 0.1])
        rhs_time = time.time() - start

        estimated_total = rhs_time * (0.1 / avg_step)
        print(f"   Tiempo estimado total: {estimated_total/60:.1f} minutos")

if __name__ == "__main__":
    print("🚀 ANÁLISIS DE PERFORMANCE: Acoplamiento BSSN-Hidro")
    print("=" * 60)

    # 1. Profile una evaluación completa
    avg_rhs_time = profile_single_rhs()

    # 2. Profile componentes
    profile_components()

    # 3. Analizar paso de tiempo
    check_step_size()

    print("\n" + "=" * 60)
    print("✅ Análisis completo. Revisa los resultados para identificar cuellos de botella.")