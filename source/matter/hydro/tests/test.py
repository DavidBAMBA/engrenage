#!/usr/bin/env python3
"""
test_updated.py — Valencia FULL reference-metric tests using engrenage infrastructure

Updated version using:
- Grid from source.core.grid
- StateVector from source.core.statevector
- PerfectFluid from source.matter.hydro.perfect_fluid
- Proper BSSN integration (frozen for Minkowski)

Maintains the same test structure as original test.py with multiple reconstructor comparisons.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add source path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, repo_root)

# Engrenage core
from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground

# BSSN
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS, idx_phi, idx_K, idx_lapse

# Hydro
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.cons2prim import prim_to_cons
from source.matter.hydro.atmosphere import AtmosphereParams


# ============================================================================
# GRID AND INFRASTRUCTURE SETUP
# ============================================================================

def build_hydro_and_grid(n_interior=256, r_max=1.0, gamma=1.4, reconstructor="mp5",
                          spacetime_mode="fixed_minkowski"):
    """
    Build grid and hydro infrastructure using engrenage architecture.

    This is the correct way to build the infrastructure, following TOVEvolution.py pattern.

    Args:
        n_interior: Number of interior points
        r_max: Maximum radius
        gamma: Adiabatic index for EOS
        reconstructor: Reconstruction method ("minmod", "mp5", "wenoz", etc.)
        spacetime_mode: "fixed_minkowski" or "dynamic"

    Returns:
        grid: Grid object with full engrenage infrastructure
        hydro: PerfectFluid object
        background: Background geometry
    """
    # 1. Create spacing
    spacing = LinearSpacing(n_interior + 2 * NUM_GHOSTS, r_max)

    # 2. Create EOS
    eos = IdealGasEOS(gamma=gamma)

    # 3. Create atmosphere parameters
    atmosphere = AtmosphereParams(
        rho_floor=1e-13,
        p_floor=1e-15,
        v_max=0.999999,
        W_max=1e3
    )

    # 4. Create hydro object (PerfectFluid)
    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode=spacetime_mode,
        atmosphere=atmosphere,
        reconstructor=create_reconstruction(reconstructor),
        riemann_solver=HLLRiemannSolver()
    )

    # 5. Create state vector
    state_vector = StateVector(hydro)

    # 6. Create grid
    grid = Grid(spacing, state_vector)

    # 7. Create background geometry
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    return grid, hydro, background


def fill_ghosts_primitives(rho, v, p, ng=NUM_GHOSTS):
    """
    Apply boundary conditions: parity at center, outflow at outer boundary.

    This function is NOT in engrenage core, it's test-specific.
    """
    N = len(rho)

    # Left boundary (center): parity conditions
    for i in range(ng):
        mir = 2 * ng - 1 - i
        rho[i] = rho[mir]   # even
        p[i] = p[mir]       # even
        v[i] = -v[mir]      # odd

    # Right boundary (outflow): zero-gradient
    last = N - ng - 1
    for k in range(1, ng + 1):
        idx = last + k
        rho[idx] = rho[last]
        p[idx] = p[last]
        v[idx] = v[last]

    return rho, v, p


def primitives_to_conservatives(rho0, vr, p, grid, hydro):
    """Convert primitives to conservatives using engrenage infrastructure."""
    gamma_rr = np.ones_like(rho0)  # Minkowski
    D, Sr, tau = prim_to_cons(rho0, vr, p, gamma_rr, hydro.eos)
    return D, Sr, tau


def conservatives_to_primitives(state_2d, grid, hydro, bssn_vars):
    """Convert conservatives to primitives using engrenage infrastructure."""
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    return hydro._get_primitives(bssn_vars, grid.r)


def max_signal_speed(primitives, eos, cfl_guard=1e-6):
    """Compute maximum signal speed for CFL condition."""
    rho0 = primitives['rho0']
    vr = primitives['vr']
    p = primitives['p']
    eps = eos.eps_from_rho_p(rho0, p)
    h = 1.0 + eps + p / np.maximum(rho0, 1e-300)
    cs2 = np.clip(eos.gamma * p / np.maximum(rho0 * h, 1e-300), 0.0, 1.0 - 1e-10)
    cs = np.sqrt(cs2)
    return np.max(np.abs(vr) + cs) + cfl_guard


def volume_integrals(state_2d, grid, hydro):
    """Compute total mass and energy with 4π r^2 weighting."""
    ng = NUM_GHOSTS
    r_int = grid.r[ng:-ng]
    D = state_2d[hydro.idx_D, ng:-ng]
    tau = state_2d[hydro.idx_tau, ng:-ng]
    dr = grid.min_dr
    mass = 4 * np.pi * np.sum(D * r_int * r_int) * dr
    energy = 4 * np.pi * np.sum((tau + D) * r_int * r_int) * dr
    return mass, energy


# ============================================================================
# RK3 TIME INTEGRATION USING ENGRENAGE ARCHITECTURE
# ============================================================================

def get_rhs_minkowski(state_flat, grid, hydro, background, bssn_fixed, bssn_d1_fixed):
    """
    Compute RHS for matter evolution in fixed Minkowski spacetime.

    Following TOVEvolution.py pattern but for Minkowski.

    Args:
        state_flat: Flattened state vector (NUM_BSSN_VARS + 3) * N
        grid: Grid object
        hydro: PerfectFluid object
        background: Background geometry
        bssn_fixed: Fixed BSSN variables (Minkowski)
        bssn_d1_fixed: Fixed BSSN derivatives

    Returns:
        rhs_flat: Flattened RHS vector
    """
    state_2d = state_flat.reshape((grid.NUM_VARS, grid.N))
    grid.fill_boundaries(state_2d)

    # Build BSSN vars (frozen Minkowski)
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(bssn_fixed)

    # Set matter vars and compute hydro RHS
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    hydro_rhs = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1_fixed, background)

    # Full RHS (BSSN frozen = 0, only hydro evolves)
    rhs_2d = np.zeros_like(state_2d)
    rhs_2d[NUM_BSSN_VARS:, :] = hydro_rhs

    return rhs_2d.flatten()


def rk3_step(state_2d, grid, hydro, background, bssn_fixed, bssn_d1_fixed, cfl=0.3):
    """
    Single RK3 (Shu-Osher) timestep for fixed Minkowski spacetime.

    Following TOVEvolution.py pattern but using RK3 instead of RK4.

    Args:
        state_2d: State array (NUM_VARS, N) with BSSN + hydro
        grid: Grid object
        hydro: PerfectFluid object
        background: Background geometry
        bssn_fixed: Fixed BSSN variables (Minkowski)
        bssn_d1_fixed: Fixed BSSN derivatives
        cfl: CFL number

    Returns:
        state_new: Updated state array (NUM_VARS, N)
        dt: Timestep used
    """
    # Build BSSN vars for primitives computation
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(bssn_fixed)

    # Get primitives for timestep calculation
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    primitives = hydro._get_primitives(bssn_vars, grid.r)

    # Compute timestep
    amax = max_signal_speed(primitives, hydro.eos)
    dt = cfl * grid.min_dr / amax

    state_flat = state_2d.flatten()

    # RK3 Stage 1
    k1 = get_rhs_minkowski(state_flat, grid, hydro, background, bssn_fixed, bssn_d1_fixed)
    state1 = state_flat + dt * k1

    # RK3 Stage 2
    k2 = get_rhs_minkowski(state1, grid, hydro, background, bssn_fixed, bssn_d1_fixed)
    state2 = 0.75 * state_flat + 0.25 * (state1 + dt * k2)

    # RK3 Stage 3
    k3 = get_rhs_minkowski(state2, grid, hydro, background, bssn_fixed, bssn_d1_fixed)
    state_new_flat = (1.0/3.0) * state_flat + (2.0/3.0) * (state2 + dt * k3)

    state_new = state_new_flat.reshape((grid.NUM_VARS, grid.N))

    return state_new, dt


# ============================================================================
# TESTS (EXACT SAME STRUCTURE AS ORIGINAL)
# ============================================================================

def test_uniform_state():
    print("\n" + "="*60)
    print("TEST 1: Estado uniforme (Minkowski, engrenage infrastructure)")
    print("="*60)

    # Build infrastructure using engrenage architecture
    grid, hydro, background = build_hydro_and_grid(
        n_interior=100, r_max=1.0, gamma=1.4, reconstructor="minmod"
    )

    # Initial primitives (uniform state)
    rho0 = np.ones(grid.N) * 1.0
    vr = np.zeros(grid.N)
    p = np.ones(grid.N) * 0.1
    rho0, vr, p = fill_ghosts_primitives(rho0, vr, p)

    # Convert to conservatives
    D, Sr, tau = primitives_to_conservatives(rho0, vr, p, grid, hydro)

    # Create full state vector with Minkowski metric
    state_2d = np.zeros((grid.NUM_VARS, grid.N))
    state_2d[idx_lapse, :] = 1.0  # α = 1
    state_2d[idx_phi, :] = 0.0    # φ = 0
    state_2d[idx_K, :] = 0.0      # K = 0
    state_2d[hydro.idx_D, :] = D
    state_2d[hydro.idx_Sr, :] = Sr
    state_2d[hydro.idx_tau, :] = tau

    # Fixed BSSN metric and derivatives
    bssn_fixed = state_2d[:NUM_BSSN_VARS, :].copy()
    bssn_d1_fixed = grid.get_d1_metric_quantities(state_2d)

    # Evolution loop
    t, Tfinal = 0.0, 0.1
    steps = 0
    while t < Tfinal and steps < 2000:
        state_2d, dt = rk3_step(state_2d, grid, hydro, background, bssn_fixed, bssn_d1_fixed, cfl=0.5)
        t += dt
        steps += 1

    # Check variation (interior)
    ng = NUM_GHOSTS
    D_final = state_2d[hydro.idx_D, ng:-ng]
    Sr_final = state_2d[hydro.idx_Sr, ng:-ng]
    tau_final = state_2d[hydro.idx_tau, ng:-ng]

    vD = np.max(np.abs(D_final - 1.0))
    vSr = np.max(np.abs(Sr_final - 0.0))
    vtau = np.max(np.abs(tau_final - 0.15))  # tau ≈ p/(gamma-1) for uniform state
    print(f"max|ΔD|={vD:.3e}, max|Sr|={vSr:.3e}, max|Δτ|={vtau:.3e}")
    ok = (vD < 5e-8) and (vSr < 5e-8)
    print("✓ PASA" if ok else "✗ FALLA")
    return ok


def test_cons2prim_roundtrip():
    print("\n" + "="*60)
    print("TEST 2: Conversión Conservadas ↔ Primitivas (roundtrip)")
    print("="*60)

    # Build infrastructure
    grid, hydro, background = build_hydro_and_grid(
        n_interior=128, r_max=1.0, gamma=1.4, reconstructor="minmod"
    )

    # Random primitives
    rho0 = np.random.uniform(0.3e-4, 2.0e-2, grid.N)
    vr = np.random.uniform(-0.8, 0.8, grid.N)
    p = np.random.uniform(0.05e-4, 1.0e-1, grid.N)
    rho0, vr, p = fill_ghosts_primitives(rho0, vr, p)

    # Convert to conservatives
    D, Sr, tau = primitives_to_conservatives(rho0, vr, p, grid, hydro)

    # Create state vector
    state_2d = np.zeros((grid.NUM_VARS, grid.N))
    state_2d[idx_lapse, :] = 1.0
    state_2d[idx_phi, :] = 0.0
    state_2d[hydro.idx_D, :] = D
    state_2d[hydro.idx_Sr, :] = Sr
    state_2d[hydro.idx_tau, :] = tau

    # Convert back to primitives
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
    primitives = conservatives_to_primitives(state_2d, grid, hydro, bssn_vars)

    # Check roundtrip error
    e_rho = np.max(np.abs(primitives['rho0'] - rho0))
    e_v = np.max(np.abs(primitives['vr'] - vr))
    e_p = np.max(np.abs(primitives['p'] - p))
    print(f"max|Δrho|={e_rho:.2e}, max|Δv|={e_v:.2e}, max|Δp|={e_p:.2e}")
    ok = (e_rho < 1e-9) and (e_v < 1e-9) and (e_p < 1e-9)
    print("✓ PASA" if ok else "✗ FALLA")
    return ok


def test_riemann_sod():
    """Exact same test as original with multiple reconstructors."""
    print("\n" + "="*60)
    print("TEST 3: Sod radial - Comparación reconstructores (Minkowski, engrenage infrastructure)")
    print("="*60)

    # Lista de métodos de reconstrucción a probar
    methods = ["mp5"]  # Can add: "wenoz", "mp5_hires"
    colors = ["blue"]
    labels = ["MP5"]
    linestyles = ["-"]

    # Guardar resultados para cada método
    results = {}

    for i, method in enumerate(methods):
        print(f"\nEjecutando con {labels[i]}...")

        # Para MP5 alta resolución, usar más puntos
        if method == "mp5_hires":
            grid, hydro, background = build_hydro_and_grid(
                n_interior=2000, r_max=1.0, gamma=1.4, reconstructor="mp5"
            )
        else:
            grid, hydro, background = build_hydro_and_grid(
                n_interior=500, r_max=1.0, gamma=1.4, reconstructor=method
            )

        # Discontinuidad en el punto medio del dominio interior
        r_mid = 0.5 * (grid.r[NUM_GHOSTS] + grid.r[-NUM_GHOSTS-1])
        rho0 = np.where(grid.r < r_mid, 10.0, 1.0)
        p = np.where(grid.r < r_mid, 40000.0/3.0, 1.0e-6)
        vr = np.zeros(grid.N)

        rho0, vr, p = fill_ghosts_primitives(rho0, vr, p)

        # Conservadas iniciales
        D, Sr, tau = primitives_to_conservatives(rho0, vr, p, grid, hydro)

        # Create full state vector with Minkowski metric
        state_2d = np.zeros((grid.NUM_VARS, grid.N))
        state_2d[idx_lapse, :] = 1.0
        state_2d[idx_phi, :] = 0.0
        state_2d[idx_K, :] = 0.0
        state_2d[hydro.idx_D, :] = D
        state_2d[hydro.idx_Sr, :] = Sr
        state_2d[hydro.idx_tau, :] = tau

        # Fixed BSSN metric and derivatives
        bssn_fixed = state_2d[:NUM_BSSN_VARS, :].copy()
        bssn_d1_fixed = grid.get_d1_metric_quantities(state_2d)

        # Evolución temporal
        t, Tfinal = 0.0, 0.35
        steps = 0
        while t < Tfinal and steps < 5000:
            state_2d, dt = rk3_step(state_2d, grid, hydro, background, bssn_fixed, bssn_d1_fixed, cfl=0.3)
            t += dt
            steps += 1

        # Extract primitives from final state
        bssn_vars = BSSNVars(grid.N)
        bssn_vars.set_bssn_vars(bssn_fixed)
        primitives = conservatives_to_primitives(state_2d, grid, hydro, bssn_vars)

        # Guardar resultados
        ng = NUM_GHOSTS
        rin = grid.r[ng:-ng]
        results[method] = {
            'r': rin.copy(),
            'rho': primitives['rho0'][ng:-ng].copy(),
            'p': primitives['p'][ng:-ng].copy(),
            'v': primitives['vr'][ng:-ng].copy(),
            't': t,
            'steps': steps
        }

        # Métricas de calidad
        rho_in = primitives['rho0'][ng:-ng]
        grad = np.gradient(rho_in, rin)
        variation = np.std(rho_in)/np.mean(rho_in)
        contact = np.any(np.abs(grad) > 0.5)
        print(f"  {method}: t≈{t:.3f}, pasos={steps}, variación ρ={variation:.3f}, contacto={contact}")

    # PLOTEO (same as original)
    # Use grid from last method for initial conditions plot
    r_ref = grid.r
    r_mid_ref = 0.5 * (r_ref[NUM_GHOSTS] + r_ref[-NUM_GHOSTS-1])
    rho0_init = np.where(r_ref < r_mid_ref, 10.0, 1.0)
    p_init = np.where(r_ref < r_mid_ref, 40000.0/3.0, 1.0e-6)
    v_init = np.zeros(grid.N)
    rho0_init, v_init, p_init = fill_ghosts_primitives(rho0_init, v_init, p_init)

    ng = NUM_GHOSTS
    rho0_init_in = rho0_init[ng:-ng]
    p_init_in = p_init[ng:-ng]
    v_init_in = v_init[ng:-ng]
    rin_ref = r_ref[ng:-ng]

    avg_time = np.mean([results[m]['t'] for m in methods])

    # Figura 1: Densidad
    fig1, (ax1, ax1_zoom) = plt.subplots(1, 2, figsize=(14, 6))

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

    ax1_zoom.plot(rin_ref, rho0_init_in, 'k:', label='ρ inicial', alpha=0.7, linewidth=2)
    for i, method in enumerate(methods):
        ax1_zoom.plot(results[method]['r'], results[method]['rho'],
                     color=colors[i], label=labels[i], linewidth=2, linestyle=linestyles[i])
    ax1_zoom.set_xlabel('r')
    ax1_zoom.set_ylabel('Densidad ρ₀')
    ax1_zoom.set_xlim(0.6, 0.7)
    # Auto-compute ylim from data in zoom region
    rho_zoom_data = []
    for method in methods:
        mask_zoom = (results[method]['r'] >= 0.6) & (results[method]['r'] <= 0.7)
        if np.any(mask_zoom):
            rho_zoom_data.extend(results[method]['rho'][mask_zoom])
    if rho_zoom_data:
        rho_min, rho_max = np.min(rho_zoom_data), np.max(rho_zoom_data)
        # Add 10% margin
        rho_range = rho_max - rho_min
        rho_min_plot = rho_min - 0.1 * rho_range
        rho_max_plot = rho_max + 0.1 * rho_range
        ax1_zoom.set_ylim(rho_min_plot, rho_max_plot)
    ax1_zoom.grid(True, alpha=0.3)
    ax1_zoom.legend()
    ax1_zoom.set_title('Densidad - Zoom Shock')

    fig1.suptitle(f"Sod: Densidad (t ≈ {avg_time:.3f})")
    plt.tight_layout()
    plt.savefig("test2sod_density_comparison.png", dpi=150, bbox_inches="tight")

    # Figura 2: Presión
    fig2, (ax2, ax2_zoom) = plt.subplots(1, 2, figsize=(14, 6))

    ax2.plot(rin_ref, p_init_in, 'k:', label='p inicial', alpha=0.7, linewidth=2)
    for i, method in enumerate(methods):
        ax2.plot(results[method]['r'], results[method]['p'],
                color=colors[i], label=labels[i], linewidth=2, linestyle=linestyles[i])
    ax2.set_xlabel('r')
    ax2.set_ylabel('Presión p')
    ax2.set_xlim(0, 1)
    #ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('Presión - Dominio Completo')

    ax2_zoom.plot(rin_ref, p_init_in, 'k:', label='p inicial', alpha=0.7, linewidth=2)
    for i, method in enumerate(methods):
        ax2_zoom.plot(results[method]['r'], results[method]['p'],
                     color=colors[i], label=labels[i], linewidth=2, linestyle=linestyles[i])
    ax2_zoom.set_xlabel('r')
    ax2_zoom.set_ylabel('Presión p')
    ax2_zoom.set_xlim(0.6, 0.7)
    # Auto-compute ylim from data in zoom region
    p_zoom_data = []
    for method in methods:
        mask_zoom = (results[method]['r'] >= 0.6) & (results[method]['r'] <= 0.7)
        if np.any(mask_zoom):
            p_zoom_data.extend(results[method]['p'][mask_zoom])
    if p_zoom_data:
        p_min, p_max = np.min(p_zoom_data), np.max(p_zoom_data)
        # Add 20% margin in log space
        p_range = np.log10(p_max / p_min)
        p_min_plot = p_min / (10 ** (0.2 * p_range))
        p_max_plot = p_max * (10 ** (0.2 * p_range))
        ax2_zoom.set_ylim(p_min_plot, p_max_plot)
    ax2_zoom.set_yscale('log')
    ax2_zoom.grid(True, alpha=0.3)
    ax2_zoom.legend()
    ax2_zoom.set_title('Presión - Zoom Shock')

    fig2.suptitle(f"Sod: Presión (t ≈ {avg_time:.3f})")
    plt.tight_layout()
    plt.savefig("test2sod_pressure_comparison.png", dpi=150, bbox_inches="tight")

    # Figura 3: Velocidad
    fig3, (ax3, ax3_zoom) = plt.subplots(1, 2, figsize=(14, 6))

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

    ax3_zoom.plot(rin_ref, v_init_in, 'k:', label='v inicial', alpha=0.7, linewidth=2)
    for i, method in enumerate(methods):
        ax3_zoom.plot(results[method]['r'], results[method]['v'],
                     color=colors[i], label=labels[i], linewidth=2, linestyle=linestyles[i])
    ax3_zoom.set_xlabel('r')
    ax3_zoom.set_ylabel('Velocidad v^r')
    ax3_zoom.set_xlim(0.5, 0.7)
    # Auto-compute ylim from data in zoom region
    v_zoom_data = []
    for method in methods:
        mask_zoom = (results[method]['r'] >= 0.5) & (results[method]['r'] <= 0.7)
        if np.any(mask_zoom):
            v_zoom_data.extend(results[method]['v'][mask_zoom])
    if v_zoom_data:
        v_min, v_max = np.min(v_zoom_data), np.max(v_zoom_data)
        # Add 10% margin
        v_range = v_max - v_min
        v_min_plot = v_min - 0.1 * v_range
        v_max_plot = v_max + 0.1 * v_range
        ax3_zoom.set_ylim(v_min_plot, v_max_plot)
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


def test_blast_wave_compare(case='weak'):
    """Exact same test as original with multiple reconstructors."""
    print("\n" + "="*60)
    print(f"TEST Blast radial - Comparación reconstructores ({case}, Minkowski, engrenage infrastructure)")
    print("="*60)

    # Blast parameters
    if case.lower() == 'weak':
        p_in, p_out = 1.0, 0.1
        rho_in, rho_out = 1.0, 0.125
    elif case.lower() == 'strong':
        p_in, p_out = 133.33, 0.125
        rho_in, rho_out = 10.0, 1.0
    else:
        raise ValueError("case debe ser 'weak' o 'strong'")

    # Métodos a comparar
    methods = ["minmod", "mp5"]
    colors = ["tab:red", "tab:blue"]
    labels = ["MINMOD", "MP5"]
    linestyles = ["-", "-"]

    # Resultados
    results = {}

    for i, method in enumerate(methods):
        print(f"\nEjecutando {labels[i]}...")

        # Build infrastructure
        if method == "mp5_hires":
            grid, hydro, background = build_hydro_and_grid(
                n_interior=200, r_max=1.0, gamma=1.4, reconstructor="mp5"
            )
        else:
            grid, hydro, background = build_hydro_and_grid(
                n_interior=100, r_max=1.0, gamma=1.4, reconstructor=method
            )

        # Parámetros del blast
        ng = NUM_GHOSTS
        r_mid = 0.5 * (grid.r[ng] + grid.r[-ng-1])
        rho0 = np.where(grid.r < r_mid, rho_in, rho_out).astype(float)
        p = np.where(grid.r < r_mid, p_in, p_out).astype(float)
        vr = np.zeros(grid.N, dtype=float)

        # Ghosts
        rho0, vr, p = fill_ghosts_primitives(rho0, vr, p)

        # Conservadas iniciales
        D, Sr, tau = primitives_to_conservatives(rho0, vr, p, grid, hydro)

        # Create full state vector with Minkowski metric
        state_2d = np.zeros((grid.NUM_VARS, grid.N))
        state_2d[idx_lapse, :] = 1.0
        state_2d[idx_phi, :] = 0.0
        state_2d[idx_K, :] = 0.0
        state_2d[hydro.idx_D, :] = D
        state_2d[hydro.idx_Sr, :] = Sr
        state_2d[hydro.idx_tau, :] = tau

        # Fixed BSSN metric and derivatives
        bssn_fixed = state_2d[:NUM_BSSN_VARS, :].copy()
        bssn_d1_fixed = grid.get_d1_metric_quantities(state_2d)

        # Evolution
        t, Tfinal = 0.0, 0.4
        steps = 0
        while t < Tfinal and steps < 10000:
            state_2d, dt = rk3_step(state_2d, grid, hydro, background, bssn_fixed, bssn_d1_fixed, cfl=0.3)
            t += dt
            steps += 1

        # Extract primitives from final state
        bssn_vars = BSSNVars(grid.N)
        bssn_vars.set_bssn_vars(bssn_fixed)
        primitives = conservatives_to_primitives(state_2d, grid, hydro, bssn_vars)

        # Guardar
        rin = grid.r[ng:-ng]
        results[method] = {
            'r': rin.copy(),
            'rho': primitives['rho0'][ng:-ng].copy(),
            'p': primitives['p'][ng:-ng].copy(),
            'v': primitives['vr'][ng:-ng].copy(),
            'E': (state_2d[hydro.idx_tau, ng:-ng] + state_2d[hydro.idx_D, ng:-ng]).copy(),
            't': t,
            'steps': steps,
        }
        print(f"  {method}: t≈{t:.3f}, pasos={steps}")

    # PLOTEO (same as original)
    avg_time = np.mean([results[m]['t'] for m in methods])
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    lw_main = 2.5

    # Densidad
    ax = axes[0, 0]
    for i, method in enumerate(methods):
        ax.plot(results[method]['r'], results[method]['rho'], color=colors[i], label=labels[i], linewidth=lw_main, linestyle=linestyles[i])
    ax.set_xlabel('r')
    ax.set_ylabel('Densidad ρ₀')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('ρ')

    # Presión
    ax = axes[0, 1]
    for i, method in enumerate(methods):
        ax.plot(results[method]['r'], results[method]['p'], color=colors[i], label=labels[i], linewidth=lw_main, linestyle=linestyles[i])
    ax.set_xlabel('r')
    ax.set_ylabel('Presión p')
    ax.set_xlim(0, 1)
    ax.set_yscale('linear')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('p')

    # Velocidad
    ax = axes[1, 0]
    for i, method in enumerate(methods):
        ax.plot(results[method]['r'], results[method]['v'], color=colors[i], label=labels[i], linewidth=lw_main, linestyle=linestyles[i])
    ax.set_xlabel('r')
    ax.set_ylabel('Velocidad v^r')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('v')

    # Energía (densidad): tau + D
    ax = axes[1, 1]
    for i, method in enumerate(methods):
        ax.plot(results[method]['r'], results[method]['E'], color=colors[i], label=labels[i], linewidth=lw_main, linestyle=linestyles[i])
    ax.set_xlabel('r')
    ax.set_ylabel('Energía dens. (τ + D)')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('E')

    fig.suptitle(f"Blast {case}: Comparación MINMOD vs MP5 (t ≈ {avg_time:.3f})")
    plt.tight_layout()
    plt.savefig(f"blast_{case}_comparison.png", dpi=150, bbox_inches="tight")

    # Criterio básico de OK
    ok_methods = []
    for method in methods:
        rho_in = results[method]['rho']
        r_in = results[method]['r']
        variation = float(np.std(rho_in)/max(np.mean(rho_in), 1e-12))
        grad = np.gradient(rho_in, r_in)
        contact = bool(np.any(np.abs(grad) > 0.5))
        ok_methods.append((variation > 0.1) and contact)
    all_ok = all(ok_methods)
    print("✓ PASA" if all_ok else "✗ FALLA")
    return all_ok


def test_conservation_short():
    print("\n" + "="*60)
    print("TEST 4: Conservación global (masa/energía)")
    print("="*60)

    # Build infrastructure
    grid, hydro, background = build_hydro_and_grid(
        n_interior=100, r_max=1.0, gamma=4.0/3.0, reconstructor="minmod"
    )

    # Initial primitives (smooth perturbation)
    rho0 = 1.0 + 0.5 * np.exp(-((grid.r - 0.6)**2) / 0.01)
    p = 0.1 * rho0
    vr = 0.02 * np.sin(4 * np.pi * (grid.r - grid.r[NUM_GHOSTS]))
    rho0, vr, p = fill_ghosts_primitives(rho0, vr, p)

    # Convert to conservatives
    D, Sr, tau = primitives_to_conservatives(rho0, vr, p, grid, hydro)

    # Create full state vector with Minkowski metric
    state_2d = np.zeros((grid.NUM_VARS, grid.N))
    state_2d[idx_lapse, :] = 1.0
    state_2d[idx_phi, :] = 0.0
    state_2d[idx_K, :] = 0.0
    state_2d[hydro.idx_D, :] = D
    state_2d[hydro.idx_Sr, :] = Sr
    state_2d[hydro.idx_tau, :] = tau

    # Fixed BSSN metric and derivatives
    bssn_fixed = state_2d[:NUM_BSSN_VARS, :].copy()
    bssn_d1_fixed = grid.get_d1_metric_quantities(state_2d)

    # Initial mass and energy
    m0, e0 = volume_integrals(state_2d, grid, hydro)

    # Evolution loop
    steps = 0
    while steps < 200:
        state_2d, dt = rk3_step(state_2d, grid, hydro, background, bssn_fixed, bssn_d1_fixed, cfl=0.25)
        steps += 1

    # Final mass and energy
    m1, e1 = volume_integrals(state_2d, grid, hydro)
    dm = abs(m1 - m0) / max(m0, 1e-15)
    de = abs(e1 - e0) / max(e0, 1e-15)
    print(f"ΔM/M={dm:.3e}, ΔE/E={de:.3e}")
    ok = (dm < 5e-3) and (de < 5e-3)
    print("✓ PASA" if ok else "✗ FALLA")
    return ok


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("="*60)
    print("SUITE — Valencia FULL Reference-Metric (engrenage infrastructure)")
    print("="*60)

    results = []
    results.append(("Uniforme", test_uniform_state()))
    results.append(("cons2prim", test_cons2prim_roundtrip()))
    results.append(("Conservación", test_conservation_short()))
    results.append(("Sod radial", test_riemann_sod()))
    results.append(("Blast weak", test_blast_wave_compare(case="weak")))
    results.append(("Blast strong", test_blast_wave_compare(case="strong")))

    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"{name:14s}: {'✓ PASÓ' if ok else '✗ FALLÓ'}")
    print("-"*40)
    print(f"Total: {passed}/{len(results)}")
