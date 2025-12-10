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
from source.bssn.bssnstatevariables import NUM_BSSN_VARS, idx_phi, idx_hrr, idx_htt, idx_hpp, idx_K, idx_arr, idx_att, idx_app, idx_lapse

# Hydro
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.cons2prim import cons_to_prim, prim_to_cons
from source.matter.hydro.atmosphere import AtmosphereParams


# ============================================================================
# GRID AND INFRASTRUCTURE SETUP
# ============================================================================

class SimpleGrid:
    """Simple grid wrapper to match original test.py interface."""
    def __init__(self, dr):
        self.dr = float(dr)


def build_grid(n_interior=512, r_min=1e-3, r_max=1.0, ng=NUM_GHOSTS):
    """
    Build grid using engrenage infrastructure.
    Returns (r_array, simple_grid_obj, n_interior) to match original interface.
    """
    # Create spacing and grid
    spacing = LinearSpacing(n_interior + 2 * ng, r_max)

    # We need a minimal hydro object to create StateVector
    # Use a temporary one just for grid creation
    eos_temp = IdealGasEOS(gamma=1.4)
    hydro_temp = PerfectFluid(
        eos=eos_temp,
        spacetime_mode="fixed_minkowski",
        atmosphere=AtmosphereParams(),
        reconstructor=create_reconstruction("minmod"),
        riemann_solver=HLLRiemannSolver()
    )

    state_vector = StateVector(hydro_temp)
    grid = Grid(spacing, state_vector)

    # Return r array and simple grid object
    dr_avg = (grid.r[-1] - grid.r[0]) / (grid.N - 1)
    return grid.r, SimpleGrid(dr_avg), n_interior


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


def to_conserved(rho0, v, p, eos):
    """Convert primitives to conservatives using prim_to_cons."""
    D, Sr, tau = prim_to_cons(rho0, v, p, np.ones_like(rho0), eos)
    return D, Sr, tau


def to_primitives(D, Sr, tau, eos, p_guess=None):
    """Convert conservatives to primitives using cons_to_prim."""
    # cons_to_prim expects:
    #   U: tuple (D, Sr, tau)
    #   metric: tuple (alpha, beta_r, gamma_rr)
    # Returns: (rho0, vr, p, eps, W, h, success)
    rho0, vr, p, eps, W, h, success = cons_to_prim(
        (D, Sr, tau),
        eos,
        metric=(np.ones_like(D), np.zeros_like(D), np.ones_like(D)),  # (alpha, beta_r, gamma_rr)
        p_guess=p_guess,
    )
    return rho0, vr, p


def max_signal_speed(rho0, v, p, eos, cfl_guard=1e-6):
    """Compute maximum signal speed for CFL condition."""
    eps = eos.eps_from_rho_p(rho0, p)
    h = 1.0 + eps + p / np.maximum(rho0, 1e-300)
    cs2 = np.clip(eos.gamma * p / np.maximum(rho0 * h, 1e-300), 0.0, 1.0 - 1e-10)
    cs = np.sqrt(cs2)
    return np.max(np.abs(v) + cs) + cfl_guard


def volume_integrals(D, tau, r, grid):
    """Compute total mass and energy with 4π r^2 weighting."""
    ng = NUM_GHOSTS
    rin = r[ng:-ng]
    Din = D[ng:-ng]
    taun = tau[ng:-ng]
    mass = 4 * np.pi * np.sum(Din * rin * rin) * grid.dr
    energ = 4 * np.pi * np.sum((taun + Din) * rin * rin) * grid.dr
    return mass, energ


# ============================================================================
# RK3 TIME INTEGRATION USING PERFECT FLUID
# ============================================================================

def rk3_step(valencia_or_hydro, D, Sr, tau, rho0, v, p, r, grid, eos, recon, rsolve, cfl=0.2,
             spacetime_mode="fixed_minkowski"):
    """
    RK3 Shu-Osher step using PerfectFluid infrastructure.

    This function signature matches the original test.py but uses engrenage internally.

    Args:
        valencia_or_hydro: Can be ValenciaReferenceMetric (ignored) or PerfectFluid instance
        D, Sr, tau: Conservative variables
        rho0, v, p: Primitive variables
        r: Radial coordinate array
        grid: Grid object (with dr attribute)
        eos: Equation of state
        recon: Reconstructor instance
        rsolve: Riemann solver instance
        cfl: CFL number
        spacetime_mode: "fixed_minkowski" or "dynamic"

    Returns:
        dt, D_new, Sr_new, tau_new, rho0_new, v_new, p_new
    """
    N = len(r)

    # Create PerfectFluid instance if needed
    if not isinstance(valencia_or_hydro, PerfectFluid):
        hydro = PerfectFluid(
            eos=eos,
            spacetime_mode=spacetime_mode,
            atmosphere=AtmosphereParams(
                rho_floor=1e-13,
                p_floor=1e-15,
                v_max=0.999999,
                W_max=1e3
            ),
            reconstructor=recon,
            riemann_solver=rsolve
        )
    else:
        hydro = valencia_or_hydro

    # Compute timestep
    amax = max_signal_speed(rho0, v, p, eos)
    dt = cfl * grid.dr / amax

    # Create full state vector
    state_2d = np.zeros((NUM_BSSN_VARS + 3, N))

    # Set flat Minkowski metric (frozen)
    state_2d[idx_lapse, :] = 1.0
    state_2d[idx_phi, :] = 0.0
    state_2d[idx_hrr, :] = 0.0
    state_2d[idx_htt, :] = 0.0
    state_2d[idx_hpp, :] = 0.0
    state_2d[idx_K, :] = 0.0
    state_2d[idx_arr, :] = 0.0
    state_2d[idx_att, :] = 0.0
    state_2d[idx_app, :] = 0.0

    # Set hydro variables
    state_2d[hydro.idx_D, :] = D
    state_2d[hydro.idx_Sr, :] = Sr
    state_2d[hydro.idx_tau, :] = tau

    # Create BSSN vars and background
    bssn_vars = BSSNVars(N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
    background = FlatSphericalBackground(r)
    hydro.background = background

    # Frozen BSSN for Minkowski
    bssn_fixed = state_2d[:NUM_BSSN_VARS, :].copy()

    # Create temporary grid for d1 computation
    spacing_temp = LinearSpacing(N, r[-1])
    state_vector_temp = StateVector(hydro)
    grid_temp = Grid(spacing_temp, state_vector_temp)
    bssn_d1_fixed = grid_temp.get_d1_metric_quantities(state_2d)

    def compute_rhs(state_flat):
        """Compute RHS for matter evolution."""
        state = state_flat.reshape((NUM_BSSN_VARS + 3, N))

        # Update matter vars
        hydro.set_matter_vars(state, bssn_vars, grid_temp)

        # Compute RHS (only matter evolves, BSSN frozen)
        rhs_D, rhs_Sr, rhs_tau = hydro.get_matter_rhs(r, bssn_vars, bssn_d1_fixed, background)

        # Build full RHS (BSSN = 0)
        rhs = np.zeros_like(state)
        rhs[hydro.idx_D, :] = rhs_D
        rhs[hydro.idx_Sr, :] = rhs_Sr
        rhs[hydro.idx_tau, :] = rhs_tau

        return rhs.flatten()

    state_flat = state_2d.flatten()

    # RK3 Stage 1
    k1 = compute_rhs(state_flat)
    state1 = state_flat + dt * k1

    # RK3 Stage 2
    k2 = compute_rhs(state1)
    state2 = 0.75 * state_flat + 0.25 * (state1 + dt * k2)

    # RK3 Stage 3
    k3 = compute_rhs(state2)
    state_new = (1.0/3.0) * state_flat + (2.0/3.0) * (state2 + dt * k3)

    # Reshape and extract
    state_2d_new = state_new.reshape((NUM_BSSN_VARS + 3, N))

    D_new = state_2d_new[hydro.idx_D, :]
    Sr_new = state_2d_new[hydro.idx_Sr, :]
    tau_new = state_2d_new[hydro.idx_tau, :]

    # Convert to primitives
    rho0_new, v_new, p_new = to_primitives(D_new, Sr_new, tau_new, eos, p_guess=p)
    rho0_new, v_new, p_new = fill_ghosts_primitives(rho0_new, v_new, p_new)

    return dt, D_new, Sr_new, tau_new, rho0_new, v_new, p_new


# ============================================================================
# TESTS (EXACT SAME STRUCTURE AS ORIGINAL)
# ============================================================================

def test_uniform_state():
    print("\n" + "="*60)
    print("TEST 1: Estado uniforme (Minkowski, engrenage infrastructure)")
    print("="*60)

    r, grid, Nin = build_grid(n_interior=300, r_min=1e-3, r_max=1.0)
    N = len(r)
    eos = IdealGasEOS(gamma=1.4)
    recon = create_reconstruction("minmod")
    rsolve = HLLRiemannSolver()
    vr = np.zeros(N)
    rho0 = np.ones(N) * 1.0
    p = np.ones(N) * 0.1
    rho0, vr, p = fill_ghosts_primitives(rho0, vr, p)
    D, Sr, tau = to_conserved(rho0, vr, p, eos)

    # Use None as placeholder (PerfectFluid will be created inside rk3_step)
    val = None
    t, Tfinal = 0.0, 0.1
    steps = 0
    while t < Tfinal and steps < 2000:
        dt, D, Sr, tau, rho0, vr, p = rk3_step(val, D, Sr, tau, rho0, vr, p, r, grid, eos, recon, rsolve, cfl=0.5)
        t += dt
        steps += 1

    # Variación relativa (interior)
    ng = NUM_GHOSTS
    vD = np.max(np.abs(D[ng:-ng] - 1.0))
    vSr = np.max(np.abs(Sr[ng:-ng] - 0.0))
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
    v = np.random.uniform(-0.8, 0.8, N)
    p = np.random.uniform(0.05e4, 1.0e-1, N)
    rho0, v, p = fill_ghosts_primitives(rho0, v, p)
    D, Sr, tau = to_conserved(rho0, v, p, eos)
    r2, v2, p2 = to_primitives(D, Sr, tau, eos)
    e_rho = np.max(np.abs(r2 - rho0))
    e_v = np.max(np.abs(v2 - v))
    e_p = np.max(np.abs(p2 - p))
    print(f"max|Δrho|={e_rho:.2e}, max|Δv|={e_v:.2e}, max|Δp|={e_p:.2e}")
    ok = (e_rho < 1e-9) and (e_v < 1e-9) and (e_p < 1e-9)
    print("✓ PASA" if ok else "✗ FALLA")
    return ok


def test_riemann_sod():
    """Exact same test as original with multiple reconstructors."""
    print("\n" + "="*60)
    print("TEST 3: Sod radial - Comparación reconstructores (Minkowski, engrenage infrastructure)")
    print("="*60)

    # Configuración base
    r, grid, Nin = build_grid(n_interior=1000, r_min=1e-3, r_max=1.0)
    N = len(r)
    gamma =1.4
    eos = IdealGasEOS(gamma=gamma)
    rsolve = HLLRiemannSolver()

    # Discontinuidad en el punto medio del dominio interior
    r_mid = 0.5*(r[NUM_GHOSTS] + r[-NUM_GHOSTS-1])
    rho0_base = np.where(r < r_mid, 10.0, 1.0)
    p_base = np.where(r < r_mid, 40.0/3.0, 1.0e-6)
    v_base = np.zeros(N)

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
            r_hires, grid_hires, Nin_hires = build_grid(n_interior=2000, r_min=1e-3, r_max=1.0)
            N_hires = len(r_hires)

            r_mid_hires = 0.5*(r_hires[NUM_GHOSTS] + r_hires[-NUM_GHOSTS-1])
            rho0 = np.where(r_hires < r_mid_hires, 10.0, 1.0)
            p = np.where(r_hires < r_mid_hires, 40.0/3.0, 1.0e-6)
            v = np.zeros(N_hires)

            rho0, v, p = fill_ghosts_primitives(rho0, v, p)

            r_current = r_hires
            grid_current = grid_hires
            actual_method = "mp5"
        else:
            rho0 = rho0_base.copy()
            p = p_base.copy()
            v = v_base.copy()

            rho0, v, p = fill_ghosts_primitives(rho0, v, p)

            r_current = r
            grid_current = grid
            actual_method = method

        # Conservadas iniciales
        D, Sr, tau = to_conserved(rho0, v, p, eos)

        # Crear reconstructor específico
        recon = create_reconstruction(actual_method)
        val = None

        # Evolución temporal
        t, Tfinal = 0.0, 0.35
        steps = 0
        while t < Tfinal and steps < 5000:
            dt, D, Sr, tau, rho0, v, p = rk3_step(
                val, D, Sr, tau, rho0, v, p, r_current, grid_current, eos, recon, rsolve, cfl=0.3
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
        grad = np.gradient(rho_in, rin)
        variation = np.std(rho_in)/np.mean(rho_in)
        contact = np.any(np.abs(grad) > 0.5)
        print(f"  {method}: t≈{t:.3f}, pasos={steps}, variación ρ={variation:.3f}, contacto={contact}")

    # PLOTEO (same as original)
    rho0_init = np.where(r < r_mid, 10.0, 1.0)
    p_init = np.where(r < r_mid, 40.0/3.0, 1.0e-6)
    v_init = np.zeros(N)
    rho0_init, v_init, p_init = fill_ghosts_primitives(rho0_init, v_init, p_init)

    ng = NUM_GHOSTS
    rho0_init_in = rho0_init[ng:-ng]
    p_init_in = p_init[ng:-ng]
    v_init_in = v_init[ng:-ng]
    rin_ref = r[ng:-ng]

    avg_time = np.mean([results[m]['t'] for m in methods])

    # Figura 1: Densidad
    fig1, (ax1, ax1_zoom) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(rin_ref, rho0_init_in, 'k:', label='ρ inicial', alpha=0.7, linewidth=2)
    for i, method in enumerate(methods):
        ax1.plot(results[method]['r'], results[method]['rho'],
                color=colors[i], label=labels[i], linewidth=2, linestyle=linestyles[i])
    ax1.set_xlabel('r')
    ax1.set_ylabel(' ρ₀')
    ax1.set_xlim(0, 1)
    ax1.legend()

    ax1_zoom.plot(rin_ref, rho0_init_in, 'k:', label='ρ inicial', alpha=0.7, linewidth=2)
    for i, method in enumerate(methods):
        ax1_zoom.plot(results[method]['r'], results[method]['rho'],
                     color=colors[i], label=labels[i], linewidth=2, linestyle=linestyles[i])
    ax1_zoom.set_xlabel('r')
    ax1_zoom.set_ylabel(' ρ₀')
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
    ax1_zoom.legend()

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
    ax2.set_ylabel(' p')
    ax2.set_xlim(0, 1)
    #ax2.set_yscale('log')
    ax2.legend()

    ax2_zoom.plot(rin_ref, p_init_in, 'k:', label='p inicial', alpha=0.7, linewidth=2)
    for i, method in enumerate(methods):
        ax2_zoom.plot(results[method]['r'], results[method]['p'],
                     color=colors[i], label=labels[i], linewidth=2, linestyle=linestyles[i])
    ax2_zoom.set_xlabel('r')
    ax2_zoom.set_ylabel(' p')
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
    ax2_zoom.legend()

    plt.tight_layout()
    plt.savefig("test2sod_pressure_comparison.png", dpi=150, bbox_inches="tight")

    # Figura 3: Velocidad
    fig3, (ax3, ax3_zoom) = plt.subplots(1, 2, figsize=(14, 6))

    ax3.plot(rin_ref, v_init_in, 'k:', label='v inicial', alpha=0.7, linewidth=2)
    for i, method in enumerate(methods):
        ax3.plot(results[method]['r'], results[method]['v'],
                color=colors[i], label=labels[i], linewidth=2, linestyle=linestyles[i])
    ax3.set_xlabel('r')
    ax3.set_ylabel(' v^r')
    ax3.set_xlim(0, 1)
    ax3.legend()

    ax3_zoom.plot(rin_ref, v_init_in, 'k:', label='v inicial', alpha=0.7, linewidth=2)
    for i, method in enumerate(methods):
        ax3_zoom.plot(results[method]['r'], results[method]['v'],
                     color=colors[i], label=labels[i], linewidth=2, linestyle=linestyles[i])
    ax3_zoom.set_xlabel('r')
    ax3_zoom.set_ylabel(' v^r')
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
    ax3_zoom.legend()

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

    # Configuración base
    r, grid, Nin = build_grid(n_interior=1000, r_min=1e-3, r_max=1.0)
    N = len(r)
    eos = IdealGasEOS(gamma=1.4)
    rsolve = HLLRiemannSolver()

    # Parámetros del blast
    ng = NUM_GHOSTS
    r_mid = 0.5*(r[ng] + r[-ng-1])
    if case.lower() == 'weak':
        p_in, p_out = 1.0, 0.1
        rho_in, rho_out = 1.0, 0.125
    elif case.lower() == 'strong':
        p_in, p_out = 133.33, 0.125
        rho_in, rho_out = 10.0, 1.0
    else:
        raise ValueError("case debe ser 'weak' o 'strong'")

    rho0_base = np.where(r < r_mid, rho_in, rho_out).astype(float)
    p_base = np.where(r < r_mid, p_in, p_out).astype(float)
    v_base = np.zeros(N, dtype=float)

    # Métodos a comparar
    methods = ["mp5"]
    colors = ["red"]
    labels = ["MP5"]
    linestyles = ["-"]

    # Resultados
    results = {}

    for i, method in enumerate(methods):
        print(f"\nEjecutando {labels[i]}...")

        # Opción de alta resolución (desactivada por defecto)
        if method == "mp5_hires":
            r_hires, grid_hires, Nin_hires = build_grid(n_interior=200, r_min=1e-3, r_max=1.0)
            r_current = r_hires
            grid_current = grid_hires
            rho0 = np.where(r_hires < r_mid, rho_in, rho_out).astype(float)
            p = np.where(r_hires < r_mid, p_in, p_out).astype(float)
            v = np.zeros_like(r_hires)
            actual_method = "mp5"
        else:
            r_current = r
            grid_current = grid
            rho0 = rho0_base.copy()
            p = p_base.copy()
            v = v_base.copy()
            actual_method = method

        # Ghosts
        rho0, v, p = fill_ghosts_primitives(rho0, v, p)

        # Conservadas iniciales
        D, Sr, tau = to_conserved(rho0, v, p, eos)

        # Recon y evolución
        recon = create_reconstruction(actual_method)
        val = None

        t, Tfinal = 0.0, 0.4
        steps = 0
        while t < Tfinal and steps < 10000:
            dt, D, Sr, tau, rho0, v, p = rk3_step(
                val, D, Sr, tau, rho0, v, p, r_current, grid_current, eos, recon, rsolve, cfl=0.3
            )
            t += dt
            steps += 1

        # Compute additional quantities
        rho_int = rho0[ng:-ng]
        p_int = p[ng:-ng]
        v_int = v[ng:-ng]

        # Lorentz factor: W = 1/sqrt(1 - v²)
        v_squared = v_int ** 2
        W_int = 1.0 / np.sqrt(np.maximum(1.0 - v_squared, 1e-16))

        # Specific internal energy: epsilon
        eps_int = eos.eps_from_rho_p(rho_int, p_int)

        # Speed of sound (ideal gas): c_s² = Γ P / (ρ h)
        h_int = 1.0 + eps_int + p_int / np.maximum(rho_int, 1e-30)
        cs2_int = eos.gamma * p_int / np.maximum(rho_int * h_int, 1e-30)
        cs2_int = np.clip(cs2_int, 0.0, 1.0 - 1e-10)  # Keep 0 < c_s² < 1
        cs_int = np.sqrt(cs2_int)

        # Relativistic Mach number: M = (W v) / (W_s c_s)
        # where W_s = 1/sqrt(1 - c_s²) is the sound Lorentz factor
        W_s_int = 1.0 / np.sqrt(np.maximum(1.0 - cs2_int, 1e-16))
        Mach_rel = (W_int * np.abs(v_int)) / np.maximum(W_s_int * cs_int, 1e-30)

        # Guardar
        rin = r_current[ng:-ng]
        results[method] = {
            'r': rin.copy(),
            'rho': rho_int.copy(),
            'p': p_int.copy(),
            'v': v_int.copy(),
            'W': W_int.copy(),
            'eps': eps_int.copy(),
            'Mach': Mach_rel.copy(),
            't': t,
            'steps': steps,
        }
        print(f"  {method}: t≈{t:.3f}, pasos={steps}")

    # PLOTEO: 3x2 layout con primitivas
    avg_time = np.mean([results[m]['t'] for m in methods])
    fig, axes = plt.subplots(3, 2, figsize=(10, 14))
    lw_main = 2.5

    # Fila 1, Col 1: Densidad ρ₀
    ax = axes[0, 0]
    for i, method in enumerate(methods):
        ax.plot(results[method]['r'], results[method]['rho'], color=colors[i], label=labels[i], linewidth=lw_main, linestyle=linestyles[i])
    ax.set_xlabel('r', fontsize=11)
    ax.set_ylabel('Densidad ρ₀', fontsize=11)
    ax.set_xlim(0, 1)
    ax.legend()

    # Fila 1, Col 2: Presión p
    ax = axes[0, 1]
    for i, method in enumerate(methods):
        ax.plot(results[method]['r'], results[method]['p'], color=colors[i], label=labels[i], linewidth=lw_main, linestyle=linestyles[i])
    ax.set_xlabel('r', fontsize=11)
    ax.set_ylabel('Presión P', fontsize=11)
    ax.set_xlim(0, 1)
    ax.legend()

    # Fila 2, Col 1: Velocidad v^r
    ax = axes[1, 0]
    for i, method in enumerate(methods):
        ax.plot(results[method]['r'], results[method]['v'], color=colors[i], label=labels[i], linewidth=lw_main, linestyle=linestyles[i])
    ax.set_xlabel('r', fontsize=11)
    ax.set_ylabel(r' $v^r$', fontsize=11)
    ax.set_xlim(0, 1)
    ax.legend()

    # Fila 2, Col 2: Factor de Lorentz W
    ax = axes[1, 1]
    for i, method in enumerate(methods):
        ax.plot(results[method]['r'], results[method]['W'], color=colors[i], label=labels[i], linewidth=lw_main, linestyle=linestyles[i])
    ax.set_xlabel('r', fontsize=11)
    ax.set_ylabel(r' $W$', fontsize=11)
    ax.set_xlim(0, 1)
    ax.legend()

    # Fila 3, Col 1: Energía específica interna ε
    ax = axes[2, 0]
    for i, method in enumerate(methods):
        ax.plot(results[method]['r'], results[method]['eps'], color=colors[i], label=labels[i], linewidth=lw_main, linestyle=linestyles[i])
    ax.set_xlabel('r', fontsize=11)
    ax.set_ylabel(r' $\varepsilon$', fontsize=11)
    ax.set_xlim(0, 1)
    ax.legend()

    # Fila 3, Col 2: Número de Mach relativista
    ax = axes[2, 1]
    for i, method in enumerate(methods):
        ax.plot(results[method]['r'], results[method]['Mach'], color=colors[i], label=labels[i], linewidth=lw_main, linestyle=linestyles[i])
    # Add horizontal line at Mach = 1 (sonic line)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Sónico (M=1)')
    ax.set_xlabel('r', fontsize=11)
    ax.set_ylabel(r'Mach $\mathcal{M}$', fontsize=11)
    ax.set_xlim(0, 1)
    ax.legend()

    fig.suptitle(f"Blast {case}: Comparación MINMOD vs MP5 (t ≈ {avg_time:.3f})", fontsize=14, fontweight='bold')
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
    r, grid, Nin = build_grid(n_interior=300, r_min=1e-3, r_max=1.0)
    N = len(r)
    eos = IdealGasEOS(gamma=4.0/3.0)
    recon = create_reconstruction("minmod")
    rsolve = HLLRiemannSolver()

    rho0 = 1.0 + 0.5*np.exp(-((r-0.6)**2)/0.01)
    p = 0.1*rho0
    v = 0.02*np.sin(4*np.pi*(r - r[NUM_GHOSTS]))
    rho0, v, p = fill_ghosts_primitives(rho0, v, p)
    D, Sr, tau = to_conserved(rho0, v, p, eos)

    val = None
    m0, e0 = volume_integrals(D, tau, r, grid)
    steps = 0
    while steps < 200:
        dt, D, Sr, tau, rho0, v, p = rk3_step(val, D, Sr, tau, rho0, v, p, r, grid, eos, recon, rsolve, cfl=0.25)
        steps += 1
    m1, e1 = volume_integrals(D, tau, r, grid)
    dm = abs(m1-m0)/max(m0, 1e-15)
    de = abs(e1-e0)/max(e0, 1e-15)
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
