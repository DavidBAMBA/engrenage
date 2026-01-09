#!/usr/bin/env python3
"""
test.py — Valencia FULL reference-metric tests using engrenage infrastructure

Updated version using the same patterns as TOVEvolution.py and BHEvolution.ipynb:
- Grid from source.core.grid
- StateVector from source.core.statevector
- PerfectFluid from source.matter.hydro.perfect_fluid
- Proper BSSN initialization for Minkowski spacetime
- grid.fill_boundaries() for boundary conditions

This runs with fixed_minkowski spacetime mode, so BSSN variables are frozen
and only hydro variables evolve.
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
from source.bssn.bssnstatevariables import (
    NUM_BSSN_VARS,
    idx_phi, idx_hrr, idx_htt, idx_hpp,
    idx_K, idx_arr, idx_att, idx_app,
    idx_lambdar, idx_shiftr, idx_br, idx_lapse
)

# Hydro
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.cons2prim import Cons2PrimSolver, prim_to_cons
from source.matter.hydro.atmosphere import AtmosphereParams


# ============================================================================
# GRID AND STATE INITIALIZATION (following TOVEvolution.py pattern)
# ============================================================================

def create_hydro_and_grid(n_interior=512, r_max=1.0, gamma=1.4,
                          reconstructor_name="minmod",
                          spacetime_mode="fixed_minkowski"):
    """
    Create PerfectFluid, StateVector, and Grid following engrenage patterns.

    This is the proper way to set up the simulation, as done in TOVEvolution.py.

    Args:
        n_interior: Number of interior points (total will be n_interior + 2*NUM_GHOSTS)
        r_max: Maximum radius
        gamma: Adiabatic index for EOS
        reconstructor_name: "minmod", "mp5", "wenoz", etc.
        spacetime_mode: "fixed_minkowski" or "dynamic"

    Returns:
        grid: Grid object with all methods (fill_boundaries, derivatives, etc.)
        hydro: PerfectFluid object
        background: FlatSphericalBackground
        eos: Equation of state
    """
    # Total points including ghosts
    num_points = n_interior + 2 * NUM_GHOSTS

    # Create spacing and EOS
    spacing = LinearSpacing(num_points, r_max)
    eos = IdealGasEOS(gamma=gamma)

    # Atmosphere parameters (centralized floor management)
    atmosphere = AtmosphereParams(
        rho_floor=1e-13,
        p_floor=1e-15,
        v_max=0.999999,
        W_max=1e3,
        conservative_floor_safety=0.999
    )

    # Create reconstructor and Riemann solver
    reconstructor = create_reconstruction(reconstructor_name)
    riemann_solver = HLLRiemannSolver(atmosphere=atmosphere)

    # Create PerfectFluid (hydro object)
    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode=spacetime_mode,
        atmosphere=atmosphere,
        reconstructor=reconstructor,
        riemann_solver=riemann_solver
    )

    # Create StateVector and Grid (following TOVEvolution.py pattern)
    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)

    # Create background
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    return grid, hydro, background, eos


def initialize_minkowski_bssn(N):
    """
    Initialize BSSN variables for flat Minkowski spacetime.

    For Minkowski in spherical coordinates:
    - phi = 0 (conformal factor e^{4phi} = 1)
    - h_ij = 0 (metric deviation from reference)
    - K = 0 (trace of extrinsic curvature)
    - A_ij = 0 (traceless part)
    - lambda^r = 0 (contracted Christoffel)
    - shift = 0 (beta^r = 0)
    - B^r = 0 (time derivative of shift)
    - lapse = 1 (alpha = 1)

    Args:
        N: Number of grid points

    Returns:
        bssn_state: (NUM_BSSN_VARS, N) array with BSSN variables
    """
    bssn_state = np.zeros((NUM_BSSN_VARS, N))

    # Only lapse is non-zero (alpha = 1)
    bssn_state[idx_lapse, :] = 1.0

    # All other BSSN vars are already zero:
    # phi = 0, h_ij = 0, K = 0, A_ij = 0, lambda^r = 0, shift^r = 0, B^r = 0

    return bssn_state


def create_initial_state(grid, rho0, v, p, eos):
    """
    Create full state vector (BSSN + hydro) from primitive variables.

    Args:
        grid: Grid object
        rho0: Rest mass density array
        v: Velocity array (v^r)
        p: Pressure array
        eos: Equation of state

    Returns:
        state_2d: (NUM_VARS, N) state array ready for evolution
    """
    N = grid.N

    # Initialize full state vector
    state_2d = np.zeros((grid.NUM_VARS, N))

    # Set BSSN variables for Minkowski
    state_2d[:NUM_BSSN_VARS, :] = initialize_minkowski_bssn(N)

    # Convert primitives to conservatives
    # For Minkowski: gamma_rr = 1, e6phi = 1, alpha = 1
    gamma_rr = np.ones(N)
    D, Sr, tau = prim_to_cons(rho0, v, p, gamma_rr, eos)

    # Set hydro variables (indices from PerfectFluid)
    idx_D = NUM_BSSN_VARS
    idx_Sr = NUM_BSSN_VARS + 1
    idx_tau = NUM_BSSN_VARS + 2

    state_2d[idx_D, :] = D
    state_2d[idx_Sr, :] = Sr
    state_2d[idx_tau, :] = tau

    # Apply boundary conditions using engrenage's fill_boundaries
    grid.fill_boundaries(state_2d)

    return state_2d


def get_rhs_minkowski(state_2d, grid, background, hydro, bssn_fixed, bssn_d1_fixed):
    """
    Compute RHS for fixed Minkowski evolution (frozen spacetime).

    This follows the same pattern as get_rhs_cowling in TOVEvolution.py.

    Args:
        state_2d: (NUM_VARS, N) current state
        grid: Grid object
        background: FlatSphericalBackground
        hydro: PerfectFluid object
        bssn_fixed: Fixed BSSN state
        bssn_d1_fixed: Fixed BSSN first derivatives

    Returns:
        rhs: (NUM_VARS, N) time derivatives
    """
    # Apply boundary conditions
    grid.fill_boundaries(state_2d)

    # Create BSSNVars from fixed state
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(bssn_fixed)

    # Set matter variables in hydro object
    hydro.set_matter_vars(state_2d, bssn_vars, grid)

    # Compute hydro RHS
    hydro_rhs = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1_fixed, background)

    # Build full RHS (BSSN frozen, only hydro evolves)
    rhs = np.zeros_like(state_2d)
    rhs[NUM_BSSN_VARS:, :] = hydro_rhs

    return rhs


# ============================================================================
# RK3 TIME INTEGRATION (using engrenage infrastructure)
# ============================================================================

def rk3_step(state_2d, dt, grid, background, hydro, bssn_fixed, bssn_d1_fixed):
    """
    Single RK3 Shu-Osher step using engrenage infrastructure.

    This is cleaner than the original - uses grid.fill_boundaries() and
    doesn't recreate Grid objects each step.

    Args:
        state_2d: (NUM_VARS, N) current state
        dt: Timestep
        grid: Grid object (reused, not recreated)
        background: FlatSphericalBackground
        hydro: PerfectFluid object
        bssn_fixed: Fixed BSSN state for Minkowski
        bssn_d1_fixed: Fixed BSSN derivatives

    Returns:
        state_new: (NUM_VARS, N) updated state
    """
    # RK3 Stage 1
    k1 = get_rhs_minkowski(state_2d, grid, background, hydro, bssn_fixed, bssn_d1_fixed)
    state1 = state_2d + dt * k1
    grid.fill_boundaries(state1)

    # RK3 Stage 2
    k2 = get_rhs_minkowski(state1, grid, background, hydro, bssn_fixed, bssn_d1_fixed)
    state2 = 0.75 * state_2d + 0.25 * (state1 + dt * k2)
    grid.fill_boundaries(state2)

    # RK3 Stage 3
    k3 = get_rhs_minkowski(state2, grid, background, hydro, bssn_fixed, bssn_d1_fixed)
    state_new = (1.0/3.0) * state_2d + (2.0/3.0) * (state2 + dt * k3)
    grid.fill_boundaries(state_new)

    return state_new


def compute_dt(rho0, v, p, eos, grid, cfl=0.3):
    """Compute timestep from CFL condition."""
    eps = eos.eps_from_rho_p(rho0, p)
    h = 1.0 + eps + p / np.maximum(rho0, 1e-300)
    cs2 = np.clip(eos.gamma * p / np.maximum(rho0 * h, 1e-300), 0.0, 1.0 - 1e-10)
    cs = np.sqrt(cs2)
    amax = np.max(np.abs(v) + cs) + 1e-10
    return cfl * grid.min_dr / amax


def extract_primitives(state_2d, grid, hydro, bssn_vars):
    """Extract primitive variables from state using cons2prim."""
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    rho0, v, p, eps, W, h, success = hydro._get_primitives(bssn_vars, grid.r)
    return rho0, v, p


def volume_integrals(D, tau, r, dr):
    """Compute total mass and energy with 4 pi r^2 weighting."""
    ng = NUM_GHOSTS
    rin = r[ng:-ng]
    Din = D[ng:-ng]
    taun = tau[ng:-ng]
    mass = 4 * np.pi * np.sum(Din * rin * rin) * dr
    energ = 4 * np.pi * np.sum((taun + Din) * rin * rin) * dr
    return mass, energ


# ============================================================================
# TESTS
# ============================================================================

def test_uniform_state():
    """Test uniform state remains uniform (should stay constant in Minkowski)."""
    print("\n" + "="*60)
    print("TEST 1: Uniform state (Minkowski, engrenage infrastructure)")
    print("="*60)

    # Create grid and hydro using proper engrenage setup
    grid, hydro, background, eos = create_hydro_and_grid(
        n_interior=300, r_max=1.0, gamma=1.4,
        reconstructor_name="minmod",
        spacetime_mode="fixed_minkowski"
    )

    # Uniform initial conditions
    rho0 = np.ones(grid.N) * 1.0
    v = np.zeros(grid.N)
    p = np.ones(grid.N) * 0.1

    # Create initial state with proper BSSN initialization
    state_2d = create_initial_state(grid, rho0, v, p, eos)

    # Store initial values for comparison
    D_init = state_2d[NUM_BSSN_VARS, :].copy()
    Sr_init = state_2d[NUM_BSSN_VARS + 1, :].copy()

    # Fixed BSSN for Minkowski evolution
    bssn_fixed = state_2d[:NUM_BSSN_VARS, :].copy()
    bssn_d1_fixed = grid.get_d1_metric_quantities(state_2d)

    # Time evolution
    t, Tfinal = 0.0, 0.1
    steps = 0
    while t < Tfinal and steps < 2000:
        dt = compute_dt(rho0, v, p, eos, grid, cfl=0.5)
        state_2d = rk3_step(state_2d, dt, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

        # Update primitives for next timestep
        bssn_vars = BSSNVars(grid.N)
        bssn_vars.set_bssn_vars(bssn_fixed)
        rho0, v, p = extract_primitives(state_2d, grid, hydro, bssn_vars)

        t += dt
        steps += 1

    # Check variation (interior only)
    ng = NUM_GHOSTS
    D_final = state_2d[NUM_BSSN_VARS, :]
    Sr_final = state_2d[NUM_BSSN_VARS + 1, :]

    vD = np.max(np.abs(D_final[ng:-ng] - D_init[ng:-ng]))
    vSr = np.max(np.abs(Sr_final[ng:-ng] - Sr_init[ng:-ng]))

    print(f"Steps: {steps}, t_final: {t:.4f}")
    print(f"max|Delta D|={vD:.3e}, max|Delta Sr|={vSr:.3e}")

    ok = (vD < 5e-8) and (vSr < 5e-8)
    print("PASS" if ok else "FAIL")
    return ok


def test_cons2prim_roundtrip():
    """Test conservative <-> primitive conversion roundtrip."""
    print("\n" + "="*60)
    print("TEST 2: Conservative <-> Primitive roundtrip")
    print("="*60)

    N = 128 + 2 * NUM_GHOSTS
    eos = IdealGasEOS(gamma=1.4)

    # Atmosphere for cons2prim
    atmosphere = AtmosphereParams(
        rho_floor=1e-13,
        p_floor=1e-15,
        v_max=0.999999,
        W_max=1e3
    )

    # Random initial primitives
    np.random.seed(42)
    rho0 = np.random.uniform(0.01, 2.0, N)
    v = np.random.uniform(-0.8, 0.8, N)
    p = np.random.uniform(0.01, 1.0, N)

    # For Minkowski: gamma_rr = 1
    gamma_rr = np.ones(N)

    # Prim -> Cons
    D, Sr, tau = prim_to_cons(rho0, v, p, gamma_rr, eos)

    # Cons -> Prim
    solver = Cons2PrimSolver(eos, atmosphere=atmosphere)
    result = solver.convert(D, Sr, tau, gamma_rr, p_guess=p)
    rho0_rec, v_rec, p_rec = result[0], result[1], result[2]

    e_rho = np.max(np.abs(rho0_rec - rho0))
    e_v = np.max(np.abs(v_rec - v))
    e_p = np.max(np.abs(p_rec - p))

    print(f"max|Delta rho|={e_rho:.2e}, max|Delta v|={e_v:.2e}, max|Delta p|={e_p:.2e}")

    ok = (e_rho < 1e-9) and (e_v < 1e-9) and (e_p < 1e-9)
    print("PASS" if ok else "FAIL")
    return ok


def test_conservation_short():
    """Test global mass and energy conservation."""
    print("\n" + "="*60)
    print("TEST 3: Global conservation (mass/energy)")
    print("="*60)

    # Create grid and hydro
    grid, hydro, background, eos = create_hydro_and_grid(
        n_interior=300, r_max=1.0, gamma=4.0/3.0,
        reconstructor_name="minmod",
        spacetime_mode="fixed_minkowski"
    )

    # Gaussian density perturbation
    r = grid.r
    rho0 = 1.0 + 0.5 * np.exp(-((r - 0.6)**2) / 0.01)
    p = 0.1 * rho0
    v = 0.02 * np.sin(4 * np.pi * (r - r[NUM_GHOSTS]))

    # Create initial state
    state_2d = create_initial_state(grid, rho0, v, p, eos)

    # Fixed BSSN
    bssn_fixed = state_2d[:NUM_BSSN_VARS, :].copy()
    bssn_d1_fixed = grid.get_d1_metric_quantities(state_2d)

    # Initial integrals
    D = state_2d[NUM_BSSN_VARS, :]
    tau = state_2d[NUM_BSSN_VARS + 2, :]
    m0, e0 = volume_integrals(D, tau, r, grid.min_dr)

    # Time evolution
    steps = 0
    while steps < 200:
        dt = compute_dt(rho0, v, p, eos, grid, cfl=0.25)
        state_2d = rk3_step(state_2d, dt, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

        # Update primitives
        bssn_vars = BSSNVars(grid.N)
        bssn_vars.set_bssn_vars(bssn_fixed)
        rho0, v, p = extract_primitives(state_2d, grid, hydro, bssn_vars)

        steps += 1

    # Final integrals
    D = state_2d[NUM_BSSN_VARS, :]
    tau = state_2d[NUM_BSSN_VARS + 2, :]
    m1, e1 = volume_integrals(D, tau, r, grid.min_dr)

    dm = abs(m1 - m0) / max(m0, 1e-15)
    de = abs(e1 - e0) / max(e0, 1e-15)

    print(f"Delta M/M = {dm:.3e}, Delta E/E = {de:.3e}")

    ok = (dm < 2e-2) and (de < 2e-2)  # 2% tolerance for outflow BC
    print("PASS" if ok else "FAIL")
    return ok


def test_riemann_sod():
    """Sod shock tube test with MP5 reconstruction."""
    print("\n" + "="*60)
    print("TEST 4: Sod shock tube (Minkowski, engrenage infrastructure)")
    print("="*60)

    # Create grid and hydro with MP5
    grid, hydro, background, eos = create_hydro_and_grid(
        n_interior=100, r_max=1.0, gamma=1.4,
        reconstructor_name="mp5",
        spacetime_mode="fixed_minkowski"
    )

    r = grid.r

    # Sod initial conditions
    r_mid = 0.5 * (r[NUM_GHOSTS] + r[-NUM_GHOSTS-1])
    rho0 = np.where(r < r_mid, 10.0, 1.0).astype(float)
    p = np.where(r < r_mid, 4000.0/3.0, 1.0e-6).astype(float)
    v = np.zeros(grid.N)

    # Create initial state
    state_2d = create_initial_state(grid, rho0, v, p, eos)

    # Fixed BSSN
    bssn_fixed = state_2d[:NUM_BSSN_VARS, :].copy()
    bssn_d1_fixed = grid.get_d1_metric_quantities(state_2d)

    # Store initial for plotting
    rho_init = rho0.copy()

    # Time evolution
    t, Tfinal = 0.0, 0.35
    steps = 0
    while t < Tfinal and steps < 5000:
        dt = compute_dt(rho0, v, p, eos, grid, cfl=0.3)
        state_2d = rk3_step(state_2d, dt, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

        # Update primitives
        bssn_vars = BSSNVars(grid.N)
        bssn_vars.set_bssn_vars(bssn_fixed)
        rho0, v, p = extract_primitives(state_2d, grid, hydro, bssn_vars)

        t += dt
        steps += 1

    print(f"Sod test: t={t:.3f}, steps={steps}")

    # Plot results
    ng = NUM_GHOSTS
    rin = r[ng:-ng]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Density
    axes[0].plot(rin, rho_init[ng:-ng], 'k:', label='Initial', linewidth=2)
    axes[0].plot(rin, rho0[ng:-ng], 'b-', label='Final', linewidth=2)
    axes[0].set_xlabel('r')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].set_title(f'Sod: Density (t={t:.3f})')

    # Pressure
    axes[1].plot(rin, p[ng:-ng], 'b-', linewidth=2)
    axes[1].set_xlabel('r')
    axes[1].set_ylabel('Pressure')
    axes[1].set_title('Pressure')

    # Velocity
    axes[2].plot(rin, v[ng:-ng], 'b-', linewidth=2)
    axes[2].set_xlabel('r')
    axes[2].set_ylabel('Velocity')
    axes[2].set_title('Velocity')

    plt.tight_layout()
    plt.savefig("test_sod_engrenage.png", dpi=150, bbox_inches="tight")
    print("Plot saved: test_sod_engrenage.png")

    # Check for shock structure
    grad = np.gradient(rho0[ng:-ng], rin)
    contact = np.any(np.abs(grad) > 0.5)
    variation = np.std(rho0[ng:-ng]) / np.mean(rho0[ng:-ng])

    ok = (variation > 0.1) and contact
    print(f"Variation: {variation:.3f}, Contact detected: {contact}")
    print("PASS" if ok else "FAIL")
    return ok


def test_blast_wave(case='weak'):
    """Blast wave test with MP5 reconstruction."""
    print("\n" + "="*60)
    print(f"TEST 5: Blast wave ({case}, Minkowski, engrenage infrastructure)")
    print("="*60)

    # Create grid and hydro with MP5
    grid, hydro, background, eos = create_hydro_and_grid(
        n_interior=1000, r_max=1.0, gamma=1.4,
        reconstructor_name="mp5",
        spacetime_mode="fixed_minkowski"
    )

    r = grid.r
    ng = NUM_GHOSTS

    # Blast parameters
    r_mid = 0.5 * (r[ng] + r[-ng-1])
    if case.lower() == 'weak':
        p_in, p_out = 1.0, 0.1
        rho_in, rho_out = 1.0, 0.125
    elif case.lower() == 'strong':
        p_in, p_out = 133.33, 0.125
        rho_in, rho_out = 10.0, 1.0
    else:
        raise ValueError("case must be 'weak' or 'strong'")

    rho0 = np.where(r < r_mid, rho_in, rho_out).astype(float)
    p = np.where(r < r_mid, p_in, p_out).astype(float)
    v = np.zeros(grid.N)

    # Create initial state
    state_2d = create_initial_state(grid, rho0, v, p, eos)

    # Fixed BSSN
    bssn_fixed = state_2d[:NUM_BSSN_VARS, :].copy()
    bssn_d1_fixed = grid.get_d1_metric_quantities(state_2d)

    # Time evolution
    t, Tfinal = 0.0, 0.4
    steps = 0
    while t < Tfinal and steps < 10000:
        dt = compute_dt(rho0, v, p, eos, grid, cfl=0.3)
        state_2d = rk3_step(state_2d, dt, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

        # Update primitives
        bssn_vars = BSSNVars(grid.N)
        bssn_vars.set_bssn_vars(bssn_fixed)
        rho0, v, p = extract_primitives(state_2d, grid, hydro, bssn_vars)

        t += dt
        steps += 1

    print(f"Blast {case}: t={t:.3f}, steps={steps}")

    # Compute derived quantities
    rin = r[ng:-ng]
    rho_in = rho0[ng:-ng]
    p_in = p[ng:-ng]
    v_in = v[ng:-ng]

    # Lorentz factor
    W_in = 1.0 / np.sqrt(np.maximum(1.0 - v_in**2, 1e-16))

    # Specific internal energy
    eps_in = eos.eps_from_rho_p(rho_in, p_in)

    # Speed of sound
    h_in = 1.0 + eps_in + p_in / np.maximum(rho_in, 1e-30)
    cs2_in = np.clip(eos.gamma * p_in / np.maximum(rho_in * h_in, 1e-30), 0.0, 1.0 - 1e-10)
    cs_in = np.sqrt(cs2_in)

    # Relativistic Mach number
    W_s_in = 1.0 / np.sqrt(np.maximum(1.0 - cs2_in, 1e-16))
    Mach_in = (W_in * np.abs(v_in)) / np.maximum(W_s_in * cs_in, 1e-30)

    # Plot 3x2 layout
    fig, axes = plt.subplots(3, 2, figsize=(10, 14))
    lw = 2.5

    # Density
    axes[0, 0].plot(rin, rho_in, 'b-', linewidth=lw)
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel('Density')

    # Pressure
    axes[0, 1].plot(rin, p_in, 'b-', linewidth=lw)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('Pressure')

    # Velocity
    axes[1, 0].plot(rin, v_in, 'b-', linewidth=lw)
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel(r'$v^r$')

    # Lorentz factor
    axes[1, 1].plot(rin, W_in, 'b-', linewidth=lw)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel(r'$W$')

    # Internal energy
    axes[2, 0].plot(rin, eps_in, 'b-', linewidth=lw)
    axes[2, 0].set_xlabel('r')
    axes[2, 0].set_ylabel(r'$\varepsilon$')

    # Mach number
    axes[2, 1].plot(rin, Mach_in, 'b-', linewidth=lw)
    axes[2, 1].axhline(y=1.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='Sonic')
    axes[2, 1].set_xlabel('r')
    axes[2, 1].set_ylabel(r'Mach $\mathcal{M}$')
    axes[2, 1].legend()

    fig.suptitle(f"Blast wave ({case}) - t = {t:.3f}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"test_blast_{case}_engrenage.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved: test_blast_{case}_engrenage.png")

    # Check for expected structure
    variation = float(np.std(rho_in) / max(np.mean(rho_in), 1e-12))
    grad = np.gradient(rho_in, rin)
    contact = bool(np.any(np.abs(grad) > 0.5))

    ok = (variation > 0.1) and contact
    print(f"Variation: {variation:.3f}, Contact: {contact}")
    print("PASS" if ok else "FAIL")
    return ok


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("="*60)
    print("SUITE - Valencia Reference-Metric (engrenage infrastructure)")
    print("="*60)
    print(f"Using NUM_GHOSTS = {NUM_GHOSTS}")
    print(f"Using NUM_BSSN_VARS = {NUM_BSSN_VARS}")

    results = []
    results.append(("Uniform", test_uniform_state()))
    results.append(("cons2prim", test_cons2prim_roundtrip()))
    results.append(("Conservation", test_conservation_short()))
    results.append(("Sod", test_riemann_sod()))
    results.append(("Blast weak", test_blast_wave(case="weak")))
    results.append(("Blast strong", test_blast_wave(case="strong")))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"{name:14s}: {'PASS' if ok else 'FAIL'}")
    print("-"*40)
    print(f"Total: {passed}/{len(results)}")
