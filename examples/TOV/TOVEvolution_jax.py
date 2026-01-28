"""
TOV Star Evolution using JAX backend (Cowling approximation).

This script mirrors TOVEvolution.py but uses the JAX-native hydro pipeline.
The entire RHS is JIT-compiled and the time loop uses jax.lax.scan,
eliminating per-timestep Python overhead.

Usage:
    python examples/TOV/TOVEvolution_jax.py
"""

import numpy as np
import sys
import os
import time
from functools import partial

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Engrenage core - add parent directories to path
sys.path.insert(0, os.path.join(script_dir, '..', '..'))

# Configure JAX before any JAX imports
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground

# BSSN
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS
from source.bssn.tensoralgebra import get_bar_gamma_LL, get_bar_A_LL, get_hat_D_bar_gamma_LL

# Hydro (for initial data setup and reference)
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import PolytropicEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.atmosphere import AtmosphereParams

# JAX hydro pipeline
from source.matter.hydro.jax.valencia_jax import (
    CowlingGeometry,
    compute_hydro_rhs_cowling,
)
from source.core.timeintegration_jax import (
    rk4_step,
    apply_atmosphere_reset_jax,
    evolve_scan,
    evolve_python_loop,
)

# TOV initial data
from examples.TOV.tov_solver import load_or_solve_tov_iso
import examples.TOV.tov_initial_data_interpolated as tov_id


# =============================================================================
# JAX Boundary conditions (functional, JIT-compatible)
# =============================================================================

@partial(jax.jit, static_argnums=(3,))
def fill_boundaries_jax(D, Sr, tau, num_ghosts):
    """
    Apply boundary conditions for hydro variables in JAX (functional).

    Inner boundary (r=0): parity reflection
        D(ghost)   = +D(mirror)    (even)
        Sr(ghost)  = -Sr(mirror)   (odd)
        tau(ghost) = +tau(mirror)  (even)

    Outer boundary: zero-gradient (outflow)
    """
    NG = num_ghosts

    # Inner boundary: parity reflection
    # Mirror indices: ghost[0]=mirror[5], ghost[1]=mirror[4], ghost[2]=mirror[3]
    # i.e., state[:NG] = parity * state[2*NG-1 : NG-1 : -1]
    mirror = jnp.flip(jnp.array([D[NG:2*NG], Sr[NG:2*NG], tau[NG:2*NG]]), axis=1)

    D = D.at[:NG].set(mirror[0])         # parity +1
    Sr = Sr.at[:NG].set(-mirror[1])       # parity -1
    tau = tau.at[:NG].set(mirror[2])      # parity +1

    # Outer boundary: zero-gradient
    D = D.at[-NG:].set(D[-NG - 1])
    Sr = Sr.at[-NG:].set(Sr[-NG - 1])
    tau = tau.at[-NG:].set(tau[-NG - 1])

    return D, Sr, tau


# =============================================================================
# Geometry extraction: BSSN -> CowlingGeometry
# =============================================================================

def build_cowling_geometry(initial_state_2d, grid, background):
    """
    Extract static geometry from BSSN variables and build a CowlingGeometry.

    This is done once (in NumPy) and the result is transferred to JAX arrays.
    """
    N = grid.N
    r = grid.r

    # Extract BSSN variables
    bssn_vars = BSSNVars(N)
    bssn_vars.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])

    # Compute BSSN derivatives
    bssn_d1 = grid.get_d1_metric_quantities(initial_state_2d)

    # Core geometry
    alpha = np.asarray(bssn_vars.lapse, dtype=np.float64)
    beta_U = np.asarray(bssn_vars.shift_U, dtype=np.float64) * background.inverse_scaling_vector
    phi = np.asarray(bssn_vars.phi, dtype=np.float64)
    e4phi = np.exp(4.0 * phi)
    e6phi = np.exp(6.0 * phi)

    bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
    gamma_LL = e4phi[:, None, None] * bar_gamma_LL
    gamma_UU = np.linalg.inv(gamma_LL)

    # Extrinsic curvature K_ij
    K_scalar = np.asarray(bssn_vars.K, dtype=np.float64)
    bar_A_LL = get_bar_A_LL(r, bssn_vars, background)
    K_LL = e4phi[:, None, None] * bar_A_LL + (K_scalar / 3.0)[:, None, None] * gamma_LL

    # Lapse derivative
    dalpha_dx = np.asarray(bssn_d1.lapse)

    # Shift derivative and covariant derivative
    dbeta_dx = (
        background.inverse_scaling_vector[:, :, None] * np.asarray(bssn_d1.shift_U)
        + bssn_vars.shift_U[:, :, None] * background.d1_inverse_scaling_vector
    )
    hat_chris = background.hat_christoffel
    hatD_beta_U = np.transpose(dbeta_dx, (0, 2, 1)) + np.einsum('xjik,xk->xij', hat_chris, beta_U)

    # Covariant derivative of metric
    dphi_dx = np.asarray(bssn_d1.phi)
    hat_D_bar_gamma = get_hat_D_bar_gamma_LL(r, bssn_vars.h_LL, bssn_d1.h_LL, background)
    hatD_gamma_LL = e4phi[:, None, None, None] * (
        4.0 * dphi_dx[:, :, None, None] * bar_gamma_LL[:, None, :, :]
        + np.transpose(hat_D_bar_gamma, (0, 3, 1, 2))
    )

    # Build CowlingGeometry (transfers to JAX device)
    geom = CowlingGeometry(
        alpha=alpha,
        beta_r=beta_U[:, 0],
        gamma_rr=gamma_LL[:, 0, 0],
        e6phi=e6phi,
        dx=float(grid.derivs.dx),
        num_ghosts=NUM_GHOSTS,
        K_LL=K_LL,
        dalpha_dx=dalpha_dx,
        hatD_beta_U=hatD_beta_U,
        hatD_gamma_LL=hatD_gamma_LL,
        hat_christoffel=hat_chris,
        beta_U=beta_U,
        gamma_LL=gamma_LL,
        gamma_UU=gamma_UU,
        e4phi=e4phi,
    )

    return geom


# =============================================================================
# Main
# =============================================================================

def main():
    """Main execution."""

    # ==================================================================
    # CONFIGURATION (same as TOVEvolution.py)
    # ==================================================================
    r_max = 100.0
    num_points = 1000
    K = 100.0
    Gamma = 2.0
    rho_central = 1.28e-3
    t_final = 10  # Short test

    RECONSTRUCTOR_NAME = "mp5"
    RIEMANN_SOLVER = "hll"
    SOLVER_METHOD = "newton"

    # Atmosphere
    rho_floor_base = 1e-12 * rho_central
    p_floor_base = K * (rho_floor_base) ** Gamma
    ATMOSPHERE = AtmosphereParams(
        rho_floor=rho_floor_base,
        p_floor=p_floor_base,
    )

    # ==================================================================
    # SETUP (reuse engrenage infrastructure for initial data)
    # ==================================================================
    spacing = LinearSpacing(num_points, r_max)
    eos = PolytropicEOS(K=K, gamma=Gamma)
    base_recon = create_reconstruction(RECONSTRUCTOR_NAME)
    riemann = HLLRiemannSolver(atmosphere=ATMOSPHERE)

    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere=ATMOSPHERE,
        reconstructor=base_recon,
        riemann_solver=riemann,
        solver_method=SOLVER_METHOD,
    )

    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    cfl_factor = 0.1
    dt = cfl_factor * grid.min_dr
    num_steps_total = int(t_final / dt)

    print("=" * 70)
    print("TOV Star Evolution - JAX Backend (Cowling Approximation)")
    print("=" * 70)
    print(f"Grid: N={grid.N}, r_max={r_max}, dr_min={grid.min_dr}")
    print(f"EOS: K={K}, Gamma={Gamma} (Polytropic)")
    print(f"Reconstruction: {RECONSTRUCTOR_NAME}")
    print(f"Riemann solver: {RIEMANN_SOLVER}")
    print(f"dt={dt:.6f} (CFL={cfl_factor}), t_final={t_final}, steps={num_steps_total}")
    print(f"JAX devices: {jax.devices()}")
    print()

    # ==================================================================
    # SOLVE TOV & CREATE INITIAL DATA
    # ==================================================================
    print("Solving TOV equations...")
    tov_solution = load_or_solve_tov_iso(
        K=K, Gamma=Gamma, rho_central=rho_central,
        r_max=r_max, accuracy="high"
    )
    print(f"TOV: M={tov_solution.M_star:.6f}, R_iso={tov_solution.R_iso:.3f}")

    print("Creating initial data...")
    initial_state_2d, prim_tuple = tov_id.create_initial_data_iso(
        tov_solution, grid, background, eos,
        atmosphere=ATMOSPHERE,
        polytrope_K=K, polytrope_Gamma=Gamma,
        interp_order=11,
    )

    # ==================================================================
    # BUILD COWLING GEOMETRY (one-time NumPy computation)
    # ==================================================================
    print("Building CowlingGeometry from BSSN variables...")
    t0_geom = time.perf_counter()
    geom = build_cowling_geometry(initial_state_2d, grid, background)
    print(f"  Geometry built in {time.perf_counter() - t0_geom:.3f}s")

    # ==================================================================
    # EXTRACT INITIAL HYDRO STATE -> JAX ARRAYS
    # ==================================================================
    D0 = jnp.asarray(initial_state_2d[NUM_BSSN_VARS + 0, :])
    Sr0 = jnp.asarray(initial_state_2d[NUM_BSSN_VARS + 1, :])
    tau0 = jnp.asarray(initial_state_2d[NUM_BSSN_VARS + 2, :])

    print(f"  Initial D_c   = {float(D0[NUM_GHOSTS]):.6e}")
    print(f"  Initial Sr_c  = {float(Sr0[NUM_GHOSTS]):.6e}")
    print(f"  Initial tau_c = {float(tau0[NUM_GHOSTS]):.6e}")

    # ==================================================================
    # CONFIGURE JAX RHS AND ATMOSPHERE PARAMETERS
    # ==================================================================
    eos_type = 'polytropic'
    _rho_floor = float(ATMOSPHERE.rho_floor)
    _p_floor = float(ATMOSPHERE.p_floor)
    _v_max = float(ATMOSPHERE.v_max)
    _gm1 = Gamma - 1.0

    eos_params = {'gamma': Gamma, 'K': K}
    atm_params = {
        'rho_floor': _rho_floor,
        'p_floor': _p_floor,
        'v_max': _v_max,
        'W_max': 10.0,
        'tol': 1e-12,
        'max_iter': 500,
    }

    # ==================================================================
    # BUILD JIT-COMPILED STEP FUNCTION
    # ==================================================================
    # Define full step (RK4 + atm reset) as a closure capturing all static
    # parameters. @jax.jit traces through everything including the nested
    # @jit functions, producing one fused XLA program.
    # ==================================================================

    @jax.jit
    def jit_step(D, Sr, tau, geom_arg):
        """One complete RK4 step + atmosphere reset, JIT-compiled."""
        def _rhs(D_in, Sr_in, tau_in):
            D_bc, Sr_bc, tau_bc = fill_boundaries_jax(D_in, Sr_in, tau_in, NUM_GHOSTS)
            return compute_hydro_rhs_cowling(
                D_bc, Sr_bc, tau_bc, geom_arg,
                eos_type, eos_params, atm_params,
                RECONSTRUCTOR_NAME, RIEMANN_SOLVER,
            )

        # RK4 stages
        dD1, dSr1, dtau1 = _rhs(D, Sr, tau)

        D2 = D + 0.5 * dt * dD1
        Sr2 = Sr + 0.5 * dt * dSr1
        tau2 = tau + 0.5 * dt * dtau1
        dD2, dSr2, dtau2 = _rhs(D2, Sr2, tau2)

        D3 = D + 0.5 * dt * dD2
        Sr3 = Sr + 0.5 * dt * dSr2
        tau3 = tau + 0.5 * dt * dtau2
        dD3, dSr3, dtau3 = _rhs(D3, Sr3, tau3)

        D4 = D + dt * dD3
        Sr4 = Sr + dt * dSr3
        tau4 = tau + dt * dtau3
        dD4, dSr4, dtau4 = _rhs(D4, Sr4, tau4)

        D_new = D + (dt / 6.0) * (dD1 + 2.0*dD2 + 2.0*dD3 + dD4)
        Sr_new = Sr + (dt / 6.0) * (dSr1 + 2.0*dSr2 + 2.0*dSr3 + dSr4)
        tau_new = tau + (dt / 6.0) * (dtau1 + 2.0*dtau2 + 2.0*dtau3 + dtau4)

        # Atmosphere reset inline
        eps_atm = K * _rho_floor**_gm1 / _gm1
        p_atm = K * _rho_floor**Gamma
        h_atm = 1.0 + eps_atm + p_atm / _rho_floor

        D_atm = geom_arg.e6phi * _rho_floor
        tau_atm = geom_arg.e6phi * (_rho_floor * h_atm - p_atm - _rho_floor)

        threshold = 100.0 * _rho_floor * geom_arg.e6phi
        atm_mask = D_new < threshold

        D_new = jnp.where(atm_mask, D_atm, D_new)
        Sr_new = jnp.where(atm_mask, 0.0, Sr_new)
        tau_new = jnp.where(atm_mask, tau_atm, tau_new)

        return D_new, Sr_new, tau_new

    # ==================================================================
    # JIT WARMUP
    # ==================================================================
    print("\nJIT compiling full step (RK4 + BC + RHS + atm)...")
    t0_jit = time.perf_counter()
    D_test, Sr_test, tau_test = jit_step(D0, Sr0, tau0, geom)
    jax.block_until_ready((D_test, Sr_test, tau_test))
    jit_time = time.perf_counter() - t0_jit
    print(f"  Full step JIT compilation: {jit_time:.2f}s")

    # Measure actual step execution time
    print("Benchmarking step execution...")
    t0_bench = time.perf_counter()
    n_bench = 20
    for _ in range(n_bench):
        D_test, Sr_test, tau_test = jit_step(D_test, Sr_test, tau_test, geom)
    jax.block_until_ready((D_test, Sr_test, tau_test))
    bench_time = (time.perf_counter() - t0_bench) / n_bench * 1000
    print(f"  Single step execution: {bench_time:.2f} ms")
    rk4_jit_time = jit_time

    # ==================================================================
    # EVOLUTION (Python loop with JIT-compiled step)
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"Starting evolution: {num_steps_total} steps")
    print(f"{'='*70}")
    print("Using Python loop + JIT-compiled full step...")

    print_every = max(1, num_steps_total // 20)
    D, Sr, tau = D0, Sr0, tau0
    t0_evol = time.perf_counter()

    for step in range(num_steps_total):
        D, Sr, tau = jit_step(D, Sr, tau, geom)

        if (step + 1) % print_every == 0:
            jax.block_until_ready(D)
            elapsed = time.perf_counter() - t0_evol
            pct = 100.0 * (step + 1) / num_steps_total
            rate = (step + 1) / elapsed
            eta = (num_steps_total - step - 1) / rate if rate > 0 else 0
            D_c_val = float(jax.device_get(D)[NUM_GHOSTS])
            print(f"  Step {step+1:>7d}/{num_steps_total} ({pct:5.1f}%) | "
                  f"wall={elapsed:7.1f}s | ETA={eta:6.1f}s | "
                  f"rate={rate:.0f} steps/s | D_c={D_c_val:.6e}")

    jax.block_until_ready((D, Sr, tau))
    evol_time = time.perf_counter() - t0_evol
    D_final, Sr_final, tau_final = D, Sr, tau

    print(f"\nEvolution complete!")
    print(f"  Total wall time: {evol_time:.2f}s")
    print(f"  Steps/second: {num_steps_total / evol_time:.0f}")

    # ==================================================================
    # RECONSTRUCT FULL STATE FOR DIAGNOSTICS
    # ==================================================================
    D_f_np = np.asarray(D_final)
    Sr_f_np = np.asarray(Sr_final)
    tau_f_np = np.asarray(tau_final)

    D0_np = np.asarray(D0)
    Sr0_np = np.asarray(Sr0)
    tau0_np = np.asarray(tau0)

    def jax_hydro_to_full_state(D_np, Sr_np, tau_np, bssn_data):
        """Embed JAX hydro arrays into a full (NUM_VARS, N) state array."""
        state = np.zeros((NUM_BSSN_VARS + 3, grid.N))
        state[:NUM_BSSN_VARS, :] = bssn_data
        state[NUM_BSSN_VARS + 0, :] = D_np
        state[NUM_BSSN_VARS + 1, :] = Sr_np
        state[NUM_BSSN_VARS + 2, :] = tau_np
        return state

    bssn_fixed = initial_state_2d[:NUM_BSSN_VARS, :].copy()
    state_initial = jax_hydro_to_full_state(D0_np, Sr0_np, tau0_np, bssn_fixed)
    state_final = jax_hydro_to_full_state(D_f_np, Sr_f_np, tau_f_np, bssn_fixed)

    # Extract primitives using existing Numba cons2prim
    def extract_primitives(state_2d):
        bssn = BSSNVars(grid.N)
        bssn.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
        hydro.set_matter_vars(state_2d, bssn, grid)
        rho0, vr, p, eps, W, h, success = hydro._get_primitives(bssn, grid.r)
        return rho0, vr, p, eps, W, h, success

    print("\nRecovering primitives for diagnostics...")
    rho0_0, vr_0, p_0, _, _, _, success_0 = extract_primitives(state_initial)
    rho0_f, vr_f, p_f, _, _, _, success_f = extract_primitives(state_final)

    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

    # ==================================================================
    # DIAGNOSTICS
    # ==================================================================
    print(f"\n{'='*70}")
    print("DIAGNOSTICS")
    print(f"{'='*70}")

    print(f"\n  CONSERVATIVE VARIABLES:")
    print(f"  D_c(t=0)     = {D0_np[NUM_GHOSTS]:.6e}")
    print(f"  D_c(t=final) = {D_f_np[NUM_GHOSTS]:.6e}")
    dDc = abs(D_f_np[NUM_GHOSTS] - D0_np[NUM_GHOSTS]) / abs(D0_np[NUM_GHOSTS])
    print(f"  ΔD_c/D_c     = {dDc:.3e}")

    delta_D = np.max(np.abs(D_f_np[interior] - D0_np[interior]) / (np.abs(D0_np[interior]) + 1e-30))
    delta_Sr = np.max(np.abs(Sr_f_np[interior] - Sr0_np[interior]) / (np.abs(Sr0_np[interior]) + 1e-30))
    delta_tau = np.max(np.abs(tau_f_np[interior] - tau0_np[interior]) / (np.abs(tau0_np[interior]) + 1e-30))
    print(f"  max|ΔD|/|D|   = {delta_D:.3e}")
    print(f"  max|ΔSr|/|Sr| = {delta_Sr:.3e}")
    print(f"  max|Δτ|/|τ|   = {delta_tau:.3e}")

    print(f"\n  PRIMITIVE VARIABLES:")
    print(f"  ρ_c(t=0)     = {rho0_0[NUM_GHOSTS]:.6e}")
    print(f"  ρ_c(t=final) = {rho0_f[NUM_GHOSTS]:.6e}")
    drho_c = abs(rho0_f[NUM_GHOSTS] - rho0_0[NUM_GHOSTS]) / rho0_0[NUM_GHOSTS]
    print(f"  Δρ_c/ρ_c     = {drho_c:.3e}")
    print(f"  max|v^r|(t=0)     = {np.max(np.abs(vr_0)):.3e}")
    print(f"  max|v^r|(t=final) = {np.max(np.abs(vr_f)):.3e}")

    delta_rho = np.max(np.abs(rho0_f[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20))
    delta_p = np.max(np.abs(p_f[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20))
    print(f"  max|Δρ|/ρ     = {delta_rho:.3e}")
    print(f"  max|ΔP|/P     = {delta_p:.3e}")

    print(f"\n  CONS2PRIM:")
    print(f"  Success at t=0:     {np.sum(success_0)}/{grid.N}")
    print(f"  Success at t=final: {np.sum(success_f)}/{grid.N}")

    n_nan = np.sum(np.isnan(D_f_np)) + np.sum(np.isnan(Sr_f_np)) + np.sum(np.isnan(tau_f_np))
    if n_nan > 0:
        print(f"\n  WARNING: {n_nan} NaN values detected!")
    else:
        print(f"\n  No NaNs detected - evolution stable.")

    # ==================================================================
    # SAVE DATA
    # ==================================================================
    output_dir = os.path.join(script_dir, 'jax_output')
    os.makedirs(output_dir, exist_ok=True)

    data_path = os.path.join(output_dir, 'evolution_jax.npz')
    np.savez(data_path,
             # Grid
             r=grid.r,
             num_points=num_points,
             r_max=r_max,
             num_ghosts=NUM_GHOSTS,
             # Parameters
             K=K, Gamma=Gamma, rho_central=rho_central,
             dt=dt, t_final=t_final, num_steps=num_steps_total,
             cfl_factor=cfl_factor,
             reconstructor=RECONSTRUCTOR_NAME,
             riemann_solver=RIEMANN_SOLVER,
             # Initial state (primitives + conservatives)
             D_initial=D0_np, Sr_initial=Sr0_np, tau_initial=tau0_np,
             rho_initial=rho0_0, vr_initial=vr_0, p_initial=p_0,
             # Final state (primitives + conservatives)
             D_final=D_f_np, Sr_final=Sr_f_np, tau_final=tau_f_np,
             rho_final=rho0_f, vr_final=vr_f, p_final=p_f,
             # BSSN (frozen)
             bssn_fixed=bssn_fixed,
             # Timing
             wall_time=evol_time,
             jit_time_rhs=jit_time,
             jit_time_rk4=rk4_jit_time,
    )
    print(f"\nData saved to: {data_path}")

    # ==================================================================
    # PLOTS
    # ==================================================================
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    PLOT_SUFFIX = "_jax"
    r = grid.r
    R_star = tov_solution.R_iso

    # --- Plot 1: Profiles initial vs final (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.semilogy(r[interior], rho0_0[interior], 'b-', label='t=0', linewidth=1.5)
    ax.semilogy(r[interior], rho0_f[interior], 'r--', label=f't={t_final}', linewidth=1.5)
    ax.axvline(R_star, color='gray', ls=':', alpha=0.7, label=f'R={R_star:.1f}')
    ax.set_xlabel('r')
    ax.set_ylabel(r'$\rho_0$')
    ax.set_title('Baryon Density')
    ax.legend()

    ax = axes[0, 1]
    ax.semilogy(r[interior], p_0[interior], 'b-', label='t=0', linewidth=1.5)
    ax.semilogy(r[interior], p_f[interior], 'r--', label=f't={t_final}', linewidth=1.5)
    ax.axvline(R_star, color='gray', ls=':', alpha=0.7)
    ax.set_xlabel('r')
    ax.set_ylabel('P')
    ax.set_title('Pressure')
    ax.legend()

    ax = axes[1, 0]
    ax.plot(r[interior], vr_0[interior], 'b-', label='t=0', linewidth=1.5)
    ax.plot(r[interior], vr_f[interior], 'r--', label=f't={t_final}', linewidth=1.5)
    ax.axvline(R_star, color='gray', ls=':', alpha=0.7)
    ax.set_xlabel('r')
    ax.set_ylabel(r'$v^r$')
    ax.set_title('Radial Velocity')
    ax.legend()

    ax = axes[1, 1]
    mask = rho0_0[interior] > 100.0 * ATMOSPHERE.rho_floor
    err = np.abs(rho0_f[interior] - rho0_0[interior]) / (rho0_0[interior] + 1e-30)
    ax.semilogy(r[interior][mask], err[mask], 'k-', linewidth=1.0)
    ax.axvline(R_star, color='gray', ls=':', alpha=0.7)
    ax.set_xlabel('r')
    ax.set_ylabel(r'$|\Delta\rho|/\rho$')
    ax.set_title('Relative Density Error')

    fig.suptitle(f'TOV Evolution (JAX Cowling) - N={num_points}, t={t_final}, '
                 f'{RECONSTRUCTOR_NAME}/{RIEMANN_SOLVER}', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'tov_evolution{PLOT_SUFFIX}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {plot_path}")

    # --- Plot 2: Conservative variables ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (q0, qf, name) in zip(axes, [
        (D0_np, D_f_np, r'$\tilde{D}$'),
        (Sr0_np, Sr_f_np, r'$\tilde{S}_r$'),
        (tau0_np, tau_f_np, r'$\tilde{\tau}$'),
    ]):
        ax.plot(r[interior], q0[interior], 'b-', label='t=0', linewidth=1.5)
        ax.plot(r[interior], qf[interior], 'r--', label=f't={t_final}', linewidth=1.5)
        ax.axvline(R_star, color='gray', ls=':', alpha=0.7)
        ax.set_xlabel('r')
        ax.set_ylabel(name)
        ax.legend()

    fig.suptitle('Conservative Variables (JAX)', fontsize=13)
    plt.tight_layout()
    plot_path2 = os.path.join(plots_dir, f'tov_conservatives{PLOT_SUFFIX}.png')
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {plot_path2}")

    # ==================================================================
    # TIMING SUMMARY
    # ==================================================================
    print(f"\n{'='*70}")
    print("TIMING SUMMARY")
    print(f"{'='*70}")
    print(f"  JIT compilation (RHS):  {jit_time:.2f}s")
    print(f"  JIT compilation (RK4):  {rk4_jit_time:.2f}s")
    print(f"  Evolution:              {evol_time:.2f}s ({num_steps_total} steps)")
    print(f"  Steps/second:           {num_steps_total / evol_time:.0f}")
    print(f"  Time per RK4 step:      {evol_time / num_steps_total * 1e3:.3f} ms")
    print(f"{'='*70}")
    print("JAX evolution complete.")


if __name__ == "__main__":
    main()
