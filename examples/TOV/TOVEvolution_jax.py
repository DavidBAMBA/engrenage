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
from source.matter.hydro.eos import IdealGasEOS, PolytropicEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.atmosphere import AtmosphereParams

# JAX hydro pipeline
from source.matter.hydro.jax.valencia_jax import (
    CowlingGeometry,
    compute_hydro_rhs_cowling,
)
from source.core.timeintegration_jax import (  # noqa: F401 - available for future lax.scan use
    evolve_scan,
)

# TOV initial data
from examples.TOV.tov_solver import load_or_solve_tov_iso
import examples.TOV.tov_initial_data_interpolated as tov_id
import examples.TOV.utils_TOVEvolution as utils
from examples.TOV.utils_TOVEvolution import (SimulationDataManager, get_star_folder_name,
                                              compute_baryon_mass, compute_constraints)


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
    # CONFIGURATION (production parameters matching TOVEvolution.py)
    # ==================================================================
    r_max = 100.0
    num_points = int(os.environ.get("NUM_POINTS", 1000))
    K = 100.0
    Gamma = 2.0
    rho_central = 1.28e-3
    t_final = 10  # Start with short evolution for testing
    FOLDER_NAME_EVOL = f"tov_evolution_data_rmax{r_max}_jax_reconstructor_mc"  # Folder name for evolution data

    # Data saving
    PLOT_SUFFIX = "_cow"  # Suffix for plots and data files (matches Numba convention)
    SNAPSHOT_INTERVAL = 100  # Save full domain every N timesteps (None to disable)
    EVOLUTION_INTERVAL = 100  # Save time series every N timesteps
    ENABLE_DATA_SAVING = True
    SAVE_TIMESERIES = True

    # Reconstruction: "wenoz" (wz), "weno5" (w5), "mp5" (mp5), "minmod" (md), "mc" (mc)
    RECONSTRUCTOR_NAME = "mc"

    # Cons2prim solver: "newton" (fast, needs good guess) or "kastaun" (robust, guaranteed convergence)
    SOLVER_METHOD = "newton"

    # Riemann solver: "hll", "hllc", or "llf"
    RIEMANN_SOLVER = "hll"

    # Evolution mode: "cowling" or "dynamic"
    EVOLUTION_MODE = "dynamic"  # JAX only supports Cowling for now

    # Atmosphere config
    rho_floor_base = 1e-13
    p_floor_base = K * (rho_floor_base)**Gamma
    ATMOSPHERE = AtmosphereParams(
        rho_floor=rho_floor_base,
        p_floor=p_floor_base
    )

    # ==================================================================
    # SETUP (reuse engrenage infrastructure for initial data)
    # ==================================================================
    spacing = LinearSpacing(num_points, r_max)
    eos = IdealGasEOS(gamma=Gamma)
    #eos = PolytropicEOS(K=K, gamma=Gamma)
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
    print(f"Evolution mode: {EVOLUTION_MODE}")
    print(f"Grid: N={grid.N}, r_max={r_max}, dr_min={grid.min_dr}")
    print(f"EOS: Gamma={Gamma} (Ideal Gas)")
    print(f"Reconstruction: {RECONSTRUCTOR_NAME}")
    print(f"Riemann solver: {RIEMANN_SOLVER}")
    print(f"Cons2prim solver: {SOLVER_METHOD}")
    print(f"dt={dt:.6f} (CFL={cfl_factor}), t_final={t_final}, steps={num_steps_total}")
    print(f"Data saving: {ENABLE_DATA_SAVING} (snapshots every {SNAPSHOT_INTERVAL if SNAPSHOT_INTERVAL else 'disabled'} steps)")
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
    # INITIALIZE DATA MANAGER (same structure as TOVEvolution.py)
    # ==================================================================
    DATA_ROOT_DIR = os.path.join(script_dir, FOLDER_NAME_EVOL)
    star_folder = get_star_folder_name(
        rho_central, num_points, K, Gamma, EVOLUTION_MODE, RECONSTRUCTOR_NAME
    )
    OUTPUT_DIR = os.path.join(DATA_ROOT_DIR, star_folder)

    data_manager = SimulationDataManager(
        OUTPUT_DIR, grid, hydro,
        enable_saving=ENABLE_DATA_SAVING,
        suffix=PLOT_SUFFIX,
        restart_mode=False
    )

    # Save metadata
    if ENABLE_DATA_SAVING:
        data_manager.save_metadata(
            tov_solution, ATMOSPHERE, dt, "rk4",
            K=K, Gamma=Gamma, rho_central=rho_central,
            r_max=r_max, num_points=num_points, t_final=t_final,
            reconstructor=RECONSTRUCTOR_NAME, solver_method=SOLVER_METHOD,
            riemann_solver=RIEMANN_SOLVER, evolution_mode=EVOLUTION_MODE,
            cfl_factor=cfl_factor
        )

    # Store reference primitives for error computation
    # prim_tuple returns (rho0, vr, p, eps) - for static TOV, vr=0 so W=1
    rho_ref, vr_ref, p_ref, eps_ref = prim_tuple
    W_ref = np.ones(grid.N)  # Static initial data
    h_ref = 1.0 + eps_ref + p_ref / (rho_ref + 1e-30)
    success_ref = np.ones(grid.N, dtype=bool)

    # Save initial snapshot (step 0) with constraints - matching reference behavior
    if ENABLE_DATA_SAVING:
        Ham_0, Mom_0 = compute_constraints(initial_state_2d, grid, background, hydro)
        data_manager.save_snapshot(0, 0.0, initial_state_2d, rho_ref, vr_ref, p_ref,
                                   eps_ref, W_ref, h_ref, Ham=Ham_0, Mom=Mom_0)
        data_manager.add_evolution_point(0, 0.0, initial_state_2d,
                                        rho_ref, vr_ref, p_ref, eps_ref, W_ref, h_ref, success_ref,
                                        rho_ref, vr_ref, p_ref, eps_ref, W_ref, h_ref, success_ref,
                                        Ham=Ham_0, Mom=Mom_0)

    # Time series tracking (collected at EVOLUTION_INTERVAL for efficiency)
    Mb0 = compute_baryon_mass(grid, initial_state_2d, rho_ref, vr_ref, p_ref, eps_ref, W_ref, h_ref)
    all_times = [0.0]
    all_Mb = [Mb0]
    all_rho_c = [float(rho_ref[NUM_GHOSTS])]
    all_v_c = [float(vr_ref[NUM_GHOSTS])]

    # Preserve config value before loop variable shadowing
    rho_central_config = rho_central

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
    eos_type = 'ideal_gas'
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
        'tol': 1e-10,
        'max_iter': 100,
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
                SOLVER_METHOD,
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
    step_time_ms = bench_time

    # ==================================================================
    # EVOLUTION (single unified loop with checkpoint tracking)
    # ==================================================================
    checkpoint_steps = {
        1: max(1, num_steps_total // 3),
        2: max(2, 2 * num_steps_total // 3),
        3: num_steps_total,
    }

    print(f"\n{'='*70}")
    print(f"Evolution checkpoints:")
    for cp_id, cp_step in checkpoint_steps.items():
        print(f"  step {cp_step:6d}: {cp_id}/3 (~{cp_step*dt:.3e})")
    print(f"{'='*70}\n")

    checkpoint_states = {}
    checkpoint_times = {}

    # BSSN data stays fixed in Cowling mode
    bssn_fixed = initial_state_2d[:NUM_BSSN_VARS, :].copy()

    def build_full_state(D_arr, Sr_arr, tau_arr):
        """Build full state array from hydro variables."""
        state = np.zeros((NUM_BSSN_VARS + 3, grid.N))
        state[:NUM_BSSN_VARS, :] = bssn_fixed
        state[NUM_BSSN_VARS + 0, :] = np.asarray(D_arr)
        state[NUM_BSSN_VARS + 1, :] = np.asarray(Sr_arr)
        state[NUM_BSSN_VARS + 2, :] = np.asarray(tau_arr)
        return state

    def extract_primitives(state_2d):
        """Extract primitives from state using Numba pipeline."""
        bssn = BSSNVars(grid.N)
        bssn.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
        hydro.set_matter_vars(state_2d, bssn, grid)
        rho0, vr, p, eps, W, h, success = hydro._get_primitives(bssn, grid.r)
        return rho0, vr, p, eps, W, h, success

    # Start timing
    t0_evol = time.perf_counter()

    D, Sr, tau = D0, Sr0, tau0
    t = 0.0
    PRINT_INTERVAL = 1000
    NAN_CHECK_INTERVAL = 1000

    # Print header
    print("===== Evolution diagnostics (per step) =====")
    print("Columns: step | t | rho_central | max_drho/rho@r | max_vr@r | max_Sr@r | c2p_fails")
    print("-" * 120)

    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
    r_interior = grid.r[interior]
    checkpoint_set = set(checkpoint_steps.values())

    for step in range(1, num_steps_total + 1):
        D, Sr, tau = jit_step(D, Sr, tau, geom)
        t += dt

        # NaN detection (periodic, lightweight)
        if step % NAN_CHECK_INTERVAL == 0:
            jax.block_until_ready(D)
            if bool(jnp.any(jnp.isnan(D))):
                print(f"\nERROR: NaN detected at step {step}, t={t:.6e}. Halting evolution.")
                break

        # Diagnostics and data saving
        should_print = (step % PRINT_INTERVAL == 0)
        should_save_evol = ENABLE_DATA_SAVING and EVOLUTION_INTERVAL and (step % EVOLUTION_INTERVAL == 0)
        should_save_snap = ENABLE_DATA_SAVING and SNAPSHOT_INTERVAL and (step % SNAPSHOT_INTERVAL == 0)

        if should_print or should_save_evol or should_save_snap:
            state_2d = build_full_state(D, Sr, tau)
            rho0, vr, p, eps, W, h, success = extract_primitives(state_2d)

            Ham, Mom = None, None
            if should_save_evol or should_save_snap:
                Ham, Mom = compute_constraints(state_2d, grid, background, hydro)

            if should_save_evol:
                data_manager.add_evolution_point(
                    step, t, state_2d,
                    rho0, vr, p, eps, W, h, success,
                    rho_ref, vr_ref, p_ref, eps_ref, W_ref, h_ref, success_ref,
                    Ham=Ham, Mom=Mom
                )
                Mb = compute_baryon_mass(grid, state_2d, rho0, vr, p, eps, W, h)
                all_times.append(t)
                all_Mb.append(Mb)
                all_rho_c.append(float(rho0[NUM_GHOSTS]))
                all_v_c.append(float(vr[NUM_GHOSTS]))

                if step % (EVOLUTION_INTERVAL * 10) == 0:
                    data_manager.flush_evolution_buffer()

            if should_save_snap:
                data_manager.save_snapshot(step, t, state_2d, rho0, vr, p, eps, W, h,
                                           Ham=Ham, Mom=Mom)

            if should_print:
                rho_central = float(rho0[NUM_GHOSTS])

                rho_int = rho0[interior]
                rho_init_int = rho_ref[interior]
                rel_rho_err = np.abs(rho_int - rho_init_int) / (np.abs(rho_init_int) + 1e-20)
                idx_max_rho = np.argmax(rel_rho_err)
                max_rel_rho_err = float(rel_rho_err[idx_max_rho])
                r_max_rho = float(r_interior[idx_max_rho])

                vr_int = vr[interior]
                idx_max_v = np.argmax(np.abs(vr_int))
                max_abs_v = float(vr_int[idx_max_v])
                r_max_v = float(r_interior[idx_max_v])

                Sr_np = np.asarray(state_2d[NUM_BSSN_VARS + 1, interior])
                idx_max_Sr = np.argmax(np.abs(Sr_np))
                max_Sr = float(np.abs(Sr_np[idx_max_Sr]))
                r_max_Sr = float(r_interior[idx_max_Sr])

                c2p_fail = int(np.sum(~success))

                print(f"step {step:5d}  t={t:.2e}:  rho_c={rho_central:.6e}  "
                      f"max_drho/rho={max_rel_rho_err:.2e}@r={r_max_rho:.2f}  "
                      f"max_vr={max_abs_v:.3e}@r={r_max_v:.2f}  "
                      f"max_Sr={max_Sr:.2e}@r={r_max_Sr:.2f}  c2p_fail={c2p_fail}")

        # Save checkpoint
        if step in checkpoint_set:
            jax.block_until_ready((D, Sr, tau))
            cp_id = [k for k, v in checkpoint_steps.items() if v == step][0]
            checkpoint_states[cp_id] = (np.array(D), np.array(Sr), np.array(tau))
            checkpoint_times[cp_id] = t
            data_manager.flush_evolution_buffer()
            print(f"  -> Reached checkpoint {cp_id}, t={t:.6e}")

    evol_time = time.perf_counter() - t0_evol
    D_final, Sr_final, tau_final = D, Sr, tau
    data_manager.flush_evolution_buffer()

    # Fill missing checkpoints if halted early
    for cp_id in [1, 2, 3]:
        if cp_id not in checkpoint_states:
            jax.block_until_ready((D, Sr, tau))
            checkpoint_states[cp_id] = (np.array(D), np.array(Sr), np.array(tau))
            checkpoint_times[cp_id] = t

    print(f"\nEvolution complete!")
    print(f"  Total wall time: {evol_time:.2f}s")
    print(f"  Steps/second: {num_steps_total / evol_time:.0f}")

    # Finalize data manager
    data_manager.finalize(execution_time_seconds=evol_time)

    # ==================================================================
    # SAVE TIME SERIES TO FILE (matching reference behavior)
    # ==================================================================
    if ENABLE_DATA_SAVING and SAVE_TIMESERIES and len(all_times) > 0:
        timeseries_path = os.path.join(OUTPUT_DIR, "timeseries.npz")
        np.savez(timeseries_path,
                 times=np.array(all_times),
                 rho_central=np.array(all_rho_c),
                 v_central=np.array(all_v_c),
                 Mb=np.array(all_Mb),
                 num_points=num_points,
                 K=K,
                 Gamma=Gamma,
                 rho_central_initial=rho_central_config,
                 r_max=r_max,
                 dt=dt,
                 num_steps=num_steps_total)
        print(f"\nTime series saved to: {timeseries_path}")

    # ==================================================================
    # RECONSTRUCT FULL STATE FOR DIAGNOSTICS
    # ==================================================================
    D_f_np = np.asarray(D_final)
    Sr_f_np = np.asarray(Sr_final)
    tau_f_np = np.asarray(tau_final)

    D0_np = np.asarray(D0)
    Sr0_np = np.asarray(Sr0)
    tau0_np = np.asarray(tau0)

    state_initial = build_full_state(D0_np, Sr0_np, tau0_np)
    state_final = build_full_state(D_f_np, Sr_f_np, tau_f_np)

    print("\nRecovering primitives for diagnostics...")
    rho0_0, vr_0, p_0, _, _, _, success_0 = extract_primitives(state_initial)
    rho0_f, vr_f, p_f, _, _, _, success_f = extract_primitives(state_final)

    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

    # ==================================================================
    # COWLING APPROXIMATION CHECK
    # ==================================================================
    print(f"\n{'='*70}")
    print("COWLING APPROXIMATION CHECK")
    print(f"{'='*70}")

    # In Cowling mode, BSSN variables should remain exactly constant
    # since we only evolve hydro, not the metric
    bssn_0 = initial_state_2d[:NUM_BSSN_VARS, :]
    bssn_f = state_final[:NUM_BSSN_VARS, :]

    bssn_change = np.max(np.abs(bssn_f - bssn_0) / (np.abs(bssn_0) + 1e-20))
    print(f"  Max BSSN change (should be 0): {bssn_change:.3e}")

    if bssn_change < 1e-14:
        print("  ✓ BSSN variables constant - Cowling approximation verified")
    else:
        print(f"  ⚠ BSSN variables changed by {bssn_change:.3e}")
        print("    (This should not happen in Cowling mode!)")

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
    # PLOTS (using utils_TOVEvolution for consistency with TOVEvolution.py)
    # ==================================================================
    R_star = tov_solution.R_iso

    # Build states for each checkpoint: t=0, 1/3, 2/3, final
    state_cp0 = build_full_state(D0, Sr0, tau0)
    state_cp1 = build_full_state(*checkpoint_states[1])
    state_cp2 = build_full_state(*checkpoint_states[2])
    state_cp3 = build_full_state(*checkpoint_states[3])

    states = [state_cp0, state_cp1, state_cp2, state_cp3]
    times_cp = [0.0, checkpoint_times[1], checkpoint_times[2], checkpoint_times[3]]

    # Plot evolution using utils (same format as TOVEvolution.py)
    print("\nGenerating plots...")
    utils.plot_evolution(
        states, times_cp, grid, hydro, rho_ref, p_ref,
        suffix=PLOT_SUFFIX, R_star=R_star
    )

    # Plot BSSN variables evolution (in Cowling mode, should be constant)
    utils.plot_bssn_evolution(
        state_cp0, state_cp3, grid,
        t_0=0.0, t_final=checkpoint_times[3],
        suffix=PLOT_SUFFIX
    )

    print("\n" + "="*70)
    print("JAX Evolution complete.")
    print(f"\nData saved to: {OUTPUT_DIR}")
    print(f"  - tov_snapshots{PLOT_SUFFIX}.h5   (full domain snapshots + constraints)")
    print(f"  - tov_evolution{PLOT_SUFFIX}.h5   (time series + constraints)")
    print(f"  - tov_metadata{PLOT_SUFFIX}.json  (simulation parameters)")
    print(f"  - timeseries.npz                (rho_c, v_c, Mb vs time)")
    print(f"\nPlots saved to: {plots_dir}")
    print(f"  - tov_evolution{PLOT_SUFFIX}.png  (hydro at checkpoints)")
    print(f"  - tov_bssn_evolution{PLOT_SUFFIX}.png (BSSN check)")
    print("="*70)

    # ==================================================================
    # TIMING SUMMARY
    # ==================================================================
    print(f"\n{'='*70}")
    print("TIMING SUMMARY")
    print(f"{'='*70}")
    print(f"  JIT compilation:        {jit_time:.2f}s")
    print(f"  Benchmark step time:    {step_time_ms:.2f} ms")
    print(f"  Evolution:              {evol_time:.2f}s ({num_steps_total} steps)")
    print(f"  Steps/second:           {num_steps_total / evol_time:.0f}")
    print(f"  Time per RK4 step:      {evol_time / num_steps_total * 1e3:.3f} ms (includes diagnostics overhead)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
