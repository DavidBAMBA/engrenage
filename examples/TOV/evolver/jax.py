"""
JAX-based evolution engine for TOV star.

Contains JIT-compiled step functions and the evolution loop for the GPU/JAX backend.
Supports both Cowling (fixed spacetime) and Dynamic (full BSSN+hydro) modes.

This module is only imported when JAX_RUN=True, so all JAX imports are at the
top level (no conditional guards needed).
"""

import numpy as np
import os
import time
import jax
import jax.numpy as jnp
from functools import partial


# =============================================================================
# JAX helper functions (moved from TOVEvolution.py)
# =============================================================================

@partial(jax.jit, static_argnums=(3,))
def fill_boundaries_jax(D, Sr, tau, num_ghosts):
    """Apply boundary conditions for hydro variables (JAX, functional).

    Inner boundary (r=0): parity reflection
        D(ghost)   = +D(mirror)    (even)
        Sr(ghost)  = -Sr(mirror)   (odd)
        tau(ghost) = +tau(mirror)  (even)
    Outer boundary: zero-gradient (outflow)
    """
    NG = num_ghosts
    mirror = jnp.flip(jnp.array([D[NG:2*NG], Sr[NG:2*NG], tau[NG:2*NG]]), axis=1)
    D   = D.at[:NG].set(mirror[0])
    Sr  = Sr.at[:NG].set(-mirror[1])
    tau = tau.at[:NG].set(mirror[2])
    D   = D.at[-NG:].set(D[-NG - 1])
    Sr  = Sr.at[-NG:].set(Sr[-NG - 1])
    tau = tau.at[-NG:].set(tau[-NG - 1])
    return D, Sr, tau


def build_cowling_geometry(initial_state_2d, grid, background):
    """Extract static geometry from BSSN variables and build HydroGeometry
    plus source/connection data for Cowling mode.

    Done once in NumPy; results are transferred to JAX arrays.

    Returns:
        geom:            HydroGeometry namedtuple (JAX arrays)
        source_data:     (K_LL, dalpha_dx, hatD_beta_U, hatD_gamma_LL) jnp arrays
        connection_data: (hat_christoffel,) jnp arrays
    """
    from source.bssn.bssnvars import BSSNVars
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS
    from source.bssn.tensoralgebra import get_bar_gamma_LL, get_bar_A_LL, get_hat_D_bar_gamma_LL
    from source.matter.hydro.jax.valencia_jax import HydroGeometry

    N = grid.N
    r = grid.r

    bssn_vars = BSSNVars(N)
    bssn_vars.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
    bssn_d1 = grid.get_d1_metric_quantities(initial_state_2d)

    alpha  = np.asarray(bssn_vars.lapse, dtype=np.float64)
    beta_U = np.asarray(bssn_vars.shift_U, dtype=np.float64) * background.inverse_scaling_vector
    phi    = np.asarray(bssn_vars.phi, dtype=np.float64)
    e4phi  = np.exp(4.0 * phi)
    e6phi  = np.exp(6.0 * phi)

    bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
    gamma_LL     = e4phi[:, None, None] * bar_gamma_LL
    gamma_UU     = np.linalg.inv(gamma_LL)

    K_scalar = np.asarray(bssn_vars.K, dtype=np.float64)
    bar_A_LL = get_bar_A_LL(r, bssn_vars, background)
    K_LL     = e4phi[:, None, None] * bar_A_LL + (K_scalar / 3.0)[:, None, None] * gamma_LL

    dalpha_dx = np.asarray(bssn_d1.lapse)
    dbeta_dx  = (
        background.inverse_scaling_vector[:, :, None] * np.asarray(bssn_d1.shift_U)
        + bssn_vars.shift_U[:, :, None] * background.d1_inverse_scaling_vector
    )
    hat_chris   = background.hat_christoffel
    hatD_beta_U = np.transpose(dbeta_dx, (0, 2, 1)) + np.einsum('xjik,xk->xij', hat_chris, beta_U)

    dphi_dx = np.asarray(bssn_d1.phi)
    hat_D_bar_gamma = get_hat_D_bar_gamma_LL(r, bssn_vars.h_LL, bssn_d1.h_LL, background)
    hatD_gamma_LL = e4phi[:, None, None, None] * (
        4.0 * dphi_dx[:, :, None, None] * bar_gamma_LL[:, None, :, :]
        + np.transpose(hat_D_bar_gamma, (0, 3, 1, 2))
    )

    geom = HydroGeometry(
        alpha=jnp.asarray(alpha),
        beta_r=jnp.asarray(beta_U[:, 0]),
        gamma_rr=jnp.asarray(gamma_LL[:, 0, 0]),
        e6phi=jnp.asarray(e6phi),
        e4phi=jnp.asarray(e4phi),
        beta_U=jnp.asarray(beta_U),
        gamma_LL=jnp.asarray(gamma_LL),
        gamma_UU=jnp.asarray(gamma_UU),
    )
    source_data     = (jnp.asarray(K_LL), jnp.asarray(dalpha_dx),
                       jnp.asarray(hatD_beta_U), jnp.asarray(hatD_gamma_LL))
    connection_data = (jnp.asarray(hat_chris),)
    return geom, source_data, connection_data


# =============================================================================
# Main evolution function
# =============================================================================

def evolve_jax(cfg, initial_state_2d, prim_tuple, tov_solution,
               grid, background, hydro, atmosphere,
               plots_dir, output_dir,
               restart_info=None, t_start=0.0, step_offset=0):
    """JAX-based TOV evolution with JIT-compiled RK4 step.

    Supports Cowling (fixed spacetime) and Dynamic (full BSSN+hydro) modes
    with optional restart from checkpoints.

    Args:
        cfg:              TOVConfig instance.
        initial_state_2d: (NUM_VARS, N) initial state array.
        prim_tuple:       (rho0, vr, p, eps) reference primitives.
        tov_solution:     TOV solution object.
        grid:             Grid instance.
        background:       FlatSphericalBackground instance.
        hydro:            PerfectFluid instance.
        atmosphere:       AtmosphereParams instance.
        plots_dir:        Directory for plot output.
        output_dir:       Directory for HDF5/JSON data output.
        restart_info:     Dict from find_latest_snapshot() or None.
        t_start:          Starting time (non-zero when restarting).
        step_offset:      Starting step count (non-zero when restarting).
    """
    from source.bssn.bssnvars import BSSNVars
    from source.bssn.bssnstatevariables import (
        NUM_BSSN_VARS, idx_lapse, idx_phi, idx_K,
        BSSN_PARITY, BSSN_ASYMP_POWER, BSSN_ASYMP_OFFSET,
    )
    from source.core.spacing import NUM_GHOSTS
    from source.bssn.jax.bssngeometry import build_bssn_background, build_derivative_stencils
    from source.bssn.jax.boundaries_jax import fill_bssn_boundaries_jax
    from source.core.rhsevolution_jax import get_rhs_bssn_hydro_jax, NUM_HYDRO_VARS, HYDRO_PARITY
    from source.matter.hydro.jax.valencia_jax import compute_hydro_rhs_cowling
    from source.bssn.ahfinder import get_horizon_diagnostics
    from examples.TOV.utils.diagnostics import compute_baryon_mass, compute_constraints
    from examples.TOV.utils.data import SimulationDataManager
    from examples.TOV import plotting as _plot

    IS_DYNAMIC  = (cfg.evolution_mode == "dynamic")
    PLOT_SUFFIX = cfg.plot_suffix

    dt = cfg.cfl_factor * grid.min_dr

    if restart_info is not None:
        t_remaining     = cfg.t_final - t_start
        num_steps_total = int(t_remaining / dt)
        print(f"  RESTART MODE: Resuming from t={t_start:.6e}, "
              f"evolving remaining t={t_remaining:.3e} ({num_steps_total} steps)")
    else:
        num_steps_total = int(cfg.t_final / dt)

    N        = grid.N
    num_vars = grid.NUM_VARS

    if IS_DYNAMIC:
        print(f"\n{'='*70}")
        print("EVOLUTION MODE: DYNAMIC (Full BSSN + Hydro) — JAX")
        print("  - Spacetime evolves with matter")
        print("  - 1+log slicing for lapse")
        print("  - Gamma-driver shift")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print("EVOLUTION MODE: COWLING (Fixed Spacetime) — JAX")
        print("  - BSSN variables frozen at t=0")
        print("  - Only hydro evolves")
        print(f"{'='*70}")

    print(f"\ndt={dt:.6f} (CFL={cfg.cfl_factor}), t_final={cfg.t_final}, steps={num_steps_total}")
    print(f"JAX devices: {jax.devices()}")

    # ------------------------------------------------------------------
    # Data manager
    # ------------------------------------------------------------------
    restart_mode = (restart_info is not None)
    data_manager = SimulationDataManager(
        output_dir, grid, hydro,
        enable_saving=cfg.enable_data_saving,
        suffix=PLOT_SUFFIX,
        restart_mode=restart_mode,
    )

    if cfg.enable_data_saving:
        data_manager.save_metadata(
            tov_solution, atmosphere, dt, "rk4",
            K=cfg.K, Gamma=cfg.Gamma, rho_central=cfg.rho_central,
            r_max=cfg.r_max, num_points=cfg.num_points, t_final=cfg.t_final,
            reconstructor=cfg.reconstructor, solver_method=cfg.solver_method,
            riemann_solver=cfg.riemann_solver, evolution_mode=cfg.evolution_mode,
            cfl_factor=cfg.cfl_factor,
        )

    # Reference primitives
    rho_ref, vr_ref, p_ref, eps_ref = prim_tuple
    W_ref       = np.ones(N)
    h_ref       = 1.0 + eps_ref + p_ref / (rho_ref + 1e-30)
    success_ref = np.ones(N, dtype=bool)

    if cfg.enable_data_saving and restart_info is None:
        Ham_0, Mom_0 = compute_constraints(initial_state_2d, grid, background, hydro)
        data_manager.save_snapshot(0, 0.0, initial_state_2d, rho_ref, vr_ref, p_ref,
                                   eps_ref, W_ref, h_ref, Ham=Ham_0, Mom=Mom_0)
        data_manager.add_evolution_point(0, 0.0, initial_state_2d,
                                         rho_ref, vr_ref, p_ref, eps_ref, W_ref, h_ref, success_ref,
                                         rho_ref, vr_ref, p_ref, eps_ref, W_ref, h_ref, success_ref,
                                         Ham=Ham_0, Mom=Mom_0)

    Mb0 = compute_baryon_mass(grid, initial_state_2d, rho_ref, vr_ref, p_ref, eps_ref, W_ref, h_ref)
    if restart_info is None:
        all_times = [0.0];  all_Mb = [Mb0]
        all_rho_c = [float(rho_ref[NUM_GHOSTS])];  all_v_c = [float(vr_ref[NUM_GHOSTS])]
    else:
        all_times = [];  all_Mb = [];  all_rho_c = [];  all_v_c = []

    # ------------------------------------------------------------------
    # Atmosphere constants
    # ------------------------------------------------------------------
    eos_type   = 'ideal_gas'
    _rho_floor = float(atmosphere.rho_floor)
    _p_floor   = float(atmosphere.p_floor)
    _v_max     = float(atmosphere.v_max)
    _gm1       = cfg.Gamma - 1.0
    _eps_atm   = cfg.K * _rho_floor**_gm1 / _gm1
    _p_atm     = cfg.K * _rho_floor**cfg.Gamma
    _h_atm     = 1.0 + _eps_atm + _p_atm / _rho_floor
    _atm_threshold_factor = 100.0 * _rho_floor

    # ------------------------------------------------------------------
    # Mode-dependent setup
    # ------------------------------------------------------------------
    if IS_DYNAMIC:
        print("\nBuilding BSSNBackground and DerivativeStencils...")
        t0_setup       = time.perf_counter()
        state_jax      = jnp.array(initial_state_2d)
        bssn_bg        = build_bssn_background(grid, background)
        deriv_stencils = build_derivative_stencils(grid)
        dr_jax         = jnp.array(grid.dr)
        dx_hydro       = float(grid.derivs.dx)

        eta = 1.0;  sigma_base = 1.0;  max_iter = 200;  tol = 1e-12

        parity_jax = jnp.array(np.concatenate([
            np.array(BSSN_PARITY, dtype=np.float64),
            np.array(HYDRO_PARITY, dtype=np.float64),
        ]))
        asymp_power_jax = jnp.array(np.concatenate([
            np.array(BSSN_ASYMP_POWER, dtype=np.float64),
            np.zeros(NUM_HYDRO_VARS),
        ]))
        asymp_offset_jax = jnp.array(np.concatenate([
            np.array(BSSN_ASYMP_OFFSET, dtype=np.float64),
            np.zeros(NUM_HYDRO_VARS),
        ]))

        _tau_atm_phys = _rho_floor * _h_atm - _p_atm - _rho_floor
        r_ghost       = grid.r[-NUM_GHOSTS:]
        M_star        = tov_solution.M_star
        boundary_ref  = np.zeros((num_vars, NUM_GHOSTS))
        for i, rg in enumerate(r_ghost):
            factor    = 1.0 + M_star / (2.0 * rg)
            boundary_ref[idx_phi,          i] = np.log(factor)
            boundary_ref[idx_lapse,        i] = (1.0 - M_star / (2.0 * rg)) / factor
            boundary_ref[NUM_BSSN_VARS,    i] = factor**6 * _rho_floor
            boundary_ref[NUM_BSSN_VARS + 2, i] = factor**6 * _tau_atm_phys
        boundary_ref_jax = jnp.array(boundary_ref)
        print(f"  Setup in {time.perf_counter() - t0_setup:.3f}s")

        @jax.jit
        def apply_bcs(state):
            return fill_bssn_boundaries_jax(
                state, bssn_bg.r, NUM_GHOSTS,
                parity_jax, asymp_power_jax, asymp_offset_jax,
                "dirichlet", boundary_ref=boundary_ref_jax,
            )

        @jax.jit
        def rhs_fn(state):
            return get_rhs_bssn_hydro_jax(
                state, bssn_bg, deriv_stencils, dr_jax,
                NUM_GHOSTS, num_vars,
                sigma_base, eta,
                eos_type, cfg.Gamma, cfg.K,
                _rho_floor, _p_floor, _v_max, 10.0,
                cfg.reconstructor, cfg.solver_method, max_iter,
                tol, dx_hydro,
                "zero_gradient",
                fix_shift=False,
            )

        @jax.jit
        def jit_step_dynamic(state, dt_val):
            k1 = rhs_fn(state)
            k2 = rhs_fn(apply_bcs(state + 0.5 * dt_val * k1))
            k3 = rhs_fn(apply_bcs(state + 0.5 * dt_val * k2))
            k4 = rhs_fn(apply_bcs(state + dt_val * k3))
            state_new = apply_bcs(state + (dt_val / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4))

            phi   = state_new[idx_phi]
            e6phi = jnp.exp(6.0 * phi)
            D     = state_new[NUM_BSSN_VARS]
            Sr    = state_new[NUM_BSSN_VARS + 1]
            tau_v = state_new[NUM_BSSN_VARS + 2]

            D_atm   = e6phi * _rho_floor
            tau_atm = e6phi * (_rho_floor * _h_atm - _p_atm - _rho_floor)
            mask    = D < (_atm_threshold_factor * e6phi)
            D     = jnp.where(mask, D_atm, D)
            Sr    = jnp.where(mask, 0.0, Sr)
            tau_v = jnp.where(mask, tau_atm, tau_v)

            state_new = state_new.at[NUM_BSSN_VARS].set(D)
            state_new = state_new.at[NUM_BSSN_VARS + 1].set(Sr)
            state_new = state_new.at[NUM_BSSN_VARS + 2].set(tau_v)
            return state_new

    else:
        print("\nBuilding HydroGeometry from BSSN variables...")
        t0_geom = time.perf_counter()
        geom, source_data, connection_data = build_cowling_geometry(initial_state_2d, grid, background)
        dx_val = float(grid.derivs.dx)
        print(f"  Geometry built in {time.perf_counter() - t0_geom:.3f}s")

        D0   = jnp.asarray(initial_state_2d[NUM_BSSN_VARS + 0, :])
        Sr0  = jnp.asarray(initial_state_2d[NUM_BSSN_VARS + 1, :])
        tau0 = jnp.asarray(initial_state_2d[NUM_BSSN_VARS + 2, :])

        eos_params = {'gamma': cfg.Gamma, 'K': cfg.K}
        atm_params = {'rho_floor': _rho_floor, 'p_floor': _p_floor, 'v_max': _v_max,
                      'W_max': 10.0, 'tol': 1e-12, 'max_iter': 200}

        @jax.jit
        def jit_step_cowling(D, Sr, tau, geom_arg):
            def _rhs(D_in, Sr_in, tau_in):
                Db, Sb, tb = fill_boundaries_jax(D_in, Sr_in, tau_in, NUM_GHOSTS)
                return compute_hydro_rhs_cowling(
                    Db, Sb, tb, geom_arg, dx_val, NUM_GHOSTS,
                    eos_type, eos_params, atm_params,
                    cfg.reconstructor, cfg.solver_method,
                    source_data=source_data, connection_data=connection_data,
                )

            dD1, dS1, dt1 = _rhs(D, Sr, tau)
            D2 = D + 0.5*dt*dD1;  S2 = Sr + 0.5*dt*dS1;  t2 = tau + 0.5*dt*dt1
            dD2, dS2, dt2 = _rhs(D2, S2, t2)
            D3 = D + 0.5*dt*dD2;  S3 = Sr + 0.5*dt*dS2;  t3 = tau + 0.5*dt*dt2
            dD3, dS3, dt3 = _rhs(D3, S3, t3)
            D4 = D + dt*dD3;      S4 = Sr + dt*dS3;       t4 = tau + dt*dt3
            dD4, dS4, dt4 = _rhs(D4, S4, t4)

            D_n = D   + (dt/6.0) * (dD1 + 2*dD2 + 2*dD3 + dD4)
            S_n = Sr  + (dt/6.0) * (dS1 + 2*dS2 + 2*dS3 + dS4)
            t_n = tau + (dt/6.0) * (dt1 + 2*dt2 + 2*dt3 + dt4)

            D_atm   = geom_arg.e6phi * _rho_floor
            tau_atm = geom_arg.e6phi * (_rho_floor * _h_atm - _p_atm - _rho_floor)
            mask    = D_n < (_atm_threshold_factor * geom_arg.e6phi)
            D_n = jnp.where(mask, D_atm, D_n)
            S_n = jnp.where(mask, 0.0,   S_n)
            t_n = jnp.where(mask, tau_atm, t_n)
            return D_n, S_n, t_n

    # ------------------------------------------------------------------
    # Print initial state
    # ------------------------------------------------------------------
    print(f"  Initial D_c   = {float(initial_state_2d[NUM_BSSN_VARS, NUM_GHOSTS]):.6e}")
    print(f"  Initial Sr_c  = {float(initial_state_2d[NUM_BSSN_VARS+1, NUM_GHOSTS]):.6e}")
    print(f"  Initial tau_c = {float(initial_state_2d[NUM_BSSN_VARS+2, NUM_GHOSTS]):.6e}")

    # ------------------------------------------------------------------
    # JIT warmup + benchmark
    # ------------------------------------------------------------------
    print("\nJIT compiling full step (RK4 + BC + RHS + atm)...")
    t0_jit = time.perf_counter()

    if IS_DYNAMIC:
        dt_jax     = jnp.float64(dt)
        state_test = jit_step_dynamic(state_jax, dt_jax).block_until_ready()
        jit_time   = time.perf_counter() - t0_jit
        print(f"  Full step JIT compilation: {jit_time:.2f}s")
        print("Benchmarking step execution...")
        t0_bench = time.perf_counter()
        for _ in range(10):
            state_test = jit_step_dynamic(state_test, dt_jax)
        state_test.block_until_ready()
        bench_time = (time.perf_counter() - t0_bench) / 10 * 1000
    else:
        D_test, Sr_test, tau_test = jit_step_cowling(D0, Sr0, tau0, geom)
        jax.block_until_ready((D_test, Sr_test, tau_test))
        jit_time = time.perf_counter() - t0_jit
        print(f"  Full step JIT compilation: {jit_time:.2f}s")
        print("Benchmarking step execution...")
        t0_bench = time.perf_counter()
        for _ in range(20):
            D_test, Sr_test, tau_test = jit_step_cowling(D_test, Sr_test, tau_test, geom)
        jax.block_until_ready((D_test, Sr_test, tau_test))
        bench_time = (time.perf_counter() - t0_bench) / 20 * 1000

    print(f"  Single step execution: {bench_time:.2f} ms")
    step_time_ms = bench_time

    # ------------------------------------------------------------------
    # Checkpoint setup
    # ------------------------------------------------------------------
    if restart_info is None:
        checkpoint_steps = {
            1: max(1, num_steps_total // 3),
            2: max(2, 2 * num_steps_total // 3),
            3: num_steps_total,
        }
    else:
        checkpoint_steps = {3: num_steps_total}

    print(f"\n{'='*70}")
    print("Evolution checkpoints:")
    for cp_id, cp_step in checkpoint_steps.items():
        print(f"  step {cp_step + step_offset:6d}: {cp_id}/3 (t~{(cp_step*dt)+t_start:.3e})")
    print(f"{'='*70}\n")

    checkpoint_states_dict = {}
    checkpoint_times       = {}
    bssn_fixed = initial_state_2d[:NUM_BSSN_VARS, :].copy()

    def extract_primitives(state_2d):
        bv = BSSNVars(N)
        bv.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
        hydro.set_matter_vars(state_2d, bv, grid)
        return hydro._get_primitives(bv, grid.r)

    if IS_DYNAMIC:
        def get_state_2d():
            return np.array(state)
        def get_checkpoint_state():
            return np.array(state)
    else:
        def get_state_2d():
            s = np.zeros((NUM_BSSN_VARS + 3, N))
            s[:NUM_BSSN_VARS, :] = bssn_fixed
            s[NUM_BSSN_VARS + 0, :] = np.asarray(D)
            s[NUM_BSSN_VARS + 1, :] = np.asarray(Sr)
            s[NUM_BSSN_VARS + 2, :] = np.asarray(tau)
            return s
        def get_checkpoint_state():
            return (np.array(D), np.array(Sr), np.array(tau))

    # ------------------------------------------------------------------
    # Evolution loop
    # ------------------------------------------------------------------
    t0_evol = time.perf_counter()
    t       = 0.0
    PRINT_INTERVAL    = 500
    NAN_CHECK_INTERVAL = 500

    if IS_DYNAMIC:
        state = state_jax
    else:
        D, Sr, tau = D0, Sr0, tau0

    ah_times_list  = []
    ah_radius_list = []
    bh_mass_list   = []

    print("===== Evolution diagnostics (per step) =====")
    if IS_DYNAMIC:
        print("Columns: step | t | rho_central | max_drho/rho@r | max_vr@r | alpha_c | K_c | dMb/Mb | c2p_fails")
    else:
        print("Columns: step | t | rho_central | max_drho/rho@r | max_vr@r | max_Sr@r | c2p_fails")
    print("-" * 120)

    interior       = slice(NUM_GHOSTS, -NUM_GHOSTS)
    r_interior     = grid.r[interior]
    checkpoint_set = set(checkpoint_steps.values())
    state_after_first_step = None

    for step in range(1, num_steps_total + 1):
        if IS_DYNAMIC:
            state = jit_step_dynamic(state, dt_jax)
        else:
            D, Sr, tau = jit_step_cowling(D, Sr, tau, geom)
        t += dt

        if step == 1:
            state_after_first_step = get_state_2d()

        if step % NAN_CHECK_INTERVAL == 0:
            if IS_DYNAMIC:
                state.block_until_ready()
                if bool(jnp.any(jnp.isnan(state[idx_phi]))):
                    print(f"\nERROR: NaN at step {step}, t={t:.6e}. Halting.")
                    break
            else:
                jax.block_until_ready(D)
                if bool(jnp.any(jnp.isnan(D))):
                    print(f"\nERROR: NaN at step {step}, t={t:.6e}. Halting.")
                    break

        should_print     = (step % PRINT_INTERVAL == 0)
        should_save_evol = cfg.enable_data_saving and cfg.evolution_interval and (step % cfg.evolution_interval == 0)
        should_save_snap = cfg.enable_data_saving and cfg.snapshot_interval  and (step % cfg.snapshot_interval  == 0)

        if should_print or should_save_evol or should_save_snap:
            state_2d = get_state_2d()
            rho0, vr, p, eps, W, h, success = extract_primitives(state_2d)

            Ham, Mom = None, None
            if should_save_evol or should_save_snap:
                Ham, Mom = compute_constraints(state_2d, grid, background, hydro)

            if should_save_evol:
                data_manager.add_evolution_point(
                    step + step_offset, t + t_start, state_2d,
                    rho0, vr, p, eps, W, h, success,
                    rho_ref, vr_ref, p_ref, eps_ref, W_ref, h_ref, success_ref,
                    Ham=Ham, Mom=Mom,
                )
                Mb = compute_baryon_mass(grid, state_2d, rho0, vr, p, eps, W, h)
                all_times.append(t + t_start);  all_Mb.append(Mb)
                all_rho_c.append(float(rho0[NUM_GHOSTS]));  all_v_c.append(float(vr[NUM_GHOSTS]))
                if step % (cfg.evolution_interval * 10) == 0:
                    data_manager.flush_evolution_buffer()

            if should_save_snap:
                data_manager.save_snapshot(step + step_offset, t + t_start,
                                           state_2d, rho0, vr, p, eps, W, h, Ham=Ham, Mom=Mom)

            if should_print:
                rho_c_curr   = float(rho0[NUM_GHOSTS])
                rho_int      = rho0[interior]
                rho_init_int = rho_ref[interior]
                matter_mask  = rho_init_int > 100.0 * atmosphere.rho_floor

                if np.any(matter_mask):
                    rel_err         = np.abs(rho_int[matter_mask] - rho_init_int[matter_mask]) / np.abs(rho_init_int[matter_mask])
                    idx_m           = np.argmax(rel_err)
                    max_rel_rho_err = float(rel_err[idx_m])
                    r_max_rho       = float(r_interior[matter_mask][idx_m])
                else:
                    max_rel_rho_err = 0.0;  r_max_rho = 0.0

                vr_int    = vr[interior]
                idx_mv    = np.argmax(np.abs(vr_int))
                max_abs_v = float(vr_int[idx_mv])
                r_max_v   = float(r_interior[idx_mv])

                c2p_fail = int(np.sum(~success))
                if c2p_fail > 0:
                    c2p_info = f"c2p_fail={c2p_fail}@r={np.array2string(grid.r[~success], precision=2, separator=',')}"
                else:
                    c2p_info = "c2p_fail=0"

                if IS_DYNAMIC:
                    lapse_c  = float(state_2d[idx_lapse, NUM_GHOSTS])
                    K_c      = float(state_2d[idx_K, NUM_GHOSTS])
                    dMb      = abs(all_Mb[-1] - Mb0) / Mb0 if len(all_Mb) > 1 else 0.0
                    elapsed  = time.perf_counter() - t0_evol
                    t_actual = t + t_start
                    frac     = (t_actual - t_start) / (cfg.t_final - t_start) if restart_info else t / cfg.t_final
                    print(f"step {step + step_offset:6d}  t={t_actual:.1e}  rho_c={rho_c_curr:.6e}  "
                          f"max_drho/rho={max_rel_rho_err:.2e}@r={r_max_rho:.2f}  "
                          f"max_vr={max_abs_v:.1e}@r={r_max_v:.1f}  "
                          f"alpha_c={lapse_c:.6f}  K_c={K_c:.6e}  dMb/Mb={dMb:.2e}  "
                          f"{c2p_info}  [{frac*100:.0f}% {elapsed:.0f}s]")

                    if (lapse_c < 0.05 or cfg.collapse_perturbation) and step % PRINT_INTERVAL == 0:
                        try:
                            state_flat = state_2d.reshape(-1)
                            t_now = t + t_start
                            _, ah_r_now, bh_m_now = get_horizon_diagnostics(
                                state_flat, np.array([t_now]), grid, background, hydro)
                            ah_times_list.append(t_now)
                            ah_radius_list.append(float(ah_r_now[0]))
                            bh_mass_list.append(float(bh_m_now[0]))
                            if ah_r_now[0] > 0:
                                print(f"    -> AH: r_AH={ah_r_now[0]:.4f}, M_BH={bh_m_now[0]:.4f}")
                        except Exception:
                            pass
                else:
                    Sr_np      = np.asarray(state_2d[NUM_BSSN_VARS + 1, interior])
                    idx_mSr    = np.argmax(np.abs(Sr_np))
                    max_Sr     = float(np.abs(Sr_np[idx_mSr]))
                    r_max_Sr   = float(r_interior[idx_mSr])
                    t_actual   = t + t_start
                    print(f"step {step + step_offset:5d}  t={t_actual:.2e}:  rho_c={rho_c_curr:.6e}  "
                          f"max_drho/rho={max_rel_rho_err:.2e}@r={r_max_rho:.2f}  "
                          f"max_vr={max_abs_v:.3e}@r={r_max_v:.2f}  "
                          f"max_Sr={max_Sr:.2e}@r={r_max_Sr:.2f}  {c2p_info}")

        if step in checkpoint_set:
            if IS_DYNAMIC:
                state.block_until_ready()
            else:
                jax.block_until_ready((D, Sr, tau))
            cp_id = [k for k, v in checkpoint_steps.items() if v == step][0]
            checkpoint_states_dict[cp_id] = get_checkpoint_state()
            checkpoint_times[cp_id]       = t + t_start
            data_manager.flush_evolution_buffer()
            print(f"  -> Reached checkpoint {cp_id}, t={t + t_start:.6e}")

    evol_time = time.perf_counter() - t0_evol
    data_manager.flush_evolution_buffer()

    for cp_id in [1, 2, 3]:
        if cp_id not in checkpoint_states_dict:
            if IS_DYNAMIC:
                state.block_until_ready()
            else:
                jax.block_until_ready((D, Sr, tau))
            checkpoint_states_dict[cp_id] = get_checkpoint_state()
            checkpoint_times[cp_id]       = t + t_start

    print(f"\nEvolution complete! {evol_time:.2f}s  ({num_steps_total/evol_time:.0f} steps/s)")

    # ------------------------------------------------------------------
    # Save time series
    # ------------------------------------------------------------------
    times_full = np.array(all_times);  rho_c_full = np.array(all_rho_c)
    v_c_full   = np.array(all_v_c);   Mb_full    = np.array(all_Mb)

    if cfg.enable_data_saving and cfg.save_timeseries and len(times_full) > 0:
        ts_path = os.path.join(output_dir, "timeseries.npz")
        np.savez(ts_path, times=times_full, rho_central=rho_c_full, v_central=v_c_full,
                 Mb=Mb_full, num_points=cfg.num_points, K=cfg.K, Gamma=cfg.Gamma,
                 rho_central_initial=cfg.rho_central, r_max=cfg.r_max, dt=dt,
                 num_steps=num_steps_total)
        print(f"\nTime series saved to: {ts_path}")

    # ------------------------------------------------------------------
    # Reconstruct full states
    # ------------------------------------------------------------------
    if IS_DYNAMIC:
        state_cp0 = initial_state_2d
        state_cp1 = checkpoint_states_dict[1]
        state_cp2 = checkpoint_states_dict[2]
        state_cp3 = checkpoint_states_dict[3]
    else:
        def _build_full(tup):
            s = np.zeros((NUM_BSSN_VARS + 3, N))
            s[:NUM_BSSN_VARS, :] = bssn_fixed
            s[NUM_BSSN_VARS:, :]  = np.array(tup).reshape(3, N)
            return s
        state_cp0 = np.zeros((NUM_BSSN_VARS + 3, N))
        state_cp0[:NUM_BSSN_VARS, :] = bssn_fixed
        state_cp0[NUM_BSSN_VARS:, :]  = initial_state_2d[NUM_BSSN_VARS:, :]
        state_cp1 = _build_full(checkpoint_states_dict[1])
        state_cp2 = _build_full(checkpoint_states_dict[2])
        state_cp3 = _build_full(checkpoint_states_dict[3])

    print("\nRecovering primitives at all checkpoints...")
    rho0_0,   vr_0,   p_0,   eps_0,   W_0,   h_0,   success_0   = extract_primitives(state_cp0)
    rho0_cp1, vr_cp1, p_cp1, eps_cp1, W_cp1, h_cp1, success_cp1 = extract_primitives(state_cp1)
    rho0_cp2, vr_cp2, p_cp2, eps_cp2, W_cp2, h_cp2, success_cp2 = extract_primitives(state_cp2)
    rho0_cp3, vr_cp3, p_cp3, eps_cp3, W_cp3, h_cp3, success_cp3 = extract_primitives(state_cp3)

    t_cp1 = checkpoint_times[1];  t_cp2 = checkpoint_times[2];  t_cp3 = checkpoint_times[3]

    delta_rho_cp1 = np.abs(rho0_cp1[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)
    delta_rho_cp2 = np.abs(rho0_cp2[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)
    delta_rho_cp3 = np.abs(rho0_cp3[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)
    delta_P_cp1   = np.abs(p_cp1[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)
    delta_P_cp2   = np.abs(p_cp2[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)
    delta_P_cp3   = np.abs(p_cp3[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)

    Mb_cp0 = compute_baryon_mass(grid, state_cp0, rho0_0,   vr_0,   p_0,   eps_0,   W_0,   h_0)
    Mb_cp1 = compute_baryon_mass(grid, state_cp1, rho0_cp1, vr_cp1, p_cp1, eps_cp1, W_cp1, h_cp1)
    Mb_cp2 = compute_baryon_mass(grid, state_cp2, rho0_cp2, vr_cp2, p_cp2, eps_cp2, W_cp2, h_cp2)
    Mb_cp3 = compute_baryon_mass(grid, state_cp3, rho0_cp3, vr_cp3, p_cp3, eps_cp3, W_cp3, h_cp3)

    # AH diagnostic
    ah_times_diag  = np.array(ah_times_list)  if ah_times_list  else None
    ah_radius_diag = np.array(ah_radius_list) if ah_radius_list else None
    bh_mass_diag   = np.array(bh_mass_list)   if bh_mass_list   else None

    if IS_DYNAMIC and ah_times_diag is not None and len(ah_times_diag) > 0:
        print(f"\n{'='*70}\nAPPARENT HORIZON DIAGNOSTIC\n{'='*70}")
        ah_mask   = ah_radius_diag > 0
        n_with_ah = int(np.sum(ah_mask))
        print(f"  Checked {len(ah_times_diag)} snapshots, AH found in {n_with_ah}")
        if n_with_ah > 0:
            fi = np.argmax(ah_mask)
            print(f"  ** AH first at t={ah_times_diag[fi]:.4e}, r={ah_radius_diag[fi]:.4f}, M={bh_mass_diag[fi]:.4f} **")

    # Evolution diagnostics
    print(f"\n{'='*70}\nEVOLUTION DIAGNOSTICS\n{'='*70}")
    print(f"\n1. VELOCITY EVOLUTION:")
    print(f"   Max |v^r| t=0:     {np.max(np.abs(vr_0)):.3e}")
    print(f"   Max |v^r| t=1/3:   {np.max(np.abs(vr_cp1)):.3e}")
    print(f"   Max |v^r| t=2/3:   {np.max(np.abs(vr_cp2)):.3e}")
    print(f"   Max |v^r| t=final: {np.max(np.abs(vr_cp3)):.3e}")

    print(f"\n2. CENTRAL DENSITY:")
    print(f"   rho_c(t=0)     = {rho0_0[NUM_GHOSTS]:.6e}")
    print(f"   rho_c(t=1/3)   = {rho0_cp1[NUM_GHOSTS]:.6e}  delta={abs(rho0_cp1[NUM_GHOSTS]-rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")
    print(f"   rho_c(t=2/3)   = {rho0_cp2[NUM_GHOSTS]:.6e}  delta={abs(rho0_cp2[NUM_GHOSTS]-rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")
    print(f"   rho_c(t=final) = {rho0_cp3[NUM_GHOSTS]:.6e}  delta={abs(rho0_cp3[NUM_GHOSTS]-rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")

    print(f"\n3. DENSITY ERROR (max): 1/3={np.max(delta_rho_cp1):.3e}  2/3={np.max(delta_rho_cp2):.3e}  final={np.max(delta_rho_cp3):.3e}")
    print(f"4. PRESSURE ERROR (max): 1/3={np.max(delta_P_cp1):.3e}  2/3={np.max(delta_P_cp2):.3e}  final={np.max(delta_P_cp3):.3e}")

    print(f"\n5. CONS2PRIM: t=0={np.sum(success_0)}/{N}  1/3={np.sum(success_cp1)}/{N}  2/3={np.sum(success_cp2)}/{N}  final={np.sum(success_cp3)}/{N}")
    if not np.all(success_cp3):
        failed_idx = np.where(~success_cp3)[0]
        print(f"   Failed points: {failed_idx[:5]}")

    print(f"\n6. BARYON MASS CONSERVATION:")
    print(f"   Mb(t=0)     = {Mb_cp0:.6f}")
    print(f"   Mb(t=1/3)   = {Mb_cp1:.6f}   dMb/Mb={abs(Mb_cp1 - Mb_cp0)/Mb_cp0:.3e}")
    print(f"   Mb(t=2/3)   = {Mb_cp2:.6f}   dMb/Mb={abs(Mb_cp2 - Mb_cp0)/Mb_cp0:.3e}")
    print(f"   Mb(t=final) = {Mb_cp3:.6f}   dMb/Mb={abs(Mb_cp3 - Mb_cp0)/Mb_cp0:.3e}")

    # Mode-specific
    if IS_DYNAMIC:
        print(f"\n{'='*70}\nCONSTRAINT VIOLATION DIAGNOSTICS\n{'='*70}")
        try:
            Ham_0c, Mom_0c = compute_constraints(state_cp0, grid, background, hydro)
            Ham_fc, Mom_fc = compute_constraints(state_cp3, grid, background, hydro)
            max_H_0 = np.max(np.abs(Ham_0c[interior]));  max_H_f = np.max(np.abs(Ham_fc[interior]))
            max_M_0 = np.max(np.abs(Mom_0c[interior, 0]));  max_M_f = np.max(np.abs(Mom_fc[interior, 0]))
            print(f"  Ham: t=0={max_H_0:.3e}  t=f={max_H_f:.3e}" +
                  (f"  growth={max_H_f/max_H_0:.2f}x" if max_H_0 > 1e-20 else ""))
            print(f"  Mom: t=0={max_M_0:.3e}  t=f={max_M_f:.3e}" +
                  (f"  growth={max_M_f/max_M_0:.2f}x" if max_M_0 > 1e-20 else ""))
        except Exception as e:
            print(f"  Constraint computation failed: {e}")
        print(f"  alpha_c: t=0={state_cp0[idx_lapse, NUM_GHOSTS]:.6f}  t=f={state_cp3[idx_lapse, NUM_GHOSTS]:.6f}")
        print(f"{'='*70}")
    else:
        bssn_change = np.max(np.abs(state_cp3[:NUM_BSSN_VARS] - initial_state_2d[:NUM_BSSN_VARS]) /
                             (np.abs(initial_state_2d[:NUM_BSSN_VARS]) + 1e-20))
        print(f"\nCOWLING CHECK: Max BSSN change = {bssn_change:.3e}" +
              ("  (OK)" if bssn_change < 1e-14 else "  WARNING: non-zero!"))

    n_nan = np.sum(np.isnan(np.asarray(state_cp3[NUM_BSSN_VARS, :])))
    print(f"\nNaN check: {'WARNING: ' + str(n_nan) + ' NaNs' if n_nan else 'No NaNs.'}")

    # ------------------------------------------------------------------
    # Finalize + plots
    # ------------------------------------------------------------------
    execution_time = time.perf_counter() - t0_evol
    if cfg.enable_data_saving:
        data_manager.finalize(execution_time_seconds=execution_time)

    R_star   = tov_solution.R_iso
    states   = [state_cp0, state_cp1, state_cp2, state_cp3]
    times_cp = [0.0, checkpoint_times[1], checkpoint_times[2], checkpoint_times[3]]

    if not cfg.skip_plots:
        print("\nGenerating plots...")
        _safe_plot(lambda: _plot.plot_tov_diagnostics(tov_solution, cfg.r_max, suffix=PLOT_SUFFIX),
                   "plot_tov_diagnostics")
        _safe_plot(lambda: _plot.plot_tov_vs_initial_data_zoom(
                       tov_solution, initial_state_2d, grid,
                       primitives=prim_tuple, window=0.5, suffix=PLOT_SUFFIX),
                   "plot_tov_vs_initial_data_zoom")
        _safe_plot(lambda: _plot.plot_evolution(
                       states, times_cp, grid, hydro, rho_ref, p_ref,
                       Mb_series=Mb_full, rho_c_series=rho_c_full,
                       times_series=times_full, suffix=PLOT_SUFFIX, R_star=R_star),
                   "plot_evolution")
        _safe_plot(lambda: _plot.plot_bssn_evolution(
                       state_cp0, state_cp3, grid,
                       t_0=0.0, t_final=checkpoint_times[3], suffix=PLOT_SUFFIX),
                   "plot_bssn_evolution")

        if state_after_first_step is not None:
            _safe_plot(lambda: _plot.plot_first_step(
                           state_cp0, state_after_first_step, grid, hydro,
                           tov_solution=tov_solution, suffix=PLOT_SUFFIX),
                       "plot_first_step")
            _safe_plot(lambda: _plot.plot_center_zoom(
                           state_cp0, state_after_first_step, grid, hydro,
                           window=0.5, suffix=PLOT_SUFFIX),
                       "plot_center_zoom")
            _safe_plot(lambda: _plot.plot_surface_zoom(
                           tov_solution, state_cp0, state_after_first_step,
                           grid, hydro, primitives_t0=None, window=0.5, suffix=PLOT_SUFFIX),
                       "plot_surface_zoom")

        if IS_DYNAMIC and cfg.enable_data_saving:
            _safe_plot(lambda: _plot.plot_constraints_evolution(output_dir, suffix=PLOT_SUFFIX),
                       "plot_constraints_evolution")

        # AH custom plot
        if IS_DYNAMIC and ah_radius_diag is not None and np.any(ah_radius_diag > 0):
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                mask = ah_radius_diag > 0
                ax1.plot(ah_times_diag[mask], ah_radius_diag[mask], 'bo-', label='AH radius')
                ax1.set_ylabel('AH radius'); ax1.legend(); ax1.grid(True, alpha=0.3)
                ax1.set_title('Apparent Horizon Diagnostics')
                ax2.plot(ah_times_diag[mask], bh_mass_diag[mask], 'ro-', label='BH mass')
                if tov_solution is not None:
                    ax2.axhline(tov_solution.M_star, color='gray', linestyle='--',
                                label=f'M_TOV={tov_solution.M_star:.4f}')
                ax2.set_xlabel('t'); ax2.set_ylabel('BH mass'); ax2.legend(); ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                ah_path = os.path.join(plots_dir, f'tov_ah_diagnostic{PLOT_SUFFIX}.png')
                plt.savefig(ah_path, dpi=150); plt.close()
                print(f"    Saved: {ah_path}")
            except Exception as e:
                print(f"  AH plot failed: {e}")

    # Timing summary
    print(f"\n{'='*70}\nTIMING SUMMARY\n{'='*70}")
    print(f"  JIT compilation:    {jit_time:.2f}s")
    print(f"  Benchmark step:     {step_time_ms:.2f} ms")
    print(f"  Pure evolution:     {evol_time:.2f}s ({num_steps_total} steps)")
    print(f"  Total (incl. diag): {execution_time:.2f}s")
    print(f"  Steps/second:       {num_steps_total / evol_time:.0f}")
    print(f"{'='*70}")
    print(f"BENCHMARK_RESULT: jit_s={jit_time:.3f} per_step_ms={step_time_ms:.3f} "
          f"total_s={evol_time:.3f} n_steps={num_steps_total}")
    print(f"\nData saved to: {output_dir}")


def _safe_plot(fn, name):
    """Call fn(), printing a warning on failure."""
    try:
        fn()
    except Exception as e:
        print(f"  {name} failed: {e}")
