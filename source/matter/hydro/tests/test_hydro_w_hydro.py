#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hydro-without-Hydro Test with JAX Backend (Dynamic Mode)

Evolves BSSN with static matter sources from TOV solution.
The matter (hydro) variables are held fixed at the TOV equilibrium values,
while BSSN variables evolve. This tests that the spacetime remains stable
when sourced by a static perfect fluid.

Uses JAX backend with dynamic mode (full BSSN + hydro coupling) and
boundary conditions from TOVEvolution.py.
"""

import os
import sys
import numpy as np
import time
from pathlib import Path
from functools import partial

os.environ['ENGRENAGE_BACKEND'] = 'jax'

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground

from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import (
    NUM_BSSN_VARS,
    BSSN_PARITY, BSSN_ASYMP_POWER, BSSN_ASYMP_OFFSET,
    idx_phi, idx_hrr, idx_htt, idx_hpp,
    idx_K, idx_arr, idx_att, idx_app,
    idx_lambdar, idx_shiftr, idx_br, idx_lapse
)

from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.atmosphere import AtmosphereParams

from source.bssn.jax.bssngeometry import build_bssn_background, build_derivative_stencils
from source.bssn.jax.boundaries_jax import fill_bssn_boundaries_jax
from source.core.rhsevolution_jax import get_rhs_bssn_hydro_jax, NUM_HYDRO_VARS, HYDRO_PARITY

from examples.TOV.tov_solver import load_or_solve_tov_iso
import examples.TOV.tov_initial_data_interpolated as tov_id


def locate_repo_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (cand / 'source').is_dir():
            return cand
    return start

THIS = Path(__file__).resolve()
REPO = locate_repo_root(THIS.parent)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def get_rhs_hwh_jax(state, bssn_bg, deriv_stencils, dr_jax,
                     NUM_GHOSTS, num_vars,
                     sigma_base, eta,
                     eos_type, Gamma, K,
                     rho_floor, p_floor, v_max, W_max,
                     recon_name, solver_method, max_iter, tol, dx_hydro,
                     fix_shift):
    """
    RHS for Hydro-without-Hydro in JAX dynamic mode.

    BSSN variables evolve normally, but hydro variables are reset to TOV
    equilibrium values after each RHS computation.
    """
    rhs = get_rhs_bssn_hydro_jax(
        state, bssn_bg, deriv_stencils, dr_jax,
        NUM_GHOSTS, num_vars,
        sigma_base, eta,
        eos_type, Gamma, K,
        rho_floor, p_floor, v_max, W_max,
        recon_name, solver_method, max_iter, tol, dx_hydro,
        "zero_gradient",
        fix_shift=fix_shift
    )
    return rhs


def run_hwh_jax_test(
    K=100.0,
    Gamma=2.0,
    rho_central=1.28e-3,
    r_max=16.0,
    num_points=300,
    t_final=50.0,
    cfl=0.1,
    progress=True,
    save_interval=50,
    RECONSTRUCTOR_NAME="wenoz",
    SOLVER_METHOD="newton"
):
    """
    Run Hydro-without-Hydro test with JAX backend (dynamic mode).
    """
    print("="*70)
    print("HYDRO-WITHOUT-HYDRO TEST - JAX DYNAMIC MODE")
    print("="*70)
    print(f"  K = {K}, Gamma = {Gamma}")
    print(f"  rho_central = {rho_central:.2e}")
    print(f"  Grid: N = {num_points}, r_max = {r_max}")
    print(f"  Evolution: t_final = {t_final}, CFL = {cfl}")
    print(f"  Reconstructor: {RECONSTRUCTOR_NAME}")
    print(f"  Solver: {SOLVER_METHOD}")
    print("="*70)

    spacing = LinearSpacing(num_points, r_max)
    eos = IdealGasEOS(gamma=Gamma)

    rho_floor_base = 1e-16
    p_floor_base = K * (rho_floor_base)**Gamma
    atmosphere = AtmosphereParams(
        rho_floor=rho_floor_base,
        p_floor=p_floor_base,
        v_max=0.999,
        W_max=100.0
    )

    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere=atmosphere,
        reconstructor=create_reconstruction(RECONSTRUCTOR_NAME),
        riemann_solver=HLLRiemannSolver(atmosphere=atmosphere),
        solver_method=SOLVER_METHOD
    )

    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    print(f"\nGrid created: N = {grid.N}, dr_min = {grid.min_dr:.4f}")
    print(f"JAX devices: {jax.devices()}")

    print("\nSolving TOV equations...")
    tov_solution = load_or_solve_tov_iso(
        K=K, Gamma=Gamma, rho_central=rho_central,
        r_max=r_max, accuracy="high"
    )
    print(f"  M_star = {tov_solution.M_star:.6f}")
    print(f"  R_iso = {tov_solution.R_iso:.3f}")
    print(f"  C = {tov_solution.C:.4f}")

    print("\nCreating initial data...")
    initial_state, prim_tuple = tov_id.create_initial_data_iso(
        tov_solution, grid, background, eos,
        atmosphere=atmosphere,
        polytrope_K=K, polytrope_Gamma=Gamma,
        interp_order=11
    )

    N = grid.N
    num_vars = grid.NUM_VARS

    dt = cfl * grid.min_dr
    num_steps = int(t_final / dt)
    print(f"\nTimestep: dt = {dt:.6f}")
    print(f"Total steps: {num_steps}")

    print("\nBuilding BSSNBackground and DerivativeStencils...")
    t0_setup = time.perf_counter()
    state_jax = jnp.array(initial_state)
    bssn_bg = build_bssn_background(grid, background)
    deriv_stencils = build_derivative_stencils(grid)
    dr_jax = jnp.array(grid.dr)
    dx_hydro = float(grid.derivs.dx)

    eta = 1.0
    sigma_base = 1.0
    max_iter = 200
    tol = 1e-12

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

    _rho_floor = float(atmosphere.rho_floor)
    _p_floor = float(atmosphere.p_floor)
    _v_max = float(atmosphere.v_max)
    _gm1 = Gamma - 1.0
    _eps_atm = K * _rho_floor**_gm1 / _gm1
    _p_atm = K * _rho_floor**Gamma
    _h_atm = 1.0 + _eps_atm + _p_atm / _rho_floor

    _tau_atm_phys = _rho_floor * _h_atm - _p_atm - _rho_floor
    r_ghost = grid.r[-NUM_GHOSTS:]
    M_star = tov_solution.M_star
    boundary_ref = np.zeros((num_vars, NUM_GHOSTS))
    for i, rg in enumerate(r_ghost):
        factor = 1.0 + M_star / (2.0 * rg)
        phi_val = np.log(factor)
        alpha_val = (1.0 - M_star / (2.0 * rg)) / factor
        e6phi_val = factor**6
        boundary_ref[idx_phi, i] = phi_val
        boundary_ref[idx_lapse, i] = alpha_val
        boundary_ref[NUM_BSSN_VARS, i] = e6phi_val * _rho_floor
        boundary_ref[NUM_BSSN_VARS + 2, i] = e6phi_val * _tau_atm_phys
    boundary_ref_jax = jnp.array(boundary_ref)

    print(f"  Setup in {time.perf_counter() - t0_setup:.3f}s")

    @jax.jit
    def apply_bcs(state):
        return fill_bssn_boundaries_jax(
            state, bssn_bg.r, NUM_GHOSTS,
            parity_jax, asymp_power_jax, asymp_offset_jax,
            "dirichlet", boundary_ref=boundary_ref_jax
        )



    tov_r_jax = jnp.array(tov_solution.r_iso)
    tov_rho_jax = jnp.array(tov_solution.rho_baryon)
    tov_p_jax = jnp.array(tov_solution.P)
    r_jax = jnp.array(grid.r)

    eos_type = 'ideal_gas'
    max_iter = 200
    tol = 1e-12
    dx_hydro = float(grid.derivs.dx)
    W_max = 10.0

    @jax.jit
    def rhs_fn(state):
        return get_rhs_bssn_hydro_jax(
            state, bssn_bg, deriv_stencils, dr_jax,
            NUM_GHOSTS, num_vars,
            sigma_base, eta,
            eos_type, Gamma, K,
            _rho_floor, _p_floor, _v_max, W_max,
            RECONSTRUCTOR_NAME, SOLVER_METHOD, max_iter,
            tol, dx_hydro,
            "zero_gradient",
            fix_shift=False
        )

    @jax.jit
    def jit_step_hwh(state, dt_val):
        """
        RK4 step for Hydro-without-Hydro.

        Hydro variables are RESET BEFORE computing RHS to maintain
        fixed TOV equilibrium sources for BSSN evolution.
        This is the correct behavior: BSSN evolves with static matter sources.
        """
        def rhs_with_reset(state_in):
            state_bssn_only = reset_hydro_to_tov(state_in)
            rhs = rhs_fn(state_bssn_only)
            rhs = rhs.at[NUM_BSSN_VARS:].set(0.0)
            return rhs

        k1 = rhs_with_reset(state)

        state_2 = apply_bcs(state + 0.5 * dt_val * k1)
        k2 = rhs_with_reset(state_2)

        state_3 = apply_bcs(state + 0.5 * dt_val * k2)
        k3 = rhs_with_reset(state_3)

        state_4 = apply_bcs(state + dt_val * k3)
        k4 = rhs_with_reset(state_4)

        state_new = apply_bcs(state + (dt_val / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4))

        state_final = reset_hydro_to_tov(state_new)

        return state_final

    @jax.jit
    def reset_hydro_to_tov(state):
        """
        Reset hydro variables to static TOV equilibrium values.
        
        Uses FIXED phi from the current state (dynamic geometry),
        but with TOV fluid variables (rho, p, v=0).
        This maintains the correct stress-energy tensor for the evolving spacetime.
        """
        phi = state[idx_phi]
        e6phi = jnp.exp(6.0 * phi)

        rho0 = jnp.interp(r_jax, tov_r_jax, tov_rho_jax, left=_rho_floor, right=_rho_floor)
        pressure = jnp.interp(r_jax, tov_r_jax, tov_p_jax, left=_p_floor, right=_p_floor)

        gm1 = Gamma - 1.0
        eps = K * rho0**gm1 / gm1
        h = 1.0 + eps + pressure / jnp.maximum(rho0, 1e-300)
        W = 1.0
        D = rho0 * W
        Sr = jnp.zeros_like(D)
        tau = rho0 * h * W**2 - pressure - D

        D_atm = e6phi * _rho_floor
        tau_atm = e6phi * (_rho_floor * h - _p_floor - _rho_floor)

        D = jnp.where(D < D_atm, D_atm, D)
        tau = jnp.where(tau < tau_atm, tau_atm, tau)

        state = state.at[NUM_BSSN_VARS].set(D)
        state = state.at[NUM_BSSN_VARS + 1].set(Sr)
        state = state.at[NUM_BSSN_VARS + 2].set(tau)

        return state

    dt_jax = jnp.float64(dt)

    print("\nJIT compiling step function...")
    t0_jit = time.perf_counter()
    state_test = jit_step_hwh(state_jax, dt_jax)
    state_test.block_until_ready()
    jit_time = time.perf_counter() - t0_jit
    print(f"  JIT compilation: {jit_time:.2f}s")

    print("Benchmarking step execution...")
    t0_bench = time.perf_counter()
    n_bench = 10
    for _ in range(n_bench):
        state_test = jit_step_hwh(state_test, dt_jax)
    state_test.block_until_ready()
    bench_time = (time.perf_counter() - t0_bench) / n_bench * 1000
    print(f"  Single step: {bench_time:.2f} ms")

    center = NUM_GHOSTS

    times = [0.0]
    lapse_c = [float(initial_state[idx_lapse, center])]
    phi_c = [float(initial_state[idx_phi, center])]
    K_c = [float(initial_state[idx_K, center])]
    hrr_c = [float(initial_state[idx_hrr, center])]

    times_detailed = [0.0]
    states_detailed = [initial_state.copy()]

    print("\n" + "="*70)
    print("Starting HWH evolution (JAX dynamic mode)...")
    print("="*70)

    state = state_jax
    t = 0.0

    PRINT_INTERVAL = 100

    for step in range(1, num_steps + 1):
        state = jit_step_hwh(state, dt_jax)
        t += dt

        state_np = np.array(state)

        if (step + 1) % 10 == 0:
            times.append(t)
            lapse_c.append(state_np[idx_lapse, center])
            phi_c.append(state_np[idx_phi, center])
            K_c.append(state_np[idx_K, center])
            hrr_c.append(state_np[idx_hrr, center])

        if (step + 1) % save_interval == 0:
            times_detailed.append(t)
            states_detailed.append(state_np.copy())

        if progress and (step + 1) % PRINT_INTERVAL == 0:
            print(f"  step {step+1:5d}  t = {t:.3f}  "
                  f"alpha_c = {state_np[idx_lapse, center]:.6f}  "
                  f"phi_c = {state_np[idx_phi, center]:+.3e}  "
                  f"K_c = {state_np[idx_K, center]:+.3e}")

        if step % 500 == 0:
            state.block_until_ready()
            if bool(jnp.any(jnp.isnan(state[idx_phi]))):
                print(f"\nERROR: NaN detected at step {step}, t={t:.6e}. Halting evolution.")
                break

    final_state = np.array(state)

    print("\n" + "="*70)
    print("HWH EVOLUTION COMPLETE (JAX)")
    print("="*70)
    print(f"  Final time: t = {t:.3f}")
    print(f"  Steps: {num_steps}")
    print(f"\n  Central values:")
    print(f"    alpha: {lapse_c[0]:.6f} -> {lapse_c[-1]:.6f}  (delta = {lapse_c[-1]-lapse_c[0]:+.3e})")
    print(f"    phi:   {phi_c[0]:+.3e} -> {phi_c[-1]:+.3e}  (delta = {phi_c[-1]-phi_c[0]:+.3e})")
    print(f"    K:     {K_c[0]:+.3e} -> {K_c[-1]:+.3e}  (delta = {K_c[-1]-K_c[0]:+.3e})")
    print("="*70)

    return dict(
        time=t,
        steps=num_steps,
        r=grid.r,
        state=final_state,
        initial_state=initial_state,
        times=np.array(times),
        lapse_center=np.array(lapse_c),
        phi_center=np.array(phi_c),
        K_center=np.array(K_c),
        hrr_center=np.array(hrr_c),
        times_detailed=np.array(times_detailed),
        states_detailed=states_detailed,
        tov=tov_solution,
        K=K,
        Gamma=Gamma,
        rho_central=rho_central
    )


if __name__ == "__main__":
    result = run_hwh_jax_test(
        K=100.0,
        Gamma=2.0,
        rho_central=0.2,
        r_max=8.0,
        num_points=300,
        t_final=100.0,
        cfl=0.1,
        save_interval=50,
        progress=True,
        RECONSTRUCTOR_NAME="wenoz",
        SOLVER_METHOD="newton"
    )
