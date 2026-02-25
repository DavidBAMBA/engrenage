"""
Numba-based evolution engine for TOV star.

Contains RHS functions and the evolution loop for the CPU backend.
Supports both Cowling (fixed spacetime) and Dynamic (full BSSN+hydro) modes.
"""

import numpy as np
import os
import time


# =============================================================================
# RHS functions
# =============================================================================

def get_rhs_cowling(t, y, grid, background, hydro, bssn_fixed, bssn_d1_fixed):
    """RHS for Cowling evolution (fixed spacetime), hydro variables only.

    BSSN variables are frozen; only the hydro block of the RHS is non-zero.
    """
    from source.bssn.bssnvars import BSSNVars
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS

    state = y.reshape((grid.NUM_VARS, grid.N))
    grid.fill_boundaries(state)

    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(bssn_fixed)
    hydro.set_matter_vars(state, bssn_vars, grid)

    hydro_rhs = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1_fixed, background)

    rhs = np.zeros_like(state)
    rhs[NUM_BSSN_VARS:, :] = hydro_rhs
    return rhs.flatten()


class _DummyProgressBar:
    """Dummy progress bar that does nothing (used as RK4 substep placeholder)."""
    def update(self, n):
        pass


def get_rhs_dynamic(t, y, grid, background, hydro, progress_bar=None, time_state=None):
    """RHS for full BSSN + hydro evolution (no Cowling approximation)."""
    from source.core.rhsevolution import get_rhs

    if progress_bar is None:
        progress_bar = _DummyProgressBar()
    if time_state is None:
        time_state = [0.0, 1.0]  # [last_t, dt] — dummy values

    return get_rhs(t, y, grid, background, hydro, progress_bar, time_state)


def _apply_atmosphere_reset(state_2d, grid, hydro, atmosphere, rho_threshold=None):
    """Apply atmospheric floors to densitized conservative variables.

    Resets to atmosphere if:
      1. D < threshold (density floor)
      2. tau + D < 0  (unphysical, would fail cons2prim)
    """
    from source.bssn.bssnvars import BSSNVars
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS

    rho_hard_floor = rho_threshold or 100.0 * atmosphere.rho_floor
    idx_D  = NUM_BSSN_VARS
    idx_Sr = NUM_BSSN_VARS + 1
    idx_tau = NUM_BSSN_VARS + 2

    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
    e6phi = np.exp(6.0 * bssn_vars.phi)

    D   = state_2d[idx_D, :]
    tau = state_2d[idx_tau, :]

    atm_mask = (D < rho_hard_floor * e6phi) | (tau + D < 0)
    if np.any(atm_mask):
        state_2d[idx_D,  atm_mask] = atmosphere.rho_floor * e6phi[atm_mask]
        state_2d[idx_Sr, atm_mask] = 0.0
        state_2d[idx_tau, atm_mask] = atmosphere.tau_atm * e6phi[atm_mask]

    return state_2d


def rk4_step(state_flat, dt, grid, background, hydro, bssn_fixed, bssn_d1_fixed, atmosphere):
    """Single RK4 timestep for Cowling mode with atmosphere reset after the full step."""
    k1 = get_rhs_cowling(0, state_flat, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    state_2 = (state_flat + 0.5 * dt * k1).reshape((grid.NUM_VARS, grid.N)).flatten()
    k2 = get_rhs_cowling(0, state_2, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    state_3 = (state_flat + 0.5 * dt * k2).reshape((grid.NUM_VARS, grid.N)).flatten()
    k3 = get_rhs_cowling(0, state_3, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    state_4 = (state_flat + dt * k3).reshape((grid.NUM_VARS, grid.N)).flatten()
    k4 = get_rhs_cowling(0, state_4, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    state_new = (state_flat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)).reshape((grid.NUM_VARS, grid.N))
    return _apply_atmosphere_reset(state_new, grid, hydro, atmosphere).flatten()


def rk4_step_dynamic(state_flat, dt, grid, background, hydro, atmosphere,
                     progress_bar=None, time_state=None):
    """Single RK4 timestep for Dynamic mode with atmosphere reset after the full step."""
    state = state_flat.reshape((grid.NUM_VARS, grid.N))

    k1 = get_rhs_dynamic(0, state_flat, grid, background, hydro, progress_bar, time_state)
    state_1 = state + 0.5 * dt * k1.reshape((grid.NUM_VARS, grid.N))

    k2 = get_rhs_dynamic(0, state_1.flatten(), grid, background, hydro, progress_bar, time_state)
    state_2 = state + 0.5 * dt * k2.reshape((grid.NUM_VARS, grid.N))

    k3 = get_rhs_dynamic(0, state_2.flatten(), grid, background, hydro, progress_bar, time_state)
    state_3 = state + dt * k3.reshape((grid.NUM_VARS, grid.N))

    k4 = get_rhs_dynamic(0, state_3.flatten(), grid, background, hydro, progress_bar, time_state)

    state_new = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4).reshape((grid.NUM_VARS, grid.N))
    return _apply_atmosphere_reset(state_new, grid, hydro, atmosphere).flatten()


# =============================================================================
# Main evolution function
# =============================================================================

def evolve_numba(cfg, initial_state_2d, prim_tuple, tov_solution,
                 grid, background, hydro, atmosphere,
                 plots_dir, output_dir,
                 restart_info=None, t_start=0.0, step_offset=0):
    """Numba-based TOV evolution with fixed-timestep RK4.

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
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS, idx_lapse, idx_phi
    from source.core.spacing import NUM_GHOSTS
    from source.bssn.constraintsdiagnostic import get_constraints_diagnostic
    from examples.TOV.utils.diagnostics import (
        compute_baryon_mass, compute_constraints, evolve_fixed_timestep,
    )
    from examples.TOV.utils.data import SimulationDataManager
    from examples.TOV import plotting as _plot

    IS_DYNAMIC   = (cfg.evolution_mode == "dynamic")
    PLOT_SUFFIX  = cfg.plot_suffix

    # ------------------------------------------------------------------
    # Setup depends on evolution mode
    # ------------------------------------------------------------------
    if IS_DYNAMIC:
        print("\n" + "=" * 70)
        print("EVOLUTION MODE: DYNAMIC (Full BSSN + Hydro) — Numba")
        print("  - Spacetime evolves with matter")
        print("  - 1+log slicing for lapse")
        print("  - Gamma-driver shift")
        print("=" * 70)

        bssn_fixed    = initial_state_2d[:NUM_BSSN_VARS, :].copy()
        bssn_d1_fixed = grid.get_d1_metric_quantities(initial_state_2d)

        def rk4_step_wrapper(state_flat, dt, grid, background, hydro,
                             bssn_fixed_unused, bssn_d1_fixed_unused, atmosphere):
            return rk4_step_dynamic(state_flat, dt, grid, background, hydro, atmosphere)

        selected_rk4_step = rk4_step_wrapper

    else:  # cowling
        print("\n" + "=" * 70)
        print("EVOLUTION MODE: COWLING (Fixed Spacetime) — Numba")
        print("  - BSSN variables frozen at t=0")
        print("  - Only hydro evolves")
        print("=" * 70)

        bssn_fixed    = initial_state_2d[:NUM_BSSN_VARS, :].copy()
        bssn_d1_fixed = grid.get_d1_metric_quantities(initial_state_2d)
        selected_rk4_step = rk4_step

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

    if cfg.integration_method == 'fixed':
        dt               = cfg.cfl_factor * grid.min_dr
        num_steps_total  = int(cfg.t_final / dt)
        time_remaining   = cfg.t_final - t_start
        num_steps_remaining = int(time_remaining / dt)

        if restart_info is not None:
            print(f"\nRestart mode:")
            print(f"  Current time: {t_start:.6e}")
            print(f"  Target time:  {cfg.t_final:.6e}")
            print(f"  Remaining:    {time_remaining:.6e} ({num_steps_remaining} steps)")
            print(f"  dt={dt:.6e} (CFL={cfg.cfl_factor})")
        else:
            print(f"\nEvolving with fixed dt={dt:.6f} (CFL={cfg.cfl_factor}) to "
                  f"t_final={cfg.t_final} ({num_steps_total} steps) using RK4")

        evolution_start_time = time.time()

        if cfg.enable_data_saving:
            tov_sol_to_save = tov_solution
            if restart_info is not None:
                from examples.TOV.utils.io import load_metadata
                metadata = load_metadata(output_dir, suffix=PLOT_SUFFIX)
                tov_sol_to_save = tov_solution if tov_solution is not None else metadata.get('tov_solution', {})
            data_manager.save_metadata(
                tov_sol_to_save, atmosphere, dt, cfg.integration_method,
                K=cfg.K, Gamma=cfg.Gamma, rho_central=cfg.rho_central,
                r_max=cfg.r_max, num_points=cfg.num_points, t_final=cfg.t_final,
                reconstructor=cfg.reconstructor, solver_method=cfg.solver_method,
                riemann_solver=cfg.riemann_solver, evolution_mode=cfg.evolution_mode,
                cfl_factor=cfg.cfl_factor,
            )

        # First-step diagnostics (skip for restart)
        state_t1 = None
        t_1 = None
        if restart_info is None:
            state_t1 = selected_rk4_step(
                initial_state_2d.flatten(), dt, grid, background, hydro,
                bssn_fixed, bssn_d1_fixed, atmosphere,
            ).reshape((grid.NUM_VARS, grid.N))
            t_1 = dt

        # Checkpoint structure
        checkpoint_states = {}
        checkpoint_times  = {}
        all_series = []

        if restart_info is None:
            checkpoint_1 = max(1, num_steps_total // 3)
            checkpoint_2 = max(2, 2 * num_steps_total // 3)
            checkpoint_3 = num_steps_total

            print(f"\n{'='*70}")
            print(f"Evolution checkpoints (for plotting):")
            print(f"  t=0:         initial state")
            print(f"  step {checkpoint_1:6d}:  1/3 (~{checkpoint_1*dt:.3e})")
            print(f"  step {checkpoint_2:6d}:  2/3 (~{checkpoint_2*dt:.3e})")
            print(f"  step {checkpoint_3:6d}:  final (~{checkpoint_3*dt:.3e})")
            print(f"{'='*70}\n")
        else:
            checkpoint_1 = step_offset + max(1, num_steps_remaining // 3)
            checkpoint_2 = step_offset + max(2, 2 * num_steps_remaining // 3)
            checkpoint_3 = num_steps_total

            print(f"\n{'='*70}")
            print(f"Evolution (restart mode):")
            print(f"  Starting step: {step_offset}")
            print(f"  Starting time: {t_start:.6e}")
            print(f"  Target step:   {checkpoint_3}")
            print(f"  Target time:   {cfg.t_final:.6e}")
            print(f"{'='*70}\n")

        if restart_info is None:
            # Normal mode: 3-checkpoint structure
            print(f"Evolving to checkpoint 1 (step {checkpoint_1})...")
            state_cp1, steps_cp1, t_cp1, series_1 = evolve_fixed_timestep(
                initial_state_2d, dt, checkpoint_1, grid, background,
                hydro, bssn_fixed, bssn_d1_fixed, atmosphere, selected_rk4_step,
                method='rk4', reference_state=initial_state_2d,
                data_manager=data_manager,
                snapshot_interval=cfg.snapshot_interval,
                evolution_interval=cfg.evolution_interval,
            )
            checkpoint_states[1] = state_cp1.copy()
            checkpoint_times[1]  = t_cp1
            all_series.append(series_1)
            print(f"  -> Reached step {steps_cp1}, t={t_cp1:.6e}")

            if steps_cp1 == checkpoint_1:
                remaining = checkpoint_2 - checkpoint_1
                print(f"\nEvolving to checkpoint 2 (step {checkpoint_2})...")
                state_cp2, steps_cp2, t_cp2, series_2 = evolve_fixed_timestep(
                    state_cp1, dt, remaining, grid, background,
                    hydro, bssn_fixed, bssn_d1_fixed, atmosphere, selected_rk4_step,
                    method='rk4', t_start=t_cp1, reference_state=initial_state_2d,
                    step_offset=checkpoint_1,
                    data_manager=data_manager,
                    snapshot_interval=cfg.snapshot_interval,
                    evolution_interval=cfg.evolution_interval,
                )
                checkpoint_states[2] = state_cp2.copy()
                checkpoint_times[2]  = t_cp2
                all_series.append(series_2)
                print(f"  -> Reached step {checkpoint_1 + steps_cp2}, t={t_cp2:.6e}")
            else:
                state_cp2 = state_cp1
                t_cp2 = t_cp1
                checkpoint_states[2] = state_cp2.copy()
                checkpoint_times[2]  = t_cp2

            if steps_cp1 == checkpoint_1 and (checkpoint_1 + steps_cp2) == checkpoint_2:
                remaining = checkpoint_3 - checkpoint_2
                print(f"\nEvolving to checkpoint 3 (step {checkpoint_3}, final)...")
                state_cp3, steps_cp3, t_cp3, series_3 = evolve_fixed_timestep(
                    state_cp2, dt, remaining, grid, background,
                    hydro, bssn_fixed, bssn_d1_fixed, atmosphere, selected_rk4_step,
                    method='rk4', t_start=t_cp2, reference_state=initial_state_2d,
                    step_offset=checkpoint_2,
                    data_manager=data_manager,
                    snapshot_interval=cfg.snapshot_interval,
                    evolution_interval=cfg.evolution_interval,
                )
                checkpoint_states[3] = state_cp3.copy()
                checkpoint_times[3]  = t_cp3
                all_series.append(series_3)
                steps_final = checkpoint_2 + steps_cp3
                print(f"  -> Reached step {steps_final}, t={t_cp3:.6e}")
            else:
                state_cp3 = state_cp2
                t_cp3 = t_cp2
                checkpoint_states[3] = state_cp3.copy()
                checkpoint_times[3]  = t_cp3
                steps_final = checkpoint_1 + steps_cp2

            state_t10000 = checkpoint_states[3]
            t_10000 = checkpoint_times[3]
            num_steps = steps_final

        else:
            # Restart mode: simple direct evolution
            print(f"Evolving from step {step_offset} to step {checkpoint_3}...")
            state_final, steps_done, t_final_actual, series_restart = evolve_fixed_timestep(
                initial_state_2d, dt, num_steps_remaining, grid, background,
                hydro, bssn_fixed, bssn_d1_fixed, atmosphere, selected_rk4_step,
                method='rk4', t_start=t_start, reference_state=initial_state_2d,
                step_offset=step_offset,
                data_manager=data_manager,
                snapshot_interval=cfg.snapshot_interval,
                evolution_interval=cfg.evolution_interval,
            )
            checkpoint_states[1] = state_final.copy()
            checkpoint_times[1]  = t_final_actual
            checkpoint_states[2] = state_final.copy()
            checkpoint_times[2]  = t_final_actual
            checkpoint_states[3] = state_final.copy()
            checkpoint_times[3]  = t_final_actual
            all_series.append(series_restart)

            state_t10000 = state_final
            t_10000 = t_final_actual
            num_steps = steps_done
            steps_final = steps_done

            print(f"  -> Reached step {step_offset + steps_done}, t={t_final_actual:.6e}")

        # ------------------------------------------------------------------
        # Build full time series arrays
        # ------------------------------------------------------------------
        try:
            if len(all_series) == 3:
                times_full = np.concatenate([all_series[0]['t'], all_series[1]['t'][1:], all_series[2]['t'][1:]])
                Mb_full    = np.concatenate([all_series[0]['Mb'], all_series[1]['Mb'][1:], all_series[2]['Mb'][1:]])
                rho_c_full = np.concatenate([all_series[0]['rho_c'], all_series[1]['rho_c'][1:], all_series[2]['rho_c'][1:]])
                v_c_full   = np.concatenate([all_series[0]['v_c'], all_series[1]['v_c'][1:], all_series[2]['v_c'][1:]])
            elif len(all_series) == 2:
                times_full = np.concatenate([all_series[0]['t'], all_series[1]['t'][1:]])
                Mb_full    = np.concatenate([all_series[0]['Mb'], all_series[1]['Mb'][1:]])
                rho_c_full = np.concatenate([all_series[0]['rho_c'], all_series[1]['rho_c'][1:]])
                v_c_full   = np.concatenate([all_series[0]['v_c'], all_series[1]['v_c'][1:]])
            elif len(all_series) == 1:
                times_full = all_series[0]['t']
                Mb_full    = all_series[0]['Mb']
                rho_c_full = all_series[0]['rho_c']
                v_c_full   = all_series[0]['v_c']
            else:
                times_full = Mb_full = rho_c_full = v_c_full = np.array([])
        except Exception as e:
            print(f"Warning: Failed to concatenate series: {e}")
            times_full = Mb_full = rho_c_full = v_c_full = np.array([])

        # ------------------------------------------------------------------
        # Save time series
        # ------------------------------------------------------------------
        if cfg.save_timeseries and len(times_full) > 0:
            timeseries_path = os.path.join(output_dir, "timeseries.npz")
            os.makedirs(output_dir, exist_ok=True)
            np.savez(timeseries_path,
                     times=times_full,
                     rho_central=rho_c_full,
                     v_central=v_c_full,
                     Mb=Mb_full,
                     num_points=cfg.num_points,
                     K=cfg.K,
                     Gamma=cfg.Gamma,
                     rho_central_initial=cfg.rho_central,
                     r_max=cfg.r_max,
                     dt=dt,
                     num_steps=num_steps_total)
            print(f"\nTime series saved to: {timeseries_path}")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    evol_time = time.time() - evolution_start_time

    bssn_ref = BSSNVars(grid.N)
    bssn_ref.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(initial_state_2d, bssn_ref, grid)
    rho_ref, _, p_ref, _, _, _, _ = hydro._get_primitives(bssn_ref, grid.r)

    states = [initial_state_2d,
              checkpoint_states[1], checkpoint_states[2], checkpoint_states[3]]
    times  = [0.0, checkpoint_times[1], checkpoint_times[2], checkpoint_times[3]]
    R_star = tov_solution.R_iso

    if not cfg.skip_plots:
        try:
            if 'times_full' in locals() and len(times_full) > 0:
                _plot.plot_evolution(states, times, grid, hydro, rho_ref, p_ref,
                                     Mb_series=Mb_full, rho_c_series=rho_c_full,
                                     times_series=times_full,
                                     suffix=PLOT_SUFFIX, R_star=R_star)
            else:
                _plot.plot_evolution(states, times, grid, hydro, rho_ref, p_ref,
                                     suffix=PLOT_SUFFIX, R_star=R_star)
        except Exception as e:
            print(f"  plot_evolution failed: {e}")

        try:
            _plot.plot_bssn_evolution(initial_state_2d, checkpoint_states[3], grid,
                                      t_0=0.0, t_final=checkpoint_times[3],
                                      suffix=PLOT_SUFFIX)
        except Exception as e:
            print(f"  plot_bssn_evolution failed: {e}")

    # Constraint diagnostics (Dynamic mode only)
    if cfg.evolution_mode == "dynamic":
        print("\n" + "=" * 70)
        print("CONSTRAINT VIOLATION DIAGNOSTICS")
        print("=" * 70)

        interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
        Ham_0, Mom_0 = get_constraints_diagnostic(
            initial_state_2d.flatten(), 0.0, grid, background, hydro)
        Ham_f, Mom_f = get_constraints_diagnostic(
            checkpoint_states[3].flatten(), checkpoint_times[3], grid, background, hydro)

        max_H_0 = np.max(np.abs(Ham_0[0, interior]))
        max_M_0 = np.max(np.abs(Mom_0[0, interior, 0]))
        max_H_f = np.max(np.abs(Ham_f[0, interior]))
        max_M_f = np.max(np.abs(Mom_f[0, interior, 0]))

        print(f"Hamiltonian constraint |H|:")
        print(f"  t=0:     max|H| = {max_H_0:.3e}")
        print(f"  t=final: max|H| = {max_H_f:.3e}")
        if max_H_0 > 1e-20:
            print(f"  Growth factor: {max_H_f/max_H_0:.2f}x")
        print(f"\nMomentum constraint |M_r|:")
        print(f"  t=0:     max|M_r| = {max_M_0:.3e}")
        print(f"  t=final: max|M_r| = {max_M_f:.3e}")
        if max_M_0 > 1e-20:
            print(f"  Growth factor: {max_M_f/max_M_0:.2f}x")
        print("=" * 70)

    # Detailed statistics at each checkpoint
    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
    t_cp1 = checkpoint_times[1]
    t_cp2 = checkpoint_times[2]
    t_cp3 = checkpoint_times[3]

    def _get_prims(state_2d):
        bv = BSSNVars(grid.N)
        bv.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
        hydro.set_matter_vars(state_2d, bv, grid)
        return hydro._get_primitives(bv, grid.r)

    rho0_0, vr_0, p_0, _, _, _, success_0 = _get_prims(initial_state_2d)
    rho0_cp1, vr_cp1, p_cp1, _, _, _, success_cp1 = _get_prims(checkpoint_states[1])
    rho0_cp2, vr_cp2, p_cp2, _, _, _, success_cp2 = _get_prims(checkpoint_states[2])
    rho0_cp3, vr_cp3, p_cp3, _, _, _, success_cp3 = _get_prims(checkpoint_states[3])

    if state_t1 is not None:
        rho0_1, vr_1, p_1, _, _, _, success_1 = _get_prims(state_t1)
        delta_rho_1 = np.abs(rho0_1[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)
        delta_P_1   = np.abs(p_1[interior]    - p_0[interior])    / (np.abs(p_0[interior])    + 1e-20)
        max_err_rho_1 = np.max(delta_rho_1)
        max_err_P_1   = np.max(delta_P_1)

    delta_rho_cp1 = np.abs(rho0_cp1[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)
    delta_rho_cp2 = np.abs(rho0_cp2[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)
    delta_rho_cp3 = np.abs(rho0_cp3[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)

    delta_P_cp1 = np.abs(p_cp1[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)
    delta_P_cp2 = np.abs(p_cp2[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)
    delta_P_cp3 = np.abs(p_cp3[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)

    max_err_rho_cp1 = np.max(delta_rho_cp1)
    max_err_rho_cp2 = np.max(delta_rho_cp2)
    max_err_rho_cp3 = np.max(delta_rho_cp3)
    max_err_P_cp1   = np.max(delta_P_cp1)
    max_err_P_cp2   = np.max(delta_P_cp2)
    max_err_P_cp3   = np.max(delta_P_cp3)

    if state_t1 is not None:
        growth_rho = max_err_rho_cp3 / max_err_rho_1 if max_err_rho_1 > 1e-15 else 0.0
        growth_P   = max_err_P_cp3   / max_err_P_1   if max_err_P_1   > 1e-15 else 0.0

    print(f"\n{'='*70}")
    print(f"EVOLUTION DIAGNOSTICS")
    print(f"  t=0 -> t={t_cp1:.6e} (1/3) -> t={t_cp2:.6e} (2/3) -> t={t_cp3:.6e} (final)")
    if state_t1 is not None:
        print(f"  (first step: t={t_1:.6e}, included for diagnostics)")
    print(f"{'='*70}")

    print(f"\n1. VELOCITY EVOLUTION:")
    print(f"   Max |v^r| at t=0:              {np.max(np.abs(vr_0)):.3e}")
    if state_t1 is not None:
        print(f"   Max |v^r| at t={t_1:.6e}:    {np.max(np.abs(vr_1)):.3e}")
    print(f"   Max |v^r| at t={t_cp1:.6e} (1/3):  {np.max(np.abs(vr_cp1)):.3e}")
    print(f"   Max |v^r| at t={t_cp2:.6e} (2/3):  {np.max(np.abs(vr_cp2)):.3e}")
    print(f"   Max |v^r| at t={t_cp3:.6e} (final): {np.max(np.abs(vr_cp3)):.3e}")

    print(f"\n2. CENTRAL DENSITY:")
    print(f"   rho_c at t=0:                  {rho0_0[NUM_GHOSTS]:.6e}")
    if state_t1 is not None:
        print(f"   rho_c at t={t_1:.6e}:    {rho0_1[NUM_GHOSTS]:.6e}")
    print(f"   rho_c at t={t_cp1:.6e} (1/3):  {rho0_cp1[NUM_GHOSTS]:.6e}")
    print(f"   rho_c at t={t_cp2:.6e} (2/3):  {rho0_cp2[NUM_GHOSTS]:.6e}")
    print(f"   rho_c at t={t_cp3:.6e} (final): {rho0_cp3[NUM_GHOSTS]:.6e}")
    if state_t1 is not None:
        print(f"   delta_rho_c/rho_c (first step):  {abs(rho0_1[NUM_GHOSTS]  - rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")
    print(f"   delta_rho_c/rho_c (1/3):        {abs(rho0_cp1[NUM_GHOSTS] - rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")
    print(f"   delta_rho_c/rho_c (2/3):        {abs(rho0_cp2[NUM_GHOSTS] - rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")
    print(f"   delta_rho_c/rho_c (final):      {abs(rho0_cp3[NUM_GHOSTS] - rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")

    print(f"\n3. DENSITY ERROR (max over domain):")
    if state_t1 is not None:
        print(f"   Max |d_rho|/rho at t={t_1:.6e}:     {max_err_rho_1:.3e}")
    print(f"   Max |d_rho|/rho at t={t_cp1:.6e} (1/3):   {max_err_rho_cp1:.3e}")
    print(f"   Max |d_rho|/rho at t={t_cp2:.6e} (2/3):   {max_err_rho_cp2:.3e}")
    print(f"   Max |d_rho|/rho at t={t_cp3:.6e} (final):  {max_err_rho_cp3:.3e}")
    if state_t1 is not None:
        print(f"   Growth factor (final/first): {growth_rho:.1f}x")

    print(f"\n4. PRESSURE ERROR (max over domain):")
    if state_t1 is not None:
        print(f"   Max |dP|/P at t={t_1:.6e}:     {max_err_P_1:.3e}")
    print(f"   Max |dP|/P at t={t_cp1:.6e} (1/3):   {max_err_P_cp1:.3e}")
    print(f"   Max |dP|/P at t={t_cp2:.6e} (2/3):   {max_err_P_cp2:.3e}")
    print(f"   Max |dP|/P at t={t_cp3:.6e} (final):  {max_err_P_cp3:.3e}")
    if state_t1 is not None:
        print(f"   Growth factor (final/first): {growth_P:.1f}x")

    print(f"\n5. CONS2PRIM STATUS:")
    print(f"   Success at t=0:                {np.sum(success_0)}/{grid.N}")
    if state_t1 is not None:
        print(f"   Success at t={t_1:.6e}:  {np.sum(success_1)}/{grid.N}")
    print(f"   Success at t={t_cp1:.6e} (1/3):    {np.sum(success_cp1)}/{grid.N}")
    print(f"   Success at t={t_cp2:.6e} (2/3):    {np.sum(success_cp2)}/{grid.N}")
    print(f"   Success at t={t_cp3:.6e} (final):  {np.sum(success_cp3)}/{grid.N}")
    if not np.all(success_cp3):
        failed_idx = np.where(~success_cp3)[0]
        print(f"   Failed points: {failed_idx[:5]} (first 5)")
        print(f"   Failed radii:  {grid.r[failed_idx[:5]]}")

    # Finalize data saving
    if cfg.enable_data_saving:
        data_manager.finalize(execution_time_seconds=evol_time)

    # Constraint plots (Dynamic mode only)
    if cfg.evolution_mode == "dynamic" and cfg.enable_data_saving and not cfg.skip_plots:
        print("\n" + "=" * 70)
        print("Plotting constraint violation evolution...")
        print("=" * 70)
        _plot.plot_constraints_evolution(output_dir, suffix=PLOT_SUFFIX)

    # Summary
    print("\n" + "=" * 70)
    print("Evolution complete. Plots saved:")
    print(f"  Wall time: {evol_time:.2f}s")
    print("=" * 70)
    _per_step_ms = evol_time / max(num_steps_total, 1) * 1000
    print(f"BENCHMARK_RESULT: jit_s=0.000 per_step_ms={_per_step_ms:.3f} "
          f"total_s={evol_time:.3f} n_steps={num_steps_total}")
