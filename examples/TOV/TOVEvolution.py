import numpy as np
import sys
import os
import time

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Engrenage core - add parent directories to path
sys.path.insert(0, os.path.join(script_dir, '..', '..'))
from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground

# BSSN
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS
from source.bssn.tensoralgebra import get_bar_gamma_LL
from source.bssn.constraintsdiagnostic import get_constraints_diagnostic

# Full BSSN+Hydro RHS (for dynamic mode)
from source.core.rhsevolution import get_rhs

# Hydro
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import PolytropicEOS, IdealGasEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver, LLFRiemannSolver
from source.matter.hydro.cons2prim import prim_to_cons
from source.matter.hydro.atmosphere import AtmosphereParams
from source.matter.hydro.geometry import GeometryState

# Local TOV modules (isotropic coordinates)
from examples.TOV.tov_solver import load_or_solve_tov_iso
import examples.TOV.tov_initial_data_interpolated as tov_id

# TOV utilities and plotting
import examples.TOV.utils_TOVEvolution as utils
from examples.TOV.utils_TOVEvolution import (SimulationDataManager, evolve_fixed_timestep,
                                              get_star_folder_name)



def get_rhs_cowling(t, y, grid, background, hydro, bssn_fixed, bssn_d1_fixed):
    """
    RHS for Cowling evolution (fixed spacetime) of hydro variables only.
    
    """
    state = y.reshape((grid.NUM_VARS, grid.N))
    grid.fill_boundaries(state)

    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(bssn_fixed)
    hydro.set_matter_vars(state, bssn_vars, grid)

    # Compute hydro RHS
    hydro_rhs = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1_fixed, background)

    # Full RHS (BSSN frozen, only hydro evolves)
    rhs = np.zeros_like(state)
    rhs[NUM_BSSN_VARS:, :] = hydro_rhs

    return rhs.flatten()


class _DummyProgressBar:
    """Dummy progress bar that does nothing (for RK4 substeps)."""
    def update(self, n):
        pass


def get_rhs_dynamic(t, y, grid, background, hydro, progress_bar=None, time_state=None):
    """
    RHS for full BSSN + hydro evolution (no Cowling approximation).

    Both spacetime (BSSN) and matter (hydro) variables evolve together.
    Uses 1+log slicing for lapse and gamma-driver for shift.
    """
    # Create dummy progress bar and time state if not provided
    if progress_bar is None:
        progress_bar = _DummyProgressBar()
    if time_state is None:
        time_state = [0.0, 1.0]  # [last_t, dt] - dummy values

    # Call the full RHS from rhsevolution.py
    return get_rhs(t, y, grid, background, hydro, progress_bar, time_state)

def _apply_atmosphere_reset(state_2d, grid, hydro, atmosphere, rho_threshold=None):
    """Aplica floors atmosféricos a variables conservativas densitizadas."""
    from source.matter.hydro.atmosphere import FloorApplicator

    rho_hard_floor = rho_threshold or 100.0 * atmosphere.rho_floor
    idx_D, idx_Sr, idx_tau = NUM_BSSN_VARS, NUM_BSSN_VARS + 1, NUM_BSSN_VARS + 2

    # Métrica
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
    bar_gamma_LL = get_bar_gamma_LL(grid.r, bssn_vars.h_LL, hydro.background)
    e4phi = np.exp(4.0 * bssn_vars.phi)
    e6phi = np.exp(6.0 * bssn_vars.phi)
    gamma_rr = e4phi * bar_gamma_LL[:, 0, 0]

    # Hard reset atmosférico
    atm_mask = state_2d[idx_D, :] < rho_hard_floor * e6phi
    if np.any(atm_mask):
        state_2d[idx_D, atm_mask] = atmosphere.rho_floor * e6phi[atm_mask]
        state_2d[idx_Sr, atm_mask] = 0.0
        state_2d[idx_tau, atm_mask] = atmosphere.tau_atm * e6phi[atm_mask]

    # Floors en región no-atmosférica
    non_atm = ~atm_mask
    if not np.any(non_atm):
        return state_2d

    # Recuperar primitivas y aplicar floors
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    rho0, vr, p, *_ = hydro._get_primitives(bssn_vars, grid.r)
    
    floor_app = FloorApplicator(atmosphere, hydro.eos)
    rho0, vr, p = floor_app.apply_primitive_floors(rho0, vr, p, gamma_rr)

    # Recomputar conservativas donde no es atmósfera
    geom = GeometryState(
        alpha=np.ones_like(e6phi),   # no usado por prim_to_cons
        beta_r=np.zeros_like(e6phi), # no usado por prim_to_cons
        gamma_rr=gamma_rr,
        e6phi=e6phi
    )
    D_new, Sr_new, tau_new = prim_to_cons(rho0, vr, p, geom, hydro.eos)
    
    state_2d[idx_D, non_atm] = D_new[non_atm]
    state_2d[idx_Sr, non_atm] = Sr_new[non_atm]
    state_2d[idx_tau, non_atm] = tau_new[non_atm]

    return state_2d


def rk4_step(state_flat, dt, grid, background, hydro, bssn_fixed, bssn_d1_fixed,
             atmosphere):
    """
    Single RK4 timestep with atmosphere reset applied at EACH substage.
    Crucial for stability in GRHD.
    """
    # --- Stage 1 ---
    k1 = get_rhs_cowling(0, state_flat, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    # --- Stage 2 ---
    state_2 = state_flat + 0.5 * dt * k1
    s2 = state_2.reshape((grid.NUM_VARS, grid.N))
    state_2 = s2.flatten()

    k2 = get_rhs_cowling(0, state_2, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    # --- Stage 3 ---
    state_3 = state_flat + 0.5 * dt * k2
    s3 = state_3.reshape((grid.NUM_VARS, grid.N))
    state_3 = s3.flatten()

    k3 = get_rhs_cowling(0, state_3, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    # --- Stage 4 ---
    state_4 = state_flat + dt * k3
    s4 = state_4.reshape((grid.NUM_VARS, grid.N))
    state_4 = s4.flatten()

    k4 = get_rhs_cowling(0, state_4, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    # --- Combine ---
    state_new = state_flat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    snew = state_new.reshape((grid.NUM_VARS, grid.N))

    # Apply atmosphere reset after full step
    snew_reset = _apply_atmosphere_reset(snew, grid, hydro, atmosphere)

    return snew_reset.flatten()


def rk4_step_dynamic(state_flat, dt, grid, background, hydro, atmosphere,
                     progress_bar=None, time_state=None):
    """
    Single RK4 timestep for full BSSN + hydro evolution.
    Atmosphere reset applied after EACH substage to prevent spurious expansion.
    """
    state = state_flat.reshape((grid.NUM_VARS, grid.N))

    # --- Stage 1 ---
    k1 = get_rhs_dynamic(0, state_flat, grid, background, hydro, progress_bar, time_state)
    state_1 = state + 0.5 * dt * k1.reshape((grid.NUM_VARS, grid.N))
    #state_1 = _apply_atmosphere_reset(state_1, grid, hydro, atmosphere)

    # --- Stage 2 ---
    k2 = get_rhs_dynamic(0, state_1.flatten(), grid, background, hydro, progress_bar, time_state)
    state_2 = state + 0.5 * dt * k2.reshape((grid.NUM_VARS, grid.N))
    #state_2 = _apply_atmosphere_reset(state_2, grid, hydro, atmosphere)

    # --- Stage 3 ---
    k3 = get_rhs_dynamic(0, state_2.flatten(), grid, background, hydro, progress_bar, time_state)
    state_3 = state + dt * k3.reshape((grid.NUM_VARS, grid.N))
    #state_3 = _apply_atmosphere_reset(state_3, grid, hydro, atmosphere)

    # --- Stage 4 ---
    k4 = get_rhs_dynamic(0, state_3.flatten(), grid, background, hydro, progress_bar, time_state)

    # --- Combine ---
    state_new = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4).reshape((grid.NUM_VARS, grid.N))

    # Apply final atmosphere reset
    state_new = _apply_atmosphere_reset(state_new, grid, hydro, atmosphere)

    return state_new.flatten()


def main():
    """Main execution."""

    # ==================================================================
    # CONFIGURATION
    # ==================================================================
    r_max = 100.0
    num_points = 2000
    K = 100.0
    Gamma = 2.0
    rho_central = 1.28e-3
    t_final = 2000
    FOLDER_NAME_EVOL = "tov_evolution_data_${num_points}_rmax${r_max}"


    # Reconstructor: "wenoz" (wz), "weno5" (w5), "mp5" (mp5), "minmod" (md), "mc" (mc)
    RECONSTRUCTOR_NAME = "mp5"  # "wenoz", "weno5", "mp5", "minmod", "mc"

    # Cons2prim solver: "newton" (fast, needs good guess) or "kastaun" (robust, guaranteed convergence)
    SOLVER_METHOD = "newton"  # "newton" or "kastaun"

    # Riemann solver: "hll" or "llf"
    RIEMANN_SOLVER = "hll"  # "hll" or "llf"

    # Options: "cowling" or "dynamic"
    EVOLUTION_MODE = "cowling"  # "cowling" or "dynamic"

    # atmosphere config
    rho_floor_base = 1e-9 * rho_central
    p_floor_base = K * (rho_floor_base)**Gamma
    ATMOSPHERE = AtmosphereParams(
        rho_floor=rho_floor_base,
        p_floor=p_floor_base
        )

    # ==================================================================
    # SETUP
    # ==================================================================
    spacing = LinearSpacing(num_points, r_max)
    eos = PolytropicEOS(K=K, gamma=Gamma)
    #eos = IdealGasEOS(gamma=Gamma)  # Use ideal gas EOS for evolution
    
    # 1. RECONSTRUCTOR BASE (uses RECONSTRUCTOR_NAME from configuration)
    base_recon = create_reconstruction(RECONSTRUCTOR_NAME)

    # 2. RIEMANN SOLVER
    if RIEMANN_SOLVER.lower() == "hll":
        riemann = HLLRiemannSolver(atmosphere=ATMOSPHERE)
    else:
        riemann = LLFRiemannSolver(atmosphere=ATMOSPHERE)

    # 3. PASAR
    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere=ATMOSPHERE,
        reconstructor=base_recon,
        riemann_solver=riemann,
        solver_method=SOLVER_METHOD
    )

    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background


    # Create suffix for plot filenames based on evolution mode
    PLOT_SUFFIX = "_cow" if EVOLUTION_MODE == "cowling" else "_dyn"

    # Print header based on evolution mode
    print("="*70)
    if EVOLUTION_MODE == "dynamic":
        print("TOV Star Evolution - Full BSSN + Hydro (Dynamic Spacetime)")
    else:
        print("TOV Star Evolution - Cowling Approximation (Fixed Spacetime)")
    print("="*70)
    print("Using ISOTROPIC coordinates (conformally flat spatial metric)")

    # ==================================================================
    # DATA SAVING CONFIGURATION
    # ==================================================================
    ENABLE_DATA_SAVING = True  # Set to True to save data to files
    DATA_ROOT_DIR = os.path.join(script_dir,FOLDER_NAME_EVOL)  # Root directory for all data

    # Create star-specific folder based on parameters (includes evolution mode and reconstructor)
    star_folder = get_star_folder_name(rho_central, num_points, K, Gamma, EVOLUTION_MODE, RECONSTRUCTOR_NAME)
    OUTPUT_DIR = os.path.join(DATA_ROOT_DIR, star_folder)

    SNAPSHOT_INTERVAL = 100  # Save full domain every N timesteps (None to disable)
    EVOLUTION_INTERVAL = 100  # Save time series every N timesteps (None to disable)

    # Time series saving (density profiles and central density evolution)
    SAVE_TIMESERIES = True  # Set to True to save time series data to .npz file

    if ENABLE_DATA_SAVING:
        print(f"Data saving enabled:")
        print(f"  Root directory: {DATA_ROOT_DIR}")
        print(f"  Star folder: {star_folder}")
        print(f"  Full path: {OUTPUT_DIR}")
        print(f"  Snapshot interval: {SNAPSHOT_INTERVAL} timesteps")
        print(f"  Evolution tracking: {EVOLUTION_INTERVAL} timesteps")
        print(f"  Save time series: {SAVE_TIMESERIES}")
        print()

    # ==================================================================
    print("=" * 70)
    print("ATMOSPHERE CONFIGURATION")
    print("=" * 70)
    print(f"  rho_floor = {ATMOSPHERE.rho_floor:.2e}")
    print(f"  p_floor   = {ATMOSPHERE.p_floor:.2e}")
    print(f"  tau_atm   = {ATMOSPHERE.tau_atm:.2e}")
    print(f"  v_max     = {ATMOSPHERE.v_max}")
    print()

    # Time integration method
    # 'fixed': RK4 with fixed timestep 
    integration_method = 'fixed'  # 'fixed' 

    print(f"Grid: N={grid.N}, r_max={r_max}, dr_min={grid.min_dr}")
    print(f"EOS: K={K}, Gamma={Gamma}")
    print(f"Cons2prim solver: {SOLVER_METHOD}")
    print(f"Riemann solver: {RIEMANN_SOLVER}\n")

    # ==================================================================
    # SOLVE TOV DIRECTLY ON EVOLUTION GRID (for discrete equilibrium)
    # ==================================================================
    print("Solving TOV equations...")

    tov_solution = load_or_solve_tov_iso(
        K=K, Gamma=Gamma, rho_central=rho_central,
        r_max=r_max, accuracy="high"
    )
    print(f"TOV Solution: M={tov_solution.M_star:.6f}, R_iso={tov_solution.R_iso:.3f}, R_schw={tov_solution.R_schw:.3f}, C={tov_solution.C:.4f}\n")

    #utils.plot_tov_diagnostics(tov_solution, r_max, suffix=PLOT_SUFFIX)

    # ==================================================================
    # INITIAL DATA (HIGH-ORDER INTERPOLATION)
    # ==================================================================
    print("Creating initial data from TOV solution...")
    # 1. Interpolate ρ, P ONLY up to stellar radius R
    # 2. Outside R: use atmosphere values directly (no interpolation)
    # 3. Interpolate geometry (alpha, exp4φ) everywhere
    # 4. Stencil NEVER crosses the stellar surface (avoids Gibbs phenomenon)

    initial_state_2d, prim_tuple = tov_id.create_initial_data_iso(
        tov_solution, grid, background, eos,
        atmosphere=ATMOSPHERE,
        polytrope_K=K, polytrope_Gamma=Gamma,
        interp_order=11
    )

    # Diagnostics: check discrete hydrostatic balance at t=0
    utils.diagnose_t0_residuals(initial_state_2d, grid, background, hydro)

    # Initial-data diagnostics 
    tov_id.plot_initial_comparison(tov_solution, initial_state_2d, grid, prim_tuple,
                                   output_dir=plots_dir, suffix=PLOT_SUFFIX)

    # Zoom comparison: TOV solution vs interpolated initial data 
    #utils.plot_tov_vs_initial_data_zoom(tov_solution, initial_state_2d, grid, prim_tuple,
     #                                   window=0.1, suffix=PLOT_SUFFIX)

    # ==================================================================
    # EVOLUTION
    # ==================================================================
    # Setup depends on evolution mode
    if EVOLUTION_MODE == "dynamic":
        print("\n" + "="*70)
        print("EVOLUTION MODE: DYNAMIC (Full BSSN + Hydro)")
        print("  - Spacetime evolves with matter")
        print("  - 1+log slicing for lapse")
        print("  - Gamma-driver for shift")
        print("="*70)

        # For dynamic mode, we still compute these for compatibility but they won't be used
        # to freeze the metric - they're just passed through to evolve_fixed_timestep
        bssn_fixed = initial_state_2d[:NUM_BSSN_VARS, :].copy()  # Not actually used to freeze
        bssn_d1_fixed = grid.get_d1_metric_quantities(initial_state_2d)

        # Wrapper for dynamic RK4 step (ignores bssn_fixed, bssn_d1_fixed)
        def rk4_step_wrapper(state_flat, dt, grid, background, hydro,
                            bssn_fixed_unused, bssn_d1_fixed_unused, atmosphere):
            return rk4_step_dynamic(state_flat, dt, grid, background, hydro, atmosphere)

        selected_rk4_step = rk4_step_wrapper
        cfl_factor = 0.1  # More conservative for full BSSN evolution

    else:  # cowling mode
        print("\n" + "="*70)
        print("EVOLUTION MODE: COWLING (Fixed Spacetime)")
        print("  - BSSN variables frozen at t=0")
        print("  - Only hydro evolves")
        print("="*70)

        bssn_fixed = initial_state_2d[:NUM_BSSN_VARS, :].copy()
        bssn_d1_fixed = grid.get_d1_metric_quantities(initial_state_2d)
        selected_rk4_step = rk4_step
        cfl_factor = 0.1  # Standard CFL for Cowling

    # Initialize data manager for saving
    data_manager = SimulationDataManager(OUTPUT_DIR, grid, hydro,
                                        enable_saving=ENABLE_DATA_SAVING,
                                        suffix=PLOT_SUFFIX)

    if integration_method == 'fixed':
        dt = cfl_factor * grid.min_dr  # CFL condition
        num_steps_total = int(t_final / dt)  # Calculate steps from t_final
        print(f"\nEvolving with fixed dt={dt:.6f} (CFL={cfl_factor}) to t_final={t_final} ({num_steps_total} steps) using RK4")

        # Start timing the evolution
        evolution_start_time = time.time()

        # Save metadata now that we have dt
        if ENABLE_DATA_SAVING:
            data_manager.save_metadata(
                tov_solution, ATMOSPHERE, dt, integration_method,
                K=K, Gamma=Gamma, rho_central=rho_central,
                r_max=r_max, num_points=num_points, t_final=t_final,
                reconstructor=RECONSTRUCTOR_NAME, solver_method=SOLVER_METHOD,
                riemann_solver=RIEMANN_SOLVER, evolution_mode=EVOLUTION_MODE,
                cfl_factor=cfl_factor
            )

        # Single step for comparison
        state_t1 = selected_rk4_step(initial_state_2d.flatten(), dt, grid, background, hydro,
                                     bssn_fixed, bssn_d1_fixed, ATMOSPHERE).reshape((grid.NUM_VARS, grid.N))
        t_1 = dt

        # Plot only the first step changes
        #utils.plot_first_step(initial_state_2d, state_t1, grid, hydro, tov_solution,
        #                     suffix=PLOT_SUFFIX)
        #utils.plot_surface_zoom(tov_solution, initial_state_2d, state_t1, grid, hydro,
        #                       primitives_t0=prim_tuple, window=0.1, suffix=PLOT_SUFFIX)
        #utils.plot_center_zoom(initial_state_2d, state_t1, grid, hydro,
        #                      window=0.5, suffix=PLOT_SUFFIX)

        # Define checkpoints at 1/3, 2/3, and final of total steps
        checkpoint_1 = max(1, num_steps_total // 3)
        checkpoint_2 = max(2, 2 * num_steps_total // 3)
        checkpoint_3 = num_steps_total

        print(f"\n{'='*70}")
        print(f"Evolution checkpoints (for plotting):")
        print(f"  t=0:         initial state")
        print(f"  step {checkpoint_1:6d}:  1/3 of evolution (~{checkpoint_1*dt:.3e} time units)")
        print(f"  step {checkpoint_2:6d}:  2/3 of evolution (~{checkpoint_2*dt:.3e} time units)")
        print(f"  step {checkpoint_3:6d}:  final state     (~{checkpoint_3*dt:.3e} time units)")
        print(f"{'='*70}\n")

        # Storage for checkpoint states
        checkpoint_states = {}
        checkpoint_times = {}
        all_series = []

        # Evolve to checkpoint 1
        print(f"Evolving to checkpoint 1 (step {checkpoint_1})...")
        state_cp1, steps_cp1, t_cp1, series_1 = evolve_fixed_timestep(
            initial_state_2d, dt, checkpoint_1, grid, background,
            hydro, bssn_fixed, bssn_d1_fixed, ATMOSPHERE, selected_rk4_step,
            method='rk4', reference_state=initial_state_2d,
            data_manager=data_manager,
            snapshot_interval=SNAPSHOT_INTERVAL,
            evolution_interval=EVOLUTION_INTERVAL)
        checkpoint_states[1] = state_cp1.copy()
        checkpoint_times[1] = t_cp1
        all_series.append(series_1)
        print(f"  -> Reached step {steps_cp1}, t={t_cp1:.6e}")

        # Evolve to checkpoint 2
        if steps_cp1 == checkpoint_1:
            remaining_steps = checkpoint_2 - checkpoint_1
            print(f"\nEvolving to checkpoint 2 (step {checkpoint_2})...")
            state_cp2, steps_cp2, t_cp2, series_2 = evolve_fixed_timestep(
                state_cp1, dt, remaining_steps, grid, background,
                hydro, bssn_fixed, bssn_d1_fixed, ATMOSPHERE, selected_rk4_step,
                method='rk4', t_start=t_cp1, reference_state=initial_state_2d,
                step_offset=checkpoint_1,
                data_manager=data_manager,
                snapshot_interval=SNAPSHOT_INTERVAL,
                evolution_interval=EVOLUTION_INTERVAL)
            checkpoint_states[2] = state_cp2.copy()
            checkpoint_times[2] = t_cp2
            all_series.append(series_2)
            print(f"  -> Reached step {checkpoint_1 + steps_cp2}, t={t_cp2:.6e}")
        else:
            # Stopped early
            state_cp2 = state_cp1
            t_cp2 = t_cp1
            checkpoint_states[2] = state_cp2.copy()
            checkpoint_times[2] = t_cp2

        # Evolve to checkpoint 3 (final)
        if steps_cp1 == checkpoint_1 and (checkpoint_1 + steps_cp2) == checkpoint_2:
            remaining_steps = checkpoint_3 - checkpoint_2
            print(f"\nEvolving to checkpoint 3 (step {checkpoint_3}, final)...")
            state_cp3, steps_cp3, t_cp3, series_3 = evolve_fixed_timestep(
                state_cp2, dt, remaining_steps, grid, background,
                hydro, bssn_fixed, bssn_d1_fixed, ATMOSPHERE, selected_rk4_step,
                method='rk4', t_start=t_cp2, reference_state=initial_state_2d,
                step_offset=checkpoint_2,
                data_manager=data_manager,
                snapshot_interval=SNAPSHOT_INTERVAL,
                evolution_interval=EVOLUTION_INTERVAL)
            checkpoint_states[3] = state_cp3.copy()
            checkpoint_times[3] = t_cp3
            all_series.append(series_3)
            steps_final = checkpoint_2 + steps_cp3
            print(f"  -> Reached step {steps_final}, t={t_cp3:.6e}")
        else:
            # Stopped early
            state_cp3 = state_cp2
            t_cp3 = t_cp2
            checkpoint_states[3] = state_cp3.copy()
            checkpoint_times[3] = t_cp3
            steps_final = checkpoint_1 + steps_cp2

        # Assign final state for backward compatibility
        state_t10000 = checkpoint_states[3]
        t_10000 = checkpoint_times[3]
        num_steps = steps_final

        # Build full-series arrays for mass, central density, and central velocity
        try:
            if len(all_series) == 3:
                # All three segments completed
                times_full = np.concatenate([all_series[0]['t'], all_series[1]['t'][1:], all_series[2]['t'][1:]])
                Mb_full = np.concatenate([all_series[0]['Mb'], all_series[1]['Mb'][1:], all_series[2]['Mb'][1:]])
                rho_c_full = np.concatenate([all_series[0]['rho_c'], all_series[1]['rho_c'][1:], all_series[2]['rho_c'][1:]])
                v_c_full = np.concatenate([all_series[0]['v_c'], all_series[1]['v_c'][1:], all_series[2]['v_c'][1:]])
            elif len(all_series) == 2:
                # Only first two segments completed
                times_full = np.concatenate([all_series[0]['t'], all_series[1]['t'][1:]])
                Mb_full = np.concatenate([all_series[0]['Mb'], all_series[1]['Mb'][1:]])
                rho_c_full = np.concatenate([all_series[0]['rho_c'], all_series[1]['rho_c'][1:]])
                v_c_full = np.concatenate([all_series[0]['v_c'], all_series[1]['v_c'][1:]])
            elif len(all_series) == 1:
                # Only first segment completed
                times_full = all_series[0]['t']
                Mb_full = all_series[0]['Mb']
                rho_c_full = all_series[0]['rho_c']
                v_c_full = all_series[0]['v_c']
            else:
                times_full = np.array([])
                Mb_full = np.array([])
                rho_c_full = np.array([])
                v_c_full = np.array([])
        except Exception as e:
            print(f"Warning: Failed to concatenate series: {e}")
            times_full = np.array([])
            Mb_full = np.array([])
            rho_c_full = np.array([])
            v_c_full = np.array([])

        # ==================================================================
        # SAVE TIME SERIES TO FILE (optional)
        # ==================================================================
        if SAVE_TIMESERIES and len(times_full) > 0:
            # Simple filename - folder already contains parameters
            timeseries_filename = "timeseries.npz"
            timeseries_path = os.path.join(OUTPUT_DIR, timeseries_filename)

            # Ensure output directory exists
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            # Save time series data
            np.savez(timeseries_path,
                     times=times_full,
                     rho_central=rho_c_full,
                     v_central=v_c_full,
                     Mb=Mb_full,
                     # Simulation parameters
                     num_points=num_points,
                     K=K,
                     Gamma=Gamma,
                     rho_central_initial=rho_central,
                     r_max=r_max,
                     dt=dt,
                     num_steps=num_steps_total)
            print(f"\nTime series saved to: {timeseries_path}")

    # ==================================================================
    # DIAGNOSTICS
    # ==================================================================
    # Combined 3x2 figure: keep 4 panels and replace bottom row with the two new time-series
    # Get reference primitives for error computation
    bssn_ref = BSSNVars(grid.N)
    bssn_ref.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(initial_state_2d, bssn_ref, grid)
    rho_ref, _, p_ref, _, _, _, _ = hydro._get_primitives(bssn_ref, grid.r)

    # Build states and times lists for plot_evolution
    # Use checkpoints: t=0, 1/3, 2/3, final (uniformly distributed)
    states = [initial_state_2d, checkpoint_states[1], checkpoint_states[2], checkpoint_states[3]]
    times = [0.0, checkpoint_times[1], checkpoint_times[2], checkpoint_times[3]]

    # Get stellar radius for plot markers (isotropic coordinates)
    R_star = tov_solution.R_iso

    try:
        if 'times_full' in locals() and len(times_full) > 0:
            utils.plot_evolution(states, times, grid, hydro, rho_ref, p_ref,
                                 Mb_series=Mb_full, rho_c_series=rho_c_full,
                                 suffix=PLOT_SUFFIX, R_star=R_star)
        else:
            utils.plot_evolution(states, times, grid, hydro, rho_ref, p_ref,
                                suffix=PLOT_SUFFIX, R_star=R_star)
    except Exception:
        utils.plot_evolution(states, times, grid, hydro, rho_ref, p_ref,
                            suffix=PLOT_SUFFIX, R_star=R_star)

    # Plot BSSN variables evolution
    # In Cowling mode: should be constant (verification)
    # In Dynamic mode: should show metric evolution
    utils.plot_bssn_evolution(initial_state_2d, checkpoint_states[3], grid,
                             t_0=0.0, t_final=checkpoint_times[3],
                             suffix=PLOT_SUFFIX)

    # ==================================================================
    # CONSTRAINT DIAGNOSTICS (Dynamic mode only)
    # ==================================================================
    if EVOLUTION_MODE == "dynamic":
        print("\n" + "="*70)
        print("CONSTRAINT VIOLATION DIAGNOSTICS")
        print("="*70)

        interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

        # Compute constraints at initial and final states
        Ham_0, Mom_0 = get_constraints_diagnostic(
            initial_state_2d.flatten(), 0.0, grid, background, hydro)
        Ham_f, Mom_f = get_constraints_diagnostic(
            checkpoint_states[3].flatten(), checkpoint_times[3], grid, background, hydro)

        # Extract max violations (interior only)
        max_H_0 = np.max(np.abs(Ham_0[0, interior]))
        max_M_0 = np.max(np.abs(Mom_0[0, interior, 0]))  # Radial component
        max_H_f = np.max(np.abs(Ham_f[0, interior]))
        max_M_f = np.max(np.abs(Mom_f[0, interior, 0]))

        print(f"Hamiltonian constraint |H|:")
        print(f"  t=0:     max|H| = {max_H_0:.3e}")
        print(f"  t=final: max|H| = {max_H_f:.3e}")
        print(f"  Growth factor: {max_H_f/max_H_0:.2f}x" if max_H_0 > 1e-20 else "  (initial ~0)")

        print(f"\nMomentum constraint |M_r|:")
        print(f"  t=0:     max|M_r| = {max_M_0:.3e}")
        print(f"  t=final: max|M_r| = {max_M_f:.3e}")
        print(f"  Growth factor: {max_M_f/max_M_0:.2f}x" if max_M_0 > 1e-20 else "  (initial ~0)")
        print("="*70)

    # Print detailed statistics - extract primitives for each state
    bssn_0 = BSSNVars(grid.N)
    bssn_0.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(initial_state_2d, bssn_0, grid)
    rho0_0, vr_0, p_0, _, _, _, success_0 = hydro._get_primitives(bssn_0, grid.r)

    bssn_1 = BSSNVars(grid.N)
    bssn_1.set_bssn_vars(state_t1[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t1, bssn_1, grid)
    rho0_1, vr_1, p_1, _, _, _, success_1 = hydro._get_primitives(bssn_1, grid.r)

    bssn_cp1 = BSSNVars(grid.N)
    bssn_cp1.set_bssn_vars(checkpoint_states[1][:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(checkpoint_states[1], bssn_cp1, grid)
    rho0_cp1, vr_cp1, p_cp1, _, _, _, success_cp1 = hydro._get_primitives(bssn_cp1, grid.r)

    bssn_cp2 = BSSNVars(grid.N)
    bssn_cp2.set_bssn_vars(checkpoint_states[2][:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(checkpoint_states[2], bssn_cp2, grid)
    rho0_cp2, vr_cp2, p_cp2, _, _, _, success_cp2 = hydro._get_primitives(bssn_cp2, grid.r)

    bssn_cp3 = BSSNVars(grid.N)
    bssn_cp3.set_bssn_vars(checkpoint_states[3][:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(checkpoint_states[3], bssn_cp3, grid)
    rho0_cp3, vr_cp3, p_cp3, _, _, _, success_cp3 = hydro._get_primitives(bssn_cp3, grid.r)

    # Interior points only (exclude ghosts)
    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

    # Compute errors at checkpoints
    delta_rho_1 = np.abs(rho0_1[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)
    delta_rho_cp1 = np.abs(rho0_cp1[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)
    delta_rho_cp2 = np.abs(rho0_cp2[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)
    delta_rho_cp3 = np.abs(rho0_cp3[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)

    delta_P_1 = np.abs(p_1[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)
    delta_P_cp1 = np.abs(p_cp1[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)
    delta_P_cp2 = np.abs(p_cp2[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)
    delta_P_cp3 = np.abs(p_cp3[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)

    # Error growth factor
    max_err_rho_1 = np.max(delta_rho_1)
    max_err_rho_cp1 = np.max(delta_rho_cp1)
    max_err_rho_cp2 = np.max(delta_rho_cp2)
    max_err_rho_cp3 = np.max(delta_rho_cp3)
    growth_rho = max_err_rho_cp3 / max_err_rho_1 if max_err_rho_1 > 1e-15 else 0

    max_err_P_1 = np.max(delta_P_1)
    max_err_P_cp1 = np.max(delta_P_cp1)
    max_err_P_cp2 = np.max(delta_P_cp2)
    max_err_P_cp3 = np.max(delta_P_cp3)
    growth_P = max_err_P_cp3 / max_err_P_1 if max_err_P_1 > 1e-15 else 0

    t_cp1 = checkpoint_times[1]
    t_cp2 = checkpoint_times[2]
    t_cp3 = checkpoint_times[3]

    print(f"\n{'='*70}")
    print(f"EVOLUTION DIAGNOSTICS")
    print(f"  t=0 → t={t_cp1:.6e} (1/3) → t={t_cp2:.6e} (2/3) → t={t_cp3:.6e} (final)")
    print(f"  (first step: t={t_1:.6e}, included for diagnostics)")
    print(f"{'='*70}")

    print(f"\n1. VELOCITY EVOLUTION:")
    print(f"   Max |v^r| at t=0:              {np.max(np.abs(vr_0)):.3e}")
    print(f"   Max |v^r| at t={t_1:.6e}:    {np.max(np.abs(vr_1)):.3e}")
    print(f"   Max |v^r| at t={t_cp1:.6e} (1/3):  {np.max(np.abs(vr_cp1)):.3e}")
    print(f"   Max |v^r| at t={t_cp2:.6e} (2/3):  {np.max(np.abs(vr_cp2)):.3e}")
    print(f"   Max |v^r| at t={t_cp3:.6e} (final): {np.max(np.abs(vr_cp3)):.3e}")

    print(f"\n2. CENTRAL DENSITY:")
    print(f"   ρ_c at t=0:                  {rho0_0[NUM_GHOSTS]:.6e}")
    print(f"   ρ_c at t={t_1:.6e}:    {rho0_1[NUM_GHOSTS]:.6e}")
    print(f"   ρ_c at t={t_cp1:.6e} (1/3):  {rho0_cp1[NUM_GHOSTS]:.6e}")
    print(f"   ρ_c at t={t_cp2:.6e} (2/3):  {rho0_cp2[NUM_GHOSTS]:.6e}")
    print(f"   ρ_c at t={t_cp3:.6e} (final): {rho0_cp3[NUM_GHOSTS]:.6e}")
    print(f"   Δρ_c/ρ_c (first step):       {abs(rho0_1[NUM_GHOSTS] - rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")
    print(f"   Δρ_c/ρ_c (1/3):              {abs(rho0_cp1[NUM_GHOSTS] - rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")
    print(f"   Δρ_c/ρ_c (2/3):              {abs(rho0_cp2[NUM_GHOSTS] - rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")
    print(f"   Δρ_c/ρ_c (final):            {abs(rho0_cp3[NUM_GHOSTS] - rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")

    print(f"\n3. DENSITY ERROR (max over domain):")
    print(f"   Max |Δρ|/ρ at t={t_1:.6e}:     {max_err_rho_1:.3e}")
    print(f"   Max |Δρ|/ρ at t={t_cp1:.6e} (1/3):   {max_err_rho_cp1:.3e}")
    print(f"   Max |Δρ|/ρ at t={t_cp2:.6e} (2/3):   {max_err_rho_cp2:.3e}")
    print(f"   Max |Δρ|/ρ at t={t_cp3:.6e} (final):  {max_err_rho_cp3:.3e}")
    print(f"   Growth factor (final/first): {growth_rho:.1f}x")

    print(f"\n4. PRESSURE ERROR (max over domain):")
    print(f"   Max |ΔP|/P at t={t_1:.6e}:     {max_err_P_1:.3e}")
    print(f"   Max |ΔP|/P at t={t_cp1:.6e} (1/3):   {max_err_P_cp1:.3e}")
    print(f"   Max |ΔP|/P at t={t_cp2:.6e} (2/3):   {max_err_P_cp2:.3e}")
    print(f"   Max |ΔP|/P at t={t_cp3:.6e} (final):  {max_err_P_cp3:.3e}")
    print(f"   Growth factor (final/first): {growth_P:.1f}x")

    print(f"\n5. CONS2PRIM STATUS:")
    print(f"   Success at t=0:                {np.sum(success_0)}/{grid.N}")
    print(f"   Success at t={t_1:.6e}:  {np.sum(success_1)}/{grid.N}")
    print(f"   Success at t={t_cp1:.6e} (1/3):    {np.sum(success_cp1)}/{grid.N}")
    print(f"   Success at t={t_cp2:.6e} (2/3):    {np.sum(success_cp2)}/{grid.N}")
    print(f"   Success at t={t_cp3:.6e} (final):  {np.sum(success_cp3)}/{grid.N}")

    if not np.all(success_cp3):
        failed_idx = np.where(~success_cp3)[0]
        print(f"   Failed points: {failed_idx[:5]} (first 5)")
        print(f"   Failed radii:  {grid.r[failed_idx[:5]]}")

    # Finalize data saving
    if ENABLE_DATA_SAVING:
        execution_time = time.time() - evolution_start_time
        data_manager.finalize(execution_time_seconds=execution_time)

    # ==================================================================
    # PLOT CONSTRAINT VIOLATIONS (Dynamic mode only)
    # ==================================================================
    if EVOLUTION_MODE == "dynamic" and ENABLE_DATA_SAVING:
        print("\n" + "="*70)
        print("Plotting constraint violation evolution...")
        print("="*70)
        utils.plot_constraints_evolution(OUTPUT_DIR, suffix=PLOT_SUFFIX)

    print("\n" + "="*70)
    print("Evolution complete. Plots saved:")
    print("  1. tov_solution.png                - TOV solution (ρ, P, M, alpha)")
    print("  2. tov_initial_data_comparison.png - TOV vs Initial data at t=0")
    print(f"  3. tov_evolution.png               - Hydro evolution at checkpoints:")
    print(f"                                       t=0 → t={t_cp1:.3e} (1/3) → t={t_cp2:.3e} (2/3) → t={t_cp3:.3e} (final)")
    if EVOLUTION_MODE == "dynamic":
        print(f"  4. tov_bssn_evolution.png          - BSSN variables: t=0 → t={t_cp3:.3e} (metric evolution)")
        print(f"  5. constraints_evolution{PLOT_SUFFIX}.png - BSSN constraint violations: max|H|, L2(H), max|M_r|, L2(M_r)")
    else:
        print(f"  4. tov_bssn_evolution.png          - BSSN variables: t=0 → t={t_cp3:.3e} (Cowling check - should be constant)")
    print("="*70)


if __name__ == "__main__":
    main()
