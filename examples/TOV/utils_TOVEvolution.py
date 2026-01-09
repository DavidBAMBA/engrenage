"""
TOV Evolution Utilities

This module contains utilities for TOV evolution simulations:
- Data management (SimulationDataManager)
- Diagnostic functions
- Plotting functions (all visualization)
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, simpson, trapezoid
from scipy.interpolate import interp1d
import h5py
import json
from datetime import datetime

# Import required BSSN and Hydro components
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import (NUM_BSSN_VARS, idx_phi, idx_hrr, idx_htt, idx_hpp,
                                             idx_K, idx_arr, idx_att, idx_app,
                                             idx_lambdar, idx_shiftr, idx_br, idx_lapse)
from source.core.spacing import NUM_GHOSTS

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)


# =============================================================================
# DATA MANAGEMENT
# =============================================================================

class SimulationDataManager:
    """Manages data storage for long TOV simulations."""

    def __init__(self, output_dir, grid, hydro, enable_saving=False, suffix=""):
        """
        Initialize data manager.

        Args:
            output_dir: Directory for output files
            grid: Grid object
            hydro: Hydro object
            enable_saving: If True, saves data to files
            suffix: Suffix for file naming (e.g., "_iso" for isotropic coordinates)
        """
        self.enable_saving = enable_saving
        self.suffix = suffix
        if not self.enable_saving:
            return

        self.output_dir = output_dir
        self.grid = grid
        self.hydro = hydro

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize HDF5 files with coordinate system suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.snapshot_file = os.path.join(output_dir, f"tov_snapshots{suffix}_{timestamp}.h5")
        self.evolution_file = os.path.join(output_dir, f"tov_evolution{suffix}_{timestamp}.h5")
        self.metadata_file = os.path.join(output_dir, f"tov_metadata{suffix}_{timestamp}.json")

        # Initialize evolution data lists (for buffering)
        self.evolution_buffer = {
            'step': [],
            'time': [],
            'rho_central': [],
            'p_central': [],
            'max_rho_error': [],
            'max_p_error': [],
            'max_velocity': [],
            'l1_rho_error': [],
            'l2_rho_error': [],
            'max_D': [],
            'max_Sr': [],
            'max_tau': [],
            'c2p_fails': []
        }

        # Initialize HDF5 files
        self._init_hdf5_files()

    def _init_hdf5_files(self):
        """Initialize HDF5 files with proper structure."""
        if not self.enable_saving:
            return

        # Snapshot file
        with h5py.File(self.snapshot_file, 'w') as f:
            # Create groups
            f.create_group('snapshots')
            f.create_group('grid')

            # Save grid data
            f['grid/r'] = self.grid.r
            f['grid/N'] = self.grid.N
            f['grid/r_max'] = self.grid.r[-1]

        # Evolution file
        with h5py.File(self.evolution_file, 'w') as f:
            # Create expandable datasets for time series
            for key in self.evolution_buffer.keys():
                f.create_dataset(key, shape=(0,), maxshape=(None,),
                               dtype=np.float64, chunks=True)

    def save_metadata(self, tov_solution, atmosphere_params, dt, integration_method, K=None, Gamma=None, rho_central=None):
        """Save simulation metadata."""
        if not self.enable_saving:
            return

        # Handle both TOVSolution objects and dicts
        if hasattr(tov_solution, 'M_star'):
            M_star = tov_solution.M_star
            R = tov_solution.R
            C = tov_solution.C
            K_val = tov_solution.K if K is None else K
            Gamma_val = tov_solution.Gamma if Gamma is None else Gamma
            rho_c_val = tov_solution.rho_central if rho_central is None else rho_central
        else:
            M_star = tov_solution['M_star']
            R = tov_solution['R']
            C = tov_solution['C']
            K_val = K if K is not None else tov_solution.get('K', 0)
            Gamma_val = Gamma if Gamma is not None else tov_solution.get('Gamma', 0)
            rho_c_val = rho_central if rho_central is not None else tov_solution.get('rho_central', 0)

        metadata = {
            'tov_solution': {
                'M_star': float(M_star),
                'R': float(R),
                'C': float(C),
                'K': float(K_val),
                'Gamma': float(Gamma_val),
                'rho_central': float(rho_c_val)
            },
            'atmosphere': {
                'rho_floor': float(atmosphere_params.rho_floor),
                'p_floor': float(atmosphere_params.p_floor),
                'v_max': float(atmosphere_params.v_max),
                'W_max': float(atmosphere_params.W_max)
            },
            'simulation': {
                'coordinate_system': 'isotropic' if '_iso' in self.suffix else 'schwarzschild',
                'dt': float(dt) if dt is not None else None,
                'integration_method': integration_method,
                'grid_N': int(self.grid.N),
                'grid_r_max': float(self.grid.r[-1])
            },
            'timestamp': datetime.now().isoformat()
        }

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def save_snapshot(self, step, time, state_2d, rho0=None, vr=None, p=None, eps=None, W=None, h=None):
        """Save full domain snapshot."""
        if not self.enable_saving:
            return

        with h5py.File(self.snapshot_file, 'a') as f:
            snap_name = f'step_{step:08d}'
            if snap_name in f['snapshots']:
                print(f"Warning: Snapshot {snap_name} already exists, skipping...")
                return
            snap_group = f['snapshots'].create_group(snap_name)

            snap_group.attrs['step'] = step
            snap_group.attrs['time'] = time

            # Save conservative variables
            cons_group = snap_group.create_group('conservatives')
            cons_group['D'] = state_2d[self.hydro.idx_D, :]
            cons_group['Sr'] = state_2d[self.hydro.idx_Sr, :]
            cons_group['tau'] = state_2d[self.hydro.idx_tau, :]

            # Save all BSSN variables (12 total)
            bssn_group = snap_group.create_group('bssn')
            bssn_group['phi'] = state_2d[0, :]       # idx_phi: conformal factor
            bssn_group['hrr'] = state_2d[1, :]       # idx_hrr: metric deviation rr
            bssn_group['htt'] = state_2d[2, :]       # idx_htt: metric deviation tt
            bssn_group['hpp'] = state_2d[3, :]       # idx_hpp: metric deviation pp
            bssn_group['K'] = state_2d[4, :]         # idx_K: mean curvature
            bssn_group['arr'] = state_2d[5, :]       # idx_arr: A_rr
            bssn_group['att'] = state_2d[6, :]       # idx_att: A_tt
            bssn_group['app'] = state_2d[7, :]       # idx_app: A_pp
            bssn_group['lambdar'] = state_2d[8, :]   # idx_lambdar: lambda^r
            bssn_group['shiftr'] = state_2d[9, :]    # idx_shiftr: shift beta^r
            bssn_group['br'] = state_2d[10, :]       # idx_br: B^r
            bssn_group['lapse'] = state_2d[11, :]    # idx_lapse: alpha

            # Save primitives if provided
            if rho0 is not None:
                prim_group = snap_group.create_group('primitives')
                prim_group['rho0'] = rho0
                prim_group['p'] = p
                prim_group['eps'] = eps
                prim_group['vr'] = vr
                prim_group['W'] = W

    def add_evolution_point(self, step, time, state_2d, rho0, vr, p, eps, W, h, success,
                           rho0_ref, vr_ref, p_ref, eps_ref, W_ref, h_ref, success_ref):
        """Add a point to the evolution time series."""
        if not self.enable_saving:
            return

        interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

        # Calculate errors
        delta_rho = np.abs(rho0[interior] - rho0_ref[interior])
        rel_delta_rho = delta_rho / (np.abs(rho0_ref[interior]) + 1e-20)

        delta_p = np.abs(p[interior] - p_ref[interior])
        rel_delta_p = delta_p / (np.abs(p_ref[interior]) + 1e-20)

        # L1 and L2 norms of density error (inside star only)
        star_mask = rho0_ref[interior] > 10 * self.hydro.atmosphere.rho_floor
        if np.any(star_mask):
            l1_rho = np.mean(delta_rho[star_mask])
            l2_rho = np.sqrt(np.mean(delta_rho[star_mask]**2))
        else:
            l1_rho = 0.0
            l2_rho = 0.0

        # Store in buffer
        self.evolution_buffer['step'].append(step)
        self.evolution_buffer['time'].append(time)
        self.evolution_buffer['rho_central'].append(rho0[NUM_GHOSTS])
        self.evolution_buffer['p_central'].append(p[NUM_GHOSTS])
        self.evolution_buffer['max_rho_error'].append(np.max(rel_delta_rho))
        self.evolution_buffer['max_p_error'].append(np.max(rel_delta_p))
        self.evolution_buffer['max_velocity'].append(np.max(np.abs(vr)))
        self.evolution_buffer['l1_rho_error'].append(l1_rho)
        self.evolution_buffer['l2_rho_error'].append(l2_rho)
        self.evolution_buffer['max_D'].append(np.max(state_2d[self.hydro.idx_D, :]))
        self.evolution_buffer['max_Sr'].append(np.max(np.abs(state_2d[self.hydro.idx_Sr, :])))
        self.evolution_buffer['max_tau'].append(np.max(state_2d[self.hydro.idx_tau, :]))
        self.evolution_buffer['c2p_fails'].append(np.sum(~success))

    def flush_evolution_buffer(self):
        """Write buffered evolution data to HDF5 file."""
        if not self.enable_saving or not self.evolution_buffer['step']:
            return

        with h5py.File(self.evolution_file, 'a') as f:
            for key, values in self.evolution_buffer.items():
                dataset = f[key]
                old_size = dataset.shape[0]
                new_size = old_size + len(values)
                dataset.resize(new_size, axis=0)
                dataset[old_size:new_size] = values

        # Clear buffer
        for key in self.evolution_buffer:
            self.evolution_buffer[key] = []

    def finalize(self):
        """Finalize data storage (flush buffers, close files)."""
        if not self.enable_saving:
            return

        self.flush_evolution_buffer()
        print(f"\nData saved to:")
        print(f"  - Snapshots: {self.snapshot_file}")
        print(f"  - Evolution: {self.evolution_file}")
        print(f"  - Metadata:  {self.metadata_file}")


# =============================================================================
# DIAGNOSTIC FUNCTIONS
# =============================================================================

def diagnose_t0_residuals(state_2d, grid, background, hydro):
    """Compute and print t=0 RHS residuals to locate imbalance."""
    # Build BSSN containers (Cowling)
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
    bssn_d1 = grid.get_d1_metric_quantities(state_2d)

    # Matter vars
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    rhs_D, rhs_Sr, rhs_tau = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1, background)

    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
    r_int = grid.r[interior]

    # Report maxima
    i_max = NUM_GHOSTS + int(np.argmax(np.abs(rhs_Sr[interior])))
    print("\nInitial RHS diagnostics (t=0):")
    print(f"  max |dS_r/dt| at r={grid.r[i_max]:.6f} (i={i_max}) -> {rhs_Sr[i_max]:.3e}")
    print(f"  max |dD/dt|   = {np.max(np.abs(rhs_D[interior])):.3e}")
    print(f"  max |dτ/dt|   = {np.max(np.abs(rhs_tau[interior])):.3e}")

    # Coarse surface estimate from primitives
    rho0, vr, p, eps, W, h, _ = hydro._get_primitives(bssn_vars, grid.r)
    mask_interior = rho0 > 1e-6
    if np.any(mask_interior):
        i_surf = np.where(mask_interior)[0][-1]
        print(f"  estimated stellar surface near r={grid.r[i_surf]:.6f} (i={i_surf})")
        window = slice(max(NUM_GHOSTS, i_surf-5), min(grid.N-NUM_GHOSTS, i_surf+6))
        print("  dS_r/dt in 11-pt window around surface:")
        for ii in range(window.start, window.stop):
            print(f"    i={ii:5d}, r={grid.r[ii]:8.5f}, dS_r/dt={rhs_Sr[ii]: .3e}")
    else:
        print("  WARNING: could not locate interior points above threshold.")


# =============================================================================
# EVOLUTION FUNCTIONS
# =============================================================================

def evolve_fixed_timestep(state_initial, dt, num_steps, grid, background, hydro,
                          bssn_fixed, bssn_d1_fixed, atmosphere, rk4_step_func,
                          method='rk4', t_start=0.0,
                          reference_state=None, step_offset=0, data_manager=None,
                          snapshot_interval=None, evolution_interval=None):
    """Evolve with fixed timestep using RK4.

    Args:
        state_initial: Initial state array
        dt: Timestep
        num_steps: Number of steps to evolve
        grid: Grid object
        background: Background spacetime
        hydro: Hydro object
        bssn_fixed: Fixed BSSN variables (Cowling approximation)
        bssn_d1_fixed: Fixed BSSN derivatives
        atmosphere: AtmosphereParams object
        rk4_step_func: RK4 step function (callable)
        method: Integration method (default: 'rk4')
        t_start: Starting time for this evolution segment (default: 0.0)
        reference_state: Reference state for error calculation (default: state_initial)
                        Use this to maintain consistent error measurement across multiple segments
        step_offset: Offset for step numbering in output (default: 0)
        data_manager: SimulationDataManager object for data saving (optional)
        snapshot_interval: Save full domain snapshot every N steps (optional)
        evolution_interval: Save evolution data every N steps (optional)

    Returns:
        tuple: (final_state, actual_steps, actual_time, series_dict)
    """
    state_flat = state_initial.flatten()

    # Prebuild BSSN container for primitives computation (Cowling)
    bssn_vars_fixed = BSSNVars(grid.N)
    bssn_vars_fixed.set_bssn_vars(bssn_fixed)

    def primitives_from_state(state_flattened):
        """Extract primitive variables from state vector.

        Returns:
            tuple: (rho0, vr, p, eps, W, h, success, state_2d)
        """
        s2d = state_flattened.reshape((grid.NUM_VARS, grid.N))
        hydro.set_matter_vars(s2d, bssn_vars_fixed, grid)
        rho0, vr, p, eps, W, h, success = hydro._get_primitives(bssn_vars_fixed, grid.r)
        return rho0, vr, p, eps, W, h, success, s2d

    # Diagnostics at start
    rho0_prev, vr_prev, p_prev, eps_prev, W_prev, h_prev, success_prev, s_prev = primitives_from_state(state_flat)

    # Store reference state for error calculation (use provided reference or current initial)
    if reference_state is None:
        reference_state = state_initial
    rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial, success_initial, s_initial = primitives_from_state(reference_state.flatten())

    # Save initial snapshot if data manager provided
    if data_manager and data_manager.enable_saving:
        data_manager.save_snapshot(step_offset, t_start, state_initial, rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial)
        data_manager.add_evolution_point(step_offset, t_start, state_initial,
                                        rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial, success_initial,
                                        rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial, success_initial)

    # Timeseries for mass and central density
    times_series = [t_start]
    Mb0 = compute_baryon_mass(grid, s_initial, rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial)
    Mb_series = [Mb0]
    rho_c0 = rho0_initial[NUM_GHOSTS]
    rho_c_series = [rho_c0]


    print("\n===== Evolution diagnostics (per step) =====")
    print("Columns: step | t | ρ_central | max_Δρ/ρ@r | max_vʳ@r | max_Sʳ@r | c2p_fails")
    print("  (@r indicates the radial position where the maximum occurs)")
    print("-" * 140)

    for step in range(num_steps):
        # Advance one RK4 step (with well-balanced correction)
        state_flat_next = rk4_step_func(state_flat, dt, grid, background, hydro,
                                   bssn_fixed, bssn_d1_fixed, atmosphere)

        # Compute primitives BEFORE and AFTER to measure change
        rho0_next, vr_next, p_next, eps_next, W_next, h_next, success_next, s_next = primitives_from_state(state_flat_next)

        # Interior slice (exclude ghosts)
        interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

        rho_next_int = rho0_next[interior]
        rho_init_int = rho0_initial[interior]
        p_next_int = p_next[interior]
        v_next_int = vr_next[interior]

        D_next = s_next[NUM_BSSN_VARS + 0, interior]
        Sr_next = s_next[NUM_BSSN_VARS + 1, interior]
        tau_next = s_next[NUM_BSSN_VARS + 2, interior]

        # Compute more informative stats
        rho_central = float(rho0_next[NUM_GHOSTS])  # Central density

        # Grid radii (interior only, matching slicing above)
        r_interior = grid.r[interior]

        # Maximum relative density error vs initial state
        rel_rho_err = np.abs(rho_next_int - rho_init_int) / (np.abs(rho_init_int) + 1e-20)
        idx_max_rho_err = np.argmax(rel_rho_err)
        max_rel_rho_err = float(rel_rho_err[idx_max_rho_err])
        r_max_rho_err = float(r_interior[idx_max_rho_err])

        # Maximum velocity
        idx_max_v = np.argmax(np.abs(v_next_int))
        max_abs_v = float(v_next_int[idx_max_v])
        r_max_v = float(r_interior[idx_max_v])

        # Maximum conserved variables (more useful than minimum)
        idx_max_D = np.argmax(D_next)
        max_D = float(D_next[idx_max_D])
        r_max_D = float(r_interior[idx_max_D])

        idx_max_Sr = np.argmax(np.abs(Sr_next))
        max_Sr = float(np.abs(Sr_next[idx_max_Sr]))
        r_max_Sr = float(r_interior[idx_max_Sr])

        idx_max_tau = np.argmax(np.abs(tau_next))
        max_tau = float(np.abs(tau_next[idx_max_tau]))
        r_max_tau = float(r_interior[idx_max_tau])

        # Cons2prim failures
        c2p_fail_count = int(np.sum(~success_next))

        t_curr = t_start + (step + 1) * dt
        step_num = step_offset + step + 1
        if step_num % 200 == 0:
            print(f"step {step_num:4d}  t={t_curr:.2e}:  ρ_c={rho_central:.6e}  max_Δρ/ρ={max_rel_rho_err:.2e}@r={r_max_rho_err:.2f}  "
              f"max_vʳ={max_abs_v:.3e}@r={r_max_v:.2f}  max_Sʳ={max_Sr:.2e}@r={r_max_Sr:.2f}  "
              f"c2p_fail={c2p_fail_count}")

        # Save data if requested
        if data_manager and data_manager.enable_saving:
            # Save evolution data at specified interval
            if evolution_interval and step_num % evolution_interval == 0:
                data_manager.add_evolution_point(step_num, t_curr, s_next,
                                                rho0_next, vr_next, p_next, eps_next, W_next, h_next, success_next,
                                                rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial, success_initial)

                # Periodic buffer flush
                if step_num % (evolution_interval * 10) == 0:
                    data_manager.flush_evolution_buffer()

            # Save full snapshot at specified interval
            if snapshot_interval and step_num % snapshot_interval == 0:
                data_manager.save_snapshot(step_num, t_curr, s_next, rho0_next, vr_next, p_next, eps_next, W_next, h_next)

        # Append to time series (mass and central density)
        Mb_next = compute_baryon_mass(grid, s_next, rho0_next, vr_next, p_next, eps_next, W_next, h_next)
        times_series.append(t_curr)
        Mb_series.append(Mb_next)
        rho_c_series.append(float(rho0_next[NUM_GHOSTS]))

        # Detect first signs of instability / non-physical values
        issues = []
        if not np.all(np.isfinite(rho0_next)) or not np.all(np.isfinite(p_next)):
            issues.append("NaN/Inf in primitives")
        if np.any(rho0_next < 0):
            issues.append("negative rho0")
        if np.any(p_next < 0):
            issues.append("negative pressure")
        if np.any(np.abs(vr_next) >= 1.0):
            issues.append("superluminal v")
        if np.any(D_next < 0):
            issues.append("negative D")
        if np.any((tau_next + D_next) < 0):
            issues.append("tau + D < 0")

        # If problems detected, print focused context (location and local values)
        if issues:
            print("  -> Detected issues:", ", ".join(issues))
            # Locate worst offenders
            try:
                idx_v = NUM_GHOSTS + int(np.argmax(np.abs(vr_next[interior])))
            except Exception:
                idx_v = NUM_GHOSTS
            try:
                idx_rho_min = NUM_GHOSTS + int(np.argmin(rho0_next[interior]))
            except Exception:
                idx_rho_min = NUM_GHOSTS
            try:
                idx_tauD_min = NUM_GHOSTS + int(np.argmin((s_next[NUM_BSSN_VARS+2, interior] + s_next[NUM_BSSN_VARS+0, interior])))
            except Exception:
                idx_tauD_min = NUM_GHOSTS

            idxs = sorted(set([idx_v, idx_rho_min, idx_tauD_min]))
            for ii in idxs:
                rloc = grid.r[ii]
                print(f"     at r={rloc:.6f} (i={ii}): "
                      f"rho0={rho0_next[ii]:.6e}, P={p_next[ii]:.6e}, vr={vr_next[ii]:.6e}, "
                      f"D={s_next[NUM_BSSN_VARS+0, ii]:.6e}, Sr={s_next[NUM_BSSN_VARS+1, ii]:.6e}, tau={s_next[NUM_BSSN_VARS+2, ii]:.6e}")

            # Stop early so we can inspect before blow-up cascades
            print("  -> Halting evolution early due to detected instability.")
            state_flat = state_flat_next  # return the last state
            actual_steps = step + 1
            actual_time = t_start + actual_steps * dt
            return state_flat.reshape((grid.NUM_VARS, grid.N)), actual_steps, actual_time, {
                't': np.array(times_series),
                'Mb': np.array(Mb_series),
                'rho_c': np.array(rho_c_series),
            }

        # Prepare next step
        state_flat = state_flat_next


    # Final flush of buffers if data manager is provided
    if data_manager and data_manager.enable_saving:
        data_manager.flush_evolution_buffer()

    actual_time = t_start + num_steps * dt
    return state_flat.reshape((grid.NUM_VARS, grid.N)), num_steps, actual_time, {
        't': np.array(times_series),
        'Mb': np.array(Mb_series),
        'rho_c': np.array(rho_c_series),
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_tov_diagnostics(tov_solution, r_max, suffix=""):
    """Plot TOV solution diagnostics."""
    # Handle both TOVSolution objects and dicts
    if hasattr(tov_solution, 'r'):
        r = tov_solution.r
        R_star = tov_solution.R
        M_star = tov_solution.M_star
        rho_baryon = tov_solution.rho_baryon
        P = tov_solution.P
        M = tov_solution.M
        alpha = tov_solution.alpha
        exp4phi = tov_solution.exp4phi
    else:
        r = tov_solution['r']
        R_star = tov_solution['R']
        M_star = tov_solution['M_star']
        rho_baryon = tov_solution['rho_baryon']
        P = tov_solution['P']
        M = tov_solution['M']
        alpha = tov_solution['alpha']
        exp4phi = tov_solution['exp4phi']

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Density
    axes[0, 0].plot(r, rho_baryon, color='navy')
    axes[0, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5, label=f'R={R_star:.2f}')
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel('rho_0')
    axes[0, 0].set_title('Baryon Density')
    axes[0, 0].set_xlim(0, r_max)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Pressure
    axes[0, 1].plot(r, P, color='darkgreen')
    axes[0, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure')
    axes[0, 1].set_xlim(0, r_max)
    axes[0, 1].grid(True, alpha=0.3)

    # Enclosed Mass
    axes[0, 2].plot(r, M, color='maroon')
    axes[0, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 2].axhline(M_star, color='gray', linestyle='--', alpha=0.3, label=f'M={M_star:.3f}')
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel('M(r)')
    axes[0, 2].set_title('Enclosed Mass')
    axes[0, 2].set_xlim(0, r_max)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Lapse alpha(r)
    axes[1, 0].plot(r, alpha, color='purple')
    axes[1, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel('alpha')
    axes[1, 0].set_title('Lapse Function')
    axes[1, 0].set_xlim(0, r_max)
    axes[1, 0].grid(True, alpha=0.3)

    # Phi(r)
    phi = 0.25 * np.log(exp4phi)
    axes[1, 1].plot(r, phi, color='teal')
    axes[1, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel('phi')
    axes[1, 1].set_title('Conformal Factor phi')
    axes[1, 1].set_xlim(0, r_max)
    axes[1, 1].grid(True, alpha=0.3)

    # a(r) metric function
    a_metric = np.sqrt(exp4phi)
    axes[1, 2].plot(r, a_metric, color='orange')
    axes[1, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel('a(r)')
    axes[1, 2].set_title('Metric a(r)')
    axes[1, 2].set_xlim(0, r_max)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(plots_dir, f'tov_solution{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_first_step(state_t0, state_t1, grid, hydro, tov_solution=None, suffix=""):
    """Plot t=0 vs t=1×dt to inspect the first update."""
    bssn_0 = BSSNVars(grid.N)
    bssn_0.set_bssn_vars(state_t0[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t0, bssn_0, grid)
    rho0_0, vr_0, p_0, eps_0, W_0, h_0, _ = hydro._get_primitives(bssn_0, grid.r)

    bssn_1 = BSSNVars(grid.N)
    bssn_1.set_bssn_vars(state_t1[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t1, bssn_1, grid)
    rho0_1, vr_1, p_1, eps_1, W_1, h_1, _ = hydro._get_primitives(bssn_1, grid.r)

    r = grid.r
    r_int = r[NUM_GHOSTS:-NUM_GHOSTS]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: Primitives
    axes[0, 0].plot(r, rho0_0, 'b-', label='t=0', linewidth=1.5)
    axes[0, 0].plot(r, rho0_1, 'r--', label='t=dt', linewidth=1.5)
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel('rho0')
    axes[0, 0].set_title('Baryon Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(r, p_0, 'b-', label='t=0', linewidth=1.5)
    axes[0, 1].plot(r, p_1, 'r--', label='t=dt', linewidth=1.5)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(r, vr_0, 'b-', label='t=0', linewidth=1.5)
    axes[0, 2].plot(r, vr_1, 'r--', label='t=dt', linewidth=1.5)
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel('v^r')
    axes[0, 2].set_title('Radial Velocity')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Conservative variables
    D_0 = state_t0[hydro.idx_D, :]
    D_1 = state_t1[hydro.idx_D, :]
    Sr_0 = state_t0[hydro.idx_Sr, :]
    Sr_1 = state_t1[hydro.idx_Sr, :]
    tau_0 = state_t0[hydro.idx_tau, :]
    tau_1 = state_t1[hydro.idx_tau, :]

    axes[1, 0].plot(r, D_0, 'b-', label='t=0', linewidth=1.5)
    axes[1, 0].plot(r, D_1, 'r--', label='t=dt', linewidth=1.5)
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel('D')
    axes[1, 0].set_title('Conserved D')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(r, Sr_0, 'b-', label='t=0', linewidth=1.5)
    axes[1, 1].plot(r, Sr_1, 'r--', label='t=dt', linewidth=1.5)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel('Sr')
    axes[1, 1].set_title('Conserved Sr')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(r, tau_0, 'b-', label='t=0', linewidth=1.5)
    axes[1, 2].plot(r, tau_1, 'r--', label='t=dt', linewidth=1.5)
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel('tau')
    axes[1, 2].set_title('Conserved tau')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(plots_dir, f'tov_first_step{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_center_zoom(state_t0, state_t1, grid, hydro, window=0.5, suffix=""):
    """Plot zoom near origin comparing t=0 vs t=dt.

    Args:
        state_t0: State at t=0
        state_t1: State at t=dt
        grid: Grid object
        hydro: Hydro object
        window: Maximum radius to show (default 0.5)
        suffix: Suffix for output filename
    """
    bssn_0 = BSSNVars(grid.N)
    bssn_0.set_bssn_vars(state_t0[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t0, bssn_0, grid)
    rho0_0, vr_0, p_0, eps_0, W_0, h_0, _ = hydro._get_primitives(bssn_0, grid.r)

    bssn_1 = BSSNVars(grid.N)
    bssn_1.set_bssn_vars(state_t1[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t1, bssn_1, grid)
    rho0_1, vr_1, p_1, eps_1, W_1, h_1, _ = hydro._get_primitives(bssn_1, grid.r)

    r = grid.r

    # Conservative variables
    D_0 = state_t0[hydro.idx_D, :]
    D_1 = state_t1[hydro.idx_D, :]
    Sr_0 = state_t0[hydro.idx_Sr, :]
    Sr_1 = state_t1[hydro.idx_Sr, :]
    tau_0 = state_t0[hydro.idx_tau, :]
    tau_1 = state_t1[hydro.idx_tau, :]

    # Create mask for window
    mask = r <= window

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: Primitives
    axes[0, 0].plot(r[mask], rho0_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[0, 0].plot(r[mask], rho0_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel(r'$\rho_0$')
    axes[0, 0].set_title('Baryon Density (center)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(r[mask], p_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[0, 1].plot(r[mask], p_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure (center)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(r[mask], vr_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[0, 2].plot(r[mask], vr_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel(r'$v^r$')
    axes[0, 2].set_title('Radial Velocity (center)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Conservative variables
    axes[1, 0].plot(r[mask], D_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[1, 0].plot(r[mask], D_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel('D')
    axes[1, 0].set_title('Conserved D (center)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(r[mask], Sr_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[1, 1].plot(r[mask], Sr_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel(r'$S_r$')
    axes[1, 1].set_title('Conserved Sr (center)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(r[mask], tau_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[1, 2].plot(r[mask], tau_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel(r'$\tau$')
    axes[1, 2].set_title('Conserved tau (center)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f'Center Zoom: r ∈ [0, {window}]', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(plots_dir, f'tov_center_zoom{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_surface_zoom(tov_solution, state_t0, state_t1, grid, hydro, primitives_t0=None, window=0.5, suffix=""):
    """Plot zoom near stellar surface."""
    # Handle both TOVSolution objects and dicts
    if hasattr(tov_solution, 'R'):
        R_star = tov_solution.R
    else:
        R_star = tov_solution['R']

    bssn_0 = BSSNVars(grid.N)
    bssn_0.set_bssn_vars(state_t0[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t0, bssn_0, grid)
    rho0_0, vr_0, p_0, eps_0, W_0, h_0, _ = hydro._get_primitives(bssn_0, grid.r)

    bssn_1 = BSSNVars(grid.N)
    bssn_1.set_bssn_vars(state_t1[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t1, bssn_1, grid)
    rho0_1, vr_1, p_1, eps_1, W_1, h_1, _ = hydro._get_primitives(bssn_1, grid.r)

    r = grid.r

    # Create mask for window around surface
    r_min = R_star - window
    r_max = R_star + window
    mask = (r >= r_min) & (r <= r_max)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: Primitives
    axes[0, 0].plot(r[mask], rho0_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[0, 0].plot(r[mask], rho0_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[0, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 0].set_yscale('log')
    
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel(r'$\rho_0$')
    axes[0, 0].set_title('Baryon Density (surface)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(r[mask], p_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[0, 1].plot(r[mask], p_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[0, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Pressure (surface)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(r[mask], vr_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[0, 2].plot(r[mask], vr_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[0, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel(r'$v^r$')
    axes[0, 2].set_title('Radial Velocity (surface)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Conservative variables
    D_0 = state_t0[hydro.idx_D, :]
    D_1 = state_t1[hydro.idx_D, :]
    Sr_0 = state_t0[hydro.idx_Sr, :]
    Sr_1 = state_t1[hydro.idx_Sr, :]
    tau_0 = state_t0[hydro.idx_tau, :]
    tau_1 = state_t1[hydro.idx_tau, :]

    axes[1, 0].plot(r[mask], D_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[1, 0].plot(r[mask], D_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[1, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel('D')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Conserved D (surface)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(r[mask], Sr_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[1, 1].plot(r[mask], Sr_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[1, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel(r'$S_r$')
    axes[1, 1].set_title('Conserved Sr (surface)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(r[mask], tau_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[1, 2].plot(r[mask], tau_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[1, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel(r'$\tau$')
    axes[1, 2].set_yscale('log')
    axes[1, 2].set_title('Conserved tau (surface)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f'Surface Zoom: r ∈ [{r_min:.2f}, {r_max:.2f}], R*={R_star:.2f}', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(plots_dir, f'tov_surface_zoom{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_bssn_evolution(state_t0, state_tfinal, grid, t_0=0.0, t_final=1.0, suffix=""):
    """Plot BSSN variables at t=0 vs t=final to verify Cowling approximation.

    Uses proper BSSN variable indices:
    - idx_phi = 0: conformal factor
    - idx_hrr = 1: metric deviation h_rr
    - idx_K = 4: mean curvature
    - idx_shiftr = 9: shift beta^r
    - idx_br = 10: B^r auxiliary
    - idx_lapse = 11: lapse alpha
    """
    r = grid.r

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # phi (conformal factor)
    axes[0, 0].plot(r, state_t0[idx_phi, :], 'b-', label='t=0', linewidth=1.5)
    axes[0, 0].plot(r, state_tfinal[idx_phi, :], 'r--', label=f't={t_final}', linewidth=1.5)
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel(r'$\phi$')
    axes[0, 0].set_title('Conformal Factor φ')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # h_rr (metric deviation)
    axes[0, 1].plot(r, state_t0[idx_hrr, :], 'b-', label='t=0', linewidth=1.5)
    axes[0, 1].plot(r, state_tfinal[idx_hrr, :], 'r--', label=f't={t_final}', linewidth=1.5)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel(r'$h_{rr}$')
    axes[0, 1].set_title('Metric Deviation h_rr')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # alpha (lapse) - CORRECTED: use idx_lapse
    axes[0, 2].plot(r, state_t0[idx_lapse, :], 'b-', label='t=0', linewidth=1.5)
    axes[0, 2].plot(r, state_tfinal[idx_lapse, :], 'r--', label=f't={t_final}', linewidth=1.5)
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel(r'$\alpha$')
    axes[0, 2].set_title('Lapse α')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # beta^r (shift) - CORRECTED: use idx_shiftr
    axes[1, 0].plot(r, state_t0[idx_shiftr, :], 'b-', label='t=0', linewidth=1.5)
    axes[1, 0].plot(r, state_tfinal[idx_shiftr, :], 'r--', label=f't={t_final}', linewidth=1.5)
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel(r'$\beta^r$')
    axes[1, 0].set_title('Shift βʳ')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # B^r (shift auxiliary) - CORRECTED: use idx_br
    axes[1, 1].plot(r, state_t0[idx_br, :], 'b-', label='t=0', linewidth=1.5)
    axes[1, 1].plot(r, state_tfinal[idx_br, :], 'r--', label=f't={t_final}', linewidth=1.5)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel(r'$B^r$')
    axes[1, 1].set_title('Shift Auxiliary Bʳ')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # K (mean curvature) - CORRECTED: use idx_K
    axes[1, 2].plot(r, state_t0[idx_K, :], 'b-', label='t=0', linewidth=1.5)
    axes[1, 2].plot(r, state_tfinal[idx_K, :], 'r--', label=f't={t_final}', linewidth=1.5)
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel('K')
    axes[1, 2].set_title('Extrinsic Curvature K')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle('BSSN Variables: Cowling Check (should be identical)', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(plots_dir, f'tov_bssn_evolution{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_mass_and_central_density(times, Mb_series, rho_c_series, suffix=""):
    """Plot baryon mass and central density evolution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mass deviation
    Mb_0 = Mb_series[0]
    delta_Mb = np.array(Mb_series) - Mb_0

    axes[0].semilogy(times, np.abs(delta_Mb) + 1e-20, 'b-', linewidth=1.5)
    axes[0].set_xlabel('t')
    axes[0].set_ylabel(r'$|M_b - M_{b,0}|$')
    axes[0].set_title('Baryon Mass Deviation')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')   

    # Central density deviation
    rho_c_0 = rho_c_series[0]
    delta_rho_c = (np.array(rho_c_series) - rho_c_0) / rho_c_0

    axes[1].plot(times, delta_rho_c, 'r-', linewidth=1.5)
    axes[1].set_xlabel('t')
    axes[1].set_ylabel(r'$(\rho_c - \rho_{c,0})/\rho_{c,0}$')
    axes[1].set_title('Central Density Relative Change')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()
    out_path = os.path.join(plots_dir, f'tov_mass_central_density{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_evolution(states, times, grid, hydro, rho_ref, p_ref,
                   Mb_series=None, rho_c_series=None, suffix="", R_star=None):
    """Plot evolution at multiple checkpoints with 3-checkpoint structure.

    Args:
        states: List of state arrays at different times [state_t0, state_t1, state_t2, state_tfinal]
        times: List of times corresponding to states [0, t1, t2, tfinal]
        grid: Grid object
        hydro: Hydro object
        rho_ref: Reference density for error computation
        p_ref: Reference pressure for error computation
        Mb_series: Optional time series of baryon mass
        rho_c_series: Optional time series of central density
        suffix: Suffix for output filename
        R_star: Optional stellar radius for vertical line marker
    """
    n_states = len(states)
    r = grid.r
    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

    # Extract primitives for each state
    primitives = []
    for state in states:
        bssn = BSSNVars(grid.N)
        bssn.set_bssn_vars(state[:NUM_BSSN_VARS, :])
        hydro.set_matter_vars(state, bssn, grid)
        rho0, vr, p, eps, W, h, _ = hydro._get_primitives(bssn, grid.r)
        primitives.append((rho0, vr, p, eps, W, h))

    # Create figure with 3 rows
    fig = plt.figure(figsize=(16, 14))

    # Row 1: Density at all checkpoints
    ax1 = plt.subplot(3, 2, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, n_states))
    for i, (prim, t) in enumerate(zip(primitives, times)):
        ax1.plot(r[interior], prim[0][interior], color=colors[i],
                label=f't={t:.2f}', linewidth=1.5)
    if R_star is not None:
        ax1.axvline(R_star, color='gray', linestyle=':', alpha=0.7, label=f'R={R_star:.2f}')
    ax1.set_xlabel('r')
    ax1.set_ylabel(r'$\rho_0$')
    ax1.set_title('Baryon Density Evolution')
    ax1.set_yscale('log')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Row 1: Pressure at all checkpoints
    ax2 = plt.subplot(3, 2, 2)
    for i, (prim, t) in enumerate(zip(primitives, times)):
        ax2.plot(r[interior], prim[2][interior], color=colors[i],
                label=f't={t:.2f}', linewidth=1.5)
    if R_star is not None:
        ax2.axvline(R_star, color='gray', linestyle=':', alpha=0.7, label=f'R={R_star:.2f}')
    ax2.set_xlabel('r')
    ax2.set_ylabel('P')
    ax2.set_title('Pressure Evolution')
    ax2.set_yscale('log')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Row 2: Velocity at all checkpoints
    ax3 = plt.subplot(3, 2, 3)
    for i, (prim, t) in enumerate(zip(primitives, times)):
        ax3.plot(r[interior], prim[1][interior], color=colors[i],
                label=f't={t:.2f}', linewidth=1.5)
    if R_star is not None:
        ax3.axvline(R_star, color='gray', linestyle=':', alpha=0.7, label=f'R={R_star:.2f}')
    ax3.set_xlabel('r')
    ax3.set_ylabel(r'$v^r$')
    ax3.set_title('Radial Velocity Evolution')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Row 2: Density error
    ax4 = plt.subplot(3, 2, 4)
    for i, (prim, t) in enumerate(zip(primitives, times)):
        if i == 0:
            continue  # Skip t=0 for error
        delta_rho = np.abs(prim[0][interior] - rho_ref[interior])
        rel_error = delta_rho / (np.abs(rho_ref[interior]) + 1e-20)
        ax4.semilogy(r[interior], rel_error + 1e-20, color=colors[i],
                    label=f't={t:.2f}', linewidth=1.5)
    if R_star is not None:
        ax4.axvline(R_star, color='gray', linestyle=':', alpha=0.7, label=f'R={R_star:.2f}')
    ax4.set_xlabel('r')
    ax4.set_ylabel(r'$|\Delta\rho|/\rho_0$')
    ax4.set_title('Relative Density Error')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    # Row 3: Time series (if available)
    if Mb_series is not None and len(Mb_series) > 1:
        times_series = np.linspace(0, times[-1], len(Mb_series))

        ax5 = plt.subplot(3, 2, 5)
        Mb_0 = Mb_series[0]
        delta_Mb = np.abs(np.array(Mb_series) - Mb_0)
        ax5.semilogy(times_series, delta_Mb + 1e-20, 'b-', linewidth=1.5)
        ax5.set_xlabel('t')
        ax5.set_ylabel(r'$|M_b - M_{b,0}|$')
        ax5.set_title('Baryon Mass Deviation')
        ax5.grid(True, alpha=0.3)

    if rho_c_series is not None and len(rho_c_series) > 1:
        times_series = np.linspace(0, times[-1], len(rho_c_series))

        ax6 = plt.subplot(3, 2, 6)
        rho_c_0 = rho_c_series[0]
        delta_rho_c = (np.array(rho_c_series) - rho_c_0) / rho_c_0
        ax6.plot(times_series, delta_rho_c, 'r-', linewidth=1.5)
        ax6.set_xlabel('t')
        ax6.set_ylabel(r'$(\rho_c - \rho_{c,0})/\rho_{c,0}$')
        ax6.set_title('Central Density Relative Change')
        ax6.grid(True, alpha=0.3)

    plt.suptitle(f'TOV Evolution: 3-Checkpoint Structure', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(plots_dir, f'tov_evolution{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    #plt.close(fig)
    plt.show()
    print(f"Saved: {out_path}")


def compute_baryon_mass(grid, state, rho0, vr, p, eps, W, h):
    """Compute baryon (rest) mass M = ∫ ρ0 W √γ d^3x = 4π ∫ ρ0 W ψ^6 r^2 dr.

    Uses interior points (excludes ghost zones).

    Args:
        grid: Grid object
        state: State vector (NUM_VARS x N)
        rho0, vr, p, eps, W, h: Primitive variables
    """
    from scipy.integrate import simpson
    interior = slice(NUM_GHOSTS, grid.N - NUM_GHOSTS)
    r = grid.r[interior]
    rho0_int = rho0[interior]
    W_int = W[interior]

    phi = state[idx_phi, interior]
    psi = np.exp(phi)
    integrand = rho0_int * W_int * (psi**6) * (r**2)
    return 4.0 * np.pi * simpson(integrand, x=r)


def plot_tov_vs_initial_data_zoom(tov_solution, initial_state_2d, grid, primitives, window=0.5, suffix=""):
    """Zoom near the stellar surface R to compare TOV solution vs interpolated initial data.

    This plot helps identify interpolation errors and differences between the analytic
    TOV solution and the discretized initial data on the evolution grid.

    Plots overlays for (ρ0, P, v^r, D, S_r, τ) in a window [R−window, R+window].

    Args:
        tov_solution: TOVSolution object
        initial_state_2d: Initial state array
        grid: Grid object
        primitives: Tuple (rho0, vr, p, eps) from initial data
        window: Half-width of zoom window around stellar surface
        suffix: Suffix for output filename
    """
    from scipy.interpolate import interp1d

    # Handle both TOVSolution objects and dicts
    if hasattr(tov_solution, 'R'):
        R = float(tov_solution.R)
        r_tov = tov_solution.r
        rho_tov = tov_solution.rho_baryon
        P_tov = tov_solution.P
    else:
        R = float(tov_solution['R'])
        r_tov = tov_solution['r']
        rho_tov = tov_solution['rho_baryon']
        P_tov = tov_solution['P']

    r = grid.r
    mask = (r >= R - window) & (r <= R + window)
    if not np.any(mask):
        return

    # Unpack primitives tuple
    rho0_init, vr_init, p_init, eps_init = primitives

    # Get conservatives from initial data
    D_init = initial_state_2d[NUM_BSSN_VARS + 0, :]
    Sr_init = initial_state_2d[NUM_BSSN_VARS + 1, :]
    tau_init = initial_state_2d[NUM_BSSN_VARS + 2, :]

    # Interpolate TOV solution to zoom region
    rho_tov_interp = interp1d(r_tov, rho_tov, kind='cubic', fill_value=0.0, bounds_error=False)
    P_tov_interp = interp1d(r_tov, P_tov, kind='cubic', fill_value=0.0, bounds_error=False)

    rho_tov_zoom = rho_tov_interp(r)
    P_tov_zoom = P_tov_interp(r)

    rZ = r[mask]
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))

    # Row 1: ρ0, P, v^r
    ax[0, 0].semilogy(rZ, np.maximum(rho_tov_zoom[mask], 1e-20), 'k-', linewidth=2, label='TOV (analytic)')
    ax[0, 0].semilogy(rZ, np.maximum(rho0_init[mask], 1e-20), 'b--', linewidth=1.5, label='Initial Data')
    ax[0, 0].axvline(R, color='gray', ls=':', linewidth=1.5, label=f'R={R:.3f}')
    ax[0, 0].set_title('rho_0 (zoom near surface)', fontsize=11)
    ax[0, 0].legend(fontsize=9)
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].set_ylabel('rho_0', fontsize=10)

    ax[0, 1].semilogy(rZ, np.maximum(P_tov_zoom[mask], 1e-20), 'k-', linewidth=2)
    ax[0, 1].semilogy(rZ, np.maximum(p_init[mask], 1e-20), 'b--', linewidth=1.5)
    ax[0, 1].axvline(R, color='gray', ls=':', linewidth=1.5)
    ax[0, 1].set_title('P (zoom near surface)', fontsize=11)
    ax[0, 1].grid(True, alpha=0.3)
    ax[0, 1].set_ylabel('P', fontsize=10)

    # v^r should be zero in TOV equilibrium
    ax[0, 2].plot(rZ, vr_init[mask], 'b-', linewidth=1.5, label='Initial Data')
    ax[0, 2].axhline(0, color='k', ls='--', linewidth=2, label='TOV (v=0)')
    ax[0, 2].axvline(R, color='gray', ls=':', linewidth=1.5)
    ax[0, 2].set_title('v^r (zoom near surface)', fontsize=11)
    ax[0, 2].legend(fontsize=9)
    ax[0, 2].grid(True, alpha=0.3)
    ax[0, 2].set_ylabel('v^r', fontsize=10)

    # Row 2: D, Sr, tau (conservative variables)
    ax[1, 0].semilogy(rZ, np.maximum(D_init[mask], 1e-22), 'b-', linewidth=1.5)
    ax[1, 0].axvline(R, color='gray', ls=':', linewidth=1.5)
    ax[1, 0].set_title('D (conserved density)', fontsize=11)
    ax[1, 0].grid(True, alpha=0.3)
    ax[1, 0].set_ylabel('D', fontsize=10)

    ax[1, 1].plot(rZ, Sr_init[mask], 'b-', linewidth=1.5)
    ax[1, 1].axhline(0, color='k', ls='--', linewidth=1, label='Expected (S^r=0)')
    ax[1, 1].axvline(R, color='gray', ls=':', linewidth=1.5)
    ax[1, 1].set_title('S_r (conserved momentum)', fontsize=11)
    ax[1, 1].legend(fontsize=9)
    ax[1, 1].grid(True, alpha=0.3)
    ax[1, 1].set_ylabel('S_r', fontsize=10)

    ax[1, 2].semilogy(rZ, np.maximum(np.abs(tau_init[mask]), 1e-22), 'b-', linewidth=1.5)
    ax[1, 2].axvline(R, color='gray', ls=':', linewidth=1.5)
    ax[1, 2].set_title('tau (conserved energy)', fontsize=11)
    ax[1, 2].grid(True, alpha=0.3)
    ax[1, 2].set_ylabel('|tau|', fontsize=10)

    for a in ax.ravel():
        a.set_xlabel('r [M]', fontsize=10)

    plt.suptitle(f'TOV Solution vs Initial Data: Surface Zoom [R-{window}, R+{window}]',
                 fontsize=13, y=0.995)
    plt.tight_layout()
    out_path = os.path.join(plots_dir, f'tov_vs_initial_zoom{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def create_center_zoom_video(snapshot_file, output_path=None, window=0.5, fps=10, suffix=""):
    """Create a video/gif of the center zoom evolution from HDF5 snapshots.

    This function reads snapshots from an HDF5 file and creates an animated
    visualization showing the evolution of primitives and conservatives near
    the center (including ghost cells).

    Args:
        snapshot_file: Path to HDF5 snapshot file
        output_path: Output path for the video. If None, uses default in plots_dir
        window: Maximum radius to show (default 0.5)
        fps: Frames per second for the video (default 10)
        suffix: Suffix for output filename
    """
    import matplotlib.animation as animation

    with h5py.File(snapshot_file, 'r') as f:
        r = f['grid/r'][:]
        snapshots = sorted(f['snapshots'].keys())
        n_frames = len(snapshots)

        if n_frames < 2:
            print("Not enough snapshots to create video")
            return

        # Read initial state (t=0) for reference
        snap0 = f['snapshots'][snapshots[0]]
        rho0_ref = snap0['primitives/rho0'][:]
        p_ref = snap0['primitives/p'][:]
        vr_ref = snap0['primitives/vr'][:]
        D_ref = snap0['conservatives/D'][:]
        Sr_ref = snap0['conservatives/Sr'][:]
        tau_ref = snap0['conservatives/tau'][:]

        # Pre-load all data
        times, steps = [], []
        rho0_all, p_all, vr_all = [], [], []
        D_all, Sr_all, tau_all = [], [], []

        for snap_name in snapshots:
            snap = f['snapshots'][snap_name]
            times.append(snap.attrs['time'])
            steps.append(snap.attrs['step'])
            rho0_all.append(snap['primitives/rho0'][:])
            p_all.append(snap['primitives/p'][:])
            vr_all.append(snap['primitives/vr'][:])
            D_all.append(snap['conservatives/D'][:])
            Sr_all.append(snap['conservatives/Sr'][:])
            tau_all.append(snap['conservatives/tau'][:])

    # Mask for window (including ghost cells)
    mask = r <= window

    # Set up figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    lines_curr = []

    # Row 1: Primitives
    axes[0, 0].plot(r[mask], rho0_ref[mask], 'b-', label='t=0', linewidth=1.5)
    l, = axes[0, 0].plot(r[mask], rho0_ref[mask], 'r--', label='t=?', linewidth=1.5)
    axes[0, 0].set_xlabel('r'); axes[0, 0].set_ylabel(r'$\rho_0$')
    axes[0, 0].set_title('Baryon Density (center)'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    lines_curr.append(l)

    axes[0, 1].plot(r[mask], p_ref[mask], 'b-', label='t=0', linewidth=1.5)
    l, = axes[0, 1].plot(r[mask], p_ref[mask], 'r--', label='t=?', linewidth=1.5)
    axes[0, 1].set_xlabel('r'); axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure (center)'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    lines_curr.append(l)

    axes[0, 2].plot(r[mask], vr_ref[mask], 'b-', label='t=0', linewidth=1.5)
    l, = axes[0, 2].plot(r[mask], vr_ref[mask], 'r--', label='t=?', linewidth=1.5)
    axes[0, 2].set_xlabel('r'); axes[0, 2].set_ylabel(r'$v^r$')
    axes[0, 2].set_title('Radial Velocity (center)'); axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)
    lines_curr.append(l)

    # Row 2: Conservatives
    axes[1, 0].plot(r[mask], D_ref[mask], 'b-', label='t=0', linewidth=1.5)
    l, = axes[1, 0].plot(r[mask], D_ref[mask], 'r--', label='t=?', linewidth=1.5)
    axes[1, 0].set_xlabel('r'); axes[1, 0].set_ylabel('D')
    axes[1, 0].set_title('Conserved D (center)'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    lines_curr.append(l)

    axes[1, 1].plot(r[mask], Sr_ref[mask], 'b-', label='t=0', linewidth=1.5)
    l, = axes[1, 1].plot(r[mask], Sr_ref[mask], 'r--', label='t=?', linewidth=1.5)
    axes[1, 1].set_xlabel('r'); axes[1, 1].set_ylabel(r'$S_r$')
    axes[1, 1].set_title('Conserved Sr (center)'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
    lines_curr.append(l)

    axes[1, 2].plot(r[mask], tau_ref[mask], 'b-', label='t=0', linewidth=1.5)
    l, = axes[1, 2].plot(r[mask], tau_ref[mask], 'r--', label='t=?', linewidth=1.5)
    axes[1, 2].set_xlabel('r'); axes[1, 2].set_ylabel(r'$\tau$')
    axes[1, 2].set_title('Conserved tau (center)'); axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)
    lines_curr.append(l)

    title = fig.suptitle(f'Center Zoom: r ∈ [0, {window}]  |  t=0.00, step=0', fontsize=14)
    plt.tight_layout()

    def animate(frame):
        t, step = times[frame], steps[frame]
        data = [rho0_all[frame], p_all[frame], vr_all[frame],
                D_all[frame], Sr_all[frame], tau_all[frame]]
        for i, line in enumerate(lines_curr):
            line.set_ydata(data[i][mask])
            line.set_label(f't={t:.2f}')
        for ax in axes.flatten():
            ax.legend()
            ax.relim()
            ax.autoscale_view()
        title.set_text(f'Center Zoom: r ∈ [0, {window}]  |  t={t:.2f}, step={step}')
        return lines_curr

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000//fps, blit=False)

    if output_path is None:
        output_path = os.path.join(plots_dir, f'tov_center_zoom_evolution{suffix}.gif')

    print(f"Creating animation with {n_frames} frames...")
    writer = 'pillow' if output_path.endswith('.gif') else 'ffmpeg'
    anim.save(output_path, writer=writer, fps=fps)
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path
