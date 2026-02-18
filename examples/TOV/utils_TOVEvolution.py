"""
TOV Evolution Utilities

This module contains utilities for TOV evolution simulations:
- Data management (SimulationDataManager)
- Diagnostic functions
- Plotting functions (all visualization)
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, simpson, trapezoid
from scipy.interpolate import interp1d
import h5py
import json
from datetime import datetime

# Add repository root to path to import source modules
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels: TOV -> examples -> repo_root
sys.path.insert(0, repo_root)

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

def get_star_folder_name(rho_central, num_points, K, Gamma, evolution_mode="cowling", reconstructor="wenoz"):
    """
    Generate folder name based on star parameters, evolution mode, and reconstructor.

    Format: tov_star_rhoc{rho}_N{N}_K{K}_G{Gamma}_{mode}_{recon}

    Args:
        rho_central: Central density (e.g., 1.28e-3)
        num_points: Number of grid points
        K: Polytropic constant
        Gamma: Adiabatic index
        evolution_mode: "cowling" or "dynamic"
        reconstructor: "wenoz", "weno5", "mp5", or "minmod"

    Returns:
        str: Folder name like "tov_star_rhoc1p28em3_N100_K100_G2_cow_wz"
    """
    # Format rho_central: 1.28e-3 -> "1p28em3"
    rho_str = f"{rho_central:.2e}"
    # Replace '.' with 'p', remove '+', replace '-' with 'm'
    rho_str = rho_str.replace(".", "p").replace("+", "").replace("-", "m")

    # Format K and Gamma (as integers if they are whole numbers)
    K_str = str(int(K)) if K == int(K) else f"{K:.1f}".replace(".", "p")
    G_str = str(int(Gamma)) if Gamma == int(Gamma) else f"{Gamma:.1f}".replace(".", "p")

    # Mode suffix: cow for cowling, dyn for dynamic
    mode_suffix = "cow" if evolution_mode == "cowling" else "dyn"

    # Reconstructor suffix mapping
    recon_map = {
        "wenoz": "wz",
        "weno5": "w5",
        "mp5": "mp5",
        "minmod": "md"
    }
    recon_suffix = recon_map.get(reconstructor.lower(), "wz")

    return f"tov_star_rhoc{rho_str}_N{num_points}_K{K_str}_G{G_str}_{mode_suffix}_{recon_suffix}"


class SimulationDataManager:
    """Manages data storage for long TOV simulations."""

    def __init__(self, output_dir, grid, hydro, enable_saving=False, suffix="",
                 restart_mode=False):
        """
        Initialize data manager.

        Args:
            output_dir: Directory for output files (should be the star-specific folder)
            grid: Grid object
            hydro: Hydro object
            enable_saving: If True, saves data to files
            suffix: Suffix for file naming (e.g., "_iso" for isotropic coordinates)
            restart_mode: If True, append to existing files instead of overwriting
        """
        self.enable_saving = enable_saving
        self.suffix = suffix
        self.restart_mode = restart_mode
        if not self.enable_saving:
            return

        self.output_dir = output_dir
        self.grid = grid
        self.hydro = hydro

        # Create output directory (overwrites if exists)
        os.makedirs(output_dir, exist_ok=True)

        # Initialize HDF5 files (no timestamp - will overwrite previous runs)
        self.snapshot_file = os.path.join(output_dir, f"tov_snapshots{suffix}.h5")
        self.evolution_file = os.path.join(output_dir, f"tov_evolution{suffix}.h5")
        self.metadata_file = os.path.join(output_dir, f"tov_metadata{suffix}.json")

        # Only remove old files if NOT in restart mode
        if not restart_mode:
            for f in [self.snapshot_file, self.evolution_file, self.metadata_file]:
                if os.path.exists(f):
                    os.remove(f)

        # Initialize evolution data lists (for buffering)
        self.evolution_buffer = {
            'step': [],
            'time': [],
            'rho_central': [],
            'v_central': [],
            'p_central': [],
            'baryon_mass': [],
            'max_rho_error': [],
            'max_p_error': [],
            'l1_rho_error': [],
            'l2_rho_error': [],
            'max_D': [],
            'max_Sr': [],
            'max_tau': [],
            # Constraint violations
            'max_Ham': [],
            'l2_Ham': [],
            'max_Mom_r': [],
            'l2_Mom_r': []
        }

        # Initialize HDF5 files (only if not restart mode)
        if not restart_mode:
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

    def save_metadata(self, tov_solution, atmosphere_params, dt, integration_method,
                      K=None, Gamma=None, rho_central=None, r_max=None, num_points=None,
                      t_final=None, reconstructor=None, solver_method=None,
                      riemann_solver=None, evolution_mode=None, cfl_factor=None):
        """Save simulation metadata.

        Args:
            tov_solution: TOV solution object or dict
            atmosphere_params: AtmosphereParams object
            dt: Timestep
            integration_method: Integration method name
            K: Polytropic constant
            Gamma: Adiabatic index
            rho_central: Central density
            r_max: Maximum radius of domain
            num_points: Number of grid points
            t_final: Final simulation time
            reconstructor: Reconstruction method name (e.g., "mp5", "wenoz")
            solver_method: Cons2prim solver method (e.g., "newton", "kastaun")
            riemann_solver: Riemann solver name (e.g., "hll", "llf")
            evolution_mode: Evolution mode (e.g., "cowling", "dynamic")
            cfl_factor: CFL number used for timestep
        """
        if not self.enable_saving:
            return

        # Handle both TOVSolution objects and dicts
        if hasattr(tov_solution, 'M_star'):
            M_star = tov_solution.M_star
            R = tov_solution.R_iso
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
                'coordinate_system': 'isotropic',
                'dt': float(dt) if dt is not None else None,
                'integration_method': integration_method,
                'grid_N': int(self.grid.N),
                'grid_r_max': float(self.grid.r[-1])
            },
            'numerical_methods': {
                'reconstructor': reconstructor,
                'solver_method': solver_method,
                'riemann_solver': riemann_solver,
                'evolution_mode': evolution_mode,
                'cfl_factor': float(cfl_factor) if cfl_factor is not None else None
            },
            'configuration': {
                'r_max': float(r_max) if r_max is not None else float(self.grid.r[-1]),
                'num_points': int(num_points) if num_points is not None else int(self.grid.N),
                't_final': float(t_final) if t_final is not None else None
            },
            'timestamp': datetime.now().isoformat()
        }

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def save_snapshot(self, step, time, state_2d, rho0=None, vr=None, p=None, eps=None, W=None, h=None,
                     Ham=None, Mom=None):
        """Save full domain snapshot with optional constraints."""
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

            # Save constraints if provided
            if Ham is not None and Mom is not None:
                const_group = snap_group.create_group('constraints')
                const_group['Ham'] = Ham
                const_group['Mom_r'] = Mom[:, 0]  # Radial component
                const_group['Mom_theta'] = Mom[:, 1]
                const_group['Mom_phi'] = Mom[:, 2]

    def add_evolution_point(self, step, time, state_2d, rho0, vr, p, eps, W, h, success,
                           rho0_ref, vr_ref, p_ref, eps_ref, W_ref, h_ref, success_ref,
                           Ham=None, Mom=None):
        """Add a point to the evolution time series with optional constraints."""
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

        # Compute baryon mass
        M_b = compute_baryon_mass(self.grid, state_2d, rho0, vr, p, eps, W, h)

        # Store in buffer
        self.evolution_buffer['step'].append(step)
        self.evolution_buffer['time'].append(time)
        self.evolution_buffer['rho_central'].append(rho0[NUM_GHOSTS])
        self.evolution_buffer['v_central'].append(vr[NUM_GHOSTS])
        self.evolution_buffer['p_central'].append(p[NUM_GHOSTS])
        self.evolution_buffer['baryon_mass'].append(M_b)
        self.evolution_buffer['max_rho_error'].append(np.max(rel_delta_rho))
        self.evolution_buffer['max_p_error'].append(np.max(rel_delta_p))
        self.evolution_buffer['l1_rho_error'].append(l1_rho)
        self.evolution_buffer['l2_rho_error'].append(l2_rho)
        self.evolution_buffer['max_D'].append(np.max(state_2d[self.hydro.idx_D, :]))
        self.evolution_buffer['max_Sr'].append(np.max(np.abs(state_2d[self.hydro.idx_Sr, :])))
        self.evolution_buffer['max_tau'].append(np.max(state_2d[self.hydro.idx_tau, :]))

        # Store constraint violations if provided
        if Ham is not None and Mom is not None:
            self.evolution_buffer['max_Ham'].append(np.max(np.abs(Ham[interior])))
            self.evolution_buffer['l2_Ham'].append(np.sqrt(np.mean(Ham[interior]**2)))
            self.evolution_buffer['max_Mom_r'].append(np.max(np.abs(Mom[interior, 0])))
            self.evolution_buffer['l2_Mom_r'].append(np.sqrt(np.mean(Mom[interior, 0]**2)))
        else:
            self.evolution_buffer['max_Ham'].append(0.0)
            self.evolution_buffer['l2_Ham'].append(0.0)
            self.evolution_buffer['max_Mom_r'].append(0.0)
            self.evolution_buffer['l2_Mom_r'].append(0.0)

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

    def finalize(self, execution_time_seconds=None):
        """Finalize data storage (flush buffers, close files).

        Args:
            execution_time_seconds: Total execution time in seconds (optional)
        """
        if not self.enable_saving:
            return

        self.flush_evolution_buffer()

        # Update metadata with execution time if provided
        if execution_time_seconds is not None:
            self._update_execution_time(execution_time_seconds)

        print(f"\nData saved to:")
        print(f"  - Snapshots: {self.snapshot_file}")
        print(f"  - Evolution: {self.evolution_file}")
        print(f"  - Metadata:  {self.metadata_file}")

    def _update_execution_time(self, execution_time_seconds):
        """Update metadata file with execution time.

        Args:
            execution_time_seconds: Total execution time in seconds
        """
        if not os.path.exists(self.metadata_file):
            return

        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)

        # Add execution time info
        metadata['simulation']['execution_time_seconds'] = float(execution_time_seconds)

        # Format as human-readable string
        hours, remainder = divmod(execution_time_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            time_str = f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"
        elif minutes > 0:
            time_str = f"{int(minutes)}m {seconds:.1f}s"
        else:
            time_str = f"{seconds:.2f}s"
        metadata['simulation']['execution_time_formatted'] = time_str

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


# =============================================================================
# RESTART/CHECKPOINT FUNCTIONS
# =============================================================================

def load_metadata(output_dir, suffix="_cow"):
    """
    Load simulation metadata from JSON file.

    Args:
        output_dir: Output directory path
        suffix: File suffix ("_cow" or "_dyn")

    Returns:
        dict: Metadata dictionary, or None if file doesn't exist
    """
    metadata_file = os.path.join(output_dir, f'tov_metadata{suffix}.json')
    if not os.path.exists(metadata_file):
        return None

    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load metadata from {metadata_file}: {e}")
        return None


def find_latest_snapshot(output_dir, suffix="_cow"):
    """
    Find the most recent snapshot in output directory.

    Args:
        output_dir: Output directory path
        suffix: File suffix ("_cow" or "_dyn")

    Returns:
        dict or None: {
            'snapshot_file': str (full path),
            'step': int,
            'time': float,
            'step_name': str
        }
        Returns None if file doesn't exist or has no snapshots
    """
    snapshot_file = os.path.join(output_dir, f'tov_snapshots{suffix}.h5')

    if not os.path.exists(snapshot_file):
        return None

    try:
        with h5py.File(snapshot_file, 'r') as f:
            snaps_group = f.get('snapshots')
            if snaps_group is None or len(snaps_group) == 0:
                return None

            # Get sorted list of snapshots and take the last one
            snap_keys = sorted(snaps_group.keys())
            latest_snap_name = snap_keys[-1]
            latest_snap = snaps_group[latest_snap_name]

            step = int(latest_snap.attrs['step'])
            time = float(latest_snap.attrs['time'])

            return {
                'snapshot_file': snapshot_file,
                'step': step,
                'time': time,
                'step_name': latest_snap_name
            }
    except Exception as e:
        print(f"Warning: Failed to find latest snapshot in {snapshot_file}: {e}")
        return None


def load_snapshot_from_hdf5(snapshot_file, step_name=None):
    """
    Load complete snapshot from HDF5 file.

    Args:
        snapshot_file: Path to tov_snapshots_cow.h5 or tov_snapshots_dyn.h5
        step_name: Name of snapshot (e.g., 'step_00001250')
                  If None, loads the latest available snapshot

    Returns:
        dict: {
            'state_2d': np.array (NUM_VARS=15, N),
            'step': int,
            'time': float,
            'r': np.array (N,),
            'N': int,
            'step_name': str
        }

    Raises:
        FileNotFoundError: If snapshot_file doesn't exist
        KeyError: If step_name not found in file
    """
    if not os.path.exists(snapshot_file):
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_file}")

    with h5py.File(snapshot_file, 'r') as f:
        # Load grid
        r = np.array(f['grid/r'])
        N = int(f['grid/N'][()])
        r_max = float(f['grid/r_max'][()])

        # If step_name not specified, find latest
        if step_name is None:
            snaps_group = f.get('snapshots')
            if snaps_group is None or len(snaps_group) == 0:
                raise KeyError("No snapshots found in file")
            snap_keys = sorted(snaps_group.keys())
            step_name = snap_keys[-1]

        # Load snapshot
        snap = f['snapshots'][step_name]
        step = int(snap.attrs['step'])
        time = float(snap.attrs['time'])

        # Reconstruct state_2d: shape (NUM_VARS=15, N)
        # [0:12] BSSN variables, [12:15] hydro conservatives
        state_2d = np.zeros((15, N), dtype=np.float64)

        # Load BSSN variables (12 variables)
        bssn_group = snap['bssn']
        bssn_order = ['phi', 'hrr', 'htt', 'hpp', 'K', 'arr', 'att', 'app',
                      'lambdar', 'shiftr', 'br', 'lapse']

        for idx, var_name in enumerate(bssn_order):
            state_2d[idx, :] = np.array(bssn_group[var_name])

        # Load hydro conservative variables (3 variables)
        cons_group = snap['conservatives']
        state_2d[12, :] = np.array(cons_group['D'])       # D
        state_2d[13, :] = np.array(cons_group['Sr'])      # Sr
        state_2d[14, :] = np.array(cons_group['tau'])     # tau

        return {
            'state_2d': state_2d,
            'step': step,
            'time': time,
            'r': r,
            'N': N,
            'r_max': r_max,
            'step_name': step_name
        }


def validate_restart_consistency(snapshot_data, config):
    """
    Validate that loaded snapshot is compatible with current configuration.

    Args:
        snapshot_data: dict returned by load_snapshot_from_hdf5()
        config: dict with current configuration {
            'num_points': int,
            'r_max': float,
            'K': float,
            'Gamma': float,
            'rho_floor': float,
            'p_floor': float
        }

    Raises:
        ValueError: If critical inconsistencies found
    """
    # Check grid size
    if snapshot_data['N'] != config['num_points']:
        raise ValueError(
            f"Grid size mismatch: snapshot has N={snapshot_data['N']}, "
            f"but config requires N={config['num_points']}"
        )

    # Check domain size (r_max)
    r_max_snapshot = snapshot_data['r_max']
    r_max_config = config['r_max']

    if not np.isclose(r_max_snapshot, r_max_config, rtol=1e-10):
        raise ValueError(
            f"Domain size mismatch: snapshot has r_max={r_max_snapshot:.6e}, "
            f"but config requires r_max={r_max_config:.6e}"
        )

    # Check state_2d shape
    if snapshot_data['state_2d'].shape != (15, config['num_points']):
        raise ValueError(
            f"State shape mismatch: expected (15, {config['num_points']}), "
            f"got {snapshot_data['state_2d'].shape}"
        )

    # Check that grid arrays match
    if len(snapshot_data['r']) != config['num_points']:
        raise ValueError(
            f"Grid array length mismatch: snapshot has {len(snapshot_data['r'])} points, "
            f"config requires {config['num_points']}"
        )

    # Warnings for parameter mismatches (non-critical)
    if not np.isclose(snapshot_data['r'][-1], config['r_max'], rtol=1e-10):
        print(f"Warning: Grid endpoint mismatch: {snapshot_data['r'][-1]:.6e} vs {config['r_max']:.6e}")


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
        # Compute constraints at initial state
        Ham_0, Mom_0 = compute_constraints(state_initial, grid, background, hydro)

        data_manager.save_snapshot(step_offset, t_start, state_initial, rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial,
                                   Ham=Ham_0, Mom=Mom_0)
        data_manager.add_evolution_point(step_offset, t_start, state_initial,
                                        rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial, success_initial,
                                        rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial, success_initial,
                                        Ham=Ham_0, Mom=Mom_0)

    # Timeseries for mass, central density, and central velocity
    times_series = [t_start]
    Mb0 = compute_baryon_mass(grid, s_initial, rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial)
    Mb_series = [Mb0]
    rho_c0 = rho0_initial[NUM_GHOSTS]
    rho_c_series = [rho_c0]
    v_c0 = vr_initial[NUM_GHOSTS]
    v_c_series = [v_c0]


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
        if step_num % 100 == 0:
            print(f"step {step_num:4d}  t={t_curr:.2e}:  ρ_c={rho_central:.6e}  max_Δρ/ρ={max_rel_rho_err:.2e}@r={r_max_rho_err:.2f}  "
              f"max_vʳ={max_abs_v:.3e}@r={r_max_v:.2f}  max_Sʳ={max_Sr:.2e}@r={r_max_Sr:.2f}  "
              f"c2p_fail={c2p_fail_count}")

        # Save data if requested
        if data_manager and data_manager.enable_saving:
            # Compute constraints for this timestep (expensive, only when saving)
            Ham, Mom = None, None
            if evolution_interval and (step_num % evolution_interval == 0):
                Ham, Mom = compute_constraints(s_next, grid, background, hydro)
            if snapshot_interval and (step_num % snapshot_interval == 0) and (Ham is None):
                Ham, Mom = compute_constraints(s_next, grid, background, hydro)

            # Save evolution data at specified interval
            if evolution_interval and step_num % evolution_interval == 0:
                data_manager.add_evolution_point(step_num, t_curr, s_next,
                                                rho0_next, vr_next, p_next, eps_next, W_next, h_next, success_next,
                                                rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial, success_initial,
                                                Ham=Ham, Mom=Mom)

                # Periodic buffer flush
                if step_num % (evolution_interval * 10) == 0:
                    data_manager.flush_evolution_buffer()

            # Save full snapshot at specified interval
            if snapshot_interval and step_num % snapshot_interval == 0:
                data_manager.save_snapshot(step_num, t_curr, s_next, rho0_next, vr_next, p_next, eps_next, W_next, h_next,
                                          Ham=Ham, Mom=Mom)

        # Append to time series (mass, central density, and central velocity)
        Mb_next = compute_baryon_mass(grid, s_next, rho0_next, vr_next, p_next, eps_next, W_next, h_next)
        times_series.append(t_curr)
        Mb_series.append(Mb_next)
        rho_c_series.append(float(rho0_next[NUM_GHOSTS]))
        v_c_series.append(float(vr_next[NUM_GHOSTS]))

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
                'v_c': np.array(v_c_series),
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
        'v_c': np.array(v_c_series),
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
    # axes[0, 0].grid(True, alpha=0.3)

    # Pressure
    axes[0, 1].plot(r, P, color='darkgreen')
    axes[0, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure')
    axes[0, 1].set_xlim(0, r_max)
    # axes[0, 1].grid(True, alpha=0.3)

    # Enclosed Mass
    axes[0, 2].plot(r, M, color='maroon')
    axes[0, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 2].axhline(M_star, color='gray', linestyle='--', alpha=0.3, label=f'M={M_star:.3f}')
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel('M(r)')
    axes[0, 2].set_title('Enclosed Mass')
    axes[0, 2].set_xlim(0, r_max)
    axes[0, 2].legend()
    # axes[0, 2].grid(True, alpha=0.3)

    # Lapse alpha(r)
    axes[1, 0].plot(r, alpha, color='purple')
    axes[1, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel('alpha')
    axes[1, 0].set_title('Lapse Function')
    axes[1, 0].set_xlim(0, r_max)
    # axes[1, 0].grid(True, alpha=0.3)

    # Phi(r)
    phi = 0.25 * np.log(exp4phi)
    axes[1, 1].plot(r, phi, color='teal')
    axes[1, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel('phi')
    axes[1, 1].set_title('Conformal Factor phi')
    axes[1, 1].set_xlim(0, r_max)
    # axes[1, 1].grid(True, alpha=0.3)

    # a(r) metric function
    a_metric = np.sqrt(exp4phi)
    axes[1, 2].plot(r, a_metric, color='orange')
    axes[1, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel('a(r)')
    axes[1, 2].set_title('Metric a(r)')
    axes[1, 2].set_xlim(0, r_max)
    # axes[1, 2].grid(True, alpha=0.3)

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
    # axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(r, p_0, 'b-', label='t=0', linewidth=1.5)
    axes[0, 1].plot(r, p_1, 'r--', label='t=dt', linewidth=1.5)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure')
    axes[0, 1].legend()
    # axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(r, vr_0, 'b-', label='t=0', linewidth=1.5)
    axes[0, 2].plot(r, vr_1, 'r--', label='t=dt', linewidth=1.5)
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel('v^r')
    axes[0, 2].set_title('Radial Velocity')
    axes[0, 2].legend()
    # axes[0, 2].grid(True, alpha=0.3)

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
    # axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(r, Sr_0, 'b-', label='t=0', linewidth=1.5)
    axes[1, 1].plot(r, Sr_1, 'r--', label='t=dt', linewidth=1.5)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel('Sr')
    axes[1, 1].set_title('Conserved Sr')
    axes[1, 1].legend()
    # axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(r, tau_0, 'b-', label='t=0', linewidth=1.5)
    axes[1, 2].plot(r, tau_1, 'r--', label='t=dt', linewidth=1.5)
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel('tau')
    axes[1, 2].set_title('Conserved tau')
    axes[1, 2].legend()
    # axes[1, 2].grid(True, alpha=0.3)

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
    # axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(r[mask], p_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[0, 1].plot(r[mask], p_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure (center)')
    axes[0, 1].legend()
    # axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(r[mask], vr_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[0, 2].plot(r[mask], vr_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel(r'$v^r$')
    axes[0, 2].set_title('Radial Velocity (center)')
    axes[0, 2].legend()
    # axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Conservative variables
    axes[1, 0].plot(r[mask], D_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[1, 0].plot(r[mask], D_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel('D')
    axes[1, 0].set_title('Conserved D (center)')
    axes[1, 0].legend()
    # axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(r[mask], Sr_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[1, 1].plot(r[mask], Sr_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel(r'$S_r$')
    axes[1, 1].set_title('Conserved Sr (center)')
    axes[1, 1].legend()
    # axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(r[mask], tau_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[1, 2].plot(r[mask], tau_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel(r'$\tau$')
    axes[1, 2].set_title('Conserved tau (center)')
    axes[1, 2].legend()
    # axes[1, 2].grid(True, alpha=0.3)

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
    # axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(r[mask], p_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[0, 1].plot(r[mask], p_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[0, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Pressure (surface)')
    axes[0, 1].legend()
    # axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(r[mask], vr_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[0, 2].plot(r[mask], vr_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[0, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel(r'$v^r$')
    axes[0, 2].set_title('Radial Velocity (surface)')
    axes[0, 2].legend()
    # axes[0, 2].grid(True, alpha=0.3)

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
    # axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(r[mask], Sr_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[1, 1].plot(r[mask], Sr_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[1, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel(r'$S_r$')
    axes[1, 1].set_title('Conserved Sr (surface)')
    axes[1, 1].legend()
    # axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(r[mask], tau_0[mask], 'b-', label='t=0', linewidth=1.5)
    axes[1, 2].plot(r[mask], tau_1[mask], 'r--', label='t=dt', linewidth=1.5)
    axes[1, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel(r'$\tau$')
    axes[1, 2].set_yscale('log')
    axes[1, 2].set_title('Conserved tau (surface)')
    axes[1, 2].legend()
    # axes[1, 2].grid(True, alpha=0.3)

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
    # axes[0, 0].grid(True, alpha=0.3)

    # h_rr (metric deviation)
    axes[0, 1].plot(r, state_t0[idx_hrr, :], 'b-', label='t=0', linewidth=1.5)
    axes[0, 1].plot(r, state_tfinal[idx_hrr, :], 'r--', label=f't={t_final}', linewidth=1.5)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel(r'$h_{rr}$')
    axes[0, 1].set_title('Metric Deviation h_rr')
    axes[0, 1].legend()
    # axes[0, 1].grid(True, alpha=0.3)

    # alpha (lapse) - CORRECTED: use idx_lapse
    axes[0, 2].plot(r, state_t0[idx_lapse, :], 'b-', label='t=0', linewidth=1.5)
    axes[0, 2].plot(r, state_tfinal[idx_lapse, :], 'r--', label=f't={t_final}', linewidth=1.5)
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel(r'$\alpha$')
    axes[0, 2].set_title('Lapse alpha')
    axes[0, 2].legend()
    # axes[0, 2].grid(True, alpha=0.3)

    # beta^r (shift) - CORRECTED: use idx_shiftr
    axes[1, 0].plot(r, state_t0[idx_shiftr, :], 'b-', label='t=0', linewidth=1.5)
    axes[1, 0].plot(r, state_tfinal[idx_shiftr, :], 'r--', label=f't={t_final}', linewidth=1.5)
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel(r'$\beta^r$')
    axes[1, 0].set_title('Shift βʳ')
    axes[1, 0].legend()
    # axes[1, 0].grid(True, alpha=0.3)

    # B^r (shift auxiliary) - CORRECTED: use idx_br
    axes[1, 1].plot(r, state_t0[idx_br, :], 'b-', label='t=0', linewidth=1.5)
    axes[1, 1].plot(r, state_tfinal[idx_br, :], 'r--', label=f't={t_final}', linewidth=1.5)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel(r'$B^r$')
    axes[1, 1].set_title('Shift Auxiliary Bʳ')
    axes[1, 1].legend()
    # axes[1, 1].grid(True, alpha=0.3)

    # K (mean curvature) - CORRECTED: use idx_K
    axes[1, 2].plot(r, state_t0[idx_K, :], 'b-', label='t=0', linewidth=1.5)
    axes[1, 2].plot(r, state_tfinal[idx_K, :], 'r--', label=f't={t_final}', linewidth=1.5)
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel('K')
    axes[1, 2].set_title('Extrinsic Curvature K')
    axes[1, 2].legend()
    # axes[1, 2].grid(True, alpha=0.3)

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
    # axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')   

    # Central density deviation
    rho_c_0 = rho_c_series[0]
    delta_rho_c = (np.array(rho_c_series) - rho_c_0) / rho_c_0

    axes[1].plot(times, delta_rho_c, 'r-', linewidth=1.5)
    axes[1].set_xlabel('t')
    axes[1].set_ylabel(r'$(\rho_c - \rho_{c,0})/\rho_{c,0}$')
    axes[1].set_title('Central Density Relative Change')
    # axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()
    out_path = os.path.join(plots_dir, f'tov_mass_central_density{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_evolution(states, times, grid, hydro, rho_ref, p_ref,
                   Mb_series=None, rho_c_series=None,
                   times_series=None, suffix="", R_star=None):
    """Plot evolution at multiple checkpoints: density, pressure, baryon mass, central density.

    Args:
        states: List of state arrays at different times [state_t0, state_t1, state_t2, state_tfinal]
        times: List of times corresponding to states [0, t1, t2, tfinal]
        grid: Grid object
        hydro: Hydro object
        rho_ref: Reference density for error computation
        p_ref: Reference pressure for error computation
        Mb_series: Optional array of baryon mass values
        rho_c_series: Optional array of central density values
        times_series: Optional array of times for Mb_series/rho_c_series
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

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Density profiles at checkpoints
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, n_states))
    for i, (prim, t) in enumerate(zip(primitives, times)):
        ax.plot(r[interior], prim[0][interior], color=colors[i],
                label=f't={t:.2f}', linewidth=1.5)
    if R_star is not None:
        ax.axvline(R_star, color='gray', linestyle=':', alpha=0.7, label=f'R={R_star:.2f}')
    ax.set_xlabel('r')
    ax.set_ylabel(r'$\rho_0$')
    ax.set_title('Baryon Density Evolution')
    ax.set_yscale('log')
    ax.legend(loc='upper right')

    # (0,1) Pressure profiles at checkpoints
    ax = axes[0, 1]
    for i, (prim, t) in enumerate(zip(primitives, times)):
        ax.plot(r[interior], prim[2][interior], color=colors[i],
                label=f't={t:.2f}', linewidth=1.5)
    if R_star is not None:
        ax.axvline(R_star, color='gray', linestyle=':', alpha=0.7, label=f'R={R_star:.2f}')
    ax.set_xlabel('r')
    ax.set_ylabel('P')
    ax.set_title('Pressure Evolution')
    ax.set_yscale('log')
    ax.legend(loc='upper right')

    # (1,0) Baryon mass deviation vs time
    ax = axes[1, 0]
    if Mb_series is not None and len(Mb_series) > 1:
        Mb_arr = np.asarray(Mb_series)
        if times_series is not None:
            t_ser = np.asarray(times_series)
        else:
            t_ser = np.linspace(0, times[-1], len(Mb_arr))
        Mb_0 = Mb_arr[0]
        delta_Mb = (Mb_arr - Mb_0) / Mb_0
        ax.plot(t_ser, delta_Mb, 'b-', linewidth=1.5)
        ax.set_ylabel(r'$\Delta M_b / M_{b,0}$')
    else:
        ax.text(0.5, 0.5, 'No time series data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
    ax.set_xlabel('t')
    ax.set_title('Baryon Mass Conservation')

    # (1,1) Central density vs time
    ax = axes[1, 1]
    if rho_c_series is not None and len(rho_c_series) > 1:
        rho_c_arr = np.asarray(rho_c_series)
        if times_series is not None:
            t_ser = np.asarray(times_series)
        else:
            t_ser = np.linspace(0, times[-1], len(rho_c_arr))
        rho_c_0 = rho_c_arr[0]
        delta_rho_c = (rho_c_arr - rho_c_0) / rho_c_0
        ax.plot(t_ser, delta_rho_c, 'r-', linewidth=1.5)
        ax.set_ylabel(r'$(\rho_c - \rho_{c,0})/\rho_{c,0}$')
    else:
        ax.text(0.5, 0.5, 'No time series data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
    ax.set_xlabel('t')
    ax.set_title('Central Density Relative Change')

    plt.suptitle(f'TOV Evolution', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(plots_dir, f'tov_evolution{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
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


def compute_constraints(state_2d, grid, background, matter):
    """Compute Hamiltonian and Momentum constraints using constraintsdiagnostic.py.

    Args:
        state_2d: State array (NUM_VARS x N)
        grid: Grid object
        background: Background object
        matter: Matter object (hydro)

    Returns:
        tuple: (Ham, Mom) where
            - Ham: Hamiltonian constraint array (N,)
            - Mom: Momentum constraint array (N, 3)
    """
    from source.bssn.constraintsdiagnostic import get_constraints_diagnostic

    # Reshape state to what get_constraints_diagnostic expects
    state_flat = state_2d.flatten()

    # Calculate constraints (returns arrays with shape (1, N) and (1, N, 3))
    Ham, Mom = get_constraints_diagnostic(state_flat, 0.0, grid, background, matter)

    # Return squeezed arrays (N,) and (N, 3)
    return Ham[0, :], Mom[0, :, :]


def plot_constraints_evolution(output_dir, suffix=""):
    """Plot constraint violations evolution from saved HDF5 data.

    Reads from tov_evolution{suffix}.h5 and creates plots showing:
    - max(|H|) and L2(H) vs time
    - max(|M_r|) and L2(M_r) vs time

    Args:
        output_dir: Directory containing the evolution file
        suffix: Suffix for the evolution file (e.g., "_dyn")
    """
    evolution_file = os.path.join(output_dir, f'tov_evolution{suffix}.h5')

    if not os.path.exists(evolution_file):
        print(f"\n⚠️  Warning: Evolution file not found: {evolution_file}")
        print(f"   Cannot plot constraints. Make sure to run evolution with data saving enabled.")
        return

    # Read constraint data
    with h5py.File(evolution_file, 'r') as f:
        # Check if constraint data exists
        if 'max_Ham' not in f:
            print(f"\n⚠️  Warning: No constraint data found in {evolution_file}")
            print(f"   This evolution was run before constraint monitoring was added.")
            print(f"   Re-run the evolution to generate constraint data.")
            return

        times = np.array(f['time'])
        max_Ham = np.array(f['max_Ham'])
        l2_Ham = np.array(f['l2_Ham'])
        max_Mom_r = np.array(f['max_Mom_r'])
        l2_Mom_r = np.array(f['l2_Mom_r'])

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Hamiltonian constraint - max
    axes[0, 0].semilogy(times, np.maximum(np.abs(max_Ham), 1e-20), 'r-', linewidth=1.5)
    axes[0, 0].set_xlabel('t [M]')
    axes[0, 0].set_ylabel(r'max$|$H$|$')
    axes[0, 0].set_title('Hamiltonian Constraint: Maximum Violation')
    axes[0, 0].grid(True, alpha=0.3)

    # Hamiltonian constraint - L2
    axes[0, 1].semilogy(times, np.maximum(l2_Ham, 1e-20), 'b-', linewidth=1.5)
    axes[0, 1].set_xlabel('t [M]')
    axes[0, 1].set_ylabel(r'L$_2$(H)')
    axes[0, 1].set_title('Hamiltonian Constraint: L2 Norm')
    axes[0, 1].grid(True, alpha=0.3)

    # Momentum constraint - max
    axes[1, 0].semilogy(times, np.maximum(np.abs(max_Mom_r), 1e-20), 'r-', linewidth=1.5)
    axes[1, 0].set_xlabel('t [M]')
    axes[1, 0].set_ylabel(r'max$|$M$_r|$')
    axes[1, 0].set_title('Momentum Constraint (radial): Maximum Violation')
    axes[1, 0].grid(True, alpha=0.3)

    # Momentum constraint - L2
    axes[1, 1].semilogy(times, np.maximum(l2_Mom_r, 1e-20), 'b-', linewidth=1.5)
    axes[1, 1].set_xlabel('t [M]')
    axes[1, 1].set_ylabel(r'L$_2$(M$_r$)')
    axes[1, 1].set_title('Momentum Constraint (radial): L2 Norm')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'BSSN Constraint Violations Evolution', fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(plots_dir, f'constraints_evolution{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


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
    if hasattr(tov_solution, 'R_iso'):
        R = float(tov_solution.R_iso)
        r_tov = tov_solution.r_iso
        rho_tov = tov_solution.rho_baryon
        P_tov = tov_solution.P
    else:
        R = float(tov_solution['R_iso'])
        r_tov = tov_solution['r_iso']
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
    # ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].set_ylabel('rho_0', fontsize=10)

    ax[0, 1].semilogy(rZ, np.maximum(P_tov_zoom[mask], 1e-20), 'k-', linewidth=2)
    ax[0, 1].semilogy(rZ, np.maximum(p_init[mask], 1e-20), 'b--', linewidth=1.5)
    ax[0, 1].axvline(R, color='gray', ls=':', linewidth=1.5)
    ax[0, 1].set_title('P (zoom near surface)', fontsize=11)
    # ax[0, 1].grid(True, alpha=0.3)
    ax[0, 1].set_ylabel('P', fontsize=10)

    # v^r should be zero in TOV equilibrium
    ax[0, 2].plot(rZ, vr_init[mask], 'b-', linewidth=1.5, label='Initial Data')
    ax[0, 2].axhline(0, color='k', ls='--', linewidth=2, label='TOV (v=0)')
    ax[0, 2].axvline(R, color='gray', ls=':', linewidth=1.5)
    ax[0, 2].set_title('v^r (zoom near surface)', fontsize=11)
    ax[0, 2].legend(fontsize=9)
    # ax[0, 2].grid(True, alpha=0.3)
    ax[0, 2].set_ylabel('v^r', fontsize=10)

    # Row 2: D, Sr, tau (conservative variables)
    ax[1, 0].semilogy(rZ, np.maximum(D_init[mask], 1e-22), 'b-', linewidth=1.5)
    ax[1, 0].axvline(R, color='gray', ls=':', linewidth=1.5)
    ax[1, 0].set_title('D (conserved density)', fontsize=11)
    # ax[1, 0].grid(True, alpha=0.3)
    ax[1, 0].set_ylabel('D', fontsize=10)

    ax[1, 1].plot(rZ, Sr_init[mask], 'b-', linewidth=1.5)
    ax[1, 1].axhline(0, color='k', ls='--', linewidth=1, label='Expected (S^r=0)')
    ax[1, 1].axvline(R, color='gray', ls=':', linewidth=1.5)
    ax[1, 1].set_title('S_r (conserved momentum)', fontsize=11)
    ax[1, 1].legend(fontsize=9)
    # ax[1, 1].grid(True, alpha=0.3)
    ax[1, 1].set_ylabel('S_r', fontsize=10)

    ax[1, 2].semilogy(rZ, np.maximum(np.abs(tau_init[mask]), 1e-22), 'b-', linewidth=1.5)
    ax[1, 2].axvline(R, color='gray', ls=':', linewidth=1.5)
    ax[1, 2].set_title('tau (conserved energy)', fontsize=11)
    # ax[1, 2].grid(True, alpha=0.3)
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


