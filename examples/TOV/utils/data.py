"""
Data management for TOV evolution simulations.

Handles HDF5 storage for snapshots and time series.
"""

import numpy as np
import os
import h5py
import json
from datetime import datetime


class SimulationDataManager:
    """Manages data storage for long TOV simulations."""

    def __init__(self, output_dir, grid, hydro, enable_saving=False, suffix="",
                 restart_mode=False):
        """
        Initialize data manager.

        Args:
            output_dir: Directory for output files
            grid: Grid object
            hydro: Hydro object
            enable_saving: If True, saves data to files
            suffix: Suffix for file naming
            restart_mode: If True, append to existing files
        """
        self.enable_saving = enable_saving
        self.suffix = suffix
        self.restart_mode = restart_mode
        if not self.enable_saving:
            return

        self.output_dir = output_dir
        self.grid = grid
        self.hydro = hydro

        os.makedirs(output_dir, exist_ok=True)

        self.snapshot_file = os.path.join(output_dir, f"tov_snapshots{suffix}.h5")
        self.evolution_file = os.path.join(output_dir, f"tov_evolution{suffix}.h5")
        self.metadata_file = os.path.join(output_dir, f"tov_metadata{suffix}.json")

        if not restart_mode:
            for f in [self.snapshot_file, self.evolution_file, self.metadata_file]:
                if os.path.exists(f):
                    os.remove(f)

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
            'max_Ham': [],
            'l2_Ham': [],
            'max_Mom_r': [],
            'l2_Mom_r': []
        }

        if not restart_mode:
            self._init_hdf5_files()

    def _init_hdf5_files(self):
        """Initialize HDF5 files with proper structure."""
        if not self.enable_saving:
            return

        with h5py.File(self.snapshot_file, 'w') as f:
            f.create_group('snapshots')
            f.create_group('grid')
            f['grid/r'] = self.grid.r
            f['grid/N'] = self.grid.N
            f['grid/r_max'] = self.grid.r[-1]

        with h5py.File(self.evolution_file, 'w') as f:
            for key in self.evolution_buffer.keys():
                f.create_dataset(key, shape=(0,), maxshape=(None,),
                               dtype=np.float64, chunks=True)

    def save_metadata(self, tov_solution, atmosphere_params, dt, integration_method,
                      K=None, Gamma=None, rho_central=None, r_max=None, num_points=None,
                      t_final=None, reconstructor=None, solver_method=None,
                      riemann_solver=None, evolution_mode=None, cfl_factor=None):
        """Save simulation metadata."""
        if not self.enable_saving:
            return

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

            cons_group = snap_group.create_group('conservatives')
            cons_group['D'] = state_2d[self.hydro.idx_D, :]
            cons_group['Sr'] = state_2d[self.hydro.idx_Sr, :]
            cons_group['tau'] = state_2d[self.hydro.idx_tau, :]

            bssn_group = snap_group.create_group('bssn')
            from source.bssn.bssnstatevariables import (
                idx_phi, idx_hrr, idx_htt, idx_hpp, idx_K,
                idx_arr, idx_att, idx_app, idx_lambdar,
                idx_shiftr, idx_br, idx_lapse
            )
            bssn_group['phi'] = state_2d[idx_phi, :]
            bssn_group['hrr'] = state_2d[idx_hrr, :]
            bssn_group['htt'] = state_2d[idx_htt, :]
            bssn_group['hpp'] = state_2d[idx_hpp, :]
            bssn_group['K'] = state_2d[idx_K, :]
            bssn_group['arr'] = state_2d[idx_arr, :]
            bssn_group['att'] = state_2d[idx_att, :]
            bssn_group['app'] = state_2d[idx_app, :]
            bssn_group['lambdar'] = state_2d[idx_lambdar, :]
            bssn_group['shiftr'] = state_2d[idx_shiftr, :]
            bssn_group['br'] = state_2d[idx_br, :]
            bssn_group['lapse'] = state_2d[idx_lapse, :]

            if rho0 is not None:
                prim_group = snap_group.create_group('primitives')
                prim_group['rho0'] = rho0
                prim_group['p'] = p
                prim_group['eps'] = eps
                prim_group['vr'] = vr
                prim_group['W'] = W

            if Ham is not None and Mom is not None:
                const_group = snap_group.create_group('constraints')
                const_group['Ham'] = Ham
                const_group['Mom_r'] = Mom[:, 0]
                const_group['Mom_theta'] = Mom[:, 1]
                const_group['Mom_phi'] = Mom[:, 2]

    def add_evolution_point(self, step, time, state_2d, rho0, vr, p, eps, W, h, success,
                           rho0_ref, vr_ref, p_ref, eps_ref, W_ref, h_ref, success_ref,
                           Ham=None, Mom=None):
        """Add a point to the evolution time series."""
        if not self.enable_saving:
            return

        from source.core.spacing import NUM_GHOSTS

        interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

        delta_rho = np.abs(rho0[interior] - rho0_ref[interior])
        rel_delta_rho = delta_rho / (np.abs(rho0_ref[interior]) + 1e-20)

        delta_p = np.abs(p[interior] - p_ref[interior])
        rel_delta_p = delta_p / (np.abs(p_ref[interior]) + 1e-20)

        star_mask = rho0_ref[interior] > 10 * self.hydro.atmosphere.rho_floor
        if np.any(star_mask):
            l1_rho = np.mean(delta_rho[star_mask])
            l2_rho = np.sqrt(np.mean(delta_rho[star_mask]**2))
        else:
            l1_rho = 0.0
            l2_rho = 0.0

        from .diagnostics import compute_baryon_mass
        M_b = compute_baryon_mass(self.grid, state_2d, rho0, vr, p, eps, W, h)

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

        for key in self.evolution_buffer:
            self.evolution_buffer[key] = []

    def finalize(self, execution_time_seconds=None):
        """Finalize data storage."""
        if not self.enable_saving:
            return

        self.flush_evolution_buffer()

        if execution_time_seconds is not None:
            self._update_execution_time(execution_time_seconds)

        print(f"\nData saved to:")
        print(f"  - Snapshots: {self.snapshot_file}")
        print(f"  - Evolution: {self.evolution_file}")
        print(f"  - Metadata:  {self.metadata_file}")

    def _update_execution_time(self, execution_time_seconds):
        """Update metadata with execution time."""
        if not os.path.exists(self.metadata_file):
            return

        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)

        metadata['simulation']['execution_time_seconds'] = float(execution_time_seconds)

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
