"""
I/O utilities for TOV evolution simulations.

Handles restart, checkpoint loading, and metadata management.
"""

import numpy as np
import os
import h5py
import json


def get_star_folder_name(rho_central, num_points, K, Gamma, evolution_mode="cowling", reconstructor="wenoz"):
    """
    Generate folder name based on star parameters.

    Format: tov_star_rhoc{rho}_N{N}_K{K}_G{Gamma}_{mode}_{recon}
    """
    rho_str = f"{rho_central:.2e}"
    rho_str = rho_str.replace(".", "p").replace("+", "").replace("-", "m")

    K_str = str(int(K)) if K == int(K) else f"{K:.1f}".replace(".", "p")
    G_str = str(int(Gamma)) if Gamma == int(Gamma) else f"{Gamma:.1f}".replace(".", "p")

    mode_suffix = "cow" if evolution_mode == "cowling" else "dyn"

    recon_map = {
        "wenoz": "wz",
        "weno5": "w5",
        "mp5": "mp5",
        "minmod": "md"
    }
    recon_suffix = recon_map.get(reconstructor.lower(), "wz")

    return f"tov_star_rhoc{rho_str}_N{num_points}_K{K_str}_G{G_str}_{mode_suffix}_{recon_suffix}"


def load_metadata(output_dir, suffix="_cow"):
    """Load simulation metadata from JSON file."""
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
    """Find the most recent snapshot in output directory."""
    snapshot_file = os.path.join(output_dir, f'tov_snapshots{suffix}.h5')

    if not os.path.exists(snapshot_file):
        return None

    try:
        with h5py.File(snapshot_file, 'r') as f:
            snaps_group = f.get('snapshots')
            if snaps_group is None or len(snaps_group) == 0:
                return None

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
    """Load complete snapshot from HDF5 file."""
    if not os.path.exists(snapshot_file):
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_file}")

    with h5py.File(snapshot_file, 'r') as f:
        r = np.array(f['grid/r'])
        N = int(f['grid/N'][()])
        r_max = float(f['grid/r_max'][()])

        if step_name is None:
            snaps_group = f.get('snapshots')
            if snaps_group is None or len(snaps_group) == 0:
                raise KeyError("No snapshots found in file")
            snap_keys = sorted(snaps_group.keys())
            step_name = snap_keys[-1]

        snap = f['snapshots'][step_name]
        step = int(snap.attrs['step'])
        time = float(snap.attrs['time'])

        state_2d = np.zeros((15, N), dtype=np.float64)

        bssn_group = snap['bssn']
        bssn_order = ['phi', 'hrr', 'htt', 'hpp', 'K', 'arr', 'att', 'app',
                      'lambdar', 'shiftr', 'br', 'lapse']

        for idx, var_name in enumerate(bssn_order):
            state_2d[idx, :] = np.array(bssn_group[var_name])

        cons_group = snap['conservatives']
        state_2d[12, :] = np.array(cons_group['D'])
        state_2d[13, :] = np.array(cons_group['Sr'])
        state_2d[14, :] = np.array(cons_group['tau'])

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
    """Validate that loaded snapshot is compatible with current configuration."""
    if snapshot_data['N'] != config['num_points']:
        raise ValueError(
            f"Grid size mismatch: snapshot has N={snapshot_data['N']}, "
            f"but config requires N={config['num_points']}"
        )

    r_max_snapshot = snapshot_data['r_max']
    r_max_config = config['r_max']

    if not np.isclose(r_max_snapshot, r_max_config, rtol=1e-10):
        raise ValueError(
            f"Domain size mismatch: snapshot has r_max={r_max_snapshot:.6e}, "
            f"but config requires r_max={r_max_config:.6e}"
        )

    if snapshot_data['state_2d'].shape != (15, config['num_points']):
        raise ValueError(
            f"State shape mismatch: expected (15, {config['num_points']}), "
            f"got {snapshot_data['state_2d'].shape}"
        )

    if len(snapshot_data['r']) != config['num_points']:
        raise ValueError(
            f"Grid array length mismatch: snapshot has {len(snapshot_data['r'])} points, "
            f"config requires {config['num_points']}"
        )

    if not np.isclose(snapshot_data['r'][-1], config['r_max'], rtol=1e-10):
        print(f"Warning: Grid endpoint mismatch: {snapshot_data['r'][-1]:.6e} vs {config['r_max']:.6e}")
