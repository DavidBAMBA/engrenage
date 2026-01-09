#!/usr/bin/env python3
"""Utility to plot the density profile ρ₀(r) from a stored state."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def add_repo_to_sys_path():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def load_state_and_primitives(state_path: Path, grid, hydro):
    from source.bssn.bssnvars import BSSNVars
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS

    state_flat = np.load(state_path)
    expected_size = grid.NUM_VARS * grid.N
    if state_flat.size != expected_size:
        raise ValueError(
            f"State file size {state_flat.size} does not match grid ({expected_size})."
        )

    state = state_flat.reshape((grid.NUM_VARS, grid.N))

    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state, bssn_vars, grid)
    rho0, vr, p, eps, W, h, success = hydro._get_primitives(bssn_vars, grid.r)
    prims = {'rho0': rho0, 'vr': vr, 'p': p, 'eps': eps, 'W': W, 'h': h, 'success': success}
    return state, prims


def build_grid_and_hydro(spacing_type: str, r_max: float, min_dr: float, max_dr: float,
                         K: float, Gamma: float):
    from source.core.spacing import CubicSpacing, LinearSpacing, SinhSpacing
    from source.core.statevector import StateVector
    from source.core.grid import Grid
    from source.backgrounds.sphericalbackground import FlatSphericalBackground
    from source.matter.hydro.perfect_fluid import PerfectFluid
    from source.matter.hydro.eos import IdealGasEOS
    from source.matter.hydro.atmosphere import AtmosphereParams
    from source.matter.hydro.reconstruction import create_reconstruction
    from source.matter.hydro.riemann import HLLRiemannSolver

    spacing_type = spacing_type.lower()
    if spacing_type == "cubic":
        params = CubicSpacing.get_parameters(r_max, min_dr, max_dr)
        spacing = CubicSpacing(**params)
    elif spacing_type == "linear":
        params = LinearSpacing.get_parameters(r_max=r_max, min_dr=min_dr)
        spacing = LinearSpacing(**params)
    elif spacing_type == "sinh":
        params = SinhSpacing.get_parameters(r_max=r_max, min_dr=min_dr, max_dr=max_dr)
        spacing = SinhSpacing(**params)
    else:
        raise ValueError(f"Unsupported spacing type '{spacing_type}'.")

    eos = IdealGasEOS(gamma=Gamma)
    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere=AtmosphereParams(rho_floor=1e-13),
        reconstructor=create_reconstruction("mp5"),
        riemann_solver=HLLRiemannSolver(),
    )

    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    hydro.background = FlatSphericalBackground(grid.r)
    return grid, hydro


def plot_domain_density(grid, prims, output_path: Path = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grid.r, prims['rho0'], color='navy', linewidth=2)
    ax.set_xlabel('r')
    ax.set_ylabel('ρ₀(r)')
    ax.set_title('Density profile in the domain')
    ax.grid(True, alpha=0.3)
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot TOV diagnostics and density profiles.")
    parser.add_argument('--state', type=Path, required=True,
                        help=".npy file with flattened state vector to plot density profile.")
    parser.add_argument('--spacing', choices=['cubic', 'linear', 'sinh'], default='cubic',
                        help="Spacing type to reconstruct the grid when plotting domain density.")
    parser.add_argument('--r-max', type=float, default=20.0)
    parser.add_argument('--min-dr', type=float, default=0.001)
    parser.add_argument('--max-dr', type=float, default=1.0)
    parser.add_argument('--K', type=float, default=100.0)
    parser.add_argument('--Gamma', type=float, default=2.0)
    parser.add_argument('--output-dir', type=Path, default=None,
                        help="Directory to save plots. If omitted, figures are shown interactively.")
    args = parser.parse_args()

    repo_root = add_repo_to_sys_path()

    plots_out = None
    if args.output_dir is not None:
        plots_out = args.output_dir.resolve()
        plots_out.mkdir(parents=True, exist_ok=True)

    grid, hydro = build_grid_and_hydro(
        args.spacing, args.r_max, args.min_dr, args.max_dr, args.K, args.Gamma
    )
    _, prims = load_state_and_primitives(args.state.resolve(), grid, hydro)
    output_path = plots_out / 'density_profile.png' if plots_out is not None else None
    plot_domain_density(grid, prims, output_path)


if __name__ == "__main__":
    main()
