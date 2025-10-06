#!/usr/bin/env python3
"""Plot the density profile ρ₀(r) from a stored state vector."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def add_repo_to_path():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def build_grid_and_hydro(spacing_type: str, r_max: float, min_dr: float, max_dr: float,
                         K: float, Gamma: float, n_interior: int = None):
    from source.core.spacing import CubicSpacing, LinearSpacing, SinhSpacing
    from source.core.statevector import StateVector
    from source.core.grid import Grid
    from source.backgrounds.sphericalbackground import FlatSphericalBackground
    from source.matter.hydro.perfect_fluid import PerfectFluid
    from source.matter.hydro.eos import IdealGasEOS
    from source.matter.hydro.reconstruction import create_reconstruction
    from source.matter.hydro.riemann import HLLERiemannSolver

    spacing_type = spacing_type.lower()
    if spacing_type == "cubic":
        params = CubicSpacing.get_parameters(r_max, min_dr, max_dr)
        if n_interior is not None:
            params['num_points'] = n_interior
        spacing = CubicSpacing(**params)
    elif spacing_type == "linear":
        params = LinearSpacing.get_parameters(r_max=r_max, min_dr=min_dr)
        if n_interior is not None:
            params['num_points'] = n_interior
        spacing = LinearSpacing(**params)
    elif spacing_type == "sinh":
        params = SinhSpacing.get_parameters(r_max=r_max, min_dr=min_dr, max_dr=max_dr)
        if n_interior is not None:
            params['num_points'] = n_interior
        spacing = SinhSpacing(**params)
    else:
        raise ValueError(f"Unsupported spacing '{spacing_type}'")

    eos = IdealGasEOS(gamma=Gamma)
    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere_rho=1e-13,
        reconstructor=create_reconstruction("mp5"),
        riemann_solver=HLLERiemannSolver(),
    )

    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    hydro.background = FlatSphericalBackground(grid.r)
    return grid, hydro


def load_primitives(state_path: Path, grid, hydro):
    from source.bssn.bssnvars import BSSNVars
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS

    state_flat = np.load(state_path)
    expected = grid.NUM_VARS * grid.N
    if state_flat.size != expected:
        raise ValueError(
            f"State vector size {state_flat.size} does not match grid ({expected})."
        )

    state = state_flat.reshape((grid.NUM_VARS, grid.N))
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state, bssn_vars, grid)
    prims = hydro._get_primitives(bssn_vars, grid.r)
    return prims


def main():
    parser = argparse.ArgumentParser(description="Plot ρ₀(r) from a saved state.")
    parser.add_argument('--state', type=Path, required=True,
                        help=".npy file containing the flattened state vector.")
    parser.add_argument('--spacing', choices=['cubic', 'linear', 'sinh'], default='cubic')
    parser.add_argument('--r-max', type=float, default=20.0)
    parser.add_argument('--min-dr', type=float, default=0.001)
    parser.add_argument('--max-dr', type=float, default=1.0)
    parser.add_argument('--K', type=float, default=100.0)
    parser.add_argument('--Gamma', type=float, default=2.0)
    parser.add_argument('--n-interior', type=int, default=None,
                        help="Number of interior grid points (if different from default).")
    parser.add_argument('--output', type=Path, default=None,
                        help="Optional path to save the figure instead of displaying it.")
    args = parser.parse_args()

    add_repo_to_path()

    grid, hydro = build_grid_and_hydro(
        args.spacing, args.r_max, args.min_dr, args.max_dr, args.K, args.Gamma, args.n_interior
    )

    prims = load_primitives(args.state.resolve(), grid, hydro)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grid.r, prims['rho0'], color='navy', linewidth=2)
    ax.set_xlabel('r')
    ax.set_ylabel('ρ₀(r)')
    ax.set_title('Density profile')
    ax.grid(True, alpha=0.3)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    main()
