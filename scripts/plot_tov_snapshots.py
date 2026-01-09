#!/usr/bin/env python3
"""
Plot quantities from TOV evolution snapshots produced by
examples/TOVEvolution_corrected.py.

Inputs:
  - A directory containing files named like: state_t{time}.npy
    (e.g., tov_snapshots/20251002_190442/state_t1.000.npy)

Outputs (by default in plots/tov):
  - For --central: rho_central_evolution.png
  - For --profiles: state_t{time}.png per snapshot (ρ0, p, v^r, lapse)

Notes:
  - Snapshots store the full flattened state vector (BSSN + hydro), not r.
    We rebuild a simple radial grid (linear by default) only to set up the
    background geometry needed to convert conservatives to primitives.
    For the Cowling setup used in TOV, gamma_rr = e^{4phi} and does not
    depend on r; using a linear spacing for plotting is fine.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Add repo root to import 'source'
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from source.core.spacing import LinearSpacing, CubicSpacing, SpacingExtent, NUM_GHOSTS
from source.core.grid import Grid
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.atmosphere import AtmosphereParams
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver


SNAP_RE = re.compile(r"state_t([0-9]+(?:\.[0-9]+)?)\.npy$")


def find_snapshots(snap_dir: Path) -> List[Tuple[float, Path]]:
    files = []
    for p in sorted(snap_dir.glob('state_t*.npy')):
        m = SNAP_RE.search(p.name)
        if not m:
            continue
        try:
            t = float(m.group(1))
        except ValueError:
            continue
        files.append((t, p))
    files.sort(key=lambda x: x[0])
    return files


def build_grid_for_snapshot(N: int, r_max: float, spacing: str = 'linear') -> Grid:
    # Build minimal hydro object to get NUM_VARS
    hydro = PerfectFluid(
        eos=IdealGasEOS(gamma=2.0),
        spacetime_mode='dynamic',
        atmosphere=AtmosphereParams(),
        reconstructor=create_reconstruction('mp5'),
        riemann_solver=HLLRiemannSolver(),
    )
    # Choose spacing
    if spacing == 'cubic':
        # Attempt to match TOV script parameters (min_dr=1e-3, max_dr=0.1)
        try:
            params = CubicSpacing.get_parameters(r_max=r_max, min_dr=1e-3, max_dr=0.1, extent=SpacingExtent.HALF)
            sp = CubicSpacing(**params)
            # Fallback to linear if N mismatches
            if (sp[0]).size != N:
                sp = LinearSpacing(N, r_max, SpacingExtent.HALF)
        except Exception:
            sp = LinearSpacing(N, r_max, SpacingExtent.HALF)
    else:
        sp = LinearSpacing(N, r_max, SpacingExtent.HALF)

    return Grid(sp, StateVector(hydro))


def primitives_from_snapshot(state_1d: np.ndarray, grid: Grid):
    # Unflatten to (NUM_VARS, N)
    num_vars = grid.NUM_VARS
    N = grid.N
    state = state_1d.reshape((num_vars, N))

    # Slice BSSN/hydro
    bssn = BSSNVars(N)
    bssn.set_bssn_vars(state[:NUM_BSSN_VARS, :])

    # Hydro helper (EOS gamma matches TOVEvolution_corrected main)
    hydro = PerfectFluid(
        eos=IdealGasEOS(gamma=2.0),
        spacetime_mode='dynamic',
        atmosphere=AtmosphereParams(),
    )
    hydro.set_matter_vars(state, bssn, grid)

    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    rho0, vr, p, eps, W, h, success = hydro._get_primitives(bssn, grid.r)
    prim = {'rho0': rho0, 'vr': vr, 'p': p, 'eps': eps, 'W': W, 'h': h, 'success': success}
    return prim, state, bssn


def plot_profiles_for_snapshot(t: float, prim: dict, bssn: BSSNVars, grid: Grid, outdir: Path):
    ng = NUM_GHOSTS
    r_in = grid.r[ng:-ng]
    rho = prim['rho0'][ng:-ng]
    p = prim['p'][ng:-ng]
    vr = prim['vr'][ng:-ng]
    lapse = np.asarray(bssn.lapse)[ng:-ng]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax[0, 0].plot(r_in, rho, color='firebrick', lw=2)
    ax[0, 0].set_xlabel('r')
    ax[0, 0].set_ylabel('ρ₀')
    ax[0, 0].set_title('Baryon density')
    ax[0, 0].grid(True, alpha=0.3)

    ax[0, 1].plot(r_in, p, color='navy', lw=2)
    ax[0, 1].set_xlabel('r')
    ax[0, 1].set_ylabel('p')
    ax[0, 1].set_yscale('log')
    ax[0, 1].set_title('Pressure')
    ax[0, 1].grid(True, which='both', alpha=0.3)

    ax[1, 0].plot(r_in, vr, color='darkgreen', lw=2)
    ax[1, 0].set_xlabel('r')
    ax[1, 0].set_ylabel('v^r')
    ax[1, 0].set_title('Radial velocity')
    ax[1, 0].grid(True, alpha=0.3)

    ax[1, 1].plot(r_in, lapse, color='black', lw=2)
    ax[1, 1].set_xlabel('r')
    ax[1, 1].set_ylabel('α (lapse)')
    ax[1, 1].set_title('Lapse')
    ax[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f'TOV snapshot profiles at t = {t:.3f}')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / f'state_t{t:.3f}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  • Saved {fname}")


def plot_central_density(times: np.ndarray, rho_center: np.ndarray, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(times, rho_center, color='black', lw=2)
    plt.xlabel('t')
    plt.ylabel('ρ_c(t)')
    plt.title('Central density evolution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = outdir / 'rho_central_evolution.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  • Saved {fname}")


def main():
    ap = argparse.ArgumentParser(description='Plot TOV snapshots produced by TOVEvolution_corrected.py')
    ap.add_argument('snap_dir', type=str, help='Directory containing state_t*.npy files')
    ap.add_argument('--outdir', type=str, default='plots/tov', help='Output directory for plots')
    ap.add_argument('--spacing', choices=['linear', 'cubic'], default='linear', help='Grid spacing to rebuild r')
    ap.add_argument('--rmax', type=float, default=20.0, help='Domain r_max used for plotting')
    ap.add_argument('--profiles', action='store_true', help='Generate per-snapshot profile plots')
    ap.add_argument('--central', action='store_true', help='Generate central density time series plot')

    args = ap.parse_args()
    snap_dir = Path(args.snap_dir)
    outdir = Path(args.outdir)

    if not snap_dir.exists():
        print(f"[ERROR] Snapshot directory not found: {snap_dir}")
        sys.exit(1)

    pairs = find_snapshots(snap_dir)
    if not pairs:
        print(f"[ERROR] No snapshot files found in {snap_dir}")
        sys.exit(2)

    # Pre-infer N from first snapshot
    first_state = np.load(pairs[0][1])
    # Build a temporary hydro to get NUM_VARS
    tmp_hydro = PerfectFluid(IdealGasEOS(gamma=2.0), atmosphere=AtmosphereParams())
    num_vars = StateVector(tmp_hydro).NUM_VARS
    if first_state.size % num_vars != 0:
        print(f"[ERROR] Snapshot size {first_state.size} not divisible by NUM_VARS={num_vars}")
        sys.exit(3)
    N = first_state.size // num_vars

    grid = build_grid_for_snapshot(N=N, r_max=args.rmax, spacing=args.spacing)
    print(f"Using grid: N={grid.N}, r_max={args.rmax}, spacing={args.spacing}")

    times = []
    rho_c = []

    for t, path in pairs:
        y = np.load(path)
        prim, state2d, bssn = primitives_from_snapshot(y, grid)

        # Record central density
        center_idx = NUM_GHOSTS
        times.append(t)
        rho_c.append(prim['rho0'][center_idx])

        if args.profiles:
            plot_profiles_for_snapshot(t, prim, bssn, grid, outdir)

    if args.central:
        plot_central_density(np.array(times), np.array(rho_c), outdir)

    if not (args.central or args.profiles):
        print("[INFO] No action specified. Use --profiles and/or --central.")


if __name__ == '__main__':
    main()

