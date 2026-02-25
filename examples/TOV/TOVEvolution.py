"""
TOV Star Evolution — entry point.

Evolves a TOV star using either the Numba (CPU) or JAX (GPU/CPU) backend.
Backend is selected via JAX_RUN env variable (default: 1 = JAX).

Usage:
    python examples/TOV/TOVEvolution.py
    JAX_RUN=0 python examples/TOV/TOVEvolution.py       # Numba backend
    JAX_RUN=1 NUM_POINTS=200 T_FINAL=100 python ...     # JAX, small run
"""

import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — must happen before any engrenage imports
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', '..'))

# ---------------------------------------------------------------------------
# TOV config and utilities
# ---------------------------------------------------------------------------
from examples.TOV.config import TOVConfig
from examples.TOV.utils.io import (
    find_latest_snapshot,
    load_snapshot_from_hdf5,
    validate_restart_consistency,
)
from examples.TOV.utils.diagnostics import diagnose_t0_residuals

# ---------------------------------------------------------------------------
# Physics imports
# ---------------------------------------------------------------------------
from source.core.grid import Grid
from source.core.spacing import LinearSpacing
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground

from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS
from source.bssn.tensoralgebra import get_bar_gamma_LL

from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver, LLFRiemannSolver, HLLCRiemannSolver
from source.matter.hydro.cons2prim import prim_to_cons
from source.matter.hydro.geometry import GeometryState

# ---------------------------------------------------------------------------
# TOV solver and initial data
# ---------------------------------------------------------------------------
from examples.TOV.tov_solver import load_or_solve_tov_iso
import examples.TOV.tov_initial_data_interpolated as tov_id


# =============================================================================
# Helper: physics setup
# =============================================================================

def setup_physics(cfg):
    """Build Grid, background, and hydro from TOVConfig."""
    spacing = LinearSpacing(cfg.num_points, cfg.r_max)
    eos = cfg.get_eos()
    atmosphere = cfg.get_atmosphere_params()

    base_recon = create_reconstruction(cfg.reconstructor)

    riemann_name = cfg.riemann_solver.lower()
    if riemann_name == "hll":
        riemann = HLLRiemannSolver(atmosphere=atmosphere)
    elif riemann_name == "hllc":
        riemann = HLLCRiemannSolver(atmosphere=atmosphere)
    else:
        riemann = LLFRiemannSolver(atmosphere=atmosphere)

    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere=atmosphere,
        reconstructor=base_recon,
        riemann_solver=riemann,
        solver_method=cfg.solver_method,
    )

    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    return grid, background, hydro


# =============================================================================
# Helper: restart detection
# =============================================================================

def check_restart(cfg, output_dir):
    """Detect previous snapshot and return (restart_info, t_start, step_offset)."""
    if not (cfg.enable_restart and cfg.enable_data_saving):
        return None, 0.0, 0

    print("=" * 70)
    print("CHECKING FOR RESTART DATA")
    print("=" * 70)

    restart_info = find_latest_snapshot(output_dir, suffix=cfg.plot_suffix)

    if restart_info is None:
        print("NO RESTART DATA FOUND - Starting from TOV solution\n")
        return None, 0.0, 0

    print(f"Restart snapshot found:")
    print(f"  Step: {restart_info['step']}")
    print(f"  Time: {restart_info['time']:.6e}")
    print(f"  File: {restart_info['snapshot_file']}")

    snapshot_data = load_snapshot_from_hdf5(
        restart_info['snapshot_file'],
        restart_info['step_name'],
    )
    validate_restart_consistency(snapshot_data, {
        'num_points': cfg.num_points,
        'r_max': cfg.r_max,
    })

    restart_info['state_2d'] = snapshot_data['state_2d']
    t_start = snapshot_data['time']
    step_offset = snapshot_data['step']

    print(f"\n✓ Restart validated successfully")
    print(f"  Resuming from step {step_offset}, t={t_start:.6e}")
    print(f"  Will evolve to t_final={cfg.t_final:.6e}")
    print("=" * 70 + "\n")

    return restart_info, t_start, step_offset


# =============================================================================
# Helper: initial data
# =============================================================================

def build_initial_data(cfg, grid, background, eos, atmosphere, hydro,
                       restart_info, plots_dir):
    """
    Solve TOV, create initial data, apply perturbation if requested.

    Returns (initial_state_2d, prim_tuple, tov_solution).
    """
    print("Solving TOV equations...")
    tov_solution = load_or_solve_tov_iso(
        K=cfg.K, Gamma=cfg.Gamma, rho_central=cfg.rho_central,
        r_max=cfg.r_max, accuracy="high",
    )
    print(f"TOV Solution: M={tov_solution.M_star:.6f}, "
          f"R_iso={tov_solution.R_iso:.3f}, "
          f"R_schw={tov_solution.R_schw:.3f}, "
          f"C={tov_solution.C:.4f}\n")

    if restart_info is not None:
        # State comes from snapshot; recompute prim_tuple for diagnostics only.
        initial_state_2d = restart_info['state_2d']
        _, prim_tuple = tov_id.create_initial_data_iso(
            tov_solution, grid, background, eos,
            atmosphere=atmosphere,
            polytrope_K=cfg.K, polytrope_Gamma=cfg.Gamma,
            interp_order=11,
        )
        return initial_state_2d, prim_tuple, tov_solution

    # --- Fresh start ---
    print("Creating initial data from TOV solution...")
    initial_state_2d, prim_tuple = tov_id.create_initial_data_iso(
        tov_solution, grid, background, eos,
        atmosphere=atmosphere,
        polytrope_K=cfg.K, polytrope_Gamma=cfg.Gamma,
        interp_order=11,
    )

    # Collapse perturbation (Font et al. 2001): inward radial velocity
    if cfg.collapse_perturbation:
        A_pert = cfg.collapse_amplitude
        r_star = tov_solution.R_iso
        r = grid.r

        mask = r < r_star
        delta_vr = np.zeros_like(r)
        delta_vr[mask] = -A_pert * (r[mask] / r_star) * np.sin(np.pi * r[mask] / r_star)

        rho0, vr, p, eps = prim_tuple
        vr_pert = vr + delta_vr

        bssn_tmp = BSSNVars(grid.N)
        bssn_tmp.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
        bar_gamma_tmp = get_bar_gamma_LL(grid.r, bssn_tmp.h_LL, background)
        geom_tmp = GeometryState.from_bssn_1d(
            alpha=bssn_tmp.lapse,
            beta_r=bssn_tmp.shift_U[:, 0] * background.inverse_scaling_vector[:, 0],
            phi=bssn_tmp.phi,
            gamma_rr=np.exp(4.0 * bssn_tmp.phi) * bar_gamma_tmp[:, 0, 0],
        )
        D_new, Sr_new, tau_new = prim_to_cons(rho0, vr_pert, p, geom_tmp, eos)
        initial_state_2d[NUM_BSSN_VARS + 0, :] = D_new
        initial_state_2d[NUM_BSSN_VARS + 1, :] = Sr_new
        initial_state_2d[NUM_BSSN_VARS + 2, :] = tau_new
        prim_tuple = (rho0, vr_pert, p, eps)
        print(f"  Collapse perturbation applied: A={A_pert}, "
              f"max|delta_vr|={np.max(np.abs(delta_vr)):.3e}")

    # t=0 diagnostics
    diagnose_t0_residuals(initial_state_2d, grid, background, hydro)

    if not cfg.skip_plots:
        tov_id.plot_initial_comparison(
            tov_solution, initial_state_2d, grid, prim_tuple,
            output_dir=plots_dir, suffix=cfg.plot_suffix,
        )
        tov_id.plot_hamiltonian_constraint_iso(
            tov_solution, initial_state_2d, grid, background, hydro,
            cfg.K, cfg.Gamma, cfg.rho_central,
            output_dir=plots_dir, show=False,
        )

    return initial_state_2d, prim_tuple, tov_solution


# =============================================================================
# Main
# =============================================================================

def main():
    cfg = TOVConfig()

    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Backend setup (must happen before any JAX imports)
    if cfg.jax_run:
        os.environ['ENGRENAGE_BACKEND'] = 'jax'
        import jax
        jax.config.update("jax_enable_x64", True)

    # Print header
    print("=" * 70)
    if cfg.jax_run and cfg.evolution_mode == "dynamic":
        print("TOV Star Evolution - JAX Backend (Full BSSN + Hydro)")
    elif cfg.jax_run:
        print("TOV Star Evolution - JAX Backend (Cowling Approximation)")
    elif cfg.evolution_mode == "dynamic":
        print("TOV Star Evolution - Full BSSN + Hydro (Dynamic Spacetime)")
    else:
        print("TOV Star Evolution - Cowling Approximation (Fixed Spacetime)")
    print("=" * 70)
    print("Using ISOTROPIC coordinates (conformally flat spatial metric)")
    print()

    # Setup
    grid, background, hydro = setup_physics(cfg)
    atmosphere = cfg.get_atmosphere_params()
    output_dir = cfg.get_output_dir(script_dir)

    # Info
    print("=" * 70)
    print("ATMOSPHERE CONFIGURATION")
    print("=" * 70)
    print(f"  rho_floor = {atmosphere.rho_floor:.2e}")
    print(f"  p_floor   = {atmosphere.p_floor:.2e}")
    print(f"  tau_atm   = {atmosphere.tau_atm:.2e}")
    print(f"  v_max     = {atmosphere.v_max}")
    print()

    if cfg.enable_data_saving:
        print(f"Data saving enabled:")
        print(f"  Output dir: {output_dir}")
        print(f"  Snapshot interval: {cfg.snapshot_interval} timesteps")
        print(f"  Evolution interval: {cfg.evolution_interval} timesteps")
        print()

    print(f"Grid:    N={grid.N}, r_max={cfg.r_max}, dr_min={grid.min_dr:.4e}")
    print(f"EOS:     K={cfg.K}, Gamma={cfg.Gamma}")
    print(f"Star:    rho_c={cfg.rho_central:.2e}")
    print(f"Recon:   {cfg.reconstructor}  |  Riemann: {cfg.riemann_solver}  |  C2P: {cfg.solver_method}")
    print(f"Mode:    {cfg.evolution_mode}  |  Backend: {'JAX' if cfg.jax_run else 'Numba'}")
    if cfg.jax_run:
        import jax
        print(f"Devices: {jax.devices()}")
    print()

    # Restart detection
    restart_info, t_start, step_offset = check_restart(cfg, output_dir)

    # Initial data
    initial_state_2d, prim_tuple, tov_solution = build_initial_data(
        cfg, grid, background, cfg.get_eos(), atmosphere, hydro,
        restart_info, plots_dir,
    )

    # Dispatch to backend
    evolve_kwargs = dict(
        cfg=cfg,
        initial_state_2d=initial_state_2d,
        prim_tuple=prim_tuple,
        tov_solution=tov_solution,
        grid=grid,
        background=background,
        hydro=hydro,
        atmosphere=atmosphere,
        plots_dir=plots_dir,
        output_dir=output_dir,
        restart_info=restart_info,
        t_start=t_start,
        step_offset=step_offset,
    )

    if cfg.jax_run:
        from examples.TOV.evolver.jax import evolve_jax
        evolve_jax(**evolve_kwargs)
    else:
        from examples.TOV.evolver.numba import evolve_numba
        evolve_numba(**evolve_kwargs)


if __name__ == '__main__':
    main()
