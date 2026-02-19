"""
TOV Star Evolution with backend dispatch (Numba or JAX).

Evolves a TOV star using either the Numba-based or JAX-based hydro pipeline.
Backend is selected via the JAX_RUN flag below:

    JAX_RUN = True   - JAX backend (auto-detect GPU)
    JAX_RUN = False   - Numba/CPU backend

Usage:
    python examples/TOV/TOVEvolution.py
"""

import numpy as np
import sys
import os
import time

# ==================================================================
# BACKEND TOGGLE (hardcoded)
# ==================================================================
JAX_RUN = os.environ.get("JAX_RUN", "1").lower() in ("1", "true", "yes")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Engrenage core - add parent directories to path
sys.path.insert(0, os.path.join(script_dir, '..', '..'))

# Set backend environment variable BEFORE any engrenage imports
if JAX_RUN:
    os.environ['ENGRENAGE_BACKEND'] = 'jax'

from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground

# BSSN
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS
from source.bssn.tensoralgebra import get_bar_gamma_LL, get_bar_A_LL, get_hat_D_bar_gamma_LL
from source.bssn.constraintsdiagnostic import get_constraints_diagnostic

# Full BSSN+Hydro RHS (for dynamic mode)
from source.core.rhsevolution import get_rhs

# Hydro
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import PolytropicEOS, IdealGasEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver, LLFRiemannSolver, HLLCRiemannSolver
from source.matter.hydro.cons2prim import prim_to_cons
from source.matter.hydro.atmosphere import AtmosphereParams
from source.matter.hydro.geometry import GeometryState

# Local TOV modules (isotropic coordinates)
from examples.TOV.tov_solver import load_or_solve_tov_iso
import examples.TOV.tov_initial_data_interpolated as tov_id

# TOV utilities and plotting
import examples.TOV.utils_TOVEvolution as utils
from examples.TOV.utils_TOVEvolution import (SimulationDataManager, evolve_fixed_timestep,
                                              get_star_folder_name,
                                              compute_baryon_mass, compute_constraints)

# Conditional JAX imports
if JAX_RUN:
    from functools import partial
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from source.matter.hydro.jax.valencia_jax import (
        HydroGeometry,
        compute_hydro_rhs_cowling,
    )
    # Dynamic mode imports (BSSN + hydro coupling)
    from source.bssn.jax.bssngeometry import build_bssn_background, build_derivative_stencils
    from source.bssn.jax.boundaries_jax import fill_bssn_boundaries_jax
    from source.core.rhsevolution_jax import get_rhs_bssn_hydro_jax, NUM_HYDRO_VARS, HYDRO_PARITY
    from source.bssn.bssnstatevariables import (
        BSSN_PARITY, BSSN_ASYMP_POWER, BSSN_ASYMP_OFFSET,
        idx_phi, idx_lapse, idx_K,
    )


# =============================================================================
# Numba RHS and time-stepping functions
# =============================================================================

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
    """Apply atmospheric floors to densitized conservative variables.

    Resets to atmosphere if:
    1. D < threshold (density floor)
    2. tau + D < 0 (unphysical, would fail cons2prim)

    Matches JAX behavior but adds safety check for tau+D constraint.
    """
    rho_hard_floor = rho_threshold or 100.0 * atmosphere.rho_floor
    idx_D, idx_Sr, idx_tau = NUM_BSSN_VARS, NUM_BSSN_VARS + 1, NUM_BSSN_VARS + 2

    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
    e6phi = np.exp(6.0 * bssn_vars.phi)

    D = state_2d[idx_D, :]
    tau = state_2d[idx_tau, :]

    # Atmosphere mask: D below threshold OR tau+D unphysical
    atm_mask = (D < rho_hard_floor * e6phi) | (tau + D < 0)

    if np.any(atm_mask):
        state_2d[idx_D, atm_mask] = atmosphere.rho_floor * e6phi[atm_mask]
        state_2d[idx_Sr, atm_mask] = 0.0
        state_2d[idx_tau, atm_mask] = atmosphere.tau_atm * e6phi[atm_mask]

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


# =============================================================================
# JAX-only functions (boundary conditions and geometry extraction)
# =============================================================================

if JAX_RUN:
    @partial(jax.jit, static_argnums=(3,))
    def fill_boundaries_jax(D, Sr, tau, num_ghosts):
        """
        Apply boundary conditions for hydro variables in JAX (functional).

        Inner boundary (r=0): parity reflection
            D(ghost)   = +D(mirror)    (even)
            Sr(ghost)  = -Sr(mirror)   (odd)
            tau(ghost) = +tau(mirror)  (even)

        Outer boundary: zero-gradient (outflow)
        """
        NG = num_ghosts

        # Inner boundary: parity reflection
        mirror = jnp.flip(jnp.array([D[NG:2*NG], Sr[NG:2*NG], tau[NG:2*NG]]), axis=1)

        D = D.at[:NG].set(mirror[0])         # parity +1
        Sr = Sr.at[:NG].set(-mirror[1])       # parity -1
        tau = tau.at[:NG].set(mirror[2])      # parity +1

        # Outer boundary: zero-gradient
        D = D.at[-NG:].set(D[-NG - 1])
        Sr = Sr.at[-NG:].set(Sr[-NG - 1])
        tau = tau.at[-NG:].set(tau[-NG - 1])

        return D, Sr, tau


    def build_cowling_geometry(initial_state_2d, grid, background):
        """
        Extract static geometry from BSSN variables and build a HydroGeometry
        plus source/connection data for Cowling mode.

        This is done once (in NumPy) and the results are transferred to JAX arrays.

        Returns:
            geom: HydroGeometry namedtuple
            source_data: (K_LL, dalpha_dx, hatD_beta_U, hatD_gamma_LL) tuple of jnp arrays
            connection_data: (hat_christoffel,) tuple of jnp arrays
        """
        N = grid.N
        r = grid.r

        # Extract BSSN variables
        bssn_vars = BSSNVars(N)
        bssn_vars.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])

        # Compute BSSN derivatives
        bssn_d1 = grid.get_d1_metric_quantities(initial_state_2d)

        # Core geometry
        alpha = np.asarray(bssn_vars.lapse, dtype=np.float64)
        beta_U = np.asarray(bssn_vars.shift_U, dtype=np.float64) * background.inverse_scaling_vector
        phi = np.asarray(bssn_vars.phi, dtype=np.float64)
        e4phi = np.exp(4.0 * phi)
        e6phi = np.exp(6.0 * phi)

        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        gamma_LL = e4phi[:, None, None] * bar_gamma_LL
        gamma_UU = np.linalg.inv(gamma_LL)

        # Extrinsic curvature K_ij
        K_scalar = np.asarray(bssn_vars.K, dtype=np.float64)
        bar_A_LL = get_bar_A_LL(r, bssn_vars, background)
        K_LL = e4phi[:, None, None] * bar_A_LL + (K_scalar / 3.0)[:, None, None] * gamma_LL

        # Lapse derivative
        dalpha_dx = np.asarray(bssn_d1.lapse)

        # Shift derivative and covariant derivative
        dbeta_dx = (
            background.inverse_scaling_vector[:, :, None] * np.asarray(bssn_d1.shift_U)
            + bssn_vars.shift_U[:, :, None] * background.d1_inverse_scaling_vector
        )
        hat_chris = background.hat_christoffel
        hatD_beta_U = np.transpose(dbeta_dx, (0, 2, 1)) + np.einsum('xjik,xk->xij', hat_chris, beta_U)

        # Covariant derivative of metric
        dphi_dx = np.asarray(bssn_d1.phi)
        hat_D_bar_gamma = get_hat_D_bar_gamma_LL(r, bssn_vars.h_LL, bssn_d1.h_LL, background)
        hatD_gamma_LL = e4phi[:, None, None, None] * (
            4.0 * dphi_dx[:, :, None, None] * bar_gamma_LL[:, None, :, :]
            + np.transpose(hat_D_bar_gamma, (0, 3, 1, 2))
        )

        # Build HydroGeometry (transfers to JAX device via jnp.asarray)
        geom = HydroGeometry(
            alpha=jnp.asarray(alpha),
            beta_r=jnp.asarray(beta_U[:, 0]),
            gamma_rr=jnp.asarray(gamma_LL[:, 0, 0]),
            e6phi=jnp.asarray(e6phi),
            e4phi=jnp.asarray(e4phi),
            beta_U=jnp.asarray(beta_U),
            gamma_LL=jnp.asarray(gamma_LL),
            gamma_UU=jnp.asarray(gamma_UU),
        )

        source_data = (
            jnp.asarray(K_LL),
            jnp.asarray(dalpha_dx),
            jnp.asarray(hatD_beta_U),
            jnp.asarray(hatD_gamma_LL),
        )
        connection_data = (jnp.asarray(hat_chris),)

        return geom, source_data, connection_data


# =============================================================================
# Main
# =============================================================================

def main():
    """Main execution."""

    # ==================================================================
    # CONFIGURATION
    # ==================================================================
    r_max = 100.0
    num_points = int(os.environ.get("NUM_POINTS", 4000))
    K = 100.0
    Gamma = 2.0
    rho_central = 1.28e-3
    t_final = float(os.environ.get("T_FINAL", "4000"))
    FOLDER_NAME_EVOL = f"tov_evolution_data_refact_rmax{r_max}" + ("_jax" if JAX_RUN else "")

    SNAPSHOT_INTERVAL = 500  # Save full domain every N timesteps (None to disable)
    EVOLUTION_INTERVAL = 500  # Save time series every N timesteps (None to disable)

    # Reconstructor: "wenoz" (wz), "weno5" (w5), "mp5" (mp5), "minmod" (md), "mc" (mc)
    RECONSTRUCTOR_NAME = "mp5"  # "wenoz", "weno5", "mp5", "minmod", "mc"

    # Cons2prim solver: "newton" (fast, needs good guess) or "kastaun" (robust, guaranteed convergence)
    SOLVER_METHOD = "newton"  # "newton" or "kastaun"

    # Riemann solver: "hll", "hllc", or "llf"
    RIEMANN_SOLVER = "hll"  # "hll", "hllc", or "llf"

    # Evolution mode: "cowling" or "dynamic"
    EVOLUTION_MODE = os.environ.get("EVOLUTION_MODE", "dynamic")

    # ==================================================================
    # RESTART CONFIGURATION
    # ==================================================================
    ENABLE_RESTART = os.environ.get("ENABLE_RESTART", "1").lower() in ("1", "true", "yes")

    # Atmosphere config
    rho_floor_base = 1e-16
    p_floor_base = K * (rho_floor_base)**Gamma
    ATMOSPHERE = AtmosphereParams(
        rho_floor=rho_floor_base,
        p_floor=p_floor_base
    )

    # Data saving
    ENABLE_DATA_SAVING = os.environ.get("ENABLE_DATA_SAVING", "1").lower() in ("1", "true", "yes")
    SAVE_TIMESERIES = ENABLE_DATA_SAVING
    SKIP_PLOTS = os.environ.get("SKIP_PLOTS", "0").lower() in ("1", "true", "yes")

    # ==================================================================
    # SETUP
    # ==================================================================
    spacing = LinearSpacing(num_points, r_max)
    eos = IdealGasEOS(gamma=Gamma)

    # 1. RECONSTRUCTOR BASE (dispatches Numba/JAX automatically)
    base_recon = create_reconstruction(RECONSTRUCTOR_NAME)

    # 2. RIEMANN SOLVER
    if RIEMANN_SOLVER.lower() == "hll":
        riemann = HLLRiemannSolver(atmosphere=ATMOSPHERE)
    elif RIEMANN_SOLVER.lower() == "hllc":
        riemann = HLLCRiemannSolver(atmosphere=ATMOSPHERE)
    else:
        riemann = LLFRiemannSolver(atmosphere=ATMOSPHERE)

    # 3. HYDRO (dispatches cons2prim solver based on backend)
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

    # Create suffix for plot filenames based on evolution mode and backend
    PLOT_SUFFIX = ("_cow" if EVOLUTION_MODE == "cowling" else "_dyn") + ("_jax" if JAX_RUN else "")

    # Print header
    print("="*70)
    if JAX_RUN and EVOLUTION_MODE == "dynamic":
        print("TOV Star Evolution - JAX Backend (Full BSSN + Hydro)")
    elif JAX_RUN:
        print("TOV Star Evolution - JAX Backend (Cowling Approximation)")
    elif EVOLUTION_MODE == "dynamic":
        print("TOV Star Evolution - Full BSSN + Hydro (Dynamic Spacetime)")
    else:
        print("TOV Star Evolution - Cowling Approximation (Fixed Spacetime)")
    print("="*70)
    print("Using ISOTROPIC coordinates (conformally flat spatial metric)")

    # ==================================================================
    # DATA SAVING CONFIGURATION
    # ==================================================================
    DATA_ROOT_DIR = os.path.join(script_dir, FOLDER_NAME_EVOL)

    # Create star-specific folder based on parameters
    star_folder = get_star_folder_name(rho_central, num_points, K, Gamma, EVOLUTION_MODE, RECONSTRUCTOR_NAME)
    OUTPUT_DIR = os.path.join(DATA_ROOT_DIR, star_folder)

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

    # Time integration
    integration_method = 'fixed'
    cfl_factor = 0.1

    print(f"Grid: N={grid.N}, r_max={r_max}, dr_min={grid.min_dr}")
    print(f"EOS: K={K}, Gamma={Gamma}")
    print(f"Cons2prim solver: {SOLVER_METHOD}")
    print(f"Riemann solver: {RIEMANN_SOLVER}")
    if JAX_RUN:
        print(f"Backend: JAX ({jax.devices()})")
    else:
        print(f"Backend: Numba")
    print()

    # ==================================================================
    # RESTART DETECTION
    # ==================================================================
    restart_info = None
    initial_state_2d = None
    tov_solution = None
    prim_tuple = None
    t_start = 0.0
    step_offset = 0

    if ENABLE_RESTART and ENABLE_DATA_SAVING:
        print("="*70)
        print("CHECKING FOR RESTART DATA")
        print("="*70)
        restart_info = utils.find_latest_snapshot(OUTPUT_DIR, suffix=PLOT_SUFFIX)

    if restart_info is not None:
        print(f"Restart snapshot found:")
        print(f"  Step: {restart_info['step']}")
        print(f"  Time: {restart_info['time']:.6e}")
        print(f"  File: {restart_info['snapshot_file']}")

        # Load snapshot
        snapshot_data = utils.load_snapshot_from_hdf5(
            restart_info['snapshot_file'],
            restart_info['step_name']
        )

        # Validate consistency
        metadata = utils.load_metadata(OUTPUT_DIR, suffix=PLOT_SUFFIX)
        utils.validate_restart_consistency(snapshot_data, {
            'num_points': num_points,
            'r_max': r_max,
            'K': K,
            'Gamma': Gamma,
            'rho_floor': ATMOSPHERE.rho_floor,
            'p_floor': ATMOSPHERE.p_floor
        })

        # Set initial state from snapshot
        initial_state_2d = snapshot_data['state_2d']
        t_start = snapshot_data['time']
        step_offset = snapshot_data['step']

        print(f"\n✓ Restart validated successfully")
        print(f"  Resuming from step {step_offset}, t={t_start:.6e}")
        print(f"  Will evolve to t_final={t_final:.6e}")
        print("="*70 + "\n")

    else:
        if ENABLE_RESTART:
            print("="*70)
            print("NO RESTART DATA FOUND - Starting from TOV solution")
            print("="*70 + "\n")

    # ==================================================================
    # SOLVE TOV (always, since the solution is cached and fast)
    # ==================================================================
    print("Solving TOV equations...")
    tov_solution = load_or_solve_tov_iso(
        K=K, Gamma=Gamma, rho_central=rho_central,
        r_max=r_max, accuracy="high"
    )
    print(f"TOV Solution: M={tov_solution.M_star:.6f}, R_iso={tov_solution.R_iso:.3f}, R_schw={tov_solution.R_schw:.3f}, C={tov_solution.C:.4f}\n")

    # ==================================================================
    # INITIAL DATA
    # ==================================================================
    if initial_state_2d is None:
        print("Creating initial data from TOV solution...")
        initial_state_2d, prim_tuple = tov_id.create_initial_data_iso(
            tov_solution, grid, background, eos,
            atmosphere=ATMOSPHERE,
            polytrope_K=K, polytrope_Gamma=Gamma,
            interp_order=11
        )

        # Diagnostics: check discrete hydrostatic balance at t=0
        utils.diagnose_t0_residuals(initial_state_2d, grid, background, hydro)

        # Initial-data diagnostics
        if not SKIP_PLOTS:
            tov_id.plot_initial_comparison(tov_solution, initial_state_2d, grid, prim_tuple,
                                           output_dir=plots_dir, suffix=PLOT_SUFFIX)

            # Hamiltonian constraint diagnostic
            tov_id.plot_hamiltonian_constraint_iso(tov_solution, initial_state_2d, grid, background, hydro,
                                                    K, Gamma, rho_central, output_dir=plots_dir, show=False)

    # ==================================================================
    # EVOLUTION
    # ==================================================================
    if JAX_RUN:
        _evolve_jax(
            initial_state_2d, prim_tuple, tov_solution,
            grid, background, hydro, ATMOSPHERE,
            K, Gamma, rho_central, cfl_factor,
            t_final, num_points, r_max,
            RECONSTRUCTOR_NAME, SOLVER_METHOD, RIEMANN_SOLVER, EVOLUTION_MODE,
            ENABLE_DATA_SAVING, SAVE_TIMESERIES,
            SNAPSHOT_INTERVAL, EVOLUTION_INTERVAL,
            OUTPUT_DIR, PLOT_SUFFIX, plots_dir,
            restart_info, t_start, step_offset,
            SKIP_PLOTS,
        )
    else:
        _evolve_numba(
            initial_state_2d, prim_tuple, tov_solution,
            grid, background, hydro, ATMOSPHERE,
            K, Gamma, rho_central, cfl_factor,
            t_final, num_points, r_max,
            RECONSTRUCTOR_NAME, SOLVER_METHOD, RIEMANN_SOLVER, EVOLUTION_MODE,
            ENABLE_DATA_SAVING, SAVE_TIMESERIES,
            SNAPSHOT_INTERVAL, EVOLUTION_INTERVAL,
            OUTPUT_DIR, PLOT_SUFFIX, plots_dir,
            integration_method, restart_info, t_start, step_offset,
            SKIP_PLOTS,
        )


# =============================================================================
# Numba evolution path
# =============================================================================

def _evolve_numba(initial_state_2d, prim_tuple, tov_solution,
                  grid, background, hydro, ATMOSPHERE,
                  K, Gamma, rho_central, cfl_factor,
                  t_final, num_points, r_max,
                  RECONSTRUCTOR_NAME, SOLVER_METHOD, RIEMANN_SOLVER, EVOLUTION_MODE,
                  ENABLE_DATA_SAVING, SAVE_TIMESERIES,
                  SNAPSHOT_INTERVAL, EVOLUTION_INTERVAL,
                  OUTPUT_DIR, PLOT_SUFFIX, plots_dir,
                  integration_method, restart_info, t_start, step_offset,
                  SKIP_PLOTS=False):
    """Numba-based evolution with checkpoint structure and restart support."""

    # Setup depends on evolution mode
    if EVOLUTION_MODE == "dynamic":
        print("\n" + "="*70)
        print("EVOLUTION MODE: DYNAMIC (Full BSSN + Hydro)")
        print("  - Spacetime evolves with matter")
        print("  - 1+log slicing for lapse")
        print("  - Gamma-driver for shift")
        print("="*70)

        bssn_fixed = initial_state_2d[:NUM_BSSN_VARS, :].copy()
        bssn_d1_fixed = grid.get_d1_metric_quantities(initial_state_2d)

        # Wrapper for dynamic RK4 step
        def rk4_step_wrapper(state_flat, dt, grid, background, hydro,
                            bssn_fixed_unused, bssn_d1_fixed_unused, atmosphere):
            return rk4_step_dynamic(state_flat, dt, grid, background, hydro, atmosphere)

        selected_rk4_step = rk4_step_wrapper

    else:  # cowling mode
        print("\n" + "="*70)
        print("EVOLUTION MODE: COWLING (Fixed Spacetime)")
        print("  - BSSN variables frozen at t=0")
        print("  - Only hydro evolves")
        print("="*70)

        bssn_fixed = initial_state_2d[:NUM_BSSN_VARS, :].copy()
        bssn_d1_fixed = grid.get_d1_metric_quantities(initial_state_2d)
        selected_rk4_step = rk4_step

    # Initialize data manager
    data_manager = SimulationDataManager(OUTPUT_DIR, grid, hydro,
                                        enable_saving=ENABLE_DATA_SAVING,
                                        suffix=PLOT_SUFFIX,
                                        restart_mode=(restart_info is not None))

    if integration_method == 'fixed':
        dt = cfl_factor * grid.min_dr
        num_steps_total = int(t_final / dt)

        # For restart: calculate remaining steps
        time_remaining = t_final - t_start
        num_steps_remaining = int(time_remaining / dt)

        if restart_info is not None:
            print(f"\nRestart mode:")
            print(f"  Current time: {t_start:.6e}")
            print(f"  Target time:  {t_final:.6e}")
            print(f"  Remaining:    {time_remaining:.6e} ({num_steps_remaining} steps)")
            print(f"  dt={dt:.6e} (CFL={cfl_factor})")
        else:
            print(f"\nEvolving with fixed dt={dt:.6f} (CFL={cfl_factor}) to t_final={t_final} ({num_steps_total} steps) using RK4")

        # Start timing
        evolution_start_time = time.time()

        # Save metadata
        if ENABLE_DATA_SAVING:
            tov_sol_to_save = tov_solution
            if restart_info is not None:
                metadata = utils.load_metadata(OUTPUT_DIR, suffix=PLOT_SUFFIX)
                tov_sol_to_save = tov_solution if tov_solution is not None else metadata.get('tov_solution', {})
            data_manager.save_metadata(
                tov_sol_to_save, ATMOSPHERE, dt, integration_method,
                K=K, Gamma=Gamma, rho_central=rho_central,
                r_max=r_max, num_points=num_points, t_final=t_final,
                reconstructor=RECONSTRUCTOR_NAME, solver_method=SOLVER_METHOD,
                riemann_solver=RIEMANN_SOLVER, evolution_mode=EVOLUTION_MODE,
                cfl_factor=cfl_factor
            )

        # First-step diagnostics (skip for restart)
        state_t1 = None
        t_1 = None
        if restart_info is None:
            state_t1 = selected_rk4_step(initial_state_2d.flatten(), dt, grid, background, hydro,
                                         bssn_fixed, bssn_d1_fixed, ATMOSPHERE).reshape((grid.NUM_VARS, grid.N))
            t_1 = dt

        # Storage for checkpoint states
        checkpoint_states = {}
        checkpoint_times = {}
        all_series = []

        # Choose evolution strategy based on restart
        if restart_info is None:
            # Normal mode: use checkpoint structure
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
        else:
            checkpoint_1 = step_offset + max(1, num_steps_remaining // 3)
            checkpoint_2 = step_offset + max(2, 2 * num_steps_remaining // 3)
            checkpoint_3 = num_steps_total

            print(f"\n{'='*70}")
            print(f"Evolution (restart mode):")
            print(f"  Starting step: {step_offset}")
            print(f"  Starting time: {t_start:.6e}")
            print(f"  Target step:   {checkpoint_3}")
            print(f"  Target time:   {t_final:.6e}")
            print(f"{'='*70}\n")

        if restart_info is None:
            # ===== NORMAL MODE: Use checkpoint structure =====
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
                state_cp3 = state_cp2
                t_cp3 = t_cp2
                checkpoint_states[3] = state_cp3.copy()
                checkpoint_times[3] = t_cp3
                steps_final = checkpoint_1 + steps_cp2

            state_t10000 = checkpoint_states[3]
            t_10000 = checkpoint_times[3]
            num_steps = steps_final

        else:
            # ===== RESTART MODE: Simple direct evolution =====
            print(f"Evolving from step {step_offset} to step {checkpoint_3}...")
            state_final, steps_done, t_final_actual, series_restart = evolve_fixed_timestep(
                initial_state_2d, dt, num_steps_remaining, grid, background,
                hydro, bssn_fixed, bssn_d1_fixed, ATMOSPHERE, selected_rk4_step,
                method='rk4', t_start=t_start, reference_state=initial_state_2d,
                step_offset=step_offset,
                data_manager=data_manager,
                snapshot_interval=SNAPSHOT_INTERVAL,
                evolution_interval=EVOLUTION_INTERVAL)

            checkpoint_states[1] = state_final.copy()
            checkpoint_times[1] = t_final_actual
            checkpoint_states[2] = state_final.copy()
            checkpoint_times[2] = t_final_actual
            checkpoint_states[3] = state_final.copy()
            checkpoint_times[3] = t_final_actual
            all_series.append(series_restart)

            state_t10000 = state_final
            t_10000 = t_final_actual
            num_steps = steps_done
            steps_final = steps_done

            print(f"  -> Reached step {step_offset + steps_done}, t={t_final_actual:.6e}")

        # Build full-series arrays
        try:
            if len(all_series) == 3:
                times_full = np.concatenate([all_series[0]['t'], all_series[1]['t'][1:], all_series[2]['t'][1:]])
                Mb_full = np.concatenate([all_series[0]['Mb'], all_series[1]['Mb'][1:], all_series[2]['Mb'][1:]])
                rho_c_full = np.concatenate([all_series[0]['rho_c'], all_series[1]['rho_c'][1:], all_series[2]['rho_c'][1:]])
                v_c_full = np.concatenate([all_series[0]['v_c'], all_series[1]['v_c'][1:], all_series[2]['v_c'][1:]])
            elif len(all_series) == 2:
                times_full = np.concatenate([all_series[0]['t'], all_series[1]['t'][1:]])
                Mb_full = np.concatenate([all_series[0]['Mb'], all_series[1]['Mb'][1:]])
                rho_c_full = np.concatenate([all_series[0]['rho_c'], all_series[1]['rho_c'][1:]])
                v_c_full = np.concatenate([all_series[0]['v_c'], all_series[1]['v_c'][1:]])
            elif len(all_series) == 1:
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
        # SAVE TIME SERIES
        # ==================================================================
        if SAVE_TIMESERIES and len(times_full) > 0:
            timeseries_filename = "timeseries.npz"
            timeseries_path = os.path.join(OUTPUT_DIR, timeseries_filename)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            np.savez(timeseries_path,
                     times=times_full,
                     rho_central=rho_c_full,
                     v_central=v_c_full,
                     Mb=Mb_full,
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
    evol_time = time.time() - evolution_start_time

    # Reference primitives for error computation
    bssn_ref = BSSNVars(grid.N)
    bssn_ref.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(initial_state_2d, bssn_ref, grid)
    rho_ref, _, p_ref, _, _, _, _ = hydro._get_primitives(bssn_ref, grid.r)

    # Build states and times for plot_evolution
    states = [initial_state_2d, checkpoint_states[1], checkpoint_states[2], checkpoint_states[3]]
    times = [0.0, checkpoint_times[1], checkpoint_times[2], checkpoint_times[3]]

    R_star = tov_solution.R_iso

    if not SKIP_PLOTS:
        try:
            if 'times_full' in locals() and len(times_full) > 0:
                utils.plot_evolution(states, times, grid, hydro, rho_ref, p_ref,
                                     Mb_series=Mb_full, rho_c_series=rho_c_full,
                                     times_series=times_full,
                                     suffix=PLOT_SUFFIX, R_star=R_star)
            else:
                utils.plot_evolution(states, times, grid, hydro, rho_ref, p_ref,
                                    suffix=PLOT_SUFFIX, R_star=R_star)
        except Exception:
            utils.plot_evolution(states, times, grid, hydro, rho_ref, p_ref,
                                suffix=PLOT_SUFFIX, R_star=R_star)

        # Plot BSSN variables evolution
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
        print(f"  Growth factor: {max_H_f/max_H_0:.2f}x" if max_H_0 > 1e-20 else "  (initial ~0)")

        print(f"\nMomentum constraint |M_r|:")
        print(f"  t=0:     max|M_r| = {max_M_0:.3e}")
        print(f"  t=final: max|M_r| = {max_M_f:.3e}")
        print(f"  Growth factor: {max_M_f/max_M_0:.2f}x" if max_M_0 > 1e-20 else "  (initial ~0)")
        print("="*70)

    # Print detailed statistics
    bssn_0 = BSSNVars(grid.N)
    bssn_0.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(initial_state_2d, bssn_0, grid)
    rho0_0, vr_0, p_0, _, _, _, success_0 = hydro._get_primitives(bssn_0, grid.r)

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

    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

    # First-step diagnostics (only available in non-restart mode)
    if state_t1 is not None:
        bssn_1 = BSSNVars(grid.N)
        bssn_1.set_bssn_vars(state_t1[:NUM_BSSN_VARS, :])
        hydro.set_matter_vars(state_t1, bssn_1, grid)
        rho0_1, vr_1, p_1, _, _, _, success_1 = hydro._get_primitives(bssn_1, grid.r)

        delta_rho_1 = np.abs(rho0_1[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)
        delta_P_1 = np.abs(p_1[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)
        max_err_rho_1 = np.max(delta_rho_1)
        max_err_P_1 = np.max(delta_P_1)

    # Compute errors at checkpoints
    delta_rho_cp1 = np.abs(rho0_cp1[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)
    delta_rho_cp2 = np.abs(rho0_cp2[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)
    delta_rho_cp3 = np.abs(rho0_cp3[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)

    delta_P_cp1 = np.abs(p_cp1[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)
    delta_P_cp2 = np.abs(p_cp2[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)
    delta_P_cp3 = np.abs(p_cp3[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)

    max_err_rho_cp1 = np.max(delta_rho_cp1)
    max_err_rho_cp2 = np.max(delta_rho_cp2)
    max_err_rho_cp3 = np.max(delta_rho_cp3)

    max_err_P_cp1 = np.max(delta_P_cp1)
    max_err_P_cp2 = np.max(delta_P_cp2)
    max_err_P_cp3 = np.max(delta_P_cp3)

    if state_t1 is not None:
        growth_rho = max_err_rho_cp3 / max_err_rho_1 if max_err_rho_1 > 1e-15 else 0
        growth_P = max_err_P_cp3 / max_err_P_1 if max_err_P_1 > 1e-15 else 0

    t_cp1 = checkpoint_times[1]
    t_cp2 = checkpoint_times[2]
    t_cp3 = checkpoint_times[3]

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
        print(f"   delta_rho_c/rho_c (first step):       {abs(rho0_1[NUM_GHOSTS] - rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")
    print(f"   delta_rho_c/rho_c (1/3):              {abs(rho0_cp1[NUM_GHOSTS] - rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")
    print(f"   delta_rho_c/rho_c (2/3):              {abs(rho0_cp2[NUM_GHOSTS] - rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")
    print(f"   delta_rho_c/rho_c (final):            {abs(rho0_cp3[NUM_GHOSTS] - rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")

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
    if ENABLE_DATA_SAVING:
        execution_time = time.time() - evolution_start_time
        data_manager.finalize(execution_time_seconds=execution_time)

    # ==================================================================
    # PLOT CONSTRAINT VIOLATIONS (Dynamic mode only)
    # ==================================================================
    if EVOLUTION_MODE == "dynamic" and ENABLE_DATA_SAVING and not SKIP_PLOTS:
        print("\n" + "="*70)
        print("Plotting constraint violation evolution...")
        print("="*70)
        utils.plot_constraints_evolution(OUTPUT_DIR, suffix=PLOT_SUFFIX)

    print("\n" + "="*70)
    print("Evolution complete. Plots saved:")
    print("  1. tov_solution.png                    - TOV solution")
    print("  2. tov_initial_data_comparison.png     - TOV vs Initial data at t=0")
    print("  3. tov_hamiltonian_constraint_iso.png  - Hamiltonian constraint at t=0")
    print(f"  4. tov_evolution.png                   - Hydro evolution at checkpoints:")
    print(f"                                           t=0 -> t={t_cp1:.3e} (1/3) -> t={t_cp2:.3e} (2/3) -> t={t_cp3:.3e} (final)")
    if EVOLUTION_MODE == "dynamic":
        print(f"  5. tov_bssn_evolution.png              - BSSN variables: t=0 -> t={t_cp3:.3e} (metric evolution)")
        print(f"  6. constraints_evolution{PLOT_SUFFIX}.png     - BSSN constraint violations")
    else:
        print(f"  5. tov_bssn_evolution.png              - BSSN variables: t=0 -> t={t_cp3:.3e} (Cowling check)")
    print(f"  Wall time: {evol_time:.2f}s")
    print("="*70)
    _per_step_ms = evol_time / max(num_steps_total, 1) * 1000
    print(f"BENCHMARK_RESULT: jit_s=0.000 per_step_ms={_per_step_ms:.3f} total_s={evol_time:.3f} n_steps={num_steps_total}")


# =============================================================================
# JAX evolution path
# =============================================================================

def _evolve_jax(initial_state_2d, prim_tuple, tov_solution,
                grid, background, hydro, ATMOSPHERE,
                K, Gamma, rho_central, cfl_factor,
                t_final, num_points, r_max,
                RECONSTRUCTOR_NAME, SOLVER_METHOD, RIEMANN_SOLVER, EVOLUTION_MODE,
                ENABLE_DATA_SAVING, SAVE_TIMESERIES,
                SNAPSHOT_INTERVAL, EVOLUTION_INTERVAL,
                OUTPUT_DIR, PLOT_SUFFIX, plots_dir,
                restart_info=None, t_start=0.0, step_offset=0,
                SKIP_PLOTS=False):
    """JAX-based evolution with JIT-compiled step function.

    Supports both Cowling (fixed spacetime) and Dynamic (full BSSN+hydro) modes.
    Now also supports restart from checkpoints.
    """
    IS_DYNAMIC = (EVOLUTION_MODE == "dynamic")

    dt = cfl_factor * grid.min_dr

    # Adjust num_steps for restart
    if restart_info is not None:
        t_remaining = t_final - t_start
        num_steps_total = int(t_remaining / dt)
        print(f"  RESTART MODE: Resuming from t={t_start:.6e}, evolving remaining t={t_remaining:.3e} ({num_steps_total} steps)")
    else:
        num_steps_total = int(t_final / dt)
    N = grid.N
    num_vars = grid.NUM_VARS

    if IS_DYNAMIC:
        print(f"\n{'='*70}")
        print("EVOLUTION MODE: DYNAMIC (Full BSSN + Hydro) — JAX")
        print("  - Spacetime evolves with matter")
        print("  - 1+log slicing for lapse")
        print("  - Shift fixed to zero (TOV static gauge)")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print("EVOLUTION MODE: COWLING (Fixed Spacetime) — JAX")
        print("  - BSSN variables frozen at t=0")
        print("  - Only hydro evolves")
        print(f"{'='*70}")

    print(f"\ndt={dt:.6f} (CFL={cfl_factor}), t_final={t_final}, steps={num_steps_total}")
    print(f"JAX devices: {jax.devices()}")

    # Initialize data manager
    restart_mode = (restart_info is not None)
    data_manager = SimulationDataManager(
        OUTPUT_DIR, grid, hydro,
        enable_saving=ENABLE_DATA_SAVING,
        suffix=PLOT_SUFFIX,
        restart_mode=restart_mode
    )

    # Save metadata
    if ENABLE_DATA_SAVING:
        data_manager.save_metadata(
            tov_solution, ATMOSPHERE, dt, "rk4",
            K=K, Gamma=Gamma, rho_central=rho_central,
            r_max=r_max, num_points=num_points, t_final=t_final,
            reconstructor=RECONSTRUCTOR_NAME, solver_method=SOLVER_METHOD,
            riemann_solver=RIEMANN_SOLVER, evolution_mode=EVOLUTION_MODE,
            cfl_factor=cfl_factor
        )

    # Store reference primitives
    rho_ref, vr_ref, p_ref, eps_ref = prim_tuple
    W_ref = np.ones(N)  # Static initial data
    h_ref = 1.0 + eps_ref + p_ref / (rho_ref + 1e-30)
    success_ref = np.ones(N, dtype=bool)

    # Save initial snapshot (step 0) - only if not restarting
    if ENABLE_DATA_SAVING and restart_info is None:
        Ham_0, Mom_0 = compute_constraints(initial_state_2d, grid, background, hydro)
        data_manager.save_snapshot(0, 0.0, initial_state_2d, rho_ref, vr_ref, p_ref,
                                   eps_ref, W_ref, h_ref, Ham=Ham_0, Mom=Mom_0)
        data_manager.add_evolution_point(0, 0.0, initial_state_2d,
                                        rho_ref, vr_ref, p_ref, eps_ref, W_ref, h_ref, success_ref,
                                        rho_ref, vr_ref, p_ref, eps_ref, W_ref, h_ref, success_ref,
                                        Ham=Ham_0, Mom=Mom_0)

    # Time series tracking
    Mb0 = compute_baryon_mass(grid, initial_state_2d, rho_ref, vr_ref, p_ref, eps_ref, W_ref, h_ref)
    if restart_info is None:
        all_times = [0.0]
        all_Mb = [Mb0]
        all_rho_c = [float(rho_ref[NUM_GHOSTS])]
        all_v_c = [float(vr_ref[NUM_GHOSTS])]
    else:
        # Restart: start with empty lists (will append as evolution progresses)
        all_times = []
        all_Mb = []
        all_rho_c = []
        all_v_c = []

    # Preserve config value before loop variable shadowing
    rho_central_config = rho_central

    # ==================================================================
    # ATMOSPHERE CONSTANTS (shared by both modes)
    # ==================================================================
    eos_type = 'ideal_gas'
    _rho_floor = float(ATMOSPHERE.rho_floor)
    _p_floor = float(ATMOSPHERE.p_floor)
    _v_max = float(ATMOSPHERE.v_max)
    _gm1 = Gamma - 1.0

    # Precompute atmosphere values
    _eps_atm = K * _rho_floor**_gm1 / _gm1
    _p_atm = K * _rho_floor**Gamma
    _h_atm = 1.0 + _eps_atm + _p_atm / _rho_floor
    _atm_threshold_factor = 100.0 * _rho_floor

    # ==================================================================
    # MODE-DEPENDENT SETUP
    # ==================================================================
    if IS_DYNAMIC:
        # ----- DYNAMIC MODE: Build BSSN background + derivative matrices -----
        print("\nBuilding BSSNBackground and DerivativeStencils...")
        t0_setup = time.perf_counter()
        state_jax = jnp.array(initial_state_2d)
        bssn_bg = build_bssn_background(grid, background)
        deriv_stencils = build_derivative_stencils(grid)
        dr_jax = jnp.array(grid.dr)
        dx_hydro = float(grid.derivs.dx)

        # BSSN gauge parameters
        eta = 0.0           # Gamma driver damping (eta=0 for static spacetime, following GRoovy)
        sigma_base = 1.0    # KO dissipation base coefficient
        max_iter = 100
        tol = 1e-12

        # Boundary condition arrays (BSSN + hydro)
        parity_jax = jnp.array(np.concatenate([
            np.array(BSSN_PARITY, dtype=np.float64),
            np.array(HYDRO_PARITY, dtype=np.float64),
        ]))
        asymp_power_jax = jnp.array(np.concatenate([
            np.array(BSSN_ASYMP_POWER, dtype=np.float64),
            np.zeros(NUM_HYDRO_VARS),
        ]))
        asymp_offset_jax = jnp.array(np.concatenate([
            np.array(BSSN_ASYMP_OFFSET, dtype=np.float64),
            np.zeros(NUM_HYDRO_VARS),
        ]))

        # Dirichlet BC: exact Schwarzschild values at ghost cell radii
        _tau_atm_phys = _rho_floor * _h_atm - _p_atm - _rho_floor
        r_ghost = grid.r[-NUM_GHOSTS:]
        M_star = tov_solution.M_star
        boundary_ref = np.zeros((num_vars, NUM_GHOSTS))
        for i, rg in enumerate(r_ghost):
            factor = 1.0 + M_star / (2.0 * rg)
            phi_val = np.log(factor)           # ln(1 + M/(2r))
            alpha_val = (1.0 - M_star / (2.0 * rg)) / factor
            e6phi_val = factor**6
            boundary_ref[idx_phi, i] = phi_val
            # h_ij=0, K=0, A_ij=0, lambda=0, shift=0, B=0 (already zeros)
            boundary_ref[idx_lapse, i] = alpha_val
            boundary_ref[NUM_BSSN_VARS, i] = e6phi_val * _rho_floor       # D
            boundary_ref[NUM_BSSN_VARS + 2, i] = e6phi_val * _tau_atm_phys  # tau
        boundary_ref_jax = jnp.array(boundary_ref)

        print(f"  Setup in {time.perf_counter() - t0_setup:.3f}s")

        # ----- JIT-compiled step function (Dynamic) -----
        @jax.jit
        def apply_bcs(state):
            return fill_bssn_boundaries_jax(
                state, bssn_bg.r, NUM_GHOSTS,
                parity_jax, asymp_power_jax, asymp_offset_jax,
                "dirichlet", boundary_ref=boundary_ref_jax
            )

        @jax.jit
        def rhs_fn(state):
            return get_rhs_bssn_hydro_jax(
                state, bssn_bg, deriv_stencils, dr_jax,
                NUM_GHOSTS, num_vars,
                sigma_base, eta,
                eos_type, Gamma, K,
                _rho_floor, _p_floor, _v_max, 10.0,
                RECONSTRUCTOR_NAME, SOLVER_METHOD, max_iter,
                tol, dx_hydro,
                "zero_gradient",  # RHS BC: smooth. State BC handles pinning via apply_bcs.
                fix_shift=True    # TOV: 1+log lapse, shift=0 (no Gamma-driver)
            )

        @jax.jit
        def jit_step_dynamic(state, dt_val):
            """RK4 step + BCs + atmosphere reset for dynamic mode."""
            k1 = rhs_fn(state)
            k2 = rhs_fn(apply_bcs(state + 0.5 * dt_val * k1))
            k3 = rhs_fn(apply_bcs(state + 0.5 * dt_val * k2))
            k4 = rhs_fn(apply_bcs(state + dt_val * k3))
            state_new = apply_bcs(state + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

            # Atmosphere reset for hydro variables (phi is dynamic)
            phi = state_new[idx_phi]
            e6phi = jnp.exp(6.0 * phi)

            D = state_new[NUM_BSSN_VARS]
            Sr = state_new[NUM_BSSN_VARS + 1]
            tau_var = state_new[NUM_BSSN_VARS + 2]

            D_atm_val = e6phi * _rho_floor
            tau_atm_val = e6phi * (_rho_floor * _h_atm - _p_atm - _rho_floor)
            threshold = _atm_threshold_factor * e6phi
            atm_mask = D < threshold

            D = jnp.where(atm_mask, D_atm_val, D)
            Sr = jnp.where(atm_mask, 0.0, Sr)
            tau_var = jnp.where(atm_mask, tau_atm_val, tau_var)

            state_new = state_new.at[NUM_BSSN_VARS].set(D)
            state_new = state_new.at[NUM_BSSN_VARS + 1].set(Sr)
            state_new = state_new.at[NUM_BSSN_VARS + 2].set(tau_var)

            return state_new

    else:
        # ----- COWLING MODE: Build static geometry -----
        print("\nBuilding HydroGeometry from BSSN variables...")
        t0_geom = time.perf_counter()
        geom, source_data, connection_data = build_cowling_geometry(initial_state_2d, grid, background)
        dx_val = float(grid.derivs.dx)
        print(f"  Geometry built in {time.perf_counter() - t0_geom:.3f}s")

        # Extract initial hydro state
        D0 = jnp.asarray(initial_state_2d[NUM_BSSN_VARS + 0, :])
        Sr0 = jnp.asarray(initial_state_2d[NUM_BSSN_VARS + 1, :])
        tau0 = jnp.asarray(initial_state_2d[NUM_BSSN_VARS + 2, :])

        eos_params = {'gamma': Gamma, 'K': K}
        atm_params = {
            'rho_floor': _rho_floor,
            'p_floor': _p_floor,
            'v_max': _v_max,
            'W_max': 10.0,
            'tol': 1e-10,
            'max_iter': 100,
        }

        # ----- JIT-compiled step function (Cowling) -----
        @jax.jit
        def jit_step_cowling(D, Sr, tau, geom_arg):
            """One complete RK4 step + atmosphere reset, JIT-compiled."""
            def _rhs(D_in, Sr_in, tau_in):
                D_bc, Sr_bc, tau_bc = fill_boundaries_jax(D_in, Sr_in, tau_in, NUM_GHOSTS)
                return compute_hydro_rhs_cowling(
                    D_bc, Sr_bc, tau_bc, geom_arg,
                    dx_val, NUM_GHOSTS,
                    eos_type, eos_params, atm_params,
                    RECONSTRUCTOR_NAME, SOLVER_METHOD,
                    source_data=source_data,
                    connection_data=connection_data,
                )

            # RK4 stages
            dD1, dSr1, dtau1 = _rhs(D, Sr, tau)

            D2 = D + 0.5 * dt * dD1
            Sr2 = Sr + 0.5 * dt * dSr1
            tau2 = tau + 0.5 * dt * dtau1
            dD2, dSr2, dtau2 = _rhs(D2, Sr2, tau2)

            D3 = D + 0.5 * dt * dD2
            Sr3 = Sr + 0.5 * dt * dSr2
            tau3 = tau + 0.5 * dt * dtau2
            dD3, dSr3, dtau3 = _rhs(D3, Sr3, tau3)

            D4 = D + dt * dD3
            Sr4 = Sr + dt * dSr3
            tau4 = tau + dt * dtau3
            dD4, dSr4, dtau4 = _rhs(D4, Sr4, tau4)

            D_new = D + (dt / 6.0) * (dD1 + 2.0*dD2 + 2.0*dD3 + dD4)
            Sr_new = Sr + (dt / 6.0) * (dSr1 + 2.0*dSr2 + 2.0*dSr3 + dSr4)
            tau_new = tau + (dt / 6.0) * (dtau1 + 2.0*dtau2 + 2.0*dtau3 + dtau4)

            # Atmosphere reset inline
            D_atm = geom_arg.e6phi * _rho_floor
            tau_atm = geom_arg.e6phi * (_rho_floor * _h_atm - _p_atm - _rho_floor)

            threshold = _atm_threshold_factor * geom_arg.e6phi
            atm_mask = D_new < threshold

            D_new = jnp.where(atm_mask, D_atm, D_new)
            Sr_new = jnp.where(atm_mask, 0.0, Sr_new)
            tau_new = jnp.where(atm_mask, tau_atm, tau_new)

            return D_new, Sr_new, tau_new

    # ==================================================================
    # PRINT INITIAL STATE
    # ==================================================================
    print(f"  Initial D_c   = {float(initial_state_2d[NUM_BSSN_VARS, NUM_GHOSTS]):.6e}")
    print(f"  Initial Sr_c  = {float(initial_state_2d[NUM_BSSN_VARS+1, NUM_GHOSTS]):.6e}")
    print(f"  Initial tau_c = {float(initial_state_2d[NUM_BSSN_VARS+2, NUM_GHOSTS]):.6e}")

    # ==================================================================
    # JIT WARMUP
    # ==================================================================
    print("\nJIT compiling full step (RK4 + BC + RHS + atm)...")
    t0_jit = time.perf_counter()

    if IS_DYNAMIC:
        dt_jax = jnp.float64(dt)
        state_test = jit_step_dynamic(state_jax, dt_jax).block_until_ready()
        jit_time = time.perf_counter() - t0_jit
        print(f"  Full step JIT compilation: {jit_time:.2f}s")

        # Benchmark
        print("Benchmarking step execution...")
        t0_bench = time.perf_counter()
        n_bench = 10
        for _ in range(n_bench):
            state_test = jit_step_dynamic(state_test, dt_jax)
        state_test.block_until_ready()
        bench_time = (time.perf_counter() - t0_bench) / n_bench * 1000
    else:
        D_test, Sr_test, tau_test = jit_step_cowling(D0, Sr0, tau0, geom)
        jax.block_until_ready((D_test, Sr_test, tau_test))
        jit_time = time.perf_counter() - t0_jit
        print(f"  Full step JIT compilation: {jit_time:.2f}s")

        # Benchmark
        print("Benchmarking step execution...")
        t0_bench = time.perf_counter()
        n_bench = 20
        for _ in range(n_bench):
            D_test, Sr_test, tau_test = jit_step_cowling(D_test, Sr_test, tau_test, geom)
        jax.block_until_ready((D_test, Sr_test, tau_test))
        bench_time = (time.perf_counter() - t0_bench) / n_bench * 1000

    print(f"  Single step execution: {bench_time:.2f} ms")
    step_time_ms = bench_time

    # ==================================================================
    # EVOLUTION SETUP
    # ==================================================================
    if restart_info is None:
        # Normal mode: 3 checkpoints for diagnostics
        checkpoint_steps = {
            1: max(1, num_steps_total // 3),
            2: max(2, 2 * num_steps_total // 3),
            3: num_steps_total,
        }
    else:
        # Restart mode: only final checkpoint (for diagnostics)
        checkpoint_steps = {
            3: num_steps_total,
        }

    print(f"\n{'='*70}")
    print(f"Evolution checkpoints:")
    for cp_id, cp_step in checkpoint_steps.items():
        t_checkpoint = (cp_step * dt) + t_start
        print(f"  step {cp_step + step_offset:6d}: {cp_id}/3 (t~{t_checkpoint:.3e})")
    print(f"{'='*70}\n")

    checkpoint_states_dict = {}
    checkpoint_times = {}

    # BSSN data (fixed for Cowling, evolving for Dynamic)
    bssn_fixed = initial_state_2d[:NUM_BSSN_VARS, :].copy()

    def extract_primitives(state_2d):
        """Extract primitives from state using Numba pipeline."""
        bssn = BSSNVars(N)
        bssn.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
        hydro.set_matter_vars(state_2d, bssn, grid)
        rho0, vr, p, eps, W, h, success = hydro._get_primitives(bssn, grid.r)
        return rho0, vr, p, eps, W, h, success

    if IS_DYNAMIC:
        def get_state_2d():
            """Get current state as numpy (15, N) array."""
            return np.array(state)

        def get_checkpoint_state():
            """Save checkpoint state."""
            return np.array(state)
    else:
        def get_state_2d():
            """Build full state from Cowling hydro variables."""
            s = np.zeros((NUM_BSSN_VARS + 3, N))
            s[:NUM_BSSN_VARS, :] = bssn_fixed
            s[NUM_BSSN_VARS + 0, :] = np.asarray(D)
            s[NUM_BSSN_VARS + 1, :] = np.asarray(Sr)
            s[NUM_BSSN_VARS + 2, :] = np.asarray(tau)
            return s

        def get_checkpoint_state():
            """Save checkpoint state (hydro tuple for Cowling)."""
            return (np.array(D), np.array(Sr), np.array(tau))

    # ==================================================================
    # EVOLUTION LOOP
    # ==================================================================
    t0_evol = time.perf_counter()
    t = 0.0
    PRINT_INTERVAL = 500
    NAN_CHECK_INTERVAL = 500

    if IS_DYNAMIC:
        state = state_jax
    else:
        D, Sr, tau = D0, Sr0, tau0

    print("===== Evolution diagnostics (per step) =====")
    if IS_DYNAMIC:
        print("Columns: step | t | rho_central | max_drho/rho@r | max_vr@r | alpha_c | K_c | dMb/Mb | c2p_fails")
    else:
        print("Columns: step | t | rho_central | max_drho/rho@r | max_vr@r | max_Sr@r | c2p_fails")
    print("-" * 120)

    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
    r_interior = grid.r[interior]
    checkpoint_set = set(checkpoint_steps.values())

    # --- Capture state after first step for diagnostic plots ---
    state_after_first_step = None

    for step in range(1, num_steps_total + 1):
        # --- Step ---
        if IS_DYNAMIC:
            state = jit_step_dynamic(state, dt_jax)
        else:
            D, Sr, tau = jit_step_cowling(D, Sr, tau, geom)
        t += dt

        # --- Save state after first step for diagnostic plots ---
        if step == 1:
            if IS_DYNAMIC:
                state_after_first_step = np.array(state)
            else:
                state_after_first_step = get_state_2d()

        # NaN detection (periodic, lightweight)
        if step % NAN_CHECK_INTERVAL == 0:
            if IS_DYNAMIC:
                state.block_until_ready()
                if bool(jnp.any(jnp.isnan(state[idx_phi]))):
                    print(f"\nERROR: NaN detected at step {step}, t={t:.6e}. Halting evolution.")
                    break
            else:
                jax.block_until_ready(D)
                if bool(jnp.any(jnp.isnan(D))):
                    print(f"\nERROR: NaN detected at step {step}, t={t:.6e}. Halting evolution.")
                    break

        # Diagnostics and data saving
        should_print = (step % PRINT_INTERVAL == 0)
        should_save_evol = ENABLE_DATA_SAVING and EVOLUTION_INTERVAL and (step % EVOLUTION_INTERVAL == 0)
        should_save_snap = ENABLE_DATA_SAVING and SNAPSHOT_INTERVAL and (step % SNAPSHOT_INTERVAL == 0)

        if should_print or should_save_evol or should_save_snap:
            state_2d = get_state_2d()
            rho0, vr, p, eps, W, h, success = extract_primitives(state_2d)

            Ham, Mom = None, None
            if should_save_evol or should_save_snap:
                Ham, Mom = compute_constraints(state_2d, grid, background, hydro)

            if should_save_evol:
                data_manager.add_evolution_point(
                    step + step_offset, t + t_start, state_2d,
                    rho0, vr, p, eps, W, h, success,
                    rho_ref, vr_ref, p_ref, eps_ref, W_ref, h_ref, success_ref,
                    Ham=Ham, Mom=Mom
                )
                Mb = compute_baryon_mass(grid, state_2d, rho0, vr, p, eps, W, h)
                all_times.append(t + t_start)
                all_Mb.append(Mb)
                all_rho_c.append(float(rho0[NUM_GHOSTS]))
                all_v_c.append(float(vr[NUM_GHOSTS]))

                if step % (EVOLUTION_INTERVAL * 10) == 0:
                    data_manager.flush_evolution_buffer()

            if should_save_snap:
                data_manager.save_snapshot(step + step_offset, t + t_start, state_2d, rho0, vr, p, eps, W, h,
                                           Ham=Ham, Mom=Mom)

            if should_print:
                rho_central = float(rho0[NUM_GHOSTS])
                rho_int = rho0[interior]
                rho_init_int = rho_ref[interior]

                # Only compute relative error where rho > 100*rho_floor (exclude atmosphere)
                rho_threshold = 100.0 * ATMOSPHERE.rho_floor
                matter_mask = rho_init_int > rho_threshold

                if np.any(matter_mask):
                    rel_rho_err_matter = np.abs(rho_int[matter_mask] - rho_init_int[matter_mask]) / np.abs(rho_init_int[matter_mask])
                    idx_max_matter = np.argmax(rel_rho_err_matter)
                    max_rel_rho_err = float(rel_rho_err_matter[idx_max_matter])
                    r_max_rho = float(r_interior[matter_mask][idx_max_matter])
                else:
                    max_rel_rho_err = 0.0
                    r_max_rho = 0.0

                vr_int = vr[interior]
                idx_max_v = np.argmax(np.abs(vr_int))
                max_abs_v = float(vr_int[idx_max_v])
                r_max_v = float(r_interior[idx_max_v])

                c2p_fail = int(np.sum(~success))

                if IS_DYNAMIC:
                    lapse_c = float(state_2d[idx_lapse, NUM_GHOSTS])
                    K_c = float(state_2d[idx_K, NUM_GHOSTS])
                    dMb = abs(all_Mb[-1] - Mb0) / Mb0 if len(all_Mb) > 1 else 0.0

                    elapsed = time.perf_counter() - t0_evol
                    t_actual = t + t_start
                    frac = (t_actual - t_start) / (t_final - t_start) if restart_info else t / t_final

                    print(f"step {step + step_offset:6d}  t={t_actual:.1e}  rho_c={rho_central:.6e}  "
                          f"max_drho/rho={max_rel_rho_err:.2e}@r={r_max_rho:.2f}  "
                          f"max_vr={max_abs_v:.1e}@r={r_max_v:.1f}  "
                          f"alpha_c={lapse_c:.6f}  K_c={K_c:.6e}  dMb/Mb={dMb:.2e}  "
                          f"c2p_fail={c2p_fail}  [{frac*100:.0f}% {elapsed:.0f}s]")
                else:
                    Sr_np = np.asarray(state_2d[NUM_BSSN_VARS + 1, interior])
                    idx_max_Sr = np.argmax(np.abs(Sr_np))
                    max_Sr = float(np.abs(Sr_np[idx_max_Sr]))
                    r_max_Sr = float(r_interior[idx_max_Sr])

                    t_actual = t + t_start
                    print(f"step {step + step_offset:5d}  t={t_actual:.2e}:  rho_c={rho_central:.6e}  "
                          f"max_drho/rho={max_rel_rho_err:.2e}@r={r_max_rho:.2f}  "
                          f"max_vr={max_abs_v:.3e}@r={r_max_v:.2f}  "
                          f"max_Sr={max_Sr:.2e}@r={r_max_Sr:.2f}  c2p_fail={c2p_fail}")

        # Save checkpoint
        if step in checkpoint_set:
            if IS_DYNAMIC:
                state.block_until_ready()
            else:
                jax.block_until_ready((D, Sr, tau))
            cp_id = [k for k, v in checkpoint_steps.items() if v == step][0]
            checkpoint_states_dict[cp_id] = get_checkpoint_state()
            checkpoint_times[cp_id] = t + t_start
            data_manager.flush_evolution_buffer()
            print(f"  -> Reached checkpoint {cp_id}, t={t + t_start:.6e}")

    evol_time = time.perf_counter() - t0_evol
    data_manager.flush_evolution_buffer()

    # Fill missing checkpoints if halted early
    for cp_id in [1, 2, 3]:
        if cp_id not in checkpoint_states_dict:
            if IS_DYNAMIC:
                state.block_until_ready()
            else:
                jax.block_until_ready((D, Sr, tau))
            checkpoint_states_dict[cp_id] = get_checkpoint_state()
            checkpoint_times[cp_id] = t + t_start

    print(f"\nEvolution complete!")
    print(f"  Pure evolution time: {evol_time:.2f}s")
    print(f"  Steps/second: {num_steps_total / evol_time:.0f}")

    # ==================================================================
    # SAVE TIME SERIES
    # ==================================================================
    times_full = np.array(all_times)
    rho_c_full = np.array(all_rho_c)
    v_c_full = np.array(all_v_c)
    Mb_full = np.array(all_Mb)

    if ENABLE_DATA_SAVING and SAVE_TIMESERIES and len(times_full) > 0:
        timeseries_path = os.path.join(OUTPUT_DIR, "timeseries.npz")
        np.savez(timeseries_path,
                 times=times_full,
                 rho_central=rho_c_full,
                 v_central=v_c_full,
                 Mb=Mb_full,
                 num_points=num_points,
                 K=K,
                 Gamma=Gamma,
                 rho_central_initial=rho_central_config,
                 r_max=r_max,
                 dt=dt,
                 num_steps=num_steps_total)
        print(f"\nTime series saved to: {timeseries_path}")

    # ==================================================================
    # RECONSTRUCT FULL STATES FOR DIAGNOSTICS
    # ==================================================================
    if IS_DYNAMIC:
        state_cp0 = initial_state_2d
        state_cp3_full = np.array(state)
        state_cp1 = checkpoint_states_dict[1]
        state_cp2 = checkpoint_states_dict[2]
        state_cp3 = checkpoint_states_dict[3]
    else:
        def _build_full(hydro_tuple):
            s = np.zeros((NUM_BSSN_VARS + 3, N))
            s[:NUM_BSSN_VARS, :] = bssn_fixed
            s[NUM_BSSN_VARS + 0, :] = hydro_tuple[0]
            s[NUM_BSSN_VARS + 1, :] = hydro_tuple[1]
            s[NUM_BSSN_VARS + 2, :] = hydro_tuple[2]
            return s

        state_cp0 = np.zeros((NUM_BSSN_VARS + 3, N))
        state_cp0[:NUM_BSSN_VARS, :] = bssn_fixed
        state_cp0[NUM_BSSN_VARS:, :] = initial_state_2d[NUM_BSSN_VARS:, :]

        state_cp1 = _build_full(checkpoint_states_dict[1])
        state_cp2 = _build_full(checkpoint_states_dict[2])
        state_cp3 = _build_full(checkpoint_states_dict[3])
        state_cp3_full = state_cp3  # same for Cowling (no evolving BSSN)

    # Reference primitives for error computation
    bssn_ref = BSSNVars(N)
    bssn_ref.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(initial_state_2d, bssn_ref, grid)
    rho_ref_diag, _, p_ref_diag, _, _, _, _ = hydro._get_primitives(bssn_ref, grid.r)

    # Recover primitives at all checkpoints
    print("\nRecovering primitives at all checkpoints...")
    rho0_0, vr_0, p_0, eps_0, W_0, h_0, success_0 = extract_primitives(state_cp0)
    rho0_cp1, vr_cp1, p_cp1, eps_cp1, W_cp1, h_cp1, success_cp1 = extract_primitives(state_cp1)
    rho0_cp2, vr_cp2, p_cp2, eps_cp2, W_cp2, h_cp2, success_cp2 = extract_primitives(state_cp2)
    rho0_cp3, vr_cp3, p_cp3, eps_cp3, W_cp3, h_cp3, success_cp3 = extract_primitives(state_cp3)

    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
    t_cp1 = checkpoint_times[1]
    t_cp2 = checkpoint_times[2]
    t_cp3 = checkpoint_times[3]

    # Compute errors at checkpoints
    delta_rho_cp1 = np.abs(rho0_cp1[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)
    delta_rho_cp2 = np.abs(rho0_cp2[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)
    delta_rho_cp3 = np.abs(rho0_cp3[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)

    delta_P_cp1 = np.abs(p_cp1[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)
    delta_P_cp2 = np.abs(p_cp2[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)
    delta_P_cp3 = np.abs(p_cp3[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)

    max_err_rho_cp1 = np.max(delta_rho_cp1)
    max_err_rho_cp2 = np.max(delta_rho_cp2)
    max_err_rho_cp3 = np.max(delta_rho_cp3)

    max_err_P_cp1 = np.max(delta_P_cp1)
    max_err_P_cp2 = np.max(delta_P_cp2)
    max_err_P_cp3 = np.max(delta_P_cp3)

    # Baryon mass at checkpoints
    Mb_cp0 = compute_baryon_mass(grid, state_cp0, rho0_0, vr_0, p_0, eps_0, W_0, h_0)
    Mb_cp1 = compute_baryon_mass(grid, state_cp1, rho0_cp1, vr_cp1, p_cp1, eps_cp1, W_cp1, h_cp1)
    Mb_cp2 = compute_baryon_mass(grid, state_cp2, rho0_cp2, vr_cp2, p_cp2, eps_cp2, W_cp2, h_cp2)
    Mb_cp3 = compute_baryon_mass(grid, state_cp3, rho0_cp3, vr_cp3, p_cp3, eps_cp3, W_cp3, h_cp3)

    # ==================================================================
    # EVOLUTION DIAGNOSTICS (same format as Numba path)
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"EVOLUTION DIAGNOSTICS")
    print(f"  t=0 -> t={t_cp1:.6e} (1/3) -> t={t_cp2:.6e} (2/3) -> t={t_cp3:.6e} (final)")
    print(f"{'='*70}")

    print(f"\n1. VELOCITY EVOLUTION:")
    print(f"   Max |v^r| at t=0:              {np.max(np.abs(vr_0)):.3e}")
    print(f"   Max |v^r| at t={t_cp1:.6e} (1/3):  {np.max(np.abs(vr_cp1)):.3e}")
    print(f"   Max |v^r| at t={t_cp2:.6e} (2/3):  {np.max(np.abs(vr_cp2)):.3e}")
    print(f"   Max |v^r| at t={t_cp3:.6e} (final): {np.max(np.abs(vr_cp3)):.3e}")

    print(f"\n2. CENTRAL DENSITY:")
    print(f"   rho_c at t=0:                  {rho0_0[NUM_GHOSTS]:.6e}")
    print(f"   rho_c at t={t_cp1:.6e} (1/3):  {rho0_cp1[NUM_GHOSTS]:.6e}")
    print(f"   rho_c at t={t_cp2:.6e} (2/3):  {rho0_cp2[NUM_GHOSTS]:.6e}")
    print(f"   rho_c at t={t_cp3:.6e} (final): {rho0_cp3[NUM_GHOSTS]:.6e}")
    print(f"   delta_rho_c/rho_c (1/3):              {abs(rho0_cp1[NUM_GHOSTS] - rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")
    print(f"   delta_rho_c/rho_c (2/3):              {abs(rho0_cp2[NUM_GHOSTS] - rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")
    print(f"   delta_rho_c/rho_c (final):            {abs(rho0_cp3[NUM_GHOSTS] - rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")

    print(f"\n3. DENSITY ERROR (max over domain):")
    print(f"   Max |d_rho|/rho at t={t_cp1:.6e} (1/3):   {max_err_rho_cp1:.3e}")
    print(f"   Max |d_rho|/rho at t={t_cp2:.6e} (2/3):   {max_err_rho_cp2:.3e}")
    print(f"   Max |d_rho|/rho at t={t_cp3:.6e} (final):  {max_err_rho_cp3:.3e}")

    print(f"\n4. PRESSURE ERROR (max over domain):")
    print(f"   Max |dP|/P at t={t_cp1:.6e} (1/3):   {max_err_P_cp1:.3e}")
    print(f"   Max |dP|/P at t={t_cp2:.6e} (2/3):   {max_err_P_cp2:.3e}")
    print(f"   Max |dP|/P at t={t_cp3:.6e} (final):  {max_err_P_cp3:.3e}")

    print(f"\n5. CONS2PRIM STATUS:")
    print(f"   Success at t=0:                {np.sum(success_0)}/{N}")
    print(f"   Success at t={t_cp1:.6e} (1/3):    {np.sum(success_cp1)}/{N}")
    print(f"   Success at t={t_cp2:.6e} (2/3):    {np.sum(success_cp2)}/{N}")
    print(f"   Success at t={t_cp3:.6e} (final):  {np.sum(success_cp3)}/{N}")

    if not np.all(success_cp3):
        failed_idx = np.where(~success_cp3)[0]
        print(f"   Failed points: {failed_idx[:5]} (first 5)")
        print(f"   Failed radii:  {grid.r[failed_idx[:5]]}")

    print(f"\n6. BARYON MASS CONSERVATION:")
    print(f"   Mb(t=0)     = {Mb_cp0:.6f}")
    print(f"   Mb(t=1/3)   = {Mb_cp1:.6f}   dMb/Mb = {abs(Mb_cp1 - Mb_cp0) / Mb_cp0:.3e}")
    print(f"   Mb(t=2/3)   = {Mb_cp2:.6f}   dMb/Mb = {abs(Mb_cp2 - Mb_cp0) / Mb_cp0:.3e}")
    print(f"   Mb(t=final) = {Mb_cp3:.6f}   dMb/Mb = {abs(Mb_cp3 - Mb_cp0) / Mb_cp0:.3e}")

    # ==================================================================
    # MODE-SPECIFIC DIAGNOSTICS
    # ==================================================================
    if IS_DYNAMIC:
        print(f"\n{'='*70}")
        print("CONSTRAINT VIOLATION DIAGNOSTICS")
        print(f"{'='*70}")

        try:
            Ham_0c, Mom_0c = compute_constraints(state_cp0, grid, background, hydro)
            Ham_fc, Mom_fc = compute_constraints(state_cp3, grid, background, hydro)

            max_H_0 = np.max(np.abs(Ham_0c[interior]))
            max_H_f = np.max(np.abs(Ham_fc[interior]))
            max_M_0 = np.max(np.abs(Mom_0c[interior, 0]))
            max_M_f = np.max(np.abs(Mom_fc[interior, 0]))

            print(f"Hamiltonian constraint |H|:")
            print(f"  t=0:     max|H| = {max_H_0:.3e}")
            print(f"  t=final: max|H| = {max_H_f:.3e}")
            print(f"  Growth factor: {max_H_f/max_H_0:.2f}x" if max_H_0 > 1e-20 else "  (initial ~0)")

            print(f"\nMomentum constraint |M_r|:")
            print(f"  t=0:     max|M_r| = {max_M_0:.3e}")
            print(f"  t=final: max|M_r| = {max_M_f:.3e}")
            print(f"  Growth factor: {max_M_f/max_M_0:.2f}x" if max_M_0 > 1e-20 else "  (initial ~0)")
        except Exception as e:
            print(f"  Constraint computation failed: {e}")

        print(f"\n  Lapse at center:")
        print(f"    alpha_c(t=0)     = {state_cp0[idx_lapse, NUM_GHOSTS]:.6f}")
        print(f"    alpha_c(t=final) = {state_cp3[idx_lapse, NUM_GHOSTS]:.6f}")
        print(f"{'='*70}")

    else:
        # Cowling approximation check
        print(f"\n{'='*70}")
        print("COWLING APPROXIMATION CHECK")
        print(f"{'='*70}")

        bssn_0_data = initial_state_2d[:NUM_BSSN_VARS, :]
        bssn_f_data = state_cp3[:NUM_BSSN_VARS, :]

        bssn_change = np.max(np.abs(bssn_f_data - bssn_0_data) / (np.abs(bssn_0_data) + 1e-20))
        print(f"  Max BSSN change (should be 0): {bssn_change:.3e}")

        if bssn_change < 1e-14:
            print("  BSSN variables constant - Cowling approximation verified")
        else:
            print(f"  BSSN variables changed by {bssn_change:.3e}")
            print("    (This should not happen in Cowling mode!)")

    # NaN check
    D_f_np = np.asarray(state_cp3[NUM_BSSN_VARS, :])
    Sr_f_np = np.asarray(state_cp3[NUM_BSSN_VARS + 1, :])
    tau_f_np = np.asarray(state_cp3[NUM_BSSN_VARS + 2, :])
    n_nan = np.sum(np.isnan(D_f_np)) + np.sum(np.isnan(Sr_f_np)) + np.sum(np.isnan(tau_f_np))
    if n_nan > 0:
        print(f"\n  WARNING: {n_nan} NaN values detected!")
    else:
        print(f"\n  No NaNs detected - evolution stable.")

    # ==================================================================
    # Finalize data saving (AFTER diagnostics, matching Numba)
    # ==================================================================
    execution_time = time.perf_counter() - t0_evol
    if ENABLE_DATA_SAVING:
        data_manager.finalize(execution_time_seconds=execution_time)

    # ==================================================================
    # PLOTS (using utils_TOVEvolution for consistency)
    # ==================================================================
    R_star = tov_solution.R_iso

    states = [state_cp0, state_cp1, state_cp2, state_cp3]
    times_cp = [0.0, checkpoint_times[1], checkpoint_times[2], checkpoint_times[3]]

    if not SKIP_PLOTS:
        print("\nGenerating plots...")

        # 1. Plot TOV solution diagnostics
        try:
            print("  - TOV solution diagnostics...")
            utils.plot_tov_diagnostics(tov_solution, r_max, suffix=PLOT_SUFFIX)
        except Exception as e:
            print(f"  plot_tov_diagnostics failed: {e}")

        # 1b. Plot TOV vs initial data comparison (zoom at surface)
        try:
            print("  - TOV vs initial data (surface zoom)...")
            utils.plot_tov_vs_initial_data_zoom(tov_solution, initial_state_2d, grid,
                                               primitives=prim_tuple, window=0.5, suffix=PLOT_SUFFIX)
        except Exception as e:
            print(f"  plot_tov_vs_initial_data_zoom failed: {e}")

        # 2. Plot evolution at checkpoints
        try:
            utils.plot_evolution(
                states, times_cp, grid, hydro, rho_ref, p_ref,
                Mb_series=Mb_full, rho_c_series=rho_c_full,
                times_series=times_full,
                suffix=PLOT_SUFFIX, R_star=R_star
            )
        except Exception as e:
            print(f"  plot_evolution failed: {e}")

        # 3. Plot BSSN evolution
        try:
            utils.plot_bssn_evolution(
                state_cp0, state_cp3, grid,
                t_0=0.0, t_final=checkpoint_times[3],
                suffix=PLOT_SUFFIX
            )
        except Exception as e:
            print(f"  plot_bssn_evolution failed: {e}")

        # 4. First step diagnostic plots (if state was captured)
        if state_after_first_step is not None:
            try:
                print("  - First step comparison (t=0 vs t=dt)...")
                utils.plot_first_step(state_cp0, state_after_first_step, grid, hydro,
                                     tov_solution=tov_solution, suffix=PLOT_SUFFIX)
            except Exception as e:
                print(f"  plot_first_step failed: {e}")

            try:
                print("  - Center zoom (first step)...")
                utils.plot_center_zoom(state_cp0, state_after_first_step, grid, hydro,
                                      window=0.5, suffix=PLOT_SUFFIX)
            except Exception as e:
                print(f"  plot_center_zoom failed: {e}")

            try:
                print("  - Surface zoom (first step)...")
                utils.plot_surface_zoom(tov_solution, state_cp0, state_after_first_step,
                                       grid, hydro, primitives_t0=None, window=0.5, suffix=PLOT_SUFFIX)
            except Exception as e:
                print(f"  plot_surface_zoom failed: {e}")

        # 5. Constraint evolution plot (dynamic mode only)
        if IS_DYNAMIC and ENABLE_DATA_SAVING:
            try:
                utils.plot_constraints_evolution(OUTPUT_DIR, suffix=PLOT_SUFFIX)
            except Exception as e:
                print(f"  plot_constraints_evolution failed: {e}")

    # ==================================================================
    # TIMING SUMMARY
    # ==================================================================
    print(f"\n{'='*70}")
    print("TIMING SUMMARY")
    print(f"{'='*70}")
    print(f"  JIT compilation:        {jit_time:.2f}s")
    print(f"  Benchmark step time:    {step_time_ms:.2f} ms")
    print(f"  Pure evolution:         {evol_time:.2f}s ({num_steps_total} steps)")
    print(f"  Total (incl. diag):    {execution_time:.2f}s")
    print(f"  Steps/second:           {num_steps_total / evol_time:.0f}")
    print(f"  Time per RK4 step:      {evol_time / num_steps_total * 1e3:.3f} ms")
    print(f"  Wall time: {execution_time:.2f}s")
    print(f"{'='*70}")
    print(f"BENCHMARK_RESULT: jit_s={jit_time:.3f} per_step_ms={step_time_ms:.3f} total_s={evol_time:.3f} n_steps={num_steps_total}")

    # ==================================================================
    # PLOTS SUMMARY
    # ==================================================================
    print(f"\n{'='*70}")
    print("DIAGNOSTIC PLOTS GENERATED")
    print(f"{'='*70}")
    print(f"  1. tov_solution{PLOT_SUFFIX}.png              - TOV solution diagnostics")
    print(f"  2. tov_initial_data_zoom{PLOT_SUFFIX}.png     - TOV vs initial data (surface)")
    print(f"  3. tov_evolution{PLOT_SUFFIX}.png             - Hydro evolution at checkpoints")
    print(f"  4. tov_bssn_evolution{PLOT_SUFFIX}.png        - BSSN metric evolution")
    if state_after_first_step is not None:
        print(f"  5. tov_first_step{PLOT_SUFFIX}.png            - First step (t=0 vs t=dt)")
        print(f"  6. tov_center_zoom{PLOT_SUFFIX}.png           - Center zoom (first step)")
        print(f"  7. tov_surface_zoom{PLOT_SUFFIX}.png          - Surface zoom (first step)")
    if IS_DYNAMIC and ENABLE_DATA_SAVING:
        print(f"  8. constraints_evolution{PLOT_SUFFIX}.png  - BSSN constraint violations")
    print(f"{'='*70}")

    print(f"\nData saved to: {OUTPUT_DIR}")
    print(f"  - tov_snapshots{PLOT_SUFFIX}.h5   (full domain snapshots + constraints)")
    print(f"  - tov_evolution{PLOT_SUFFIX}.h5   (time series + constraints)")
    print(f"  - tov_metadata{PLOT_SUFFIX}.json  (simulation parameters)")
    print(f"  - timeseries.npz                (rho_c, v_c, Mb vs time)")


if __name__ == "__main__":
    main()
