import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.integrate import ode, trapezoid, simpson
from scipy.interpolate import interp1d
import h5py
import json
from datetime import datetime

# Create directory for plots
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Engrenage core
sys.path.insert(0, '/home/yo/repositories/engrenage')
from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground, i_r, i_t, i_p

# BSSN
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import (NUM_BSSN_VARS, idx_phi, idx_hrr, idx_htt, idx_hpp,
                                             idx_K, idx_arr, idx_att, idx_app, idx_lapse)
from source.bssn.tensoralgebra import get_bar_gamma_LL
from source.bssn.constraintsdiagnostic import get_constraints_diagnostic

# Hydro
from source.matter.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import PolytropicEOS, IdealGasEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.cons2prim import prim_to_cons
from source.matter.hydro.atmosphere import AtmosphereParams  
from source.bssn.tensoralgebra import get_bar_gamma_LL

from tov_solver import TOVSolver
import tov_initial_data_interpolated as tov_id

# Import utilities (plotting, data management, helper functions)
import utils_TOVEvolution as utils


def diagnose_t0_residuals(state_2d, grid, background, hydro):
    """Compute and print t=0 RHS residuals (especially dS_r/dt) to locate imbalance.

    This helps identify discrete hydrostatic imbalance (typically strongest near the surface).
    Also computes Hamiltonian constraint violation.
    """
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

    # Compute Hamiltonian constraint
    print("\nHamiltonian constraint at t=0:")
    Ham, Mom = compute_hamiltonian_constraint(state_2d, grid, background, hydro)
    max_Ham = np.max(np.abs(Ham[interior]))
    i_max_Ham = NUM_GHOSTS + int(np.argmax(np.abs(Ham[interior])))
    print(f"  max |Ham| = {max_Ham:.3e} at r={grid.r[i_max_Ham]:.6f} (i={i_max_Ham})")

    # L2 norm of Ham (physical measure)
    L2_Ham = np.sqrt(np.mean(Ham[interior]**2))
    print(f"  L2(Ham)   = {L2_Ham:.3e}")

    # Coarse surface estimate from primitives
    prim = hydro._get_primitives(bssn_vars, grid.r)
    mask_interior = prim['rho0'] > 1e-6
    if np.any(mask_interior):
        i_surf = np.where(mask_interior)[0][-1]
        print(f"\n  estimated stellar surface near r={grid.r[i_surf]:.6f} (i={i_surf})")
        window = slice(max(NUM_GHOSTS, i_surf-5), min(grid.N-NUM_GHOSTS, i_surf+6))
        print("  dS_r/dt in 11-pt window around surface:")
        for ii in range(window.start, window.stop):
            print(f"    i={ii:5d}, r={grid.r[ ii ]:8.5f}, dS_r/dt={rhs_Sr[ii]: .3e}")
    else:
        print("  WARNING: could not locate interior points above threshold.")


def get_rhs_bssn(t, y, grid, background, hydro):
    """RHS for full BSSN + Hydro evolution.

    Standard finite volume workflow:
    1. Fill conservative ghost zones (parity + outflow)
    2. Compute primitives via cons2prim
    3. Reconstruct primitives to cell faces
    4. Solve Riemann problem for fluxes
    5. Compute RHS with connection and source terms
    6. Evolve BSSN variables dynamically
    """
    from source.bssn.bssnrhs import get_bssn_rhs

    state = y.reshape((grid.NUM_VARS, grid.N))
    grid.fill_boundaries(state)

    # ATMOSPHERE TREATMENT WITH TRANSITION ZONE
    # Strategy: Instead of abrupt freeze at atmosphere threshold, use smooth transition
    # to avoid artificial "wall" that reflects flux from stellar surface.
    #
    # IMPORTANT: When modifying STATE directly (not RHS), only modify PHYSICAL cells
    # [NUM_GHOSTS:-NUM_GHOSTS] to preserve ghost cell parity conditions set by fill_boundaries.

    # Define physical cells slice
    phys = slice(NUM_GHOSTS, -NUM_GHOSTS)

    # Full density array (used for RHS calculations which cover all cells)
    D_all = state[NUM_BSSN_VARS + 0, :]

    # Define density thresholds
    rho_floor = hydro.atmosphere.rho_floor
    rho_atm = 100.0 * rho_floor  # Pure atmosphere (completely frozen)
    rho_transition = 1000.0 * rho_floor  # Start of transition zone

    # Masks for RHS modifications (use full domain D_all)
    pure_atm_mask = D_all < rho_atm  # Pure atmosphere: freeze completely
    transition_mask = (D_all >= rho_atm) & (D_all < rho_transition)  # Transition: damping

    # In pure atmosphere: force Sr=0 (no momentum) - ONLY in PHYSICAL cells
    # Create physical-only version of mask for state modification
    pure_atm_mask_phys = D_all[phys] < rho_atm
    if np.any(pure_atm_mask_phys):
        state[NUM_BSSN_VARS + 1, phys][pure_atm_mask_phys] = 0.0

    # Build BSSN containers (dynamic)
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state[:NUM_BSSN_VARS, :])
    bssn_d1 = grid.get_d1_metric_quantities(state)

    hydro.set_matter_vars(state, bssn_vars, grid)

    # Get hydro RHS
    hydro_rhs = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1, background)

    # Apply RHS modifications based on density (RHS covers all cells including ghosts)

    # 1. Pure atmosphere: freeze completely (RHS=0)
    if np.any(pure_atm_mask):
        hydro_rhs[0, pure_atm_mask] = 0.0  # dD/dt = 0
        hydro_rhs[1, pure_atm_mask] = 0.0  # dSr/dt = 0
        hydro_rhs[2, pure_atm_mask] = 0.0  # dtau/dt = 0

    # 2. Transition zone: smooth damping
    # Damping factor: α(ρ) = (ρ - rho_atm) / (rho_transition - rho_atm)
    # α=0 at rho_atm (full damping), α=1 at rho_transition (no damping)
    if np.any(transition_mask):
        alpha = (D_all[transition_mask] - rho_atm) / (rho_transition - rho_atm)
        alpha = np.clip(alpha, 0.0, 1.0)  # Ensure [0,1]

        # Apply damping to Sr evolution (damp momentum growth)
        # Keep D and tau evolution to allow mass/energy drainage
        hydro_rhs[1, transition_mask] *= alpha  # Gradual reduction of dSr/dt

    # 3. ADDITIONAL: Velocity damping near surface to suppress spurious oscillations
    # This addresses discrete hydrostatic imbalance that creates growing velocities
    # at the stellar surface. Use exponential damping based on density.
    # Damping rate: Γ(ρ) = Γ_max * exp(-(ρ/ρ_damp)^2) where ρ_damp ~ surface density

    # Only damp where there's actual momentum
    Sr_all = state[NUM_BSSN_VARS + 1, :]
    has_momentum = np.abs(Sr_all) > 1e-30

    # Characteristic density for damping (around surface, ~1e-5 to 1e-6)
    rho_surface_characteristic = 1e-5

    # Damping region: low density but above atmosphere (ρ > rho_transition)
    damp_region_mask = (D_all > rho_transition) & (D_all < 10.0 * rho_surface_characteristic) & has_momentum

    if np.any(damp_region_mask):
        # Exponential damping coefficient
        rho_damp = D_all[damp_region_mask]
        gamma_damp = 0.01 * np.exp(-(rho_damp / rho_surface_characteristic)**2)

        # Apply damping: dSr/dt -= Γ * Sr (exponential decay)
        hydro_rhs[1, damp_region_mask] -= gamma_damp * Sr_all[damp_region_mask]

    # Get BSSN RHS (now evolves dynamically)
    # First get second derivatives
    bssn_d2 = grid.get_d2_metric_quantities(state)

    # Get energy-momentum tensor from matter
    emtensor = hydro.get_emtensor(grid.r, bssn_vars, background)

    # Create container for BSSN RHS (modified in place by get_bssn_rhs)
    bssn_rhs = BSSNVars(grid.N)

    # Call get_bssn_rhs with correct signature
    get_bssn_rhs(bssn_rhs, grid.r, bssn_vars, bssn_d1, bssn_d2, background, emtensor)

    # Extract BSSN RHS values into array form
    bssn_rhs_array = bssn_rhs.set_bssn_state_vars()

    # Full RHS (BSSN + hydro both evolve)
    rhs = np.zeros_like(state)
    rhs[:NUM_BSSN_VARS, :] = bssn_rhs_array
    rhs[NUM_BSSN_VARS:, :] = hydro_rhs
    return rhs.flatten()




def compute_hamiltonian_constraint(state_2d, grid, background, hydro):
    """
    Compute Hamiltonian constraint for a single state.

    Args:
        state_2d: State array (NUM_VARS, N)
        grid: Grid object
        background: Background object
        hydro: Hydro object

    Returns:
        Ham: Hamiltonian constraint array (N,)
    """
    # Use the get_constraints_diagnostic function with a single state
    Ham, Mom = get_constraints_diagnostic(state_2d.flatten(), 0.0, grid, background, hydro)
    return Ham[0, :], Mom[0, :, :]  # Return first (and only) time slice


def compute_update_mask(state_old, state_new, atmosphere, bssn_vars):
    """
    Compute update mask following GRHydro strategy.

    Determines which cells can be safely updated without violating physical constraints.
    Cells where the update would push D or tau below atmosphere are marked as False.

    This prevents spurious oscillations from repeated floor application
    (see GRHydro documentation section 8.4).

    Args:
        state_old: State before update (NUM_VARS, N)
        state_new: State after RK4 update (NUM_VARS, N)
        atmosphere: AtmosphereParams object
        bssn_vars: BSSNVars object (needed for e6phi = exp(6φ))

    Returns:
        mask: Boolean array (N,) - True = safe to update, False = would violate constraints
    """
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS

    # Extract DENSITIZED conservatives from state
    D_tilde_old = state_old[NUM_BSSN_VARS + 0, :]
    D_tilde_new = state_new[NUM_BSSN_VARS + 0, :]
    tau_tilde_old = state_old[NUM_BSSN_VARS + 2, :]
    tau_tilde_new = state_new[NUM_BSSN_VARS + 2, :]

    # Compute e6phi for de-densitization: e^{6φ}
    phi = np.asarray(bssn_vars.phi, dtype=float)
    e6phi = np.exp(6.0 * phi)

    # DE-DENSITIZE to get physical variables for comparison with floors
    # D = D̃ / e^{6φ}, τ = τ̃ / e^{6φ}
    inv_e6phi = 1.0 / np.maximum(e6phi, 1e-30)
    D_old = D_tilde_old * inv_e6phi
    D_new = D_tilde_new * inv_e6phi
    tau_old = tau_tilde_old * inv_e6phi
    tau_new = tau_tilde_new * inv_e6phi

    # Initialize mask (all True by default)
    mask = np.ones(len(D_old), dtype=bool)

    # Only check physical cells (ghost cells handled by fill_boundaries)
    phys = slice(NUM_GHOSTS, len(D_old) - NUM_GHOSTS)

    # Cell is safe to update if:
    # 1. D stays above floor (with 10% tolerance to avoid edge cases)
    # 2. tau stays above floor (with 10% tolerance)
    # Using tolerance prevents repeated triggering on cells near floor
    # NOW comparing PHYSICAL (non-densitized) variables with floors
    tolerance = 0.9

    mask[phys] = (
        (D_new[phys] >= atmosphere.rho_floor * tolerance) &
        (tau_new[phys] >= atmosphere.tau_atm * tolerance)
    )

    return mask


def reset_to_atmosphere(state_2d, failed_mask, atmosphere, eos, bssn_vars):
    """
    Reset failed cells to atmosphere values following GRHydro strategy.

    For cells marked as failed (where update would violate constraints),
    set all hydro variables to atmosphere values:
    - D̃ = e^{6φ} * rho_floor (DENSITIZED)
    - S̃r = 0 (no velocity, already densitized)
    - τ̃ = e^{6φ} * (rho_floor * eps_atm) (DENSITIZED)

    Args:
        state_2d: State array (NUM_VARS, N) - modified in place
        failed_mask: Boolean array (N,) - True for cells to reset
        atmosphere: AtmosphereParams object
        eos: Equation of state (for computing eps_atm)
        bssn_vars: BSSNVars object (needed for e6phi = exp(6φ))
    """
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS

    # Only operate on physical cells
    phys = slice(NUM_GHOSTS, state_2d.shape[1] - NUM_GHOSTS)
    failed_phys = failed_mask[phys]

    if not np.any(failed_phys):
        return  # No cells to reset

    # Compute e6phi for densitization: e^{6φ}
    phi = np.asarray(bssn_vars.phi, dtype=float)
    e6phi = np.exp(6.0 * phi)

    # Compute atmosphere epsilon (small value)
    eps_atm = eos.eps_from_rho_p(atmosphere.rho_floor, atmosphere.p_floor) if hasattr(eos, 'eps_from_rho_p') else 1e-10

    # DENSITIZE atmosphere values before writing to state
    # State stores D̃, S̃r, τ̃ (densitized conservatives)
    D_tilde_atm = e6phi[phys][failed_phys] * atmosphere.rho_floor
    Sr_tilde_atm = 0.0  # No momentum (already zero in both representations)
    tau_tilde_atm = e6phi[phys][failed_phys] * (atmosphere.rho_floor * eps_atm)

    # Set DENSITIZED atmosphere values (only in physical cells that failed)
    state_2d[NUM_BSSN_VARS + 0, phys][failed_phys] = D_tilde_atm  # D̃
    state_2d[NUM_BSSN_VARS + 1, phys][failed_phys] = Sr_tilde_atm  # S̃r
    state_2d[NUM_BSSN_VARS + 2, phys][failed_phys] = tau_tilde_atm  # τ̃


def rk4_step(state_flat, dt, grid, background, hydro, atmosphere):
    """
    Single RK4 timestep with GRHydro-style atmosphere treatment.

    Following GRHydro philosophy (section 8.4):
    - Cells that would violate constraints are NOT updated
    - Failed cells are reset to atmosphere values
    - Floors are applied only at the end (not after each substep)
    - This reduces spurious oscillations from repeated floor application

    Args:
        atmosphere: AtmosphereParams object
    """
    from source.matter.hydro.atmosphere import apply_floors_to_state

    # Get gamma from the EOS to ensure consistency
    gamma = hydro.eos.gamma

    # Store initial state for update mask
    state_old = state_flat.reshape((grid.NUM_VARS, grid.N))

    # Stage 1
    k1 = get_rhs_bssn(0, state_flat, grid, background, hydro)

    # Stage 2 (NO floor - cons2prim applies internal floors)
    state_2 = state_flat + 0.5 * dt * k1
    k2 = get_rhs_bssn(0, state_2, grid, background, hydro)

    # Stage 3 (NO floor - cons2prim applies internal floors)
    state_3 = state_flat + 0.5 * dt * k2
    k3 = get_rhs_bssn(0, state_3, grid, background, hydro)

    # Stage 4 (NO floor - cons2prim applies internal floors)
    state_4 = state_flat + dt * k3
    k4 = get_rhs_bssn(0, state_4, grid, background, hydro)

    # Combine
    state_new = state_flat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    snew = state_new.reshape((grid.NUM_VARS, grid.N))

    # Construct BSSN vars from state_old for accessing φ (needed for densitization)
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_old[:NUM_BSSN_VARS, :])

    # GRHydro-style update mask: Don't update cells that would violate constraints
    update_mask = compute_update_mask(state_old, snew, atmosphere, bssn_vars)

    # For cells that would violate constraints, keep old values
    for var_idx in range(grid.NUM_VARS):
        snew[var_idx, ~update_mask] = state_old[var_idx, ~update_mask]

    # Reset failed cells to atmosphere
    reset_to_atmosphere(snew, ~update_mask, atmosphere, hydro.eos, bssn_vars)

    # CRITICAL: Restore ghost cells with parity after RK4 integration
    # The RHS includes ghost cells, so after integration they evolve instead of staying static.
    # This call restores proper parity conditions at inner boundary (r=0).
    grid.fill_boundaries(snew)

    # Apply atmosphere floors after full step
    # This handles cells that passed the update mask but still need floor correction
    # (e.g., velocity limits, pressure floors)
    apply_floors_to_state(snew, grid, hydro)

    return snew.flatten()


def _compute_baryon_mass(grid, state, primitives):
    """Compute baryon (rest) mass M = ∫ ρ0 W √γ d^3x = 4π ∫ ρ0 W ψ^6 r^2 dr.

    Uses interior points (excludes ghost zones)."""
    interior = slice(NUM_GHOSTS, grid.N - NUM_GHOSTS)
    r = grid.r[interior]
    rho0 = primitives['rho0'][interior]
    # Prefer W from primitives (accounts for metric), fallback to 1/sqrt(1-v^2)
    if 'W' in primitives and primitives['W'] is not None:
        W = primitives['W'][interior]
    else:
        v = primitives['vr'][interior]
        W = 1.0 / np.sqrt(np.maximum(1.0 - v*v, 1e-14))

    phi = state[idx_phi, interior]
    psi = np.exp(phi)
    integrand = rho0 * W * (psi**6) * (r**2)
    return 4.0 * np.pi * simpson(integrand, x=r)


def evolve_fixed_timestep(state_initial, dt, num_steps, grid, background, hydro,
                          atmosphere, method='rk4', t_start=0.0,
                          reference_state=None, step_offset=0, data_manager=None,
                          snapshot_interval=None, evolution_interval=None):
    """Evolve with fixed timestep using RK4.

    Args:
        atmosphere: AtmosphereParams object
        t_start: Starting time for this evolution segment (default: 0.0)
        reference_state: Reference state for error calculation (default: state_initial)
                        Use this to maintain consistent error measurement across multiple segments
        step_offset: Offset for step numbering in output (default: 0)
        data_manager: utils.SimulationDataManager object for data saving (optional)
        snapshot_interval: Save full domain snapshot every N steps (optional)
        evolution_interval: Save evolution data every N steps (optional)
    """
    state_flat = state_initial.flatten()

    def primitives_from_state(state_flattened):
        s2d = state_flattened.reshape((grid.NUM_VARS, grid.N))
        bssn_vars = BSSNVars(grid.N)
        bssn_vars.set_bssn_vars(s2d[:NUM_BSSN_VARS, :])
        hydro.set_matter_vars(s2d, bssn_vars, grid)
        return hydro._get_primitives(bssn_vars, grid.r), s2d

    # Diagnostics at start
    prim_prev, s_prev = primitives_from_state(state_flat)

    # Store reference state for error calculation (use provided reference or current initial)
    if reference_state is None:
        reference_state = state_initial
    prim_initial, s_initial = primitives_from_state(reference_state.flatten())

    # Save initial snapshot if data manager provided
    if data_manager and data_manager.enable_saving:
        # Compute Hamiltonian constraint for initial state
        Ham_initial, Mom_initial = compute_hamiltonian_constraint(state_initial, grid, background, hydro)
        data_manager.save_snapshot(step_offset, t_start, state_initial, prim_initial, Ham=Ham_initial, Mom=Mom_initial)
        data_manager.add_evolution_point(step_offset, t_start, state_initial, prim_initial, prim_initial, Ham=Ham_initial)

    # Timeseries for mass and central density
    times_series = [t_start]
    Mb0 = _compute_baryon_mass(grid, s_initial, prim_initial)
    Mb_series = [Mb0]
    rho_c0 = prim_initial['rho0'][NUM_GHOSTS]
    rho_c_series = [rho_c0]

    print("\n===== Evolution diagnostics (per step) =====")
    print("Columns: step | t | ρ_central | max_Δρ/ρ | max_vʳ | max_D | max_Sʳ | max_τ | c2p_fails")
    print("-" * 100)

    for step in range(num_steps):
        # Advance one RK4 step
        state_flat_next = rk4_step(state_flat, dt, grid, background, hydro, atmosphere)

        # Compute primitives BEFORE and AFTER to measure change
        prim_next, s_next = primitives_from_state(state_flat_next)

        # Interior slice (exclude ghosts)
        interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

        rho_prev = prim_prev['rho0'][interior]
        rho_next = prim_next['rho0'][interior]
        rho_init = prim_initial['rho0'][interior]
        p_prev = prim_prev['p'][interior]
        p_next = prim_next['p'][interior]
        v_prev = prim_prev['vr'][interior]
        v_next = prim_next['vr'][interior]

        D_prev = s_prev[NUM_BSSN_VARS + 0, interior]
        Sr_prev = s_prev[NUM_BSSN_VARS + 1, interior]
        tau_prev = s_prev[NUM_BSSN_VARS + 2, interior]
        D_next = s_next[NUM_BSSN_VARS + 0, interior]
        Sr_next = s_next[NUM_BSSN_VARS + 1, interior]
        tau_next = s_next[NUM_BSSN_VARS + 2, interior]

        # Compute more informative stats
        rho_central = float(prim_next['rho0'][NUM_GHOSTS])  # Central density

        # Maximum relative density error vs initial state
        rel_rho_err = np.abs(rho_next - rho_init) / (np.abs(rho_init) + 1e-20)
        max_rel_rho_err = float(np.max(rel_rho_err))
        idx_max_rel_rho = NUM_GHOSTS + int(np.argmax(rel_rho_err))
        r_max_rel_rho = grid.r[idx_max_rel_rho]

        # Maximum velocity
        max_abs_v = float(np.max(np.abs(v_next)))
        idx_max_v = NUM_GHOSTS + int(np.argmax(np.abs(v_next)))
        r_max_v = grid.r[idx_max_v]

        # Maximum conserved variables (more useful than minimum)
        max_D = float(np.max(D_next))
        idx_max_D = NUM_GHOSTS + int(np.argmax(D_next))
        r_max_D = grid.r[idx_max_D]

        max_Sr = float(np.max(np.abs(Sr_next)))
        idx_max_Sr = NUM_GHOSTS + int(np.argmax(np.abs(Sr_next)))
        r_max_Sr = grid.r[idx_max_Sr]

        max_tau = float(np.max(np.abs(tau_next)))
        idx_max_tau = NUM_GHOSTS + int(np.argmax(np.abs(tau_next)))
        r_max_tau = grid.r[idx_max_tau]

        # Cons2prim failures
        c2p_fail_count = int(np.sum(~prim_next['success']))

        t_curr = t_start + (step + 1) * dt
        step_num = step_offset + step + 1
        if step_num % 200 == 0:
            print(f"step {step_num:4d}  t={t_curr:.6e}:  ρ_c={rho_central:.6e}  max_Δρ/ρ={max_rel_rho_err:.3e} (r={r_max_rel_rho:.2f})  "
              f"max_vʳ={max_abs_v:.3e} (r={r_max_v:.2f})  max_D={max_D:.3e} (r={r_max_D:.2f})  max_Sʳ={max_Sr:.3e} (r={r_max_Sr:.2f})  "
              f"max_τ={max_tau:.3e} (r={r_max_tau:.2f})  c2p_fail={c2p_fail_count}")

        # Save data if requested
        if data_manager and data_manager.enable_saving:
            # Compute Ham if needed for saving
            Ham_next = None
            Mom_next = None
            if (evolution_interval and step_num % evolution_interval == 0) or \
               (snapshot_interval and step_num % snapshot_interval == 0):
                Ham_next, Mom_next = compute_hamiltonian_constraint(s_next, grid, background, hydro)

            # Save evolution data at specified interval
            if evolution_interval and step_num % evolution_interval == 0:
                data_manager.add_evolution_point(step_num, t_curr, s_next, prim_next, prim_initial, Ham=Ham_next)

                # Periodic buffer flush
                if step_num % (evolution_interval * 10) == 0:
                    data_manager.flush_evolution_buffer()

            # Save full snapshot at specified interval
            if snapshot_interval and step_num % snapshot_interval == 0:
                data_manager.save_snapshot(step_num, t_curr, s_next, prim_next, Ham=Ham_next, Mom=Mom_next)

        # Append to time series (mass and central density)
        Mb_next = _compute_baryon_mass(grid, s_next, prim_next)
        times_series.append(t_curr)
        Mb_series.append(Mb_next)
        rho_c_series.append(float(prim_next['rho0'][NUM_GHOSTS]))

        # Detect first signs of instability / non-physical values
        issues = []
        if not np.all(np.isfinite(rho_next)) or not np.all(np.isfinite(p_next)):
            issues.append("NaN/Inf in primitives")
        if np.any(rho_next < 0):
            issues.append("negative rho0")
        if np.any(p_next < 0):
            issues.append("negative pressure")
        if np.any(np.abs(v_next) >= 1.0):
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
                idx_v = NUM_GHOSTS + int(np.argmax(np.abs(prim_next['vr'][interior])))
            except Exception:
                idx_v = NUM_GHOSTS
            try:
                idx_rho_min = NUM_GHOSTS + int(np.argmin(prim_next['rho0'][interior]))
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
                      f"rho0={prim_next['rho0'][ii]:.6e}, P={prim_next['p'][ii]:.6e}, vr={prim_next['vr'][ii]:.6e}, "
                      f"D={s_next[NUM_BSSN_VARS+0, ii]:.6e}, Sr={s_next[NUM_BSSN_VARS+1, ii]:.6e}, tau={s_next[NUM_BSSN_VARS+2, ii]:.6e}")

            # Also report relative changes at those points (prev -> next)
            def rel(a, b):
                return (b - a) / (np.abs(a) + 1e-30)
            for ii in idxs:
                print(f"       Δrel:  Δrho0={rel(prim_prev['rho0'][ii], prim_next['rho0'][ii]):.3e}, "
                      f"ΔP={rel(prim_prev['p'][ii], prim_next['p'][ii]):.3e}, "
                      f"Δvr={prim_next['vr'][ii]-prim_prev['vr'][ii]:.3e}, "
                      f"ΔD={rel(s_prev[NUM_BSSN_VARS+0, ii], s_next[NUM_BSSN_VARS+0, ii]):.3e}, "
                      f"ΔSr={s_next[NUM_BSSN_VARS+1, ii]-s_prev[NUM_BSSN_VARS+1, ii]:.3e}, "
                      f"Δtau={rel(s_prev[NUM_BSSN_VARS+2, ii], s_next[NUM_BSSN_VARS+2, ii]):.3e}")

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
        prim_prev, s_prev = prim_next, s_next


    # Final flush of buffers if data manager is provided
    if data_manager and data_manager.enable_saving:
        data_manager.flush_evolution_buffer()

    actual_time = t_start + num_steps * dt
    return state_flat.reshape((grid.NUM_VARS, grid.N)), num_steps, actual_time, {
        't': np.array(times_series),
        'Mb': np.array(Mb_series),
        'rho_c': np.array(rho_c_series),
    }


def evolve_adaptive(state_initial, t_final, grid, background, hydro,
                   method='RK45', rtol=1e-6, atol=1e-8):
    """Evolve with adaptive timestep using scipy.integrate.solve_ivp."""
    from scipy.integrate import solve_ivp

    # Wrapper for RHS compatible with solve_ivp
    def rhs_wrapper(t, y):
        return get_rhs_bssn(t, y, grid, background, hydro)

    state_flat = state_initial.flatten()

    print(f"  Using solve_ivp with method={method}, rtol={rtol}, atol={atol}")

    solution = solve_ivp(
        rhs_wrapper,
        t_span=(0, t_final),
        y0=state_flat,
        method=method,  # 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
        rtol=rtol,
        atol=atol,
        dense_output=False
    )

    print(f"  solve_ivp: {solution.nfev} function evaluations, status={solution.status}")

    return solution.y[:, -1].reshape((grid.NUM_VARS, grid.N))


def main():
    """Main execution."""
    print("="*70)
    print("TOV Star Evolution - Full BSSN + Hydro")
    print("="*70)

    # ==================================================================
    # CONFIGURATION
    # ==================================================================
    r_max = 16.0
    num_points = 500
    K = 100.0
    Gamma = 2.0
    rho_central = 1.28e-3  # Central density
    num_steps_total = 10000  # Quick test run


    # ==================================================================
    # DATA SAVING CONFIGURATION
    # ==================================================================
    ENABLE_DATA_SAVING = True  # Set to True to save data to files
    OUTPUT_DIR = os.path.join(plots_dir, "tov_evolution_data_bssn")  # Directory for output data
    SNAPSHOT_INTERVAL = 10  # Save full domain every N timesteps (None to disable)
    EVOLUTION_INTERVAL = 10  # Save time series every N timesteps (None to disable)

    if ENABLE_DATA_SAVING:
        print(f"Data saving enabled:")
        print(f"  Output directory: {OUTPUT_DIR}")
        print(f"  Snapshot interval: {SNAPSHOT_INTERVAL} timesteps")
        print(f"  Evolution tracking: {EVOLUTION_INTERVAL} timesteps")
        print()

    # ==================================================================
    # ATMOSPHERE CONFIGURATION (Following   approach)
    # ==================================================================
    # Atmosphere floors ( /  -style): choose small but nonzero floors
    # Central density is our maximum density: rho_central = 1.28e-3
    rho_max = rho_central
    rho_floor_value = 1.0e-10#1.0e-8 * rho_max  # Following  : 1e-10 * rho_max

    ATMOSPHERE = AtmosphereParams(
        rho_floor=rho_floor_value,
        p_floor=1.0e-10,
        v_max=0.9999,
        W_max=100.0,
        tau_atm_factor=1.0,
        conservative_floor_safety=0.999999,
        mb=1.0
        )

    print("=" * 70)
    print("ATMOSPHERE CONFIGURATION")
    print("=" * 70)
    print(f"  rho_max       = {rho_max:.2e} (central density)")
    print(f"  rho_floor     = {ATMOSPHERE.rho_floor:.2e} (1e-10 * rho_max)")
    print(f"  p_floor       = {ATMOSPHERE.p_floor:.2e} (polytropic EOS)")
    print(f"  tau_atm       = {ATMOSPHERE.tau_atm:.2e}")
    print(f"  v_max         = {ATMOSPHERE.v_max}")
    print()

    # Time integration method
    # 'fixed': RK4 with fixed timestep 
    # 'adaptive': solve_ivp with adaptive timestep (slower, more accurate)
    integration_method = 'fixed'  # 'fixed' or 'adaptive'

    # ==================================================================
    # SETUP
    # ==================================================================
    spacing = LinearSpacing(num_points, r_max)
    # Evolve with ideal-gas EOS using the same Gamma as the TOV polytrope
    eos = IdealGasEOS(gamma=Gamma)
    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere=ATMOSPHERE,
        reconstructor=create_reconstruction("mp5"),
        riemann_solver=HLLRiemannSolver(atmosphere=ATMOSPHERE)
    )

    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    print(f"Grid: N={grid.N}, r_max={r_max}, dr_min={grid.min_dr}")
    print(f"EOS: K={K}, Gamma={Gamma}\n")

    # ==================================================================
    # SOLVE TOV DIRECTLY ON EVOLUTION GRID (for discrete equilibrium)
    # ==================================================================
    print("Solving TOV equations (Schwarzschild coordinates)...")
    tov_solver = TOVSolver(K=K, Gamma=Gamma)
    tov_solution = tov_solver.solve(rho_central, r_max=r_max)
    print(f"TOV Solution: M={tov_solution['M_star']:.6f}, R={tov_solution['R']:.3f}, C={tov_solution['C']:.4f}\n")
    utils.plot_tov_diagnostics(tov_solution, r_max)

    # ==================================================================
    # INITIAL DATA (HIGH-ORDER INTERPOLATION)
    # ==================================================================
    print("Creating initial data from TOV solution...")
    initial_state_2d = tov_id.create_initial_data_interpolated(
        tov_solution, grid, background, eos,
        atmosphere=ATMOSPHERE,
        interp_order=11
    )

    # Diagnostics: check discrete hydrostatic balance at t=0
    diagnose_t0_residuals(initial_state_2d, grid, background, hydro)

    # Initial-data diagnostics
    tov_id.plot_initial_comparison(tov_solution, initial_state_2d, grid, hydro)

    # Zoom comparison: TOV solution vs interpolated initial data
    utils.plot_tov_vs_initial_data_zoom(tov_solution, initial_state_2d, grid, hydro, window=0.1)

    # ==================================================================
    # EVOLUTION
    # ==================================================================
    # Initialize data manager for saving
    data_manager = utils.SimulationDataManager(OUTPUT_DIR, grid, hydro, enable_saving=ENABLE_DATA_SAVING)


    if integration_method == 'fixed':
        dt = 0.2 * grid.min_dr  # CFL condition
        print(f"\nEvolving with fixed dt={dt:.6f} (CFL=0.1) for {num_steps_total} steps using RK4")

        # Save metadata now that we have dt
        if ENABLE_DATA_SAVING:
            data_manager.save_metadata(tov_solution, ATMOSPHERE, dt, integration_method, K=K, Gamma=Gamma, rho_central=rho_central)

        # Plot only the first step changes (for diagnostics)
        state_t1_diag = rk4_step(initial_state_2d.flatten(), dt, grid, background, hydro,
                           ATMOSPHERE).reshape((grid.NUM_VARS, grid.N))
        utils.plot_first_step(initial_state_2d, state_t1_diag, grid, hydro, tov_solution)
        utils.plot_surface_zoom(tov_solution, initial_state_2d, state_t1_diag, grid, hydro, window=0.1)

        # Define checkpoints at 1/3, 2/3, and 1 of total steps
        checkpoint_1 = num_steps_total // 3
        checkpoint_2 = 2 * num_steps_total // 3
        checkpoint_3 = num_steps_total

        print(f"\nEvolution checkpoints:")
        print(f"  t=0:         initial state")
        print(f"  step {checkpoint_1}:    1/3 of evolution")
        print(f"  step {checkpoint_2}:   2/3 of evolution")
        print(f"  step {checkpoint_3}: final state")

        # Evolve to checkpoint 1 (1/3 of total)
        print(f"\nEvolving to step {checkpoint_1} (1/3)...")
        state_t1, steps_1, t_1, series_1 = evolve_fixed_timestep(
            initial_state_2d, dt, checkpoint_1, grid, background,
            hydro, ATMOSPHERE, method='rk4',
            reference_state=initial_state_2d,
            data_manager=data_manager,
            snapshot_interval=SNAPSHOT_INTERVAL,
            evolution_interval=EVOLUTION_INTERVAL)
        print(f"  -> Reached step {steps_1}, t={t_1:.6e}")

        # Continue evolution to checkpoint 2 (2/3 of total)
        if num_steps_total > checkpoint_1 and steps_1 == checkpoint_1:
            remaining_steps_2 = checkpoint_2 - checkpoint_1
            print(f"\nContinuing evolution from step {checkpoint_1} to {checkpoint_2} (2/3)...")
            state_t2, steps_more_2, t_2, series_2 = evolve_fixed_timestep(
                state_t1, dt, remaining_steps_2, grid, background,
                hydro, ATMOSPHERE, method='rk4',
                t_start=t_1, reference_state=initial_state_2d,
                step_offset=checkpoint_1,
                data_manager=data_manager,
                snapshot_interval=SNAPSHOT_INTERVAL,
                evolution_interval=EVOLUTION_INTERVAL)
            steps_2 = checkpoint_1 + steps_more_2
            print(f"  -> Reached step {steps_2}, t={t_2:.6e}")
        else:
            # Evolution stopped early
            state_t2 = state_t1
            t_2 = t_1
            steps_2 = steps_1
            series_2 = None

        # Continue evolution to checkpoint 3 (final)
        if num_steps_total > checkpoint_2 and steps_2 == checkpoint_2:
            remaining_steps_3 = checkpoint_3 - checkpoint_2
            print(f"\nContinuing evolution from step {checkpoint_2} to {checkpoint_3} (final)...")
            state_tfinal, steps_more_3, t_final, series_3 = evolve_fixed_timestep(
                state_t2, dt, remaining_steps_3, grid, background,
                hydro, ATMOSPHERE, method='rk4',
                t_start=t_2, reference_state=initial_state_2d,
                step_offset=checkpoint_2,
                data_manager=data_manager,
                snapshot_interval=SNAPSHOT_INTERVAL,
                evolution_interval=EVOLUTION_INTERVAL)
            steps_final = checkpoint_2 + steps_more_3
            print(f"  -> Reached step {steps_final}, t={t_final:.6e}")
        else:
            # Evolution stopped early
            state_tfinal = state_t2
            t_final = t_2
            steps_final = steps_2
            series_3 = None

        # For plotting: use all four checkpoints (t=0, t=1/3, t=2/3, t=final)
        # Keep state_t1 as the 1/3 checkpoint
        # state_t2 is the 2/3 checkpoint
        # state_tfinal is the final state
        state_t100 = state_t2  # Renamed for compatibility with existing plot calls
        t_100 = t_2
        state_t10000 = state_tfinal
        t_10000 = t_final
        num_steps = steps_final

        # Build full-series arrays for mass and central density
        try:
            series_list = [series_1]
            if series_2 is not None:
                series_list.append(series_2)
            if series_3 is not None:
                series_list.append(series_3)

            if len(series_list) == 1:
                times_full = series_1['t']
                Mb_full = series_1['Mb']
                rho_c_full = series_1['rho_c']
            else:
                times_full = series_list[0]['t']
                Mb_full = series_list[0]['Mb']
                rho_c_full = series_list[0]['rho_c']
                for series in series_list[1:]:
                    times_full = np.concatenate([times_full, series['t'][1:]])
                    Mb_full = np.concatenate([Mb_full, series['Mb'][1:]])
                    rho_c_full = np.concatenate([rho_c_full, series['rho_c'][1:]])
        except Exception:
            times_full = np.array([])
            Mb_full = np.array([])
            rho_c_full = np.array([])

    elif integration_method == 'adaptive':
        # NOTE: Adaptive methods are slower but can be more accurate
        # Available methods: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
        t_10000 = 10.00  # Final time
        t_1 = t_10000 / 3.0  # Time for 1/3 snapshot
        t_100 = 2.0 * t_10000 / 3.0  # Time for 2/3 snapshot

        print(f"\nEvolving to t={t_10000} using solve_ivp (adaptive)")
        print(f"Evolution checkpoints:")
        print(f"  t=0:     initial state")
        print(f"  t={t_1:.6e}:  1/3 of evolution")
        print(f"  t={t_100:.6e}:  2/3 of evolution")
        print(f"  t={t_10000:.6e}: final state")

        # Define an effective number of steps for labeling/plots
        dt = 0.5 * grid.min_dr
        num_steps = max(1, int(round(t_10000 / dt)))

        # Adaptive evolution to t=1/3
        state_t1 = evolve_adaptive(initial_state_2d, t_1, grid, background, hydro,
                                    method='RK45', rtol=1e-5, atol=1e-7)

        # Adaptive evolution to t=2/3
        state_t100 = evolve_adaptive(initial_state_2d, t_100, grid, background, hydro,
                                    method='RK45', rtol=1e-5, atol=1e-7)

        # Adaptive evolution to t=final
        state_t10000 = evolve_adaptive(initial_state_2d, t_10000, grid, background, hydro,
                                      method='RK45', rtol=1e-5, atol=1e-7)

        # No time series data for adaptive method
        times_full = np.array([])
        Mb_full = np.array([])
        rho_c_full = np.array([])

    # ==================================================================
    # DIAGNOSTICS
    # ==================================================================
    # Determine labels for plotting based on actual evolution
    if integration_method == 'fixed':
        # For fixed timestep: use step numbers for clarity
        if t_100 == t_10000:
            # Same state (evolution stopped early or only ran to checkpoint)
            label_100 = f't=2/3 (step={steps_final})'
            label_10000 = f't=final (step={steps_final})'
        else:
            label_100 = f't=2/3 (step={steps_2})'
            label_10000 = f't=final (step={steps_final})'
    else:
        # For adaptive timestep: use time labels
        label_100 = 't=2/3'
        label_10000 = 't=final'

    # Combined 3x2 figure: keep 4 panels and replace bottom row with the two new time-series
    try:
        if 'times_full' in locals() and len(times_full) > 0:
            utils.plot_evolution(initial_state_2d, state_t1, state_t100, state_t10000, grid, hydro,
                           t_1, t_100, t_10000, label_100=label_100, label_10000=label_10000,
                           times_series=times_full, Mb_series=Mb_full, rho_c_series=rho_c_full)
        else:
            utils.plot_evolution(initial_state_2d, state_t1, state_t100, state_t10000, grid, hydro,
                           t_1, t_100, t_10000, label_100=label_100, label_10000=label_10000)
    except Exception:
        utils.plot_evolution(initial_state_2d, state_t1, state_t100, state_t10000, grid, hydro,
                       t_1, t_100, t_10000, label_100=label_100, label_10000=label_10000)

    # Plot BSSN variables evolution to verify BSSN dynamics
    utils.plot_bssn_evolution(initial_state_2d, state_t10000, grid, t_0=0.0, t_final=t_10000)

    # Plot Hamiltonian constraint evolution
    if ENABLE_DATA_SAVING and data_manager:
        try:
            utils.plot_hamiltonian_constraint_evolution(
                data_manager.snapshot_file,
                data_manager.evolution_file,
                grid,
                output_dir=plots_dir
            )
        except Exception as e:
            print(f"Warning: Could not generate Hamiltonian constraint plot: {e}")

    # Print detailed statistics
    bssn_0 = BSSNVars(grid.N)
    bssn_0.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(initial_state_2d, bssn_0, grid)
    prim_0 = hydro._get_primitives(bssn_0, grid.r)

    bssn_1 = BSSNVars(grid.N)
    bssn_1.set_bssn_vars(state_t1[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t1, bssn_1, grid)
    prim_1 = hydro._get_primitives(bssn_1, grid.r)

    bssn_100 = BSSNVars(grid.N)
    bssn_100.set_bssn_vars(state_t100[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t100, bssn_100, grid)
    prim_100 = hydro._get_primitives(bssn_100, grid.r)

    bssn_10000 = BSSNVars(grid.N)
    bssn_10000.set_bssn_vars(state_t10000[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t10000, bssn_10000, grid)
    prim_10000 = hydro._get_primitives(bssn_10000, grid.r)

    # Calculate actual final time (already stored in t_10000)
    t_final_actual = t_10000

    # Interior points only (exclude ghosts)
    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

    # Compute errors
    delta_rho_1 = np.abs(prim_1['rho0'][interior] - prim_0['rho0'][interior]) / (np.abs(prim_0['rho0'][interior]) + 1e-20)
    delta_rho_100 = np.abs(prim_100['rho0'][interior] - prim_0['rho0'][interior]) / (np.abs(prim_0['rho0'][interior]) + 1e-20)
    delta_rho_10000 = np.abs(prim_10000['rho0'][interior] - prim_0['rho0'][interior]) / (np.abs(prim_0['rho0'][interior]) + 1e-20)

    delta_P_1 = np.abs(prim_1['p'][interior] - prim_0['p'][interior]) / (np.abs(prim_0['p'][interior]) + 1e-20)
    delta_P_100 = np.abs(prim_100['p'][interior] - prim_0['p'][interior]) / (np.abs(prim_0['p'][interior]) + 1e-20)
    delta_P_10000 = np.abs(prim_10000['p'][interior] - prim_0['p'][interior]) / (np.abs(prim_0['p'][interior]) + 1e-20)

    # Error growth factor
    max_err_rho_1 = np.max(delta_rho_1)
    idx_max_err_rho_1 = NUM_GHOSTS + int(np.argmax(delta_rho_1))
    r_max_err_rho_1 = grid.r[idx_max_err_rho_1]

    max_err_rho_100 = np.max(delta_rho_100)
    idx_max_err_rho_100 = NUM_GHOSTS + int(np.argmax(delta_rho_100))
    r_max_err_rho_100 = grid.r[idx_max_err_rho_100]

    max_err_rho_10000 = np.max(delta_rho_10000)
    idx_max_err_rho_10000 = NUM_GHOSTS + int(np.argmax(delta_rho_10000))
    r_max_err_rho_10000 = grid.r[idx_max_err_rho_10000]

    growth_rho = max_err_rho_10000 / max_err_rho_1 if max_err_rho_1 > 1e-15 else 0

    max_err_P_1 = np.max(delta_P_1)
    idx_max_err_P_1 = NUM_GHOSTS + int(np.argmax(delta_P_1))
    r_max_err_P_1 = grid.r[idx_max_err_P_1]

    max_err_P_100 = np.max(delta_P_100)
    idx_max_err_P_100 = NUM_GHOSTS + int(np.argmax(delta_P_100))
    r_max_err_P_100 = grid.r[idx_max_err_P_100]

    max_err_P_10000 = np.max(delta_P_10000)
    idx_max_err_P_10000 = NUM_GHOSTS + int(np.argmax(delta_P_10000))
    r_max_err_P_10000 = grid.r[idx_max_err_P_10000]

    growth_P = max_err_P_10000 / max_err_P_1 if max_err_P_1 > 1e-15 else 0

    # Calculate max velocity and positions
    max_v_0 = np.max(np.abs(prim_0['vr'][interior]))
    idx_max_v_0 = NUM_GHOSTS + int(np.argmax(np.abs(prim_0['vr'][interior])))
    r_max_v_0 = grid.r[idx_max_v_0]

    max_v_1 = np.max(np.abs(prim_1['vr'][interior]))
    idx_max_v_1 = NUM_GHOSTS + int(np.argmax(np.abs(prim_1['vr'][interior])))
    r_max_v_1 = grid.r[idx_max_v_1]

    max_v_100 = np.max(np.abs(prim_100['vr'][interior]))
    idx_max_v_100 = NUM_GHOSTS + int(np.argmax(np.abs(prim_100['vr'][interior])))
    r_max_v_100 = grid.r[idx_max_v_100]

    max_v_10000 = np.max(np.abs(prim_10000['vr'][interior]))
    idx_max_v_10000 = NUM_GHOSTS + int(np.argmax(np.abs(prim_10000['vr'][interior])))
    r_max_v_10000 = grid.r[idx_max_v_10000]

    print(f"\n{'='*70}")
    print(f"EVOLUTION DIAGNOSTICS (t=0 → t={t_1:.6e} → t={t_100:.6e} → t={t_10000:.6e})")
    print(f"{'='*70}")

    print(f"\n1. VELOCITY EVOLUTION:")
    print(f"   Max |v^r| at t=0:             {max_v_0:.3e} (r={r_max_v_0:.2f})")
    print(f"   Max |v^r| at t={t_1:.6e}:   {max_v_1:.3e} (r={r_max_v_1:.2f})")
    if t_100 != t_10000:
        print(f"   Max |v^r| at t={t_100:.6e}:   {max_v_100:.3e} (r={r_max_v_100:.2f})")
    print(f"   Max |v^r| at t={t_10000:.6e}: {max_v_10000:.3e} (r={r_max_v_10000:.2f})")

    print(f"\n2. CENTRAL DENSITY:")
    print(f"   ρ_c at t=0:                 {prim_0['rho0'][NUM_GHOSTS]:.6e}")
    print(f"   ρ_c at t={t_1:.6e}:   {prim_1['rho0'][NUM_GHOSTS]:.6e}")
    if t_100 != t_10000:
        print(f"   ρ_c at t={t_100:.6e}:   {prim_100['rho0'][NUM_GHOSTS]:.6e}")
    print(f"   ρ_c at t={t_10000:.6e}: {prim_10000['rho0'][NUM_GHOSTS]:.6e}")
    print(f"   Δρ_c/ρ_c (t={t_1:.6e}):    {abs(prim_1['rho0'][NUM_GHOSTS] - prim_0['rho0'][NUM_GHOSTS])/prim_0['rho0'][NUM_GHOSTS]:.3e}")
    if t_100 != t_10000:
        print(f"   Δρ_c/ρ_c (t={t_100:.6e}):  {abs(prim_100['rho0'][NUM_GHOSTS] - prim_0['rho0'][NUM_GHOSTS])/prim_0['rho0'][NUM_GHOSTS]:.3e}")
    print(f"   Δρ_c/ρ_c (t={t_10000:.6e}):{abs(prim_10000['rho0'][NUM_GHOSTS] - prim_0['rho0'][NUM_GHOSTS])/prim_0['rho0'][NUM_GHOSTS]:.3e}")

    print(f"\n3. DENSITY ERROR (max over domain):")
    print(f"   Max |Δρ|/ρ at t={t_1:.6e}:    {max_err_rho_1:.3e} (r={r_max_err_rho_1:.2f})")
    if t_100 != t_10000:
        print(f"   Max |Δρ|/ρ at t={t_100:.6e}:  {max_err_rho_100:.3e} (r={r_max_err_rho_100:.2f})")
    print(f"   Max |Δρ|/ρ at t={t_10000:.6e}: {max_err_rho_10000:.3e} (r={r_max_err_rho_10000:.2f})")
    print(f"   Growth factor (final/1):        {growth_rho:.1f}x")

    print(f"\n4. PRESSURE ERROR (max over domain):")
    print(f"   Max |ΔP|/P at t={t_1:.6e}:    {max_err_P_1:.3e} (r={r_max_err_P_1:.2f})")
    if t_100 != t_10000:
        print(f"   Max |ΔP|/P at t={t_100:.6e}:  {max_err_P_100:.3e} (r={r_max_err_P_100:.2f})")
    print(f"   Max |ΔP|/P at t={t_10000:.6e}: {max_err_P_10000:.3e} (r={r_max_err_P_10000:.2f})")
    print(f"   Growth factor (final/1):        {growth_P:.1f}x")

    print(f"\n5. CONS2PRIM STATUS:")
    print(f"   Success at t=0:               {np.sum(prim_0['success'])}/{grid.N}")
    print(f"   Success at t={t_1:.6e}:   {np.sum(prim_1['success'])}/{grid.N}")
    if t_100 != t_10000:
        print(f"   Success at t={t_100:.6e}:   {np.sum(prim_100['success'])}/{grid.N}")
    print(f"   Success at t={t_10000:.6e}: {np.sum(prim_10000['success'])}/{grid.N}")

    if not np.all(prim_10000['success']):
        failed_idx = np.where(~prim_10000['success'])[0]
        print(f"   Failed points: {failed_idx[:5]} (first 5)")
        print(f"   Failed radii:  {grid.r[failed_idx[:5]]}")

    # Finalize data saving
    if ENABLE_DATA_SAVING:
        data_manager.finalize()

    print("\n" + "="*70)
    print("Evolution complete. Plots saved:")
    print("  1. tov_solution.png                   - TOV solution (ρ, P, M, α)")
    print("  2. tov_initial_data_comparison.png    - TOV vs Initial data at t=0")
    if t_100 != t_10000:
        print(f"  3. tov_evolution.png                  - Hydro evolution: t=0 → t={t_1:.6e} → t={t_100:.6e} → t={t_10000:.6e}")
    else:
        print(f"  3. tov_evolution.png                  - Hydro evolution: t=0 → t={t_1:.6e} → t={t_10000:.6e}")
    print(f"  4. tov_bssn_evolution.png             - BSSN variables: t=0 → t={t_10000:.6e} (Full evolution)")
    if ENABLE_DATA_SAVING:
        print(f"  5. tov_hamiltonian_constraint.png     - Hamiltonian constraint evolution")
    print("="*70)


if __name__ == "__main__":
    main()
