"""
TOV Evolution Utilities

This module contains utilities for TOV evolution simulations:
- Data management (SimulationDataManager)
- Helper functions (_compute_baryon_mass, enforce_origin_symmetry, _apply_atmosphere_reset)
- Plotting functions (all visualization)
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.interpolate import interp1d
import h5py
import json
from datetime import datetime

# Import required BSSN and Hydro components
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import (NUM_BSSN_VARS, idx_phi, idx_hrr, idx_htt, idx_hpp,
                                             idx_K, idx_arr, idx_att, idx_app, idx_lapse)
from source.core.spacing import NUM_GHOSTS
from source.bssn.tensoralgebra import get_bar_gamma_LL
from source.matter.hydro.cons2prim import prim_to_cons

# Create directory for plots
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

    def __init__(self, output_dir, grid, hydro, enable_saving=False):
        """
        Initialize data manager.

        Args:
            output_dir: Directory for output files
            grid: Grid object
            hydro: Hydro object
            enable_saving: If True, saves data to files
        """
        self.enable_saving = enable_saving
        if not self.enable_saving:
            return

        self.output_dir = output_dir
        self.grid = grid
        self.hydro = hydro

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize HDF5 files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.snapshot_file = os.path.join(output_dir, f"tov_snapshots_{timestamp}.h5")
        self.evolution_file = os.path.join(output_dir, f"tov_evolution_{timestamp}.h5")
        self.metadata_file = os.path.join(output_dir, f"tov_metadata_{timestamp}.json")

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
            'c2p_fails': [],
            'max_Ham': [],
            'l2_Ham': []
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

        metadata = {
            'tov_solution': {
                'M_star': float(tov_solution['M_star']),
                'R': float(tov_solution['R']),
                'C': float(tov_solution['C']),
                'K': float(K) if K is not None else float(tov_solution.get('K', 0)),
                'Gamma': float(Gamma) if Gamma is not None else float(tov_solution.get('Gamma', 0)),
                'rho_central': float(rho_central) if rho_central is not None else float(tov_solution.get('rho_central', 0))
            },
            'atmosphere': {
                'rho_floor': float(atmosphere_params.rho_floor),
                'p_floor': float(atmosphere_params.p_floor),
                'v_max': float(atmosphere_params.v_max),
                'W_max': float(atmosphere_params.W_max)
            },
            'simulation': {
                'dt': float(dt) if dt is not None else None,
                'integration_method': integration_method,
                'grid_N': int(self.grid.N),
                'grid_r_max': float(self.grid.r[-1])
            },
            'timestamp': datetime.now().isoformat()
        }

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def save_snapshot(self, step, time, state_2d, primitives=None, Ham=None, Mom=None):
        """Save full domain snapshot."""
        if not self.enable_saving:
            return

        with h5py.File(self.snapshot_file, 'a') as f:
            # Check if snapshot already exists, if so skip or overwrite
            snap_name = f'step_{step:08d}'
            if snap_name in f['snapshots']:
                print(f"Warning: Snapshot {snap_name} already exists, skipping...")
                return
            snap_group = f['snapshots'].create_group(snap_name)

            # Save metadata for this snapshot
            snap_group.attrs['step'] = step
            snap_group.attrs['time'] = time

            # Save conservative variables
            cons_group = snap_group.create_group('conservatives')
            cons_group['D'] = state_2d[self.hydro.idx_D, :]
            cons_group['Sr'] = state_2d[self.hydro.idx_Sr, :]
            cons_group['tau'] = state_2d[self.hydro.idx_tau, :]

            # Save BSSN variables
            bssn_group = snap_group.create_group('bssn')
            bssn_group['phi'] = state_2d[0, :]
            bssn_group['a'] = state_2d[1, :]
            bssn_group['alpha'] = state_2d[2, :]
            bssn_group['betaR'] = state_2d[3, :]
            bssn_group['Br'] = state_2d[4, :]
            bssn_group['K'] = state_2d[5, :]

            # Save primitives if provided
            if primitives is not None:
                prim_group = snap_group.create_group('primitives')
                prim_group['rho0'] = primitives['rho0']
                prim_group['p'] = primitives['p']
                prim_group['eps'] = primitives['eps']
                prim_group['vr'] = primitives['vr']
                prim_group['W'] = primitives['W']

            # Save constraints if provided
            if Ham is not None:
                const_group = snap_group.create_group('constraints')
                const_group['Ham'] = Ham
                if Mom is not None:
                    const_group['Mom'] = Mom

    def add_evolution_point(self, step, time, state_2d, primitives, reference_primitives, Ham=None):
        """Add a point to the evolution time series."""
        if not self.enable_saving:
            return

        interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

        # Calculate errors
        delta_rho = np.abs(primitives['rho0'][interior] - reference_primitives['rho0'][interior])
        rel_delta_rho = delta_rho / (np.abs(reference_primitives['rho0'][interior]) + 1e-20)

        delta_p = np.abs(primitives['p'][interior] - reference_primitives['p'][interior])
        rel_delta_p = delta_p / (np.abs(reference_primitives['p'][interior]) + 1e-20)

        # L1 and L2 norms of density error (inside star only)
        star_mask = reference_primitives['rho0'][interior] > 10 * self.hydro.atmosphere.rho_floor
        if np.any(star_mask):
            l1_rho = np.mean(delta_rho[star_mask])
            l2_rho = np.sqrt(np.mean(delta_rho[star_mask]**2))
        else:
            l1_rho = 0.0
            l2_rho = 0.0

        # Calculate Hamiltonian constraint metrics if provided
        if Ham is not None:
            max_Ham = np.max(np.abs(Ham[interior]))
            l2_Ham = np.sqrt(np.mean(Ham[interior]**2))
        else:
            max_Ham = 0.0
            l2_Ham = 0.0

        # Store in buffer
        self.evolution_buffer['step'].append(step)
        self.evolution_buffer['time'].append(time)
        self.evolution_buffer['rho_central'].append(primitives['rho0'][NUM_GHOSTS])
        self.evolution_buffer['p_central'].append(primitives['p'][NUM_GHOSTS])
        self.evolution_buffer['max_rho_error'].append(np.max(rel_delta_rho))
        self.evolution_buffer['max_p_error'].append(np.max(rel_delta_p))
        self.evolution_buffer['max_velocity'].append(np.max(np.abs(primitives['vr'])))
        self.evolution_buffer['l1_rho_error'].append(l1_rho)
        self.evolution_buffer['l2_rho_error'].append(l2_rho)
        self.evolution_buffer['max_D'].append(np.max(state_2d[self.hydro.idx_D, :]))
        self.evolution_buffer['max_Sr'].append(np.max(np.abs(state_2d[self.hydro.idx_Sr, :])))
        self.evolution_buffer['max_tau'].append(np.max(state_2d[self.hydro.idx_tau, :]))
        self.evolution_buffer['c2p_fails'].append(np.sum(~primitives['success']))
        self.evolution_buffer['max_Ham'].append(max_Ham)
        self.evolution_buffer['l2_Ham'].append(l2_Ham)

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
# HELPER FUNCTIONS
# =============================================================================

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


def enforce_origin_symmetry(state_2d):
    """
    Enforce spherical symmetry at origin: set momentum S^r = 0 at first interior cell.

    By spherical symmetry, velocity must be exactly zero at r=0. This prevents
    spurious mass creation from connection terms -Γ*F near origin where Γ ~ 200/M.

    Args:
        state_2d: State array with shape (NUM_VARS, N)

    Returns:
        Modified state_2d with S^r = 0 at first interior cell
    """
    # S^r_tilde is at index NUM_BSSN_VARS + 1 = 13
    # We enforce S^r = 0 at first interior cell
    state_2d[NUM_BSSN_VARS + 1, NUM_GHOSTS] = 0.0  # S^r = 0 at r ≈ 0
    return state_2d


def _apply_atmosphere_reset(state_2d, grid, hydro, atmosphere, rho_threshold=None):
    """
    Apply atmosphere floors using    strategy.

    Strategy (following   / +):
    1. Recover primitives from conservatives (cons2prim)
    2. Apply floors to PRIMITIVES (ρ, v, P)
    3. Apply conservative variable consistency checks (tau floor, S^2 constraint)
    4. Recompute conservatives from floored primitives if needed

    This ensures thermodynamic consistency: floored conservatives correspond
    to a valid physical state.

    Args:
        state_2d: State vector (2D array)
        grid: Grid object
        hydro: Hydrodynamics object
        atmosphere: AtmosphereParams object
        rho_threshold: Threshold below which to apply atmosphere (default: 10 * rho_floor)

    Returns:
        state_2d with floors applied
    """
    from source.matter.hydro.atmosphere import FloorApplicator

    if rho_threshold is None:
        # Use more aggressive threshold to prevent spurious wave generation
        rho_threshold = 10.0 * atmosphere.rho_floor

    # Extract conservatives
    D = state_2d[NUM_BSSN_VARS + 0, :]
    Sr = state_2d[NUM_BSSN_VARS + 1, :]
    tau = state_2d[NUM_BSSN_VARS + 2, :]

    # Build BSSN geometry (needed for cons2prim and floor application)
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])

    # Get metric for floor application
    bar_gamma_LL = get_bar_gamma_LL(grid.r, bssn_vars.h_LL, hydro.background)
    phi = np.asarray(bssn_vars.phi, dtype=float)
    e4phi = np.exp(4.0 * phi)
    gamma_rr = e4phi * bar_gamma_LL[:, 0, 0]  # Physical γ_rr

    # Recover primitives (cons2prim already has some floor logic built in)
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    prim = hydro._get_primitives(bssn_vars, grid.r)

    rho0 = prim['rho0']
    vr = prim['vr']
    p = prim['p']

    # Create floor applicator
    floor_app = FloorApplicator(atmosphere, hydro.eos)

    # STEP 1: Apply primitive floors (ρ, v, P)
    rho0_floor, vr_floor, p_floor = floor_app.apply_primitive_floors(rho0, vr, p, gamma_rr)

    # STEP 2: Identify atmosphere regions (where there is NO real fluid)
    atm_mask = D < rho_threshold

    # CRITICAL: In atmosphere, velocity MUST be zero (no fluid to move!)
    if np.any(atm_mask):
        vr_floor[atm_mask] = 0.0  # Force v=0 in atmosphere

    # STEP 3: Apply conservative consistency floors (tau, S_i constraints)
    D_floor, Sr_floor, tau_floor, cons_floor_applied = floor_app.apply_conservative_floors(
        D, Sr, tau, gamma_rr
    )

    # STEP 4: Recompute conservatives from floored primitives where needed
    prim_floor_applied = (
        (np.abs(rho0_floor - rho0) > 1e-14) |
        (np.abs(vr_floor - vr) > 1e-14) |
        (np.abs(p_floor - p) > 1e-14)
    )

    # Combine all masks to identify points that need floor application
    needs_floor = prim_floor_applied | atm_mask | cons_floor_applied

    if np.any(needs_floor):
        # Recompute conservatives from floored primitives
        D_new, Sr_new, tau_new = prim_to_cons(rho0_floor, vr_floor, p_floor, gamma_rr, hydro.eos)

        # Apply conservative floors again to ensure consistency
        D_new, Sr_new, tau_new, _ = floor_app.apply_conservative_floors(
            D_new, Sr_new, tau_new, gamma_rr
        )

        # Update state ONLY at points that need floors
        state_2d[NUM_BSSN_VARS + 0, needs_floor] = D_new[needs_floor]
        state_2d[NUM_BSSN_VARS + 1, needs_floor] = Sr_new[needs_floor]
        state_2d[NUM_BSSN_VARS + 2, needs_floor] = tau_new[needs_floor]

    # STEP 5: Force zero momentum in outer ghost cells when in atmosphere
    outer_ghosts = slice(-NUM_GHOSTS, None)
    outer_atm_mask = D[outer_ghosts] < rho_threshold

    if np.any(outer_atm_mask):
        state_2d[NUM_BSSN_VARS + 1, outer_ghosts][outer_atm_mask] = 0.0

    return state_2d


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================


def plot_first_step(state_t0, state_t1, grid, hydro, tov_solution=None):
    """Plot only t=0 vs t=1×dt to inspect the first update."""
    bssn_0 = BSSNVars(grid.N)
    bssn_0.set_bssn_vars(state_t0[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t0, bssn_0, grid)
    prim_0 = hydro._get_primitives(bssn_0, grid.r)

    bssn_1 = BSSNVars(grid.N)
    bssn_1.set_bssn_vars(state_t1[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t1, bssn_1, grid)
    prim_1 = hydro._get_primitives(bssn_1, grid.r)

    r_int = grid.r

    # Compute mass using integral M(r) = 4π ∫_0^r ρ r^2 dr
    rho_0 = prim_0['rho0']
    rho_1 = prim_1['rho0']
    # Full grid mass calculation
    M_0_full = 4.0 * np.pi * cumulative_trapezoid(rho_0 * grid.r**2, grid.r, initial=0.0)
    M_1_full = 4.0 * np.pi * cumulative_trapezoid(rho_1 * grid.r**2, grid.r, initial=0.0)

    # ========================================================================
    # ANALYSIS: Print values at key locations
    # ========================================================================
    print("\n" + "="*80)
    print("FIRST STEP ANALYSIS: Changes from t=0 to t=1×dt")
    print("="*80)

    # Find indices for key locations
    idx_center = NUM_GHOSTS  # First interior point (center)
    idx_left = NUM_GHOSTS + 10  # Near left boundary (interior)

    # Find index closest to stellar radius (if TOV solution provided)
    if tov_solution is not None:
        R_star = tov_solution['R']
        idx_star = np.argmin(np.abs(grid.r - R_star))
        print(f"\nStellar radius R = {R_star:.4f} at grid index {idx_star} (r = {grid.r[idx_star]:.4f})")
    else:
        # Estimate stellar surface as where density drops below threshold
        rho_threshold = 1e-6
        interior_mask = rho_0 > rho_threshold
        if np.any(interior_mask):
            idx_star = np.where(interior_mask)[0][-1]
            R_star = grid.r[idx_star]
            print(f"\nEstimated stellar surface at r = {R_star:.4f} (ρ < {rho_threshold})")
        else:
            idx_star = len(grid.r) // 2
            R_star = grid.r[idx_star]
            print(f"\nUsing midpoint as reference: r = {R_star:.4f}")

    idx_right = -NUM_GHOSTS - 1  # Last interior point (far boundary)

    def analyze_point(idx, label):
        """Analyze all quantities at a given index."""
        r = grid.r[idx]
        print(f"\n{'-'*80}")
        print(f"{label}: r = {r:.6f} (index {idx})")
        print(f"{'-'*80}")

        # Hydro primitives
        rho0_0, rho0_1 = rho_0[idx], rho_1[idx]
        p0, p1 = prim_0['p'][idx], prim_1['p'][idx]
        vr0, vr1 = prim_0['vr'][idx], prim_1['vr'][idx]

        # BSSN variables
        phi0, phi1 = state_t0[idx_phi, idx], state_t1[idx_phi, idx]
        alpha0, alpha1 = state_t0[idx_lapse, idx], state_t1[idx_lapse, idx]
        hrr0, hrr1 = state_t0[idx_hrr, idx], state_t1[idx_hrr, idx]
        K0, K1 = state_t0[idx_K, idx], state_t1[idx_K, idx]
        arr0, arr1 = state_t0[idx_arr, idx], state_t1[idx_arr, idx]

        # Mass
        M0, M1 = M_0_full[idx], M_1_full[idx]

        # Conservative variables
        D0 = state_t0[NUM_BSSN_VARS + 0, idx]
        D1 = state_t1[NUM_BSSN_VARS + 0, idx]
        Sr0 = state_t0[NUM_BSSN_VARS + 1, idx]
        Sr1 = state_t1[NUM_BSSN_VARS + 1, idx]
        tau0 = state_t0[NUM_BSSN_VARS + 2, idx]
        tau1 = state_t1[NUM_BSSN_VARS + 2, idx]

        def rel_change(v0, v1):
            """Relative change with safe division."""
            if abs(v0) < 1e-30:
                return f"{v1:.6e}" if abs(v1) > 1e-30 else "0"
            return f"{(v1-v0)/v0:.6e}"

        def abs_change(v0, v1):
            """Absolute change."""
            return f"{v1-v0:.6e}"

        print(f"\nPRIMITIVE VARIABLES:")
        print(f"  ρ₀:     {rho0_0:.8e} → {rho0_1:.8e}   Δrel = {rel_change(rho0_0, rho0_1)}")
        print(f"  P:      {p0:.8e} → {p1:.8e}   Δrel = {rel_change(p0, p1)}")
        print(f"  vʳ:     {vr0:.8e} → {vr1:.8e}   Δabs = {abs_change(vr0, vr1)}")

        print(f"\nCONSERVATIVE VARIABLES:")
        print(f"  D:      {D0:.8e} → {D1:.8e}   Δrel = {rel_change(D0, D1)}")
        print(f"  Sʳ:     {Sr0:.8e} → {Sr1:.8e}   Δabs = {abs_change(Sr0, Sr1)}")
        print(f"  τ:      {tau0:.8e} → {tau1:.8e}   Δrel = {rel_change(tau0, tau1)}")

        print(f"\nBSSN VARIABLES:")
        print(f"  φ:      {phi0:.8e} → {phi1:.8e}   Δrel = {rel_change(phi0, phi1)}")
        print(f"  α:      {alpha0:.8e} → {alpha1:.8e}   Δrel = {rel_change(alpha0, alpha1)}")
        print(f"  hʳʳ:    {hrr0:.8e} → {hrr1:.8e}   Δrel = {rel_change(hrr0, hrr1)}")
        print(f"  K:      {K0:.8e} → {K1:.8e}   Δabs = {abs_change(K0, K1)}")
        print(f"  aʳʳ:    {arr0:.8e} → {arr1:.8e}   Δabs = {abs_change(arr0, arr1)}")

        print(f"\nDERIVED QUANTITIES:")
        print(f"  M(r):   {M0:.8e} → {M1:.8e}   Δrel = {rel_change(M0, M1)}")

        # Compute specific internal energy and enthalpy
        eps0 = hydro.eos.eps_from_rho_p(rho0_0, p0)
        eps1 = hydro.eos.eps_from_rho_p(rho0_1, p1)
        h0 = 1.0 + eps0 + p0/max(rho0_0, 1e-30)
        h1 = 1.0 + eps1 + p1/max(rho0_1, 1e-30)
        print(f"  ε:      {eps0:.8e} → {eps1:.8e}   Δrel = {rel_change(eps0, eps1)}")
        print(f"  h:      {h0:.8e} → {h1:.8e}   Δrel = {rel_change(h0, h1)}")

    # Analyze key points
    analyze_point(idx_center, "CENTER (r≈0)")
    analyze_point(idx_left, "LEFT INTERIOR")
    analyze_point(idx_star, "STELLAR SURFACE")
    analyze_point(idx_right, "FAR BOUNDARY")

    # Global statistics
    print(f"\n{'='*80}")
    print("GLOBAL STATISTICS:")
    print(f"{'='*80}")

    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

    def safe_rel_error(arr0, arr1):
        """Compute relative error safely."""
        mask = np.abs(arr0) > 1e-30
        rel_err = np.zeros_like(arr0)
        rel_err[mask] = np.abs((arr1[mask] - arr0[mask]) / arr0[mask])
        return rel_err

    rho_rel_err = safe_rel_error(rho_0[interior], rho_1[interior])
    p_rel_err = safe_rel_error(prim_0['p'][interior], prim_1['p'][interior])
    phi_rel_err = safe_rel_error(state_t0[idx_phi, interior], state_t1[idx_phi, interior])
    alpha_rel_err = safe_rel_error(state_t0[idx_lapse, interior], state_t1[idx_lapse, interior])

    print(f"\nMax relative changes (interior points):")
    print(f"  ρ₀:     {np.max(rho_rel_err):.6e}")
    print(f"  P:      {np.max(p_rel_err):.6e}")
    print(f"  φ:      {np.max(phi_rel_err):.6e}")
    print(f"  α:      {np.max(alpha_rel_err):.6e}")
    print(f"  |vʳ|:   {np.max(np.abs(prim_1['vr'][interior])):.6e}")

    print(f"\nLocations of maximum changes:")
    idx_max_rho = NUM_GHOSTS + np.argmax(rho_rel_err)
    idx_max_p = NUM_GHOSTS + np.argmax(p_rel_err)
    idx_max_v = NUM_GHOSTS + np.argmax(np.abs(prim_1['vr'][interior]))
    print(f"  Max Δρ/ρ at r = {grid.r[idx_max_rho]:.6f} (index {idx_max_rho})")
    print(f"  Max ΔP/P at r = {grid.r[idx_max_p]:.6f} (index {idx_max_p})")
    print(f"  Max |vʳ| at r = {grid.r[idx_max_v]:.6f} (index {idx_max_v})")

    print("="*80 + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Density
    axes[0, 0].semilogy(r_int, prim_0['rho0'], 'b-', linewidth=2, label='t=0')
    axes[0, 0].semilogy(r_int, prim_1['rho0'], 'r--', linewidth=1.7, label='t=1×dt')
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel(r'$\rho_0$')
    axes[0, 0].set_title('Density: first step')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Pressure
    axes[0, 1].semilogy(r_int, np.maximum(prim_0['p'], 1e-20), 'b-', linewidth=2, label='t=0')
    axes[0, 1].semilogy(r_int, np.maximum(prim_1['p'], 1e-20), 'r--', linewidth=1.7, label='t=1×dt')
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure: first step')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Velocity
    axes[0, 2].plot(r_int, prim_0['vr'], 'b-', linewidth=2, label='t=0')
    axes[0, 2].plot(r_int, prim_1['vr'], 'r--', linewidth=1.7, label='t=1×dt')
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel(r'$v^r$')
    axes[0, 2].set_title('Velocity: first step')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Lapse
    axes[1, 0].plot(r_int, state_t0[idx_lapse, :], 'b-', linewidth=2, label='t=0')
    axes[1, 0].plot(r_int, state_t1[idx_lapse, :], 'r--', linewidth=1.7, label='t=1×dt')
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel(r'$\alpha$')
    axes[1, 0].set_title('Lapse: first step')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Mass
    axes[1, 1].plot(r_int, M_0_full, 'b-', linewidth=2, label='t=0')
    axes[1, 1].plot(r_int, M_1_full, 'r--', linewidth=1.7, label='t=1×dt')
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel('M(r)')
    axes[1, 1].set_title('Enclosed Mass: first step')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Phi
    axes[1, 2].plot(r_int, state_t0[idx_phi, :], 'b-', linewidth=2, label='t=0')
    axes[1, 2].plot(r_int, state_t1[idx_phi, :], 'r--', linewidth=1.7, label='t=1×dt')
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel(r'$\phi$')
    axes[1, 2].set_title('Conformal Factor: first step')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'tov_first_step.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_surface_zoom(tov_solution, state_t0, state_t1, grid, hydro, window=0.5):
    """Zoom near the stellar surface R to compare t=0 vs t=1×dt.

    Plots overlays for (ρ0, P, v^r, D, S_r, τ) in a window [R−window, R+window].
    """
    R = float(tov_solution['R'])
    r = grid.r
    mask = (r >= R - window) & (r <= R + window)
    if not np.any(mask):
        return

    bssn0 = BSSNVars(grid.N)
    bssn0.set_bssn_vars(state_t0[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t0, bssn0, grid)
    prim0 = hydro._get_primitives(bssn0, r)

    bssn1 = BSSNVars(grid.N)
    bssn1.set_bssn_vars(state_t1[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t1, bssn1, grid)
    prim1 = hydro._get_primitives(bssn1, r)

    D0, Sr0, tau0 = state_t0[NUM_BSSN_VARS + 0, :], state_t0[NUM_BSSN_VARS + 1, :], state_t0[NUM_BSSN_VARS + 2, :]
    D1, Sr1, tau1 = state_t1[NUM_BSSN_VARS + 0, :], state_t1[NUM_BSSN_VARS + 1, :], state_t1[NUM_BSSN_VARS + 2, :]

    rZ = r[mask]
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    # Row 1: rho0, P, vr
    ax[0, 0].semilogy(rZ, np.maximum(prim0['rho0'][mask], 1e-20), 'b-', label='t=0')
    ax[0, 0].semilogy(rZ, np.maximum(prim1['rho0'][mask], 1e-20), 'r--', label='t=1×dt')
    ax[0, 0].axvline(R, color='gray', ls=':'); ax[0, 0].set_title('ρ0 (zoom)'); ax[0, 0].legend(); ax[0, 0].grid(True, alpha=0.3)

    ax[0, 1].semilogy(rZ, np.maximum(prim0['p'][mask], 1e-20), 'b-')
    ax[0, 1].semilogy(rZ, np.maximum(prim1['p'][mask], 1e-20), 'r--')
    ax[0, 1].axvline(R, color='gray', ls=':'); ax[0, 1].set_title('P (zoom)'); ax[0, 1].grid(True, alpha=0.3)

    ax[0, 2].plot(rZ, prim0['vr'][mask], 'b-')
    ax[0, 2].plot(rZ, prim1['vr'][mask], 'r--')
    ax[0, 2].axvline(R, color='gray', ls=':'); ax[0, 2].set_title('v^r (zoom)'); ax[0, 2].grid(True, alpha=0.3)

    # Row 2: D, Sr, tau
    ax[1, 0].semilogy(rZ, np.maximum(D0[mask], 1e-22), 'b-')
    ax[1, 0].semilogy(rZ, np.maximum(D1[mask], 1e-22), 'r--')
    ax[1, 0].axvline(R, color='gray', ls=':'); ax[1, 0].set_title('D (zoom)'); ax[1, 0].grid(True, alpha=0.3)

    ax[1, 1].plot(rZ, Sr0[mask], 'b-')
    ax[1, 1].plot(rZ, Sr1[mask], 'r--')
    ax[1, 1].axvline(R, color='gray', ls=':'); ax[1, 1].set_title('S_r (zoom)'); ax[1, 1].grid(True, alpha=0.3)

    ax[1, 2].semilogy(rZ, np.maximum(np.abs(tau0[mask]), 1e-22), 'b-')
    ax[1, 2].semilogy(rZ, np.maximum(np.abs(tau1[mask]), 1e-22), 'r--')
    ax[1, 2].axvline(R, color='gray', ls=':'); ax[1, 2].set_title('τ (zoom)'); ax[1, 2].grid(True, alpha=0.3)

    for a in ax.ravel():
        a.set_xlabel('r')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'tov_surface_zoom.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_tov_vs_initial_data_zoom(tov_solution, initial_state_2d, grid, hydro, window=0.5):
    """Zoom near the stellar surface R to compare TOV solution vs interpolated initial data.

    This plot helps identify interpolation errors and differences between the analytic
    TOV solution and the discretized initial data on the evolution grid.

    Plots overlays for (ρ0, P, v^r, D, S_r, τ) in a window [R−window, R+window].
    """
    R = float(tov_solution['R'])
    r = grid.r
    mask = (r >= R - window) & (r <= R + window)
    if not np.any(mask):
        return

    # Get primitives from initial data
    bssn_init = BSSNVars(grid.N)
    bssn_init.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(initial_state_2d, bssn_init, grid)
    prim_init = hydro._get_primitives(bssn_init, r)

    # Get conservatives from initial data
    D_init = initial_state_2d[NUM_BSSN_VARS + 0, :]
    Sr_init = initial_state_2d[NUM_BSSN_VARS + 1, :]
    tau_init = initial_state_2d[NUM_BSSN_VARS + 2, :]

    # Interpolate TOV solution to zoom region
    r_tov = tov_solution['r']
    rho_tov = tov_solution['rho_baryon']
    P_tov = tov_solution['P']

    rho_tov_interp = interp1d(r_tov, rho_tov, kind='cubic', fill_value=0.0, bounds_error=False)
    P_tov_interp = interp1d(r_tov, P_tov, kind='cubic', fill_value=0.0, bounds_error=False)

    rho_tov_zoom = rho_tov_interp(r)
    P_tov_zoom = P_tov_interp(r)

    rZ = r[mask]
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))

    # Row 1: ρ0, P, v^r
    ax[0, 0].semilogy(rZ, np.maximum(rho_tov_zoom[mask], 1e-20), 'k-', linewidth=2, label='TOV (analytic)')
    ax[0, 0].semilogy(rZ, np.maximum(prim_init['rho0'][mask], 1e-20), 'b--', linewidth=1.5, label='Initial Data')
    ax[0, 0].axvline(R, color='gray', ls=':', linewidth=1.5, label=f'R={R:.3f}')
    ax[0, 0].set_title('ρ₀ (zoom near surface)', fontsize=11)
    ax[0, 0].legend(fontsize=9)
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].set_ylabel('ρ₀', fontsize=10)

    ax[0, 1].semilogy(rZ, np.maximum(P_tov_zoom[mask], 1e-20), 'k-', linewidth=2)
    ax[0, 1].semilogy(rZ, np.maximum(prim_init['p'][mask], 1e-20), 'b--', linewidth=1.5)
    ax[0, 1].axvline(R, color='gray', ls=':', linewidth=1.5)
    ax[0, 1].set_title('P (zoom near surface)', fontsize=11)
    ax[0, 1].grid(True, alpha=0.3)
    ax[0, 1].set_ylabel('P', fontsize=10)

    # v^r should be zero in TOV equilibrium
    ax[0, 2].plot(rZ, prim_init['vr'][mask], 'b-', linewidth=1.5, label='Initial Data')
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
    ax[1, 2].set_title('τ (conserved energy)', fontsize=11)
    ax[1, 2].grid(True, alpha=0.3)
    ax[1, 2].set_ylabel('|τ|', fontsize=10)

    for a in ax.ravel():
        a.set_xlabel('r [M]', fontsize=10)

    plt.suptitle(f'TOV Solution vs Initial Data: Surface Zoom [R−{window}, R+{window}]',
                 fontsize=13, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'tov_vs_initial_zoom.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_tov_diagnostics(tov_solution, r_max):
    """Plot TOV solution diagnostics (without conformal factor)."""
    r = tov_solution['r']
    R_star = tov_solution['R']
    M_star = tov_solution['M_star']

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Density
    axes[0, 0].semilogy(r, tov_solution['rho_baryon'], color='navy')
    axes[0, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5, label=f'R={R_star:.2f}')
    axes[0, 0].set_xlabel(r"$r$")
    axes[0, 0].set_ylabel(r"$\rho_0$")
    axes[0, 0].set_title('Baryon Density')
    axes[0, 0].set_xlim(0, r_max)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Pressure
    axes[0, 1].semilogy(r, tov_solution['P'], color='darkgreen')
    axes[0, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel(r"$r$")
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure')
    axes[0, 1].set_xlim(0, r_max)
    axes[0, 1].grid(True, alpha=0.3)

    # Enclosed Mass
    axes[0, 2].plot(r, tov_solution['M'], color='maroon')
    axes[0, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[0, 2].axhline(M_star, color='gray', linestyle='--', alpha=0.3, label=f'M={M_star:.3f}')
    axes[0, 2].set_xlabel(r"$r$")
    axes[0, 2].set_ylabel('M(r)')
    axes[0, 2].set_title('Enclosed Mass')
    axes[0, 2].set_xlim(0, r_max)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Lapse alpha(r)
    axes[1, 0].plot(r, tov_solution['alpha'], color='purple')
    axes[1, 0].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel(r"$r$")
    axes[1, 0].set_ylabel(r'$\alpha$')
    axes[1, 0].set_title('Lapse Function')
    axes[1, 0].set_xlim(0, r_max)
    axes[1, 0].grid(True, alpha=0.3)

    # Phi(r)
    phi = 0.25 * np.log(tov_solution['exp4phi'])
    axes[1, 1].plot(r, phi, color='teal')
    axes[1, 1].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel(r"$r$")
    axes[1, 1].set_ylabel(r'$\phi$')
    axes[1, 1].set_title(r'Conformal Factor $\phi$')
    axes[1, 1].set_xlim(0, r_max)
    axes[1, 1].grid(True, alpha=0.3)

    # a(r) metric function: a = exp(2*phi) = sqrt(exp4phi)
    a_metric = np.sqrt(tov_solution['exp4phi'])
    axes[1, 2].plot(r, a_metric, color='orange')
    axes[1, 2].axvline(R_star, color='gray', linestyle=':', alpha=0.5)
    axes[1, 2].set_xlabel(r"$r$")
    axes[1, 2].set_ylabel(r'$a(r)$')
    axes[1, 2].set_title('Metric $a(r)$')
    axes[1, 2].set_xlim(0, r_max)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'tov_solution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_bssn_evolution(state_t0, state_tfinal, grid, t_0=0.0, t_final=1.0):
    """Plot BSSN variables at initial and final time to verify Cowling approximation.

    In Cowling approximation, BSSN variables should remain constant.
    This plot helps verify that the spacetime is indeed frozen.

    Note on spherical symmetry:
    - In spherical symmetry, h_θθ should equal h_φφ (isotropy)
    - Similarly, A_θθ should equal A_φφ
    - We plot all components to verify this consistency
    - Only h_rr and A_rr are truly independent radial variables

    Args:
        state_t0: Initial state
        state_tfinal: Final state
        grid: Grid object
        t_0: Initial time (for labeling)
        t_final: Final time (for labeling)
    """
    r_int = grid.r

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    # Row 1: Conformal factor φ, Lapse α, Trace of extrinsic curvature K
    # φ
    axes[0, 0].plot(r_int, state_t0[idx_phi, :], 'b-', linewidth=2, label=f't={t_0:.6e}')
    axes[0, 0].plot(r_int, state_tfinal[idx_phi, :], 'r--', linewidth=1.7, label=f't={t_final:.6e}')
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel(r'$\phi$')
    axes[0, 0].set_title('Conformal Factor')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # α
    axes[0, 1].plot(r_int, state_t0[idx_lapse, :], 'b-', linewidth=2, label=f't={t_0:.6e}')
    axes[0, 1].plot(r_int, state_tfinal[idx_lapse, :], 'r--', linewidth=1.7, label=f't={t_final:.6e}')
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel(r'$\alpha$')
    axes[0, 1].set_title('Lapse Function')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # K
    axes[0, 2].plot(r_int, state_t0[idx_K, :], 'b-', linewidth=2, label=f't={t_0:.6e}')
    axes[0, 2].plot(r_int, state_tfinal[idx_K, :], 'r--', linewidth=1.7, label=f't={t_final:.6e}')
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel(r'$K$')
    axes[0, 2].set_title('Trace of Extrinsic Curvature')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Conformal metric components h_rr, h_θθ, h_φφ
    # h_rr
    axes[1, 0].plot(r_int, state_t0[idx_hrr, :], 'b-', linewidth=2, label=f't={t_0:.6e}')
    axes[1, 0].plot(r_int, state_tfinal[idx_hrr, :], 'r--', linewidth=1.7, label=f't={t_final:.6e}')
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel(r'$h_{rr}$')
    axes[1, 0].set_title('Conformal Metric $h_{rr}$')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # h_θθ
    axes[1, 1].plot(r_int, state_t0[idx_htt, :], 'b-', linewidth=2, label=f't={t_0:.6e}')
    axes[1, 1].plot(r_int, state_tfinal[idx_htt, :], 'r--', linewidth=1.7, label=f't={t_final:.6e}')
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel(r'$h_{\theta\theta}$')
    axes[1, 1].set_title(r'Conformal Metric $h_{\theta\theta}$')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # h_φφ
    axes[1, 2].plot(r_int, state_t0[idx_hpp, :], 'b-', linewidth=2, label=f't={t_0:.6e}')
    axes[1, 2].plot(r_int, state_tfinal[idx_hpp, :], 'r--', linewidth=1.7, label=f't={t_final:.6e}')
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel(r'$h_{\phi\phi}$')
    axes[1, 2].set_title(r'Conformal Metric $h_{\phi\phi}$')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # Row 3: Traceless extrinsic curvature components A_rr, A_θθ, A_φφ
    # A_rr
    axes[2, 0].plot(r_int, state_t0[idx_arr, :], 'b-', linewidth=2, label=f't={t_0:.6e}')
    axes[2, 0].plot(r_int, state_tfinal[idx_arr, :], 'r--', linewidth=1.7, label=f't={t_final:.6e}')
    axes[2, 0].set_xlabel('r')
    axes[2, 0].set_ylabel(r'$A_{rr}$')
    axes[2, 0].set_title('Traceless Extrinsic Curvature $A_{rr}$')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # A_θθ
    axes[2, 1].plot(r_int, state_t0[idx_att, :], 'b-', linewidth=2, label=f't={t_0:.6e}')
    axes[2, 1].plot(r_int, state_tfinal[idx_att, :], 'r--', linewidth=1.7, label=f't={t_final:.6e}')
    axes[2, 1].set_xlabel('r')
    axes[2, 1].set_ylabel(r'$A_{\theta\theta}$')
    axes[2, 1].set_title(r'Traceless Extrinsic Curvature $A_{\theta\theta}$')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    # A_φφ
    axes[2, 2].plot(r_int, state_t0[idx_app, :], 'b-', linewidth=2, label=f't={t_0:.6e}')
    axes[2, 2].plot(r_int, state_tfinal[idx_app, :], 'r--', linewidth=1.7, label=f't={t_final:.6e}')
    axes[2, 2].set_xlabel('r')
    axes[2, 2].set_ylabel(r'$A_{\phi\phi}$')
    axes[2, 2].set_title(r'Traceless Extrinsic Curvature $A_{\phi\phi}$')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    plt.suptitle(f'BSSN Variables Evolution (Cowling Approximation)\nt={t_0:.6e} → t={t_final:.6e}',
                 fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'tov_bssn_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Compute and print maximum changes to verify Cowling approximation
    print("\n" + "="*70)
    print("BSSN VARIABLES VERIFICATION (Cowling Approximation)")
    print("="*70)
    print("Maximum absolute changes (should be ~0 for frozen spacetime):\n")

    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

    def max_change(var_idx, var_name):
        change = np.abs(state_tfinal[var_idx, interior] - state_t0[var_idx, interior])
        max_val = np.max(np.abs(state_t0[var_idx, interior]))
        max_abs_change = np.max(change)
        max_rel_change = max_abs_change / (max_val + 1e-30)
        print(f"  {var_name:10s}: max |Δ| = {max_abs_change:.6e}  max |Δ/val| = {max_rel_change:.6e}")
        return max_abs_change

    max_change(idx_phi, 'φ')
    max_change(idx_lapse, 'α')
    max_change(idx_K, 'K')
    max_change(idx_hrr, 'h_rr')
    max_change(idx_htt, 'h_θθ')
    max_change(idx_hpp, 'h_φφ')
    max_change(idx_arr, 'A_rr')
    max_change(idx_att, 'A_θθ')
    max_change(idx_app, 'A_φφ')

    # Verify spherical symmetry: h_θθ should equal h_φφ, A_θθ should equal A_φφ
    print("\nSpherical symmetry verification (should be ~0):")
    diff_h = np.abs(state_t0[idx_htt, interior] - state_t0[idx_hpp, interior])
    diff_A = np.abs(state_t0[idx_att, interior] - state_t0[idx_app, interior])
    print(f"  max |h_θθ - h_φφ| at t={t_0:.6e}: {np.max(diff_h):.6e}")
    print(f"  max |A_θθ - A_φφ| at t={t_0:.6e}: {np.max(diff_A):.6e}")

    diff_h_final = np.abs(state_tfinal[idx_htt, interior] - state_tfinal[idx_hpp, interior])
    diff_A_final = np.abs(state_tfinal[idx_att, interior] - state_tfinal[idx_app, interior])
    print(f"  max |h_θθ - h_φφ| at t={t_final:.6e}: {np.max(diff_h_final):.6e}")
    print(f"  max |A_θθ - A_φφ| at t={t_final:.6e}: {np.max(diff_A_final):.6e}")

    print("="*70 + "\n")


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


def plot_mass_and_central_density(times, Mb_series, rho_c_series, out_path):
    """Plot (1) log10 |M - M0| and (2) (ρ_c/ρ_c(0) - 1) vs time, Figure-12 style."""
    if len(times) == 0:
        return

    t = np.array(times)
    Mb = np.array(Mb_series)
    rho_c = np.array(rho_c_series)

    M0 = Mb[0]
    rho_c0 = rho_c[0]

    # Avoid log10(0)
    dM = np.abs(Mb - M0)
    dM = np.maximum(dM, 1e-16)
    log_dM = np.log10(dM)

    rel_rho_c = rho_c / (rho_c0 + 1e-30) - 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: log10 |M - M0|
    axes[0].plot(t, log_dM, color='tab:red', lw=1.5, label='Engrenage')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel(r'log10(|M - M$_0$|)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Right: ρ_c/ρ_c(0) - 1
    axes[1].plot(t, rel_rho_c, color='tab:green', lw=1.2, label='Engrenage')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel(r'$\rho_c/\rho_c(0) - 1$')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_evolution(state_t0, state_t1, state_t100, state_t10000, grid, hydro,
                   t_1, t_100, t_10000, label_100='t_2/3', label_10000='t_final',
                   times_series=None, Mb_series=None, rho_c_series=None):
    """Evolution plot with 6 panels showing snapshots at t=0, t=1/3, t=2/3, and t=final.

    Top 4 panels: (ρ, P, v^r, |Δρ|/ρ) at four evolution times
    Bottom row:
      - Left: log10(|M - M0|) vs t (if time series provided; else legacy ΔM/M0 sparse)
      - Right: ρ_c/ρ_c(0) - 1 vs t (if time series provided; else legacy L1 density error)
    """
    bssn_0 = BSSNVars(grid.N)
    bssn_0.set_bssn_vars(state_t0[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t0, bssn_0, grid)
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

    # Compute baryon mass for each snapshot
    def compute_baryon_mass(prim, state):
        return _compute_baryon_mass(grid, state, prim)

    M_b_0 = compute_baryon_mass(prim_0, state_t0)
    M_b_1 = compute_baryon_mass(prim_1, state_t1)
    M_b_100 = compute_baryon_mass(prim_100, state_t100)
    M_b_10000 = compute_baryon_mass(prim_10000, state_t10000)

    # Compute L1 norm of density error
    def compute_l1_error(rho_current, rho_initial):
        """Compute L1 norm of density error"""
        interior = slice(NUM_GHOSTS, grid.N - NUM_GHOSTS)
        return np.sum(np.abs(rho_current[interior] - rho_initial[interior])) / np.sum(np.abs(rho_initial[interior]))

    l1_1 = compute_l1_error(prim_1['rho0'], prim_0['rho0'])
    l1_100 = compute_l1_error(prim_100['rho0'], prim_0['rho0'])
    l1_10000 = compute_l1_error(prim_10000['rho0'], prim_0['rho0'])

    r_int = grid.r

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    # Density
    axes[0, 0].plot(r_int, prim_0['rho0'], 'b-', linewidth=2, label='t=0')
    axes[0, 0].plot(r_int, prim_1['rho0'], 'orange', linestyle='--', linewidth=1.5, label=f't=1/3 (t={t_1:.6e})')
    if t_100 != t_10000:  # Only plot if different from final
        axes[0, 0].plot(r_int, prim_100['rho0'], 'green', linestyle='-.', linewidth=1.5, label=f't=2/3 (t={t_100:.6e})')
    axes[0, 0].plot(r_int, prim_10000['rho0'], 'red', linestyle=':', linewidth=1.5, label=f't=final (t={t_10000:.6e})')
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel(r'$\rho_0$')
    axes[0, 0].set_title('Baryon Density Evolution')
    axes[0, 0].legend()
    #axes[0, 0].set_ylim(0.0, 0.002)
    axes[0, 0].grid(True, alpha=0.3)

    # Pressure
    axes[0, 1].plot(r_int, np.maximum(prim_0['p'], 1e-20), 'b-', linewidth=2, label='t=0')
    axes[0, 1].plot(r_int, np.maximum(prim_1['p'], 1e-20), 'orange', linestyle='--', linewidth=1.5, label=f't=1/3 (t={t_1:.6e})')
    if t_100 != t_10000:
        axes[0, 1].plot(r_int, np.maximum(prim_100['p'], 1e-20), 'green', linestyle='-.', linewidth=1.5, label=f't=2/3 (t={t_100:.6e})')
    axes[0, 1].plot(r_int, np.maximum(prim_10000['p'], 1e-20), 'red', linestyle=':', linewidth=1.5, label=f't=final (t={t_10000:.6e})')
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('P')
    axes[0, 1].set_title('Pressure Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Velocity
    axes[1, 0].plot(r_int, prim_0['vr'], 'b-', linewidth=2, label='t=0')
    axes[1, 0].plot(r_int, prim_1['vr'], 'orange', linestyle='--', linewidth=1.5, label=f't=1/3 (t={t_1:.6e})')
    if t_100 != t_10000:
        axes[1, 0].plot(r_int, prim_100['vr'], 'green', linestyle='-.', linewidth=1.5, label=f't=2/3 (t={t_100:.6e})')
    axes[1, 0].plot(r_int, prim_10000['vr'], 'red', linestyle=':', linewidth=1.5, label=f't=final (t={t_10000:.6e})')
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel(r'$v^r$')
    axes[1, 0].set_title('Radial Velocity Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Relative density error
    delta_rho_1 = np.abs(prim_1['rho0'] - prim_0['rho0']) / (np.abs(prim_0['rho0']) + 1e-20)
    delta_rho_100 = np.abs(prim_100['rho0'] - prim_0['rho0']) / (np.abs(prim_0['rho0']) + 1e-20)
    delta_rho_10000 = np.abs(prim_10000['rho0'] - prim_0['rho0']) / (np.abs(prim_0['rho0']) + 1e-20)

    axes[1, 1].semilogy(r_int, delta_rho_1, 'orange', linestyle='-', linewidth=1.5, label=f't=1/3 (t={t_1:.6e})')
    if t_100 != t_10000:
        axes[1, 1].semilogy(r_int, delta_rho_100, 'green', linestyle='-', linewidth=1.5, label=f't=2/3 (t={t_100:.6e})')
    axes[1, 1].semilogy(r_int, delta_rho_10000, 'red', linestyle='-', linewidth=1.5, label=f't=final (t={t_10000:.6e})')
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel(r'$|\Delta\rho|/\rho$')
    axes[1, 1].set_title('Relative Density Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Bottom-left: Mass deviation (preferred: time series)
    if times_series is not None and Mb_series is not None and len(times_series) > 1:
        t_arr = np.array(times_series)
        Mb_arr = np.array(Mb_series)
        dM = np.maximum(np.abs(Mb_arr - Mb_arr[0]), 1e-16)
        axes[2, 0].plot(t_arr, np.log10(dM), color='tab:red', lw=1.4)
        axes[2, 0].set_ylabel(r'log10(|M - M$_0$|)', fontsize=11)
        axes[2, 0].set_xlabel('t', fontsize=11)
        axes[2, 0].set_title('Total Rest Mass Deviation', fontsize=12)
        axes[2, 0].grid(True, alpha=0.3)
    else:
        # Legacy sparse ΔM/M0 points
        times_sparse = [0, t_1, t_100, t_10000]
        M_b_values = [M_b_0, M_b_1, M_b_100, M_b_10000]
        M_b_drift = [(M - M_b_0) / (M_b_0 + 1e-30) for M in M_b_values]
        axes[2, 0].plot(times_sparse, M_b_drift, 'bo-', linewidth=2, markersize=6)
        axes[2, 0].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes[2, 0].set_xlabel('Time', fontsize=11)
        axes[2, 0].set_ylabel(r'$\Delta M_b / M_b$', fontsize=11)
        axes[2, 0].set_title(f'Baryon Mass Conservation (M_b₀={M_b_0:.6f})', fontsize=12, fontweight='bold')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))

    # Bottom-right: Central density fractional change (preferred: time series)
    if times_series is not None and rho_c_series is not None and len(times_series) > 1:
        t_arr = np.array(times_series)
        rho_c_arr = np.array(rho_c_series)
        rel = rho_c_arr / (rho_c_arr[0] + 1e-30) - 1.0
        axes[2, 1].plot(t_arr, rel, color='tab:green', lw=1.2)
        axes[2, 1].set_xlabel('t', fontsize=11)
        axes[2, 1].set_ylabel(r'$\rho_c/\rho_c(0) - 1$', fontsize=11)
        axes[2, 1].set_title('Central Density Deviation', fontsize=12)
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))
    else:
        # Legacy L1 density error vs time (sparse)
        times = [0, t_1, t_100, t_10000]
        l1_values = [0, l1_1, l1_100, l1_10000]
        axes[2, 1].semilogy(times, l1_values, 'ro-', linewidth=2, markersize=8)
        axes[2, 1].set_xlabel('Time', fontsize=11)
        axes[2, 1].set_ylabel(r'$\|\Delta\rho\|_1 / \|\rho_0\|_1$', fontsize=11)
        axes[2, 1].set_title('L1 Density Error (vs initial state)', fontsize=12, fontweight='bold')
        axes[2, 1].grid(True, alpha=0.3, which='both')

    plt.suptitle(f'Evolution: t=0 → t=1/3 ({t_1:.6e}) → t=2/3 ({t_100:.6e}) → t=final ({t_10000:.6e})', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'tov_evolution.png'), dpi=150, bbox_inches='tight')
    plt.show()
    # plt.close(fig)


def plot_hamiltonian_constraint_evolution(snapshot_file, evolution_file, grid, output_dir=plots_dir):
    """
    Plot Hamiltonian constraint evolution from HDF5 data files.

    Creates a figure with:
    - Left panel: Ham(r) spatial profiles at multiple times
    - Right panel: Time series of max|Ham| and L2(Ham)

    Args:
        snapshot_file: Path to snapshot HDF5 file
        evolution_file: Path to evolution HDF5 file
        grid: Grid object
        output_dir: Directory to save plot
    """
    print("\nGenerating Hamiltonian constraint evolution plot...")

    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
    r_interior = grid.r[interior]

    # Load snapshots with Ham data
    snapshot_times = []
    Ham_profiles = []

    with h5py.File(snapshot_file, 'r') as f:
        snap_group = f['snapshots']
        step_names = sorted(snap_group.keys())

        # Sample evenly across available snapshots (max 5 profiles)
        n_samples = min(5, len(step_names))
        indices = np.linspace(0, len(step_names)-1, n_samples, dtype=int)

        for idx in indices:
            step_name = step_names[idx]
            snap = snap_group[step_name]

            if 'constraints' in snap.keys() and 'Ham' in snap['constraints'].keys():
                t = snap.attrs['time']
                Ham = snap['constraints']['Ham'][:]
                snapshot_times.append(t)
                Ham_profiles.append(Ham)
            else:
                print(f"Warning: No Ham data in {step_name}")

    # Load evolution time series
    evolution_times = []
    max_Ham_series = []
    l2_Ham_series = []

    try:
        with h5py.File(evolution_file, 'r') as f:
            if 'max_Ham' in f.keys() and 'l2_Ham' in f.keys():
                evolution_times = f['time'][:]
                max_Ham_series = f['max_Ham'][:]
                l2_Ham_series = f['l2_Ham'][:]
            else:
                print("Warning: No Ham time series in evolution file")
    except Exception as e:
        print(f"Warning: Could not load evolution file: {e}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Ham(r) profiles
    if Ham_profiles:
        colors = plt.cm.viridis(np.linspace(0, 1, len(Ham_profiles)))
        for i, (t, Ham) in enumerate(zip(snapshot_times, Ham_profiles)):
            axes[0].plot(r_interior, Ham[interior],
                        color=colors[i], linewidth=1.5,
                        label=f't={t:.3e}')

        axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[0].set_xlabel('r', fontsize=12)
        axes[0].set_ylabel('Ham', fontsize=12)
        axes[0].set_title('Hamiltonian Constraint Spatial Profile', fontsize=13, fontweight='bold')
        axes[0].legend(loc='best', fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Use symlog scale if values span multiple orders of magnitude
        Ham_all = np.concatenate([H[interior] for H in Ham_profiles])
        Ham_range = np.ptp(Ham_all)
        if Ham_range > 0 and np.max(np.abs(Ham_all)) / np.min(np.abs(Ham_all[Ham_all != 0]) + 1e-30) > 100:
            axes[0].set_yscale('symlog', linthresh=1e-10)
    else:
        axes[0].text(0.5, 0.5, 'No Ham snapshots available',
                    ha='center', va='center', transform=axes[0].transAxes,
                    fontsize=12)
        axes[0].set_xlabel('r')
        axes[0].set_ylabel('Ham')
        axes[0].set_title('Hamiltonian Constraint Spatial Profile', fontsize=13, fontweight='bold')

    # Right panel: Time series
    if len(evolution_times) > 0:
        ax2 = axes[1]
        ax2_twin = ax2.twinx()

        # Plot max|Ham| on left y-axis
        line1 = ax2.semilogy(evolution_times, np.maximum(max_Ham_series, 1e-16),
                             'b-', linewidth=1.5, label='max|Ham|')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('max|Ham|', fontsize=12, color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.grid(True, alpha=0.3, which='both')

        # Plot L2(Ham) on right y-axis
        line2 = ax2_twin.semilogy(evolution_times, np.maximum(l2_Ham_series, 1e-16),
                                  'r-', linewidth=1.5, label='L2(Ham)')
        ax2_twin.set_ylabel('L2(Ham)', fontsize=12, color='r')
        ax2_twin.tick_params(axis='y', labelcolor='r')

        # Combined legend
        lines = [line1[0], line2[0]]
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left', fontsize=10)

        axes[1].set_title('Constraint Violation vs Time', fontsize=13, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'No Ham time series available',
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=12)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Ham metrics')
        axes[1].set_title('Constraint Violation vs Time', fontsize=13, fontweight='bold')

    plt.suptitle('Hamiltonian Constraint Evolution', fontsize=15, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'tov_hamiltonian_constraint.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()
    # plt.close(fig)
