import numpy as np
import sys
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Engrenage core - add parent directories to path
sys.path.insert(0, os.path.join(script_dir, '..', '..'))
from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground

# BSSN
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS
from source.bssn.tensoralgebra import get_bar_gamma_LL

# Hydro
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver#, HLLCRiemannSolver
from source.matter.hydro.cons2prim import prim_to_cons
from source.matter.hydro.atmosphere import AtmosphereParams

# Local TOV modules
# Schwarzschild coordinates
from examples.TOV.tov_solver import load_or_solve_tov
import examples.TOV.tov_initial_data_interpolated as tov_id_schw

# Isotropic coordinates
from examples.TOV.tov_solver_iso import load_or_solve_tov_iso
import examples.TOV.tov_initial_data_interpolated_iso as tov_id_iso

# TOV utilities and plotting
import examples.TOV.utils_TOVEvolution as utils
from examples.TOV.utils_TOVEvolution import SimulationDataManager
from examples.TOV.plot_TOVEvolution import compute_qnm_spectrum, extract_decay_rate



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


def _apply_atmosphere_reset(state_2d, grid, hydro, atmosphere, rho_threshold=None):
    """
    Aplica floors atmosféricos y correcciones a variables conservativas.
    """
    from source.matter.hydro.atmosphere import FloorApplicator

    # Definir umbral para reset atmosférico
    if rho_threshold is None:
        rho_hard_floor = 10.0 * atmosphere.rho_floor

    # Indices de variables conservativas
    idx_D = NUM_BSSN_VARS + 0
    idx_Sr = NUM_BSSN_VARS + 1
    idx_tau = NUM_BSSN_VARS + 2

    # Extraer vistas para lectura
    D = state_2d[idx_D, :]

    # --- HARD ATMOSPHERE RESET ---
    # Resetear valores muy bajos al piso absoluto
    atm_mask = D < rho_hard_floor

    if np.any(atm_mask):
        state_2d[idx_D, atm_mask] = atmosphere.rho_floor
        state_2d[idx_Sr, atm_mask] = 0.0
        state_2d[idx_tau, atm_mask] = atmosphere.tau_atm
        
        # Actualizar la variable local D para la siguiente fase
        D[atm_mask] = atmosphere.rho_floor

    # Si todo es atmósfera, terminamos
    if np.all(atm_mask):
        return state_2d

    # --- STANDARD FLOORS (Illinois Strategy) ---
    # Solo para el interior de la estrella donde la densidad es alta
    non_atm_mask = ~atm_mask

    if not np.any(non_atm_mask):
        return state_2d

    # Necesitamos métrica para los floors físicos
    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
    
    # Calcular gamma_rr físico
    bar_gamma_LL = get_bar_gamma_LL(grid.r, bssn_vars.h_LL, hydro.background)
    phi = np.asarray(bssn_vars.phi, dtype=float)
    e4phi = np.exp(4.0 * phi)
    gamma_rr = e4phi * bar_gamma_LL[:, 0, 0]

    # Recuperar primitivas solo donde importa
    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    rho0, vr, p, eps, W, h, success = hydro._get_primitives(bssn_vars, grid.r)

    floor_app = FloorApplicator(atmosphere, hydro.eos)

    # Aplicar floors a primitivas (rho, p, v)
    rho0_floor, vr_floor, p_floor = floor_app.apply_primitive_floors(rho0, vr, p, gamma_rr)

    # Aplicar consistencia a conservativas (Tau floor, S^2 limit)
    # Usamos las conservativas actuales del state_2d
    D_curr = state_2d[idx_D, :]
    Sr_curr = state_2d[idx_Sr, :]
    tau_curr = state_2d[idx_tau, :]

    D_floor, Sr_floor, tau_floor, cons_floor_applied = floor_app.apply_conservative_floors(
        D_curr, Sr_curr, tau_curr, gamma_rr
    )

    # Detectar dónde se aplicaron cambios
    prim_floor_applied = (
        (np.abs(rho0_floor - rho0) > 1e-14) |
        (np.abs(vr_floor - vr) > 1e-14) |
        (np.abs(p_floor - p) > 1e-14)
    )

    # Combinar: aplicar recomputo solo en celdas NO atmósfera que violaron floors
    needs_recompute = prim_floor_applied & non_atm_mask
    needs_cons_fix = cons_floor_applied & non_atm_mask

    # 1. Si primitivas cambiaron, recomputar conservativas
    if np.any(needs_recompute):
        D_new, Sr_new, tau_new = prim_to_cons(
            rho0_floor, vr_floor, p_floor, gamma_rr, hydro.eos, 
            e6phi=np.exp(6.0*phi), alpha=bssn_vars.lapse
        )
        state_2d[idx_D, needs_recompute] = D_new[needs_recompute]
        state_2d[idx_Sr, needs_recompute] = Sr_new[needs_recompute]
        state_2d[idx_tau, needs_recompute] = tau_new[needs_recompute]

    # 2. Si conservativas violaban reglas físicas, aplicar corrección directa
    if np.any(needs_cons_fix):
        # Aplicar los valores corregidos por apply_conservative_floors
        state_2d[idx_D, needs_cons_fix] = D_floor[needs_cons_fix]
        state_2d[idx_Sr, needs_cons_fix] = Sr_floor[needs_cons_fix]
        state_2d[idx_tau, needs_cons_fix] = tau_floor[needs_cons_fix]

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
    # APLICAR FLOOR AQUI
    #s2 = _apply_atmosphere_reset(s2, grid, hydro, atmosphere)
    state_2 = s2.flatten()

    k2 = get_rhs_cowling(0, state_2, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    # --- Stage 3 ---
    state_3 = state_flat + 0.5 * dt * k2
    s3 = state_3.reshape((grid.NUM_VARS, grid.N))
    # APLICAR FLOOR AQUI
    #s3 = _apply_atmosphere_reset(s3, grid, hydro, atmosphere)
    state_3 = s3.flatten()

    k3 = get_rhs_cowling(0, state_3, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    # --- Stage 4 ---
    state_4 = state_flat + dt * k3
    s4 = state_4.reshape((grid.NUM_VARS, grid.N))
    # APLICAR FLOOR AQUI
    #s4 = _apply_atmosphere_reset(s4, grid, hydro, atmosphere)
    state_4 = s4.flatten()

    k4 = get_rhs_cowling(0, state_4, grid, background, hydro, bssn_fixed, bssn_d1_fixed)

    # --- Combine ---
    state_new = state_flat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    snew = state_new.reshape((grid.NUM_VARS, grid.N))

    # Apply atmosphere reset after full step
    snew_reset = _apply_atmosphere_reset(snew, grid, hydro, atmosphere)

    return snew_reset.flatten()


def evolve_fixed_timestep(state_initial, dt, num_steps, grid, background, hydro,
                          bssn_fixed, bssn_d1_fixed, atmosphere, method='rk4', t_start=0.0,
                          reference_state=None, step_offset=0, data_manager=None,
                          snapshot_interval=None, evolution_interval=None):
    """Evolve with fixed timestep using RK4.

    Args:
        atmosphere: AtmosphereParams object
        t_start: Starting time for this evolution segment (default: 0.0)
        reference_state: Reference state for error calculation (default: state_initial)
                        Use this to maintain consistent error measurement across multiple segments
        step_offset: Offset for step numbering in output (default: 0)
        data_manager: SimulationDataManager object for data saving (optional)
        snapshot_interval: Save full domain snapshot every N steps (optional)
        evolution_interval: Save evolution data every N steps (optional)
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
        data_manager.save_snapshot(step_offset, t_start, state_initial, rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial)
        data_manager.add_evolution_point(step_offset, t_start, state_initial,
                                        rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial, success_initial,
                                        rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial, success_initial)

    # Timeseries for mass and central density
    times_series = [t_start]
    Mb0 = utils.compute_baryon_mass(grid, s_initial, rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial)
    Mb_series = [Mb0]
    rho_c0 = rho0_initial[NUM_GHOSTS]
    rho_c_series = [rho_c0]


    print("\n===== Evolution diagnostics (per step) =====")
    print("Columns: step | t | ρ_central | max_Δρ/ρ@r | max_vʳ@r | max_D@r | max_Sʳ@r | c2p_fails")
    print("  (@r indicates the radial position where the maximum occurs)")
    print("-" * 140)

    for step in range(num_steps):
        # Advance one RK4 step (with well-balanced correction)
        state_flat_next = rk4_step(state_flat, dt, grid, background, hydro,
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
        max_abs_v = float(np.abs(v_next_int[idx_max_v]))
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
        if step_num % 200 == 0:
            print(f"step {step_num:4d}  t={t_curr:.2e}:  ρ_c={rho_central:.6e}  max_Δρ/ρ={max_rel_rho_err:.2e}@r={r_max_rho_err:.2f}  "
              f"max_vʳ={max_abs_v:.3e}@r={r_max_v:.2f}  max_D={max_D:.2e}@r={r_max_D:.2f}  max_Sʳ={max_Sr:.2e}@r={r_max_Sr:.2f}  "
              f"c2p_fail={c2p_fail_count}")

        # Save data if requested
        if data_manager and data_manager.enable_saving:
            # Save evolution data at specified interval
            if evolution_interval and step_num % evolution_interval == 0:
                data_manager.add_evolution_point(step_num, t_curr, s_next,
                                                rho0_next, vr_next, p_next, eps_next, W_next, h_next, success_next,
                                                rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial, success_initial)

                # Periodic buffer flush
                if step_num % (evolution_interval * 10) == 0:
                    data_manager.flush_evolution_buffer()

            # Save full snapshot at specified interval
            if snapshot_interval and step_num % snapshot_interval == 0:
                data_manager.save_snapshot(step_num, t_curr, s_next, rho0_next, vr_next, p_next, eps_next, W_next, h_next)

        # Append to time series (mass and central density)
        Mb_next = utils.compute_baryon_mass(grid, s_next, rho0_next, vr_next, p_next, eps_next, W_next, h_next)
        times_series.append(t_curr)
        Mb_series.append(Mb_next)
        rho_c_series.append(float(rho0_next[NUM_GHOSTS]))

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
    }


def main():
    """Main execution."""
    print("="*70)
    print("TOV Star Evolution - Cowling Approximation")
    print("="*70)

    # ==================================================================
    # CONFIGURATION
    # ==================================================================
    r_max = 16.0
    num_points = 300
    K = 100.0
    Gamma = 2.0
    rho_central = 1.28e-3
    num_steps_total = 10000

    # ==================================================================
    # COORDINATE SYSTEM SELECTION
    # ==================================================================
    # Options: "isotropic" or "schwarzschild"
    #   - "isotropic": Uses isotropic coordinates where spatial metric is
    #                  conformally flat: γ_ij = e^{4φ} ĝ_ij, h_ij = 0
    #   - "schwarzschild": Uses Schwarzschild-like coordinates (standard TOV)
    COORDINATE_SYSTEM = "isotropic"  # Change to "schwarzschild" for Schwarzschild coords

    # Select appropriate modules based on coordinate system
    if COORDINATE_SYSTEM == "isotropic":
        tov_id = tov_id_iso
        coord_suffix = "_iso"
        print("Using ISOTROPIC coordinates (conformally flat spatial metric)")
    elif COORDINATE_SYSTEM == "schwarzschild":
        tov_id = tov_id_schw
        coord_suffix = ""
        print("Using SCHWARZSCHILD coordinates (standard TOV)")
    else:
        raise ValueError(f"Unknown coordinate system: {COORDINATE_SYSTEM}. Use 'isotropic' or 'schwarzschild'")


    # ==================================================================
    # DATA SAVING CONFIGURATION
    # ==================================================================
    ENABLE_DATA_SAVING = True  # Set to True to save data to files
    OUTPUT_DIR = os.path.join(script_dir, "tov_evolution_data")  # Directory for output data
    SNAPSHOT_INTERVAL = 100  # Save full domain every N timesteps (None to disable)
    EVOLUTION_INTERVAL = 100  # Save time series every N timesteps (None to disable)

    if ENABLE_DATA_SAVING:
        print(f"Data saving enabled:")
        print(f"  Output directory: {OUTPUT_DIR}")
        print(f"  Snapshot interval: {SNAPSHOT_INTERVAL} timesteps")
        print(f"  Evolution tracking: {EVOLUTION_INTERVAL} timesteps")
        print()

    # ==================================================================
    # ATMOSPHERE CONFIGURATION (Centralized floor management)
    # ==================================================================
    # Define atmosphere parameters ONCE - all subsystems will use these
    ATMOSPHERE = AtmosphereParams(
        rho_floor=1.0e-10,  # Rest mass density floor
        p_floor=K*(1.0e-10)**Gamma,     # Pressure floor
        v_max=0.999,           # Maximum velocity
        W_max=100.0,            # Maximum Lorentz factor
        conservative_floor_safety=0.999  # Safety factor for S^2 constraint
    )

    print("=" * 70)
    print("ATMOSPHERE CONFIGURATION")
    print("=" * 70)
    print(f"  rho_floor = {ATMOSPHERE.rho_floor:.2e}")
    print(f"  p_floor   = {ATMOSPHERE.p_floor:.2e}")
    print(f"  tau_atm   = {ATMOSPHERE.tau_atm:.2e}")
    print(f"  v_max     = {ATMOSPHERE.v_max}")
    print()

    # Time integration method
    # 'fixed': RK4 with fixed timestep 
    integration_method = 'fixed'  # 'fixed' 

    # ==================================================================
    # SETUP
    # ==================================================================
    spacing = LinearSpacing(num_points, r_max)
    eos = IdealGasEOS(gamma=Gamma)
    
    # 1. RECONSTRUCTOR BASE (Usa WENO-Z, es más preciso que minmod para el interior)
    base_recon = create_reconstruction("wenoz")
    
    # 3. PASAR EL WRAPPER 
    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere=ATMOSPHERE,
        reconstructor=base_recon,
        riemann_solver=HLLRiemannSolver()
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
    print("Solving TOV equations...")

    if COORDINATE_SYSTEM == "isotropic":
        # Isotropic coordinates: use cached solver
        tov_solution = load_or_solve_tov_iso(
            K=K, Gamma=Gamma, rho_central=rho_central,
            r_max=r_max, accuracy="high"
        )
        print(f"TOV Solution (ISO): M={tov_solution.M_star:.6f}, R_iso={tov_solution.R_iso:.3f}, R_schw={tov_solution.R_schw:.3f}, C={tov_solution.C:.4f}\n")
    else:
        # Schwarzschild coordinates: use cached solver
        tov_solution = load_or_solve_tov(
            K=K, Gamma=Gamma, rho_central=rho_central,
            r_max=r_max, accuracy="high"
        )
        print(f"TOV Solution (SCHW): M={tov_solution.M_star:.6f}, R={tov_solution.R:.3f}, C={tov_solution.C:.4f}\n")

    utils.plot_tov_diagnostics(tov_solution, r_max, suffix=coord_suffix)

    # ==================================================================
    # INITIAL DATA (HIGH-ORDER INTERPOLATION)
    # ==================================================================
    print("Creating initial data from TOV solution...")
    # 1. Interpolate ρ, P ONLY up to stellar radius R
    # 2. Outside R: use atmosphere values directly (no interpolation)
    # 3. Interpolate geometry (α, exp4φ) everywhere
    # 4. Stencil NEVER crosses the stellar surface (avoids Gibbs phenomenon)

    if COORDINATE_SYSTEM == "isotropic":
        # Isotropic: use create_initial_data_iso
        initial_state_2d, prim_tuple = tov_id.create_initial_data_iso(
            tov_solution, grid, background, eos,
            atmosphere=ATMOSPHERE,
            polytrope_K=K, polytrope_Gamma=Gamma,
            interp_order=11
        )
    else:
        # Schwarzschild: use create_initial_data_interpolated
        initial_state_2d, prim_tuple = tov_id.create_initial_data_interpolated(
            tov_solution, grid, background, eos,
            atmosphere=ATMOSPHERE,
            polytrope_K=K, polytrope_Gamma=Gamma,
            interp_order=11
        )

    # Diagnostics: check discrete hydrostatic balance at t=0
    utils.diagnose_t0_residuals(initial_state_2d, grid, background, hydro)

    # Initial-data diagnostics 
    tov_id.plot_initial_comparison(tov_solution, initial_state_2d, grid, prim_tuple,
                                   output_dir=plots_dir, suffix=coord_suffix)

    # Zoom comparison: TOV solution vs interpolated initial data 
    utils.plot_tov_vs_initial_data_zoom(tov_solution, initial_state_2d, grid, prim_tuple,
                                        window=0.1, suffix=coord_suffix)

    # ==================================================================
    # EVOLUTION
    # ==================================================================
    bssn_fixed = initial_state_2d[:NUM_BSSN_VARS, :].copy()
    bssn_d1_fixed = grid.get_d1_metric_quantities(initial_state_2d)

    # Initialize data manager for saving
    data_manager = SimulationDataManager(OUTPUT_DIR, grid, hydro,
                                        enable_saving=ENABLE_DATA_SAVING,
                                        suffix=coord_suffix)


    if integration_method == 'fixed':
        dt = 0.1 * grid.min_dr  # CFL condition
        print(f"\nEvolving with fixed dt={dt:.6f} (CFL=0.1) for {num_steps_total} steps using RK4")

        # Save metadata now that we have dt
        if ENABLE_DATA_SAVING:
            data_manager.save_metadata(tov_solution, ATMOSPHERE, dt, integration_method, K=K, Gamma=Gamma, rho_central=rho_central)

        # Single step for comparison
        state_t1 = rk4_step(initial_state_2d.flatten(), dt, grid, background, hydro,
                           bssn_fixed, bssn_d1_fixed, ATMOSPHERE).reshape((grid.NUM_VARS, grid.N))
        t_1 = dt

        # Plot only the first step changes
        utils.plot_first_step(initial_state_2d, state_t1, grid, hydro, tov_solution,
                             suffix=coord_suffix)
        utils.plot_surface_zoom(tov_solution, initial_state_2d, state_t1, grid, hydro,
                               primitives_t0=prim_tuple, window=0.1, suffix=coord_suffix)
        utils.plot_center_zoom(initial_state_2d, state_t1, grid, hydro,
                              window=0.5, suffix=coord_suffix)

        # Define checkpoints at 1/3, 2/3, and final of total steps
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

        # Storage for checkpoint states
        checkpoint_states = {}
        checkpoint_times = {}
        all_series = []

        # Evolve to checkpoint 1
        print(f"Evolving to checkpoint 1 (step {checkpoint_1})...")
        state_cp1, steps_cp1, t_cp1, series_1 = evolve_fixed_timestep(
            initial_state_2d, dt, checkpoint_1, grid, background,
            hydro, bssn_fixed, bssn_d1_fixed, ATMOSPHERE, method='rk4',
            reference_state=initial_state_2d,
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
                hydro, bssn_fixed, bssn_d1_fixed, ATMOSPHERE, method='rk4',
                t_start=t_cp1, reference_state=initial_state_2d,
                step_offset=checkpoint_1,
                data_manager=data_manager,
                snapshot_interval=SNAPSHOT_INTERVAL,
                evolution_interval=EVOLUTION_INTERVAL)
            checkpoint_states[2] = state_cp2.copy()
            checkpoint_times[2] = t_cp2
            all_series.append(series_2)
            print(f"  -> Reached step {checkpoint_1 + steps_cp2}, t={t_cp2:.6e}")
        else:
            # Stopped early
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
                hydro, bssn_fixed, bssn_d1_fixed, ATMOSPHERE, method='rk4',
                t_start=t_cp2, reference_state=initial_state_2d,
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
            # Stopped early
            state_cp3 = state_cp2
            t_cp3 = t_cp2
            checkpoint_states[3] = state_cp3.copy()
            checkpoint_times[3] = t_cp3
            steps_final = checkpoint_1 + steps_cp2

        # Assign final state for backward compatibility
        state_t10000 = checkpoint_states[3]
        t_10000 = checkpoint_times[3]
        num_steps = steps_final

        # Build full-series arrays for mass and central density
        try:
            if len(all_series) == 3:
                # All three segments completed
                times_full = np.concatenate([all_series[0]['t'], all_series[1]['t'][1:], all_series[2]['t'][1:]])
                Mb_full = np.concatenate([all_series[0]['Mb'], all_series[1]['Mb'][1:], all_series[2]['Mb'][1:]])
                rho_c_full = np.concatenate([all_series[0]['rho_c'], all_series[1]['rho_c'][1:], all_series[2]['rho_c'][1:]])
            elif len(all_series) == 2:
                # Only first two segments completed
                times_full = np.concatenate([all_series[0]['t'], all_series[1]['t'][1:]])
                Mb_full = np.concatenate([all_series[0]['Mb'], all_series[1]['Mb'][1:]])
                rho_c_full = np.concatenate([all_series[0]['rho_c'], all_series[1]['rho_c'][1:]])
            elif len(all_series) == 1:
                # Only first segment completed
                times_full = all_series[0]['t']
                Mb_full = all_series[0]['Mb']
                rho_c_full = all_series[0]['rho_c']
            else:
                times_full = np.array([])
                Mb_full = np.array([])
                rho_c_full = np.array([])
        except Exception as e:
            print(f"Warning: Failed to concatenate series: {e}")
            times_full = np.array([])
            Mb_full = np.array([])
            rho_c_full = np.array([])

    # ==================================================================
    # DIAGNOSTICS
    # ==================================================================
    # Combined 3x2 figure: keep 4 panels and replace bottom row with the two new time-series
    # Get reference primitives for error computation
    bssn_ref = BSSNVars(grid.N)
    bssn_ref.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(initial_state_2d, bssn_ref, grid)
    rho_ref, _, p_ref, _, _, _, _ = hydro._get_primitives(bssn_ref, grid.r)

    # Build states and times lists for plot_evolution
    # Use checkpoints: t=0, 1/3, 2/3, final (uniformly distributed)
    states = [initial_state_2d, checkpoint_states[1], checkpoint_states[2], checkpoint_states[3]]
    times = [0.0, checkpoint_times[1], checkpoint_times[2], checkpoint_times[3]]

    try:
        if 'times_full' in locals() and len(times_full) > 0:
            utils.plot_evolution(states, times, grid, hydro, rho_ref, p_ref,
                                 Mb_series=Mb_full, rho_c_series=rho_c_full,
                                 suffix=coord_suffix)
        else:
            utils.plot_evolution(states, times, grid, hydro, rho_ref, p_ref,
                                suffix=coord_suffix)
    except Exception:
        utils.plot_evolution(states, times, grid, hydro, rho_ref, p_ref,
                            suffix=coord_suffix)

    # Plot BSSN variables evolution to verify Cowling approximation
    utils.plot_bssn_evolution(initial_state_2d, checkpoint_states[3], grid,
                             t_0=0.0, t_final=checkpoint_times[3],
                             suffix=coord_suffix)

    # Print detailed statistics - extract primitives for each state
    bssn_0 = BSSNVars(grid.N)
    bssn_0.set_bssn_vars(initial_state_2d[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(initial_state_2d, bssn_0, grid)
    rho0_0, vr_0, p_0, _, _, _, success_0 = hydro._get_primitives(bssn_0, grid.r)

    bssn_1 = BSSNVars(grid.N)
    bssn_1.set_bssn_vars(state_t1[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state_t1, bssn_1, grid)
    rho0_1, vr_1, p_1, _, _, _, success_1 = hydro._get_primitives(bssn_1, grid.r)

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

    # Interior points only (exclude ghosts)
    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

    # Compute errors at checkpoints
    delta_rho_1 = np.abs(rho0_1[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)
    delta_rho_cp1 = np.abs(rho0_cp1[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)
    delta_rho_cp2 = np.abs(rho0_cp2[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)
    delta_rho_cp3 = np.abs(rho0_cp3[interior] - rho0_0[interior]) / (np.abs(rho0_0[interior]) + 1e-20)

    delta_P_1 = np.abs(p_1[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)
    delta_P_cp1 = np.abs(p_cp1[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)
    delta_P_cp2 = np.abs(p_cp2[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)
    delta_P_cp3 = np.abs(p_cp3[interior] - p_0[interior]) / (np.abs(p_0[interior]) + 1e-20)

    # Error growth factor
    max_err_rho_1 = np.max(delta_rho_1)
    max_err_rho_cp1 = np.max(delta_rho_cp1)
    max_err_rho_cp2 = np.max(delta_rho_cp2)
    max_err_rho_cp3 = np.max(delta_rho_cp3)
    growth_rho = max_err_rho_cp3 / max_err_rho_1 if max_err_rho_1 > 1e-15 else 0

    max_err_P_1 = np.max(delta_P_1)
    max_err_P_cp1 = np.max(delta_P_cp1)
    max_err_P_cp2 = np.max(delta_P_cp2)
    max_err_P_cp3 = np.max(delta_P_cp3)
    growth_P = max_err_P_cp3 / max_err_P_1 if max_err_P_1 > 1e-15 else 0

    t_cp1 = checkpoint_times[1]
    t_cp2 = checkpoint_times[2]
    t_cp3 = checkpoint_times[3]

    print(f"\n{'='*70}")
    print(f"EVOLUTION DIAGNOSTICS")
    print(f"  t=0 → t={t_cp1:.6e} (1/3) → t={t_cp2:.6e} (2/3) → t={t_cp3:.6e} (final)")
    print(f"  (first step: t={t_1:.6e}, included for diagnostics)")
    print(f"{'='*70}")

    print(f"\n1. VELOCITY EVOLUTION:")
    print(f"   Max |v^r| at t=0:              {np.max(np.abs(vr_0)):.3e}")
    print(f"   Max |v^r| at t={t_1:.6e}:    {np.max(np.abs(vr_1)):.3e}")
    print(f"   Max |v^r| at t={t_cp1:.6e} (1/3):  {np.max(np.abs(vr_cp1)):.3e}")
    print(f"   Max |v^r| at t={t_cp2:.6e} (2/3):  {np.max(np.abs(vr_cp2)):.3e}")
    print(f"   Max |v^r| at t={t_cp3:.6e} (final): {np.max(np.abs(vr_cp3)):.3e}")

    print(f"\n2. CENTRAL DENSITY:")
    print(f"   ρ_c at t=0:                  {rho0_0[NUM_GHOSTS]:.6e}")
    print(f"   ρ_c at t={t_1:.6e}:    {rho0_1[NUM_GHOSTS]:.6e}")
    print(f"   ρ_c at t={t_cp1:.6e} (1/3):  {rho0_cp1[NUM_GHOSTS]:.6e}")
    print(f"   ρ_c at t={t_cp2:.6e} (2/3):  {rho0_cp2[NUM_GHOSTS]:.6e}")
    print(f"   ρ_c at t={t_cp3:.6e} (final): {rho0_cp3[NUM_GHOSTS]:.6e}")
    print(f"   Δρ_c/ρ_c (first step):       {abs(rho0_1[NUM_GHOSTS] - rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")
    print(f"   Δρ_c/ρ_c (1/3):              {abs(rho0_cp1[NUM_GHOSTS] - rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")
    print(f"   Δρ_c/ρ_c (2/3):              {abs(rho0_cp2[NUM_GHOSTS] - rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")
    print(f"   Δρ_c/ρ_c (final):            {abs(rho0_cp3[NUM_GHOSTS] - rho0_0[NUM_GHOSTS])/rho0_0[NUM_GHOSTS]:.3e}")

    print(f"\n3. DENSITY ERROR (max over domain):")
    print(f"   Max |Δρ|/ρ at t={t_1:.6e}:     {max_err_rho_1:.3e}")
    print(f"   Max |Δρ|/ρ at t={t_cp1:.6e} (1/3):   {max_err_rho_cp1:.3e}")
    print(f"   Max |Δρ|/ρ at t={t_cp2:.6e} (2/3):   {max_err_rho_cp2:.3e}")
    print(f"   Max |Δρ|/ρ at t={t_cp3:.6e} (final):  {max_err_rho_cp3:.3e}")
    print(f"   Growth factor (final/first): {growth_rho:.1f}x")

    print(f"\n4. PRESSURE ERROR (max over domain):")
    print(f"   Max |ΔP|/P at t={t_1:.6e}:     {max_err_P_1:.3e}")
    print(f"   Max |ΔP|/P at t={t_cp1:.6e} (1/3):   {max_err_P_cp1:.3e}")
    print(f"   Max |ΔP|/P at t={t_cp2:.6e} (2/3):   {max_err_P_cp2:.3e}")
    print(f"   Max |ΔP|/P at t={t_cp3:.6e} (final):  {max_err_P_cp3:.3e}")
    print(f"   Growth factor (final/first): {growth_P:.1f}x")

    print(f"\n5. CONS2PRIM STATUS:")
    print(f"   Success at t=0:                {np.sum(success_0)}/{grid.N}")
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
        data_manager.finalize()

    # ==================================================================
    # QUASI-NORMAL MODE ANALYSIS
    # ==================================================================
    # Perform QNM frequency analysis on the full time series
    if 'times_full' in locals() and 'rho_c_full' in locals() and len(times_full) > 100:
        print("\n" + "="*70)
        print("PERFORMING QUASI-NORMAL MODE ANALYSIS")
        print("="*70)

        try:
            # 1. Compute QNM spectrum
            freqs_kHz, psd, peaks = compute_qnm_spectrum(times_full, rho_c_full,
                                                         output_dir=plots_dir,
                                                         suffix=coord_suffix)

            # 2. Extract decay rate if F-mode was detected
            if len(peaks) > 0:
                # Use the dominant peak (first peak) as F-mode
                f_mode_idx = peaks[0]
                f_mode_freq = freqs_kHz[f_mode_idx]

                print(f"\nExtracting decay rate for F-mode at {f_mode_freq:.3f} kHz...")
                tau_fit, decay_rate, fitted_freq = extract_decay_rate(
                    times_full, rho_c_full, f_mode_freq,
                    output_dir=plots_dir,
                    suffix=coord_suffix
                )

                if not np.isnan(tau_fit):
                    print(f"\n" + "="*70)
                    print(f"QNM ANALYSIS SUMMARY")
                    print(f"="*70)
                    print(f"F-mode frequency:  {f_mode_freq:.3f} kHz (from spectrum)")
                    print(f"Fitted frequency:  {fitted_freq:.3f} kHz (from decay fit)")
                    print(f"Decay time:        {tau_fit:.3e} M_sun")
                    print(f"Decay rate:        {decay_rate:.3e} M_sun^-1")
                    print(f"="*70)
            else:
                print("\nNo peaks detected - skipping decay rate analysis.")

        except Exception as e:
            print(f"\nWarning: QNM analysis failed: {e}")
            print("Continuing with evolution diagnostics...")
    else:
        print("\nSkipping QNM analysis (insufficient data points or time series not available)")

    print("\n" + "="*70)
    print("Evolution complete. Plots saved:")
    print("  1. tov_solution.png                - TOV solution (ρ, P, M, α)")
    print("  2. tov_initial_data_comparison.png - TOV vs Initial data at t=0")
    print(f"  3. tov_evolution.png               - Hydro evolution at checkpoints:")
    print(f"                                       t=0 → t={t_cp1:.3e} (1/3) → t={t_cp2:.3e} (2/3) → t={t_cp3:.3e} (final)")
    print(f"  4. tov_bssn_evolution.png          - BSSN variables: t=0 → t={t_cp3:.3e} (Cowling check)")
    print("  5. tov_qnm_spectrum.png            - QNM frequency spectrum (F-mode, overtones)")
    print("  6. tov_qnm_decay.png               - F-mode decay rate analysis")
    print("="*70)


if __name__ == "__main__":
    main()
