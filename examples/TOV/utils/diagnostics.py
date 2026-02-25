"""
Diagnostic utilities for TOV evolution.

Compute constraints, baryon mass, and initial residuals.
"""

import numpy as np
from scipy.integrate import simpson


def diagnose_t0_residuals(state_2d, grid, background, hydro):
    """Compute and print t=0 RHS residuals."""
    from source.bssn.bssnvars import BSSNVars
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS
    from source.core.spacing import NUM_GHOSTS

    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(state_2d[:NUM_BSSN_VARS, :])
    bssn_d1 = grid.get_d1_metric_quantities(state_2d)

    hydro.set_matter_vars(state_2d, bssn_vars, grid)
    rhs_D, rhs_Sr, rhs_tau = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1, background)

    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
    r_int = grid.r[interior]

    i_max = NUM_GHOSTS + int(np.argmax(np.abs(rhs_Sr[interior])))
    print("\nInitial RHS diagnostics (t=0):")
    print(f"  max |dS_r/dt| at r={grid.r[i_max]:.6f} (i={i_max}) -> {rhs_Sr[i_max]:.3e}")
    print(f"  max |dD/dt|   = {np.max(np.abs(rhs_D[interior])):.3e}")
    print(f"  max |dτ/dt|   = {np.max(np.abs(rhs_tau[interior])):.3e}")


def compute_baryon_mass(grid, state, rho0, vr, p, eps, W, h):
    """Compute baryon (rest) mass M = 4π ∫ ρ0 W ψ^6 r^2 dr."""
    from source.bssn.bssnstatevariables import idx_phi
    from source.core.spacing import NUM_GHOSTS

    interior = slice(NUM_GHOSTS, grid.N - NUM_GHOSTS)
    r = grid.r[interior]
    rho0_int = rho0[interior]
    W_int = W[interior]

    phi = state[idx_phi, interior]
    psi = np.exp(phi)
    integrand = rho0_int * W_int * (psi**6) * (r**2)
    return 4.0 * np.pi * simpson(integrand, x=r)


def compute_constraints(state_2d, grid, background, matter):
    """Compute Hamiltonian and Momentum constraints."""
    from source.bssn.constraintsdiagnostic import get_constraints_diagnostic

    state_flat = state_2d.flatten()
    Ham, Mom = get_constraints_diagnostic(state_flat, 0.0, grid, background, matter)
    return Ham[0, :], Mom[0, :, :]


def evolve_fixed_timestep(state_initial, dt, num_steps, grid, background, hydro,
                          bssn_fixed, bssn_d1_fixed, atmosphere, rk4_step_func,
                          method='rk4', t_start=0.0,
                          reference_state=None, step_offset=0, data_manager=None,
                          snapshot_interval=None, evolution_interval=None):
    """Evolve with fixed timestep using RK4."""
    from source.bssn.bssnvars import BSSNVars
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS, idx_lapse
    from source.core.spacing import NUM_GHOSTS

    state_flat = state_initial.flatten()

    bssn_vars_fixed = BSSNVars(grid.N)
    bssn_vars_fixed.set_bssn_vars(bssn_fixed)

    def primitives_from_state(state_flattened):
        s2d = state_flattened.reshape((grid.NUM_VARS, grid.N))
        hydro.set_matter_vars(s2d, bssn_vars_fixed, grid)
        rho0, vr, p, eps, W, h, success = hydro._get_primitives(bssn_vars_fixed, grid.r)
        return rho0, vr, p, eps, W, h, success, s2d

    rho0_prev, vr_prev, p_prev, eps_prev, W_prev, h_prev, success_prev, s_prev = primitives_from_state(state_flat)

    if reference_state is None:
        reference_state = state_initial
    rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial, success_initial, s_initial = primitives_from_state(reference_state.flatten())

    if data_manager and data_manager.enable_saving:
        Ham_0, Mom_0 = compute_constraints(state_initial, grid, background, hydro)
        data_manager.save_snapshot(step_offset, t_start, state_initial, rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial,
                                 Ham=Ham_0, Mom=Mom_0)
        data_manager.add_evolution_point(step_offset, t_start, state_initial,
                                        rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial, success_initial,
                                        rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial, success_initial,
                                        Ham=Ham_0, Mom=Mom_0)

    times_series = [t_start]
    Mb0 = compute_baryon_mass(grid, s_initial, rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial)
    Mb_series = [Mb0]
    rho_c0 = rho0_initial[NUM_GHOSTS]
    rho_c_series = [rho_c0]
    v_c0 = vr_initial[NUM_GHOSTS]
    v_c_series = [v_c0]

    print("\n===== Evolution diagnostics (per step) =====")
    print("Columns: step | t | ρ_central | max_Δρ/ρ@r | max_vʳ@r | max_Sʳ@r | c2p_fails")
    print("  (@r indicates the radial position where the maximum occurs)")
    print("-" * 140)

    for step in range(num_steps):
        state_flat_next = rk4_step_func(state_flat, dt, grid, background, hydro,
                                   bssn_fixed, bssn_d1_fixed, atmosphere)

        rho0_next, vr_next, p_next, eps_next, W_next, h_next, success_next, s_next = primitives_from_state(state_flat_next)

        interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

        rho_next_int = rho0_next[interior]
        rho_init_int = rho0_initial[interior]
        p_next_int = p_next[interior]
        v_next_int = vr_next[interior]

        D_next = s_next[NUM_BSSN_VARS + 0, interior]
        Sr_next = s_next[NUM_BSSN_VARS + 1, interior]
        tau_next = s_next[NUM_BSSN_VARS + 2, interior]

        rho_central = float(rho0_next[NUM_GHOSTS])

        r_interior = grid.r[interior]

        rel_rho_err = np.abs(rho_next_int - rho_init_int) / (np.abs(rho_init_int) + 1e-20)
        idx_max_rho_err = np.argmax(rel_rho_err)
        max_rel_rho_err = float(rel_rho_err[idx_max_rho_err])
        r_max_rho_err = float(r_interior[idx_max_rho_err])

        idx_max_v = np.argmax(np.abs(v_next_int))
        max_abs_v = float(v_next_int[idx_max_v])
        r_max_v = float(r_interior[idx_max_v])

        idx_max_D = np.argmax(D_next)
        max_D = float(D_next[idx_max_D])
        r_max_D = float(r_interior[idx_max_D])

        idx_max_Sr = np.argmax(np.abs(Sr_next))
        max_Sr = float(np.abs(Sr_next[idx_max_Sr]))
        r_max_Sr = float(r_interior[idx_max_Sr])

        idx_max_tau = np.argmax(np.abs(tau_next))
        max_tau = float(np.abs(tau_next[idx_max_tau]))
        r_max_tau = float(r_interior[idx_max_tau])

        c2p_fail_count = int(np.sum(~success_next))

        t_curr = t_start + (step + 1) * dt
        step_num = step_offset + step + 1
        if step_num % 100 == 0:
            print(f"step {step_num:4d}  t={t_curr:.2e}:  ρ_c={rho_central:.6e}  max_Δρ/ρ={max_rel_rho_err:.2e}@r={r_max_rho_err:.2f}  "
              f"max_vʳ={max_abs_v:.3e}@r={r_max_v:.2f}  max_Sʳ={max_Sr:.2e}@r={r_max_Sr:.2f}  "
              f"c2p_fail={c2p_fail_count}")

        if data_manager and data_manager.enable_saving:
            Ham, Mom = None, None
            if evolution_interval and (step_num % evolution_interval == 0):
                Ham, Mom = compute_constraints(s_next, grid, background, hydro)
            if snapshot_interval and (step_num % snapshot_interval == 0) and (Ham is None):
                Ham, Mom = compute_constraints(s_next, grid, background, hydro)

            if evolution_interval and step_num % evolution_interval == 0:
                data_manager.add_evolution_point(step_num, t_curr, s_next,
                                                rho0_next, vr_next, p_next, eps_next, W_next, h_next, success_next,
                                                rho0_initial, vr_initial, p_initial, eps_initial, W_initial, h_initial, success_initial,
                                                Ham=Ham, Mom=Mom)

                if step_num % (evolution_interval * 10) == 0:
                    data_manager.flush_evolution_buffer()

            if snapshot_interval and step_num % snapshot_interval == 0:
                data_manager.save_snapshot(step_num, t_curr, s_next, rho0_next, vr_next, p_next, eps_next, W_next, h_next,
                                          Ham=Ham, Mom=Mom)

        Mb_next = compute_baryon_mass(grid, s_next, rho0_next, vr_next, p_next, eps_next, W_next, h_next)
        times_series.append(t_curr)
        Mb_series.append(Mb_next)
        rho_c_series.append(float(rho0_next[NUM_GHOSTS]))
        v_c_series.append(float(vr_next[NUM_GHOSTS]))

        lapse_c = float(s_next[idx_lapse, NUM_GHOSTS])
        if lapse_c < 0.05:
            print(f"\n*** COLLAPSE DETECTED: alpha_c={lapse_c:.4f} < 0.05 at t={t_curr:.4e} ***")
            state_flat = state_flat_next
            actual_steps = step + 1
            actual_time = t_start + actual_steps * dt
            return state_flat.reshape((grid.NUM_VARS, grid.N)), actual_steps, actual_time, {
                't': np.array(times_series),
                'Mb': np.array(Mb_series),
                'rho_c': np.array(rho_c_series),
                'v_c': np.array(v_c_series),
            }

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

        if issues:
            print("  -> Detected issues:", ", ".join(issues))
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

            print("  -> Halting evolution early due to detected instability.")
            state_flat = state_flat_next
            actual_steps = step + 1
            actual_time = t_start + actual_steps * dt
            return state_flat.reshape((grid.NUM_VARS, grid.N)), actual_steps, actual_time, {
                't': np.array(times_series),
                'Mb': np.array(Mb_series),
                'rho_c': np.array(rho_c_series),
                'v_c': np.array(v_c_series),
            }

        state_flat = state_flat_next

    if data_manager and data_manager.enable_saving:
        data_manager.flush_evolution_buffer()

    actual_time = t_start + num_steps * dt
    return state_flat.reshape((grid.NUM_VARS, grid.N)), num_steps, actual_time, {
        't': np.array(times_series),
        'Mb': np.array(Mb_series),
        'rho_c': np.array(rho_c_series),
        'v_c': np.array(v_c_series),
    }
