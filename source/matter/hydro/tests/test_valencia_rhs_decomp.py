import numpy as np


def test_valencia_rhs_decomposition_matches_full_rhs():
    """
    Verify that the Valencia RHS equals the sum of its parts:
      -flux divergence
      +connection terms
      +physical sources

    on a TOV initial datum (Cowling). Checks all three equations: D, S_r, tau.
    """
    # Imports
    from source.core.spacing import LinearSpacing, NUM_GHOSTS
    from source.core.grid import Grid
    from source.core.statevector import StateVector
    from source.backgrounds.sphericalbackground import FlatSphericalBackground
    from source.matter.perfect_fluid import PerfectFluid
    from source.matter.hydro.eos import IdealGasEOS
    from source.matter.hydro.riemann import HLLRiemannSolver
    from source.matter.hydro.reconstruction import create_reconstruction
    from source.matter.hydro.atmosphere import AtmosphereParams
    from source.bssn.bssnvars import BSSNVars
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS
    from source.bssn.tensoralgebra import get_bar_gamma_LL
    from source.matter.hydro.grhd_equations import GRHDEquations
    from source.matter.hydro.stress_energy import StressEnergyTensor, StressEnergyTensor4D
    from examples.TOV.tov_solver import TOVSolver
    import examples.TOV.tov_initial_data_interpolated as tov_id

    # Configuration (moderate size for runtime)
    r_max = 16.0
    N = 300
    K = 100.0
    Gamma = 2.0
    rho_c = 1.28e-3

    # Grid and background
    spacing = LinearSpacing(N, r_max)
    eos = IdealGasEOS(gamma=Gamma)
    atm = AtmosphereParams(rho_floor=1e-12, p_floor=1e-15)
    hydro = PerfectFluid(
        eos=eos,
        spacetime_mode="dynamic",
        atmosphere=atm,
        reconstructor=create_reconstruction("mp5"),
        riemann_solver=HLLRiemannSolver(),
    )
    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    # TOV + initial data
    tov = TOVSolver(K=K, Gamma=Gamma).solve(rho_c, r_max=r_max)
    state = tov_id.create_initial_data_interpolated(
        tov, grid, background, eos,
        atmosphere=atm, polytrope_K=K, polytrope_Gamma=Gamma, interp_order=11
    )

    # BSSN and primitives
    bssn = BSSNVars(grid.N)
    bssn.set_bssn_vars(state[:NUM_BSSN_VARS, :])
    bssn_d1 = grid.get_d1_metric_quantities(state)
    hydro.set_matter_vars(state, bssn, grid)
    prim = hydro._get_primitives(bssn, grid.r)

    r = grid.r
    rho0 = prim['rho0']
    p = prim['p']
    v_U = np.zeros((grid.N, 3))
    v_U[:, 0] = prim['vr']

    # Full RHS from hydro (Cowling)
    rhs_D_full, rhs_Sr_full, rhs_tau_full = hydro.get_matter_rhs(r, bssn, bssn_d1, background)

    # Valencia geometry extract
    val = hydro.valencia
    val._extract_geometry(r, bssn, hydro.spacetime_mode, background, grid)
    adm_geom = val.adm_geometry
    val_geom = val.valencia_geometry
    grhd_eq = GRHDEquations(eos, atmosphere=val.atmosphere, boundary_mode=val.boundary_mode)

    # Compute W, h exactly as in compute_rhs
    v2 = np.einsum('xij,xi,xj->x', val.gamma_LL, v_U, v_U)
    W = 1.0 / np.sqrt(np.maximum(1.0 - v2, 1e-16))
    eps = eos.eps_from_rho_p(rho0, p)
    h = 1.0 + eps + p / np.maximum(rho0, 1e-30)

    # Interface fluxes via GRHDEquations helper
    phi_arr = np.asarray(bssn.phi, dtype=float)
    if len(val_geom.alpha) > 1:
        alpha_face = 0.5 * (val_geom.alpha[:-1] + val_geom.alpha[1:])
        beta_face = 0.5 * (val_geom.beta_U[:-1] + val_geom.beta_U[1:])
        gamma_face = 0.5 * (val_geom.gamma_LL[:-1] + val_geom.gamma_LL[1:])
        phi_face = 0.5 * (phi_arr[:-1] + phi_arr[1:])
    else:
        alpha_face = val_geom.alpha.copy()
        beta_face = val_geom.beta_U.copy()
        gamma_face = val_geom.gamma_LL.copy()
        phi_face = phi_arr.copy()
    e6phi_face = np.exp(6.0 * phi_face)

    class _FaceGeometry:
        __slots__ = ("alpha", "beta_U", "gamma_LL", "e6phi")

        def __init__(self, alpha, beta_U, gamma_LL, e6phi):
            self.alpha = alpha
            self.beta_U = beta_U
            self.gamma_LL = gamma_LL
            self.e6phi = e6phi

    face_geom = _FaceGeometry(alpha_face, beta_face, gamma_face, e6phi_face)

    UL_batch, UR_batch, primL_batch, primR_batch = grhd_eq.reconstruct_and_convert(
        rho0, v_U[:, 0], p, r, hydro.reconstructor, face_geom
    )
    F_D_face, F_S_face, F_tau_face = grhd_eq.solve_riemann_and_densitize(
        UL_batch, UR_batch, primL_batch, primR_batch, face_geom, hydro.riemann_solver, hydro.spacetime_mode
    )

    # Flux divergence (interior only)
    inv_dr = 1.0 / (val.dr + 1e-30)
    Ntot = grid.N
    div_D = np.zeros(Ntot)
    div_S = np.zeros((Ntot, 3))
    div_tau = np.zeros(Ntot)
    for i in range(NUM_GHOSTS, Ntot - NUM_GHOSTS):
        div_D[i] = -(F_D_face[i] - F_D_face[i - 1]) * inv_dr
        div_S[i, :] = -(F_S_face[i, :] - F_S_face[i - 1, :]) * inv_dr
        div_tau[i] = -(F_tau_face[i] - F_tau_face[i - 1]) * inv_dr

    # Sources (physical)
    src_S_vec, src_tau, _ = grhd_eq.compute_source_terms(
        rho0, v_U, p, W, h, val_geom, bssn, bssn_d1, background, hydro.spacetime_mode, r, return_debug=True
    )

    # Connection terms (  form)
    # u^μ and T^{μν}
    u4U = adm_geom.compute_4velocity(v_U, W)
    stress = StressEnergyTensor(adm_geom, rho0, v_U, p, W, h)
    T00_vals, T0i_vals, Tij_vals = stress.compute_T4UU()
    T4UU = StressEnergyTensor4D.from_components(T00_vals, T0i_vals, Tij_vals)
    T0_0_vals, T0_j_vals, Ti_j_vals = stress.compute_T4UD()
    T4UD = StressEnergyTensor4D.from_components(T0_0_vals, T0_j_vals, Ti_j_vals)

    # Geometry factors
    phi = np.asarray(bssn.phi, dtype=float)
    e6phi = np.exp(6.0 * phi)
    alpha = val.alpha

    # rho_* and partial fluxes at cell centers
    rho_star = alpha * e6phi * rho0 * u4U[:, 0]
    fD_U = np.zeros((Ntot, 3))
    fTau_U = np.zeros((Ntot, 3))
    for j in range(3):
        fD_U[:, j] = rho_star * v_U[:, j]
        fTau_U[:, j] = alpha * alpha * e6phi * T4UU.T0i[:, j] - rho_star * v_U[:, j]

    # Momentum partial flux tensor (no √ĝ): F̃^j_i = α e^{6φ} T^j_i
    F_S_no_ghat = np.zeros((Ntot, 3, 3))
    for j in range(3):
        for i in range(3):
            F_S_no_ghat[:, j, i] = alpha * e6phi * T4UD.Tij[:, j, i]

    # Trace and connection contractions
    hat_chris = background.hat_christoffel
    Gamma_trace = np.einsum('xkkj->xj', hat_chris)
    conn_D = -np.einsum('xj,xj->x', Gamma_trace, fD_U)
    conn_tau = -np.einsum('xj,xj->x', Gamma_trace, fTau_U)
    conn_S = (
        -np.einsum('xl,xli->xi', Gamma_trace, F_S_no_ghat)
        + np.einsum('xlji,xjl->xi', hat_chris, F_S_no_ghat)
    )

    # Sum contributions
    rhs_D_rec = div_D + conn_D
    rhs_Sr_rec = div_S[:, 0] + conn_S[:, 0] + src_S_vec[:, 0]
    rhs_tau_rec = div_tau + conn_tau + src_tau

    # Compare on interior (exclude ghosts)
    sl = slice(NUM_GHOSTS, -NUM_GHOSTS)

    def max_abs_diff(a, b):
        return float(np.max(np.abs(a[sl] - b[sl])))

    tol = 1e-10
    dD = max_abs_diff(rhs_D_full, rhs_D_rec)
    dSr = max_abs_diff(rhs_Sr_full, rhs_Sr_rec)
    dtau = max_abs_diff(rhs_tau_full, rhs_tau_rec)

    assert dD < tol, f"D RHS mismatch: {dD:.3e} > {tol}"
    assert dSr < tol, f"Sr RHS mismatch: {dSr:.3e} > {tol}"
    assert dtau < tol, f"tau RHS mismatch: {dtau:.3e} > {tol}"
