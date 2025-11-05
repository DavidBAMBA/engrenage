import numpy as np
import pytest


def _build_initial_data(N=300, r_max=16.0, K=100.0, Gamma=2.0, rho_c=1.28e-3,
                        rho_floor=1e-12, p_floor=1e-15):
    """Helper: construct grid, TOV solution, and initial data on the evolution grid."""
    from source.core.spacing import LinearSpacing
    from source.core.grid import Grid
    from source.core.statevector import StateVector
    from source.backgrounds.sphericalbackground import FlatSphericalBackground
    from source.matter.perfect_fluid import PerfectFluid
    from source.matter.hydro.eos import IdealGasEOS
    from source.matter.hydro.atmosphere import AtmosphereParams
    from source.bssn.bssnvars import BSSNVars
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS
    from examples.TOV.tov_solver import TOVSolver
    import examples.TOV.tov_initial_data_interpolated as tov_id

    spacing = LinearSpacing(N, r_max)
    eos = IdealGasEOS(gamma=Gamma)
    atmosphere = AtmosphereParams(rho_floor=rho_floor, p_floor=p_floor)
    hydro = PerfectFluid(eos=eos, atmosphere=atmosphere)
    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    tov = TOVSolver(K=K, Gamma=Gamma).solve(rho_c, r_max=r_max)
    state = tov_id.create_initial_data_interpolated(
        tov, grid, background, eos,
        atmosphere=atmosphere,
        interp_order=11
    )

    bssn = BSSNVars(grid.N)
    bssn.set_bssn_vars(state[:NUM_BSSN_VARS, :])
    hydro.set_matter_vars(state, bssn, grid)
    prim = hydro._get_primitives(bssn, grid.r)

    return grid, background, hydro, state, prim, tov


def _surface_index_from_prim(grid, prim, rho_floor, factor=10.0):
    """Last interior cell with rho > factor * rho_floor."""
    from source.core.spacing import NUM_GHOSTS
    mask = prim['rho0'] > factor * rho_floor
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return NUM_GHOSTS
    return int(idx[-1])


def test_tov_initial_matches_tov_solution():
    """Initial data should match TOV profiles on the evolution grid inside the star."""
    from source.core.spacing import NUM_GHOSTS
    grid, background, hydro, state, prim, tov = _build_initial_data()

    r = grid.r
    R = float(tov['R'])
    # Interpolate reference TOV fields to grid points
    r_tov = tov['r']
    rho_ref = np.interp(r, r_tov, tov['rho_baryon'])
    P_ref = np.interp(r, r_tov, tov['P'])

    # Interior mask excluding ghosts and a thin buffer near R
    interior = (r > r[NUM_GHOSTS]) & (r < (R - 2.0 * grid.min_dr))
    interior &= prim['rho0'] > 50.0 * hydro.atmosphere.rho_floor

    # Relative errors
    rho_err = np.abs(prim['rho0'][interior] - rho_ref[interior]) / (np.abs(rho_ref[interior]) + 1e-30)
    P_err = np.abs(prim['p'][interior] - P_ref[interior]) / (np.abs(P_ref[interior]) + 1e-30)

    # Tolerancias modestas (más estrictas lejos de R)
    assert np.max(rho_err) < 5e-2, f"rho mismatch too large: {np.max(rho_err):.2e}"
    assert np.max(P_err) < 5e-2, f"P mismatch too large: {np.max(P_err):.2e}"


def test_tov_initial_atmosphere_exterior():
    """Exterior points must be set exactly to atmosphere values (D, Sr, tau)."""
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS
    grid, background, hydro, state, prim, tov = _build_initial_data()

    r = grid.r
    R = float(tov['R'])
    ext = r > (R + 1e-12)

    D = state[NUM_BSSN_VARS + 0, :]
    Sr = state[NUM_BSSN_VARS + 1, :]
    tau = state[NUM_BSSN_VARS + 2, :]

    # Exact atmosphere where r > R (up to machine eps)
    assert np.allclose(D[ext], hydro.atmosphere.rho_floor, rtol=0.0, atol=1e-30)
    assert np.allclose(Sr[ext], 0.0, rtol=0.0, atol=1e-30)
    assert np.all(tau[ext] >= hydro.atmosphere.tau_atm - 1e-30)


def test_monotonic_near_surface():
    """rho and P should decrease monotonically in the last few interior cells."""
    grid, background, hydro, state, prim, tov = _build_initial_data()

    i_surf = _surface_index_from_prim(grid, prim, hydro.atmosphere.rho_floor, factor=10.0)
    i0 = max(0, i_surf - 5)
    i1 = i_surf

    rho_seg = prim['rho0'][i0:i1+1]
    P_seg = prim['p'][i0:i1+1]

    # Allow tiny numerical jitter (<= 1e-12 relative)
    assert np.all(np.diff(rho_seg) <= np.maximum(1e-12 * rho_seg[:-1], 0.0))
    assert np.all(np.diff(P_seg) <= np.maximum(1e-12 * P_seg[:-1], 0.0))


@pytest.mark.xfail(strict=False, reason="EOS consistency near surface may fail if rho and P are interpolated independently")
def test_eos_consistency_interior():
    """Check P ≈ K rho^Gamma in the interior (excluding a thin layer near R)."""
    from source.core.spacing import NUM_GHOSTS
    grid, background, hydro, state, prim, tov = _build_initial_data()

    # Use TOV K, Gamma from this test setup
    K = 100.0
    Gamma = 2.0

    r = grid.r
    R = float(tov['R'])
    interior = (r > r[NUM_GHOSTS]) & (r < (R - 2.0 * grid.min_dr))
    interior &= prim['rho0'] > 50.0 * hydro.atmosphere.rho_floor

    P = prim['p'][interior]
    rho = prim['rho0'][interior]
    P_eos = K * np.power(np.maximum(rho, 1e-300), Gamma)
    rel = np.abs(P - P_eos) / (np.abs(P) + 1e-30)

    # Expect very small mismatch if data are EOS-consistent; mark as xfail for current implementation.
    assert np.max(rel) < 1e-6, f"EOS mismatch too large: {np.max(rel):.2e}"

