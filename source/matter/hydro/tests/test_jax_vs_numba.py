"""
Cross-validation tests: JAX hydro pipeline vs Numba hydro pipeline.

Tests that the JAX-native implementations produce numerically identical
results to the Numba implementations for all hydro pipeline components.

Usage:
    pytest source/matter/hydro/tests/test_jax_vs_numba.py -v

Requires JAX to be installed.
"""

import numpy as np
import pytest
import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# Skip all tests if JAX is not installed
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# =============================================================================
# Fixtures: Create TOV-like test data
# =============================================================================

@pytest.fixture(scope="module")
def tov_setup():
    """Create a small TOV-like setup for testing."""
    from source.core.grid import Grid
    from source.core.spacing import LinearSpacing, NUM_GHOSTS
    from source.core.statevector import StateVector
    from source.backgrounds.sphericalbackground import FlatSphericalBackground
    from source.bssn.bssnvars import BSSNVars
    from source.bssn.bssnstatevariables import NUM_BSSN_VARS
    from source.matter.hydro.perfect_fluid import PerfectFluid
    from source.matter.hydro.eos import PolytropicEOS
    from source.matter.hydro.reconstruction import create_reconstruction
    from source.matter.hydro.riemann import HLLRiemannSolver
    from source.matter.hydro.atmosphere import AtmosphereParams
    from examples.TOV.tov_solver import load_or_solve_tov_iso
    import examples.TOV.tov_initial_data_interpolated as tov_id

    # Small grid for fast tests
    r_max = 100.0
    num_points = 200
    K = 100.0
    Gamma = 2.0
    rho_central = 1.28e-3

    rho_floor = 1e-12 * rho_central
    p_floor = K * rho_floor**Gamma
    atmosphere = AtmosphereParams(rho_floor=rho_floor, p_floor=p_floor)

    spacing = LinearSpacing(num_points, r_max)
    eos = PolytropicEOS(K=K, gamma=Gamma)
    recon = create_reconstruction("mp5")
    riemann = HLLRiemannSolver(atmosphere=atmosphere)

    hydro = PerfectFluid(
        eos=eos, spacetime_mode="dynamic",
        atmosphere=atmosphere, reconstructor=recon,
        riemann_solver=riemann, solver_method="newton",
    )

    state_vector = StateVector(hydro)
    grid = Grid(spacing, state_vector)
    background = FlatSphericalBackground(grid.r)
    hydro.background = background

    # TOV initial data
    tov = load_or_solve_tov_iso(K=K, Gamma=Gamma, rho_central=rho_central,
                                 r_max=r_max, accuracy="high")
    initial_state, _ = tov_id.create_initial_data_iso(
        tov, grid, background, eos,
        atmosphere=atmosphere,
        polytrope_K=K, polytrope_Gamma=Gamma,
        interp_order=11,
    )

    return {
        'initial_state': initial_state,
        'grid': grid,
        'background': background,
        'hydro': hydro,
        'eos': eos,
        'atmosphere': atmosphere,
        'K': K,
        'Gamma': Gamma,
        'NUM_BSSN_VARS': NUM_BSSN_VARS,
        'NUM_GHOSTS': NUM_GHOSTS,
    }


# =============================================================================
# Test: EOS functions
# =============================================================================

class TestEOS:
    def test_pressure_polytropic(self, tov_setup):
        """Test polytropic pressure: JAX vs NumPy."""
        from source.matter.hydro.jax.eos_jax import pressure_polytropic

        K, Gamma = tov_setup['K'], tov_setup['Gamma']
        rho = np.linspace(1e-15, 1e-3, 100)
        p_np = K * rho**Gamma
        p_jax = np.asarray(pressure_polytropic(jnp.array(rho), K, Gamma))
        np.testing.assert_allclose(p_jax, p_np, rtol=1e-12)

    def test_sound_speed_polytropic(self, tov_setup):
        """Test polytropic sound speed: JAX vs formula."""
        from source.matter.hydro.jax.eos_jax import sound_speed_squared_polytropic

        K, Gamma = tov_setup['K'], tov_setup['Gamma']
        rho = np.linspace(1e-12, 1e-3, 100)
        eps = K * rho**(Gamma - 1.0) / (Gamma - 1.0)
        h = 1.0 + Gamma * K * rho**(Gamma - 1.0) / (Gamma - 1.0)
        cs2_expected = Gamma * K * rho**(Gamma - 1.0) / h
        cs2_expected = np.clip(cs2_expected, 0.0, 1.0)

        cs2_jax = np.asarray(sound_speed_squared_polytropic(jnp.array(rho), K, Gamma))
        np.testing.assert_allclose(cs2_jax, cs2_expected, rtol=1e-12)

    def test_prim_to_cons_polytropic(self, tov_setup):
        """Test prim_to_cons: JAX vs Numba."""
        from source.matter.hydro.jax.eos_jax import prim_to_cons_jax
        from source.matter.hydro.cons2prim import prim_to_cons
        from source.matter.hydro.geometry import GeometryState

        K, Gamma = tov_setup['K'], tov_setup['Gamma']
        eos = tov_setup['eos']

        N = 50
        rho = np.linspace(1e-10, 1e-3, N)
        vr = np.linspace(-0.1, 0.1, N)
        p = K * rho**Gamma
        gamma_rr = np.ones(N) * 1.1
        e6phi = np.ones(N) * 1.05

        geom = GeometryState(
            alpha=np.ones(N), beta_r=np.zeros(N),
            gamma_rr=gamma_rr, e6phi=e6phi,
        )
        D_nb, Sr_nb, tau_nb = prim_to_cons(rho, vr, p, geom, eos)

        D_jx, Sr_jx, tau_jx = prim_to_cons_jax(
            jnp.array(rho), jnp.array(vr), jnp.array(p),
            jnp.array(gamma_rr), jnp.array(e6phi),
            'polytropic', Gamma, K,
        )

        np.testing.assert_allclose(np.asarray(D_jx), D_nb, rtol=1e-10)
        np.testing.assert_allclose(np.asarray(Sr_jx), Sr_nb, rtol=1e-10)
        np.testing.assert_allclose(np.asarray(tau_jx), tau_nb, rtol=1e-10)


# =============================================================================
# Test: Full RHS comparison
# =============================================================================

class TestRHS:
    def test_single_rhs_evaluation(self, tov_setup):
        """Compare one RHS evaluation: JAX vs Numba on TOV initial data."""
        from source.bssn.bssnvars import BSSNVars
        from source.bssn.bssnstatevariables import NUM_BSSN_VARS
        from source.bssn.tensoralgebra import get_bar_gamma_LL, get_bar_A_LL, get_hat_D_bar_gamma_LL
        from source.core.spacing import NUM_GHOSTS
        from source.matter.hydro.jax.valencia_jax import CowlingGeometry, compute_hydro_rhs_cowling

        initial_state = tov_setup['initial_state'].copy()
        grid = tov_setup['grid']
        background = tov_setup['background']
        hydro = tov_setup['hydro']
        K = tov_setup['K']
        Gamma = tov_setup['Gamma']
        atmosphere = tov_setup['atmosphere']

        # --- Numba RHS ---
        state = initial_state.copy()
        grid.fill_boundaries(state)

        bssn_vars = BSSNVars(grid.N)
        bssn_vars.set_bssn_vars(state[:NUM_BSSN_VARS, :])
        hydro.set_matter_vars(state, bssn_vars, grid)

        bssn_d1 = grid.get_d1_metric_quantities(state)
        rhs_numba = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1, background)

        # --- JAX RHS ---
        # Build geometry
        N = grid.N
        r = grid.r

        bssn_vars2 = BSSNVars(N)
        bssn_vars2.set_bssn_vars(state[:NUM_BSSN_VARS, :])
        bssn_d1_2 = grid.get_d1_metric_quantities(state)

        alpha = np.asarray(bssn_vars2.lapse, dtype=np.float64)
        beta_U = np.asarray(bssn_vars2.shift_U, dtype=np.float64) * background.inverse_scaling_vector
        phi = np.asarray(bssn_vars2.phi, dtype=np.float64)
        e4phi = np.exp(4.0 * phi)
        e6phi = np.exp(6.0 * phi)
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars2.h_LL, background)
        gamma_LL = e4phi[:, None, None] * bar_gamma_LL
        gamma_UU = np.linalg.inv(gamma_LL)

        K_scalar = np.asarray(bssn_vars2.K, dtype=np.float64)
        bar_A_LL = get_bar_A_LL(r, bssn_vars2, background)
        K_LL = e4phi[:, None, None] * bar_A_LL + (K_scalar / 3.0)[:, None, None] * gamma_LL

        dalpha_dx = np.asarray(bssn_d1_2.lapse)
        dbeta_dx = (
            background.inverse_scaling_vector[:, :, None] * np.asarray(bssn_d1_2.shift_U)
            + bssn_vars2.shift_U[:, :, None] * background.d1_inverse_scaling_vector
        )
        hat_chris = background.hat_christoffel
        hatD_beta_U = np.transpose(dbeta_dx, (0, 2, 1)) + np.einsum('xjik,xk->xij', hat_chris, beta_U)

        dphi_dx = np.asarray(bssn_d1_2.phi)
        hat_D_bar_gamma = get_hat_D_bar_gamma_LL(r, bssn_vars2.h_LL, bssn_d1_2.h_LL, background)
        hatD_gamma_LL = e4phi[:, None, None, None] * (
            4.0 * dphi_dx[:, :, None, None] * bar_gamma_LL[:, None, :, :]
            + np.transpose(hat_D_bar_gamma, (0, 3, 1, 2))
        )

        geom = CowlingGeometry(
            alpha=alpha, beta_r=beta_U[:, 0],
            gamma_rr=gamma_LL[:, 0, 0], e6phi=e6phi,
            dx=float(grid.derivs.dx), num_ghosts=NUM_GHOSTS,
            K_LL=K_LL, dalpha_dx=dalpha_dx,
            hatD_beta_U=hatD_beta_U, hatD_gamma_LL=hatD_gamma_LL,
            hat_christoffel=hat_chris,
            beta_U=beta_U, gamma_LL=gamma_LL, gamma_UU=gamma_UU,
            e4phi=e4phi,
        )

        D = jnp.asarray(state[NUM_BSSN_VARS + 0, :])
        Sr = jnp.asarray(state[NUM_BSSN_VARS + 1, :])
        tau = jnp.asarray(state[NUM_BSSN_VARS + 2, :])

        eos_params = {'gamma': Gamma, 'K': K}
        atm_params = {
            'rho_floor': float(atmosphere.rho_floor),
            'p_floor': float(atmosphere.p_floor),
            'v_max': float(atmosphere.v_max),
            'W_max': 10.0,
            'tol': 1e-12,
            'max_iter': 500,
        }

        rhs_D, rhs_Sr, rhs_tau = compute_hydro_rhs_cowling(
            D, Sr, tau, geom,
            'polytropic', eos_params, atm_params,
            'mp5', 'hll',
        )

        rhs_jax = np.array([np.asarray(rhs_D), np.asarray(rhs_Sr), np.asarray(rhs_tau)])

        # Compare interior points only (ghost cells differ due to boundary treatment)
        interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
        for i, name in enumerate(['D', 'Sr', 'tau']):
            nb = rhs_numba[i, interior]
            jx = rhs_jax[i, interior]
            # Use relative tolerance where values are significant
            mask = np.abs(nb) > 1e-20
            if np.any(mask):
                rel_err = np.max(np.abs((jx[mask] - nb[mask]) / nb[mask]))
                print(f"  RHS {name}: max relative error = {rel_err:.3e}")
                # Allow some tolerance due to different cons2prim implementations
                assert rel_err < 1e-4, f"RHS {name} mismatch: rel_err={rel_err:.3e}"
