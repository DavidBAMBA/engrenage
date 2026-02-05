"""
Comparación OPTIMIZADA: NumPy/Numba vs JAX en TOV Evolution
FIX: Mantiene arrays en JAX durante todo el loop (sin conversiones)
"""
import numpy as np
import sys
import os
import time

# IMPORTANTE: Forzar CPU ANTES de importar JAX
os.environ['JAX_PLATFORMS'] = 'cpu'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import PolytropicEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.atmosphere import AtmosphereParams
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS
from examples.TOV.tov_solver import load_or_solve_tov_iso
import examples.TOV.tov_initial_data_interpolated as tov_id

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from source.matter.hydro.jax.valencia_jax import CowlingGeometry, compute_hydro_rhs_cowling

print("="*70)
print("COMPARACIÓN OPTIMIZADA: NumPy/Numba vs JAX")
print("="*70)

# Configuración
r_max = 100.0
num_points = 400
K, Gamma = 100.0, 2.0
rho_central = 1.28e-3
n_steps = 100  # 100 pasos para mejor estadística
dt = 0.005

print(f"\nConfiguración:")
print(f"  Grid: N = {num_points}")
print(f"  Evolución: {n_steps} pasos RK4, dt = {dt}")
print(f"  JAX device: {jax.devices()[0]}")

# Setup
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
    riemann_solver=riemann, solver_method="newton"
)

state_vector = StateVector(hydro)
grid = Grid(spacing, state_vector)
background = FlatSphericalBackground(grid.r)
hydro.background = background

print(f"\nCargando TOV...")
tov = load_or_solve_tov_iso(K=K, Gamma=Gamma, rho_central=rho_central,
                             r_max=r_max, accuracy="high")

initial_state, _ = tov_id.create_initial_data_iso(
    tov, grid, background, eos, atmosphere=atmosphere,
    polytrope_K=K, polytrope_Gamma=Gamma, interp_order=11
)

# =============================================================================
# BACKEND 1: NumPy/Numba
# =============================================================================
print(f"\n{'='*70}")
print("BACKEND 1: NumPy/Numba")
print(f"{'='*70}")

state_numpy = initial_state.copy()
grid.fill_boundaries(state_numpy)

bssn_vars = BSSNVars(grid.N)
bssn_vars.set_bssn_vars(state_numpy[:NUM_BSSN_VARS, :])
bssn_d1 = grid.get_d1_metric_quantities(state_numpy)
bssn_fixed = state_numpy[:NUM_BSSN_VARS, :].copy()

def get_rhs_numpy(t, y, grid, background, hydro, bssn_fixed, bssn_d1):
    state = y.reshape((grid.NUM_VARS, grid.N))
    grid.fill_boundaries(state)

    bssn_vars = BSSNVars(grid.N)
    bssn_vars.set_bssn_vars(bssn_fixed)

    hydro.set_matter_vars(state, bssn_vars, grid)
    hydro_rhs = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1, background)

    rhs = np.zeros_like(state)
    rhs[NUM_BSSN_VARS:, :] = hydro_rhs
    return rhs.flatten()

def rk4_step_numpy(state_flat, dt, grid, background, hydro, bssn_fixed, bssn_d1):
    k1 = get_rhs_numpy(0, state_flat, grid, background, hydro, bssn_fixed, bssn_d1)
    state_2 = state_flat + 0.5 * dt * k1
    k2 = get_rhs_numpy(0, state_2, grid, background, hydro, bssn_fixed, bssn_d1)
    state_3 = state_flat + 0.5 * dt * k2
    k3 = get_rhs_numpy(0, state_3, grid, background, hydro, bssn_fixed, bssn_d1)
    state_4 = state_flat + dt * k3
    k4 = get_rhs_numpy(0, state_4, grid, background, hydro, bssn_fixed, bssn_d1)
    return state_flat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

state_flat_numpy = state_numpy.flatten()

print("Evolucionando...")
t_start = time.perf_counter()

for step in range(n_steps):
    state_flat_numpy = rk4_step_numpy(state_flat_numpy, dt, grid, background,
                                       hydro, bssn_fixed, bssn_d1)
    if (step + 1) % 20 == 0:
        print(f"  Paso {step+1}/{n_steps}")

t_numpy_total = time.perf_counter() - t_start

print(f"\n✓ NumPy/Numba:")
print(f"  Tiempo total: {t_numpy_total:.3f} s")
print(f"  Tiempo/paso: {t_numpy_total/n_steps*1000:.2f} ms")

# =============================================================================
# BACKEND 2: JAX OPTIMIZADO (sin conversiones en el loop)
# =============================================================================
print(f"\n{'='*70}")
print("BACKEND 2: JAX (optimizado)")
print(f"{'='*70}")

state_jax = initial_state.copy()
bssn_vars_jax = BSSNVars(grid.N)
bssn_vars_jax.set_bssn_vars(state_jax[:NUM_BSSN_VARS, :])

# Geometría JAX
alpha = jnp.asarray(bssn_vars_jax.lapse)
phi = jnp.asarray(bssn_vars_jax.phi)
e6phi = jnp.exp(6.0 * phi)
gamma_rr = jnp.exp(4.0 * phi)

geom_jax = CowlingGeometry(
    alpha=alpha, beta_r=jnp.zeros(grid.N),
    gamma_rr=gamma_rr, e6phi=e6phi,
    dx=float(grid.derivs.dx), num_ghosts=NUM_GHOSTS
)

eos_params = {'gamma': Gamma, 'K': K}
atm_params = {
    'rho_floor': float(atmosphere.rho_floor),
    'p_floor': float(atmosphere.p_floor),
    'v_max': float(atmosphere.v_max),
    'W_max': 10.0, 'tol': 1e-12, 'max_iter': 500,
}

# RK4 PURAMENTE EN JAX (sin conversiones!)
@jax.jit
def rk4_step_jax_pure(D, Sr, tau, geom, eos_type, eos_gamma, eos_K,
                       rho_floor, p_floor, v_max, W_max, tol, max_iter,
                       recon_method, riemann_type, dt):
    """RK4 completamente en JAX - sin conversiones NumPy."""
    # k1
    rhs_D1, rhs_Sr1, rhs_tau1 = compute_hydro_rhs_cowling._compute_hydro_rhs_impl(
        D, Sr, tau, geom, eos_type, eos_gamma, eos_K,
        rho_floor, p_floor, v_max, W_max, tol, max_iter,
        recon_method, riemann_type, False, False  # use_connections, use_sources
    )

    # k2
    D2 = D + 0.5 * dt * rhs_D1
    Sr2 = Sr + 0.5 * dt * rhs_Sr1
    tau2 = tau + 0.5 * dt * rhs_tau1

    rhs_D2, rhs_Sr2, rhs_tau2 = compute_hydro_rhs_cowling._compute_hydro_rhs_impl(
        D2, Sr2, tau2, geom, eos_type, eos_gamma, eos_K,
        rho_floor, p_floor, v_max, W_max, tol, max_iter,
        recon_method, riemann_type, False, False
    )

    # k3
    D3 = D + 0.5 * dt * rhs_D2
    Sr3 = Sr + 0.5 * dt * rhs_Sr2
    tau3 = tau + 0.5 * dt * rhs_tau2

    rhs_D3, rhs_Sr3, rhs_tau3 = compute_hydro_rhs_cowling._compute_hydro_rhs_impl(
        D3, Sr3, tau3, geom, eos_type, eos_gamma, eos_K,
        rho_floor, p_floor, v_max, W_max, tol, max_iter,
        recon_method, riemann_type, False, False
    )

    # k4
    D4 = D + dt * rhs_D3
    Sr4 = Sr + dt * rhs_Sr3
    tau4 = tau + dt * rhs_tau3

    rhs_D4, rhs_Sr4, rhs_tau4 = compute_hydro_rhs_cowling._compute_hydro_rhs_impl(
        D4, Sr4, tau4, geom, eos_type, eos_gamma, eos_K,
        rho_floor, p_floor, v_max, W_max, tol, max_iter,
        recon_method, riemann_type, False, False
    )

    # Update
    D_new = D + (dt / 6.0) * (rhs_D1 + 2*rhs_D2 + 2*rhs_D3 + rhs_D4)
    Sr_new = Sr + (dt / 6.0) * (rhs_Sr1 + 2*rhs_Sr2 + 2*rhs_Sr3 + rhs_Sr4)
    tau_new = tau + (dt / 6.0) * (rhs_tau1 + 2*rhs_tau2 + 2*rhs_tau3 + rhs_tau4)

    return D_new, Sr_new, tau_new

# Estado inicial en JAX
D_jax = jnp.asarray(state_jax[NUM_BSSN_VARS + 0, :])
Sr_jax = jnp.asarray(state_jax[NUM_BSSN_VARS + 1, :])
tau_jax = jnp.asarray(state_jax[NUM_BSSN_VARS + 2, :])

# Warmup
print("Compilando JAX...")
t_compile_start = time.perf_counter()
D_jax, Sr_jax, tau_jax = rk4_step_jax_pure(
    D_jax, Sr_jax, tau_jax, geom_jax, 'polytropic',
    eos_params['gamma'], eos_params['K'],
    atm_params['rho_floor'], atm_params['p_floor'], atm_params['v_max'],
    atm_params['W_max'], atm_params['tol'], atm_params['max_iter'],
    'mp5', 'hll', dt
)
D_jax.block_until_ready()
t_compile = time.perf_counter() - t_compile_start
print(f"  Compilación: {t_compile*1000:.0f} ms")

# Reset
D_jax = jnp.asarray(state_jax[NUM_BSSN_VARS + 0, :])
Sr_jax = jnp.asarray(state_jax[NUM_BSSN_VARS + 1, :])
tau_jax = jnp.asarray(state_jax[NUM_BSSN_VARS + 2, :])

print("Evolucionando...")
t_start = time.perf_counter()

for step in range(n_steps):
    D_jax, Sr_jax, tau_jax = rk4_step_jax_pure(
        D_jax, Sr_jax, tau_jax, geom_jax, 'polytropic',
        eos_params['gamma'], eos_params['K'],
        atm_params['rho_floor'], atm_params['p_floor'], atm_params['v_max'],
        atm_params['W_max'], atm_params['tol'], atm_params['max_iter'],
        'mp5', 'hll', dt
    )
    if (step + 1) % 20 == 0:
        D_jax.block_until_ready()  # Sync para timing preciso
        print(f"  Paso {step+1}/{n_steps}")

D_jax.block_until_ready()
t_jax_total = time.perf_counter() - t_start

print(f"\n✓ JAX:")
print(f"  Tiempo total: {t_jax_total:.3f} s")
print(f"  Tiempo/paso: {t_jax_total/n_steps*1000:.2f} ms")

# =============================================================================
# COMPARACIÓN
# =============================================================================
speedup = t_numpy_total / t_jax_total

print(f"\n{'='*70}")
print("COMPARACIÓN FINAL")
print(f"{'='*70}")

print(f"\n  {'Backend':<20} {'Tiempo total':<15} {'Tiempo/paso'}")
print(f"  {'-'*55}")
print(f"  {'NumPy/Numba':<20} {t_numpy_total:<15.3f} s {t_numpy_total/n_steps*1000:.2f} ms")
print(f"  {'JAX (CPU)':<20} {t_jax_total:<15.3f} s {t_jax_total/n_steps*1000:.2f} ms")

print(f"\n  {'='*55}")
print(f"  🚀 SPEEDUP: {speedup:.2f}x")
print(f"  {'='*55}")

if speedup > 2.0:
    print(f"\n  ✅ ¡JAX es {speedup:.1f}x MÁS RÁPIDO!")
    print(f"  ⚡ Tiempo ahorrado: {t_numpy_total - t_jax_total:.2f} s en {n_steps} pasos")
elif speedup > 1.2:
    print(f"\n  ✓ JAX es {speedup:.1f}x más rápido (mejora moderada)")
else:
    print(f"\n  ⚠ Speedup: {speedup:.1f}x (mejora marginal)")

print(f"\n  Para evolución de 1000 pasos:")
print(f"    NumPy/Numba: ~{t_numpy_total*1000/n_steps:.1f} s")
print(f"    JAX:         ~{t_jax_total*1000/n_steps:.1f} s")
print(f"    Ahorro:      ~{(t_numpy_total-t_jax_total)*1000/n_steps:.1f} s")

print(f"\n{'='*70}")
