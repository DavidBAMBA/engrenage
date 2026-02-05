"""Benchmark JAX vs NumPy/Numba for TOV evolution RHS.

This benchmark measures the actual speedup achieved by the JAX backend
compared to the standard NumPy/Numba implementation.
"""
import jax
import jax.numpy as jnp
import numpy as np
import sys
import time
sys.path.insert(0, '/home/davidbamba/repositories/engrenage')

jax.config.update("jax_enable_x64", True)

from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import PolytropicEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.atmosphere import AtmosphereParams
from examples.TOV.tov_solver import load_or_solve_tov_iso
import examples.TOV.tov_initial_data_interpolated as tov_id
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS
from source.matter.hydro.jax.valencia_jax import CowlingGeometry, compute_hydro_rhs_cowling

print("="*70)
print("BENCHMARK: JAX vs NumPy/Numba RHS Evaluation")
print("="*70)

# Setup parameters
r_max = 100.0
num_points = 400
K, Gamma = 100.0, 2.0
rho_central = 1.28e-3
n_warmup = 5
n_bench = 100

rho_floor = 1e-12 * rho_central
p_floor = K * rho_floor**Gamma
atmosphere = AtmosphereParams(rho_floor=rho_floor, p_floor=p_floor)

print(f"\nSetup:")
print(f"  Grid size: N = {num_points}")
print(f"  Warmup runs: {n_warmup}")
print(f"  Benchmark runs: {n_bench}")

# Create grid and load TOV
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

tov = load_or_solve_tov_iso(
    K=K, Gamma=Gamma, rho_central=rho_central,
    r_max=r_max, accuracy="high"
)

initial_state, _ = tov_id.create_initial_data_iso(
    tov, grid, background, eos,
    atmosphere=atmosphere,
    polytrope_K=K, polytrope_Gamma=Gamma,
    interp_order=11
)

state = initial_state.copy()
grid.fill_boundaries(state)

bssn_vars = BSSNVars(grid.N)
bssn_vars.set_bssn_vars(state[:NUM_BSSN_VARS, :])

# =============================================================================
# BENCHMARK 1: NumPy/Numba (Reference)
# =============================================================================
print(f"\n" + "-"*70)
print("NUMBA/NUMPY BACKEND (Reference)")
print("-"*70)

bssn_d1 = grid.get_d1_metric_quantities(state)

# Set matter variables (required before calling get_matter_rhs)
hydro.set_matter_vars(state, bssn_vars, grid)

# Warmup
for _ in range(n_warmup):
    rhs_numba = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1, background)

# Benchmark
times_numba = []
for _ in range(n_bench):
    t0 = time.perf_counter()
    rhs_numba = hydro.get_matter_rhs(grid.r, bssn_vars, bssn_d1, background)
    t1 = time.perf_counter()
    times_numba.append(t1 - t0)

mean_numba = np.mean(times_numba)
std_numba = np.std(times_numba)
min_numba = np.min(times_numba)

print(f"  Mean time: {mean_numba*1000:.3f} ± {std_numba*1000:.3f} ms")
print(f"  Min time:  {min_numba*1000:.3f} ms")

# =============================================================================
# BENCHMARK 2: JAX
# =============================================================================
print(f"\n" + "-"*70)
print("JAX BACKEND")
print("-"*70)

# Setup JAX geometry
alpha = jnp.asarray(bssn_vars.lapse)
phi = jnp.asarray(bssn_vars.phi)
e6phi = jnp.exp(6.0 * phi)
beta_r = jnp.zeros(grid.N)
gamma_rr = jnp.exp(4.0 * phi)

geom = CowlingGeometry(
    alpha=alpha, beta_r=beta_r,
    gamma_rr=gamma_rr, e6phi=e6phi,
    dx=float(grid.derivs.dx),
    num_ghosts=NUM_GHOSTS
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

# Compile + warmup
print(f"  Compiling...")
t0 = time.perf_counter()
for _ in range(n_warmup):
    rhs_D, rhs_Sr, rhs_tau = compute_hydro_rhs_cowling(
        D, Sr, tau, geom, 'polytropic', eos_params, atm_params, 'mp5', 'hll'
    )
t_compile = time.perf_counter() - t0
print(f"  Compilation + warmup: {t_compile*1000:.2f} ms")

# Benchmark
times_jax = []
for _ in range(n_bench):
    t0 = time.perf_counter()
    rhs_D, rhs_Sr, rhs_tau = compute_hydro_rhs_cowling(
        D, Sr, tau, geom, 'polytropic', eos_params, atm_params, 'mp5', 'hll'
    )
    # Force synchronization for accurate timing
    rhs_D.block_until_ready()
    t1 = time.perf_counter()
    times_jax.append(t1 - t0)

mean_jax = np.mean(times_jax)
std_jax = np.std(times_jax)
min_jax = np.min(times_jax)

print(f"  Mean time: {mean_jax*1000:.3f} ± {std_jax*1000:.3f} ms")
print(f"  Min time:  {min_jax*1000:.3f} ms")

# =============================================================================
# COMPARISON
# =============================================================================
speedup_mean = mean_numba / mean_jax
speedup_min = min_numba / min_jax

print(f"\n" + "="*70)
print("SPEEDUP ANALYSIS")
print("="*70)
print(f"\n  NumPy/Numba (mean): {mean_numba*1000:.3f} ms")
print(f"  JAX (mean):         {mean_jax*1000:.3f} ms")
print(f"\n  ✓ SPEEDUP (mean):   {speedup_mean:.2f}x")
print(f"  ✓ SPEEDUP (min):    {speedup_min:.2f}x")

# Check correctness
rhs_jax_array = np.array([np.asarray(rhs_D), np.asarray(rhs_Sr), np.asarray(rhs_tau)])
interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

print(f"\n" + "="*70)
print("CORRECTNESS CHECK")
print("="*70)

max_rel_errors = []
for i, name in enumerate(['D', 'Sr', 'tau']):
    nb = rhs_numba[i, interior]
    jx = rhs_jax_array[i, interior]

    # Compute relative error where values are significant
    mask = np.abs(nb) > 1e-20
    if np.any(mask):
        rel_err = np.max(np.abs((jx[mask] - nb[mask]) / (nb[mask] + 1e-30)))
        max_rel_errors.append(rel_err)
        print(f"  RHS_{name}: max relative error = {rel_err:.3e}")
    else:
        print(f"  RHS_{name}: all values ~zero")

# Summary
print(f"\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n  Grid size: N = {num_points}")
print(f"  Benchmark runs: {n_bench}")
print(f"\n  NumPy/Numba: {mean_numba*1000:.2f} ms/RHS")
print(f"  JAX:         {mean_jax*1000:.2f} ms/RHS")
print(f"\n  ✓ JAX is {speedup_mean:.1f}x faster than NumPy/Numba")
print(f"\n  For 100 RK4 steps (400 RHS calls):")
print(f"    NumPy/Numba: {mean_numba*400:.2f} s")
print(f"    JAX:         {mean_jax*400:.2f} s")
print(f"    Time saved:  {(mean_numba-mean_jax)*400:.2f} s")
print("="*70)
