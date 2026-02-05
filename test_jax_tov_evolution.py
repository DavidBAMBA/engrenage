"""Integration test: Full TOV evolution with JAX backend."""
import jax
import jax.numpy as jnp
import numpy as np
import sys
import time
sys.path.insert(0, '/home/davidbamba/repositories/engrenage')

jax.config.update("jax_enable_x64", True)

# Import core modules
from source.core.grid import Grid
from source.core.spacing import LinearSpacing
from source.core.statevector import StateVector
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.matter.hydro.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import PolytropicEOS
from source.matter.hydro.reconstruction import create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.atmosphere import AtmosphereParams
from examples.TOV.tov_solver import load_or_solve_tov_iso
import examples.TOV.tov_initial_data_interpolated as tov_id

# Import BSSN modules
from source.bssn.bssnvars import BSSNVars
from source.bssn.bssnstatevariables import NUM_BSSN_VARS
from source.core.spacing import NUM_GHOSTS

# Import JAX backend
from source.matter.hydro.jax.valencia_jax import CowlingGeometry, compute_hydro_rhs_cowling

print("="*70)
print("JAX TOV EVOLUTION INTEGRATION TEST")
print("="*70)

# Setup
r_max = 2.0
num_points = 100
K, Gamma = 100.0, 2.0
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
    riemann_solver=riemann, solver_method="newton"
)

state_vector = StateVector(hydro)
grid = Grid(spacing, state_vector)
background = FlatSphericalBackground(grid.r)
hydro.background = background

print(f"\nGrid setup:")
print(f"  N = {grid.N}")
print(f"  r_max = {r_max}")
print(f"  dr = {grid.derivs.dx:.3f}")

# Load TOV
print(f"\nSolving TOV equation...")
tov = load_or_solve_tov_iso(
    K=K, Gamma=Gamma, rho_central=rho_central,
    r_max=r_max, accuracy="high"
)
print(f"  TOV loaded from cache")

# Create initial data
print(f"\nCreating initial data...")
initial_state, _ = tov_id.create_initial_data_iso(
    tov, grid, background, eos,
    atmosphere=atmosphere,
    polytrope_K=K, polytrope_Gamma=Gamma,
    interp_order=11
)

state = initial_state.copy()
grid.fill_boundaries(state)

# Setup BSSN variables
bssn_vars = BSSNVars(grid.N)
bssn_vars.set_bssn_vars(state[:NUM_BSSN_VARS, :])

# Create JAX geometry (simple - no connections/sources for Cowling)
print(f"\nCreating JAX geometry...")
alpha = jnp.asarray(bssn_vars.lapse)
phi = jnp.asarray(bssn_vars.phi)
e6phi = jnp.exp(6.0 * phi)

# For 1D spherical, beta_r and gamma_rr are simple
beta_r = jnp.zeros(grid.N)  # No shift in Cowling
gamma_rr = jnp.exp(4.0 * phi)  # Isotropic gauge

geom = CowlingGeometry(
    alpha=alpha, beta_r=beta_r,
    gamma_rr=gamma_rr, e6phi=e6phi,
    dx=float(grid.derivs.dx),
    num_ghosts=NUM_GHOSTS
)

# Extract hydro variables
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

print(f"\nTesting RHS evaluation...")
print(f"  Conservative variables at center:")
print(f"    D[0] = {D[0]:.6e}")
print(f"    Sr[0] = {Sr[0]:.6e}")
print(f"    tau[0] = {tau[0]:.6e}")

# Warmup (JIT compilation)
print(f"\nWarmup (JIT compilation)...")
t0 = time.perf_counter()
rhs_D, rhs_Sr, rhs_tau = compute_hydro_rhs_cowling(
    D, Sr, tau, geom, 'polytropic', eos_params, atm_params, 'mp5', 'hll'
)
t_compile = time.perf_counter() - t0
print(f"  Compilation time: {t_compile*1000:.2f} ms")

# Timed evaluation
print(f"\nTimed RHS evaluation (5 runs)...")
times = []
for i in range(5):
    t0 = time.perf_counter()
    rhs_D, rhs_Sr, rhs_tau = compute_hydro_rhs_cowling(
        D, Sr, tau, geom, 'polytropic', eos_params, atm_params, 'mp5', 'hll'
    )
    t1 = time.perf_counter()
    times.append(t1 - t0)
    print(f"  Run {i+1}: {(t1-t0)*1000:.2f} ms")

mean_time = np.mean(times)
std_time = np.std(times)

print(f"\n  Mean time: {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")

# Check RHS magnitudes
interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
rhs_D_int = np.array(rhs_D[interior])
rhs_Sr_int = np.array(rhs_Sr[interior])
rhs_tau_int = np.array(rhs_tau[interior])

print(f"\nRHS magnitudes (interior):")
print(f"  |RHS_D|_max = {np.max(np.abs(rhs_D_int)):.6e}")
print(f"  |RHS_Sr|_max = {np.max(np.abs(rhs_Sr_int)):.6e}")
print(f"  |RHS_tau|_max = {np.max(np.abs(rhs_tau_int)):.6e}")

# ==================================================================
# EVOLUTION TEST - RK4 Integration
# ==================================================================
print(f"\n" + "="*70)
print("EVOLUTION TEST")
print("="*70)

# Define RK4 step function (calls JIT-compiled RHS internally)
def rk4_step_jax(D_in, Sr_in, tau_in, geom, dt, eos_params, atm_params):
    """Single RK4 timestep using JAX backend (RHS is JIT-compiled internally)."""
    # k1
    rhs_D1, rhs_Sr1, rhs_tau1 = compute_hydro_rhs_cowling(
        D_in, Sr_in, tau_in, geom, 'polytropic', eos_params, atm_params, 'mp5', 'hll'
    )

    # k2
    D2 = D_in + 0.5 * dt * rhs_D1
    Sr2 = Sr_in + 0.5 * dt * rhs_Sr1
    tau2 = tau_in + 0.5 * dt * rhs_tau1
    rhs_D2, rhs_Sr2, rhs_tau2 = compute_hydro_rhs_cowling(
        D2, Sr2, tau2, geom, 'polytropic', eos_params, atm_params, 'mp5', 'hll'
    )

    # k3
    D3 = D_in + 0.5 * dt * rhs_D2
    Sr3 = Sr_in + 0.5 * dt * rhs_Sr2
    tau3 = tau_in + 0.5 * dt * rhs_tau2
    rhs_D3, rhs_Sr3, rhs_tau3 = compute_hydro_rhs_cowling(
        D3, Sr3, tau3, geom, 'polytropic', eos_params, atm_params, 'mp5', 'hll'
    )

    # k4
    D4 = D_in + dt * rhs_D3
    Sr4 = Sr_in + dt * rhs_Sr3
    tau4 = tau_in + dt * rhs_tau3
    rhs_D4, rhs_Sr4, rhs_tau4 = compute_hydro_rhs_cowling(
        D4, Sr4, tau4, geom, 'polytropic', eos_params, atm_params, 'mp5', 'hll'
    )

    # Update
    D_new = D_in + (dt / 6.0) * (rhs_D1 + 2*rhs_D2 + 2*rhs_D3 + rhs_D4)
    Sr_new = Sr_in + (dt / 6.0) * (rhs_Sr1 + 2*rhs_Sr2 + 2*rhs_Sr3 + rhs_Sr4)
    tau_new = tau_in + (dt / 6.0) * (rhs_tau1 + 2*rhs_tau2 + 2*rhs_tau3 + rhs_tau4)

    return D_new, Sr_new, tau_new

# Evolution parameters
cfl_factor = 0.1
dt = cfl_factor * grid.derivs.dx
n_steps = 100  # Short test evolution

print(f"\nEvolving for {n_steps} steps (dt={dt:.6e})...")

# Warmup compilation
print("  Compiling RK4 step...")
t0_compile = time.perf_counter()
D_test, Sr_test, tau_test = rk4_step_jax(D, Sr, tau, geom, dt, eos_params, atm_params)
jax.block_until_ready((D_test, Sr_test, tau_test))
compile_time = time.perf_counter() - t0_compile
print(f"  Compilation time: {compile_time:.2f} s")

# Evolution
D_evol, Sr_evol, tau_evol = D, Sr, tau
t0_evol = time.perf_counter()

for step in range(n_steps):
    D_evol, Sr_evol, tau_evol = rk4_step_jax(
        D_evol, Sr_evol, tau_evol, geom, dt, eos_params, atm_params
    )
    if (step + 1) % 20 == 0:
        jax.block_until_ready(D_evol)
        print(f"  Step {step+1}/{n_steps}")

jax.block_until_ready((D_evol, Sr_evol, tau_evol))
evol_time = time.perf_counter() - t0_evol

print(f"\n✓ Evolution complete!")
print(f"  Wall time: {evol_time:.2f} s")
print(f"  Steps/second: {n_steps/evol_time:.0f}")
print(f"  Time per step: {evol_time/n_steps*1000:.2f} ms")

# Check for stability
D_initial = np.array(D)
D_final = np.array(D_evol)
Sr_final = np.array(Sr_evol)
tau_final = np.array(tau_evol)

if np.any(np.isnan(D_final)) or np.any(np.isnan(Sr_final)) or np.any(np.isnan(tau_final)):
    print("  ❌ NaNs detected - evolution unstable!")
else:
    print("  ✓ No NaNs - evolution stable")
    D_change = np.max(np.abs(D_final - D_initial)) / np.max(np.abs(D_initial))
    print(f"  Max relative change in D: {D_change:.3e}")

    # Check interior points only
    D_int_change = np.max(np.abs(D_final[interior] - D_initial[interior])) / np.max(np.abs(D_initial[interior]))
    print(f"  Max relative change (interior): {D_int_change:.3e}")

print("\n" + "="*70)
print("✓ JAX TOV EVOLUTION TEST COMPLETE")
print("="*70)
print(f"\nRHS evaluation: {mean_time*1000:.2f} ms (N={grid.N})")
print(f"Evolution: {n_steps} steps in {evol_time:.2f} s ({n_steps/evol_time:.0f} steps/s)")
