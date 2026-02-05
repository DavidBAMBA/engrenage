"""
Comparación directa: NumPy/Numba vs JAX en TOV Evolution
Ejecuta el mismo código con ambos backends y compara resultados.
"""
import numpy as np
import sys
import os
import time
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

print("="*70)
print("COMPARACIÓN: NumPy/Numba vs JAX - TOV Evolution")
print("="*70)

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
r_max = 100.0
num_points = 400
K, Gamma = 100.0, 2.0
rho_central = 1.28e-3
n_steps = 50  # 50 pasos RK4 para comparar
dt = 0.005

print(f"\nConfiguración:")
print(f"  Grid: N = {num_points}, r_max = {r_max}")
print(f"  TOV: K = {K}, Γ = {Gamma}, ρc = {rho_central}")
print(f"  Evolución: {n_steps} pasos RK4, dt = {dt}")

# Setup común
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

print(f"Creando initial data...")
initial_state, _ = tov_id.create_initial_data_iso(
    tov, grid, background, eos, atmosphere=atmosphere,
    polytrope_K=K, polytrope_Gamma=Gamma, interp_order=11
)

# =============================================================================
# VERSIÓN 1: NumPy/Numba (ACTUAL)
# =============================================================================
print(f"\n{'='*70}")
print("BACKEND 1: NumPy/Numba (actual)")
print(f"{'='*70}")

state_numpy = initial_state.copy()
grid.fill_boundaries(state_numpy)

bssn_vars = BSSNVars(grid.N)
bssn_vars.set_bssn_vars(state_numpy[:NUM_BSSN_VARS, :])
bssn_d1 = grid.get_d1_metric_quantities(state_numpy)

# Fixed BSSN (Cowling)
bssn_fixed = state_numpy[:NUM_BSSN_VARS, :].copy()

def get_rhs_numpy(t, y, grid, background, hydro, bssn_fixed, bssn_d1):
    """RHS con NumPy/Numba (tu código actual)."""
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
    """RK4 con NumPy/Numba."""
    k1 = get_rhs_numpy(0, state_flat, grid, background, hydro, bssn_fixed, bssn_d1)

    state_2 = state_flat + 0.5 * dt * k1
    k2 = get_rhs_numpy(0, state_2, grid, background, hydro, bssn_fixed, bssn_d1)

    state_3 = state_flat + 0.5 * dt * k2
    k3 = get_rhs_numpy(0, state_3, grid, background, hydro, bssn_fixed, bssn_d1)

    state_4 = state_flat + dt * k3
    k4 = get_rhs_numpy(0, state_4, grid, background, hydro, bssn_fixed, bssn_d1)

    state_new = state_flat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return state_new

# Evolución con NumPy/Numba
state_flat_numpy = state_numpy.flatten()
t = 0.0

print("Evolucionando...")
t_start = time.perf_counter()

for step in range(n_steps):
    state_flat_numpy = rk4_step_numpy(state_flat_numpy, dt, grid, background,
                                       hydro, bssn_fixed, bssn_d1)
    t += dt

    if (step + 1) % 10 == 0:
        print(f"  Paso {step+1}/{n_steps}, t = {t:.3f}")

t_numpy_total = time.perf_counter() - t_start

state_final_numpy = state_flat_numpy.reshape((grid.NUM_VARS, grid.N))

print(f"\n✓ NumPy/Numba completado:")
print(f"  Tiempo total: {t_numpy_total:.3f} s")
print(f"  Tiempo/paso: {t_numpy_total/n_steps*1000:.1f} ms")

# Variables finales
D_final_numpy = state_final_numpy[NUM_BSSN_VARS + 0, grid.N//2]
Sr_final_numpy = state_final_numpy[NUM_BSSN_VARS + 1, grid.N//2]
tau_final_numpy = state_final_numpy[NUM_BSSN_VARS + 2, grid.N//2]

print(f"  Estado final (centro):")
print(f"    D = {D_final_numpy:.6e}")
print(f"    Sr = {Sr_final_numpy:.6e}")
print(f"    tau = {tau_final_numpy:.6e}")

# =============================================================================
# VERSIÓN 2: JAX
# =============================================================================
print(f"\n{'='*70}")
print("BACKEND 2: JAX")
print(f"{'='*70}")

# Setup JAX
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # CPU mejor para N=400

from source.matter.hydro.jax.valencia_jax import CowlingGeometry, compute_hydro_rhs_cowling

print(f"JAX device: {jax.devices()[0]}")

# Reset state
state_jax = initial_state.copy()
grid.fill_boundaries(state_jax)

bssn_vars_jax = BSSNVars(grid.N)
bssn_vars_jax.set_bssn_vars(state_jax[:NUM_BSSN_VARS, :])

# Crear geometría JAX
alpha = jnp.asarray(bssn_vars_jax.lapse)
phi = jnp.asarray(bssn_vars_jax.phi)
e6phi = jnp.exp(6.0 * phi)
gamma_rr = jnp.exp(4.0 * phi)
beta_r = jnp.zeros(grid.N)

geom_jax = CowlingGeometry(
    alpha=alpha, beta_r=beta_r,
    gamma_rr=gamma_rr, e6phi=e6phi,
    dx=float(grid.derivs.dx),
    num_ghosts=NUM_GHOSTS
)

eos_params = {'gamma': Gamma, 'K': K}
atm_params = {
    'rho_floor': float(atmosphere.rho_floor),
    'p_floor': float(atmosphere.p_floor),
    'v_max': float(atmosphere.v_max),
    'W_max': 10.0,
    'tol': 1e-12,
    'max_iter': 500,
}

def get_rhs_jax(t, y, geom_jax, eos_params, atm_params):
    """RHS con JAX."""
    state = y.reshape((grid.NUM_VARS, grid.N))

    D = jnp.asarray(state[NUM_BSSN_VARS + 0, :])
    Sr = jnp.asarray(state[NUM_BSSN_VARS + 1, :])
    tau = jnp.asarray(state[NUM_BSSN_VARS + 2, :])

    rhs_D, rhs_Sr, rhs_tau = compute_hydro_rhs_cowling(
        D, Sr, tau, geom_jax, 'polytropic', eos_params, atm_params, 'mp5', 'hll'
    )

    rhs = np.zeros((grid.NUM_VARS, grid.N))
    rhs[NUM_BSSN_VARS + 0, :] = np.array(rhs_D)
    rhs[NUM_BSSN_VARS + 1, :] = np.array(rhs_Sr)
    rhs[NUM_BSSN_VARS + 2, :] = np.array(rhs_tau)

    return rhs.flatten()

def rk4_step_jax(state_flat, dt, geom_jax, eos_params, atm_params):
    """RK4 con JAX."""
    k1 = get_rhs_jax(0, state_flat, geom_jax, eos_params, atm_params)

    state_2 = state_flat + 0.5 * dt * k1
    k2 = get_rhs_jax(0, state_2, geom_jax, eos_params, atm_params)

    state_3 = state_flat + 0.5 * dt * k2
    k3 = get_rhs_jax(0, state_3, geom_jax, eos_params, atm_params)

    state_4 = state_flat + dt * k3
    k4 = get_rhs_jax(0, state_4, geom_jax, eos_params, atm_params)

    state_new = state_flat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return state_new

# Warmup (compilación)
print("Compilando JAX (primera llamada)...")
state_flat_jax = state_jax.flatten()
t_compile_start = time.perf_counter()
state_flat_jax = rk4_step_jax(state_flat_jax, dt, geom_jax, eos_params, atm_params)
t_compile = time.perf_counter() - t_compile_start
print(f"  Compilación: {t_compile*1000:.0f} ms")

# Reset y evolución con JAX
state_jax = initial_state.copy()
state_flat_jax = state_jax.flatten()
t = 0.0

print("Evolucionando...")
t_start = time.perf_counter()

for step in range(n_steps):
    state_flat_jax = rk4_step_jax(state_flat_jax, dt, geom_jax, eos_params, atm_params)
    t += dt

    if (step + 1) % 10 == 0:
        print(f"  Paso {step+1}/{n_steps}, t = {t:.3f}")

t_jax_total = time.perf_counter() - t_start

state_final_jax = state_flat_jax.reshape((grid.NUM_VARS, grid.N))

print(f"\n✓ JAX completado:")
print(f"  Tiempo total: {t_jax_total:.3f} s")
print(f"  Tiempo/paso: {t_jax_total/n_steps*1000:.1f} ms")

# Variables finales
D_final_jax = state_final_jax[NUM_BSSN_VARS + 0, grid.N//2]
Sr_final_jax = state_final_jax[NUM_BSSN_VARS + 1, grid.N//2]
tau_final_jax = state_final_jax[NUM_BSSN_VARS + 2, grid.N//2]

print(f"  Estado final (centro):")
print(f"    D = {D_final_jax:.6e}")
print(f"    Sr = {Sr_final_jax:.6e}")
print(f"    tau = {tau_final_jax:.6e}")

# =============================================================================
# COMPARACIÓN FINAL
# =============================================================================
print(f"\n{'='*70}")
print("COMPARACIÓN FINAL")
print(f"{'='*70}")

speedup = t_numpy_total / t_jax_total

print(f"\n  {'Backend':<20} {'Tiempo total':<15} {'Tiempo/paso':<15}")
print(f"  {'-'*50}")
print(f"  {'NumPy/Numba':<20} {t_numpy_total:<15.3f} s {t_numpy_total/n_steps*1000:<15.1f} ms")
print(f"  {'JAX':<20} {t_jax_total:<15.3f} s {t_jax_total/n_steps*1000:<15.1f} ms")

print(f"\n  {'='*50}")
print(f"  🚀 SPEEDUP: {speedup:.2f}x")
print(f"  {'='*50}")

if speedup > 2.0:
    print(f"\n  ✅ ¡JAX es {speedup:.1f}x MÁS RÁPIDO! Mejora significativa.")
elif speedup > 1.2:
    print(f"\n  ✓ JAX es {speedup:.1f}x más rápido. Mejora moderada.")
else:
    print(f"\n  ⚠ JAX es {speedup:.1f}x. Mejora marginal (overhead de conversión).")

# Verificar correctness
print(f"\n{'='*70}")
print("VERIFICACIÓN DE CORRECTNESS")
print(f"{'='*70}")

rel_err_D = abs(D_final_jax - D_final_numpy) / abs(D_final_numpy)
rel_err_Sr = abs(Sr_final_jax - Sr_final_numpy) / abs(Sr_final_numpy + 1e-30)
rel_err_tau = abs(tau_final_jax - tau_final_numpy) / abs(tau_final_numpy)

print(f"\n  Error relativo en estado final:")
print(f"    D:   {rel_err_D:.3e}")
print(f"    Sr:  {rel_err_Sr:.3e}")
print(f"    tau: {rel_err_tau:.3e}")

if rel_err_D < 1e-6 and rel_err_tau < 1e-6:
    print(f"\n  ✅ Resultados coinciden (error < 1e-6)")
elif rel_err_D < 1e-3 and rel_err_tau < 1e-3:
    print(f"\n  ✓ Resultados razonablemente cercanos (error < 1e-3)")
else:
    print(f"\n  ⚠ Diferencias significativas - revisar implementación")

print(f"\n{'='*70}")
print("RESUMEN")
print(f"{'='*70}")
print(f"""
Para {n_steps} pasos RK4 en TOV evolution (N={num_points}):

  NumPy/Numba: {t_numpy_total:.2f} s
  JAX:         {t_jax_total:.2f} s

  ⚡ Tiempo ahorrado: {t_numpy_total - t_jax_total:.2f} s ({speedup:.1f}x más rápido)

  Para evolución completa (1000 pasos):
    NumPy/Numba: ~{t_numpy_total*1000/n_steps:.1f} s
    JAX:         ~{t_jax_total*1000/n_steps:.1f} s
    Ahorro:      ~{(t_numpy_total-t_jax_total)*1000/n_steps:.1f} s
""")

print(f"{'='*70}")
print("✅ Comparación completa!")
print(f"{'='*70}")
