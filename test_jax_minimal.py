"""Minimal test to isolate JAX compilation errors."""
import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, '/home/davidbamba/repositories/engrenage')

jax.config.update("jax_enable_x64", True)

from source.matter.hydro.jax.valencia_jax import CowlingGeometry, compute_hydro_rhs_cowling

# Minimal geometry
N = 100
geom = CowlingGeometry(
    alpha=jnp.ones(N), beta_r=jnp.zeros(N),
    gamma_rr=jnp.ones(N), e6phi=jnp.ones(N),
    dx=0.5, num_ghosts=3
)

# Dummy state
D = jnp.ones(N) * 1e-4
Sr = jnp.zeros(N)
tau = jnp.ones(N) * 1e-5

eos_params = {'gamma': 2.0, 'K': 100.0}
atm_params = {'rho_floor': 1e-12, 'p_floor': 1e-12, 'v_max': 0.99,
              'W_max': 10.0, 'tol': 1e-12, 'max_iter': 500}

print("Testing JAX compilation...")
try:
    rhs_D, rhs_Sr, rhs_tau = compute_hydro_rhs_cowling(
        D, Sr, tau, geom, 'polytropic', eos_params, atm_params, 'wenoz', 'hll'
    )
    print(f"✓ SUCCESS: {rhs_D.shape}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
