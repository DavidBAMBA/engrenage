"""Test JAX with connections enabled to trigger the static branch issue."""
import jax
import jax.numpy as jnp
import numpy as np
import sys
sys.path.insert(0, '/home/davidbamba/repositories/engrenage')

jax.config.update("jax_enable_x64", True)

from source.matter.hydro.jax.valencia_jax import CowlingGeometry, compute_hydro_rhs_cowling

# Geometry with connections enabled
N = 100
NG = 3

# Basic geometry
alpha = jnp.ones(N)
beta_r = jnp.zeros(N)
gamma_rr = jnp.ones(N)
e6phi = jnp.ones(N)

# Add Christoffel symbols to trigger has_connections=True
hat_christoffel = jnp.zeros((N, 3, 3, 3))

# Create geometry WITH connections
geom_with_conn = CowlingGeometry(
    alpha=alpha, beta_r=beta_r,
    gamma_rr=gamma_rr, e6phi=e6phi,
    dx=0.5, num_ghosts=NG,
    hat_christoffel=hat_christoffel  # This triggers has_connections=True
)

# Dummy state
D = jnp.ones(N) * 1e-4
Sr = jnp.zeros(N)
tau = jnp.ones(N) * 1e-5

eos_params = {'gamma': 2.0, 'K': 100.0}
atm_params = {'rho_floor': 1e-12, 'p_floor': 1e-12, 'v_max': 0.99,
              'W_max': 10.0, 'tol': 1e-12, 'max_iter': 500}

print(f"Geometry has_connections: {geom_with_conn.has_connections}")
print(f"Geometry has_sources: {geom_with_conn.has_sources}")
print("\nTesting JAX compilation with connections enabled...")

try:
    # First call - should compile successfully
    rhs_D, rhs_Sr, rhs_tau = compute_hydro_rhs_cowling(
        D, Sr, tau, geom_with_conn, 'polytropic', eos_params, atm_params, 'wenoz', 'hll'
    )
    print(f"✓ First call SUCCESS: {rhs_D.shape}")

    # Now try with geometry WITHOUT connections
    geom_no_conn = CowlingGeometry(
        alpha=alpha, beta_r=beta_r,
        gamma_rr=gamma_rr, e6phi=e6phi,
        dx=0.5, num_ghosts=NG
        # No hat_christoffel - has_connections=False
    )

    print(f"\nNow testing with has_connections=False...")
    rhs_D2, rhs_Sr2, rhs_tau2 = compute_hydro_rhs_cowling(
        D, Sr, tau, geom_no_conn, 'polytropic', eos_params, atm_params, 'wenoz', 'hll'
    )
    print(f"✓ Second call SUCCESS: {rhs_D2.shape}")
    print("\n✓ Both geometries work! The pytree aux_data handles this correctly.")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
