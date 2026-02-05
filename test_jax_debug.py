"""Debug JAX RHS computation step by step."""
import jax
import jax.numpy as jnp
import numpy as np
import sys
sys.path.insert(0, '/home/davidbamba/repositories/engrenage')

jax.config.update("jax_enable_x64", True)

from source.matter.hydro.jax.valencia_jax import CowlingGeometry, compute_hydro_rhs_cowling

# Simple test: flat spacetime, uniform state
N = 100
NG = 3

# Flat spacetime
alpha = jnp.ones(N)
beta_r = jnp.zeros(N)
gamma_rr = jnp.ones(N)
e6phi = jnp.ones(N)

# Simple geometry (no connections, no sources)
geom = CowlingGeometry(
    alpha=alpha, beta_r=beta_r,
    gamma_rr=gamma_rr, e6phi=e6phi,
    dx=0.5, num_ghosts=NG
)

# Uniform, subsonic state
rho_0 = 1e-4
p_0 = 1e-5
v_0 = 0.1

# Convert to conservatives (manual for polytropic)
K, Gamma = 100.0, 2.0
eps_0 = K * rho_0 ** (Gamma - 1) / (Gamma - 1)
h_0 = 1.0 + eps_0 + p_0 / rho_0
W_0 = 1.0 / np.sqrt(1.0 - v_0**2)

D = jnp.ones(N) * (rho_0 * W_0)
Sr = jnp.ones(N) * (rho_0 * h_0 * W_0**2 * v_0 * gamma_rr[0])
tau = jnp.ones(N) * (rho_0 * h_0 * W_0**2 - p_0 - rho_0 * W_0)

eos_params = {'gamma': Gamma, 'K': K}
atm_params = {'rho_floor': 1e-12, 'p_floor': 1e-12, 'v_max': 0.99,
              'W_max': 10.0, 'tol': 1e-12, 'max_iter': 500}

print("Testing uniform state RHS...")
print(f"  Input D: {D[N//2]:.6e}")
print(f"  Input Sr: {Sr[N//2]:.6e}")
print(f"  Input tau: {tau[N//2]:.6e}")

try:
    rhs_D, rhs_Sr, rhs_tau = compute_hydro_rhs_cowling(
        D, Sr, tau, geom, 'polytropic', eos_params, atm_params, 'mp5', 'hll'
    )

    print(f"\n✓ RHS computation successful")
    print(f"  RHS D: {rhs_D[N//2]:.6e}")
    print(f"  RHS Sr: {rhs_Sr[N//2]:.6e}")
    print(f"  RHS tau: {rhs_tau[N//2]:.6e}")

    # For uniform state in flat spacetime, RHS should be ~zero
    interior = slice(NG, -NG)
    max_rhs_D = np.max(np.abs(np.array(rhs_D[interior])))
    max_rhs_Sr = np.max(np.abs(np.array(rhs_Sr[interior])))
    max_rhs_tau = np.max(np.abs(np.array(rhs_tau[interior])))

    print(f"\nMax RHS magnitudes (interior):")
    print(f"  D: {max_rhs_D:.6e}")
    print(f"  Sr: {max_rhs_Sr:.6e}")
    print(f"  tau: {max_rhs_tau:.6e}")

    if max_rhs_D < 1e-10 and max_rhs_Sr < 1e-10 and max_rhs_tau < 1e-10:
        print("\n✓ PASS: Uniform state gives ~zero RHS (as expected)")
    else:
        print("\n⚠ WARNING: Uniform state should have ~zero RHS in flat spacetime")

except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
