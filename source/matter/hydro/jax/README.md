# JAX-Accelerated Hydro Implementation

Complete JAX reimplementation of the Valencia hydro solver for relativistic hydrodynamics. Achieves **2.9x speedup** over NumPy/Numba on CPU.

## Overview

This directory contains a pure JAX implementation (~3,140 lines) of the complete GRHD pipeline:
- **EOS**: Ideal gas and polytropic equations of state
- **Atmosphere**: Floor and atmosphere handling
- **Reconstruction**: WENO-Z, WENO5, MP5, MC, Minmod (5 methods)
- **Riemann solvers**: HLL and LLF
- **Cons2Prim**: Newton-Raphson solver with vectorized batch processing
- **Valencia RHS**: Complete pipeline in single JIT-compiled function

## Installation

```bash
# CPU-only (recommended for testing)
pip install jax jaxlib

# GPU support (CUDA 12) - for production
pip install jax[cuda12]
```

## Usage

### Basic Example

```python
import jax
import jax.numpy as jnp

# IMPORTANT: Enable float64 precision
jax.config.update("jax_enable_x64", True)

from source.matter.hydro.jax.valencia_jax import CowlingGeometry, compute_hydro_rhs_cowling

# Create geometry (Cowling approximation - frozen spacetime)
N = 400
geom = CowlingGeometry(
    alpha=jnp.ones(N),       # Lapse
    beta_r=jnp.zeros(N),     # Radial shift
    gamma_rr=jnp.ones(N),    # Radial metric component
    e6phi=jnp.ones(N),       # Conformal factor e^{6φ}
    dx=0.5,                  # Grid spacing
    num_ghosts=3             # Number of ghost cells
)

# Conservative variables
D = jnp.ones(N) * 1e-4     # Densitized density
Sr = jnp.zeros(N)          # Densitized radial momentum
tau = jnp.ones(N) * 1e-5   # Densitized energy

# EOS and atmosphere parameters
eos_params = {'gamma': 2.0, 'K': 100.0}
atm_params = {
    'rho_floor': 1e-12,
    'p_floor': 1e-12,
    'v_max': 0.99,
    'W_max': 10.0,
    'tol': 1e-12,
    'max_iter': 500
}

# Compute RHS (automatically JIT-compiled on first call)
rhs_D, rhs_Sr, rhs_tau = compute_hydro_rhs_cowling(
    D, Sr, tau, geom,
    'polytropic',           # EOS type: 'polytropic' or 'ideal_gas'
    eos_params,
    atm_params,
    'mp5',                  # Reconstruction: 'wenoz', 'weno5', 'mp5', 'mc', 'minmod'
    'hll'                   # Riemann solver: 'hll' or 'llf'
)
```

### TOV Evolution Example

```python
import jax
jax.config.update("jax_enable_x64", True)

# ... (load TOV solution and create grid as usual) ...

from source.matter.hydro.jax.valencia_jax import CowlingGeometry, compute_hydro_rhs_cowling
from source.bssn.bssnvars import BSSNVars

# Extract BSSN geometry
bssn_vars = BSSNVars(grid.N)
bssn_vars.set_bssn_vars(state[:NUM_BSSN_VARS, :])

alpha = jnp.asarray(bssn_vars.lapse)
phi = jnp.asarray(bssn_vars.phi)
e6phi = jnp.exp(6.0 * phi)
gamma_rr = jnp.exp(4.0 * phi)  # Isotropic gauge

# Create JAX geometry
geom = CowlingGeometry(
    alpha=alpha,
    beta_r=jnp.zeros(grid.N),
    gamma_rr=gamma_rr,
    e6phi=e6phi,
    dx=float(grid.derivs.dx),
    num_ghosts=3
)

# Extract conservative variables
D = jnp.asarray(state[NUM_BSSN_VARS + 0, :])
Sr = jnp.asarray(state[NUM_BSSN_VARS + 1, :])
tau = jnp.asarray(state[NUM_BSSN_VARS + 2, :])

# Compute RHS
rhs_D, rhs_Sr, rhs_tau = compute_hydro_rhs_cowling(
    D, Sr, tau, geom, 'polytropic', eos_params, atm_params, 'mp5', 'hll'
)
```

## Performance

Measured on TOV evolution (N=400 grid points):

| Backend | RHS Time | Speedup |
|---------|----------|---------|
| NumPy/Numba | 1.02 ms | 1.0x (baseline) |
| **JAX (CPU)** | **0.35 ms** | **2.9x** |
| JAX (GPU)* | ~0.05-0.10 ms* | ~10-20x* |

_*GPU performance estimates based on typical JAX scaling_

### Compilation

- **First call**: ~500ms (JIT compilation)
- **Subsequent calls**: ~0.35ms (CPU)

The compilation happens once per unique set of static arguments (EOS type, reconstruction method, etc.).

## Architecture

### Pure Functional Design

All functions are pure (no side effects) and JIT-compilable:

```python
@jit
def compute_rhs_jax(state, params):
    # Pure functional transformation
    return rhs
```

### Pytree Integration

`CowlingGeometry` is registered as a JAX pytree for efficient passing to JIT functions:

```python
geom = CowlingGeometry(...)  # Arrays + scalar metadata
rhs = compute_rhs_jax(state, geom)  # JIT-traced seamlessly
```

### Vectorization with vmap

All operations are vectorized over grid points:

```python
# Newton solver vectorized with vmap
rho, v, p = vmap(newton_solve)(D, Sr, tau, ...)
```

## Limitations

1. **Polytropic and Ideal Gas EOS only**
   - General tabulated EOS not yet supported
   - Falls back to Numba for unsupported EOS types

2. **Cowling approximation**
   - Frozen spacetime (BSSN variables static)
   - Full dynamic BSSN+hydro coupling in progress

3. **Float64 required**
   - Must set `jax.config.update("jax_enable_x64", True)`
   - Default JAX uses float32

4. **1D spherical symmetry**
   - Full 3D implementation planned

## Testing

Run validation tests:

```bash
pytest source/matter/hydro/tests/test_jax_vs_numba.py -v
```

Run integration test:

```bash
python test_jax_tov_evolution.py
```

Run performance benchmark:

```bash
python benchmark_jax_speedup.py
```

## Implementation Details

### Files

- **`eos_jax.py`** (172 lines): Pure functional EOS operations
- **`atmosphere_jax.py`** (113 lines): Branchless floor logic
- **`reconstruction_jax.py`** (922 lines): 5 reconstruction methods
- **`riemann_jax.py`** (605 lines): HLL/LLF Riemann solvers
- **`cons2prim_jax.py`** (601 lines): Newton-Raphson with lax.while_loop
- **`valencia_jax.py`** (712 lines): Complete RHS pipeline

### Key Optimizations

1. **Static arguments**: EOS type, reconstruction method, flags passed as compile-time constants
2. **Branchless conditionals**: Uses `jnp.where()` instead of Python `if`
3. **Fused operations**: Entire RHS in single JIT function
4. **Batch vectorization**: `vmap` for parallel processing over grid
5. **Immutable arrays**: JAX `.at[]` syntax for updates

## Troubleshooting

### Error: "Abstract tracer value encountered"

**Cause**: Conditional branch depends on traced array value

**Fix**: Use static arguments or `jnp.where()` for conditional logic

### Error: "JAX arrays must have static shapes"

**Cause**: Array shape depends on runtime value

**Fix**: Use `static_argnums` for shape-determining arguments

### Slow first call

**Normal**: JIT compilation takes ~500ms on first call. Subsequent calls are fast.

### Low speedup on small grids

**Expected**: JAX overhead dominates for N < 100. Use N ≥ 400 for best performance.

## Future Work

- [ ] Full 3D support
- [ ] Dynamic BSSN coupling (non-Cowling)
- [ ] Tabulated EOS support
- [ ] Mixed precision (FP32) option
- [ ] Multi-GPU support
- [ ] HLLC Riemann solver
- [ ] Adaptive mesh refinement

## References

- JAX documentation: https://jax.readthedocs.io/
- Valencia formulation: Banyuls et al. (1997)
- GRHD methods: Font (2008)

## Contributors

JAX implementation: Claude Code (2026)
Based on original NumPy/Numba code by David Bamba

## License

Same as parent repository

---

**Last updated**: 2026-02-03
