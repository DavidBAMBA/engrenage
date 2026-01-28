"""
JAX backend for engrenage relativistic hydrodynamics.

This package provides JAX-native implementations of all hydro kernels,
enabling GPU acceleration and full-program JIT compilation via jax.lax.scan.

Modules:
    eos_jax: Pure-functional EOS (ideal gas + polytropic)
    atmosphere_jax: Functional floor application (no mutation)
    cons2prim_jax: Conservative-to-primitive solver (vmap + while_loop)
    reconstruction_jax: Spatial reconstruction (Minmod, MC, MP5, WENO5, WENO-Z)
    riemann_jax: HLL/LLF Riemann solvers
    valencia_jax: Full Valencia RHS (cons2prim + recon + riemann + sources)
"""
