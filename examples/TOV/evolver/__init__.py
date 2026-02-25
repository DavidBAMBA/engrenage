"""
Evolution engines for TOV star simulations.

Import directly from submodules to avoid loading both backends:

    from examples.TOV.evolver.numba import evolve_numba
    from examples.TOV.evolver.jax import evolve_jax
"""

__all__ = ["evolve_numba", "evolve_jax"]
