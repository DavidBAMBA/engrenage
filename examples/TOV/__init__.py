"""
TOV Star Evolution Package

Modules:
    - config: Configuration classes
    - evolver: Evolution engines (Numba, JAX)
    - utils: Data management and I/O
    - plotting: Visualization utilities
"""

from .config import TOVConfig

__all__ = ["TOVConfig"]
