"""
Backend dispatcher for engrenage.

Allows switching between Numba (CPU) and JAX (CPU/GPU) backends via environment variable.

Usage:
    # Default: Numba backend
    python simulation.py

    # JAX backend (auto-detect GPU)
    ENGRENAGE_BACKEND=jax python simulation.py

    # Force specific backend
    ENGRENAGE_BACKEND=numba python simulation.py
    ENGRENAGE_BACKEND=jax-cpu python simulation.py
    ENGRENAGE_BACKEND=jax-gpu python simulation.py
"""

import os
from enum import Enum
from functools import lru_cache


class Backend(Enum):
    NUMBA = "numba"
    JAX_CPU = "jax-cpu"
    JAX_GPU = "jax-gpu"


@lru_cache(maxsize=1)
def get_backend() -> Backend:
    """
    Determine the compute backend based on environment variable.

    Returns:
        Backend enum value
    """
    backend = os.environ.get('ENGRENAGE_BACKEND', 'numba').lower()

    if backend == 'numba':
        return Backend.NUMBA

    if backend in ('jax', 'jax-auto'):
        try:
            import jax
            jax.config.update("jax_enable_x64", True)
            devices = jax.devices()
            gpu_available = any(d.platform == 'gpu' for d in devices)
            if gpu_available:
                print(f"[engrenage] JAX backend: GPU detected ({jax.devices('gpu')[0]})")
                return Backend.JAX_GPU
            else:
                print("[engrenage] JAX backend: CPU mode (no GPU detected)")
                return Backend.JAX_CPU
        except ImportError:
            print("[engrenage] WARNING: JAX not installed, falling back to Numba")
            return Backend.NUMBA
        except Exception as e:
            print(f"[engrenage] WARNING: JAX initialization failed ({e}), falling back to Numba")
            return Backend.NUMBA

    if backend == 'jax-cpu':
        try:
            import jax
            jax.config.update("jax_enable_x64", True)
            jax.config.update('jax_platform_name', 'cpu')
            print("[engrenage] JAX backend: CPU mode (forced)")
            return Backend.JAX_CPU
        except ImportError:
            print("[engrenage] WARNING: JAX not installed, falling back to Numba")
            return Backend.NUMBA

    if backend == 'jax-gpu':
        try:
            import jax
            jax.config.update("jax_enable_x64", True)
            devices = jax.devices('gpu')
            if not devices:
                print("[engrenage] WARNING: No GPU found, falling back to JAX CPU")
                return Backend.JAX_CPU
            print(f"[engrenage] JAX backend: GPU mode ({devices[0]})")
            return Backend.JAX_GPU
        except ImportError:
            print("[engrenage] WARNING: JAX not installed, falling back to Numba")
            return Backend.NUMBA
        except RuntimeError:
            print("[engrenage] WARNING: No GPU available, falling back to JAX CPU")
            return Backend.JAX_CPU

    print(f"[engrenage] WARNING: Unknown backend '{backend}', using Numba")
    return Backend.NUMBA


def is_jax_backend() -> bool:
    """Check if using any JAX backend."""
    return get_backend() in (Backend.JAX_CPU, Backend.JAX_GPU)


def is_gpu_backend() -> bool:
    """Check if using GPU backend."""
    return get_backend() == Backend.JAX_GPU


def get_array_module():
    """
    Get the appropriate array module (numpy or jax.numpy).

    Returns:
        module: numpy or jax.numpy depending on backend
    """
    if is_jax_backend():
        import jax.numpy as jnp
        return jnp
    else:
        import numpy as np
        return np
