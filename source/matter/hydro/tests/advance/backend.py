"""
Backend dispatcher for engrenage hydro module.

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


def get_backend():
    """
    Determine the compute backend based on environment variable.

    Returns:
        str: One of 'numba', 'jax-cpu', or 'jax-gpu'
    """
    backend = os.environ.get('ENGRENAGE_BACKEND', 'numba').lower()

    if backend == 'numba':
        return 'numba'

    if backend in ('jax', 'jax-auto'):
        # Auto-detect JAX GPU availability
        try:
            import jax
            devices = jax.devices()
            gpu_available = any(d.platform == 'gpu' for d in devices)
            result = 'jax-gpu' if gpu_available else 'jax-cpu'
            if gpu_available:
                print(f"[engrenage] JAX backend: GPU detected ({jax.devices('gpu')[0]})")
            else:
                print("[engrenage] JAX backend: CPU mode (no GPU detected)")
            return result
        except ImportError:
            print("[engrenage] WARNING: JAX not installed, falling back to Numba")
            return 'numba'
        except Exception as e:
            print(f"[engrenage] WARNING: JAX initialization failed ({e}), falling back to Numba")
            return 'numba'

    if backend == 'jax-cpu':
        try:
            import jax
            # Force CPU
            jax.config.update('jax_platform_name', 'cpu')
            print("[engrenage] JAX backend: CPU mode (forced)")
            return 'jax-cpu'
        except ImportError:
            print("[engrenage] WARNING: JAX not installed, falling back to Numba")
            return 'numba'

    if backend == 'jax-gpu':
        try:
            import jax
            devices = jax.devices('gpu')
            if not devices:
                print("[engrenage] WARNING: No GPU found, falling back to JAX CPU")
                return 'jax-cpu'
            print(f"[engrenage] JAX backend: GPU mode ({devices[0]})")
            return 'jax-gpu'
        except ImportError:
            print("[engrenage] WARNING: JAX not installed, falling back to Numba")
            return 'numba'
        except RuntimeError:
            print("[engrenage] WARNING: No GPU available, falling back to JAX CPU")
            return 'jax-cpu'

    # Unknown backend, default to numba
    print(f"[engrenage] WARNING: Unknown backend '{backend}', using Numba")
    return 'numba'


# Initialize backend on module import
BACKEND = get_backend()


def is_jax_backend():
    """Check if using JAX backend."""
    return 'jax' in BACKEND


def is_gpu_backend():
    """Check if using GPU backend."""
    return BACKEND == 'jax-gpu'


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
