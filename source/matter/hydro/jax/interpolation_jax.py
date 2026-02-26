"""
JAX-native interpolation utilities for GRHD.

Provides 4th-order (degree-4, 5-point) Lagrange interpolation of metric
quantities from cell centres to cell faces. JAX-compatible version of
interpolation_utils.py.
"""

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Pre-computed Lagrange weights (exact fractions, converted to float64)
# ---------------------------------------------------------------------------

_W_INTERIOR = jnp.array([3.0, -20.0, 90.0, 60.0, -5.0]) / 128.0
_W_FACE0 = jnp.array([35.0, 140.0, -70.0, 28.0, -5.0]) / 128.0
_W_FACE1 = jnp.array([-5.0, 60.0, 90.0, -20.0, 3.0]) / 128.0
_W_FACE_LAST = jnp.array([-5.0, 28.0, -70.0, 140.0, 35.0]) / 128.0


def lagrange_interpolate_to_faces_4th_order(field):
    """
    Interpolate a 1-D cell-centred field to cell faces using 4th-order
    (degree-4, 5-point) Lagrange interpolation.

    For N cell centres the function returns N-1 face values.

    Interior faces use the symmetric-as-possible centred stencil
    [f-2, f-1, f, f+1, f+2] with weights [3,-20,90,60,-5]/128.

    The two faces nearest each boundary use a one-sided stencil that
    stays within the domain, with weights derived from the same Lagrange
    construction.

    Args:
        field: (N,) array of cell-centred values.

    Returns:
        field_face: (N-1,) array of face-interpolated values.
    """
    field = jnp.asarray(field, dtype=jnp.float64)
    N = field.size

    if N < 2:
        raise ValueError(f"Field must have at least 2 points, got {N}.")

    if N < 5:
        return 0.5 * (field[:-1] + field[1:])

    field_face = jnp.empty(N - 1, dtype=jnp.float64)

    field_face = field_face.at[2:N-2].set(
        _W_INTERIOR[0] * field[0:N-4]
        + _W_INTERIOR[1] * field[1:N-3]
        + _W_INTERIOR[2] * field[2:N-2]
        + _W_INTERIOR[3] * field[3:N-1]
        + _W_INTERIOR[4] * field[4:N]
    )

    field_face = field_face.at[0].set(jnp.dot(_W_FACE0, field[0:5]))
    field_face = field_face.at[1].set(jnp.dot(_W_FACE1, field[0:5]))
    field_face = field_face.at[N-2].set(jnp.dot(_W_FACE_LAST, field[N-5:N]))

    return field_face


def lagrange_interpolate_vector_to_faces(vector_field):
    """
    Interpolate a vector field (N, 3) to cell faces using 4th-order Lagrange.

    Each spatial component is interpolated independently.

    Args:
        vector_field: (N, 3) array of cell-centred vector values.

    Returns:
        vector_face: (N-1, 3) array of face-interpolated values.
    """
    vector_field = jnp.asarray(vector_field, dtype=jnp.float64)
    if vector_field.ndim != 2 or vector_field.shape[1] != 3:
        raise ValueError(
            f"Expected vector field with shape (N, 3), got {vector_field.shape}."
        )

    N = vector_field.shape[0]
    vector_face = jnp.empty((N - 1, 3), dtype=jnp.float64)

    for d in range(3):
        vector_face = vector_face.at[:, d].set(
            lagrange_interpolate_to_faces_4th_order(vector_field[:, d])
        )

    return vector_face


def lagrange_interpolate_tensor_to_faces(tensor_field):
    """
    Interpolate a rank-2 tensor field (N, 3, 3) to cell faces using
    4th-order Lagrange.

    Each independent component is interpolated separately.

    Args:
        tensor_field: (N, 3, 3) array of cell-centred tensor values.

    Returns:
        tensor_face: (N-1, 3, 3) array of face-interpolated values.
    """
    tensor_field = jnp.asarray(tensor_field, dtype=jnp.float64)
    if tensor_field.ndim != 3 or tensor_field.shape[1:] != (3, 3):
        raise ValueError(
            f"Expected tensor field with shape (N, 3, 3), got {tensor_field.shape}."
        )

    N = tensor_field.shape[0]
    tensor_face = jnp.empty((N - 1, 3, 3), dtype=jnp.float64)

    for i in range(3):
        for j in range(3):
            tensor_face = tensor_face.at[:, i, j].set(
                lagrange_interpolate_to_faces_4th_order(tensor_field[:, i, j])
            )

    return tensor_face
