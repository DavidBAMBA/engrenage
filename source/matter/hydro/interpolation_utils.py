"""
Interpolation utilities for GRHD.

Provides higher-order interpolation methods for metric quantities at cell faces.
"""

import numpy as np


def lagrange_interpolate_to_faces_3rd_order(field, boundary_mode="outflow"):
    """
    Interpolate 1D field to cell faces using 3rd-order Lagrange polynomial.

    Uses 4-point stencil (cubic interpolation):
    - Interior faces: centered stencil [i-1, i, i+1, i+2]
    - Boundary faces: adjusted stencil to stay in domain

    For uniform grid with face at midpoint between cells i and i+1,
    the Lagrange weights are: [-1/16, 9/16, 9/16, -1/16]

    Args:
        field: (N,) array of cell-centered values
        boundary_mode: "outflow" or "reflecting" (currently not used, reserved for future)

    Returns:
        field_face: (N-1,) array of face-interpolated values

    References:
        - Lagrange polynomial interpolation:
          f(x_face) = Σ_i L_i(x_face) * f_i
        - For uniform grid at midpoint: optimized pre-computed weights
    """
    N = len(field)
    field_face = np.zeros(N - 1)

    # Pre-computed Lagrange weights for face at midpoint (ξ = 0.5)
    # Stencil: [i-1, i, i+1, i+2] → positions [-1, 0, 1, 2]
    # Weights: [-1/16, 9/16, 9/16, -1/16]
    w = np.array([-1.0/16.0, 9.0/16.0, 9.0/16.0, -1.0/16.0])

    # Interior faces (have full 4-point stencil available)
    for i in range(1, N - 2):
        # Face between cells i and i+1
        # Stencil: [field[i-1], field[i], field[i+1], field[i+2]]
        stencil = field[i-1:i+3]
        field_face[i] = np.dot(w, stencil)

    # First face (i=0, between cells 0 and 1)
    # Use forward-biased stencil: [field[0], field[1], field[2], field[3]]
    if N >= 4:
        stencil = field[0:4]
        field_face[0] = np.dot(w, stencil)
    else:
        # Fallback to linear for small arrays
        field_face[0] = 0.5 * (field[0] + field[1])

    # Last face (i=N-2, between cells N-2 and N-1)
    # Use backward-biased stencil: [field[N-4], field[N-3], field[N-2], field[N-1]]
    if N >= 4:
        stencil = field[N-4:N]
        field_face[N-2] = np.dot(w, stencil)
    else:
        # Fallback to linear for small arrays
        field_face[N-2] = 0.5 * (field[N-2] + field[N-1])

    return field_face


def lagrange_interpolate_vector_to_faces(vector_field, boundary_mode="outflow"):
    """
    Interpolate vector field (N, 3) to faces using 3rd-order Lagrange.

    Applies 1D Lagrange interpolation to each component independently.

    Args:
        vector_field: (N, 3) array of cell-centered vector values
        boundary_mode: "outflow" or "reflecting"

    Returns:
        vector_face: (N-1, 3) array of face-interpolated vector values
    """
    N, ndim = vector_field.shape
    assert ndim == 3, "Expected vector field with 3 components"

    vector_face = np.zeros((N - 1, 3))

    # Interpolate each component independently
    for d in range(3):
        vector_face[:, d] = lagrange_interpolate_to_faces_3rd_order(
            vector_field[:, d], boundary_mode
        )

    return vector_face


def lagrange_interpolate_tensor_to_faces(tensor_field, boundary_mode="outflow"):
    """
    Interpolate rank-2 tensor (N, 3, 3) to faces using 3rd-order Lagrange.

    Applies 1D Lagrange interpolation to each component independently.

    Args:
        tensor_field: (N, 3, 3) array of cell-centered tensor values
        boundary_mode: "outflow" or "reflecting"

    Returns:
        tensor_face: (N-1, 3, 3) array of face-interpolated tensor values
    """
    N = tensor_field.shape[0]
    assert tensor_field.shape == (N, 3, 3), "Expected tensor field with shape (N, 3, 3)"

    tensor_face = np.zeros((N - 1, 3, 3))

    # Interpolate each component independently
    for i in range(3):
        for j in range(3):
            tensor_face[:, i, j] = lagrange_interpolate_to_faces_3rd_order(
                tensor_field[:, i, j], boundary_mode
            )

    return tensor_face
