"""
Interpolation utilities for GRHD.

Provides 4th-order (degree-4, 5-point) Lagrange interpolation of metric
quantities from cell centres to cell faces.

STENCIL AND WEIGHT DERIVATION
==============================
For a uniform grid with cell centres at integer positions, the face between
cells i and i+1 lies at i + 0.5.  A 5-point stencil [s, s+1, s+2, s+3, s+4]
has its nodes at positions [0,1,2,3,4] relative to s, and the face sits at
relative position  xi = (i + 0.5) - s.

Lagrange weights w_k are evaluated analytically:

  w_k(xi) = prod_{j != k} (xi - j) / prod_{j != k} (k - j)

The four cases that arise on a finite grid of N cells are:

  Interior faces (f = 2 … N-3):
      stencil [f-2 … f+2], xi = 2.5
      w = [3, -20, 90, 60, -5] / 128

  Face 0 (f = 0):
      stencil [0 … 4], xi = 0.5
      w = [35, 140, -70, 28, -5] / 128

  Face 1 (f = 1):
      stencil [0 … 4], xi = 1.5
      w = [-5, 60, 90, -20, 3] / 128

  Face N-2 (f = N-2, last face):
      stencil [N-5 … N-1], xi = 3.5
      w = [-5, 28, -70, 140, 35] / 128

All four weight sets sum to 1 (verified) and the scheme is 5th-order accurate
in h for smooth functions (degree-4 polynomial, O(h^5) error).

Note: face N-3 is handled correctly by the interior formula with stencil
[N-5 … N-1] (xi = 2.5), so only three special cases are needed.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Pre-computed Lagrange weights (exact fractions, converted to float64)
# ---------------------------------------------------------------------------

# Interior faces: stencil [f-2, f-1, f, f+1, f+2], evaluate at xi = 2.5
_W_INTERIOR = np.array([3.0, -20.0, 90.0, 60.0, -5.0]) / 128.0

# Face 0: stencil [0,1,2,3,4], evaluate at xi = 0.5
_W_FACE0 = np.array([35.0, 140.0, -70.0, 28.0, -5.0]) / 128.0

# Face 1: stencil [0,1,2,3,4], evaluate at xi = 1.5
_W_FACE1 = np.array([-5.0, 60.0, 90.0, -20.0, 3.0]) / 128.0

# Face N-2 (last face): stencil [N-5,...,N-1], evaluate at xi = 3.5
_W_FACE_LAST = np.array([-5.0, 28.0, -70.0, 140.0, 35.0]) / 128.0


def lagrange_interpolate_to_faces_4th_order(field, boundary_mode="outflow"):
    """
    Interpolate a 1-D cell-centred field to cell faces using 4th-order
    (degree-4, 5-point) Lagrange interpolation.

    For N cell centres the function returns N-1 face values.

    Interior faces use the symmetric-as-possible centred stencil
    [f-2, f-1, f, f+1, f+2] with weights [3,-20,90,60,-5]/128.

    The two faces nearest each boundary use a one-sided stencil that
    stays within the domain, with weights derived from the same Lagrange
    construction (see module docstring).

    Args:
        field:         (N,) array of cell-centred values.
        boundary_mode: Reserved for future use ("outflow" | "reflecting").
                       Currently the same weights are applied regardless;
                       the caller is responsible for ghost-cell values.

    Returns:
        field_face: (N-1,) array of face-interpolated values.

    Raises:
        ValueError: if N < 2.
    """
    field = np.asarray(field, dtype=np.float64)
    N = field.size

    if N < 2:
        raise ValueError(f"Field must have at least 2 points, got {N}.")

    if N < 5:
        # Not enough points for a 5-point stencil: fall back to linear.
        return 0.5 * (field[:-1] + field[1:])

    field_face = np.empty(N - 1, dtype=np.float64)

    # ------------------------------------------------------------------
    # Interior faces  f = 2 … N-3  (vectorised, no Python loop)
    # stencil columns: field[0:N-4], field[1:N-3], ..., field[4:N]
    # ------------------------------------------------------------------
    field_face[2:N-2] = (
        _W_INTERIOR[0] * field[0:N-4]
        + _W_INTERIOR[1] * field[1:N-3]
        + _W_INTERIOR[2] * field[2:N-2]
        + _W_INTERIOR[3] * field[3:N-1]
        + _W_INTERIOR[4] * field[4:N]
    )

    # ------------------------------------------------------------------
    # Boundary faces: forward/backward one-sided stencils
    # ------------------------------------------------------------------

    # Face 0  (between cells 0 and 1)
    field_face[0] = np.dot(_W_FACE0, field[0:5])

    # Face 1  (between cells 1 and 2)
    field_face[1] = np.dot(_W_FACE1, field[0:5])

    # Face N-2  (between cells N-2 and N-1, i.e. the last face)
    field_face[N-2] = np.dot(_W_FACE_LAST, field[N-5:N])

    return field_face


def lagrange_interpolate_vector_to_faces(vector_field, boundary_mode="outflow"):
    """
    Interpolate a vector field (N, 3) to cell faces using 4th-order Lagrange.

    Each spatial component is interpolated independently.

    Args:
        vector_field:  (N, 3) array of cell-centred vector values.
        boundary_mode: Passed through to the scalar interpolator.

    Returns:
        vector_face: (N-1, 3) array of face-interpolated values.
    """
    vector_field = np.asarray(vector_field, dtype=np.float64)
    if vector_field.ndim != 2 or vector_field.shape[1] != 3:
        raise ValueError(
            f"Expected vector field with shape (N, 3), got {vector_field.shape}."
        )

    N = vector_field.shape[0]
    vector_face = np.empty((N - 1, 3), dtype=np.float64)

    for d in range(3):
        vector_face[:, d] = lagrange_interpolate_to_faces_4th_order(
            vector_field[:, d], boundary_mode
        )

    return vector_face


def lagrange_interpolate_tensor_to_faces(tensor_field, boundary_mode="outflow"):
    """
    Interpolate a rank-2 tensor field (N, 3, 3) to cell faces using
    4th-order Lagrange.

    Each independent component is interpolated separately.

    Args:
        tensor_field:  (N, 3, 3) array of cell-centred tensor values.
        boundary_mode: Passed through to the scalar interpolator.

    Returns:
        tensor_face: (N-1, 3, 3) array of face-interpolated values.
    """
    tensor_field = np.asarray(tensor_field, dtype=np.float64)
    if tensor_field.ndim != 3 or tensor_field.shape[1:] != (3, 3):
        raise ValueError(
            f"Expected tensor field with shape (N, 3, 3), got {tensor_field.shape}."
        )

    N = tensor_field.shape[0]
    tensor_face = np.empty((N - 1, 3, 3), dtype=np.float64)

    for i in range(3):
        for j in range(3):
            tensor_face[:, i, j] = lagrange_interpolate_to_faces_4th_order(
                tensor_field[:, i, j], boundary_mode
            )

    return tensor_face


# ---------------------------------------------------------------------------
# Self-verification (runs on import only in debug mode, or call explicitly)
# ---------------------------------------------------------------------------

def _verify_weights():
    """
    Check that all weight sets sum to 1 and reproduce polynomials of
    degree <= 4 exactly on a small test grid.
    """
    assert abs(sum(_W_INTERIOR) - 1.0) < 1e-14, "Interior weights do not sum to 1"
    assert abs(sum(_W_FACE0) - 1.0) < 1e-14, "Face-0 weights do not sum to 1"
    assert abs(sum(_W_FACE1) - 1.0) < 1e-14, "Face-1 weights do not sum to 1"
    assert abs(sum(_W_FACE_LAST) - 1.0) < 1e-14, "Face-last weights do not sum to 1"

    # Polynomial exactness: f(x) = x^4 should be reproduced exactly.
    N = 20
    x = np.arange(N, dtype=np.float64)
    f = x ** 4
    f_face = lagrange_interpolate_to_faces_4th_order(f)
    x_face = x[:-1] + 0.5           # exact face positions
    f_face_exact = x_face ** 4
    err = np.max(np.abs(f_face - f_face_exact))
    assert err < 1e-8, f"Polynomial exactness failed: max error = {err:.3e}"

    print("interpolation_utils: all weight checks passed.")
    print(f"  max |error| on x^4 (N={N}): {err:.2e}")


if __name__ == "__main__":
    _verify_weights()