"""Two-vector attitude geometry utilities.

All functions here are DETERMINISTIC (no randomness) and operate on unit
vectors expressed as plain length-3 numpy arrays. Angles returned/accepted
by these functions are in RADIANS unless a name ends in ``_deg``.

See docs/geometry_and_information.md for the engineering discussion of why
one vector constrains only 2 of 3 rotational degrees of freedom, and why
two non-collinear vectors recover full 3-axis attitude information.
"""

from __future__ import annotations

import numpy as np

from .rotations import measurement_jacobian, normalize

# Default numerical tolerance used to call a singular value "zero" when
# computing rank via SVD. This is a scale-free tolerance appropriate for
# unit-vector Jacobians (whose singular values are O(1)).
DEFAULT_RANK_TOL = 1e-8


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle [rad] between two unit vectors, in [0, pi]."""
    v1 = normalize(v1)
    v2 = normalize(v2)
    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.arccos(cos_theta))


def cross_mag(v1: np.ndarray, v2: np.ndarray) -> float:
    """Magnitude of the cross product of two unit vectors, |v1 x v2|.

    Equal to sin(angle_between(v1, v2)) for unit vectors; this is a
    convenient, cheap degeneracy indicator (0 when collinear/anti-collinear,
    maximal at 90 deg separation).
    """
    v1 = normalize(v1)
    v2 = normalize(v2)
    return float(np.linalg.norm(np.cross(v1, v2)))


def triad_basis(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Construct an orthonormal TRIAD-style basis from two unit vectors.

    ``v1`` is treated as the more-trusted ("primary") vector. Returns a 3x3
    matrix whose columns are the right-handed orthonormal basis
    [t1, t2, t3] with::

        t1 = v1
        t2 = normalize(v1 x v2)
        t3 = t1 x t2

    Raises ValueError if v1 and v2 are (numerically) collinear, since no
    basis can be formed from parallel/anti-parallel vectors.
    """
    v1 = normalize(v1)
    v2 = normalize(v2)
    cross = np.cross(v1, v2)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-10:
        raise ValueError(
            "Cannot construct a TRIAD basis from (near-)collinear vectors "
            f"(|v1 x v2| = {cross_norm:.3e})."
        )
    t1 = v1
    t2 = cross / cross_norm
    t3 = np.cross(t1, t2)
    return np.column_stack([t1, t2, t3])


def degeneracy_indicator(v1: np.ndarray, v2: np.ndarray) -> float:
    """A simple [0, 1] conditioning indicator for a two-vector pair.

    Returns |sin(angle_between(v1, v2))|: 0 at exact collinearity/anti-
    collinearity (fully degenerate for 3-axis recovery), 1 at exactly 90 deg
    separation (best-conditioned two-vector geometry).
    """
    return cross_mag(v1, v2)


def stacked_measurement_jacobian(body_vectors: list[np.ndarray]) -> np.ndarray:
    """Stack the H_v = -skew(v_b) Jacobians for a list of body-frame vectors.

    Returns a (3*N, 3) matrix; N=1 gives the single-vector Jacobian.
    """
    if len(body_vectors) == 0:
        raise ValueError("Need at least one body-frame vector.")
    blocks = [measurement_jacobian(normalize(v)) for v in body_vectors]
    return np.vstack(blocks)


def singular_values(body_vectors: list[np.ndarray]) -> np.ndarray:
    """Singular values (descending) of the stacked measurement Jacobian."""
    H = stacked_measurement_jacobian(body_vectors)
    return np.linalg.svd(H, compute_uv=False)


def geometry_rank(body_vectors: list[np.ndarray], tol: float = DEFAULT_RANK_TOL) -> int:
    """Rank of the instantaneous attitude-information geometry via SVD.

    A singular value below ``tol`` is treated as numerically zero. For unit
    vectors, singular values are O(1), so the default tolerance is safe
    without additional scaling.
    """
    sv = singular_values(body_vectors)
    return int(np.sum(sv > tol))


def condition_number(body_vectors: list[np.ndarray]) -> float:
    """Condition number (max/min singular value) of the stacked Jacobian.

    Returns ``np.inf`` if the minimum singular value is (numerically) zero.
    """
    sv = singular_values(body_vectors)
    sv_min = sv[-1]
    sv_max = sv[0]
    if sv_min < 1e-14:
        return float("inf")
    return float(sv_max / sv_min)
