"""Rotation utilities: skew-symmetric (cross-product) matrices, attitude
rotation-vector parametrization, and the small-angle measurement Jacobian.

CONVENTION (frozen; see docs/conventions.md section "Attitude-error
interpretation"):

- A rotation vector ``phi`` (rad) parametrizes the inertial-to-body DCM as
  the standard Rodrigues active-rotation matrix::

      A(phi) = Rotation.from_rotvec(phi).as_matrix()
      v_b = A(phi) @ v_i

- Composition of a small additional rotation ``delta_theta`` on top of a
  base rotation ``phi`` obeys, to first order::

      A(phi + delta_theta) ~= (I + skew(delta_theta)) @ A(phi)

  i.e. the small-angle DCM correction convention used throughout this
  package is ``A(delta_theta) ~= I + skew(delta_theta)`` (NOT ``I -
  skew(...)``). This is a definitional choice (documented, not derived);
  it is what makes the measurement Jacobian come out to ``H_v = -skew(v_b)``
  below, matching the Milestone 1 specification and standard multiplicative
  extended Kalman filter (MEKF) literature (e.g. Markley & Crassidis).

- Consequently, for a body-frame unit-vector measurement ``v_b = A(phi) @
  v_i``, the sensitivity of ``v_b`` to a small attitude-error rotation
  vector ``delta_theta`` (added on top of the current attitude) is::

      d(v_b)/d(delta_theta) = -skew(v_b) =: H_v

  This is verified numerically against a central finite difference of the
  exact rotation in ``tests/test_rotations.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric (cross-product) matrix of vector v.

    Defined so that ``skew(v) @ w == np.cross(v, w)`` for all w (right-handed
    cross product convention).
    """
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def dcm_from_rotvec(phi: np.ndarray) -> np.ndarray:
    """Inertial-to-body DCM for rotation vector ``phi`` [rad].

    Uses the standard Rodrigues active-rotation matrix (see module
    docstring for the sign convention frozen for this package).
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    return Rotation.from_rotvec(phi).as_matrix()


def rotate_inertial_to_body(v_inertial: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Rotate an inertial-frame unit vector into the body frame via ``phi``."""
    v_inertial = np.asarray(v_inertial, dtype=float).reshape(3)
    return dcm_from_rotvec(phi) @ v_inertial


def normalize(v: np.ndarray) -> np.ndarray:
    """Return v / ||v||. Raises ValueError for a (near-)zero vector."""
    v = np.asarray(v, dtype=float).reshape(-1)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero (or near-zero) vector.")
    return v / norm


def measurement_jacobian(v_body: np.ndarray) -> np.ndarray:
    """Small-angle measurement Jacobian H_v = -skew(v_body) (see module docstring).

    Maps a small attitude-error rotation vector [rad] to the resulting
    first-order change in the body-frame unit-vector measurement.
    """
    v_body = np.asarray(v_body, dtype=float).reshape(3)
    return -skew(v_body)
