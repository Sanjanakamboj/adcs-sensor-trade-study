"""Local instantaneous attitude-information / covariance proxy.

IMPORTANT SCOPE NOTE: everything in this module is a LOCAL, INSTANTANEOUS
GEOMETRY/NOISE PROXY built from small-angle measurement Jacobians. It is
**not** a full estimator (EKF) covariance: there is no time propagation, no
gyro process-noise coupling, no measurement history, and no filter gain
dynamics. It answers only "given these measurement vectors and their noise,
how well-conditioned is the *instantaneous* 3-axis attitude information",
which is exactly the Milestone 1 question (sensor physics + geometry, not a
trade matrix and not an estimator design).

Model
-----
For a set of unit-vector measurements v_b,i with isotropic angular
measurement noise variance sigma_i^2 (rad^2) about each axis transverse to
the vector (small-angle tangent-space model, R_i = sigma_i^2 * I_3), the
per-measurement information contribution is::

    H_i = -skew(v_b,i)              (3x3, see rotations.py)
    J_i = H_i^T @ R_i^-1 @ H_i = (1/sigma_i^2) * H_i^T @ H_i

Total information: J = sum_i J_i. Covariance proxy: P = J^-1 where J is
full rank (rank 3); undefined (returned as None) otherwise.

Star tracker special case
--------------------------
A star tracker measures a full 3-axis attitude directly (it triangulates
many stars and outputs a quaternion/DCM), not a single line-of-sight
vector. Modeling it as "one more unit-vector measurement" would understate
its information content and is not physically representative. Instead we
model its instantaneous information as ISOTROPIC over all 3 attitude axes::

    J_star = (1/sigma_star^2) * I_3

This is a documented SIMPLIFICATION: a real star tracker's own internal
covariance (from its star-centroiding/triangulation solution) is generally
only approximately isotropic and can depend weakly on the star-field
geometry in its FOV; we do not model that internal geometry here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import DEFAULT_RANK_TOL
from .rotations import normalize

_ISO_TOL = 1e-9


@dataclass(frozen=True)
class InformationReport:
    """Summary of a local attitude-information / covariance-proxy result."""

    J: np.ndarray  # 3x3 information matrix [rad^-2]
    P: np.ndarray | None  # 3x3 covariance proxy [rad^2], None if singular
    rank: int
    eigenvalues_J: np.ndarray  # ascending, information eigenvalues [rad^-2]
    eigenvectors_J: np.ndarray  # columns = principal axes (information frame)
    condition_number_J: float
    rms_attitude_error_rad: float | None  # sqrt(trace(P)/3)
    worst_axis_error_rad: float | None  # sqrt(max eigenvalue of P)

    @property
    def rms_attitude_error_deg(self) -> float | None:
        if self.rms_attitude_error_rad is None:
            return None
        return float(np.degrees(self.rms_attitude_error_rad))

    @property
    def worst_axis_error_deg(self) -> float | None:
        if self.worst_axis_error_rad is None:
            return None
        return float(np.degrees(self.worst_axis_error_rad))


def vector_measurement_information(v_body: np.ndarray, sigma_rad: float) -> np.ndarray:
    """Information contribution J_i = (1/sigma^2) * H_i^T H_i for one unit vector."""
    if sigma_rad <= 0:
        raise ValueError(f"sigma_rad must be > 0, got {sigma_rad}")
    v_body = normalize(v_body)
    H = -_skew_local(v_body)
    return (1.0 / sigma_rad**2) * (H.T @ H)


def _skew_local(v: np.ndarray) -> np.ndarray:
    # local import-free skew to avoid a circular import at module load time
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def combined_vector_information(
    body_vectors: list[np.ndarray], sigmas_rad: list[float]
) -> np.ndarray:
    """Sum information contributions from a list of unit-vector measurements."""
    if len(body_vectors) != len(sigmas_rad):
        raise ValueError("body_vectors and sigmas_rad must have equal length.")
    if len(body_vectors) == 0:
        raise ValueError("Need at least one measurement vector.")
    J = np.zeros((3, 3))
    for v, sigma in zip(body_vectors, sigmas_rad):
        J += vector_measurement_information(v, sigma)
    return J


def star_tracker_information(sigma_star_rad: float) -> np.ndarray:
    """Isotropic 3-axis information matrix for a direct-attitude star tracker."""
    if sigma_star_rad <= 0:
        raise ValueError(f"sigma_star_rad must be > 0, got {sigma_star_rad}")
    return (1.0 / sigma_star_rad**2) * np.eye(3)


def summarize_information(J: np.ndarray, rank_tol: float = DEFAULT_RANK_TOL) -> InformationReport:
    """Build a full InformationReport (rank, covariance proxy, eigen-summary) from J."""
    J = np.asarray(J, dtype=float)
    if J.shape != (3, 3):
        raise ValueError(f"J must be 3x3, got shape {J.shape}")
    if not np.allclose(J, J.T, atol=1e-8):
        raise ValueError("Information matrix J must be symmetric.")

    eigvals, eigvecs = np.linalg.eigh(J)  # ascending order for symmetric matrices
    # Rank via singular values of J itself (J is PSD, so singular values == |eigenvalues|)
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(sv > rank_tol))

    sv_min, sv_max = sv[-1], sv[0]
    cond = float("inf") if sv_min < 1e-14 else float(sv_max / sv_min)

    P = None
    rms_err = None
    worst_err = None
    if rank == 3:
        P = np.linalg.inv(J)
        rms_err = float(np.sqrt(np.trace(P) / 3.0))
        worst_err = float(np.sqrt(np.max(np.linalg.eigvalsh(P))))

    return InformationReport(
        J=J,
        P=P,
        rank=rank,
        eigenvalues_J=eigvals,
        eigenvectors_J=eigvecs,
        condition_number_J=cond,
        rms_attitude_error_rad=rms_err,
        worst_axis_error_rad=worst_err,
    )
