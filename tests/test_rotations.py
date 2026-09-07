"""Tests for adcs_sensor_trade.rotations: skew matrices, DCM parametrization,
the frozen small-angle measurement-Jacobian sign convention, and normalize().
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from adcs_sensor_trade.rotations import (
    dcm_from_rotvec,
    measurement_jacobian,
    normalize,
    rotate_inertial_to_body,
    skew,
)


def test_skew_matches_cross_product():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.normal(size=3)
        b = rng.normal(size=3)
        assert np.allclose(skew(a) @ b, np.cross(a, b), atol=1e-10)


def test_skew_is_antisymmetric():
    rng = np.random.default_rng(2)
    a = rng.normal(size=3)
    S = skew(a)
    assert np.allclose(S, -S.T, atol=1e-12)


def test_skew_diagonal_is_zero():
    a = np.array([3.0, -1.0, 2.0])
    assert np.allclose(np.diag(skew(a)), 0.0)


def test_dcm_from_rotvec_is_orthonormal():
    rng = np.random.default_rng(3)
    phi = rng.normal(size=3) * 0.7
    A = dcm_from_rotvec(phi)
    assert np.allclose(A @ A.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(A), 1.0, atol=1e-10)


def test_dcm_from_zero_rotvec_is_identity():
    A = dcm_from_rotvec(np.zeros(3))
    assert np.allclose(A, np.eye(3), atol=1e-12)


def test_rotate_inertial_to_body_preserves_norm():
    rng = np.random.default_rng(4)
    v_i = normalize(rng.normal(size=3))
    phi = rng.normal(size=3) * 0.5
    v_b = rotate_inertial_to_body(v_i, phi)
    assert np.isclose(np.linalg.norm(v_b), 1.0, atol=1e-10)


def test_normalize_unit_norm():
    v = np.array([3.0, 4.0, 0.0])
    u = normalize(v)
    assert np.isclose(np.linalg.norm(u), 1.0)
    assert np.allclose(u, [0.6, 0.8, 0.0])


def test_normalize_rejects_zero_vector():
    with pytest.raises(ValueError):
        normalize(np.zeros(3))


def test_normalize_rejects_near_zero_vector():
    with pytest.raises(ValueError):
        normalize(np.array([1e-14, -1e-14, 1e-15]))


def test_measurement_jacobian_is_negative_skew():
    v = np.array([0.0, 0.0, 1.0])
    H = measurement_jacobian(v)
    assert np.allclose(H, -skew(v))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_measurement_jacobian_matches_finite_difference(seed):
    """Verify H_v = -skew(v_b) against a central finite difference of the
    exact rotation, using the LEFT-multiplicative small-rotation
    perturbation convention frozen in rotations.py: A(phi_true) ~=
    R(delta_theta) @ A(phi_est) for a small correction delta_theta.

    This is the rigorous numerical sign-convention check required by the
    Milestone 1 spec: it does not merely assert the analytic formula, it
    confirms it against independently-computed finite differences of the
    exact (not linearized) rotation matrices.
    """
    rng = np.random.default_rng(seed)
    phi0 = rng.normal(size=3) * 0.6
    v_i = normalize(rng.normal(size=3))
    A0 = dcm_from_rotvec(phi0)
    v_b0 = A0 @ v_i

    H_analytic = measurement_jacobian(v_b0)

    eps = 1e-6
    H_numeric = np.zeros((3, 3))
    for i in range(3):
        theta = np.zeros(3)
        theta[i] = eps
        Rp = Rotation.from_rotvec(theta).as_matrix()
        Rm = Rotation.from_rotvec(-theta).as_matrix()
        v_plus = Rp @ A0 @ v_i
        v_minus = Rm @ A0 @ v_i
        H_numeric[:, i] = (v_plus - v_minus) / (2 * eps)

    assert np.allclose(H_numeric, H_analytic, atol=1e-6)


def test_measurement_jacobian_shape_and_antisymmetry():
    v = normalize(np.array([1.0, 2.0, 3.0]))
    H = measurement_jacobian(v)
    assert H.shape == (3, 3)
    assert np.allclose(H, -H.T)
