"""Tests for adcs_sensor_trade.geometry: two-vector attitude geometry."""

import numpy as np
import pytest

from adcs_sensor_trade.geometry import (
    angle_between,
    condition_number,
    cross_mag,
    degeneracy_indicator,
    geometry_rank,
    singular_values,
    triad_basis,
)


def test_angle_between_orthogonal_vectors():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert np.isclose(angle_between(v1, v2), np.pi / 2)


def test_angle_between_parallel_vectors_is_zero():
    v1 = np.array([1.0, 0.0, 0.0])
    assert np.isclose(angle_between(v1, v1), 0.0, atol=1e-10)


def test_angle_between_antiparallel_vectors_is_pi():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([-1.0, 0.0, 0.0])
    assert np.isclose(angle_between(v1, v2), np.pi, atol=1e-10)


def test_cross_mag_matches_sine_of_angle():
    rng = np.random.default_rng(10)
    v1 = rng.normal(size=3)
    v2 = rng.normal(size=3)
    theta = angle_between(v1, v2)
    assert np.isclose(cross_mag(v1, v2), np.sin(theta), atol=1e-10)


def test_triad_basis_orthonormal_and_right_handed():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.3])
    T = triad_basis(v1, v2)
    assert np.allclose(T.T @ T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(T), 1.0, atol=1e-10)


def test_triad_basis_rejects_collinear_vectors():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([2.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        triad_basis(v1, v2)


def test_degeneracy_indicator_bounds():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert np.isclose(degeneracy_indicator(v1, v1), 0.0, atol=1e-10)
    assert np.isclose(degeneracy_indicator(v1, v2), 1.0, atol=1e-10)


def test_single_vector_rank_is_two():
    v1 = np.array([1.0, 0.0, 0.0])
    assert geometry_rank([v1]) == 2


def test_two_noncollinear_vectors_rank_is_three():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert geometry_rank([v1, v2]) == 3


def test_two_collinear_vectors_rank_is_two_and_singular():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    assert geometry_rank([v1, v2]) == 2
    sv = singular_values([v1, v2])
    assert sv[-1] < 1e-8
    assert condition_number([v1, v2]) == float("inf")


def test_two_anticollinear_vectors_singular():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([-1.0, 0.0, 0.0])
    assert geometry_rank([v1, v2]) == 2
    assert condition_number([v1, v2]) == float("inf")


def test_condition_number_best_at_90_degrees():
    v1 = np.array([1.0, 0.0, 0.0])
    angles_deg = [10, 30, 60, 90, 120, 150, 170]
    conds = []
    for a in angles_deg:
        theta = np.radians(a)
        v2 = np.array([np.cos(theta), np.sin(theta), 0.0])
        conds.append(condition_number([v1, v2]))
    # minimum condition number should occur at (or nearest to) 90 deg
    best_idx = int(np.argmin(conds))
    assert angles_deg[best_idx] == 90


def test_condition_number_monotonic_degradation_toward_zero_deg():
    v1 = np.array([1.0, 0.0, 0.0])
    angles_deg = np.array([1, 5, 10, 20, 40, 60, 90])
    conds = []
    for a in angles_deg:
        theta = np.radians(a)
        v2 = np.array([np.cos(theta), np.sin(theta), 0.0])
        conds.append(condition_number([v1, v2]))
    conds = np.array(conds)
    # as angle increases away from 0 (toward 90), condition number should
    # be non-increasing (monotonic improvement in conditioning)
    diffs = np.diff(conds)
    assert np.all(diffs <= 1e-9)


def test_condition_number_monotonic_degradation_toward_180_deg():
    v1 = np.array([1.0, 0.0, 0.0])
    angles_deg = np.array([90, 120, 140, 160, 170, 175, 179])
    conds = []
    for a in angles_deg:
        theta = np.radians(a)
        v2 = np.array([np.cos(theta), np.sin(theta), 0.0])
        conds.append(condition_number([v1, v2]))
    conds = np.array(conds)
    # as angle increases from 90 toward 180, condition number should be
    # non-decreasing (monotonic degradation in conditioning)
    diffs = np.diff(conds)
    assert np.all(diffs >= -1e-9)


def test_geometry_rank_rejects_empty_list():
    with pytest.raises(ValueError):
        geometry_rank([])
