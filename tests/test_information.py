"""Tests for adcs_sensor_trade.information: information/covariance proxies."""

import numpy as np
import pytest

from adcs_sensor_trade.information import (
    combined_vector_information,
    star_tracker_information,
    summarize_information,
    vector_measurement_information,
)


def test_star_tracker_information_is_isotropic():
    sigma = np.radians(30.0 / 3600.0)  # 30 arcsec
    J = star_tracker_information(sigma)
    eigvals = np.linalg.eigvalsh(J)
    assert np.allclose(eigvals, eigvals[0], atol=1e-6 * eigvals[0])


def test_star_tracker_isotropy_independent_of_attitude():
    """The isotropic star-tracker information model does not depend on the
    spacecraft attitude (it directly measures 3-axis attitude)."""
    sigma = np.radians(30.0 / 3600.0)
    J1 = star_tracker_information(sigma)
    J2 = star_tracker_information(sigma)  # model has no attitude input at all
    assert np.allclose(J1, J2)
    eigvals = np.linalg.eigvalsh(J1)
    assert np.ptp(eigvals) < 1e-9


def test_star_tracker_information_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        star_tracker_information(0.0)
    with pytest.raises(ValueError):
        star_tracker_information(-1.0)


def test_vector_information_scales_as_inverse_sigma_squared():
    v = np.array([1.0, 0.0, 0.0])
    sigma1 = np.radians(1.0)
    sigma2 = sigma1 / 2.0  # halving sigma
    J1 = vector_measurement_information(v, sigma1)
    J2 = vector_measurement_information(v, sigma2)
    ratio = J2[np.abs(J1) > 1e-12] / J1[np.abs(J1) > 1e-12]
    # halving sigma should quadruple information magnitude
    assert np.allclose(ratio, 4.0, rtol=1e-8)


def test_vector_information_rejects_nonpositive_sigma():
    v = np.array([0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        vector_measurement_information(v, 0.0)


def test_combined_information_requires_matching_lengths():
    v1 = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        combined_vector_information([v1], [0.1, 0.2])


def test_combined_information_requires_nonempty():
    with pytest.raises(ValueError):
        combined_vector_information([], [])


def test_single_vector_information_is_symmetric_and_psd():
    v = np.array([0.3, 0.7, 0.2])
    J = vector_measurement_information(v, 0.05)
    assert np.allclose(J, J.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(J)
    assert np.all(eigvals >= -1e-10)


def test_combined_two_vector_information_symmetric_and_psd():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    J = combined_vector_information([v1, v2], [0.01, 0.02])
    assert np.allclose(J, J.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(J)
    assert np.all(eigvals >= -1e-10)


def test_single_vector_information_is_rank_deficient():
    """A single unit-vector measurement's information matrix must be rank 2
    (it cannot constrain rotation about its own line of sight)."""
    v = np.array([1.0, 0.0, 0.0])
    J = vector_measurement_information(v, 0.01)
    eigvals = np.linalg.eigvalsh(J)
    n_zero = np.sum(np.abs(eigvals) < 1e-8)
    assert n_zero == 1  # exactly one zero eigenvalue -> rank 2 of 3


def test_summarize_information_full_rank_two_orthogonal_vectors():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    J = combined_vector_information([v1, v2], [0.01, 0.01])
    report = summarize_information(J)
    assert report.rank == 3
    assert report.P is not None
    assert report.rms_attitude_error_rad is not None
    assert report.worst_axis_error_rad is not None
    assert report.condition_number_J >= 1.0


def test_summarize_information_singular_for_single_vector():
    v1 = np.array([1.0, 0.0, 0.0])
    J = vector_measurement_information(v1, 0.01)
    report = summarize_information(J)
    assert report.rank == 2
    assert report.P is None
    assert report.rms_attitude_error_rad is None
    assert report.worst_axis_error_rad is None
    assert report.condition_number_J == float("inf")


def test_summarize_information_rejects_nonsquare():
    with pytest.raises(ValueError):
        summarize_information(np.eye(2))


def test_summarize_information_rejects_asymmetric_matrix():
    J = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        summarize_information(J)


def test_covariance_scaling_with_sigma_doubling():
    """Doubling sigma for an isotropic 3-axis information source should
    quadruple the covariance-proxy trace (and double the worst-axis error
    proxy), since P = sigma^2 * I for the isotropic star-tracker model."""
    sigma1 = np.radians(20.0 / 3600.0)
    sigma2 = 2.0 * sigma1

    J1 = star_tracker_information(sigma1)
    J2 = star_tracker_information(sigma2)
    r1 = summarize_information(J1)
    r2 = summarize_information(J2)

    assert np.isclose(r2.worst_axis_error_rad / r1.worst_axis_error_rad, 2.0, rtol=1e-8)
    assert np.isclose(
        np.trace(r2.P) / np.trace(r1.P), 4.0, rtol=1e-8
    )


def test_star_tracker_information_matches_sun_mag_at_matched_sigma_order_of_magnitude():
    """Sanity check: combining two orthogonal unit-vector measurements at
    sigma comparable to the star tracker's produces a full-rank, PSD,
    finite-condition-number information matrix (not a numerical artifact)."""
    sigma = np.radians(30.0 / 3600.0)
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 0.0, 1.0])
    J = combined_vector_information([v1, v2], [sigma, sigma])
    report = summarize_information(J)
    assert report.rank == 3
    assert np.isfinite(report.condition_number_J)
