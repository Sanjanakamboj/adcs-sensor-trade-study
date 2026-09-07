"""Tests for adcs_sensor_trade.sensors.star_tracker."""

import numpy as np
import pytest

from adcs_sensor_trade.sensors.star_tracker import (
    StarTracker,
    StarTrackerConfig,
)


def test_default_config_is_valid_and_illustrative_baseline():
    cfg = StarTrackerConfig()
    assert 0.0 < cfg.sigma_arcsec < 100.0  # "tens of arcsec" class
    assert 0.5 <= cfg.update_rate_hz <= 10.0


def test_from_arcsec_constructor_roundtrip():
    cfg = StarTrackerConfig.from_arcsec(25.0)
    assert np.isclose(cfg.sigma_arcsec, 25.0, rtol=1e-8)


def test_deterministic_measurement_with_zero_sigma():
    cfg = StarTrackerConfig(sigma_rad=0.0)
    st = StarTracker(cfg)
    phi_true = np.array([0.1, -0.2, 0.05])
    rng = np.random.default_rng(0)
    meas = st.measure(phi_true, rng=rng)
    from adcs_sensor_trade.rotations import dcm_from_rotvec

    assert meas.available
    assert np.allclose(meas.dcm_body_from_inertial, dcm_from_rotvec(phi_true), atol=1e-12)


def test_measurement_without_rng_is_deterministic_even_with_nonzero_sigma():
    cfg = StarTrackerConfig(sigma_rad=np.radians(1.0))
    st = StarTracker(cfg)
    phi_true = np.array([0.0, 0.0, 0.0])
    meas = st.measure(phi_true, rng=None)
    assert np.allclose(meas.dcm_body_from_inertial, np.eye(3), atol=1e-12)


def test_reproducibility_same_seed_gives_identical_measurement():
    cfg = StarTrackerConfig(sigma_rad=np.radians(30.0 / 3600.0))
    st = StarTracker(cfg)
    phi_true = np.array([0.2, 0.1, -0.3])

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    m1 = st.measure(phi_true, rng=rng1)
    m2 = st.measure(phi_true, rng=rng2)
    assert np.allclose(m1.dcm_body_from_inertial, m2.dcm_body_from_inertial, atol=1e-15)
    assert np.allclose(m1.quaternion_xyzw, m2.quaternion_xyzw, atol=1e-15)


def test_different_seeds_give_different_measurements():
    cfg = StarTrackerConfig(sigma_rad=np.radians(30.0 / 3600.0))
    st = StarTracker(cfg)
    phi_true = np.zeros(3)
    m1 = st.measure(phi_true, rng=np.random.default_rng(1))
    m2 = st.measure(phi_true, rng=np.random.default_rng(2))
    assert not np.allclose(m1.dcm_body_from_inertial, m2.dcm_body_from_inertial)


def test_noise_magnitude_scales_with_sigma():
    """Larger sigma should produce, on average, larger attitude-error
    rotation angle between the measured and true DCM."""
    from scipy.spatial.transform import Rotation

    phi_true = np.zeros(3)
    n_trials = 400

    def mean_error_angle(sigma_rad, seed):
        cfg = StarTrackerConfig(sigma_rad=sigma_rad)
        st = StarTracker(cfg)
        rng = np.random.default_rng(seed)
        angles = []
        for _ in range(n_trials):
            m = st.measure(phi_true, rng=rng)
            dA = m.dcm_body_from_inertial  # since phi_true=0, A_true=I
            angle = np.linalg.norm(Rotation.from_matrix(dA).as_rotvec())
            angles.append(angle)
        return np.mean(angles)

    small = mean_error_angle(np.radians(10.0 / 3600.0), seed=7)
    large = mean_error_angle(np.radians(60.0 / 3600.0), seed=7)
    assert large > small


def test_exclusion_zone_makes_sensor_unavailable():
    cfg = StarTrackerConfig(exclusion_half_angle_rad=np.radians(30.0))
    st = StarTracker(cfg)
    phi_true = np.zeros(3)
    meas = st.measure(phi_true, angle_to_bright_source_rad=np.radians(10.0))
    assert not meas.available
    assert meas.dcm_body_from_inertial is None


def test_outside_exclusion_zone_is_available():
    cfg = StarTrackerConfig(exclusion_half_angle_rad=np.radians(30.0))
    st = StarTracker(cfg)
    phi_true = np.zeros(3)
    meas = st.measure(phi_true, angle_to_bright_source_rad=np.radians(90.0))
    assert meas.available


def test_dropout_probability_one_always_unavailable():
    cfg = StarTrackerConfig(dropout_probability=1.0)
    st = StarTracker(cfg)
    rng = np.random.default_rng(0)
    meas = st.measure(np.zeros(3), rng=rng)
    assert not meas.available


def test_dropout_probability_zero_default_available():
    cfg = StarTrackerConfig()
    st = StarTracker(cfg)
    rng = np.random.default_rng(0)
    meas = st.measure(np.zeros(3), rng=rng)
    assert meas.available


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_rad": -1.0},
        {"update_rate_hz": 0.0},
        {"update_rate_hz": -2.0},
        {"exclusion_half_angle_rad": -0.1},
        {"exclusion_half_angle_rad": 4.0},  # > pi
        {"dropout_probability": -0.1},
        {"dropout_probability": 1.5},
    ],
)
def test_invalid_config_raises_value_error(kwargs):
    with pytest.raises(ValueError):
        StarTrackerConfig(**kwargs)


def test_from_arcsec_rejects_negative():
    with pytest.raises(ValueError):
        StarTrackerConfig.from_arcsec(-5.0)
