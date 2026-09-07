"""Tests for adcs_sensor_trade.sensors.magnetometer."""

import numpy as np
import pytest

from adcs_sensor_trade.geometry import geometry_rank
from adcs_sensor_trade.sensors.magnetometer import (
    Magnetometer,
    MagnetometerConfig,
    demonstrate_single_vector_rank,
)


def test_default_config_valid():
    cfg = MagnetometerConfig()
    assert cfg.sigma_deg > 0
    assert cfg.update_rate_hz > 0
    assert np.allclose(cfg.bias_tesla, 0.0)
    assert np.allclose(cfg.scale_factor, 1.0)


def test_deterministic_measurement_zero_sigma_no_bias_no_scale():
    cfg = MagnetometerConfig(sigma_rad=0.0)
    mag = Magnetometer(cfg)
    b_i = np.array([0.0, 0.0, 3.0e-5])
    meas = mag.measure(b_i, np.zeros(3), rng=None)
    assert meas.available
    assert np.allclose(meas.b_field_direction_body, [0.0, 0.0, 1.0], atol=1e-12)
    assert np.allclose(meas.b_field_vector_body_tesla, b_i, atol=1e-15)


def test_bias_is_added():
    cfg = MagnetometerConfig(sigma_rad=0.0, bias_tesla=np.array([1e-6, 0.0, 0.0]))
    mag = Magnetometer(cfg)
    b_i = np.array([0.0, 0.0, 3.0e-5])
    meas = mag.measure(b_i, np.zeros(3), rng=None)
    assert np.isclose(meas.b_field_vector_body_tesla[0], 1e-6, atol=1e-12)


def test_scale_factor_is_applied():
    cfg = MagnetometerConfig(sigma_rad=0.0, scale_factor=np.array([1.0, 1.0, 2.0]))
    mag = Magnetometer(cfg)
    b_i = np.array([0.0, 0.0, 3.0e-5])
    meas = mag.measure(b_i, np.zeros(3), rng=None)
    assert np.isclose(meas.b_field_vector_body_tesla[2], 6.0e-5, rtol=1e-10)


def test_direction_output_is_unit_vector_even_with_noise():
    cfg = MagnetometerConfig(sigma_rad=np.radians(1.0))
    mag = Magnetometer(cfg)
    b_i = np.array([1.0, 1.0, 1.0])
    rng = np.random.default_rng(11)
    meas = mag.measure(b_i, np.zeros(3), rng=rng)
    assert np.isclose(np.linalg.norm(meas.b_field_direction_body), 1.0, atol=1e-10)


def test_zero_inertial_field_rejected():
    cfg = MagnetometerConfig()
    mag = Magnetometer(cfg)
    with pytest.raises(ValueError):
        mag.measure(np.zeros(3), np.zeros(3), rng=None)


def test_reproducibility_same_seed():
    cfg = MagnetometerConfig(sigma_rad=np.radians(0.5))
    b_i = np.array([0.0, 0.0, 3.0e-5])
    m1 = Magnetometer(cfg).measure(b_i, np.zeros(3), rng=np.random.default_rng(9))
    m2 = Magnetometer(cfg).measure(b_i, np.zeros(3), rng=np.random.default_rng(9))
    assert np.allclose(m1.b_field_direction_body, m2.b_field_direction_body, atol=1e-15)


def test_single_b_field_vector_rank_is_two():
    b_body = np.array([0.0, 0.0, 1.0])
    assert demonstrate_single_vector_rank(b_body) == 2
    assert geometry_rank([b_body]) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_rad": -0.1},
        {"update_rate_hz": 0.0},
        {"update_rate_hz": -5.0},
        {"scale_factor": np.array([1.0, 0.0, 1.0])},
        {"scale_factor": np.array([1.0, -1.0, 1.0])},
    ],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        MagnetometerConfig(**kwargs)
