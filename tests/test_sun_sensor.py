"""Tests for adcs_sensor_trade.sensors.sun_sensor."""

import numpy as np
import pytest

from adcs_sensor_trade.geometry import geometry_rank
from adcs_sensor_trade.sensors.sun_sensor import (
    SunSensor,
    SunSensorConfig,
    demonstrate_single_vector_rank,
)


def test_default_config_valid():
    cfg = SunSensorConfig()
    assert cfg.sigma_deg > 0
    assert 0 < cfg.fov_half_angle_deg <= 180


def test_deterministic_measurement_zero_sigma():
    cfg = SunSensorConfig(sigma_rad=0.0)
    sensor = SunSensor(cfg)
    sun_i = np.array([1.0, 0.0, 0.0])
    phi_true = np.zeros(3)
    meas = sensor.measure(sun_i, phi_true, rng=None)
    assert meas.available
    assert np.allclose(meas.sun_vector_body, sun_i, atol=1e-12)


def test_measurement_is_unit_vector_even_with_noise():
    cfg = SunSensorConfig(sigma_rad=np.radians(1.0))
    sensor = SunSensor(cfg)
    sun_i = np.array([1.0, 0.0, 0.0])
    rng = np.random.default_rng(5)
    meas = sensor.measure(sun_i, np.zeros(3), rng=rng)
    assert np.isclose(np.linalg.norm(meas.sun_vector_body), 1.0, atol=1e-10)


def test_out_of_fov_is_unavailable():
    cfg = SunSensorConfig(fov_half_angle_rad=np.radians(30.0))
    sensor = SunSensor(cfg)
    sun_i = np.array([0.0, 1.0, 0.0])  # 90 deg from boresight [1,0,0]
    meas = sensor.measure(sun_i, np.zeros(3))
    assert not meas.available
    assert meas.sun_vector_body is None


def test_in_fov_is_available():
    cfg = SunSensorConfig(fov_half_angle_rad=np.radians(60.0))
    sensor = SunSensor(cfg)
    sun_i = np.array([1.0, 0.0, 0.0])  # aligned with boresight
    meas = sensor.measure(sun_i, np.zeros(3))
    assert meas.available


def test_eclipse_forces_unavailable():
    cfg = SunSensorConfig()
    sensor = SunSensor(cfg)
    sun_i = np.array([1.0, 0.0, 0.0])
    meas = sensor.measure(sun_i, np.zeros(3), in_eclipse=True)
    assert not meas.available


def test_reproducibility_same_seed():
    cfg = SunSensorConfig(sigma_rad=np.radians(0.5))
    sun_i = np.array([1.0, 0.0, 0.0])
    m1 = SunSensor(cfg).measure(sun_i, np.zeros(3), rng=np.random.default_rng(3))
    m2 = SunSensor(cfg).measure(sun_i, np.zeros(3), rng=np.random.default_rng(3))
    assert np.allclose(m1.sun_vector_body, m2.sun_vector_body, atol=1e-15)


def test_single_sun_vector_rank_is_two():
    sun_body = np.array([1.0, 0.0, 0.0])
    assert demonstrate_single_vector_rank(sun_body) == 2
    assert geometry_rank([sun_body]) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_rad": -0.1},
        {"fov_half_angle_rad": 0.0},
        {"fov_half_angle_rad": -1.0},
        {"fov_half_angle_rad": 4.0},  # > pi
        {"update_rate_hz": 0.0},
        {"update_rate_hz": -1.0},
    ],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        SunSensorConfig(**kwargs)
