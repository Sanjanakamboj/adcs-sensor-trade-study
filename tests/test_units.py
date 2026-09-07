"""Tests for adcs_sensor_trade.units conversion helpers."""

import math

import pytest

from adcs_sensor_trade.units import (
    arcsec_to_rad,
    deg_to_rad,
    rad_to_arcsec,
    rad_to_deg,
)


def test_deg_to_rad_known_value():
    assert math.isclose(deg_to_rad(180.0), math.pi)


def test_rad_to_deg_known_value():
    assert math.isclose(rad_to_deg(math.pi), 180.0)


def test_arcsec_to_rad_known_value():
    # 3600 arcsec = 1 degree
    assert math.isclose(arcsec_to_rad(3600.0), deg_to_rad(1.0), rel_tol=1e-12)


def test_rad_to_arcsec_roundtrip():
    for val_arcsec in [0.0, 1.0, 30.0, 3600.0, 12345.6]:
        rad = arcsec_to_rad(val_arcsec)
        back = rad_to_arcsec(rad)
        assert math.isclose(back, val_arcsec, rel_tol=1e-10)


@pytest.mark.parametrize("deg", [0.0, 1.0, 45.0, 90.0, 180.0, 359.0])
def test_deg_rad_roundtrip(deg):
    assert math.isclose(rad_to_deg(deg_to_rad(deg)), deg, rel_tol=1e-12, abs_tol=1e-12)
