"""Tests for adcs_sensor_trade.orbit_environment (Milestone 2)."""

from __future__ import annotations

import numpy as np
import pytest

from adcs_sensor_trade.orbit_environment import (
    EARTH_RADIUS_KM,
    attitude_dcm_nadir_pointing,
    attitude_rotvec_nadir_pointing,
    b_field_direction_inertial,
    b_field_inertial_tesla,
    dipole_axis_inertial,
    eclipse_fraction,
    environment_state,
    environment_states_batch,
    is_in_eclipse,
    orbit_position_hat,
    orbit_velocity_hat,
    orbital_period_s,
    star_tracker_bright_source_angle_rad,
    sun_and_mag_body_vectors,
    sun_vector_inertial,
)


# ---------------------------------------------------------------------------
# Orbital period
# ---------------------------------------------------------------------------

def test_orbital_period_positive_and_increases_with_altitude():
    t_low = orbital_period_s(400.0)
    t_high = orbital_period_s(800.0)
    assert t_low > 0
    assert t_high > t_low


def test_orbital_period_matches_known_600km_value():
    # Standard circular LEO period ~ 96-97 minutes at 600 km altitude.
    t_min = orbital_period_s(600.0) / 60.0
    assert 90.0 < t_min < 100.0


@pytest.mark.parametrize("bad_altitude", [0.0, -100.0])
def test_orbital_period_rejects_bad_altitude(bad_altitude):
    with pytest.raises(ValueError):
        orbital_period_s(bad_altitude)


# ---------------------------------------------------------------------------
# Eclipse fraction / eclipse boolean
# ---------------------------------------------------------------------------

def test_eclipse_fraction_zero_beta_is_largest():
    f0 = eclipse_fraction(600.0, 0.0)
    f30 = eclipse_fraction(600.0, 30.0)
    assert f0 > f30 > 0.0


def test_eclipse_fraction_zero_above_beta_star():
    r = EARTH_RADIUS_KM + 600.0
    beta_star_deg = np.degrees(np.arcsin(EARTH_RADIUS_KM / r))
    assert eclipse_fraction(600.0, beta_star_deg + 5.0) == 0.0


def test_eclipse_fraction_symmetric_in_beta_sign():
    assert np.isclose(eclipse_fraction(600.0, 25.0), eclipse_fraction(600.0, -25.0))


def test_eclipse_fraction_monotonic_non_increasing_in_abs_beta():
    betas = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    fracs = np.array([eclipse_fraction(600.0, b) for b in betas])
    assert np.all(np.diff(fracs) <= 1e-12)


def test_eclipse_fraction_in_valid_range():
    for beta in [-80.0, -10.0, 0.0, 10.0, 80.0]:
        f = eclipse_fraction(600.0, beta)
        assert 0.0 <= f < 0.5


@pytest.mark.parametrize("bad_beta", [-91.0, 91.0, 180.0])
def test_eclipse_fraction_rejects_bad_beta(bad_beta):
    with pytest.raises(ValueError):
        eclipse_fraction(600.0, bad_beta)


def test_eclipse_fraction_rejects_bad_altitude():
    with pytest.raises(ValueError):
        eclipse_fraction(-1.0, 0.0)


def test_is_in_eclipse_measured_fraction_matches_closed_form():
    beta = 15.0
    expected = eclipse_fraction(600.0, beta)
    phases = np.linspace(0.0, 2.0 * np.pi, 20000, endpoint=False)
    measured = np.mean([is_in_eclipse(p, 600.0, beta) for p in phases])
    assert abs(measured - expected) < 0.002


def test_is_in_eclipse_false_when_fraction_zero():
    assert is_in_eclipse(np.pi, 600.0, 89.0) is False


def test_is_in_eclipse_centered_on_local_midnight():
    # At beta = 0, eclipse fraction is largest and must include phase = pi.
    assert is_in_eclipse(np.pi, 600.0, 0.0) is True
    # Local noon (phase = 0) is never eclipsed.
    assert is_in_eclipse(0.0, 600.0, 0.0) is False


# ---------------------------------------------------------------------------
# Sun vector / orbit position & velocity geometry
# ---------------------------------------------------------------------------

def test_sun_vector_inertial_is_unit():
    for beta in [-45.0, 0.0, 30.0, 89.0]:
        v = sun_vector_inertial(beta)
        assert np.isclose(np.linalg.norm(v), 1.0)


def test_sun_vector_inertial_z_component_matches_sin_beta():
    v = sun_vector_inertial(30.0)
    assert np.isclose(v[2], np.sin(np.radians(30.0)))


def test_orbit_position_and_velocity_are_orthonormal():
    for phase in np.linspace(0, 2 * np.pi, 13):
        r_hat = orbit_position_hat(phase)
        v_hat = orbit_velocity_hat(phase)
        assert np.isclose(np.linalg.norm(r_hat), 1.0)
        assert np.isclose(np.linalg.norm(v_hat), 1.0)
        assert np.isclose(np.dot(r_hat, v_hat), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Attitude DCM
# ---------------------------------------------------------------------------

def test_attitude_dcm_is_orthonormal_rotation_matrix():
    for phase in np.linspace(0, 2 * np.pi, 17):
        A = attitude_dcm_nadir_pointing(phase)
        assert np.allclose(A @ A.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(A), 1.0, atol=1e-8)


def test_attitude_rotvec_reproduces_dcm():
    from adcs_sensor_trade.rotations import dcm_from_rotvec

    for phase in [0.0, 0.7, 2.1, 4.5]:
        A = attitude_dcm_nadir_pointing(phase)
        phi = attitude_rotvec_nadir_pointing(phase)
        assert np.allclose(dcm_from_rotvec(phi), A, atol=1e-8)


def test_body_y_axis_is_fixed_orbit_normal_direction():
    # By construction (nadir = -r_hat, x_b = v_hat, both in-plane), the body
    # +Y axis should be a FIXED direction (independent of phase) for a
    # planar circular orbit.
    ys = [attitude_dcm_nadir_pointing(p)[1, :] for p in np.linspace(0, 2 * np.pi, 9)]
    for y in ys[1:]:
        assert np.allclose(y, ys[0], atol=1e-10)


# ---------------------------------------------------------------------------
# Star tracker bright-source angle
# ---------------------------------------------------------------------------

def test_star_tracker_bright_source_angle_min_equals_beta_at_local_noon():
    beta = 20.0
    angle = star_tracker_bright_source_angle_rad(0.0, beta)
    assert np.isclose(np.degrees(angle), beta, atol=1e-6)


def test_star_tracker_bright_source_angle_far_from_sun_at_local_midnight():
    beta = 20.0
    angle_noon = star_tracker_bright_source_angle_rad(0.0, beta)
    angle_midnight = star_tracker_bright_source_angle_rad(np.pi, beta)
    assert angle_midnight > angle_noon


def test_star_tracker_bright_source_angle_range():
    for phase in np.linspace(0, 2 * np.pi, 25):
        angle = star_tracker_bright_source_angle_rad(phase, 30.0)
        assert 0.0 <= angle <= np.pi


# ---------------------------------------------------------------------------
# Tilted-dipole magnetic field model
# ---------------------------------------------------------------------------

def test_dipole_axis_is_unit_and_tilted_correctly():
    axis = dipole_axis_inertial(11.5)
    assert np.isclose(np.linalg.norm(axis), 1.0)
    assert np.isclose(np.degrees(np.arccos(np.clip(axis[2], -1, 1))), 11.5, atol=1e-6)


def test_dipole_axis_zero_tilt_is_orbit_normal():
    axis = dipole_axis_inertial(0.0)
    assert np.allclose(axis, [0.0, 0.0, 1.0], atol=1e-12)


@pytest.mark.parametrize("bad_tilt", [-1.0, 91.0])
def test_dipole_axis_rejects_bad_tilt(bad_tilt):
    with pytest.raises(ValueError):
        dipole_axis_inertial(bad_tilt)


def test_b_field_direction_inertial_is_unit_and_varies_with_phase():
    dirs = [b_field_direction_inertial(p) for p in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    for d in dirs:
        assert np.isclose(np.linalg.norm(d), 1.0)
    # Direction must actually change over the orbit (not constant).
    assert not all(np.allclose(dirs[0], d, atol=1e-6) for d in dirs[1:])


def test_b_field_inertial_tesla_scales_with_magnitude():
    b1 = b_field_inertial_tesla(1.0, magnitude_tesla=3.0e-5)
    b2 = b_field_inertial_tesla(1.0, magnitude_tesla=6.0e-5)
    assert np.isclose(np.linalg.norm(b2), 2.0 * np.linalg.norm(b1))


def test_b_field_inertial_tesla_rejects_bad_magnitude():
    with pytest.raises(ValueError):
        b_field_inertial_tesla(0.0, magnitude_tesla=-1.0)


# ---------------------------------------------------------------------------
# Sun-mag separation angle genuinely varies over the orbit (regression test
# for the "same DCM applied to both vectors -> constant separation" bug
# caught and fixed during Milestone 2 development)
# ---------------------------------------------------------------------------

def test_sun_mag_separation_angle_varies_over_orbit():
    beta = 20.0
    seps = []
    for phase in np.linspace(0, 2 * np.pi, 40, endpoint=False):
        sun_b, mag_b = sun_and_mag_body_vectors(phase, beta)
        cos_sep = np.clip(np.dot(sun_b, mag_b), -1.0, 1.0)
        seps.append(np.degrees(np.arccos(cos_sep)))
    seps = np.array(seps)
    assert seps.max() - seps.min() > 5.0  # meaningfully varying, not flat


def test_sun_and_mag_body_vectors_are_unit():
    for phase in [0.0, 1.0, 3.0, 5.5]:
        sun_b, mag_b = sun_and_mag_body_vectors(phase, 20.0)
        assert np.isclose(np.linalg.norm(sun_b), 1.0)
        assert np.isclose(np.linalg.norm(mag_b), 1.0)


# ---------------------------------------------------------------------------
# environment_state / environment_states_batch consistency
# ---------------------------------------------------------------------------

def test_environment_state_rejects_bad_period():
    with pytest.raises(ValueError):
        environment_state(0.0, 0.0, 600.0, 20.0)


def test_environment_state_fields_self_consistent():
    period = orbital_period_s(600.0)
    env = environment_state(period / 4.0, period, 600.0, 20.0)
    assert np.isclose(np.linalg.norm(env.sun_body_true), 1.0)
    assert np.isclose(np.linalg.norm(env.mag_body_true), 1.0)
    assert isinstance(env.in_eclipse, (bool, np.bool_))
    assert 0.0 <= env.star_tracker_bright_source_angle_rad <= np.pi


def test_environment_states_batch_matches_scalar_environment_state():
    period = orbital_period_s(600.0)
    ts = np.linspace(0, period, 37, endpoint=False)
    batch = environment_states_batch(ts, period, 600.0, 20.0)
    for i, t in enumerate(ts):
        env = environment_state(t, period, 600.0, 20.0)
        assert np.allclose(env.attitude_rotvec_rad, batch.attitude_rotvec_rad[i], atol=1e-10)
        assert env.in_eclipse == bool(batch.in_eclipse[i])
        assert np.isclose(
            env.star_tracker_bright_source_angle_rad,
            batch.star_tracker_bright_source_angle_rad[i],
            atol=1e-10,
        )
        assert np.isclose(env.b_field_inertial_tesla[0], batch.b_field_inertial_tesla[i, 0], atol=1e-15)


def test_environment_states_batch_separation_angle_matches_manual_computation():
    period = orbital_period_s(600.0)
    ts = np.array([0.0, period / 8, period / 3, period / 2])
    batch = environment_states_batch(ts, period, 600.0, 20.0)
    for i, t in enumerate(ts):
        sun_b, mag_b = sun_and_mag_body_vectors(2 * np.pi * t / period, 20.0)
        cos_sep = np.clip(np.dot(sun_b, mag_b), -1.0, 1.0)
        expected_deg = np.degrees(np.arccos(cos_sep))
        assert np.isclose(expected_deg, batch.separation_angle_deg[i], atol=1e-8)


def test_environment_states_batch_rejects_bad_inputs():
    with pytest.raises(ValueError):
        environment_states_batch(np.array([0.0]), -1.0, 600.0, 20.0)
    with pytest.raises(ValueError):
        environment_states_batch(np.array([0.0]), 100.0, 600.0, 91.0)


# ---------------------------------------------------------------------------
# Reproducibility (fully deterministic, no RNG anywhere in this module)
# ---------------------------------------------------------------------------

def test_orbit_environment_is_fully_deterministic_across_repeated_calls():
    period = orbital_period_s(600.0)
    ts = np.linspace(0, period, 101)
    b1 = environment_states_batch(ts, period, 600.0, 20.0)
    b2 = environment_states_batch(ts, period, 600.0, 20.0)
    assert np.array_equal(b1.attitude_rotvec_rad, b2.attitude_rotvec_rad)
    assert np.array_equal(b1.in_eclipse, b2.in_eclipse)
    assert np.array_equal(b1.separation_angle_deg, b2.separation_angle_deg)
