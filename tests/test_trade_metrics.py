"""Tests for adcs_sensor_trade.trade_metrics (Milestone 3).

IMPORTANT: several tests here are REGRESSION tests against the ACCEPTED
Milestone 1/2 headline numbers (see the Milestone 3 task specification).
They exist to prove that trade_metrics.py's extraction of real numbers
from the (unmodified) M1/M2 modules has not silently diverged from the
accepted results.
"""

from __future__ import annotations

import pytest

from adcs_sensor_trade.trade_metrics import (
    baseline_availability_config,
    star_tracker_metrics,
    sunmag_metrics,
)

TOL = 1e-3  # tight relative/absolute tolerance for regression checks


def test_baseline_config_matches_verification_script_baseline():
    cfg = baseline_availability_config()
    assert cfg.altitude_km == pytest.approx(600.0)
    assert cfg.beta_angle_deg == pytest.approx(20.0)
    assert cfg.n_orbits == pytest.approx(1.0)
    assert cfg.dt_s == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Star tracker regression checks against ACCEPTED M1/M2 numbers
# ---------------------------------------------------------------------------


def test_star_tracker_worst_axis_error_matches_accepted_30_arcsec():
    m = star_tracker_metrics()
    assert m.worst_axis_error_deg == pytest.approx(30.0 / 3600.0, abs=1e-6)


def test_star_tracker_condition_number_is_isotropic_1():
    m = star_tracker_metrics()
    assert m.condition_number == pytest.approx(1.0)


def test_star_tracker_full_attitude_observable():
    m = star_tracker_metrics()
    assert m.full_attitude_observable is True


def test_star_tracker_availability_matches_accepted_87_30_percent():
    m = star_tracker_metrics()
    assert m.availability_fraction * 100 == pytest.approx(87.30, abs=0.01)


def test_star_tracker_longest_outage_matches_accepted_369s():
    m = star_tracker_metrics()
    assert m.longest_outage_s == pytest.approx(369.0, abs=0.5)


def test_star_tracker_usable_updates_per_orbit_matches_accepted_10130():
    m = star_tracker_metrics()
    assert m.updates_per_orbit == pytest.approx(10130.0, abs=1.0)


def test_star_tracker_effective_cadence_matches_accepted_1_7462_hz():
    m = star_tracker_metrics()
    assert m.cadence_hz == pytest.approx(1.7462, abs=1e-3)


def test_star_tracker_poorly_conditioned_fraction_is_zero():
    m = star_tracker_metrics()
    assert m.poorly_conditioned_fraction == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Sun+mag regression checks against ACCEPTED M1/M2 numbers
# ---------------------------------------------------------------------------


def test_sunmag_availability_matches_accepted_23_18_percent():
    m = sunmag_metrics()
    assert m.availability_fraction * 100 == pytest.approx(23.18, abs=0.01)


def test_sunmag_longest_outage_matches_accepted_3939s():
    m = sunmag_metrics()
    assert m.longest_outage_s == pytest.approx(3939.0, abs=0.5)


def test_sunmag_usable_updates_per_orbit_matches_accepted_6723():
    m = sunmag_metrics()
    assert m.updates_per_orbit == pytest.approx(6723.0, abs=1.0)


def test_sunmag_effective_cadence_matches_accepted_1_1589_hz():
    m = sunmag_metrics()
    assert m.cadence_hz == pytest.approx(1.1589, abs=1e-3)


def test_sunmag_median_condition_number_matches_accepted_1_903():
    m = sunmag_metrics()
    assert m.condition_number == pytest.approx(1.903, abs=1e-3)


def test_sunmag_poorly_conditioned_fraction_matches_accepted_zero_percent():
    m = sunmag_metrics()
    assert m.poorly_conditioned_fraction * 100 == pytest.approx(0.0, abs=1e-6)


def test_sunmag_full_attitude_observable_when_available():
    m = sunmag_metrics()
    assert m.full_attitude_observable is True


def test_condition_number_at_90_deg_separation_matches_accepted_2_0():
    # Direct M1 regression check (independent of the availability timeline):
    # exercises geometry.py/information.py at the accepted 90 deg
    # separation case (cond(J) = 2.0, worst-axis = 0.5 deg for equal 0.5 deg
    # sigmas).
    from adcs_sensor_trade.information import combined_vector_information, summarize_information
    from adcs_sensor_trade.sensors.sun_sensor import SunSensorConfig
    from adcs_sensor_trade.sensors.magnetometer import MagnetometerConfig
    from adcs_sensor_trade.sweep import sun_mag_body_vectors

    sun_body, mag_body = sun_mag_body_vectors(90.0)
    sigma_sun = SunSensorConfig().sigma_rad
    sigma_mag = MagnetometerConfig().sigma_rad
    J = combined_vector_information([sun_body, mag_body], [sigma_sun, sigma_mag])
    report = summarize_information(J)
    assert report.condition_number_J == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# M3 resource / cost-complexity fields present on ArchitectureMetrics
# ---------------------------------------------------------------------------


def test_star_tracker_metrics_carries_resource_and_cost_fields():
    m = star_tracker_metrics()
    assert m.mass_kg > 0
    assert m.avg_power_w > 0
    assert m.peak_power_w >= m.avg_power_w
    assert 1.0 <= m.cost_score <= 5.0
    assert 1.0 <= m.complexity_score <= 5.0


def test_sunmag_metrics_resource_totals_equal_component_sums():
    m = sunmag_metrics()
    from adcs_sensor_trade.resource_model import sunmag_architecture_totals

    totals = sunmag_architecture_totals()
    assert m.mass_kg == pytest.approx(totals.mass_kg)
    assert m.avg_power_w == pytest.approx(totals.avg_power_w)


def test_metrics_are_reproducible_across_two_calls():
    m1 = star_tracker_metrics()
    m2 = star_tracker_metrics()
    assert m1.availability_fraction == m2.availability_fraction
    assert m1.longest_outage_s == m2.longest_outage_s
    assert m1.cadence_hz == m2.cadence_hz

    s1 = sunmag_metrics()
    s2 = sunmag_metrics()
    assert s1.availability_fraction == s2.availability_fraction
    assert s1.condition_number == s2.condition_number
