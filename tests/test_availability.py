"""Tests for adcs_sensor_trade.availability (Milestone 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adcs_sensor_trade.availability import (
    DEFAULT_CONDITION_NUMBER_THRESHOLD,
    OrbitAvailabilityConfig,
    async_sample_times,
    compute_full_attitude_update_cadence,
    default_sync_window_s,
    outage_intervals,
    run_availability_timeline,
    sensitivity_eclipse_beta_angle,
    sensitivity_exclusion_half_angle,
    sensitivity_fov_half_angle,
    sensitivity_update_rate,
    star_tracker_usable_updates,
    summarize_boolean_series,
)
from adcs_sensor_trade.sensors.magnetometer import MagnetometerConfig
from adcs_sensor_trade.sensors.star_tracker import StarTrackerConfig
from adcs_sensor_trade.sensors.sun_sensor import SunSensorConfig


# ---------------------------------------------------------------------------
# OrbitAvailabilityConfig validation
# ---------------------------------------------------------------------------

def test_default_config_is_valid():
    cfg = OrbitAvailabilityConfig()
    assert cfg.period_s > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"altitude_km": 0.0},
        {"altitude_km": -100.0},
        {"beta_angle_deg": -91.0},
        {"beta_angle_deg": 91.0},
        {"n_orbits": 0.0},
        {"n_orbits": -1.0},
        {"dt_s": 0.0},
        {"dt_s": -1.0},
        {"condition_number_threshold": 1.0},
        {"condition_number_threshold": 0.5},
        {"rank_tol": 0.0},
        {"b_field_magnitude_tesla": 0.0},
        {"dip_tilt_deg": -1.0},
        {"dip_tilt_deg": 91.0},
    ],
)
def test_invalid_config_raises_value_error(kwargs):
    with pytest.raises(ValueError):
        OrbitAvailabilityConfig(**kwargs)


def test_dt_larger_than_period_raises_value_error():
    with pytest.raises(ValueError):
        OrbitAvailabilityConfig(altitude_km=600.0, dt_s=1.0e6)


# ---------------------------------------------------------------------------
# run_availability_timeline: basic structure and reproducibility
# ---------------------------------------------------------------------------

def test_timeline_returns_expected_columns_and_row_count():
    cfg = OrbitAvailabilityConfig(dt_s=10.0, n_orbits=1.0)
    df = run_availability_timeline(cfg)
    expected_cols = {
        "t_s",
        "phase_deg",
        "in_eclipse",
        "star_tracker_available",
        "star_tracker_reason",
        "sun_sensor_available",
        "sun_sensor_reason",
        "magnetometer_available",
        "magnetometer_reason",
        "separation_angle_deg",
        "sunmag_vectors_usable",
        "sunmag_rank",
        "sunmag_condition_number",
        "sunmag_full_rank",
        "sunmag_well_conditioned",
        "sunmag_worst_axis_error_deg",
        "sunmag_available",
        "sunmag_reason",
    }
    assert expected_cols.issubset(set(df.columns))
    expected_n = int(cfg.n_orbits * cfg.period_s / cfg.dt_s)
    assert abs(len(df) - expected_n) <= 1


def test_timeline_is_deterministic_across_repeated_calls():
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    df1 = run_availability_timeline(cfg)
    df2 = run_availability_timeline(cfg)
    pd.testing.assert_frame_equal(df1, df2)


def test_timeline_two_orbits_is_roughly_double_one_orbit_length():
    cfg1 = OrbitAvailabilityConfig(dt_s=10.0, n_orbits=1.0)
    cfg2 = OrbitAvailabilityConfig(dt_s=10.0, n_orbits=2.0)
    df1 = run_availability_timeline(cfg1)
    df2 = run_availability_timeline(cfg2)
    assert abs(len(df2) - 2 * len(df1)) <= 2


# ---------------------------------------------------------------------------
# Eclipse / FOV / exclusion reason tagging
# ---------------------------------------------------------------------------

def test_star_tracker_reason_is_ok_or_sun_exclusion_only():
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    df = run_availability_timeline(cfg)
    assert set(df["star_tracker_reason"].unique()).issubset({"ok", "sun_exclusion"})


def test_star_tracker_reason_matches_availability_boolean():
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    df = run_availability_timeline(cfg)
    assert ((df["star_tracker_reason"] == "ok") == df["star_tracker_available"]).all()


def test_sun_sensor_reason_is_one_of_expected_values():
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    df = run_availability_timeline(cfg)
    assert set(df["sun_sensor_reason"].unique()).issubset({"ok", "eclipse", "fov"})


def test_eclipse_rows_are_always_sun_sensor_unavailable():
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    df = run_availability_timeline(cfg)
    eclipsed = df[df["in_eclipse"]]
    assert not eclipsed.empty
    assert (~eclipsed["sun_sensor_available"]).all()
    assert (eclipsed["sun_sensor_reason"] == "eclipse").all()


def test_magnetometer_always_available_in_this_model():
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    df = run_availability_timeline(cfg)
    assert df["magnetometer_available"].all()
    assert (df["magnetometer_reason"] == "ok").all()


def test_exclusion_half_angle_zero_means_star_tracker_never_excluded():
    star_cfg = StarTrackerConfig(exclusion_half_angle_rad=0.0)
    cfg = OrbitAvailabilityConfig(dt_s=10.0, star_tracker_config=star_cfg)
    df = run_availability_timeline(cfg)
    assert df["star_tracker_available"].all()


def test_exclusion_half_angle_pi_means_star_tracker_always_excluded():
    star_cfg = StarTrackerConfig(exclusion_half_angle_rad=np.pi)
    cfg = OrbitAvailabilityConfig(dt_s=10.0, star_tracker_config=star_cfg)
    df = run_availability_timeline(cfg)
    assert not df["star_tracker_available"].any()


def test_fov_half_angle_pi_means_sun_sensor_never_fov_blocked():
    sun_cfg = SunSensorConfig(fov_half_angle_rad=np.pi)
    cfg = OrbitAvailabilityConfig(dt_s=10.0, sun_sensor_config=sun_cfg)
    df = run_availability_timeline(cfg)
    # Every unavailable sample must now be attributable to eclipse, not FOV.
    unavailable = df[~df["sun_sensor_available"]]
    assert (unavailable["sun_sensor_reason"] == "eclipse").all()


# ---------------------------------------------------------------------------
# Full-rank vs. well-conditioned distinction
# ---------------------------------------------------------------------------

def test_full_rank_implies_vectors_usable():
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    df = run_availability_timeline(cfg)
    full_rank = df[df["sunmag_full_rank"]]
    assert (full_rank["sunmag_vectors_usable"]).all()


def test_well_conditioned_implies_full_rank():
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    df = run_availability_timeline(cfg)
    well_cond = df[df["sunmag_well_conditioned"]]
    assert (well_cond["sunmag_full_rank"]).all()


def test_sunmag_available_equals_well_conditioned():
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    df = run_availability_timeline(cfg)
    assert (df["sunmag_available"] == df["sunmag_well_conditioned"]).all()


def test_high_condition_number_threshold_never_flags_poorly_conditioned():
    cfg = OrbitAvailabilityConfig(dt_s=10.0, condition_number_threshold=1.0e9)
    df = run_availability_timeline(cfg)
    usable_full_rank = df[df["sunmag_full_rank"]]
    assert (usable_full_rank["sunmag_well_conditioned"]).all()


def test_tight_condition_number_threshold_can_flag_poorly_conditioned():
    cfg_loose = OrbitAvailabilityConfig(dt_s=10.0, condition_number_threshold=1.0e9)
    cfg_tight = OrbitAvailabilityConfig(dt_s=10.0, condition_number_threshold=1.01)
    df_loose = run_availability_timeline(cfg_loose)
    df_tight = run_availability_timeline(cfg_tight)
    # A stricter threshold can only shrink (never grow) the well-conditioned set.
    assert df_tight["sunmag_well_conditioned"].sum() <= df_loose["sunmag_well_conditioned"].sum()


def test_reason_tag_poorly_conditioned_appears_with_tight_threshold():
    cfg = OrbitAvailabilityConfig(dt_s=10.0, condition_number_threshold=1.001)
    df = run_availability_timeline(cfg)
    assert (df["sunmag_reason"] == "poorly_conditioned").any()


def test_rank_deficient_reason_only_when_vectors_usable_but_rank_below_3():
    # Force exact collinearity is hard in this orbit model, but we can at
    # least verify the reason-tagging LOGIC is internally consistent for
    # every sample produced.
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    df = run_availability_timeline(cfg)
    rank_def = df[df["sunmag_reason"] == "rank_deficient"]
    assert (rank_def["sunmag_vectors_usable"]).all()
    assert (rank_def["sunmag_full_rank"] == False).all()  # noqa: E712


def test_default_condition_number_threshold_constant_is_greater_than_one():
    assert DEFAULT_CONDITION_NUMBER_THRESHOLD > 1.0


# ---------------------------------------------------------------------------
# Singular / healthy two-vector geometry sanity (reuse of M1 geometry.py)
# ---------------------------------------------------------------------------

def test_geometry_module_singular_case_reused_correctly():
    from adcs_sensor_trade.geometry import geometry_rank

    collinear = [np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    assert geometry_rank(collinear) == 2
    anti_collinear = [np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])]
    assert geometry_rank(anti_collinear) == 2


def test_geometry_module_healthy_near_90_deg_case_reused_correctly():
    from adcs_sensor_trade.geometry import condition_number, geometry_rank

    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert geometry_rank([v1, v2]) == 3
    assert condition_number([v1, v2]) < 3.0


# ---------------------------------------------------------------------------
# Outage-interval detection (synthetic boolean timelines with known outages)
# ---------------------------------------------------------------------------

def test_outage_intervals_no_outages():
    available = np.ones(10, dtype=bool)
    assert outage_intervals(available, 1.0) == []


def test_outage_intervals_all_unavailable():
    available = np.zeros(5, dtype=bool)
    intervals = outage_intervals(available, 2.0)
    assert len(intervals) == 1
    assert intervals[0]["start_index"] == 0
    assert intervals[0]["end_index"] == 5
    assert intervals[0]["duration_s"] == 10.0


def test_outage_intervals_single_known_gap():
    # available = [T,T,F,F,F,T,T]
    available = np.array([True, True, False, False, False, True, True])
    intervals = outage_intervals(available, 1.0)
    assert len(intervals) == 1
    assert intervals[0]["start_index"] == 2
    assert intervals[0]["end_index"] == 5
    assert intervals[0]["duration_s"] == 3.0


def test_outage_intervals_multiple_gaps():
    # F,T,F,F,T,T,F
    available = np.array([False, True, False, False, True, True, False])
    intervals = outage_intervals(available, 1.0)
    assert len(intervals) == 3
    durations = [iv["duration_s"] for iv in intervals]
    assert durations == [1.0, 2.0, 1.0]


def test_outage_intervals_starts_and_ends_unavailable():
    available = np.array([False, False, True, True, False])
    intervals = outage_intervals(available, 0.5)
    assert len(intervals) == 2
    assert intervals[0]["duration_s"] == 1.0
    assert intervals[1]["duration_s"] == 0.5


def test_outage_intervals_rejects_bad_dt():
    with pytest.raises(ValueError):
        outage_intervals(np.array([True, False]), 0.0)


def test_outage_intervals_rejects_empty_array():
    with pytest.raises(ValueError):
        outage_intervals(np.array([], dtype=bool), 1.0)


def test_summarize_boolean_series_known_case():
    available = np.array([True, True, False, False, False, True, True, False])
    summary = summarize_boolean_series(available, 1.0)
    assert np.isclose(summary["fraction_available"], 4.0 / 8.0)
    assert summary["num_outages"] == 2
    assert summary["longest_outage_s"] == 3.0
    assert summary["mean_outage_s"] == 2.0
    assert summary["median_outage_s"] == 2.0


def test_summarize_boolean_series_all_available():
    available = np.ones(20, dtype=bool)
    summary = summarize_boolean_series(available, 1.0)
    assert summary["fraction_available"] == 1.0
    assert summary["num_outages"] == 0
    assert summary["longest_outage_s"] == 0.0


def test_summarize_boolean_series_rejects_empty():
    with pytest.raises(ValueError):
        summarize_boolean_series(np.array([], dtype=bool), 1.0)


def test_outage_intervals_matches_summarize_longest_outage_for_real_timeline():
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    df = run_availability_timeline(cfg)
    summary = summarize_boolean_series(df["sun_sensor_available"].values, cfg.dt_s)
    intervals = outage_intervals(df["sun_sensor_available"].values, cfg.dt_s)
    if intervals:
        assert summary["longest_outage_s"] == max(iv["duration_s"] for iv in intervals)
    else:
        assert summary["longest_outage_s"] == 0.0


# ---------------------------------------------------------------------------
# Asynchronous sampling / update-cadence
# ---------------------------------------------------------------------------

def test_async_sample_times_count_and_spacing():
    times = async_sample_times(2.0, 10.0)
    assert times[0] == 0.0
    assert np.allclose(np.diff(times), 0.5)
    assert times[-1] <= 10.0


def test_async_sample_times_rejects_bad_inputs():
    with pytest.raises(ValueError):
        async_sample_times(0.0, 10.0)
    with pytest.raises(ValueError):
        async_sample_times(2.0, 0.0)


def test_default_sync_window_is_half_slower_sensor_period():
    w = default_sync_window_s(5.0, 10.0)
    assert np.isclose(w, 0.5 / 5.0)
    w2 = default_sync_window_s(10.0, 5.0)
    assert np.isclose(w2, 0.5 / 5.0)  # symmetric: slower rate governs


def test_default_sync_window_rejects_bad_rates():
    with pytest.raises(ValueError):
        default_sync_window_s(0.0, 10.0)


def test_compute_full_attitude_update_cadence_basic_structure():
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    result = compute_full_attitude_update_cadence(cfg)
    assert result["count"] >= 0
    assert result["duration_s"] > 0
    assert result["cadence_hz"] == result["count"] / result["duration_s"]
    assert result["updates_per_orbit"] == result["count"] / cfg.n_orbits
    assert isinstance(result["events"], pd.DataFrame)
    assert len(result["events"]) > 0


def test_compute_full_attitude_update_cadence_deterministic():
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    r1 = compute_full_attitude_update_cadence(cfg)
    r2 = compute_full_attitude_update_cadence(cfg)
    assert r1["count"] == r2["count"]
    assert r1["cadence_hz"] == r2["cadence_hz"]


def test_compute_full_attitude_update_cadence_zero_when_sun_sensor_disabled():
    # A star tracker exclusion-style trick: make FOV essentially unreachable
    # by shrinking it to ~0, forcing the Sun sensor (and hence Sun+mag) to
    # never be usable.
    sun_cfg = SunSensorConfig(fov_half_angle_rad=1e-6)
    cfg = OrbitAvailabilityConfig(dt_s=10.0, sun_sensor_config=sun_cfg)
    result = compute_full_attitude_update_cadence(cfg)
    assert result["count"] == 0


def test_faster_magnetometer_alone_does_not_increase_usable_updates():
    """Regression test for the documented physical expectation (Milestone 2
    question 7): since a magnetometer alone is rank-2 and Sun-sensor
    availability is the bottleneck, raising the magnetometer's rate alone
    should not materially change the Sun+mag usable-update count."""
    cfg_slow_mag = OrbitAvailabilityConfig(
        dt_s=10.0, magnetometer_config=MagnetometerConfig(update_rate_hz=10.0)
    )
    cfg_fast_mag = OrbitAvailabilityConfig(
        dt_s=10.0, magnetometer_config=MagnetometerConfig(update_rate_hz=100.0)
    )
    r_slow = compute_full_attitude_update_cadence(cfg_slow_mag)
    r_fast = compute_full_attitude_update_cadence(cfg_fast_mag)
    assert r_slow["count"] == r_fast["count"]


def test_star_tracker_usable_updates_scales_with_rate():
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    r_low = star_tracker_usable_updates(cfg, star_rate_hz=1.0)
    r_high = star_tracker_usable_updates(cfg, star_rate_hz=4.0)
    assert r_high["updates_per_orbit"] > r_low["updates_per_orbit"]


def test_star_tracker_usable_updates_rejects_bad_rate():
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    with pytest.raises(ValueError):
        star_tracker_usable_updates(cfg, star_rate_hz=0.0)


# ---------------------------------------------------------------------------
# Sensitivity studies: monotonic-behavior checks
# ---------------------------------------------------------------------------

def test_sensitivity_exclusion_half_angle_is_monotonic_non_increasing():
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    df = sensitivity_exclusion_half_angle(cfg, np.array([5.0, 15.0, 30.0, 45.0, 60.0, 90.0]))
    diffs = np.diff(df["star_tracker_availability_fraction"].values)
    assert np.all(diffs <= 1e-9)


def test_sensitivity_fov_half_angle_is_monotonic_non_decreasing():
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    df = sensitivity_fov_half_angle(cfg, np.array([15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 179.0]))
    diffs = np.diff(df["sun_sensor_availability_fraction"].values)
    assert np.all(diffs >= -1e-9)


def test_sensitivity_eclipse_beta_angle_eclipse_fraction_monotonic():
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    df = sensitivity_eclipse_beta_angle(cfg, np.array([0.0, 15.0, 30.0, 45.0, 60.0]))
    diffs = np.diff(df["eclipse_fraction"].values)
    assert np.all(diffs <= 1e-12)


def test_sensitivity_update_rate_star_tracker_increasing():
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    df = sensitivity_update_rate(cfg, star_rates_hz=np.array([1.0, 2.0, 4.0]))
    vals = df["updates_per_orbit"].values
    assert np.all(np.diff(vals) > 0)


def test_sensitivity_update_rate_returns_empty_when_nothing_requested():
    cfg = OrbitAvailabilityConfig(dt_s=10.0)
    df = sensitivity_update_rate(cfg)
    assert df.empty
