"""Tests for adcs_sensor_trade.sweep: the deterministic separation-angle sweep."""

import numpy as np

from adcs_sensor_trade.sweep import run_separation_sweep, sun_mag_body_vectors


SIGMA_SUN = np.radians(0.5)
SIGMA_MAG = np.radians(0.5)
SIGMA_STAR = np.radians(30.0 / 3600.0)


def test_sun_mag_body_vectors_are_unit_and_separated_correctly():
    for angle in [1, 45, 90, 135, 179]:
        s, m = sun_mag_body_vectors(angle)
        assert np.isclose(np.linalg.norm(s), 1.0)
        assert np.isclose(np.linalg.norm(m), 1.0)
        cos_theta = np.dot(s, m)
        assert np.isclose(cos_theta, np.cos(np.radians(angle)), atol=1e-10)


def test_sweep_rank_three_away_from_singularities():
    angles = np.array([1, 30, 60, 90, 120, 150, 179])
    df = run_separation_sweep(angles, SIGMA_SUN, SIGMA_MAG, SIGMA_STAR)
    assert (df["rank"] == 3).all()


def test_sweep_best_condition_number_near_90_deg():
    angles = np.arange(10, 171, 10)
    df = run_separation_sweep(angles, SIGMA_SUN, SIGMA_MAG, SIGMA_STAR)
    idx_min = df["sunmag_condition_number_J"].idxmin()
    best_angle = df.loc[idx_min, "separation_angle_deg"]
    assert abs(best_angle - 90.0) <= 10.0


def test_sweep_condition_number_degrades_monotonically_toward_edges():
    angles_low = np.array([1, 5, 10, 20, 40, 60, 90])
    df_low = run_separation_sweep(angles_low, SIGMA_SUN, SIGMA_MAG, SIGMA_STAR)
    diffs_low = np.diff(df_low["sunmag_condition_number_J"].values)
    assert np.all(diffs_low <= 1e-9)

    angles_high = np.array([90, 120, 140, 160, 170, 175, 179])
    df_high = run_separation_sweep(angles_high, SIGMA_SUN, SIGMA_MAG, SIGMA_STAR)
    diffs_high = np.diff(df_high["sunmag_condition_number_J"].values)
    assert np.all(diffs_high >= -1e-9)


def test_sweep_star_tracker_reference_is_flat_across_angles():
    angles = np.array([1, 45, 90, 135, 179])
    df = run_separation_sweep(angles, SIGMA_SUN, SIGMA_MAG, SIGMA_STAR)
    star_vals = df["star_worst_axis_error_deg"].values
    assert np.ptp(star_vals) < 1e-9  # flat: independent of separation angle


def test_sweep_singularity_at_exactly_0_and_180_degrees():
    df = run_separation_sweep(np.array([0.0, 180.0]), SIGMA_SUN, SIGMA_MAG, SIGMA_STAR)
    assert (df["rank"] == 2).all()
    assert (df["min_singular_value"] < 1e-8).all()
    assert np.all(np.isinf(df["sunmag_condition_number_J"].values))


def test_sweep_reproducible_across_repeated_calls():
    angles = np.linspace(1, 179, 25)
    df1 = run_separation_sweep(angles, SIGMA_SUN, SIGMA_MAG, SIGMA_STAR)
    df2 = run_separation_sweep(angles, SIGMA_SUN, SIGMA_MAG, SIGMA_STAR)
    numeric_cols = [
        "min_singular_value",
        "sunmag_worst_axis_error_deg",
        "sunmag_rms_error_deg",
        "star_worst_axis_error_deg",
    ]
    for col in numeric_cols:
        assert np.allclose(df1[col].values, df2[col].values, equal_nan=True)
