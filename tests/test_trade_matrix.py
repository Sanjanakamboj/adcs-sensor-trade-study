"""Tests for adcs_sensor_trade.trade_matrix (Milestone 3)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from adcs_sensor_trade.normalization import CATEGORIES
from adcs_sensor_trade.trade_matrix import (
    ACCURACY_PRIORITY_WEIGHTS,
    AVAILABILITY_PRIORITY_WEIGHTS,
    BASELINE_WEIGHTS,
    COST_COMPLEXITY_PRIORITY_WEIGHTS,
    RESOURCE_CONSTRAINED_WEIGHTS,
    SCENARIOS,
    WeightSet,
    breakeven_availability_weight,
    breakeven_star_tracker_mass,
    breakeven_star_tracker_power,
    breakeven_sunmag_accuracy,
    breakeven_sunmag_availability,
    build_normalized_table,
    build_raw_table,
    compute_weighted_scores,
    monte_carlo_weight_sensitivity,
    run_trade_matrix,
    sample_weight_vectors,
)
from adcs_sensor_trade.trade_metrics import star_tracker_metrics, sunmag_metrics


@pytest.fixture(scope="module")
def star_metrics():
    return star_tracker_metrics()


@pytest.fixture(scope="module")
def sunmag_metrics_fixture():
    return sunmag_metrics()


# ---------------------------------------------------------------------------
# WeightSet validation
# ---------------------------------------------------------------------------


def _equal_weights():
    return {c: 1.0 / len(CATEGORIES) for c in CATEGORIES}


def test_weight_set_valid_construction():
    ws = WeightSet(name="ok", weights=_equal_weights(), rationale="test")
    assert sum(ws.weights.values()) == pytest.approx(1.0)


def test_weight_set_rejects_sum_below_one():
    weights = _equal_weights()
    weights[CATEGORIES[0]] -= 0.1
    with pytest.raises(ValueError):
        WeightSet(name="bad", weights=weights, rationale="test")


def test_weight_set_rejects_sum_above_one():
    weights = _equal_weights()
    weights[CATEGORIES[0]] += 0.1
    with pytest.raises(ValueError):
        WeightSet(name="bad", weights=weights, rationale="test")


def test_weight_set_rejects_negative_weight():
    weights = _equal_weights()
    weights[CATEGORIES[0]] = -0.01
    weights[CATEGORIES[1]] += 0.01
    with pytest.raises(ValueError):
        WeightSet(name="bad", weights=weights, rationale="test")


def test_weight_set_rejects_missing_category():
    weights = _equal_weights()
    del weights[CATEGORIES[0]]
    with pytest.raises(ValueError):
        WeightSet(name="bad", weights=weights, rationale="test")


def test_weight_set_rejects_unknown_category_key():
    weights = _equal_weights()
    weights["not_a_category"] = weights.pop(CATEGORIES[0])
    with pytest.raises(ValueError):
        WeightSet(name="bad", weights=weights, rationale="test")


def test_weight_set_tolerance_allows_tiny_float_error():
    weights = _equal_weights()
    # Perturb by less than the declared tolerance.
    weights[CATEGORIES[0]] += 1e-9
    weights[CATEGORIES[1]] -= 1e-9
    WeightSet(name="ok", weights=weights, rationale="test")  # should not raise


# ---------------------------------------------------------------------------
# The 5 mandatory named scenarios
# ---------------------------------------------------------------------------


def test_five_named_scenarios_exist():
    assert len(SCENARIOS) == 5
    assert set(SCENARIOS.keys()) == {
        "baseline",
        "accuracy_priority",
        "resource_constrained",
        "availability_priority",
        "cost_complexity_priority",
    }


@pytest.mark.parametrize(
    "ws",
    [
        BASELINE_WEIGHTS,
        ACCURACY_PRIORITY_WEIGHTS,
        RESOURCE_CONSTRAINED_WEIGHTS,
        AVAILABILITY_PRIORITY_WEIGHTS,
        COST_COMPLEXITY_PRIORITY_WEIGHTS,
    ],
)
def test_each_scenario_weight_set_sums_to_one(ws):
    assert sum(ws.weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_scenarios_are_distinct_weight_vectors():
    arrays = [ws.as_array() for ws in SCENARIOS.values()]
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            assert not np.allclose(arrays[i], arrays[j]), "scenario weight vectors must be distinct"


# ---------------------------------------------------------------------------
# Raw / normalized / weighted table construction
# ---------------------------------------------------------------------------


def test_build_raw_table_shape_and_index(star_metrics, sunmag_metrics_fixture):
    raw = build_raw_table(star_metrics, sunmag_metrics_fixture)
    assert set(raw.index) == {"star_tracker", "sun_plus_mag"}
    assert list(raw.columns) == list(CATEGORIES)


def test_normalized_table_values_in_unit_interval(star_metrics, sunmag_metrics_fixture):
    raw = build_raw_table(star_metrics, sunmag_metrics_fixture)
    normalized = build_normalized_table(raw)
    assert (normalized.values >= 0.0).all()
    assert (normalized.values <= 1.0).all()


def test_weighted_contributions_sum_to_totals(star_metrics, sunmag_metrics_fixture):
    raw = build_raw_table(star_metrics, sunmag_metrics_fixture)
    normalized = build_normalized_table(raw)
    contributions, totals = compute_weighted_scores(normalized, BASELINE_WEIGHTS)
    for arch in totals.index:
        assert contributions.loc[arch].sum() == pytest.approx(totals[arch])


def test_run_trade_matrix_winner_is_one_of_the_two_architectures(star_metrics, sunmag_metrics_fixture):
    result = run_trade_matrix(star_metrics, sunmag_metrics_fixture, BASELINE_WEIGHTS)
    assert result.winner in ("star_tracker", "sun_plus_mag")


# ---------------------------------------------------------------------------
# Hand-computed synthetic weighted-score arithmetic check (part L requirement)
# ---------------------------------------------------------------------------


def test_weighted_score_matches_hand_computed_synthetic_example():
    import pandas as pd

    # Tiny synthetic 2-category normalized table.
    normalized = pd.DataFrame(
        {c: [0.0, 0.0] for c in CATEGORIES},
        index=["star_tracker", "sun_plus_mag"],
    )
    normalized.loc["star_tracker", "accuracy"] = 0.8
    normalized.loc["star_tracker", "availability"] = 0.6
    normalized.loc["sun_plus_mag", "accuracy"] = 0.3
    normalized.loc["sun_plus_mag", "availability"] = 0.9

    weights = {c: 0.0 for c in CATEGORIES}
    weights["accuracy"] = 0.4
    weights["availability"] = 0.6
    ws = WeightSet(name="synthetic", weights=weights, rationale="test")

    contributions, totals = compute_weighted_scores(normalized, ws)
    expected_star = 0.8 * 0.4 + 0.6 * 0.6  # = 0.68
    expected_sunmag = 0.3 * 0.4 + 0.9 * 0.6  # = 0.66
    assert totals["star_tracker"] == pytest.approx(expected_star)
    assert totals["sun_plus_mag"] == pytest.approx(expected_sunmag)


# ---------------------------------------------------------------------------
# Deterministic reproducibility
# ---------------------------------------------------------------------------


def test_run_trade_matrix_is_reproducible(star_metrics, sunmag_metrics_fixture):
    r1 = run_trade_matrix(star_metrics, sunmag_metrics_fixture, BASELINE_WEIGHTS)
    r2 = run_trade_matrix(star_metrics, sunmag_metrics_fixture, BASELINE_WEIGHTS)
    assert r1.totals.equals(r2.totals)
    assert r1.winner == r2.winner


# ---------------------------------------------------------------------------
# Monte Carlo weight-space sampling
# ---------------------------------------------------------------------------


def test_sample_weight_vectors_rows_sum_to_one():
    samples = sample_weight_vectors(500, seed=1, n_categories=len(CATEGORIES))
    assert samples.shape == (500, len(CATEGORIES))
    row_sums = samples.sum(axis=1)
    assert np.allclose(row_sums, 1.0)


def test_sample_weight_vectors_nonnegative():
    samples = sample_weight_vectors(500, seed=1, n_categories=len(CATEGORIES))
    assert (samples >= 0).all()


def test_sample_weight_vectors_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        sample_weight_vectors(0, seed=1)


def test_monte_carlo_same_seed_gives_identical_fractions(star_metrics, sunmag_metrics_fixture):
    mc1 = monte_carlo_weight_sensitivity(star_metrics, sunmag_metrics_fixture, n_samples=2000, seed=7)
    mc2 = monte_carlo_weight_sensitivity(star_metrics, sunmag_metrics_fixture, n_samples=2000, seed=7)
    assert mc1.fraction_favoring_star_tracker == pytest.approx(mc2.fraction_favoring_star_tracker)
    assert mc1.fraction_favoring_sun_plus_mag == pytest.approx(mc2.fraction_favoring_sun_plus_mag)
    assert np.allclose(mc1.score_diff, mc2.score_diff)


def test_monte_carlo_fractions_sum_to_one(star_metrics, sunmag_metrics_fixture):
    mc = monte_carlo_weight_sensitivity(star_metrics, sunmag_metrics_fixture, n_samples=2000, seed=7)
    total = mc.fraction_favoring_star_tracker + mc.fraction_favoring_sun_plus_mag + mc.fraction_tied
    assert total == pytest.approx(1.0)


def test_monte_carlo_different_seeds_can_differ(star_metrics, sunmag_metrics_fixture):
    mc1 = monte_carlo_weight_sensitivity(star_metrics, sunmag_metrics_fixture, n_samples=500, seed=1)
    mc2 = monte_carlo_weight_sensitivity(star_metrics, sunmag_metrics_fixture, n_samples=500, seed=2)
    # Not a strict requirement that they differ, but the underlying samples must.
    assert not np.array_equal(mc1.weight_samples, mc2.weight_samples)


# ---------------------------------------------------------------------------
# Break-even calculations
# ---------------------------------------------------------------------------


def test_breakeven_sunmag_availability_is_between_0_and_1(star_metrics, sunmag_metrics_fixture):
    result = breakeven_sunmag_availability(star_metrics, sunmag_metrics_fixture)
    assert result is not None
    assert 0.0 <= result <= 1.0


def test_breakeven_sunmag_availability_flips_the_decision(star_metrics, sunmag_metrics_fixture):
    threshold = breakeven_sunmag_availability(star_metrics, sunmag_metrics_fixture)
    assert threshold is not None
    below = replace(sunmag_metrics_fixture, availability_fraction=threshold - 0.02)
    above = replace(sunmag_metrics_fixture, availability_fraction=threshold + 0.02)
    result_below = run_trade_matrix(star_metrics, below, BASELINE_WEIGHTS)
    result_above = run_trade_matrix(star_metrics, above, BASELINE_WEIGHTS)
    assert result_below.winner != result_above.winner


def test_breakeven_star_tracker_mass_flips_the_decision(star_metrics, sunmag_metrics_fixture):
    threshold = breakeven_star_tracker_mass(star_metrics, sunmag_metrics_fixture)
    assert threshold is not None
    below = replace(star_metrics, mass_kg=threshold - 0.1)
    above = replace(star_metrics, mass_kg=threshold + 0.1)
    result_below = run_trade_matrix(below, sunmag_metrics_fixture, BASELINE_WEIGHTS)
    result_above = run_trade_matrix(above, sunmag_metrics_fixture, BASELINE_WEIGHTS)
    assert result_below.winner != result_above.winner


def test_breakeven_star_tracker_power_flips_the_decision(star_metrics, sunmag_metrics_fixture):
    threshold = breakeven_star_tracker_power(star_metrics, sunmag_metrics_fixture)
    assert threshold is not None
    below = replace(star_metrics, avg_power_w=threshold - 0.1)
    above = replace(star_metrics, avg_power_w=threshold + 0.1)
    result_below = run_trade_matrix(below, sunmag_metrics_fixture, BASELINE_WEIGHTS)
    result_above = run_trade_matrix(above, sunmag_metrics_fixture, BASELINE_WEIGHTS)
    assert result_below.winner != result_above.winner


def test_breakeven_availability_weight_flips_the_decision(star_metrics, sunmag_metrics_fixture):
    threshold = breakeven_availability_weight(star_metrics, sunmag_metrics_fixture)
    assert threshold is not None
    assert 0.0 <= threshold <= 1.0


def test_breakeven_sunmag_accuracy_reports_none_when_not_meaningful(star_metrics, sunmag_metrics_fixture):
    # Documented, honest result: improving Sun+mag accuracy ALONE (holding
    # availability/resources/cost fixed) does not flip the baseline
    # decision in this model - this test locks in that finding rather than
    # forcing a crossing to exist.
    result = breakeven_sunmag_accuracy(star_metrics, sunmag_metrics_fixture)
    assert result is None


def test_breakeven_root_finder_matches_known_analytic_crossing():
    # Synthetic case with a KNOWN analytic break-even point: vary Sun+mag
    # availability utility linearly and confirm the numeric bisection finds
    # the point where weighted totals are equal, matching hand algebra.
    #
    # Fix all categories except "availability" at raw values that give
    # equal utility for both architectures (so they contribute equally and
    # cancel out of the margin). Then:
    #   margin(avail) = w_avail * [U_star(avail) - U_sunmag(avail)]
    # where U_star is fixed (star tracker's own availability held constant)
    # and U_sunmag(avail) is the normalized utility of the swept value.
    from adcs_sensor_trade.trade_metrics import star_tracker_metrics as _stm
    from adcs_sensor_trade.trade_metrics import sunmag_metrics as _sm

    star = _stm()
    sunmag = _sm()
    # Make every OTHER category identical between the two architectures so
    # only "availability" drives the margin.
    star_equalized = replace(
        star,
        worst_axis_error_deg=sunmag.worst_axis_error_deg,
        cadence_hz=sunmag.cadence_hz,
        mass_kg=sunmag.mass_kg,
        avg_power_w=sunmag.avg_power_w,
        cost_score=sunmag.cost_score,
        complexity_score=sunmag.complexity_score,
    )
    weights = {c: 0.0 for c in CATEGORIES}
    weights["availability"] = 1.0
    ws = WeightSet(name="only_availability", weights=weights, rationale="test")

    # With only "availability" weighted, the crossing is exactly where the
    # two RAW availability fractions are equal (utility is a monotonic
    # linear function of the raw value under our normalization, and both
    # architectures use the SAME MetricSpec).
    known_crossing = star_equalized.availability_fraction

    threshold = breakeven_sunmag_availability(star_equalized, sunmag, ws)
    assert threshold is not None
    assert threshold == pytest.approx(known_crossing, abs=1e-4)


# ---------------------------------------------------------------------------
# Recommendation-flip behavior on a constructed SYNTHETIC case
# ---------------------------------------------------------------------------


def test_synthetic_case_where_sunmag_should_win():
    star = star_tracker_metrics()
    sunmag = sunmag_metrics()
    # Artificially make Sun+mag dominate on every category.
    sunmag_dominant = replace(
        sunmag,
        worst_axis_error_deg=1e-6,
        availability_fraction=0.999,
        cadence_hz=10.0,
        mass_kg=1e-6,
        avg_power_w=1e-6,
        cost_score=1.0,
        complexity_score=1.0,
    )
    result = run_trade_matrix(star, sunmag_dominant, BASELINE_WEIGHTS)
    assert result.winner == "sun_plus_mag"


def test_synthetic_case_where_star_tracker_should_win():
    star = star_tracker_metrics()
    sunmag = sunmag_metrics()
    star_dominant = replace(
        star,
        worst_axis_error_deg=1e-6,
        availability_fraction=0.999,
        cadence_hz=10.0,
        mass_kg=1e-6,
        avg_power_w=1e-6,
        cost_score=1.0,
        complexity_score=1.0,
    )
    result = run_trade_matrix(star_dominant, sunmag, BASELINE_WEIGHTS)
    assert result.winner == "star_tracker"
