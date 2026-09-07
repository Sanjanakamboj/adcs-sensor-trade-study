"""Tests for adcs_sensor_trade.normalization (Milestone 3)."""

from __future__ import annotations

import numpy as np
import pytest

from adcs_sensor_trade.normalization import (
    CATEGORIES,
    METRIC_SPECS,
    Direction,
    MetricSpec,
    NormalizationMethod,
    normalize_relative_minmax_for_reference,
    normalize_value,
)


def test_metric_spec_rejects_bad_clip_bounds():
    with pytest.raises(ValueError):
        MetricSpec(
            name="bad",
            direction=Direction.HIGHER_IS_BETTER,
            method=NormalizationMethod.ABSOLUTE_REQUIREMENT_MINMAX,
            clip_min=5.0,
            clip_max=1.0,
            units="",
            rationale="",
        )


def test_metric_spec_rejects_equal_clip_bounds():
    with pytest.raises(ValueError):
        MetricSpec(
            name="bad",
            direction=Direction.HIGHER_IS_BETTER,
            method=NormalizationMethod.ABSOLUTE_REQUIREMENT_MINMAX,
            clip_min=1.0,
            clip_max=1.0,
            units="",
            rationale="",
        )


def test_all_declared_metric_specs_cover_all_categories():
    assert set(METRIC_SPECS.keys()) == set(CATEGORIES)


@pytest.mark.parametrize("category", list(CATEGORIES))
def test_boundary_values_map_to_0_and_1(category):
    spec = METRIC_SPECS[category]
    lo_util = normalize_value(spec.clip_min, spec)
    hi_util = normalize_value(spec.clip_max, spec)
    if spec.direction is Direction.HIGHER_IS_BETTER:
        assert lo_util == pytest.approx(0.0)
        assert hi_util == pytest.approx(1.0)
    else:
        assert lo_util == pytest.approx(1.0)
        assert hi_util == pytest.approx(0.0)


@pytest.mark.parametrize("category", list(CATEGORIES))
def test_clipping_saturates_outside_bounds(category):
    spec = METRIC_SPECS[category]
    below = spec.clip_min - 10 * (spec.clip_max - spec.clip_min)
    above = spec.clip_max + 10 * (spec.clip_max - spec.clip_min)
    util_below = normalize_value(below, spec)
    util_above = normalize_value(above, spec)
    assert 0.0 <= util_below <= 1.0
    assert 0.0 <= util_above <= 1.0
    # Saturation: below-range and above-range values must map to the SAME
    # utility as the clip bound itself.
    assert util_below == pytest.approx(normalize_value(spec.clip_min, spec))
    assert util_above == pytest.approx(normalize_value(spec.clip_max, spec))


@pytest.mark.parametrize("category", list(CATEGORIES))
def test_monotonicity_across_the_declared_range(category):
    spec = METRIC_SPECS[category]
    xs = np.linspace(spec.clip_min, spec.clip_max, 25)
    utils = [normalize_value(x, spec) for x in xs]
    diffs = np.diff(utils)
    if spec.direction is Direction.HIGHER_IS_BETTER:
        assert np.all(diffs >= -1e-12), "higher-is-better utility must be non-decreasing in raw value"
    else:
        assert np.all(diffs <= 1e-12), "lower-is-better utility must be non-increasing in raw value"


def test_equal_inputs_case_gives_equal_utility_not_winner_take_all():
    # This is exactly the artifact this trade study avoids: feeding the
    # SAME raw value for both "architectures" through the SAME MetricSpec
    # must yield the SAME utility, never an arbitrary 0/1 split.
    spec = METRIC_SPECS["mass"]
    value = 1.2
    u1 = normalize_value(value, spec)
    u2 = normalize_value(value, spec)
    assert u1 == pytest.approx(u2)


def test_higher_is_better_direction_increases_utility_with_raw_value():
    spec = METRIC_SPECS["availability"]
    assert normalize_value(0.9, spec) > normalize_value(0.1, spec)


def test_lower_is_better_direction_decreases_utility_with_raw_value():
    spec = METRIC_SPECS["mass"]
    assert normalize_value(0.1, spec) > normalize_value(2.9, spec)


def test_fixed_ordinal_map_anchors_are_stable_regardless_of_comparison():
    spec = METRIC_SPECS["cost"]
    # A score of 1 always maps to utility 1.0, and 5 always to 0.0,
    # regardless of what any "other" architecture's score is - i.e. this is
    # NOT a relative/winner-take-all computation.
    assert normalize_value(1, spec) == pytest.approx(1.0)
    assert normalize_value(5, spec) == pytest.approx(0.0)
    assert normalize_value(3, spec) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Documented counter-example: the naive relative winner-take-all method that
# this trade study explicitly avoids for the real trade matrix.
# ---------------------------------------------------------------------------


def test_naive_relative_minmax_produces_the_winner_take_all_artifact():
    # Demonstrates WHY normalize_relative_minmax_for_reference() is not used
    # in trade_matrix.py: an arbitrarily small physical difference still
    # gets mapped to a maximal 0/1 utility gap.
    values = {"a": 1.0000, "b": 1.0001}
    out = normalize_relative_minmax_for_reference(values, Direction.LOWER_IS_BETTER)
    assert out["a"] == pytest.approx(1.0)
    assert out["b"] == pytest.approx(0.0)


def test_naive_relative_minmax_equal_inputs_gives_half():
    values = {"a": 2.0, "b": 2.0}
    out = normalize_relative_minmax_for_reference(values, Direction.HIGHER_IS_BETTER)
    assert out["a"] == pytest.approx(0.5)
    assert out["b"] == pytest.approx(0.5)


def test_naive_relative_minmax_rejects_empty_input():
    with pytest.raises(ValueError):
        normalize_relative_minmax_for_reference({}, Direction.HIGHER_IS_BETTER)
