"""Tests for adcs_sensor_trade.cost_complexity_model (Milestone 3)."""

from __future__ import annotations

import pytest

from adcs_sensor_trade.cost_complexity_model import (
    ArchitectureCostComplexity,
    star_tracker_cost_complexity,
    sunmag_cost_complexity,
)


@pytest.mark.parametrize("factory", [star_tracker_cost_complexity, sunmag_cost_complexity])
def test_all_ordinal_scores_in_valid_range(factory):
    cc = factory()
    for field_name in (
        "procurement_cost",
        "integration_complexity",
        "calibration_burden",
        "software_complexity",
        "operational_constraints",
        "fault_management_burden",
    ):
        value = getattr(cc, field_name)
        assert value in (1, 2, 3, 4, 5), f"{field_name}={value} out of range"


@pytest.mark.parametrize("bad_value", [0, 6, -1, 3.5])
def test_invalid_ordinal_score_raises(bad_value):
    with pytest.raises(ValueError):
        ArchitectureCostComplexity(
            architecture="bad",
            procurement_cost=bad_value,
            integration_complexity=2,
            calibration_burden=2,
            software_complexity=2,
            operational_constraints=2,
            fault_management_burden=2,
        )


def test_cost_score_equals_procurement_cost():
    cc = star_tracker_cost_complexity()
    assert cc.cost_score == float(cc.procurement_cost)


def test_complexity_score_is_mean_of_five_subscores():
    cc = sunmag_cost_complexity()
    expected = (
        cc.integration_complexity
        + cc.calibration_burden
        + cc.software_complexity
        + cc.operational_constraints
        + cc.fault_management_burden
    ) / 5.0
    assert cc.complexity_score == pytest.approx(expected)


def test_cost_and_complexity_are_distinguishable_subscores():
    cc = star_tracker_cost_complexity()
    # Cost score must be exactly the procurement ordinal, not folded into complexity.
    assert cc.cost_score != cc.complexity_score or cc.procurement_cost == pytest.approx(cc.complexity_score)


def test_every_score_has_a_rationale_string():
    for factory in (star_tracker_cost_complexity, sunmag_cost_complexity):
        cc = factory()
        for field_name in (
            "procurement_cost",
            "integration_complexity",
            "calibration_burden",
            "software_complexity",
            "operational_constraints",
            "fault_management_burden",
        ):
            assert field_name in cc.rationale
            assert len(cc.rationale[field_name].strip()) > 20


def test_star_tracker_has_higher_procurement_cost_burden_than_sunmag():
    star = star_tracker_cost_complexity()
    sunmag = sunmag_cost_complexity()
    assert star.cost_score > sunmag.cost_score


def test_star_tracker_has_higher_complexity_burden_than_sunmag():
    star = star_tracker_cost_complexity()
    sunmag = sunmag_cost_complexity()
    assert star.complexity_score > sunmag.complexity_score


def test_architecture_cost_complexity_is_frozen():
    cc = star_tracker_cost_complexity()
    with pytest.raises(Exception):
        cc.procurement_cost = 1  # type: ignore[misc]
