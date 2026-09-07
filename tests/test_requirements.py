"""Tests for adcs_sensor_trade.requirements (Milestone 3)."""

from __future__ import annotations

from dataclasses import replace

from adcs_sensor_trade.requirements import REQUIREMENTS, all_requirements_pass, evaluate_requirements
from adcs_sensor_trade.trade_metrics import star_tracker_metrics, sunmag_metrics


def test_star_tracker_passes_all_requirements_at_baseline():
    m = star_tracker_metrics()
    results = evaluate_requirements(m)
    assert all(results.values()), results
    assert all_requirements_pass(m) is True


def test_sunmag_fails_availability_and_outage_requirements_at_baseline():
    # Honest regression check: given the accepted M2 numbers (23.18%
    # availability, 3939 s longest outage), Sun+mag FAILS these two
    # representative hard requirements. This is reported as-is, not tuned
    # away.
    m = sunmag_metrics()
    results = evaluate_requirements(m)
    assert results["min_full_attitude_availability_fraction"] is False
    assert results["max_longest_outage_s"] is False
    assert all_requirements_pass(m) is False


def test_all_requirement_names_are_unique():
    names = [r.name for r in REQUIREMENTS]
    assert len(names) == len(set(names))


def test_every_requirement_has_a_rationale():
    for r in REQUIREMENTS:
        assert len(r.rationale.strip()) > 20
        assert len(r.description.strip()) > 5


def test_constructed_synthetic_pass_case():
    m = star_tracker_metrics()
    good = replace(
        m,
        worst_axis_error_deg=0.01,
        availability_fraction=0.99,
        longest_outage_s=1.0,
        cadence_hz=5.0,
    )
    assert all_requirements_pass(good) is True


def test_constructed_synthetic_fail_case_accuracy():
    m = star_tracker_metrics()
    bad = replace(m, worst_axis_error_deg=10.0)
    results = evaluate_requirements(bad)
    assert results["max_worst_axis_error_deg"] is False


def test_constructed_synthetic_fail_case_availability():
    m = star_tracker_metrics()
    bad = replace(m, availability_fraction=0.01)
    results = evaluate_requirements(bad)
    assert results["min_full_attitude_availability_fraction"] is False


def test_constructed_synthetic_fail_case_outage():
    m = star_tracker_metrics()
    bad = replace(m, longest_outage_s=100000.0)
    results = evaluate_requirements(bad)
    assert results["max_longest_outage_s"] is False


def test_constructed_synthetic_fail_case_cadence():
    m = star_tracker_metrics()
    bad = replace(m, cadence_hz=0.01)
    results = evaluate_requirements(bad)
    assert results["min_full_attitude_cadence_hz"] is False


def test_requirement_boundary_values_pass_inclusively():
    m = star_tracker_metrics()
    boundary = replace(
        m,
        worst_axis_error_deg=2.0,
        availability_fraction=0.30,
        longest_outage_s=600.0,
        cadence_hz=1.0,
    )
    assert all_requirements_pass(boundary) is True


def test_evaluate_requirements_returns_bool_values_not_numpy_bool():
    m = star_tracker_metrics()
    results = evaluate_requirements(m)
    for v in results.values():
        assert isinstance(v, bool)
