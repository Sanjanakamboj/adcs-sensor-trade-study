"""Tests for adcs_sensor_trade.resource_model (Milestone 3)."""

from __future__ import annotations

import pytest

from adcs_sensor_trade.resource_model import (
    ComponentResourceProfile,
    SourcedValue,
    ValueSource,
    magnetometer_component,
    star_tracker_architecture_totals,
    star_tracker_component,
    sun_sensor_component,
    sunmag_architecture_totals,
)


def test_sourced_value_requires_valuesource_enum():
    with pytest.raises(ValueError):
        SourcedValue(1.0, "not_a_source")  # type: ignore[arg-type]


def test_sourced_value_accepts_valid_source():
    sv = SourcedValue(1.0, ValueSource.ILLUSTRATIVE_ASSUMPTION)
    assert sv.value == 1.0


@pytest.mark.parametrize("component_fn", [star_tracker_component, sun_sensor_component, magnetometer_component])
def test_component_profiles_are_nonnegative(component_fn):
    comp = component_fn()
    assert comp.mass_kg.value >= 0
    assert comp.avg_power_w.value >= 0
    assert comp.peak_power_w.value >= 0
    assert comp.volume_proxy_cm3.value >= 0
    assert comp.unit_count > 0


@pytest.mark.parametrize("component_fn", [star_tracker_component, sun_sensor_component, magnetometer_component])
def test_component_interface_burden_is_ordinal_1_to_5(component_fn):
    comp = component_fn()
    assert 1.0 <= comp.interface_burden.value <= 5.0


def test_component_rejects_negative_mass():
    with pytest.raises(ValueError):
        ComponentResourceProfile(
            name="bad",
            mass_kg=SourcedValue(-1.0, ValueSource.ILLUSTRATIVE_ASSUMPTION),
            avg_power_w=SourcedValue(1.0, ValueSource.ILLUSTRATIVE_ASSUMPTION),
            peak_power_w=SourcedValue(1.0, ValueSource.ILLUSTRATIVE_ASSUMPTION),
            volume_proxy_cm3=SourcedValue(1.0, ValueSource.ILLUSTRATIVE_ASSUMPTION),
            unit_count=1,
            interface_burden=SourcedValue(2.0, ValueSource.ILLUSTRATIVE_ASSUMPTION),
        )


def test_component_rejects_zero_unit_count():
    with pytest.raises(ValueError):
        ComponentResourceProfile(
            name="bad",
            mass_kg=SourcedValue(1.0, ValueSource.ILLUSTRATIVE_ASSUMPTION),
            avg_power_w=SourcedValue(1.0, ValueSource.ILLUSTRATIVE_ASSUMPTION),
            peak_power_w=SourcedValue(1.0, ValueSource.ILLUSTRATIVE_ASSUMPTION),
            volume_proxy_cm3=SourcedValue(1.0, ValueSource.ILLUSTRATIVE_ASSUMPTION),
            unit_count=0,
            interface_burden=SourcedValue(2.0, ValueSource.ILLUSTRATIVE_ASSUMPTION),
        )


def test_component_rejects_out_of_range_interface_burden():
    with pytest.raises(ValueError):
        ComponentResourceProfile(
            name="bad",
            mass_kg=SourcedValue(1.0, ValueSource.ILLUSTRATIVE_ASSUMPTION),
            avg_power_w=SourcedValue(1.0, ValueSource.ILLUSTRATIVE_ASSUMPTION),
            peak_power_w=SourcedValue(1.0, ValueSource.ILLUSTRATIVE_ASSUMPTION),
            volume_proxy_cm3=SourcedValue(1.0, ValueSource.ILLUSTRATIVE_ASSUMPTION),
            unit_count=1,
            interface_burden=SourcedValue(6.0, ValueSource.ILLUSTRATIVE_ASSUMPTION),
        )


def test_component_total_properties_scale_by_unit_count():
    comp = sun_sensor_component()
    assert comp.total_mass_kg == pytest.approx(comp.mass_kg.value * comp.unit_count)
    assert comp.total_avg_power_w == pytest.approx(comp.avg_power_w.value * comp.unit_count)
    assert comp.total_peak_power_w == pytest.approx(comp.peak_power_w.value * comp.unit_count)
    assert comp.total_volume_proxy_cm3 == pytest.approx(comp.volume_proxy_cm3.value * comp.unit_count)


# ---------------------------------------------------------------------------
# Architecture-level DERIVED totals: the key requirement (part A of spec) is
# that Sun+mag totals are the SUM of the individual component values, never
# a separately hardcoded lump number.
# ---------------------------------------------------------------------------


def test_sunmag_architecture_mass_is_sum_of_components():
    sun = sun_sensor_component()
    mag = magnetometer_component()
    totals = sunmag_architecture_totals()
    expected_mass = sun.total_mass_kg + mag.total_mass_kg
    assert totals.mass_kg == pytest.approx(expected_mass)


def test_sunmag_architecture_avg_power_is_sum_of_components():
    sun = sun_sensor_component()
    mag = magnetometer_component()
    totals = sunmag_architecture_totals()
    expected = sun.total_avg_power_w + mag.total_avg_power_w
    assert totals.avg_power_w == pytest.approx(expected)


def test_sunmag_architecture_peak_power_is_sum_of_components():
    sun = sun_sensor_component()
    mag = magnetometer_component()
    totals = sunmag_architecture_totals()
    expected = sun.total_peak_power_w + mag.total_peak_power_w
    assert totals.peak_power_w == pytest.approx(expected)


def test_sunmag_architecture_volume_is_sum_of_components():
    sun = sun_sensor_component()
    mag = magnetometer_component()
    totals = sunmag_architecture_totals()
    expected = sun.total_volume_proxy_cm3 + mag.total_volume_proxy_cm3
    assert totals.volume_proxy_cm3 == pytest.approx(expected)


def test_sunmag_architecture_unit_count_is_sum_of_components():
    sun = sun_sensor_component()
    mag = magnetometer_component()
    totals = sunmag_architecture_totals()
    assert totals.unit_count == sun.unit_count + mag.unit_count


def test_sunmag_architecture_totals_source_is_derived():
    totals = sunmag_architecture_totals()
    assert totals.source == ValueSource.DERIVED


def test_star_tracker_architecture_totals_matches_single_component():
    comp = star_tracker_component()
    totals = star_tracker_architecture_totals()
    assert totals.mass_kg == pytest.approx(comp.total_mass_kg)
    assert totals.avg_power_w == pytest.approx(comp.total_avg_power_w)
    assert totals.peak_power_w == pytest.approx(comp.total_peak_power_w)


def test_architecture_totals_are_nonnegative():
    for totals in (star_tracker_architecture_totals(), sunmag_architecture_totals()):
        assert totals.mass_kg >= 0
        assert totals.avg_power_w >= 0
        assert totals.peak_power_w >= 0
        assert totals.unit_count > 0


def test_sunmag_has_lower_mass_and_power_than_star_tracker_in_this_illustrative_model():
    # A specific, checkable claim of this illustrative resource model (not a
    # universal truth about all star trackers / Sun+mag suites): documented
    # in docs/trade_study_methodology.md. Regression-guards the assumption.
    star = star_tracker_architecture_totals()
    sunmag = sunmag_architecture_totals()
    assert sunmag.mass_kg < star.mass_kg
    assert sunmag.avg_power_w < star.avg_power_w
