"""Illustrative spacecraft-resource model for individual sensor components
and the two candidate architectures (Milestone 3).

SCOPE / HONESTY NOTE
--------------------
Every numeric value produced by this module is a representative,
ILLUSTRATIVE engineering assumption for a small/mid-class smallsat sensor
component, or a value DERIVED (by simple, documented arithmetic) from such
assumptions. Nothing here is a vendor datasheet number, a procurement
quote, or a flight-heritage claim - see docs/trade_study_methodology.md for
the full rationale behind each value. Every numeric field is tagged with a
:class:`ValueSource` so downstream consumers (and readers) can tell at a
glance which numbers are "real" (computed from the accepted Milestone 1/2
modules) and which are illustrative engineering assumptions.

The Sun+mag ARCHITECTURE totals are always DERIVED by summing the
individual Sun-sensor and magnetometer COMPONENT profiles - they are never
hardcoded as a separate lump number, per the Milestone 3 requirement that
architecture-level resource totals be a true function of component-level
assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ValueSource(str, Enum):
    """Provenance label for every numeric value in the M3 resource/cost model."""

    M1_M2_MODELED = "m1_m2_modeled"
    ILLUSTRATIVE_ASSUMPTION = "illustrative_assumption"
    DERIVED = "derived"


@dataclass(frozen=True)
class SourcedValue:
    """A single numeric value plus its provenance label and a short note."""

    value: float
    source: ValueSource
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source, ValueSource):
            raise ValueError(f"source must be a ValueSource, got {self.source!r}")


def _validate_nonnegative(name: str, sv: SourcedValue) -> None:
    if sv.value < 0:
        raise ValueError(f"{name} must be >= 0, got {sv.value}")


@dataclass(frozen=True)
class ComponentResourceProfile:
    """Illustrative representative resource profile for ONE sensor component.

    Parameters
    ----------
    name
        Component name (e.g. "star_tracker", "sun_sensor", "magnetometer").
    mass_kg, avg_power_w, peak_power_w
        Representative mass [kg] and average/peak electrical power [W] for
        ONE unit of this component.
    volume_proxy_cm3
        A simple packaging-volume proxy [cm^3] for ONE unit (not a detailed
        mechanical envelope - see docs/trade_study_methodology.md).
    unit_count
        Number of units of this component baselined for the architecture
        (e.g. two Sun sensors for near-full-sky coverage).
    interface_burden
        Ordinal 1-5 proxy for interface/processing burden on the spacecraft
        OBC and harness (1 = simple/low burden, 5 = demanding/high burden).
        This is a COST/COMPLEXITY-adjacent proxy kept in the resource model
        because it is naturally a per-component physical/interface property
        (harness complexity, data rate, dedicated processing needs).
    """

    name: str
    mass_kg: SourcedValue
    avg_power_w: SourcedValue
    peak_power_w: SourcedValue
    volume_proxy_cm3: SourcedValue
    unit_count: int
    interface_burden: SourcedValue

    def __post_init__(self) -> None:
        for field_name in ("mass_kg", "avg_power_w", "peak_power_w", "volume_proxy_cm3"):
            _validate_nonnegative(field_name, getattr(self, field_name))
        if self.unit_count <= 0:
            raise ValueError(f"unit_count must be > 0, got {self.unit_count}")
        if not (1.0 <= self.interface_burden.value <= 5.0):
            raise ValueError(
                f"interface_burden must be an ordinal in [1, 5], got {self.interface_burden.value}"
            )

    @property
    def total_mass_kg(self) -> float:
        return self.mass_kg.value * self.unit_count

    @property
    def total_avg_power_w(self) -> float:
        return self.avg_power_w.value * self.unit_count

    @property
    def total_peak_power_w(self) -> float:
        return self.peak_power_w.value * self.unit_count

    @property
    def total_volume_proxy_cm3(self) -> float:
        return self.volume_proxy_cm3.value * self.unit_count


# ---------------------------------------------------------------------------
# Individual component baselines (ILLUSTRATIVE ASSUMPTIONS)
#
# See docs/trade_study_methodology.md "Resource assumptions" section for the
# full written rationale behind each of these numbers. In short: these are
# representative order-of-magnitude values for the SIZE CLASS of component
# implied by the Milestone 1 sensor models (a "tens of arcsec" small/mid
# class star tracker; a digital coarse/fine Sun sensor; a boom-mountable
# fluxgate-class magnetometer) for a small/mid-size smallsat bus. They are
# NOT vendor specifications and NOT tied to any specific product.
# ---------------------------------------------------------------------------


def star_tracker_component() -> ComponentResourceProfile:
    """Illustrative representative small/mid-class star-tracker component."""
    src = ValueSource.ILLUSTRATIVE_ASSUMPTION
    return ComponentResourceProfile(
        name="star_tracker",
        mass_kg=SourcedValue(0.45, src, "representative small/mid-class star-tracker head mass"),
        avg_power_w=SourcedValue(1.5, src, "representative average operating power"),
        peak_power_w=SourcedValue(2.5, src, "representative peak power during acquisition/startup"),
        volume_proxy_cm3=SourcedValue(250.0, src, "representative head + baffle envelope proxy"),
        unit_count=1,
        interface_burden=SourcedValue(
            4.0, src, "precision boresight alignment, baffle keep-out geometry, thermal stability"
        ),
    )


def sun_sensor_component() -> ComponentResourceProfile:
    """Illustrative representative digital coarse/fine Sun-sensor component."""
    src = ValueSource.ILLUSTRATIVE_ASSUMPTION
    return ComponentResourceProfile(
        name="sun_sensor",
        mass_kg=SourcedValue(0.05, src, "representative single-head digital Sun-sensor mass"),
        avg_power_w=SourcedValue(0.2, src, "representative average operating power"),
        peak_power_w=SourcedValue(0.3, src, "representative peak power"),
        volume_proxy_cm3=SourcedValue(20.0, src, "representative single-head envelope proxy"),
        unit_count=2,
        interface_burden=SourcedValue(
            2.0, src, "simple analog/digital interface, straightforward panel mounting"
        ),
    )


def magnetometer_component() -> ComponentResourceProfile:
    """Illustrative representative boom-mountable magnetometer component."""
    src = ValueSource.ILLUSTRATIVE_ASSUMPTION
    return ComponentResourceProfile(
        name="magnetometer",
        mass_kg=SourcedValue(0.15, src, "representative fluxgate-class magnetometer mass"),
        avg_power_w=SourcedValue(0.3, src, "representative average operating power"),
        peak_power_w=SourcedValue(0.4, src, "representative peak power"),
        volume_proxy_cm3=SourcedValue(40.0, src, "representative sensor-head envelope proxy"),
        unit_count=1,
        interface_burden=SourcedValue(
            2.0, src, "simple analog interface; magnetic-cleanliness/boom accommodation adds modest burden"
        ),
    )


# ---------------------------------------------------------------------------
# Architecture-level totals (DERIVED from components above)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchitectureResourceTotals:
    """Architecture-level resource totals, DERIVED from component profiles."""

    architecture: str
    mass_kg: float
    avg_power_w: float
    peak_power_w: float
    volume_proxy_cm3: float
    unit_count: int
    interface_burden: float
    source: ValueSource
    components: tuple[ComponentResourceProfile, ...]

    def __post_init__(self) -> None:
        if self.mass_kg < 0 or self.avg_power_w < 0 or self.peak_power_w < 0:
            raise ValueError("Architecture resource totals must be non-negative.")
        if self.unit_count <= 0:
            raise ValueError(f"unit_count must be > 0, got {self.unit_count}")


def _sum_components(components: tuple[ComponentResourceProfile, ...]) -> ArchitectureResourceTotals:
    if len(components) == 0:
        raise ValueError("Need at least one component to build architecture totals.")
    mass = sum(c.total_mass_kg for c in components)
    avg_p = sum(c.total_avg_power_w for c in components)
    peak_p = sum(c.total_peak_power_w for c in components)
    vol = sum(c.total_volume_proxy_cm3 for c in components)
    units = sum(c.unit_count for c in components)
    # Interface burden is a per-COMPONENT-TYPE ordinal proxy (not scaled by
    # unit count): averaged across the distinct component types present in
    # the architecture. Documented rationale: adding a second identical Sun
    # sensor does not meaningfully change per-type interface complexity, it
    # is the diversity of interface TYPES that drives integration burden.
    burden = sum(c.interface_burden.value for c in components) / len(components)
    source = ValueSource.DERIVED if len(components) > 1 else components[0].interface_burden.source
    return ArchitectureResourceTotals(
        architecture="",  # filled in by caller
        mass_kg=mass,
        avg_power_w=avg_p,
        peak_power_w=peak_p,
        volume_proxy_cm3=vol,
        unit_count=units,
        interface_burden=burden,
        source=source,
        components=components,
    )


def star_tracker_architecture_totals() -> ArchitectureResourceTotals:
    """Architecture A totals: a single star-tracker component."""
    totals = _sum_components((star_tracker_component(),))
    return ArchitectureResourceTotals(
        architecture="star_tracker",
        mass_kg=totals.mass_kg,
        avg_power_w=totals.avg_power_w,
        peak_power_w=totals.peak_power_w,
        volume_proxy_cm3=totals.volume_proxy_cm3,
        unit_count=totals.unit_count,
        interface_burden=totals.interface_burden,
        source=totals.source,
        components=totals.components,
    )


def sunmag_architecture_totals() -> ArchitectureResourceTotals:
    """Architecture B totals: DERIVED by summing Sun-sensor + magnetometer components."""
    totals = _sum_components((sun_sensor_component(), magnetometer_component()))
    return ArchitectureResourceTotals(
        architecture="sun_plus_mag",
        mass_kg=totals.mass_kg,
        avg_power_w=totals.avg_power_w,
        peak_power_w=totals.peak_power_w,
        volume_proxy_cm3=totals.volume_proxy_cm3,
        unit_count=totals.unit_count,
        interface_burden=totals.interface_burden,
        source=ValueSource.DERIVED,
        components=totals.components,
    )
