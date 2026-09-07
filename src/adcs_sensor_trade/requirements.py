"""Representative, ILLUSTRATIVE mission-level hard requirements (Milestone 3).

These are a SEPARATE set of pass/fail checks, evaluated independently of
the weighted trade score (part H of the Milestone 3 spec). An architecture
failing a hard requirement is flagged as such REGARDLESS of how well it
scores on the weighted trade matrix - the two systems are deliberately
decoupled.

Thresholds are plausible, illustrative engineering values for a
representative small/mid-class 3-axis-stabilized smallsat, chosen for
physical plausibility BEFORE looking at which architecture would pass or
fail them (see docs/trade_study_methodology.md "Hard requirements" section
for the full rationale). No threshold below was adjusted after the fact to
manufacture a particular outcome - where a result is surprising (e.g. an
architecture failing on availability or outage duration), it is reported
as such.

Resource ceilings are deliberately OMITTED as hard requirements here: there
is no defensible mission-independent "you MUST be under X kg / X W" cutoff
for a representative, unspecified smallsat bus the way there is for e.g. a
minimum usable-attitude-availability fraction - resource pressure is
already captured (as a preference, not a hard cutoff) in the weighted trade
matrix's mass/power categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .trade_metrics import ArchitectureMetrics


@dataclass(frozen=True)
class Requirement:
    """One representative, illustrative mission-level hard requirement."""

    name: str
    description: str
    check: Callable[[ArchitectureMetrics], bool]
    rationale: str


REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement(
        name="max_worst_axis_error_deg",
        description="Worst-axis attitude-knowledge error proxy <= 2.0 deg",
        check=lambda m: m.worst_axis_error_deg <= 2.0,
        rationale=(
            "A basic 3-axis-stabilized smallsat needs at least coarse "
            "(better than a couple of degrees) 3-axis attitude knowledge "
            "to be a physically meaningful ADCS solution at all; below "
            "this bar the sensor suite would not be providing usable "
            "3-axis attitude information for even a modest payload."
        ),
    ),
    Requirement(
        name="min_full_attitude_availability_fraction",
        description="Full-attitude information available >= 30% of the orbit",
        check=lambda m: m.availability_fraction >= 0.30,
        rationale=(
            "An illustrative representative requirement that an absolute "
            "3-axis attitude-determination update be obtainable often "
            "enough (at least ~30% of each orbit) to periodically "
            "reconverge / bound the growth of an attitude filter that "
            "otherwise propagates on gyro data alone between absolute "
            "updates (see the gyro-propagation discussion in "
            "docs/trade_study_methodology.md)."
        ),
    ),
    Requirement(
        name="max_longest_outage_s",
        description="Longest continuous full-attitude outage <= 600 s (10 min)",
        check=lambda m: m.longest_outage_s <= 600.0,
        rationale=(
            "An illustrative representative bound on how long an attitude "
            "filter may be forced to propagate on gyro data alone without "
            "any absolute update, before accumulated gyro drift becomes a "
            "significant concern for a modest-precision gyro."
        ),
    ),
    Requirement(
        name="min_full_attitude_cadence_hz",
        description="Effective usable full-attitude update cadence >= 1.0 Hz",
        check=lambda m: m.cadence_hz >= 1.0,
        rationale=(
            "An illustrative representative minimum cadence for absolute "
            "attitude updates to usefully bound estimator drift for a "
            "basic 3-axis ADCS, independent of the (separate, closed-loop "
            "control-bandwidth) question explicitly deferred in "
            "docs/availability_methodology.md."
        ),
    ),
)


def evaluate_requirements(metrics: ArchitectureMetrics) -> dict[str, bool]:
    """PASS (True) / FAIL (False) for every requirement, for one architecture."""
    return {r.name: bool(r.check(metrics)) for r in REQUIREMENTS}


def all_requirements_pass(metrics: ArchitectureMetrics) -> bool:
    return all(evaluate_requirements(metrics).values())
