"""Normalized ordinal cost/complexity scoring model (Milestone 3).

NO fabricated dollar figures appear anywhere in this module. Every burden
is a normalized ORDINAL score in {1, 2, 3, 4, 5}: 1 = favorable / low
burden, 5 = unfavorable / high burden. Every score is accompanied by a
written rationale string; see docs/trade_study_methodology.md for the full
engineering discussion of each value (this module stores the rationale
inline so it stays attached to the number it justifies, and the docs page
reproduces it in prose).

"Cost" and "complexity" are kept as distinguishable sub-scores:

- ``procurement_cost``: cost/procurement burden (schedule + $ burden proxy,
  ordinal only - no fabricated pricing).
- The remaining five ordinals (integration, calibration, software,
  operational, fault-management) are engineering COMPLEXITY/burden scores,
  averaged into a single ``complexity_score`` for the trade matrix while
  remaining individually inspectable here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _validate_ordinal(name: str, value: int) -> None:
    if value not in (1, 2, 3, 4, 5):
        raise ValueError(f"{name} must be an ordinal in {{1,2,3,4,5}}, got {value}")


@dataclass(frozen=True)
class ArchitectureCostComplexity:
    """Ordinal 1-5 cost/complexity burden scores for one architecture.

    Every field is a normalized ordinal score (1 = favorable/low burden,
    5 = unfavorable/high burden), NOT a dollar amount.
    """

    architecture: str
    procurement_cost: int
    integration_complexity: int
    calibration_burden: int
    software_complexity: int
    operational_constraints: int
    fault_management_burden: int
    rationale: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "procurement_cost",
            "integration_complexity",
            "calibration_burden",
            "software_complexity",
            "operational_constraints",
            "fault_management_burden",
        ):
            _validate_ordinal(name, getattr(self, name))

    @property
    def cost_score(self) -> float:
        """Cost/procurement burden sub-score (1-5, kept distinct from complexity)."""
        return float(self.procurement_cost)

    @property
    def complexity_score(self) -> float:
        """Mean of the five engineering-complexity/burden ordinals (1-5)."""
        return (
            self.integration_complexity
            + self.calibration_burden
            + self.software_complexity
            + self.operational_constraints
            + self.fault_management_burden
        ) / 5.0


def star_tracker_cost_complexity() -> ArchitectureCostComplexity:
    """Illustrative ordinal cost/complexity scores for the star-tracker architecture.

    See docs/trade_study_methodology.md for the full rationale; short
    version in the ``rationale`` dict below.
    """
    return ArchitectureCostComplexity(
        architecture="star_tracker",
        procurement_cost=4,
        integration_complexity=4,
        calibration_burden=3,
        software_complexity=2,
        operational_constraints=4,
        fault_management_burden=3,
        rationale={
            "procurement_cost": (
                "Precision optics, an APS/CMOS focal-plane detector, and a "
                "space-qualified star-identification processing chain make a "
                "star tracker a comparatively expensive, longer-lead-time "
                "procurement relative to a Sun sensor or magnetometer."
            ),
            "integration_complexity": (
                "Requires precise boresight alignment to the spacecraft "
                "attitude reference frame, a baffle/keep-out zone that must "
                "be respected by nearby deployables and other sensor fields "
                "of view, and thermal stability requirements for the "
                "optical bench - all of which drive mechanical/ADCS "
                "integration effort above a simple bolt-on sensor."
            ),
            "calibration_burden": (
                "Precision lens-distortion and focal-plane calibration is "
                "largely performed by the vendor at the factory; on-orbit "
                "work is mostly boresight-alignment verification rather "
                "than a full recalibration, so the burden on the "
                "integrating team is moderate rather than high."
            ),
            "software_complexity": (
                "Star identification/centroiding is normally a turnkey "
                "vendor-supplied algorithm; the integrating team mainly "
                "needs quaternion/DCM interface handling on the OBC, which "
                "is comparatively simple software work."
            ),
            "operational_constraints": (
                "Sun/Earth/Moon bright-source keep-out constraints "
                "(modeled directly in Milestone 2) actively restrict "
                "attitude planning and slew timing whenever the boresight "
                "would sweep near a bright source, a real operational "
                "planning burden."
            ),
            "fault_management_burden": (
                "Star trackers are commonly flown single-string on "
                "smallsats; loss of star identification (blinding, cosmic-"
                "ray upsets, tracking loss) requires a defined fallback / "
                "coarse-attitude mode, a moderate fault-management burden."
            ),
        },
    )


def sunmag_cost_complexity() -> ArchitectureCostComplexity:
    """Illustrative ordinal cost/complexity scores for the Sun+mag architecture."""
    return ArchitectureCostComplexity(
        architecture="sun_plus_mag",
        procurement_cost=1,
        integration_complexity=2,
        calibration_burden=2,
        software_complexity=3,
        operational_constraints=4,
        fault_management_burden=2,
        rationale={
            "procurement_cost": (
                "Sun sensors and magnetometers are commodity, low-cost, "
                "widely available smallsat components with short lead "
                "times relative to a star tracker."
            ),
            "integration_complexity": (
                "Sun-sensor panel mounting is straightforward; the "
                "magnetometer typically wants boom deployment or careful "
                "placement for magnetic cleanliness, which adds some "
                "mechanical integration work but nothing approaching a "
                "star tracker's boresight/baffle/thermal requirements."
            ),
            "calibration_burden": (
                "Both sensors have a small number of well-understood "
                "calibration parameters (Sun-sensor boresight offset; "
                "magnetometer hard/soft-iron bias and scale factor) with "
                "standard, low-effort ground calibration procedures - "
                "simpler than a star tracker's precision optical "
                "calibration."
            ),
            "software_complexity": (
                "Recovering full 3-axis attitude from two vectors needs an "
                "explicit two-vector attitude-determination algorithm "
                "(e.g. TRIAD/QUEST) plus rank/conditioning-aware logic to "
                "know when the geometry is degenerate (modeled directly in "
                "this trade study's geometry.py/availability.py) - more "
                "custom software than the star tracker's turnkey "
                "quaternion output."
            ),
            "operational_constraints": (
                "Eclipse (Sun sensor blind) and Sun-sensor FOV geometry "
                "(modeled directly in Milestone 2) are the dominant "
                "availability bottleneck for this architecture and "
                "actively constrain WHEN a full-attitude update can be "
                "obtained, an operational planning burden comparable to "
                "the star tracker's keep-out constraint."
            ),
            "fault_management_burden": (
                "Three simple, largely independent commodity units "
                "(2 Sun-sensor heads + 1 magnetometer) degrade gracefully "
                "to a rank-2/coarse attitude mode on a single-unit "
                "failure rather than an abrupt loss of all attitude "
                "information, a comparatively low fault-management burden."
            ),
        },
    )
