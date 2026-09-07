"""Transparent metric normalization for the Milestone 3 weighted trade matrix.

For EACH metric in the trade matrix we explicitly declare, as a documented
:class:`MetricSpec` (not a bare comment):

- whether higher or lower raw values are better (:class:`Direction`),
- the normalization METHOD used (:class:`NormalizationMethod`), and
- the clip bounds the raw value is mapped from.

Both methods used here apply the SAME linear-with-clip transform; what
differs is what the clip bounds MEAN:

- ``ABSOLUTE_REQUIREMENT_MINMAX``: the bounds are a defensible ABSOLUTE
  engineering floor/ceiling (a physical bound, or an illustrative
  requirement-class threshold) chosen independently of which architecture
  happens to score better - NOT "whichever option is better gets 1.0".
  This is the PREFERRED method used for every physical/continuous metric
  in this trade study (accuracy, availability, cadence, mass, power).
- ``FIXED_ORDINAL_MAP``: for the cost/complexity ordinal scores, 1 and 5
  are already fixed, universally-anchored endpoints of the 1-5 scale by
  construction (not derived from the two architectures being compared), so
  a fixed linear map [1, 5] -> [0, 1] is a defensible, non-"winner-take-all"
  choice: it would produce the exact same normalized utility for a score of
  4 regardless of what the other architecture happens to score.

We deliberately do NOT use a naive relative "winner gets 1.0, loser gets
0.0" min-max anywhere in this trade study, because with only two
architectures being compared that method would silently launder ANY
difference, however small or large, into a maximal utility gap - exactly
the "winner-take-all artifact" this module is designed to avoid. (See
:func:`normalize_relative_minmax_for_reference` below, kept ONLY as a
documented counter-example used by a test to demonstrate why it is
avoided - it is never called by the production trade-matrix code path.)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Direction(str, Enum):
    """Whether higher or lower raw metric values are better."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class NormalizationMethod(str, Enum):
    """Which normalization method a :class:`MetricSpec` uses."""

    ABSOLUTE_REQUIREMENT_MINMAX = "absolute_requirement_minmax"
    FIXED_ORDINAL_MAP = "fixed_ordinal_map"


@dataclass(frozen=True)
class MetricSpec:
    """Documented normalization declaration for one trade-matrix metric."""

    name: str
    direction: Direction
    method: NormalizationMethod
    clip_min: float
    clip_max: float
    units: str
    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.direction, Direction):
            raise ValueError(f"direction must be a Direction, got {self.direction!r}")
        if not isinstance(self.method, NormalizationMethod):
            raise ValueError(f"method must be a NormalizationMethod, got {self.method!r}")
        if self.clip_max <= self.clip_min:
            raise ValueError(
                f"clip_max ({self.clip_max}) must be > clip_min ({self.clip_min}) for metric {self.name!r}"
            )


def normalize_value(value: float, spec: MetricSpec) -> float:
    """Map a raw metric value to a utility in [0, 1] per ``spec``.

    Both declared methods use the same linear-with-clip transform; the
    METHOD label only documents what the clip bounds represent (see module
    docstring). The value is clipped to ``[spec.clip_min, spec.clip_max]``
    before mapping, so values outside the declared bounds saturate rather
    than producing utilities outside [0, 1].
    """
    clipped = min(max(float(value), spec.clip_min), spec.clip_max)
    span = spec.clip_max - spec.clip_min
    frac = (clipped - spec.clip_min) / span
    if spec.direction is Direction.LOWER_IS_BETTER:
        frac = 1.0 - frac
    return float(frac)


def normalize_relative_minmax_for_reference(values: dict[str, float], direction: Direction) -> dict[str, float]:
    """Naive relative winner-take-all min-max, kept ONLY as a documented counter-example.

    Given a small set of raw values, maps the best one to utility 1.0 and
    the worst to utility 0.0 (equal values all map to 0.5). This is the
    method this trade study explicitly AVOIDS for the real trade matrix
    (see module docstring) because with two options it manufactures a
    maximal utility gap regardless of how large the actual physical
    difference is. It is exercised by a test that demonstrates exactly that
    artifact, and is never called from ``trade_matrix.py``.
    """
    if len(values) == 0:
        raise ValueError("values must be non-empty.")
    raw = list(values.values())
    lo, hi = min(raw), max(raw)
    if hi - lo < 1e-15:
        return {k: 0.5 for k in values}
    out = {}
    for k, v in values.items():
        frac = (v - lo) / (hi - lo)
        if direction is Direction.LOWER_IS_BETTER:
            frac = 1.0 - frac
        out[k] = float(frac)
    return out


# ---------------------------------------------------------------------------
# The declared metric specs used by the Milestone 3 trade matrix.
#
# Clip-bound rationale (full discussion in docs/trade_study_methodology.md):
#
# - accuracy: 0 deg (ideal) to 2.0 deg (illustrative coarse 3-axis
#   attitude-KNOWLEDGE ceiling for a basic 3-axis-stabilized smallsat -
#   chosen as a representative "still usable, but poor" bound, independent
#   of which architecture happens to be better or worse than it).
# - availability: 0% to 100% of orbit - a physical absolute bound, not a
#   choice at all.
# - cadence: 0 Hz to 2.0 Hz (illustrative ceiling for a determination-only
#   full-attitude update cadence; beyond this an attitude FILTER fed by
#   absolute updates has diminishing marginal benefit relative to gyro
#   propagation between updates for this class of mission - see
#   docs/trade_study_methodology.md).
# - mass: 0 kg (ideal) to 3.0 kg (illustrative ADCS sensor-suite mass
#   ceiling for a representative ~50-150 kg smallsat bus).
# - power: 0 W (ideal) to 5.0 W (illustrative ADCS sensor-suite average
#   power ceiling for the same representative bus).
# - cost / complexity: fixed ordinal anchors [1, 5] (see class docstring).
# ---------------------------------------------------------------------------

METRIC_SPECS: dict[str, MetricSpec] = {
    "accuracy": MetricSpec(
        name="accuracy",
        direction=Direction.LOWER_IS_BETTER,
        method=NormalizationMethod.ABSOLUTE_REQUIREMENT_MINMAX,
        clip_min=0.0,
        clip_max=2.0,
        units="deg (worst-axis 1-sigma attitude-knowledge error proxy)",
        rationale=(
            "Lower worst-axis error is better. 0 deg is the physical ideal; "
            "2.0 deg is an illustrative 'still usable but poor' coarse "
            "3-axis attitude-knowledge ceiling for a basic 3-axis-"
            "stabilized smallsat, chosen independently of which "
            "architecture happens to fall above or below it."
        ),
    ),
    "availability": MetricSpec(
        name="availability",
        direction=Direction.HIGHER_IS_BETTER,
        method=NormalizationMethod.ABSOLUTE_REQUIREMENT_MINMAX,
        clip_min=0.0,
        clip_max=1.0,
        units="fraction of orbit with full-attitude information available",
        rationale=(
            "Higher availability is better. [0, 1] is the physical bound "
            "on a fraction of orbit; no engineering judgment call is "
            "needed for this metric's bounds."
        ),
    ),
    "cadence": MetricSpec(
        name="cadence",
        direction=Direction.HIGHER_IS_BETTER,
        method=NormalizationMethod.ABSOLUTE_REQUIREMENT_MINMAX,
        clip_min=0.0,
        clip_max=2.0,
        units="Hz (effective usable full-attitude update cadence)",
        rationale=(
            "Higher effective full-attitude update cadence is better. "
            "2.0 Hz is an illustrative ceiling for a determination-only "
            "update cadence in this class of mission (see "
            "docs/trade_study_methodology.md); it is a bound on the METRIC, "
            "not tuned to either architecture's actual cadence."
        ),
    ),
    "mass": MetricSpec(
        name="mass",
        direction=Direction.LOWER_IS_BETTER,
        method=NormalizationMethod.ABSOLUTE_REQUIREMENT_MINMAX,
        clip_min=0.0,
        clip_max=3.0,
        units="kg (architecture total)",
        rationale=(
            "Lower mass is better. 3.0 kg is an illustrative ADCS "
            "sensor-suite mass ceiling for a representative ~50-150 kg "
            "smallsat bus."
        ),
    ),
    "power": MetricSpec(
        name="power",
        direction=Direction.LOWER_IS_BETTER,
        method=NormalizationMethod.ABSOLUTE_REQUIREMENT_MINMAX,
        clip_min=0.0,
        clip_max=5.0,
        units="W (architecture total average power)",
        rationale=(
            "Lower average power is better. 5.0 W is an illustrative ADCS "
            "sensor-suite average-power ceiling for the same representative "
            "bus."
        ),
    ),
    "cost": MetricSpec(
        name="cost",
        direction=Direction.LOWER_IS_BETTER,
        method=NormalizationMethod.FIXED_ORDINAL_MAP,
        clip_min=1.0,
        clip_max=5.0,
        units="ordinal 1-5 (1=favorable/low burden, 5=unfavorable/high burden)",
        rationale=(
            "Lower procurement/cost burden is better. 1 and 5 are the "
            "fixed, universally-anchored endpoints of the ordinal scale "
            "itself, not values derived from comparing the two "
            "architectures - a fixed linear map is therefore appropriate "
            "here, unlike a relative winner-take-all min-max."
        ),
    ),
    "complexity": MetricSpec(
        name="complexity",
        direction=Direction.LOWER_IS_BETTER,
        method=NormalizationMethod.FIXED_ORDINAL_MAP,
        clip_min=1.0,
        clip_max=5.0,
        units="ordinal 1-5 (mean of 5 burden sub-scores; 1=favorable, 5=unfavorable)",
        rationale=(
            "Lower implementation/operational complexity burden is better. "
            "Same fixed-anchor rationale as the cost metric."
        ),
    ),
}

CATEGORIES: tuple[str, ...] = tuple(METRIC_SPECS.keys())
