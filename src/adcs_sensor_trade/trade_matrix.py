"""Weighted decision-matrix engine for the Milestone 3 sensor trade study.

Combines the metric-extraction layer (``trade_metrics.py``) and the
normalization layer (``normalization.py``) into a reusable weighted
decision matrix: raw table -> normalized utility table -> weighted
contribution table -> total score per architecture.

Also implements the MANDATORY weight-sensitivity / decision-robustness
study (5 named scenarios + Monte Carlo simplex sampling) and a small set
of numeric break-even root-finders.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .normalization import CATEGORIES, METRIC_SPECS, normalize_value
from .trade_metrics import ArchitectureMetrics

ARCHITECTURES: tuple[str, ...] = ("star_tracker", "sun_plus_mag")


def _metric_value_for_category(metrics: ArchitectureMetrics, category: str) -> float:
    mapping = {
        "accuracy": metrics.worst_axis_error_deg,
        "availability": metrics.availability_fraction,
        "cadence": metrics.cadence_hz,
        "mass": metrics.mass_kg,
        "power": metrics.avg_power_w,
        "cost": metrics.cost_score,
        "complexity": metrics.complexity_score,
    }
    if category not in mapping:
        raise ValueError(f"Unknown category {category!r}; expected one of {CATEGORIES}")
    return mapping[category]


# ---------------------------------------------------------------------------
# Weighting
# ---------------------------------------------------------------------------

_WEIGHT_SUM_TOL = 1e-6


@dataclass(frozen=True)
class WeightSet:
    """A named, documented mission-priority weighting over the trade categories.

    ``weights`` MUST have exactly the keys in :data:`CATEGORIES`, all
    values >= 0, summing to 1.0 within ``_WEIGHT_SUM_TOL``.
    """

    name: str
    weights: dict[str, float]
    rationale: str

    def __post_init__(self) -> None:
        keys = set(self.weights.keys())
        expected = set(CATEGORIES)
        if keys != expected:
            raise ValueError(
                f"WeightSet {self.name!r} keys {sorted(keys)} do not match the required "
                f"categories {sorted(expected)}"
            )
        for k, v in self.weights.items():
            if v < 0:
                raise ValueError(f"WeightSet {self.name!r}: weight for {k!r} must be >= 0, got {v}")
        total = sum(self.weights.values())
        if abs(total - 1.0) > _WEIGHT_SUM_TOL:
            raise ValueError(
                f"WeightSet {self.name!r} weights must sum to 1.0 (within {_WEIGHT_SUM_TOL}), "
                f"got {total}"
            )

    def as_array(self) -> np.ndarray:
        return np.array([self.weights[c] for c in CATEGORIES], dtype=float)


# ---------------------------------------------------------------------------
# Baseline mission-priority weighting + the 5 MANDATORY named scenarios.
#
# Full written rationale for every scenario lives in
# docs/trade_study_methodology.md ("Baseline mission priorities" and
# "Weighting scenarios" sections); short rationale is carried inline here
# so it stays attached to the numbers.
# ---------------------------------------------------------------------------

BASELINE_WEIGHTS = WeightSet(
    name="baseline",
    weights={
        "accuracy": 0.20,
        "availability": 0.20,
        "cadence": 0.10,
        "mass": 0.15,
        "power": 0.15,
        "cost": 0.10,
        "complexity": 0.10,
    },
    rationale=(
        "A balanced representative smallsat priority set: attitude "
        "accuracy and operational availability/continuity are weighted "
        "equally and highest (0.20 each) as the two core ADCS-performance "
        "drivers; mass and power are weighted next (0.15 each) reflecting "
        "real but secondary resource pressure on a smallsat bus; update "
        "cadence, cost, and complexity are weighted lowest (0.10 each) as "
        "real but comparatively second-order concerns for a mission that "
        "has not (in this baseline) declared an extreme resource, cost, or "
        "cadence constraint."
    ),
)

ACCURACY_PRIORITY_WEIGHTS = WeightSet(
    name="accuracy_priority",
    weights={
        "accuracy": 0.45,
        "availability": 0.15,
        "cadence": 0.10,
        "mass": 0.10,
        "power": 0.10,
        "cost": 0.05,
        "complexity": 0.05,
    },
    rationale=(
        "A mission with a demanding pointing-KNOWLEDGE requirement (e.g. a "
        "narrow-FOV imaging or laser-communication payload) that makes "
        "instantaneous attitude accuracy the dominant driver, at the "
        "expense of resource/cost/complexity weight."
    ),
)

RESOURCE_CONSTRAINED_WEIGHTS = WeightSet(
    name="resource_constrained",
    weights={
        "accuracy": 0.10,
        "availability": 0.10,
        "cadence": 0.05,
        "mass": 0.35,
        "power": 0.30,
        "cost": 0.05,
        "complexity": 0.05,
    },
    rationale=(
        "A very small/mass- and power-constrained smallsat (e.g. a 1-3U "
        "CubeSat class bus) where mass and power margin dominate the "
        "sensor-suite decision over accuracy or availability."
    ),
)

AVAILABILITY_PRIORITY_WEIGHTS = WeightSet(
    name="availability_priority",
    weights={
        "accuracy": 0.10,
        "availability": 0.35,
        "cadence": 0.20,
        "mass": 0.10,
        "power": 0.10,
        "cost": 0.075,
        "complexity": 0.075,
    },
    rationale=(
        "A mission with a strict operational-continuity requirement (e.g. "
        "near-continuous fine-pointing for a payload that cannot tolerate "
        "long attitude-determination gaps) that weights availability and "
        "update cadence heavily over raw instantaneous accuracy."
    ),
)

COST_COMPLEXITY_PRIORITY_WEIGHTS = WeightSet(
    name="cost_complexity_priority",
    weights={
        "accuracy": 0.10,
        "availability": 0.10,
        "cadence": 0.05,
        "mass": 0.10,
        "power": 0.10,
        "cost": 0.30,
        "complexity": 0.25,
    },
    rationale=(
        "A schedule/budget-constrained student or early-stage program "
        "mission that weights procurement cost and implementation/"
        "operational complexity burden above absolute performance."
    ),
)

SCENARIOS: dict[str, WeightSet] = {
    ws.name: ws
    for ws in (
        BASELINE_WEIGHTS,
        ACCURACY_PRIORITY_WEIGHTS,
        RESOURCE_CONSTRAINED_WEIGHTS,
        AVAILABILITY_PRIORITY_WEIGHTS,
        COST_COMPLEXITY_PRIORITY_WEIGHTS,
    )
}


# ---------------------------------------------------------------------------
# Raw / normalized / weighted tables
# ---------------------------------------------------------------------------


def build_raw_table(star_metrics: ArchitectureMetrics, sunmag_metrics: ArchitectureMetrics) -> pd.DataFrame:
    """Raw metric table: rows = architectures, columns = trade categories."""
    rows = {}
    for m in (star_metrics, sunmag_metrics):
        rows[m.architecture] = {c: _metric_value_for_category(m, c) for c in CATEGORIES}
    return pd.DataFrame(rows).T[list(CATEGORIES)]


def build_normalized_table(raw_table: pd.DataFrame) -> pd.DataFrame:
    """Normalized [0, 1] utility table, same shape as ``raw_table``."""
    out = pd.DataFrame(index=raw_table.index, columns=raw_table.columns, dtype=float)
    for category in raw_table.columns:
        spec = METRIC_SPECS[category]
        for arch in raw_table.index:
            out.loc[arch, category] = normalize_value(raw_table.loc[arch, category], spec)
    return out


def compute_weighted_scores(
    normalized_table: pd.DataFrame, weight_set: WeightSet
) -> tuple[pd.DataFrame, pd.Series]:
    """Weighted-contribution table (utility * weight) and total score per architecture."""
    weights = pd.Series(weight_set.weights)[list(normalized_table.columns)]
    contributions = normalized_table.mul(weights, axis=1)
    totals = contributions.sum(axis=1)
    return contributions, totals


@dataclass(frozen=True)
class TradeMatrixResult:
    weight_set: WeightSet
    raw_table: pd.DataFrame
    normalized_table: pd.DataFrame
    weighted_contributions: pd.DataFrame
    totals: pd.Series
    winner: str

    @property
    def score_margin(self) -> float:
        """star_tracker total minus sun_plus_mag total (documented sign convention)."""
        return float(self.totals["star_tracker"] - self.totals["sun_plus_mag"])


def run_trade_matrix(
    star_metrics: ArchitectureMetrics,
    sunmag_metrics: ArchitectureMetrics,
    weight_set: WeightSet = BASELINE_WEIGHTS,
) -> TradeMatrixResult:
    """Run the full raw -> normalized -> weighted -> total pipeline for one weighting."""
    raw = build_raw_table(star_metrics, sunmag_metrics)
    normalized = build_normalized_table(raw)
    contributions, totals = compute_weighted_scores(normalized, weight_set)
    winner = str(totals.idxmax())
    return TradeMatrixResult(
        weight_set=weight_set,
        raw_table=raw,
        normalized_table=normalized,
        weighted_contributions=contributions,
        totals=totals,
        winner=winner,
    )


# ---------------------------------------------------------------------------
# Part F: Monte Carlo weight-space robustness study
# ---------------------------------------------------------------------------


def sample_weight_vectors(n_samples: int, seed: int, n_categories: int = len(CATEGORIES)) -> np.ndarray:
    """Uniform-over-the-simplex weight samples via ``rng.dirichlet(alpha=ones(n))``.

    Returns an (n_samples, n_categories) array; each row sums to 1.0 and is
    non-negative by construction of the Dirichlet distribution.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(n_categories), size=n_samples)


@dataclass(frozen=True)
class MonteCarloResult:
    n_samples: int
    seed: int
    weight_samples: np.ndarray  # (n_samples, n_categories)
    score_diff: np.ndarray  # star_tracker_score - sun_plus_mag_score, per sample
    fraction_favoring_star_tracker: float
    fraction_favoring_sun_plus_mag: float
    fraction_tied: float


def monte_carlo_weight_sensitivity(
    star_metrics: ArchitectureMetrics,
    sunmag_metrics: ArchitectureMetrics,
    n_samples: int = 20000,
    seed: int = 42,
) -> MonteCarloResult:
    """Monte Carlo weight-space robustness study.

    Samples ``n_samples`` weight vectors uniformly over the 7-category
    simplex (fixed ``seed`` for full reproducibility) and, for each,
    computes the weighted total score for both architectures using the
    SAME normalized utilities as the baseline trade matrix (only the
    weights vary - the underlying metrics/normalization are held fixed).
    Reports the fraction of weight-space samples favoring each
    architecture and the SCORE-DIFFERENCE distribution, defined
    consistently as (star_tracker score) - (sun_plus_mag score), matching
    :attr:`TradeMatrixResult.score_margin`.
    """
    raw = build_raw_table(star_metrics, sunmag_metrics)
    normalized = build_normalized_table(raw)
    star_utility = normalized.loc["star_tracker", list(CATEGORIES)].to_numpy(dtype=float)
    sunmag_utility = normalized.loc["sun_plus_mag", list(CATEGORIES)].to_numpy(dtype=float)

    samples = sample_weight_vectors(n_samples, seed, n_categories=len(CATEGORIES))
    star_scores = samples @ star_utility
    sunmag_scores = samples @ sunmag_utility
    diff = star_scores - sunmag_scores

    tie_tol = 1e-12
    frac_star = float(np.mean(diff > tie_tol))
    frac_sunmag = float(np.mean(diff < -tie_tol))
    frac_tied = float(np.mean(np.abs(diff) <= tie_tol))

    return MonteCarloResult(
        n_samples=n_samples,
        seed=seed,
        weight_samples=samples,
        score_diff=diff,
        fraction_favoring_star_tracker=frac_star,
        fraction_favoring_sun_plus_mag=frac_sunmag,
        fraction_tied=frac_tied,
    )


# ---------------------------------------------------------------------------
# Part G: numeric break-even analysis
# ---------------------------------------------------------------------------


def _score_margin_with_overrides(
    star_metrics: ArchitectureMetrics,
    sunmag_metrics: ArchitectureMetrics,
    weight_set: WeightSet,
    star_overrides: dict | None = None,
    sunmag_overrides: dict | None = None,
) -> float:
    """star_tracker total minus sun_plus_mag total, with raw-field overrides applied."""
    from dataclasses import replace

    star = replace(star_metrics, **(star_overrides or {}))
    sunmag = replace(sunmag_metrics, **(sunmag_overrides or {}))
    result = run_trade_matrix(star, sunmag, weight_set)
    return result.score_margin


def _find_root_or_none(f, lo: float, hi: float, xtol: float = 1e-6) -> float | None:
    """Bisection root-find via ``scipy.optimize.brentq``; None if no sign change in [lo, hi]."""
    f_lo, f_hi = f(lo), f(hi)
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if (f_lo > 0) == (f_hi > 0):
        return None
    return float(brentq(f, lo, hi, xtol=xtol))


def breakeven_sunmag_availability(
    star_metrics: ArchitectureMetrics,
    sunmag_metrics: ArchitectureMetrics,
    weight_set: WeightSet = BASELINE_WEIGHTS,
) -> float | None:
    """Minimum Sun+mag ``availability_fraction`` (holding everything else fixed) at
    which the weighted score margin flips from favoring the star tracker to
    favoring Sun+mag (or vice versa). Returns None if no crossing exists in [0, 1].
    """

    def margin(avail: float) -> float:
        return _score_margin_with_overrides(
            star_metrics, sunmag_metrics, weight_set, sunmag_overrides={"availability_fraction": avail}
        )

    return _find_root_or_none(margin, 0.0, 1.0)


def breakeven_sunmag_accuracy(
    star_metrics: ArchitectureMetrics,
    sunmag_metrics: ArchitectureMetrics,
    weight_set: WeightSet = BASELINE_WEIGHTS,
) -> float | None:
    """Sun+mag worst-axis error [deg] (holding everything else fixed) at which the
    weighted score margin flips. Returns None if no crossing exists in (0, current].
    """
    hi = sunmag_metrics.worst_axis_error_deg

    def margin(err_deg: float) -> float:
        return _score_margin_with_overrides(
            star_metrics, sunmag_metrics, weight_set, sunmag_overrides={"worst_axis_error_deg": err_deg}
        )

    return _find_root_or_none(margin, 1e-6, hi)


def breakeven_star_tracker_mass(
    star_metrics: ArchitectureMetrics,
    sunmag_metrics: ArchitectureMetrics,
    weight_set: WeightSet = BASELINE_WEIGHTS,
    hi_kg: float = 20.0,
) -> float | None:
    """Star-tracker mass [kg] (holding everything else fixed) at which the weighted
    score margin flips from favoring the star tracker to favoring Sun+mag.
    Returns None if no crossing exists in [current mass, hi_kg].
    """
    lo = star_metrics.mass_kg

    def margin(mass_kg: float) -> float:
        return _score_margin_with_overrides(
            star_metrics, sunmag_metrics, weight_set, star_overrides={"mass_kg": mass_kg}
        )

    return _find_root_or_none(margin, lo, hi_kg)


def breakeven_star_tracker_power(
    star_metrics: ArchitectureMetrics,
    sunmag_metrics: ArchitectureMetrics,
    weight_set: WeightSet = BASELINE_WEIGHTS,
    hi_w: float = 20.0,
) -> float | None:
    """Star-tracker average power [W] (holding everything else fixed) at which the
    weighted score margin flips. Returns None if no crossing exists in
    [current power, hi_w].
    """
    lo = star_metrics.avg_power_w

    def margin(power_w: float) -> float:
        return _score_margin_with_overrides(
            star_metrics, sunmag_metrics, weight_set, star_overrides={"avg_power_w": power_w}
        )

    return _find_root_or_none(margin, lo, hi_w)


def breakeven_availability_weight(
    star_metrics: ArchitectureMetrics,
    sunmag_metrics: ArchitectureMetrics,
    base_weight_set: WeightSet = BASELINE_WEIGHTS,
) -> float | None:
    """Minimum 'availability' category weight (scanning w_avail in [0, 1], with all
    OTHER category weights scaled down proportionally to keep the total at 1.0,
    preserving their baseline RELATIVE proportions) at which the weighted score
    margin flips. Returns None if no crossing exists in [0, 1].
    """
    other_categories = [c for c in CATEGORIES if c != "availability"]
    base_other_total = sum(base_weight_set.weights[c] for c in other_categories)

    def weights_for(w_avail: float) -> WeightSet:
        remaining = 1.0 - w_avail
        if base_other_total <= 0:
            raise ValueError("Baseline weight set has zero weight on non-availability categories.")
        scale = remaining / base_other_total
        weights = {c: base_weight_set.weights[c] * scale for c in other_categories}
        weights["availability"] = w_avail
        return WeightSet(name="_scan_availability_weight", weights=weights, rationale="internal scan")

    def margin(w_avail: float) -> float:
        ws = weights_for(w_avail)
        return _score_margin_with_overrides(star_metrics, sunmag_metrics, ws)

    return _find_root_or_none(margin, 0.0, 1.0)
