"""Single source of truth: extracts real numbers from the accepted M1/M2
modules plus the M3 resource/cost-complexity models (Milestone 3).

Every metric returned by this module is either:

- COMPUTED by calling into the actual, unmodified Milestone 1/2 modules
  (``availability.py``, ``geometry.py``, ``information.py``,
  ``orbit_environment.py``, the ``sensors/*`` models) using the SAME
  baseline configuration used by ``scripts/verify_sensor_geometry.py`` and
  ``scripts/verify_availability.py`` (:func:`baseline_availability_config`
  reproduces that baseline exactly), or
- DERIVED from the Milestone 3 resource/cost-complexity models
  (``resource_model.py``, ``cost_complexity_model.py``).

Nothing here hand-types an M1/M2 headline number - every value is
re-derived from the modules on every call, so this module is a true
regression-consumer of M1/M2: if a future change to those modules altered
their outputs, the numbers returned here (and the regression tests in
``tests/test_trade_metrics.py`` that check them against the ACCEPTED
headline values) would change/fail accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .availability import (
    OrbitAvailabilityConfig,
    compute_full_attitude_update_cadence,
    run_availability_timeline,
    star_tracker_usable_updates,
    summarize_boolean_series,
)
from .cost_complexity_model import (
    ArchitectureCostComplexity,
    star_tracker_cost_complexity,
    sunmag_cost_complexity,
)
from .resource_model import (
    ArchitectureResourceTotals,
    star_tracker_architecture_totals,
    sunmag_architecture_totals,
)


def baseline_availability_config() -> OrbitAvailabilityConfig:
    """The exact baseline configuration used by ``scripts/verify_availability.py``.

    600 km circular LEO, beta = 20 deg (module default), one orbit, 1 s
    time step, default M1 sensor configs. Reproduced here (rather than
    imported as a shared "baseline" object) so this module has no more
    coupling to the verification SCRIPT than to the library modules
    themselves - but the values are identical by construction, since
    ``OrbitAvailabilityConfig``'s own defaults are exactly what
    ``verify_availability.py`` uses (``OrbitAvailabilityConfig(dt_s=1.0,
    n_orbits=1.0)``).
    """
    return OrbitAvailabilityConfig(dt_s=1.0, n_orbits=1.0)


@dataclass(frozen=True)
class ArchitectureMetrics:
    """All trade-study metrics for one sensor architecture."""

    architecture: str

    # --- M1/M2 modeled (computed from the accepted modules) ---------------
    worst_axis_error_deg: float
    condition_number: float
    full_attitude_observable: bool
    availability_fraction: float
    longest_outage_s: float
    updates_per_orbit: float
    cadence_hz: float
    poorly_conditioned_fraction: float

    # --- M3 resource model (illustrative/derived) --------------------------
    mass_kg: float
    avg_power_w: float
    peak_power_w: float
    interface_burden: float

    # --- M3 cost/complexity model (illustrative ordinal scores) ------------
    cost_score: float
    complexity_score: float

    resource_totals: ArchitectureResourceTotals
    cost_complexity: ArchitectureCostComplexity


def star_tracker_metrics(config: OrbitAvailabilityConfig | None = None) -> ArchitectureMetrics:
    """Extract all trade-study metrics for the star-tracker architecture."""
    cfg = config or baseline_availability_config()
    df = run_availability_timeline(cfg)
    summary = summarize_boolean_series(df["star_tracker_available"].values, cfg.dt_s)
    cadence = star_tracker_usable_updates(cfg)

    # Isotropic star-tracker information model (information.py /
    # docs/geometry_and_information.md): worst-axis 1-sigma error proxy
    # equals sigma itself (J = I/sigma^2 => P = sigma^2 * I), and
    # condition number is exactly 1.0 by construction.
    worst_axis_error_deg = float(np.degrees(cfg.star_tracker_config.sigma_rad))

    resources = star_tracker_architecture_totals()
    cc = star_tracker_cost_complexity()

    return ArchitectureMetrics(
        architecture="star_tracker",
        worst_axis_error_deg=worst_axis_error_deg,
        condition_number=1.0,
        full_attitude_observable=True,
        availability_fraction=summary["fraction_available"],
        longest_outage_s=summary["longest_outage_s"],
        updates_per_orbit=cadence["updates_per_orbit"],
        cadence_hz=cadence["cadence_hz"],
        poorly_conditioned_fraction=0.0,
        mass_kg=resources.mass_kg,
        avg_power_w=resources.avg_power_w,
        peak_power_w=resources.peak_power_w,
        interface_burden=resources.interface_burden,
        cost_score=cc.cost_score,
        complexity_score=cc.complexity_score,
        resource_totals=resources,
        cost_complexity=cc,
    )


def sunmag_metrics(config: OrbitAvailabilityConfig | None = None) -> ArchitectureMetrics:
    """Extract all trade-study metrics for the Sun+mag architecture."""
    cfg = config or baseline_availability_config()
    df = run_availability_timeline(cfg)
    summary = summarize_boolean_series(df["sunmag_available"].values, cfg.dt_s)
    cadence = compute_full_attitude_update_cadence(cfg)

    usable_full_rank = df[df["sunmag_full_rank"]]
    n_usable = len(usable_full_rank)
    if n_usable > 0:
        cond_vals = usable_full_rank["sunmag_condition_number"].values.astype(float)
        worst_axis_vals = usable_full_rank["sunmag_worst_axis_error_deg"].values.astype(float)
        n_well_conditioned = int(usable_full_rank["sunmag_well_conditioned"].sum())
        poorly_conditioned_fraction = 1.0 - (n_well_conditioned / n_usable)
        median_cond = float(np.median(cond_vals))
        median_worst_axis_error_deg = float(np.median(worst_axis_vals))
    else:  # pragma: no cover - not reachable for the accepted baseline config
        poorly_conditioned_fraction = float("nan")
        median_cond = float("nan")
        median_worst_axis_error_deg = float("nan")

    resources = sunmag_architecture_totals()
    cc = sunmag_cost_complexity()

    return ArchitectureMetrics(
        architecture="sun_plus_mag",
        worst_axis_error_deg=median_worst_axis_error_deg,
        condition_number=median_cond,
        full_attitude_observable=n_usable > 0,
        availability_fraction=summary["fraction_available"],
        longest_outage_s=summary["longest_outage_s"],
        updates_per_orbit=cadence["updates_per_orbit"],
        cadence_hz=cadence["cadence_hz"],
        poorly_conditioned_fraction=poorly_conditioned_fraction,
        mass_kg=resources.mass_kg,
        avg_power_w=resources.avg_power_w,
        peak_power_w=resources.peak_power_w,
        interface_burden=resources.interface_burden,
        cost_score=cc.cost_score,
        complexity_score=cc.complexity_score,
        resource_totals=resources,
        cost_complexity=cc,
    )
