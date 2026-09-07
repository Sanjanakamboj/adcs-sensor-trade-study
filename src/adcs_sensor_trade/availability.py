"""Deterministic sensor-availability and update-cadence simulation.

Builds on ``orbit_environment.py`` (orbit/eclipse/attitude geometry) and
reuses the Milestone 1 sensor models (``sensors/*``) and attitude-
information geometry machinery (``geometry.py``, ``information.py``)
unchanged. Nothing here is a Monte Carlo simulation: every quantity is
computed from a deterministic time grid and deterministic sensor
availability logic (RNG is never used - ``rng=None`` is passed to every
sensor ``.measure()`` call, which the M1 sensor models document as their
deterministic/no-noise mode).

See ``docs/availability_methodology.md`` for the full write-up of
definitions (availability, full-rank, well-conditioned, synchronization
window) and their justification.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .geometry import DEFAULT_RANK_TOL, condition_number, geometry_rank
from .information import combined_vector_information, summarize_information
from .orbit_environment import (
    DEFAULT_B_FIELD_MAGNITUDE_TESLA,
    DEFAULT_BETA_ANGLE_DEG,
    DEFAULT_DIP_TILT_DEG,
    environment_state,
    environment_states_batch,
    orbital_period_s,
)
from .rotations import rotate_inertial_to_body
from .sensors.magnetometer import Magnetometer, MagnetometerConfig
from .sensors.star_tracker import StarTracker, StarTrackerConfig
from .sensors.sun_sensor import SunSensor, SunSensorConfig

# Documented conditioning threshold (see docs/availability_methodology.md
# for the justification): a two-vector information-matrix condition number
# above this value is labeled "poorly conditioned". Condition number is the
# ratio of worst-axis to best-axis attitude-information eigenvalues, so its
# SQUARE ROOT is the ratio of worst-axis to best-axis 1-sigma attitude-error
# proxy. cond(J) = 10 => worst/best axis error ratio = sqrt(10) ~= 3.16x,
# a threshold commonly used as a practical "meaningfully anisotropic"
# cutoff in attitude-determination geometry engineering practice. This
# corresponds to a Sun-mag separation angle of about 18 deg or about 162
# deg (see docs/availability_methodology.md derivation).
DEFAULT_CONDITION_NUMBER_THRESHOLD = 10.0


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


@dataclass(frozen=True)
class OrbitAvailabilityConfig:
    """Configuration for one deterministic availability-timeline run.

    Parameters
    ----------
    altitude_km, beta_angle_deg
        Orbit/eclipse geometry (see ``orbit_environment.py``).
    n_orbits
        Number of orbital periods to simulate. Must be > 0.
    dt_s
        Fixed time step [s] for the availability timeline. Must be > 0 and
        small relative to the orbital period (validated loosely: must be <
        the orbital period itself).
    star_tracker_config, sun_sensor_config, magnetometer_config
        Sensor configuration objects (Milestone 1 dataclasses), reused
        unmodified.
    b_field_magnitude_tesla, dip_tilt_deg
        Representative inertial magnetic-field magnitude [T] and
        tilted-dipole axis tilt [deg] (see ``orbit_environment.py`` module
        docstring for the model and its IGRF-deferral caveat).
    condition_number_threshold
        Two-vector information condition-number threshold above which a
        sample is labeled "poorly conditioned" (see module-level constant
        docstring and docs/availability_methodology.md).
    rank_tol
        SVD rank tolerance passed through to ``geometry.geometry_rank``.
    """

    altitude_km: float = 600.0
    beta_angle_deg: float = DEFAULT_BETA_ANGLE_DEG
    n_orbits: float = 1.0
    dt_s: float = 1.0
    star_tracker_config: StarTrackerConfig = field(default_factory=StarTrackerConfig)
    sun_sensor_config: SunSensorConfig = field(default_factory=SunSensorConfig)
    magnetometer_config: MagnetometerConfig = field(default_factory=MagnetometerConfig)
    b_field_magnitude_tesla: float = DEFAULT_B_FIELD_MAGNITUDE_TESLA
    dip_tilt_deg: float = DEFAULT_DIP_TILT_DEG
    condition_number_threshold: float = DEFAULT_CONDITION_NUMBER_THRESHOLD
    rank_tol: float = DEFAULT_RANK_TOL

    def __post_init__(self) -> None:
        _validate_positive("altitude_km", self.altitude_km)
        if not (-90.0 <= self.beta_angle_deg <= 90.0):
            raise ValueError(f"beta_angle_deg must be in [-90, 90], got {self.beta_angle_deg}")
        _validate_positive("n_orbits", self.n_orbits)
        _validate_positive("dt_s", self.dt_s)
        period_s = orbital_period_s(self.altitude_km)
        if self.dt_s >= period_s:
            raise ValueError(
                f"dt_s ({self.dt_s} s) must be smaller than the orbital period "
                f"({period_s:.1f} s)."
            )
        if self.condition_number_threshold <= 1.0:
            raise ValueError(
                "condition_number_threshold must be > 1.0 (1.0 is the perfectly "
                f"conditioned lower bound), got {self.condition_number_threshold}"
            )
        if self.rank_tol <= 0:
            raise ValueError(f"rank_tol must be > 0, got {self.rank_tol}")
        _validate_positive("b_field_magnitude_tesla", self.b_field_magnitude_tesla)
        if not (0.0 <= self.dip_tilt_deg <= 90.0):
            raise ValueError(f"dip_tilt_deg must be in [0, 90], got {self.dip_tilt_deg}")

    @property
    def period_s(self) -> float:
        return orbital_period_s(self.altitude_km)


def run_availability_timeline(config: OrbitAvailabilityConfig) -> pd.DataFrame:
    """Simulate a deterministic sensor-availability timeline over ``config.n_orbits`` orbits.

    Returns a DataFrame with one row per time sample, tagging WHY each
    sensor/architecture is or is not available at that sample (not just a
    boolean), plus the instantaneous Sun-mag separation angle and
    (where computable) two-vector attitude-information quality metrics.
    """
    period_s = config.period_s
    total_time_s = config.n_orbits * period_s
    t_grid = np.arange(0.0, total_time_s, config.dt_s)

    batch = environment_states_batch(
        t_grid,
        period_s,
        config.altitude_km,
        config.beta_angle_deg,
        config.b_field_magnitude_tesla,
        config.dip_tilt_deg,
    )

    star = StarTracker(config.star_tracker_config)
    sun = SunSensor(config.sun_sensor_config)
    mag = Magnetometer(config.magnetometer_config)

    rows = []
    for i, t_s in enumerate(t_grid):
        phi = batch.attitude_rotvec_rad[i]
        in_eclipse = bool(batch.in_eclipse[i])
        bright_angle = float(batch.star_tracker_bright_source_angle_rad[i])
        b_field_i = batch.b_field_inertial_tesla[i]

        # --- Star tracker -------------------------------------------------
        star_available = star.is_available(bright_angle)
        star_reason = "ok" if star_available else "sun_exclusion"

        # --- Sun sensor -----------------------------------------------------
        sun_meas = sun.measure(batch.sun_vector_inertial, phi, rng=None, in_eclipse=in_eclipse)
        if sun_meas.available:
            sun_reason = "ok"
        elif in_eclipse:
            sun_reason = "eclipse"
        else:
            sun_reason = "fov"

        # --- Magnetometer ---------------------------------------------------
        # Documented assumption: the magnetometer is modeled as ALWAYS
        # available (no eclipse/FOV/exclusion dependence for a body-mounted
        # magnetic-field sensor). See docs/availability_methodology.md for
        # the rationale (deferred: local/spacecraft-generated field
        # interference, saturation near high-field anomalies).
        mag_meas = mag.measure(b_field_i, phi, rng=None)
        mag_available = mag_meas.available
        mag_reason = "ok"

        # --- Sun-mag combined geometry (TRUE, noise-free vectors) -----------
        separation_angle_deg = float(batch.separation_angle_deg[i])
        sunmag_vectors_usable = bool(sun_meas.available and mag_available)
        rank = None
        cond = None
        worst_axis_error_deg = None
        if sunmag_vectors_usable:
            vecs = [sun_meas.sun_vector_body, mag_meas.b_field_direction_body]
            rank = geometry_rank(vecs, tol=config.rank_tol)
            cond = condition_number(vecs)
            if rank == 3:
                J = combined_vector_information(
                    vecs,
                    [config.sun_sensor_config.sigma_rad, config.magnetometer_config.sigma_rad],
                )
                report = summarize_information(J, rank_tol=config.rank_tol)
                worst_axis_error_deg = report.worst_axis_error_deg

        full_rank = bool(sunmag_vectors_usable and rank == 3)
        well_conditioned = bool(
            full_rank and cond is not None and cond <= config.condition_number_threshold
        )

        if not sunmag_vectors_usable:
            combined_reason = sun_reason if not sun_meas.available else mag_reason
        elif not full_rank:
            combined_reason = "rank_deficient"
        elif not well_conditioned:
            combined_reason = "poorly_conditioned"
        else:
            combined_reason = "ok"

        rows.append(
            {
                "t_s": float(t_s),
                "phase_deg": float(np.degrees(batch.phase_rad[i])),
                "in_eclipse": in_eclipse,
                "star_tracker_available": bool(star_available),
                "star_tracker_reason": star_reason,
                "sun_sensor_available": bool(sun_meas.available),
                "sun_sensor_reason": sun_reason,
                "magnetometer_available": bool(mag_available),
                "magnetometer_reason": mag_reason,
                "separation_angle_deg": separation_angle_deg,
                "sunmag_vectors_usable": sunmag_vectors_usable,
                "sunmag_rank": rank,
                "sunmag_condition_number": cond,
                "sunmag_full_rank": full_rank,
                "sunmag_well_conditioned": well_conditioned,
                "sunmag_worst_axis_error_deg": worst_axis_error_deg,
                "sunmag_available": well_conditioned,
                "sunmag_reason": combined_reason,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Outage-interval detection
# ---------------------------------------------------------------------------

def outage_intervals(available: np.ndarray, dt_s: float) -> list[dict]:
    """Find contiguous False ("unavailable") runs in a boolean timeline.

    ``available`` is a 1-D boolean array sampled uniformly at spacing
    ``dt_s`` [s]. Returns a list of dicts, one per outage, each with
    ``start_index``, ``end_index`` (exclusive), ``start_s``, and
    ``duration_s``.

    LIMITATION: an outage that straddles the first/last sample of the array
    (e.g. wraps across an artificial single-orbit boundary) is reported as
    two separate partial outages rather than one continuous one. For
    ``n_orbits > 1`` this only affects the very first/last sample of the
    whole run, not internal orbit boundaries.
    """
    available = np.asarray(available, dtype=bool)
    if available.ndim != 1:
        raise ValueError("available must be a 1-D boolean array.")
    if available.size == 0:
        raise ValueError("available must be non-empty.")
    _validate_positive("dt_s", dt_s)

    intervals = []
    n = available.size
    i = 0
    while i < n:
        if not available[i]:
            start = i
            while i < n and not available[i]:
                i += 1
            end = i
            intervals.append(
                {
                    "start_index": start,
                    "end_index": end,
                    "start_s": start * dt_s,
                    "duration_s": (end - start) * dt_s,
                }
            )
        else:
            i += 1
    return intervals


def summarize_boolean_series(available: np.ndarray, dt_s: float) -> dict:
    """Summarize an availability boolean timeline: fraction available, outages, etc."""
    available = np.asarray(available, dtype=bool)
    if available.size == 0:
        raise ValueError("available must be non-empty.")
    intervals = outage_intervals(available, dt_s)
    durations = np.array([iv["duration_s"] for iv in intervals], dtype=float)
    return {
        "fraction_available": float(np.mean(available)),
        "num_outages": len(intervals),
        "longest_outage_s": float(durations.max()) if durations.size else 0.0,
        "median_outage_s": float(np.median(durations)) if durations.size else 0.0,
        "mean_outage_s": float(np.mean(durations)) if durations.size else 0.0,
        "outage_intervals": intervals,
    }


# ---------------------------------------------------------------------------
# Asynchronous update-rate / cadence analysis
# ---------------------------------------------------------------------------

def async_sample_times(rate_hz: float, duration_s: float) -> np.ndarray:
    """Sample times [s] for a sensor sampled at a fixed rate, starting at t=0."""
    _validate_positive("rate_hz", rate_hz)
    _validate_positive("duration_s", duration_s)
    dt = 1.0 / rate_hz
    n = int(np.floor(duration_s / dt)) + 1
    return np.arange(n) * dt


def default_sync_window_s(sun_rate_hz: float, mag_rate_hz: float) -> float:
    """Default synchronization window: half the period of the SLOWER sensor.

    Rationale (see docs/availability_methodology.md): a full-attitude
    update requires pairing one Sun sample with the temporally-nearest
    magnetometer sample. Since the magnetometer is (by baseline
    assumption) faster than the Sun sensor, every Sun sample is guaranteed
    to have a magnetometer sample within half of the SLOWER (Sun) sensor's
    own sample period - this is the largest window that still reflects
    "staleness no worse than the rate-limiting sensor's own cadence",
    rather than an arbitrarily generous window.
    """
    _validate_positive("sun_rate_hz", sun_rate_hz)
    _validate_positive("mag_rate_hz", mag_rate_hz)
    slower_rate_hz = min(sun_rate_hz, mag_rate_hz)
    return 0.5 / slower_rate_hz


def compute_full_attitude_update_cadence(
    config: OrbitAvailabilityConfig, sync_window_s: float | None = None
) -> dict:
    """Quantify the effective Sun+mag full-attitude update cadence.

    Builds independent asynchronous sample-time grids for the Sun sensor
    and magnetometer (each at its own configured rate). For every Sun
    sample, finds the temporally-nearest magnetometer sample; if it falls
    within ``sync_window_s`` [s] of the Sun sample AND the resulting
    two-vector geometry at (approximately) that instant is available,
    full-rank, and well-conditioned, it counts as one usable full-attitude
    update. See :func:`default_sync_window_s` for the default window and
    its justification.

    Returns a dict with ``count``, ``duration_s``, ``cadence_hz``,
    ``updates_per_orbit``, and the per-Sun-sample ``events`` DataFrame.
    """
    period_s = config.period_s
    duration_s = config.n_orbits * period_s
    sun_rate_hz = config.sun_sensor_config.update_rate_hz
    mag_rate_hz = config.magnetometer_config.update_rate_hz

    if sync_window_s is None:
        sync_window_s = default_sync_window_s(sun_rate_hz, mag_rate_hz)
    _validate_positive("sync_window_s", sync_window_s)

    sun_times = async_sample_times(sun_rate_hz, duration_s)
    mag_times = async_sample_times(mag_rate_hz, duration_s)

    # Vectorized nearest-neighbor lookup (mag_times is sorted/uniform, so
    # np.searchsorted gives an O(N log M) alternative to an O(N*M) full
    # pairwise search, which matters once either rate gets large in the
    # update-rate sensitivity sweeps).
    right_idx = np.clip(np.searchsorted(mag_times, sun_times, side="left"), 0, len(mag_times) - 1)
    left_idx = np.clip(right_idx - 1, 0, len(mag_times) - 1)
    left_dist = np.abs(mag_times[left_idx] - sun_times)
    right_dist = np.abs(mag_times[right_idx] - sun_times)
    nearest_idx = np.where(left_dist <= right_dist, left_idx, right_idx)

    sun = SunSensor(config.sun_sensor_config)
    mag = Magnetometer(config.magnetometer_config)

    batch_sun = environment_states_batch(
        sun_times, period_s, config.altitude_km, config.beta_angle_deg,
        config.b_field_magnitude_tesla, config.dip_tilt_deg,
    )
    t_mag_for_sun = mag_times[nearest_idx]
    batch_mag = environment_states_batch(
        t_mag_for_sun, period_s, config.altitude_km, config.beta_angle_deg,
        config.b_field_magnitude_tesla, config.dip_tilt_deg,
    )

    rows = []
    usable_count = 0
    for i, t_sun in enumerate(sun_times):
        t_mag = t_mag_for_sun[i]
        staleness_s = abs(t_mag - t_sun)
        within_window = staleness_s <= sync_window_s

        sun_meas = sun.measure(
            batch_sun.sun_vector_inertial,
            batch_sun.attitude_rotvec_rad[i],
            rng=None,
            in_eclipse=bool(batch_sun.in_eclipse[i]),
        )
        mag_meas = mag.measure(
            batch_mag.b_field_inertial_tesla[i], batch_mag.attitude_rotvec_rad[i], rng=None
        )

        usable = False
        rank = None
        cond = None
        if within_window and sun_meas.available and mag_meas.available:
            vecs = [sun_meas.sun_vector_body, mag_meas.b_field_direction_body]
            rank = geometry_rank(vecs, tol=config.rank_tol)
            cond = condition_number(vecs)
            usable = bool(rank == 3 and cond <= config.condition_number_threshold)

        if usable:
            usable_count += 1

        rows.append(
            {
                "t_sun_s": float(t_sun),
                "t_mag_s": float(t_mag),
                "staleness_s": float(staleness_s),
                "within_sync_window": within_window,
                "sun_available": bool(sun_meas.available),
                "mag_available": bool(mag_meas.available),
                "rank": rank,
                "condition_number": cond,
                "usable_full_attitude_update": usable,
            }
        )

    events = pd.DataFrame(rows)
    return {
        "count": usable_count,
        "duration_s": float(duration_s),
        "cadence_hz": float(usable_count / duration_s) if duration_s > 0 else 0.0,
        "updates_per_orbit": float(usable_count / config.n_orbits),
        "sync_window_s": float(sync_window_s),
        "events": events,
    }


def star_tracker_usable_updates(
    config: OrbitAvailabilityConfig, star_rate_hz: float | None = None
) -> dict:
    """Usable full-attitude update count/cadence for the star tracker ALONE.

    The star tracker directly provides full 3-axis attitude on every
    sample where it is not exclusion-blocked - no separate rank/
    conditioning check is needed (see ``information.py`` isotropic-model
    rationale). ``star_rate_hz`` overrides the configured update rate,
    for use in update-rate sensitivity sweeps.
    """
    rate_hz = star_rate_hz if star_rate_hz is not None else config.star_tracker_config.update_rate_hz
    _validate_positive("rate_hz", rate_hz)
    duration_s = config.n_orbits * config.period_s
    times = async_sample_times(rate_hz, duration_s)
    star = StarTracker(config.star_tracker_config)

    batch = environment_states_batch(
        times,
        config.period_s,
        config.altitude_km,
        config.beta_angle_deg,
        config.b_field_magnitude_tesla,
        config.dip_tilt_deg,
    )
    count = sum(
        1
        for angle in batch.star_tracker_bright_source_angle_rad
        if star.is_available(float(angle))
    )

    return {
        "rate_hz": float(rate_hz),
        "count": count,
        "duration_s": float(duration_s),
        "cadence_hz": float(count / duration_s) if duration_s > 0 else 0.0,
        "updates_per_orbit": float(count / config.n_orbits),
    }


# ---------------------------------------------------------------------------
# Sensitivity studies (all deterministic; see docs/availability_methodology.md
# item G for the full discussion and expected monotonic-behavior checks)
# ---------------------------------------------------------------------------

def sensitivity_exclusion_half_angle(
    base_config: OrbitAvailabilityConfig, half_angles_deg: np.ndarray
) -> pd.DataFrame:
    """Star-tracker availability vs. bright-source exclusion half-angle.

    Physical expectation: wider exclusion half-angle -> LOWER star-tracker
    availability (monotonically non-increasing).
    """
    rows = []
    base_star = base_config.star_tracker_config
    for half_angle_deg in half_angles_deg:
        star_cfg = StarTrackerConfig(
            sigma_rad=base_star.sigma_rad,
            update_rate_hz=base_star.update_rate_hz,
            exclusion_half_angle_rad=np.radians(float(half_angle_deg)),
            dropout_probability=base_star.dropout_probability,
        )
        cfg = replace(base_config, star_tracker_config=star_cfg)
        df = run_availability_timeline(cfg)
        rows.append(
            {
                "exclusion_half_angle_deg": float(half_angle_deg),
                "star_tracker_availability_fraction": float(df["star_tracker_available"].mean()),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_fov_half_angle(
    base_config: OrbitAvailabilityConfig, fov_half_angles_deg: np.ndarray
) -> pd.DataFrame:
    """Sun-sensor / Sun+mag availability vs. Sun-sensor FOV half-angle.

    Physical expectation: wider FOV -> HIGHER Sun-sensor (and Sun+mag)
    availability (monotonically non-decreasing).
    """
    rows = []
    base_sun = base_config.sun_sensor_config
    for fov_deg in fov_half_angles_deg:
        sun_cfg = SunSensorConfig(
            sigma_rad=base_sun.sigma_rad,
            fov_half_angle_rad=np.radians(float(fov_deg)),
            update_rate_hz=base_sun.update_rate_hz,
        )
        cfg = replace(base_config, sun_sensor_config=sun_cfg)
        df = run_availability_timeline(cfg)
        rows.append(
            {
                "fov_half_angle_deg": float(fov_deg),
                "sun_sensor_availability_fraction": float(df["sun_sensor_available"].mean()),
                "sunmag_availability_fraction": float(df["sunmag_available"].mean()),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_eclipse_beta_angle(
    base_config: OrbitAvailabilityConfig, beta_angles_deg: np.ndarray
) -> pd.DataFrame:
    """Eclipse fraction / Sun-sensor / Sun+mag availability vs. beta angle.

    Physical expectation: |beta| increasing -> eclipse fraction decreasing
    (monotonically non-increasing in |beta|, exactly zero above
    ``beta_star``, see :func:`orbit_environment.eclipse_fraction`) ->
    Sun-sensor and Sun+mag availability trending UP as eclipse shrinks
    (star-tracker exclusion geometry also changes with beta - see
    ``orbit_environment.star_tracker_bright_source_angle_rad`` - so the
    star-tracker column is reported too but is not expected to be
    monotonic with |beta| the same way).
    """
    from .orbit_environment import eclipse_fraction

    rows = []
    for beta_deg in beta_angles_deg:
        cfg = replace(base_config, beta_angle_deg=float(beta_deg))
        df = run_availability_timeline(cfg)
        rows.append(
            {
                "beta_angle_deg": float(beta_deg),
                "eclipse_fraction": eclipse_fraction(cfg.altitude_km, float(beta_deg)),
                "star_tracker_availability_fraction": float(df["star_tracker_available"].mean()),
                "sun_sensor_availability_fraction": float(df["sun_sensor_available"].mean()),
                "sunmag_availability_fraction": float(df["sunmag_available"].mean()),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_update_rate(
    base_config: OrbitAvailabilityConfig,
    star_rates_hz: np.ndarray | None = None,
    sun_rates_hz: np.ndarray | None = None,
    mag_rates_hz: np.ndarray | None = None,
) -> pd.DataFrame:
    """Usable full-attitude updates/orbit vs. sensor update rate.

    Varies star-tracker rate (star-tracker-alone updates/orbit), Sun-sensor
    rate, and magnetometer rate (Sun+mag effective full-attitude update
    cadence in each case), one parameter at a time, holding the rest at
    ``base_config``'s values. Used to check: (a) star-tracker/Sun-sensor
    rate increases raise usable full-attitude updates/orbit, and (b) a
    faster magnetometer ALONE does not materially help (see
    docs/availability_methodology.md, question 7).
    """
    rows = []
    if star_rates_hz is not None:
        for rate in star_rates_hz:
            res = star_tracker_usable_updates(base_config, float(rate))
            rows.append(
                {
                    "parameter": "star_tracker_rate_hz",
                    "rate_hz": float(rate),
                    "updates_per_orbit": res["updates_per_orbit"],
                }
            )
    if sun_rates_hz is not None:
        base_sun = base_config.sun_sensor_config
        for rate in sun_rates_hz:
            sun_cfg = SunSensorConfig(
                sigma_rad=base_sun.sigma_rad,
                fov_half_angle_rad=base_sun.fov_half_angle_rad,
                update_rate_hz=float(rate),
            )
            cfg = replace(base_config, sun_sensor_config=sun_cfg)
            cad = compute_full_attitude_update_cadence(cfg)
            rows.append(
                {
                    "parameter": "sun_sensor_rate_hz",
                    "rate_hz": float(rate),
                    "updates_per_orbit": cad["updates_per_orbit"],
                }
            )
    if mag_rates_hz is not None:
        base_mag = base_config.magnetometer_config
        for rate in mag_rates_hz:
            mag_cfg = MagnetometerConfig(
                sigma_rad=base_mag.sigma_rad,
                update_rate_hz=float(rate),
                bias_tesla=base_mag.bias_tesla,
                scale_factor=base_mag.scale_factor,
            )
            cfg = replace(base_config, magnetometer_config=mag_cfg)
            cad = compute_full_attitude_update_cadence(cfg)
            rows.append(
                {
                    "parameter": "magnetometer_rate_hz",
                    "rate_hz": float(rate),
                    "updates_per_orbit": cad["updates_per_orbit"],
                }
            )
    return pd.DataFrame(rows)
