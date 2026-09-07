# Milestone 2 Availability Methodology

This document describes the orbit/environment model, availability
definitions, and quality-conditioning threshold used by
`src/adcs_sensor_trade/orbit_environment.py` and
`src/adcs_sensor_trade/availability.py`, and their justification. It
extends (and reuses, unmodified) the Milestone 1 sensor-physics and
attitude-information-geometry machinery documented in
`docs/conventions.md`, `docs/sensor_models.md`, and
`docs/geometry_and_information.md`.

**Read this before treating any Milestone 2 number as more than
illustrative.** Every model choice below is deliberately simple and
closed-form so the study stays transparent and reproducible - none of it
is a substitute for high-fidelity ephemeris, a real geomagnetic field
model, or a flight-qualified attitude estimator.

## 1. Orbit/environment model

- **Orbit**: circular, two-body Keplerian, altitude 600 km (illustrative
  LEO, not a specific mission). Period is computed from
  `T = 2*pi*sqrt(a^3/mu)`. No J2, drag, or other perturbations.
- **Beta angle**: the Sun direction is held FIXED in inertial space at a
  constant angle ("beta angle", default 20 deg) out of the orbit plane for
  the whole simulated span (one to a few orbits, ~90-100 minutes each).
  Real beta angle drifts over weeks due to precession; that drift is
  negligible over the spans analyzed here but is not modeled.
- **Eclipse**: the standard circular-orbit CYLINDRICAL shadow
  approximation (Vallado/Wertz closed form, no penumbra, no Earth
  oblateness):

  ```
  beta_star = asin(R_earth / r)
  eclipse_fraction = 0                                             if |beta| >= beta_star
                    = (1/pi) * arccos( sqrt(r^2 - R_earth^2) / (r*cos(beta)) )   otherwise
  ```

  The eclipse arc is modeled as centered on orbital phase = 180 deg
  ("local midnight"), symmetric, with total angular width
  `2*pi*eclipse_fraction`.
- **Attitude profile**: an idealized NADIR-POINTING (LVLH) profile is
  assumed for the entire study: body +Z = nadir (Earth-pointing), body +X
  = velocity direction, body +Y completes a right-handed triad (works out
  to a fixed ~orbit-normal direction for a circular orbit). This is a
  simple, explicit, deterministic choice made purely to give the Sun- and
  magnetic-field vectors a well-defined time history in the body frame -
  **no real ADCS control loop or attitude dynamics are modeled**. A
  different mission attitude profile (e.g. Sun-pointing, inertial-pointing)
  would give different specific numbers, though the qualitative
  availability mechanisms (eclipse, FOV, exclusion, rank/conditioning)
  would still apply.
- **Star-tracker mounting**: assumed boresighted along body -Z
  (anti-nadir, deep-space pointing) - a common real mounting strategy
  specifically chosen to avoid Earth/Sun stray light. Its exclusion
  condition is therefore the angle between the anti-nadir direction and
  the Sun.
- **Sun-sensor mounting**: boresighted along body +X (velocity direction),
  matching the Milestone 1 convention (`docs/conventions.md`).
- **Magnetic field**: a TILTED-DIPOLE DIRECTION model. A dipole axis fixed
  in inertial space, tilted `dip_tilt_deg` (default 11.5 deg, the commonly
  cited approximate tilt of Earth's real magnetic dipole axis from its
  rotation axis) away from the orbit-normal direction. Field DIRECTION at
  orbital position `r_hat` uses the standard dipole formula
  `3*(m.r_hat)*r_hat - m` (normalized); field MAGNITUDE is held at a
  single representative constant (3.0e-5 T, as in Milestone 1), since only
  DIRECTION matters for the rank/conditioning geometry used here. This
  reproduces the qualitatively-correct feature that the geomagnetic field
  direction (unlike the fixed Sun direction) changes substantially over
  one orbit - but it is explicitly **not** a full IGRF/orbit-resolved
  geomagnetic field model (no magnitude variation, no higher-order
  multipoles, no true rotation-axis/orbit-inclination distinction from the
  orbit-plane geometry used here).

  **Defect caught and fixed during development**: an earlier version of
  this model used a single FIXED inertial B-field vector (as Milestone 1
  does for its single-snapshot analysis). Because both the Sun and B
  vectors would then be rotated into the body frame by the *same* attitude
  DCM at every instant, the Sun-mag separation angle came out perfectly
  CONSTANT over the whole orbit (rotating two fixed vectors by the same
  rotation preserves the angle between them) - failing to show the
  required orbital variation and silently making every sensitivity result
  built on it meaningless. The tilted-dipole model above was introduced
  specifically to give the B-field direction its own *inertial* variation
  with orbital position, independent of the attitude rotation, which
  restores a physically-meaningful varying separation angle (confirmed
  numerically: separation ranges roughly 88-121 deg over one orbit at the
  default beta = 20 deg, see `tests/test_orbit_environment.py::test_sun_mag_separation_angle_varies_over_orbit`).

## 2. Availability definitions

- **Star tracker available** iff the anti-nadir-to-Sun angle is outside
  the configured exclusion half-angle (`StarTracker.is_available`, reused
  unmodified from Milestone 1).
- **Sun sensor available** iff NOT in eclipse AND the Sun is within the
  configured FOV half-angle of the +X boresight
  (`SunSensor.is_eclipsed`/`.measure`, reused unmodified).
- **Magnetometer available**: modeled as ALWAYS available. Rationale: a
  body-mounted magnetometer has no eclipse or FOV dependence by physical
  principle (it measures a field, not a line of sight). Explicitly
  DEFERRED: spacecraft-generated magnetic interference, local field
  anomalies/saturation, and any other real-world magnetometer outage
  mode - none of those are modeled here, and this is a deliberate
  simplification, not a claim that real magnetometers never have outages.
- **Sun+mag full-attitude available** iff (a) both the Sun sensor and
  magnetometer report a usable vector, AND (b) the resulting two-vector
  geometry is full rank (`geometry.geometry_rank == 3`, reused unmodified),
  AND (c) the geometry is "well-conditioned" (condition number below the
  threshold defined in Section 3). This explicitly distinguishes "powered
  on"/"geometrically visible" from "actually gives full, reasonably
  well-conditioned attitude observability" - collapsing those two would
  overstate availability.
- Each sample is tagged with a **reason** when unavailable: star tracker
  -> `sun_exclusion`; Sun sensor -> `eclipse` or `fov`; Sun+mag -> whichever
  of `eclipse`/`fov` blocked the Sun sensor, else `rank_deficient` (usable
  vectors but rank < 3), else `poorly_conditioned` (full rank but above the
  conditioning threshold), else `ok`.

## 3. Conditioning ("well-conditioned") threshold

The two-vector information-matrix condition number `cond(J)` is the ratio
of the worst-axis to best-axis attitude-information EIGENVALUES. Since the
attitude-error covariance proxy `P = J^-1` has eigenvalues that are the
RECIPROCALS of `J`'s eigenvalues, the ratio of worst-axis to best-axis
1-sigma attitude-error proxy (`sqrt` of a covariance eigenvalue) is
`sqrt(cond(J))`.

**Threshold used**: `cond(J) <= 10` is labeled "well-conditioned". This
corresponds to `sqrt(10) ~= 3.16`, i.e. a worst-axis attitude error no more
than ~3.16x the best-axis error - a threshold commonly used in
attitude-determination and general numerical-conditioning engineering
practice as a "meaningfully anisotropic but not yet badly degraded" cutoff
(by contrast, `cond(J) = 100` would already correspond to a 10x
worst/best-axis error ratio, which is a much harsher degradation). Using
the Milestone 1 separation-angle sweep (`sweep.py`), `cond(J) = 10` for the
Sun+mag geometry corresponds to a Sun-mag separation angle of about 18 deg
or about 162 deg - i.e. once the two vectors get within about 18 degrees
of collinear/anti-collinear, the geometry is judged too anisotropic to
call "healthy". This is a chosen, documented engineering judgment call,
not a physical law - a different mission's pointing-accuracy requirements
could justify a different threshold, and the threshold is a configurable
parameter (`OrbitAvailabilityConfig.condition_number_threshold`) precisely
so it is never silently hard-coded.

## 4. Asynchronous update-rate treatment and the synchronization window

The star tracker, Sun sensor, and magnetometer sample independently at
their own fixed rates (2 Hz / 5 Hz / 10 Hz baseline). A Sun+mag
full-attitude update requires pairing ONE Sun sample with the temporally
nearest magnetometer sample. If that pairing's time gap ("staleness")
exceeds a synchronization window, the pair is rejected (too stale to treat
as a simultaneous two-vector measurement); otherwise the pair's true
vectors are evaluated for availability, rank, and conditioning exactly as
in Section 2/3, and if all pass, one usable full-attitude update is
counted.

**Default synchronization window**: half the sample period of the SLOWER
of the two sensors (`default_sync_window_s`). Since the magnetometer is,
by the Milestone 1 baseline, faster than the Sun sensor (10 Hz vs 5 Hz),
this evaluates to `0.5 / 5 Hz = 100 ms`. Rationale: every Sun sample is
then guaranteed a magnetometer sample within `100 ms` (since the
magnetometer's own sample spacing, `100 ms` at 10 Hz, is no larger than the
window) - this reflects "staleness no worse than the rate-limiting
sensor's own cadence" without being an arbitrarily generous window chosen
to inflate the usable-update count. The window is a configurable parameter
of `compute_full_attitude_update_cadence`, not a hidden constant.

**Important, by-construction result** (Milestone 2 question 7): raising
the magnetometer's rate ALONE (holding the Sun sensor's rate fixed) does
NOT materially raise the usable full-attitude update count, because a
magnetometer sample by itself is rank-2 (see `information.py`) - the
bottleneck for a full-attitude update is always the availability/rate of
the SLOWER, rank-completing sensor (the Sun sensor here), not the faster
one. This is verified numerically in
`tests/test_availability.py::test_faster_magnetometer_alone_does_not_increase_usable_updates`
and in the update-rate sensitivity sweep (Section 5g of the Milestone 2
script output).

## 5. Outage-interval detection

`availability.outage_intervals` scans a uniformly-sampled boolean
timeline for contiguous `False` ("unavailable") runs, returning each
run's start index/time and duration. **Limitation**: an outage that
straddles the very first/last sample of a simulated span (e.g. an
artificial single-orbit boundary that happens to fall inside a real
outage) is reported as two separate partial outages rather than one
continuous one. This is a genuine limitation of finite-window batch
analysis, not a bug, and is why some Milestone 2 report numbers show,
e.g., "2 outages" for a physical mechanism that would recur once per
orbit in steady state - the boundary artifact is confirmed and explained
per-case in the script output rather than hidden.

## 6. Limitations (full list; see also the M2 script's answer to Question 9)

- Circular, unperturbed two-body orbit; no J2/drag/precession.
- Fixed beta angle over the simulated span (valid for ~1-2 orbits, not
  weeks).
- Idealized fixed nadir-pointing attitude profile; no real ADCS control
  loop, no attitude dynamics, no other candidate pointing profile explored
  quantitatively (though the mechanisms generalize).
- Cylindrical (no penumbra, no oblateness) eclipse model.
- Tilted-dipole magnetic FIELD DIRECTION model with constant magnitude;
  not IGRF, no higher-order multipoles.
- Magnetometer modeled as always available (no interference/saturation
  modes).
- No gyro propagation or full estimator (MEKF/EKF) - "availability" here
  means an instantaneous measurement is usable, not that a real filter's
  covariance stays bounded through an outage.
- No mass, power, or cost modeling - deferred to Milestone 3.
- No vendor/product-specific hardware qualification claims anywhere in
  this codebase.
