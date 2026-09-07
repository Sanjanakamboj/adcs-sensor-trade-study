# Sensor Models — Milestone 1

This document records the representative baseline parameters used for each
sensor model and why they were chosen. **All numeric values below are
ILLUSTRATIVE / representative of a small-to-mid-class spacecraft sensor
suite — they are NOT vendor datasheet values for any specific product**,
and should not be quoted as such outside this portfolio project.

## Star tracker (`sensors/star_tracker.py`)

| Parameter | Value | Rationale |
|---|---|---|
| 1-sigma per-axis attitude noise | 30 arcsec | Squarely in the "tens of arcsec" class commonly quoted for small/mid-class star trackers (roughly 5–60 arcsec 1-sigma across the market segment); 30 arcsec is a round, representative mid-point. |
| Update rate | 2 Hz | Representative of the common 1–5 Hz update-rate class for this sensor category. |
| Bright-source exclusion half-angle | 30 deg | Simple abstraction of a Sun/Earth-limb keep-out cone; real baffle designs vary (often 20–40+ deg), 30 deg is a round representative value. |
| Dropout probability | 0.0 (configurable) | Nominal case has no random dropout; the parameter exists so later Monte Carlo work (deferred) can model transient star-ID failures. |

**Modeling choice**: the star tracker outputs a **full 3-axis attitude**
(quaternion/DCM), not a single line-of-sight vector, because internally it
triangulates the positions of many catalog stars. Its instantaneous
information content is modeled as **isotropic**, `J_star = (1/sigma_star^2)
* I_3` (see `docs/geometry_and_information.md` and `information.py` for the
justification and the explicit statement that this is a simplification of
a real unit's internal star-triangulation covariance).

## Sun sensor (`sensors/sun_sensor.py`)

| Parameter | Value | Rationale |
|---|---|---|
| 1-sigma angular noise | 0.5 deg | Representative of a digital fine/coarse sun sensor (roughly 0.1–1 deg class); 0.5 deg is a round mid-point. |
| Field of view half-angle | ±60 deg | Representative single-head FOV; several heads are typically combined for near-4π coverage on a real spacecraft (not modeled here — Milestone 1 treats a single head). |
| Update rate | 5 Hz | Representative for a simple analog/digital sun sensor. |

**Key geometric fact demonstrated**: a single Sun-sensor unit-vector
measurement constrains only **2 of 3** rotational degrees of freedom —
rotation of the body about the Sun-line itself produces zero change in the
measured Sun direction. See `sun_sensor.demonstrate_single_vector_rank`
and `tests/test_sun_sensor.py::test_single_sun_vector_rank_is_two`.

## Magnetometer (`sensors/magnetometer.py`)

| Parameter | Value | Rationale |
|---|---|---|
| 1-sigma angular noise | 0.5 deg | Representative of a 3-axis fluxgate magnetometer's direction accuracy after basic calibration. |
| Update rate | 10 Hz | Representative of a typical fluxgate magnetometer sampling rate. |
| Bias | 0 (configurable 3-vector, T) | Nominal case has no bias; the parameter models a constant hard-iron-like offset for future calibration/estimation work. |
| Scale factor | 1.0 per axis (configurable) | Nominal case has unity scale; the parameter models per-axis gain error. |
| Representative inertial field magnitude | 3.0e-5 T (30 microtesla) | Representative order-of-magnitude LEO geomagnetic field strength (real field varies roughly 25–65 microtesla with position/epoch). |

**SIMPLIFICATION (explicitly deferred)**: Milestone 1 uses a single
constant representative inertial magnetic field vector. A real mission
would use a full geomagnetic field model (e.g. **IGRF**) evaluated at the
spacecraft's instantaneous orbital position and epoch — the field direction
in the *orbital* frame sweeps through a large range of angles over one
orbit, which is itself an important input to a real sensor-suite trade
(magnetometer/Sun geometry is not static). **A full IGRF-based, orbit-
resolved field model is deferred to a later milestone.**

**Key geometric fact demonstrated**: a single magnetometer unit-vector
measurement, like the Sun sensor, constrains only **2 of 3** rotational
degrees of freedom — rotation about the local field-line direction is
unobservable from one reading. See
`magnetometer.demonstrate_single_vector_rank` and
`tests/test_magnetometer.py::test_single_b_field_vector_rank_is_two`.

## RNG policy (applies to all three sensors)

Every `measure(...)` method takes an optional `rng: numpy.random.Generator`.
No sensor touches numpy's global random state. Passing `rng=None` (or
configuring `sigma_rad=0.0`) produces a fully deterministic measurement —
this is how Milestone 1's deterministic tests and geometry sweep disable
noise while keeping the exact same code path used for later (deferred)
Monte Carlo studies.
