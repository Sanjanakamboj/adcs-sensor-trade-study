# ADCS-06 — ADCS Sensor Trade Study

Portfolio engineering project comparing two spacecraft attitude-
determination sensor architectures:

1. **Star tracker + gyro**
2. **Sun sensor + magnetometer + gyro**

The final deliverable, across multiple future milestones, is a
quantitative trade matrix and sensor-suite recommendation.

## Milestone 1 scope (THIS repository state)

Milestone 1 builds and verifies:

- Sensor **physics models** for a star tracker, a coarse/fine sun sensor,
  and a magnetometer (`src/adcs_sensor_trade/sensors/`), with configurable
  noise, update rate, FOV/availability, and (where applicable) bias/scale
  errors.
- **Instantaneous two-vector attitude-information geometry**
  (`geometry.py`, `information.py`, `sweep.py`): rank/conditioning of the
  attitude-information Jacobian, a local information/covariance proxy, and
  a deterministic Sun–magnetic separation-angle sweep.
- A full **pytest** suite (110 tests) and an analysis script
  (`scripts/verify_sensor_geometry.py`) that generates a numerical report
  and figures under `results/`.

**Milestone 1 explicitly does NOT build a weighted trade matrix and does
NOT make a final sensor-suite recommendation.** That is deferred to a
later milestone (see "Recommended next milestone" below).

## Representative assumptions (illustrative, NOT vendor specifications)

| Sensor | Key parameters |
|---|---|
| Star tracker | 1σ = 30 arcsec/axis, 2 Hz update rate, 30 deg bright-source exclusion half-angle |
| Sun sensor | 1σ = 0.5 deg, ±60 deg FOV half-angle, 5 Hz update rate |
| Magnetometer | 1σ = 0.5 deg, 10 Hz update rate, nominal zero bias/unity scale, representative LEO field magnitude 3.0e-5 T |

See `docs/sensor_models.md` for full rationale, and `docs/conventions.md`
for the frozen engineering conventions (frames, units, sign conventions,
RNG policy) used throughout the codebase.

## Repository layout

```
src/adcs_sensor_trade/   installable package (rotations, geometry, information, sensors, sweep)
tests/                   pytest suite (110 tests)
scripts/                 scripts/verify_sensor_geometry.py — the M1 analysis/report script
docs/                    conventions.md, sensor_models.md, geometry_and_information.md
results/                 generated report (m1_report.txt) and figures (fig*.png)
```

## Setup & verification commands

```bash
cd "ADCS sensor trade study"
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest -q                              # run the test suite (110 tests, all passing)
python3 scripts/verify_sensor_geometry.py   # generate results/m1_report.txt and results/fig*.png
```

Both commands are deterministic (fixed random seed `20240607` in the
script; the test suite is seeded per-test) — running either twice produces
identical numeric output and byte-identical figures.

## Key Milestone 1 numerical findings

(Full detail in `docs/geometry_and_information.md` and
`results/m1_report.txt`.)

- A single Sun-sensor or magnetometer vector measurement is **rank 2 of
  3** — it cannot instantaneously resolve rotation about its own
  line-of-sight/field-line direction.
- Two non-collinear vectors recover **full rank-3** attitude information;
  conditioning is best near 90 deg separation and becomes singular exactly
  at 0/180 deg (collinear/anti-collinear).
- At a healthy 90 deg Sun–magnetic separation: Sun+mag condition number =
  2.00, worst-axis 1σ attitude-error proxy = 0.50 deg, vs. the (isotropic,
  geometry-independent) star-tracker reference of condition number = 1.00,
  worst-axis error = 30 arcsec (0.0083 deg).
- Sun+mag conditioning degrades sharply approaching collinearity — e.g.
  condition number ≈ 525.6 at 5 deg / 175 deg separation, vs. 2.00 at 90
  deg — while the simplified isotropic star-tracker model is, by
  construction, independent of this angle.
- Scaling sanity checks confirm the expected `1/sigma^2` information
  scaling and `sigma` covariance-proxy scaling.

## Written engineering conclusions

See `docs/geometry_and_information.md` for the full discussion. In brief:
a star tracker gives full 3-axis attitude directly (by triangulating many
stars) and is modeled as geometry-independent (isotropic); a single Sun
sensor or magnetometer vector is fundamentally rank-2 and needs a second,
non-collinear vector to recover full attitude, with accuracy that is
strongly geometry-dependent (best near 90 deg separation, singular at
0/180 deg).

**This analysis does not yet say anything about:** mass, power, cost,
eclipse/Sun-exclusion availability duty cycle, blinding/stray light,
update-rate effects on closed-loop control bandwidth, or full estimator
(EKF) performance with gyro propagation and measurement history.

## Final architecture selection is DEFERRED

**Milestone 1 makes no final sensor-suite recommendation.** The two
architectures are only compared here on instantaneous sensor-physics and
attitude-information geometry grounds — this is a necessary but explicitly
partial input to the eventual trade.

## Recommended Milestone 2 scope

Availability (eclipse/Sun-exclusion duty cycle, dropout), update-rate
effects, mass/power/cost modeling, and full estimator-performance
(EKF-with-gyro-propagation) trade study, building toward the eventual
quantitative sensor-suite trade matrix and recommendation.
