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
src/adcs_sensor_trade/   installable package (rotations, geometry, information, sensors, sweep,
                          orbit_environment [M2], availability [M2])
tests/                   pytest suite (213 tests: 110 M1 + 103 M2)
scripts/                 verify_sensor_geometry.py (M1) and verify_availability.py (M2)
docs/                    conventions.md, sensor_models.md, geometry_and_information.md,
                          availability_methodology.md [M2]
results/                 generated reports (m1_report.txt, m2_report.txt) and figures (fig*.png)
```

## Setup & verification commands

```bash
cd "ADCS sensor trade study"
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest -q                                   # run the full test suite (213 tests, all passing)
python3 scripts/verify_sensor_geometry.py   # generate results/m1_report.txt and results/fig1-5*.png
python3 scripts/verify_availability.py      # generate results/m2_report.txt and results/fig6-11*.png
```

All commands are deterministic (fixed random seeds; nothing in either
analysis script or in `orbit_environment.py`/`availability.py` uses
randomness at all - every sensor `.measure()` call is invoked with
`rng=None`) — running any of them twice produces identical numeric output
and byte-identical figures.

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

## Milestone 2 scope (added in this repository state)

**Milestone 2 adds orbit/time-based OPERATIONAL AVAILABILITY analysis on
top of the Milestone 1 results above.** This is a distinct kind of result
from Milestone 1 and the two must not be conflated:

- **Milestone 1 results** (above): deterministic, INSTANTANEOUS geometry
  snapshots - "given these two vectors right now, how good is the
  attitude information?" No time, no orbit, no eclipse.
- **Milestone 2 results** (below): deterministic, ORBIT/TIME-BASED
  availability and update-cadence analysis over one representative orbit -
  "what fraction of an orbit can each architecture actually observe
  attitude, and how often?" Still fully deterministic and reproducible,
  but built on an explicitly ILLUSTRATIVE orbit/eclipse/attitude/magnetic-
  field model (see `docs/availability_methodology.md`) - **not** a flight
  prediction for any real mission.

New code: `src/adcs_sensor_trade/orbit_environment.py` (simplified LEO
orbit/eclipse/attitude/tilted-dipole-field model) and
`src/adcs_sensor_trade/availability.py` (availability timeline, outage-
interval detection, asynchronous update-cadence, sensitivity studies),
reusing the Milestone 1 sensor models and geometry/information machinery
unchanged. New tests (103 tests: `tests/test_orbit_environment.py`,
`tests/test_availability.py`) bring the total suite to **213 tests**. New
analysis script: `scripts/verify_availability.py`, generating
`results/m2_report.txt` and figures `results/fig6*.png`-`fig11*.png`.

### Representative Milestone 2 assumptions (illustrative)

| Parameter | Value |
|---|---|
| Orbit altitude / period | 600 km circular / 5801.2 s (96.69 min) |
| Beta angle | 20 deg |
| Eclipse fraction (closed-form, cylindrical shadow) | 35.79% of orbit |
| Attitude profile | idealized nadir-pointing (LVLH) |
| Star-tracker mounting / exclusion half-angle | anti-nadir (-Z) boresight / 30 deg |
| Sun-sensor FOV half-angle | 60 deg |
| Magnetic field model | tilted dipole DIRECTION (11.5 deg tilt), constant 3.0e-5 T magnitude |
| Update rates | star tracker 2 Hz, Sun sensor 5 Hz, magnetometer 10 Hz |
| Sun+mag synchronization window | 100 ms (half the slower sensor's period) |
| Conditioning threshold ("well-conditioned") | cond(J) <= 10 (worst/best-axis error ratio <= 3.16x) |

### Key Milestone 2 numerical findings (actual computed values)

(Full detail in `docs/availability_methodology.md` and
`results/m2_report.txt`.)

- **Availability over one representative orbit**: star tracker available
  **87.30%** of the orbit (longest outage 369 s / 6.15 min, caused by Sun
  exclusion at local noon); Sun sensor available **23.18%** (longest outage
  3939 s / 65.65 min, caused by eclipse); magnetometer **100%** (modeled as
  always available - see methodology doc); Sun+mag full-attitude available
  **23.18%** (tracks the Sun sensor, its bottleneck; longest outage 3939 s,
  same eclipse root cause).
- **Quality while geometrically available**: of the orbit time where
  Sun+mag vectors are usable AND full-rank, **0.00%** is poorly conditioned
  (cond(J) > 10) at the baseline beta = 20 deg / FOV = 60 deg - Sun-mag
  separation stays in a healthy 88-121 deg range throughout the visible
  window in this configuration. Condition number while available: median
  1.90, P95 2.05, max 2.05.
- **Update cadence**: star tracker delivers **10,130 usable full-attitude
  updates/orbit** at its own 2 Hz rate (1.75 Hz effective, since exclusion
  blocks ~12.7% of samples); Sun+mag delivers **6,723 usable full-attitude
  updates/orbit** (1.159 Hz effective) given asynchronous 5 Hz/10 Hz
  sampling and the 100 ms sync window.
- **Magnetometer rate does NOT matter (by construction)**: raising the
  magnetometer's rate from 5 to 50 Hz leaves Sun+mag usable updates/orbit
  exactly flat at 6,723 - a magnetometer measurement alone is rank-2, so
  the Sun sensor's own availability/rate remains the bottleneck.
- **Sensitivity checks all confirmed monotonic, matching physical
  expectation**: star-tracker availability falls monotonically from 100%
  (5 deg exclusion) to 50% (90 deg exclusion); Sun-sensor/Sun+mag
  availability rises monotonically from 0% (15 deg FOV) to 64.2% (179 deg
  FOV); eclipse fraction falls monotonically from 36.7% (beta=0) to 0%
  (beta >= ~66 deg, the closed-form beta_star cutoff for 600 km altitude).
  One defect was found and fixed during development: an earlier fixed-
  inertial-B-field model made the Sun-mag separation angle constant over
  the orbit (a same-DCM-rotates-both-vectors modeling bug) - replaced with
  a tilted-dipole direction model (see `docs/availability_methodology.md`
  Section 1 for the full account).
- **Which architecture has stronger operational availability/continuity**
  (availability + quality ONLY, explicitly NOT mass/power/cost): the
  **star tracker** - 87.30% available with a 6.15-minute longest outage
  driven solely by Sun exclusion, versus Sun+mag at 23.18% available with a
  65.65-minute longest outage driven by eclipse and Sun-sensor FOV
  geometry.

### Strongest Milestone 2 figures

- `results/fig6_availability_timeline.png` - one-orbit availability
  timeline for all four architectures side by side.
- `results/fig8_architecture_comparison.png` - availability-fraction and
  longest-outage bar-chart comparison.
- `results/fig11_update_rate_cadence_tradeoff.png` - usable full-attitude
  update cadence vs. sensor update rate, showing the flat magnetometer-rate
  curve directly.

### Milestone 2 also explicitly does NOT

Build a weighted trade matrix, model mass/power/cost, run a full
gyro-propagated MEKF/EKF, simulate closed-loop ADCS control, or make any
hardware-qualification claim. See `docs/availability_methodology.md`
Section 6 and `results/m2_report.txt` Q9 for the full limitations list.

## Recommended Milestone 3 scope

Mass/power/cost modeling for both candidate architectures, and the
eventual weighted trade matrix combining Milestone 1 (instantaneous
geometry/accuracy), Milestone 2 (operational availability/continuity), and
mass/power/cost into a final, explicit sensor-suite recommendation. A full
gyro-propagated estimator (MEKF/EKF) performance study and closed-loop
control-bandwidth analysis remain candidates for further deferral beyond
Milestone 3 if scope requires it.
