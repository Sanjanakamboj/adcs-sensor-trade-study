# ADCS-06 — ADCS Sensor Trade Study

## 1. Problem / objective

Small satellites need to know their own 3-axis orientation ("attitude")
to point solar panels, antennas, and payloads. Two common
attitude-sensing architectures compete for this job on a smallsat:

1. **A star tracker** — a camera that identifies patterns of stars and
   outputs a full, precise 3-axis attitude solution directly.
2. **A Sun sensor + magnetometer suite** — two simple, cheap, low-power
   sensors that each measure one direction (to the Sun, and along the
   local magnetic field); combined, two non-collinear directions can
   also recover a full 3-axis attitude, just less precisely and less
   often.

This project builds a from-scratch, fully reproducible engineering study
— sensor physics, orbit/availability simulation, and a weighted decision
trade — to answer: **given modeled accuracy, operational availability,
illustrative spacecraft resource cost, and implementation complexity,
which architecture should a representative smallsat use, and why?**

## 2. Final recommendation

**For a representative smallsat with balanced (baseline) mission
priorities, the star tracker is recommended as the primary 3-axis
attitude-determination sensor.** It is the only architecture that passes
every representative hard requirement evaluated here, and it scores
higher on the baseline weighted trade matrix (**0.764 vs. 0.664**) —
driven mainly by its large operational-availability/continuity advantage,
not by its raw instantaneous-accuracy advantage alone.

**This recommendation is conditional, not absolute.** It flips to the Sun
sensor + magnetometer suite for a genuinely mass/power-constrained or
cost/complexity-constrained mission (2 of the 4 alternative named
priority scenarios studied here already flip the winner), and a Monte
Carlo sweep over 20,000 random mission-priority weightings favors the
star tracker only **56.3%** of the time (43.7% favor Sun+mag) — a
mission-priority-dependent result, not an overwhelming one. See Section 7
below and `docs/trade_study_methodology.md` for the full, honest
discussion of that sensitivity.

**Qualitative caveat that applies to both architectures:** neither this
study nor any part of this codebase simulates a gyroscope or an attitude
filter (EKF/MEKF). Between absolute updates, a real spacecraft would
propagate attitude using a gyro; Sun+mag's much longer and more frequent
outages place materially more burden on that (unmodeled) gyro's drift
performance than the star tracker's short, rare outages. This is a real
system-level consideration the numeric score does not capture.

## 3. Milestone 1 — instantaneous geometry & information quality

*Full detail: `docs/geometry_and_information.md`, `docs/sensor_models.md`,
`results/m1_report.txt`, figures `results/fig1*.png`-`fig5*.png`.*

Deterministic, **instantaneous** (no orbit, no time) attitude-information
geometry, built from small-angle measurement-Jacobian theory:

- A single Sun-sensor or magnetometer vector measurement is **rank 2 of
  3** — it cannot resolve rotation about its own line-of-sight/field-line
  direction. A star tracker outputs full 3-axis attitude directly
  (triangulating many stars), modeled as isotropic and geometry-
  independent.
- Two non-collinear vectors (Sun + magnetic field) recover full **rank-3**
  attitude information; conditioning is best near 90 deg separation and
  singular exactly at 0/180 deg.
- At a healthy 90 deg Sun-magnetic separation: Sun+mag condition number =
  **2.00**, worst-axis 1-sigma error proxy = **0.50 deg**, vs. the
  (isotropic) star-tracker reference of condition number = **1.00**,
  worst-axis error = **30 arcsec (0.0083 deg)**.
- Sun+mag conditioning degrades sharply approaching collinearity — e.g.
  condition number **≈ 525.6** at 5 deg/175 deg separation.

**Milestone 1 explicitly made no final sensor-suite recommendation** —
resource, cost, and operational-availability questions were still open.

## 4. Milestone 2 — operational availability & continuity

*Full detail: `docs/availability_methodology.md`, `results/m2_report.txt`,
figures `results/fig6*.png`-`fig11*.png`.*

Deterministic, **orbit/time-based** availability and update-cadence
simulation over a representative 600 km circular LEO orbit (period 96.69
min, beta = 20 deg, eclipse fraction 35.79%):

| Architecture | Availability | Longest outage | Usable updates/orbit (effective cadence) |
|---|---|---|---|
| Star tracker | **87.30%** | 369 s (Sun exclusion) | 10,130 (1.7462 Hz) |
| Sun sensor (alone, rank-2) | 23.18% | 3939 s (eclipse) | n/a (never full attitude alone) |
| Magnetometer (alone, modeled always-available, rank-2) | 100% | 0 s | n/a (never full attitude alone) |
| Sun + magnetometer (full attitude) | **23.18%** | 3939 s (eclipse) | 6,723 (1.1589 Hz) |

Sun+mag conditioning while geometrically available: median cond(J) =
1.903, P95 = 2.045, max = 2.047; **0.00%** of that time is poorly
conditioned. A faster magnetometer alone (5 -> 50 Hz) does **not** raise
Sun+mag's usable update cadence — by construction a magnetometer alone is
rank-2, so the Sun sensor's own eclipse/FOV-limited availability remains
the bottleneck. **Availability/continuity alone favors the star tracker.**
**Milestone 2 still made no final recommendation** — mass, power, cost,
and implementation-complexity questions remained open.

## 5. Milestone 3 — resource, cost/complexity & the weighted decision trade

*Full detail: `docs/trade_study_methodology.md`, `results/m3_report.txt`,
figures `results/fig12*.png`-`fig18*.png`, CSVs `results/m3_*.csv`.*

Milestone 3 adds an illustrative spacecraft **resource model** (mass,
power, volume proxy — `resource_model.py`), a normalized ordinal **cost/
complexity model** (`cost_complexity_model.py`, NO fabricated pricing), a
transparent **normalization** layer (`normalization.py`), a reusable
**weighted trade-matrix engine** (`trade_matrix.py`), a **Monte Carlo
weight-sensitivity/robustness study**, a numeric **break-even analysis**,
and a set of representative **hard requirements** (`requirements.py`) —
then combines all of it with the real M1/M2 numbers above (re-derived,
never hand-typed — see `trade_metrics.py`) into the final trade matrix and
recommendation in Section 2.

Illustrative resource totals (see `docs/trade_study_methodology.md` for
full component-level rationale; Sun+mag totals are DERIVED by summing
individual Sun-sensor x2 + magnetometer x1 component profiles, never a
separate lump number):

| Architecture | Mass [kg] | Avg power [W] | Peak power [W] | Cost burden [1-5] | Complexity burden [1-5] |
|---|---|---|---|---|---|
| Star tracker | 0.450 | 1.500 | 2.500 | 4.0 | 3.2 |
| Sun + magnetometer | 0.250 | 0.700 | 1.000 | 1.0 | 2.6 |

## 6. Final trade matrix

| Architecture | Accuracy proxy [deg] | Conditioning cond(J) | Availability | Longest outage | Cadence [Hz] | Mass [kg] | Avg power [W] | Cost burden | Complexity burden | Hard requirements | Normalized score (baseline) | Weighted score (baseline) | Recommendation status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Star tracker | 0.0083 | 1.00 | 87.30% | 369 s | 1.7462 | 0.450 | 1.500 | 4.0 | 3.2 | **PASS (4/4)** | see `results/m3_normalized_utilities.csv` | **0.7636** | **Baseline recommendation** |
| Sun + magnetometer | 0.6729 | 1.903 | 23.18% | 3939 s | 1.1589 | 0.250 | 0.700 | 1.0 | 2.6 | FAIL (2/4: availability, outage) | see `results/m3_normalized_utilities.csv` | 0.6635 | Conditional alternative |

(Full raw/normalized/weighted tables: `results/m3_raw_metrics.csv`,
`results/m3_normalized_utilities.csv`, `results/m3_weighted_scores.csv`.)

## 7. Decision robustness summary

| Weighting scenario | Star tracker score | Sun+mag score | Winner |
|---|---|---|---|
| Baseline | 0.7636 | 0.6635 | Star tracker |
| Accuracy priority | 0.8564 | 0.6490 | Star tracker |
| Resource constrained | 0.7730 | 0.7773 | **Sun+mag (flips)** |
| Availability priority | 0.7872 | 0.5610 | Star tracker |
| Cost / complexity priority | 0.5730 | 0.7462 | **Sun+mag (flips)** |

**2 of the 4** non-baseline named scenarios flip the recommendation.
A Monte Carlo sweep of 20,000 weight vectors sampled uniformly over the
7-category priority simplex (`numpy.random.default_rng(seed=42)`,
`rng.dirichlet`, fully reproducible) favors the star tracker in
**56.29%** of samples and Sun+mag in **43.71%**. This is reported here,
honestly, as **mission-priority-dependent, not robust** — see
`docs/trade_study_methodology.md` Section 8 for the full discussion and
`results/fig15_scenario_sensitivity.png` / `fig16_monte_carlo_score_difference.png`.

Break-even findings (numeric root-finding, baseline weighting): Sun+mag
would need availability to rise to **~73.2%** (from 23.2%), OR the star
tracker's mass to rise above **~2.45 kg** (from 0.45 kg), OR its average
power to rise above **~4.84 W** (from 1.50 W), OR the availability-
category weight to fall below **~5.2%** (from the baseline 20%) for the
decision to flip. Improving Sun+mag's raw accuracy ALONE does **not**
flip the decision in this model — an honest null result, not forced.

## 8. Reproduction instructions

```bash
cd "ADCS sensor trade study"
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest -q                                   # full test suite: 344 tests, all passing
python3 scripts/verify_sensor_geometry.py   # M1: results/m1_report.txt, results/fig1-5*.png
python3 scripts/verify_availability.py      # M2: results/m2_report.txt, results/fig6-11*.png
python3 scripts/run_trade_study.py          # M3: results/m3_report.txt, results/fig12-18*.png, results/m3_*.csv
```

All three scripts are fully deterministic — no randomness is used in
`orbit_environment.py`/`availability.py` (every sensor `.measure()` call
uses `rng=None`), and Milestone 3's one Monte Carlo study uses a fixed
seed (42) — running any script twice produces bit-identical numeric
output.

## 9. Limitations (all three milestones)

- **Sensor/orbit/environment models are illustrative simplifications**,
  not flight-fidelity or vendor-specific: isotropic star-tracker noise;
  constant-magnitude tilted-dipole magnetic-field DIRECTION (not a full
  IGRF model); idealized nadir-pointing attitude profile; cylindrical
  (no-penumbra) eclipse shadow; circular two-body orbit (no J2/drag).
- **No full gyro-propagated estimator (MEKF/EKF)** anywhere in this
  project. Outages are treated as instantaneous measurement gaps, not
  propagated through a real attitude filter with process noise and gyro
  drift. This is discussed only qualitatively (Section 2 above,
  Q11 in `results/m3_report.txt`) — never simulated.
- **No closed-loop ADCS/control-bandwidth analysis.** "Cadence" here
  means usable ABSOLUTE full-attitude update cadence, not a control-loop
  rate requirement.
- **Resource and cost/complexity numbers are illustrative engineering
  assumptions**, not vendor datasheets, procurement quotes, or
  flight-heritage claims — see `docs/trade_study_methodology.md` for
  every value's rationale.
- **No vendor/product selection, fabricated commercial pricing, or
  flight-qualification claim** is made anywhere in this project.
- The weighted trade matrix answers "which architecture better serves a
  STATED set of mission priorities" — Sections 6-7 show explicitly that
  the answer is priority-dependent, not universal.

---

## Appendix: repository layout

```
src/adcs_sensor_trade/   installable package
                         M1: rotations, geometry, information, sensors/, sweep
                         M2: orbit_environment, availability
                         M3: resource_model, cost_complexity_model, trade_metrics,
                             normalization, trade_matrix, requirements
tests/                   pytest suite (344 tests: 213 M1+M2, 131 M3)
scripts/                 verify_sensor_geometry.py (M1), verify_availability.py (M2),
                         run_trade_study.py (M3)
docs/                    conventions.md, sensor_models.md, geometry_and_information.md,
                         availability_methodology.md, trade_study_methodology.md
results/                 generated reports (m1/m2/m3_report.txt), figures (fig*.png),
                         and M3 CSV tables (m3_*.csv)
```

## Appendix: representative sensor assumptions (illustrative, NOT vendor specifications)

| Sensor | Key parameters |
|---|---|
| Star tracker | 1-sigma = 30 arcsec/axis, 2 Hz update rate, 30 deg bright-source exclusion half-angle |
| Sun sensor | 1-sigma = 0.5 deg, +/-60 deg FOV half-angle, 5 Hz update rate |
| Magnetometer | 1-sigma = 0.5 deg, 10 Hz update rate, nominal zero bias/unity scale, representative LEO field magnitude 3.0e-5 T |

See `docs/sensor_models.md` for full rationale, `docs/conventions.md` for
frozen engineering conventions, and `docs/trade_study_methodology.md` for
the Milestone 3 resource/cost/complexity assumptions.
