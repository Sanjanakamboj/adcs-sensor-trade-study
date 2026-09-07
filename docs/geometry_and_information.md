# Two-Vector Attitude Geometry & Instantaneous Information — Milestone 1

This document explains the geometry/information analysis implemented in
`src/adcs_sensor_trade/geometry.py`, `information.py`, and `sweep.py`, and
records the key numerical results produced by
`scripts/verify_sensor_geometry.py` (full report at
`results/m1_report.txt`, figures at `results/fig*.png`).

## Why one vector is not enough

A body-frame unit-vector measurement `v_b = A(phi) @ v_i` has small-angle
measurement Jacobian `H_v = -skew(v_b)` (see `docs/conventions.md` §7 for
the sign-convention derivation). A skew-symmetric matrix built from a
single nonzero vector is **always exactly rank 2** — its null space is the
line spanned by the vector itself, because `skew(v) @ v = v × v = 0`.
Physically: rotating the body about the vector's own direction produces no
change in the measured direction, so that rotational degree of freedom is
instantaneously unobservable from one such reading. This holds identically
for a Sun sensor and a magnetometer (both are single unit-vector sensors).

A star tracker, by contrast, does not report a single line-of-sight vector
— it triangulates many catalog stars simultaneously and solves directly
for a full attitude quaternion/DCM. We model its instantaneous information
as isotropic, `J_star = (1/sigma_star^2) I_3` — full rank 3 by
construction (a documented simplification of its own internal
star-triangulation covariance; see `docs/sensor_models.md`).

## Two vectors: when do they recover full attitude?

Stacking two vectors' Jacobians gives a 6x3 matrix `H = [H_1; H_2]`. Its
rank is 3 (full attitude information) whenever the two vectors are
non-collinear, and exactly 2 (singular) when they are collinear or
anti-collinear. `geometry.geometry_rank` computes this via SVD with a
numerical tolerance (`DEFAULT_RANK_TOL = 1e-8`), and
`geometry.condition_number` reports the resulting conditioning
(`sigma_max / sigma_min` of the stacked Jacobian).

## Key numerical results (fixed seed 20240607; see `results/m1_report.txt`)

Representative baseline assumptions used (see `docs/sensor_models.md` for
full rationale): star tracker sigma = 30 arcsec/axis @ 2 Hz; Sun sensor
sigma = 0.5 deg, FOV half-angle 60 deg, @ 5 Hz; magnetometer sigma = 0.5
deg @ 10 Hz, zero nominal bias/scale error; inertial Sun vector `[1,0,0]`;
representative inertial B-field vector `[0,0,3e-5] T` (30 microtesla, LEO
order-of-magnitude).

**Single-vector rank check:**
- rank(single Sun vector) = 2
- rank(single magnetometer vector) = 2
- rank(two orthogonal vectors) = 3

**Healthy geometry (90 deg Sun–magnetic separation):**

| | Star tracker (isotropic) | Sun + magnetometer (90 deg) |
|---|---|---|
| Condition number of J | 1.000 | 2.000 |
| RMS attitude-error proxy | 30.00 arcsec (0.008333 deg) | 0.4564 deg |
| Worst-axis 1σ error proxy | 30.00 arcsec (0.008333 deg) | 0.5000 deg |

**Separation-angle sweep** (`sun-magnetic separation angle` vs. Sun+mag
condition number and worst-axis error proxy; star tracker is the flat
angle-independent reference under the isotropic model):

| Angle [deg] | cond(J) | Worst-axis error [deg] | Rank |
|---|---|---|---|
| 5 | 525.6 | 8.105 | 3 |
| 45 | 6.83 | 0.924 | 3 |
| 90 | 2.00 | 0.500 | 3 |
| 135 | 6.83 | 0.924 | 3 |
| 175 | 525.6 | 8.105 | 3 |

**Exact singularity check:**
- 0 deg: rank = 2, min singular value = 0.000e+00, cond(J) = inf
- 180 deg: rank = 2, min singular value ≈ 8.66e-17 (numerical zero), cond(J) = inf

**Scaling sanity checks:**
- Halving the star tracker's sigma multiplies its isotropic information by
  4.00 (matches the expected `1/sigma^2` scaling).
- Doubling the star tracker's sigma multiplies its worst-axis error proxy
  by 2.00 (matches expected `sigma` scaling of the covariance-proxy
  standard deviation).

## Figures (see `results/`)

1. `fig1_information_ellipsoids.png` — attitude-error covariance-proxy
   ellipsoids: a perfect sphere for the isotropic star-tracker model vs. a
   mildly oblate spheroid for Sun+mag at 90 deg separation.
2. `fig2_condition_number_vs_angle.png` — condition number of J vs.
   Sun–magnetic separation angle (log scale), U-shaped with minimum at 90
   deg, blowing up toward 0/180 deg; flat star-tracker reference line.
3. `fig3_worst_axis_error_vs_angle.png` — worst-axis 1σ attitude-error
   proxy [deg] vs. separation angle, same U-shape, log scale.
4. `fig4_rank_and_singular_value_vs_angle.png` — minimum singular value of
   the stacked measurement Jacobian and geometry rank vs. separation angle,
   showing the rank-3 plateau collapsing to rank 2 exactly at 0/180 deg.
5. `fig5_vector_geometry_3d.png` — 3D body-frame vector-geometry
   visualization for a healthy (90 deg) case vs. a near-degenerate (5 deg)
   case.

## Written engineering conclusions

- **Why can a star tracker provide full instantaneous 3-axis attitude
  information?** It triangulates many catalog stars simultaneously and
  solves directly for a full attitude, not a single line-of-sight vector;
  under our isotropic simplification, `J_star = (1/sigma^2) I_3` is full
  rank by construction.
- **Why can a single Sun sensor not?** Its measurement Jacobian
  `H = -skew(v_b)` is rank-2: rotation about the Sun-line itself produces
  no change in the measured direction.
- **Why can a single magnetometer not?** Same argument — it is also a
  single unit-vector measurement, so rotation about the local field-line
  direction is unobservable from one reading.
- **Under what geometry does Sun + magnetometer recover full attitude?**
  Whenever the two vectors are non-collinear (0 < separation < 180 deg),
  the stacked Jacobian reaches rank 3.
- **When does that architecture become poorly conditioned?** As the
  separation angle approaches 0 deg or 180 deg; it is exactly singular at
  0/180 deg.
- **Which architecture has more uniform instantaneous accuracy under these
  simplified assumptions?** The star tracker — its isotropic model is
  independent of geometry/attitude by construction, whereas Sun+mag varies
  strongly (over 100x condition-number range) with the instantaneous
  Sun-magnetic separation angle.
- **What this analysis does NOT yet say:** mass, power, cost, eclipse/Sun-
  exclusion availability duty cycle, blinding/stray light, the effect of
  update rate on real closed-loop control bandwidth, or full estimator
  (EKF) performance with gyro propagation and measurement history. All of
  these are explicitly **deferred to a later milestone**.

**No final sensor-suite recommendation is made in this milestone.**
