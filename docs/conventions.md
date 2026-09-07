# Engineering Conventions (frozen for Milestone 1)

These conventions are frozen BEFORE any sensor model or geometry code was
written, and every module in `src/adcs_sensor_trade/` follows them. If a
future milestone needs to change one of these, it must be called out
explicitly as a breaking convention change.

## 1. Frames

- **Inertial frame** (`_i` / `_inertial` suffix): a fixed, non-rotating
  reference frame (e.g. ECI-like). All "known" vectors that are inputs to
  the sensor models (Sun direction, magnetic field direction) are specified
  in this frame.
- **Body frame** (`_b` / `_body` suffix): the spacecraft-fixed frame in
  which sensors report their measurements. The body +X axis is used as the
  nominal sensor boresight direction for FOV-constrained sensors (Sun
  sensor) unless a sensor explicitly models a different mounting.
- No orbital/attitude dynamics propagation is modeled in Milestone 1 — all
  analysis is at a single, instantaneous attitude / geometry snapshot (or a
  deterministic sweep over such snapshots).

## 2. Unit vectors

All measurement vectors (Sun direction, magnetic field direction, TRIAD
basis vectors) are **unit vectors** (`||v|| = 1`) unless explicitly labeled
otherwise (e.g. a magnetometer's raw vector output, which carries physical
field magnitude, bias, and scale-factor error in Tesla).

## 3. Angle units

**Radians internally, everywhere in code.** Degrees and arcseconds are
permitted ONLY:
- as clearly-named keyword arguments / constructors at the sensor
  configuration boundary (e.g. `StarTrackerConfig.from_arcsec(...)`,
  `*_deg` properties), and
- at plotting / report-generation boundaries (`scripts/verify_sensor_geometry.py`,
  axis labels, printed reports).

`adcs_sensor_trade.units` is the single source of truth for all
deg/rad/arcsec conversions; no ad hoc `* pi / 180` conversions appear
elsewhere in the codebase.

## 4. Right-handed cross products

`adcs_sensor_trade.rotations.skew(v)` is defined so that
`skew(v) @ w == np.cross(v, w)` for all `w` — the standard right-handed
cross-product (skew-symmetric) matrix. This is verified directly in
`tests/test_rotations.py::test_skew_matches_cross_product`.

## 5. Measurement-vector convention

- Vectors are plain 1-D numpy arrays of length 3 (treated as columns when
  used in matrix products, e.g. `A @ v`).
- Naming: `<quantity>_inertial` or `<quantity>_i` for inertial-frame
  vectors, `<quantity>_body` or `<quantity>_b` for body-frame vectors.
- A DCM named `A` (or `dcm_body_from_inertial`) maps inertial-frame
  components to body-frame components: `v_body = A @ v_inertial`.

## 6. Attitude rotation-vector parametrization

A rotation vector `phi` [rad] parametrizes the inertial-to-body DCM as the
standard Rodrigues **active**-rotation matrix:

```
A(phi) = scipy.spatial.transform.Rotation.from_rotvec(phi).as_matrix()
v_body = A(phi) @ v_inertial
```

This is a definitional choice (not a physical derivation) that fixes the
sign of everything downstream. It is documented in full, with the
derivation of the small-angle measurement Jacobian, in
`src/adcs_sensor_trade/rotations.py`.

## 7. Attitude-error interpretation

We use a **multiplicative, left-composed** small-angle attitude-error
model, standard in spacecraft attitude estimation (e.g. Markley & Crassidis,
*Fundamentals of Spacecraft Attitude Determination and Control*):

```
A(phi_true) ≈ R(delta_theta) @ A(phi_est)      for small delta_theta
R(delta_theta) ≈ I + skew(delta_theta)
```

`delta_theta` [rad] is a **small-angle rotation vector correction** to be
added on top of a current attitude estimate to move it toward the true
attitude. Under this convention, the sensitivity of a body-frame unit
vector measurement `v_body = A(phi) @ v_inertial` to `delta_theta` is:

```
H_v := d(v_body)/d(delta_theta) = -skew(v_body)
```

This is the measurement Jacobian used everywhere in `geometry.py` and
`information.py`. It is verified against an independent central finite
difference of the *exact* (non-linearized) rotation in
`tests/test_rotations.py::test_measurement_jacobian_matches_finite_difference`
— the sign convention is checked numerically, not merely asserted.

A **"1-sigma attitude-error proxy"** anywhere in this codebase means: the
standard deviation of one component of the small-angle attitude-error
rotation vector `delta_theta`, as predicted by the local information/
covariance proxy of `information.py`. It is NOT a full nonlinear attitude
uncertainty and is only meaningful for reasonably well-conditioned
geometry (see `docs/geometry_and_information.md`).

## 8. Deterministic vs. noisy quantities

- Everything in `geometry.py`, `information.py`, and `sweep.py` is
  **deterministic** — no randomness, given fixed inputs.
- Sensor `*.measure(...)` methods accept an **optional** `rng:
  numpy.random.Generator` argument. If `rng is None`, or the configured
  noise sigma is `0.0`, the measurement is **deterministic** (no noise
  added) — this is how Milestone 1's deterministic tests and the geometry
  sweep script disable noise. If an `rng` is supplied and sigma > 0, noise
  is drawn from that generator only.
- **No sensor or analysis function in this package ever calls
  `numpy.random` global functions** (e.g. `np.random.randn`). Every
  stochastic call takes an explicit `numpy.random.Generator` instance, so
  Monte Carlo trials (a later milestone) can use independent, reproducible
  streams. This is checked implicitly by the reproducibility tests (same
  seed → identical output).

## 9. SI units

All physical quantities are SI throughout the codebase: meters, seconds,
radians, tesla. The only exception is the deg/arcsec convenience layer
described in §3, which stays at the configuration/reporting boundary.
