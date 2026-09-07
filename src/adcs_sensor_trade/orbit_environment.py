"""Simplified, deterministic LEO circular-orbit / environment model.

SCOPE AND HONESTY NOTE (read before using this module): this is a
representative, illustrative orbit/attitude/eclipse model built for a
Milestone 2 *availability* study. It is explicitly **NOT** high-fidelity
ephemeris. In particular:

- The orbit is a perfect circle, propagated with simple two-body Keplerian
  mean motion. No J2, drag, or other perturbations are modeled.
- The Sun direction is held FIXED in inertial space at a constant "beta
  angle" (angle between the Sun vector and the orbit plane) for the whole
  simulated span. Real beta angle drifts slowly (weeks) due to orbital
  precession and the Earth's motion around the Sun; over the 1-few-orbit
  spans analyzed here (~90-100 minutes each) that drift is negligible, but
  this is a simplification, not a claim of long-term validity.
- Eclipse is modeled with the standard circular-orbit CYLINDRICAL shadow
  approximation (no penumbra, no oblateness of the Earth's shadow). This is
  the same closed-form approximation used in introductory orbital mechanics
  texts (e.g. Vallado, Wertz) for quick eclipse-fraction estimates.
- The spacecraft attitude profile is a fixed, idealized NADIR-POINTING
  (local-vertical/local-horizontal, "LVLH") profile: body +Z points at nadir
  (Earth center), body +X points along the velocity vector, body +Y
  completes a right-handed triad (~orbit-normal direction). This is a
  common, simple, and explicit attitude assumption chosen purely to give
  the Sun- and magnetic-field vectors a well-defined, deterministic
  time-history in the body frame for the purposes of exclusion/FOV/rank
  geometry. No real ADCS control loop or attitude dynamics are modeled.
- The inertial magnetic field is modeled with a simple TILTED-DIPOLE
  DIRECTION model: a dipole axis fixed in inertial space, tilted by a
  representative ``dip_tilt_deg`` (default 11.5 deg, the commonly cited
  approximate tilt of Earth's magnetic dipole axis from its rotation axis)
  away from the orbit-normal direction, with field DIRECTION at orbital
  position ``r_hat`` given by the standard dipole formula
  ``3*(m . r_hat)*r_hat - m`` (normalized). Field MAGNITUDE is held at a
  single representative constant (as in Milestone 1's magnetometer model,
  since only direction matters for the rank/conditioning geometry used
  here). This reproduces the qualitatively-correct feature that a real
  geomagnetic field direction, unlike the fixed Sun direction, changes
  substantially over one orbit - but it is explicitly NOT a full
  IGRF/orbit-resolved geomagnetic field model (no field-magnitude
  variation, no higher-order multipoles, no true rotation-axis/orbit
  geometry distinction). A full IGRF-based field model remains explicitly
  DEFERRED (see ``sensors/magnetometer.py``).
- The star tracker is assumed boresighted along body -Z (anti-nadir, i.e.
  pointed at deep space, away from the Earth) - a common real-world mounting
  strategy specifically chosen to avoid Earth/Sun stray light. Its
  "bright-source exclusion" angle is therefore the angle between the
  anti-nadir direction and the Sun vector.

None of this is a substitute for SGP4/precise ephemeris propagation, and no
such propagation is implemented (or intended) here.

Frame/unit conventions match ``docs/conventions.md``: radians internally,
plain length-3 numpy arrays for vectors, ``A`` maps inertial to body
(``v_body = A @ v_inertial``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .rotations import normalize

# ---------------------------------------------------------------------------
# Physical constants (representative, standard values)
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM = 6378.137
MU_EARTH_KM3_S2 = 398600.4418  # standard gravitational parameter of Earth

# Representative default assumptions for the M2 availability study.
# THESE ARE ILLUSTRATIVE, NOT A SPECIFIC MISSION ORBIT.
DEFAULT_ALTITUDE_KM = 600.0
# Chosen strictly less than the star tracker's default 30 deg exclusion
# half-angle so the default configuration actually exercises a Sun-
# exclusion outage (see docs/availability_methodology.md): the minimum
# star-tracker/Sun angle over an orbit equals beta_angle_deg exactly (at
# local noon, phase = 0), so beta_angle_deg >= exclusion_half_angle would
# never trigger exclusion at all.
DEFAULT_BETA_ANGLE_DEG = 20.0
DEFAULT_B_FIELD_MAGNITUDE_TESLA = 3.0e-5
DEFAULT_DIP_TILT_DEG = 11.5  # representative Earth magnetic-dipole tilt from rotation axis


def orbital_period_s(altitude_km: float) -> float:
    """Circular-orbit Keplerian period [s] for the given altitude [km].

    Two-body, non-perturbed mean motion: T = 2*pi*sqrt(a^3 / mu).
    """
    if altitude_km <= 0:
        raise ValueError(f"altitude_km must be > 0, got {altitude_km}")
    a = EARTH_RADIUS_KM + altitude_km
    return float(2.0 * np.pi * np.sqrt(a**3 / MU_EARTH_KM3_S2))


def _validate_beta_deg(beta_angle_deg: float) -> None:
    if not (-90.0 <= beta_angle_deg <= 90.0):
        raise ValueError(f"beta_angle_deg must be in [-90, 90], got {beta_angle_deg}")


def eclipse_fraction(altitude_km: float, beta_angle_deg: float) -> float:
    """Fraction of one orbit (in [0, 1)) spent in the cylindrical Earth shadow.

    Standard circular-orbit cylindrical-shadow closed form (e.g. Wertz,
    *Spacecraft Attitude Determination and Control*, or Vallado): with orbit
    radius ``r = R_earth + altitude`` and beta angle ``beta``, let
    ``beta_star = asin(R_earth / r)`` be the beta angle above which the
    orbit never enters the (cylindrical) shadow at all. For ``|beta| <
    beta_star``::

        eclipse_fraction = (1/pi) * arccos( sqrt(r^2 - R_earth^2) / (r * cos(beta)) )

    and 0 otherwise. No penumbra, no oblateness.
    """
    if altitude_km <= 0:
        raise ValueError(f"altitude_km must be > 0, got {altitude_km}")
    _validate_beta_deg(beta_angle_deg)

    r = EARTH_RADIUS_KM + altitude_km
    beta = np.radians(beta_angle_deg)
    beta_star = np.arcsin(EARTH_RADIUS_KM / r)
    if abs(beta) >= beta_star:
        return 0.0
    ratio = np.sqrt(r**2 - EARTH_RADIUS_KM**2) / (r * np.cos(beta))
    ratio = float(np.clip(ratio, -1.0, 1.0))
    return float(np.arccos(ratio) / np.pi)


def is_in_eclipse(phase_rad: float, altitude_km: float, beta_angle_deg: float) -> bool:
    """True if orbital phase ``phase_rad`` (0 at "local noon") is in eclipse.

    The eclipse arc is modeled as centered on phase = pi ("local midnight",
    directly behind the Earth as seen from the Sun), with total angular
    width ``2*pi*eclipse_fraction``, symmetric about phase = pi. This
    follows directly from the geometry used to derive
    :func:`eclipse_fraction` (the satellite is in shadow only near the
    anti-Sun side of its orbit).
    """
    frac = eclipse_fraction(altitude_km, beta_angle_deg)
    if frac <= 0.0:
        return False
    half_width_rad = np.pi * frac
    phase = np.mod(phase_rad, 2.0 * np.pi)
    # Signed angular distance from phase = pi, wrapped into (-pi, pi].
    delta = np.angle(np.exp(1j * (phase - np.pi)))
    return bool(abs(delta) <= half_width_rad)


def sun_vector_inertial(beta_angle_deg: float) -> np.ndarray:
    """Unit Sun-direction vector in inertial coordinates for a given beta angle.

    Convention: the orbit plane is spanned by inertial X/Y (orbit normal =
    inertial +Z). The Sun vector is placed at angle ``beta_angle_deg`` out
    of the orbit plane, with its in-plane projection along inertial +X
    (this in-plane projection direction defines orbital phase = 0, "local
    noon", by construction).
    """
    _validate_beta_deg(beta_angle_deg)
    beta = np.radians(beta_angle_deg)
    return np.array([np.cos(beta), 0.0, np.sin(beta)])


def orbit_position_hat(phase_rad: float) -> np.ndarray:
    """Unit position vector (Earth center -> spacecraft) at the given orbital phase."""
    return np.array([np.cos(phase_rad), np.sin(phase_rad), 0.0])


def orbit_velocity_hat(phase_rad: float) -> np.ndarray:
    """Unit velocity direction (prograde) at the given orbital phase."""
    return np.array([-np.sin(phase_rad), np.cos(phase_rad), 0.0])


def attitude_dcm_nadir_pointing(phase_rad: float) -> np.ndarray:
    """Body-from-inertial DCM for the idealized nadir-pointing (LVLH) attitude profile.

    body +Z = nadir (Earth-pointing), body +X = velocity direction, body +Y
    completes a right-handed triad (~orbit-normal direction for a circular
    orbit). Returns A such that ``v_body = A @ v_inertial``.
    """
    r_hat = orbit_position_hat(phase_rad)
    v_hat = orbit_velocity_hat(phase_rad)
    z_b = -r_hat
    x_b = v_hat
    y_b = np.cross(z_b, x_b)
    return np.vstack([x_b, y_b, z_b])


def attitude_rotvec_nadir_pointing(phase_rad: float) -> np.ndarray:
    """Rotation-vector [rad] representation of :func:`attitude_dcm_nadir_pointing`.

    Provided so the M2 availability simulation can reuse the existing M1
    sensor ``.measure()`` APIs (which take a rotation vector, see
    ``rotations.py``) rather than re-implementing FOV/eclipse logic.
    """
    A = attitude_dcm_nadir_pointing(phase_rad)
    return Rotation.from_matrix(A).as_rotvec()


def star_tracker_bright_source_angle_rad(phase_rad: float, beta_angle_deg: float) -> float:
    """Angle [rad] between the star tracker boresight (anti-nadir) and the Sun.

    The star tracker is assumed mounted along body -Z (anti-nadir, deep-space
    pointing; see module docstring). The anti-nadir direction expressed in
    inertial coordinates is ``+orbit_position_hat(phase_rad)`` (since body
    +Z = -orbit_position_hat is nadir).
    """
    _validate_beta_deg(beta_angle_deg)
    r_hat = orbit_position_hat(phase_rad)
    sun_i = sun_vector_inertial(beta_angle_deg)
    cos_angle = np.clip(np.dot(r_hat, sun_i), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def dipole_axis_inertial(dip_tilt_deg: float = DEFAULT_DIP_TILT_DEG) -> np.ndarray:
    """Unit dipole-axis vector, tilted ``dip_tilt_deg`` from the orbit normal (+Z).

    The tilt is applied in the inertial X-Z plane purely as a simple,
    explicit, arbitrary-but-fixed geometric choice (see module docstring).
    """
    if not (0.0 <= dip_tilt_deg <= 90.0):
        raise ValueError(f"dip_tilt_deg must be in [0, 90], got {dip_tilt_deg}")
    tilt = np.radians(dip_tilt_deg)
    return np.array([np.sin(tilt), 0.0, np.cos(tilt)])


def b_field_direction_inertial(
    phase_rad: float, dip_tilt_deg: float = DEFAULT_DIP_TILT_DEG
) -> np.ndarray:
    """Unit inertial magnetic-field DIRECTION at orbital phase, via a tilted-dipole model.

    Standard magnetic-dipole field-direction formula ``3*(m.r_hat)*r_hat -
    m`` (normalized), evaluated at the orbital position ``r_hat(phase)``.
    This varies with orbital position (hence with time), unlike the fixed
    Sun direction - see module docstring for the model and its limits.
    """
    m_hat = dipole_axis_inertial(dip_tilt_deg)
    r_hat = orbit_position_hat(phase_rad)
    raw = 3.0 * np.dot(m_hat, r_hat) * r_hat - m_hat
    return normalize(raw)


def b_field_inertial_tesla(
    phase_rad: float,
    magnitude_tesla: float = DEFAULT_B_FIELD_MAGNITUDE_TESLA,
    dip_tilt_deg: float = DEFAULT_DIP_TILT_DEG,
) -> np.ndarray:
    """Representative inertial B-field vector [T] (tilted-dipole direction, constant magnitude)."""
    if magnitude_tesla <= 0:
        raise ValueError(f"magnitude_tesla must be > 0, got {magnitude_tesla}")
    return magnitude_tesla * b_field_direction_inertial(phase_rad, dip_tilt_deg)


def sun_and_mag_body_vectors(
    phase_rad: float,
    beta_angle_deg: float,
    dip_tilt_deg: float = DEFAULT_DIP_TILT_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """True (noise-free) Sun- and magnetic-field UNIT direction vectors in the body frame.

    This is a purely geometric quantity (does not depend on whether a
    sensor could actually detect the vector, e.g. eclipse/FOV/exclusion) -
    useful for plotting the true Sun-mag separation angle continuously over
    the orbit regardless of instantaneous sensor availability.
    """
    A = attitude_dcm_nadir_pointing(phase_rad)
    sun_i = sun_vector_inertial(beta_angle_deg)
    b_hat_i = b_field_direction_inertial(phase_rad, dip_tilt_deg)
    sun_b = A @ sun_i
    mag_b = A @ b_hat_i
    return sun_b, mag_b


@dataclass(frozen=True)
class EnvironmentState:
    """Snapshot of the orbit/environment model at one instant in time."""

    t_s: float
    phase_rad: float
    in_eclipse: bool
    sun_vector_inertial: np.ndarray
    b_field_inertial_tesla: np.ndarray
    sun_body_true: np.ndarray
    mag_body_true: np.ndarray
    attitude_rotvec_rad: np.ndarray
    star_tracker_bright_source_angle_rad: float


def environment_state(
    t_s: float,
    period_s: float,
    altitude_km: float,
    beta_angle_deg: float,
    b_field_magnitude_tesla: float = DEFAULT_B_FIELD_MAGNITUDE_TESLA,
    dip_tilt_deg: float = DEFAULT_DIP_TILT_DEG,
) -> EnvironmentState:
    """Build a full :class:`EnvironmentState` at time ``t_s`` [s] since epoch."""
    if period_s <= 0:
        raise ValueError(f"period_s must be > 0, got {period_s}")
    _validate_beta_deg(beta_angle_deg)

    phase = 2.0 * np.pi * (t_s / period_s)
    sun_i = sun_vector_inertial(beta_angle_deg)
    b_i = b_field_inertial_tesla(phase, b_field_magnitude_tesla, dip_tilt_deg)
    sun_b, mag_b = sun_and_mag_body_vectors(phase, beta_angle_deg, dip_tilt_deg)
    in_eclipse = is_in_eclipse(phase, altitude_km, beta_angle_deg)
    angle_bright = star_tracker_bright_source_angle_rad(phase, beta_angle_deg)
    phi = attitude_rotvec_nadir_pointing(phase)

    return EnvironmentState(
        t_s=float(t_s),
        phase_rad=float(np.mod(phase, 2.0 * np.pi)),
        in_eclipse=in_eclipse,
        sun_vector_inertial=sun_i,
        b_field_inertial_tesla=b_i,
        sun_body_true=sun_b,
        mag_body_true=mag_b,
        attitude_rotvec_rad=phi,
        star_tracker_bright_source_angle_rad=angle_bright,
    )


@dataclass(frozen=True)
class EnvironmentBatch:
    """Vectorized (array-valued) counterpart of :class:`EnvironmentState`.

    Every field is a numpy array of length N (matching the input
    ``t_s_array``), except ``sun_vector_inertial`` which is a single
    constant unit vector (the Sun direction does not vary with time in
    this model). Produced by :func:`environment_states_batch`, which
    computes the same physics as repeated :func:`environment_state` calls
    but vectorized (single batched attitude-rotation-vector conversion
    instead of one scipy call per sample) - purely a performance
    optimization used by ``availability.py`` for large time grids /
    sensitivity sweeps; the underlying model is unchanged.
    """

    t_s: np.ndarray
    phase_rad: np.ndarray
    in_eclipse: np.ndarray
    sun_vector_inertial: np.ndarray
    b_field_inertial_tesla: np.ndarray
    attitude_rotvec_rad: np.ndarray
    star_tracker_bright_source_angle_rad: np.ndarray
    separation_angle_deg: np.ndarray


def environment_states_batch(
    t_s_array: np.ndarray,
    period_s: float,
    altitude_km: float,
    beta_angle_deg: float,
    b_field_magnitude_tesla: float = DEFAULT_B_FIELD_MAGNITUDE_TESLA,
    dip_tilt_deg: float = DEFAULT_DIP_TILT_DEG,
) -> EnvironmentBatch:
    """Vectorized batch evaluation of the orbit/environment model.

    Numerically identical to calling :func:`environment_state` once per
    entry of ``t_s_array`` (same closed-form equations), just computed with
    array operations plus a single batched attitude-DCM-to-rotation-vector
    conversion instead of one scipy call per sample.
    """
    if period_s <= 0:
        raise ValueError(f"period_s must be > 0, got {period_s}")
    _validate_beta_deg(beta_angle_deg)
    if b_field_magnitude_tesla <= 0:
        raise ValueError(f"b_field_magnitude_tesla must be > 0, got {b_field_magnitude_tesla}")

    t = np.asarray(t_s_array, dtype=float).reshape(-1)
    phase = 2.0 * np.pi * (t / period_s)

    r_hat = np.stack([np.cos(phase), np.sin(phase), np.zeros_like(phase)], axis=-1)  # (N,3)
    v_hat = np.stack([-np.sin(phase), np.cos(phase), np.zeros_like(phase)], axis=-1)  # (N,3)
    z_b = -r_hat
    x_b = v_hat
    y_b = np.cross(z_b, x_b)
    A_batch = np.stack([x_b, y_b, z_b], axis=1)  # (N,3,3), rows = body axes

    rotvecs = Rotation.from_matrix(A_batch).as_rotvec()  # (N,3), single batched scipy call

    sun_i = sun_vector_inertial(beta_angle_deg)
    m_hat = dipole_axis_inertial(dip_tilt_deg)
    dot = r_hat @ m_hat
    raw = 3.0 * dot[:, None] * r_hat - m_hat[None, :]
    b_hat_i = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    b_i = b_field_magnitude_tesla * b_hat_i  # (N,3)

    frac = eclipse_fraction(altitude_km, beta_angle_deg)
    if frac <= 0.0:
        in_eclipse = np.zeros_like(phase, dtype=bool)
    else:
        half_width = np.pi * frac
        delta = np.angle(np.exp(1j * (phase - np.pi)))
        in_eclipse = np.abs(delta) <= half_width

    cos_bright = np.clip(r_hat @ sun_i, -1.0, 1.0)
    angle_bright = np.arccos(cos_bright)

    sun_body = np.einsum("nij,j->ni", A_batch, sun_i)
    mag_body = np.einsum("nij,nj->ni", A_batch, b_hat_i)
    cos_sep = np.clip(np.sum(sun_body * mag_body, axis=1), -1.0, 1.0)
    separation_angle_deg = np.degrees(np.arccos(cos_sep))

    return EnvironmentBatch(
        t_s=t,
        phase_rad=np.mod(phase, 2.0 * np.pi),
        in_eclipse=in_eclipse,
        sun_vector_inertial=sun_i,
        b_field_inertial_tesla=b_i,
        attitude_rotvec_rad=rotvecs,
        star_tracker_bright_source_angle_rad=angle_bright,
        separation_angle_deg=separation_angle_deg,
    )
