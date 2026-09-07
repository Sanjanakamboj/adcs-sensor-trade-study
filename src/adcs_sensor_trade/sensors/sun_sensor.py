"""Coarse/fine sun-sensor model.

A sun sensor measures a single unit line-of-sight vector to the Sun,
expressed in the body frame. A single such vector constrains only 2 of the
3 rotational degrees of freedom of the spacecraft attitude: rotation of the
body ABOUT the sun-line itself produces no change in the measured Sun
direction, so that degree of freedom is instantaneously unobservable from
one Sun-sensor reading alone. See :func:`demonstrate_single_vector_rank`
below and ``tests/test_geometry.py`` / ``tests/test_information.py`` for a
numerical demonstration (rank == 2 for a single vector's stacked measurement
Jacobian).

See docs/sensor_models.md for the representative baseline values used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import geometry_rank
from ..rotations import rotate_inertial_to_body
from ..units import deg_to_rad, rad_to_deg

# Representative baseline values for a digital coarse/fine sun sensor.
# THESE ARE ILLUSTRATIVE, NOT VENDOR SPECIFICATIONS. See docs/sensor_models.md.
DEFAULT_SIGMA_DEG = 0.5  # 1-sigma angular noise, fine-sun-sensor class
DEFAULT_FOV_HALF_ANGLE_DEG = 60.0  # +/- 60 deg full cone half-angle
DEFAULT_UPDATE_RATE_HZ = 5.0


@dataclass
class SunSensorConfig:
    """Configuration for :class:`SunSensor`.

    Parameters
    ----------
    sigma_rad : float
        1-sigma angular measurement noise [rad] on the measured Sun
        direction. 0.0 disables noise (deterministic mode).
    fov_half_angle_rad : float
        Half-angle [rad] of the sensor's field-of-view cone about its
        boresight. The Sun is only measurable when within this cone.
    update_rate_hz : float
        Sensor update rate [Hz]. Must be > 0.
    """

    sigma_rad: float = deg_to_rad(DEFAULT_SIGMA_DEG)
    fov_half_angle_rad: float = deg_to_rad(DEFAULT_FOV_HALF_ANGLE_DEG)
    update_rate_hz: float = DEFAULT_UPDATE_RATE_HZ

    def __post_init__(self) -> None:
        if self.sigma_rad < 0:
            raise ValueError(f"sigma_rad must be >= 0, got {self.sigma_rad}")
        if not (0.0 < self.fov_half_angle_rad <= np.pi):
            raise ValueError(
                f"fov_half_angle_rad must be in (0, pi], got {self.fov_half_angle_rad}"
            )
        if self.update_rate_hz <= 0:
            raise ValueError(f"update_rate_hz must be > 0, got {self.update_rate_hz}")

    @property
    def sigma_deg(self) -> float:
        return rad_to_deg(self.sigma_rad)

    @property
    def fov_half_angle_deg(self) -> float:
        return rad_to_deg(self.fov_half_angle_rad)


@dataclass
class SunSensorMeasurement:
    """Result of a single SunSensor.measure() call."""

    available: bool
    sun_vector_body: np.ndarray | None  # unit vector, None if unavailable
    sigma_rad: float


class SunSensor:
    """Coarse/fine sun sensor: single unit-vector measurement model.

    RNG policy: takes an explicit ``numpy.random.Generator``; never touches
    numpy's global random state (see docs/conventions.md).
    """

    def __init__(self, config: SunSensorConfig | None = None) -> None:
        self.config = config or SunSensorConfig()

    def is_eclipsed(self, in_eclipse: bool) -> bool:
        """Availability is False whenever the spacecraft is in eclipse.

        Kept as an explicit, trivial function (rather than inlined) so the
        eclipse/exclusion state is a documented, testable concept distinct
        from the FOV geometry check.
        """
        return not in_eclipse

    def measure(
        self,
        sun_vector_inertial: np.ndarray,
        true_phi_rotvec_rad: np.ndarray,
        rng: np.random.Generator | None = None,
        in_eclipse: bool = False,
    ) -> SunSensorMeasurement:
        """Simulate a Sun-sensor measurement.

        Parameters
        ----------
        sun_vector_inertial : array_like, shape (3,)
            Known unit vector to the Sun, expressed in the inertial frame.
        true_phi_rotvec_rad : array_like, shape (3,)
            True inertial-to-body attitude rotation vector [rad].
        rng : numpy.random.Generator, optional
            Noise source. If None or sigma_rad == 0, deterministic.
        in_eclipse : bool
            If True, the sensor is unavailable regardless of geometry.
        """
        if self.is_eclipsed(in_eclipse) is False:
            return SunSensorMeasurement(False, None, self.config.sigma_rad)

        sun_body_true = rotate_inertial_to_body(sun_vector_inertial, true_phi_rotvec_rad)

        # FOV check: sensor boresight is assumed to be the body +X axis by
        # convention (see docs/conventions.md); angle from boresight to the
        # true Sun direction must be within the FOV half-angle.
        boresight_body = np.array([1.0, 0.0, 0.0])
        angle_from_boresight = np.arccos(np.clip(np.dot(boresight_body, sun_body_true), -1.0, 1.0))
        if angle_from_boresight > self.config.fov_half_angle_rad:
            return SunSensorMeasurement(False, None, self.config.sigma_rad)

        if rng is not None and self.config.sigma_rad > 0.0:
            noise = rng.normal(loc=0.0, scale=self.config.sigma_rad, size=3)
            measured = sun_body_true + noise
            measured = measured / np.linalg.norm(measured)
        else:
            measured = sun_body_true

        return SunSensorMeasurement(True, measured, self.config.sigma_rad)


def demonstrate_single_vector_rank(sun_vector_body: np.ndarray) -> int:
    """Return the rank of the attitude-information geometry from ONE Sun vector.

    Numerically demonstrates that a single Sun-sensor unit-vector
    measurement constrains only rank-2 of the 3 rotational degrees of
    freedom (rotation about the sun-line is unobservable). Expected: 2.
    """
    return geometry_rank([sun_vector_body])
