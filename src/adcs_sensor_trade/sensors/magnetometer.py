"""Magnetometer model.

A magnetometer measures the local geomagnetic field vector in the body
frame. Like a single Sun-sensor reading, a single magnetometer vector
measurement constrains only 2 of the 3 rotational degrees of freedom:
rotation about the local field-line direction is instantaneously
unobservable. See :func:`demonstrate_single_vector_rank`.

SIMPLIFICATION: the inertial magnetic field vector used here is a single
representative CONSTANT vector for Milestone 1. A real mission would use a
full geomagnetic field model (e.g. IGRF) evaluated at the spacecraft's
orbital position and epoch, which varies significantly around an orbit.
That full IGRF-based, orbit-resolved field model is explicitly DEFERRED to
a later milestone (see docs/sensor_models.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import geometry_rank
from ..rotations import rotate_inertial_to_body
from ..units import rad_to_deg

# Representative baseline values. THESE ARE ILLUSTRATIVE, NOT VENDOR SPECS.
# See docs/sensor_models.md.
DEFAULT_SIGMA_DEG = 0.5  # 1-sigma angular noise on the measured field direction
DEFAULT_UPDATE_RATE_HZ = 10.0
DEFAULT_BIAS_TESLA = np.array([0.0, 0.0, 0.0])  # nominal: no bias
DEFAULT_SCALE_FACTOR = np.array([1.0, 1.0, 1.0])  # nominal: unity scale, per axis

# Representative low-Earth-orbit geomagnetic field magnitude (Tesla). Actual
# field varies with position/epoch (~25-65 microtesla); this is a simple
# representative order-of-magnitude constant for Milestone 1.
DEFAULT_B_FIELD_MAGNITUDE_TESLA = 3.0e-5


@dataclass
class MagnetometerConfig:
    """Configuration for :class:`Magnetometer`.

    Parameters
    ----------
    sigma_rad : float
        1-sigma angular measurement noise [rad] on the measured field
        direction (applied as a small-angle perturbation, consistent with
        the other sensor models). 0.0 disables noise.
    update_rate_hz : float
        Sensor update rate [Hz]. Must be > 0.
    bias_tesla : array_like, shape (3,)
        Constant body-frame bias vector [T] added to the measured field.
    scale_factor : array_like, shape (3,)
        Per-axis multiplicative scale-factor error (diagonal scale matrix).
        Must be > 0 per axis (a zero or negative scale factor is unphysical
        for this simple model).
    """

    sigma_rad: float = np.radians(DEFAULT_SIGMA_DEG)
    update_rate_hz: float = DEFAULT_UPDATE_RATE_HZ
    bias_tesla: np.ndarray = None  # set in __post_init__
    scale_factor: np.ndarray = None  # set in __post_init__

    def __post_init__(self) -> None:
        if self.sigma_rad < 0:
            raise ValueError(f"sigma_rad must be >= 0, got {self.sigma_rad}")
        if self.update_rate_hz <= 0:
            raise ValueError(f"update_rate_hz must be > 0, got {self.update_rate_hz}")
        if self.bias_tesla is None:
            self.bias_tesla = DEFAULT_BIAS_TESLA.copy()
        else:
            self.bias_tesla = np.asarray(self.bias_tesla, dtype=float).reshape(3)
        if self.scale_factor is None:
            self.scale_factor = DEFAULT_SCALE_FACTOR.copy()
        else:
            self.scale_factor = np.asarray(self.scale_factor, dtype=float).reshape(3)
            if np.any(self.scale_factor <= 0):
                raise ValueError(
                    f"scale_factor entries must be > 0, got {self.scale_factor}"
                )

    @property
    def sigma_deg(self) -> float:
        return rad_to_deg(self.sigma_rad)


@dataclass
class MagnetometerMeasurement:
    """Result of a single Magnetometer.measure() call."""

    available: bool
    b_field_vector_body_tesla: np.ndarray | None  # raw vector [T], with bias/scale
    b_field_direction_body: np.ndarray | None  # unit vector, bias/scale removed... see note
    sigma_rad: float


class Magnetometer:
    """Magnetometer: body-frame magnetic field vector measurement model.

    RNG policy: takes an explicit ``numpy.random.Generator``; never touches
    numpy's global random state (see docs/conventions.md).
    """

    def __init__(self, config: MagnetometerConfig | None = None) -> None:
        self.config = config or MagnetometerConfig()

    def measure(
        self,
        b_field_inertial_tesla: np.ndarray,
        true_phi_rotvec_rad: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> MagnetometerMeasurement:
        """Simulate a magnetometer measurement.

        Parameters
        ----------
        b_field_inertial_tesla : array_like, shape (3,)
            Known (representative-constant, see module docstring) inertial
            magnetic field vector [T].
        true_phi_rotvec_rad : array_like, shape (3,)
            True inertial-to-body attitude rotation vector [rad].
        rng : numpy.random.Generator, optional
            Noise source. If None or sigma_rad == 0, deterministic.

        Returns
        -------
        MagnetometerMeasurement
            ``b_field_vector_body_tesla`` includes bias and scale-factor
            error applied to the (possibly noisy) true field direction and
            magnitude. ``b_field_direction_body`` is the pure unit-vector
            DIRECTION used for attitude-geometry purposes (angular noise
            only, no bias/scale — those are radiometric/calibration errors
            that a ground calibration would remove from the direction
            estimate to first order; kept separate from the geometry
            analysis in geometry.py / information.py, which consumes unit
            directions).
        """
        b_inertial = np.asarray(b_field_inertial_tesla, dtype=float).reshape(3)
        b_mag = np.linalg.norm(b_inertial)
        if b_mag < 1e-15:
            raise ValueError("b_field_inertial_tesla must be non-zero.")
        b_hat_inertial = b_inertial / b_mag

        b_hat_body_true = rotate_inertial_to_body(b_hat_inertial, true_phi_rotvec_rad)

        if rng is not None and self.config.sigma_rad > 0.0:
            noise = rng.normal(loc=0.0, scale=self.config.sigma_rad, size=3)
            b_hat_body_meas = b_hat_body_true + noise
            b_hat_body_meas = b_hat_body_meas / np.linalg.norm(b_hat_body_meas)
        else:
            b_hat_body_meas = b_hat_body_true

        # Raw vector output: true magnitude, with bias and scale-factor
        # error applied (representative simple calibration-error model).
        b_vector_raw = self.config.scale_factor * (b_mag * b_hat_body_meas) + self.config.bias_tesla

        return MagnetometerMeasurement(
            available=True,
            b_field_vector_body_tesla=b_vector_raw,
            b_field_direction_body=b_hat_body_meas,
            sigma_rad=self.config.sigma_rad,
        )


def demonstrate_single_vector_rank(b_field_direction_body: np.ndarray) -> int:
    """Return the rank of the attitude-information geometry from ONE B-field vector.

    Numerically demonstrates that a single magnetometer unit-vector
    measurement constrains only rank-2 of the 3 rotational degrees of
    freedom. Expected: 2.
    """
    return geometry_rank([b_field_direction_body])
