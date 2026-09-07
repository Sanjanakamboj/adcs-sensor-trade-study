"""Star-tracker sensor model.

A star tracker outputs a full 3-axis attitude (quaternion / DCM), not a
single line-of-sight vector, because it triangulates the observed pattern
of many catalog stars in its field of view. See docs/sensor_models.md for
the representative baseline numbers used here and their justification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from ..rotations import dcm_from_rotvec
from ..units import arcsec_to_rad, rad_to_arcsec

# Representative baseline values for a small/mid-class star tracker.
# THESE ARE ILLUSTRATIVE, NOT VENDOR SPECIFICATIONS. See docs/sensor_models.md.
DEFAULT_SIGMA_ARCSEC = 30.0  # 1-sigma per-axis attitude noise, "tens of arcsec" class
DEFAULT_UPDATE_RATE_HZ = 2.0  # typical 1-5 Hz class update rate
DEFAULT_EXCLUSION_HALF_ANGLE_RAD = np.radians(30.0)  # e.g. Sun/Earth keep-out half-angle


@dataclass
class StarTrackerConfig:
    """Configuration for :class:`StarTracker`.

    Parameters
    ----------
    sigma_rad : float
        1-sigma per-axis attitude measurement noise [rad]. Use
        :func:`from_arcsec` to build this from an arcsec specification.
        Set to 0.0 for deterministic (noise-free) Milestone 1 tests.
    update_rate_hz : float
        Sensor update rate [Hz]. Must be > 0.
    exclusion_half_angle_rad : float
        Half-angle [rad] of a bright-source (e.g. Sun) keep-out cone within
        which the star tracker is considered unavailable (simple
        availability abstraction; no detailed baffle/stray-light model).
    dropout_probability : float
        Probability in [0, 1] of a random independent dropout on any given
        call to :meth:`StarTracker.measure`, even when nominally in FOV and
        outside the exclusion zone (models e.g. transient star-ID failures).
    """

    sigma_rad: float = arcsec_to_rad(DEFAULT_SIGMA_ARCSEC)
    update_rate_hz: float = DEFAULT_UPDATE_RATE_HZ
    exclusion_half_angle_rad: float = DEFAULT_EXCLUSION_HALF_ANGLE_RAD
    dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma_rad < 0:
            raise ValueError(f"sigma_rad must be >= 0, got {self.sigma_rad}")
        if self.update_rate_hz <= 0:
            raise ValueError(f"update_rate_hz must be > 0, got {self.update_rate_hz}")
        if not (0.0 <= self.exclusion_half_angle_rad <= np.pi):
            raise ValueError(
                "exclusion_half_angle_rad must be in [0, pi], got "
                f"{self.exclusion_half_angle_rad}"
            )
        if not (0.0 <= self.dropout_probability <= 1.0):
            raise ValueError(
                f"dropout_probability must be in [0, 1], got {self.dropout_probability}"
            )

    @classmethod
    def from_arcsec(cls, sigma_arcsec: float, **kwargs) -> "StarTrackerConfig":
        """Build a config from a 1-sigma per-axis noise spec in arcsec."""
        if sigma_arcsec < 0:
            raise ValueError(f"sigma_arcsec must be >= 0, got {sigma_arcsec}")
        return cls(sigma_rad=arcsec_to_rad(sigma_arcsec), **kwargs)

    @property
    def sigma_arcsec(self) -> float:
        return rad_to_arcsec(self.sigma_rad)


@dataclass
class StarTrackerMeasurement:
    """Result of a single StarTracker.measure() call."""

    available: bool
    dcm_body_from_inertial: np.ndarray | None  # None if not available
    quaternion_xyzw: np.ndarray | None  # None if not available
    sigma_rad: float


class StarTracker:
    """Star tracker: full 3-axis attitude measurement model.

    RNG policy: this class takes an explicit ``numpy.random.Generator``
    instance for every stochastic call. It never touches numpy's global
    random state, so multiple sensors / Monte Carlo trials can be run with
    independent, reproducible streams (see docs/conventions.md).
    """

    def __init__(self, config: StarTrackerConfig | None = None) -> None:
        self.config = config or StarTrackerConfig()

    def is_available(
        self,
        angle_to_bright_source_rad: float,
        rng: np.random.Generator | None = None,
    ) -> bool:
        """Simple availability abstraction.

        Unavailable if the boresight-to-bright-source angle is inside the
        exclusion half-angle, OR (if an rng is supplied) a random dropout
        is drawn according to ``dropout_probability``.
        """
        if angle_to_bright_source_rad < self.config.exclusion_half_angle_rad:
            return False
        if rng is not None and self.config.dropout_probability > 0.0:
            if rng.random() < self.config.dropout_probability:
                return False
        return True

    def measure(
        self,
        true_phi_rotvec_rad: np.ndarray,
        rng: np.random.Generator | None = None,
        angle_to_bright_source_rad: float = np.pi,
    ) -> StarTrackerMeasurement:
        """Simulate a star-tracker attitude measurement.

        Parameters
        ----------
        true_phi_rotvec_rad : array_like, shape (3,)
            True inertial-to-body attitude rotation vector [rad] (see
            rotations.py convention).
        rng : numpy.random.Generator, optional
            Source of randomness. If None, or if ``self.config.sigma_rad``
            is 0, the measurement is deterministic (no noise added).
        angle_to_bright_source_rad : float
            Angle used for the exclusion-zone availability check. Defaults
            to pi (i.e. "no bright source nearby" / always in nominal FOV).

        Returns
        -------
        StarTrackerMeasurement
        """
        available = self.is_available(angle_to_bright_source_rad, rng=rng)
        if not available:
            return StarTrackerMeasurement(
                available=False,
                dcm_body_from_inertial=None,
                quaternion_xyzw=None,
                sigma_rad=self.config.sigma_rad,
            )

        A_true = dcm_from_rotvec(true_phi_rotvec_rad)

        if rng is not None and self.config.sigma_rad > 0.0:
            noise_rotvec = rng.normal(loc=0.0, scale=self.config.sigma_rad, size=3)
            # Multiplicative small-angle perturbation, consistent with the
            # H_v = -skew(v_b) convention documented in rotations.py:
            # A_meas = R(noise) @ A_true.
            A_noise = Rotation.from_rotvec(noise_rotvec).as_matrix()
            A_meas = A_noise @ A_true
        else:
            A_meas = A_true

        quat_xyzw = Rotation.from_matrix(A_meas).as_quat()
        return StarTrackerMeasurement(
            available=True,
            dcm_body_from_inertial=A_meas,
            quaternion_xyzw=quat_xyzw,
            sigma_rad=self.config.sigma_rad,
        )
