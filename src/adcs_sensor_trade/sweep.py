"""Deterministic Sun-magnetic separation angle sweep.

Reusable function behind scripts/verify_sensor_geometry.py: sweeps the
angle between the (body-frame) Sun-direction and magnetic-field-direction
unit vectors and reports the resulting Sun+magnetometer information/
conditioning at each angle, alongside the (angle-independent, under our
simplified isotropic model) star-tracker reference.

All quantities here are deterministic (no RNG) local geometry/noise
proxies; see information.py for the modeling caveats.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geometry import condition_number, geometry_rank, singular_values
from .information import combined_vector_information, star_tracker_information, summarize_information


@dataclass(frozen=True)
class SweepConfig:
    sigma_sun_rad: float
    sigma_mag_rad: float
    sigma_star_rad: float
    angles_deg: np.ndarray


def sun_mag_body_vectors(separation_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (sun_body, mag_body) unit vectors separated by the given angle.

    Sun vector is fixed along body +X; the magnetic-field vector is placed
    in the body X-Y plane at ``separation_deg`` from the Sun vector. This
    is purely a geometric construction for the sweep (no attitude/rotation
    needed since only the RELATIVE angle between the two body-frame
    directions matters for the information/conditioning analysis).
    """
    sun_body = np.array([1.0, 0.0, 0.0])
    theta = np.radians(separation_deg)
    mag_body = np.array([np.cos(theta), np.sin(theta), 0.0])
    return sun_body, mag_body


def run_separation_sweep(
    angles_deg: np.ndarray,
    sigma_sun_rad: float,
    sigma_mag_rad: float,
    sigma_star_rad: float,
) -> pd.DataFrame:
    """Run the deterministic Sun-magnetic separation-angle sweep.

    Returns a pandas DataFrame with one row per angle, containing rank,
    condition number, min singular value, worst-axis attitude-error proxy
    [deg], and RMS attitude-error proxy [deg] for the Sun+mag architecture,
    plus the (angle-independent) star-tracker reference values for
    comparison.
    """
    rows = []
    J_star = star_tracker_information(sigma_star_rad)
    star_report = summarize_information(J_star)

    for angle_deg in angles_deg:
        sun_body, mag_body = sun_mag_body_vectors(float(angle_deg))
        rank = geometry_rank([sun_body, mag_body])
        cond = condition_number([sun_body, mag_body])
        sv = singular_values([sun_body, mag_body])

        J_sm = combined_vector_information(
            [sun_body, mag_body], [sigma_sun_rad, sigma_mag_rad]
        )
        sm_report = summarize_information(J_sm)

        rows.append(
            {
                "separation_angle_deg": float(angle_deg),
                "rank": rank,
                "min_singular_value": float(sv[-1]),
                "geometry_condition_number": cond,
                "sunmag_condition_number_J": sm_report.condition_number_J,
                "sunmag_worst_axis_error_deg": sm_report.worst_axis_error_deg,
                "sunmag_rms_error_deg": sm_report.rms_attitude_error_deg,
                "star_condition_number_J": star_report.condition_number_J,
                "star_worst_axis_error_deg": star_report.worst_axis_error_deg,
                "star_rms_error_deg": star_report.rms_attitude_error_deg,
            }
        )
    return pd.DataFrame(rows)
