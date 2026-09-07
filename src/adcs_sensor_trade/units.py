"""Unit conversion utilities.

Convention (see docs/conventions.md): all angles are stored and computed in
RADIANS everywhere inside this package. Degrees and arcseconds are permitted
ONLY at the input/reporting boundary (sensor configuration constructors that
explicitly accept a "*_arcsec" or "*_deg" keyword, and plotting/report code).
These helpers are the single source of truth for those conversions.
"""

from __future__ import annotations

import math

ARCSEC_PER_DEGREE = 3600.0
DEGREES_PER_RADIAN = 180.0 / math.pi
ARCSEC_PER_RADIAN = ARCSEC_PER_DEGREE * DEGREES_PER_RADIAN


def arcsec_to_rad(arcsec: float) -> float:
    """Convert an angle in arcseconds to radians."""
    return arcsec / ARCSEC_PER_RADIAN


def rad_to_arcsec(rad: float) -> float:
    """Convert an angle in radians to arcseconds."""
    return rad * ARCSEC_PER_RADIAN


def deg_to_rad(deg: float) -> float:
    """Convert an angle in degrees to radians."""
    return math.radians(deg)


def rad_to_deg(rad: float) -> float:
    """Convert an angle in radians to degrees."""
    return math.degrees(rad)
