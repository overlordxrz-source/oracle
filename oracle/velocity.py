"""Is a ship underway, and which way is it heading? Read from its wake.

Foam is bright in visible light but weak in near-IR (water absorbs NIR), while painted
hulls are relatively bright in NIR. So a moving ship on Sentinel-2 is a NIR-bright hull
followed by a blue-bright, NIR-dark foam trail along its axis. Oracle looks for foam in
a corridor along the hull axis: if there is clearly more on one side, the ship is
underway and heading away from it.

Why not speed from inter-band parallax? Sentinel-2 bands are acquired up to ~2 s apart
(B08 0.264 s, B03 0.527 s, B04 1.005 s after B02; Binet et al. 2022), so moving objects
shift between bands. That works for aircraft (~200 m shifts), but for ships Oracle
measured it on the Malacca and Singapore straits and rejected it:
  - visible/NIR band pairs: hull paint and foam change the ship's shape from band to
    band by more than the ~1 px of motion, giving errors of 6-19 kn;
  - red-edge/NIR pairs (B06/B07/B8A, similar spectra) agree with each other, but
    inter-band misregistration (up to ~0.3 px) leaves anchored ships reading 1-13 kn.
Speeds therefore come from tracking (position change between observations) or AIS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class WakeMotion:
    underway: bool
    course_deg: float | None  # direction of travel 0-360, None if not underway / ambiguous
    wake_m: float  # length of foam trail detected behind the hull


def wake_motion(
    blue: np.ndarray,
    nir: np.ndarray,
    hull: np.ndarray,
    background: np.ndarray,
    axis_deg: float,
    length_m: float,
    width_m: float,
    res: float,
) -> WakeMotion | None:
    """Foam asymmetry along the hull axis -> underway flag + course.

    ``blue``/``nir``: reflectance windows. ``hull``: boolean blob mask. ``background``:
    open-water pixels for noise statistics. Angles are degrees from north.
    """
    if not background.any() or hull.sum() < 3:
        return None
    b_bg, n_bg = np.nanmedian(blue[background]), np.nanmedian(nir[background])
    b_sd = 1.4826 * np.nanmedian(np.abs(blue[background] - b_bg)) + 1e-4
    eb = np.nan_to_num(blue - b_bg)
    en = np.nan_to_num(nir - n_bg)
    rows, cols = np.indices(blue.shape)
    hr, hc = np.nonzero(hull)
    cy, cx = hr.mean(), hc.mean()
    ux, uy = math.sin(math.radians(axis_deg)), math.cos(math.radians(axis_deg))  # east, north
    east, north = (cols - cx) * res, -(rows - cy) * res
    along = east * ux + north * uy
    across = -east * uy + north * ux
    corridor = (np.abs(across) <= max(width_m, 2 * res)) & (np.abs(along) <= 0.5 * length_m + 3 * length_m)
    # foam: clearly brighter than water in blue, and blue-dominated rather than NIR-bright
    foam = corridor & ~_dilate(hull, 1) & (eb > 3 * b_sd) & (eb > en)
    half = 0.5 * length_m
    ahead = foam & (along > half)
    behind = foam & (along < -half)
    fa, fb = int(ahead.sum()), int(behind.sum())
    total = fa + fb
    if total * res * res < max(2 * res * res, 0.15 * length_m * width_m):
        return WakeMotion(False, None, 0.0)
    if max(fa, fb) < 3 * max(min(fa, fb), 1):
        return WakeMotion(True, None, round(total * res * res / max(width_m, res), 1))  # foam both ends
    sign = 1.0 if fb > fa else -1.0  # foam behind (-axis side) -> heading +axis
    course = (math.degrees(math.atan2(sign * ux, sign * uy)) + 360) % 360
    tail = behind if fb > fa else ahead
    wake = float(np.abs(along[tail]).max() - half)
    return WakeMotion(True, round(course, 1), round(wake, 1))


def _dilate(m: np.ndarray, n: int) -> np.ndarray:
    from scipy import ndimage

    return ndimage.binary_dilation(m, iterations=n)
