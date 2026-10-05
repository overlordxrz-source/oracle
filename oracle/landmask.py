"""Static land/water mask from ESA WorldCover 2021 (10 m, global, CC BY 4.0, public on AWS).

Radar can't separate wind-roughened sea from land by brightness alone, so SAR vessel
detection uses this map instead. Class 80 is water (seas included) and 0 is no-data
offshore; a 3x3 degree tile that doesn't exist is open ocean.
"""

from __future__ import annotations

import math

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

from .imagery import Grid, read_href

WORLDCOVER = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
WATER = 80


def tile_names(bbox: tuple[float, float, float, float]) -> list[str]:
    w, s, e, n = bbox
    out = []
    for lat in range(int(math.floor(s / 3) * 3), int(math.floor(n / 3) * 3) + 1, 3):
        for lon in range(int(math.floor(w / 3) * 3), int(math.floor(e / 3) * 3) + 1, 3):
            out.append(f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}")
    return out


_cache: dict[tuple, np.ndarray | None] = {}


def water_mask(grid: Grid) -> np.ndarray | None:
    """Boolean water mask on ``grid``; None if WorldCover can't be reached."""
    key = (grid.crs.to_wkt(), tuple(grid.transform)[:6], grid.width, grid.height)
    if key not in _cache:
        if len(_cache) > 32:
            _cache.clear()
        _cache[key] = _compute(grid)
    return _cache[key]


def _compute(grid: Grid) -> np.ndarray | None:
    bbox = transform_bounds(grid.crs, "EPSG:4326", *grid.bounds, densify_pts=21)
    land = np.zeros((grid.height, grid.width), bool)
    reached = False
    for t in tile_names(bbox):
        try:
            a = read_href(WORLDCOVER.format(tile=t), grid, [1], Resampling.nearest)[0]
        except rasterio.errors.RasterioIOError as exc:
            if "404" in str(exc) or "does not exist" in str(exc) or "No such file" in str(exc):
                reached = True  # no tile here: open ocean
                continue
            return None
        reached = True
        land |= np.isfinite(a) & (a != WATER)
    return ~land if reached else None
