"""Bring your own imagery: any georeferenced GeoTIFF/COG on disk or at a URL."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

from ..geo import AOI
from ..models import Render, Scene, parse_dt


def local_scene(path: str, when: str | None = None, sensor: str = "optical") -> tuple[Scene, AOI]:
    """Wrap a georeferenced raster as a Scene (and its footprint as an AOI).

    8-bit RGB(A) rasters render as-is; anything else is treated as single-band SAR /
    grayscale and stretched. ``when`` overrides the acquisition time (ISO date).
    """
    with rasterio.open(path) as ds:
        if ds.crs is None:
            raise ValueError(f"{path} has no georeferencing (CRS); Oracle can't place it on the map")
        bbox = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
        res = abs(ds.res[0])
        if ds.crs.is_geographic:
            res *= 111_320 * math.cos(math.radians((bbox[1] + bbox[3]) / 2))
        tags = ds.tags()
        rgb = ds.count >= 3 and ds.dtypes[0] == "uint8"
    t = when or tags.get("TIFFTAG_DATETIME") or tags.get("datetime")
    dt = _parse_time(t)
    render = Render(kind="rgb8", hrefs=[path], bands=[1, 2, 3]) if rgb else Render(kind="sar", hrefs=[path])
    scene = Scene(
        id=f"local-{Path(path).stem}",
        source="local",
        platform="local",
        sensor=sensor if rgb else "sar",
        datetime=dt,
        gsd=round(res, 3),
        bbox=tuple(bbox),
        render=render,
        item_url=path,
        attribution=f"user imagery: {Path(path).name}",
    )
    return scene, AOI(tuple(bbox), Path(path).name)


def _parse_time(t: str | None) -> datetime:
    if not t:
        return datetime.now(timezone.utc)
    t = t.strip()
    if len(t) >= 19 and t[4] == ":" and t[7] == ":":  # TIFF style "YYYY:MM:DD HH:MM:SS"
        t = t[:10].replace(":", "-") + "T" + t[11:19]
    return parse_dt(t)
