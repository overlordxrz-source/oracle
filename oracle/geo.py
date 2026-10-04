"""Areas of interest, geocoding and the small amount of map math Oracle needs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import httpx
from pyproj import CRS

from .config import HTTP_TIMEOUT, USER_AGENT

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class AOI:
    """Area of interest in WGS84. ``bbox`` is (west, south, east, north)."""

    bbox: tuple[float, float, float, float]
    name: str = ""

    @property
    def center(self) -> tuple[float, float]:
        w, s, e, n = self.bbox
        return ((s + n) / 2, (w + e) / 2)  # (lat, lon)

    @property
    def geometry(self) -> dict:
        w, s, e, n = self.bbox
        return {
            "type": "Polygon",
            "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
        }

    def size_km(self) -> tuple[float, float]:
        w, s, e, n = self.bbox
        lat = math.radians((s + n) / 2)
        return (
            (e - w) * 111.320 * math.cos(lat),
            (n - s) * 110.574,
        )

    def utm_crs(self) -> CRS:
        lat, lon = self.center
        return utm_crs_for(lat, lon)

    @classmethod
    def from_point(cls, lat: float, lon: float, radius_km: float = 2.0, name: str = "") -> AOI:
        dlat = radius_km / 110.574
        dlon = radius_km / (111.320 * max(math.cos(math.radians(lat)), 1e-6))
        return cls((lon - dlon, lat - dlat, lon + dlon, lat + dlat), name)


def utm_crs_for(lat: float, lon: float) -> CRS:
    zone = int((lon + 180) // 6) % 60 + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def bbox_intersects(a: tuple, b: tuple) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def bbox_overlap_fraction(aoi: tuple, other: tuple) -> float:
    """Fraction of ``aoi`` covered by ``other``'s bounding box (0..1)."""
    w, s = max(aoi[0], other[0]), max(aoi[1], other[1])
    e, n = min(aoi[2], other[2]), min(aoi[3], other[3])
    if e <= w or n <= s:
        return 0.0
    area = (aoi[2] - aoi[0]) * (aoi[3] - aoi[1])
    return ((e - w) * (n - s)) / area if area > 0 else 1.0


def tile_bounds(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    """XYZ (slippy map) tile -> WGS84 bbox."""
    n = 2.0**z

    def lat(yy: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))

    return (x / n * 360.0 - 180.0, lat(y + 1), (x + 1) / n * 360.0 - 180.0, lat(y))


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2**z
    lat = max(min(lat, 85.0511), -85.0511)
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def mercator_tile_bounds(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    """XYZ tile -> EPSG:3857 bounds."""
    origin = 20037508.342789244
    size = 2 * origin / 2**z
    return (-origin + x * size, origin - (y + 1) * size, -origin + (x + 1) * size, origin - y * size)


_FLOAT = r"[-+]?\d+(?:\.\d+)?"


def parse_aoi(text: str, radius_km: float = 2.0) -> AOI:
    """Parse ``lat,lon`` | ``west,south,east,north`` | a place name.

    ``lat,lon`` becomes a square of +/- ``radius_km``. Place names are geocoded with
    OpenStreetMap Nominatim (1 request; respect its usage policy).
    """
    t = text.strip()
    nums = re.findall(_FLOAT, t)
    if re.fullmatch(rf"\s*{_FLOAT}\s*[, ]\s*{_FLOAT}\s*", t):
        lat, lon = float(nums[0]), float(nums[1])
        _check_latlon(lat, lon)
        return AOI.from_point(lat, lon, radius_km, name=t)
    if re.fullmatch(rf"\s*{_FLOAT}(\s*[, ]\s*{_FLOAT}){{3}}\s*", t):
        w, s, e, n = map(float, nums)
        if not (w < e and s < n):
            raise ValueError("bbox must be west,south,east,north")
        _check_latlon(s, w)
        _check_latlon(n, e)
        return AOI((w, s, e, n), name=t)
    return geocode(t, radius_km)


def _check_latlon(lat: float, lon: float) -> None:
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"coordinate out of range: lat={lat} lon={lon} (did you swap them?)")


def geocode(query: str, radius_km: float = 2.0) -> AOI:
    """Place name -> AOI via OpenStreetMap (Nominatim, falling back to Photon)."""
    errors = []
    for fn in (_nominatim, _photon):
        try:
            hit = fn(query)
        except httpx.HTTPError as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else type(exc).__name__
            errors.append(f"{fn.__name__.strip('_')}: {status}")
            continue
        if hit:
            lat, lon, name = hit
            return AOI.from_point(lat, lon, radius_km, name=name)
        errors.append(f"{fn.__name__.strip('_')}: no match")
    raise ValueError(f"could not geocode {query!r} ({'; '.join(errors)}); pass lat,lon instead")


def _nominatim(query: str) -> tuple[float, float, str] | None:
    r = httpx.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "jsonv2", "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    hits = r.json()
    if not hits:
        return None
    return float(hits[0]["lat"]), float(hits[0]["lon"]), hits[0].get("display_name", query)


def _photon(query: str) -> tuple[float, float, str] | None:
    r = httpx.get(
        "https://photon.komoot.io/api/",
        params={"q": query, "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    feats = r.json().get("features") or []
    if not feats:
        return None
    lon, lat = feats[0]["geometry"]["coordinates"][:2]
    p = feats[0]["properties"]
    name = ", ".join(str(p[k]) for k in ("name", "city", "country") if p.get(k)) or query
    return float(lat), float(lon), name
