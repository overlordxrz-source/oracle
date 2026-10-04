"""Microsoft Planetary Computer: Sentinel-1 RTC radar, Landsat, NAIP.

Data is public; reads need a short-lived anonymous SAS token per collection, which
``sign()`` fetches and caches (no account or key required).
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from urllib.parse import urlparse

from ..geo import AOI
from ..http import get_json
from ..models import Render, Scene, parse_dt
from .base import SourceInfo, asset_href, stac_search

API = "https://planetarycomputer.microsoft.com/api/stac/v1"
TOKEN_API = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
PREVIEW = "https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png"

_tokens: dict[str, tuple[str, float]] = {}
_lock = threading.Lock()


def sign(href: str, collection: str) -> str:
    """Append a (cached) SAS token for ``collection`` to an Azure blob href."""
    if not urlparse(href).netloc.endswith(".blob.core.windows.net") or "?" in href:
        return href
    with _lock:
        tok, exp = _tokens.get(collection, ("", 0.0))
        if time.time() > exp - 300:
            d = get_json(f"{TOKEN_API}/{collection}")
            tok = d["token"]
            exp_s = d.get("msft:expiry")
            exp = parse_dt(exp_s).timestamp() if exp_s else time.time() + 3000
            _tokens[collection] = (tok, exp)
    return f"{href}?{tok}"


def _base(item: dict, collection: str) -> dict:
    return {
        "id": item["id"],
        "datetime": parse_dt(item["properties"]["datetime"]),
        "bbox": tuple(item["bbox"]),
        "geometry": item.get("geometry"),
        "item_url": f"{API}/collections/{collection}/items/{item['id']}",
        # PC's rendered previews look good for Landsat; S1 RTC previews come out near-black.
        "thumbnail": f"{PREVIEW}?collection={collection}&item={item['id']}&format=png&max_size=256"
        if collection == "landsat-c2-l2"
        else None,
    }


class Sentinel1:
    info = SourceInfo(
        key="sentinel-1",
        name="Sentinel-1 RTC radar (ESA Copernicus via Planetary Computer)",
        sensor="sar",
        resolution="10 m (C-band SAR, sees through cloud and at night)",
        coverage="global land + coasts",
        revisit="~6-12 days, archive from 2014",
        license="Copernicus open licence",
        best_gsd=10.0,
    )
    collection = "sentinel-1-rtc"

    def search(self, aoi: AOI, start: datetime, end: datetime, *, max_cloud=None, limit: int = 50) -> list[Scene]:
        out = []
        for item in stac_search(API, [self.collection], aoi, start, end, max_items=limit):
            p = item["properties"]
            vv = asset_href(item, "vv", "hh")
            if not vv:
                continue
            vh = asset_href(item, "vh", "hv")
            out.append(
                Scene(
                    **_base(item, self.collection),
                    source="sentinel-1",
                    platform=p.get("platform", "sentinel-1"),
                    sensor="sar",
                    gsd=10.0,
                    license="Copernicus Sentinel data, open licence",
                    attribution="Contains modified Copernicus Sentinel data (RTC by Catalyst/Microsoft)",
                    render=Render(kind="sar", hrefs=[vv], power=True, vmin=-25, vmax=5, sign=self.collection),
                    extra={
                        "polarizations": p.get("sar:polarizations"),
                        "orbit_state": p.get("sat:orbit_state"),
                        "mode": p.get("sar:instrument_mode"),
                        "cross_pol": vh,
                    },
                )
            )
        return out


class Landsat:
    info = SourceInfo(
        key="landsat",
        name="Landsat 4-9 Collection 2 L2 (USGS via Planetary Computer)",
        sensor="optical",
        resolution="30 m",
        coverage="global land",
        revisit="~8 days (L8+L9), archive from 1982",
        license="Public domain (USGS)",
        best_gsd=30.0,
    )
    collection = "landsat-c2-l2"

    def search(self, aoi: AOI, start: datetime, end: datetime, *, max_cloud=None, limit: int = 50) -> list[Scene]:
        query = {"eo:cloud_cover": {"lte": max_cloud}} if max_cloud is not None else None
        out = []
        for item in stac_search(API, [self.collection], aoi, start, end, query=query, max_items=limit):
            p = item["properties"]
            rgb = [asset_href(item, b) for b in ("red", "green", "blue")]
            if not all(rgb):
                continue
            out.append(
                Scene(
                    **_base(item, self.collection),
                    source="landsat",
                    platform=p.get("platform", "landsat"),
                    sensor="optical",
                    gsd=30.0,
                    cloud_cover=p.get("eo:cloud_cover"),
                    off_nadir=p.get("view:off_nadir"),
                    license="Public domain",
                    attribution="Landsat imagery courtesy of the U.S. Geological Survey",
                    render=Render(
                        kind="reflectance",
                        hrefs=rgb,
                        scale=0.0000275,
                        offset=-0.2,
                        vmin=0.0,
                        vmax=0.3,
                        sign=self.collection,
                    ),
                    extra={"nir": asset_href(item, "nir08")},
                )
            )
        return out


class NAIP:
    info = SourceInfo(
        key="naip",
        name="NAIP aerial (USDA via Planetary Computer)",
        sensor="optical",
        resolution="0.3-1 m",
        coverage="continental United States only",
        revisit="every 2-3 years per state, archive from 2010",
        license="Public domain (USDA)",
        best_gsd=0.3,
    )
    collection = "naip"

    def search(self, aoi: AOI, start: datetime, end: datetime, *, max_cloud=None, limit: int = 50) -> list[Scene]:
        groups: dict[str, Scene] = {}
        for item in stac_search(API, [self.collection], aoi, start, end, max_items=limit * 4):
            p = item["properties"]
            href = asset_href(item, "image")
            if not href:
                continue
            base = _base(item, self.collection)
            # NAIP ships as ~6x7 km quarter-quads; merge same-day tiles into one scene.
            key = base["datetime"].strftime("%Y-%m-%d") + p.get("naip:state", "")
            if key in groups:
                g = groups[key]
                g.render.hrefs.append(href)
                g.bbox = _union(g.bbox, base["bbox"])
                g.geometry = None
                continue
            groups[key] = Scene(
                **base,
                source="naip",
                platform="aerial",
                sensor="optical",
                gsd=float(p.get("gsd", 0.6)),
                license="Public domain",
                attribution="USDA Farm Service Agency NAIP",
                render=Render(kind="rgb8", hrefs=[href], bands=[1, 2, 3], sign=self.collection),
                extra={"state": p.get("naip:state"), "year": p.get("naip:year")},
            )
        return list(groups.values())[:limit]


def _union(a: tuple, b: tuple) -> tuple:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
