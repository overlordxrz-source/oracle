"""Esri World Imagery Wayback: every published version of Esri's global sub-metre basemap.

~200 releases since 2014. Each release is a mosaic, so a point's imagery only changes
in some releases; ``versions()`` walks the archive with the ``tilemap`` service to
find the distinct captures at a location, then reads each one's metadata (capture
date, resolution, sensor). Much of it is 30-60 cm Maxar/Vantor imagery.

Display only: tiles are shown in the map viewer straight from Esri, with
attribution, under Esri's terms of use. Oracle does not bulk-download them.
"""

from __future__ import annotations

import functools
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from ..geo import AOI, lonlat_to_tile
from ..http import client, get_json
from ..models import Render, Scene
from .base import SourceInfo

CONFIG_URL = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
TILEMAP = "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tilemap/{r}/{z}/{y}/{x}/1/1"
ATTRIBUTION = "Esri World Imagery Wayback; Esri, Maxar/Vantor, Earthstar Geographics, and the GIS User Community"
# Metadata layers by resolution: 4=30cm(z19) 5=60cm(z18) 6=1.2m(z17) 7=2.4m(z16) 8=4.8m(z15)
_META_LAYERS = (4, 5, 6, 7, 8)
_FIELDS = "SRC_DATE2,SRC_RES,SRC_ACC,SRC_DESC,NICE_NAME,NICE_DESC"


@functools.lru_cache(maxsize=1)
def releases() -> list[dict]:
    """All releases, newest first: {release, date, title, tiles, metadata}."""
    cfg = get_json(CONFIG_URL)
    out = []
    for num, r in cfg.items():
        m = re.search(r"(\d{4}-\d{2}-\d{2})", r["itemTitle"])
        if not m:
            continue
        out.append(
            {
                "release": int(num),
                "date": m.group(1),
                "title": r["itemTitle"],
                "tiles": r["itemURL"].replace("{level}", "{z}").replace("{row}", "{y}").replace("{col}", "{x}"),
                "metadata": r["metadataLayerUrl"],
            }
        )
    out.sort(key=lambda r: r["date"], reverse=True)
    return out


def versions(lat: float, lon: float, zoom: int = 17, max_versions: int = 40) -> list[dict]:
    """Distinct imagery captures at a point, newest first, with capture metadata."""
    rels = releases()
    by_num = {r["release"]: i for i, r in enumerate(rels)}
    x, y = lonlat_to_tile(lon, lat, zoom)
    found: list[dict] = []
    i = 0
    with client() as c:
        while i < len(rels) and len(found) < max_versions:
            r = c.get(TILEMAP.format(r=rels[i]["release"], z=zoom, y=y, x=x))
            if r.status_code != 200:
                break
            d = r.json()
            if not d.get("data") or d["data"][0] != 1:
                break  # no imagery at this zoom in this and older releases
            src = (d.get("select") or [rels[i]["release"]])[0]
            j = by_num.get(src, i)
            found.append(dict(rels[j]))  # copy: releases() is cached
            i = j + 1  # skip every release that just re-served this same tile
    with ThreadPoolExecutor(max_workers=8) as pool, client() as c:
        for v, meta in zip(found, pool.map(lambda v: _metadata(c, v["metadata"], lat, lon), found), strict=True):
            v.update(meta)
    # A zoom-17 tile changes when *any* part of it is re-imaged, so several releases can
    # still show the same capture at this exact point. Keep one row per capture.
    unique, seen = [], set()
    for v in found:
        key = (v["captured"], v["sensor"], v["resolution"]) if v["captured"] else v["release"]
        if key not in seen:
            seen.add(key)
            unique.append(v)
    return unique


def _metadata(c, layer_url: str, lat: float, lon: float) -> dict:
    for layer in _META_LAYERS:
        try:
            r = c.get(
                f"{layer_url}/{layer}/query",
                params={
                    "f": "json",
                    "geometry": f"{lon},{lat}",
                    "geometryType": "esriGeometryPoint",
                    "inSR": "4326",
                    "spatialRel": "esriSpatialRelIntersects",
                    "outFields": _FIELDS,
                    "returnGeometry": "false",
                },
            )
            feats = r.json().get("features") or []
        except Exception:  # noqa: BLE001 - metadata is best effort
            continue
        if feats:
            a = feats[0]["attributes"]
            cap = (
                datetime.fromtimestamp(a["SRC_DATE2"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                if a.get("SRC_DATE2")
                else None
            )
            return {
                "captured": cap,
                "resolution": a.get("SRC_RES"),
                "accuracy": a.get("SRC_ACC"),
                "sensor": a.get("SRC_DESC"),
                "provider": " ".join(x for x in (a.get("NICE_DESC"), a.get("NICE_NAME")) if x),
            }
    return {"captured": None, "resolution": None, "accuracy": None, "sensor": None, "provider": None}


class Wayback:
    info = SourceInfo(
        key="wayback",
        name="Esri World Imagery Wayback (view-only basemap archive)",
        sensor="optical",
        resolution="0.15-1 m in most populated areas",
        coverage="global mosaic, ~200 versions since 2014",
        revisit="irregular per location (months to years)",
        license="Esri terms of use (display with attribution)",
        best_gsd=0.3,
    )

    def search(self, aoi: AOI, start: datetime, end: datetime, *, max_cloud: float | None = None, limit: int = 50) -> list[Scene]:
        lat, lon = aoi.center
        out = []
        for v in versions(lat, lon):
            when = v.get("captured") or v["date"]
            t = datetime.fromisoformat(when).replace(tzinfo=timezone.utc)
            if not (start <= t <= end):
                continue
            out.append(
                Scene(
                    id=f"wayback-{v['release']}",
                    source="wayback",
                    platform=(v.get("sensor") or "mosaic"),
                    sensor="optical",
                    datetime=t,
                    gsd=float(v.get("resolution") or 1.0),
                    bbox=aoi.bbox,
                    item_url=f"https://livingatlas.arcgis.com/wayback/#active={v['release']}&mapCenter={lon:.5f},{lat:.5f},17",
                    license="Esri terms of use",
                    attribution=ATTRIBUTION,
                    render=Render(kind="xyz", hrefs=[v["tiles"]]),
                    extra={
                        "release": v["release"],
                        "release_date": v["date"],
                        "provider": v.get("provider"),
                        "accuracy_m": v.get("accuracy"),
                        "capture_date_known": bool(v.get("captured")),
                    },
                )
            )
        return out[:limit]
