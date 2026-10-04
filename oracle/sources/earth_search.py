"""Sentinel-2 L2A via Element 84 Earth Search (public COGs on AWS, no account needed)."""

from __future__ import annotations

from datetime import datetime

from ..geo import AOI
from ..models import Render, Scene, parse_dt
from .base import SourceInfo, asset_href, stac_search

API = "https://earth-search.aws.element84.com/v1"
# Collection 1 is the consistent reprocessing; the legacy collection fills gaps in the archive.
COLLECTIONS = ["sentinel-2-c1-l2a", "sentinel-2-l2a"]


class Sentinel2:
    info = SourceInfo(
        key="sentinel-2",
        name="Sentinel-2 L2A (ESA Copernicus via Earth Search)",
        sensor="optical",
        resolution="10 m",
        coverage="global land + coastal waters",
        revisit="~2-3 days (S2A/B/C), archive from 2015",
        license="Copernicus open licence",
        best_gsd=10.0,
    )

    def search(
        self,
        aoi: AOI,
        start: datetime,
        end: datetime,
        *,
        max_cloud: float | None = None,
        limit: int = 50,
    ) -> list[Scene]:
        query = {"eo:cloud_cover": {"lte": max_cloud}} if max_cloud is not None else None
        scenes: dict[tuple[str, str], Scene] = {}
        for collection in COLLECTIONS:
            for item in stac_search(API, [collection], aoi, start, end, query=query, max_items=limit):
                s = _to_scene(item, collection)
                if s is None:
                    continue
                key = (s.extra.get("mgrs", s.id), s.datetime.strftime("%Y%m%d"))
                # c1 is searched first and wins duplicates.
                scenes.setdefault(key, s)
        return sorted(scenes.values(), key=lambda s: s.datetime, reverse=True)[:limit]


def _to_scene(item: dict, collection: str) -> Scene | None:
    visual = asset_href(item, "visual")
    if not visual:
        return None
    p = item["properties"]
    bands = {k: asset_href(item, k) for k in ("red", "green", "blue", "nir", "scl", "swir16") if asset_href(item, k)}
    nir_meta = (item["assets"].get("nir", {}).get("raster:bands") or [{}])[0]
    return Scene(
        id=item["id"],
        source="sentinel-2",
        platform=p.get("platform", "sentinel-2"),
        sensor="optical",
        datetime=parse_dt(p["datetime"]),
        gsd=10.0,
        bbox=tuple(item["bbox"]),
        geometry=item.get("geometry"),
        cloud_cover=p.get("eo:cloud_cover"),
        off_nadir=p.get("view:incidence_angle"),
        thumbnail=asset_href(item, "thumbnail"),
        item_url=f"{API}/collections/{collection}/items/{item['id']}",
        license="Copernicus Sentinel data, open licence",
        attribution="Contains modified Copernicus Sentinel data",
        render=Render(kind="rgb8", hrefs=[visual], bands=[1, 2, 3]),
        extra={
            "collection": collection,
            "mgrs": p.get("grid:code") or p.get("s2:mgrs_tile"),
            "bands": bands,
            "reflectance_scale": nir_meta.get("scale", 0.0001),
            "reflectance_offset": nir_meta.get("offset", 0.0),
        },
    )
