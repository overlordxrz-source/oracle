"""Maxar Open Data Program: 30-50 cm WorldView/GeoEye imagery released for disasters.

Coverage is limited to event areas (earthquakes, floods, wildfires, conflicts ...),
usually with pre- and post-event collects. Licence CC BY-NC 4.0.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from urllib.parse import urljoin

from pyproj import Transformer

from ..geo import AOI, bbox_intersects
from ..http import fetch_many_json, get_json
from ..models import Render, Scene, parse_dt
from .base import SourceInfo
from .static_index import StaticIndex

ROOT = "https://maxar-opendata.s3.amazonaws.com/events/catalog.json"

# Maxar ARD grid ("MXRA"): a quadtree per UTM zone, origin (-9,740,000, 10,240,000) in the
# zone's *northern* UTM CRS, 20,480 km across, so zoom-12 tiles are exactly 5 km.
_ARD_X0, _ARD_Y0, _ARD_EXTENT = -9_740_000.0, 10_240_000.0, 20_480_000.0
_transformers: dict[int, Transformer] = {}


def ard_tile_bbox(zone: int, quadkey: str) -> tuple[float, float, float, float]:
    """MXRA quadkey in UTM ``zone`` -> WGS84 bbox."""
    tx = ty = 0
    for ch in quadkey:
        d = int(ch)
        tx, ty = tx * 2 + (d & 1), ty * 2 + ((d >> 1) & 1)
    size = _ARD_EXTENT / 2 ** len(quadkey)
    x0, y1 = _ARD_X0 + tx * size, _ARD_Y0 - ty * size
    if zone not in _transformers:
        _transformers[zone] = Transformer.from_crs(32600 + zone, 4326, always_xy=True)
    return _transformers[zone].transform_bounds(x0, y1 - size, x0 + size, y1, densify_pts=5)


def _build() -> list[dict]:
    root = get_json(ROOT)
    events = [urljoin(ROOT, link["href"]) for link in root["links"] if link["rel"] == "child"]
    ev_docs = fetch_many_json(events, progress="maxar events")
    acq_urls: list[tuple[str, str]] = []
    for url, doc in ev_docs.items():
        if not doc:
            continue
        for link in doc.get("links", []):
            if link["rel"] == "child":
                acq_urls.append((doc["id"], urljoin(url, link["href"])))
    acq_docs = fetch_many_json([u for _, u in acq_urls], progress="maxar acquisitions")
    recs = []
    for event, url in acq_urls:
        doc = acq_docs.get(url)
        if not doc:
            continue
        try:
            when = doc["extent"]["temporal"]["interval"][0][0]
        except (KeyError, IndexError):
            continue
        for link in doc.get("links", []):
            if link["rel"] != "item":
                continue
            item_url = urljoin(url, link["href"])
            # .../ard/<utm zone>/<quadkey>/<date>/<catalog id>.json
            zone, quadkey = item_url.split("/")[-4:-2]
            try:
                bbox = [round(v, 6) for v in ard_tile_bbox(int(zone), quadkey)]
            except ValueError:
                continue
            recs.append({"e": event, "a": doc["id"], "t": when, "b": bbox, "u": item_url})
    return recs


INDEX = StaticIndex("maxar", _build)


class Maxar:
    info = SourceInfo(
        key="maxar",
        name="Maxar Open Data (WorldView / GeoEye)",
        sensor="optical",
        resolution="0.3-0.5 m",
        coverage="disaster & crisis event areas only (~60 events)",
        revisit="event driven, pre + post imagery",
        license="CC BY-NC 4.0",
        best_gsd=0.3,
    )

    def search(self, aoi: AOI, start: datetime, end: datetime, *, max_cloud: float | None = None, limit: int = 50) -> list[Scene]:
        hits: dict[str, list[dict]] = defaultdict(list)
        for r in INDEX.records():
            t = parse_dt(r["t"])
            if not (start <= t <= end):
                continue
            if bbox_intersects(tuple(r["b"]), aoi.bbox):
                hits[r["a"]].append(r)
        if not hits:
            return []
        acqs = sorted(hits.values(), key=lambda rs: rs[0]["t"], reverse=True)[:limit]
        items = fetch_many_json([r["u"] for rs in acqs for r in rs], concurrency=16)
        out = []
        for rs in acqs:
            docs = [(r["u"], items.get(r["u"])) for r in rs]
            docs = [(u, d) for u, d in docs if d]
            if not docs:
                continue
            s = _to_scene(rs[0], docs)
            if s and (max_cloud is None or (s.cloud_cover or 0) <= max_cloud):
                out.append(s)
        return out


def _to_scene(rec: dict, docs: list[tuple[str, dict]]) -> Scene | None:
    hrefs, polys, gsds, clouds, nadirs = [], [], [], [], []
    bbox = None
    p0 = docs[0][1]["properties"]
    for url, d in docs:
        a = d.get("assets", {}).get("visual")
        if not a:
            continue
        hrefs.append(urljoin(url, a["href"]))
        p = d["properties"]
        gsds.append(p.get("gsd", 0.5))
        clouds.append(p.get("tile:clouds_percent", 0))
        if p.get("view:off_nadir") is not None:
            nadirs.append(p["view:off_nadir"])
        if d.get("geometry"):
            polys.append(d["geometry"]["coordinates"])
        b = d["bbox"]
        bbox = b if bbox is None else (min(bbox[0], b[0]), min(bbox[1], b[1]), max(bbox[2], b[2]), max(bbox[3], b[3]))
    if not hrefs:
        return None
    return Scene(
        id=f"maxar-{rec['a']}",
        source="maxar",
        platform=p0.get("platform", "maxar"),
        sensor="optical",
        datetime=parse_dt(rec["t"]),
        gsd=round(min(gsds), 2),
        bbox=tuple(bbox),
        geometry={"type": "MultiPolygon", "coordinates": polys} if polys else None,
        cloud_cover=round(sum(clouds) / len(clouds), 1),
        off_nadir=round(sum(nadirs) / len(nadirs), 1) if nadirs else None,
        item_url=docs[0][0],
        license="CC BY-NC 4.0",
        attribution="Imagery (c) Maxar Technologies, Maxar Open Data Program",
        render=Render(kind="rgb8", hrefs=hrefs, bands=[1, 2, 3]),
        extra={"event": rec["e"], "catalog_id": rec["a"], "tiles": len(hrefs)},
    )
