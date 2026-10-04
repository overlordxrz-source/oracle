"""Umbra Open Data: X-band SAR down to ~16-25 cm, thousands of collects since 2023.

The bucket's STAC tree has empty asset hrefs, so the index is built from the bucket
listing itself plus each collect's metadata JSON. Licence CC BY 4.0.
"""

from __future__ import annotations

import html
import re
from collections import defaultdict
from datetime import datetime
from urllib.parse import quote

from ..geo import AOI, bbox_intersects
from ..http import client, fetch_many_json
from ..models import Render, Scene, parse_dt
from .base import SourceInfo
from .static_index import StaticIndex

BUCKET = "https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com"
PREFIX = "sar-data/tasks/"


def _url(key: str) -> str:
    return f"{BUCKET}/{quote(key)}"


def _list_keys() -> list[str]:
    keys: list[str] = []
    token = None
    with client() as c:
        while True:
            params = {"list-type": "2", "prefix": PREFIX, "max-keys": "1000"}
            if token:
                params["continuation-token"] = token
            r = c.get(BUCKET + "/", params=params)
            r.raise_for_status()
            keys += [html.unescape(k) for k in re.findall(r"<Key>([^<]*)</Key>", r.text)]
            m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", r.text)
            if not m:
                return keys
            token = html.unescape(m.group(1))


def _build() -> list[dict]:
    dirs: dict[str, dict[str, str]] = defaultdict(dict)
    for k in _list_keys():
        d, name = k.rsplit("/", 1)
        if name.endswith("_GEC.tif"):
            dirs[d]["gec"] = k
        elif ".stac.v2" in name and name.endswith(".json"):
            dirs[d].setdefault("stac", k)
        elif name.endswith("_METADATA.json"):
            dirs[d]["meta"] = k
    todo = {}
    for d, files in dirs.items():
        if "gec" in files and ("stac" in files or "meta" in files):
            todo[d] = _url(files.get("stac") or files["meta"])
    docs = fetch_many_json(todo.values(), concurrency=48, progress="umbra collects")
    recs = []
    for d, meta_url in todo.items():
        doc = docs.get(meta_url)
        if not doc:
            continue
        rec = _from_stac(doc) if "stac" in dirs[d] else _from_metadata(doc)
        if not rec:
            continue
        parts = d.split("/")
        rec.update({"id": parts[-1], "task": parts[2] if len(parts) > 4 else "", "h": _url(dirs[d]["gec"]), "u": meta_url})
        recs.append(rec)
    return recs


def _from_stac(d: dict) -> dict | None:
    p = d.get("properties", {})
    if not d.get("bbox") or not p.get("datetime"):
        return None
    res = [p.get("umbra:best_resolution_range_meters"), p.get("umbra:best_resolution_azimuth_meters")]
    res = [r for r in res if r]
    b = d["bbox"]
    return {
        "t": p["datetime"],
        "b": [b[0], b[1], b[3], b[4]] if len(b) == 6 else b,
        "g": d.get("geometry"),
        "r": round(max(res), 2) if res else 0.5,
        "pl": p.get("platform"),
        "i": p.get("view:incidence_angle"),
    }


def _from_metadata(d: dict) -> dict | None:
    try:
        c = d["collects"][0]
        ring = [pt[:2] for pt in c["footprintPolygonLla"]["coordinates"][0]]
    except (KeyError, IndexError, TypeError):
        return None
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    return {
        "t": c["startAtUTC"],
        "b": [min(xs), min(ys), max(xs), max(ys)],
        "g": {"type": "Polygon", "coordinates": [ring]},
        "r": d.get("targetIpr") or d.get("baseIpr") or 0.5,
        "pl": d.get("umbraSatelliteName"),
        "i": c.get("angleIncidenceDegrees"),
    }


INDEX = StaticIndex("umbra", _build)


class Umbra:
    info = SourceInfo(
        key="umbra",
        name="Umbra Open Data (X-band SAR)",
        sensor="sar",
        resolution="0.16-1 m",
        coverage="~8,000 collects over selected sites (ports, airbases, mines, disasters)",
        revisit="irregular; some sites revisited weekly, 2023-present",
        license="CC BY 4.0",
        best_gsd=0.16,
    )

    def search(self, aoi: AOI, start: datetime, end: datetime, *, max_cloud: float | None = None, limit: int = 50) -> list[Scene]:
        out = []
        for r in INDEX.records():
            if not bbox_intersects(tuple(r["b"]), aoi.bbox):
                continue
            t = parse_dt(r["t"])
            if not (start <= t <= end):
                continue
            out.append(
                Scene(
                    id=f"umbra-{r['id']}",
                    source="umbra",
                    platform=(r.get("pl") or "umbra").lower().replace("_", "-"),
                    sensor="sar",
                    datetime=t,
                    gsd=float(r["r"]),
                    bbox=tuple(r["b"]),
                    geometry=r.get("g"),
                    off_nadir=r.get("i"),
                    item_url=r["u"],
                    license="CC BY 4.0",
                    attribution="Umbra Space Open Data",
                    render=Render(kind="sar", hrefs=[r["h"]], power=False),
                    extra={"task": r.get("task")},
                )
            )
        out.sort(key=lambda s: s.datetime, reverse=True)
        return out[:limit]
