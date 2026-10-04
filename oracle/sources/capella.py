"""Capella Space Open Data: X-band SAR, ~0.5 m spotlight, ~2,000+ collects worldwide.

Radar images through cloud, smoke and darkness. Licence CC BY 4.0.
"""

from __future__ import annotations

from datetime import datetime
from urllib.parse import urljoin

from ..geo import AOI, bbox_intersects
from ..http import fetch_many_json, get_json
from ..models import Render, Scene, parse_dt
from .base import SourceInfo
from .static_index import StaticIndex

GEO_COLLECTION = (
    "https://capella-open-data.s3.us-west-2.amazonaws.com/stac/"
    "capella-open-data-by-product-type/capella-open-data-geo/collection.json"
)


def _build() -> list[dict]:
    col = get_json(GEO_COLLECTION)
    urls = [urljoin(GEO_COLLECTION, link["href"]) for link in col["links"] if link["rel"] == "item"]
    docs = fetch_many_json(urls, progress="capella items")
    recs = []
    for url, d in docs.items():
        if not d:
            continue
        p = d["properties"]
        asset = next((a for k, a in d["assets"].items() if k in ("HH", "VV", "HV", "VH") and a.get("href")), None)
        if not asset:
            continue
        res = [p.get("capella:resolution_ground_range"), p.get("sar:resolution_azimuth")]
        res = [r for r in res if r]
        recs.append(
            {
                "id": d["id"],
                "t": p["datetime"],
                "b": d["bbox"],
                "g": d.get("geometry"),
                "h": asset["href"],
                "th": (d["assets"].get("thumbnail") or {}).get("href"),
                "r": round(max(res), 2) if res else p.get("sar:pixel_spacing_range", 0.5),
                "pl": p.get("platform"),
                "m": p.get("sar:instrument_mode"),
                "uc": p.get("capella:use_case"),
                "o": p.get("sat:orbit_state"),
                "i": p.get("view:incidence_angle"),
                "u": url,
            }
        )
    return recs


INDEX = StaticIndex("capella", _build)


class Capella:
    info = SourceInfo(
        key="capella",
        name="Capella Space Open Data (X-band SAR)",
        sensor="sar",
        resolution="0.5-1 m",
        coverage="~2,000+ selected sites worldwide (ports, airports, cities, disasters)",
        revisit="irregular, 2020-present",
        license="CC BY 4.0",
        best_gsd=0.5,
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
                    id=r["id"],
                    source="capella",
                    platform=r.get("pl") or "capella",
                    sensor="sar",
                    datetime=t,
                    gsd=float(r["r"]),
                    bbox=tuple(r["b"]),
                    geometry=r.get("g"),
                    off_nadir=r.get("i"),
                    thumbnail=r.get("th"),
                    item_url=r["u"],
                    license="CC BY 4.0",
                    attribution="Capella Space Open Data",
                    render=Render(kind="sar", hrefs=[r["h"]], power=False),
                    extra={"mode": r.get("m"), "use_case": r.get("uc"), "orbit_state": r.get("o")},
                )
            )
        out.sort(key=lambda s: s.datetime, reverse=True)
        return out[:limit]
