"""Source interface and the generic STAC API search used by the dynamic catalogs."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from ..geo import AOI
from ..http import client
from ..models import Scene


@dataclass(frozen=True)
class SourceInfo:
    key: str
    name: str
    sensor: str  # optical | sar
    resolution: str  # human readable, e.g. "10 m"
    coverage: str
    revisit: str
    license: str
    best_gsd: float  # metres, used to order sources


class Source(Protocol):
    info: SourceInfo

    def search(
        self,
        aoi: AOI,
        start: datetime,
        end: datetime,
        *,
        max_cloud: float | None = None,
        limit: int = 50,
    ) -> list[Scene]: ...


def iso_range(start: datetime, end: datetime) -> str:
    return f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end.strftime('%Y-%m-%dT%H:%M:%SZ')}"


def stac_search(
    api: str,
    collections: list[str],
    aoi: AOI,
    start: datetime,
    end: datetime,
    *,
    query: dict[str, Any] | None = None,
    max_items: int = 200,
) -> Iterator[dict]:
    """POST /search against a STAC API, following ``next`` links, newest first."""
    body: dict[str, Any] = {
        "collections": collections,
        "intersects": aoi.geometry,
        "datetime": iso_range(start, end),
        "limit": min(max_items, 100),
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }
    if query:
        body["query"] = query
    seen = 0
    url = api.rstrip("/") + "/search"
    method = "POST"
    with client() as c:
        while True:
            r = c.post(url, json=body) if method == "POST" else c.get(url)
            r.raise_for_status()
            page = r.json()
            for feat in page.get("features", []):
                yield feat
                seen += 1
                if seen >= max_items:
                    return
            nxt = next((link for link in page.get("links", []) if link.get("rel") == "next"), None)
            if not nxt or not page.get("features"):
                return
            url = nxt["href"]
            method = nxt.get("method", "GET").upper()
            if method == "POST":
                # Earth Search and Planetary Computer both hand back the full body plus a
                # paging token; merging is correct for either style.
                body = {**body, **(nxt.get("body") or {})}


def asset_href(item: dict, *keys: str) -> str | None:
    for k in keys:
        a = item.get("assets", {}).get(k)
        if a and a.get("href"):
            return a["href"]
    return None
