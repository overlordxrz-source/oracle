"""Federated search: query every source in parallel, normalize, rank."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

from .geo import AOI, bbox_overlap_fraction
from .models import Scene
from .sources import SOURCES

SORTS = ("best", "resolution", "date", "cloud")


@dataclass
class SearchResult:
    aoi: AOI
    scenes: list[Scene]
    errors: dict[str, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)


def search(
    aoi: AOI,
    start: datetime,
    end: datetime,
    *,
    sources: list[str] | None = None,
    max_cloud: float | None = None,
    limit: int = 30,
    sort: str = "best",
    sensor: str | None = None,
    min_coverage: float = 0.0,
) -> SearchResult:
    keys = sources or list(SOURCES)
    picked = {k: SOURCES[k] for k in keys if k in SOURCES}
    unknown = set(keys) - set(picked)
    if unknown:
        raise KeyError(f"unknown source(s): {', '.join(sorted(unknown))}; choose from {', '.join(SOURCES)}")
    if sensor:
        picked = {k: s for k, s in picked.items() if s.info.sensor == sensor}

    errors: dict[str, str] = {}
    counts: dict[str, int] = {}
    scenes: list[Scene] = []
    with ThreadPoolExecutor(max_workers=len(picked) or 1) as pool:
        futs = {k: pool.submit(s.search, aoi, start, end, max_cloud=max_cloud, limit=limit) for k, s in picked.items()}
        for k, f in futs.items():
            try:
                got = f.result()
            except Exception as exc:  # noqa: BLE001 - one source down must not sink the search
                errors[k] = f"{type(exc).__name__}: {exc}"
                continue
            counts[k] = len(got)
            scenes.extend(got)

    for s in scenes:
        s.extra["coverage"] = 1.0 if s.source == "wayback" else round(bbox_overlap_fraction(aoi.bbox, s.bbox), 3)
    scenes = merge_passes([s for s in scenes if s.extra["coverage"] > 0], aoi)
    scenes = [s for s in scenes if s.extra["coverage"] >= min_coverage]
    for k in counts:
        counts[k] = sum(s.source == k for s in scenes)
    return SearchResult(aoi, rank(scenes, sort), errors, counts)


PASS_WINDOW_S = 180  # granules of one overpass are seconds apart; revisits are hours+ apart


def merge_passes(scenes: list[Scene], aoi: AOI) -> list[Scene]:
    """Collapse catalog tiles from the same overpass (e.g. adjacent Sentinel-2 MGRS tiles
    over one AOI) into one scene: the best-covering tile, mosaicked with the rest."""
    out: list[Scene] = []
    by_key: dict[tuple, list[Scene]] = {}
    for s in sorted(scenes, key=lambda s: s.datetime):
        if s.source not in ("sentinel-2", "sentinel-1", "landsat"):
            out.append(s)
            continue
        key = (s.source, s.platform)
        groups = by_key.setdefault(key, [])
        if groups and (s.datetime - groups[-1][0].datetime).total_seconds() <= PASS_WINDOW_S:
            groups[-1].append(s)
        else:
            groups.append([s])
    for groups in by_key.values():
        for g in groups:
            out.append(_merge(g, aoi) if len(g) > 1 else g[0])
    return out


def _merge(group: list[Scene], aoi: AOI) -> Scene:
    group = sorted(group, key=lambda s: s.extra["coverage"], reverse=True)
    best = group[0]
    if best.render.kind in ("rgb8", "sar"):  # one file per footprint: mosaic them
        for s in group[1:]:
            best.render.hrefs.extend(h for h in s.render.hrefs if h not in best.render.hrefs)
        best.extra["coverage"] = round(_union_coverage(aoi, [s.bbox for s in group]), 3)
        b = [s.bbox for s in group]
        best.bbox = (min(x[0] for x in b), min(x[1] for x in b), max(x[2] for x in b), max(x[3] for x in b))
        best.geometry = None
    best.extra["merged_ids"] = [s.id for s in group[1:]]
    if best.extra.get("bands"):  # per-band mosaics for analysis (change detection)
        best.extra["band_mosaic"] = [s.extra["bands"] for s in group[1:] if s.extra.get("bands")]
    return best


def _union_coverage(aoi: AOI, bboxes: list[tuple], n: int = 24) -> float:
    w, s, e, nn = aoi.bbox
    hit = 0
    for i in range(n):
        y = s + (i + 0.5) * (nn - s) / n
        for j in range(n):
            x = w + (j + 0.5) * (e - w) / n
            hit += any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in bboxes)
    return hit / (n * n)


def rank(scenes: list[Scene], sort: str = "best") -> list[Scene]:
    if sort not in SORTS:
        raise ValueError(f"sort must be one of {SORTS}")
    if sort == "date":
        key = lambda s: -s.datetime.timestamp()  # noqa: E731
    elif sort == "resolution":
        key = lambda s: (s.gsd, -s.datetime.timestamp())  # noqa: E731
    elif sort == "cloud":
        key = lambda s: (s.cloud_cover or 0, s.gsd, -s.datetime.timestamp())  # noqa: E731
    else:
        # Resolution tier (each halving of GSD is a tier), then covers most of the AOI,
        # then usable (<50% cloud), then newest.
        key = lambda s: (  # noqa: E731
            round(math.log2(max(s.gsd, 0.05))),
            s.extra.get("coverage", 1.0) < 0.5,
            (s.cloud_cover or 0) > 50,
            -s.datetime.timestamp(),
        )
    return sorted(scenes, key=key)
