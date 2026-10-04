"""Watchlist: re-check places of interest and report every new image (and its ships).

State lives in ``$ORACLE_CACHE/watch``: ``watchlist.json`` (what to watch),
``seen.json`` (scene ids already reported) and one folder of PNGs per site.
Optional alerts go to a webhook (Discord, Slack or anything that accepts JSON).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

import httpx

from .config import USER_AGENT, WATCH_DIR
from .detect import detect_ships
from .geo import AOI
from .http import log
from .imagery import NoData, chip
from .search import search


@dataclass
class Site:
    name: str
    bbox: tuple[float, float, float, float]
    sources: list[str] = field(default_factory=lambda: ["sentinel-2", "sentinel-1", "maxar", "umbra", "capella"])
    max_cloud: float | None = 40.0
    ships: bool = False
    min_length: float = 50.0


def _load(name: str, default):
    p = WATCH_DIR / name
    return json.loads(p.read_text()) if p.exists() else default


def _save(name: str, data) -> None:
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    tmp = WATCH_DIR / (name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(WATCH_DIR / name)


def sites() -> list[Site]:
    return [Site(**{**d, "bbox": tuple(d["bbox"])}) for d in _load("watchlist.json", [])]


def add(site: Site) -> None:
    current = [s for s in sites() if s.name != site.name]
    _save("watchlist.json", [asdict(s) for s in current + [site]])


def remove(name: str) -> bool:
    current = sites()
    kept = [s for s in current if s.name != name]
    _save("watchlist.json", [asdict(s) for s in kept])
    return len(kept) != len(current)


def run_once(lookback_days: int = 14, webhook: str | None = None) -> list[dict]:
    """Check every site; return (and alert on) scenes not seen before."""
    webhook = webhook or os.environ.get("ORACLE_WEBHOOK")
    seen: dict[str, list[str]] = _load("seen.json", {})
    now = datetime.now(timezone.utc)
    events = []
    for site in sites():
        aoi = AOI(site.bbox, site.name)
        try:
            res = search(
                aoi,
                now - timedelta(days=lookback_days),
                now,
                sources=site.sources,
                max_cloud=site.max_cloud,
                limit=20,
                sort="date",
                min_coverage=0.3,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"[{site.name}] search failed: {exc}")
            continue
        for src, err in res.errors.items():
            log(f"[{site.name}] {src}: {err}")
        known = set(seen.get(site.name, []))
        for scene in res.scenes:
            if scene.id in known or scene.render.kind == "xyz":
                continue
            ev = {
                "site": site.name,
                "scene": scene.id,
                "source": scene.source,
                "datetime": scene.datetime.isoformat(),
                "gsd": scene.gsd,
                "cloud": scene.cloud_cover,
            }
            out_dir = WATCH_DIR / _slug(site.name)
            try:
                c = chip(scene, aoi, max_pixels=2048)
                ev["image"] = str(c.save_png(out_dir / f"{scene.date}_{scene.source}_{_slug(scene.id)[:48]}.png"))
            except NoData as exc:
                ev["image_error"] = str(exc)
            if site.ships and (scene.source == "sentinel-2" or scene.sensor == "sar"):
                try:
                    det = detect_ships(scene, aoi, min_length=site.min_length)
                    ev["ships"] = len(det.detections)
                    ev["ships_250m"] = sum(d.length_m >= 250 for d in det.detections)
                    paths = det.save(out_dir, f"{scene.date}_{scene.source}_ships")
                    ev["ships_geojson"] = str(paths["geojson"])
                except (ValueError, NoData) as exc:
                    ev["ships_error"] = str(exc)
            known.add(scene.id)
            events.append(ev)
            log("  NEW " + _describe(ev))
            if webhook:
                _notify(webhook, ev)
        seen[site.name] = sorted(known)
    _save("seen.json", seen)
    return events


def run_forever(interval_s: int, lookback_days: int = 14, webhook: str | None = None) -> None:
    while True:
        log(f"[watch] checking {len(sites())} site(s) at {datetime.now(timezone.utc):%Y-%m-%d %H:%M}Z")
        run_once(lookback_days, webhook)
        time.sleep(interval_s)


def _describe(ev: dict) -> str:
    s = f"{ev['site']}: {ev['source']} {ev['datetime'][:16]}Z {ev['gsd']:g} m"
    if ev.get("cloud") is not None:
        s += f" cloud {ev['cloud']:.0f}%"
    if "ships" in ev:
        s += f" | {ev['ships']} vessels ({ev['ships_250m']} >= 250 m)"
    return s


def _notify(url: str, ev: dict) -> None:
    text = "Oracle: new image - " + _describe(ev)
    try:
        httpx.post(url, json={"content": text, "text": text, "event": ev}, headers={"User-Agent": USER_AGENT}, timeout=20)
    except httpx.HTTPError as exc:
        log(f"  webhook failed: {exc}")


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


__all__ = ["Site", "sites", "add", "remove", "run_once", "run_forever"]
