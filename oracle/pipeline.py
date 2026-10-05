"""The intelligence loop: imagery -> detections -> object database -> tracks -> events.

``process_site`` is idempotent: scenes already processed for a site/detector are
skipped, and tracks/events are rebuilt deterministically from everything stored. So it
can run on a schedule (``oracle sweep --loop``) over any number of sites.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from . import ais, analytics
from .detect import detect_ships
from .geo import AOI
from .http import log
from .imagery import NoData
from .models import Scene
from .observations import Observation
from .presets import DETECTORS, PRESETS, SOURCES
from .search import search
from .store import Store
from .tracking import track_site

_yolo_lock = threading.Lock()  # one CPU-hungry YOLO run at a time; I/O-bound work stays parallel
CLEAR_ENOUGH = 0.85  # a scene only counts as "looked and didn't see it" if this clear


@dataclass
class Site:
    name: str
    bbox: tuple[float, float, float, float]
    kind: str = "maritime"
    detectors: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    max_cloud: float = 40.0
    min_length: float = 40.0

    def __post_init__(self) -> None:
        self.detectors = self.detectors or DETECTORS.get(self.kind, ["ships"])
        self.sources = self.sources or sorted({s for d in self.detectors for s in SOURCES[d]})

    @property
    def area_m2(self) -> float:
        w, h = AOI(self.bbox).size_km()
        return max(w * h * 1e6, 1e4)

    def save(self, store: Store) -> None:
        store.save_site(
            self.name,
            self.bbox,
            self.kind,
            detectors=self.detectors,
            sources=self.sources,
            max_cloud=self.max_cloud,
            min_length=self.min_length,
        )

    @classmethod
    def from_row(cls, d: dict) -> Site:
        return cls(
            name=d["name"],
            bbox=tuple(d["bbox"]),
            kind=d.get("kind") or "maritime",
            detectors=d.get("detectors") or [],
            sources=d.get("sources") or [],
            max_cloud=d.get("max_cloud", 40.0),
            min_length=d.get("min_length", 40.0),
        )


def add_preset(store: Store, preset: str) -> list[Site]:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {', '.join(PRESETS)}")
    out = []
    for name, (lat, lon, r_km, kind) in PRESETS[preset].items():
        s = Site(name, AOI.from_point(lat, lon, r_km).bbox, kind)
        s.save(store)
        out.append(s)
    return out


def detector_for(scene: Scene, site: Site) -> str | None:
    if "ships" in site.detectors and (scene.source == "sentinel-2" or scene.sensor == "sar"):
        return "ships"
    if "objects" in site.detectors and scene.sensor == "optical" and scene.gsd <= 1.0 and scene.render.kind != "xyz":
        return "objects"
    return None


def run_detector(det: str, scene: Scene, aoi: AOI, site: Site) -> tuple[list[Observation], float]:
    """-> (observations, clear fraction of the AOI)."""
    if det == "ships":
        res = detect_ships(scene, aoi, min_length=site.min_length)
        return res.observations(), res.clear_fraction
    from .objdet import detect_objects

    with _yolo_lock:
        return detect_objects(scene, aoi), max(0.0, 1 - (scene.cloud_cover or 0) / 100)


def process_site(
    store: Store,
    site: Site,
    since: datetime,
    until: datetime | None = None,
    max_scenes: int = 8,
) -> dict:
    until = until or datetime.now(timezone.utc)
    aoi = AOI(site.bbox, site.name)
    summary = {"site": site.name, "new_scenes": 0, "new_observations": 0, "errors": []}
    try:
        res = search(aoi, since, until, sources=site.sources, max_cloud=site.max_cloud, limit=40, sort="date", min_coverage=0.3)
    except Exception as exc:  # noqa: BLE001
        summary["errors"].append(f"search: {exc}")
        return summary
    for k, e in res.errors.items():
        summary["errors"].append(f"{k}: {e}")
    todo = []
    for scene in res.scenes:
        det = detector_for(scene, site)
        if det and not store.processed(site.name, scene.id, det):
            todo.append((scene, det))
    for scene, det in todo[:max_scenes]:
        store.add_scene(scene)
        try:
            obs, clear = run_detector(det, scene, aoi, site)
            n = store.add_observations(obs, site.name)
            store.record_run(site.name, scene.id, det, n, clear=clear)
            summary["new_scenes"] += 1
            summary["new_observations"] += n
            log(f"[{site.name}] {scene.source} {scene.date} {det}: {n} objects ({clear:.0%} clear)")
        except (NoData, ValueError) as exc:
            store.record_run(site.name, scene.id, det, 0, status="ok", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - keep sweeping other scenes
            store.record_run(site.name, scene.id, det, 0, status="error", error=f"{type(exc).__name__}: {exc}")
            summary["errors"].append(f"{scene.id}: {exc}")
    summary.update(refresh_site(store, site))
    return summary


def refresh_site(store: Store, site: Site) -> dict:
    """Re-run AIS matching, tracking and event generation over everything stored for a site."""
    obs = store.observations(site=site.name)
    if store.conn.execute("SELECT 1 FROM ais LIMIT 1").fetchone():
        ais.match(store, [o for o in obs if not o.attrs.get("ais") and not o.attrs.get("dark")])
        obs = store.observations(site=site.name)
    runs = [r for r in store.runs(site.name) if r["status"] == "ok"]
    coverage, scene_times, clear = {}, {}, {}
    for r in runs:
        if r.get("scene_time"):
            scene_times[r["scene_id"]] = datetime.fromisoformat(r["scene_time"])
        clear[r["scene_id"]] = r.get("clear") if r.get("clear") is not None else 1.0
        if r.get("w") is not None and clear[r["scene_id"]] >= CLEAR_ENOUGH:
            w, s, e, n = site.bbox
            coverage[r["scene_id"]] = (max(w, r["w"]), max(s, r["s"]), min(e, r["e"]), min(n, r["n"]))
    tracks, alts = track_site(obs, site.area_m2, coverage)
    links = {o.id: (o.attrs["track_id"], o.attrs["link_prob"]) for t in tracks for o in t.obs}
    store.save_tracking(site.name, [t.summary(site.name) for t in tracks], links, alts)
    for t in tracks:
        for o in t.obs:
            o.attrs["alternatives"] = alts.get(o.id) or None
    events = analytics.generate(site.name, tracks, obs, scene_times, clear, coverage=coverage)
    new_events = store.replace_events(site.name, events)
    return {"tracks": len(tracks), "events": len(events), "new_events": new_events}


def sweep(
    store: Store,
    sites: list[Site] | None = None,
    lookback_days: int = 30,
    max_scenes: int = 8,
    workers: int = 4,
) -> list[dict]:
    sites = sites if sites is not None else [Site.from_row(r) for r in store.sites()]
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    out = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(process_site, store, s, since, None, max_scenes): s for s in sites}
        for f in as_completed(futs):
            try:
                out.append(f.result())
            except Exception as exc:  # noqa: BLE001
                out.append({"site": futs[f].name, "errors": [f"{type(exc).__name__}: {exc}"]})
    return sorted(out, key=lambda r: r["site"])
