"""Oracle's capabilities as tools: what the agent (or a script) can ask the system to do.

Every tool takes JSON arguments and returns a JSON result. Every fact a tool produces is
registered as numbered evidence (E1, E2, ...) so conclusions keep a provenance chain:

  observation  a direct measurement in a specific image (one detection, with its chip)
  derived      an algorithm's output over observations (tracks, change regions, events,
               counts, anomalies); it inherits the uncertainty of what it was built from
  reference    catalog, orbital and gazetteer facts (which images exist, next passes)

Results quote the evidence ids next to the facts. Answers cite them as [E7], and the
agent checks every citation against this registry before finishing.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .geo import AOI, parse_aoi
from .models import Scene, parse_dt
from .store import Store

MAX_RESULT_CHARS = 14_000


class ToolError(Exception):
    """A tool could not do what was asked; the message goes back to the caller."""


@dataclass
class Evidence:
    id: str
    kind: str  # observation | derived | reference
    tool: str
    summary: str
    data: dict = field(default_factory=dict)
    time: str | None = None
    lat: float | None = None
    lon: float | None = None
    links: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- schemas

_WHERE = {
    "where": {"type": "string", "description": "Place name, 'lat,lon', or 'west,south,east,north' bbox."},
    "radius_km": {"type": "number", "description": "Half-width of the square around a point or place."},
}
_DATE = {"type": "string", "description": "YYYY-MM-DD (UTC)."}


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


TOOLS: list[dict] = [
    {
        "name": "geocode",
        "description": "Resolve a place name to coordinates and a bounding box (OpenStreetMap). Cheap. Use it first "
        "when the question names a place, so later tools can use precise 'lat,lon'.",
        "input_schema": _schema({"place": {"type": "string"}, "radius_km": _WHERE["radius_km"]}, ["place"]),
    },
    {
        "name": "search_imagery",
        "description": "List satellite/aerial images covering an area in a time window, newest first: Sentinel-2 "
        "(10 m optical, every 2-5 days), Sentinel-1 (10 m radar, day/night/cloud, every 6-12 days), Landsat (30 m), "
        "Maxar open data (0.3-0.5 m, disaster events only), NAIP (0.6 m, USA only), Umbra/Capella (sub-metre "
        "radar, sparse), Esri Wayback (sub-metre, view-only, no analysis). Returns scene ids usable by detect_* "
        "tools. Cheap (a few seconds).",
        "input_schema": _schema(
            {
                **_WHERE,
                "start": _DATE,
                "end": _DATE,
                "days_back": {"type": "number", "description": "Alternative to start/end: the last N days."},
                "sources": {"type": "array", "items": {"type": "string"}},
                "max_cloud": {"type": "number"},
            },
            ["where"],
        ),
    },
    {
        "name": "detect_ships",
        "description": "Run the vessel detector on one Sentinel-2 or Sentinel-1 image (newest clear one unless "
        "scene_id/date is given). Finds hulls >= ~40 m with length, heading axis, and (Sentinel-2) underway/course "
        "from wakes; 10 m pixels cannot identify ship types. Results are stored, tracked across dates and checked "
        "for events (arrivals, dark vessels, ship-to-ship rendezvous, unusual anchorage). 20-90 s. Max ~50 km radius.",
        "input_schema": _schema(
            {
                **_WHERE,
                "source": {"type": "string", "enum": ["sentinel-2", "sentinel-1"]},
                "scene_id": {"type": "string"},
                "date": _DATE,
                "site": {"type": "string", "description": "Store under this monitored site name."},
            },
            ["where"],
        ),
    },
    {
        "name": "detect_objects",
        "description": "Run YOLO object detection (aircraft, helicopters, vessels, cars, trucks, storage tanks, plus "
        "optional open-vocabulary prompts) on the newest sub-metre image (Maxar open data or NAIP). Only works "
        "where such imagery exists; check search_imagery first. Slow (1-5 min); keep radius <= 1 km.",
        "input_schema": _schema(
            {
                **_WHERE,
                "scene_id": {"type": "string"},
                "date": _DATE,
                "classes": {"type": "array", "items": {"type": "string"}},
                "prompts": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['tank', 'tent']"},
                "site": {"type": "string"},
            },
            ["where"],
        ),
    },
    {
        "name": "detect_change",
        "description": "What physically changed on the ground: compares the newest clear image with a multi-image "
        "baseline. Sentinel-2 classes: new_water, water_loss (reclamation), burn, vegetation_loss (clearing, "
        "earthworks), new_bright_surface (construction, pads, containers), surface_change. Sentinel-1 (all-weather): "
        "radar_increase (new structures/metal), radar_decrease (removal, flooding). 15-60 s. Max ~20 km radius. "
        "baseline='anniversary' compares with the same season last year.",
        "input_schema": _schema(
            {
                **_WHERE,
                "source": {"type": "string", "enum": ["sentinel-2", "sentinel-1"]},
                "after": _DATE,
                "before": _DATE,
                "baseline": {"type": "string", "enum": ["recent", "anniversary"]},
                "site": {"type": "string"},
            },
            ["where"],
        ),
    },
    {
        "name": "query_objects",
        "description": "Counts of stored detections per image date, source and class for an area or site (from "
        "earlier detector runs and monitoring sweeps). Cheap; use before running detectors to see existing history.",
        "input_schema": _schema(
            {
                **_WHERE,
                "site": {"type": "string"},
                "days": {"type": "number"},
                "classes": {"type": "array", "items": {"type": "string"}},
            },
            [],
        ),
    },
    {
        "name": "get_tracks",
        "description": "Objects tracked across dates in an area/site: class, length, sightings, first/last seen, "
        "status, displacement, speed between images and the mean same-object probability. Cheap.",
        "input_schema": _schema(
            {
                **_WHERE,
                "site": {"type": "string"},
                "cls": {"type": "string"},
                "min_obs": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            [],
        ),
    },
    {
        "name": "get_track",
        "description": "One track in full: every sighting (time, sensor, position, length, underway, AIS match) "
        "with the probability that it is the same object, plus competing hypotheses. Cheap.",
        "input_schema": _schema({"track_id": {"type": "string"}}, ["track_id"]),
    },
    {
        "name": "get_events",
        "description": "Ranked events already derived for an area/site: arrivals/departures, count anomalies, dark "
        "vessels (no AIS), loitering, fast movers, ship-to-ship rendezvous candidates, unusual locations, and "
        "change detections. Cheap; check this early.",
        "input_schema": _schema(
            {
                **_WHERE,
                "site": {"type": "string"},
                "days": {"type": "number"},
                "kinds": {"type": "array", "items": {"type": "string"}},
                "min_severity": {"type": "number"},
            },
            [],
        ),
    },
    {
        "name": "next_passes",
        "description": "When Sentinel-2, Sentinel-1 and Landsat can next image a point (SGP4 orbit prediction on "
        "public TLEs, checked against each sensor's swath geometry). Use for collection gaps and 'when will we know'.",
        "input_schema": _schema(
            {"lat": {"type": "number"}, "lon": {"type": "number"}, "days": {"type": "number"}}, ["lat", "lon"]
        ),
    },
    {
        "name": "wayback_history",
        "description": "Dates of distinct sub-metre captures (Esri World Imagery Wayback) at a point: for viewing "
        "history by eye in the app; not analysable. Cheap.",
        "input_schema": _schema({"lat": {"type": "number"}, "lon": {"type": "number"}}, ["lat", "lon"]),
    },
    {
        "name": "list_sites",
        "description": "Monitored sites with stored object counts and the highest event severity in 30 days.",
        "input_schema": _schema({}, []),
    },
    {
        "name": "monitor_site",
        "description": "Add an area as a monitored site and process its recent imagery now (detectors, tracking, "
        "events). Use to build history for a place that has none. Slow (1-5 min for 4 images).",
        "input_schema": _schema(
            {
                "name": {"type": "string"},
                **_WHERE,
                "kind": {"type": "string", "enum": ["maritime", "naval", "airbase", "ground"]},
                "lookback_days": {"type": "integer"},
                "max_scenes": {"type": "integer"},
            },
            ["name", "where"],
        ),
    },
    {
        "name": "ais_positions",
        "description": "AIS ship positions loaded into Oracle (only if the user imported AIS CSVs) in an area and "
        "time window: MMSI, name, type, speed, course. Lets you name vessels that broadcast, and spot those that don't.",
        "input_schema": _schema({**_WHERE, "start": _DATE, "end": _DATE}, ["where"]),
    },
]
TOOL_NAMES = {t["name"] for t in TOOLS}


# --------------------------------------------------------------------------- toolbox


class Toolbox:
    def __init__(self, store: Store | None = None):
        self.store = store or Store()
        self.evidence: dict[str, Evidence] = {}
        self.scenes: dict[str, Scene] = {}

    # ---------------------------------------------------------------- evidence
    def add(self, kind: str, tool: str, summary: str, **kw: Any) -> str:
        eid = f"E{len(self.evidence) + 1}"
        self.evidence[eid] = Evidence(eid, kind, tool, summary, **kw)
        return eid

    # ---------------------------------------------------------------- dispatch
    def call(self, name: str, args: dict) -> dict:
        if name not in TOOL_NAMES:
            raise ToolError(f"unknown tool {name!r}")
        spec = next(t for t in TOOLS if t["name"] == name)
        args = dict(args or {})
        props = spec["input_schema"]["properties"]
        for k in spec["input_schema"]["required"]:
            if args.get(k) in (None, ""):
                raise ToolError(f"{name}: missing required argument {k!r}")
        unknown = set(args) - set(props)
        if unknown:
            raise ToolError(f"{name}: unknown argument(s) {', '.join(sorted(unknown))}")
        for k, v in list(args.items()):
            t = props[k].get("type")
            try:
                if t == "number" and v is not None:
                    args[k] = float(v)
                elif t == "integer" and v is not None:
                    args[k] = int(v)
            except (TypeError, ValueError):
                raise ToolError(f"{name}: {k} must be a {t}") from None
        try:
            out = getattr(self, f"t_{name}")(**args)
        except ToolError:
            raise
        except (ValueError, KeyError, LookupError) as exc:
            raise ToolError(f"{name}: {exc}") from None
        return out

    @staticmethod
    def compact(result: dict) -> str:
        """JSON for the model, trimmed to a sane size (long lists are cut, with a note)."""
        text = json.dumps(result, default=str, separators=(",", ":"))
        if len(text) <= MAX_RESULT_CHARS:
            return text
        r = dict(result)
        for k, v in sorted(r.items(), key=lambda kv: -len(json.dumps(kv[1], default=str))):
            if isinstance(v, list) and len(v) > 5:
                keep = max(5, len(v) * MAX_RESULT_CHARS // len(text) // 2)
                r[k] = v[:keep]
                r[f"{k}_truncated"] = f"showing {keep} of {len(v)}"
            text = json.dumps(r, default=str, separators=(",", ":"))
            if len(text) <= MAX_RESULT_CHARS:
                break
        return text[: MAX_RESULT_CHARS * 2]

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _aoi(where: str | None, radius_km: float | None, default_r: float, max_r: float) -> AOI:
        if not where:
            raise ToolError("give 'where' (place, 'lat,lon' or bbox)")
        r = float(radius_km or default_r)
        if r > max_r:
            raise ToolError(f"radius_km {r:g} too large for this tool (max {max_r:g})")
        aoi = parse_aoi(where, r)
        w, h = aoi.size_km()
        if max(w, h) > 2 * max_r + 1:
            raise ToolError(f"area {w:.0f} x {h:.0f} km too large for this tool (max ~{2 * max_r:g} km across)")
        return aoi

    @staticmethod
    def _window(start: str | None, end: str | None, days_back: float | None, default_days: float) -> tuple[datetime, datetime]:
        now = datetime.now(timezone.utc)
        t1 = parse_dt(end + "T23:59:59Z") if end else now
        t0 = parse_dt(start + "T00:00:00Z") if start else t1 - timedelta(days=days_back or default_days)
        if t0 >= t1:
            raise ToolError("start must be before end")
        return t0, t1

    def _site_for(self, site: str | None, aoi: AOI, kind: str) -> str:
        from .pipeline import Site

        rows = {s["name"]: s for s in self.store.sites()}
        if site and site in rows:
            return site
        name = site or f"agent {aoi.center[0]:.3f},{aoi.center[1]:.3f}"
        if name not in rows:
            Site(name, aoi.bbox, kind).save(self.store)
        return name

    def _scene_for(self, aoi: AOI, scene_id: str | None, date: str | None, sources: list[str], optical_hr: bool = False) -> Scene:
        from .search import search

        if scene_id:
            if scene_id in self.scenes:
                return self.scenes[scene_id]
            s = self.store.scene(scene_id)
            if s:
                return s
        if date:
            t0, t1 = parse_dt(date + "T00:00:00Z"), parse_dt(date + "T23:59:59Z")
        else:
            t1 = datetime.now(timezone.utc)
            t0 = t1 - timedelta(days=3650 if optical_hr else 45)
        res = search(aoi, t0, t1, sources=sources, limit=40, sort="date", min_coverage=0.3)
        scenes = res.scenes
        if optical_hr:
            scenes = [s for s in scenes if s.sensor == "optical" and s.gsd <= 1.0 and s.render.kind != "xyz"]
        else:
            scenes = [s for s in scenes if (s.cloud_cover or 0) <= 40]
        if scene_id:
            scenes = [s for s in scenes if s.id == scene_id] or scenes
        if not scenes:
            raise ToolError(f"no suitable {'/'.join(sources)} image for that area and time")
        for s in scenes:
            self.scenes[s.id] = s
        return scenes[0]

    def _refresh(self, site: str) -> dict:
        from .pipeline import Site, refresh_site

        row = next(s for s in self.store.sites() if s["name"] == site)
        return refresh_site(self.store, Site.from_row(row))

    def _obs_evidence(self, tool: str, o, extra: str = "") -> str:
        size = f"{o.length_m:.0f} m " if o.length_m else ""
        return self.add(
            "observation",
            tool,
            f"{size}{o.cls} at {o.lat:.5f},{o.lon:.5f} on {o.time:%Y-%m-%d %H:%M}Z ({o.source}, conf {o.confidence:.2f}){extra}",
            data={"obs_id": o.id, "scene_id": o.scene_id, "length_m": o.length_m, "course_deg": o.course_deg},
            time=o.time.isoformat(),
            lat=o.lat,
            lon=o.lon,
            links={"chip": f"/api/obs/{o.id}/chip.png"},
        )

    # ---------------------------------------------------------------- tools
    def t_geocode(self, place: str, radius_km: float | None = None) -> dict:
        from .geo import geocode

        aoi = geocode(place, radius_km or 5.0)
        lat, lon = aoi.center
        eid = self.add("reference", "geocode", f"{place} -> {lat:.5f},{lon:.5f} ({aoi.name[:80]})", lat=lat, lon=lon)
        return {"place": aoi.name, "lat": round(lat, 6), "lon": round(lon, 6), "bbox": aoi.bbox, "evidence": eid}

    def t_search_imagery(
        self,
        where: str,
        radius_km: float | None = None,
        start: str | None = None,
        end: str | None = None,
        days_back: float | None = None,
        sources: list[str] | None = None,
        max_cloud: float | None = None,
    ) -> dict:
        from .search import search

        aoi = self._aoi(where, radius_km, 5.0, 100.0)
        t0, t1 = self._window(start, end, days_back, 30)
        res = search(aoi, t0, t1, sources=sources or None, max_cloud=max_cloud, limit=30, sort="date", min_coverage=0.3)
        rows = []
        for s in res.scenes[:40]:
            self.scenes[s.id] = s
            rows.append(
                {
                    "scene_id": s.id,
                    "source": s.source,
                    "sensor": s.sensor,
                    "time": s.datetime.isoformat(timespec="minutes"),
                    "gsd_m": s.gsd,
                    "cloud": None if s.cloud_cover is None else round(s.cloud_cover),
                    "coverage": s.extra.get("coverage"),
                    "analysable": s.render.kind != "xyz",
                }
            )
        by_source = Counter(r["source"] for r in rows)
        newest = {}
        for r in rows:
            newest.setdefault(r["source"], r["time"][:10])
        eid = self.add(
            "reference",
            "search_imagery",
            f"{len(rows)} images {t0:%Y-%m-%d}..{t1:%Y-%m-%d}: "
            + ", ".join(f"{k} {v} (newest {newest[k]})" for k, v in by_source.most_common()),
            data={"by_source": dict(by_source), "newest": newest},
            lat=aoi.center[0],
            lon=aoi.center[1],
        )
        return {"window": [t0.date().isoformat(), t1.date().isoformat()], "scenes": rows, "errors": res.errors, "evidence": eid}

    def t_detect_ships(
        self,
        where: str,
        radius_km: float | None = None,
        source: str | None = None,
        scene_id: str | None = None,
        date: str | None = None,
        site: str | None = None,
    ) -> dict:
        from .detect import detect_ships

        aoi = self._aoi(where, radius_km, 5.0, 50.0)
        src = source or "sentinel-2"
        scene = self._scene_for(aoi, scene_id, date, [src])
        if scene.source not in ("sentinel-2", "sentinel-1") and scene.sensor != "sar":
            raise ToolError(f"detect_ships runs on sentinel-2 or radar scenes, not {scene.source}")
        t = time.time()
        res = detect_ships(scene, aoi)
        obs = res.observations()
        name = self._site_for(site, aoi, "maritime")
        self.store.add_scene(scene)
        n = self.store.add_observations(obs, name)
        self.store.record_run(name, scene.id, "ships", n, clear=res.clear_fraction)
        refresh = self._refresh(name)
        stored = {o.id: o for o in self.store.observations(site=name, bbox=aoi.bbox) if o.scene_id == scene.id}
        obs = sorted(stored.values(), key=lambda o: -(o.length_m or 0))
        bins = Counter(_len_bin(o.length_m) for o in obs)
        underway = sum(1 for o in obs if o.attrs.get("underway"))
        big = sum(1 for o in obs if (o.length_m or 0) >= 200)
        summary = self.add(
            "derived",
            "detect_ships",
            f"{len(obs)} vessels in {scene.source} {scene.datetime:%Y-%m-%d %H:%M}Z "
            f"({res.clear_fraction:.0%} of area clear): {big} of 200 m or more, {underway} underway, "
            f"{len(obs) - underway} stationary or unknown",
            data={"scene_id": scene.id, "count": len(obs), "length_bins": dict(bins), "clear": res.clear_fraction},
            time=scene.datetime.isoformat(),
            lat=aoi.center[0],
            lon=aoi.center[1],
        )
        top = []
        for o in obs[:15]:
            extra = ""
            if o.attrs.get("underway"):
                extra = f", underway course {o.course_deg:.0f}" if o.course_deg is not None else ", underway"
            top.append(
                {
                    "evidence": self._obs_evidence("detect_ships", o, extra),
                    "lat": round(o.lat, 5),
                    "lon": round(o.lon, 5),
                    "length_m": o.length_m,
                    "underway": o.attrs.get("underway"),
                    "course_deg": o.course_deg,
                    "confidence": round(o.confidence, 2),
                    "track_id": o.attrs.get("track_id"),
                }
            )
        return {
            "scene": {"id": scene.id, "source": scene.source, "time": scene.datetime.isoformat(timespec="minutes")},
            "site": name,
            "clear_fraction": res.clear_fraction,
            "vessels": len(obs),
            "length_bins_m": dict(bins),
            "underway": underway,
            "largest": top,
            "notes": res.warnings,
            "tracking": refresh,
            "seconds": round(time.time() - t),
            "evidence": summary,
        }

    def t_detect_objects(
        self,
        where: str,
        radius_km: float | None = None,
        scene_id: str | None = None,
        date: str | None = None,
        classes: list[str] | None = None,
        prompts: list[str] | None = None,
        site: str | None = None,
    ) -> dict:
        from .objdet import detect_objects

        aoi = self._aoi(where, radius_km, 0.5, 1.5)
        scene = self._scene_for(aoi, scene_id, date, ["maxar", "naip", "umbra", "capella"], optical_hr=True)
        t = time.time()
        obs = detect_objects(scene, aoi, classes=classes or None, prompts=prompts or None)
        name = self._site_for(site, aoi, "ground")
        self.store.add_scene(scene)
        n = self.store.add_observations(obs, name)
        self.store.record_run(name, scene.id, "objects", n)
        self._refresh(name)
        counts = Counter(o.cls for o in obs)
        summary = self.add(
            "derived",
            "detect_objects",
            f"YOLO on {scene.source} {scene.gsd:g} m {scene.datetime:%Y-%m-%d}: "
            + (", ".join(f"{v} {k}" for k, v in counts.most_common()) or "nothing found"),
            data={"scene_id": scene.id, "counts": dict(counts)},
            time=scene.datetime.isoformat(),
            lat=aoi.center[0],
            lon=aoi.center[1],
        )
        order = {"aircraft": 0, "helicopter": 1, "vessel": 2, "large-vehicle": 3, "storage-tank": 4, "vehicle": 5}
        top = []
        for o in sorted(obs, key=lambda o: (order.get(o.cls, 9), -o.confidence))[:15]:
            top.append(
                {
                    "evidence": self._obs_evidence("detect_objects", o),
                    "cls": o.cls,
                    "label": o.attrs.get("label"),
                    "lat": round(o.lat, 6),
                    "lon": round(o.lon, 6),
                    "length_m": o.length_m,
                    "confidence": round(o.confidence, 2),
                }
            )
        return {
            "scene": {"id": scene.id, "source": scene.source, "gsd_m": scene.gsd, "time": scene.datetime.isoformat()},
            "site": name,
            "counts": dict(counts),
            "examples": top,
            "seconds": round(time.time() - t),
            "evidence": summary,
        }

    def t_detect_change(
        self,
        where: str,
        radius_km: float | None = None,
        source: str | None = None,
        after: str | None = None,
        before: str | None = None,
        baseline: str | None = None,
        site: str | None = None,
    ) -> dict:
        from .change import KINDS, detect_change, remember, to_events
        from .imagery import NoData

        aoi = self._aoi(where, radius_km, 3.0, 20.0)
        try:
            res = remember(
                detect_change(
                    aoi,
                    source or "sentinel-2",
                    after=parse_dt(after + "T23:59:59Z") if after else None,
                    before=parse_dt(before + "T23:59:59Z") if before else None,
                    baseline=baseline or "recent",
                )
            )
        except NoData as exc:
            raise ToolError(f"detect_change: {exc}") from None
        if site:
            self.store.add_events(to_events(res, site))
        s = res.summary()
        summary = self.add(
            "derived",
            "detect_change",
            f"{s['method']}: {res.after.datetime:%Y-%m-%d} vs baseline "
            f"{', '.join(b['time'][:10] for b in s['before'])}; {len(res.regions)} change regions "
            + ", ".join(f"{k} {v['regions']} ({v['area_m2'] / 1e4:.1f} ha)" for k, v in s["by_kind"].items()),
            data={"change_id": res.id, "by_kind": s["by_kind"], "valid_fraction": s["valid_fraction"]},
            time=res.after.datetime.isoformat(),
            lat=aoi.center[0],
            lon=aoi.center[1],
            links={"overlay": f"/api/change/{res.id}/change.png"},
        )
        regions = []
        for r in res.regions[:12]:
            bb = ",".join(f"{v:.6f}" for v in r.bbox)
            eid = self.add(
                "derived",
                "detect_change",
                f"{r.kind} {r.area_m2 / 1e4:.2f} ha at {r.lat:.5f},{r.lon:.5f} (conf {r.confidence:.0%}) "
                + "; ".join(f"{m} {v[0]:g}->{v[1]:g}" for m, v in r.values.items()),
                data={"kind": r.kind, "area_m2": r.area_m2, "values": r.values, "bbox": r.bbox},
                time=res.after.datetime.isoformat(),
                lat=r.lat,
                lon=r.lon,
                links={
                    "before": f"/api/change/{res.id}/before.png?bbox={bb}",
                    "after": f"/api/change/{res.id}/after.png?bbox={bb}",
                },
            )
            regions.append(
                {
                    "evidence": eid,
                    "kind": r.kind,
                    "meaning": KINDS[r.kind][0],
                    "area_ha": round(r.area_m2 / 1e4, 2),
                    "lat": round(r.lat, 5),
                    "lon": round(r.lon, 5),
                    "confidence": round(r.confidence, 2),
                    "values": r.values,
                }
            )
        return {
            **{k: s[k] for k in ("method", "after", "before", "valid_fraction", "by_kind", "notes")},
            "regions": regions,
            "evidence": summary,
        }

    def t_query_objects(
        self,
        where: str | None = None,
        radius_km: float | None = None,
        site: str | None = None,
        days: float | None = None,
        classes: list[str] | None = None,
    ) -> dict:
        if not (where or site):
            raise ToolError("give 'where' or 'site'")
        bbox = self._aoi(where, radius_km, 5.0, 200.0).bbox if where else None
        start = datetime.now(timezone.utc) - timedelta(days=days) if days else None
        obs = self.store.observations(bbox=bbox, site=site, start=start, classes=classes or None, limit=200_000)
        per: dict[tuple, Counter] = defaultdict(Counter)
        for o in obs:
            per[(o.time.strftime("%Y-%m-%d"), o.source)][o.cls] += 1
        series = [{"date": d, "source": s, "counts": dict(c)} for (d, s), c in sorted(per.items())]
        total = Counter(o.cls for o in obs)
        eid = self.add(
            "derived",
            "query_objects",
            f"{len(obs)} stored detections over {len(series)} image dates"
            + (f" ({', '.join(f'{v} {k}' for k, v in total.most_common(4))})" if total else ""),
            data={"total": dict(total), "dates": len(series)},
        )
        return {"total": dict(total), "series": series[-60:], "evidence": eid}

    def t_get_tracks(
        self,
        where: str | None = None,
        radius_km: float | None = None,
        site: str | None = None,
        cls: str | None = None,
        min_obs: int | None = None,
        limit: int | None = None,
    ) -> dict:
        bbox = self._aoi(where, radius_km, 5.0, 200.0).bbox if where else None
        rows = [t for t in self.store.tracks(site=site, bbox=bbox) if t["n_obs"] >= (min_obs or 2)]
        if cls:
            rows = [t for t in rows if t["cls"] == cls]
        rows.sort(key=lambda t: (-t["n_obs"], -(t["length_m"] or 0)))
        out = []
        for t in rows[: limit or 15]:
            eid = self.add(
                "derived",
                "get_tracks",
                f"track {t['id']}: {t['length_m'] or '?'} m {t['cls']}, {t['n_obs']} sightings "
                f"{t['first_seen'][:10]}..{t['last_seen'][:10]}, {t['status']}, mean same-object p "
                f"{(t['mean_link_prob'] or 0):.0%}",
                data={"track_id": t["id"]},
                time=t["last_seen"],
                lat=t["lat"],
                lon=t["lon"],
            )
            out.append(
                {
                    "evidence": eid,
                    "track_id": t["id"],
                    "cls": t["cls"],
                    "length_m": t["length_m"],
                    "sightings": t["n_obs"],
                    "first_seen": t["first_seen"][:16],
                    "last_seen": t["last_seen"][:16],
                    "status": t["status"],
                    "speed_kn": None if t["speed_ms"] is None else round(t["speed_ms"] * 1.944, 1),
                    "distance_km": None if t["distance_m"] is None else round(t["distance_m"] / 1000, 2),
                    "mean_link_prob": t["mean_link_prob"],
                    "site": t["site"],
                }
            )
        return {"tracks": out, "total_matching": len(rows)}

    def t_get_track(self, track_id: str) -> dict:
        t = self.store.track(track_id)
        if not t:
            raise ToolError(f"no track {track_id}")
        sightings = []
        for o in t["observations"]:
            a = o["attrs"]
            eid = self.add(
                "observation",
                "get_track",
                f"track {track_id} sighting {o['time'][:16]}Z {o['source']} at {o['lat']:.5f},{o['lon']:.5f}, "
                f"{o['length_m'] or '?'} m, same-object p={a.get('link_prob')}",
                data={"obs_id": o["id"]},
                time=o["time"],
                lat=o["lat"],
                lon=o["lon"],
                links={"chip": f"/api/obs/{o['id']}/chip.png"},
            )
            sightings.append(
                {
                    "evidence": eid,
                    "time": o["time"][:16],
                    "source": o["source"],
                    "lat": round(o["lat"], 5),
                    "lon": round(o["lon"], 5),
                    "length_m": o["length_m"],
                    "same_object_p": a.get("link_prob"),
                    "underway": a.get("underway"),
                    "ais": a.get("ais"),
                    "dark": a.get("dark"),
                }
            )
        return {
            "track_id": track_id,
            "cls": t["cls"],
            "status": t["status"],
            "sightings": sightings,
            "competing_hypotheses": sorted(t["alternatives"], key=lambda a: -a["prob"])[:8],
        }

    def t_get_events(
        self,
        where: str | None = None,
        radius_km: float | None = None,
        site: str | None = None,
        days: float | None = None,
        kinds: list[str] | None = None,
        min_severity: float | None = None,
    ) -> dict:
        bbox = self._aoi(where, radius_km, 10.0, 300.0).bbox if where else None
        since = datetime.now(timezone.utc) - timedelta(days=days or 30)
        rows = []
        for e in self.store.events(since=since, site=site, limit=5000):
            d = e["detail"]
            if bbox and not ("lat" in d and bbox[0] <= d["lon"] <= bbox[2] and bbox[1] <= d["lat"] <= bbox[3]):
                if not (e["site"] and self._site_in(e["site"], bbox)):
                    continue
            if kinds and not any(e["kind"].startswith(k) for k in kinds):
                continue
            if e["severity"] < (min_severity or 0):
                continue
            rows.append(e)
        out = []
        for e in rows[:25]:
            d = e["detail"]
            eid = self.add(
                "derived",
                "get_events",
                f"[{e['kind']}] {e['title']} ({e['time'][:16]}Z, severity {e['severity']:.2f})",
                data={"event_id": e["id"], "track_id": e.get("track_id"), "obs_id": e.get("obs_id")},
                time=e["time"],
                lat=d.get("lat"),
                lon=d.get("lon"),
                links={"chip": f"/api/obs/{e['obs_id']}/chip.png"} if e.get("obs_id") else {},
            )
            out.append(
                {
                    "evidence": eid,
                    "kind": e["kind"],
                    "title": e["title"],
                    "time": e["time"][:16],
                    "severity": e["severity"],
                    "site": e["site"],
                    "lat": d.get("lat"),
                    "lon": d.get("lon"),
                    "track_id": e.get("track_id"),
                    "detail": {k: v for k, v in d.items() if k not in ("lat", "lon", "alternatives", "track_ids", "values")},
                }
            )
        return {"events": out, "total_matching": len(rows), "window_days": days or 30}

    def _site_in(self, site: str, bbox: tuple) -> bool:
        for s in self.store.sites():
            if s["name"] == site:
                w, so, e, n = s["bbox"]
                return w <= bbox[2] and e >= bbox[0] and so <= bbox[3] and n >= bbox[1]
        return False

    def t_next_passes(self, lat: float, lon: float, days: float | None = None) -> dict:
        from .passes import next_passes

        try:
            ps = [p for p in next_passes(lat, lon, days=min(days or 7, 21)) if p.likely]
        except Exception as exc:  # noqa: BLE001 - orbit data download can fail
            raise ToolError(f"orbit prediction unavailable: {exc}") from None
        rows = []
        for p in ps[:20]:
            rows.append(
                {
                    "time": p.time.isoformat(timespec="minutes"),
                    "satellite": p.satellite,
                    "family": p.family,
                    "off_track_km": p.cross_track_km,
                    "direction": p.direction,
                    "sun_elevation": p.sun_elevation,
                }
            )
        first = {}
        for r in rows:
            first.setdefault(r["family"], r["time"])
        eid = self.add(
            "reference",
            "next_passes",
            "next likely looks: " + ", ".join(f"{k} {v[:16]}Z" for k, v in first.items()) if first else "no likely passes",
            data={"first": first},
            lat=lat,
            lon=lon,
        )
        return {"passes": rows, "first_by_family": first, "evidence": eid}

    def t_wayback_history(self, lat: float, lon: float) -> dict:
        from .sources.wayback import versions

        vs = versions(lat, lon)
        rows = [
            {
                "captured": v.get("captured"),
                "sensor": v.get("sensor"),
                "resolution_m": v.get("resolution"),
                "release": v.get("date"),
            }
            for v in vs
        ]
        eid = self.add(
            "reference",
            "wayback_history",
            f"{len(rows)} distinct sub-metre captures in Esri Wayback"
            + (f", newest {rows[0]['captured']}" if rows and rows[0]["captured"] else ""),
            lat=lat,
            lon=lon,
        )
        return {"captures": rows[:30], "note": "view-only basemap history (look at it in the app)", "evidence": eid}

    def t_list_sites(self) -> dict:
        st = self.store
        counts = dict(st.conn.execute("SELECT site, COUNT(*) FROM observations GROUP BY site").fetchall())
        sev: dict[str, float] = defaultdict(float)
        for e in st.events(since=datetime.now(timezone.utc) - timedelta(days=30), limit=5000):
            sev[e["site"]] = max(sev[e["site"]], e["severity"])
        rows = [
            {
                "name": s["name"],
                "kind": s.get("kind"),
                "bbox": s["bbox"],
                "observations": counts.get(s["name"], 0),
                "max_severity_30d": sev.get(s["name"], 0.0),
            }
            for s in st.sites()
        ]
        return {"sites": rows}

    def t_monitor_site(
        self,
        name: str,
        where: str,
        radius_km: float | None = None,
        kind: str | None = None,
        lookback_days: int | None = None,
        max_scenes: int | None = None,
    ) -> dict:
        from .pipeline import Site, process_site

        aoi = self._aoi(where, radius_km, 5.0, 50.0)
        site = Site(name, aoi.bbox, kind or "maritime")
        site.save(self.store)
        since = datetime.now(timezone.utc) - timedelta(days=lookback_days or 30)
        r = process_site(self.store, site, since, None, min(max_scenes or 4, 8))
        eid = self.add(
            "derived",
            "monitor_site",
            f"site {name}: processed {r.get('new_scenes', 0)} new images, {r.get('new_observations', 0)} objects, "
            f"{r.get('tracks', 0)} tracks, {r.get('events', 0)} events",
            data=r,
            lat=aoi.center[0],
            lon=aoi.center[1],
        )
        return {**r, "evidence": eid}

    def t_ais_positions(
        self, where: str, radius_km: float | None = None, start: str | None = None, end: str | None = None
    ) -> dict:
        aoi = self._aoi(where, radius_km, 5.0, 100.0)
        t0, t1 = self._window(start, end, None, 7)
        w, s, e, n = aoi.bbox
        rows = self.store.conn.execute(
            "SELECT mmsi, name, vtype, length_m, COUNT(*) n, MIN(time) t0, MAX(time) t1, AVG(sog) sog, "
            "AVG(lat) lat, AVG(lon) lon FROM ais WHERE time BETWEEN ? AND ? AND lon BETWEEN ? AND ? "
            "AND lat BETWEEN ? AND ? GROUP BY mmsi ORDER BY n DESC LIMIT 60",
            (t0.isoformat(), t1.isoformat(), w, e, s, n),
        ).fetchall()
        if not rows and not self.store.conn.execute("SELECT 1 FROM ais LIMIT 1").fetchone():
            raise ToolError("no AIS data loaded (the user can import CSVs with `oracle ais load`)")
        vessels = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in dict(r).items()} for r in rows]
        eid = self.add(
            "reference", "ais_positions", f"{len(vessels)} AIS-broadcasting vessels in the area {t0:%Y-%m-%d}..{t1:%Y-%m-%d}"
        )
        return {"vessels": vessels, "evidence": eid}


def _len_bin(L: float | None) -> str:
    if not L:
        return "unknown"
    for hi, label in ((50, "<50"), (100, "50-100"), (200, "100-200"), (300, "200-300")):
        if L < hi:
            return label
    return ">=300"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p = math.radians
    a = math.sin(p(lat2 - lat1) / 2) ** 2 + math.cos(p(lat1)) * math.cos(p(lat2)) * math.sin(p(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))
