"""Turn tracks and observations into ranked, human-readable events.

Event kinds (severity 0-1, higher = more interesting):
  arrival / departure      a tracked object appeared after the site's baseline image,
                           or stopped being seen in images that covered it
  count_spike / count_drop the number of objects of a class in a scene is far outside
                           the site's own history (robust z-score vs earlier scenes)
  dark_vessel              hull with no AIS position nearby while AIS coverage exists
  fast_vessel              track speed between observations above ~23 kn
  loitering                vessel present and stationary for 14+ days
  possible_task_group      >= 250 m hull with several 100-220 m hulls within 15 km
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from datetime import datetime

import numpy as np

from .observations import AIRCRAFT, HELICOPTER, VESSEL, Observation
from .tracking import Track, family


def _eid(*parts: object) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _size_word(o: Observation) -> str:
    return f"{o.length_m:.0f} m " if o.length_m else ""


def _cap(size: str) -> str:
    return size.strip().capitalize() or "A"


def _where(o: Observation) -> dict:
    return {"lat": o.lat, "lon": o.lon}


def _object_severity(o: Observation) -> float:
    if o.cls == VESSEL:
        L = o.length_m or 0
        return 0.85 if L >= 250 else 0.6 if L >= 150 else 0.45 if L >= 80 else 0.25
    if o.cls in (AIRCRAFT, HELICOPTER):
        return 0.6
    return 0.2


MAX_INDIVIDUAL = 4  # per scene; the rest of a busy turnover is rolled into one event


def generate(
    site: str,
    tracks: list[Track],
    obs: list[Observation],
    scene_times: dict[str, datetime],
    clear: dict[str, float] | None = None,
    clear_enough: float = 0.85,
) -> list[dict]:
    """Events for one site. ``clear``: scene_id -> cloud-free fraction of the site."""
    events: list[dict] = []
    if not obs:
        return events
    clear = clear or {}
    # Baseline per class family = first scene that imaged the site clearly. Anything first
    # seen *after* a clear look is an arrival; before that we can't tell.
    baseline: dict[str, datetime] = {}
    for o in obs:
        if clear.get(o.scene_id, 1.0) >= clear_enough:
            f = family(o.cls)
            baseline[f] = min(baseline.get(f, o.time), o.time)

    # In crowded waters (anchorages, straits) "big hull with medium hulls nearby" is noise.
    per_scene = defaultdict(int)
    for o in obs:
        if o.cls == VESSEL and clear.get(o.scene_id, 1.0) >= clear_enough:
            per_scene[o.scene_id] += 1
    crowded = bool(per_scene) and float(np.median(list(per_scene.values()))) > 12

    arrivals: dict[str, list[tuple[Track, Observation]]] = defaultdict(list)
    departures: dict[str, list[tuple[Track, Observation]]] = defaultdict(list)
    for t in tracks:
        first, last = t.obs[0], t.last
        if t.fam not in ("vessel", "aircraft"):
            continue
        if t.fam in baseline and first.time > baseline[t.fam]:
            arrivals[first.scene_id].append((t, first))
        if t.status == "departed":
            departures[last.scene_id].append((t, last))
    scale = 0.6 if crowded else 1.0  # turnover is routine in a busy anchorage or strait
    for sid, items in arrivals.items():
        events += _rollup(
            "arrival", site, sid, items, "New {size}{cls} at {site}", "{n} new {cls}s at {site} (largest {big})", scale
        )
    for sid, items in departures.items():
        events += _rollup(
            "departure",
            site,
            sid,
            items,
            "{Size} {cls} no longer seen at {site}",
            "{n} {cls}s no longer seen at {site} (largest {big})",
            scale,
        )

    for t in tracks:
        last = t.last
        s = t.summary(site)
        if t.fam == "vessel" and s["speed_ms"] and s["speed_ms"] > 12 and (s["mean_link_prob"] or 0) >= 0.6:
            events.append(
                {
                    "id": _eid("fast", t.id),
                    "site": site,
                    "time": last.time.isoformat(),
                    "kind": "fast_vessel",
                    "severity": 0.5,
                    "title": f"Fast {_size_word(last)}vessel ({s['speed_ms'] * 1.944:.0f} kn between images) at {site}",
                    "detail": {**_where(last), "speed_kn": round(s["speed_ms"] * 1.944, 1), "link_prob": s["mean_link_prob"]},
                    "obs_id": last.id,
                    "track_id": t.id,
                }
            )
        if t.fam == "vessel" and s["attrs"]["dwell_days"] >= 14 and (s["speed_ms"] or 0) < 0.3 and len(t.obs) >= 3:
            events.append(
                {
                    "id": _eid("loiter", t.id, len(t.obs)),
                    "site": site,
                    "time": last.time.isoformat(),
                    "kind": "loitering",
                    "severity": 0.35 + 0.15 * ((last.length_m or 0) >= 150),
                    "title": f"{_cap(_size_word(last))} vessel stationary for {s['attrs']['dwell_days']:.0f} days at {site}",
                    "detail": {**_where(last), "dwell_days": s["attrs"]["dwell_days"], "n_obs": len(t.obs)},
                    "obs_id": last.id,
                    "track_id": t.id,
                }
            )

    for o in obs:
        if o.attrs.get("dark"):
            events.append(
                {
                    "id": _eid("dark", o.id),
                    "site": site,
                    "time": o.time.isoformat(),
                    "kind": "dark_vessel",
                    "severity": 0.55 + 0.25 * ((o.length_m or 0) >= 150),
                    "title": f"{_size_word(o).strip().capitalize() or 'A'} vessel with no AIS at {site}",
                    "detail": {**_where(o), "nearest_ais_m": o.attrs.get("nearest_ais_m"), "source": o.source},
                    "obs_id": o.id,
                    "track_id": o.attrs.get("track_id"),
                }
            )
        if not crowded and o.cls == VESSEL and (o.length_m or 0) >= 250 and (o.attrs.get("nearby_medium_vessels") or 0) >= 2:
            events.append(
                {
                    "id": _eid("group", o.id),
                    "site": site,
                    "time": o.time.isoformat(),
                    "kind": "possible_task_group",
                    "severity": 0.4,
                    "title": f"{o.length_m:.0f} m hull with {o.attrs['nearby_medium_vessels']} medium hulls nearby at {site}",
                    "detail": {**_where(o), "note": "weak hint; meaningless in busy shipping lanes"},
                    "obs_id": o.id,
                    "track_id": o.attrs.get("track_id"),
                }
            )

    clear_times = {sid: t for sid, t in scene_times.items() if clear.get(sid, 1.0) >= clear_enough}
    events += count_anomalies(site, [o for o in obs if o.scene_id in clear_times], clear_times)
    return events


def _rollup(kind: str, site: str, sid: str, items: list, one: str, many: str, scale: float = 1.0) -> list[dict]:
    items = sorted(items, key=lambda x: -(x[1].length_m or 0))
    out = []
    for t, o in items[:MAX_INDIVIDUAL]:
        size = _size_word(o)
        sev = (_object_severity(o) - (0.1 if kind == "departure" else 0.0)) * scale
        out.append(
            {
                "id": _eid(kind, t.id),
                "site": site,
                "time": o.time.isoformat(),
                "kind": kind,
                "severity": round(max(sev, 0.1), 2),
                "title": one.format(size=size, Size=_cap(size), cls=o.cls, site=site),
                "detail": {
                    **_where(o),
                    "source": o.source,
                    "length_m": o.length_m,
                    "n_obs": len(t.obs),
                    "alternatives": o.attrs.get("alternatives"),
                },
                "obs_id": o.id,
                "track_id": t.id,
            }
        )
    if len(items) > MAX_INDIVIDUAL:
        o = items[0][1]
        big = f"{o.length_m:.0f} m" if o.length_m else "?"
        out.append(
            {
                "id": _eid(kind, "rollup", site, sid),
                "site": site,
                "time": o.time.isoformat(),
                "kind": kind + "s",
                "severity": round(min(0.3 + 0.02 * len(items), 0.6) * scale, 2),
                "title": many.format(n=len(items), cls=o.cls, site=site, big=big),
                "detail": {"count": len(items), "scene_id": sid, "track_ids": [t.id for t, _ in items]},
            }
        )
    return out


def count_anomalies(site: str, obs: list[Observation], scene_times: dict[str, datetime], min_history: int = 4) -> list[dict]:
    """Robust z-score of each scene's per-class count against the scenes before it.

    Scenes from different sensors aren't comparable (10 m vs 30 cm), so the series is
    per (class family, source).
    """
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for o in obs:
        counts[(family(o.cls), o.source)][o.scene_id] += 1
    out = []
    for (fam, source), per_scene in counts.items():
        series = [(sid, n) for sid, n in sorted(per_scene.items(), key=lambda kv: scene_times.get(kv[0], datetime.min))]
        if len(series) < min_history + 1:
            continue
        values = [v for _, v in series]
        for k in range(min_history, len(values)):
            hist = np.array(values[:k], float)
            med = float(np.median(hist))
            mad = float(np.median(np.abs(hist - med))) * 1.4826
            z = (values[k] - med) / (mad + math.sqrt(max(med, 1.0)))
            if abs(z) >= 3:
                sid = series[k][0]
                kind = "count_spike" if z > 0 else "count_drop"
                out.append(
                    {
                        "id": _eid(kind, site, fam, sid),
                        "site": site,
                        "time": scene_times[sid].isoformat(),
                        "kind": kind,
                        "severity": round(min(1.0, 0.3 + abs(z) / 10), 2),
                        "title": f"{fam.capitalize()} count {'up' if z > 0 else 'down'} to {values[k]} (usual ~{med:.0f}) at {site}",  # noqa: E501
                        "detail": {"count": values[k], "median": med, "z": round(z, 1), "source": source, "scene_id": sid},
                    }
                )
    return out
