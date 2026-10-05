"""Pattern-of-life analytics: behaviour that only shows up against a place's own history.

ship-to-ship (STS) rendezvous
    Two hulls lying alongside each other (centres closer than their combined beams plus
    ~60 m, roughly parallel, overlapping along their length, and not one passing the
    other) are the signature of an STS transfer, bunkering or a hull being serviced.
    On 10 m imagery a rafted pair often merges into one unusually fat blob, so those are
    flagged too. A pair seen together on several dates scores higher; crowded anchorages
    score lower because rafting there is routine.

unusual location
    For each class family Oracle builds a kernel density of where objects have been in
    every earlier clear image that covered the spot (vessels ~300 m kernel, aircraft
    ~60 m, vehicles ~25 m). A new stationary object where history says an object is
    present in under ~2% of looks is unusual: a ship anchored in a lane nobody anchors
    in, aircraft on an apron that is always empty, vehicles in a field. It needs a
    baseline (6+ clear images covering the point), so it stays silent at new sites.

Everything here only ranks candidates for a human (or the agent) to look at. Rafting
is often mundane and "unusual" only means "unusual for this place in our data".
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.spatial import cKDTree

from .observations import VESSEL, Observation
from .tracking import family

STS_MIN_LENGTH_M = 60.0
STS_GAP_M = 60.0
STS_MAX_AXIS_DIFF = 25.0
CROWDED_SCENE = 40  # vessels in one clear image: an anchorage, where rafting is routine
KDE_BANDWIDTH_M = {"vessel": 300.0, "aircraft": 60.0, "vehicle": 25.0}
UNUSUAL_RATE = 0.02  # objects expected within one bandwidth, per look
MIN_LOOKS = 6
MIN_HISTORY = 30


def _eid(*parts: object) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _xy(obs: list[Observation], lat0: float, lon0: float) -> np.ndarray:
    """Local metres (equirectangular around the site centre; fine over a few hundred km)."""
    k = math.cos(math.radians(lat0)) * 111_320.0
    return np.array([[(o.lon - lon0) * k, (o.lat - lat0) * 110_540.0] for o in obs], float).reshape(-1, 2)


def _beam(o: Observation) -> float:
    if o.width_m and o.length_m and o.width_m < o.length_m:
        return float(o.width_m)
    return (o.length_m or 60.0) / 6.0


def _axis_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _underway(o: Observation) -> bool | None:
    u = o.attrs.get("underway")
    return None if u is None else bool(u)


def sts_candidates(
    site: str,
    obs: list[Observation],
    clear: dict[str, float] | None = None,
    clear_enough: float = 0.85,
) -> list[dict]:
    clear = clear or {}
    vessels = [o for o in obs if o.cls == VESSEL and (o.length_m or 0) >= STS_MIN_LENGTH_M]
    if not vessels:
        return []
    lat0 = float(np.mean([o.lat for o in vessels]))
    lon0 = float(np.mean([o.lon for o in vessels]))
    per_scene: dict[str, list[Observation]] = defaultdict(list)
    for o in vessels:
        per_scene[o.scene_id].append(o)
    n_in_scene = defaultdict(int)
    for o in obs:
        if o.cls == VESSEL:
            n_in_scene[o.scene_id] += 1

    pairs: list[tuple[Observation, Observation, dict]] = []
    for group in per_scene.values():
        if len(group) < 2:
            continue
        xy = _xy(group, lat0, lon0)
        reach = max((o.length_m or 0) for o in group) / 2 + 2 * max(_beam(o) for o in group) + STS_GAP_M
        for i, j in cKDTree(xy).query_pairs(reach):
            a, b = group[i], group[j]
            geo = _alongside(a, b, xy[j] - xy[i])
            if geo:
                pairs.append((a, b, geo))

    # How often has the same pair of tracks been seen together?
    together: dict[tuple, set] = defaultdict(set)
    for a, b, _ in pairs:
        ta, tb = a.attrs.get("track_id"), b.attrs.get("track_id")
        if ta and tb:
            together[tuple(sorted((ta, tb)))].add(a.scene_id)

    events = []
    for a, b, geo in pairs:
        big, small = (a, b) if (a.length_m or 0) >= (b.length_m or 0) else (b, a)
        key = tuple(sorted((a.attrs.get("track_id") or a.id, b.attrs.get("track_id") or b.id)))
        repeats = len(together.get(key, ())) if all(x.attrs.get("track_id") for x in (a, b)) else 1
        crowded = n_in_scene[a.scene_id] > CROWDED_SCENE
        sev = 0.45
        sev += 0.15 * ((small.length_m or 0) >= 150)  # tanker-sized on both sides
        sev += 0.1 * min(repeats - 1, 2)
        sev -= 0.2 * crowded
        sev -= 0.1 * (clear.get(a.scene_id, 1.0) < clear_enough)
        lat, lon = (a.lat + b.lat) / 2, (a.lon + b.lon) / 2
        events.append(
            {
                "id": _eid("sts", *sorted((a.id, b.id))),
                "site": site,
                "time": a.time.isoformat(),
                "kind": "sts_candidate",
                "severity": round(min(max(sev, 0.15), 0.9), 2),
                "title": f"{big.length_m:.0f} m and {small.length_m:.0f} m hulls alongside each other at {site}",
                "detail": {
                    "lat": lat,
                    "lon": lon,
                    "separation_m": geo["separation_m"],
                    "gap_m": geo["gap_m"],
                    "axis_diff_deg": geo["axis_diff_deg"],
                    "seen_together": repeats,
                    "crowded": crowded,
                    "source": a.source,
                    "obs_ids": [a.id, b.id],
                    "note": "rafted pair: STS transfer, bunkering or servicing; routine in anchorages",
                },
                "obs_id": big.id,
                "track_id": big.attrs.get("track_id"),
            }
        )

    # Rafted pairs that the 10 m detector saw as one fat hull.
    for o in vessels:
        L, W = o.length_m or 0, o.width_m or 0
        if o.source in ("sentinel-2", "sentinel-1", "landsat") and L >= 120 and W >= 0.3 * L:
            events.append(
                {
                    "id": _eid("sts1", o.id),
                    "site": site,
                    "time": o.time.isoformat(),
                    "kind": "sts_candidate",
                    "severity": round(0.3 - 0.15 * (n_in_scene[o.scene_id] > CROWDED_SCENE), 2),
                    "title": f"Unusually wide {L:.0f} x {W:.0f} m hull (possible rafted pair) at {site}",
                    "detail": {"lat": o.lat, "lon": o.lon, "aspect": round(L / max(W, 1), 2), "source": o.source},
                    "obs_id": o.id,
                    "track_id": o.attrs.get("track_id"),
                }
            )
    return events


def _alongside(a: Observation, b: Observation, d_ab: np.ndarray) -> dict | None:
    if _underway(a) is not None and _underway(b) is not None and _underway(a) != _underway(b):
        return None  # one moving past the other
    diff = _axis_diff(a.axis_deg, b.axis_deg)
    if diff is not None and diff > STS_MAX_AXIS_DIFF:
        return None
    axes = [x for x in (a.axis_deg, b.axis_deg) if x is not None]
    if axes:
        th = math.radians(axes[0])  # axis is clockwise from north
        u = np.array([math.sin(th), math.cos(th)])
        along = abs(float(d_ab @ u))
        lateral = abs(float(d_ab[0] * u[1] - d_ab[1] * u[0]))
    else:
        along, lateral = 0.0, float(np.hypot(*d_ab))
    half = max(a.length_m or 0, b.length_m or 0) / 2
    gap = lateral - (_beam(a) + _beam(b)) / 2
    if along > half or gap > STS_GAP_M:
        return None
    return {
        "separation_m": round(float(np.hypot(*d_ab)), 1),
        "gap_m": round(max(gap, 0.0), 1),
        "axis_diff_deg": None if diff is None else round(diff, 1),
    }


def unusual_locations(
    site: str,
    obs: list[Observation],
    scene_times: dict[str, datetime],
    clear: dict[str, float] | None = None,
    coverage: dict[str, tuple] | None = None,
    clear_enough: float = 0.85,
    recent: int = 3,
) -> list[dict]:
    """Stationary objects where the site's own history says objects almost never are."""
    clear = clear or {}
    coverage = coverage or {}
    looks = sorted(
        (sid for sid in scene_times if clear.get(sid, 1.0) >= clear_enough),
        key=lambda s: scene_times[s],
    )
    if len(looks) <= MIN_LOOKS:
        return []
    by_scene: dict[str, list[Observation]] = defaultdict(list)
    for o in obs:
        by_scene[o.scene_id].append(o)
    lat0 = float(np.mean([o.lat for o in obs])) if obs else 0.0
    lon0 = float(np.mean([o.lon for o in obs])) if obs else 0.0
    events = []
    for k in range(max(MIN_LOOKS, len(looks) - recent), len(looks)):
        sid = looks[k]
        hist_ids = looks[:k]
        for fam, h in KDE_BANDWIDTH_M.items():
            new = [o for o in by_scene.get(sid, []) if family(o.cls) == fam and _underway(o) is not True]
            if not new:
                continue
            hist = [o for s in hist_ids for o in by_scene.get(s, []) if family(o.cls) == fam]
            if len(hist) < MIN_HISTORY:
                continue
            hxy = _xy(hist, lat0, lon0)
            tree = cKDTree(hxy)
            for o, p in zip(new, _xy(new, lat0, lon0), strict=True):
                looked = [s for s in hist_ids if _covers(coverage.get(s), o)]
                if len(looked) < MIN_LOOKS:
                    continue
                idx = tree.query_ball_point(p, 3 * h)
                d2 = np.sum((hxy[idx] - p) ** 2, axis=1) if idx else np.zeros(0)
                rate = float(np.exp(-d2 / (2 * h * h)).sum()) / len(looked)
                if rate >= UNUSUAL_RATE:
                    continue
                nearest = float(tree.query(p)[0])
                sev = 0.35 + 0.15 * min(nearest / (10 * h), 1.0) + 0.15 * (fam == "vessel" and (o.length_m or 0) >= 150)
                sev += 0.1 * (fam == "aircraft")
                sev -= 0.1 * (_underway(o) is None and fam == "vessel")  # SAR: can't tell if it was moving
                events.append(
                    {
                        "id": _eid("unusual", o.id),
                        "site": site,
                        "time": o.time.isoformat(),
                        "kind": "unusual_location",
                        "severity": round(min(sev, 0.85), 2),
                        "title": f"{_size(o)}{o.cls} where none usually is at {site} (nearest past sighting {nearest:,.0f} m)",
                        "detail": {
                            "lat": o.lat,
                            "lon": o.lon,
                            "expected_per_look": round(rate, 4),
                            "looks": len(looked),
                            "nearest_past_m": round(nearest),
                            "bandwidth_m": h,
                            "source": o.source,
                        },
                        "obs_id": o.id,
                        "track_id": o.attrs.get("track_id"),
                    }
                )
    return events


def _covers(bbox: tuple | None, o: Observation) -> bool:
    if bbox is None:
        return True  # coverage unknown: assume the site footprint
    w, s, e, n = bbox
    return w <= o.lon <= e and s <= o.lat <= n


def _size(o: Observation) -> str:
    return f"{o.length_m:.0f} m " if o.length_m else ""


def density_grid(
    obs: list[Observation], bbox: tuple[float, float, float, float], fam: str, n_looks: int, cell_m: float | None = None
) -> dict:
    """Pattern-of-life heatmap: expected objects of ``fam`` per look per km^2 (for the UI/agent)."""
    from scipy.ndimage import gaussian_filter

    h = KDE_BANDWIDTH_M.get(fam, 100.0)
    cell = cell_m or h
    w, s, e, n = bbox
    lat0 = (s + n) / 2
    kx = math.cos(math.radians(lat0)) * 111_320.0
    nx = max(1, min(400, int((e - w) * kx / cell)))
    ny = max(1, min(400, int((n - s) * 110_540.0 / cell)))
    grid = np.zeros((ny, nx))
    for o in obs:
        if family(o.cls) != fam:
            continue
        i = int((n - o.lat) / (n - s) * ny)
        j = int((o.lon - w) / (e - w) * nx)
        if 0 <= i < ny and 0 <= j < nx:
            grid[i, j] += 1
    cell_x, cell_y = (e - w) * kx / nx, (n - s) * 110_540.0 / ny
    grid = gaussian_filter(grid, sigma=(h / cell_y, h / cell_x)) / max(n_looks, 1) / (cell_x * cell_y / 1e6)
    return {"bbox": bbox, "family": fam, "looks": n_looks, "per_km2_per_look": np.round(grid, 4).tolist()}
