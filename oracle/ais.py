"""AIS cross-matching: which detected hulls are broadcasting, and which are "dark"?

Ships over ~300 GT must transmit AIS. A large hull seen by a satellite with no AIS
position nearby at that moment, in an area where AIS coverage clearly exists, is a
classic OSINT lead: sanctions evasion, smuggling, military, or plain AIS dropouts.

Load any AIS CSV export. Recognised column names (case-insensitive):
  MMSI | mmsi;  BaseDateTime | timestamp | time | datetime;  LAT | lat | latitude;
  LON | lon | longitude;  SOG | sog | speed;  COG | cog | course;
  VesselName | name | shipname;  Length | length;  VesselType | type | shiptype
That covers NOAA MarineCadastre (US waters, free), Danish Maritime Authority (free),
aisstream.io dumps, Global Fishing Watch exports and most commercial exports.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from .models import parse_dt
from .observations import VESSEL, Observation
from .store import Store

ALIASES = {
    "mmsi": ("mmsi",),
    "time": ("basedatetime", "timestamp", "time", "datetime", "date_time", "time_utc"),
    "lat": ("lat", "latitude"),
    "lon": ("lon", "lng", "long", "longitude"),
    "sog": ("sog", "speed"),
    "cog": ("cog", "course"),
    "name": ("vesselname", "name", "shipname", "vessel_name"),
    "length": ("length", "length_m"),
    "vtype": ("vesseltype", "type", "shiptype", "ship_type"),
}


def load_csv(
    store: Store,
    path: str | Path,
    bbox: tuple[float, float, float, float] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> int:
    """Import AIS positions (optionally clipped to bbox/time). Returns rows stored."""
    rows, n = [], 0
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        cols = {k.lower().strip(): k for k in reader.fieldnames or []}
        pick = {}
        for key, names in ALIASES.items():
            pick[key] = next((cols[a] for a in names if a in cols), None)
        missing = [k for k in ("mmsi", "time", "lat", "lon") if not pick[k]]
        if missing:
            raise ValueError(f"AIS file lacks columns: {', '.join(missing)} (have: {', '.join(cols)})")
        for r in reader:
            try:
                lat, lon = float(r[pick["lat"]]), float(r[pick["lon"]])
                t = parse_dt(r[pick["time"]])
            except (ValueError, TypeError):
                continue
            if bbox and not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
                continue
            if (start and t < start) or (end and t > end):
                continue
            rows.append(
                (
                    int(float(r[pick["mmsi"]])),
                    t.isoformat(),
                    lat,
                    lon,
                    _f(r, pick["sog"]),
                    _f(r, pick["cog"]),
                    (r.get(pick["name"]) or "").strip() if pick["name"] else "",
                    _f(r, pick["length"]),
                    (r.get(pick["vtype"]) or "") if pick["vtype"] else "",
                )
            )
            if len(rows) >= 50_000:
                n += _flush(store, rows)
                rows = []
    return n + _flush(store, rows)


def _f(r: dict, col: str | None) -> float | None:
    if not col or r.get(col) in (None, ""):
        return None
    try:
        return float(r[col])
    except ValueError:
        return None


def _flush(store: Store, rows: list[tuple]) -> int:
    with store.tx() as c:
        c.executemany("INSERT OR IGNORE INTO ais VALUES(?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def positions_near(store: Store, o: Observation, window_s: float, pad_deg: float) -> dict[int, list[dict]]:
    t0 = (o.time - timedelta(seconds=window_s)).isoformat()
    t1 = (o.time + timedelta(seconds=window_s)).isoformat()
    cur = store.conn.execute(
        "SELECT * FROM ais WHERE time BETWEEN ? AND ? AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
        (t0, t1, o.lat - pad_deg, o.lat + pad_deg, o.lon - pad_deg, o.lon + pad_deg),
    )
    tracks: dict[int, list[dict]] = defaultdict(list)
    for r in cur:
        d = dict(r)
        d["t"] = parse_dt(d["time"])
        tracks[d["mmsi"]].append(d)
    return tracks


def interpolate(points: list[dict], when: datetime) -> tuple[float, float, float] | None:
    """Position of one vessel at ``when`` -> (lat, lon, uncertainty m)."""
    pts = sorted(points, key=lambda p: p["t"])
    before = [p for p in pts if p["t"] <= when]
    after = [p for p in pts if p["t"] > when]
    if before and after:
        a, b = before[-1], after[0]
        span = (b["t"] - a["t"]).total_seconds() or 1.0
        f = (when - a["t"]).total_seconds() / span
        gap = min((when - a["t"]).total_seconds(), (b["t"] - when).total_seconds())
        return a["lat"] + f * (b["lat"] - a["lat"]), a["lon"] + f * (b["lon"] - a["lon"]), 50 + 0.5 * gap
    p = before[-1] if before else after[0]
    dt = (when - p["t"]).total_seconds()
    lat, lon = p["lat"], p["lon"]
    if p.get("sog") and p.get("cog") is not None and p["sog"] < 60:
        d = p["sog"] * 0.514444 * dt  # dead reckoning along COG
        lat += d * math.cos(math.radians(p["cog"])) / 110_574
        lon += d * math.sin(math.radians(p["cog"])) / (111_320 * math.cos(math.radians(lat)))
    return lat, lon, 100 + 2.0 * abs(dt)


def match(store: Store, obs: list[Observation], window_s: float = 3600, min_dark_length: float = 60.0) -> dict[str, dict]:
    """Attach the best AIS match to each vessel observation (and flag dark candidates).

    Returns {obs_id: result}; results are also written into each observation's attrs
    in the store: ``ais`` = {mmsi, name, distance_m, p, sog, cog} or ``dark`` = True.
    """
    out = {}
    for o in obs:
        if o.cls != VESSEL:
            continue
        tracks = positions_near(store, o, window_s, pad_deg=0.25)
        if not tracks:
            continue  # no AIS coverage here and now: can't call anything dark
        cands = []
        for mmsi, pts in tracks.items():
            pos = interpolate(pts, o.time)
            if pos is None:
                continue
            k = math.cos(math.radians(o.lat))
            d = math.hypot((pos[1] - o.lon) * 111_320 * k, (pos[0] - o.lat) * 110_574)
            sig = math.hypot(pos[2], 3 * max(o.attrs.get("gsd", 10.0), 10.0))
            like = math.exp(-0.5 * (d / sig) ** 2)
            ais_len = next((p["length_m"] for p in pts if p.get("length_m")), None)
            if ais_len and o.length_m:
                like *= math.exp(-0.5 * ((ais_len - o.length_m) / max(25.0, 0.2 * ais_len)) ** 2)
            cands.append((like, d, mmsi, pts))
        total = sum(c[0] for c in cands) + 0.05  # 0.05: "none of these" (dark / unmatched)
        best = max(cands, default=None, key=lambda c: c[0])
        if best and best[0] / total >= 0.5 and best[1] < 3000:
            pts = best[3]
            last = min(pts, key=lambda p: abs((p["t"] - o.time).total_seconds()))
            res = {
                "mmsi": best[2],
                "name": last.get("name") or "",
                "distance_m": round(best[1]),
                "p": round(best[0] / total, 3),
                "sog": last.get("sog"),
                "cog": last.get("cog"),
                "vtype": last.get("vtype") or "",
            }
            store.update_obs_attrs(o.id, ais=res, dark=False)
        elif (o.length_m or 0) >= min_dark_length:
            res = {"dark": True, "nearest_ais_m": round(best[1]) if best else None}
            store.update_obs_attrs(o.id, dark=True, nearest_ais_m=res["nearest_ais_m"])
        else:
            continue
        out[o.id] = res
    return out
