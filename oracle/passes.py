"""Collection planning: when will the free imaging satellites next look at a place?

Orbits come from public TLEs (CelesTrak, cached 12 h, with a fallback mirror) and are
propagated with SGP4. For each satellite Oracle finds every closest approach to the
target in the window and checks the sensor's actual imaging geometry:

  Sentinel-2 / Landsat  nadir-centred swath (290 km / 185 km), descending (morning)
                        pass, sun above ~10 deg; both only image land and coasts
  Sentinel-1            right-looking radar; the IW swath sits ~345-617 km to the right
                        of the ground track (incidence 29-46 deg), day or night

"Likely" means the geometry allows an acquisition. Whether one is actually scheduled
depends on the mission's acquisition plan (Sentinel-1 especially), so treat the output as
an upper bound. Good enough to answer "when is the next free look at X?" and "how
stale can my picture get?".
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from .config import CACHE_DIR
from .http import client, log

EARTH_R = 6371.0
SATELLITES = {
    # name: (norad id, family)
    "SENTINEL-2A": (40697, "sentinel-2"),
    "SENTINEL-2B": (42063, "sentinel-2"),
    "SENTINEL-2C": (60989, "sentinel-2"),
    "SENTINEL-1A": (39634, "sentinel-1"),
    "SENTINEL-1C": (62261, "sentinel-1"),
    "SENTINEL-1D": (66315, "sentinel-1"),
    "LANDSAT 8": (39084, "landsat"),
    "LANDSAT 9": (49260, "landsat"),
}
GEOMETRY = {
    "sentinel-2": {"swath_km": 290.0, "descending": True, "daylight": True},
    "landsat": {"swath_km": 185.0, "descending": True, "daylight": True},
    "sentinel-1": {"near_km": 345.0, "far_km": 617.0, "right": True},
}
TLE_CACHE = CACHE_DIR / "tle.txt"
TLE_MAX_AGE_S = 12 * 3600


@dataclass
class Pass:
    satellite: str
    family: str
    time: datetime
    cross_track_km: float
    side: str  # left / right of the ground track
    direction: str  # ascending / descending
    sun_elevation: float
    likely: bool
    note: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["time"] = self.time.isoformat(timespec="seconds")
        return d


def _fetch_tles() -> dict[str, tuple[str, str]]:
    text = ""
    if TLE_CACHE.exists() and time.time() - TLE_CACHE.stat().st_mtime < TLE_MAX_AGE_S:
        text = TLE_CACHE.read_text()
    if not text:
        parts = []
        with client() as c:
            for norad in sorted({v[0] for v in SATELLITES.values()}):
                for url in (
                    f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad}&FORMAT=TLE",
                    f"https://tle.ivanstanojevic.me/api/tle/{norad}",
                ):
                    try:
                        r = c.get(url, timeout=30)
                        if r.status_code != 200:
                            continue
                        if url.endswith(str(norad)) and r.headers.get("content-type", "").startswith("application/"):
                            d = r.json()
                            parts.append(f"{d['name']}\n{d['line1']}\n{d['line2']}")
                        elif r.text.strip():
                            parts.append(r.text.strip())
                        break
                    except Exception as exc:  # noqa: BLE001 - try the mirror
                        log(f"  tle {norad}: {type(exc).__name__}")
        text = "\n".join(parts)
        if text:
            TLE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            TLE_CACHE.write_text(text)
        elif TLE_CACHE.exists():
            text = TLE_CACHE.read_text()  # stale beats nothing
    out: dict[str, tuple[str, str]] = {}
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    by_norad = {v[0]: k for k, v in SATELLITES.items()}
    for i, ln in enumerate(lines):
        if ln.startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            norad = int(ln[2:7])
            if norad in by_norad:
                out[by_norad[norad]] = (ln, lines[i + 1])
    return out


def _gmst(jd: np.ndarray) -> np.ndarray:
    return np.radians((280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360.0)


def sun_elevation(lat: float, lon: float, when: datetime) -> float:
    """Approximate solar elevation (deg); good to ~1 deg, plenty for day/night."""
    d = (when - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400
    g = math.radians((357.529 + 0.98560028 * d) % 360)
    q = (280.459 + 0.98564736 * d) % 360
    lam = math.radians(q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 0.00000036 * d)
    dec = math.asin(math.sin(eps) * math.sin(lam))
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    gmst = math.radians((280.46061837 + 360.98564736629 * d) % 360)
    ha = gmst + math.radians(lon) - ra
    phi = math.radians(lat)
    return math.degrees(math.asin(math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.cos(ha)))


def next_passes(
    lat: float,
    lon: float,
    days: float = 7.0,
    families: list[str] | None = None,
    start: datetime | None = None,
    step_s: float = 20.0,
) -> list[Pass]:
    from sgp4.api import Satrec, jday

    start = start or datetime.now(timezone.utc)
    n = int(days * 86400 / step_s)
    offs = np.arange(n) * step_s / 86400.0
    jd0, fr0 = jday(start.year, start.month, start.day, start.hour, start.minute, start.second)
    jd = np.full(n, jd0)
    fr = fr0 + offs
    tgt = np.array(
        [
            math.cos(math.radians(lat)) * math.cos(math.radians(lon)),
            math.cos(math.radians(lat)) * math.sin(math.radians(lon)),
            math.sin(math.radians(lat)),
        ]
    )
    out: list[Pass] = []
    for name, (l1, l2) in _fetch_tles().items():
        fam = SATELLITES[name][1]
        if families and fam not in families:
            continue
        sat = Satrec.twoline2rv(l1, l2)
        err, r, v = sat.sgp4_array(jd, fr)
        ok = err == 0
        if not ok.any():
            continue
        th = _gmst(jd + fr)
        c, s = np.cos(th), np.sin(th)
        # TEME -> Earth-fixed (rotation about z by GMST; polar motion ignored)
        x = c * r[:, 0] + s * r[:, 1]
        y = -s * r[:, 0] + c * r[:, 1]
        z = r[:, 2]
        vx = c * v[:, 0] + s * v[:, 1]
        vy = -s * v[:, 0] + c * v[:, 1]
        vz = v[:, 2]
        pos = np.stack([x, y, z], 1)
        up = pos / np.linalg.norm(pos, axis=1, keepdims=True)
        ang = np.arccos(np.clip(up @ tgt, -1, 1))
        dist = ang * EARTH_R
        # closest approaches = local minima of ground distance, within ~800 km
        mins = np.where((dist[1:-1] < dist[:-2]) & (dist[1:-1] <= dist[2:]) & (dist[1:-1] < 800) & ok[1:-1])[0] + 1
        for i in mins:
            vel = np.array([vx[i], vy[i], vz[i]])
            right = np.cross(vel, up[i])
            side = "right" if right @ (tgt - up[i]) > 0 else "left"
            when = start + timedelta(days=float(offs[i]))
            direction = "descending" if vz[i] < 0 else "ascending"
            sun = sun_elevation(lat, lon, when)
            g = GEOMETRY[fam]
            if fam == "sentinel-1":
                likely = side == "right" and g["near_km"] <= dist[i] <= g["far_km"]
                note = "in IW swath (right-looking)" if likely else "outside radar swath"
            else:
                in_swath = dist[i] <= g["swath_km"] / 2
                likely = in_swath and direction == "descending" and sun > 10
                note = (
                    "in swath, daylight descending pass"
                    if likely
                    else "outside swath"
                    if not in_swath
                    else "wrong pass direction/night"
                )
            out.append(
                Pass(
                    name,
                    fam,
                    when.replace(microsecond=0),
                    round(float(dist[i]), 1),
                    side,
                    direction,
                    round(sun, 1),
                    bool(likely),
                    note,
                )
            )
    out.sort(key=lambda p: p.time)
    return out


def next_looks(lat: float, lon: float, days: float = 7.0) -> dict[str, dict | None]:
    """First likely acquisition per family (for briefs and the agent)."""
    res: dict[str, dict | None] = {"sentinel-2": None, "sentinel-1": None, "landsat": None}
    for p in next_passes(lat, lon, days):
        if p.likely and res.get(p.family) is None:
            res[p.family] = p.to_dict()
    return res
