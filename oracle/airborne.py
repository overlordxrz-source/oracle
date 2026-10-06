"""Aircraft in flight on free Sentinel-2 imagery: detection, speed, heading and altitude.

Sentinel-2's push-broom detectors record each band at a slightly different moment:
B08 0.264 s, B03 0.527 s and B04 1.005 s after B02. Everything on the ground lines up;
anything moving does not. A jet doing 230 m/s is 230 m (23 pixels) further along in
red than in blue, so it shows up as four blobs on a straight line, spaced exactly in the
ratio of the band delays: 0 : 0.263 : 0.524 : 1. That pattern is the detector. A car or
a ship moves a few metres in a second and never matches.

Altitude adds a second shift. Each band looks at the ground at a slightly different
along-track angle, so an object at height h appears displaced against the satellite's
ground track by h * (Vg / H) * dt (Vg ~ 6.6 km/s ground-track speed, H 786 km). The
apparent band-to-band velocity is therefore

    u = v_aircraft - (h * Vg / H) * track_direction

With the aircraft's heading known (the fuselage's long axis in the image), the two
unknowns, true speed and altitude, can be solved. Oracle reports the apparent velocity
always, and speed plus altitude when the geometry allows (the heading must not be
parallel to the satellite's track). Track direction comes from the orbit (SGP4 on the
satellite's TLE at the image time), or ~193 deg for a descending pass if unavailable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from pyproj import Transformer
from scipy import ndimage

from .geo import AOI
from .imagery import Grid, NoData
from .models import Scene

BAND_DT = {"blue": 0.0, "nir": 0.264, "green": 0.527, "red": 1.005}  # seconds after B02
PARALLAX_PER_M = 6630.0 / 786_000.0  # Vg / H: apparent shift (m/s) per metre of height
MIN_SPEED = 45.0  # m/s apparent: below this it's ground traffic or misregistration
MAX_SPEED = 420.0  # m/s apparent (incl. up to ~13 km of parallax)
PEAK_SIGMA = 5.0


@dataclass
class Aircraft:
    lat: float
    lon: float
    time: str
    apparent_speed_ms: float
    apparent_heading_deg: float
    contrast: float
    bands_matched: int
    fit_rms_px: float
    axis_deg: float | None = None
    speed_ms: float | None = None
    heading_deg: float | None = None
    altitude_m: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 5 if k in ("lat", "lon") else 1)
        return d


def track_bearing(scene: Scene) -> float:
    """Satellite ground-track bearing (deg) at the image time, from its orbit."""
    try:
        from sgp4.api import Satrec, jday

        from .passes import _fetch_tles, _gmst

        name = {"sentinel-2a": "SENTINEL-2A", "sentinel-2b": "SENTINEL-2B", "sentinel-2c": "SENTINEL-2C"}.get(
            (scene.platform or "").lower()
        )
        tles = _fetch_tles()
        if name not in tles:
            raise KeyError(name)
        sat = Satrec.twoline2rv(*tles[name])
        t = scene.datetime
        out = []
        for dt in (0.0, 2.0):
            jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + dt)
            err, r, _ = sat.sgp4(jd, fr)
            th = float(_gmst(np.array([jd + fr]))[0])
            x = math.cos(th) * r[0] + math.sin(th) * r[1]
            y = -math.sin(th) * r[0] + math.cos(th) * r[1]
            z = r[2]
            out.append((math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x))))
        (la1, lo1), (la2, lo2) = out
        k = math.cos(math.radians(la1))
        return (math.degrees(math.atan2((lo2 - lo1) * k, la2 - la1)) + 360) % 360
    except Exception:  # noqa: BLE001 - fall back to a typical descending track
        return 193.0


def _peaks(x: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compact anomalies -> (sub-pixel rows/cols (n,2), |z| at peaks, signed z map)."""
    hp = x - ndimage.gaussian_filter(np.nan_to_num(x, nan=float(np.nanmedian(x))), 3)
    hp[~valid] = 0
    v = hp[valid]
    mad = float(np.median(np.abs(v - np.median(v)))) * 1.4826 or 1e-4
    zs = hp / mad
    z = np.abs(zs)
    mx = ndimage.maximum_filter(z, size=5)
    rr, cc = np.nonzero((z == mx) & (z > PEAK_SIGMA) & valid)
    pts = []
    for r, c in zip(rr, cc, strict=True):  # sub-pixel centroid in a 3x3 window
        r0, r1, c0, c1 = max(r - 1, 0), min(r + 2, z.shape[0]), max(c - 1, 0), min(c + 2, z.shape[1])
        w = z[r0:r1, c0:c1]
        gy, gx = np.mgrid[r0:r1, c0:c1]
        pts.append(((gy * w).sum() / w.sum(), (gx * w).sum() / w.sum()))
    return np.array(pts, float).reshape(-1, 2), z[rr, cc], zs


def _sample(zs: np.ndarray, p: np.ndarray) -> float:
    """Signed z with the largest magnitude in the 3x3 around p."""
    r, c = int(round(p[0])), int(round(p[1]))
    w = zs[max(r - 1, 0) : r + 2, max(c - 1, 0) : c + 2]
    if not w.size:
        return 0.0
    k = np.unravel_index(np.argmax(np.abs(w)), w.shape)
    return float(w[k])


# Band pairs far enough apart in time that a moving aircraft can't overlap itself.
EXCLUSIVE = [("blue", "green"), ("blue", "red"), ("nir", "red"), ("green", "red"), ("nir", "blue")]


def _is_mover(zmaps: dict, pos: dict) -> tuple[bool, float]:
    """The moving-object signature: present in each band at that band's position, absent
    from the other bands there (a static feature shows in all bands at one spot), same
    polarity, comparable strength."""
    own = {b: _sample(zmaps[b], pos[b]) for b in pos}
    mags = [abs(v) for v in own.values()]
    if min(mags) < PEAK_SIGMA * 0.8 or max(mags) > 5 * min(mags):
        return False, 0.0
    if len({np.sign(v) for v in own.values()}) > 1:
        return False, 0.0
    worst = 0.0
    for a, b in EXCLUSIVE:
        if np.hypot(*(pos[a] - pos[b])) < 3.0:  # blobs overlap at low speed: can't judge
            continue
        for x, y in ((a, b), (b, a)):
            ratio = abs(_sample(zmaps[y], pos[x])) / abs(own[x])
            worst = max(worst, ratio)
    return worst <= 0.45, worst


def _axis(img: np.ndarray, r: float, c: float) -> float | None:
    """Long-axis bearing (deg, 0-180) of the blob around (r, c), if it's elongated."""
    r, c = int(round(r)), int(round(c))
    sl = (slice(max(r - 5, 0), r + 6), slice(max(c - 5, 0), c + 6))
    w = np.nan_to_num(img[sl])
    w = np.abs(w - np.median(w))
    m = w > 0.5 * w.max()
    lab, _ = ndimage.label(m)
    cl = lab[min(r, 5), min(c, 5)] if lab.shape[0] > 5 and lab.shape[1] > 5 else 0
    if not cl:
        return None
    yy, xx = np.nonzero(lab == cl)
    # At 10 m an airliner is 4-7 pixels: only trust a clearly resolved, clearly elongated
    # fuselage. (A looser test gave a 9 km "altitude" for a departure that was ~1 km up.)
    if len(yy) < 8:
        return None
    cov = np.cov(np.stack([xx, -yy]).astype(float))
    ev, vec = np.linalg.eigh(cov)
    if ev[0] <= 0 or ev[1] / ev[0] < 9.0:  # axis ratio >= 3
        return None
    vx, vy = vec[:, 1]
    return (math.degrees(math.atan2(vx, vy)) + 180) % 180


def detect_airborne(scene: Scene, aoi: AOI, max_pixels: int = 3000) -> list[Aircraft]:
    """Aircraft in flight in a Sentinel-2 scene over ``aoi``."""
    from .change import _read_band

    if scene.source != "sentinel-2":
        raise ValueError("airborne detection uses Sentinel-2's band timing")
    grid = Grid.for_aoi(aoi, 10.0, max_pixels=max_pixels)
    if grid.res > 10.5:
        raise ValueError(f"area too large (max ~{max_pixels * 10 / 1000:.0f} km across at 10 m)")
    scale = scene.extra.get("reflectance_scale", 0.0001)
    offset = scene.extra.get("reflectance_offset", 0.0)
    bands = {b: _read_band(scene, b, grid) * scale + offset for b in BAND_DT}
    valid = np.all([np.isfinite(a) for a in bands.values()], axis=0)
    if valid.mean() < 0.2:
        raise NoData("scene doesn't cover the area")
    peaks = {b: _peaks(a, valid) for b, a in bands.items()}
    zmaps = {b: v[2] for b, v in peaks.items()}
    from scipy.spatial import cKDTree

    trees = {b: (cKDTree(p) if len(p) else None) for b, (p, _, _) in peaks.items()}
    p2, z2, _ = peaks["blue"]
    p4 = peaks["red"][0]
    if not len(p2) or trees["red"] is None or trees["green"] is None or trees["nir"] is None:
        return []
    rmin, rmax = MIN_SPEED * BAND_DT["red"] / grid.res, MAX_SPEED * BAND_DT["red"] / grid.res
    found: list[tuple[float, np.ndarray, np.ndarray, int, float, int]] = []
    for i, a in enumerate(p2):
        for j in trees["red"].query_ball_point(a, rmax):
            d = p4[j] - a
            if np.hypot(*d) < rmin:
                continue
            pos, resid = {"blue": a, "red": p4[j]}, []
            for b in ("nir", "green"):
                pred = a + (BAND_DT[b] / BAND_DT["red"]) * d
                dist, k = trees[b].query(pred)
                if dist > 1.6:
                    break
                pos[b] = peaks[b][0][k]
                resid.append(dist)
            else:
                ok, _leak = _is_mover(zmaps, pos)
                if ok:
                    found.append((float(z2[i]), a, d, 4, float(np.sqrt(np.mean(np.square(resid)))), j))
    # One aircraft can pair with several red blobs: keep the best fit per blue blob and per red blob.
    found.sort(key=lambda f: (-f[3], f[4], -f[0]))
    used_b, used_r, out = set(), set(), []
    tr = Transformer.from_crs(grid.crs, 4326, always_xy=True)
    bearing = None
    for z, a, d, matched, rms, j in found:
        key_b = (round(a[0]), round(a[1]))
        if key_b in used_b or j in used_r:
            continue
        used_b.add(key_b)
        used_r.add(j)
        bearing = track_bearing(scene) if bearing is None else bearing
        east, north = d[1] * grid.res / BAND_DT["red"], -d[0] * grid.res / BAND_DT["red"]
        mid = a + d / 2
        x, y = grid.transform @ (mid[1] + 0.5, mid[0] + 0.5)
        lon, lat = tr.transform(x, y)
        ac = Aircraft(
            lat=lat,
            lon=lon,
            time=scene.datetime.isoformat(timespec="seconds"),
            apparent_speed_ms=math.hypot(east, north),
            apparent_heading_deg=(math.degrees(math.atan2(east, north)) + 360) % 360,
            contrast=z,
            bands_matched=matched,
            fit_rms_px=rms,
        )
        axis = _axis(bands["nir"], *(a + d * BAND_DT["nir"] / BAND_DT["red"]))
        ac.axis_deg = axis
        _solve(ac, east, north, axis, bearing)
        out.append(ac)
    return out


def _solve(ac: Aircraft, east: float, north: float, axis: float | None, track_deg: float) -> None:
    """u = s * h_hat - p * t_hat  (p = altitude * Vg/H >= 0): solve for s and p."""
    tb = math.radians(track_deg)
    t_hat = np.array([math.sin(tb), math.cos(tb)])
    u = np.array([east, north])
    if axis is None:
        # Without a heading, bracket: ground level (p=0) up to a typical 11 km cruise.
        lo = np.hypot(*(u + 0 * t_hat))
        hi = np.hypot(*(u + 11_000 * PARALLAX_PER_M * t_hat))
        ac.note = f"heading not measurable; true speed {min(lo, hi):.0f}-{max(lo, hi):.0f} m/s for 0-11 km altitude"
        return
    best = None
    for hdg in (axis, axis + 180):
        h_hat = np.array([math.sin(math.radians(hdg)), math.cos(math.radians(hdg))])
        A = np.stack([h_hat, -t_hat], 1)
        if abs(np.linalg.det(A)) < 0.25:  # heading within ~15 deg of the track: ill-posed
            continue
        s, p = np.linalg.solve(A, u)
        if s > 0 and p > -5 * PARALLAX_PER_M * 1000:  # allow small negative from noise
            cand = (s, max(p, 0.0), hdg % 360)
            if best is None or abs(cand[1]) < abs(best[1]):
                best = cand
    if best is None:
        ac.note = "heading nearly parallel to the satellite track: speed/altitude not separable"
        return
    s, p, hdg = best
    ac.speed_ms, ac.heading_deg, ac.altitude_m = float(s), float(hdg), float(p / PARALLAX_PER_M)
    if ac.altitude_m > 16_000 or ac.speed_ms > 330:
        ac.note = "solution implausible (noisy heading); trust the apparent velocity only"
        ac.speed_ms = ac.altitude_m = None
