"""Cross-date multi-object tracking: "is this the same object we saw last time, and how sure are we?"

For every pair (existing track T, new observation o) Oracle computes a likelihood ratio
of two hypotheses: H1 "o is T, seen again" vs H0 "o is an unrelated object".

  kinematics   Under H1 the object either stayed put (2-D Gaussian on the position
               error of both observations) or moved: if T has a velocity estimate, it is
               propagated with uncertainty growing with time; if not, anywhere within
               class max-speed x dt is equally likely. A known course (from the wake) that
               agrees with the direction of displacement raises the likelihood. Under H0
               the object is anywhere in the site, uniformly.
  attributes   Length (and, for stationary objects, orientation) must agree within the
               sensors' measurement error. Under H0 they follow a broad prior.
  persistence  Prior odds that the object is still around after dt (class time scale:
               vessels days, vehicles hours-days, aircraft weeks, tanks years).

Each observation's probability of being each candidate is its LR normalised against
all competing tracks, competing observations and "new object" (a JPDA-style marginal).
The final assignment is the global optimum (Hungarian algorithm on -log LR). Links below
50% start a new track, with the near-misses kept as "possible same object" alternatives.

The output is meant to rank and inform, not to assert identity. Two 10 m Sentinel-2
blobs five days apart in a busy strait will rightly get low probabilities.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy.optimize import linear_sum_assignment

from .observations import AIRCRAFT, HELICOPTER, LARGE_VEHICLE, MAX_SPEED, STORAGE_TANK, VEHICLE, VESSEL, Observation

FAMILY = {
    VESSEL: "vessel",
    AIRCRAFT: "aircraft",
    HELICOPTER: "aircraft",
    VEHICLE: "vehicle",
    LARGE_VEHICLE: "vehicle",
    STORAGE_TANK: "tank",
}
P_STATIONARY = {"vessel": 0.4, "aircraft": 0.7, "vehicle": 0.4, "tank": 0.99, "other": 0.95}
PERSIST_DAYS = {"vessel": 5.0, "aircraft": 20.0, "vehicle": 1.0, "tank": 3650.0, "other": 365.0}
LOST_AFTER_DAYS = {"vessel": 30.0, "aircraft": 90.0, "vehicle": 10.0, "tank": 3650.0, "other": 730.0}
LENGTH_RANGE_M = {"vessel": 400.0, "aircraft": 80.0, "vehicle": 25.0, "tank": 120.0, "other": 200.0}
FAMILY_SPEED = {
    "vessel": MAX_SPEED[VESSEL],
    "aircraft": MAX_SPEED[AIRCRAFT],
    "vehicle": MAX_SPEED[VEHICLE],
    "tank": 0.0,
    "other": 0.0,
}
LINK_P = 0.5
ALT_P = 0.05


def family(cls: str) -> str:
    return FAMILY.get(cls, "other")


def gsd_of(o: Observation) -> float:
    g = o.attrs.get("gsd")
    if g:
        return float(g)
    return {"sentinel-2": 10.0, "sentinel-1": 10.0, "landsat": 30.0}.get(o.source, 1.0)


def pos_sigma(o: Observation) -> float:
    """1-sigma position error (m): pixel size plus centroid fuzz of long blobs/wakes."""
    return max(1.5 * gsd_of(o), 2.0) + 0.05 * (o.length_m or 0.0)


def len_sigma(o: Observation) -> float:
    if o.source in ("sentinel-1",):  # radar side-lobes and wakes inflate lengths a lot
        return max(40.0, 0.25 * (o.length_m or 0.0))
    return max(2.0 * gsd_of(o), 0.1 * (o.length_m or 0.0), 1.0)


def dist_m(a: Observation, b: Observation) -> float:
    k = math.cos(math.radians((a.lat + b.lat) / 2))
    return math.hypot((b.lon - a.lon) * 111_320 * k, (b.lat - a.lat) * 110_574)


def bearing(a: Observation, b: Observation) -> float:
    k = math.cos(math.radians((a.lat + b.lat) / 2))
    return (math.degrees(math.atan2((b.lon - a.lon) * k, b.lat - a.lat)) + 360) % 360


def _gauss2(d: float, s: float) -> float:
    return math.exp(-0.5 * (d / s) ** 2) / (2 * math.pi * s * s)


def _gauss1(x: float, s: float) -> float:
    return math.exp(-0.5 * (x / s) ** 2) / (math.sqrt(2 * math.pi) * s)


@dataclass
class Track:
    id: str
    fam: str
    obs: list[Observation] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)
    misses: int = 0
    status: str = "active"

    @property
    def last(self) -> Observation:
        return self.obs[-1]

    def velocity(self) -> tuple[float, float, float] | None:
        """(vx, vy, sigma) in m/s from the last two observations, if they're informative."""
        if len(self.obs) < 2:
            return None
        a, b = self.obs[-2], self.obs[-1]
        dt = (b.time - a.time).total_seconds()
        if dt <= 0 or dt > 3 * 86400:
            return None
        d = dist_m(a, b)
        sig = math.hypot(pos_sigma(a), pos_sigma(b))
        if d < 3 * sig:
            return (0.0, 0.0, sig / dt)
        brg = math.radians(bearing(a, b))
        return (d / dt * math.sin(brg), d / dt * math.cos(brg), sig / dt)

    def summary(self, site: str) -> dict:
        times = [o.time for o in self.obs]
        lengths = [o.length_m for o in self.obs if o.length_m]
        dist = sum(dist_m(a, b) for a, b in zip(self.obs, self.obs[1:], strict=False))
        speeds = []
        for a, b in zip(self.obs, self.obs[1:], strict=False):
            dt = (b.time - a.time).total_seconds()
            if 0 < dt <= 3 * 86400:
                speeds.append(dist_m(a, b) / dt)
        v = self.velocity()
        course = None
        if v and math.hypot(v[0], v[1]) > 0:
            course = round((math.degrees(math.atan2(v[0], v[1])) + 360) % 360, 1)
        elif self.last.course_deg is not None:
            course = self.last.course_deg
        return {
            "id": self.id,
            "site": site,
            "cls": self.last.cls,
            "status": self.status,
            "first_seen": min(times).isoformat(),
            "last_seen": max(times).isoformat(),
            "n_obs": len(self.obs),
            "lat": self.last.lat,
            "lon": self.last.lon,
            "length_m": round(float(np.median(lengths)), 1) if lengths else None,
            "speed_ms": round(float(np.median(speeds)), 2) if speeds else None,
            "course_deg": course,
            "distance_m": round(dist, 1),
            "mean_link_prob": round(float(np.mean(self.probs[1:])), 3) if len(self.probs) > 1 else None,
            "attrs": {
                "dwell_days": round((max(times) - min(times)).total_seconds() / 86400, 2),
                "sources": sorted({o.source for o in self.obs}),
                "path": [[o.lon, o.lat, o.time.isoformat()] for o in self.obs],
                "misses": self.misses,
            },
        }


def likelihood_ratio(t: Track, o: Observation, area_m2: float) -> float:
    """LR of "o is track t seen again" vs "o is an unrelated object" (see module doc)."""
    last = t.last
    dt = (o.time - last.time).total_seconds()
    if dt <= 0:
        return 0.0
    fam = t.fam
    d = dist_m(last, o)
    sig = math.hypot(pos_sigma(last), pos_sigma(o))
    p_s = P_STATIONARY.get(fam, 0.5)
    if last.attrs.get("underway") or o.attrs.get("underway"):
        p_s = 0.1
    l_stat = _gauss2(d, sig)
    vmax = FAMILY_SPEED.get(fam, 0.0)
    v = t.velocity()
    if v and (v[0] or v[1]):
        # dead-reckon from the last measured velocity
        px = v[0] * dt
        py = v[1] * dt
        k = math.cos(math.radians(last.lat))
        ex = (o.lon - last.lon) * 111_320 * k - px
        ey = (o.lat - last.lat) * 110_574 - py
        l_move = _gauss2(math.hypot(ex, ey), math.hypot(sig, v[2] * dt + 0.2 * math.hypot(px, py)))
    elif vmax > 0:
        reach = min(vmax * dt, 4 * math.sqrt(area_m2)) + 2 * sig
        l_move = 1.0 / (math.pi * reach * reach) if d <= reach else 0.0
    else:
        l_move = 0.0
    if l_move and d > 3 * sig:
        course = last.course_deg if last.course_deg is not None else o.course_deg
        if course is not None:
            delta = math.radians(bearing(last, o) - course)
            l_move *= 1.0 + 0.8 * math.cos(delta)  # mean 1 over a uniform bearing
    lr = (p_s * l_stat + (1 - p_s) * l_move) * area_m2
    # attributes
    if last.length_m and o.length_m:
        s_l = math.hypot(len_sigma(last), len_sigma(o))
        lr *= min(LENGTH_RANGE_M.get(fam, 200.0) * _gauss1(o.length_m - last.length_m, s_l), 20.0)
    if last.axis_deg is not None and o.axis_deg is not None and d <= 3 * sig:
        da = abs(o.axis_deg - last.axis_deg) % 180
        da = min(da, 180 - da)
        lr *= p_s * min(180 * _gauss1(da, 8.0), 15.0) + (1 - p_s)
    # persistence prior
    tau = PERSIST_DAYS.get(fam, 30.0) * 86400
    stay = math.exp(-dt / tau)
    lr *= min(max(stay / max(1 - stay, 1e-6), 0.02), 50.0)
    return lr


def associate(
    tracks: list[Track], batch: list[Observation], area_m2: float
) -> tuple[dict[int, tuple[int, float]], list[dict[int, float]]]:
    """Assign observations in one scene to tracks.

    Returns ({obs_index: (track_index, prob)}, per-observation {track_index: prob} for
    all candidates with prob >= ALT_P).
    """
    if not tracks or not batch:
        return {}, [{} for _ in batch]
    lr = np.array([[likelihood_ratio(t, o, area_m2) for o in batch] for t in tracks])
    col = lr.sum(0)
    row = lr.sum(1)
    probs = lr / (1.0 + col[None, :] + row[:, None] - lr + 1e-12)
    cost = np.where(lr > 1.0, -np.log(np.maximum(lr, 1e-300)), 1e6)
    ri, ci = linear_sum_assignment(cost)
    chosen = {}
    for i, j in zip(ri, ci, strict=True):
        if cost[i, j] < 1e6 and probs[i, j] >= LINK_P:
            chosen[j] = (i, float(probs[i, j]))
    cands = [{i: float(probs[i, j]) for i in range(len(tracks)) if probs[i, j] >= ALT_P} for j in range(len(batch))]
    return chosen, cands


def track_site(
    obs: list[Observation],
    area_m2: float,
    coverage: dict[str, tuple[float, float, float, float]] | None = None,
) -> tuple[list[Track], dict[str, dict[str, float]]]:
    """Rebuild tracks for one site from all its observations (deterministic, time ordered).

    ``coverage`` maps scene_id -> bbox of what that scene actually imaged (cloud-free
    enough), used to count misses: a track not re-detected in two scenes that covered
    it is marked "departed".
    Returns tracks and, per observation id, {track_id: prob} alternatives.
    """
    tracks: list[Track] = []
    alts: dict[str, dict[str, float]] = {}
    by_fam: dict[str, list[Observation]] = defaultdict(list)
    for o in obs:
        by_fam[family(o.cls)].append(o)
    for fam, items in by_fam.items():
        scenes: dict[str, list[Observation]] = defaultdict(list)
        for o in items:
            scenes[o.scene_id].append(o)
        order = sorted(scenes, key=lambda sid: min(o.time for o in scenes[sid]))
        fam_tracks: list[Track] = []
        for sid in order:
            batch = scenes[sid]
            when = min(o.time for o in batch)
            live = [
                t
                for t in fam_tracks
                if t.status == "active" and (when - t.last.time).total_seconds() <= LOST_AFTER_DAYS.get(fam, 30) * 86400
            ]
            chosen, cands = associate(live, batch, area_m2)
            hit = set()
            for j, o in enumerate(batch):
                if j in chosen:
                    i, p = chosen[j]
                    live[i].obs.append(o)
                    live[i].probs.append(p)
                    live[i].misses = 0
                    hit.add(live[i].id)
                    tid = live[i].id
                else:
                    t = Track(id=_track_id(o), fam=fam, obs=[o], probs=[1.0])
                    fam_tracks.append(t)
                    tid, p = t.id, 1.0
                o.attrs["track_id"], o.attrs["link_prob"] = tid, round(p, 3)
                alts[o.id] = {live[i].id: round(p_, 3) for i, p_ in cands[j].items() if live[i].id != tid}
            box = (coverage or {}).get(sid)
            for t in live:
                if t.id in hit or t.last.scene_id == sid:
                    continue
                if box is None or (box[0] <= t.last.lon <= box[2] and box[1] <= t.last.lat <= box[3]):
                    t.misses += 1
                    if t.misses >= 2:
                        t.status = "departed"
        for t in fam_tracks:
            if t.status == "active" and order:
                gap = (datetime.fromisoformat(max(o.time for o in items).isoformat()) - t.last.time).total_seconds()
                if gap > LOST_AFTER_DAYS.get(fam, 30) * 86400:
                    t.status = "lost"
        tracks += fam_tracks
    return tracks, alts


def _track_id(o: Observation) -> str:
    return "T" + hashlib.sha1(o.id.encode()).hexdigest()[:11]
