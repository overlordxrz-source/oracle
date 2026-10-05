"""Persistent object database (SQLite + R*Tree): scenes, observations, tracks, events, AIS.

One file, no server, handles hundreds of thousands of observations comfortably. Default
location ``$ORACLE_CACHE/oracle.db`` (override with ``ORACLE_DB``).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import CACHE_DIR
from .models import Scene, parse_dt
from .observations import Observation

SCHEMA = """
CREATE TABLE IF NOT EXISTS scenes(
  id TEXT PRIMARY KEY, source TEXT, platform TEXT, sensor TEXT, time TEXT, gsd REAL,
  w REAL, s REAL, e REAL, n REAL, cloud REAL, data TEXT);
CREATE TABLE IF NOT EXISTS observations(
  rid INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, scene_id TEXT, site TEXT, source TEXT,
  detector TEXT, cls TEXT, time TEXT, lat REAL, lon REAL, confidence REAL, length_m REAL,
  width_m REAL, axis_deg REAL, course_deg REAL, speed_ms REAL, speed_err_ms REAL,
  polygon TEXT, attrs TEXT, track_id TEXT, link_prob REAL);
CREATE INDEX IF NOT EXISTS obs_time ON observations(time);
CREATE INDEX IF NOT EXISTS obs_site ON observations(site, cls, time);
CREATE INDEX IF NOT EXISTS obs_track ON observations(track_id);
CREATE VIRTUAL TABLE IF NOT EXISTS obs_rtree USING rtree(rid, minx, maxx, miny, maxy);
CREATE TABLE IF NOT EXISTS tracks(
  id TEXT PRIMARY KEY, site TEXT, cls TEXT, status TEXT, first_seen TEXT, last_seen TEXT,
  n_obs INTEGER, lat REAL, lon REAL, length_m REAL, speed_ms REAL, course_deg REAL,
  distance_m REAL, mean_link_prob REAL, attrs TEXT);
CREATE INDEX IF NOT EXISTS tracks_site ON tracks(site, status);
CREATE TABLE IF NOT EXISTS links(
  obs_id TEXT, track_id TEXT, prob REAL, chosen INTEGER, PRIMARY KEY(obs_id, track_id));
CREATE TABLE IF NOT EXISTS sites(
  name TEXT PRIMARY KEY, w REAL, s REAL, e REAL, n REAL, kind TEXT, data TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS runs(
  site TEXT, scene_id TEXT, detector TEXT, time TEXT, n INTEGER, status TEXT, error TEXT, clear REAL,
  PRIMARY KEY(site, scene_id, detector));
CREATE TABLE IF NOT EXISTS events(
  id TEXT PRIMARY KEY, site TEXT, time TEXT, kind TEXT, severity REAL, title TEXT,
  detail TEXT, obs_id TEXT, track_id TEXT, created TEXT);
CREATE INDEX IF NOT EXISTS events_time ON events(time);
CREATE TABLE IF NOT EXISTS ais(
  mmsi INTEGER, time TEXT, lat REAL, lon REAL, sog REAL, cog REAL, name TEXT,
  length_m REAL, vtype TEXT, PRIMARY KEY(mmsi, time));
CREATE INDEX IF NOT EXISTS ais_time ON ais(time);
CREATE TABLE IF NOT EXISTS investigations(
  id TEXT PRIMARY KEY, question TEXT, created TEXT, finished TEXT, status TEXT, engine TEXT,
  answer TEXT, data TEXT);
"""

OBS_COLS = (
    "id scene_id site source detector cls time lat lon confidence length_m width_m axis_deg "
    "course_deg speed_ms speed_err_ms polygon attrs track_id link_prob"
).split()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_path() -> Path:
    return Path(os.environ.get("ORACLE_DB", CACHE_DIR / "oracle.db"))


class Store:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.conn.executescript(SCHEMA)  # executescript manages its own transaction

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=60, isolation_level=None)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = c
        return c

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        c = self.conn
        c.execute("BEGIN IMMEDIATE")
        try:
            yield c
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------ scenes
    def add_scene(self, s: Scene) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO scenes VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    s.id,
                    s.source,
                    s.platform,
                    s.sensor,
                    s.datetime.isoformat(),
                    s.gsd,
                    *s.bbox,
                    s.cloud_cover,
                    json.dumps(s.to_dict()),
                ),
            )

    def scene(self, scene_id: str) -> Scene | None:
        r = self.conn.execute("SELECT data FROM scenes WHERE id=?", (scene_id,)).fetchone()
        return Scene.from_dict(json.loads(r["data"])) if r else None

    # ------------------------------------------------------------------ observations
    def add_observations(self, obs: Iterable[Observation], site: str | None = None) -> int:
        n = 0
        with self.tx() as c:
            for o in obs:
                row = self._obs_row(o, site)
                cur = c.execute(
                    f"INSERT OR IGNORE INTO observations({','.join(OBS_COLS)}) VALUES({','.join('?' * len(OBS_COLS))})",
                    row,
                )
                if cur.rowcount:
                    c.execute("INSERT INTO obs_rtree VALUES(?,?,?,?,?)", (cur.lastrowid, o.lon, o.lon, o.lat, o.lat))
                    n += 1
        return n

    @staticmethod
    def _obs_row(o: Observation, site: str | None) -> tuple:
        return (
            o.id,
            o.scene_id,
            site,
            o.source,
            o.detector,
            o.cls,
            o.time.isoformat(),
            o.lat,
            o.lon,
            o.confidence,
            o.length_m,
            o.width_m,
            o.axis_deg,
            o.course_deg,
            o.speed_ms,
            o.speed_err_ms,
            json.dumps(o.polygon) if o.polygon else None,
            json.dumps(o.attrs),
            o.attrs.get("track_id"),
            None,
        )

    def observations(
        self,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        site: str | None = None,
        classes: Iterable[str] | None = None,
        track_id: str | None = None,
        limit: int = 50_000,
    ) -> list[Observation]:
        q, args = ["SELECT o.* FROM observations o"], []
        where = []
        if bbox:
            q.append("JOIN obs_rtree r ON r.rid=o.rid")
            where += ["r.minx>=?", "r.maxx<=?", "r.miny>=?", "r.maxy<=?"]
            args += [bbox[0], bbox[2], bbox[1], bbox[3]]
        if start:
            where.append("o.time>=?")
            args.append(start.isoformat())
        if end:
            where.append("o.time<=?")
            args.append(end.isoformat())
        if site:
            where.append("o.site=?")
            args.append(site)
        if track_id:
            where.append("o.track_id=?")
            args.append(track_id)
        cls = list(classes or [])
        if cls:
            where.append(f"o.cls IN ({','.join('?' * len(cls))})")
            args += cls
        if where:
            q.append("WHERE " + " AND ".join(where))
        q.append("ORDER BY o.time LIMIT ?")
        args.append(limit)
        return [self._row_obs(r) for r in self.conn.execute(" ".join(q), args)]

    @staticmethod
    def _row_obs(r: sqlite3.Row) -> Observation:
        attrs = json.loads(r["attrs"] or "{}")
        if r["track_id"]:
            attrs["track_id"] = r["track_id"]
            attrs["link_prob"] = r["link_prob"]
        attrs["site"] = r["site"]
        return Observation(
            cls=r["cls"],
            lat=r["lat"],
            lon=r["lon"],
            time=parse_dt(r["time"]),
            scene_id=r["scene_id"],
            source=r["source"],
            detector=r["detector"],
            confidence=r["confidence"],
            length_m=r["length_m"],
            width_m=r["width_m"],
            axis_deg=r["axis_deg"],
            course_deg=r["course_deg"],
            speed_ms=r["speed_ms"],
            speed_err_ms=r["speed_err_ms"],
            polygon=json.loads(r["polygon"]) if r["polygon"] else None,
            attrs=attrs,
            id=r["id"],
        )

    def update_obs_attrs(self, obs_id: str, **attrs: Any) -> None:
        with self.tx() as c:
            r = c.execute("SELECT attrs FROM observations WHERE id=?", (obs_id,)).fetchone()
            if r:
                cur = json.loads(r["attrs"] or "{}")
                cur.update(attrs)
                c.execute("UPDATE observations SET attrs=? WHERE id=?", (json.dumps(cur), obs_id))

    # ------------------------------------------------------------------ tracks
    def save_track(self, t: dict) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO tracks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    t["id"],
                    t["site"],
                    t["cls"],
                    t["status"],
                    t["first_seen"],
                    t["last_seen"],
                    t["n_obs"],
                    t["lat"],
                    t["lon"],
                    t.get("length_m"),
                    t.get("speed_ms"),
                    t.get("course_deg"),
                    t.get("distance_m"),
                    t.get("mean_link_prob"),
                    json.dumps(t.get("attrs", {})),
                ),
            )

    def link(self, obs_id: str, track_id: str, prob: float, alternatives: dict[str, float] | None = None) -> None:
        with self.tx() as c:
            c.execute("UPDATE observations SET track_id=?, link_prob=? WHERE id=?", (track_id, prob, obs_id))
            c.execute("INSERT OR REPLACE INTO links VALUES(?,?,?,1)", (obs_id, track_id, prob))
            for tid, p in (alternatives or {}).items():
                if tid != track_id:
                    c.execute("INSERT OR REPLACE INTO links VALUES(?,?,?,0)", (obs_id, tid, p))

    def save_tracking(
        self, site: str, summaries: list[dict], links: dict[str, tuple[str, float]], alternatives: dict[str, dict[str, float]]
    ) -> None:
        """Replace a site's tracks and observation links in one transaction."""
        with self.tx() as c:
            c.execute("DELETE FROM links WHERE obs_id IN (SELECT id FROM observations WHERE site=?)", (site,))
            c.execute("DELETE FROM tracks WHERE site=?", (site,))
            c.executemany(
                "INSERT INTO tracks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        t["id"],
                        t["site"],
                        t["cls"],
                        t["status"],
                        t["first_seen"],
                        t["last_seen"],
                        t["n_obs"],
                        t["lat"],
                        t["lon"],
                        t.get("length_m"),
                        t.get("speed_ms"),
                        t.get("course_deg"),
                        t.get("distance_m"),
                        t.get("mean_link_prob"),
                        json.dumps(t.get("attrs", {})),
                    )
                    for t in summaries
                ],
            )
            c.executemany(
                "UPDATE observations SET track_id=?, link_prob=? WHERE id=?", [(tid, p, oid) for oid, (tid, p) in links.items()]
            )
            rows = [(oid, tid, p, 1) for oid, (tid, p) in links.items()]
            rows += [(oid, tid, p, 0) for oid, alts in alternatives.items() for tid, p in alts.items()]
            c.executemany("INSERT OR REPLACE INTO links VALUES(?,?,?,?)", rows)

    def tracks(self, site: str | None = None, status: str | None = None, bbox=None, limit: int = 20_000) -> list[dict]:
        q, args = "SELECT * FROM tracks WHERE 1=1", []
        if site:
            q += " AND site=?"
            args.append(site)
        if status:
            q += " AND status=?"
            args.append(status)
        if bbox:
            q += " AND lon>=? AND lon<=? AND lat>=? AND lat<=?"
            args += [bbox[0], bbox[2], bbox[1], bbox[3]]
        q += " ORDER BY last_seen DESC LIMIT ?"
        args.append(limit)
        out = []
        for r in self.conn.execute(q, args):
            d = dict(r)
            d["attrs"] = json.loads(d["attrs"] or "{}")
            out.append(d)
        return out

    def track(self, track_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM tracks WHERE id=?", (track_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["attrs"] = json.loads(d["attrs"] or "{}")
        d["observations"] = [o.to_dict() for o in self.observations(track_id=track_id)]
        d["alternatives"] = [
            dict(x)
            for x in self.conn.execute(
                "SELECT obs_id, track_id, prob FROM links WHERE chosen=0 AND obs_id IN "
                "(SELECT id FROM observations WHERE track_id=?)",
                (track_id,),
            )
        ]
        return d

    # ------------------------------------------------------------------ sites
    def save_site(self, name: str, bbox: tuple, kind: str = "", **data: Any) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO sites VALUES(?,?,?,?,?,?,?,COALESCE((SELECT created FROM sites WHERE name=?),?))",
                (name, *bbox, kind, json.dumps(data), name, _now()),
            )

    def sites(self) -> list[dict]:
        out = []
        for r in self.conn.execute("SELECT * FROM sites ORDER BY name"):
            d = dict(r)
            d["bbox"] = (d.pop("w"), d.pop("s"), d.pop("e"), d.pop("n"))
            d.update(json.loads(d.pop("data") or "{}"))
            out.append(d)
        return out

    def delete_site(self, name: str) -> bool:
        with self.tx() as c:
            return c.execute("DELETE FROM sites WHERE name=?", (name,)).rowcount > 0

    # ------------------------------------------------------------------ runs / events
    def processed(self, site: str, scene_id: str, detector: str) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM runs WHERE site=? AND scene_id=? AND detector=? AND status='ok'", (site, scene_id, detector)
            ).fetchone()
            is not None
        )

    def record_run(
        self, site: str, scene_id: str, detector: str, n: int, status: str = "ok", error: str = "", clear: float | None = None
    ) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO runs VALUES(?,?,?,?,?,?,?,?)",
                (site, scene_id, detector, _now(), n, status, error, clear),
            )

    def runs(self, site: str | None = None) -> list[dict]:
        q = "SELECT r.*, s.time AS scene_time, s.source, s.w, s.s, s.e, s.n FROM runs r LEFT JOIN scenes s ON s.id=r.scene_id"
        args: list = []
        if site:
            q += " WHERE r.site=?"
            args.append(site)
        return [dict(r) for r in self.conn.execute(q + " ORDER BY s.time", args)]

    def add_events(self, events: Iterable[dict]) -> int:
        n = 0
        with self.tx() as c:
            for e in events:
                n += c.execute(
                    "INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        e["id"],
                        e.get("site"),
                        e["time"],
                        e["kind"],
                        e.get("severity", 0.5),
                        e["title"],
                        json.dumps(e.get("detail", {})),
                        e.get("obs_id"),
                        e.get("track_id"),
                        _now(),
                    ),
                ).rowcount
        return n

    def replace_events(self, site: str, events: list[dict]) -> int:
        """Make a site's events exactly ``events`` (derived data), keeping creation times of
        ones that already existed. Returns how many are new."""
        ids = [e["id"] for e in events]
        with self.tx() as c:
            c.execute("CREATE TEMP TABLE IF NOT EXISTS keep_ids(id TEXT PRIMARY KEY)")
            c.execute("DELETE FROM keep_ids")
            c.executemany("INSERT OR IGNORE INTO keep_ids VALUES(?)", [(i,) for i in ids])
            # change_* events come from change detection runs, not from the tracker: keep them.
            c.execute(
                "DELETE FROM events WHERE site=? AND substr(kind, 1, 7) != 'change_' AND id NOT IN (SELECT id FROM keep_ids)",
                (site,),
            )
        return self.add_events(events)

    def events(self, since: datetime | None = None, site: str | None = None, limit: int = 500) -> list[dict]:
        q, args = "SELECT * FROM events WHERE 1=1", []
        if since:
            q += " AND time>=?"
            args.append(since.isoformat())
        if site:
            q += " AND site=?"
            args.append(site)
        q += " ORDER BY severity DESC, time DESC LIMIT ?"
        args.append(limit)
        out = []
        for r in self.conn.execute(q, args):
            d = dict(r)
            d["detail"] = json.loads(d["detail"] or "{}")
            out.append(d)
        return out

    # ------------------------------------------------------------------ investigations
    def save_investigation(self, inv: dict) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO investigations VALUES(?,?,?,?,?,?,?,?)",
                (
                    inv["id"],
                    inv["question"],
                    inv["created"],
                    inv.get("finished"),
                    inv["status"],
                    inv.get("engine"),
                    inv.get("answer"),
                    json.dumps({k: inv.get(k) for k in ("steps", "evidence", "provenance", "error")}, default=str),
                ),
            )

    def investigation(self, inv_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM investigations WHERE id=?", (inv_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d.update(json.loads(d.pop("data") or "{}"))
        return d

    def investigations(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, question, created, finished, status, engine FROM investigations ORDER BY created DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        c = self.conn
        return {
            "observations": c.execute("SELECT COUNT(*) FROM observations").fetchone()[0],
            "tracks": c.execute("SELECT COUNT(*) FROM tracks").fetchone()[0],
            "events": c.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "scenes": c.execute("SELECT COUNT(*) FROM scenes").fetchone()[0],
            "sites": c.execute("SELECT COUNT(*) FROM sites").fetchone()[0],
            "ais_points": c.execute("SELECT COUNT(*) FROM ais").fetchone()[0],
            "by_class": {r[0]: r[1] for r in c.execute("SELECT cls, COUNT(*) FROM observations GROUP BY cls")},
        }
