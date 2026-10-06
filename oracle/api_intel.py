"""Web API for the intelligence layer: objects, tracks, events, sites, sweeps, briefs."""

from __future__ import annotations

import functools
import io
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from . import jobs
from .geo import AOI, parse_aoi
from .guard import check_scene
from .imagery import NoData, chip
from .models import Scene
from .observations import feature_collection
from .pipeline import Site, add_preset, process_site, refresh_site, sweep
from .presets import PRESETS
from .store import Store

router = APIRouter(prefix="/api")


@functools.lru_cache(maxsize=1)
def store() -> Store:
    return Store()


def _bbox(text: str | None) -> tuple[float, float, float, float] | None:
    if not text:
        return None
    try:
        w, s, e, n = map(float, text.split(","))
    except ValueError:
        raise HTTPException(400, "bbox must be west,south,east,north") from None
    return (w, s, e, n)


def _ago(days: float | None) -> datetime | None:
    return datetime.now(timezone.utc) - timedelta(days=days) if days else None


# --------------------------------------------------------------------------- objects & tracks


@router.get("/objects")
def objects(
    bbox: str | None = None,
    site: str | None = None,
    days: float | None = None,
    classes: str | None = None,
    limit: int = Query(20000, le=100000),
) -> dict:
    obs = store().observations(
        bbox=_bbox(bbox), start=_ago(days), site=site, classes=[c for c in (classes or "").split(",") if c], limit=limit
    )
    return feature_collection(obs, count=len(obs))


@router.get("/tracks")
def tracks(bbox: str | None = None, site: str | None = None, min_obs: int = 2, status: str | None = None) -> dict:
    feats = []
    for t in store().tracks(site=site, status=status, bbox=_bbox(bbox)):
        if t["n_obs"] < min_obs:
            continue
        path = t["attrs"].get("path") or []
        feats.append(
            {
                "type": "Feature",
                "id": t["id"],
                "geometry": {"type": "LineString", "coordinates": [[p[0], p[1]] for p in path]},
                "properties": {k: v for k, v in t.items() if k not in ("attrs",)}
                | {
                    "dwell_days": t["attrs"].get("dwell_days"),
                    "sources": t["attrs"].get("sources"),
                    "times": [p[2] for p in path],
                },
            }
        )
    return {"type": "FeatureCollection", "features": feats}


@router.get("/track/{track_id}")
def track(track_id: str) -> dict:
    t = store().track(track_id)
    if not t:
        raise HTTPException(404, "no such track")
    return t


@router.get("/obs/{obs_id}")
def obs_one(obs_id: str) -> dict:
    r = store().conn.execute("SELECT * FROM observations WHERE id=?", (obs_id,)).fetchone()
    if not r:
        raise HTTPException(404, "no such observation")
    return store()._row_obs(r).feature()


@router.get("/obs/{obs_id}/chip.png")
def obs_chip(obs_id: str, size_m: float = Query(500, ge=50, le=5000), px: int = Query(256, ge=64, le=1024)) -> Response:
    row = store().conn.execute("SELECT scene_id, lat, lon FROM observations WHERE id=?", (obs_id,)).fetchone()
    if not row:
        raise HTTPException(404, "no such observation")
    return Response(
        _chip_png(row["scene_id"], row["lat"], row["lon"], size_m, px),
        media_type="image/png",
        headers={"Cache-Control": "max-age=86400"},
    )


@functools.lru_cache(maxsize=512)
def _chip_png(scene_id: str, lat: float, lon: float, size_m: float, px: int) -> bytes:
    scene = store().scene(scene_id)
    if scene is None:
        raise HTTPException(404, "scene not stored")
    try:
        c = chip(scene, AOI.from_point(lat, lon, size_m / 2000), max_pixels=px)
    except NoData as exc:
        raise HTTPException(404, str(exc)) from None
    img = c.image(label=False).convert("RGB").resize((px, px))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- events, stats, brief


@router.get("/events")
def events(days: float = 30, site: str | None = None, limit: int = Query(300, le=5000), min_severity: float = 0.0) -> list[dict]:
    return [e for e in store().events(since=_ago(days), site=site, limit=limit) if e["severity"] >= min_severity]


@router.get("/stats")
def stats() -> dict:
    return store().stats()


@router.get("/brief")
def brief(days: float = 7, llm: str = "auto", site: str | None = None) -> dict:
    from .brief import brief as make

    use = {"auto": None, "on": True, "off": False}.get(llm)
    try:
        md, engine = make(store(), since=_ago(days), sites=[site] if site else None, use_llm=use)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from None
    return {"markdown": md, "engine": engine}


# --------------------------------------------------------------------------- sites


class SiteBody(BaseModel):
    name: str
    where: str | None = None
    bbox: list[float] | None = None
    radius_km: float = 5.0
    kind: str = "maritime"


@router.get("/sites")
def sites() -> list[dict]:
    st = store()
    out = []
    counts = defaultdict(int)
    for r in st.conn.execute("SELECT site, COUNT(*) n FROM observations GROUP BY site"):
        counts[r["site"]] = r["n"]
    ev = defaultdict(float)
    for e in st.events(since=_ago(30), limit=5000):
        ev[e["site"]] = max(ev[e["site"]], e["severity"])
    for s in st.sites():
        s["observations"] = counts.get(s["name"], 0)
        s["max_severity_30d"] = ev.get(s["name"], 0.0)
        out.append(s)
    return out


@router.post("/sites")
def add_site(body: SiteBody) -> dict:
    try:
        bbox = tuple(body.bbox) if body.bbox else parse_aoi(body.where or "", body.radius_km).bbox
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    if len(bbox) != 4:
        raise HTTPException(400, "bbox must be [w,s,e,n]")
    Site(body.name, bbox, body.kind).save(store())
    return {"ok": True, "name": body.name, "bbox": bbox}


@router.delete("/sites/{name}")
def delete_site(name: str) -> dict:
    return {"deleted": store().delete_site(name)}


@router.post("/sites/preset/{preset}")
def preset(preset: str) -> dict:
    if preset not in PRESETS:
        raise HTTPException(404, f"presets: {', '.join(PRESETS)}")
    return {"added": [s.name for s in add_preset(store(), preset)]}


@router.get("/sites/{name}/series")
def series(name: str) -> dict:
    """Objects per class per processed image, for activity charts."""
    st = store()
    per = defaultdict(lambda: defaultdict(int))
    for r in st.conn.execute("SELECT scene_id, cls, COUNT(*) n FROM observations WHERE site=? GROUP BY scene_id, cls", (name,)):
        per[r["scene_id"]][r["cls"]] = r["n"]
    rows = []
    for r in st.runs(name):
        if r["status"] == "ok" and r.get("scene_time"):
            rows.append(
                {
                    "time": r["scene_time"],
                    "source": r["source"],
                    "clear": r.get("clear"),
                    "counts": dict(per.get(r["scene_id"], {})),
                }
            )
    return {"site": name, "series": rows}


# --------------------------------------------------------------------------- jobs: sweep & detect


class SweepBody(BaseModel):
    sites: list[str] | None = None
    lookback_days: int = 30
    max_scenes: int = 6


@router.post("/sweep")
def start_sweep(body: SweepBody) -> dict:
    st = store()
    rows = [r for r in st.sites() if not body.sites or r["name"] in body.sites]
    if not rows:
        raise HTTPException(400, "no sites")
    jid = jobs.submit("sweep", sweep, st, [Site.from_row(r) for r in rows], body.lookback_days, body.max_scenes)
    return {"job": jid}


class DetectBody(BaseModel):
    scene: dict
    bbox: list[float]
    detector: str = "auto"  # ships | objects | auto
    prompts: list[str] | None = None
    site: str | None = None


@router.post("/detect")
def start_detect(body: DetectBody) -> dict:
    scene = Scene.from_dict(body.scene)
    check_scene(scene)
    if len(body.bbox) != 4:
        raise HTTPException(400, "bbox must be [w,s,e,n]")
    aoi = AOI(tuple(body.bbox))
    w, h = aoi.size_km()
    det = body.detector
    if det == "auto":
        det = "ships" if (scene.source == "sentinel-2" or scene.sensor == "sar") else "objects"
    limit = 2500 if det == "ships" else 25
    if w * h > limit:
        raise HTTPException(400, f"area too large for {det} in the web app ({w * h:.0f} km2 > {limit}); use the CLI")
    jid = jobs.submit("detect", _detect, scene, aoi, det, body.prompts, body.site)
    return {"job": jid}


def _detect(scene: Scene, aoi: AOI, det: str, prompts: list[str] | None, site: str | None) -> dict:
    from .detect import detect_ships

    if det == "ships":
        obs = detect_ships(scene, aoi).observations()
    else:
        from .objdet import detect_objects

        obs = detect_objects(scene, aoi, prompts=prompts)
    st = store()
    site_name = site or f"adhoc {scene.date} {aoi.center[0]:.3f},{aoi.center[1]:.3f}"
    if not any(s["name"] == site_name for s in st.sites()):
        Site(site_name, aoi.bbox, "maritime" if det == "ships" else "ground").save(st)
    st.add_scene(scene)
    n = st.add_observations(obs, site_name)
    st.record_run(site_name, scene.id, det, n)
    refresh_site(st, Site.from_row(next(s for s in st.sites() if s["name"] == site_name)))
    return feature_collection(st.observations(site=site_name, bbox=aoi.bbox), site=site_name, scene=scene.id, detector=det)


@router.post("/sites/{name}/process")
def process_one(name: str, lookback_days: int = 30, max_scenes: int = 6) -> dict:
    st = store()
    row = next((s for s in st.sites() if s["name"] == name), None)
    if not row:
        raise HTTPException(404, "no such site")
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    return {"job": jobs.submit("process", process_site, st, Site.from_row(row), since, None, max_scenes)}


# --------------------------------------------------------------------------- agent


class AskBody(BaseModel):
    question: str
    llm: str = "auto"  # auto | on | off
    web: bool = True
    max_calls: int = 20
    effort: str = "high"


@router.post("/investigate")
def start_investigation(body: AskBody) -> dict:
    from .agent import Investigation, investigate, register

    q = body.question.strip()
    if not q or len(q) > 2000:
        raise HTTPException(400, "question must be 1-2000 characters")
    if body.effort not in ("low", "medium", "high", "xhigh", "max"):
        raise HTTPException(400, "effort: low | medium | high | xhigh | max")
    inv = register(Investigation(q))
    use = {"auto": None, "on": True, "off": False}.get(body.llm)
    jid = jobs.submit(
        "investigate",
        lambda: investigate(q, store(), use, max(1, min(body.max_calls, 40)), body.effort, inv=inv, web=body.web).id,
    )
    return {"id": inv.id, "job": jid}


@router.get("/investigations")
def investigation_list(limit: int = 30) -> list[dict]:
    return store().investigations(limit)


@router.get("/investigations/{inv_id}")
def investigation(inv_id: str) -> dict:
    from .agent import live

    d = live(inv_id) or store().investigation(inv_id)
    if not d:
        raise HTTPException(404, "no such investigation")
    return d


# --------------------------------------------------------------------------- change detection


class ChangeBody(BaseModel):
    bbox: list[float] | None = None
    where: str | None = None
    radius_km: float = 3.0
    source: str = "sentinel-2"
    after: str | None = None  # YYYY-MM-DD
    before: str | None = None
    baseline: str = "recent"
    site: str | None = None


@router.post("/change")
def start_change(body: ChangeBody) -> dict:
    try:
        aoi = AOI(tuple(body.bbox)) if body.bbox else parse_aoi(body.where or "", body.radius_km)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    w, h = aoi.size_km()
    if w * h > 1600:
        raise HTTPException(400, f"area too large for the web app ({w * h:.0f} km2 > 1600); use the CLI")
    if body.source not in ("sentinel-2", "sentinel-1"):
        raise HTTPException(400, "source must be sentinel-2 or sentinel-1")
    return {"job": jobs.submit("change", _change, aoi, body)}


def _change(aoi: AOI, body: ChangeBody) -> dict:
    from .change import detect_change, remember, to_events
    from .models import parse_dt

    res = remember(
        detect_change(
            aoi,
            body.source,
            after=parse_dt(body.after + "T23:59:59Z") if body.after else None,
            before=parse_dt(body.before + "T23:59:59Z") if body.before else None,
            baseline=body.baseline,
        )
    )
    if body.site:
        store().add_events(to_events(res, body.site))
    return res.geojson()


@router.get("/change/{cid}/{which}.png")
def change_image(cid: str, which: str, bbox: str | None = None) -> Response:
    from .change import recall

    res = recall(cid)
    if res is None:
        raise HTTPException(404, "change result expired; run it again")
    if which not in ("before", "after", "overlay", "change"):
        raise HTTPException(404, "before | after | overlay | change")
    return Response(res.png(which, crop=_bbox(bbox)), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


# --------------------------------------------------------------------------- foundation model, SR, airborne


class EmbedBody(BaseModel):
    where: str | None = None
    bbox: list[float] | None = None
    radius_km: float = 10.0
    kind: str = "similar"  # similar | change | view
    examples: list[list[float]] | None = None  # [[lat, lon], ...]
    negatives: list[list[float]] | None = None
    year: int = 2025
    year_from: int = 2017
    segments: int = 0


def _area(where: str | None, bbox: list[float] | None, radius_km: float, max_km: float) -> AOI:
    try:
        aoi = AOI(tuple(bbox)) if bbox else parse_aoi(where or "", radius_km)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    w, h = aoi.size_km()
    if max(w, h) > max_km:
        raise HTTPException(400, f"area too large ({max(w, h):.0f} km across > {max_km:g})")
    return aoi


@router.post("/embed")
def start_embed(body: EmbedBody) -> dict:
    aoi = _area(body.where, body.bbox, body.radius_km, 101)
    if body.kind == "similar" and not body.examples:
        raise HTTPException(400, "give examples: [[lat, lon], ...]")
    return {"job": jobs.submit("embed", _embed, aoi, body)}


def _embed(aoi: AOI, body: EmbedBody) -> dict:
    from . import embeddings as E

    if body.kind == "similar":
        r = E.find_similar(
            aoi, [tuple(p) for p in body.examples or []], body.year, [tuple(p) for p in body.negatives or []] or None
        )
    elif body.kind == "change":
        r = E.semantic_change(aoi, body.year_from, body.year)
    else:
        r = E.embedding_view(aoi, body.year, body.segments)
    return E.remember(r).geojson()


@router.get("/embed/{rid}/overlay.png")
def embed_overlay(rid: str) -> Response:
    from .embeddings import recall

    r = recall(rid)
    if r is None:
        raise HTTPException(404, "result expired; run it again")
    return Response(r.png(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@router.get("/embed/years")
def embed_years(lat: float, lon: float) -> dict:
    from .embeddings import available_years

    return {"years": available_years(lat, lon)}


class EnhanceBody(BaseModel):
    scene: dict
    bbox: list[float]


_SR: dict[str, object] = {}


@router.post("/enhance")
def start_enhance(body: EnhanceBody) -> dict:
    scene = Scene.from_dict(body.scene)
    check_scene(scene)
    if scene.source != "sentinel-2":
        raise HTTPException(400, "super-resolution works on Sentinel-2 scenes")
    aoi = _area(None, body.bbox, 0, 7.7)
    return {"job": jobs.submit("enhance", _enhance, scene, aoi)}


def _enhance(scene: Scene, aoi: AOI) -> dict:
    import hashlib

    from pyproj import Transformer

    from .superres import NOTICE, enhance

    r = enhance(scene, aoi)
    rid = hashlib.sha1(f"{scene.id}{aoi.bbox}".encode()).hexdigest()[:12]
    _SR[rid] = r
    while len(_SR) > 6:
        _SR.pop(next(iter(_SR)))
    w, s, e, n = r.grid.bounds
    tr = Transformer.from_crs(r.grid.crs, 4326, always_xy=True)
    corners = [list(tr.transform(x, y)) for x, y in ((w, n), (e, n), (e, s), (w, s))]
    return {"id": rid, "corners": corners, "variant": r.variant, "notice": NOTICE, "scene": scene.id, "resolution_m": 2.5}


@router.get("/enhance/{rid}/{which}.png")
def enhance_png(rid: str, which: str) -> Response:
    r = _SR.get(rid)
    if r is None or which not in ("sr", "lr"):
        raise HTTPException(404, "result expired or unknown image")
    return Response(r.png(which), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


class AirborneBody(BaseModel):
    where: str | None = None
    bbox: list[float] | None = None
    radius_km: float = 10.0
    scene: dict | None = None


@router.post("/airborne")
def start_airborne(body: AirborneBody) -> dict:
    aoi = _area(body.where, body.bbox, body.radius_km, 31)
    return {"job": jobs.submit("airborne", _airborne, aoi, body.scene)}


def _airborne(aoi: AOI, scene_dict: dict | None) -> dict:
    from .airborne import detect_airborne
    from .tools import Toolbox

    if scene_dict:
        scene = Scene.from_dict(scene_dict)
        check_scene(scene)
    else:
        scene = Toolbox(store())._scene_for(aoi, None, None, ["sentinel-2"])
    store().add_scene(scene)
    acs = detect_airborne(scene, aoi)
    feats = []
    for a in acs:
        d = a.to_dict()
        d["chip"] = f"/api/scenes/{scene.id}/chip.png?lat={a.lat:.6f}&lon={a.lon:.6f}&size_m=900"
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [a.lon, a.lat]}, "properties": d})
    return {"type": "FeatureCollection", "properties": {"scene": scene.id, "time": scene.datetime.isoformat()}, "features": feats}


@router.get("/scenes/{scene_id}/chip.png")
def scene_chip(
    scene_id: str, lat: float, lon: float, size_m: float = Query(800, ge=50, le=10000), px: int = Query(384, ge=64, le=1024)
) -> Response:
    return Response(_chip_png(scene_id, lat, lon, size_m, px), media_type="image/png", headers={"Cache-Control": "max-age=86400"})


# --------------------------------------------------------------------------- orbits & patterns


@router.get("/passes")
def passes(lat: float, lon: float, days: float = Query(7.0, gt=0, le=30), all: bool = False) -> dict:
    from .passes import next_passes

    try:
        ps = next_passes(lat, lon, days=days)
    except Exception as exc:  # noqa: BLE001 - TLE download or propagation failure
        raise HTTPException(503, f"orbit prediction unavailable: {exc}") from None
    return {"lat": lat, "lon": lon, "days": days, "passes": [p.to_dict() for p in ps if p.likely or all]}


@router.get("/sites/{name}/heatmap")
def heatmap(name: str, family: str = "vessel") -> dict:
    """Pattern-of-life density: where objects of a family usually are, per look per km^2."""
    from .patterns import density_grid

    st = store()
    row = next((s for s in st.sites() if s["name"] == name), None)
    if not row:
        raise HTTPException(404, "no such site")
    looks = sum(1 for r in st.runs(name) if r["status"] == "ok" and (r.get("clear") or 1.0) >= 0.85)
    return density_grid(st.observations(site=name), tuple(row["bbox"]), family, looks)


@router.get("/jobs/{jid}")
def job(jid: str) -> dict:
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "no such job")
    return j


@router.get("/jobs")
def job_list() -> list[dict]:
    return jobs.all_jobs()
