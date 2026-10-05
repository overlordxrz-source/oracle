"""Oracle web app: map UI + JSON API + on-the-fly tiles for any scene."""

from __future__ import annotations

import base64
import functools
import io
import json
import tempfile
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from . import __version__
from .api_intel import router as intel_router
from .detect import detect_ships
from .geo import AOI, geocode, parse_aoi
from .guard import check_href, check_scene
from .imagery import NoData, chip, render_tile, transparent_png
from .models import Render, Scene, parse_dt
from .search import search
from .sources import SOURCES
from .sources.wayback import versions as wayback_versions

WEB = Path(__file__).parent / "web"

app = FastAPI(title="Oracle", version=__version__)
app.include_router(intel_router)


# --------------------------------------------------------------------------- helpers


def encode_spec(render: Render) -> str:
    raw = zlib.compress(json.dumps(render.to_dict(), separators=(",", ":")).encode(), 9)
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


@functools.lru_cache(maxsize=1024)
def decode_spec(spec: str) -> Render:
    try:
        raw = zlib.decompress(base64.urlsafe_b64decode(spec + "=" * (-len(spec) % 4)))
        render = Render.from_dict(json.loads(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"bad spec: {exc}") from None
    if render.kind == "xyz":
        raise HTTPException(400, "xyz layers are served by their provider")
    for h in render.hrefs:
        check_href(h)
    return render


def _dates(start: str | None, end: str | None, days: int | None) -> tuple[datetime, datetime]:
    t1 = parse_dt(end + "T23:59:59Z") if end else datetime.now(timezone.utc)
    if start:
        t0 = parse_dt(start + "T00:00:00Z")
    elif days:
        t0 = t1 - timedelta(days=days)
    else:
        t0 = datetime(2014, 1, 1, tzinfo=timezone.utc)
    return t0, t1


def _aoi(where: str | None, bbox: str | None, radius_km: float) -> AOI:
    try:
        if bbox:
            return parse_aoi(bbox)
        if where:
            return parse_aoi(where, radius_km)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    raise HTTPException(400, "give `where` (place or lat,lon) or `bbox`")


# --------------------------------------------------------------------------- routes


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB / "index.html").read_text()


@app.get("/api/sources")
def sources() -> list[dict]:
    return [s.info.__dict__ for s in SOURCES.values()]


@app.get("/api/geocode")
def api_geocode(q: str, radius_km: float = 3.0) -> dict:
    try:
        a = geocode(q, radius_km)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from None
    return {"name": a.name, "bbox": a.bbox, "center": a.center}


@app.get("/api/search")
def api_search(
    where: str | None = None,
    bbox: str | None = None,
    radius_km: float = 3.0,
    start: str | None = None,
    end: str | None = None,
    days: int | None = None,
    sources: str | None = None,
    max_cloud: float | None = None,
    sort: str = "best",
    limit: int = Query(30, le=200),
) -> dict:
    aoi = _aoi(where, bbox, radius_km)
    t0, t1 = _dates(start, end, days)
    keys = [s for s in (sources or "").split(",") if s] or None
    try:
        res = search(aoi, t0, t1, sources=keys, max_cloud=max_cloud, limit=limit, sort=sort)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from None
    out = []
    for s in res.scenes:
        d = s.to_dict()
        d["spec"] = encode_spec(s.render) if s.render.kind != "xyz" else None
        out.append(d)
    return {"aoi": {"bbox": aoi.bbox, "name": aoi.name}, "scenes": out, "errors": res.errors, "counts": res.counts}


@app.get("/api/tiles/{z}/{x}/{y}.png")
def tiles(z: int, x: int, y: int, spec: str) -> Response:
    decode_spec(spec)  # validate before doing any I/O
    return Response(_tile(spec, z, x, y), media_type="image/png", headers={"Cache-Control": "max-age=86400"})


@functools.lru_cache(maxsize=2048)
def _tile_cached(spec: str, z: int, x: int, y: int) -> bytes:
    return render_tile(decode_spec(spec), z, x, y)


def _tile(spec: str, z: int, x: int, y: int) -> bytes:
    if z < 3:
        return transparent_png()
    try:
        return _tile_cached(spec, z, x, y)
    except Exception:  # noqa: BLE001 - a bad tile should be blank, not a 500 storm
        return transparent_png()


class SceneBody(BaseModel):
    scene: dict
    bbox: list[float]
    k: float | None = None
    min_length: float = 25.0
    include_shore: bool = False
    format: str = "png"
    auto_contrast: bool = False


def _scene_and_aoi(body: SceneBody) -> tuple[Scene, AOI]:
    if len(body.bbox) != 4:
        raise HTTPException(400, "bbox must be [west, south, east, north]")
    scene = Scene.from_dict(body.scene)
    check_scene(scene)
    aoi = AOI(tuple(body.bbox))
    w, h = aoi.size_km()
    if w * h > 2500:
        raise HTTPException(400, "AOI too large for the web app (max ~50 x 50 km); use the CLI")
    return scene, aoi


@app.post("/api/ships")
def api_ships(body: SceneBody) -> JSONResponse:
    scene, aoi = _scene_and_aoi(body)
    try:
        res = detect_ships(scene, aoi, k=body.k, min_length=body.min_length, include_shore=body.include_shore)
    except (ValueError, NoData) as exc:
        raise HTTPException(400, str(exc)) from None
    return JSONResponse(res.geojson())


@app.post("/api/chip")
def api_chip(body: SceneBody) -> Response:
    scene, aoi = _scene_and_aoi(body)
    try:
        c = chip(scene, aoi, auto_contrast=body.auto_contrast)
    except NoData as exc:
        raise HTTPException(404, str(exc)) from None
    name = f"{scene.source}_{scene.date}_{scene.id[:40]}"
    if body.format == "tif":
        with tempfile.TemporaryDirectory() as d:
            p = c.save_geotiff(Path(d) / "chip.tif")
            data = p.read_bytes()
        return Response(data, media_type="image/tiff", headers={"Content-Disposition": f'attachment; filename="{name}.tif"'})
    buf = io.BytesIO()
    c.image(label=True).save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"Content-Disposition": f'attachment; filename="{name}.png"'})


@app.get("/api/wayback")
def api_wayback(lat: float, lon: float) -> list[dict]:
    return _wayback_cached(round(lat, 4), round(lon, 4))


@functools.lru_cache(maxsize=256)
def _wayback_cached(lat: float, lon: float) -> list[dict]:
    return wayback_versions(lat, lon)


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    return FileResponse(WEB / "favicon.svg", media_type="image/svg+xml")


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
