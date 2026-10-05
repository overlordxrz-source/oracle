"""Object detection on sub-metre optical imagery with YOLO.

Default model: Ultralytics YOLO11-OBB trained on DOTA v1 (aerial/satellite imagery,
oriented boxes): planes, ships, small and large vehicles, helicopters, storage tanks,
bridges, harbours and sports facilities. It works from about 0.2 to 1 m per pixel:
aircraft and ships remain detectable at 1 m, cars need roughly 0.5 m or better.

Open-vocabulary option: pass ``prompts=["fighter jet", "tank", ...]`` to use
YOLO-World. That's zero-shot on satellite views, so treat it as a lead generator,
not a classifier.

Inference is tiled (1024 px tiles with overlap) over the AOI at the chosen ground
sample distance. Each box is georeferenced, and duplicates from tile overlaps are
resolved by keeping only detections whose centre lies in a tile's core.

Optional dependency: ``pip install "oracle-osint[ai]"`` (ultralytics + torch).
Ultralytics code and weights are AGPL-3.0.
"""

from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from pyproj import Transformer

from .config import CACHE_DIR
from .geo import AOI
from .http import log
from .imagery import Grid, NoData, colorize, read_render
from .models import Scene
from .observations import AIRCRAFT, HELICOPTER, LARGE_VEHICLE, OTHER, STORAGE_TANK, VEHICLE, VESSEL, Observation

MODEL_DIR = CACHE_DIR / "models"
DEFAULT_MODEL = "yolo11l-obb.pt"  # s/m are faster; l was clearly best on cars (256 vs 161 for s)
WORLD_MODEL = "yolov8s-worldv2.pt"

DOTA_TO_CLASS = {
    "plane": AIRCRAFT,
    "helicopter": HELICOPTER,
    "ship": VESSEL,
    "small vehicle": VEHICLE,
    "large vehicle": LARGE_VEHICLE,
    "storage tank": STORAGE_TANK,
}


class DetectorUnavailable(RuntimeError):
    pass


@functools.lru_cache(maxsize=4)
def load_model(name: str = DEFAULT_MODEL):
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise DetectorUnavailable('object detection needs: pip install "oracle-osint[ai]"') from exc
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(name) if Path(name).exists() else MODEL_DIR / name
    return YOLO(str(path))  # downloads official weights into MODEL_DIR on first use


# YOLO is scale-sensitive: it finds objects at the pixel sizes it was trained on (DOTA).
# Measured on 0.3 m NAIP: aircraft recall peaks when imagery is resampled to 0.6-0.9 m/px
# (Davis-Monthan boneyard: 65-68 aircraft vs 57 at native 0.3 m), while cars only show
# up when upsampled to ~0.12-0.2 m/px (Tucson parking area: 256 cars at 0.13 m and 231
# at 0.2 m with YOLO11-L, 0 at 0.3 m). So the default runs two passes, each trusted for
# its own classes. Known weak spot: cars packed bumper-to-bumper (import lots) on hazy
# imagery are mostly missed by every model size.
VEHICLE_GSD = 0.15
FINE_CLASSES = {VEHICLE, LARGE_VEHICLE}
COARSE_CLASSES = {AIRCRAFT, VESSEL, STORAGE_TANK, HELICOPTER, OTHER}


@contextlib.contextmanager
def _weights_in_cache():
    """YOLO-World fetches a CLIP text model into Ultralytics' WEIGHTS_DIR (default:
    ./weights in the current directory), which it binds at import time. Point that
    binding at Oracle's cache just for this call; the user's settings file is untouched."""
    try:
        import ultralytics.nn.text_model as tm
    except ImportError:  # pragma: no cover - layout differs in other ultralytics versions
        yield
        return
    old = getattr(tm, "WEIGHTS_DIR", None)
    tm.WEIGHTS_DIR = MODEL_DIR
    try:
        yield
    finally:
        if old is not None:
            tm.WEIGHTS_DIR = old


def default_scales(scene_gsd: float, vehicles: bool = True) -> list[tuple[float, set[str] | None]]:
    if scene_gsd <= 0.7:
        coarse = [(max(0.75, scene_gsd), COARSE_CLASSES)]
        return ([(VEHICLE_GSD, FINE_CLASSES)] if vehicles else []) + coarse
    return [(scene_gsd, None)]


def detect_objects(
    scene: Scene,
    aoi: AOI,
    *,
    model: str = DEFAULT_MODEL,
    gsd: float | None = None,
    conf: float = 0.3,
    classes: Iterable[str] | None = None,
    prompts: list[str] | None = None,
    tile: int = 1024,
    overlap: int = 160,
    max_pixels: int = 20000,
    batch: int = 4,
    vehicles: bool = True,
) -> list[Observation]:
    """Run YOLO over ``aoi`` in ``scene``. ``classes`` filters canonical classes.

    ``vehicles=False`` skips the expensive car pass (upsampled to 0.13 m/px)."""
    if scene.sensor != "optical" or scene.render.kind == "xyz":
        raise ValueError(f"object detection needs downloadable optical imagery, not {scene.source}")
    if scene.gsd > 2.0:
        raise ValueError(f"{scene.source} is {scene.gsd:g} m/px; YOLO needs ~1 m or better (try maxar, naip)")
    if classes and not (set(classes) & FINE_CLASSES):
        vehicles = False
    scales = [(gsd, None)] if (gsd or prompts) else default_scales(scene.gsd, vehicles)
    if prompts and not gsd:
        scales = [(max(scene.gsd, 0.3), None)]
    obs: list[Observation] = []
    for res, keep in scales:
        grid = Grid.for_aoi(aoi, res, max_pixels=max_pixels)
        if keep is FINE_CLASSES and grid.res > 0.2:
            log(f"  AOI too large for the vehicle pass ({grid.res:.2f} m/px > 0.2); shrink it to count cars")
            continue
        rgba = colorize(scene.render, read_render(scene.render, grid))
        if not rgba[..., 3].any():
            raise NoData("no imagery inside the AOI")
        found = detect_array(
            rgba,
            grid,
            scene_id=scene.id,
            source=scene.source,
            time=scene.datetime,
            model=WORLD_MODEL if prompts else model,
            prompts=prompts,
            conf=conf,
            tile=tile,
            overlap=overlap,
            batch=batch,
        )
        obs += [o for o in found if keep is None or o.cls in keep]
    obs = dedupe(obs)
    if classes:
        wanted = set(classes)
        obs = [o for o in obs if o.cls in wanted or o.attrs.get("label") in wanted]
    return obs


def detect_array(
    rgba: np.ndarray,
    grid: Grid,
    *,
    scene_id: str,
    source: str,
    time,
    model: str = DEFAULT_MODEL,
    prompts: list[str] | None = None,
    conf: float = 0.3,
    tile: int = 1024,
    overlap: int = 160,
    batch: int = 4,
) -> list[Observation]:
    """Tile an RGBA array on ``grid``, run YOLO and georeference the boxes."""
    yolo = load_model(model)
    if prompts:
        with _weights_in_cache():
            yolo.set_classes(list(prompts))
    names = yolo.names
    h, w = rgba.shape[:2]
    step = tile - overlap
    windows = [(r, c) for r in range(0, max(h - overlap, 1), step) for c in range(0, max(w - overlap, 1), step)]
    to_wgs = Transformer.from_crs(grid.crs, 4326, always_xy=True)
    det_name = Path(model).stem + (":" + ",".join(prompts) if prompts else "")
    out: list[Observation] = []
    todo = []
    for r0, c0 in windows:
        alpha = rgba[r0 : r0 + tile, c0 : c0 + tile, 3]
        if alpha.mean() < 0.05 * 255:
            continue
        img = np.zeros((tile, tile, 3), np.uint8)  # pad edge tiles to a full tile
        part = rgba[r0 : r0 + tile, c0 : c0 + tile, :3]
        img[: part.shape[0], : part.shape[1]] = part[..., ::-1]  # RGB -> BGR for ultralytics
        todo.append((r0, c0, img))
    log(f"  yolo: {len(todo)} tile(s) of {tile}px at {grid.res:.2f} m/px")
    half = overlap // 2
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        results = yolo.predict([t[2] for t in chunk], imgsz=tile, conf=conf, verbose=False)
        for (r0, c0, _), res in zip(chunk, results, strict=True):
            # keep boxes whose centre is in this tile's core, so overlaps don't double count
            lo_r, lo_c = (half if r0 > 0 else 0), (half if c0 > 0 else 0)
            hi_r = tile - half if r0 + tile < h else tile
            hi_c = tile - half if c0 + tile < w else tile
            for det in _boxes(res, names):
                cx, cy = det["cx"], det["cy"]
                if not (lo_c <= cx < hi_c and lo_r <= cy < hi_r):
                    continue
                out.append(_to_obs(det, r0, c0, grid, to_wgs, scene_id, source, time, det_name))
    return out


def dedupe(obs: list[Observation]) -> list[Observation]:
    """Greedy same-class suppression by centre distance (multi-scale / overlap duplicates)."""
    kept: list[Observation] = []
    by_cls: dict[str, list[Observation]] = {}
    for o in sorted(obs, key=lambda o: -o.confidence):
        near = by_cls.setdefault(o.cls, [])
        k = math.cos(math.radians(o.lat))
        radius = max(0.4 * (o.length_m or 0), 3.0)
        if any(math.hypot((o.lon - p.lon) * 111_320 * k, (o.lat - p.lat) * 110_574) < radius for p in near):
            continue
        near.append(o)
        kept.append(o)
    return kept


def _boxes(res, names) -> list[dict]:
    """Normalise ultralytics OBB / axis-aligned results to dicts in tile pixels."""
    dets = []
    if getattr(res, "obb", None) is not None and len(res.obb):
        xywhr = res.obb.xywhr.cpu().numpy()
        corners = res.obb.xyxyxyxy.cpu().numpy()
        for (cx, cy, bw, bh, rot), pts, c, k in zip(
            xywhr, corners, res.obb.conf.cpu().numpy(), res.obb.cls.cpu().numpy(), strict=True
        ):
            dets.append({"cx": cx, "cy": cy, "w": bw, "h": bh, "rot": rot, "pts": pts, "conf": float(c), "label": names[int(k)]})
    elif getattr(res, "boxes", None) is not None and len(res.boxes):
        for (x0, y0, x1, y1), c, k in zip(
            res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy(), strict=True
        ):
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            dets.append(
                {
                    "cx": (x0 + x1) / 2,
                    "cy": (y0 + y1) / 2,
                    "w": x1 - x0,
                    "h": y1 - y0,
                    "rot": None,
                    "pts": pts,
                    "conf": float(c),
                    "label": names[int(k)],
                }
            )
    return dets


def _to_obs(det, r0, c0, grid: Grid, to_wgs, scene_id, source, time, det_name) -> Observation:
    res = grid.res
    x, y = grid.transform @ (c0 + float(det["cx"]), r0 + float(det["cy"]))
    lon, lat = to_wgs.transform(x, y)
    poly = []
    for px, py in det["pts"]:
        mx, my = grid.transform @ (c0 + float(px), r0 + float(py))
        plon, plat = to_wgs.transform(mx, my)
        poly.append([round(plon, 7), round(plat, 7)])
    bw, bh = float(det["w"]) * res, float(det["h"]) * res
    axis = None
    if det["rot"] is not None:
        rot = float(det["rot"]) + (0.0 if bw >= bh else math.pi / 2)  # angle of the long side, image coords
        axis = round((math.degrees(math.atan2(math.cos(rot), -math.sin(rot))) + 180) % 180, 1)
    label = det["label"]
    return Observation(
        cls=DOTA_TO_CLASS.get(label, label if det["rot"] is None else OTHER),
        lat=round(lat, 7),
        lon=round(lon, 7),
        time=time,
        scene_id=scene_id,
        source=source,
        detector=det_name,
        confidence=round(det["conf"], 3),
        length_m=round(max(bw, bh), 1),
        width_m=round(min(bw, bh), 1),
        axis_deg=axis,
        polygon=poly,
        attrs={"label": label, "gsd": round(res, 3)},
    )
