"""Change detection: what is physically different on the ground between two times?

Object detectors answer "what is here"; this answers "what changed", including things
no detector has a class for (new earthworks, a flooded field, a cleared treeline, a
burnt area, reclaimed land, new hard-standing).

Optical (Sentinel-2, 10 m, 6 bands)
  * Baseline: per-pixel median of up to N clear earlier images (cloud, shadow and snow
    masked with the scene classification layer). A multi-date baseline cancels most of
    the one-off noise (haze, a parked ship, a single bad pixel) that makes two-date
    differencing so noisy. ``baseline="anniversary"`` takes the baseline from the same
    season a year earlier, so crop cycles and leaf-fall don't read as change.
  * Indices: NDVI (vegetation), MNDWI (water), NBR (burn), brightness, SWIR.
  * Every difference is turned into a robust z-score against the whole AOI (median and
    MAD), so scene-wide shifts (season, illumination, atmosphere) cancel and only
    locally unusual change survives.
  * Classes, in priority order: new_water, water_loss, burn, vegetation_loss,
    new_bright_surface (needs SWIR to brighten too, which haze doesn't do) and
    surface_change (change vector magnitude over red/NIR/SWIR, the haze-robust bands).
    Sea pixels that are water in both images are ignored (ships have their own
    detector); ship-shaped "water loss" is dropped for the same reason.

Radar (Sentinel-1 RTC, 10 m, VV + VH, day/night, through cloud)
  * Speckle: 5x5 spatial multilook, plus a temporal mean over the baseline images.
  * Log-ratio in dB; change needs >= 3 dB and a robust z beyond the threshold.
    radar_increase (new structures, vehicles, containers, metal) and radar_decrease
    (removal, demolition, flooding: smooth water is radar-dark). Sea is masked with ESA
    WorldCover because wind changes sea backscatter far more than anything on land.
  * The baseline prefers images from the same orbit direction as the "after" image;
    mixed geometry is noted because it causes false change on slopes and buildings.

Output: connected regions (min area, morphological clean-up) with class, area, centroid,
outline, robust score, a heuristic confidence, and before/after values, plus before /
after / overlay images for evidence.
"""

from __future__ import annotations

import hashlib
import io
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from scipy import ndimage

from .geo import AOI
from .http import log
from .imagery import Grid, NoData, read_href, resolve_href
from .models import Scene
from .search import search

KINDS = {
    # kind: (meaning, RGB)
    "new_water": ("flooding, new reservoir or dredged basin", (59, 130, 246)),
    "water_loss": ("land reclamation, island building, drying, new pier", (234, 179, 8)),
    "burn": ("fire scar", (220, 38, 38)),
    "vegetation_loss": ("clearing, earthworks, construction start or harvest", (249, 115, 22)),
    "new_bright_surface": ("new construction, paving, pads, containers, tents", (244, 244, 245)),
    "surface_change": ("other change in surface material", (168, 85, 247)),
    "radar_increase": ("new structures, vehicles, containers or metal", (244, 244, 245)),
    "radar_decrease": ("removal, demolition or flooding", (59, 130, 246)),
}
KIND_ID = {k: i + 1 for i, k in enumerate(KINDS)}
S2_BANDS = ("blue", "green", "red", "nir", "swir16", "swir22")
SCL_BAD = (0, 1, 3, 8, 9, 10, 11)  # no data, saturated, cloud shadow, cloud med/high, cirrus, snow
MAX_PIXELS = 2048
MAX_REGIONS = 300
_RECENT: dict[str, ChangeResult] = {}  # id -> result, for evidence images in the web app / agent


def remember(res: ChangeResult, keep: int = 8) -> ChangeResult:
    _RECENT[res.id] = res
    while len(_RECENT) > keep:
        _RECENT.pop(next(iter(_RECENT)))
    return res


def recall(change_id: str) -> ChangeResult | None:
    return _RECENT.get(change_id)


@dataclass
class ChangeRegion:
    kind: str
    lat: float
    lon: float
    area_m2: float
    bbox: tuple[float, float, float, float]
    score: float
    confidence: float
    outline: list[list[float]]
    values: dict[str, list[float]] = field(default_factory=dict)  # metric -> [before, after]
    id: str = ""

    def feature(self) -> dict:
        ring = self.outline + self.outline[:1] if len(self.outline) >= 3 else None
        geom = {"type": "Polygon", "coordinates": [ring]} if ring else {"type": "Point", "coordinates": [self.lon, self.lat]}
        return {
            "type": "Feature",
            "id": self.id,
            "geometry": geom,
            "properties": {
                "kind": self.kind,
                "meaning": KINDS[self.kind][0],
                "lat": round(self.lat, 6),
                "lon": round(self.lon, 6),
                "area_m2": round(self.area_m2),
                "score": round(self.score, 2),
                "confidence": round(self.confidence, 2),
                "bbox": [round(v, 6) for v in self.bbox],
                "values": self.values,
            },
        }


@dataclass
class ChangeResult:
    aoi: AOI
    after: Scene
    before: list[Scene]
    grid: Grid
    regions: list[ChangeRegion]
    kind_map: np.ndarray
    valid_fraction: float
    method: str
    notes: list[str] = field(default_factory=list)
    before_rgb: np.ndarray | None = None
    after_rgb: np.ndarray | None = None
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            key = f"{self.after.id}|{','.join(s.id for s in self.before)}|{self.aoi.bbox}"
            self.id = hashlib.sha1(key.encode()).hexdigest()[:12]

    @property
    def corners(self) -> list[list[float]]:
        """Image corners (TL, TR, BR, BL) in lon/lat, for map overlays."""
        w, s, e, n = self.grid.bounds
        tr = Transformer.from_crs(self.grid.crs, 4326, always_xy=True)
        return [list(tr.transform(x, y)) for x, y in ((w, n), (e, n), (e, s), (w, s))]

    def summary(self) -> dict:
        by_kind: dict[str, dict] = {}
        for r in self.regions:
            k = by_kind.setdefault(r.kind, {"regions": 0, "area_m2": 0.0})
            k["regions"] += 1
            k["area_m2"] += r.area_m2
        return {
            "id": self.id,
            "method": self.method,
            "after": {"id": self.after.id, "time": self.after.datetime.isoformat(), "source": self.after.source},
            "before": [{"id": s.id, "time": s.datetime.isoformat()} for s in self.before],
            "valid_fraction": round(self.valid_fraction, 3),
            "by_kind": {k: {"regions": v["regions"], "area_m2": round(v["area_m2"])} for k, v in by_kind.items()},
            "notes": self.notes,
            "corners": self.corners,
        }

    def geojson(self) -> dict:
        return {"type": "FeatureCollection", "properties": self.summary(), "features": [r.feature() for r in self.regions]}

    def overlay(self) -> Image.Image:
        """Transparent RGBA: change pixels coloured by kind."""
        h, w = self.kind_map.shape
        rgba = np.zeros((h, w, 4), np.uint8)
        for k, i in KIND_ID.items():
            m = self.kind_map == i
            if m.any():
                rgba[m, :3] = KINDS[k][1]
                rgba[m, 3] = 200
        return Image.fromarray(rgba, "RGBA")

    def image(self, which: str) -> Image.Image:
        if which == "overlay":
            return self.overlay()
        rgb = self.before_rgb if which == "before" else self.after_rgb
        if rgb is None:
            raise NoData(f"no {which} image")
        img = Image.fromarray(rgb, "RGB")
        if which == "change":  # after image with change outlines
            ov = self.overlay()
            edge = np.array(ov)[..., 3] > 0
            edge = edge & ~ndimage.binary_erosion(edge, iterations=1)
            a = np.array(img)
            a[edge] = np.array(ov)[..., :3][edge]
            img = Image.fromarray(a, "RGB")
        return img

    def png(self, which: str, crop: tuple[float, float, float, float] | None = None, pad_m: float = 150.0) -> bytes:
        img = self.image(which)
        if crop:
            img = img.crop(self._pixel_box(crop, pad_m))
            if max(img.size) < 256:
                f = 256 / max(img.size)
                img = img.resize((max(1, round(img.width * f)), max(1, round(img.height * f))), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _pixel_box(self, bbox: tuple, pad_m: float) -> tuple[int, int, int, int]:
        tr = Transformer.from_crs(4326, self.grid.crs, always_xy=True)
        w, s, e, n = bbox
        xs, ys = zip(*(tr.transform(x, y) for x, y in ((w, s), (e, n), (w, n), (e, s))), strict=True)
        inv = ~self.grid.transform
        c0, r0 = inv @ (min(xs) - pad_m, max(ys) + pad_m)
        c1, r1 = inv @ (max(xs) + pad_m, min(ys) - pad_m)
        c0, r0 = max(0, int(c0)), max(0, int(r0))
        c1, r1 = min(self.grid.width, max(c0 + 1, int(math.ceil(c1)))), min(self.grid.height, max(r0 + 1, int(math.ceil(r1))))
        return c0, r0, c1, r1

    def save(self, out_dir, stem: str | None = None) -> dict:
        from pathlib import Path

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = stem or f"change_{self.after.date}_{self.after.source}"
        paths = {}
        for which in ("before", "after", "change"):
            p = out / f"{stem}_{which}.png"
            p.write_bytes(self.png(which))
            paths[which] = p
        import json

        p = out / f"{stem}.geojson"
        p.write_text(json.dumps(self.geojson(), indent=1))
        paths["geojson"] = p
        return paths


# --------------------------------------------------------------------------- scene selection


def pick_scenes(
    aoi: AOI,
    source: str = "sentinel-2",
    after: datetime | None = None,
    before: datetime | None = None,
    n_before: int = 3,
    lookback_days: int = 90,
    baseline: str = "recent",
) -> tuple[list[Scene], list[Scene]]:
    """-> (after candidates newest first, baseline candidates newest first)."""
    if source not in ("sentinel-2", "sentinel-1"):
        raise ValueError("change detection supports sentinel-2 and sentinel-1")
    now = datetime.now(timezone.utc)
    after_end = after + timedelta(days=1) if after else now
    after_start = after_end - timedelta(days=30 if after is None else 3)
    cloud = 60.0 if source == "sentinel-2" else None
    res = search(aoi, after_start, after_end, sources=[source], max_cloud=cloud, limit=20, sort="date", min_coverage=0.7)
    afters = [s for s in res.scenes if s.datetime <= after_end]
    if not afters:
        raise NoData(f"no {source} image of the area around {after_end:%Y-%m-%d}")
    ref = afters[0].datetime
    if baseline == "anniversary":
        b_end = (before or ref - timedelta(days=365)) + timedelta(days=30)
        b_start = b_end - timedelta(days=60)
    else:
        b_end = before + timedelta(days=1) if before else ref - timedelta(days=4)
        b_start = b_end - timedelta(days=lookback_days)
    res = search(aoi, b_start, b_end, sources=[source], max_cloud=cloud, limit=40, sort="date", min_coverage=0.7)
    befores = [s for s in res.scenes if s.datetime < ref - timedelta(hours=12)]
    if source == "sentinel-1":
        same = [s for s in befores if s.extra.get("orbit_state") == afters[0].extra.get("orbit_state")]
        befores = same + [s for s in befores if s not in same]
    if not befores:
        raise NoData(f"no earlier {source} image for a baseline ({b_start:%Y-%m-%d}..{b_end:%Y-%m-%d})")
    return afters, befores


# --------------------------------------------------------------------------- reading


def _read_band(scene: Scene, name: str, grid: Grid, resampling=Resampling.bilinear) -> np.ndarray:
    out = np.full((grid.height, grid.width), np.nan, np.float32)
    for bands in [scene.extra.get("bands") or {}, *(scene.extra.get("band_mosaic") or [])]:
        if name not in bands:
            continue
        a = read_href(bands[name], grid, [1], resampling)[0]
        hole = np.isnan(out)
        out[hole] = a[hole]
        if not np.isnan(out).any():
            break
    return out


def _read_s2(scene: Scene, grid: Grid) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Reflectance bands and a clear-sky mask on ``grid``."""
    scl = _read_band(scene, "scl", grid, Resampling.nearest)
    bad = np.isin(np.nan_to_num(scl, nan=0).astype(np.uint8), SCL_BAD)
    bad = ndimage.binary_dilation(bad, iterations=3)  # cloud edges and thin shadow rims
    scale = scene.extra.get("reflectance_scale", 0.0001)
    offset = scene.extra.get("reflectance_offset", 0.0)
    bands = {b: _read_band(scene, b, grid) * scale + offset for b in S2_BANDS}
    valid = ~bad & np.isfinite(scl)
    for a in bands.values():
        valid &= np.isfinite(a)
    return bands, valid


def _read_s1(scene: Scene, grid: Grid) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Multilooked linear backscatter (VV, VH if present) and validity on ``grid``."""
    out = {}
    hrefs = {"vv": scene.render.hrefs[0], "vh": scene.extra.get("cross_pol")}
    valid = None
    for pol, href in hrefs.items():
        if not href:
            continue
        a = read_href(resolve_href(href, scene.render), grid, [1])[0]
        ok = np.isfinite(a) & (a > 0)
        s = ndimage.uniform_filter(np.where(ok, a, 0.0), 5)
        n = ndimage.uniform_filter(ok.astype(np.float32), 5)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[pol] = np.where(n > 0.5, s / n, np.nan).astype(np.float32)
        valid = ok if valid is None else valid & ok
    if valid is None:
        raise NoData("no radar band")
    return out, valid


# --------------------------------------------------------------------------- core


def _rz(d: np.ndarray, m: np.ndarray, floor: float) -> np.ndarray:
    """Robust z-score of ``d`` against its own distribution over mask ``m``."""
    v = d[m]
    if v.size < 100:
        return np.zeros_like(d)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    return (d - med) / max(mad, floor)


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a - b) / (a + b)


def _rgb_s2(b: dict[str, np.ndarray]) -> np.ndarray:
    rgb = np.stack([b["red"], b["green"], b["blue"]], -1)
    rgb = np.clip(np.nan_to_num(rgb) / 0.3, 0, 1) ** (1 / 1.6)
    return (rgb * 255).astype(np.uint8)


def _rgb_db(lin: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        db = 10 * np.log10(lin)
    g = (np.clip((np.nan_to_num(db, nan=-30) + 25) / 30, 0, 1) * 255).astype(np.uint8)
    return np.stack([g, g, g], -1)


def detect_change(
    aoi: AOI,
    source: str = "sentinel-2",
    after: datetime | None = None,
    before: datetime | None = None,
    n_before: int = 3,
    baseline: str = "recent",
    k: float = 3.0,
    min_area_m2: float = 1500.0,
    scenes: tuple[Scene, list[Scene]] | None = None,
) -> ChangeResult:
    """Find what changed in ``aoi`` (see module docs). ``scenes`` skips the catalog search."""
    if scenes:
        afters, befores = [scenes[0]], list(scenes[1])
    else:
        afters, befores = pick_scenes(aoi, source, after, before, n_before, baseline=baseline)
    source = afters[0].source
    grid = Grid.for_aoi(aoi, 10.0, max_pixels=MAX_PIXELS)
    notes: list[str] = []
    if grid.res > 10.5:
        notes.append(f"large area: analysed at {grid.res:.0f} m instead of 10 m")
    reader = _read_s2 if source == "sentinel-2" else _read_s1

    # After: newest candidate that is mostly clear over the AOI.
    a_img = a_valid = None
    for cand in afters[:3]:
        bands, valid = reader(cand, grid)
        frac = float(valid.mean())
        if frac >= 0.3:
            after_scene, a_img, a_valid = cand, bands, valid
            break
        notes.append(f"skipped {cand.date}: only {frac:.0%} clear")
    if a_img is None:
        raise NoData("no clear recent image over the area")

    # Baseline: up to n_before clear images, read in parallel, composited per pixel.
    used: list[Scene] = []
    stack: list[tuple[dict, np.ndarray]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        cands = befores[: n_before * 2]
        for cand, (bands, valid) in zip(cands, pool.map(lambda s: reader(s, grid), cands), strict=True):
            if len(used) >= n_before:
                break
            if valid.mean() < 0.3:
                continue
            used.append(cand)
            stack.append((bands, valid))
    if not stack:
        raise NoData("no clear baseline image over the area")
    ref = {}
    for name in a_img:
        layers = np.stack([np.where(v, b[name], np.nan) for b, v in stack])
        if source == "sentinel-2":
            ref[name] = _nanmedian(layers)
        else:
            ref[name] = np.nanmean(layers, axis=0)  # temporal multilook (linear power)
    b_valid = np.isfinite(ref[next(iter(ref))])
    both = a_valid & b_valid
    valid_fraction = float(both.mean())
    if valid_fraction < 0.1:
        raise NoData(f"only {valid_fraction:.0%} of the area is clear in both the image and the baseline")

    if source == "sentinel-2":
        masks, metrics = _classify_s2(ref, a_img, both, k)
        method = f"sentinel-2 multi-index vs {len(used)}-image median baseline"
    else:
        if len({s.extra.get("orbit_state") for s in [after_scene, *used]}) > 1:
            notes.append("baseline mixes ascending and descending passes: expect false change on slopes/buildings")
        masks, metrics = _classify_s1(ref, a_img, both, grid, k, notes)
        method = f"sentinel-1 log-ratio vs {len(used)}-image temporal mean"

    regions, kind_map = _regions(masks, metrics, grid, k, min_area_m2, len(used))
    if source == "sentinel-2":
        shown = {c: _nanmedian(np.stack([b[c] for b, _ in stack])) for c in ("red", "green", "blue")}
        b_rgb, a_rgb = _rgb_s2(shown), _rgb_s2(a_img)
    else:
        b_rgb, a_rgb = _rgb_db(ref["vv"]), _rgb_db(a_img["vv"])
    log(f"change {source} {after_scene.date} vs {len(used)} baseline: {len(regions)} regions ({valid_fraction:.0%} comparable)")
    return ChangeResult(aoi, after_scene, used, grid, regions, kind_map, valid_fraction, method, notes, b_rgb, a_rgb)


def _nanmedian(layers: np.ndarray) -> np.ndarray:
    if layers.shape[0] == 1:
        return layers[0]
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(layers, axis=0)


def _classify_s2(b: dict, a: dict, both: np.ndarray, k: float):
    ndvi_b, ndvi_a = _ratio(b["nir"], b["red"]), _ratio(a["nir"], a["red"])
    mndwi_b, mndwi_a = _ratio(b["green"], b["swir16"]), _ratio(a["green"], a["swir16"])
    nbr_b, nbr_a = _ratio(b["nir"], b["swir22"]), _ratio(a["nir"], a["swir22"])
    br_b = (b["blue"] + b["green"] + b["red"]) / 3
    br_a = (a["blue"] + a["green"] + a["red"]) / 3
    water_b, water_a = mndwi_b > 0.1, mndwi_a > 0.1
    land_b, land_a = mndwi_b < -0.05, mndwi_a < -0.05
    land_any = both & ~(water_b & water_a)

    z_mndwi = _rz(mndwi_a - mndwi_b, both, 0.03)
    z_ndvi = _rz(ndvi_a - ndvi_b, land_any, 0.03)
    z_nbr = _rz(nbr_a - nbr_b, land_any, 0.03)
    z_br = _rz(br_a - br_b, land_any, 0.01)
    z_sw = _rz(a["swir16"] - b["swir16"], land_any, 0.01)
    # Change vector over the haze-robust bands.
    zs = [_rz(a[x] - b[x], land_any, 0.01) for x in ("red", "nir", "swir16", "swir22")]
    cva = np.sqrt(sum(z * z for z in zs) / len(zs))

    m = {}
    # Water/land flips also need SWIR to agree: water is SWIR-dark (< ~0.05 even when turbid),
    # sand, rock and concrete are SWIR-bright. That keeps glint, plumes and algae out.
    m["new_water"] = both & land_b & (mndwi_a > 0.1) & (z_mndwi > k) & (b["swir16"] > 0.12) & (a["swir16"] < 0.08)
    m["water_loss"] = both & water_b & land_a & (z_mndwi < -k) & (a["swir16"] > 0.12) & (b["swir16"] < 0.08)
    taken = m["new_water"] | m["water_loss"]
    # Burn vs clearing: both drop NBR, but char is dark while bare soil brightens in the visible.
    m["burn"] = (
        land_any
        & ~taken
        & (nbr_b - nbr_a > 0.27)
        & (ndvi_b > 0.25)
        & (z_nbr < -k)
        & ~water_a
        & (a["red"] < 0.12)
        & (br_a <= br_b + 0.02)
    )
    taken |= m["burn"]
    m["vegetation_loss"] = land_any & ~taken & (ndvi_b > 0.4) & (ndvi_a - ndvi_b < -0.25) & (z_ndvi < -k) & ~water_a
    taken |= m["vegetation_loss"]
    m["new_bright_surface"] = land_any & ~taken & (br_a - br_b > 0.04) & (z_br > k) & (z_sw > k) & (ndvi_a < 0.3) & ~water_a
    taken |= m["new_bright_surface"]
    m["surface_change"] = land_any & ~taken & (cva > k + 1) & ~water_a & ~water_b
    metrics = {
        "new_water": (z_mndwi, {"mndwi": (mndwi_b, mndwi_a), "swir16": (b["swir16"], a["swir16"])}),
        "water_loss": (-z_mndwi, {"mndwi": (mndwi_b, mndwi_a), "swir16": (b["swir16"], a["swir16"])}),
        "burn": (-z_nbr, {"nbr": (nbr_b, nbr_a), "ndvi": (ndvi_b, ndvi_a)}),
        "vegetation_loss": (-z_ndvi, {"ndvi": (ndvi_b, ndvi_a)}),
        "new_bright_surface": (z_br, {"brightness": (br_b, br_a), "swir16": (b["swir16"], a["swir16"])}),
        "surface_change": (cva, {"brightness": (br_b, br_a), "ndvi": (ndvi_b, ndvi_a)}),
    }
    return m, metrics


def _classify_s1(b: dict, a: dict, both: np.ndarray, grid: Grid, k: float, notes: list[str]):
    from .landmask import water_mask

    water = water_mask(grid)
    if water is None:
        notes.append("WorldCover unreachable: sea not masked, wind-driven sea changes may appear")
        land = both
    else:
        land = both & ~ndimage.binary_dilation(water, iterations=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = {p: 10 * np.log10(a[p] / b[p]) for p in a if p in b}
    zs = {p: _rz(np.nan_to_num(v), land, 0.5) for p, v in r.items()}
    up = np.zeros_like(both)
    down = np.zeros_like(both)
    for p, v in r.items():
        up |= (np.nan_to_num(v) >= 3) & (zs[p] > k)
        down |= (np.nan_to_num(v) <= -3) & (zs[p] < -k)
    m = {"radar_increase": land & up & ~down, "radar_decrease": land & down & ~up}
    vv_b = 10 * np.log10(np.clip(b["vv"], 1e-6, None))
    vv_a = 10 * np.log10(np.clip(a["vv"], 1e-6, None))
    score = np.max(np.stack([np.abs(z) for z in zs.values()]), axis=0)
    metrics = {
        "radar_increase": (score, {"vv_db": (vv_b, vv_a)}),
        "radar_decrease": (score, {"vv_db": (vv_b, vv_a)}),
    }
    return m, metrics


def _regions(masks: dict, metrics: dict, grid: Grid, k: float, min_area_m2: float, n_base: int):
    px_area = grid.res * grid.res
    min_px = max(2, int(round(min_area_m2 / px_area)))
    tr = Transformer.from_crs(grid.crs, 4326, always_xy=True)
    kind_map = np.zeros((grid.height, grid.width), np.uint8)
    regions: list[ChangeRegion] = []
    st = ndimage.generate_binary_structure(2, 1)
    for kind, mask in masks.items():
        if not mask.any():
            continue
        if grid.res <= 12:
            mask = ndimage.binary_opening(mask, st)  # drop isolated pixels
        lab, n = ndimage.label(mask, st)
        if not n:
            continue
        sizes = ndimage.sum_labels(mask, lab, np.arange(1, n + 1))
        objs = ndimage.find_objects(lab)
        score_map, vals = metrics[kind]
        for i in np.nonzero(sizes >= min_px)[0] + 1:
            sl = objs[i - 1]
            rr, cc = np.nonzero(lab[sl] == i)
            rr, cc = rr + sl[0].start, cc + sl[1].start
            area = len(rr) * px_area
            if kind in ("water_loss", "new_water") and _ship_shaped(rr, cc, grid.res, area):
                continue  # a moored or passing ship, handled by the vessel detector
            kind_map[rr, cc] = KIND_ID[kind]
            score = float(np.nanmean(score_map[rr, cc]))
            xs, ys = grid.transform @ (cc + 0.5, rr + 0.5)
            lon, lat = tr.transform(float(np.mean(xs)), float(np.mean(ys)))
            outline = _hull(np.asarray(xs), np.asarray(ys), tr)
            lons = [p[0] for p in outline] or [lon]
            lats = [p[1] for p in outline] or [lat]
            conf = 0.3 + 0.08 * min(score - k, 5) + 0.08 * math.log10(max(area / min_area_m2, 1)) + 0.12 * (n_base >= 3)
            regions.append(
                ChangeRegion(
                    kind=kind,
                    lat=lat,
                    lon=lon,
                    area_m2=area,
                    bbox=(min(lons), min(lats), max(lons), max(lats)),
                    score=score,
                    confidence=float(np.clip(conf, 0.05, 0.95)),
                    outline=outline,
                    values={
                        name: [round(float(np.nanmedian(bv[rr, cc])), 3), round(float(np.nanmedian(av[rr, cc])), 3)]
                        for name, (bv, av) in vals.items()
                    },
                    id=hashlib.sha1(f"{kind}|{lat:.5f}|{lon:.5f}|{area:.0f}".encode()).hexdigest()[:12],
                )
            )
    regions.sort(key=lambda r: -(r.area_m2 * r.confidence))
    return regions[:MAX_REGIONS], kind_map


def _ship_shaped(rr: np.ndarray, cc: np.ndarray, res: float, area: float) -> bool:
    if area > 30_000 or len(rr) < 3:
        return False
    cov = np.cov(np.stack([rr, cc]).astype(float))
    ev = np.sort(np.linalg.eigvalsh(cov))
    if ev[0] <= 0:
        return True
    return math.sqrt(ev[1] / ev[0]) >= 3.0


def _hull(xs: np.ndarray, ys: np.ndarray, tr: Transformer) -> list[list[float]]:
    if len(xs) < 3:
        return []
    from scipy.spatial import ConvexHull, QhullError

    pts = np.stack([xs, ys], 1)
    try:
        h = ConvexHull(pts)
    except QhullError:
        return []
    return [[round(v, 6) for v in tr.transform(*pts[i])] for i in h.vertices]


def to_events(res: ChangeResult, site: str, min_confidence: float = 0.35, top: int = 25) -> list[dict]:
    """Significant change regions as site events (kept across tracker refreshes)."""
    out = []
    for r in res.regions:
        if r.confidence < min_confidence:
            continue
        sev = 0.25 + 0.35 * r.confidence + 0.1 * min(math.log10(max(r.area_m2, 1)) - 3, 2) / 2
        out.append(
            {
                "id": hashlib.sha1(f"change|{site}|{res.after.id}|{r.id}".encode()).hexdigest()[:16],
                "site": site,
                "time": res.after.datetime.isoformat(),
                "kind": f"change_{r.kind}",
                "severity": round(min(sev, 0.8), 2),
                "title": f"{r.kind.replace('_', ' ').capitalize()} over {r.area_m2 / 1e4:.1f} ha at {site}",
                "detail": {
                    "lat": r.lat,
                    "lon": r.lon,
                    "area_m2": round(r.area_m2),
                    "confidence": round(r.confidence, 2),
                    "values": r.values,
                    "meaning": KINDS[r.kind][0],
                    "after_scene": res.after.id,
                    "baseline": [s.id for s in res.before],
                    "bbox": r.bbox,
                    "change_id": res.id,
                },
            }
        )
        if len(out) >= top:
            break
    return out
