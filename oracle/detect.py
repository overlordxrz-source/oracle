"""Vessel detection on Sentinel-2 (optical) and SAR scenes (Sentinel-1, Capella, Umbra).

Optical: ships are bright in near-infrared against dark water. Water comes from ESA's
scene classification (SCL), with small "holes" in it (the ships themselves, which SCL
often labels as cloud or land) put back. A pixel is a candidate when it is ``k``
local standard deviations brighter than the surrounding water.

SAR: steel hulls are strong radar reflectors and calm water is dark. Water is found
by Otsu-thresholding a coarse median of the backscatter; detection is the same
local-contrast (CFAR-style) test in dB.

Length/width come from the principal axes of each detected blob, so they include
bright wake/turbulence and are only good to about +/- 2 pixels (+/- 20 m on
Sentinel-2). At 10 m a carrier, a VLCC tanker and a large container ship all look
alike; context (escorts, location, news) does the identification.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from rasterio.enums import Resampling
from scipy import ndimage

from .geo import AOI
from .imagery import Grid, NoData, _font, chip, draw_caption, read_href, read_render, to_db
from .models import Scene

LARGE_M = 250.0  # carrier / VLCC / ULCV class
MEDIUM_M = (100.0, 220.0)  # destroyer / frigate / cruiser / supply ship class


@dataclass
class Detection:
    lat: float
    lon: float
    x: float  # projected (UTM) coordinates, for drawing
    y: float
    length_m: float
    width_m: float
    heading_deg: float  # axis orientation from north, 0-180 (direction of travel is ambiguous)
    area_m2: float
    contrast: float  # peak, in local standard deviations above background
    near_shore: bool
    nearby_medium_vessels: int = 0
    # Sentinel-2 only: reflectance above the surrounding water in blue / red / NIR.
    excess: dict[str, float] | None = None

    @property
    def size_class(self) -> str:
        if self.length_m >= LARGE_M:
            return "very large (>=250 m)"
        if self.length_m >= MEDIUM_M[0]:
            return "large (100-250 m)"
        if self.length_m >= 40:
            return "medium (40-100 m)"
        return "small (<40 m)"

    def feature(self) -> dict:
        props = {k: v for k, v in asdict(self).items() if k not in ("lat", "lon", "x", "y")}
        props["size_class"] = self.size_class
        return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [self.lon, self.lat]}, "properties": props}


@dataclass
class DetectionResult:
    scene: Scene
    aoi: AOI
    grid: Grid
    detections: list[Detection]
    warnings: list[str] = field(default_factory=list)

    def geojson(self) -> dict:
        return {
            "type": "FeatureCollection",
            "properties": {
                "scene": self.scene.id,
                "source": self.scene.source,
                "datetime": self.scene.datetime.isoformat(),
                "gsd": self.scene.gsd,
                "warnings": self.warnings,
            },
            "features": [d.feature() for d in self.detections],
        }

    def annotated(self, max_pixels: int = 4096) -> Image.Image:
        """Overview of the AOI with every detection boxed (red = >=250 m)."""
        c = chip(self.scene, self.aoi, max_pixels=max_pixels)
        img = c.image(label=False).convert("RGBA")
        d = ImageDraw.Draw(img)
        inv = ~c.grid.transform
        font = _font(max(11, img.width // 90))
        for det in sorted(self.detections, key=lambda t: t.length_m):
            px, py = inv @ (det.x, det.y)
            r = max(det.length_m * 0.75, 60) / c.grid.res
            r = max(r, 6)
            color = (255, 40, 40) if det.length_m >= LARGE_M else (255, 160, 0) if det.length_m >= MEDIUM_M[0] else (255, 230, 0)
            d.rectangle([px - r, py - r, px + r, py + r], outline=color, width=max(2, img.width // 800))
            if det.length_m >= MEDIUM_M[0]:
                d.text((px + r + 3, py - r), f"{det.length_m:.0f} m", fill=color, font=font)
        n_large = sum(t.length_m >= LARGE_M for t in self.detections)
        return draw_caption(
            img,
            f"{c.caption()}  |  {len(self.detections)} vessel detections, {n_large} >= {LARGE_M:.0f} m",
        )

    def contact_sheet(self, top: int = 12, crop_m: float = 900.0, cell: int = 256) -> Image.Image | None:
        """Close-ups of the largest detections at native resolution."""
        dets = sorted(self.detections, key=lambda t: t.length_m, reverse=True)[:top]
        if not dets:
            return None
        cols = min(4, len(dets))
        rows = math.ceil(len(dets) / cols)
        sheet = Image.new("RGB", (cols * cell, rows * cell), (20, 20, 20))
        font = _font(13)
        for i, det in enumerate(dets):
            try:
                c = chip(self.scene, AOI.from_point(det.lat, det.lon, crop_m / 2000), max_pixels=cell)
            except NoData:
                continue
            tile = c.image(label=False).convert("RGB").resize((cell, cell), Image.NEAREST)
            dd = ImageDraw.Draw(tile)
            dd.rectangle([0, cell - 20, cell, cell], fill=(0, 0, 0))
            dd.text((4, cell - 18), f"{det.length_m:.0f} m  {det.lat:.4f},{det.lon:.4f}", fill=(255, 255, 255), font=font)
            sheet.paste(tile, ((i % cols) * cell, (i // cols) * cell))
        return sheet

    def save(self, out_dir: str | Path, stem: str | None = None) -> dict[str, Path]:
        import json

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = stem or f"{self.scene.source}_{self.scene.date}_ships"
        paths = {"geojson": out_dir / f"{stem}.geojson", "overview": out_dir / f"{stem}.png"}
        paths["geojson"].write_text(json.dumps(self.geojson(), indent=1))
        self.annotated().save(paths["overview"])
        sheet = self.contact_sheet()
        if sheet is not None:
            paths["closeups"] = out_dir / f"{stem}_closeups.jpg"
            sheet.save(paths["closeups"], quality=90)
        return paths


# --------------------------------------------------------------------------- public API


def detect_ships(
    scene: Scene,
    aoi: AOI,
    *,
    k: float | None = None,
    min_length: float = 25.0,
    include_shore: bool = False,
) -> DetectionResult:
    """Detect vessels in ``scene`` over ``aoi``.

    ``k``: detection threshold in local standard deviations (higher = fewer, surer hits).
    ``include_shore``: keep blobs touching land, i.e. moored ships but also piers and cranes.
    """
    if scene.source == "sentinel-2":
        res = _detect_s2(scene, aoi, k=k or 6.0, min_length=min_length)
    elif scene.sensor == "sar":
        res = _detect_sar(scene, aoi, k=k or 5.0, min_length=min_length)
    else:
        raise ValueError(f"ship detection supports sentinel-2 and SAR scenes, not {scene.source}")
    if not include_shore:
        res.detections = [d for d in res.detections if not d.near_shore]
        _tag_task_groups(res.detections)
    return res


# --------------------------------------------------------------------------- optical

SCL_CLOUD = (3, 8, 9, 10)  # cloud shadow, cloud medium/high probability, cirrus
SCL_WATER = 6
SCL_VEGETATION = 4


def _detect_s2(scene: Scene, aoi: AOI, *, k: float, min_length: float, block: int = 2048) -> DetectionResult:
    bands = scene.extra.get("bands", {})
    if "nir" not in bands:
        raise ValueError("scene has no NIR band")
    full = Grid.for_aoi(aoi, 10.0, max_pixels=40_000)
    notes: list[str] = []
    if full.width * full.height > 25_000**2:
        raise ValueError("AOI too large for 10 m detection (max ~250 x 250 km); split it up")
    scale = scene.extra.get("reflectance_scale", 0.0001)
    offset = scene.extra.get("reflectance_offset", 0.0)
    dets: list[Detection] = []
    pad = 64
    for r0 in range(0, full.height, block):
        for c0 in range(0, full.width, block):
            g = _subgrid(full, c0 - pad, r0 - pad, block + 2 * pad, block + 2 * pad)
            nir = read_href(bands["nir"], g, [1])[0] * scale + offset
            if np.isnan(nir).all():
                continue
            green = read_href(bands["green"], g, [1])[0] * scale + offset if "green" in bands else None
            red = read_href(bands["red"], g, [1])[0] * scale + offset if "red" in bands else None
            blue = read_href(bands["blue"], g, [1])[0] * scale + offset if "blue" in bands else None
            scl = read_href(bands["scl"], g, [1], Resampling.nearest)[0] if "scl" in bands else None
            sea, cloud, land = _s2_masks(nir, green, red, scl)
            found = _find_targets(
                nir,
                sea,
                cloud,
                land,
                g,
                k=k,
                min_contrast=0.03,
                sigma_m=400.0,
                min_area_m2=200.0,
                shore_buffer_px=5,
                aux={name: a for name, a in (("blue", blue), ("red", red), ("nir", nir)) if a is not None},
            )
            found = [d for d in found if _hull_like(d)]
            core = (c0, r0, c0 + block, r0 + block)
            for det in found:
                col, row = ~full.transform @ (det.x, det.y)
                if core[0] <= col < core[2] and core[1] <= row < core[3]:
                    dets.append(det)
    return _finish(scene, aoi, full, dets, min_length, notes)


def _hull_like(d: Detection) -> bool:
    """Reject blobs that can't be a hull seen at 10 m.

    Shape: long blobs must be narrow (drops cumulus puffs, fish traps, merged clusters).
    Spectrum, measured on real hulls vs look-alikes: painted hulls are brighter than the
    water in red (floating vegetation and wet mud are darker or barely brighter). Cloud
    is spectrally flat (NIR/blue excess ~0.9 vs 1.3-8 for most hulls), but so are grey
    warship paint and white boats, so a flat blob is only rejected when it is also faint
    and puffy rather than elongated, or very faint.
    """
    if d.length_m >= 120 and d.width_m > 0.4 * d.length_m:
        return False
    e = d.excess
    if e and {"blue", "red", "nir"} <= e.keys():
        if e["red"] < 0 or (e["red"] < 0.02 and d.contrast < 15):
            return False  # darker than water in red, or faint with no paint signal
        flat = e["nir"] < 1.15 * e["blue"]
        if flat and (d.contrast < 10 or (d.contrast < 25 and d.width_m > 0.3 * d.length_m)):
            return False
    return True


def _s2_masks(nir: np.ndarray, green: np.ndarray | None, red: np.ndarray | None, scl: np.ndarray | None):
    valid = np.isfinite(nir)
    vegetated = np.zeros(nir.shape, bool)
    if red is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            # Strict NDVI: dark-blue or green hulls reach ~0.3-0.5 against water; islands more.
            vegetated = np.nan_to_num((nir - red) / (nir + red)) > 0.55
    if scl is not None and np.isfinite(scl).any():
        s = np.nan_to_num(scl, nan=0).astype(np.uint8)
        water = s == SCL_WATER
        cloud_raw = np.isin(s, SCL_CLOUD)
        vegetated |= s == SCL_VEGETATION
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            ndwi = (green - nir) / (green + nir) if green is not None else -nir
        water = np.nan_to_num(ndwi) > 0.0
        cloud_raw = np.zeros_like(water)
    # Bright ships are often classified as cloud: only large cloud blobs are cloud.
    lab, n = ndimage.label(cloud_raw)
    if n:
        sizes = ndimage.sum_labels(cloud_raw, lab, index=np.arange(1, n + 1))
        big = np.zeros(n + 1, bool)
        big[1:] = sizes > 400  # 4 ha: far larger than any ship's bright blob
        cloud = ndimage.binary_dilation(big[lab], iterations=4)
    else:
        cloud = np.zeros_like(water)
    # 400 px = 4 ha: comfortably bigger than a 340 x 80 m carrier, smaller than most islets.
    sea = _fill_small_holes(water, max_hole_px=400, land_evidence=vegetated)
    land = valid & ~sea & ~cloud
    return sea & valid & ~cloud, cloud, land


# --------------------------------------------------------------------------- SAR


def _detect_sar(scene: Scene, aoi: AOI, *, k: float, min_length: float) -> DetectionResult:
    res = max(scene.gsd, 1.0) if scene.source != "sentinel-1" else 10.0
    grid = Grid.for_aoi(aoi, res, max_pixels=6000)
    raw = read_render(scene.render, grid)[0]
    db = to_db(raw, scene.render.power)
    valid = np.isfinite(db)
    if not valid.any():
        raise NoData("no SAR pixels in AOI")
    notes: list[str] = []
    f = max(1, int(round(120.0 / grid.res)))
    coarse = _block_nanmedian(db, f)
    vals = coarse[np.isfinite(coarse)]
    thr, sep = _otsu(vals)
    if sep < 5.0:
        if scene.source == "sentinel-1":
            thr = -14.0  # calibrated VV: open water sits well below this
        else:
            thr = np.inf
            notes.append("no clear land/water contrast; treating the whole AOI as water (expect land clutter)")
    water_c = np.nan_to_num(coarse, nan=np.inf) < thr
    water_c = ndimage.binary_opening(water_c, iterations=1)
    water = np.repeat(np.repeat(water_c, f, axis=0), f, axis=1)
    water = np.pad(water, ((0, max(0, db.shape[0] - water.shape[0])), (0, max(0, db.shape[1] - water.shape[1]))))
    water = water[: db.shape[0], : db.shape[1]] & valid
    max_hole = int(60_000 / grid.res**2)  # ships plus their bright sidelobes / wake
    sea = _fill_small_holes(water, max_hole_px=max_hole) & valid
    land = valid & ~sea
    cloud = np.zeros_like(sea)
    dets = _find_targets(
        db,
        sea,
        cloud,
        land,
        grid,
        k=k,
        min_contrast=8.0,
        sigma_m=max(25 * grid.res, 300.0),
        min_area_m2=max(80.0, 2 * grid.res**2),
    )
    return _finish(scene, aoi, grid, dets, min_length, notes)


def _block_nanmedian(a: np.ndarray, f: int) -> np.ndarray:
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    if h == 0 or w == 0:
        return np.array([[np.nanmedian(a)]])
    blocks = a[:h, :w].reshape(h // f, f, w // f, f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks are expected
        return np.nanmedian(blocks, axis=(1, 3))


def _otsu(values: np.ndarray) -> tuple[float, float]:
    """Otsu threshold and the separation of the two class means."""
    if values.size < 10:
        return float("inf"), 0.0
    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m = np.cumsum(hist * centers)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = m / w0
        mu1 = (m[-1] - m) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
    between = np.nan_to_num(between)
    i = int(np.argmax(between))
    return float(centers[i]), float(abs(np.nan_to_num(mu1[i]) - np.nan_to_num(mu0[i])))


# --------------------------------------------------------------------------- shared core


def _fill_small_holes(mask: np.ndarray, max_hole_px: int, land_evidence: np.ndarray | None = None) -> np.ndarray:
    """Add enclosed non-water specks (candidate hulls) back into the water mask.

    Holes larger than ``max_hole_px``, or where >20% of pixels are known land
    (e.g. vegetation), are islands and stay out.
    """
    holes = ndimage.binary_fill_holes(mask) & ~mask
    lab, n = ndimage.label(holes)
    if not n:
        return mask
    idx = np.arange(1, n + 1)
    sizes = ndimage.sum_labels(holes, lab, index=idx)
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes <= max_hole_px
    if land_evidence is not None:
        land_frac = ndimage.sum_labels(land_evidence & holes, lab, index=idx) / np.maximum(sizes, 1)
        keep[1:] &= land_frac <= 0.2
    return mask | keep[lab]


def _find_targets(
    values: np.ndarray,
    sea: np.ndarray,
    cloud: np.ndarray,
    land: np.ndarray,
    grid: Grid,
    *,
    k: float,
    min_contrast: float,
    sigma_m: float,
    min_area_m2: float,
    shore_buffer_px: int = 3,
    aux: dict[str, np.ndarray] | None = None,
) -> list[Detection]:
    if sea.sum() < 100:
        return []
    v = values[sea]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826 + 1e-9
    clipped = np.where(sea, np.minimum(np.nan_to_num(values, nan=med), med + 3 * mad), 0.0).astype(np.float32)
    w = sea.astype(np.float32)
    sigma = float(np.clip(sigma_m / grid.res, 6, 80))
    gw = ndimage.gaussian_filter(w, sigma) + 1e-6
    mean = ndimage.gaussian_filter(clipped, sigma) / gw
    var = ndimage.gaussian_filter(clipped * clipped, sigma) / gw - mean**2
    std = np.sqrt(np.maximum(var, (0.5 * mad) ** 2))
    excess = np.nan_to_num(values - mean, nan=0.0)
    cand = sea & (excess > k * std) & (excess > min_contrast)
    lab, n = ndimage.label(cand, structure=np.ones((3, 3)))
    if not n:
        return []
    near_land = ndimage.binary_dilation(land | cloud, iterations=shore_buffer_px)
    to_wgs = Transformer.from_crs(grid.crs, 4326, always_xy=True)
    res = grid.res
    min_px = max(2, int(math.ceil(min_area_m2 / res**2)))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        blob = lab[sl] == i
        npx = int(blob.sum())
        if npx < min_px:
            continue
        rows, cols = np.nonzero(blob)
        rows = rows + sl[0].start
        cols = cols + sl[1].start
        xs, ys = cols * res, rows * res
        if npx >= 2:
            cov = np.cov(np.vstack([xs, ys]))
            evals, evecs = np.linalg.eigh(cov)
            l1, l2 = max(evals[1], 0), max(evals[0], 0)
            vx, vy = evecs[:, 1]
        else:
            l1 = l2 = 0.0
            vx, vy = 1.0, 0.0
        length = max(math.sqrt(12 * l1), res)
        width = max(math.sqrt(12 * l2), res)
        heading = (math.degrees(math.atan2(vx, -vy)) + 180) % 180
        cr, cc = float(rows.mean()), float(cols.mean())
        x, y = grid.transform @ (cc + 0.5, cr + 0.5)
        lon, lat = to_wgs.transform(x, y)
        peak = float(np.max((excess[sl][blob]) / std[sl][blob]))
        out.append(
            Detection(
                lat=round(lat, 6),
                lon=round(lon, 6),
                x=x,
                y=y,
                length_m=round(length, 1),
                width_m=round(width, 1),
                heading_deg=round(heading, 1),
                area_m2=round(npx * res * res, 1),
                contrast=round(peak, 1),
                near_shore=bool(near_land[sl][blob].any()),
                excess=_ring_excess(aux, lab, i, sl, sea) if aux else None,
            )
        )
    return out


def _ring_excess(aux: dict[str, np.ndarray], lab: np.ndarray, i: int, sl: tuple, sea: np.ndarray) -> dict[str, float]:
    """Blob mean minus the median of a thin ring of water around it, per band."""
    pad = 6
    r0, c0 = max(sl[0].start - pad, 0), max(sl[1].start - pad, 0)
    win = (slice(r0, sl[0].stop + pad), slice(c0, sl[1].stop + pad))
    blob = lab[win] == i
    ring = ndimage.binary_dilation(blob, iterations=4) & ~ndimage.binary_dilation(blob, iterations=1) & sea[win]
    out = {}
    for k, a in aux.items():
        w = a[win]
        bg = np.nanmedian(w[ring]) if ring.any() else np.nanmedian(w)
        out[k] = round(float(np.nanmean(w[blob]) - bg), 4)
    return out


def _finish(scene: Scene, aoi: AOI, grid: Grid, dets: list[Detection], min_length: float, notes: list[str]) -> DetectionResult:
    dets = [d for d in dets if d.length_m >= min_length]
    if len(dets) > 3000:
        notes.append(f"{len(dets)} detections: likely clutter (land, sea state, ice); raise --k or shrink the AOI")
    _tag_task_groups(dets)
    dets.sort(key=lambda d: d.length_m, reverse=True)
    return DetectionResult(scene, aoi, grid, dets, notes)


def _tag_task_groups(dets: list[Detection]) -> None:
    """Task-group hint: a very large hull with several 100-220 m hulls within 15 km."""
    big = [d for d in dets if d.length_m >= LARGE_M]
    mids = [d for d in dets if MEDIUM_M[0] <= d.length_m <= MEDIUM_M[1]]
    for b in big:
        b.nearby_medium_vessels = sum(math.hypot(m.x - b.x, m.y - b.y) <= 15_000 for m in mids)


def _subgrid(g: Grid, col: int, row: int, w: int, h: int) -> Grid:
    from affine import Affine

    c0, r0 = max(col, 0), max(row, 0)
    c1, r1 = min(col + w, g.width), min(row + h, g.height)
    t = g.transform @ Affine.translation(c0, r0)
    return Grid(g.crs, t, max(1, c1 - c0), max(1, r1 - r0))


__all__ = ["Detection", "DetectionResult", "detect_ships"]
