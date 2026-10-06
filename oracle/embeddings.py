"""Planetary embeddings: search, compare and visualise the Earth with AlphaEarth Foundations.

AlphaEarth Foundations (Google DeepMind, 2025) is a geospatial foundation model that
fuses a year of Sentinel-1/2, Landsat, elevation, climate and more into a 64-number
embedding for every 10 m pixel on land and coastal water. Similar places get similar
vectors, whatever makes them similar (a solar farm, a container terminal, an airfield,
a refinery, a burnt forest). The annual embeddings for 2017-2025 are public (CC BY 4.0)
as cloud-optimised GeoTIFFs on Source Cooperative, so Oracle reads them directly:

  find_similar     few-shot search: give one or more example points ("like this") and
                   get every place in an area whose embedding matches, ranked. With 3+
                   examples a small logistic probe is trained against the background.
  semantic_change  year-over-year cosine distance between annual embeddings: change in
                   what a place *is* (new port, cleared forest, new town), robust to
                   cloud and season because each embedding summarises a whole year.
  embedding_view   the 64 dimensions projected to RGB with PCA: a false-colour map in
                   which visually similar colours mean similar land.

Reading: the index (one GeoParquet, ~78 MB, downloaded once) is turned into a local
SQLite R*Tree. The COGs are stored bottom-up (positive y resolution), so windows are
read in source row order, flipped, then reprojected onto the analysis grid. Values are
int8, de-quantised as sign(v) * (v / 127.5)^2 and renormalised to unit length.

Attribution (required by the licence): "The AlphaEarth Foundations Satellite Embedding
dataset is produced by Google and Google DeepMind."
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window
from scipy import ndimage

from .config import CACHE_DIR
from .geo import AOI
from .http import client, log
from .imagery import Grid, NoData

INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
DATA_PREFIX = ("s3://us-west-2.opendata.source.coop/", "https://data.source.coop/")
AEF_DIR = CACHE_DIR / "aef"
INDEX_DB = AEF_DIR / "index.db"
YEARS = range(2017, 2026)
DIMS = 64
MAX_PIXELS = 1024  # per side; 64 float32 bands of 1024^2 is 256 MB
READ_THREADS = 32
ATTRIBUTION = "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."
_lock = threading.Lock()


# --------------------------------------------------------------------------- index


def _index() -> sqlite3.Connection:
    with _lock:
        if not INDEX_DB.exists():
            _build_index()
        return sqlite3.connect(INDEX_DB, check_same_thread=False)


def _build_index() -> None:
    import pyarrow.parquet as pq

    AEF_DIR.mkdir(parents=True, exist_ok=True)
    pq_path = AEF_DIR / "aef_index.parquet"
    if not pq_path.exists():
        log("downloading the AlphaEarth tile index (78 MB, once)...")
        with client() as c, c.stream("GET", INDEX_URL, timeout=600) as r:
            r.raise_for_status()
            with open(pq_path.with_suffix(".part"), "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        pq_path.with_suffix(".part").rename(pq_path)
    cols = ["path", "year", "crs", "wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north"]
    t = pq.read_table(pq_path, columns=cols).to_pydict()
    tmp = INDEX_DB.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.executescript(
        "CREATE TABLE tiles(id INTEGER PRIMARY KEY, year INTEGER, path TEXT, crs TEXT);"
        "CREATE VIRTUAL TABLE tiles_rt USING rtree(id, w, e, s, n);"
    )
    rows = zip(t["path"], t["year"], t["crs"], t["wgs84_west"], t["wgs84_south"], t["wgs84_east"], t["wgs84_north"], strict=True)
    with db:
        for i, (p, y, crs, w, s, e, n) in enumerate(rows):
            db.execute("INSERT INTO tiles VALUES(?,?,?,?)", (i, y, p.replace(*DATA_PREFIX), crs))
            db.execute("INSERT INTO tiles_rt VALUES(?,?,?,?,?)", (i, w, e, s, n))
    db.close()
    tmp.rename(INDEX_DB)
    pq_path.unlink(missing_ok=True)  # the compact index is all we need
    log(f"AlphaEarth index ready: {len(t['path'])} tiles")


def tiles_for(bbox: tuple[float, float, float, float], year: int) -> list[tuple[str, str]]:
    """-> [(https url of the COG, crs)] covering bbox in a year."""
    w, s, e, n = bbox
    db = _index()
    try:
        rows = db.execute(
            "SELECT t.path, t.crs FROM tiles t JOIN tiles_rt r ON r.id=t.id "
            "WHERE t.year=? AND r.w<=? AND r.e>=? AND r.s<=? AND r.n>=?",
            (year, e, w, n, s),
        ).fetchall()
    finally:
        db.close()
    return rows


def available_years(lat: float, lon: float) -> list[int]:
    db = _index()
    try:
        rows = db.execute(
            "SELECT DISTINCT t.year FROM tiles t JOIN tiles_rt r ON r.id=t.id WHERE r.w<=? AND r.e>=? AND r.s<=? AND r.n>=?",
            (lon, lon, lat, lat),
        ).fetchall()
    finally:
        db.close()
    return sorted(r[0] for r in rows)


# --------------------------------------------------------------------------- reading


def dequantize(raw: np.ndarray) -> np.ndarray:
    v = raw.astype(np.float32) / 127.5
    return v * np.abs(v)  # == sign(v) * v^2


def _normalize(e: np.ndarray, axis: int = 0) -> np.ndarray:
    n = np.linalg.norm(e, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 1e-6, e / n, np.nan)


BLOCK = 1024  # the COGs' internal tile size
BLOCK_DIR = AEF_DIR / "blocks"
CACHE_GB = float(os.environ.get("ORACLE_AEF_CACHE_GB", "4"))


def _fetch(url: str, win: Window, out_shape: tuple[int, int]) -> np.ndarray:
    """(64, h, w) int8 for a window, fetched band pairs in parallel.

    The 64 dimensions are separate bands (planar layout), so a window read is ~64x the
    HTTP requests of an RGB read: done serially it took ~50 s for 10x10 km, in parallel ~5 s.
    """

    def grab(bands: list[int]) -> np.ndarray:
        for attempt in range(4):
            try:
                with rasterio.open(url) as s:
                    return s.read(bands, window=win, out_shape=(len(bands), *out_shape), resampling=Resampling.nearest)
            except rasterio.errors.RasterioIOError:
                if attempt == 3:
                    raise
                time.sleep(0.5 * 2**attempt)  # transient HTTP errors under parallel load
        raise AssertionError("unreachable")

    groups = [[b, b + 1] for b in range(1, DIMS + 1, 2)]
    with ThreadPoolExecutor(max_workers=READ_THREADS) as pool:
        return np.concatenate(list(pool.map(grab, groups)))


def _block(url: str, bc: int, br: int, width: int, height: int) -> np.ndarray:
    """One full-resolution 1024x1024x64 block, from the disk cache when possible.

    A window always costs whole blocks of every band, so caching blocks makes every later
    analysis, example point or year comparison in the same area local and instant.
    """
    key = hashlib.sha1(url.encode()).hexdigest()[:20]
    path = BLOCK_DIR / key / f"{br}_{bc}.npy"
    if path.exists():
        try:
            return np.load(path, mmap_mode="r")
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
    win = Window(bc * BLOCK, br * BLOCK, min(BLOCK, width - bc * BLOCK), min(BLOCK, height - br * BLOCK))
    arr = _fetch(url, win, (int(win.height), int(win.width)))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.rename(path)
    _trim_cache()
    return arr


def _trim_cache() -> None:
    files = sorted(BLOCK_DIR.rglob("*.npy"), key=lambda f: f.stat().st_atime)
    total = sum(f.stat().st_size for f in files)
    while files and total > CACHE_GB * 1e9:
        f = files.pop(0)
        total -= f.stat().st_size
        f.unlink(missing_ok=True)


def _read_tile(url: str, grid: Grid) -> np.ndarray | None:
    """One bottom-up AEF COG onto ``grid`` -> (64, h, w) float32 unit vectors, NaN = none."""
    with rasterio.open(url) as src:
        t = src.transform
        sb = transform_bounds(grid.crs, src.crs, *grid.bounds, densify_pts=21)
        inv = ~t
        cw, rs = inv @ (sb[0], sb[1])  # rows grow northward in these files
        ce, rn = inv @ (sb[2], sb[3])
        c0, c1 = max(0, math.floor(min(cw, ce)) - 2), min(src.width, math.ceil(max(cw, ce)) + 2)
        r0, r1 = max(0, math.floor(min(rs, rn)) - 2), min(src.height, math.ceil(max(rs, rn)) + 2)
        if c1 <= c0 or r1 <= r0:
            return None
        win = Window(c0, r0, c1 - c0, r1 - r0)
        target_res_src = (sb[2] - sb[0]) / grid.width
        factor = max(1.0, target_res_src / abs(t.a) / 1.5)
        src_crs, width, height = src.crs, src.width, src.height
    if factor > 1.0:  # coarse analysis: GDAL serves it from the overviews, which are small
        ow, oh = max(1, round(win.width / factor)), max(1, round(win.height / factor))
        raw = _fetch(url, win, (oh, ow))
    else:  # full resolution: assemble from cached blocks
        ow, oh = c1 - c0, r1 - r0
        raw = np.full((DIMS, oh, ow), -128, np.int8)
        blocks = [
            (bc, br) for br in range(r0 // BLOCK, (r1 - 1) // BLOCK + 1) for bc in range(c0 // BLOCK, (c1 - 1) // BLOCK + 1)
        ]
        with ThreadPoolExecutor(max_workers=2) as pool:
            got = list(pool.map(lambda b: _block(url, b[0], b[1], width, height), blocks))
        for (bc, br), blk in zip(blocks, got, strict=True):
            y0, x0 = br * BLOCK, bc * BLOCK
            ys, ye = max(r0, y0), min(r1, y0 + blk.shape[1])
            xs, xe = max(c0, x0), min(c1, x0 + blk.shape[2])
            raw[:, ys - r0 : ye - r0, xs - c0 : xe - c0] = blk[:, ys - y0 : ye - y0, xs - x0 : xe - x0]
    valid = raw[0] != -128
    if not valid.any():
        return None
    emb = np.where(valid, dequantize(raw), np.nan)[:, ::-1, :]  # flip to north-up
    fx, fy = win.width / ow, win.height / oh
    north = t.f + (r0 + win.height) * t.e
    src_tr = Affine(t.a * fx, 0, t.c + c0 * t.a, 0, -t.e * fy, north)
    out = np.full((DIMS, grid.height, grid.width), np.nan, np.float32)
    reproject(
        np.ascontiguousarray(emb),
        out,
        src_transform=src_tr,
        src_crs=src_crs,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return out


def read(aoi: AOI, year: int, max_pixels: int = MAX_PIXELS, res: float = 10.0) -> tuple[np.ndarray, Grid]:
    """Embeddings of ``aoi`` in ``year`` -> ((h, w, 64) unit vectors, NaN outside data; grid)."""
    if year not in YEARS:
        raise ValueError(f"AlphaEarth annual embeddings cover {YEARS.start}-{YEARS.stop - 1}")
    grid = Grid.for_aoi(aoi, res, max_pixels=max_pixels)
    urls = tiles_for(aoi.bbox, year)
    if not urls:
        raise NoData(f"no AlphaEarth embeddings for this area in {year}")
    out: np.ndarray | None = None
    for url, _crs in urls:
        try:
            a = _read_tile(url, grid)
        except rasterio.errors.RasterioIOError as exc:
            log(f"  aef {url[-40:]}: {exc}")
            continue
        if a is None:
            continue
        if out is None:
            out = a
        else:
            hole = np.isnan(out[0])
            out[:, hole] = a[:, hole]
    if out is None or np.isnan(out[0]).all():
        raise NoData(f"no AlphaEarth embeddings for this area in {year}")
    return np.moveaxis(_normalize(out, 0), 0, -1), grid


def at_points(points: list[tuple[float, float]], year: int, radius_m: float = 15.0) -> np.ndarray:
    """Mean unit embedding around each (lat, lon) -> (k, 64)."""
    vecs = []
    for lat, lon in points:
        emb, _ = read(AOI.from_point(lat, lon, max(radius_m, 10.0) / 1000), year, max_pixels=16)
        v = np.nanmean(emb.reshape(-1, DIMS), axis=0)
        if np.isnan(v).any():
            raise NoData(f"no embedding at {lat:.5f},{lon:.5f} in {year}")
        vecs.append(v / np.linalg.norm(v))
    return np.array(vecs, np.float32)


# --------------------------------------------------------------------------- results


@dataclass
class Match:
    lat: float
    lon: float
    score: float  # cosine similarity (or probe probability)
    area_m2: float
    bbox: tuple[float, float, float, float]
    id: str = ""

    def feature(self) -> dict:
        return {
            "type": "Feature",
            "id": self.id,
            "geometry": {"type": "Point", "coordinates": [self.lon, self.lat]},
            "properties": {
                "score": round(self.score, 3),
                "area_m2": round(self.area_m2),
                "bbox": [round(v, 6) for v in self.bbox],
            },
        }


@dataclass
class EmbeddingResult:
    kind: str  # similar | change | view
    aoi: AOI
    grid: Grid
    years: list[int]
    image: np.ndarray  # (h, w, 4) uint8 RGBA overlay
    score: np.ndarray | None = None  # (h, w) float map
    matches: list[Match] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            key = f"{self.kind}|{self.years}|{self.aoi.bbox}|{self.stats.get('key', '')}"
            self.id = hashlib.sha1(key.encode()).hexdigest()[:12]

    @property
    def corners(self) -> list[list[float]]:
        w, s, e, n = self.grid.bounds
        tr = Transformer.from_crs(self.grid.crs, 4326, always_xy=True)
        return [list(tr.transform(x, y)) for x, y in ((w, n), (e, n), (e, s), (w, s))]

    def png(self) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(self.image, "RGBA").save(buf, format="PNG")
        return buf.getvalue()

    def summary(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "years": self.years,
            "corners": self.corners,
            "resolution_m": round(self.grid.res, 1),
            "stats": {k: v for k, v in self.stats.items() if k != "key"},
            "attribution": ATTRIBUTION,
        }

    def geojson(self) -> dict:
        return {"type": "FeatureCollection", "properties": self.summary(), "features": [m.feature() for m in self.matches]}


_RECENT: dict[str, EmbeddingResult] = {}


def remember(r: EmbeddingResult, keep: int = 8) -> EmbeddingResult:
    _RECENT[r.id] = r
    while len(_RECENT) > keep:
        _RECENT.pop(next(iter(_RECENT)))
    return r


def recall(rid: str) -> EmbeddingResult | None:
    return _RECENT.get(rid)


def _regions(score: np.ndarray, mask: np.ndarray, grid: Grid, top: int, min_px: int = 2) -> list[Match]:
    tr = Transformer.from_crs(grid.crs, 4326, always_xy=True)
    lab, n = ndimage.label(mask)
    if not n:
        return []
    objs = ndimage.find_objects(lab)
    peaks = ndimage.maximum(score, lab, np.arange(1, n + 1))
    sizes = ndimage.sum_labels(mask, lab, np.arange(1, n + 1))
    order = np.argsort(-np.asarray(peaks))
    out = []
    px = grid.res * grid.res
    for i in order:
        if sizes[i] < min_px:
            continue
        sl = objs[i]
        rr, cc = np.nonzero(lab[sl] == i + 1)
        rr, cc = rr + sl[0].start, cc + sl[1].start
        w = score[rr, cc]
        k = int(np.argmax(w))
        x, y = grid.transform @ (cc[k] + 0.5, rr[k] + 0.5)
        lon, lat = tr.transform(x, y)
        x0, y0 = grid.transform @ (cc.min(), rr.max() + 1)
        x1, y1 = grid.transform @ (cc.max() + 1, rr.min())
        (w0, s0), (e0, n0) = tr.transform(x0, y0), tr.transform(x1, y1)
        out.append(
            Match(
                lat,
                lon,
                float(peaks[i]),
                float(sizes[i] * px),
                (w0, s0, e0, n0),
                hashlib.sha1(f"{lat:.5f}{lon:.5f}".encode()).hexdigest()[:10],
            )
        )
        if len(out) >= top:
            break
    return out


def _heat_rgba(score: np.ndarray, lo: float, hi: float, rgb: tuple[int, int, int]) -> np.ndarray:
    a = np.clip((np.nan_to_num(score, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1)
    out = np.zeros((*score.shape, 4), np.uint8)
    out[..., :3] = rgb
    out[..., 3] = (a**1.5 * 230).astype(np.uint8)
    return out


# --------------------------------------------------------------------------- analyses


def find_similar(
    aoi: AOI,
    examples: list[tuple[float, float]],
    year: int = 2025,
    negatives: list[tuple[float, float]] | None = None,
    threshold: float | None = None,
    top: int = 50,
    exclude_m: float = 150.0,
) -> EmbeddingResult:
    """Places in ``aoi`` whose embedding matches the examples'.

    Similarity is cosine in a *whitened* space: centred on the area's own mean embedding
    and scaled by its principal components. Raw cosine is nearly useless for search,
    because every pixel shares a large common component: in a desert everything scores
    ~0.97 against a solar farm. After whitening, the example's own region scored 0.96
    and plain desert ~0.0 (median). On Fujairah's tank farms, other tank farms scored
    0.84-0.93 and the best non-tank look-alikes (tower blocks) 0.82-0.86: >= 0.7 is a
    look-alike worth checking, not an identification.
    With 3+ examples (or any negatives) a logistic probe on the whitened features
    replaces the prototype.
    """
    if not examples:
        raise ValueError("give at least one example point")
    pos = at_points(examples, year)
    emb, grid = read(aoi, year)
    valid = ~np.isnan(emb[..., 0])
    if valid.sum() < 100:
        raise NoData("too little embedding coverage in the search area")
    flat = np.nan_to_num(emb.reshape(-1, DIMS))
    rng = np.random.default_rng(0)
    bg = flat[valid.ravel()]
    sample = bg[rng.choice(len(bg), size=min(len(bg), 20000), replace=False)]
    mu, W = _whitener(sample)
    zx = _unit((flat - mu) @ W)
    zp = _unit((pos - mu) @ W)
    if len(pos) >= 3 or negatives:
        neg = at_points(negatives, year) if negatives else np.zeros((0, DIMS), np.float32)
        zn = _unit((neg - mu) @ W) if len(neg) else np.zeros((0, DIMS))
        zb = _unit((sample[:4000] - mu) @ W)
        score = _probe(zp, np.vstack([zn, zb]), zx).reshape(valid.shape)
        thr = threshold if threshold is not None else 0.8
        method = f"whitened logistic probe ({len(pos)} positives, {len(zn)} negatives + background)"
    else:
        proto = _unit(zp.mean(0, keepdims=True))[0]
        score = (zx @ proto).reshape(valid.shape)
        thr = threshold if threshold is not None else 0.7
        method = "whitened cosine to the example" + ("s" if len(pos) > 1 else "")
    score = np.where(valid, score, np.nan)
    mask = np.nan_to_num(score, nan=-1) >= thr
    # Don't return the examples as discoveries: drop every matching region that touches one
    # (the rest of the example's own facility is not a new find).
    tr = Transformer.from_crs(4326, grid.crs, always_xy=True)
    inv = ~grid.transform
    r = max(1, int(exclude_m / grid.res))
    lab, _ = ndimage.label(mask)
    own: set[int] = set()
    for lat, lon in examples:
        c, rr = inv @ tr.transform(lon, lat)
        c, rr = int(c), int(rr)
        win = lab[max(0, rr - r) : rr + r + 1, max(0, c - r) : c + r + 1]
        own |= set(np.unique(win[win > 0]).tolist())
    example_area = float(np.isin(lab, list(own)).sum()) * grid.res**2 if own else 0.0
    if own:
        mask &= ~np.isin(lab, list(own))
    matches = _regions(np.nan_to_num(score, nan=-1), mask, grid, top)
    lo = max(thr - 0.35, 0.0)
    return EmbeddingResult(
        "similar",
        aoi,
        grid,
        [year],
        _heat_rgba(score, lo, max(thr + 0.2, lo + 0.1), (255, 255, 255)),
        score,
        matches,
        {
            "method": method,
            "threshold": round(thr, 3),
            "examples": len(examples),
            "background_median": round(float(np.nanmedian(score)), 3),
            "example_facility_km2": round(example_area / 1e6, 3),
            "matched_area_km2": round(float(mask.sum()) * grid.res**2 / 1e6, 3),
            "key": f"{examples}|{negatives}|{threshold}",
        },
    )


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def _whitener(sample: np.ndarray, ridge: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """PCA whitening fitted on the area's own embeddings (ridge keeps tiny axes from exploding)."""
    mu = sample.mean(0)
    _, s, vt = np.linalg.svd(sample - mu, full_matrices=False)
    sd = s / np.sqrt(len(sample))
    return mu, vt.T / (sd + sd.max() * ridge)


def _probe(pos: np.ndarray, neg: np.ndarray, x: np.ndarray, l2: float = 1e-2, iters: int = 300) -> np.ndarray:
    """Tiny L2 logistic regression (positives up-weighted to balance), gradient descent."""
    X = np.vstack([pos, neg]).astype(np.float64)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    sw = np.where(y == 1, len(neg) / max(len(pos), 1), 1.0)
    sw /= sw.sum()
    w, b = np.zeros(X.shape[1]), 0.0
    lr = 2.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ (sw * g) + l2 * w)
        b -= lr * float(sw @ g)
    return (1 / (1 + np.exp(-(x @ w + b)))).astype(np.float32)


def semantic_change(
    aoi: AOI, year_a: int, year_b: int, k: float = 3.0, top: int = 40, min_area_m2: float = 2000
) -> EmbeddingResult:
    """Where did what a place *is* change between two years? Cosine distance of embeddings."""
    ea, grid = read(aoi, year_a)
    eb, _ = read(aoi, year_b)
    valid = ~np.isnan(ea[..., 0]) & ~np.isnan(eb[..., 0])
    if valid.mean() < 0.05:
        raise NoData("too little embedding coverage in one of the years")
    dist = np.where(valid, 1 - np.sum(np.nan_to_num(ea) * np.nan_to_num(eb), axis=-1), np.nan)
    v = dist[valid]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    z = (dist - med) / max(mad, 0.01)
    mask = valid & (z > k) & (dist > 0.15)
    mask = ndimage.binary_opening(mask, iterations=1) if grid.res <= 12 else mask
    matches = [m for m in _regions(np.nan_to_num(dist), mask, grid, top * 2) if m.area_m2 >= min_area_m2][:top]
    return EmbeddingResult(
        "change",
        aoi,
        grid,
        [year_a, year_b],
        _heat_rgba(dist, med + 2 * mad, float(np.nanquantile(v, 0.999)) if v.size else 1.0, (244, 114, 182)),
        dist,
        matches,
        {
            "median_distance": round(med, 3),
            "changed_area_km2": round(float(mask.sum()) * grid.res**2 / 1e6, 3),
            "share_changed": round(float(mask.sum() / max(valid.sum(), 1)), 4),
            "key": f"{k}",
        },
    )


def embedding_view(aoi: AOI, year: int = 2025, segments: int = 0) -> EmbeddingResult:
    """PCA false-colour of the embeddings (optionally k-means segments instead)."""
    emb, grid = read(aoi, year)
    valid = ~np.isnan(emb[..., 0])
    X = emb[valid]
    rng = np.random.default_rng(0)
    sample = X[rng.choice(len(X), size=min(len(X), 20000), replace=False)]
    mu = sample.mean(0)
    _, _, vt = np.linalg.svd(sample - mu, full_matrices=False)
    rgba = np.zeros((*valid.shape, 4), np.uint8)
    stats: dict = {}
    if segments:
        cents = _kmeans(sample, segments, rng)
        labels = np.argmin(((X[:, None, :] - cents[None]) ** 2).sum(-1), axis=1)
        # colour clusters by their position in PCA space so similar clusters look alike
        cc = (cents - mu) @ vt[:3].T
        cc = (cc - cc.min(0)) / np.maximum(np.ptp(cc, 0), 1e-6)
        rgba[valid, :3] = (cc[labels] * 220 + 25).astype(np.uint8)
        stats["segments"] = segments
        stats["share"] = [round(float((labels == i).mean()), 3) for i in range(segments)]
    else:
        p = (X - mu) @ vt[:3].T
        lo, hi = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
        rgba[valid, :3] = (np.clip((p - lo) / np.maximum(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    rgba[valid, 3] = 235
    stats["key"] = f"{segments}"
    return EmbeddingResult("view", aoi, grid, [year], rgba, None, [], stats)


def _kmeans(x: np.ndarray, k: int, rng, iters: int = 25) -> np.ndarray:
    c = x[rng.choice(len(x), size=k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        for i in range(k):
            if (lab == i).any():
                c[i] = x[lab == i].mean(0)
    return c
