"""Read any scene onto any grid, straight from cloud-optimized GeoTIFFs.

Only the bytes covering the requested window (at the matching overview level) are
fetched, so a 30 cm Maxar strip or a 110 km Sentinel-2 tile both render in seconds.
"""

from __future__ import annotations

import functools
import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image, ImageDraw, ImageFont
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from . import config  # noqa: F401  (applies GDAL env defaults)
from .geo import AOI, mercator_tile_bounds
from .models import Render, Scene
from .sources.planetary import sign as pc_sign

WEB_MERCATOR = CRS.from_epsg(3857)


class NoData(Exception):
    """The requested area has no valid pixels in this scene."""


@dataclass
class Grid:
    crs: CRS
    transform: Affine
    width: int
    height: int

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return array_bounds(self.height, self.width, self.transform)

    @property
    def res(self) -> float:
        return self.transform.a

    @classmethod
    def for_aoi(cls, aoi: AOI, res: float, max_pixels: int = 4096) -> Grid:
        crs = aoi.utm_crs()
        w, s, e, n = transform_bounds("EPSG:4326", crs, *aoi.bbox, densify_pts=21)
        res = max(res, (e - w) / max_pixels, (n - s) / max_pixels)
        width, height = max(1, math.ceil((e - w) / res)), max(1, math.ceil((n - s) / res))
        return cls(crs, from_origin(w, n, res, res), width, height)

    @classmethod
    def for_tile(cls, z: int, x: int, y: int, size: int = 256) -> Grid:
        w, s, e, n = mercator_tile_bounds(x, y, z)
        return cls(WEB_MERCATOR, from_origin(w, n, (e - w) / size, (n - s) / size), size, size)


def resolve_href(href: str, render: Render) -> str:
    return pc_sign(href, render.sign) if render.sign else href


def read_href(
    href: str,
    grid: Grid,
    bands: list[int],
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Read ``bands`` of one raster onto ``grid``. Returns float32 (bands, h, w); NaN = no data."""
    out = np.full((len(bands), grid.height, grid.width), np.nan, dtype=np.float32)
    with rasterio.open(href) as src:
        sb = transform_bounds(grid.crs, src.crs, *grid.bounds, densify_pts=21)
        win = from_bounds(*sb, transform=src.transform)
        c0, r0 = max(0, math.floor(win.col_off) - 2), max(0, math.floor(win.row_off) - 2)
        c1 = min(src.width, math.ceil(win.col_off + win.width) + 2)
        r1 = min(src.height, math.ceil(win.row_off + win.height) + 2)
        if c1 <= c0 or r1 <= r0:
            return out
        win = Window(c0, r0, c1 - c0, r1 - r0)
        # Read no finer than ~2x the target resolution so GDAL serves it from overviews.
        dst_res_in_src = (sb[2] - sb[0]) / grid.width
        factor = max(1.0, dst_res_in_src / abs(src.res[0]) / 2)
        ow, oh = max(1, round(win.width / factor)), max(1, round(win.height / factor))
        data = src.read(bands, window=win, out_shape=(len(bands), oh, ow), masked=True, resampling=Resampling.average)
        arr = data.astype(np.float32).filled(np.nan)
        rt = src.window_transform(win) @ Affine.scale(win.width / ow, win.height / oh)
        for i in range(len(bands)):
            reproject(
                arr[i],
                out[i],
                src_transform=rt,
                src_crs=src.crs,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=resampling,
            )
    return out


def read_render(render: Render, grid: Grid) -> np.ndarray:
    """Raw values for a scene on ``grid`` (float32, NaN = no data), mosaicking multi-file scenes."""
    if render.kind == "reflectance":
        # hrefs are bands of one product
        layers = [read_href(resolve_href(h, render), grid, [1])[0] for h in render.hrefs]
        return np.stack(layers)
    bands = render.bands if render.kind == "rgb8" else [1]
    out: np.ndarray | None = None
    for h in render.hrefs:  # hrefs are footprints to mosaic, first valid pixel wins
        try:
            a = read_href(resolve_href(h, render), grid, bands)
        except rasterio.errors.RasterioIOError:
            continue
        if out is None:
            out = a
        else:
            hole = np.isnan(out[0])
            out[:, hole] = a[:, hole]
        if not np.isnan(out[0]).any():
            break
    if out is None:
        raise NoData("no readable asset")
    return out


@functools.lru_cache(maxsize=256)
def _sar_stats(href_key: str, href: str, power: bool) -> tuple[float, float]:
    """2nd/98th percentile in dB from the smallest overview (one cheap read per file)."""
    with rasterio.open(href) as src:
        ovs = src.overviews(1)
        f = ovs[-1] if ovs else max(1, max(src.width, src.height) // 1024)
        a = src.read(1, out_shape=(max(1, src.height // f), max(1, src.width // f)), masked=True)
    v = a.compressed().astype(np.float64)
    v = v[v > 0]
    if v.size == 0:
        return (0.0, 1.0)
    db = (10 if power else 20) * np.log10(v)
    lo, hi = np.percentile(db, [2, 99.5])
    return float(lo), float(hi)


def to_db(values: np.ndarray, power: bool) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(values > 0, values, np.nan)
        return (10 if power else 20) * np.log10(v)


def colorize(render: Render, raw: np.ndarray, auto_contrast: bool = False) -> np.ndarray:
    """Raw values -> RGBA uint8 (h, w, 4)."""
    valid = ~np.isnan(raw).all(axis=0)
    if render.kind == "rgb8":
        rgb = raw[:3]
        if rgb.shape[0] < 3:
            rgb = np.repeat(rgb[:1], 3, axis=0)
    elif render.kind == "reflectance":
        refl = raw * render.scale + render.offset
        lo, hi = render.vmin or 0.0, render.vmax or 0.3
        rgb = np.clip((refl - lo) / (hi - lo), 0, 1) ** (1 / 1.4) * 255
    elif render.kind == "sar":
        db = to_db(raw[0], render.power)
        if render.vmin is not None and render.vmax is not None:
            lo, hi = render.vmin, render.vmax
        else:
            h = render.hrefs[0]
            lo, hi = _sar_stats(h.split("?")[0], resolve_href(h, render), render.power)
        g = np.clip((db - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        rgb = np.stack([g, g, g])
    else:
        raise ValueError(f"cannot render kind {render.kind!r} locally")
    rgb = np.nan_to_num(rgb, nan=0.0)
    if auto_contrast and valid.any():
        rgb = _stretch(rgb, valid)
    rgba = np.zeros((raw.shape[1], raw.shape[2], 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb, 0, 255).transpose(1, 2, 0).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 255, 0)
    return rgba


def _stretch(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.empty_like(rgb)
    for i in range(rgb.shape[0]):
        lo, hi = np.percentile(rgb[i][valid], [1, 99])
        out[i] = (rgb[i] - lo) / max(hi - lo, 1e-6) * 255
    return out


# --------------------------------------------------------------------------- tiles


def render_tile(render: Render, z: int, x: int, y: int, size: int = 256) -> bytes:
    grid = Grid.for_tile(z, x, y, size)
    try:
        raw = read_render(render, grid)
    except NoData:
        return transparent_png(size)
    return png_bytes(colorize(render, raw))


@functools.lru_cache(maxsize=4)
def transparent_png(size: int = 256) -> bytes:
    return png_bytes(np.zeros((size, size, 4), dtype=np.uint8))


def png_bytes(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=False, compress_level=3)
    return buf.getvalue()


# --------------------------------------------------------------------------- chips


@dataclass
class Chip:
    scene: Scene
    grid: Grid
    rgba: np.ndarray

    @property
    def valid_fraction(self) -> float:
        return float((self.rgba[..., 3] > 0).mean())

    def caption(self) -> str:
        s = self.scene
        return f"{s.platform.upper()}  {s.datetime:%Y-%m-%d %H:%M}Z  {s.gsd:g} m  |  {s.attribution}"

    def image(self, label: bool = True) -> Image.Image:
        img = Image.fromarray(self.rgba, "RGBA")
        if label:
            img = draw_caption(img, self.caption())
        return img

    def save_png(self, path: str | Path, label: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image(label).save(path)
        return path

    def save_geotiff(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=self.grid.width,
            height=self.grid.height,
            count=4,
            dtype="uint8",
            crs=self.grid.crs,
            transform=self.grid.transform,
            compress="deflate",
            tiled=True,
            photometric="RGB",
        ) as dst:
            dst.write(self.rgba.transpose(2, 0, 1))
            dst.colorinterp = [
                rasterio.enums.ColorInterp.red,
                rasterio.enums.ColorInterp.green,
                rasterio.enums.ColorInterp.blue,
                rasterio.enums.ColorInterp.alpha,
            ]
            dst.update_tags(
                source=self.scene.source,
                scene=self.scene.id,
                datetime=self.scene.datetime.isoformat(),
                attribution=self.scene.attribution,
                license=self.scene.license,
            )
        return path


def chip(
    scene: Scene,
    aoi: AOI,
    res: float | None = None,
    max_pixels: int = 4096,
    auto_contrast: bool = False,
) -> Chip:
    """Render ``scene`` over ``aoi`` on a north-up UTM grid at native (or given) resolution."""
    if scene.render.kind == "xyz":
        raise NoData(f"{scene.source} is a view-only basemap; open it with `oracle serve` (or {scene.item_url})")
    grid = Grid.for_aoi(aoi, res or scene.gsd, max_pixels)
    raw = read_render(scene.render, grid)
    rgba = colorize(scene.render, raw, auto_contrast=auto_contrast)
    if not rgba[..., 3].any():
        raise NoData(f"{scene.id} has no pixels inside the AOI")
    return Chip(scene, grid, rgba)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_caption(img: Image.Image, text: str) -> Image.Image:
    img = img.convert("RGBA")
    size = max(11, img.width // 70)
    font = _font(size)
    pad = size // 2
    d = ImageDraw.Draw(img)
    tb = d.textbbox((0, 0), text, font=font)
    h = tb[3] - tb[1] + 2 * pad
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle([0, img.height - h, img.width, img.height], fill=(0, 0, 0, 150))
    img = Image.alpha_composite(img, overlay)
    ImageDraw.Draw(img).text((pad, img.height - h + pad - tb[1]), text, font=font, fill=(255, 255, 255, 255))
    return img
