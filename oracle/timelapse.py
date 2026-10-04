"""Animated time series of an AOI: one frame per period, least-cloudy scene wins."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image

from .geo import AOI, bbox_overlap_fraction
from .http import log
from .imagery import Grid, NoData, colorize, draw_caption, read_render
from .models import Scene
from .sources import get

PERIODS = ("day", "week", "month", "quarter", "year")


def period_key(dt: datetime, every: str) -> str:
    if every == "day":
        return dt.strftime("%Y-%m-%d")
    if every == "week":
        y, w, _ = dt.isocalendar()
        return f"{y}-W{w:02d}"
    if every == "month":
        return dt.strftime("%Y-%m")
    if every == "quarter":
        return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
    if every == "year":
        return str(dt.year)
    raise ValueError(f"every must be one of {PERIODS}")


def pick_frames(scenes: list[Scene], every: str, min_coverage: float = 0.9) -> list[Scene]:
    best: dict[str, Scene] = {}
    for s in scenes:
        if s.extra.get("coverage", 1.0) < min_coverage:
            continue
        k = period_key(s.datetime, every)
        cur = best.get(k)
        if cur is None or ((s.cloud_cover or 0), s.gsd) < ((cur.cloud_cover or 0), cur.gsd):
            best[k] = s
    return [best[k] for k in sorted(best)]


def timelapse(
    aoi: AOI,
    start: datetime,
    end: datetime,
    out: str | Path,
    *,
    source: str = "sentinel-2",
    every: str = "month",
    max_cloud: float | None = 30,
    max_pixels: int = 1024,
    ms_per_frame: int = 700,
) -> tuple[Path, list[Scene]]:
    src = get(source)
    scenes = src.search(aoi, start, end, max_cloud=max_cloud, limit=1000)
    for s in scenes:
        s.extra["coverage"] = bbox_overlap_fraction(aoi.bbox, s.bbox)
    chosen = pick_frames(scenes, every)
    if not chosen:
        raise NoData("no scenes for that AOI / period")
    grid = Grid.for_aoi(aoi, max(s.gsd for s in chosen), max_pixels)
    frames, used = [], []
    for s in chosen:
        try:
            rgba = colorize(s.render, read_render(s.render, grid))
        except NoData:
            continue
        valid = rgba[..., 3] > 0
        if valid.mean() < 0.8:
            continue
        if s.sensor == "optical" and (rgba[..., :3][valid].min(axis=-1) > 200).mean() > 0.5:
            continue  # mostly cloud inside the AOI, whatever the scene-wide cloud % says
        img = Image.fromarray(rgba, "RGBA")
        frames.append(draw_caption(img, f"{s.datetime:%Y-%m-%d}  {s.platform.upper()}  {s.gsd:g} m").convert("RGB"))
        used.append(s)
        log(f"  frame {len(frames)}: {s.date} {s.id}")
    if not frames:
        raise NoData("every candidate frame was cloudy or incomplete")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=ms_per_frame, loop=0, optimize=True)
    return out, used


__all__ = ["timelapse", "pick_frames", "period_key", "PERIODS"]
