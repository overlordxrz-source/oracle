"""``oracle`` command line."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import __version__
from .geo import AOI, parse_aoi
from .models import parse_dt
from .search import SORTS, search
from .sources import SOURCES

ARCHIVE_START = datetime(2014, 1, 1, tzinfo=timezone.utc)


def _dates(args, default_days: int | None = None) -> tuple[datetime, datetime]:
    end = parse_dt(args.end + "T23:59:59Z") if args.end else datetime.now(timezone.utc)
    if args.date:
        d = parse_dt(args.date + "T00:00:00Z")
        return d, d + timedelta(days=1) - timedelta(seconds=1)
    if args.start:
        return parse_dt(args.start + "T00:00:00Z"), end
    days = args.days or default_days
    return (end - timedelta(days=days) if days else ARCHIVE_START), end


def _aoi(args) -> AOI:
    return parse_aoi(args.where, args.radius)


def _sources(args) -> list[str] | None:
    return [s.strip() for s in args.sources.split(",")] if args.sources else None


def _fmt_gsd(g: float) -> str:
    return f"{g * 100:.0f} cm" if g < 1 else f"{g:g} m"


def _table(scenes, n: int) -> None:
    print(f"{'#':>3}  {'source':<11} {'date (UTC)':<16} {'res':>6} {'cloud':>5} {'cover':>5}  {'platform':<12} id")
    for i, s in enumerate(scenes[:n]):
        cloud = "" if s.cloud_cover is None else f"{s.cloud_cover:.0f}%"
        cov = f"{s.extra.get('coverage', 1) * 100:.0f}%"
        when = f"{s.datetime:%Y-%m-%d %H:%M}"
        print(f"{i:>3}  {s.source:<11} {when} {_fmt_gsd(s.gsd):>6} {cloud:>5} {cov:>5}  {s.platform[:12]:<12} {s.id}")


# --------------------------------------------------------------------------- commands


def cmd_sources(args) -> None:
    for s in sorted(SOURCES.values(), key=lambda s: s.info.best_gsd):
        i = s.info
        print(f"{i.key:<11} {i.resolution:<34} {i.sensor:<7} {i.coverage}")
        print(f"{'':<11} {i.revisit}; {i.license}")


def cmd_search(args) -> None:
    aoi = _aoi(args)
    t0, t1 = _dates(args)
    res = search(
        aoi, t0, t1, sources=_sources(args), max_cloud=args.max_cloud, limit=args.limit, sort=args.sort, sensor=args.sensor
    )
    if args.json:
        json.dump({"aoi": aoi.bbox, "scenes": [s.to_dict() for s in res.scenes], "errors": res.errors}, sys.stdout, indent=1)
        print()
        return
    w, h = aoi.size_km()
    print(f"AOI {aoi.name or aoi.bbox}  ({w:.1f} x {h:.1f} km)  {t0:%Y-%m-%d} .. {t1:%Y-%m-%d}")
    print("found: " + ", ".join(f"{k} {n}" for k, n in res.counts.items()))
    for k, e in res.errors.items():
        print(f"  ! {k}: {e}", file=sys.stderr)
    _table(res.scenes, args.show)


def cmd_fetch(args) -> None:
    from .imagery import NoData, chip

    aoi = _aoi(args)
    t0, t1 = _dates(args)
    res = search(aoi, t0, t1, sources=_sources(args), max_cloud=args.max_cloud, limit=args.limit, sort=args.sort)
    candidates = [s for s in res.scenes if s.render.kind != "xyz"]
    if args.scene:
        candidates = [s for s in candidates if s.id == args.scene]
    if not candidates:
        sys.exit("no downloadable scenes matched (wayback is view-only: use `oracle serve`)")
    out = Path(args.out)
    done = 0
    for s in candidates:
        if done >= args.count:
            break
        try:
            c = chip(s, aoi, res=args.res, max_pixels=args.max_pixels, auto_contrast=args.enhance)
        except NoData as exc:
            print(f"  skip {s.id}: {exc}", file=sys.stderr)
            continue
        stem = f"{s.date}_{s.source}_{s.id[:60]}"
        written = []
        if "png" in args.format:
            written.append(c.save_png(out / f"{stem}.png", label=not args.no_label))
        if "tif" in args.format:
            written.append(c.save_geotiff(out / f"{stem}.tif"))
        print(
            f"{s.source:<11} {s.datetime:%Y-%m-%d %H:%M}Z {_fmt_gsd(s.gsd):>6}  {c.grid.width}x{c.grid.height}px  -> "
            + ", ".join(str(p) for p in written)
        )
        done += 1


def cmd_ships(args) -> None:
    from .detect import detect_ships
    from .imagery import NoData

    aoi = _aoi(args)
    t0, t1 = _dates(args, default_days=30)
    srcs = _sources(args) or ["sentinel-2"]
    res = search(
        aoi,
        t0,
        t1,
        sources=srcs,
        max_cloud=args.max_cloud,
        limit=200,
        sort="cloud" if not args.newest else "date",
        min_coverage=0.5,
    )
    scenes = [s for s in res.scenes if s.source == "sentinel-2" or s.sensor == "sar"]
    if args.scene:
        scenes = [s for s in scenes if s.id == args.scene]
    if not scenes:
        sys.exit("no Sentinel-2/SAR scenes for that place and time (try --days 60 or --sources sentinel-1)")
    for s in scenes[: args.count]:
        print(f"{s.source} {s.datetime:%Y-%m-%d %H:%M}Z  {s.id}  cloud={s.cloud_cover}", file=sys.stderr)
        try:
            det = detect_ships(s, aoi, k=args.k, min_length=args.min_length, include_shore=args.include_shore)
        except (ValueError, NoData) as exc:
            print(f"  ! {exc}", file=sys.stderr)
            continue
        for wmsg in det.warnings:
            print(f"  ! {wmsg}", file=sys.stderr)
        big = [d for d in det.detections if d.length_m >= 250]
        print(f"  {len(det.detections)} vessels >= {args.min_length:g} m, {len(big)} >= 250 m")
        for d in det.detections[: args.top]:
            extra = f"  {d.nearby_medium_vessels} 100-220 m hulls within 15 km" if d.nearby_medium_vessels else ""
            print(
                f"    {d.length_m:6.0f} m x {d.width_m:4.0f} m  axis {d.heading_deg:5.1f} deg  {d.lat:9.5f},{d.lon:10.5f}  "
                f"{d.contrast:5.1f} sigma{extra}"
            )
        paths = det.save(args.out, f"{s.date}_{s.source}_{s.id[:40]}_ships")
        print("  -> " + ", ".join(str(p) for p in paths.values()))


def cmd_history(args) -> None:
    from .sources.wayback import versions

    lat, lon = _aoi(args).center
    vs = versions(lat, lon)
    print(f"Esri Wayback captures at {lat:.5f},{lon:.5f} (newest first):")
    print(f"  {'captured':<11} {'res':>6}  {'sensor':<10} {'provider':<28} release")
    for v in vs:
        res = _fmt_gsd(v["resolution"]) if v.get("resolution") else "?"
        print(
            f"  {v.get('captured') or '?':<11} {res:>6}  {(v.get('sensor') or '?'):<10} {(v.get('provider') or ''):<28} "
            f"{v['release']} ({v['date']})"
        )
    print("view: `oracle serve` -> search with source 'wayback', or https://livingatlas.arcgis.com/wayback/")


def cmd_timelapse(args) -> None:
    from .timelapse import timelapse

    aoi = _aoi(args)
    t0, t1 = _dates(args, default_days=365)
    out, used = timelapse(
        aoi,
        t0,
        t1,
        args.out,
        source=args.source,
        every=args.every,
        max_cloud=args.max_cloud,
        max_pixels=args.max_pixels,
        ms_per_frame=args.ms,
    )
    print(f"{len(used)} frames -> {out}")


def cmd_index(args) -> None:
    from .sources import capella, maxar, umbra

    idx = {"maxar": maxar.INDEX, "capella": capella.INDEX, "umbra": umbra.INDEX}
    for k in _sources(args) or list(idx):
        if k not in idx:
            sys.exit(f"{k} has a live search API and needs no index")
        i = idx[k]
        if args.refresh or i.age_days() is None:
            i.build()
        print(f"{k:<8} {len(i.records()):>6} scenes  ({i.age_days():.1f} days old)  {i.path}")


def cmd_serve(args) -> None:
    from .server import run

    print(f"Oracle on http://{args.host}:{args.port}")
    run(args.host, args.port)


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="oracle", description="Free high-resolution satellite imagery: search, view, analyse.")
    p.add_argument("--version", action="version", version=f"oracle {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def where(sp, radius=2.0):
        sp.add_argument("where", help="'lat,lon', 'west,south,east,north', or a place name")
        sp.add_argument("-r", "--radius", type=float, default=radius, help=f"km around a point (default {radius})")

    def when(sp):
        sp.add_argument("--start", help="YYYY-MM-DD")
        sp.add_argument("--end", help="YYYY-MM-DD (default: now)")
        sp.add_argument("--date", help="a single day, YYYY-MM-DD")
        sp.add_argument("--days", type=int, help="last N days")

    def srcs(sp):
        sp.add_argument("-s", "--sources", help=f"comma list from: {', '.join(SOURCES)}")
        sp.add_argument("--max-cloud", type=float, help="max scene cloud cover %%")

    sp = sub.add_parser("sources", help="list imagery sources")
    sp.set_defaults(fn=cmd_sources)

    sp = sub.add_parser("search", help="find imagery for a place and time")
    where(sp), when(sp), srcs(sp)
    sp.add_argument("--sort", choices=SORTS, default="best")
    sp.add_argument("--sensor", choices=("optical", "sar"))
    sp.add_argument("--limit", type=int, default=30, help="max results per source")
    sp.add_argument("--show", type=int, default=40, help="rows to print")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_search)

    sp = sub.add_parser("fetch", help="download the best image(s) as PNG / GeoTIFF")
    where(sp), when(sp), srcs(sp)
    sp.add_argument("--sort", choices=SORTS, default="best")
    sp.add_argument("--scene", help="exact scene id from `oracle search`")
    sp.add_argument("-n", "--count", type=int, default=1)
    sp.add_argument("--limit", type=int, default=30)
    sp.add_argument("--res", type=float, help="output metres/pixel (default: native)")
    sp.add_argument("--max-pixels", type=int, default=8192)
    sp.add_argument("--format", default="png,tif")
    sp.add_argument("--enhance", action="store_true", help="per-image contrast stretch")
    sp.add_argument("--no-label", action="store_true")
    sp.add_argument("-o", "--out", default="oracle-out")
    sp.set_defaults(fn=cmd_fetch)

    sp = sub.add_parser("ships", help="detect vessels (Sentinel-2 or SAR)")
    where(sp, radius=10.0), when(sp), srcs(sp)
    sp.add_argument("--scene")
    sp.add_argument("-n", "--count", type=int, default=1, help="how many scenes to process")
    sp.add_argument("--newest", action="store_true", help="prefer newest over least cloudy")
    sp.add_argument("-k", type=float, help="threshold in local std devs (default 6 optical / 5 SAR)")
    sp.add_argument("--min-length", type=float, default=25.0)
    sp.add_argument("--include-shore", action="store_true", help="keep detections touching land (piers too)")
    sp.add_argument("--top", type=int, default=15, help="rows to print")
    sp.add_argument("-o", "--out", default="oracle-out")
    sp.set_defaults(fn=cmd_ships)

    sp = sub.add_parser("history", help="every sub-metre capture in the Esri Wayback archive at a point")
    where(sp)
    sp.set_defaults(fn=cmd_history)

    sp = sub.add_parser("timelapse", help="animated GIF over time")
    where(sp), when(sp)
    sp.add_argument("--source", default="sentinel-2", choices=[k for k in SOURCES if k != "wayback"])
    sp.add_argument("--every", default="month", choices=("day", "week", "month", "quarter", "year"))
    sp.add_argument("--max-cloud", type=float, default=30)
    sp.add_argument("--max-pixels", type=int, default=1024)
    sp.add_argument("--ms", type=int, default=700, help="milliseconds per frame")
    sp.add_argument("-o", "--out", default="oracle-out/timelapse.gif")
    sp.set_defaults(fn=cmd_timelapse)

    sp = sub.add_parser("index", help="build/refresh local indexes of the static catalogs")
    sp.add_argument("-s", "--sources", help="maxar,capella,umbra")
    sp.add_argument("--refresh", action="store_true")
    sp.set_defaults(fn=cmd_index)

    from .cli_intel import register

    register(sub)

    sp = sub.add_parser("serve", help="start the map web app")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except (ValueError, KeyError) as exc:
        sys.exit(f"error: {exc}")
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
