"""CLI for the intelligence layer: detect, sites, sweep, tracks, events, brief, ais, db."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .geo import parse_aoi
from .models import parse_dt


def _store():
    from .store import Store

    return Store()


def _since(text: str | None, default_days: int = 7) -> datetime:
    now = datetime.now(timezone.utc)
    if not text:
        return now - timedelta(days=default_days)
    t = text.strip().lower()
    if t[:-1].isdigit() and t[-1] in "dhw":
        n = int(t[:-1])
        return now - {"d": timedelta(days=n), "h": timedelta(hours=n), "w": timedelta(weeks=n)}[t[-1]]
    return parse_dt(text + ("T00:00:00Z" if len(text) == 10 else ""))


# --------------------------------------------------------------------------- detect


def cmd_detect(args) -> None:
    from .objdet import detect_objects
    from .search import search
    from .sources.local import local_scene

    if args.file:
        scene, aoi = local_scene(args.file, when=args.date)
        if args.where:
            aoi = parse_aoi(args.where, args.radius)
    else:
        if not args.where:
            sys.exit("give WHERE or --file")
        aoi = parse_aoi(args.where, args.radius)
        end = parse_dt(args.end + "T23:59:59Z") if args.end else datetime.now(timezone.utc)
        start = parse_dt(args.start + "T00:00:00Z") if args.start else datetime(2014, 1, 1, tzinfo=timezone.utc)
        if args.date:
            start, end = parse_dt(args.date + "T00:00:00Z"), parse_dt(args.date + "T23:59:59Z")
        srcs = [s.strip() for s in args.sources.split(",")] if args.sources else ["maxar", "naip"]
        res = search(aoi, start, end, sources=srcs, limit=50, sort="date", min_coverage=0.5)
        scenes = [s for s in res.scenes if s.sensor == "optical" and s.gsd <= 1.0 and s.render.kind != "xyz"]
        if args.scene:
            scenes = [s for s in scenes if s.id == args.scene]
        if not scenes:
            sys.exit("no sub-metre optical scene there (YOLO needs maxar/naip or --file); try `oracle search ... -s maxar,naip`")
        scene = scenes[0]
    print(f"{scene.source} {scene.datetime:%Y-%m-%d} {scene.gsd:g} m  {scene.id}", file=sys.stderr)
    t0 = time.time()
    prompts = [p.strip() for p in args.prompt.split(",")] if args.prompt else None
    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
    obs = detect_objects(
        scene,
        aoi,
        model=args.model,
        gsd=args.gsd,
        conf=args.conf,
        classes=classes,
        prompts=prompts,
        vehicles=not args.no_vehicles,
    )
    from collections import Counter

    print(
        f"{len(obs)} objects in {time.time() - t0:.0f}s: "
        + ", ".join(f"{k} {v}" for k, v in Counter(o.cls for o in obs).most_common())
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{scene.date}_{scene.source}_objects"
    from .observations import feature_collection

    (out / f"{stem}.geojson").write_text(json.dumps(feature_collection(obs, scene=scene.id), indent=1))
    _draw_objects(scene, aoi, obs, out / f"{stem}.png")
    print(f"  -> {out / stem}.geojson, {out / stem}.png")
    if args.site:
        st = _store()
        st.add_scene(scene)
        n = st.add_observations(obs, args.site)
        st.record_run(args.site, scene.id, "objects", n)
        print(f"  stored {n} new observations under site {args.site!r}")


COLORS = {
    "aircraft": (255, 70, 70),
    "helicopter": (255, 140, 0),
    "vessel": (0, 200, 255),
    "vehicle": (255, 230, 0),
    "large-vehicle": (180, 255, 0),
    "storage-tank": (200, 120, 255),
}


def _draw_objects(scene, aoi, obs, path: Path) -> None:
    from PIL import ImageDraw
    from pyproj import Transformer

    from .imagery import chip, draw_caption

    c = chip(scene, aoi, max_pixels=4096)
    img = c.image(label=False).convert("RGBA")
    d = ImageDraw.Draw(img)
    inv = ~c.grid.transform
    tr = Transformer.from_crs(4326, c.grid.crs, always_xy=True)
    for o in obs:
        pts = [inv @ tr.transform(lon, lat) for lon, lat in (o.polygon or [[o.lon, o.lat]])]
        col = COLORS.get(o.cls, (255, 255, 255))
        if len(pts) > 2:
            d.polygon(pts, outline=col + (255,), width=max(1, img.width // 1500))
    img = draw_caption(img, f"{c.caption()}  |  {len(obs)} objects (YOLO)")
    img.save(path)


# --------------------------------------------------------------------------- sites / sweep


def cmd_sites(args) -> None:
    from .pipeline import Site, add_preset
    from .presets import PRESETS

    st = _store()
    if args.action == "add":
        if not (args.name and args.where):
            sys.exit("usage: oracle sites add NAME WHERE [--kind maritime|naval|airbase]")
        aoi = parse_aoi(args.where, args.radius)
        s = Site(args.name, aoi.bbox, args.kind, detectors=_csv(args.detectors), sources=_csv(args.sources))
        s.save(st)
        print(f"site {s.name}: {s.kind}, detectors {s.detectors}, sources {s.sources}")
    elif args.action == "preset":
        names = list(PRESETS) if args.name in (None, "all") else [args.name]
        for p in names:
            added = add_preset(st, p)
            print(f"{p}: {len(added)} sites")
    elif args.action == "rm":
        print("removed" if st.delete_site(args.name) else "no such site")
    else:
        for s in st.sites():
            print(
                f"{s['name']:<32} {s.get('kind') or '':<9} {','.join(s.get('detectors') or []):<14} "
                f"{', '.join(f'{v:.3f}' for v in s['bbox'])}"
            )
        print(f"({len(st.sites())} sites; presets: {', '.join(PRESETS)})")


def _csv(text: str | None) -> list[str]:
    return [x.strip() for x in text.split(",")] if text else []


def cmd_sweep(args) -> None:
    from . import alerts
    from .brief import brief
    from .pipeline import Site, sweep

    st = _store()
    rows = st.sites()
    if args.sites:
        wanted = set(_csv(args.sites))
        rows = [r for r in rows if r["name"] in wanted]
    if not rows:
        sys.exit("no sites: add some with `oracle sites add ...` or `oracle sites preset chokepoints`")
    sites = [Site.from_row(r) for r in rows]
    while True:
        started = datetime.now(timezone.utc)
        print(f"[sweep] {len(sites)} site(s), lookback {args.lookback} d, at {started:%Y-%m-%d %H:%M}Z", file=sys.stderr)
        results = sweep(st, sites, lookback_days=args.lookback, max_scenes=args.max_scenes, workers=args.workers)
        for r in results:
            errs = f"  ! {len(r.get('errors', []))} error(s)" if r.get("errors") else ""
            print(
                f"  {r['site']:<32} +{r.get('new_scenes', 0)} images  +{r.get('new_observations', 0)} objects  "
                f"{r.get('tracks', 0)} tracks  +{r.get('new_events', 0)} new events{errs}"
            )
        fresh = [e for e in st.events(limit=2000) if e["created"] >= started.isoformat()]
        if fresh:
            print(f"  {len(fresh)} new event(s); top: " + "; ".join(e["title"] for e in fresh[:3]))
            alerts.notify(fresh, args.webhook)
        if args.brief:
            md, engine = brief(st, since=started - timedelta(days=args.lookback))
            Path(args.brief).write_text(md)
            print(f"  brief ({engine}) -> {args.brief}")
        if not args.loop:
            break
        time.sleep(args.loop)


# --------------------------------------------------------------------------- tracks / events / brief


def cmd_tracks(args) -> None:
    st = _store()
    rows = [t for t in st.tracks(site=args.site, status=args.status) if t["n_obs"] >= args.min_obs]
    if args.cls:
        rows = [t for t in rows if t["cls"] == args.cls]
    rows.sort(key=lambda t: (-t["n_obs"], -(t["length_m"] or 0)))
    print(f"{'track':<13} {'class':<8} {'len':>5} {'seen':>4} {'first':<10} {'last':<10} {'link p':>6} {'status':<9} site")
    for t in rows[: args.limit]:
        p = f"{t['mean_link_prob']:.0%}" if t["mean_link_prob"] is not None else "-"
        print(
            f"{t['id']:<13} {t['cls']:<8} {t['length_m'] or 0:>5.0f} {t['n_obs']:>4} {t['first_seen'][:10]:<10} "
            f"{t['last_seen'][:10]:<10} {p:>6} {t['status']:<9} {t['site']}"
        )
    print(f"({len(rows)} tracks)")


def cmd_track(args) -> None:
    t = _store().track(args.id)
    if not t:
        sys.exit("no such track")
    print(f"{t['id']}  {t['cls']}  ~{t['length_m']} m  {t['status']}  site {t['site']}")
    print(f"  {t['n_obs']} sightings {t['first_seen'][:16]} .. {t['last_seen'][:16]}, sources {t['attrs'].get('sources')}")
    for o in t["observations"]:
        p = o["attrs"].get("link_prob")
        extra = ""
        if o["attrs"].get("underway"):
            extra = f" underway course {o['course_deg']}" if o["course_deg"] is not None else " underway"
        if o["attrs"].get("ais"):
            a = o["attrs"]["ais"]
            extra += f" AIS {a['mmsi']} {a['name']}"
        if o["attrs"].get("dark"):
            extra += " DARK (no AIS)"
        print(
            f"  {o['time'][:16]}Z {o['source']:<11} {o['lat']:.5f},{o['lon']:.5f}  {o['length_m'] or 0:.0f} m  "
            f"same-object p={p if p is not None else '-'}{extra}"
        )
    if t["alternatives"]:
        print("  possibly the same object as:")
        for a in sorted(t["alternatives"], key=lambda a: -a["prob"])[:10]:
            print(f"    obs {a['obs_id']} could belong to track {a['track_id']} (p={a['prob']:.0%})")


def cmd_events(args) -> None:
    st = _store()
    for e in st.events(since=_since(args.since), site=args.site, limit=args.limit):
        if e["severity"] < args.min_severity:
            continue
        d = e["detail"]
        where = f"  {d['lat']:.4f},{d['lon']:.4f}" if "lat" in d else ""
        print(f"{e['severity']:.2f}  {e['time'][:16]}Z  {e['kind']:<20} {e['title']}{where}")


def cmd_brief(args) -> None:
    from .brief import brief

    md, engine = brief(
        _store(),
        since=_since(args.since),
        sites=_csv(args.sites) or None,
        use_llm=False if args.no_llm else (True if args.llm else None),
    )
    if args.out:
        Path(args.out).write_text(md)
        print(f"brief ({engine}) -> {args.out}", file=sys.stderr)
    else:
        print(md)


def cmd_ais(args) -> None:
    from . import ais
    from .pipeline import Site, refresh_site

    st = _store()
    if args.action == "load":
        bbox = tuple(map(float, args.bbox.split(","))) if args.bbox else None
        n = ais.load_csv(
            st,
            args.file,
            bbox=bbox,
            start=parse_dt(args.start + "T00:00:00Z") if args.start else None,
            end=parse_dt(args.end + "T23:59:59Z") if args.end else None,
        )
        print(f"{n} AIS positions loaded; re-matching sites...")
        for r in st.sites():
            print(f"  {r['name']}: {refresh_site(st, Site.from_row(r))}")
    else:
        print(st.stats()["ais_points"], "AIS positions in the database")


def cmd_db(args) -> None:
    st = _store()
    print(f"{st.path}")
    print(json.dumps(st.stats(), indent=1))


def register(sub) -> None:
    def where(sp, radius):
        sp.add_argument("-r", "--radius", type=float, default=radius, help=f"km around a point (default {radius})")

    sp = sub.add_parser("detect", help="YOLO object detection on sub-metre imagery (or your own GeoTIFF)")
    sp.add_argument("where", nargs="?", help="lat,lon | bbox | place (optional with --file)")
    where(sp, 0.5)
    sp.add_argument("--file", help="local/remote georeferenced GeoTIFF to analyse instead of searching")
    sp.add_argument("-s", "--sources", help="default maxar,naip")
    sp.add_argument("--start")
    sp.add_argument("--end")
    sp.add_argument("--date", help="YYYY-MM-DD (also sets the time of a --file without one)")
    sp.add_argument("--scene")
    sp.add_argument("--model", default="yolo11l-obb.pt", help="any Ultralytics weights (yolo11s-obb.pt is ~3x faster)")
    sp.add_argument("--gsd", type=float, help="force one pass at this m/px (default: multi-scale)")
    sp.add_argument("--conf", type=float, default=0.3)
    sp.add_argument("--classes", help="keep only these, e.g. aircraft,helicopter")
    sp.add_argument("--prompt", help="open-vocabulary (YOLO-World) prompts, e.g. 'fighter jet,tank,truck'")
    sp.add_argument("--no-vehicles", action="store_true", help="skip the slow 0.15 m car pass")
    sp.add_argument("--site", help="also store results in the object database under this site")
    sp.add_argument("-o", "--out", default="oracle-out")
    sp.set_defaults(fn=cmd_detect)

    sp = sub.add_parser("sites", help="places to monitor (add / list / rm / preset)")
    sp.add_argument("action", choices=("add", "list", "rm", "preset"), nargs="?", default="list")
    sp.add_argument("name", nargs="?", help="site name, or preset: chokepoints | naval-bases | airbases-us | all")
    sp.add_argument("where", nargs="?")
    where(sp, 5.0)
    sp.add_argument("--kind", default="maritime", choices=("maritime", "naval", "airbase", "ground"))
    sp.add_argument("--detectors", help="ships,objects (default from --kind)")
    sp.add_argument("-s", "--sources")
    sp.set_defaults(fn=cmd_sites)

    sp = sub.add_parser("sweep", help="process new imagery for all sites: detect, track, raise events")
    sp.add_argument("--sites", help="comma list of site names (default: all)")
    sp.add_argument("--lookback", type=int, default=30, help="days of imagery to consider")
    sp.add_argument("--max-scenes", type=int, default=8, help="new images per site per run")
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--loop", type=int, help="repeat every N seconds")
    sp.add_argument("--webhook", help="POST new events here (or set ORACLE_WEBHOOK)")
    sp.add_argument("--brief", help="also write a brief to this .md file after each run")
    sp.set_defaults(fn=cmd_sweep)

    sp = sub.add_parser("tracks", help="list tracked objects")
    sp.add_argument("--site")
    sp.add_argument("--status", choices=("active", "departed", "lost"))
    sp.add_argument("--cls")
    sp.add_argument("--min-obs", type=int, default=2)
    sp.add_argument("--limit", type=int, default=50)
    sp.set_defaults(fn=cmd_tracks)

    sp = sub.add_parser("track", help="one track: every sighting and same-object probabilities")
    sp.add_argument("id")
    sp.set_defaults(fn=cmd_track)

    sp = sub.add_parser("events", help="ranked events (arrivals, departures, anomalies, dark vessels...)")
    sp.add_argument("--since", default="7d", help="7d | 48h | 2w | YYYY-MM-DD")
    sp.add_argument("--site")
    sp.add_argument("--min-severity", type=float, default=0.0)
    sp.add_argument("--limit", type=int, default=50)
    sp.set_defaults(fn=cmd_events)

    sp = sub.add_parser("brief", help="intelligence brief (Claude if configured, else built-in)")
    sp.add_argument("--since", default="7d")
    sp.add_argument("--sites")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--llm", action="store_true", help="require Claude")
    g.add_argument("--no-llm", action="store_true", help="built-in report only")
    sp.add_argument("-o", "--out")
    sp.set_defaults(fn=cmd_brief)

    sp = sub.add_parser("ais", help="load AIS CSVs to match detections and flag dark vessels")
    sp.add_argument("action", choices=("load", "stats"))
    sp.add_argument("file", nargs="?")
    sp.add_argument("--bbox", help="west,south,east,north filter while loading")
    sp.add_argument("--start")
    sp.add_argument("--end")
    sp.set_defaults(fn=cmd_ais)

    sp = sub.add_parser("db", help="object database location and counts")
    sp.set_defaults(fn=cmd_db)
