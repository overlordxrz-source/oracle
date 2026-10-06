"""CLI for the intelligence layer: ask, detect, change, sites, sweep, tracks, events, brief, ais, passes, db, doctor."""

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


def cmd_change(args) -> None:
    from .change import KINDS, detect_change, to_events

    aoi = parse_aoi(args.where, args.radius)
    after = parse_dt(args.after + "T23:59:59Z") if args.after else None
    before = parse_dt(args.before + "T23:59:59Z") if args.before else None
    t0 = time.time()
    res = detect_change(
        aoi,
        args.source,
        after=after,
        before=before,
        n_before=args.baseline_images,
        baseline=args.baseline,
        k=args.k,
        min_area_m2=args.min_area,
    )
    s = res.summary()
    print(f"{s['method']}: {res.after.date} vs {', '.join(b['time'][:10] for b in s['before'])}")
    print(f"  {s['valid_fraction']:.0%} of the area comparable, {len(res.regions)} change regions in {time.time() - t0:.0f}s")
    for n in s["notes"]:
        print(f"  note: {n}")
    for k, v in s["by_kind"].items():
        print(f"  {k:<20} {v['regions']:>4} regions {v['area_m2'] / 1e4:>8.1f} ha   ({KINDS[k][0]})")
    for r in res.regions[: args.top]:
        vals = "  ".join(f"{m} {a[0]:g}->{a[1]:g}" for m, a in r.values.items())
        print(f"  {r.kind:<20} {r.area_m2 / 1e4:>6.2f} ha  conf {r.confidence:.0%}  {r.lat:.5f},{r.lon:.5f}  {vals}")
    paths = res.save(args.out)
    print("  -> " + ", ".join(str(p) for p in paths.values()))
    if args.site:
        st = _store()
        evs = to_events(res, args.site)
        print(f"  stored {st.add_events(evs)} change events under site {args.site!r}")


def cmd_ask(args) -> None:
    from .agent import investigate

    def show(step: dict) -> None:
        if step["type"] == "tool" and step.get("status") == "running":
            print(f"  -> {step['tool']} {json.dumps(step['input'])[:110]}", file=sys.stderr)
        elif step["type"] == "tool":
            mark = "!!" if step["status"] == "error" else "ok"
            ev = f" [{', '.join(step['evidence'][:4])}{'...' if len(step['evidence']) > 4 else ''}]" if step["evidence"] else ""
            print(f"     {mark} {step['ms'] / 1000:.1f}s {step['summary'][:150]}{ev}", file=sys.stderr)
        elif args.verbose or step["type"] == "note":
            print(f"  .. {step['summary'][:300]}", file=sys.stderr)

    use = False if args.no_llm else (True if args.llm else None)
    inv = investigate(args.question, use_llm=use, max_calls=args.max_calls, effort=args.effort, on_step=show, web=not args.no_web)
    print(f"\n[{inv.engine}] {inv.status}  (investigation {inv.id})\n", file=sys.stderr)
    print(inv.answer)
    ev = inv.toolbox.evidence
    p = inv.provenance
    if ev:
        print("\n---\nEvidence")
        for eid in p.get("cited") or []:
            if eid in ev:
                e = ev[eid]
                link = f"  {next(iter(e.links.values()))}" if e.links else ""
                print(f"  [{eid}] {e.kind:<11} {e.summary[:140]}{link}")
        print(
            f"  provenance: {len(p.get('cited', []))} cited "
            f"({', '.join(f'{v} {k}' for k, v in p.get('cited_by_kind', {}).items() if v)}), "
            f"{len(p.get('uncited', []))} uncited, unknown: {', '.join(p.get('unknown') or []) or 'none'}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(inv.to_dict(), indent=1, default=str))
        print(f"  -> {args.json}", file=sys.stderr)


def _latlon_list(items: list[str] | None) -> list[tuple[float, float]]:
    return [parse_aoi(x, 0.05).center for x in items or []]


def cmd_similar(args) -> None:
    from . import embeddings as E

    aoi = parse_aoi(args.where, args.radius)
    ex = _latlon_list(args.like)
    if not ex:
        sys.exit("give at least one --like 'lat,lon' (or a place) example")
    t0 = time.time()
    r = E.find_similar(aoi, ex, args.year, _latlon_list(args.unlike) or None, threshold=args.threshold)
    print(f"AlphaEarth {args.year} search, {r.stats['method']}, {r.grid.res:.0f} m grid, {time.time() - t0:.0f}s")
    print(f"  threshold {r.stats['threshold']}, background median {r.stats['background_median']}, {len(r.matches)} matches")
    for m in r.matches[: args.top]:
        print(f"  {m.score:5.2f}  {m.lat:.5f},{m.lon:.5f}  {m.area_m2 / 1e4:7.1f} ha")
    _save_embed(r, args.out, "similar")


def cmd_evolve(args) -> None:
    from . import embeddings as E

    aoi = parse_aoi(args.where, args.radius)
    t0 = time.time()
    r = E.semantic_change(aoi, args.year_from, args.year_to)
    st = r.stats
    print(
        f"AlphaEarth {args.year_from} -> {args.year_to}: {st['changed_area_km2']} km2 changed "
        f"({st['share_changed']:.1%}), {time.time() - t0:.0f}s"
    )
    for m in r.matches[: args.top]:
        print(f"  distance {m.score:4.2f}  {m.lat:.5f},{m.lon:.5f}  {m.area_m2 / 1e4:7.1f} ha")
    _save_embed(r, args.out, f"evolve_{args.year_from}_{args.year_to}")


def cmd_embed(args) -> None:
    from . import embeddings as E

    r = E.embedding_view(parse_aoi(args.where, args.radius), args.year, args.segments)
    _save_embed(r, args.out, f"embedding_{args.year}")


def _save_embed(r, out: str, stem: str) -> None:
    from .embeddings import ATTRIBUTION

    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.png").write_bytes(r.png())
    (d / f"{stem}.geojson").write_text(json.dumps(r.geojson(), indent=1))
    print(f"  -> {d / stem}.png (georeferenced corners in the .geojson)\n  {ATTRIBUTION}")


def cmd_enhance(args) -> None:
    from .search import search
    from .superres import enhance

    aoi = parse_aoi(args.where, args.radius)
    end = parse_dt(args.date + "T23:59:59Z") if args.date else datetime.now(timezone.utc)
    res = search(
        aoi,
        end - timedelta(days=60 if not args.date else 1),
        end,
        sources=["sentinel-2"],
        max_cloud=args.max_cloud,
        sort="date",
        min_coverage=0.9,
    )
    if not res.scenes:
        sys.exit("no clear Sentinel-2 image covering the area")
    t0 = time.time()
    r = enhance(res.scenes[0], aoi, args.variant)
    d = Path(args.out)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{r.scene.date}_superres_{r.variant}.png"
    r.side_by_side().save(p)
    print(
        f"{r.scene.id}: {r.lr.shape[2]}x{r.lr.shape[1]} px at 10 m -> {r.sr.shape[2]}x{r.sr.shape[1]} at 2.5 m "
        f"({r.variant}, {time.time() - t0:.0f}s)"
    )
    print(f"  -> {p}\n  AI-enhanced: plausible detail, not evidence")


def cmd_airborne(args) -> None:
    from .airborne import detect_airborne
    from .search import search

    aoi = parse_aoi(args.where, args.radius)
    end = parse_dt(args.date + "T23:59:59Z") if args.date else datetime.now(timezone.utc)
    res = search(
        aoi,
        end - timedelta(days=30 if not args.date else 1),
        end,
        sources=["sentinel-2"],
        max_cloud=60,
        sort="date",
        min_coverage=0.8,
    )
    for s in res.scenes[: args.images]:
        acs = detect_airborne(s, aoi)
        print(f"{s.datetime:%Y-%m-%d %H:%M}Z {s.platform}: {len(acs)} aircraft in flight")
        for a in acs:
            est = (
                f"  -> {a.speed_ms * 1.944:4.0f} kn hdg {a.heading_deg:3.0f}  ~{a.altitude_m:6,.0f} m"
                if a.speed_ms is not None
                else f"  ({a.note})"
                if a.note
                else ""
            )
            print(
                f"  {a.lat:.5f},{a.lon:.5f}  apparent {a.apparent_speed_ms * 1.944:4.0f} kn "
                f"toward {a.apparent_heading_deg:3.0f}{est}"
            )


def cmd_passes(args) -> None:
    from .passes import next_passes

    aoi = parse_aoi(args.where, 1.0)
    lat, lon = aoi.center
    fams = _csv(args.families) or None
    ps = next_passes(lat, lon, days=args.days, families=fams)
    shown = [p for p in ps if p.likely or args.all]
    print(f"overpasses of {lat:.4f},{lon:.4f} in the next {args.days:g} days (SGP4 on public TLEs)")
    print(f"{'UTC':<17} {'satellite':<12} {'family':<10} {'off-track':>9} {'side':<5} {'pass':<10} {'sun':>5}  note")
    for p in shown:
        mark = "" if p.likely else "  -"
        print(
            f"{p.time:%Y-%m-%d %H:%M}  {p.satellite:<12} {p.family:<10} {p.cross_track_km:>7.0f}km {p.side:<5} "
            f"{p.direction:<10} {p.sun_elevation:>5.0f}  {p.note}{mark}"
        )
    if not shown:
        print("  no likely acquisitions; try --days 14 or --all")
    print("(likely = imaging geometry allows it; Sentinel-1 also depends on the mission's acquisition plan)")


def cmd_doctor(args) -> None:
    """Environment report: hardware acceleration, models, credentials, network, storage."""
    import importlib.util
    import os
    import platform

    import httpx

    from . import __version__
    from .config import CACHE_DIR, USER_AGENT

    def row(k: str, v: str) -> None:
        print(f"  {k:<22} {v}")

    print(f"oracle {__version__} on {platform.platform()} (Python {platform.python_version()}, {platform.machine()})")
    print("compute")
    if importlib.util.find_spec("torch"):
        import torch

        from .objdet import pick_device, resolve_model

        row("torch", torch.__version__)
        row("cuda", str(torch.cuda.is_available()))
        mps = getattr(torch.backends, "mps", None)
        row("apple mps", str(bool(mps and mps.is_available())))
        row("yolo device", pick_device())
        row("yolo model (auto)", resolve_model("auto"))
    else:
        row("torch", 'not installed (pip install -e ".[ai]" for YOLO)')
    print("llm")
    row("anthropic sdk", "installed" if importlib.util.find_spec("anthropic") else 'missing (pip install -e ".[llm]")')
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    row("credentials", "env var set" if has_key else "none in env (ant auth login profiles also work)")
    print("foundation models")
    aef = CACHE_DIR / "aef"
    blocks = sum(f.stat().st_size for f in (aef / "blocks").rglob("*.npy")) if (aef / "blocks").exists() else 0
    row("alphaearth index", "ready" if (aef / "index.db").exists() else "downloads on first use (78 MB)")
    row("alphaearth cache", f"{blocks / 1e9:.2f} GB (cap ORACLE_AEF_CACHE_GB, default 4)")
    row("pyarrow", "installed" if importlib.util.find_spec("pyarrow") else 'missing (pip install -e ".[embed]")')
    sr = CACHE_DIR / "models" / "sen2sr"
    row(
        "sen2sr",
        ("installed" if importlib.util.find_spec("sen2sr") else 'missing (pip install -e ".[ai]")')
        + (", weights cached" if sr.exists() else ""),
    )
    print("storage")
    total = sum(f.stat().st_size for f in CACHE_DIR.rglob("*") if f.is_file()) if CACHE_DIR.exists() else 0
    row("cache", f"{CACHE_DIR} ({total / 1e6:.0f} MB)")
    st = _store()
    s = st.stats()
    row("database", f"{st.path} ({s['observations']} objects, {s['tracks']} tracks, {s['sites']} sites)")
    print("network")
    hosts = {
        "earth-search (S2)": "https://earth-search.aws.element84.com/v1",
        "planetary computer": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "maxar open data": "https://maxar-opendata.s3.amazonaws.com/events/catalog.json",
        "umbra": "https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com/stac/catalog.json",
        "esri wayback": "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json",
        "worldcover": "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N24E054_Map.tif",
        "celestrak (orbits)": "https://celestrak.org/NORAD/elements/gp.php?CATNR=40697&FORMAT=TLE",
        "alphaearth (source.coop)": "https://data.source.coop/tge-labs/aef/README.md",
        "hugging face (sen2sr)": "https://huggingface.co/api/models/tacofoundation/SEN2SR",
    }
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=15, follow_redirects=True) as c:
        for name, url in hosts.items():
            try:
                r = c.head(url) if url.endswith(".tif") else c.get(url)
                row(name, f"{r.status_code} in {r.elapsed.total_seconds() * 1000:.0f} ms")
            except httpx.HTTPError as exc:
                row(name, f"FAILED ({type(exc).__name__})")


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
    sp.add_argument(
        "--model",
        default="auto",
        help="auto (x on GPU/Apple Silicon, l on CPU) or any Ultralytics weights, e.g. yolo11s-obb.pt (~3x faster)",
    )
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

    sp = sub.add_parser("change", help="what changed on the ground (Sentinel-2 multi-index / Sentinel-1 radar)")
    sp.add_argument("where", help="lat,lon | bbox | place")
    where(sp, 3.0)
    sp.add_argument("-s", "--source", default="sentinel-2", choices=("sentinel-2", "sentinel-1"))
    sp.add_argument("--after", help="YYYY-MM-DD: newest image on/before this date (default: latest)")
    sp.add_argument("--before", help="YYYY-MM-DD: baseline images on/before this date (default: just before)")
    sp.add_argument("--baseline", default="recent", choices=("recent", "anniversary"), help="anniversary = same season last year")
    sp.add_argument("--baseline-images", type=int, default=3, help="images composited into the baseline")
    sp.add_argument("--k", type=float, default=3.0, help="robust z threshold (higher = fewer, surer changes)")
    sp.add_argument("--min-area", type=float, default=1500.0, help="smallest change region, m^2")
    sp.add_argument("--top", type=int, default=15)
    sp.add_argument("--site", help="also store significant changes as events of this site")
    sp.add_argument("-o", "--out", default="oracle-out")
    sp.set_defaults(fn=cmd_change)

    sp = sub.add_parser("ask", help="ask Oracle a question; it investigates with its tools and cites evidence")
    sp.add_argument("question")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--llm", action="store_true", help="require Claude")
    g.add_argument("--no-llm", action="store_true", help="offline playbook only")
    sp.add_argument("--max-calls", type=int, default=20, help="tool-call budget")
    sp.add_argument("--effort", default="high", choices=("low", "medium", "high", "xhigh", "max"))
    sp.add_argument("-v", "--verbose", action="store_true", help="also show the model's thinking summaries")
    sp.add_argument("--json", help="write the full investigation (steps, evidence, provenance) here")
    sp.add_argument("--no-web", action="store_true", help="don't let Claude search the web for context")
    sp.set_defaults(fn=cmd_ask)

    sp = sub.add_parser("similar", help="find places that look like your examples (AlphaEarth embeddings)")
    sp.add_argument("where", help="area to search")
    where(sp, 20.0)
    sp.add_argument("--like", action="append", help="example 'lat,lon' or place (repeatable)")
    sp.add_argument("--unlike", action="append", help="counter-example (repeatable)")
    sp.add_argument("--year", type=int, default=2025)
    sp.add_argument("--threshold", type=float)
    sp.add_argument("--top", type=int, default=20)
    sp.add_argument("-o", "--out", default="oracle-out")
    sp.set_defaults(fn=cmd_similar)

    sp = sub.add_parser("evolve", help="long-term change between two years (AlphaEarth embeddings)")
    sp.add_argument("where")
    where(sp, 5.0)
    sp.add_argument("--from", dest="year_from", type=int, default=2017)
    sp.add_argument("--to", dest="year_to", type=int, default=2025)
    sp.add_argument("--top", type=int, default=20)
    sp.add_argument("-o", "--out", default="oracle-out")
    sp.set_defaults(fn=cmd_evolve)

    sp = sub.add_parser("embed", help="false-colour map of the AlphaEarth embeddings (PCA or k-means segments)")
    sp.add_argument("where")
    where(sp, 5.0)
    sp.add_argument("--year", type=int, default=2025)
    sp.add_argument("--segments", type=int, default=0, help="k-means clusters instead of PCA colours")
    sp.add_argument("-o", "--out", default="oracle-out")
    sp.set_defaults(fn=cmd_embed)

    sp = sub.add_parser("enhance", help="AI super-resolution of Sentinel-2 to 2.5 m (SEN2SR; for viewing, not evidence)")
    sp.add_argument("where")
    where(sp, 1.5)
    sp.add_argument("--date")
    sp.add_argument("--max-cloud", type=float, default=10)
    sp.add_argument("--variant", default="auto", choices=("auto", "lite", "full"))
    sp.add_argument("-o", "--out", default="oracle-out")
    sp.set_defaults(fn=cmd_enhance)

    sp = sub.add_parser("airborne", help="aircraft in flight on Sentinel-2: speed, heading, altitude (band parallax)")
    sp.add_argument("where")
    where(sp, 10.0)
    sp.add_argument("--date")
    sp.add_argument("--images", type=int, default=3, help="how many recent images to scan")
    sp.set_defaults(fn=cmd_airborne)

    sp = sub.add_parser("passes", help="when will Sentinel-1/2 and Landsat next image a place (orbit prediction)")
    sp.add_argument("where", help="lat,lon | bbox | place")
    sp.add_argument("--days", type=float, default=7.0)
    sp.add_argument("--families", help="sentinel-2,sentinel-1,landsat (default all)")
    sp.add_argument("--all", action="store_true", help="also list passes whose geometry can't image the place")
    sp.set_defaults(fn=cmd_passes)

    sp = sub.add_parser("doctor", help="check hardware acceleration, models, credentials and network")
    sp.set_defaults(fn=cmd_doctor)
