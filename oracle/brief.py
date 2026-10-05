"""Intelligence briefing: what changed across your sites, ranked and written up.

``digest()`` turns the object database into a compact structured summary per site:
coverage, counts vs. baseline, top events, notable tracks and caveats. That digest is
rendered either:

  * by Claude (``pip install "oracle-osint[llm]"`` + ``ANTHROPIC_API_KEY`` or an
    ``ant auth login`` profile): an analyst-style brief with a bottom line up front,
    calibrated estimative language, and every claim tied to the digest; or
  * deterministically (no key / no network), as a plain Markdown report.

The model only sees the digest (numbers, coordinates, timestamps). It can't invent
detections, and the prompt tells it to say what the data can't show.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from .store import Store

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You are an imagery analyst writing a short intelligence brief from the output of an automated \
open-source satellite monitoring pipeline (free Sentinel-1/2 imagery at 10 m, open sub-metre radar and \
optical where available, YOLO object detection, and a probabilistic cross-date tracker).

You get a JSON digest. Write the brief in Markdown:
1. **Bottom line up front**: 2-4 bullets with the most significant changes across all sites.
2. One section per site that had activity, most significant first: what changed (counts vs the site's own \
baseline, notable arrivals/departures, dark vessels, anomalies), with coordinates and UTC dates for anything \
a reader might want to check. Skip quiet sites in a single closing line.
3. **Confidence and gaps**: what the data can't support, e.g. 10 m pixels can't identify a ship type, \
cloud-blocked passes, link probabilities below ~70%, no AIS data loaded, sparse revisits.

Rules: use calibrated estimative language (almost certainly / likely / roughly even chance / unlikely) and \
tie each judgement to the digest's numbers. Never name a specific vessel, aircraft, unit or operator unless \
the digest contains that identification (for example an AIS match). Do not speculate about intent. If \
nothing notable happened, say so briefly. Keep it under ~600 words."""


def digest(store: Store, since: datetime, sites: list[str] | None = None, max_events: int = 12) -> dict:
    out = {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "since": since.isoformat(), "sites": []}
    all_sites = store.sites()
    names = sites or [s["name"] for s in all_sites]
    kinds = {s["name"]: s.get("kind") for s in all_sites}
    have_ais = store.conn.execute("SELECT 1 FROM ais LIMIT 1").fetchone() is not None
    for name in names:
        runs = [r for r in store.runs(name) if r["status"] == "ok" and r.get("scene_time")]
        recent = [r for r in runs if r["scene_time"] >= since.isoformat()]
        obs = store.observations(site=name)
        evs = store.events(since=since, site=name, limit=max_events)
        per_scene: dict[tuple[str, str], int] = defaultdict(int)
        for o in obs:
            per_scene[(o.scene_id, o.cls)] += 1
        counts = {}
        for cls in sorted({o.cls for o in obs}):
            series = []
            for r in runs:
                if (r.get("clear") or 1.0) >= 0.85:
                    series.append((r["scene_time"][:10], r["source"], per_scene.get((r["scene_id"], cls), 0)))
            before = [n for t, _, n in series if t < since.date().isoformat()]
            after = [n for t, _, n in series if t >= since.date().isoformat()]
            counts[cls] = {
                "baseline_median": _median(before),
                "period_median": _median(after),
                "latest": series[-1] if series else None,
            }
        tracks = store.tracks(name)
        recent_tracks = [t for t in tracks if t["last_seen"] >= since.isoformat()]
        notable = sorted(
            (t for t in recent_tracks if t["cls"] in ("vessel", "aircraft")),
            key=lambda t: (-(t["length_m"] or 0), -t["n_obs"]),
        )[:6]
        dark = [o for o in obs if o.attrs.get("dark") and o.time >= since]
        out["sites"].append(
            {
                "site": name,
                "kind": kinds.get(name),
                "images_in_period": len(recent),
                "images_total": len(runs),
                "cloudy_images_in_period": sum((r.get("clear") or 1.0) < 0.85 for r in recent),
                "sources": dict(Counter(r["source"] for r in recent)),
                "last_image": runs[-1]["scene_time"][:16] if runs else None,
                "counts": counts,
                "events": [
                    {
                        "time": e["time"][:16],
                        "kind": e["kind"],
                        "severity": e["severity"],
                        "title": e["title"],
                        "detail": {
                            k: v
                            for k, v in e["detail"].items()
                            if k in ("lat", "lon", "length_m", "count", "median", "z", "dwell_days", "speed_kn", "nearest_ais_m")
                        },
                    }
                    for e in evs
                ],
                "notable_tracks": [
                    {
                        "track": t["id"],
                        "class": t["cls"],
                        "length_m": t["length_m"],
                        "sightings": t["n_obs"],
                        "first_seen": t["first_seen"][:10],
                        "last_seen": t["last_seen"][:10],
                        "status": t["status"],
                        "mean_link_probability": t["mean_link_prob"],
                        "sources": t["attrs"].get("sources"),
                        "lat": round(t["lat"], 4),
                        "lon": round(t["lon"], 4),
                    }
                    for t in notable
                ],
                "dark_vessels": len(dark),
                "ais_loaded": have_ais,
            }
        )
    out["sites"].sort(key=lambda s: -max([e["severity"] for e in s["events"]] or [0]))
    return out


def _median(xs: list[int]) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    m = len(xs) // 2
    return float(xs[m]) if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def render_markdown(d: dict) -> str:
    """Deterministic brief: no model, no network."""
    lines = [f"# Oracle brief, since {d['since'][:10]}", f"_generated {d['generated']} UTC_", ""]
    active = [s for s in d["sites"] if s["events"]]
    top = sorted((e | {"site": s["site"]} for s in d["sites"] for e in s["events"]), key=lambda e: -e["severity"])[:5]
    lines.append("## Top items")
    lines += [f"- **{e['site']}**: {e['title']} ({e['time']}Z)" for e in top] or ["- Nothing notable."]
    lines.append("")
    for s in active:
        lines.append(f"## {s['site']}")
        lines.append(
            f"{s['images_in_period']} images this period ({', '.join(f'{k} {v}' for k, v in s['sources'].items()) or 'none'}), "
            f"{s['cloudy_images_in_period']} too cloudy; last image {s['last_image']}Z."
        )
        for cls, c in s["counts"].items():
            if c["period_median"] is not None:
                base = f" (baseline {c['baseline_median']:.0f})" if c["baseline_median"] is not None else ""
                lines.append(f"- {cls}: typical {c['period_median']:.0f} per image{base}")
        for e in s["events"][:8]:
            loc = f" at {e['detail']['lat']:.4f}, {e['detail']['lon']:.4f}" if "lat" in e["detail"] else ""
            lines.append(f"- [{e['kind']}] {e['title']}{loc} ({e['time']}Z)")
        for t in s["notable_tracks"][:4]:
            p = f", link confidence {t['mean_link_probability']:.0%}" if t["mean_link_probability"] else ""
            lines.append(
                f"- track {t['track']}: {t['length_m']} m {t['class']}, {t['sightings']} sightings "
                f"{t['first_seen']} to {t['last_seen']} ({t['status']}{p})"
            )
        lines.append("")
    quiet = [s["site"] for s in d["sites"] if not s["events"]]
    if quiet:
        lines.append(f"_Quiet: {', '.join(quiet)}._")
    lines += [
        "",
        "_Caveats: 10 m imagery can't identify ship types; tracks are probabilistic; "
        "cloudy passes and sparse revisits leave gaps._",
    ]
    return "\n".join(lines)


def render_llm(d: dict, model: str = MODEL) -> str:
    """Brief written by Claude from the digest. Raises RuntimeError if unavailable."""
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError('LLM briefs need: pip install "oracle-osint[llm]"') from exc
    client = anthropic.Anthropic()
    payload = json.dumps(d, default=str, separators=(",", ":"))
    try:
        with client.beta.messages.stream(
            model=model,
            max_tokens=16000,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": f"Digest:\n```json\n{payload}\n```\nWrite the brief."}],
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            msg = stream.get_final_message()
    except anthropic.AuthenticationError as exc:
        raise RuntimeError("Anthropic credentials were rejected") from exc
    except TypeError as exc:  # raised before any request when no credential source resolves
        if "authentication" not in str(exc).lower():
            raise
        raise RuntimeError("no Anthropic credentials (set ANTHROPIC_API_KEY or run `ant auth login`)") from exc
    except anthropic.RateLimitError as exc:
        raise RuntimeError("Anthropic rate limit hit; try again shortly") from exc
    except anthropic.APIStatusError as exc:
        raise RuntimeError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise RuntimeError("could not reach the Anthropic API") from exc
    if msg.stop_reason == "refusal":
        raise RuntimeError("the model declined to write this brief")
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def brief(
    store: Store, since: datetime | None = None, sites: list[str] | None = None, use_llm: bool | None = None
) -> tuple[str, str]:
    """-> (markdown, engine) where engine is "claude" or "template"."""
    since = since or datetime.now(timezone.utc) - timedelta(days=7)
    d = digest(store, since, sites)
    if use_llm is not False:
        try:
            return render_llm(d), "claude"
        except RuntimeError as exc:
            if use_llm:
                raise
            note = f"\n\n_(Claude brief unavailable: {exc}; showing the built-in report.)_"
            return render_markdown(d) + note, "template"
    return render_markdown(d), "template"


__all__ = ["digest", "render_markdown", "render_llm", "brief"]
