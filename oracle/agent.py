"""Oracle Agent: ask a question; it investigates with Oracle's own tools and answers with evidence.

    oracle ask "what changed at Palm Jebel Ali this month?"
    oracle ask "are there large ships loitering off Fujairah?"

With Claude (``pip install "oracle-osint[llm]"`` + ``ANTHROPIC_API_KEY`` or an ``ant auth
login`` profile) the model plans the investigation, calls tools (geocode, imagery
search, ship/object/change detection, tracks, events, orbit predictions, AIS...), reads
the results, decides what to check next, and stops when the question is answered. A
manual tool loop (no hidden framework) keeps every step visible as a live trace.

Without an LLM, a deterministic playbook runs the same tools in a fixed order chosen by
keywords in the question and writes a templated answer, so the feature works offline.

Provenance: every tool result is registered as numbered evidence (``tools.Evidence``,
kinds observation / derived / reference). Answers must cite it as [E3]; after the
answer, Oracle checks each citation exists and reports cited/uncited evidence.

Boundary: Oracle works at the level of objects and places (ships, aircraft, vehicles as
counts and tracks, construction, floods). It does not identify or follow individual
people or private vehicles, read licence plates, or link imagery to personal identity,
and both modes refuse such questions up front.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .store import Store
from .tools import TOOLS, Toolbox, ToolError

MODEL = "claude-opus-5-5"
MAX_TOOL_CALLS = 20
MAX_TURNS = 40

SYSTEM_PROMPT = """You are Oracle, an imagery-intelligence analyst with direct, tool-based access to an \
open-source satellite monitoring system. A user asks a question about a place or activity; you investigate \
with the tools and answer with evidence.

What the system can see:
- Sentinel-2: 10 m optical, every 2-5 days, blocked by cloud. Sentinel-1: 10 m radar, day/night, through \
cloud, every 6-12 days. Landsat: 30 m. At 10 m a 300 m carrier and a 300 m tanker look alike: vessels can be \
counted, measured (+/- ~20 m), and judged underway from wakes, but not typed or named.
- Sub-metre imagery exists only in places: Maxar open data (disaster events), NAIP (USA), sparse Umbra/Capella \
radar. Esri Wayback is sub-metre but view-only. Check search_imagery before relying on detect_objects.
- AIS ship positions exist only if the user imported them (ais_positions errors otherwise).
- Foundation-model search: find_similar uses AlphaEarth Foundations embeddings (Google DeepMind; one vector \
per 10 m pixel per year, 2017-2025) to find every place in an area that resembles example points: after \
locating one tank farm, airfield, solar plant, mine or port, it finds the others. semantic_change compares \
two years to find long-term development. Matches are ranked candidates (whitened similarity >= 0.7 is \
worth checking, 0.85+ strong); confirm the ones that matter with imagery.
- detect_flying_aircraft finds aircraft in flight on Sentinel-2 from the timing gap between its colour \
bands, with velocity, and speed and altitude when the heading is measurable.
- Web search, when available, is for context: what a facility is, reported events, an explanation for an \
anomaly. It never replaces imagery. Reporting and imagery are different kinds of evidence; when they \
disagree, say so. Web sources are cited automatically.
- The object database holds earlier detections, cross-date tracks with same-object probabilities, and \
derived events, so cheap queries (get_events, query_objects, get_tracks) can answer history questions \
without new processing.

How to investigate:
- Make a short plan, then act. Start cheap (geocode, search_imagery, get_events/query_objects), then run the \
detector that answers the question on the newest clear image. Corroborate an important claim with a second \
date or sensor (radar when optical is cloudy) when the budget allows. Use next_passes to say when a gap can be \
closed. You have a budget of about {max_calls} tool calls; stop as soon as the question is answered.
- Use precise 'lat,lon' from geocode in later calls, and keep areas tight (a port or base is ~3-8 km).
- Independent tool calls run in parallel: request them in the same turn (e.g. detect_ships and \
detect_change on the same area).
- If a tool fails, adapt (other sensor, other date, smaller area) or report the gap; never fill it with guesses.

Evidence discipline (non-negotiable):
- Every tool fact carries an evidence id (E1, E2...). Cite them inline like [E4] or [E4][E9] for every \
factual claim. Never cite an id that a tool did not return. Never invent numbers, dates or coordinates.
- Separate what the imagery shows (observation) from what it probably means (inference). State inferences \
with calibrated language (almost certainly / likely / roughly even chance / unlikely) and say what they rest \
on. Do not speculate about intent, and never name a vessel, aircraft, unit or operator unless the evidence \
itself identifies it (e.g. an AIS match).

Boundaries: you analyse places and objects (counts, tracks, sizes, construction, damage, environmental \
change) for research and journalism-style open-source analysis. You do not identify, locate or follow \
individual people or private vehicles, read licence plates, or link imagery to personal identity, and you \
do not give targeting or attack-planning advice. Decline those parts briefly and answer the rest.

Answer in Markdown, under ~450 words:
**Bottom line** - 1-3 bullets answering the question directly.
**What the imagery shows** - the findings, each with UTC dates, coordinates where useful, and citations.
**Confidence and gaps** - what the data cannot support (resolution, cloud, revisit gaps, low link \
probabilities, missing AIS) and what would raise confidence.
**Next looks** - only if useful: when the next free images are expected (from next_passes)."""

BOUNDARY = re.compile(
    r"licen[cs]e\s*plates?|number\s*plates?|\balpr\b|\banpr\b|face\s*recogni|facial|"
    r"\b(track|follow|find|locate|identify|stalk)\w*\s+(a|this|that|the|my|his|her|their)?\s*"
    r"(person|man|woman|girl|boy|individual|people|ex|neighbou?r|wife|husband|boyfriend|girlfriend|employee)\b|"
    r"where\s+does\s+\w+\s+live|home\s+address|who\s+(owns|drives|lives)",
    re.I,
)
BOUNDARY_ANSWER = (
    "**Outside what Oracle does.** Oracle analyses places and objects in satellite imagery: ship and aircraft "
    "counts and tracks, construction, floods, burn scars, activity patterns. It does not identify or follow "
    "individual people or private vehicles, read licence plates, or link imagery to anyone's identity, and "
    "free imagery couldn't do it reliably anyway (a car is ~2 pixels at 1 m, and nothing here is real-time).\n\n"
    "If the underlying question is about a place (activity at a facility, vehicles at a site over time, "
    "changes in an area), ask that and Oracle will investigate it."
)

_LIVE: dict[str, Investigation] = {}
_LIVE_LOCK = threading.Lock()


@dataclass
class Investigation:
    question: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    status: str = "running"  # running | done | declined | error
    engine: str = ""
    steps: list[dict] = field(default_factory=list)
    answer: str = ""
    provenance: dict = field(default_factory=dict)
    error: str | None = None
    finished: str | None = None
    toolbox: Toolbox | None = None

    def to_dict(self) -> dict:
        ev = self.toolbox.evidence if self.toolbox else {}
        return {
            "id": self.id,
            "question": self.question,
            "created": self.created,
            "finished": self.finished,
            "status": self.status,
            "engine": self.engine,
            "steps": self.steps,
            "answer": self.answer,
            "evidence": {k: v.to_dict() for k, v in ev.items()},
            "provenance": self.provenance,
            "error": self.error,
        }


def register(inv: Investigation) -> Investigation:
    """Make an investigation visible to live() (the web app polls it while it runs)."""
    with _LIVE_LOCK:
        _LIVE[inv.id] = inv
        while len(_LIVE) > 50:
            _LIVE.pop(next(iter(_LIVE)))
    return inv


def live(inv_id: str) -> dict | None:
    with _LIVE_LOCK:
        inv = _LIVE.get(inv_id)
    return inv.to_dict() if inv else None


# --------------------------------------------------------------------------- entry point


def investigate(
    question: str,
    store: Store | None = None,
    use_llm: bool | None = None,
    max_calls: int = MAX_TOOL_CALLS,
    effort: str = "high",
    on_step: Callable[[dict], None] | None = None,
    inv: Investigation | None = None,
    web: bool = True,
) -> Investigation:
    """Run an investigation. ``use_llm``: None = Claude if available else playbook.
    ``web``: let Claude use web search for context (open-source reporting)."""
    store = store or Store()
    inv = inv or Investigation(question.strip())
    inv.toolbox = Toolbox(store)
    register(inv)
    store.save_investigation(inv.to_dict())
    try:
        if BOUNDARY.search(inv.question):
            inv.engine, inv.status, inv.answer = "boundary", "declined", BOUNDARY_ANSWER
        elif use_llm is False:
            _playbook(inv, on_step)
        else:
            try:
                _llm(inv, max_calls, effort, on_step, web)
            except _Unavailable as exc:
                if use_llm:
                    raise RuntimeError(str(exc)) from None
                _step(inv, on_step, "note", summary=f"Claude unavailable ({exc}); running the offline playbook")
                _playbook(inv, on_step)
        if inv.status == "running":
            inv.status = "done"
    except Exception as exc:  # noqa: BLE001 - recorded on the investigation
        inv.status, inv.error = "error", f"{type(exc).__name__}: {exc}"
        if not inv.answer:
            inv.answer = _compose(inv, note=f"The investigation stopped early: {inv.error}")
    inv.provenance = check_citations(inv.answer, inv.toolbox.evidence)
    inv.finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    store.save_investigation(inv.to_dict())
    return inv


def check_citations(answer: str, evidence: dict) -> dict:
    cited: list[str] = []
    for group in re.findall(r"\[([^\]]*E\d+[^\]]*)\]", answer):
        for eid in re.findall(r"E\d+", group):
            if eid not in cited:
                cited.append(eid)
    unknown = [c for c in cited if c not in evidence]
    kinds = {
        k: sum(1 for c in cited if c in evidence and evidence[c].kind == k)
        for k in ("observation", "derived", "reference", "report")
    }
    return {
        "cited": cited,
        "unknown": unknown,
        "uncited": [e for e in evidence if e not in cited],
        "cited_by_kind": kinds,
        "ok": not unknown,
    }


_STEP_LOCK = threading.Lock()


def _step(inv: Investigation, on_step, kind: str, **kw) -> dict:
    with _STEP_LOCK:
        s = {"n": len(inv.steps) + 1, "type": kind, "t": round(time.time(), 1), **kw}
        inv.steps.append(s)
    if on_step:
        on_step(s)
    return s


def _run_tool(inv: Investigation, on_step, name: str, args: dict) -> tuple[dict | None, str | None]:
    tb = inv.toolbox
    s = _step(inv, on_step, "tool", tool=name, input=args, status="running")
    t0 = time.time()
    try:
        result, err = tb.call(name, args), None
    except ToolError as exc:
        result, err = None, str(exc)
    except Exception as exc:  # noqa: BLE001 - a crashing tool is reported to the model, not fatal
        result, err = None, f"{name} failed: {type(exc).__name__}: {exc}"
    new = tb.collected()
    s.update(
        status="error" if err else "ok",
        ms=round((time.time() - t0) * 1000),
        evidence=new,
        summary=err or (tb.evidence[new[0]].summary if new else _brief(result)),
    )
    if on_step:
        on_step(s)
    return result, err


def _brief(result: dict | None) -> str:
    if not result:
        return ""
    for k in ("sites", "tracks", "events", "passes", "vessels"):
        if isinstance(result.get(k), list):
            return f"{len(result[k])} {k}"
    return ", ".join(list(result)[:4])


# --------------------------------------------------------------------------- Claude loop


class _Unavailable(Exception):
    """Claude can't be used (no SDK / credentials / API) before any tool ran."""


WEB_SEARCH = {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}


def _llm(inv: Investigation, max_calls: int, effort: str, on_step, web: bool = True) -> None:
    try:
        import anthropic
    except ImportError as exc:
        raise _Unavailable('pip install "oracle-osint[llm]"') from exc
    try:
        client = anthropic.Anthropic()
    except TypeError as exc:  # no credential source resolves
        raise _Unavailable("no Anthropic credentials (set ANTHROPIC_API_KEY or run `ant auth login`)") from exc
    inv.engine = "claude"
    system = [{"type": "text", "text": SYSTEM_PROMPT.format(max_calls=max_calls), "cache_control": {"type": "ephemeral"}}]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    messages: list[dict] = [{"role": "user", "content": f"Current time: {today} UTC.\n\nQuestion: {inv.question}"}]
    calls = 0
    final_text = ""
    for _turn in range(MAX_TURNS):
        tools = TOOLS + ([WEB_SEARCH] if web else [])
        try:
            resp = client.beta.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=system,
                tools=tools,
                messages=messages,
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except TypeError as exc:
            if "authentication" not in str(exc).lower() or calls:
                raise
            raise _Unavailable("no Anthropic credentials (set ANTHROPIC_API_KEY or run `ant auth login`)") from exc
        except anthropic.AuthenticationError as exc:
            raise _Unavailable("Anthropic credentials were rejected") from exc
        except anthropic.BadRequestError as exc:
            if web and "web_search" in str(exc).lower():  # web search not enabled for this org
                web = False
                _step(inv, on_step, "note", summary="web search unavailable for this API key; continuing without it")
                continue
            raise
        except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.APIStatusError) as exc:
            if calls == 0:
                raise _Unavailable(f"Anthropic API: {type(exc).__name__}") from exc
            raise RuntimeError(f"Anthropic API failed mid-investigation: {exc}") from exc

        if resp.stop_reason == "refusal":
            inv.status = "declined"
            inv.answer = "The model declined this question. " + BOUNDARY_ANSWER.split("\n\n", 1)[1]
            return
        texts = []
        for b in resp.content:
            if b.type == "thinking" and getattr(b, "thinking", ""):
                _step(inv, on_step, "thinking", summary=b.thinking[:800])
            elif b.type == "server_tool_use":
                q = (getattr(b, "input", None) or {}).get("query", "")
                _step(inv, on_step, "tool", tool=b.name, input={"query": q}, status="ok", ms=None, evidence=[], summary="")
            elif b.type == "text" and b.text:
                texts.append(_with_citations(inv, b))
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "pause_turn":
            continue
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses or resp.stop_reason in ("end_turn", "max_tokens", "stop_sequence"):
            final_text = "".join(texts).strip()  # cited answers arrive as several contiguous text blocks
            if resp.stop_reason == "max_tokens":
                final_text += "\n\n_(answer truncated at the output limit)_"
            break
        if "".join(texts).strip():
            _step(inv, on_step, "note", summary="".join(texts).strip()[:600])
        runnable = []
        for u in uses:
            if calls < max_calls:
                calls += 1
                runnable.append(u)
        # Independent calls in one turn run concurrently (I/O-bound: catalogs, COGs, orbit data).
        with ThreadPoolExecutor(max_workers=max(1, min(4, len(runnable)))) as pool:
            ran = pool.map(lambda u: _run_tool(inv, on_step, u.name, dict(u.input or {})), runnable)
            outs = dict(zip([u.id for u in runnable], ran, strict=True))
        results: list[dict] = []
        for u in uses:
            if u.id not in outs:
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": "tool budget exhausted", "is_error": True})
                continue
            result, err = outs[u.id]
            content = err if err else inv.toolbox.compact(result)
            results.append({"type": "tool_result", "tool_use_id": u.id, "content": content, "is_error": bool(err)})
        if calls >= max_calls:
            results.append(
                {"type": "text", "text": "Tool budget used up. Write the final answer now from the evidence gathered."}
            )
        messages.append({"role": "user", "content": results})
    else:
        final_text = final_text or "_(stopped: too many turns)_"
    inv.answer = final_text or _compose(inv, note="The model returned no text.")
    check = check_citations(inv.answer, inv.toolbox.evidence)
    if check["unknown"]:
        inv.answer += (
            f"\n\n_Provenance check: cites unknown evidence {', '.join(check['unknown'])}; treat those claims as unsupported._"
        )


def _with_citations(inv: Investigation, block) -> str:
    """Register a text block's web citations as 'report' evidence and append their ids."""
    cites = getattr(block, "citations", None) or []
    ids = []
    for c in cites:
        url = getattr(c, "url", None)
        if not url:
            continue
        tb = inv.toolbox
        existing = next((e.id for e in tb.evidence.values() if e.kind == "report" and e.links.get("url") == url), None)
        if existing is None:
            title = getattr(c, "title", None) or url
            snippet = (getattr(c, "cited_text", None) or "").strip()
            domain = re.sub(r"^https?://(www\.)?([^/]+).*$", r"\2", url)
            summary = f"{title} ({domain})" + (f': "{snippet[:200]}"' if snippet else "")
            existing = tb.add("report", "web_search", summary, links={"url": url})
        if existing not in ids:
            ids.append(existing)
    return block.text + "".join(f"[{i}]" for i in ids)


# --------------------------------------------------------------------------- offline playbook

_INTENTS = {
    "maritime": r"ship|vessel|boat|tanker|carrier|nav(y|al)|port\b|harbou?r|strait|anchorage|fleet|warship|"
    r"submarine|cargo|maritime|\bsea\b|dark vessel|sts|rendezvous|bay\b|gulf",
    "air": r"aircraft|plane|\bjets?\b|air\s?base|airport|airfield|bomber|fighter|helicopter|runway|apron",
    "ground": r"vehicle|\bcars?\b|truck|\btanks?\b|convoy|parking|equipment|containers?",
    "change": r"chang|construct|built|build|new\b|expan|flood|fire|burn|clear|deforest|earthwork|reclam|island|"
    r"damage|destroy|develop",
    "passes": r"\bwhen\b|next pass|revisit|overpass|next image|next look|satellite.*(over|pass)",
    "similar": r"similar|look(s)? like|other (places|sites|facilities|ones)|find (all|more|other)|more like|elsewhere",
    "longterm": r"since (19|20)\d\d|over the (last|past) (few |\d+ )?years|long[- ]term|in recent years|years",
    "flying": r"flying|in flight|airborne|overflight|flights|in the air|airspace",
}
_PLACE = re.compile(
    r"\b(?:at|in|near|around|over|off|of|outside|across)\s+((?:the\s+)?[A-Z][\w'’.\-]*(?:[\s,]+[A-Z][\w'’.\-]*)*)"
)
_COORD = re.compile(r"(-?\d{1,2}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)")


def extract_place(q: str) -> str | None:
    m = _COORD.search(q)
    if m:
        return f"{m.group(1)},{m.group(2)}"
    m = re.search(r"[\"“']([^\"”']{3,60})[\"”']", q)
    if m:
        return m.group(1)
    hits = _PLACE.findall(q)
    if hits:
        return max(hits, key=len).strip(" ,.?")
    caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", q)
    return max(caps, key=len) if caps else None


def intents(q: str) -> set[str]:
    return {k for k, rx in _INTENTS.items() if re.search(rx, q, re.I)}


def _playbook(inv: Investigation, on_step) -> None:
    inv.engine = "playbook"
    q = inv.question
    want = intents(q)
    place = extract_place(q)
    _step(inv, on_step, "note", summary=f"plan: place={place!r}, focus={sorted(want) or ['overview']}")
    if not place:
        inv.answer = 'I couldn\'t find a place in the question. Name one ("... at Port of Fujairah") or give lat,lon.'
        return
    if _COORD.fullmatch(place.replace(" ", "")):
        lat, lon = map(float, place.split(","))
    else:
        g, err = _run_tool(inv, on_step, "geocode", {"place": place})
        if err:
            inv.answer = f"Couldn't resolve {place!r}: {err}"
            return
        lat, lon = g["lat"], g["lon"]
    where = f"{lat:.5f},{lon:.5f}"
    if not want & {"maritime", "air", "ground", "change", "similar", "longterm", "flying"}:
        want |= {"maritime", "change"} if re.search(_INTENTS["maritime"], place, re.I) else {"change"}
    if "flying" in want:
        want.discard("air")  # airborne, not parked: the band-timing detector, not YOLO
    _run_tool(inv, on_step, "search_imagery", {"where": where, "radius_km": 5, "days_back": 30})
    _run_tool(inv, on_step, "get_events", {"where": where, "radius_km": 10, "days": 60})
    if "maritime" in want:
        # Geocoded places are often inland centroids (a city, not its anchorage): cast a wide net.
        r, err = _run_tool(inv, on_step, "detect_ships", {"where": where, "radius_km": 25, "source": "sentinel-2"})
        if err or (r and r.get("clear_fraction", 1) < 0.5):
            _run_tool(inv, on_step, "detect_ships", {"where": where, "radius_km": 25, "source": "sentinel-1"})
    if want & {"air", "ground"}:  # searches the whole sub-metre archive, not just the last 30 days
        _run_tool(inv, on_step, "detect_objects", {"where": where, "radius_km": 0.6})
    if "change" in want and "longterm" not in want:
        _, err = _run_tool(inv, on_step, "detect_change", {"where": where, "radius_km": 3, "source": "sentinel-2"})
        if err:
            _run_tool(inv, on_step, "detect_change", {"where": where, "radius_km": 3, "source": "sentinel-1"})
    if "longterm" in want:
        m = re.search(r"since ((?:19|20)\d\d)", q)
        y0 = max(2017, int(m.group(1))) if m else 2017
        _run_tool(inv, on_step, "semantic_change", {"where": where, "radius_km": 5, "year_from": y0, "year_to": 2025})
    if "similar" in want:
        _run_tool(inv, on_step, "find_similar", {"where": where, "radius_km": 30, "examples": [where]})
    if "flying" in want:
        _run_tool(inv, on_step, "detect_flying_aircraft", {"where": where, "radius_km": 12})
    _run_tool(inv, on_step, "next_passes", {"lat": lat, "lon": lon, "days": 7})
    inv.answer = _compose(inv)


def _compose(inv: Investigation, note: str = "") -> str:
    """Deterministic answer from the evidence registry, every line cited."""
    ev = inv.toolbox.evidence if inv.toolbox else {}
    by_tool: dict[str, list] = {}
    for e in ev.values():
        by_tool.setdefault(e.tool, []).append(e)
    bottom, shows, gaps = [], [], []
    ships = [e for e in by_tool.get("detect_ships", []) if e.kind == "derived"]
    for e in ships:
        bottom.append(f"{e.summary} [{e.id}]")
        big = [o for o in by_tool["detect_ships"] if o.kind == "observation" and (o.data.get("length_m") or 0) >= 200]
        for o in big[:4]:
            shows.append(f"{o.summary} [{o.id}]")
    objs = [e for e in by_tool.get("detect_objects", []) if e.kind == "derived"]
    for e in objs:
        bottom.append(f"{e.summary} [{e.id}]")
    for tool in ("find_similar", "semantic_change", "detect_flying_aircraft"):
        items = by_tool.get(tool, [])
        if items:
            bottom.append(f"{items[0].summary} [{items[0].id}]")
            for e in items[1:6]:
                shows.append(f"{e.summary} [{e.id}]")
    if by_tool.get("find_similar"):
        gaps.append("Embedding matches are look-alikes, not identifications: confirm the ones that matter in imagery.")
    if by_tool.get("detect_flying_aircraft"):
        gaps.append(
            "Airborne detections come from band-timing parallax at 10 m: velocity is reliable to ~10%, altitude is rough."
        )
    change = by_tool.get("detect_change", [])
    if change:
        bottom.append(f"{change[0].summary} [{change[0].id}]")
        for r in change[1:6]:
            shows.append(f"{r.summary} [{r.id}]")
    events = sorted(by_tool.get("get_events", []), key=lambda e: e.summary)
    for e in events[:6]:
        shows.append(f"{e.summary} [{e.id}]")
    for e in by_tool.get("search_imagery", []):
        shows.append(f"Imagery available: {e.summary} [{e.id}]")
    errs = [s for s in inv.steps if s.get("status") == "error"]
    for s in errs:
        gaps.append(f"{s['tool']} failed: {s['summary']}")
    if ships:
        gaps.append("10 m imagery measures hull length (+/- ~20 m) but cannot identify ship type or name.")
    if change:
        gaps.append("Change classes come from spectral rules; confirm notable regions by eye in the before/after chips.")
    if not (
        ships or objs or change or any(by_tool.get(t) for t in ("find_similar", "semantic_change", "detect_flying_aircraft"))
    ):
        gaps.append("No detector produced results; the area may lack recent clear imagery.")
    passes = by_tool.get("next_passes", [])
    out = ["**Bottom line**"]
    out += [f"- {b}" for b in bottom] or ["- Nothing conclusive from the available imagery."]
    if shows:
        out += ["", "**What the imagery shows**"] + [f"- {s}" for s in shows]
    out += ["", "**Confidence and gaps**"] + [f"- {g}" for g in gaps]
    if passes:
        out += ["", "**Next looks**", f"- {passes[0].summary} [{passes[0].id}]"]
    if note:
        out += ["", f"_{note}_"]
    if inv.engine == "playbook":
        out += [
            "",
            "_Offline playbook (no LLM): fixed steps and a templated answer. Configure Claude for an analyst-led investigation._",
        ]
    return "\n".join(out)


__all__ = ["investigate", "Investigation", "check_citations", "extract_place", "intents", "live", "SYSTEM_PROMPT"]


if __name__ == "__main__":  # pragma: no cover
    import sys

    print(json.dumps(investigate(" ".join(sys.argv[1:]), use_llm=False).to_dict(), indent=1, default=str)[:4000])
