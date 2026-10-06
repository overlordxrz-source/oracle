"""Offline tests for orbits, pattern-of-life, change detection and the Oracle agent."""

import math
import types
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from rasterio.transform import from_origin

from oracle import agent, change, passes, patterns
from oracle.geo import utm_crs_for
from oracle.imagery import Grid
from oracle.observations import AIRCRAFT, VESSEL, Observation
from oracle.store import Store
from oracle.tools import Toolbox, ToolError

T0 = datetime(2026, 9, 1, 7, 0, tzinfo=timezone.utc)
S2A = (
    "1 40697U 15028A   26277.65521571  .00000153  00000+0  75110-4 0  9994",
    "2 40697  98.5646 350.9109 0001084  88.6373 271.4934 14.30820120589385",
)


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "t.db")


def at(lat, lon, north_m, east_m):
    return lat + north_m / 110_540, lon + east_m / (111_320 * math.cos(math.radians(lat)))


def vessel(lat, lon, scene, t=T0, length=200.0, axis=45.0, underway=False, **kw):
    return Observation(
        VESSEL,
        lat,
        lon,
        t,
        scene,
        "sentinel-2",
        "ships",
        length_m=length,
        width_m=length / 6,
        axis_deg=axis,
        attrs={"underway": underway, **kw},
    )


# --------------------------------------------------------------------------- orbits


def test_sun_elevation_equinox_noon_is_overhead():
    assert passes.sun_elevation(0.0, 0.0, datetime(2026, 3, 20, 12, 7, tzinfo=timezone.utc)) > 85
    assert passes.sun_elevation(0.0, 0.0, datetime(2026, 3, 20, 0, 7, tzinfo=timezone.utc)) < -85


def test_sentinel2_pass_matches_observed_acquisition(monkeypatch):
    # Sentinel-2A imaged Hormuz at ~07:02 UTC on 2026-10-06 (10-day repeat of the 09-26 take).
    monkeypatch.setattr(passes, "_fetch_tles", lambda: {"SENTINEL-2A": S2A})
    ps = passes.next_passes(26.4494, 56.2028, days=2, start=datetime(2026, 10, 5, tzinfo=timezone.utc))
    likely = [p for p in ps if p.likely]
    assert likely and likely[0].family == "sentinel-2" and likely[0].direction == "descending"
    want = datetime(2026, 10, 6, 7, 2, tzinfo=timezone.utc)
    assert abs((likely[0].time - want).total_seconds()) < 180
    assert likely[0].sun_elevation > 30 and likely[0].cross_track_km < 145


# --------------------------------------------------------------------------- patterns


def test_sts_pair_alongside_is_flagged_but_passing_ship_is_not():
    lat, lon = 1.2, 104.0
    a = vessel(lat, lon, "s1", length=250, axis=45)
    # 60 m to the side (perpendicular to a 45 deg axis is 135 deg)
    b = vessel(*at(lat, lon, -60 * 0.707, 60 * 0.707), "s1", length=180, axis=48)
    passing = vessel(*at(lat, lon, 150, 150), "s1", length=200, axis=45, underway=True)
    ev = patterns.sts_candidates("X", [a, b, passing])
    pairs = [e for e in ev if e["kind"] == "sts_candidate" and "alongside" in e["title"]]
    assert len(pairs) == 1 and set(pairs[0]["detail"]["obs_ids"]) == {a.id, b.id}
    assert pairs[0]["detail"]["gap_m"] < 40


def test_sts_rejects_crossing_axes():
    a = vessel(1.2, 104.0, "s1", axis=0)
    b = vessel(*at(1.2, 104.0, 0, 50), "s1", axis=90)
    assert not [e for e in patterns.sts_candidates("X", [a, b]) if "alongside" in e["title"]]


def test_unusual_location_needs_history_and_flags_new_anchorage():
    rng = np.random.default_rng(0)
    obs, times = [], {}
    for k in range(10):
        sid = f"s{k}"
        times[sid] = T0 + timedelta(days=5 * k)
        for _ in range(5):
            obs.append(vessel(*at(1.2, 104.0, rng.normal(0, 300), rng.normal(0, 300)), sid, t=times[sid], length=150))
    odd = vessel(*at(1.2, 104.0, 4000, 6000), "s9", t=times["s9"], length=230)
    moving = vessel(*at(1.2, 104.0, 4000, 7000), "s9", t=times["s9"], underway=True)
    ev = patterns.unusual_locations("X", obs + [odd, moving], times)
    assert [e["obs_id"] for e in ev] == [odd.id]
    # Too little history: silent.
    assert patterns.unusual_locations("X", obs[:10] + [odd], {k: times[k] for k in ("s0", "s1", "s9")}) == []


def test_density_grid_peaks_where_objects_are():
    obs = [Observation(AIRCRAFT, 10.0, 20.0, T0, f"s{i}", "maxar", "yolo") for i in range(5)]
    g = np.array(patterns.density_grid(obs, (19.99, 9.99, 20.01, 10.01), "aircraft", 5)["per_km2_per_look"])
    i, j = np.unravel_index(g.argmax(), g.shape)
    assert abs(i - g.shape[0] / 2) <= 2 and abs(j - g.shape[1] / 2) <= 2


# --------------------------------------------------------------------------- change detection


def _scene_pair(n=200, seed=1):
    rng = np.random.default_rng(seed)

    def noise():
        return rng.normal(0, 0.004, (n, n)).astype(np.float32)

    land = {"blue": 0.06, "green": 0.10, "red": 0.08, "nir": 0.30, "swir16": 0.25, "swir22": 0.15}
    water = {"blue": 0.08, "green": 0.08, "red": 0.04, "nir": 0.02, "swir16": 0.02, "swir22": 0.01}
    sand = {"blue": 0.20, "green": 0.25, "red": 0.25, "nir": 0.30, "swir16": 0.35, "swir22": 0.28}
    cleared = {"blue": 0.10, "green": 0.13, "red": 0.15, "nir": 0.17, "swir16": 0.30, "swir22": 0.24}
    is_water = np.zeros((n, n), bool)
    is_water[:, 120:] = True
    before = {b: np.where(is_water, water[b], land[b]).astype(np.float32) + noise() for b in land}
    after = {b: v.copy() + noise() for b, v in before.items()}
    for b in land:
        after[b][30:55, 30:55] = cleared[b] + noise()[30:55, 30:55]  # forest cleared: 6.25 ha
        after[b][60:80, 140:160] = sand[b] + noise()[60:80, 140:160]  # reclaimed: 4 ha
        after[b][150:153, 140:170] = sand[b]  # a 300 x 30 m ship-shaped bright thing on water
    return before, after


def _grid(n=200):
    crs = utm_crs_for(25.0, 55.0)
    return Grid(crs, from_origin(300_000, 2_770_000, 10, 10), n, n)


def test_change_classes_on_synthetic_scene():
    before, after = _scene_pair()
    both = np.ones((200, 200), bool)
    masks, metrics = change._classify_s2(before, after, both, 3.0)
    regions, kind_map = change._regions(masks, metrics, _grid(), 3.0, 1500.0, 3)
    kinds = {r.kind: r for r in regions}
    assert set(kinds) == {"vegetation_loss", "water_loss"}, [(r.kind, r.area_m2) for r in regions]
    assert abs(kinds["vegetation_loss"].area_m2 - 62_500) < 8000
    assert abs(kinds["water_loss"].area_m2 - 40_000) < 6000
    assert kinds["vegetation_loss"].values["ndvi"][0] > 0.5 > kinds["vegetation_loss"].values["ndvi"][1]
    assert kind_map.max() > 0 and len(kinds["water_loss"].outline) >= 4


def test_change_events_survive_tracker_refresh(store):
    before, after = _scene_pair()
    masks, metrics = change._classify_s2(before, after, np.ones((200, 200), bool), 3.0)
    regions, kind_map = change._regions(masks, metrics, _grid(), 3.0, 1500.0, 3)
    from oracle.geo import AOI
    from oracle.models import Render, Scene

    sc = Scene("A", "sentinel-2", "s2", "optical", T0, 10.0, (55, 25, 55.1, 25.1), Render("rgb8", ["x"]))
    rgb = np.zeros((200, 200, 3), np.uint8)
    res = change.ChangeResult(AOI((55, 25, 55.1, 25.1)), sc, [sc], _grid(), regions, kind_map, 1.0, "test", [], rgb, rgb)
    evs = change.to_events(res, "site", min_confidence=0.0)
    assert evs and all(e["kind"].startswith("change_") for e in evs)
    store.add_events(evs)
    store.replace_events("site", [])  # what a tracker refresh does
    assert len(store.events(site="site")) == len(evs)
    assert res.png("change")[:4] == b"\x89PNG"
    r = regions[0]
    assert res.png("before", crop=r.bbox)[:4] == b"\x89PNG"


# --------------------------------------------------------------------------- tools & agent


def test_toolbox_validates_arguments(store):
    tb = Toolbox(store)
    with pytest.raises(ToolError, match="missing required"):
        tb.call("get_track", {})
    with pytest.raises(ToolError, match="unknown argument"):
        tb.call("list_sites", {"bogus": 1})
    with pytest.raises(ToolError, match="unknown tool"):
        tb.call("rm_rf", {})
    with pytest.raises(ToolError, match="too large"):
        tb.call("detect_change", {"where": "25,55", "radius_km": 500})


def _seed(store):
    store.save_site("dock", (55.0, 25.0, 55.1, 25.1), "maritime")
    store.add_events(
        [
            {
                "id": "e1",
                "site": "dock",
                "time": datetime.now(timezone.utc).isoformat(),
                "kind": "sts_candidate",
                "severity": 0.6,
                "title": "250 m and 180 m hulls alongside each other at dock",
                "detail": {"lat": 25.05, "lon": 55.05},
            }
        ]
    )


def test_get_events_registers_evidence(store):
    _seed(store)
    tb = Toolbox(store)
    r = tb.call("get_events", {"where": "25.05,55.05", "radius_km": 5})
    assert r["events"][0]["evidence"] == "E1" and tb.evidence["E1"].kind == "derived"
    assert tb.call("get_events", {"where": "-30,20", "radius_km": 5})["events"] == []


def test_citation_check():
    ev = {"E1": types.SimpleNamespace(kind="observation"), "E2": types.SimpleNamespace(kind="reference")}
    c = agent.check_citations("A [E1]. B [E1, E3]. C [E2][E1].", ev)
    assert c["cited"] == ["E1", "E3", "E2"] and c["unknown"] == ["E3"] and not c["ok"]
    assert c["cited_by_kind"] == {"observation": 1, "derived": 0, "reference": 1, "report": 0}


@pytest.mark.parametrize(
    "q",
    [
        "track this license plate across the city",
        "find the person who parks here every day",
        "where does John live",
        "use facial recognition on the crowd",
    ],
)
def test_boundary_declines_personal_tracking(store, q):
    inv = agent.investigate(q, store=store, use_llm=False)
    assert inv.status == "declined" and inv.engine == "boundary" and not inv.steps


@pytest.mark.parametrize(
    "q", ["how many ships are at Fujairah?", "track the carrier group near Guam", "what changed at Palm Jebel Ali"]
)
def test_boundary_allows_object_questions(q):
    assert not agent.BOUNDARY.search(q)


def test_place_and_intent_extraction():
    assert agent.extract_place("ships near 25.12, 56.3 today?") == "25.12,56.3"
    assert agent.extract_place("What changed at Palm Jebel Ali this month?") == "Palm Jebel Ali"
    assert agent.extract_place("any aircraft at 'Al Udeid'?") == "Al Udeid"
    assert agent.intents("new construction at the naval base") >= {"change", "maritime"}


def test_playbook_runs_tools_and_cites(store, monkeypatch):
    def fake_change(self, where, radius_km=None, source=None, **kw):
        e = self.add("derived", "detect_change", "s2: 2 change regions", lat=25.0, lon=55.0)
        r = self.add("derived", "detect_change", "vegetation_loss 1.2 ha at 25.0,55.0", lat=25.0, lon=55.0)
        return {"regions": [{"evidence": r}], "evidence": e}

    monkeypatch.setattr(
        Toolbox,
        "t_search_imagery",
        lambda self, **kw: {"scenes": [], "evidence": self.add("reference", "search_imagery", "3 images")},
    )
    monkeypatch.setattr(Toolbox, "t_detect_change", fake_change)
    monkeypatch.setattr(
        Toolbox,
        "t_next_passes",
        lambda self, lat, lon, days=None: {"evidence": self.add("reference", "next_passes", "S2 tomorrow")},
    )
    steps = []
    inv = agent.investigate("What changed at 25.0,55.0?", store=store, use_llm=False, on_step=steps.append)
    assert inv.status == "done" and inv.engine == "playbook"
    assert [s["tool"] for s in inv.steps if s["type"] == "tool"] == [
        "search_imagery",
        "get_events",
        "detect_change",
        "next_passes",
    ]
    assert inv.provenance["ok"] and set(inv.provenance["cited"]) >= {"E1", "E2", "E3"}
    saved = store.investigation(inv.id)
    assert saved["status"] == "done" and saved["evidence"]["E2"]["tool"] == "detect_change"


class _FakeMessages:
    def __init__(self, script):
        self.script, self.calls = script, []

    def create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})  # the loop appends to the same list
        return self.script.pop(0)


def _block(**kw):
    return types.SimpleNamespace(**kw)


def test_llm_loop_with_mocked_claude(store, monkeypatch):
    import anthropic

    _seed(store)
    script = [
        types.SimpleNamespace(
            stop_reason="tool_use",
            content=[
                _block(type="thinking", thinking="Check stored events first.", signature="x"),
                _block(type="text", text="Looking at stored events."),
                _block(type="tool_use", id="tu1", name="get_events", input={"where": "25.05,55.05", "radius_km": 5}),
                _block(type="tool_use", id="tu2", name="get_track", input={"track_id": "nope"}),
            ],
        ),
        types.SimpleNamespace(
            stop_reason="end_turn",
            content=[_block(type="text", text="**Bottom line**\n- A likely STS pair [E1]. Also [E7].")],
        ),
    ]
    fake = _FakeMessages(script)
    monkeypatch.setattr(anthropic, "Anthropic", lambda: types.SimpleNamespace(beta=types.SimpleNamespace(messages=fake)))
    inv = agent.investigate("Any ship-to-ship transfers at the dock?", store=store, use_llm=True)
    assert inv.engine == "claude" and inv.status == "done"
    first = fake.calls[0]
    assert first["model"] == "claude-opus-5-5" and first["fallbacks"] == "default"
    assert first["thinking"]["type"] == "adaptive" and first["output_config"] == {"effort": "high"}
    assert "tool_choice" not in first and {t["name"] for t in first["tools"]} >= {"detect_change", "get_events"}
    results = fake.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["tu1", "tu2"]
    assert results[0]["is_error"] is False and "E1" in results[0]["content"]
    assert results[1]["is_error"] is True and "no track" in results[1]["content"]
    assert inv.provenance["unknown"] == ["E7"] and "unknown evidence E7" in inv.answer
    assert [s["type"] for s in inv.steps][:2] == ["thinking", "note"]


def test_llm_unavailable_falls_back_to_playbook(store, monkeypatch):
    import anthropic

    def boom():
        raise TypeError("Could not resolve authentication method")

    monkeypatch.setattr(anthropic, "Anthropic", boom)
    monkeypatch.setattr(Toolbox, "call", lambda self, name, args: (_ for _ in ()).throw(ToolError("offline")))
    inv = agent.investigate("what changed at 10.0,20.0", store=store)
    assert inv.engine == "playbook" and inv.steps[0]["type"] == "note" and "unavailable" in inv.steps[0]["summary"]


def test_llm_required_but_unavailable_raises(store, monkeypatch):
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda: (_ for _ in ()).throw(TypeError("authentication")))
    inv = agent.investigate("what changed at 10.0,20.0", store=store, use_llm=True)
    assert inv.status == "error" and "credentials" in inv.error
