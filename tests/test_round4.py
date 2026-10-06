"""Offline tests for embeddings (AlphaEarth), super-resolution, airborne aircraft and agent v2."""

import math
import types
from datetime import datetime, timezone

import numpy as np
import pytest
from rasterio.transform import from_origin

from oracle import agent, airborne, embeddings, superres
from oracle.geo import AOI, utm_crs_for
from oracle.imagery import Grid
from oracle.models import Render, Scene
from oracle.store import Store
from oracle.tools import Toolbox

T0 = datetime(2026, 9, 24, 7, 2, tzinfo=timezone.utc)


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "t.db")


def _grid(n=100, res=10.0, lat=25.0, lon=55.0):
    crs = utm_crs_for(lat, lon)
    from pyproj import Transformer

    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    return Grid(crs, from_origin(x - n * res / 2, y + n * res / 2, res, res), n, n)


# --------------------------------------------------------------------------- embeddings


def test_dequantize_matches_published_mapping():
    raw = np.array([-127, -64, 0, 64, 127], np.int8)
    want = ((raw / 127.5) ** 2) * np.sign(raw)
    assert np.allclose(embeddings.dequantize(raw), want)


def _scene_embeddings(n=100, seed=0):
    """Desert background with a tight cluster of 'tank farm' pixels in three places."""
    rng = np.random.default_rng(seed)
    common = rng.normal(size=64)
    bg = common + 0.25 * rng.normal(size=(n, n, 64))
    farm = common + 0.25 * rng.normal(size=64) + 1.2 * rng.normal(size=64) / 8
    e = bg.copy()
    for r, c in ((20, 20), (70, 30), (40, 80)):
        e[r : r + 6, c : c + 6] = farm + 0.03 * rng.normal(size=(6, 6, 64))
    return e / np.linalg.norm(e, axis=-1, keepdims=True), farm / np.linalg.norm(farm)


def test_find_similar_whitened_finds_lookalikes_not_desert(monkeypatch):
    emb, farm = _scene_embeddings()
    grid = _grid()
    monkeypatch.setattr(embeddings, "read", lambda aoi, year, **kw: (emb.astype(np.float32), grid))
    monkeypatch.setattr(embeddings, "at_points", lambda pts, year, radius_m=15.0: np.array([farm] * len(pts), np.float32))
    from pyproj import Transformer

    tr = Transformer.from_crs(grid.crs, 4326, always_xy=True)
    lon, lat = tr.transform(*(grid.transform @ (23, 23)))  # the first farm is the example
    r = embeddings.find_similar(AOI.from_point(25, 55, 0.5), [(lat, lon)], 2025)
    assert len(r.matches) == 2, [(m.lat, m.lon, m.score) for m in r.matches]
    assert all(m.score > 0.6 for m in r.matches)
    assert r.stats["background_median"] < 0.3 and r.stats["example_facility_km2"] > 0
    # raw cosine can't separate them: everything shares the common component
    assert float(np.median(emb.reshape(-1, 64) @ farm)) > 0.6
    assert r.png()[:4] == b"\x89PNG"


def test_semantic_change_flags_only_the_changed_block(monkeypatch):
    rng = np.random.default_rng(1)
    a = rng.normal(size=(80, 80, 64))
    b = a + 0.05 * rng.normal(size=a.shape)
    b[30:45, 30:45] = rng.normal(size=(15, 15, 64))  # a new port
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    grid = _grid(80)
    monkeypatch.setattr(embeddings, "read", lambda aoi, year, **kw: ((a if year == 2017 else b).astype(np.float32), grid))
    r = embeddings.semantic_change(AOI.from_point(25, 55, 0.4), 2017, 2025)
    assert len(r.matches) == 1
    assert abs(r.matches[0].area_m2 - 15 * 15 * 100) < 4000
    assert 0.03 < r.stats["share_changed"] < 0.05


def test_embedding_view_and_segments(monkeypatch):
    emb, _ = _scene_embeddings(100)
    monkeypatch.setattr(embeddings, "read", lambda aoi, year, **kw: (emb.astype(np.float32), _grid(100)))
    v = embeddings.embedding_view(AOI.from_point(25, 55, 0.3), 2025)
    assert v.image.shape == (100, 100, 4) and v.image[..., 3].min() > 0
    k = embeddings.embedding_view(AOI.from_point(25, 55, 0.3), 2025, segments=4)
    assert k.stats["segments"] == 4 and abs(sum(k.stats["share"]) - 1) < 1e-6


def test_index_build_and_lookup(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    monkeypatch.setattr(embeddings, "AEF_DIR", tmp_path)
    monkeypatch.setattr(embeddings, "INDEX_DB", tmp_path / "index.db")
    pq.write_table(
        pa.table(
            {
                "path": [
                    "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2025/40N/a.tiff",
                    "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2024/40N/b.tiff",
                ],
                "year": [2025, 2024],
                "crs": ["EPSG:32640", "EPSG:32640"],
                "wgs84_west": [54.0, 54.0],
                "wgs84_south": [24.0, 24.0],
                "wgs84_east": [55.0, 55.0],
                "wgs84_north": [25.0, 25.0],
            }
        ),
        tmp_path / "aef_index.parquet",
    )
    assert embeddings.tiles_for((54.5, 24.5, 54.6, 24.6), 2025) == [
        ("https://data.source.coop/tge-labs/aef/v1/annual/2025/40N/a.tiff", "EPSG:32640")
    ]
    assert embeddings.tiles_for((56.5, 24.5, 56.6, 24.6), 2025) == []
    assert embeddings.available_years(24.5, 54.5) == [2024, 2025]


# --------------------------------------------------------------------------- super-resolution


def test_upscale_tiles_and_blends_without_seams(monkeypatch):
    import torch

    class Nearest(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest")

    monkeypatch.setattr(superres, "load", lambda variant, device=None: (Nearest(), "cpu"))
    x = np.random.default_rng(0).uniform(0, 0.3, (4, 300, 260)).astype(np.float32)
    y = superres.upscale(x, "lite")
    assert y.shape == (4, 1200, 1040)
    back = y.reshape(4, 300, 4, 260, 4).mean((2, 4))
    assert np.abs(back - x).max() < 1e-5  # blending across overlapping tiles is exact for a consistent model


def test_superres_rejects_non_sentinel2():
    sc = Scene("x", "landsat", "l9", "optical", T0, 30.0, (55, 25, 55.1, 25.1), Render("reflectance", ["a"]))
    with pytest.raises(ValueError):
        superres.enhance(sc, AOI.from_point(25.05, 55.05, 1))


# --------------------------------------------------------------------------- aircraft in flight


def _airborne_scene(monkeypatch, speed=200.0, heading=120.0, static=True, n=300):
    rng = np.random.default_rng(3)
    base = {b: 0.2 + 0.004 * rng.normal(size=(n, n)) for b in airborne.BAND_DT}
    r0, c0 = 150.0, 120.0
    vr, vc = -math.cos(math.radians(heading)) * speed / 10, math.sin(math.radians(heading)) * speed / 10  # px/s
    for b, dt in airborne.BAND_DT.items():
        r, c = r0 + vr * dt, c0 + vc * dt
        yy, xx = np.mgrid[0:n, 0:n]
        base[b] += 0.25 * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 1.2**2))
        if static:  # a bright rooftop: same place in every band
            base[b] += 0.25 * np.exp(-((yy - 60) ** 2 + (xx - 220) ** 2) / (2 * 1.2**2))
    grid = _grid(n)

    def fake_read(scene, name, g, *a, **k):
        return base[name].astype(np.float32)

    import oracle.change

    monkeypatch.setattr(oracle.change, "_read_band", fake_read)
    monkeypatch.setattr(airborne.Grid, "for_aoi", classmethod(lambda cls, aoi, res, max_pixels=0: grid))
    monkeypatch.setattr(airborne, "track_bearing", lambda scene: 193.0)
    sc = Scene("s", "sentinel-2", "sentinel-2b", "optical", T0, 10.0, (54.9, 24.9, 55.1, 25.1), Render("rgb8", ["x"]))
    sc.extra = {"bands": {}, "reflectance_scale": 1.0, "reflectance_offset": 0.0}
    return sc


def test_airborne_finds_the_mover_not_the_rooftop(monkeypatch):
    sc = _airborne_scene(monkeypatch)
    found = airborne.detect_airborne(sc, AOI.from_point(25, 55, 1.5))
    assert len(found) == 1
    a = found[0]
    assert abs(a.apparent_speed_ms - 200) < 15 and abs(a.apparent_heading_deg - 120) < 5
    assert a.bands_matched == 4


def test_airborne_ignores_slow_traffic(monkeypatch):
    sc = _airborne_scene(monkeypatch, speed=20.0)
    assert airborne.detect_airborne(sc, AOI.from_point(25, 55, 1.5)) == []


def test_parallax_solution_recovers_speed_and_altitude():
    speed, hdg, alt, track = 130.0, 300.0, 900.0, 193.0
    v = np.array([math.sin(math.radians(hdg)), math.cos(math.radians(hdg))]) * speed
    t = np.array([math.sin(math.radians(track)), math.cos(math.radians(track))])
    u = v - alt * airborne.PARALLAX_PER_M * t
    ac = airborne.Aircraft(0, 0, "", float(np.hypot(*u)), 0, 10, 4, 0.1)
    airborne._solve(ac, float(u[0]), float(u[1]), hdg % 180, track)
    assert abs(ac.speed_ms - speed) < 1 and abs(ac.altitude_m - alt) < 20 and abs(ac.heading_deg - hdg) < 1


def test_parallax_without_heading_brackets_speed():
    ac = airborne.Aircraft(0, 0, "", 136.0, 303.0, 10, 4, 0.1)
    airborne._solve(ac, -114.1, 74.1, None, 193.0)
    assert ac.speed_ms is None and "0-11 km" in ac.note


# --------------------------------------------------------------------------- agent v2


def test_tool_evidence_is_attributed_per_call_under_parallelism(store):
    tb = Toolbox(store)
    from concurrent.futures import ThreadPoolExecutor

    def run(i):
        tb._tl.ids = []
        for k in range(5):
            tb.add("derived", f"t{i}", f"fact {k}")
        return tb.collected()

    with ThreadPoolExecutor(4) as pool:
        out = list(pool.map(run, range(4)))
    assert sorted(e for ids in out for e in ids) == sorted(tb.evidence)
    assert all(len(ids) == 5 and len({tb.evidence[e].tool for e in ids}) == 1 for ids in out)


def test_web_citations_become_report_evidence(store):
    inv = agent.Investigation("q")
    inv.toolbox = Toolbox(store)
    cite = types.SimpleNamespace(
        type="web_search_result_location",
        url="https://www.example.org/news/1",
        title="Port expands",
        cited_text="New berths opened.",
    )
    block = types.SimpleNamespace(type="text", text="A new berth was reported", citations=[cite, cite])
    out = agent._with_citations(inv, block)
    assert out == "A new berth was reported[E1]"
    e = inv.toolbox.evidence["E1"]
    assert e.kind == "report" and e.links["url"] == cite.url and "example.org" in e.summary
    # the same source cited again reuses its id
    assert agent._with_citations(inv, block).endswith("[E1]")


class _Msgs:
    def __init__(self, script):
        self.script, self.calls = script, []

    def create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})
        return self.script.pop(0)


def test_llm_runs_parallel_tools_and_offers_web_search(store, monkeypatch):
    import anthropic

    store.save_site("dock", (55.0, 25.0, 55.1, 25.1), "maritime")
    B = types.SimpleNamespace
    script = [
        B(
            stop_reason="tool_use",
            content=[
                B(type="server_tool_use", id="s1", name="web_search", input={"query": "dock port news"}),
                B(type="tool_use", id="a", name="list_sites", input={}),
                B(type="tool_use", id="b", name="next_passes", input={"lat": 25.05, "lon": 55.05}),
            ],
        ),
        B(
            stop_reason="end_turn",
            content=[
                B(
                    type="text",
                    text="Port news says it grew",
                    citations=[B(url="https://news.example/x", title="Grew", cited_text="grew")],
                ),
                B(type="text", text=". Done.", citations=None),
            ],
        ),
    ]
    fake = _Msgs(script)
    monkeypatch.setattr(anthropic, "Anthropic", lambda: B(beta=B(messages=fake)))
    monkeypatch.setattr(
        Toolbox,
        "t_next_passes",
        lambda self, lat, lon, days=None: {"evidence": self.add("reference", "next_passes", "S2 tomorrow")},
    )
    inv = agent.investigate("What's happening at the dock?", store=store, use_llm=True)
    assert inv.status == "done", inv.error
    assert any(t.get("type") == "web_search_20260209" for t in fake.calls[0]["tools"])
    results = fake.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert inv.answer.startswith("Port news says it grew[E") and inv.answer.endswith(". Done.")
    kinds = {e.kind for e in inv.toolbox.evidence.values()}
    assert "report" in kinds and "reference" in kinds
    assert [s["tool"] for s in inv.steps if s["type"] == "tool"][0] == "web_search"


def test_llm_without_web(store, monkeypatch):
    import anthropic

    B = types.SimpleNamespace
    fake = _Msgs([B(stop_reason="end_turn", content=[B(type="text", text="ok", citations=None)])])
    monkeypatch.setattr(anthropic, "Anthropic", lambda: B(beta=B(messages=fake)))
    agent.investigate("anything at 1,2?", store=store, use_llm=True, web=False)
    assert all(t.get("name") != "web_search" for t in fake.calls[0]["tools"])


def test_playbook_routes_new_intents():
    assert {"similar"} <= agent.intents("find other places like this refinery")
    assert {"longterm", "change"} <= agent.intents("how has the port developed since 2019")
    assert "flying" in agent.intents("any planes in flight near Dubai?")


def test_find_similar_tool_registers_matches(store, monkeypatch):
    emb, farm = _scene_embeddings()
    grid = _grid()
    monkeypatch.setattr(embeddings, "read", lambda aoi, year, **kw: (emb.astype(np.float32), grid))
    monkeypatch.setattr(embeddings, "at_points", lambda pts, year, radius_m=15.0: np.array([farm] * len(pts), np.float32))
    tb = Toolbox(store)
    r = tb.call("find_similar", {"where": "25,55", "radius_km": 1, "examples": ["25.0,55.0"]})
    assert r["matches"] and all(m["evidence"] in tb.evidence for m in r["matches"])
    assert tb.evidence[r["evidence"]].links["overlay"].startswith("/api/embed/")
