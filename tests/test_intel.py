"""Offline tests for the intelligence layer: store, tracker, events, AIS, brief, pipeline, API."""

import math
import sys
import types
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from oracle import ais, analytics
from oracle.brief import digest, render_llm, render_markdown
from oracle.objdet import _to_obs, dedupe, default_scales
from oracle.observations import AIRCRAFT, VEHICLE, VESSEL, Observation
from oracle.pipeline import Site, detector_for
from oracle.store import Store
from oracle.tracking import Track, associate, likelihood_ratio, track_site
from oracle.velocity import wake_motion

T0 = datetime(2026, 8, 1, 3, 30, tzinfo=timezone.utc)
SITE_AREA = 8_000 * 8_000.0


def ob(lat, lon, t, scene, cls=VESSEL, length=200.0, source="sentinel-2", **attrs):
    return Observation(
        cls=cls,
        lat=lat,
        lon=lon,
        time=t,
        scene_id=scene,
        source=source,
        detector="test",
        length_m=length,
        width_m=length / 6,
        attrs={"gsd": 10.0, **attrs},
    )


def moved(lat, lon, north_m, east_m):
    return lat + north_m / 110_574, lon + east_m / (111_320 * math.cos(math.radians(lat)))


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "t.db")


# --------------------------------------------------------------------------- tracker


def test_stationary_object_links_across_dates_with_high_probability():
    a = ob(1.25, 103.9, T0, "s1")
    b = ob(*moved(1.25, 103.9, 8, -5), T0 + timedelta(days=3), "s2", length=210)
    other = ob(*moved(1.25, 103.9, 3000, 2000), T0 + timedelta(days=3), "s2", length=90)
    tracks, alts = track_site([a, b, other], SITE_AREA)
    by_obs = {o.id: t.id for t in tracks for o in t.obs}
    assert by_obs[a.id] == by_obs[b.id] != by_obs[other.id]
    assert b.attrs["link_prob"] > 0.8


def test_different_size_object_at_same_spot_is_not_the_same():
    a = ob(1.25, 103.9, T0, "s1", length=330)
    b = ob(1.25, 103.9, T0 + timedelta(days=4), "s2", length=60)
    tracks, _ = track_site([a, b], SITE_AREA)
    assert len(tracks) == 2


def test_moving_vessel_needs_kinematic_feasibility():
    a = ob(1.25, 103.9, T0, "s1", underway=True, course_deg=90.0)
    a.course_deg = 90.0
    near = ob(*moved(1.25, 103.9, 0, 9000), T0 + timedelta(hours=1), "s2", underway=True)  # 9 km/h east: feasible
    far = ob(*moved(1.25, 103.9, 0, 200_000), T0 + timedelta(hours=1), "s2")  # 200 km in 1 h: impossible
    t = Track("T", "vessel", [a], [1.0])
    assert likelihood_ratio(t, near, SITE_AREA * 100) > 1
    assert likelihood_ratio(t, far, SITE_AREA * 100) == 0


def test_association_probabilities_are_normalised_and_compete():
    a1, a2 = ob(1.25, 103.9, T0, "s1"), ob(*moved(1.25, 103.9, 60, 0), T0, "s1")
    t1, t2 = Track("T1", "vessel", [a1], [1.0]), Track("T2", "vessel", [a2], [1.0])
    b = ob(*moved(1.25, 103.9, 30, 0), T0 + timedelta(days=1), "s2")  # halfway between: ambiguous
    chosen, cands = associate([t1, t2], [b], SITE_AREA)
    probs = list(cands[0].values())
    assert all(0 <= p <= 1 for p in probs) and sum(probs) <= 1.0001
    assert max(probs) < 0.8  # genuinely ambiguous -> no overconfident link


def test_missed_in_clear_coverage_marks_departed_and_ids_are_deterministic():
    a = ob(1.25, 103.9, T0, "s1")
    filler = [ob(*moved(1.25, 103.9, 3000, 3000), T0 + timedelta(days=d), f"s{d}", length=90) for d in (2, 4)]
    cov = {f"s{d}": (103.8, 1.2, 104.0, 1.3) for d in (1, 2, 4)}
    tracks, _ = track_site([a, *filler], SITE_AREA, cov)
    ta = next(t for t in tracks if t.obs[0].id == a.id)
    assert ta.status == "departed"
    again, _ = track_site([a, *filler], SITE_AREA, cov)
    assert sorted(t.id for t in tracks) == sorted(t.id for t in again)


# --------------------------------------------------------------------------- events


def test_arrivals_need_a_clear_baseline_and_busy_turnover_is_rolled_up():
    obs = [ob(1.25, 103.9, T0, "base", length=300)]
    later = T0 + timedelta(days=5)
    obs += [ob(*moved(1.25, 103.9, 500 * i, 0), later, "new", length=100 + 40 * i) for i in range(1, 8)]
    tracks, _ = track_site(obs, SITE_AREA)
    times = {"base": T0, "new": later}
    ev = analytics.generate("S", tracks, obs, times, {"base": 1.0, "new": 1.0})
    arrivals = [e for e in ev if e["kind"] == "arrival"]
    rollups = [e for e in ev if e["kind"] == "arrivals"]
    assert len(arrivals) == analytics.MAX_INDIVIDUAL and len(rollups) == 1 and rollups[0]["detail"]["count"] == 7
    # same data, but the first look was cloudy -> no baseline -> no arrivals
    ev2 = analytics.generate("S", tracks, obs, times, {"base": 0.2, "new": 0.2})
    assert not [e for e in ev2 if e["kind"].startswith("arrival")]


def test_count_spike_detected_against_site_history():
    obs = []
    times = {}
    for d in range(8):
        sid = f"sentinel-2-{d}"
        times[sid] = T0 + timedelta(days=5 * d)
        n = 40 if d == 7 else 10 + (d % 2)
        obs += [ob(1.25 + i * 1e-3, 103.9, times[sid], sid, length=100) for i in range(n)]
    ev = analytics.count_anomalies("S", obs, times)
    assert any(e["kind"] == "count_spike" and e["detail"]["count"] == 40 for e in ev)


# --------------------------------------------------------------------------- store + AIS


def test_store_roundtrip_bbox_query_tracking_and_events(store):
    a = ob(1.25, 103.9, T0, "s1")
    b = ob(10.0, 50.0, T0, "s1")
    assert store.add_observations([a, b], "S") == 2
    assert store.add_observations([a], "S") == 0  # idempotent
    got = store.observations(bbox=(103.0, 1.0, 104.0, 2.0))
    assert [o.id for o in got] == [a.id]
    store.save_tracking(
        "S",
        [
            {
                "id": "T1",
                "site": "S",
                "cls": VESSEL,
                "status": "active",
                "first_seen": T0.isoformat(),
                "last_seen": T0.isoformat(),
                "n_obs": 1,
                "lat": 1.25,
                "lon": 103.9,
            }
        ],
        {a.id: ("T1", 1.0)},
        {a.id: {"T9": 0.2}},
    )
    t = store.track("T1")
    assert t["observations"][0]["id"] == a.id and t["alternatives"][0]["track_id"] == "T9"
    ev = [{"id": "e1", "site": "S", "time": T0.isoformat(), "kind": "arrival", "title": "x"}]
    assert store.replace_events("S", ev) == 1
    assert store.replace_events("S", ev) == 0
    assert store.replace_events("S", []) == 0 and not store.events(site="S")


def test_ais_match_and_dark_vessel(store, tmp_path):
    lit = ob(1.25, 103.9, T0, "s1", length=180)
    dark = ob(*moved(1.25, 103.9, 4000, 4000), T0, "s1", length=250)
    csv = tmp_path / "ais.csv"
    lines = ["MMSI,BaseDateTime,LAT,LON,SOG,COG,VesselName,Length"]
    for dt_min in (-6, 6):  # straddles the image time; vessel moving north at ~0 kn for simplicity
        t = (T0 + timedelta(minutes=dt_min)).strftime("%Y-%m-%dT%H:%M:%S")
        lat, lon = moved(1.25, 103.9, 40, -30)
        lines.append(f"563000001,{t},{lat},{lon},0.1,10,TEST TANKER,185")
    csv.write_text("\n".join(lines))
    assert ais.load_csv(store, csv) == 2
    store.add_observations([lit, dark], "S")
    res = ais.match(store, [lit, dark])
    assert res[lit.id]["mmsi"] == 563000001 and res[lit.id]["distance_m"] < 100
    assert res[dark.id]["dark"] is True
    assert store.observations(site="S", bbox=(103.92, 1.26, 103.95, 1.30))[0].attrs["dark"] is True


# --------------------------------------------------------------------------- brief


def test_digest_and_markdown_brief(store):
    o = ob(1.25, 103.9, T0, "s1", length=320)
    store.save_site("S", (103.8, 1.2, 104.0, 1.3), "maritime")
    store.add_observations([o], "S")
    store.replace_events(
        "S",
        [
            {
                "id": "e1",
                "site": "S",
                "time": T0.isoformat(),
                "kind": "arrival",
                "severity": 0.85,
                "title": "New 320 m vessel at S",
                "detail": {"lat": 1.25, "lon": 103.9},
            }
        ],
    )
    d = digest(store, T0 - timedelta(days=1))
    assert d["sites"][0]["events"][0]["title"].startswith("New 320 m")
    md = render_markdown(d)
    assert "New 320 m vessel" in md and "1.2500, 103.9000" in md


def test_llm_brief_uses_digest_and_handles_refusal(monkeypatch):
    calls = {}

    class Block:
        type = "text"
        text = "## BLUF\n- something"

    class Msg:
        content = [Block()]
        stop_reason = "end_turn"

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return Msg()

    class Messages:
        def stream(self, **kw):
            calls.update(kw)
            return Stream()

    fake = types.SimpleNamespace(
        Anthropic=lambda: types.SimpleNamespace(beta=types.SimpleNamespace(messages=Messages())),
        AuthenticationError=type("A", (Exception,), {}),
        RateLimitError=type("R", (Exception,), {}),
        APIStatusError=type("S", (Exception,), {}),
        APIConnectionError=type("C", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    out = render_llm({"sites": [], "since": "x", "generated": "y"})
    assert out.startswith("## BLUF")
    assert calls["model"] == "claude-opus-5-5" and calls["thinking"] == {"type": "adaptive"}
    assert '"sites":[]' in calls["messages"][0]["content"]
    Msg.stop_reason = "refusal"
    with pytest.raises(RuntimeError, match="declined"):
        render_llm({"sites": []})


# --------------------------------------------------------------------------- pipeline / objdet / wake


def test_site_kinds_pick_detectors_and_sources():
    from oracle.models import Render, Scene

    s = Site("x", (0, 0, 1, 1), "naval")
    assert s.detectors == ["ships", "objects"] and {"sentinel-2", "maxar", "naip"} <= set(s.sources)
    sc = lambda src, sensor, gsd: Scene(
        id="i",
        source=src,
        platform="p",
        sensor=sensor,
        datetime=T0,
        gsd=gsd,  # noqa: E731
        bbox=(0, 0, 1, 1),
        render=Render("rgb8", ["h"]),
    )
    assert detector_for(sc("sentinel-2", "optical", 10), s) == "ships"
    assert detector_for(sc("maxar", "optical", 0.3), s) == "objects"
    assert detector_for(sc("landsat", "optical", 30), s) is None
    assert detector_for(sc("maxar", "optical", 0.3), Site("y", (0, 0, 1, 1), "maritime")) is None


def test_yolo_postprocessing_geometry_and_dedupe():
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_origin

    from oracle.imagery import Grid

    grid = Grid(rasterio.crs.CRS.from_epsg(32612), from_origin(500000, 3560000, 0.5, 0.5), 2048, 2048)
    tr = Transformer.from_crs(grid.crs, 4326, always_xy=True)
    pts = np.array([[90, 100], [110, 100], [110, 104], [90, 104]], float)
    det = {"cx": 100.0, "cy": 102.0, "w": 80.0, "h": 20.0, "rot": 0.0, "pts": pts, "conf": 0.9, "label": "plane"}
    o = _to_obs(det, 0, 0, grid, tr, "scene", "naip", T0, "yolo")
    assert o.cls == AIRCRAFT and o.length_m == 40.0 and o.width_m == 10.0
    assert o.axis_deg == pytest.approx(90.0)  # long side along image x = east-west
    dup = Observation(**{**o.__dict__, "id": "", "confidence": 0.5, "lat": o.lat + 1e-6})
    car = Observation(**{**o.__dict__, "id": "", "cls": VEHICLE})
    assert {x.cls for x in dedupe([o, dup, car])} == {AIRCRAFT, VEHICLE} and len(dedupe([o, dup, car])) == 2
    assert default_scales(0.3)[0][0] < 0.2 and default_scales(1.0) == [(1.0, None)]


def test_wake_gives_direction_of_travel():
    res = 10.0
    blue = np.full((80, 80), 0.05)
    nir = np.full((80, 80), 0.01)
    hull = np.zeros((80, 80), bool)
    hull[38:41, 30:46] = True  # 160 m hull along east-west, centred near col 38
    nir[hull] = 0.15
    blue[hull] = 0.08
    blue[38:41, 10:30] = 0.2  # bright foam trailing to the WEST -> heading east
    background = ~hull
    background[38:41, 5:50] = False
    w = wake_motion(blue, nir, hull, background, axis_deg=90.0, length_m=160, width_m=30, res=res)
    assert w.underway and w.course_deg == pytest.approx(90.0, abs=1) and w.wake_m > 50
    calm = wake_motion(np.full((80, 80), 0.05), nir, hull, background, 90.0, 160, 30, res)
    assert not calm.underway


# --------------------------------------------------------------------------- API


def test_intel_api_sites_objects_tracks(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from oracle import api_intel
    from oracle.server import app

    st = Store(tmp_path / "api.db")
    monkeypatch.setattr(api_intel, "store", lambda: st)
    c = TestClient(app)
    assert c.post("/api/sites", json={"name": "S", "bbox": [103.8, 1.2, 104.0, 1.3], "kind": "maritime"}).status_code == 200
    a = ob(1.25, 103.9, T0, "s1")
    b = ob(*moved(1.25, 103.9, 5, 5), T0 + timedelta(days=2), "s2")
    st.add_observations([a, b], "S")
    from oracle.pipeline import refresh_site

    refresh_site(st, Site.from_row(st.sites()[0]))
    objs = c.get("/api/objects", params={"bbox": "103.8,1.2,104.0,1.3"}).json()
    assert len(objs["features"]) == 2
    tracks = c.get("/api/tracks", params={"min_obs": 2}).json()["features"]
    assert len(tracks) == 1 and tracks[0]["geometry"]["type"] == "LineString"
    assert c.get(f"/api/track/{tracks[0]['id']}").json()["n_obs"] == 2
    assert c.get(f"/api/obs/{a.id}").json()["id"] == a.id
    assert c.get("/api/sites").json()[0]["observations"] == 2
    assert c.delete("/api/sites/S").json()["deleted"] is True
    assert (
        c.post(
            "/api/detect",
            json={
                "scene": {
                    "id": "x",
                    "source": "sentinel-2",
                    "platform": "p",
                    "datetime": T0.isoformat(),
                    "gsd": 10,
                    "bbox": [0, 0, 1, 1],
                    "render": {"kind": "rgb8", "hrefs": ["http://10.0.0.1/x.tif"]},
                },
                "bbox": [0, 0, 0.01, 0.01],
            },
        ).status_code
        == 400
    )  # SSRF guard
