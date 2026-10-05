"""Offline unit tests: geometry, ranking, pass merging, parsing."""

from datetime import datetime, timedelta, timezone

import pytest

from oracle.geo import AOI, bbox_overlap_fraction, lonlat_to_tile, parse_aoi, tile_bounds, utm_crs_for
from oracle.models import Render, Scene, parse_dt
from oracle.search import _union_coverage, merge_passes, rank
from oracle.sources.maxar import ard_tile_bbox
from oracle.sources.umbra import _from_metadata, _from_stac
from oracle.timelapse import period_key, pick_frames

T0 = datetime(2026, 3, 10, 4, 0, tzinfo=timezone.utc)


def scene(source="sentinel-2", gsd=10.0, dt=T0, cloud=None, cov=1.0, sid=None, bbox=(0, 0, 1, 1), kind="rgb8", platform="s2"):
    s = Scene(
        id=sid or f"{source}-{dt:%H%M%S}-{gsd}",
        source=source,
        platform=platform,
        sensor="optical",
        datetime=dt,
        gsd=gsd,
        bbox=bbox,
        cloud_cover=cloud,
        render=Render(kind=kind, hrefs=[f"https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/{sid or gsd}.tif"]),
    )
    s.extra["coverage"] = cov
    return s


# --------------------------------------------------------------------------- geo


def test_parse_point_makes_square_of_radius():
    a = parse_aoi("1.264, 103.84", radius_km=3)
    w, h = a.size_km()
    assert w == pytest.approx(6, rel=0.01) and h == pytest.approx(6, rel=0.01)
    assert a.center == pytest.approx((1.264, 103.84))


def test_parse_bbox_and_validation():
    assert parse_aoi("100.5,13.7,100.6,13.9").bbox == (100.5, 13.7, 100.6, 13.9)
    with pytest.raises(ValueError):
        parse_aoi("100.6,13.7,100.5,13.9")  # west > east
    with pytest.raises(ValueError):
        parse_aoi("120.0, 45.0")  # lat 120 is impossible -> swapped coordinates


def test_utm_zone_and_hemisphere():
    assert utm_crs_for(1.26, 103.84).to_epsg() == 32648
    assert utm_crs_for(-33.9, 151.2).to_epsg() == 32756


def test_tile_roundtrip():
    x, y = lonlat_to_tile(103.84, 1.264, 16)
    w, s, e, n = tile_bounds(x, y, 16)
    assert w <= 103.84 <= e and s <= 1.264 <= n


def test_overlap_fraction():
    assert bbox_overlap_fraction((0, 0, 2, 2), (1, 1, 3, 3)) == pytest.approx(0.25)
    assert bbox_overlap_fraction((0, 0, 1, 1), (2, 2, 3, 3)) == 0


def test_maxar_ard_grid_matches_published_item_bbox():
    # Item 103001010CB46500 (quadkey 122022102203, zone 47) publishes this bbox; the ARD
    # tile is that footprint plus a ~150 m buffer.
    w, s, e, n = ard_tile_bbox(47, "122022102203")
    published = (100.52850802591706, 13.838038465079542, 100.57457991878897, 13.86737334)
    assert w <= published[0] and s <= published[1] and e >= published[2] - 0.002 and n >= published[3]
    assert e - w == pytest.approx(0.0466, abs=0.002)  # 5 km at 13.8 N


def test_parse_dt_variants():
    assert parse_dt("2025-02-02 03:44:47Z") == datetime(2025, 2, 2, 3, 44, 47, tzinfo=timezone.utc)
    assert parse_dt("2026-09-03T17:08:15.833575Z").tzinfo is not None
    assert parse_dt("2023-10-02T02:35:00+00:00").hour == 2


# --------------------------------------------------------------------------- ranking / merging


def test_best_rank_prefers_resolution_then_coverage_then_cloud_then_recency():
    sub_m_partial = scene("maxar", 0.3, cov=0.2)
    sub_m_full = scene("maxar", 0.35, cov=1.0, dt=T0 - timedelta(days=30))
    s2_clear_old = scene(gsd=10, cloud=5, dt=T0 - timedelta(days=5))
    s2_cloudy_new = scene(gsd=10, cloud=90, dt=T0)
    out = rank([s2_cloudy_new, s2_clear_old, sub_m_partial, sub_m_full])
    assert out == [sub_m_full, sub_m_partial, s2_clear_old, s2_cloudy_new]


def test_rank_by_date_and_bad_sort():
    a, b = scene(dt=T0), scene(dt=T0 + timedelta(days=1))
    assert rank([a, b], "date") == [b, a]
    with pytest.raises(ValueError):
        rank([a], "nope")


def test_merge_passes_collapses_same_overpass_tiles():
    aoi = AOI((0, 0, 2, 1))
    left = scene(sid="T1", bbox=(-1, -1, 1, 2), cov=0.5)
    right = scene(sid="T2", bbox=(1, -1, 3, 2), cov=0.5, dt=T0 + timedelta(seconds=4))
    later = scene(sid="T3", bbox=(-1, -1, 3, 2), dt=T0 + timedelta(days=5))
    out = merge_passes([left, right, later], aoi)
    assert len(out) == 2
    merged = next(s for s in out if s.extra.get("merged_ids"))
    assert len(merged.render.hrefs) == 2
    assert merged.extra["coverage"] == pytest.approx(1.0)


def test_merge_passes_keeps_other_sources_and_platforms_apart():
    aoi = AOI((0, 0, 1, 1))
    a = scene(sid="A", platform="sentinel-2a")
    b = scene(sid="B", platform="sentinel-2b", dt=T0 + timedelta(seconds=10))
    c = scene("maxar", 0.3, sid="C")
    assert len(merge_passes([a, b, c], aoi)) == 3


def test_union_coverage():
    aoi = AOI((0, 0, 1, 1))
    assert _union_coverage(aoi, [(0, 0, 0.5, 1)]) == pytest.approx(0.5)
    assert _union_coverage(aoi, [(0, 0, 0.5, 1), (0.5, 0, 1, 1)]) == pytest.approx(1.0)


# --------------------------------------------------------------------------- source parsing


def test_umbra_stac_bbox_with_heights():
    rec = _from_stac({"bbox": [1, 2, 0, 3, 4, 100], "properties": {"datetime": "2025-01-01T00:00:00Z"}})
    assert rec["b"] == [1, 2, 3, 4]


def test_umbra_legacy_metadata_footprint():
    doc = {
        "targetIpr": 0.5,
        "umbraSatelliteName": "UMBRA_04",
        "collects": [
            {
                "startAtUTC": "2023-10-02T02:35:00+00:00",
                "footprintPolygonLla": {"coordinates": [[[10, 1, 5], [11, 1, 5], [11, 2, 5], [10, 2, 5], [10, 1, 5]]]},
            }
        ],
    }
    rec = _from_metadata(doc)
    assert rec["b"] == [10, 1, 11, 2] and rec["r"] == 0.5 and rec["pl"] == "UMBRA_04"


def test_scene_roundtrip():
    s = scene("umbra", 0.25)
    assert Scene.from_dict(s.to_dict()).to_dict() == s.to_dict()


# --------------------------------------------------------------------------- timelapse


def test_period_keys():
    d = datetime(2026, 3, 10, tzinfo=timezone.utc)
    assert period_key(d, "month") == "2026-03"
    assert period_key(d, "quarter") == "2026-Q1"
    assert period_key(d, "week") == "2026-W11"


def test_pick_frames_least_cloudy_per_period():
    a = scene(cloud=50, dt=datetime(2026, 1, 3, tzinfo=timezone.utc))
    b = scene(cloud=5, dt=datetime(2026, 1, 20, tzinfo=timezone.utc))
    c = scene(cloud=20, dt=datetime(2026, 2, 2, tzinfo=timezone.utc))
    partial = scene(cloud=0, dt=datetime(2026, 2, 9, tzinfo=timezone.utc), cov=0.4)
    assert pick_frames([a, b, c, partial], "month") == [b, c]


def test_worldcover_tile_names():
    from oracle.landmask import tile_names

    assert tile_names((56.25, 26.39, 56.65, 26.75)) == ["N24E054"]
    assert tile_names((-0.5, -0.5, 0.5, 0.5)) == ["S03W003", "S03E000", "N00W003", "N00E000"]
