"""Smoke tests against the real public catalogs. Run with: pytest -m live"""

from datetime import datetime, timedelta, timezone

import pytest

from oracle.geo import parse_aoi
from oracle.imagery import chip
from oracle.search import search

pytestmark = pytest.mark.live
NOW = datetime.now(timezone.utc)


@pytest.mark.parametrize("source", ["sentinel-2", "sentinel-1", "landsat"])
def test_dynamic_sources_return_recent_scenes(source):
    res = search(parse_aoi("1.264,103.84", 2), NOW - timedelta(days=60), NOW, sources=[source], limit=3)
    assert not res.errors and res.scenes


def test_sentinel2_chip_renders():
    aoi = parse_aoi("1.264,103.84", 1)
    s = search(aoi, NOW - timedelta(days=90), NOW, sources=["sentinel-2"], max_cloud=30, limit=5).scenes[0]
    c = chip(s, aoi, max_pixels=256)
    assert c.valid_fraction > 0.9


def test_wayback_has_sub_metre_history():
    res = search(parse_aoi("38.8977,-77.0365", 0.5), datetime(2014, 1, 1, tzinfo=timezone.utc), NOW, sources=["wayback"])
    assert any(s.gsd <= 0.6 for s in res.scenes)
