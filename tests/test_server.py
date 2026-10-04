"""API guard rails (offline)."""

from fastapi.testclient import TestClient

from oracle.models import Render
from oracle.server import app, decode_spec, encode_spec

client = TestClient(app)


def test_spec_roundtrip():
    r = Render(kind="sar", hrefs=["https://capella-open-data.s3.amazonaws.com/data/x.tif"], power=False)
    assert decode_spec(encode_spec(r)) == r


def test_tiles_reject_hosts_outside_allowlist():
    for href in ("http://169.254.169.254/latest/meta-data", "https://evil.example.com/x.tif", "file:///etc/passwd"):
        spec = encode_spec(Render(kind="rgb8", hrefs=[href]))
        assert client.get(f"/api/tiles/12/1/1.png?spec={spec}").status_code == 400


def test_tiles_reject_garbage_spec():
    assert client.get("/api/tiles/12/1/1.png?spec=not-a-spec").status_code == 400


def test_ships_rejects_disallowed_band_hrefs():
    scene = {
        "id": "x",
        "source": "sentinel-2",
        "platform": "s2",
        "sensor": "optical",
        "datetime": "2026-01-01T00:00:00Z",
        "gsd": 10,
        "bbox": [0, 0, 1, 1],
        "render": {"kind": "rgb8", "hrefs": ["https://sentinel-cogs.s3.us-west-2.amazonaws.com/a.tif"]},
        "extra": {"bands": {"nir": "http://127.0.0.1:9999/secret.tif"}},
    }
    r = client.post("/api/ships", json={"scene": scene, "bbox": [0, 0, 0.01, 0.01]})
    assert r.status_code == 400 and "not allowed" in r.json()["detail"]


def test_index_and_sources():
    assert "ORACLE" in client.get("/").text
    keys = {s["key"] for s in client.get("/api/sources").json()}
    assert {"sentinel-2", "maxar", "umbra", "capella", "wayback"} <= keys
