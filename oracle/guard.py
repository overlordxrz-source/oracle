"""Allowlist for URLs the web API will read on a client's behalf (no SSRF)."""

from __future__ import annotations

from urllib.parse import urlparse

from fastapi import HTTPException

from .models import Scene

# Tile/chip/detect endpoints read whatever hrefs a request names, so only the public data
# buckets Oracle's sources actually use are allowed, never your local network.
ALLOWED_HOSTS = {
    "e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com",
    "sentinel-cogs.s3.us-west-2.amazonaws.com",
    "maxar-opendata.s3.amazonaws.com",
    "maxar-opendata.s3.us-west-2.amazonaws.com",
    "capella-open-data.s3.amazonaws.com",
    "capella-open-data.s3.us-west-2.amazonaws.com",
    "umbra-open-data-catalog.s3.us-west-2.amazonaws.com",
    "umbra-open-data-catalog.s3.amazonaws.com",
}
ALLOWED_SUFFIXES = (".blob.core.windows.net",)  # Planetary Computer storage accounts


def check_href(href: str) -> None:
    u = urlparse(href)
    host = (u.hostname or "").lower()
    if u.scheme != "https" or not (host in ALLOWED_HOSTS or host.endswith(ALLOWED_SUFFIXES)):
        raise HTTPException(400, f"host not allowed: {host or href}")


def check_scene(scene: Scene) -> None:
    if scene.render.kind != "xyz":
        for h in scene.render.hrefs:
            check_href(h)
    for bands in [scene.extra.get("bands") or {}, *(scene.extra.get("band_mosaic") or [])]:
        for h in bands.values():
            check_href(h)
    if scene.extra.get("cross_pol"):
        check_href(scene.extra["cross_pol"])
