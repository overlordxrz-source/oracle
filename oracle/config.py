"""Runtime configuration: cache locations, HTTP identity, GDAL tuning."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["CACHE_DIR", "INDEX_DIR", "USER_AGENT", "HTTP_TIMEOUT", "configure_gdal"]

CACHE_DIR = Path(os.environ.get("ORACLE_CACHE", Path.home() / ".cache" / "oracle"))
INDEX_DIR = CACHE_DIR / "index"

# Several public endpoints (Esri, Nominatim) reject anonymous library user agents.
# Identify honestly instead of impersonating a browser.
USER_AGENT = os.environ.get("ORACLE_USER_AGENT", "Oracle/0.1 (+https://github.com/overlordxrz-source/oracle)")
HTTP_TIMEOUT = float(os.environ.get("ORACLE_HTTP_TIMEOUT", "60"))

# Re-crawl static catalogs (Maxar / Capella / Umbra) after this many days.
INDEX_MAX_AGE_DAYS = float(os.environ.get("ORACLE_INDEX_MAX_AGE_DAYS", "7"))

_GDAL_DEFAULTS = {
    # Don't LIST the bucket "directory" next to every remote COG we open.
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.TIF,.TIFF",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_CACHEMAX": "512",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "100000000",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_HTTP_USERAGENT": USER_AGENT,
    # Public buckets: never try to sign /vsis3 requests with whatever creds are lying around.
    "AWS_NO_SIGN_REQUEST": "YES",
}


def configure_gdal() -> None:
    """Apply sane defaults for streaming cloud-optimized GeoTIFFs (user env wins)."""
    for k, v in _GDAL_DEFAULTS.items():
        os.environ.setdefault(k, v)
    # libcurl inside GDAL reads CURL_CA_BUNDLE; fall back to the requests/SSL bundle.
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        if os.environ.get(var) and not os.environ.get("CURL_CA_BUNDLE"):
            os.environ["CURL_CA_BUNDLE"] = os.environ[var]


configure_gdal()
