"""Local index for *static* STAC catalogs (Maxar, Capella, Umbra).

Those publishers expose a tree of JSON files on S3 rather than a search API, so
Oracle crawls them once, keeps a compact record per scene on disk, and searches
that. Records are plain dicts; each source decides what to keep.
"""

from __future__ import annotations

import gzip
import json
import time
from collections.abc import Callable
from pathlib import Path

from ..config import INDEX_DIR, INDEX_MAX_AGE_DAYS
from ..http import log


class StaticIndex:
    def __init__(self, name: str, builder: Callable[[], list[dict]]):
        self.name = name
        self.builder = builder
        self._records: list[dict] | None = None

    @property
    def path(self) -> Path:
        return INDEX_DIR / f"{self.name}.json.gz"

    def age_days(self) -> float | None:
        if not self.path.exists():
            return None
        return (time.time() - self.path.stat().st_mtime) / 86400

    def records(self, refresh: bool = False) -> list[dict]:
        if self._records is not None and not refresh:
            return self._records
        age = self.age_days()
        if refresh or age is None or age > INDEX_MAX_AGE_DAYS:
            try:
                self.build()
            except Exception as exc:  # noqa: BLE001
                if age is None:
                    raise
                log(f"[{self.name}] refresh failed ({exc}); using index from {age:.1f} days ago")
        if self._records is None:
            with gzip.open(self.path, "rt") as f:
                self._records = json.load(f)
        return self._records

    def build(self) -> list[dict]:
        log(f"[{self.name}] crawling catalog (one-time, cached in {self.path}) ...")
        t0 = time.time()
        recs = self.builder()
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as f:
            json.dump(recs, f, separators=(",", ":"))
        tmp.replace(self.path)
        self._records = recs
        log(f"[{self.name}] indexed {len(recs)} scenes in {time.time() - t0:.0f}s")
        return recs
