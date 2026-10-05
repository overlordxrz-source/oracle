"""Tiny in-process background job runner for slow API work (YOLO, sweeps, briefs)."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="oracle-job")
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
MAX_KEEP = 200


def submit(kind: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    jid = uuid.uuid4().hex[:12]
    job = {"id": jid, "kind": kind, "status": "queued", "created": time.time(), "result": None, "error": None}
    with _lock:
        _jobs[jid] = job
        if len(_jobs) > MAX_KEEP:
            for old in sorted(_jobs.values(), key=lambda j: j["created"])[: len(_jobs) - MAX_KEEP]:
                _jobs.pop(old["id"], None)

    def run() -> None:
        job["status"] = "running"
        job["started"] = time.time()
        try:
            job["result"] = fn(*args, **kwargs)
            job["status"] = "done"
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
            job["trace"] = traceback.format_exc(limit=5)
        job["finished"] = time.time()

    _pool.submit(run)
    return jid


def get(jid: str) -> dict | None:
    return _jobs.get(jid)


def all_jobs() -> list[dict]:
    return sorted(({k: v for k, v in j.items() if k != "result"} for j in _jobs.values()), key=lambda j: -j["created"])
