"""Push new events to a webhook (Discord, Slack, ntfy, or anything that accepts JSON)."""

from __future__ import annotations

import os

import httpx

from .config import USER_AGENT
from .http import log


def notify(events: list[dict], webhook: str | None = None, min_severity: float = 0.5) -> int:
    url = webhook or os.environ.get("ORACLE_WEBHOOK")
    if not url:
        return 0
    sent = 0
    for e in sorted(events, key=lambda e: -e.get("severity", 0)):
        if e.get("severity", 0) < min_severity:
            continue
        d = e.get("detail") or {}
        where = f" ({d['lat']:.4f}, {d['lon']:.4f})" if "lat" in d else ""
        text = f"Oracle [{e['kind']}] {e['title']}{where} {e['time'][:16]}Z"
        try:
            httpx.post(url, json={"content": text, "text": text, "event": e}, headers={"User-Agent": USER_AGENT}, timeout=20)
            sent += 1
        except httpx.HTTPError as exc:
            log(f"  webhook failed: {exc}")
    return sent
