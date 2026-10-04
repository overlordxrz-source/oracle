"""HTTP helpers: one identity, retries, and a bounded-concurrency async fetcher."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Iterable
from typing import Any

import httpx

from .config import HTTP_TIMEOUT, USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}


def client() -> httpx.Client:
    return httpx.Client(
        headers=HEADERS,
        timeout=HTTP_TIMEOUT,
        follow_redirects=True,
        transport=httpx.HTTPTransport(retries=3),
    )


def get_json(url: str, **kw: Any) -> Any:
    with client() as c:
        r = c.get(url, **kw)
        r.raise_for_status()
        return r.json()


def fetch_many_json(
    urls: Iterable[str],
    concurrency: int = 32,
    progress: str | None = None,
) -> dict[str, Any]:
    """GET many JSON documents concurrently. Failures map to ``None`` (logged, not raised)."""
    urls = list(dict.fromkeys(urls))
    return asyncio.run(_fetch_many(urls, concurrency, progress))


async def _fetch_many(urls: list[str], concurrency: int, progress: str | None) -> dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    out: dict[str, Any] = {}
    done = 0
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(
        headers=HEADERS,
        timeout=HTTP_TIMEOUT,
        follow_redirects=True,
        limits=limits,
        transport=httpx.AsyncHTTPTransport(retries=3, limits=limits),
    ) as c:

        async def one(u: str) -> None:
            nonlocal done
            async with sem:
                try:
                    r = await c.get(u)
                    r.raise_for_status()
                    out[u] = r.json()
                except Exception as exc:  # noqa: BLE001 - one bad document must not kill a crawl
                    out[u] = None
                    _log(f"  ! {u}: {exc}")
                done += 1
                if progress and (done % 200 == 0 or done == len(urls)):
                    _log(f"  {progress}: {done}/{len(urls)}")

        await asyncio.gather(*(one(u) for u in urls))
    return out


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


log: Callable[[str], None] = _log
