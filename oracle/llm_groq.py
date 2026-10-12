"""Groq provider for Oracle: OpenAI-compatible chat with tool calling.

Uses the smartest tool-capable model on the account by default
(``openai/gpt-oss-120b``). No new dependencies: plain httpx (already required).

Env:
  GROQ_API_KEY        required
  ORACLE_GROQ_MODEL   override, e.g. openai/gpt-oss-20b or qwen/qwen3.8-27b
  ORACLE_GROQ_BASE    override (default https://api.groq.com/openai/v1)
"""

from __future__ import annotations

import json
import os

GROQ_BASE = os.environ.get("ORACLE_GROQ_BASE", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_MODEL = os.environ.get("ORACLE_GROQ_MODEL", "openai/gpt-oss-120b")

# Models on this key known to support tools, smartest first.
PREFERENCE = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]


def has_credentials() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def api_key() -> str:
    k = os.environ.get("GROQ_API_KEY", "")
    if not k:
        raise RuntimeError("no Groq credentials (set GROQ_API_KEY)")
    return k


def to_openai_tools(anthropic_tools: list[dict]) -> list[dict]:
    """Anthropic {name, description, input_schema} -> OpenAI function tools."""
    out = []
    for t in anthropic_tools:
        if t.get("type", "").startswith("web_search"):
            continue  # no server-side web search on Groq; skip
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return out


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
    max_tokens: int = 8000,
    temperature: float = 0.2,
    timeout: float = 120,
) -> dict:
    """POST /chat/completions, return the raw response dict. Raises RuntimeError.

    Retries 429s with backoff (honours Retry-After): the free tier throttles
    fast when every turn resends all 17 tool schemas.
    """
    import time

    import httpx

    payload: dict = {
        "model": model or GROQ_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    waits = [5, 15, 30, 60, 60]
    attempt = 0
    while True:
        try:
            r = httpx.post(
                f"{GROQ_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
        except Exception as exc:
            raise RuntimeError(f"could not reach Groq API: {exc}") from exc
        if r.status_code in (401, 403):
            raise RuntimeError("Groq credentials were rejected (check GROQ_API_KEY)")
        if r.status_code == 429 and attempt < len(waits):
            try:
                retry_after = float(r.headers.get("retry-after", "nan"))
            except ValueError:
                retry_after = float("nan")
            import math

            wait = retry_after if math.isfinite(retry_after) and retry_after > 0 else waits[attempt]
            time.sleep(wait)
            attempt += 1
            continue
        if r.status_code == 429:
            raise RuntimeError("Groq rate limit still hit after retries; wait a minute and try again")
        break
    if r.status_code >= 400:
        try:
            detail = r.json().get("error", {}).get("message", r.text[:300])
        except Exception:
            detail = r.text[:300]
        raise RuntimeError(f"Groq API error {r.status_code}: {detail}")
    try:
        return r.json()
    except Exception as exc:
        raise RuntimeError(f"bad Groq response: {exc}") from exc


def simple_answer(system: str, user: str, model: str | None = None, max_tokens: int = 4000) -> str:
    """One-shot answer without tools (briefs)."""
    data = chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
        max_tokens=max_tokens,
    )
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError) as exc:
        raise RuntimeError(f"bad Groq response shape: {exc}") from exc


def pick_smartest(timeout: float = 30) -> str:
    """Return the smartest tool-capable model actually visible to this key."""
    import httpx

    try:
        r = httpx.get(
            f"{GROQ_BASE}/models",
            headers={"Authorization": f"Bearer {api_key()}"},
            timeout=timeout,
        )
        r.raise_for_status()
        ids = {m["id"] for m in r.json().get("data", [])}
    except Exception:
        return GROQ_MODEL
    for m in PREFERENCE:
        if m in ids:
            return m
    # else: any model advertising tools support
    try:
        for m in r.json().get("data", []):
            if "tools" in (m.get("supported_features") or []):
                return m["id"]
    except Exception:
        pass
    return GROQ_MODEL


__all__ = ["GROQ_BASE", "GROQ_MODEL", "has_credentials", "chat", "simple_answer", "to_openai_tools", "pick_smartest"]
