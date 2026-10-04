"""Imagery sources. Every one is free and needs no account or API key."""

from __future__ import annotations

from .base import Source, SourceInfo
from .capella import Capella
from .earth_search import Sentinel2
from .maxar import Maxar
from .planetary import NAIP, Landsat, Sentinel1
from .umbra import Umbra
from .wayback import Wayback

SOURCES: dict[str, Source] = {
    s.info.key: s for s in (Wayback(), Maxar(), Umbra(), Capella(), NAIP(), Sentinel2(), Sentinel1(), Landsat())
}

__all__ = ["SOURCES", "Source", "SourceInfo", "get"]


def get(key: str) -> Source:
    try:
        return SOURCES[key]
    except KeyError:
        raise KeyError(f"unknown source {key!r}; choose from {', '.join(SOURCES)}") from None
