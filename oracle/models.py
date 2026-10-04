"""Shared data model: every source normalizes its results into ``Scene`` objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Render:
    """How to turn a scene's assets into pixels.

    kind:
      ``rgb8``        one 3+ band 8-bit COG (Sentinel-2 TCI, Maxar visual, NAIP)
      ``reflectance`` separate single-band COGs scaled to reflectance (Landsat)
      ``sar``         single-band amplitude/power COG, displayed in dB
    """

    kind: str
    hrefs: list[str]
    bands: list[int] = field(default_factory=lambda: [1, 2, 3])
    scale: float = 1.0
    offset: float = 0.0
    vmin: float | None = None
    vmax: float | None = None
    power: bool = False  # SAR: values are power (10*log10) rather than amplitude (20*log10)
    sign: str | None = None  # Planetary Computer collection whose SAS token must sign the hrefs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Render:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class Scene:
    id: str
    source: str  # short key, e.g. "sentinel-2", "maxar", "umbra"
    platform: str
    sensor: str  # "optical" | "sar"
    datetime: datetime
    gsd: float  # ground sample distance, metres
    bbox: tuple[float, float, float, float]
    render: Render
    geometry: dict | None = None
    cloud_cover: float | None = None
    off_nadir: float | None = None
    thumbnail: str | None = None
    item_url: str | None = None
    license: str = ""
    attribution: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def date(self) -> str:
        return self.datetime.strftime("%Y-%m-%d")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "platform": self.platform,
            "sensor": self.sensor,
            "datetime": self.datetime.isoformat(),
            "gsd": self.gsd,
            "bbox": list(self.bbox),
            "geometry": self.geometry,
            "cloud_cover": self.cloud_cover,
            "off_nadir": self.off_nadir,
            "thumbnail": self.thumbnail,
            "item_url": self.item_url,
            "license": self.license,
            "attribution": self.attribution,
            "render": self.render.to_dict(),
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Scene:
        return cls(
            id=d["id"],
            source=d["source"],
            platform=d.get("platform", ""),
            sensor=d.get("sensor", "optical"),
            datetime=parse_dt(d["datetime"]),
            gsd=float(d["gsd"]),
            bbox=tuple(d["bbox"]),
            render=Render.from_dict(d["render"]),
            geometry=d.get("geometry"),
            cloud_cover=d.get("cloud_cover"),
            off_nadir=d.get("off_nadir"),
            thumbnail=d.get("thumbnail"),
            item_url=d.get("item_url"),
            license=d.get("license", ""),
            attribution=d.get("attribution", ""),
            extra=d.get("extra", {}),
        )


def parse_dt(value: str | datetime) -> datetime:
    """Parse the assorted ISO-ish timestamps found across catalogs into aware UTC datetimes."""
    if isinstance(value, datetime):
        dt = value
    else:
        s = value.strip().replace(" ", "T")
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
