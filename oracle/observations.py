"""One record type for everything any detector finds: an ``Observation``.

Ship blobs on Sentinel-2, CFAR hits on radar and YOLO boxes on 30 cm imagery all end
up as observations with a class, a position, a time, a size and (when it can be
measured) a velocity. The store, the tracker and the briefing only ever see these.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from .models import parse_dt

# Canonical classes. Detector-specific labels are kept in ``attrs["label"]``.
VESSEL = "vessel"
AIRCRAFT = "aircraft"
HELICOPTER = "helicopter"
VEHICLE = "vehicle"
LARGE_VEHICLE = "large-vehicle"
STORAGE_TANK = "storage-tank"
OTHER = "other"

# Rough upper bounds used for kinematic gating (m/s).
MAX_SPEED = {VESSEL: 18.0, AIRCRAFT: 300.0, HELICOPTER: 90.0, VEHICLE: 40.0, LARGE_VEHICLE: 35.0}
STATIC_CLASSES = {STORAGE_TANK, OTHER}


@dataclass
class Observation:
    cls: str
    lat: float
    lon: float
    time: datetime
    scene_id: str
    source: str
    detector: str
    confidence: float = 1.0  # detector score, 0-1
    length_m: float | None = None
    width_m: float | None = None
    axis_deg: float | None = None  # orientation of the long axis, 0-180 from north
    course_deg: float | None = None  # direction of travel, 0-360 (None = unknown)
    speed_ms: float | None = None
    speed_err_ms: float | None = None
    polygon: list[list[float]] | None = None  # [[lon, lat], ...] footprint, when known
    attrs: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            key = f"{self.scene_id}|{self.detector}|{self.cls}|{self.lat:.6f}|{self.lon:.6f}"
            self.id = hashlib.sha1(key.encode()).hexdigest()[:16]

    @property
    def speed_kn(self) -> float | None:
        return None if self.speed_ms is None else self.speed_ms * 1.943844

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["time"] = self.time.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Observation:
        d = dict(d)
        d["time"] = parse_dt(d["time"])
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def feature(self) -> dict:
        props = {k: v for k, v in self.to_dict().items() if k not in ("lat", "lon", "polygon")}
        props["speed_kn"] = None if self.speed_kn is None else round(self.speed_kn, 1)
        return {
            "type": "Feature",
            "id": self.id,
            "geometry": {"type": "Point", "coordinates": [self.lon, self.lat]},
            "properties": props,
        }


def feature_collection(obs: list[Observation], **props: Any) -> dict:
    return {"type": "FeatureCollection", "properties": props, "features": [o.feature() for o in obs]}
