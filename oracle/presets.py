"""Ready-made site lists for global sweeps. Approximate centres; radii are generous.

kinds -> detectors:
  maritime  vessel detection on Sentinel-2 / Sentinel-1 / Umbra / Capella
  naval     maritime + YOLO objects wherever sub-metre optical exists (Maxar events, NAIP)
  airbase   YOLO objects (aircraft, vehicles) on sub-metre optical. Free repeat coverage
            of that kind is essentially NAIP, hence the US list; elsewhere use Maxar Open
            Data events or your own GeoTIFFs (``oracle detect --file``).
"""

from __future__ import annotations

# name: (lat, lon, radius_km, kind)
CHOKEPOINTS = {
    "Strait of Hormuz": (26.57, 56.45, 20, "maritime"),
    "Bab-el-Mandeb": (12.58, 43.38, 15, "maritime"),
    "Suez Canal south anchorage": (29.90, 32.55, 10, "maritime"),
    "Port Said / Suez Canal north": (31.30, 32.35, 10, "maritime"),
    "Singapore Strait": (1.20, 103.85, 12, "maritime"),
    "Malacca Strait (Port Klang)": (2.95, 101.20, 15, "maritime"),
    "Bosphorus": (41.10, 29.06, 8, "maritime"),
    "Strait of Gibraltar": (35.96, -5.50, 15, "maritime"),
    "Dover Strait": (51.00, 1.50, 15, "maritime"),
    "Panama Canal Pacific anchorage": (8.88, -79.52, 10, "maritime"),
    "Kerch Strait": (45.30, 36.55, 12, "maritime"),
    "Oresund": (55.90, 12.70, 10, "maritime"),
    "Kinmen / Xiamen": (24.45, 118.40, 15, "maritime"),
    "Lombok Strait": (-8.75, 115.75, 15, "maritime"),
}

NAVAL_BASES = {
    "Norfolk Naval Station": (36.946, -76.315, 4, "naval"),
    "San Diego Naval Base": (32.684, -117.124, 4, "naval"),
    "Pearl Harbor": (21.355, -157.955, 4, "naval"),
    "Yokosuka": (35.290, 139.665, 3, "naval"),
    "Sevastopol": (44.617, 33.530, 5, "naval"),
    "Novorossiysk": (44.715, 37.790, 5, "naval"),
    "Tartus": (34.905, 35.870, 3, "naval"),
    "Severomorsk": (69.070, 33.420, 4, "naval"),
    "Baltiysk": (54.645, 19.900, 4, "naval"),
    "Kronshtadt": (59.990, 29.770, 4, "naval"),
    "Vladivostok": (43.110, 131.890, 5, "naval"),
    "Qingdao": (36.080, 120.380, 6, "naval"),
    "Yulin / Sanya": (18.225, 109.555, 4, "naval"),
    "Zhanjiang": (21.190, 110.410, 6, "naval"),
    "Toulon": (43.110, 5.920, 4, "naval"),
    "Portsmouth (UK)": (50.800, -1.110, 3, "naval"),
    "Djibouti": (11.600, 43.140, 5, "naval"),
    "Bandar Abbas": (27.140, 56.200, 5, "naval"),
}

AIRBASES_US = {
    "Davis-Monthan AFB": (32.150, -110.840, 3, "airbase"),
    "Barksdale AFB": (32.501, -93.663, 3, "airbase"),
    "Whiteman AFB": (38.730, -93.548, 3, "airbase"),
    "Ellsworth AFB": (44.145, -103.103, 3, "airbase"),
    "Minot AFB": (48.416, -101.358, 3, "airbase"),
    "Dover AFB": (39.130, -75.466, 3, "airbase"),
    "Travis AFB": (38.263, -121.927, 3, "airbase"),
    "Nellis AFB": (36.236, -115.034, 3, "airbase"),
}

PRESETS = {"chokepoints": CHOKEPOINTS, "naval-bases": NAVAL_BASES, "airbases-us": AIRBASES_US}

DETECTORS = {"maritime": ["ships"], "naval": ["ships", "objects"], "airbase": ["objects"], "ground": ["objects"]}
SOURCES = {
    "ships": ["sentinel-2", "sentinel-1", "umbra", "capella"],
    "objects": ["maxar", "naip"],
}
