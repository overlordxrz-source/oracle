# Oracle

Free satellite imagery for OSINT research. Give it any point on Earth and a time window,
and it searches every free archive in parallel and ranks what it finds by resolution.
It can render or download the imagery and stream it onto a map at full resolution.
It can also find ships in it.

No accounts, no API keys, no paid data.

```
$ oracle search "25.0108,55.0613" -r 2          # Jebel Ali port
found: wayback 9, umbra 30, capella 1, sentinel-2 30, sentinel-1 30, landsat 30
  #  source      date (UTC)          res cloud cover  platform     id
  0  umbra       2025-11-12 18:49  31 cm         99%  umbra-10     umbra-2025-11-12-18-49-17_UMBRA-10
  ...

$ oracle history "38.8977,-77.0365"              # every sub-metre capture of a point since 2011
  captured       res  sensor     provider
  2025-07-24   50 cm  WV02       Vantor Vivid Advanced
  2022-04-02   30 cm  WV03       Maxar Vivid Premium
  ...

$ oracle ships "2.55,101.55" -r 15 --days 60     # Strait of Malacca, Sentinel-2
  27 vessels >= 25 m, 5 >= 250 m
     476 m x 55 m  axis 129.4 deg   2.49947, 101.60337   87.3 sigma
     333 m x 73 m  axis 130.6 deg   2.55228, 101.54154   84.8 sigma
  -> annotated overview PNG, close-up contact sheet, GeoJSON
```

## What "free" can and can't get you

You can't get free sub-metre imagery of *any* point on *any* date. Commercial
satellites (Maxar/Vantor, Planet SkySat, Airbus) only image a place when someone pays
them to. Oracle collects every free archive that exists and picks the best one for you:

| source | resolution | coverage | revisit | licence |
|---|---|---|---|---|
| `umbra` | **16-50 cm** radar | ~8,000 collects over selected ports, airbases, mines, disasters | some sites weekly, 2023+ | CC BY 4.0 |
| `wayback` | **15 cm-1 m** optical | most populated land, as Esri's basemap archive (Maxar/Vantor, Airbus, aerial) | months to years per place, 2014+ | Esri ToU, *view only* |
| `maxar` | **30-50 cm** optical | ~60 disaster/crisis events, pre + post imagery | event driven | CC BY-NC 4.0 |
| `naip` | **30-60 cm** aerial | continental US only | every 2-3 years | public domain |
| `capella` | **0.5-1 m** radar | ~2,300 collects worldwide | irregular, 2020+ | CC BY 4.0 |
| `sentinel-2` | 10 m optical | **everywhere**, land + coastal seas | **every 2-3 days**, 2015+ | Copernicus open |
| `sentinel-1` | 10 m radar (sees through cloud, at night) | everywhere | 6-12 days, 2014+ | Copernicus open |
| `landsat` | 30 m optical | everywhere | 8 days, 1982+ | public domain |

So for a given place you usually have three options:
- **sub-metre but old**: Wayback, which shows when it was captured and by which satellite;
- **sub-metre and recent**, if you're lucky: Umbra/Capella/Maxar cover a lot of ports, bases and hot spots;
- **10 m but from the last few days**: Sentinel-1/2, which is what most open-source
  ship and aircraft-carrier tracking actually uses.

### How the carrier-on-Sentinel-2 posts are made
At 10 m a 333 m carrier is ~33 pixels long, and its wake is often longer than the hull.
Analysts use news or AIS reports to narrow the search to a region and date. Then they
scan the Sentinel-2 scenes for large hulls. A carrier usually has escorts around it in
formation. Oracle automates the scan:

```
oracle ships "5.5,95.5" -r 60 --start 2026-03-08 --end 2026-03-12 --min-length 200
```

Each very large hull gets boxed in red. Oracle reports its length, width and axis, and
counts the 100-220 m hulls within 15 km (a weak task-group hint). It also saves close-ups.
At 10 m a carrier, a VLCC tanker and a big container ship look alike, so **you** do the
identification, using context: escorts, the wake, the route, the news. If it's cloudy,
use `--sources sentinel-1` (radar).

## Install

Needs Python 3.10+. Wheels for rasterio and pyproj bundle GDAL/PROJ, so no system
packages are required.

```
git clone https://github.com/overlordxrz-source/oracle && cd oracle
python -m venv .venv && . .venv/bin/activate
pip install -e .
oracle serve            # http://127.0.0.1:8000
```

## Web app (`oracle serve`)

Type a place (or `lat,lon`), or click the map. Pick a date range and the sources you
want, then hit Search. The results come back ranked.

- **View** streams the scene onto the map from its cloud-optimized GeoTIFF at full
  resolution: zoom right in on 30 cm Maxar or 25 cm Umbra. There's an opacity slider
  for before/after comparison against the Esri basemap.
- **PNG / TIF** downloads the area as a captioned PNG or a georeferenced GeoTIFF.
- **Ships** runs vessel detection on Sentinel-2 or radar scenes. Hulls of 250 m or more
  are red, and you can click a detection to fly to it.

It listens on 127.0.0.1 by default. The tile endpoint will only read from the public
data buckets the sources use (an allowlist), so it can't be used to fetch arbitrary URLs.

## CLI

```
oracle sources                                   # what's available
oracle search  WHERE [-r KM] [--days N | --start D --end D | --date D] [-s a,b] [--max-cloud P] [--sort best|resolution|date|cloud] [--json]
oracle fetch   WHERE [...search opts] [-n 3] [--res M] [--format png,tif] [--enhance] [-o DIR]
oracle ships   WHERE [-r KM] [--days N] [-s sentinel-2|sentinel-1|umbra|capella] [--min-length M] [-k SIGMA] [--include-shore]
oracle history WHERE                             # Wayback capture history at a point
oracle timelapse WHERE --start D --end D --every week|month|quarter|year [--source sentinel-2] -o out.gif
oracle index [--refresh]                         # (re)crawl Maxar/Capella/Umbra catalogs
oracle watch add NAME WHERE [-r KM] [-s ...] [--ships]
oracle watch run [--loop 3600] [--webhook URL]  # new-image alerts (Discord/Slack-compatible)
```

`WHERE` can be `lat,lon`, `west,south,east,north`, or a place name (geocoded through
OpenStreetMap).

### Watchlist: a cheap "global watcher"
```
oracle watch add jebel-ali "25.0108,55.0613" -r 3 -s sentinel-2,sentinel-1,umbra --ships
oracle watch add sevastopol "44.615,33.525" -r 4
ORACLE_WEBHOOK=https://discord.com/api/webhooks/... oracle watch run --loop 21600
```
Every run saves a captioned image of each new scene to `~/.cache/oracle/watch/<site>/`,
along with ship detections if you enabled them, and posts a one-line alert to the webhook.

## How it works

```
oracle/
  sources/        one adapter per archive, all normalised to Scene objects
    earth_search.py   Sentinel-2 L2A    (Element 84 STAC API, public AWS COGs)
    planetary.py      Sentinel-1 RTC, Landsat, NAIP (Planetary Computer, anonymous SAS tokens)
    maxar.py          Maxar Open Data   (static STAC tree; crawled + indexed locally)
    capella.py        Capella SAR       (static STAC tree; indexed)
    umbra.py          Umbra SAR         (S3 listing + per-collect metadata; indexed)
    wayback.py        Esri Wayback      (tilemap walk -> distinct captures + metadata)
  search.py       parallel federated search, AOI coverage, same-pass merging, ranking
  imagery.py      window reads from COGs (only the needed bytes, at the right overview),
                  reprojection, mosaicking, SAR dB stretch, XYZ tiles, captioned chips
  detect.py       vessel detection (optical NIR contrast + spectral hull test; SAR CFAR)
  timelapse.py    per-period best frames -> GIF
  watch.py        watchlist + alerts
  server.py       FastAPI + Leaflet UI (web/index.html)
```

- The static catalogs are crawled once, about a minute for ~50k scenes, then cached
  gzipped in `~/.cache/oracle/index` and refreshed weekly. Maxar's ARD tiles sit on
  their own quadtree for each UTM zone. Oracle decodes those quadkeys directly, so it
  never has to fetch 40k item files.
- Nothing is downloaded in bulk. A 4 km chip from a 2 GB Maxar strip costs a few MB of
  HTTP range requests.
- **Ship detection, optical:** water comes from ESA's scene classification. Ships are
  often mislabelled as cloud or land, so small "holes" in the water mask get put back,
  unless they're vegetated (islands). A blob is a candidate if it's k sigma brighter than
  the local water in near-IR. Candidates then have to pass a hull test. Shape: long
  objects must be thin. Spectrum: hulls are brighter than the water in red, while cloud
  is spectrally flat and puffy. The thresholds were tuned on real hulls and look-alikes
  in the Malacca Strait.
- **Ship detection, radar:** the land/water split is an Otsu threshold on a coarse
  median of the backscatter. Detection is the same local-contrast test, in dB.

## Limits (read before you conclude anything)

- Lengths come from the shape of the bright blob, so they include wake and turbulence,
  and are only good to about +/- 20 m on Sentinel-2. Radar side-lobes (the bright
  crosses) also inflate them.
- There are false positives: thin cloud wisps that happen to be elongated, wind farms,
  oil platforms, fish traps. And small, dark or wooden boats get missed. Always look at
  the close-ups. Near-shore blobs (piers, moored ships) are dropped unless you pass
  `--include-shore`.
- The task-group hint means nothing in busy shipping lanes.
- Wayback dates are *capture* dates from Esri's metadata, but a release is a mosaic, so
  a neighbouring area can come from a different date. Check the capture date at your
  exact point.
- Respect the licences: Maxar Open Data is **non-commercial**, Esri imagery is for display
  with attribution and Oracle doesn't bulk-download it, and Capella/Umbra are CC BY 4.0
  (credit them).

## Roadmap ideas

- AI super-resolution for Sentinel-2 (e.g. ESA OpenSR / SEN2SR, 10 m to 2.5 m). It's
  labelled "hallucinated detail" for a reason, so only as a clearly marked layer.
- AIS cross-referencing to flag *dark* vessels (detected but not broadcasting).
- NASA FIRMS active-fire points and VIIRS night lights.
- Satellogic EarthView (1 m, one-time coverage of large regions), Copernicus Data Space.
- Change detection between two dates (new structures, aircraft counts at airbases).

## Tests

```
pip install -e ".[dev]"
pytest              # offline unit tests
pytest -m live      # smoke tests against the real catalogs
```
