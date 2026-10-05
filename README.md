# Oracle

A free satellite OSINT workbench. Ask it a question in plain language and it
investigates with its own tools, then answers with numbered evidence you can click.
Underneath, Oracle does seven things:

1. **Finds** the best free imagery of any point on Earth for any time window. It
   searches 8 archives at once and ranks them by resolution, coverage and cloud.
2. **Detects** objects in that imagery:
   - ships on 10 m Sentinel-2 and on radar;
   - aircraft, ships, vehicles, helicopters and storage tanks with YOLO on sub-metre
     imagery (or your own GeoTIFF).
3. **Tracks** objects across dates and sensors. Every link gets a *probability that
   it's the same object*.
4. **Raises events**: arrivals, departures, count anomalies, dark vessels, loitering
   and fast movers, ranked by severity across as many sites as you like.
5. **Sees change**: new construction, earthworks, land reclamation, floods, burn
   scars and cleared ground, from optical or all-weather radar against a multi-date
   baseline.
6. **Knows the pattern of life** of a place: ship-to-ship rendezvous, objects where
   the site's own history says nothing ever is, and when the next free satellite
   pass will come.
7. **Investigates and briefs.** The Oracle agent plans, calls the tools above, reads
   the results and writes an answer. Every claim cites evidence (E1, E2, ...) that
   traces back to a specific image, detection or algorithm. It runs on Claude if
   you've configured it, and as an offline playbook if not.

No accounts, no API keys, no paid data. Claude is the only optional extra.

```
$ oracle ask "Are there large ships waiting off Fujairah?" --no-llm
  -> geocode {"place": "Fujairah"}                    ok 0.4s  Fujairah -> 25.41474,56.23137 [E1]
  -> search_imagery {..., "days_back": 30}            ok 16.7s 32 images: sentinel-1 18, sentinel-2 7, landsat 7 [E2]
  -> get_events {..., "days": 60}                     ok 0.0s  [sts_candidate] Unusually wide 222 x 82 m hull (possible rafted pair) [E3, E4]
  -> detect_ships {..., "radius_km": 25}              ok 53.3s 101 vessels in sentinel-2 2026-10-04 07:02Z [E5, E6, ...]
  -> next_passes {...}                                ok 0.1s  sentinel-2 2026-10-06T07:02Z, sentinel-1 2026-10-06T14:24Z [E21]

**Bottom line**
- 101 vessels in sentinel-2 2026-10-04 07:02Z (100% of area clear): 9 of 200 m or more,
  25 underway, 76 stationary or unknown [E5]
**What the imagery shows**
- 319 m vessel at 25.23811,56.41183 on 2026-10-04 07:02Z (sentinel-2), underway course 112 [E6]
- [sts_candidate] Unusually wide 222 x 82 m hull (possible rafted pair) ... [E3]
  ...
```

```
$ oracle sites preset chokepoints                 # 14 shipping chokepoints
$ oracle sweep --lookback 20                      # imagery -> detections -> tracks -> events
  Strait of Hormuz                 +5 images  +94 objects   90 tracks  +15 new events
  Suez Canal south anchorage       +5 images  +272 objects  205 tracks  +35 new events
$ oracle track T4348aa0879d                       # one anchored ship, radar + optical
  5 sightings 2026-08-23 .. 2026-09-15, sources ['sentinel-1', 'sentinel-2']
  2026-08-23T22:47Z sentinel-1  1.26049,103.87848  218 m  same-object p=1.0
  2026-08-27T03:37Z sentinel-2  1.26023,103.87938  122 m  same-object p=0.569
  2026-09-03T11:24Z sentinel-1  1.25954,103.87936  107 m  same-object p=0.608
  2026-09-04T22:47Z sentinel-1  1.25889,103.87913   90 m  same-object p=0.991
  2026-09-15T11:24Z sentinel-1  1.25840,103.87885  222 m  same-object p=0.957
$ oracle detect "32.1545,-110.8530" -r 0.6        # YOLO on 30 cm NAIP: aircraft boneyard
  100 objects in 151s: aircraft 61, vehicle 32, large-vehicle 5, helicopter 2
$ oracle brief --since 7d                         # ranked write-up of what changed
```

## What "free" can and can't get you

You can't get free sub-metre imagery of *any* point on *any* date. Commercial satellites
only image what someone pays for. Oracle combines every free archive and picks the best:

| source | resolution | coverage | revisit | licence |
|---|---|---|---|---|
| `umbra` | **16-50 cm** radar | ~8,000 collects over selected ports, airbases, mines, disasters | some sites weekly, 2023+ | CC BY 4.0 |
| `wayback` | **15 cm-1 m** optical | most populated land: Esri's basemap archive (Maxar/Vantor, Airbus, aerial) | months to years, 2014+ | Esri ToU, *view only* |
| `maxar` | **30-50 cm** optical | ~60 disaster/crisis events, pre + post | event driven | CC BY-NC 4.0 |
| `naip` | **30-60 cm** aerial | continental US | every 2-3 years | public domain |
| `capella` | **0.5-1 m** radar | ~2,300 collects worldwide | irregular, 2020+ | CC BY 4.0 |
| `sentinel-2` | 10 m optical | **everywhere**, land + coastal seas | **2-3 days**, 2015+ | Copernicus open |
| `sentinel-1` | 10 m radar (through cloud, at night) | everywhere | 6-12 days, 2014+ | Copernicus open |
| `landsat` | 30 m optical | everywhere | 8 days, 1982+ | public domain |

So the realistic global loop is **maritime**: Sentinel-1/2 every few days everywhere,
plus sub-metre radar where Umbra/Capella task it. **Object detection** (aircraft, cars)
needs sub-metre optical imagery: Maxar event data, NAIP (US), or imagery you supply
with `oracle detect --file`.

## Install

Python 3.10+. Wheels bundle GDAL/PROJ, so you don't need any system packages.

```
git clone https://github.com/overlordxrz-source/oracle && cd oracle
python -m venv .venv && . .venv/bin/activate
pip install -e ".[all]"     # or: -e .  (core)  |  -e ".[ai]" (YOLO)  |  -e ".[llm]" (Claude briefs)
oracle serve                # http://127.0.0.1:8000
```

For Claude (the agent and briefs), set `ANTHROPIC_API_KEY` or run `ant auth login`.
Without it, `oracle ask` runs the offline playbook and briefs use the built-in report.
`oracle doctor` checks your setup: acceleration, model choice, credentials, storage,
and whether every data source is reachable.

### Hardware: it adapts on its own

YOLO picks the device and model size automatically:

| machine | device | default model |
|---|---|---|
| NVIDIA GPU | `cuda:0` | `yolo11x-obb` (largest, most accurate) |
| **Apple Silicon (M1-M5)** | `mps` (Metal) | `yolo11x-obb` |
| CPU only | `cpu` | `yolo11l-obb` (found 256 cars vs 161 for `s`) |

On a Mac, `pip install -e ".[all]"` installs a PyTorch build with Metal support, and
Oracle sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so the few unsupported ops fall back to the
CPU. A **MacBook Air has no fan**: a single detection is fast, but a long sweep of
YOLO sites will throttle after several minutes. For sustained runs use
`ORACLE_YOLO_MODEL=yolo11l-obb.pt` (or `--model`). The ship and change detectors
don't use the GPU at all; they're I/O-bound.

Overrides: `ORACLE_DEVICE=cpu|mps|cuda:0`, `ORACLE_YOLO_MODEL=<weights>`,
`ORACLE_DB=<path>`, `ORACLE_CACHE=<dir>`.

## The console (`oracle serve`)

A dark, minimal, full-screen map. The basemap is dimmed so detections and change stand
out. The command bar (⌘K or `/`) does two things: type a question and Oracle
**investigates**, or type a place or `lat,lon` and it **goes** there. The views sit in a
slim dock on the left:

- **Ask.** Live investigation: every tool call appears as it runs, with its inputs,
  timing, result and the evidence it produced. Thinking summaries appear too when
  Claude is the engine. The answer's citations are pills: click one to fly to it and
  open the evidence (the detection chip, or before/after crops for a change). Evidence
  with a location is plotted on the map. A provenance line counts cited, uncited and
  unknown evidence. Past investigations are kept.
- **Change.** Pick optical or radar and a recent or same-season-last-year baseline,
  then run on the selected area. Change regions are drawn on the map by kind, and each
  opens a before/after pair.
- **Imagery.** Search all archives for the area you clicked or typed. Results are
  ranked, and a **timeline** shows every scene by source and date: click a scene,
  step with ←/→, **play** through them, or **blink** (B) between two dates to spot
  changes. Any scene streams onto the map at full resolution from its COG. Scene
  cards also have **Find ships / Detect objects** and **PNG / GeoTIFF** export.
- **Objects.** Everything the detectors found in view, clustered so tens of thousands
  stay fast. Filter by class and confidence; click one for an image chip, its size,
  heading, wake, AIS match or dark flag, and its track.
- **Tracks.** Objects seen more than once. The inspector shows a chip for every
  sighting with a *same-object* probability bar for each link, plus the "possibly
  also" alternatives the tracker considered.
- **Sites.** Watched places with severity badges and per-image activity bars (cloudy
  passes are dimmed). Load presets (chokepoints, naval bases, US airbases), add a
  site, sweep one site or all of them, or show a site's **pattern of life**: a heatmap
  of where its vessels, aircraft or vehicles usually are.
- **Events.** Ranked events (click to fly there) and a one-click brief.

Selecting an area also shows the **next free looks**: when Sentinel-2, Sentinel-1 and
Landsat will next pass over it.

The server binds to 127.0.0.1. Endpoints that read imagery only accept URLs from the
public data buckets the sources use (an allowlist), so they can't be used to fetch
arbitrary URLs.

## CLI

```
# imagery
oracle search    WHERE [-r KM] [--days N | --start D --end D | --date D] [-s a,b] [--max-cloud P] [--sort best|resolution|date|cloud]
oracle fetch     WHERE [...] [-n 3] [--res M] [--format png,tif] [--enhance]
oracle history   WHERE                            # every sub-metre Wayback capture at a point
oracle timelapse WHERE --start D --end D --every week|month|quarter|year -o out.gif
oracle index     [--refresh]                      # (re)crawl Maxar/Capella/Umbra catalogs

# detection
oracle ships     WHERE [-r KM] [--days N] [-s sentinel-2|sentinel-1|umbra|capella] [--min-length M]
oracle detect    WHERE | --file my.tif [--classes aircraft,helicopter] [--prompt "fighter jet,truck"]
                 [--model yolo11s-obb.pt] [--no-vehicles] [--site NAME]

# ask (the agent)
oracle ask       "QUESTION" [--llm | --no-llm] [--max-calls 20] [--effort high] [-v] [--json inv.json]

# change, orbits
oracle change    WHERE [-r KM] [-s sentinel-2|sentinel-1] [--after D] [--before D] [--baseline recent|anniversary] [--site S]
oracle passes    WHERE [--days 7] [--families sentinel-2,sentinel-1,landsat] [--all]

# intelligence loop
oracle sites     add NAME WHERE [-r KM] [--kind maritime|naval|airbase|ground] | list | rm NAME
oracle sites     preset chokepoints|naval-bases|airbases-us|all
oracle sweep     [--sites a,b] [--lookback 30] [--loop 21600] [--webhook URL] [--brief brief.md]
oracle tracks    [--site S] [--min-obs 3] [--status active|departed|lost]
oracle track     ID                               # every sighting + same-object probabilities
oracle events    [--since 7d] [--site S] [--min-severity 0.5]
oracle brief     [--since 7d] [--llm | --no-llm] [-o brief.md]
oracle ais       load positions.csv [--bbox w,s,e,n] [--start D --end D]
oracle db                                         # database location + counts
oracle doctor                                     # hardware, models, credentials, network
```

`WHERE` = `lat,lon` | `west,south,east,north` | a place name (OpenStreetMap geocoding).
A "global watcher" is just `oracle sweep --loop 21600 --webhook ...` running somewhere.
It posts new high-severity events to Discord or Slack.

## How it works

```
 archives ──search──▶ scenes ──detectors──▶ observations ──tracker──▶ tracks ──analytics──▶ events ──▶ brief
 (8 free)            (ranked)  ships S2/SAR   (SQLite+R*Tree)  same-object p   arrivals, anomalies,   Claude /
                               YOLO-OBB                        per link        dark vessels, ...      built-in
```

### Detectors

- **Ships, Sentinel-2.**
  - **Water mask:** ESA's scene classification, with ship-sized "holes" put back
    (unless they're vegetated islands).
  - **Candidates:** blobs k sigma brighter than the local water in near-IR.
  - **Hull test:** long blobs must be thin; the spine must be straight (hulls stay
    within 2-5 m of a line, cloud streaks wander 7-25 m); and the spectrum must look
    like paint (brighter than water in red; flat, puffy blobs are cloud).
  - **Validation:** tuned on the Malacca and Singapore straits and the Andaman Sea.
    Every hull of 150 m or more was kept, and 5 of 6 cloud and surf look-alikes were
    rejected.
  - **Motion:** a foam wake (bright in blue, dark in NIR) behind the hull marks a ship
    underway and gives its heading. In Malacca, every underway tanker read 306-310°,
    matching the imagery.
- **Ships, radar (Sentinel-1, Umbra, Capella).** Water comes from the **ESA
  WorldCover** 10 m land mask. A backscatter threshold fails on wind-roughened sea:
  at Hormuz it found 0-4 ships per image, while the land mask found 5-14. Detection
  is a local-contrast (CFAR-style) test in dB.
- **Objects, YOLO11-OBB (DOTA).** Detects planes, ships, small and large vehicles,
  helicopters and storage tanks as oriented, georeferenced boxes. YOLO is
  scale-sensitive, so Oracle runs two passes and measured where each works:

  | target | best resolution | measured result |
  |---|---|---|
  | aircraft | 0.6-0.9 m/px | 65-68 aircraft at the Davis-Monthan boneyard, vs 57 at native 0.3 m |
  | cars | upsampled to ~0.15 m/px | 256 at 0.13 m in a Tucson lot, vs 0 at 0.3 m |

  `yolo11l-obb` is the default because it found the most cars (256 vs 161 for
  `yolo11s-obb`). `--prompt` switches to **YOLO-World** open-vocabulary detection.
  It's zero-shot on overhead views, so treat it as a lead generator: it found 7 of
  ~30 planes. Known weak spot: cars parked bumper-to-bumper on hazy imagery.

### Rejected on purpose: ship speed from one Sentinel-2 image

Sentinel-2 bands are captured up to 2.6 s apart: B08 at +0.264 s, B03 at +0.527 s
and B04 at +1.005 s after B02 (Binet et al., ISPRS 2022). So moving objects shift
between bands. That works for aircraft, which shift ~200 m. For ships, Oracle measured
it and rejected it:

- **Visible bands:** hull paint and foam change the ship's shape from band to band by
  more than the ~1 pixel of motion, so errors were 6-19 kn.
- **Red-edge bands:** they agree with each other, but band-to-band misregistration
  still made **anchored** ships read 1-13 kn.

Speeds come from the tracker (position change between sightings) or from AIS instead.

### The tracker: "is this the same object, and how sure are we?"

For every existing track T and new observation o, Oracle computes a likelihood ratio:
*o is T seen again* vs *o is an unrelated object somewhere in the site*. It multiplies:

- **Kinematics.** T either stayed put (a Gaussian on both position errors) or moved.
  If T has a measured velocity, Oracle dead-reckons with growing uncertainty;
  otherwise any point within max-speed × Δt is equally likely. A wake heading that
  agrees with the displacement boosts the score.
- **Attributes.** Length must agree within each sensor's error (radar's error is
  wide, because side-lobes inflate lengths). Parked objects must keep their
  orientation.
- **Persistence.** The prior odds that the object is still there after Δt (vessels:
  days; vehicles: hours to days; aircraft: weeks).

Each pair's ratio is normalised against every competing track, every competing
observation and "new object", which gives a probability. The final assignment is the
global optimum (the Hungarian algorithm). Links under 50% start a new track, and the
near-misses are kept as "possibly also" alternatives. A track only counts as
**departed** after it's missed in two passes that were actually clear (Oracle records
how much of each pass was cloud-free), so a cloudy image doesn't "lose" anything.
Tracks are rebuilt deterministically on every sweep.

### Events and briefs

Each site is judged against its **own** history:

- **Arrivals:** only counted after a clear baseline image.
- **Count anomalies:** a robust z-score against the site's own counts per class and
  sensor.
- **Busy sites:** in a crowded anchorage, turnover is rolled up into one event and
  down-weighted.
- **Other events:** dark vessels (when AIS is loaded), fast movers, ships loitering
  14+ days, and a task-group hint.

Briefs are built from a compact digest (counts vs baseline, top events, notable
tracks, data gaps):

- **With Claude** (`claude-opus-5-5`): an analyst-style brief with a bottom line up
  front and calibrated estimative language, tied to the digest's numbers. It's told
  not to identify specific vessels or units unless AIS did, and not to speculate on
  intent. Server-side refusal fallbacks are enabled.
- **Without:** the same digest as a Markdown report.

### Change detection: "what is physically different?"

Detectors only find classes they were trained on. Change detection finds what's new
whatever it is: earthworks, a new pier, a flooded field, a burn scar.

- **Optical (Sentinel-2, 6 bands, 10 m).** Up to three clear earlier images are
  composited per pixel (median, with cloud, shadow and snow masked by the scene
  classification) into a **baseline**. Two-date differencing is noisy (haze, a parked
  ship, one bad pixel); a multi-date baseline cancels most of that. For each index
  (NDVI, MNDWI, NBR, brightness, SWIR) the difference becomes a **robust z-score
  against the whole area**, so scene-wide shifts from season, sun or atmosphere cancel
  out. Classes, in priority order:
  - `new_water` and `water_loss`: need the water index and SWIR to agree, which keeps
    out glint, plumes and algae;
  - `burn`: a large NBR drop where the ground got darker (char), unlike cleared soil,
    which brightens;
  - `vegetation_loss`;
  - `new_bright_surface`: needs SWIR to brighten too, which haze can't do;
  - `surface_change`: change-vector magnitude over the haze-robust bands.

  Ship-shaped "water loss" is dropped, because ships have their own detector.
  `--baseline anniversary` compares with the same season last year, so harvests and
  leaf-fall don't read as change.
- **Radar (Sentinel-1, VV+VH).** Speckle is suppressed with 5×5 multilooking plus a
  temporal mean over the baseline. Change is a log-ratio of at least 3 dB plus a
  robust-z test, split into `radar_increase` (new structures, vehicles, containers) and
  `radar_decrease` (removal, flooding). The sea is masked with WorldCover, and the
  baseline prefers images from the same orbit direction as the newest image.
- **Measured.** At Palm Jebel Ali (active construction), optical and radar both
  concentrated change on the island's trunk. Zooming in on the top optical region
  showed new dark excavation and a newly filled basin. The first version also flagged
  turbid lagoon water as "water loss"; the SWIR test now removes that. The port of
  Fujairah came back with no change across two weeks.

### Pattern of life

- **Ship-to-ship (STS) rendezvous.** Two hulls lying alongside: centres closer than
  their combined beams plus ~60 m, parallel within 25°, overlapping along their
  length, and not one ship passing the other. On 10 m imagery a rafted pair can merge
  into one unusually wide hull, so those are flagged too. A pair seen together on
  several dates scores higher. Crowded anchorages score lower, because rafting there
  is routine.
- **Unusual location.** A kernel density of where each class has been in every
  earlier clear image covering the spot (vessels ~300 m, aircraft ~60 m, vehicles
  ~25 m). A new *stationary* object where history expects under 0.02 objects per look
  is flagged: a ship anchored in a lane nobody anchors in, or aircraft on an apron
  that's always empty. It needs 6+ clear looks first, so it stays silent at new sites.
- **Collection planning.** Public TLEs (CelesTrak, with a mirror and a 12 h cache)
  are propagated with SGP4. Each closest approach is checked against the real imaging
  geometry: a 290 km nadir swath for Sentinel-2, 185 km for Landsat, descending
  daylight passes only, and Sentinel-1's right-looking 345-617 km radar swath. The
  predicted Sentinel-2A pass over Hormuz at 07:02 UTC and the Sentinel-2B pass over
  Singapore at 03:37 UTC both match the archive's acquisition times.

### The Oracle agent and provenance

`oracle ask` (or the command bar) runs an investigation:

- **With Claude** (`claude-opus-5-5`, adaptive thinking, server-side refusal
  fallbacks): a manual tool-use loop over 14 tools. The tools are geocode,
  search_imagery, detect_ships, detect_objects, detect_change, query_objects,
  get_tracks, get_track, get_events, next_passes, wayback_history, list_sites,
  monitor_site and ais_positions. The model plans, starts with cheap queries, runs a
  detector on the newest clear image, corroborates important claims with a second
  date or sensor, and stops when the question is answered. A tool-call budget caps
  the run (default 20).
- **Offline playbook.** The same tools in a fixed order chosen from keywords in the
  question, with a templated answer.
- **Evidence.** Every fact a tool returns is registered as evidence and labelled:
  - `observation`: one detection in one image, with its chip;
  - `derived`: an algorithm's output, such as tracks, change regions or events;
  - `reference`: a catalog, orbital or gazetteer fact.

  The answer has to cite the ids. Afterwards Oracle checks every citation against
  the registry and flags any it doesn't know as unsupported. Investigations are
  stored with their full step trace.
- **Boundary.** Oracle works on places and objects, never people. Questions about
  following a person, reading licence plates, facial recognition or someone's home
  are declined before any tool runs, and the model is told the same.

### AIS: dark-vessel detection

`oracle ais load file.csv` accepts NOAA MarineCadastre (US, free), the Danish Maritime
Authority (free), aisstream.io dumps, Global Fishing Watch exports and most commercial
exports. Each detection is matched to AIS tracks interpolated to the image time, with
distance and length likelihoods. A big hull with **no AIS nearby where AIS coverage
exists** is flagged dark.

## Limits (read before concluding anything)

- 10 m pixels can't identify a ship type. A carrier, a VLCC and a large container ship
  look alike. Lengths include wake, about ±20 m on Sentinel-2 and more on radar.
- Track probabilities are model outputs, not identification. In busy straits with 5-day
  gaps, most ships will rightly get *low* link probabilities.
- Free sub-metre imagery is patchy. Airbase monitoring outside the US needs Maxar
  event imagery or your own imagery.
- There will be false positives (surf lines, the odd cloud streak, platforms, wind
  farms) and misses (small, dark and wooden boats). Use the image chips.
- Wayback releases are mosaics. Check the capture date at your exact point.
- **Licences:** Maxar Open Data is non-commercial. Esri imagery is display-only, with
  attribution. Capella, Umbra and WorldCover are CC BY 4.0. Ultralytics code and weights
  are **AGPL-3.0**.
- Change classes are spectral rules, not a trained classifier. "New bright surface"
  can be construction, or a stockpile, or a field after harvest. Always look at the
  before/after chips. Radar change on slopes and buildings is unreliable if the
  baseline mixes orbit directions (Oracle notes it when it does).
- "Likely" passes mean the geometry *allows* an acquisition. Sentinel-1 also depends
  on ESA's acquisition plan.
- The agent is only as good as its tools and data. It can pick a poor area: geocoders
  return city centroids, not anchorages, which is why the playbook searches 25 km for
  maritime questions. Read the step trace. Claims it can't cite are flagged.
- **Use it for research, journalism and situational awareness on public data.** It
  works on places and objects. It isn't built to follow individuals, and the agent
  declines such questions: satellite views can't identify people or plates, and don't
  point it at someone's home.

## Tests

```
pip install -e ".[dev]"
pytest              # 69 offline tests: detectors, tracker, events, AIS, store, API, briefs,
                    # orbits, STS/unusual-location, change classes, agent loop (mocked Claude), playbook
pytest -m live      # smoke tests against the real catalogs
```
