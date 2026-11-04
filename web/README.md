# Physicam web UI

Plain `index.html` + `style.css` + ES modules — no build step, no CDNs, works offline.
Served by the backend as static files, or standalone for mock development:

```
.venv/bin/python -m http.server 8765 -d web
# open http://127.0.0.1:8765/?mock=1
```

## Architecture

| File | Role |
| --- | --- |
| [index.html](index.html) | Page skeleton: top bar, input panel, video stage, agent timeline, results |
| [style.css](style.css) | Dark instrument theme; light theme via `prefers-color-scheme`; stacks ≤ 900 px |
| [app.js](app.js) | State, job lifecycle, SSE handling, video overlay, transport, mic, mock replay |
| [charts.js](charts.js) | Hand-rolled canvas charts: height + residuals strip + restitution bars, hover tooltip, click-to-seek |
| [markdown.js](markdown.js) | Safe mini markdown renderer (escapes all input; headings, lists, tables, code, links) |
| [mock/make_mock.py](mock/make_mock.py) | Builds `mock/result.json` from the real SAM fixture and copies the demo clip |

State lives in one `state` object in [app.js](app.js): the accumulated result JSON
(contract shape from `AGENTS.md`), a `Map` of tracks by id, and the fits list.
Rendering is push-based: every SSE event mutates state, then only the affected
region re-renders (a track redraws legend/charts/overlay; a fit redraws the
parameter table and charts; the answer animates in with a count-up).

## Event handling

The UI consumes `GET /api/jobs/{id}/events` (SSE) and tolerates both nested
(`{type:"track", track:{…}}`) and flat (`{type:"track", id:"t0", …}`) payloads:

| Event `type` | Effect |
| --- | --- |
| `stage` | Section header in the timeline, running spinner row, and stage-progress pill on the stage (`ingest`/`plan`/`agent`/`report`; the `agent` pill updates with the latest trace summary, and the pill fades out on `done`) |
| `trace` | Timeline row: icon + model badge (SPARK/SAM/WONDER/local) by tool name, summary, ms, expandable args/summary JSON |
| `track` | Upserts a track; overlay, legend and charts update immediately |
| `fit` | Upserts a fit; parameter table + ghost curves + impact markers update |
| `strobe` | `{url}` (or a top-level `strobe_url` on the result/event): shows the Video \| Strobe segmented control on the stage and the strobe figure at the top of the lab report |
| `answer` | Answer card with count-up; Δ vs 9.81 m/s² (in % and σ) only when the unit is m/s², otherwise a dimensionless value ± stderr plus the `e_height` apex-height cross-check when a fit provides it |
| `report` | `report_markdown` rendered with [markdown.js](markdown.js) |
| `plan` / `calibration` | Stored on the result; calibration rescales charts and HUD to meters |
| `done` | Clears the spinner, then fetches the authoritative `GET /api/jobs/{id}` |
| `error` | Error box in the timeline |

`GET /api/jobs/{id}` also works standalone: open the page with `#job=<id>` after
a reload and the finished result renders without any SSE. Clicking an example
(`GET /api/examples` → `{job_id, …}`) loads that job the same way.

## Video overlay & chart linking

- The overlay canvas is synced with `requestVideoFrameCallback` (rAF fallback)
  and draws, per track in its color: the nearest mask contour (`new Path2D(d)`
  under a frame-px → canvas-px transform), a fading centroid trail, dashed fit
  ghost curves (per-segment parabolas for bounce), impact crosses with `e`, and
  a name label. The monospace HUD shows `t · frame · y · v` in meters when a
  calibration exists, else pixels.
- Charts share the video clock: a vertical cursor follows playback on every
  chart, clicking any chart (or a restitution bar) seeks the video, and the
  residuals strip is computed against the first fit curve for each track.
- The first chart is **Height vs time** (positive up): with a calibration,
  `h = (y_ref − y)·m_per_px` in metres, without one `(y_ref − y)` in pixels,
  where `y_ref` is the largest tracked `y` (the lowest point). Fitted curves,
  impact markers and residual signs are transformed the same way. When
  `calibration.kind == "per_frame"` and a track carries `X_m`/`Y_m`, those
  arrays are plotted directly (Y up, x/speed in metres). The restitution
  chart uses categorical "impact 1…n" labels (no numeric x ticks), shows each
  bar's value above it, and draws a dashed `e = 1` reference line.
- Whenever a job or example has `video.url` (mock replay, example, live job,
  or a `#job=` reload), the empty state hides and the `<video>` shows with its
  overlay canvas, sized to the video's aspect ratio; the video uses
  `preload="auto"` and seeks to 0.001 s on `loadeddata` so the first frame is
  visible before play. The timeline auto-scrolls to the newest row and keeps
  its running spinner row until the next event.
- A dismissible "How it works" strip under the top bar shows the four steps
  (Spark watches → SAM tracks → agent measures → Wonder reports) with their
  model badge colors; dismissal is remembered in `localStorage`.
- Keyboard: `space` play/pause, `←`/`→` step one frame (at the effective fps,
  honoring the slow-motion override). Speeds 0.1×–1×; masks/trail/fit toggles.

## Mock mode

`?mock=1` needs no backend: the examples gallery shows one mock entry,
"Measure" replays a fake SSE sequence (plan → track → fit → answer → report,
≈ 6 s of timers) from `mock/result.json`, the video is `mock/clip.mp4`, and
hold-to-talk fills a canned transcript. Regenerate the data with:

```
.venv/bin/python web/mock/make_mock.py
```

It parses `tests/fixtures/bounce240_red_ball_f0-199.sam.txt.gz` with
`meta_sam_parser`, computes mask-weighted centroids and frame-coordinate
contours, fits parabolas around the max-y impact frame, and calibrates from
the ball's known 80 mm diameter. It also writes `mock/strobe.jpg` —
`result.json` carries `strobe_url: "mock/strobe.jpg"` — by extracting frames
with `imageio_ffmpeg.get_ffmpeg_exe()` and compositing ~15 mask cut-outs
(masks from `meta_sam_parser.decode_mask_to_raster`, each raster mapped onto
its `bounds` rect) onto the first frame; the mock fit also provides an
`e_height` apex-height cross-check. Note the fixture's fall segment is not a
clean passive fall, so the mock fit honestly flags `e > 1` as an upper bound
and reports g from the rebound segment.

## Backend contract the UI expects

Beyond the `AGENTS.md` result contract:

- `POST /api/jobs` — multipart `video`, `question`, optional `facts` (JSON
  string: `reference_object`, `reference_size_m`, `fps_override`, `diameter_m`);
  returns `{id}`.
- `GET /api/jobs/{id}/events` — SSE, `data:` lines of JSON with a `type` field
  as tabulated above.
- `POST /api/transcribe` — multipart `audio` blob (browser MediaRecorder,
  typically webm/opus); returns `{transcript}`.
- `GET /api/examples` — `[{id, title, subtitle, thumb_url, question, job_id}]`.
- Fit objects may include `impacts: [{index, t, frame, x, y, e, e_stderr}]`
  (frame px for `x`/`y`) and `segments: [{curve: {t, x?, y}}]`; both are used
  for markers, ghost curves and the restitution bar chart when present.
  A fit may also provide `e_height` (a number, or `{value, stderr}`) — the
  apex-height restitution cross-check shown on dimensionless answers.
- Results may include `strobe_url`; stage values are `ingest`, `plan`,
  `agent` and `report`.
