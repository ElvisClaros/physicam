# Task 03: the web UI (`web/`)

Read `AGENTS.md` first, especially the **Result JSON contract** and the SSE events. Another engineer
is building the Python backend in parallel, so **only create or modify files under `web/`**. Do not
touch `physicam/` or `tests/`.

The demo video counts for 25% of judging, and this UI is what the judges will see. It must look like
a precise scientific instrument, polished and modern, and be fun to watch. The UI has no build step:
plain `index.html`, `style.css` and ES-module JS. Do not use external CDNs. Hand-roll the charts with
canvas or SVG so it works offline.

## Layout (desktop first, must also stack cleanly ≤ 900 px)

- **Top bar**: the "Physicam" wordmark with a small aperture/trajectory logo in inline SVG; the
  tagline "Film anything that moves. Ask a question. Get a measurement."; and on the right, model
  badges `Muse Spark` · `SAM 3.1` · `Muse Wonder`.
- **Left panel (input)**:
  - Drag-and-drop zone and file picker for the video.
  - An "Examples" gallery from `GET /api/examples`, which returns
    `[{id, title, subtitle, thumb_url, question, job_id}]`. Clicking one loads that finished job
    instantly.
  - A question textarea with suggestion chips: "What is g?", "Coefficient of restitution?",
    "Period of the pendulum?", "How fast is it moving?".
  - A **hold-to-talk mic button**: MediaRecorder sends `POST /api/transcribe` (multipart `audio`
    blob), the response is `{transcript}`, and the transcript fills the question. Show
    "transcribed by Muse Voice".
  - A collapsible "Known facts" section: reference object + real size (m), capture fps override
    (for slow-motion files), and object diameter.
  - A big "Measure" button. It sends `POST /api/jobs` (multipart: `video`, `question`, optional
    `facts` JSON), which returns `{id}`. Then open the SSE stream at `GET /api/jobs/{id}/events`.
- **Center: video stage**, the hero:
  - A `<video>` with a `<canvas>` overlay, synced per frame (`requestVideoFrameCallback` if
    available, otherwise rAF while playing).
  - Overlay content per track, in the track color:
    - The **mask contour** of the nearest frame that has one (contours come as SVG path strings in
      frame pixel coordinates; use `new Path2D(d)` with a scale transform from frame px to canvas
      px), drawn as a semi-transparent fill plus a crisp stroke.
    - A fading **trajectory trail** of past centroids.
    - The **fitted model curve** as a dashed "ghost" path. For bounce fits, draw each segment
      parabola from `fits[].segments[].curve`.
    - **Impact markers**.
    - A small label with the track name.
  - HUD in a corner, monospace: `t = 0.412 s · frame 99 · y = 1.23 m · v = 3.4 m/s`, in m when
    calibration exists, otherwise px.
  - Transport controls: play/pause, ±1 frame, speed 0.1× / 0.25× / 0.5× / 1×, and overlay toggles
    (masks, trail, fit).
- **Right panel: agent timeline** (live). One row per `trace` event, showing:
  - An icon by kind or tool: watch, track, calibrate, fit, python, answer, report.
  - A model badge: SPARK, SAM, WONDER or "local".
  - A one-line summary, with expandable args/summary JSON.
  - Elapsed ms.

  The running step shows an animated indicator. Stage events (`stage`) appear as section headers.
- **Bottom: results**
  - **Answer card**: big value ± uncertainty with unit, the headline, and a summary. When the
    quantity is g (unit m/s²), show "Δ vs 9.81 m/s²: +0.3% (1.2σ)".
  - **Parameters table** per fit (value ± stderr, unit, R², RMSE px, warnings as amber chips).
  - **Charts** (canvas, high-DPI aware), with a light grid, axis labels, units and hover crosshair
    tooltips:
    1. Vertical position vs time: data points plus the fitted curve(s), with a residuals strip
       below it.
    2. Horizontal position vs time.
    3. Speed vs time (finite differences, lightly smoothed).
    4. For bounce, a bar chart of the restitution coefficient per impact with error bars.

    **Link charts and video both ways**: a vertical cursor on each chart follows the video time,
    and clicking anywhere on a chart seeks the video to that time.
  - **Lab report**: render `report_markdown` with a small built-in markdown renderer (headings,
    bold/italic, lists, tables, code, links; no raw HTML injection, escape everything).
  - **Footer cost line**: "960 SAM frames · 14k Spark tokens · ≈ $0.21".
- **Progressive rendering**: while a job runs, apply SSE events as they arrive (draw tracks as soon
  as a `track` event comes, then the fit, answer and report). On `done`, fetch `GET /api/jobs/{id}`
  and render the final state. `GET /api/jobs/{id}` must also work by itself, for example after a page
  reload with `#job=<id>` in the URL.

## Visual design
- A dark theme (deep navy/graphite background) with a crisp light theme via `prefers-color-scheme`.
- Use one accent color (electric cyan or blue) and per-track colors from the result JSON, with a
  fallback palette.
- Use system font stacks: an Inter-like sans for UI and a monospace for numbers. Use tabular
  numerals.
- Subtle borders, 12 px radii, generous spacing, no clutter. Animations stay tasteful: fade/slide
  for timeline rows, and a sweep or count-up animation when the answer value appears.
- Empty state: an elegant illustration (inline SVG of a parabolic trajectory with dots) and the
  text "Drop a video of something moving".

## Mock mode (you need it to develop without the backend)
- With `index.html?mock=1`, load `web/mock/result.json` and `web/mock/clip.mp4`, and replay a fake
  SSE sequence on timers: stages, trace steps, then the track, fit, answer and report, roughly 6 s
  in total. The examples gallery shows one mock example.
- Write `web/mock/make_mock.py`, which builds `result.json` from REAL data:
  - Parse `tests/fixtures/bounce240_red_ball_f0-199.sam.txt.gz` (gzip text, real SAM 3.1 output,
    240 fps, 1280x720) with `meta_sam_parser`, using the pattern in AGENTS.md.
  - Compute mask-weighted centroids and contours: `decode_mask_to_svg_path` gives paths in raster
    coordinates, so transform them to frame coordinates using bounds.
  - Do a simple numpy bounce analysis to get realistic fits and segments: find the impact at the
    max-y frame and fit parabolas before and after it.
  - Write a plausible trace and a short report.
  - Copy `_probe/real/bounce_clip.mp4` to `web/mock/clip.mp4` (gitignored).
- Run it with `.venv/bin/python web/mock/make_mock.py`. Serve with
  `.venv/bin/python -m http.server 8765 -d web` and check that every file loads without 404s
  (`curl` them).

## Quality checks before finishing
- `node --check` on every JS file if node is available. If not, be careful with syntax.
- No console errors in mock mode as far as you can verify (the code paths are guarded).
- Keyboard: space toggles play, ←/→ step frames.
- Add `web/README.md` with a short description of the UI architecture and the event handling.
- Add `web/mock/*.mp4` to the repo `.gitignore` (append only).
- End with a concise summary of what you built and anything the backend must provide (endpoint
  shapes).
