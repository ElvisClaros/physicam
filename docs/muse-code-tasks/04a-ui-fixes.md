# Task 04a: UI fixes and additions (`web/` only)

Read `AGENTS.md` and `web/` (README, app.js, charts.js, style.css). Another engineer is editing
`physicam/` in parallel, so **only modify files under `web/`**.

## Fixes
1. **The video is hidden behind the empty state.** In mock mode after pressing Measure, or after
   loading an example, the stage still shows the "Drop a video" illustration instead of the video.
   Whenever a job or example has `video.url`, hide the empty state and show the `<video>` with its
   overlay canvas, sized to the video's aspect ratio. The first frame must be visible before play
   (set `preload="auto"` and seek to 0.001 on `loadeddata`). Verify the fix in code paths for
   mock, example, live job and `#job=` reload.
2. **Vertical position chart → "Height"**: plot physical height (up = positive). With calibration
   (`calibration.m_per_px`), use h = (y_ref − y)·m_per_px, where y_ref is the largest tracked y (the
   lowest point), labeled "height (m)". Without calibration, use (y_ref − y) labeled "height (px)".
   Transform fitted curves and residual signs the same way. If `calibration.kind == "per_frame"`
   and the track has `X_m`/`Y_m` arrays, plot those directly (with Y up).
3. **Restitution bar chart**: use categorical x labels "impact 1…n" with no numeric x ticks (they
   currently overlap). Show the bar value above each bar, and draw an e = 1 dashed reference line.
4. **Favicon**: an inline SVG data URL in index.html (no 404).
5. **Timeline**: auto-scroll to the newest row, and keep the running row's spinner until the next
   event.
6. **Stage progress overlay** while a job runs, driven by `stage` events, as a pill on top of the
   stage: `ingest` → "Preparing video…", `plan` → "Muse Spark is watching the video…", `agent` →
   "Agent at work · SAM 3.1 tracking…" (update with the latest trace summary), `report` → "Muse
   Wonder is writing the report…". It fades out on `done`.
7. **Answer card**: show "Δ vs 9.81 m/s²" only when the unit is m/s² (gravity). For dimensionless
   answers (e, ratios), show the value with stderr, and the cross-check if the fit provides
   `e_height`.
8. **Strobe view**: if the result has `strobe_url` (or an SSE `strobe` event `{url}` arrives), add
   a segmented control on the stage, "Video | Strobe". Strobe shows the image fitted to the stage.
   Also show the strobe as a figure at the top of the lab report section.
9. **Hero polish**: a compact "How it works" strip under the top bar, dismissible, with 4 steps
   (Spark watches → SAM tracks → agent measures → Wonder reports), each with its model badge color.
10. **Mock mode** should exercise all of the above: add `strobe_url: "mock/strobe.jpg"`. Generate a
    placeholder strobe in `make_mock.py` by compositing ~15 mask cut-outs from the fixture onto the
    first frame. Extract frames with `imageio_ffmpeg.get_ffmpeg_exe()` and decode masks with
    `meta_sam_parser.decode_mask_to_raster`; the mask raster maps onto the `bounds` rect.

## Checks
- `node --check` on all JS.
- Serve with `.venv/bin/python -m http.server 8765 -d web` and curl every asset referenced by
  index.html in mock mode (200s).
- Update `web/README.md`.
- Finish with a summary.
