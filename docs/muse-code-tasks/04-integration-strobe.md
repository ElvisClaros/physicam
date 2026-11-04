# Task 04: backend–UI integration fixes, agent speed, stroboscopic composite

Read `AGENTS.md`, then `physicam/` and `web/`. The backend (task 02) and UI (tasks 03/04a) were built
in parallel against the contract. A real end-to-end run (job `42e1d5a3f33a` in `data/jobs/`, a spring
pendulum, and `live-bounce-1`, a bounce) exposed integration bugs. You may edit both `physicam/` and
`web/`. A server is running on 127.0.0.1:8000 from the host. **Do not start another server on port
8000**; use port 8010 for your own tests. Check the UI with the real result JSONs in `data/jobs/*/`.

## 1. Integration bugs (fix first)
1. **JS crash** `Cannot read properties of undefined (reading '0')` at `web/app.js` `drawOverlay`
   (~line 616) and `curveAt` (~line 382). Pendulum/oscillator fits have `curve: {t, x}` with no `y`,
   but the UI assumes `curve.y`. The crash also prevents the answer card and charts from rendering.
   - The UI must handle curves with any subset of `x`/`y`. When drawing the overlay ghost path,
     take a missing axis from the track's own interpolated position.
   - The backend should also always emit both axes where meaningful: for pendulum/oscillator, add
     the other axis as the track's smoothed position, or emit an `axis` field so the UI knows which
     axis the curve describes.
   - Wrap per-fit rendering in try/catch so one bad fit can never blank the page.
2. **Restitution/impacts contract**: the backend emits `restitution[]` + `impact_times[]`, and the UI
   expects `impacts[]` of `{t, x, y, e, e_stderr, e_height, valid}`. Make the backend emit `impacts`
   in exactly that shape, with impact positions from the track and `segments[].curve` = `{t, x, y}`.
   Update AGENTS.md's contract section to match.
3. **Portrait videos**: a 720×1280 clip makes the stage enormous. Constrain the stage to
   `max-height: min(72vh, 760px)`, keep the aspect ratio, and center it (letterbox). The overlay
   canvas must match the displayed video rect exactly.
4. **Timeline readability**: trace rows show raw JSON from tool outputs. The backend should add a
   human `title` to every trace event, e.g.:
   - "SAM 3.1 tracked “metal ball”: 491 frames (0.43–16.8 s)"
   - "SAM 3.1 found nothing for “silver metal weight”"
   - "Fit pendulum → T = 0.9633 ± 0.0003 s (R² 0.965)"
   - "Looked at 3 frames: …first 80 chars of Spark's answer"
   - "Python: mean T = 0.963 s"

   The UI shows `title` in the row and keeps args/summary in the collapsible details,
   pretty-printed and truncated.
5. **Parameters table**:
   - Do not emit duplicate params (`period` and `T`): keep `T`.
   - Put non-numeric info (`axis`) in a separate `info` dict, not `params`.
   - A param with stderr 0 because it is derived should show its propagated error or "—".
   - In the UI, show the **primary fit** (the first of `answer.fit_ids`) expanded, and the other
     fits collapsed under "Other fits (n)".
   - Show px-unit params with their converted metric value when a calibration exists.
6. **Answer card**: must render for every unit (s, m/s², dimensionless, m/s). The format for value
   ± uncertainty uses 2 significant digits on the uncertainty and matching decimals on the value.
7. The HUD text and the stage-progress pill overlap at the top-left of the stage. Move the progress
   pill to top-center or bottom-center.

## 2. Agent speed and robustness
- SAM prompts must be **1–3 words: color/material + noun** ("metal ball", "red ball", "spring"). Long
  descriptive prompts failed every time on the real run. Enforce this in the tool description and
  in the system prompt. Also truncate or simplify a prompt over 4 words before calling SAM, and say
  so in the result.
- `plan_experiment` should return 2–4 candidate SAM prompts per moving object
  (`sam_prompts: [...]`). Before the agent starts, the pipeline runs SAM on the candidates
  **concurrently** (threads; SAM allows 3000 RPM). It keeps those with tracks and hands the agent a
  ready summary ("pre-tracked: …"). `track_objects` also accepts `prompts: [..]` and runs them
  concurrently. This should cut the 3.8-minute spring run to about 1.5 minutes.
- The run_python summary should be its stdout parsed as JSON when possible.

## 3. Stroboscopic composite
This is the classic physics-textbook "strobe photo", made from SAM masks. For each track, take the
pixel-accurate masks from every k-th frame and paste the object's pixels from those frames onto a
clean background plate. Choose k so that you get about 12–25 copies with little overlap. The
background plate is the per-pixel temporal median of ~15 frames spread over the clip, which removes
the moving object. Add a subtle drop shadow or outline. Optionally draw the fitted curve as a thin
dashed line and small time labels ("0.10 s").

- Needs mask rasters per frame. If the track only keeps contours, re-parse the cached SAM text (it is
  on disk) to get rasters. Do not call the API again.
- Output `job_dir/strobe.jpg`. The pipeline creates it after tracking. Add `strobe_url` to
  result.json and emit a `strobe` SSE event.
- The UI already has a "Video | Strobe" control and a report figure (task 04a) that consume
  `strobe_url` and the `strobe` event. Wire the backend to them, and make sure the strobe is
  generated for existing jobs when re-run from cache.

## 4. Finish
- `.venv/bin/python -m pytest -q` passes. Add tests for the new contract fields (`impacts`, `info`,
  trace `title`) and for concurrent multi-prompt tracking with the FakeClient.
- Re-run the pipeline **offline from cache** on `data/jobs/42e1d5a3f33a`'s video and on the bounce
  clip, so the result.json files get the new fields; use `PHYSICAM_OFFLINE=1` or the cache.
- Verify in the served UI on port 8010 (fetch the result JSON and run app.js-equivalent checks; no
  browser is available) that no field the UI reads is missing.
- Summarize the changes.
