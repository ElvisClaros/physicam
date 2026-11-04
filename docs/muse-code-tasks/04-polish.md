# Task 04: polish, stroboscopic composite, examples, README

Read `AGENTS.md`, then `physicam/` and `web/` as they are now. The backend and UI work end to end.
This task makes the project demo-ready and judge-ready.

## A. Stroboscopic composite (`physicam/strobe.py`), a signature visual
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
- In the UI, add a "Strobe" tab or toggle on the video stage that shows the image. Use it in the
  report too: the report markdown may reference it.

## B. UI
The UI fixes (strobe view, height chart, stage progress, etc.) were done in task 04a. Only touch `web/` if the backend changes require it, such as a new result field.

## C. Examples (`examples/`)
- `examples/fetch_examples.py` downloads the CC-licensed clips from Wikimedia Commons, using the
  Commons API with a User-Agent of `physicam/0.1 (hackathon demo)`. It transcodes them to H.264 MP4
  into `examples/media/`. Write `examples/ATTRIBUTION.md` with title, author, license and URL for
  each. Clips:
  - `File:Bouncy ball 240fps.webm` (CC BY-SA 4.0), segment 6–10 s: a real 240 fps red bouncy ball.
  - `File:ElasticPendulum.webm` (CC BY-SA 4.0), full: a spring pendulum (vertical phone video, 30
    fps). SAM tracks "spring" well; "metal weight" gave no detections.
- `examples/build_examples.py` runs the full pipeline on each clip, with a question, and copies the
  finished job dir into `examples/jobs/<slug>/` (result.json, events.jsonl, video.mp4, strobe.jpg,
  thumb.jpg). It writes `examples/examples.json` for `GET /api/examples`. It must also accept user
  clips placed in `examples/media/` with a sidecar `<name>.json` of
  `{title, subtitle, question, facts}`.
- Build both examples now (live API; the cost is small).
- Keep committed media small: re-encode each example video to ≤ 6 MB (CRF 26–28, 720p max).

## D. README.md (judges read this; make it excellent but honest)
- Title, one-line pitch, and a hero screenshot placeholder (`docs/img/hero.png`; the lead engineer
  adds screenshots).
- **Why**: lab motion sensors and photogates cost hundreds of dollars and most students and makers
  don't have them, but everyone has a phone camera. Physicam turns a phone video into a
  markerless motion-capture instrument with error bars. Uses: science education labs, makers and
  engineers (vibration, cycle times, mechanism tuning), research (behavior tracking, materials
  tests).
- **How it works**: a mermaid diagram of the pipeline (Spark watches, SAM tracks, Spark agent tools,
  Wonder reports), plus a table of every Meta model and exactly how it is used.
- Physics details: sub-pixel mask centroids, calibration hierarchy (with "gravity as a ruler"), the
  models, uncertainty propagation, and outlier handling.
- **Validation**: a table of each example, the measured value, the known or expected value, and the
  error. Report results honestly, including the perspective caveat on the bounce clip.
- Quickstart: `python -m venv .venv && .venv/bin/pip install -r requirements.txt`, set
  `MODEL_API_KEY`, `python -m physicam serve`. Examples work without an API key (cached).
- **Built during the hackathon with Muse Code**: explain that the code was written by the Muse Code
  agent (muse-wonder-1.0) from the task specs in `docs/muse-code-tasks/`, with a human lead
  reviewing, validating with real footage and committing. Link the specs.
- Limitations and next steps: perspective/depth (2D only), rolling shutter, phone VFR, multi-object
  interactions, and a live phone capture PWA.
- License: MIT for the code. Third-party: meta-sam-parser (SAM License), example media CC BY-SA
  (see ATTRIBUTION.md).
- Add a `LICENSE` file (MIT, copyright "Elvis Claros").

## F. Static demo mode (GitHub Pages; judges get a live URL with no backend)
- `scripts/build_static.py` builds `site/` (gitignored), a fully static copy that works when served
  from any static host under a sub-path (e.g. `https://user.github.io/physicam/`):
  - Copy `web/` and the finished example jobs: `result.json`, `events.jsonl`, `video.mp4`,
    `strobe.jpg`, `thumb.jpg`.
  - Rewrite example `video.url` / `strobe_url` / `thumb_url` to **relative** paths, and write
    `site/static-examples.json`.
- In `web/app.js`, if `GET api/health` fails (no backend), switch to **static mode**:
  - Load `static-examples.json` and render the examples gallery.
  - Clicking an example replays its `events.jsonl` with realistic but compressed timing (about
    8–10 s total) and then shows the final result.
  - Show a small banner: "Static demo: replaying real Physicam runs. Run locally with your API key
    to measure your own videos."
  - Disable upload and Measure, with a tooltip explaining why.
- Make every fetch URL in the frontend relative (no leading `/`) so it works under a sub-path.
  Keep the FastAPI server routes working: mount static `web/` at `/` and keep the `api/...` paths.
- Test: `python scripts/build_static.py && .venv/bin/python -m http.server 8766 -d site` returns
  200s for every referenced asset.

## E. Quality
- `.venv/bin/python -m pytest -q` passes.
- Run the server and click through the examples via curl/API checks.
- Remove dead code. In `physics.detect_impacts`, line ~365 contains a leftover
  `... if False else ...` expression, and `fit_bounce` has an empty `pass` block for the height
  cross-check. Make sure e's uncertainty propagation includes the linear-coefficient covariance, not
  only the quadratic term.
- Finish with a summary of changes, the example results (values), and remaining issues.
