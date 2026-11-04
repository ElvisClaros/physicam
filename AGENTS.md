# Physicam: agent instructions

Physicam turns any phone video into a quantitative physics measurement. The user uploads a clip,
such as a falling ball, a pendulum, a bouncing ball or a cart on a ramp, and asks a question in plain
language ("what is g?", "what's the coefficient of restitution?", "how fast is it going?"). Physicam
then runs these steps:

1. **Muse Spark** watches the video (video understanding) and plans the experiment: which objects
   move, what physical model applies, and what can serve as a length reference.
2. **SAM 3.1** tracks the objects through every frame (video concept segmentation) and returns
   pixel-accurate masks. We turn them into sub-pixel centroid trajectories.
3. The **Muse Spark agent** (tool calling) calibrates pixels to meters, fits physical models with
   uncertainties, checks residuals and refits, then answers.
4. **Muse Wonder** writes a short lab report.

The product is for the **Meta Global AI Developer Hackathon** (theme: *Push the Frontier*,
"accelerate a scientific experiment / understand a physical system"). Judging covers impact, creative
use of Meta models, technical depth, and a working, impressive demo. The quality bar is high:
correct physics, honest uncertainties and a polished UI.

## Hard rules

- **Never install system packages.** Use only the project venv: `.venv/bin/python`, `.venv/bin/pip`.
  These are already installed: `numpy scipy pillow imageio-ffmpeg meta-sam-parser fastapi uvicorn
  python-multipart httpx`, plus `pytest` if missing (`.venv/bin/pip install pytest`). If you truly
  need another pure-Python package, install it into `.venv` and add it to `requirements.txt`.
- **ffmpeg**: no system ffmpeg is available. Use `imageio_ffmpeg.get_ffmpeg_exe()` for the binary.
  For probing, run `ffmpeg -i` and parse stderr; ffprobe is not bundled.
- **Secrets**: the API key comes from env `MODEL_API_KEY`, falling back to `META_API_KEY`.
  Never print it, log it or write it to disk.
- **Do not `git commit`**. The lead engineer reviews and commits.
- Python ≥ 3.11, type hints, small focused modules, docstrings on public functions. No notebooks.
- Every network call to Meta goes through `physicam/meta_client.py`.
- Cache expensive API results on disk under `data/cache/` (gitignored), keyed by sha256 of the video
  plus the request parameters, so demos replay instantly and offline.

## Verified Meta Model API facts (tested against the live API, Nov 2026)

Base URL `https://api.meta.ai/v1`, header `Authorization: Bearer <key>`.

### Files
`POST /files` multipart: `file=(name, bytes, "video/mp4")`, `purpose=user_data`. It returns
`{"id": "file-…", …}` synchronously. Videos must be MP4/MOV with H.264/H.265; VP9/WebM is rejected,
so transcode first.

### SAM 3.1 video tracking (`model: "sam-3.1"`)
```json
POST /responses
{"model": "sam-3.1", "stream": false,
 "metadata": {"include_confidence": "true"},
 "input": [{"role": "user", "content": [
   {"type": "input_text", "text": "red ball"},
   {"type": "input_video", "file_id": "file-…"}]}]}
```
- The prompt is ONE short noun phrase ("red ball", "pendulum bob", "toy car"). Avoid commands,
  lists and negation. "red ball" works much better than "ball". For several objects, send one
  request per phrase.
- For images, use `{"type": "input_image", "image_url": "data:image/png;base64,…"}` instead.
- Concatenate the text of all `output_text` parts in `output[*].content[*]`. It holds one line per
  frame: `<Nf>id<|box;x1=..;y1=..;x2=..;y2=..;w=W;h=H;c=..|><|mask;x=0;y=0;c=..;data=…|>,id2<|box…|>…`.
  Frames with no detection have no line. Zero matches gives an empty string, which is not an error.
- `usage.video_frames_processed` gives the billed frames. Price: $0.20 per 1000 frames, minimum 50
  per request. Latency: about 40 s for 960 frames of 720p.
- Max 15,000 frames and 16 objects per request.
- **Parse with the official `meta_sam_parser`**. Do not hand-roll the mask codec:
  ```python
  from meta_sam_parser import CompletedOutcome, video_segmentation_format, decode_mask_to_raster, decode_mask_to_svg_path
  parser = video_segmentation_format().create_parser()   # image_segmentation_format() for images
  parser.push(text if text.endswith("\n") else text + "\n")
  result = parser.finish(CompletedOutcome()).result
  for rec in result.records:            # kinds: "box", "mask", "text"
      if rec.kind == "mask":
          rec.object_id, rec.frame.frame_index, rec.confidence
          m = rec.mask                  # m.width, m.height: raster size, smaller than the frame
          b = rec.bounds                # left, top, right, bottom in FRAME pixels; the raster maps onto this rect
          raster = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width)  # 0/1
          # Frame coords of raster pixel (i, j): x = b.left + (j + .5) * (b.right - b.left) / m.width
          #                                      y = b.top  + (i + .5) * (b.bottom - b.top) / m.height
  ```
- **Track identity is not stable through occlusion or exit/re-entry.** When an object leaves the frame
  and comes back, it gets a new id. For a single physical object, two ids can overlap in time (the
  same ball detected twice). Tracks need stitching and deduplication: merge ids whose time ranges do
  not overlap and whose endpoints are spatially and kinematically consistent; drop duplicates with
  high IoU.
- A synthetic flat-color video (a drawn circle on a plain background) gave **zero** detections in
  video mode, although the same frame worked in image mode. Test with real footage
  (`_probe/real/bounce_clip.mp4`, 1280x720, 240 fps, 960 frames, a real red bouncy ball). Unit-test
  physics with synthetic numeric data, not synthetic videos.

### Muse Spark: video understanding + structured output (`model: "muse-spark-1.3"`)
```json
POST /responses
{"model": "muse-spark-1.3", "max_output_tokens": 4000, "reasoning": {"effort": "low"},
 "input": [{"role": "user", "content": [
    {"type": "input_text", "text": "…"},
    {"type": "input_video", "file_id": "file-…"}]}],
 "text": {"format": {"type": "json_schema", "name": "plan", "schema": {…}, "strict": true}}}
```
- Works in about 8 s for a 640px, 30 fps proxy video. Upload a downscaled proxy (`fps=30,scale=640:-2`)
  for Spark, and the full-quality normalized video for SAM.
- The answer is in the `output` item with `type == "message"`, under `content[*].text`. Reasoning
  tokens share the output budget, so keep `max_output_tokens` ≥ 4000.
- Spark's guesses of real object sizes are unreliable: the same ball was estimated as 0.15 m and
  0.07 m. Treat them as priors with large uncertainty and prefer explicit references.
- Images: `{"type": "input_image", "image_url": "data:image/jpeg;base64,…"}`.

### Muse Spark tool calling (Responses API, OpenAI-compatible)
- `tools=[{"type": "function", "name": …, "description": …, "parameters": {JSON schema}}]`.
- The output contains items of type `reasoning`, `message` and `function_call` (`name`, `arguments` as
  a JSON string, `call_id`).
- To continue, send `previous_response_id: <id>` with
  `input=[{"type": "function_call_output", "call_id": …, "output": "<json string>"}]`. Both this and
  re-sending the full item list work.

### Muse Wonder (`model: "muse-wonder-1.0"`)
Same Responses API. Slower and pricier ($4/$20 per M tokens). Use it only for the final report.

### Voice: `POST /asr/transcribe`
Multipart with two parts:
- `request`: JSON with content type application/json, e.g.
  `{"mode": "PUSH_TO_TALK", "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV"}`.
- `audio`: a WAV file, mono 16-bit PCM at 16 kHz.

The response is JSON with `transcript`, `audioDurationMs` and `turns[{startMs, endMs, transcript}]`
(turns only in `ENDPOINTING`/`DIARIZATION` mode). Convert browser webm/opus to WAV with ffmpeg first.

## Architecture (target)

```
physicam/
  config.py        # settings from env (key, base url, model ids, data dir)
  meta_client.py   # httpx wrapper: upload_file, responses(...), output_text(...), function_calls(...), transcribe(...), retries/backoff
  video.py         # probe(path) -> VideoInfo(fps, n_frames, width, height, duration, rotation); normalize(src) -> H.264 CFR mp4 (+faststart), max 1280px long side, keep fps; make_proxy() for Spark; extract_frame(t) -> PIL; sha256
  sam.py           # segment_video(video, prompt) -> list[Track]; segment_image(img, prompt); track stitching/dedup; per-frame centroid (sub-pixel, mask-weighted), bbox, area_px, equivalent diameter, orientation, simplified contour (svg path, frame coords); cached
  calibrate.py     # Calibration(m_per_px, rel_uncertainty, method, details): from reference object (SAM image seg + known size), from tracked object's known diameter, or from gravity (inverse: assume g=9.81)
  physics.py       # model fits with uncertainties (see below)
  sandbox.py       # run model-written numpy/scipy code in a subprocess (timeout, no network env, no key), with `tracks`/`calibration` injected; returns JSON
  agent.py         # Muse Spark tool-calling loop (tools below), max ~12 steps, emits events
  report.py        # Muse Wonder lab report (markdown) from the result JSON
  pipeline.py      # Job orchestration: ingest -> plan -> agent -> report; writes data/jobs/<id>/result.json; emits progress events
  server.py        # FastAPI app: REST + SSE + static web/
  cli.py / __main__.py
web/               # static single-page UI (no build step)
tests/             # pytest; tests/fixtures/bounce240_red_ball_f0-199.sam.txt.gz is real SAM output (240 fps, 1280x720)
examples/          # fetch_examples.py downloads CC-licensed demo clips (with attribution) + cached results
```

### Physics models (`physics.py`)
All fits work in pixel space with image y pointing DOWN, then convert with the calibration. Each
fit returns `Fit(model, track_id, t_range, params: dict[name -> {value, stderr, unit}], r2, rmse_px,
n_points, curve: {t, x?, y?}, residuals, warnings)`. Uncertainty combines the curve_fit covariance
(or bootstrap) with the calibration's relative uncertainty in quadrature.

- `free_fall` / `projectile`: y = y0 + vy0·t + ½·a·t², x = x0 + vx·t. Report g = a·m_per_px, plus
  launch speed and angle. Optionally fit linear drag.
- `bounce`: detect impacts (velocity sign flips / local maxima of y), fit a parabola per flight
  segment, report g per segment and pooled, and the coefficient of restitution per impact:
  e = |v_up_after| / |v_down_before| from the fitted parabolas at the impact times, cross-checked
  with sqrt(h_{n+1}/h_n) from apex heights.
- `pendulum`: damped sinusoid A·e^(−γt)·cos(ωt+φ)+c on the dominant axis. Report period T, γ and
  Q. If a length L is known or measured, report g = 4π²L/T² with the finite-amplitude correction.
- `oscillator` (spring-mass): the same damped sinusoid, giving ω0, ζ and k/m.
- `constant_velocity`: linear fit, giving speed and direction.
- `polynomial`: generic, degree n.

Fits must be robust to outliers (one pass of residual-based rejection) and to missing frames.

### Agent tools (Muse Spark function calling)
`track_objects(prompt, t_start?, t_end?)`, `calibrate_reference(prompt, real_size_m, dimension)`,
`calibrate_from_track(track_id, real_diameter_m)`, `calibrate_from_gravity()`,
`fit_model(track_id, model, t_start?, t_end?, options?)`, `run_python(code)`,
`finish(headline, value, unit, uncertainty, summary, fit_ids)`. Tool results are compact JSON
summaries; never return raw masks to the model.

### Result JSON contract (`data/jobs/<id>/result.json`, also `GET /api/jobs/<id>`)
```json
{"id": "…", "status": "done|running|error", "question": "…",
 "video": {"url": "/media/<id>/video.mp4", "fps": 240, "width": 1280, "height": 720, "n_frames": 960, "duration_s": 4.0},
 "plan": {"experiment": "…", "objects": [{"sam_prompt": "…", "role": "…"}], "model": "…", "reference": {…}, "notes": "…"},
 "calibration": {"m_per_px": 0.0012, "rel_uncertainty": 0.05, "method": "reference|track_size|gravity|none", "details": "…"},
 "tracks": [{"id": "t0", "label": "red ball", "color": "#ff4d4f",
             "frames": [0, 1, …], "t": [0.0, …], "x": [..px..], "y": [..px..], "area": [..], "bbox": [[x1, y1, x2, y2], …],
             "contours": {"0": "M…Z", "5": "M…Z"}}],
 "fits": [{"id": "f0", "track_id": "t0", "model": "bounce", "t_range": [a, b], "params": {"g": {"value": 9.79, "stderr": 0.08, "unit": "m/s^2"}},
           "info": {"axis": "y"}, "r2": 0.999, "rmse_px": 0.7, "curve": {"t": [], "x": [], "y": []},
           "segments": [{"index": 0, "t_range": [a, b], "curve": {"t": [], "x": [], "y": []}}],
           "impacts": [{"t": 0.32, "x": 640.0, "y": 540.0, "e": 0.72, "e_stderr": 0.01, "e_height": 0.71, "valid": true}],
           "restitution": [], "impact_times": [], "warnings": []}],
 "answer": {"headline": "g = 9.79 ± 0.08 m/s²", "value": 9.79, "unit": "m/s^2", "uncertainty": 0.08, "summary": "…"},
 "strobe_url": "/media/<id>/strobe.jpg",
 "report_markdown": "…",
 "trace": [{"step": 1, "kind": "plan|tool|answer|report", "name": "track_objects", "args": {}, "summary": "…", "title": "SAM 3.1 tracked “red ball”: 960 frames (0.00–4.00 s)", "ms": 41000}],
 "cost": {"sam_frames": 960, "spark_tokens": 12345, "usd_estimate": 0.21}}
```
Fit contract notes: `curve` always carries `t` plus both `x` and `y` where meaningful (pendulum/oscillator
also set `curve.axis` and `info.axis`); non-numeric info goes in `info`, not `params`; no duplicate
`period`/`T` (keep `T`). Bounce fits emit `impacts[]` in the shape above (positions from the track);
`restitution[]`/`impact_times[]` are kept for compatibility. Plans give `sam_prompts` (2–4 short
1–3-word candidates) per moving object.
SSE events (`GET /api/jobs/<id>/events`): `{"type": "stage"|"trace"|"track"|"fit"|"answer"|"report"|"strobe"|"done"|"error", …}`; a `strobe` event carries `url`/`strobe_url`.

## Commands
- Tests: `.venv/bin/python -m pytest -q`
- Server: `.venv/bin/python -m physicam serve --host 0.0.0.0 --port 8000`
- CLI: `.venv/bin/python -m physicam analyze <video> "<question>"`
- Any command that calls the live API needs `MODEL_API_KEY` or `META_API_KEY` in env. It is set in
  this container. Keep live-API usage small: reuse cached results and short clips.
