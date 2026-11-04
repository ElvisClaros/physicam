# Task 02: planning, the Muse Spark agent, pipeline, report, server

Read `AGENTS.md` (verified API facts, architecture, result contract) and the code already in
`physicam/` (config, meta_client, video, sam, calibrate, physics, cli). The web UI in `web/` is being
built in parallel against the contract in AGENTS.md. **Do not modify files in `web/`** except
`web/mock/` if strictly needed. Your job is to make the backend serve that UI.

## 1. `physicam/plan.py`: Muse Spark watches the video
`plan_experiment(client, proxy_video_path, question, facts) -> Plan`. It uses Spark 1.3 video
understanding on the 30 fps / 640 px proxy, which keeps audio so Spark can hear narration such as
"the string is 80 cm long". The output is strict `json_schema` structured output with these fields:
- `experiment` (short description) and `phenomenon` (enum: free_fall, projectile, bounce, pendulum,
  oscillator, rolling, constant_velocity, other).
- `moving_objects`: `[{sam_prompt, role, description, standard_size_m|null}]`. Ask for specific,
  visually distinctive noun phrases: color + object, such as "red ball" or "yellow tennis ball".
- `reference_candidates`: `[{sam_prompt, real_size_m, dimension, confidence, why}]`. These are
  objects of standard known size visible and roughly static: A4 sheet 0.297 × 0.210 m, credit card
  0.0856 m, letter paper, tennis ball 0.067 m, ping-pong ball 0.040 m, soda can 0.122 m tall, a ruler
  or tape measure with readable marks, and so on.
- `spoken_facts`: what the narrator says, with any numbers and units.
- `time_window_s` [start, end] where the relevant motion happens.
- `slow_motion` (bool), `capture_fps_guess|null`, `camera_static` (bool) and `quality_issues[]`.
- `recommended_model`: one of the physics.py models.
- `target_quantity`: what the question asks for, with unit.

Put the known-size table in the prompt. The user's explicit `facts` override everything: reference
object + size, capture fps (for slow-mo files whose container fps is the playback rate), and object
diameter.

## 2. `physicam/agent.py`: tool-calling agent (Muse Spark 1.3, reasoning effort "low"/"medium")
- The system prompt makes the agent a careful experimental physicist. Calibration preference:
  1. User facts or spoken facts.
  2. A static reference object of standard size.
  3. The tracked object's standard size.
  4. Gravity as the ruler (when the question is NOT g itself).
  5. Pixels.

  The agent must always report uncertainty, inspect fit quality (R², residual RMS in px, warnings),
  and refit with a different window/model when the fit is poor. It must never invent numbers: every
  number in the answer comes from tool outputs.
- Tools (JSON-schema function tools). The implementations call the existing modules and return
  compact JSON:
  - `track_objects(prompt, t_start?, t_end?)` returns stitched track summaries with ids. A track
    whose label already exists is cached in the session.
  - `calibrate_reference(prompt, real_size_m, dimension)`
  - `calibrate_from_track(track_id, real_diameter_m)`
  - `calibrate_from_gravity(track_id, t_start?, t_end?)`
  - `fit_model(track_id, model, t_start?, t_end?, options?)` returns params ± stderr, r2, rmse_px,
    warnings and segments (for bounce).
  - `inspect_frames(times_s[], question)` sends up to 4 extracted frames, as JPEG data URLs, to Spark
    vision. Use it to disambiguate tracks or read a ruler.
  - `run_python(code)` runs in the sandbox (below) with `tracks` (dict id → numpy arrays t, x, y,
    area) and `m_per_px` available, and returns the printed JSON.
  - `finish(headline, value, unit, uncertainty, summary, fit_ids[])`
- The loop uses `previous_response_id` chaining and runs at most 12 model steps and 8 minutes
  total. Emit a `trace` event per step via a callback. If the model calls tools in parallel, run
  them in order.
- **Deterministic fallback**: if the agent errors, times out, or ends without `finish`, run a
  default recipe from the plan. Track the first moving object, calibrate by the best available
  method, fit `recommended_model`, and answer with a templated summary. The fallback must also be
  what runs when `PHYSICAM_OFFLINE=1`, using only caches.

## 3. `physicam/sandbox.py`
Run code in a subprocess using the venv python, with `-I`, a 20 s timeout, a temp cwd, and an env
stripped of all keys (no `MODEL_API_KEY`/`META_API_KEY`). Inputs come as an npz plus a JSON file.
Prepend a prelude that loads `tracks` and `m_per_px` and imports numpy/scipy. Capture stdout (max
8 KB) and return `{ok, stdout, error}`.

## 4. `physicam/report.py`
`write_report(client, result) -> markdown` uses **Muse Wonder** (`muse-wonder-1.0`, reasoning effort
low or medium, timeout 120 s, fallback to Spark 1.3 on error). The input is a compact JSON of the
result: no arrays, only summaries, fits and calibration. Sections: Question · What Physicam saw ·
Method · Results (a markdown table) · Discussion (comparison with accepted values where they exist,
dominant error sources, what the residuals say) · How to measure better next time. Keep it to about
300 words.

## 5. `physicam/pipeline.py`: job orchestration
`run_job(job_dir, video_path, question, facts, emit)` runs these stages, each emitting `stage` events:
1. `ingest`: probe and normalize (H.264 CFR, max 1280 px) into `job_dir/video.mp4`, make the proxy,
   compute the time base (capture fps override!).
2. `plan`
3. `agent`
4. `report`
5. `done`

After each change, write `job_dir/result.json` following the AGENTS.md contract exactly (tracks
include contours, `color` from a fixed palette, and `t` in seconds using the capture fps). Emit
`track`, `fit`, `answer`, `report` events as the data becomes available, and keep cost accounting.
Errors are caught, `status="error"`, and an `error` event is emitted. Everything is cached by video
sha256, so rerunning the same video + question is fast.

## 6. `physicam/server.py`: FastAPI, served with `python -m physicam serve`
- `POST /api/jobs` (multipart: `video`, `question`, optional `facts` JSON string) returns `{id}`. It
  runs the job in a background thread.
- `GET /api/jobs/{id}` returns result.json, partial while running.
- `GET /api/jobs/{id}/events` streams SSE (`text/event-stream`, `data: <json>\n\n`). It replays all
  past events first, so late subscribers get everything, then streams live events and ends after
  `done`/`error`. Events persist to `job_dir/events.jsonl`, so finished jobs replay after a server
  restart. Send a heartbeat comment every 15 s.
- `GET /media/{id}/video.mp4` needs **HTTP Range support** (the video element needs it for seeking).
  Verify with `curl -r 0-99 -i` that you get a 206.
- `POST /api/transcribe` (multipart `audio`, any browser format) converts to 16 kHz mono WAV with
  ffmpeg, calls Muse Voice Transcribe, and returns `{transcript}`.
- `GET /api/examples` reads `examples/examples.json`:
  `[{id, title, subtitle, question, job_id, thumb_url}]`. Example job dirs live in
  `examples/jobs/<job_id>/`; serve their media/results the same way as normal jobs (search both
  `data/jobs` and `examples/jobs`).
- `GET /api/health`
- Static: mount `web/` at `/` (index.html), after the API routes.
- `GET /api/jobs/{id}/thumb.jpg`: the first-frame thumbnail.

## 7. CLI
`python -m physicam analyze <video> "<question>" [--facts JSON] [--job-dir DIR]` runs the pipeline,
prints events live (one line per event) and finally the answer + report path.
`python -m physicam serve [--host --port]`.

## 8. Tests (`tests/`, no network)
Use a `FakeClient` that returns canned responses:
- The SAM text from the fixture.
- A canned plan.
- A canned tool-call sequence: track, then calibrate, then fit, then finish.

Test these:
- The agent loop executes tools and produces an answer.
- The fallback works when the model returns garbage.
- The sandbox blocks env keys and times out.
- The pipeline writes a contract-valid result.json.
- Server endpoints via FastAPI TestClient: jobs, events replay, Range 206, examples.

## 9. Live validation (required; keep the cost small)
Run `python -m physicam analyze _probe/real/bounce_clip.mp4 "What is the coefficient of restitution of this ball, and what does the throw tell us about g?"`.
This is a real 240 fps clip and the file's container fps is the true capture fps. Then start the
server and check that `/`, `/api/jobs/<id>`, the SSE stream and `/media/...` work with curl.

Finish with a summary: the files, the agent's actual tool sequence on the real clip, the answer it
produced, the costs, and any problems you could not solve.
