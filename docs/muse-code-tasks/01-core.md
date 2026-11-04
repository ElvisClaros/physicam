# Task 01: core library (meta client, video, SAM tracking, calibration, physics) + tests

Read `AGENTS.md` first. It holds verified API facts and the architecture. Build the core of the
`physicam` Python package. Do not build the agent, the server or the UI yet (later tasks), but keep
interfaces ready for them.

## Deliverables

1. `requirements.txt` (pinned to the versions installed in `.venv`, plus `pytest`) and a minimal
   `pyproject.toml` (package `physicam`, Python ≥ 3.11).
2. `physicam/config.py`: `Settings` dataclass loaded from env. It holds the API key (`MODEL_API_KEY`,
   then `META_API_KEY`), `base_url`, model ids (`spark="muse-spark-1.3"`, `wonder="muse-wonder-1.0"`,
   `sam="sam-3.1"`, `asr="muse-voice-transcribe-1.0"`) and `data_dir="data"`.
3. `physicam/meta_client.py`: `MetaClient` (sync httpx) with these methods:
   - `upload_file(path) -> file_id`, cached by sha256 in `data/cache/files.json`. File ids do not
     expire.
   - `responses(body) -> dict`, with retries on 429/5xx/timeouts (exponential backoff, 4 tries) and
     a 600 s timeout.
   - Helpers: `output_text(resp)`, `function_calls(resp)`, `usage(resp)`.
   - `transcribe_wav(path, mode="PUSH_TO_TALK") -> dict`.
   - A simple usage/cost accumulator (SAM frames, tokens per model) that pipeline code can read.
4. `physicam/video.py`:
   - `probe(path) -> VideoInfo`, by parsing `ffmpeg -i` stderr: fps (prefer avg/real frame rate
     over tbr), duration, width, height, rotation (displaymatrix/rotate tag), codec. `n_frames` comes
     from fps·duration, or from a fast decode count when cheap.
   - `normalize(src, dst, max_side=1280, t_start=None, t_end=None)`: H.264 yuv420p, CFR at the
     source fps (keep high fps for slow-motion clips, which matters for precision), apply rotation,
     `+faststart`, no audio. Returns VideoInfo.
   - `make_proxy(src, dst)`: 30 fps, 640 px wide, H.264, keeps audio if present (Spark can listen to
     narration).
   - `extract_frame(path, t) -> PIL.Image` and `extract_frames(path, times)`.
   - `contact_sheet(path, n=6) -> PIL.Image`.
   - `sha256_file`.
5. `physicam/sam.py`:
   - `segment_video(client, video_path, prompt, fps, *, cache=True) -> list[Track]`: upload, call
     SAM 3.1, parse with `meta_sam_parser`, and build tracks. Cache the raw SAM text under
     `data/cache/sam/<sha>_<prompt-slug>.txt` so re-runs are free.
   - `Track` dataclass: `id, label, frames (np.int array), t (seconds = frame/fps), x, y`
     (mask-weighted sub-pixel centroid in frame px), `area` (px²), `bbox`, `conf`, `eq_diameter`
     (px), `contours` (dict frame -> svg path in frame coordinates, kept for every k-th frame so
     the total stays ≤ ~300 contours, simplified). Methods: `to_json()` and `summary()` (compact
     dict for the LLM: frames visible, t range, median size, x/y extents, gaps).
   - Contours: `decode_mask_to_svg_path(mask)` returns a path in raster coordinates. Transform it
     to frame coordinates with bounds and the scale factor. Check the transform visually or
     numerically against the bbox.
   - `stitch_tracks(tracks, fps)`: drop near-duplicates (same frames, high box IoU, keep the higher
     confidence). Merge tracks that do not overlap in time when the end of A and the start of B are
     consistent: a gap ≤ 0.5 s, and the position extrapolated with velocity plus gravity-agnostic
     constant acceleration from A's last points lands near B's first point, or both lie near the
     same image border (exit/re-entry). Keep it simple and deterministic, and document heuristics.
   - `segment_image(client, pil_image, prompt) -> list[dict]`: boxes, masks, area, and the
     raster's major/minor axis lengths in frame px (from mask second moments and the min-area-rect
     extent).
   - `parse_sam_text(text, fps)`: the pure parser, used by the cache and by tests.
6. `physicam/calibrate.py`: `Calibration(m_per_px, rel_uncertainty, method, details)` and these
   functions:
   - `from_reference(client, video_path, t, prompt, real_size_m, dimension)` with dimension in
     `"longest" | "width" | "height" | "diameter"`. Segment the reference on 3 frames spread over
     the clip and take the median. Base the uncertainty on 1 px of edge error per side plus the
     spread between frames.
   - `from_track_size(track, real_diameter_m)` uses the median equivalent diameter. Assume a 3%
     uncertainty from mask-edge bias, or larger if the size varies (e.g., motion blur).
   - `from_gravity(fit_accel_px)` sets m_per_px so that g = 9.81. This is the "gravity as the
     ruler" mode.
   - `none()` reports results in px units.
7. `physicam/physics.py`: the models listed in `AGENTS.md` (`free_fall`/`projectile`, `bounce`,
   `pendulum`, `oscillator`, `constant_velocity`, `polynomial`) with the `Fit` dataclass (JSON
   serializable), proper uncertainties (covariance with outlier rejection, combined with the
   calibration's rel. uncertainty), and helpful `warnings` (e.g., "only 9 points in flight", "large
   residual structure: maybe air drag", "motion not in image plane?"). `bounce` is the most
   important one, because the demo uses it: a robust impact detector on noisy, possibly gappy
   trajectories (the ball may leave the frame at the top), per-flight parabola fits, per-impact
   restitution, and pooled g with uncertainty.
8. `physicam/__main__.py` + `cli.py`. These subcommands:
   - `python -m physicam probe <video>`
   - `python -m physicam track <video> "<prompt>" [--t0 --t1]` prints track summaries and writes
     `data/tracks/<name>.json`.
   - `python -m physicam fit <tracks.json> <track_id> <model> [--t0 --t1] [--m-per-px X]` prints
     the fit.
9. Tests (`tests/`):
   - Physics on synthetic numeric trajectories with noise (σ = 0.7 px) and gaps: free fall recovers
     g within 1%, bounce recovers e and g, pendulum recovers T within 0.5%, constant velocity, and
     outlier rejection.
   - Parser on the real fixture `tests/fixtures/bounce240_red_ball_f0-199.sam.txt.gz`: tracks are
     built and centroids lie inside boxes.
   - Stitching logic on synthetic tracks.

## Validation on real footage (do this; small cost is fine)

Run `python -m physicam track _probe/real/bounce_clip.mp4 "red ball"`. It is a real 240 fps clip of
a red bouncy ball thrown in a room; SAM returns about 2 ids per object. Then run `fit … bounce` on
the stitched track. Report:
- The flight segments found.
- g in px/s² per segment. They should be mutually consistent; the slow-motion file is real 240 fps
  timing.
- e per impact.
- The R² values.

If the per-segment accelerations disagree by more than ~5%, investigate (wrong time base? segment
boundaries? perspective as the ball moves toward or away from the camera?) and describe it in your
final message.

## Finish
- `.venv/bin/python -m pytest -q` passes.
- Finish with a concise summary: files created, key design decisions, validation numbers, and known
  limitations.
