# Task 11b: make mask decoding fast (parallel + cached), then build the draining example

Read `AGENTS.md`, `physicam/sam.py`, `physicam/level.py`, `physicam/strobe.py` and
`physicam/pipeline.py`. Do not start servers on port 8000.

## Problem (measured)
Building the draining example (`examples/media/fiuba-draining-water.mp4`: 2471 frames at 60 fps,
3 candidate SAM prompts) ran for **more than 32 minutes at 93% CPU** without finishing. Every stage
decodes every mask with `meta_sam_parser.decode_mask_to_raster`, a pure-Python range decoder: the
`parse_sam_text` centroids, contours (`decode_mask_to_svg_path`, also pure Python), dedup, `level`
and `strobe`. That happens again for each candidate prompt. The SAM texts are already cached in
`data/cache/sam/` (prompts "brown water", "dark liquid", "water" for sha `d3f968a1…`), so no API
calls are needed to test.

## Fix
1. `physicam/maskcache.py` (new): `decoded_masks(sam_text_path_or_text, cache_key) -> MaskSet`.
   - Parse the records once.
   - Decode all mask rasters **in parallel** with `concurrent.futures.ProcessPoolExecutor`
     (`max_workers = min(os.cpu_count(), 12)`), in chunks of ~64 records per task to amortize IPC.
     Each worker returns per-record results: the raster bit-packed with `np.packbits`, plus shape,
     bounds, object id, frame and confidence.
   - Store the results in `data/cache/masks/<key>.npz` (compressed). Subsequent calls load the npz
     in well under 1 s.
   - API: iterate per object/frame, `raster(i)` unpacks lazily, and precomputed per-record stats
     (area, centroid x/y in frame coordinates, eq diameter, row-width profile summary,
     top/bottom full-width rows for level) are computed in the worker, so most consumers never
     unpack.
2. Use it everywhere masks are needed: `parse_sam_text` (centroids/area from the stats), level
   extraction, strobe, and image segmentation where applicable.
   - **Contours**: compute SVG paths only for the ≤ 300 frames actually kept (every k-th), in the
     same worker pool and cached.
   - **Dedup across prompts** must use only box records and centroid stats. It must not need
     rasters.
3. Pre-tracking of candidate prompts: when a candidate turns out to be a duplicate of a kept track
   (cross-prompt dedup), skip its contour generation entirely. Decode the duplicates' masks lazily
   or not at all: box-level dedup can be done **before** any raster decode by parsing boxes only
   (the parser yields box records without decoding).
4. Benchmark on the cached "brown water" SAM text (2471 frames). Print the cold time (decode +
   cache write) and warm time (npz load) for parse→tracks, level and strobe. **Target: cold under
   90 s and warm under 5 s for the full pipeline on cached SAM text.**
5. Then build the draining example **live** (SAM is cached, so only Spark/Wonder calls are live)
   with `examples/build_examples.py`, only for `fiuba-draining-water`. Add a `--only <slug>` flag if
   it does not exist. Verify `result.json` is complete: level, draining fit, validation vs
   `examples/validation/fiuba_tabla_agua_5mm.csv`, strobe, and report.
6. Tests: the maskcache roundtrip (packbits) equals direct decode on the real fixture, and parallel
   equals serial. `.venv/bin/python -m pytest -q` passes.
7. Summary: the benchmark numbers, and the draining example's answer: Cd ± σ, t_f vs t_f,theory,
   times to 10/25/50/75/90% vs the students', and RMS vs the manual table in cm and h/h0.
