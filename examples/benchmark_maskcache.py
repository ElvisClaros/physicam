"""Benchmark the parallel mask cache on the cached draining SAM text.

Cold = one parallel decode + cache write, shared by every stage (as in
the real pipeline); warm = npz load. Stages: parse->tracks, level
extraction and strobe on the "brown water" SAM text for
``examples/media/fiuba-draining-water.mp4`` (2471 frames at 60 fps).
No API calls: the SAM text comes from ``data/cache/sam/``.

Run: ``.venv/bin/python examples/benchmark_maskcache.py``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def _clear_mask_cache(data_dir: str, cache_key: str) -> None:
    from physicam.maskcache import mask_cache_path

    p = mask_cache_path(data_dir, cache_key)
    p.unlink(missing_ok=True)
    p.with_name(p.name[:-4] + ".contours.npz").unlink(missing_ok=True)


def main() -> None:
    """Run the cold/warm benchmark and print a small table."""
    from physicam.level import level_series, level_series_from_maskset
    from physicam.maskcache import decoded_masks, mask_cache_path, plan_contour_indices, parse_records
    from physicam.sam import mask_cache_key, parse_sam_text, stitch_tracks, MAX_CONTOURS
    from physicam.strobe import cached_sam_text, make_strobe
    from physicam.video import normalize, sha256_file

    src_video = ROOT / "examples" / "media" / "fiuba-draining-water.mp4"
    prompt = "brown water"
    data_dir = "data"
    # The pipeline segments the normalized video; its sha keys the SAM cache.
    bench_dir = ROOT / "data" / "bench_strobe"
    bench_dir.mkdir(parents=True, exist_ok=True)
    video = bench_dir / "video.mp4"
    info = normalize(src_video, video)
    sha = sha256_file(video)
    text = cached_sam_text(data_dir, sha, prompt)
    if text is None:
        raise SystemExit(f"no cached SAM text for prompt {prompt!r}")
    key = mask_cache_key(sha, prompt)
    print(f"video={src_video.name} frames={info.n_frames} fps={info.fps} prompt={prompt!r} "
          f"sam_chars={len(text)} cache_key={key}", flush=True)

    def parse_tracks():
        return stitch_tracks(parse_sam_text(text, info.fps, label=prompt,
                                            cache_key=key, data_dir=data_dir), info.fps)

    def level():
        return level_series(video, None, info.fps, prompt, data_dir=data_dir)

    # ---- cold: a single decode + cache write, shared by all stages ----
    _clear_mask_cache(data_dir, key)
    _boxes, _masks = parse_records(text)
    contour_idx = plan_contour_indices(_boxes, _masks, MAX_CONTOURS)
    t0 = time.perf_counter()
    ms = decoded_masks(text, key, data_dir=data_dir, contour_indices=contour_idx)
    decode_s = time.perf_counter() - t0
    print(f"mask decode (cold, shared): {decode_s:.2f}s for {len(ms)} records", flush=True)

    t0 = time.perf_counter()
    tracks = parse_tracks()
    parse_cold = decode_s + (time.perf_counter() - t0)
    t0 = time.perf_counter()
    lvl = level_series_from_maskset(ms, info.fps, prompt)
    level_cold = decode_s + (time.perf_counter() - t0)
    t0 = time.perf_counter()
    out_path = make_strobe(video, tracks, bench_dir, data_dir,
                           width=info.width, height=info.height)
    strobe_cold = decode_s + (time.perf_counter() - t0)
    # Pipeline cold counts the shared decode once, not once per stage.
    t0 = time.perf_counter()
    tracks = parse_tracks()
    lvl2 = level()
    make_strobe(video, tracks, bench_dir, data_dir, width=info.width, height=info.height)
    cold_total = decode_s + (time.perf_counter() - t0)
    print(f"parse->tracks: cold={parse_cold:.2f}s (tracks={[(t.n, len(t.contours)) for t in tracks]})", flush=True)
    print(f"level: cold={level_cold:.2f}s (n={lvl.n})", flush=True)
    print(f"strobe: cold={strobe_cold:.2f}s (out={out_path})", flush=True)
    print(f"pipeline cold total (decode once): {cold_total:.2f}s level_n={lvl2.n}", flush=True)

    # ---- warm: everything from the npz caches ----
    t0 = time.perf_counter()
    tracks = parse_tracks()
    parse_warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    level()
    level_warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    make_strobe(video, tracks, bench_dir, data_dir, width=info.width, height=info.height)
    strobe_warm = time.perf_counter() - t0
    warm_total = parse_warm + level_warm + strobe_warm
    print(f"parse->tracks: warm={parse_warm:.2f}s", flush=True)
    print(f"level: warm={level_warm:.2f}s", flush=True)
    print(f"strobe: warm={strobe_warm:.2f}s", flush=True)
    print(f"pipeline warm total: {warm_total:.2f}s", flush=True)
    npz = mask_cache_path(data_dir, key)
    print(f"npz: {npz} ({npz.stat().st_size / 1e6:.2f} MB)", flush=True)
    print(f"targets: cold < 90s -> {'OK' if cold_total < 90 else 'FAIL'}; "
          f"warm < 5s -> {'OK' if warm_total < 5 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
