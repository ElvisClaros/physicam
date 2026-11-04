"""Stroboscopic composite from SAM masks re-parsed from the cached SAM text.

No API calls: mask rasters come from ``data/cache/sam/`` and frames from the
local video via ffmpeg.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def cached_sam_text(data_dir: str, video_sha: str, prompt: str) -> str | None:
    """Return cached SAM text for *prompt* or None."""
    from physicam.sam import _sam_cache_path

    path = _sam_cache_path(data_dir, video_sha, prompt)
    return path.read_text() if path.exists() else None


def parse_mask_rasters(text: str, cache_key: str | None = None, data_dir: str = "data") -> dict[tuple[str, int], tuple[np.ndarray, tuple[float, float, float, float]]]:
    """Parse SAM text into {(object_id, frame): (raster 0/1, bounds)}.

    Rasters are lazily unpacked from the parallel mask cache; callers that
    only need a few frames should prefer :func:`physicam.maskcache.decoded_masks`
    directly (see :func:`make_strobe`).
    """
    from physicam.maskcache import decoded_masks
    from physicam.sam import mask_cache_key_for_text

    if not text or not text.strip():
        return {}
    ms = decoded_masks(text, cache_key or mask_cache_key_for_text(text), data_dir=data_dir)
    out: dict[tuple[str, int], tuple[np.ndarray, tuple[float, float, float, float]]] = {}
    for rec in ms.records:
        if not rec.has_mask:
            continue
        out[(rec.object_id, int(rec.frame))] = (
            ms.raster(rec.index), tuple(float(v) for v in rec.bounds))
    return out


def _maskset_for_track(data_dir: str, sha: str, label: str, text: str):
    """MaskSet for one track label, sharing the parse/level cache key."""
    from physicam.maskcache import decoded_masks
    from physicam.sam import mask_cache_key

    return decoded_masks(text, mask_cache_key(sha, str(label)), data_dir=data_dir)


def _choose_step(n: int, target: int = 18) -> int:
    if n <= 0:
        return 1
    return max(1, round(n / target))


def _extract_frames_parallel(video_path: Path, times: list) -> list:
    """Extract frames at *times* with parallel ffmpeg calls (ordered)."""
    from concurrent.futures import ThreadPoolExecutor

    from physicam.video import extract_frame

    def _one(t):
        try:
            return extract_frame(video_path, float(t))
        except Exception:
            return None

    if len(times) <= 1:
        return [_one(t) for t in times]
    with ThreadPoolExecutor(max_workers=min(6, len(times))) as pool:
        return list(pool.map(_one, times))


def _median_plate(video_path: Path, duration: float, width: int, height: int, n_samples: int = 15) -> Image.Image:
    times = [duration * (i + 0.5) / n_samples for i in range(n_samples)] if duration > 0 else [0.0]
    frames = []
    for img in _extract_frames_parallel(video_path, times):
        if img is not None:
            frames.append(np.asarray(img.resize((width, height)), dtype=np.float32))
    if not frames:
        return Image.new("RGB", (width, height), (16, 16, 16))
    return Image.fromarray(np.median(np.stack(frames), axis=0).astype(np.uint8))


def make_strobe(video_path, tracks, job_dir, data_dir: str, fits=None, width: int = 0, height: int = 0) -> Path | None:
    """Build ``job_dir/strobe.jpg`` from *tracks*; return its path or None.

    *tracks* are Track-like objects (or result-JSON dicts converted by caller).
    Masks are matched to a track by nearest centroid at each sampled frame.
    """
    from physicam.video import probe, sha256_file

    video_path = Path(video_path)
    job_dir = Path(job_dir)
    if not tracks:
        return None
    try:
        info = probe(video_path)
    except Exception:
        return None
    width = int(width or info.width)
    height = int(height or info.height)
    if width <= 0 or height <= 0:
        return None
    sha = sha256_file(video_path)
    plate = _median_plate(video_path, float(info.duration_s), width, height)
    canvas = plate.convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")

    # Fitted curves (thin dashed) under the copies.
    for fit in (fits or []):
        curve = (fit.get("curve") if isinstance(fit, dict) else getattr(fit, "curve", None)) or {}
        ct, cx, cy = curve.get("t") or [], curve.get("x") or [], curve.get("y") or []
        if ct and cx and cy:
            pts = [(float(cx[i]), float(cy[i])) for i in range(min(len(ct), len(cx), len(cy)))]
            for i in range(0, len(pts) - 1, 2):
                draw.line([pts[i], pts[i + 1]], fill=(255, 255, 255, 140), width=1)

    any_paste = False
    for track in tracks:
        label = track.label if not isinstance(track, dict) else track.get("label", "")
        text = cached_sam_text(data_dir, sha, str(label))
        if text is None:
            continue
        # Only the sampled frames' rasters are ever unpacked (lazy).
        try:
            maskset = _maskset_for_track(data_dir, sha, str(label), text)
        except Exception:
            continue
        if not any(r.has_mask for r in maskset.records):
            continue
        by_frame = maskset.by_frame()
        frames_arr = np.asarray(track.frames) if not isinstance(track, dict) else np.asarray(track.get("frames", []))
        xs = np.asarray(track.x, dtype=float) if not isinstance(track, dict) else np.asarray(track.get("x", []), dtype=float)
        ys = np.asarray(track.y, dtype=float) if not isinstance(track, dict) else np.asarray(track.get("y", []), dtype=float)
        ts = np.asarray(track.t, dtype=float) if not isinstance(track, dict) else np.asarray(track.get("t", []), dtype=float)
        if len(frames_arr) == 0:
            continue
        step = _choose_step(len(frames_arr))
        # Increase step if successive centroids overlap heavily (<0.6 diameter).
        diam = float(np.median(track.eq_diameter)) if not isinstance(track, dict) and len(track.eq_diameter) else 40.0
        while step < len(frames_arr):
            idxs = list(range(0, len(frames_arr), step))
            if len(idxs) <= 25 and all(
                    math.hypot(xs[idxs[i + 1]] - xs[idxs[i]], ys[idxs[i + 1]] - ys[idxs[i]]) >= 0.6 * max(diam, 1.0)
                    for i in range(len(idxs) - 1)):
                break
            if len(idxs) <= 12:
                break
            step += max(1, step // 2)
        # Collect this track's sampled masks, then batch-extract frames.
        sampled = []
        for idx in range(0, len(frames_arr), step):
            frame_no = int(frames_arr[idx])
            # Pick the record whose centroid is nearest the track centroid,
            # then unpack only that raster.
            best = None
            for rec in by_frame.get(frame_no, []):
                if not rec.has_mask:
                    continue
                d = math.hypot(rec.centroid_x - xs[idx], rec.centroid_y - ys[idx])
                if best is None or d < best[0]:
                    best = (d, rec)
            if best is None:
                continue
            _, best_rec = best
            raster = maskset.raster(best_rec.index)
            bounds = tuple(float(v) for v in best_rec.bounds)
            t_extract = float(ts[idx]) if idx < len(ts) else frame_no / max(info.fps, 1)
            sampled.append((idx, raster, bounds, t_extract))
        frame_imgs = _extract_frames_parallel(video_path, [s[3] for s in sampled])
        for (idx, raster, (x1, y1, x2, y2), _t_extract), frame_img in zip(sampled, frame_imgs):
            if frame_img is None:
                continue
            bw, bh = max(1, int(round(x2 - x1))), max(1, int(round(y2 - y1)))
            if frame_img.size != (width, height):
                frame_img = frame_img.resize((width, height))
            mask_img = Image.fromarray((raster * 255).astype(np.uint8)).resize((bw, bh), Image.NEAREST)
            crop = frame_img.crop((int(x1), int(y1), int(x1) + bw, int(y1) + bh))
            # Drop shadow: dark shifted silhouette, then the object, then outline.
            shadow = Image.new("RGBA", (bw + 4, bh + 4), (0, 0, 0, 0))
            sil = mask_img.point(lambda v: 110 if v else 0)
            shadow.paste(Image.new("RGBA", (bw, bh), (0, 0, 0, 110)), (3, 3), sil)
            canvas.paste(shadow, (int(x1), int(y1)), shadow)
            canvas.paste(crop, (int(x1), int(y1)), mask_img)
            outline = mask_img.filter(ImageFilter.FIND_EDGES)
            canvas.paste(Image.new("RGB", (bw, bh), (255, 255, 255)), (int(x1), int(y1)), outline.point(lambda v: 90 if v else 0))
            draw = ImageDraw.Draw(canvas, "RGBA")
            if idx < len(ts):
                draw.text((float(xs[idx]) + 6, float(ys[idx]) - 8), f"{float(ts[idx]):.2f} s",
                          fill=(255, 255, 255, 200))
            any_paste = True
    if not any_paste:
        return None
    out = job_dir / "strobe.jpg"
    job_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(out, format="JPEG", quality=88)
    return out
