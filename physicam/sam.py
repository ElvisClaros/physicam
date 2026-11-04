"""SAM 3.1 video/image segmentation, track building, stitching and dedup.

Uses the official ``meta_sam_parser`` for the mask codec (see AGENTS.md).
Raw SAM text is cached under ``data/cache/sam/`` keyed by video sha256 + prompt.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from physicam.meta_client import MetaClient

MAX_CONTOURS = 300

# ---------------------------------------------------------------------------
# Contour coordinate transform
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def raster_path_to_frame(svg_path: str, left: float, top: float,
                         frame_w: float, frame_h: float,
                         raster_w: int, raster_h: int) -> str:
    """Transform an SVG path from raster coords to frame coords.

    ``decode_mask_to_svg_path`` emits coordinates in raster pixel space
    (x in [0, raster_w], y in [0, raster_h]). The raster maps onto the frame
    rectangle [left, right] x [top, bottom], so
    ``fx = left + x * (right-left)/raster_w`` (and likewise for y).
    Numbers are rewritten in order (x, y, x, y, ...); path commands (M/L/Z)
    contain no digits and are preserved.
    """
    if not svg_path:
        return ""
    sx = frame_w / raster_w if raster_w else 1.0
    sy = frame_h / raster_h if raster_h else 1.0
    counter = [0]

    def repl(m: re.Match) -> str:
        idx = counter[0]
        counter[0] += 1
        v = float(m.group(0))
        out = left + v * sx if idx % 2 == 0 else top + v * sy
        return f"{out:.1f}"

    return _NUM_RE.sub(repl, svg_path)


def _largest_4component(mask: np.ndarray) -> np.ndarray:
    """Boolean mask of the largest 4-connected component of *mask*.

    Pure-Python BFS (rasters are small); no new dependencies. Ties keep
    the first component found in raster scan order.
    """
    fg = np.asarray(mask, dtype=bool)
    if fg.ndim != 2 or not np.any(fg):
        return np.zeros(fg.shape, dtype=bool) if fg.ndim == 2 else fg
    h, w = fg.shape
    visited = np.zeros((h, w), dtype=bool)
    best: list[tuple[int, int]] = []
    rows, cols = np.nonzero(fg)
    for r0, c0 in zip(rows.tolist(), cols.tolist()):
        if visited[r0, c0]:
            continue
        stack = [(r0, c0)]
        visited[r0, c0] = True
        comp: list[tuple[int, int]] = []
        while stack:
            i, j = stack.pop()
            comp.append((i, j))
            if i > 0 and fg[i - 1, j] and not visited[i - 1, j]:
                visited[i - 1, j] = True
                stack.append((i - 1, j))
            if i + 1 < h and fg[i + 1, j] and not visited[i + 1, j]:
                visited[i + 1, j] = True
                stack.append((i + 1, j))
            if j > 0 and fg[i, j - 1] and not visited[i, j - 1]:
                visited[i, j - 1] = True
                stack.append((i, j - 1))
            if j + 1 < w and fg[i, j + 1] and not visited[i, j + 1]:
                visited[i, j + 1] = True
                stack.append((i, j + 1))
        if len(comp) > len(best):
            best = comp
    out = np.zeros((h, w), dtype=bool)
    for i, j in best:
        out[i, j] = True
    return out


def mask_extents(raster: np.ndarray, bounds: tuple[float, float, float, float]) -> dict:
    """Oriented extents of a binary mask raster mapped to frame coordinates.

    *raster* is a 2-D 0/1 (or boolean) array smaller than the frame;
    *bounds* is ``(left, top, right, bottom)`` in frame pixels onto which
    the raster maps. Foreground pixel centres are
    ``x = left + (j + 0.5) * sx``, ``y = top + (i + 0.5) * sy`` with
    ``sx = (right-left)/raster_width``, ``sy = (bottom-top)/raster_height``.

    Principal axes come from the second central moments. Projecting all
    foreground centres onto the major/minor axes gives
    ``long_extent_px = (max - min) + pixel_size_along_axis`` (and likewise
    for the short axis), where ``pixel_size_along_axis`` is the projected
    width of one raster pixel, ``|vx|*sx + |vy|*sy``. This is the true
    side length for a rectangle, unlike the equivalent-ellipse major axis
    ``4*sqrt(lambda_max)`` which overstates a rectangle side by
    ``4/sqrt(12) ~= 1.155``. The robust variants are the exact extents
    (same formula) of the largest 4-connected component only, so stray
    disconnected pixels do not inflate the size without the bias that
    percentile trimming introduces (0.99x for a rectangle, ~0.96x for a
    disc). The old 0.5th/99.5th percentile numbers are kept only as
    ``trimmed_*`` diagnostics and are not used for calibration.
    """
    arr = np.asarray(raster)
    left, top, right, bottom = (float(v) for v in bounds)
    if arr.ndim != 2 or arr.size == 0 or not np.any(arr):
        bw, bh = max(right - left, 0.0), max(bottom - top, 0.0)
        return {
            "centroid": [(left + right) / 2.0, (top + bottom) / 2.0],
            "area_px": 0.0, "eq_diameter_px": 0.0,
            "ellipse_major_px": 0.0, "ellipse_minor_px": 0.0,
            "major_axis_px": 0.0, "minor_axis_px": 0.0,
            "orientation_deg": 0.0,
            "bbox_width_px": bw, "bbox_height_px": bh,
            "width_px": bw, "height_px": bh,
            "oriented_width_px": bw, "oriented_height_px": bh,
            "long_extent_px": 0.0, "short_extent_px": 0.0,
            "robust_long_extent_px": 0.0, "robust_short_extent_px": 0.0,
            "trimmed_long_extent_px": 0.0, "trimmed_short_extent_px": 0.0,
            "longest_px": 0.0, "shortest_px": 0.0,
        }
    rh, rw = arr.shape
    sx = (right - left) / rw if rw else 0.0
    sy = (bottom - top) / rh if rh else 0.0
    ii, jj = np.nonzero(arr)
    xs = left + (jj.astype(float) + 0.5) * sx
    ys = top + (ii.astype(float) + 0.5) * sy
    cx, cy = float(xs.mean()), float(ys.mean())
    dx, dy = xs - cx, ys - cy
    mu20 = float((dx * dx).mean())
    mu02 = float((dy * dy).mean())
    mu11 = float((dx * dy).mean())
    cov = np.array([[mu20, mu11], [mu11, mu02]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    major_vec = eigvecs[:, 0].astype(float)
    minor_vec = eigvecs[:, 1].astype(float)
    # Normalise the major-axis angle to (-90, 90] degrees.
    angle_deg = math.degrees(math.atan2(float(major_vec[1]), float(major_vec[0])))
    while angle_deg <= -90.0:
        angle_deg += 180.0
        major_vec = -major_vec
    while angle_deg > 90.0:
        angle_deg -= 180.0
        major_vec = -major_vec
    proj_major = dx * major_vec[0] + dy * major_vec[1]
    proj_minor = dx * minor_vec[0] + dy * minor_vec[1]

    def _pix(vec: np.ndarray) -> float:
        return abs(float(vec[0])) * sx + abs(float(vec[1])) * sy

    long_extent = float(proj_major.max() - proj_major.min()) + _pix(major_vec)
    short_extent = float(proj_minor.max() - proj_minor.min()) + _pix(minor_vec)
    lo_m, hi_m = np.percentile(proj_major, [0.5, 99.5])
    lo_s, hi_s = np.percentile(proj_minor, [0.5, 99.5])
    trimmed_long = float(hi_m - lo_m) + _pix(major_vec)
    trimmed_short = float(hi_s - lo_s) + _pix(minor_vec)
    # Robust extents: exact extents of the largest 4-connected component,
    # with principal axes recomputed from that component alone.
    comp = _largest_4component(np.asarray(arr) != 0)
    ci, cj = np.nonzero(comp)
    if ci.size == 0:
        robust_long, robust_short = long_extent, short_extent
        robust_angle = angle_deg
    elif ci.size == len(xs):
        robust_long, robust_short = long_extent, short_extent
        robust_angle = angle_deg
    else:
        cxs = left + (cj.astype(float) + 0.5) * sx
        cys = top + (ci.astype(float) + 0.5) * sy
        cdx, cdy = cxs - float(cxs.mean()), cys - float(cys.mean())
        ccov = np.array([[float((cdx * cdx).mean()), float((cdx * cdy).mean())],
                         [float((cdx * cdy).mean()), float((cdy * cdy).mean())]])
        _, cevecs = np.linalg.eigh(ccov)
        cmajor = cevecs[:, -1].astype(float)
        cminor = cevecs[:, 0].astype(float)
        robust_angle = math.degrees(math.atan2(float(cmajor[1]), float(cmajor[0])))
        while robust_angle <= -90.0:
            robust_angle += 180.0
            cmajor = -cmajor
        while robust_angle > 90.0:
            robust_angle -= 180.0
            cmajor = -cmajor
        pmaj = cdx * cmajor[0] + cdy * cmajor[1]
        pmin = cdx * cminor[0] + cdy * cminor[1]
        robust_long = float(pmaj.max() - pmaj.min()) + _pix(cmajor)
        robust_short = float(pmin.max() - pmin.min()) + _pix(cminor)
    area = float(len(xs)) * sx * sy
    ellipse_major = 4.0 * math.sqrt(max(float(eigvals[0]), 0.0))
    ellipse_minor = 4.0 * math.sqrt(max(float(eigvals[1]), 0.0))
    bw, bh = float(right - left), float(bottom - top)
    # Oriented width/height: assign long/short by the major axis angle.
    if abs(robust_angle) <= 45.0:
        oriented_w, oriented_h = robust_long, robust_short
        oriented_w_plain, oriented_h_plain = long_extent, short_extent
    else:
        oriented_w, oriented_h = robust_short, robust_long
        oriented_w_plain, oriented_h_plain = short_extent, long_extent
    return {
        "centroid": [cx, cy],
        "area_px": area,
        "eq_diameter_px": math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0,
        "ellipse_major_px": float(ellipse_major),
        "ellipse_minor_px": float(ellipse_minor),
        "major_axis_px": float(ellipse_major),
        "minor_axis_px": float(ellipse_minor),
        "orientation_deg": float(angle_deg),
        "bbox_width_px": bw, "bbox_height_px": bh,
        "width_px": bw, "height_px": bh,
        "oriented_width_px": float(oriented_w),
        "oriented_height_px": float(oriented_h),
        "oriented_width_plain_px": float(oriented_w_plain),
        "oriented_height_plain_px": float(oriented_h_plain),
        "long_extent_px": float(long_extent),
        "short_extent_px": float(short_extent),
        "robust_long_extent_px": float(robust_long),
        "robust_short_extent_px": float(robust_short),
        "trimmed_long_extent_px": float(trimmed_long),
        "trimmed_short_extent_px": float(trimmed_short),
        "longest_px": float(long_extent),
        "shortest_px": float(short_extent),
    }


# Backwards-friendly alias.
oriented_extents = mask_extents


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """One tracked object's trajectory in frame pixel coordinates."""

    id: str
    label: str
    frames: np.ndarray  # int frame indices
    t: np.ndarray  # seconds
    x: np.ndarray  # mask-weighted centroid, frame px
    y: np.ndarray
    area: np.ndarray  # px^2
    bbox: np.ndarray  # (n, 4) x1 y1 x2 y2 frame px
    conf: np.ndarray
    eq_diameter: np.ndarray  # px, sqrt(4*area/pi)
    contours: dict[int, str] = field(default_factory=dict)
    fps: float = 30.0
    aliases: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.frames)

    def to_json(self) -> dict:
        """JSON-serializable dict following the result contract in AGENTS.md."""
        return {
            "id": self.id, "label": self.label,
            "frames": [int(v) for v in self.frames],
            "t": [round(float(v), 6) for v in self.t],
            "x": [round(float(v), 3) for v in self.x],
            "y": [round(float(v), 3) for v in self.y],
            "area": [round(float(v), 2) for v in self.area],
            "bbox": [[round(float(c), 2) for c in row] for row in self.bbox],
            "conf": [round(float(v), 4) for v in self.conf],
            "eq_diameter": [round(float(v), 3) for v in self.eq_diameter],
            "fps": self.fps,
            "aliases": list(self.aliases),
            "contours": {str(k): v for k, v in sorted(self.contours.items())},
        }

    @classmethod
    def from_json(cls, d: dict) -> "Track":
        return cls(
            id=d["id"], label=d.get("label", d["id"]),
            frames=np.asarray(d["frames"], dtype=int),
            t=np.asarray(d["t"], dtype=float),
            x=np.asarray(d["x"], dtype=float), y=np.asarray(d["y"], dtype=float),
            area=np.asarray(d["area"], dtype=float),
            bbox=np.asarray(d["bbox"], dtype=float).reshape(-1, 4),
            conf=np.asarray(d.get("conf", [1.0] * len(d["frames"])), dtype=float),
            eq_diameter=np.asarray(d["eq_diameter"], dtype=float),
            contours={int(k): v for k, v in (d.get("contours") or {}).items()},
            fps=float(d.get("fps", 30.0)),
            aliases=[str(a) for a in (d.get("aliases") or [])],
        )

    def summary(self) -> dict:
        """Compact dict for the LLM (no raw masks)."""
        if self.n == 0:
            return {"id": self.id, "label": self.label, "n_frames": 0}
        gaps = []
        if self.n > 1:
            d = np.diff(self.frames)
            for i, g in enumerate(d):
                if g > 1:
                    gaps.append({"after_frame": int(self.frames[i]), "missing": int(g - 1)})
        return {
            "id": self.id, "label": self.label,
            "n_frames": int(self.n),
            "frame_range": [int(self.frames[0]), int(self.frames[-1])],
            "t_range": [round(float(self.t[0]), 4), round(float(self.t[-1]), 4)],
            "median_eq_diameter_px": round(float(np.median(self.eq_diameter)), 2),
            "median_area_px": round(float(np.median(self.area)), 1),
            "x_extent_px": [round(float(self.x.min()), 1), round(float(self.x.max()), 1)],
            "y_extent_px": [round(float(self.y.min()), 1), round(float(self.y.max()), 1)],
            "mean_conf": round(float(np.mean(self.conf)), 3),
            "aliases": list(self.aliases),
            "gaps": gaps[:20],
        }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _slug(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_") or "obj"


def mask_cache_key_for_text(text: str) -> str:
    """Content-addressed mask-cache key for a raw SAM text."""
    return f"text_{hashlib.sha256(text.encode()).hexdigest()[:24]}"


def parse_sam_text(text: str, fps: float, label: str = "", *, cache_key: str | None = None,
                   data_dir: str = "data", max_workers: int | None = None,
                   with_contours: bool = True) -> list[Track]:
    """Pure parser: SAM output text + fps -> tracks (unstiched, raw SAM ids).

    Centroids/areas come from the parallel mask cache
    (:mod:`physicam.maskcache`); contours are computed only for the
    <= MAX_CONTOURS frames actually kept (every k-th per track).
    """
    from physicam.maskcache import decoded_masks, parse_records, plan_contour_indices

    if not text or not text.strip():
        return []
    # Plan contour frames before decoding so they are traced from the
    # rasters in the same worker pass (no second decode, cached on disk).
    kept_idx: list[int] = []
    if with_contours:
        try:
            _boxes, _masks = parse_records(text)
            kept_idx = plan_contour_indices(_boxes, _masks, MAX_CONTOURS)
        except Exception:
            kept_idx = []
    ms = decoded_masks(text, cache_key or mask_cache_key_for_text(text),
                       data_dir=data_dir, max_workers=max_workers,
                       contour_indices=kept_idx or None)

    tracks: list[Track] = []
    by_oid: dict[str, list] = {}
    for rec in ms.records:
        by_oid.setdefault(rec.object_id, []).append(rec)
    per_track_kept: dict[str, list] = {}
    for oid in sorted(by_oid, key=lambda s: (len(s), s)):
        recs = sorted(by_oid[oid], key=lambda r: r.frame)
        stride = max(1, math.ceil(len(recs) / MAX_CONTOURS))
        per_track_kept[oid] = recs[::stride]
    svg_by_index: dict[int, str] = {}
    if with_contours and kept_idx:
        raw_svg = ms.contours
        if not raw_svg:
            try:
                raw_svg = ms.ensure_contours(kept_idx, max_workers=max_workers)
            except Exception:
                raw_svg = {}
        for idx, svg in raw_svg.items():
            if not svg:
                continue
            r = ms.records[idx]
            x1, y1, x2, y2 = (float(v) for v in r.box)
            svg_by_index[idx] = raster_path_to_frame(
                svg, x1, y1, x2 - x1, y2 - y1, r.raster_width, r.raster_height)

    for oid in sorted(by_oid, key=lambda s: (len(s), s)):
        recs = sorted(by_oid[oid], key=lambda r: r.frame)
        if not recs:
            continue
        frames = np.array([r.frame for r in recs], dtype=int)
        area_arr = np.array([r.area for r in recs], dtype=float)
        contours = {int(r.frame): svg_by_index[r.index]
                    for r in per_track_kept.get(oid, []) if r.index in svg_by_index}
        tracks.append(Track(
            id=f"t{len(tracks)}", label=label or oid,
            frames=frames, t=frames.astype(float) / fps,
            x=np.array([r.centroid_x for r in recs]), y=np.array([r.centroid_y for r in recs]),
            area=area_arr,
            bbox=np.array([[float(c) for c in r.box] for r in recs], dtype=float),
            conf=np.array([r.confidence for r in recs], dtype=float),
            eq_diameter=np.sqrt(4.0 * np.maximum(area_arr, 0.0) / math.pi),
            contours=contours, fps=float(fps),
        ))
        # Keep raw SAM id available for stitching determinism via label fallback.
        tracks[-1]._sam_id = oid  # type: ignore[attr-defined]
    return tracks


def box_only_tracks(text: str, fps: float, label: str = "") -> list[Track]:
    """Lightweight tracks from SAM box records only (no mask decoding).

    Centroids are box centres and areas are bbox areas; good enough for
    cross-prompt duplicate detection before any raster decode is paid.
    """
    from physicam.maskcache import box_records

    boxes = box_records(text)
    by_oid: dict[str, list] = {}
    for (oid, frame), (x1, y1, x2, y2, conf) in boxes.items():
        by_oid.setdefault(str(oid), []).append((int(frame), x1, y1, x2, y2, conf))
    tracks: list[Track] = []
    for oid in sorted(by_oid, key=lambda s: (len(s), s)):
        rows = sorted(by_oid[oid])
        frames = np.array([r[0] for r in rows], dtype=int)
        area_arr = np.array([max(0.0, (r[3] - r[1]) * (r[4] - r[2])) for r in rows], dtype=float)
        tracks.append(Track(
            id=f"t{len(tracks)}", label=label or oid,
            frames=frames, t=frames.astype(float) / fps,
            x=np.array([(r[1] + r[3]) / 2.0 for r in rows]),
            y=np.array([(r[2] + r[4]) / 2.0 for r in rows]),
            area=area_arr,
            bbox=np.array([[r[1], r[2], r[3], r[4]] for r in rows], dtype=float),
            conf=np.array([r[5] for r in rows], dtype=float),
            eq_diameter=np.sqrt(4.0 * np.maximum(area_arr, 0.0) / math.pi),
            contours={}, fps=float(fps),
        ))
        tracks[-1]._sam_id = oid  # type: ignore[attr-defined]
    return tracks


def filter_duplicate_prompts(prompt_texts: dict[str, str], fps: float) -> tuple[list[str], dict[str, list[str]]]:
    """Box-level cross-prompt dedup before any raster decoding.

    Returns (kept_prompts, aliases) where ``aliases`` maps a kept prompt to
    the dropped duplicate prompts. A prompt is dropped only when every
    box track it produced merged into another prompt's track.
    """
    flat: list[Track] = []
    for prompt, text in prompt_texts.items():
        if not text or not text.strip():
            continue
        for tr in box_only_tracks(text, fps, label=prompt):
            taken = {t.id for t in flat}
            while tr.id in taken:
                tr.id = f"t{int(tr.id[1:]) + 1}" if tr.id[1:].isdigit() else f"t{len(taken)}"
            flat.append(tr)
    if not flat:
        return [], {}
    deduped = deduplicate_tracks(flat)
    kept_labels = {t.label for t in deduped}
    kept = [p for p in prompt_texts if p in kept_labels]
    aliases: dict[str, list[str]] = {}
    for tr in deduped:
        for alias in tr.aliases or []:
            if alias in prompt_texts and alias not in kept:
                aliases.setdefault(tr.label, []).append(alias)
    # Prompts with no box tracks at all are kept (decoding decides).
    for p, text in prompt_texts.items():
        if p not in kept and not any(t.label == p for t in flat):
            kept.append(p)
    return kept, aliases


# ---------------------------------------------------------------------------
# Stitching / dedup
# ---------------------------------------------------------------------------

def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _drop_duplicates(tracks: list[Track]) -> list[Track]:
    """Drop near-duplicates: same frames, high mean box IoU; keep higher conf."""
    keep = [True] * len(tracks)
    for i in range(len(tracks)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(tracks)):
            if not keep[j]:
                continue
            a, b = tracks[i], tracks[j]
            fa = {int(f): k for k, f in enumerate(a.frames)}
            fb = {int(f): k for k, f in enumerate(b.frames)}
            common = sorted(set(fa) & set(fb))
            if not common:
                continue
            overlap = len(common) / max(1, min(a.n, b.n))
            ious = [_box_iou(a.bbox[fa[f]], b.bbox[fb[f]]) for f in common]
            if overlap > 0.5 and float(np.mean(ious)) > 0.5:
                drop = j if float(np.mean(a.conf)) >= float(np.mean(b.conf)) else i
                keep[drop] = False
                if drop == i:
                    break
    return [t for t, k in zip(tracks, keep) if k]


def _median_conf(track: Track) -> float:
    return float(np.median(track.conf)) if track.n and len(track.conf) else 0.0


def _same_object(a: Track, b: Track) -> bool:
    """True when *a* and *b* are the same physical object (spec §1).

    Requires overlap on >= 60% of the shorter track, then either median
    box IoU >= 0.5 or median centroid distance < 0.5 x median eq. diameter.
    """
    if a.n == 0 or b.n == 0:
        return False
    fa = {int(f): k for k, f in enumerate(a.frames)}
    fb = {int(f): k for k, f in enumerate(b.frames)}
    common = sorted(set(fa) & set(fb))
    if not common or len(common) / max(1, min(a.n, b.n)) < 0.6:
        return False
    ious = [_box_iou(a.bbox[fa[f]], b.bbox[fb[f]]) for f in common]
    if float(np.median(ious)) >= 0.5:
        return True
    dists = [math.hypot(float(a.x[fa[f]] - b.x[fb[f]]), float(a.y[fa[f]] - b.y[fb[f]])) for f in common]
    diams = [float(a.eq_diameter[fa[f]]) for f in common] + [float(b.eq_diameter[fb[f]]) for f in common]
    med_diam = float(np.median(diams)) if diams else 0.0
    return med_diam > 0 and float(np.median(dists)) < 0.5 * med_diam


def _merge_cross_prompt(kept: Track, dropped: Track) -> Track:
    """Keep *kept*, record *dropped*'s label as an alias, fill consistent gaps."""
    aliases: list[str] = []
    for name in list(kept.aliases) + [dropped.label] + list(dropped.aliases):
        if name and name != kept.label and name not in aliases:
            aliases.append(str(name))
    kept_frames = {int(f) for f in kept.frames}
    diam = float(np.median(kept.eq_diameter)) if kept.n else 0.0
    if diam <= 0 and dropped.n:
        diam = float(np.median(dropped.eq_diameter))
    rows: list[tuple] = []
    for k in range(kept.n):
        rows.append((int(kept.frames[k]), float(kept.t[k]), float(kept.x[k]), float(kept.y[k]),
                     float(kept.area[k]), kept.bbox[k], float(kept.conf[k]), float(kept.eq_diameter[k])))
    contours = dict(kept.contours)
    for k in range(dropped.n):
        frame = int(dropped.frames[k])
        if frame in kept_frames:
            continue
        if kept.n >= 2:
            ix = float(np.interp(float(dropped.t[k]), kept.t, kept.x))
            iy = float(np.interp(float(dropped.t[k]), kept.t, kept.y))
        elif kept.n == 1:
            ix, iy = float(kept.x[0]), float(kept.y[0])
        else:
            continue
        if diam > 0 and math.hypot(float(dropped.x[k]) - ix, float(dropped.y[k]) - iy) >= 1.0 * diam:
            continue
        rows.append((frame, float(dropped.t[k]), float(dropped.x[k]), float(dropped.y[k]),
                     float(dropped.area[k]), dropped.bbox[k], float(dropped.conf[k]),
                     float(dropped.eq_diameter[k])))
        if frame in dropped.contours:
            contours[frame] = dropped.contours[frame]
    rows.sort(key=lambda r: r[0])
    return Track(
        id=kept.id, label=kept.label,
        frames=np.array([r[0] for r in rows], dtype=int),
        t=np.array([r[1] for r in rows], dtype=float),
        x=np.array([r[2] for r in rows], dtype=float),
        y=np.array([r[3] for r in rows], dtype=float),
        area=np.array([r[4] for r in rows], dtype=float),
        bbox=np.array([r[5] for r in rows], dtype=float).reshape(-1, 4) if rows else np.zeros((0, 4)),
        conf=np.array([r[6] for r in rows], dtype=float),
        eq_diameter=np.array([r[7] for r in rows], dtype=float),
        contours=contours, fps=kept.fps, aliases=aliases,
    )


def deduplicate_tracks(tracks: list[Track]) -> list[Track]:
    """Deduplicate tracks across prompts (same physical object, many labels).

    Two tracks are the same object when, over overlapping frames (>= 60%
    of the shorter track), median box IoU >= 0.5 or median centroid
    distance < 0.5 x median equivalent diameter. The track with more
    frames is kept (tie-break: higher median confidence); dropped labels
    are kept in ``aliases`` and consistent dropped-only frames fill gaps.
    """
    work = list(tracks)
    changed = True
    while changed:
        changed = False
        for i in range(len(work)):
            if changed:
                break
            for j in range(i + 1, len(work)):
                a, b = work[i], work[j]
                if not _same_object(a, b):
                    continue
                if (b.n, _median_conf(b)) > (a.n, _median_conf(a)):
                    kept, dropped, ki = b, a, j
                else:
                    kept, dropped, ki = a, b, i
                merged = _merge_cross_prompt(kept, dropped)
                work = [t for k, t in enumerate(work) if k not in (i, j)]
                work.insert(min(ki, len(work)), merged)
                changed = True
                break
    return work


def _predict(track: Track, t_query: float) -> tuple[float, float]:
    """Extrapolate (x, y) at *t_query* with constant-acceleration fit on last points.

    Fits in shifted time (u = t - t_last) over up to ~0.2 s of trailing points;
    fitting quadratics in absolute time is ill-conditioned and extrapolates
    wildly, which previously broke stitching on real 240 fps footage.
    """
    window = max(8, int(round(0.2 * track.fps))) if track.fps else 8
    n = min(window, track.n)
    tt, xx, yy = track.t[-n:], track.x[-n:], track.y[-n:]
    u = tt - tt[-1]
    dt = float(t_query - tt[-1])
    if n >= 3 and tt[-1] > tt[0]:
        cx = np.polyfit(u, xx, 2)
        cy = np.polyfit(u, yy, 2)
        return float(np.polyval(cx, dt)), float(np.polyval(cy, dt))
    if n >= 2 and tt[-1] > tt[0]:
        vx = (xx[-1] - xx[0]) / (tt[-1] - tt[0])
        vy = (yy[-1] - yy[0]) / (tt[-1] - tt[0])
        return float(xx[-1] + vx * dt), float(yy[-1] + vy * dt)
    return float(xx[-1]), float(yy[-1])


def _merge_pair(a: Track, b: Track) -> Track:
    frames = np.concatenate([a.frames, b.frames])
    order = np.argsort(frames)
    contours = dict(a.contours)
    contours.update(b.contours)
    if len(contours) > MAX_CONTOURS:
        keys = sorted(contours)
        stride = math.ceil(len(keys) / MAX_CONTOURS)
        contours = {k: contours[k] for k in keys[::stride]}
    return Track(
        id=a.id, label=a.label, frames=frames[order],
        t=np.concatenate([a.t, b.t])[order],
        x=np.concatenate([a.x, b.x])[order], y=np.concatenate([a.y, b.y])[order],
        area=np.concatenate([a.area, b.area])[order],
        bbox=np.concatenate([a.bbox, b.bbox])[order],
        conf=np.concatenate([a.conf, b.conf])[order],
        eq_diameter=np.concatenate([a.eq_diameter, b.eq_diameter])[order],
        contours=contours, fps=a.fps,
        aliases=list(dict.fromkeys(list(a.aliases) + list(b.aliases))),
    )


def stitch_tracks(tracks: list[Track], fps: float) -> list[Track]:
    """Deduplicate and stitch fragmented tracks. Deterministic and simple.

    Heuristics (documented, deliberately conservative):
    1. Duplicates: two tracks sharing >50% of the shorter track's frames with
       mean box IoU > 0.5 are the same detection twice; keep higher mean conf.
    2. Merge: non-overlapping A (earlier) and B (later) with time gap <= 0.5 s
       merge when the constant-acceleration extrapolation of A's trailing
       ~0.2 s lands within max(50 px, 2x median eq. diameter) of B's first
       point, OR the raw endpoint distance is within max(30 px, 1x diameter)
       (slow exit/re-entry near the same border region).
    3. Border exit/re-entry: A ends and B starts both within
       max(30 px, 1.5x diameter) of the same image border (top/left are y=0 /
       x=0; right/bottom inferred from the max bbox extent over all tracks)
       and the along-border coordinate extrapolates consistently. This covers
       a ball leaving the top of the frame and dropping back in, where the
       perpendicular (gravity) coordinate cannot be extrapolated reliably.
    Merges repeat greedily by smallest normalised distance until none qualify.
    """
    tracks = _drop_duplicates(list(tracks))
    tracks.sort(key=lambda tr: (float(tr.t[0]) if tr.n else 0.0, tr.id))
    frame_right = max((float(np.max(tr.bbox[:, 2])) for tr in tracks if tr.n), default=0.0)
    frame_bottom = max((float(np.max(tr.bbox[:, 3])) for tr in tracks if tr.n), default=0.0)

    def border_score(a: Track, b: Track, diam: float) -> float | None:
        tol = max(30.0, 1.5 * diam)
        ax, ay = float(a.x[-1]), float(a.y[-1])
        bx, by = float(b.x[0]), float(b.y[0])
        px, py = _predict(a, float(b.t[0]))
        # (perpendicular A, perpendicular B, along-error)
        candidates = [
            (ay, by, abs(px - bx)),          # top border (y ~ 0)
            (ax, bx, abs(py - by)),          # left border (x ~ 0)
        ]
        if frame_right > 0:
            candidates.append((frame_right - ax, frame_right - bx, abs(py - by)))
        if frame_bottom > 0:
            candidates.append((frame_bottom - ay, frame_bottom - by, abs(px - bx)))
        for pa, pb, along_err in candidates:
            if pa <= tol and pb <= tol and along_err <= max(60.0, 2.5 * diam):
                return along_err / diam
        return None

    changed = True
    while changed:
        changed = False
        best: tuple[float, int, int] | None = None
        for i, a in enumerate(tracks):
            for j, b in enumerate(tracks):
                if i == j or a.n == 0 or b.n == 0:
                    continue
                if float(b.t[0]) <= float(a.t[-1]):
                    continue  # require A strictly before B (no time overlap)
                gap = float(b.t[0] - a.t[-1])
                if gap > 0.5 or gap <= 0:
                    continue
                diam = max(float(np.median(a.eq_diameter)), float(np.median(b.eq_diameter)), 1.0)
                px, py = _predict(a, float(b.t[0]))
                dist_pred = math.hypot(px - float(b.x[0]), py - float(b.y[0]))
                dist_raw = math.hypot(float(a.x[-1]) - float(b.x[0]), float(a.y[-1]) - float(b.y[0]))
                ok = dist_pred <= max(50.0, 2.0 * diam) or dist_raw <= max(30.0, 1.0 * diam)
                score = min(dist_pred, dist_raw) / diam
                if not ok:
                    bs = border_score(a, b, diam)
                    if bs is None:
                        continue
                    ok, score = True, bs
                if ok and (best is None or score < best[0]):
                    best = (score, i, j)
        if best is not None:
            _, i, j = best
            merged = _merge_pair(tracks[i], tracks[j])
            tracks = [t for k, t in enumerate(tracks) if k not in (i, j)] + [merged]
            tracks.sort(key=lambda tr: (float(tr.t[0]) if tr.n else 0.0, tr.id))
            changed = True
    for idx, tr in enumerate(tracks):
        tr.id = f"t{idx}"
    return tracks


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def _sam_cache_path(settings_data_dir: str, video_sha: str, prompt: str) -> Path:
    p = Path(settings_data_dir) / "cache" / "sam" / f"{video_sha}_{_slug(prompt)}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def sam_text_for_prompt(client: "MetaClient", video_path: str | Path, prompt: str,
                        *, cache: bool = True) -> str:
    """SAM text for *prompt*: disk cache first, live API on a miss.

    Split out of :func:`segment_video` so callers can run box-level
    dedup on candidate prompts before paying for any mask decoding.
    """
    from physicam.video import sha256_file

    video_path = Path(video_path)
    sha = sha256_file(video_path)
    cache_path = _sam_cache_path(client.settings.data_dir, sha, prompt)
    if cache and cache_path.exists():
        return cache_path.read_text()
    file_id = client.upload_file(video_path)
    body = {
        "model": client.settings.sam, "stream": False,
        "metadata": {"include_confidence": "true"},
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_video", "file_id": file_id}]}],
    }
    resp = client.responses(body)
    text = client.output_text(resp)
    if cache:
        cache_path.write_text(text)
    return text


def mask_cache_key(video_sha: str, prompt: str) -> str:
    """Mask-cache key shared by parse/level/strobe for one video+prompt."""
    return f"{video_sha}_{_slug(prompt)}"


def segment_video(client: "MetaClient", video_path: str | Path, prompt: str,
                  fps: float, *, cache: bool = True) -> list[Track]:
    """Upload, call SAM 3.1, parse, stitch. Raw text cached for free re-runs."""
    from physicam.video import sha256_file

    video_path = Path(video_path)
    sha = sha256_file(video_path)
    text = sam_text_for_prompt(client, video_path, prompt, cache=cache)
    raw = parse_sam_text(text, fps, label=prompt,
                         cache_key=mask_cache_key(sha, prompt),
                         data_dir=str(client.settings.data_dir))
    return stitch_tracks(raw, fps)


def segment_image(client: "MetaClient", pil_image: Image.Image, prompt: str) -> list[dict]:
    """Segment one image; return boxes, masks info, axis lengths in frame px."""
    import base64
    import io

    from meta_sam_parser import CompletedOutcome, image_segmentation_format

    from physicam.maskcache import decode_mask_payloads

    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    body = {
        "model": client.settings.sam, "stream": False,
        "metadata": {"include_confidence": "true"},
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}]}],
    }
    resp = client.responses(body)
    text = client.output_text(resp)
    if not text or not text.strip():
        return []
    parser = image_segmentation_format().create_parser()
    parser.push(text if text.endswith("\n") else text + "\n")
    result = parser.finish(CompletedOutcome()).result

    boxes: dict[str, tuple[float, float, float, float, float]] = {}
    masks: dict[str, object] = {}
    for rec in result.records:
        if rec.kind == "box":
            boxes[str(rec.object_id)] = (float(rec.left), float(rec.top), float(rec.right),
                                         float(rec.bottom), float(rec.confidence or 0.0))
        elif rec.kind == "mask":
            masks[str(rec.object_id)] = rec

    out: list[dict] = []
    for oid in sorted(set(boxes) | set(masks), key=lambda s: (len(s), s)):
        rec = masks.get(oid)
        box = boxes.get(oid)
        if box is not None:
            x1, y1, x2, y2, conf = box
        else:
            b = rec.bounds  # type: ignore[union-attr]
            x1, y1, x2, y2 = float(b.left), float(b.top), float(b.right), float(b.bottom)
            conf = float(rec.confidence or 0.0)  # type: ignore[union-attr]
        bw, bh = float(x2 - x1), float(y2 - y1)
        item: dict = {"object_id": oid, "bbox": [x1, y1, x2, y2], "conf": conf,
                      "width_px": bw, "height_px": bh,
                      "bbox_width_px": bw, "bbox_height_px": bh, "area_px": 0.0,
                      "ellipse_major_px": max(bw, bh), "ellipse_minor_px": min(bw, bh),
                      "major_axis_px": max(bw, bh), "minor_axis_px": min(bw, bh),
                      "orientation_deg": 0.0,
                      "long_extent_px": max(bw, bh), "short_extent_px": min(bw, bh),
                      "robust_long_extent_px": max(bw, bh), "robust_short_extent_px": min(bw, bh),
                      "trimmed_long_extent_px": max(bw, bh), "trimmed_short_extent_px": min(bw, bh),
                      "oriented_width_px": bw, "oriented_height_px": bh,
                      "longest_px": max(bw, bh), "shortest_px": min(bw, bh),
                      "eq_diameter_px": math.sqrt(4.0 * bw * bh / math.pi) if bw > 0 and bh > 0 else 0.0}
        if rec is not None:
            m = rec.mask  # type: ignore[union-attr]
            # Use mask bounds when available: the raster maps onto them.
            try:
                b = rec.bounds  # type: ignore[union-attr]
                bx1, by1, bx2, by2 = float(b.left), float(b.top), float(b.right), float(b.bottom)
            except Exception:
                bx1, by1, bx2, by2 = x1, y1, x2, y2
            # Decode via the shared maskcache worker (serial for one image).
            [(raster, _stats)] = decode_mask_payloads([(m, (bx1, by1, bx2, by2), (x1, y1, x2, y2))])
            if float(raster.sum()) > 0:
                item.update(mask_extents(raster, (bx1, by1, bx2, by2)))
                # Preserve the SAM box as the reported bbox.
                item["bbox"] = [x1, y1, x2, y2]
                item["bbox_width_px"] = bw
                item["bbox_height_px"] = bh
                item["width_px"] = bw
                item["height_px"] = bh
        out.append(item)
    return out
