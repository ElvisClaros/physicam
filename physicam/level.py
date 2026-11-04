"""Liquid-level measurement from SAM 3.1 mask rasters.

Decodes mask rasters for every frame of the liquid track from cached SAM
text (same parser as :mod:`physicam.strobe`; no new API calls when cached).

Surface/bottom definitions (see task spec):
- ``surface_y``: first mask row whose width is >= 50% of the median row
  width, refined to sub-pixel by linear interpolation of the row coverage
  fraction across the boundary. Robust to meniscus, splashes and the exit
  jet.
- ``bottom_y``: last such row; its median over the clip is the fixed bottom.
- ``h`` for Torricelli is measured above the *hole*, not the jar bottom:
  the hole level is the final steady surface level (median ``surface_y``
  over a trailing window of at least 1 s where |dh/dt| ~ 0). Facts
  ``hole_y_px`` / ``hole_height_m`` override this estimate. If the clip
  ends before draining stops, ``hole_estimated`` is False and ``h_px``
  falls back to height above the mask bottom (physics then uses the
  Torricelli-fit asymptote, with a warning).

Coordinate conventions match :mod:`physicam.sam`: frame pixels, y DOWN,
raster pixel (i, j) centre at ``x = left + (j+.5)*sx``,
``y = top + (i+.5)*sy``.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _row_surface_bottom(raster: np.ndarray, bounds) -> tuple[float, float, np.ndarray, np.ndarray, float]:
    """Sub-pixel surface/bottom and row widths for one mask raster.

    Returns (surface_y, bottom_y, row_centre_y, row_width_px, median_width_px).
    """
    left, top, right, bottom = (float(v) for v in bounds)
    rh, rw = raster.shape
    sx = (right - left) / rw if rw else 0.0
    sy = (bottom - top) / rh if rh else 0.0
    widths_counts = np.asarray(raster, dtype=float).sum(axis=1)
    widths_px = widths_counts * sx
    positive = widths_px[widths_px > 0]
    if positive.size == 0:
        return float("nan"), float("nan"), np.zeros(0), np.zeros(0), 0.0
    med_w = float(np.median(positive))
    if med_w <= 0:
        return float("nan"), float("nan"), np.zeros(0), np.zeros(0), 0.0
    centres = top + (np.arange(rh, dtype=float) + 0.5) * sy
    coverage = widths_px / med_w
    full = np.nonzero(coverage >= 0.5)[0]
    if full.size == 0:
        return float("nan"), float("nan"), centres, widths_px, med_w
    i0, i1 = int(full[0]), int(full[-1])
    # Surface: interpolate coverage crossing 0.5 between row i0-1 and i0.
    if i0 > 0 and coverage[i0] != coverage[i0 - 1]:
        frac = (0.5 - coverage[i0 - 1]) / (coverage[i0] - coverage[i0 - 1])
        frac = float(np.clip(frac, 0.0, 1.0))
        surface = float(centres[i0 - 1] + frac * sy)
    else:
        surface = float(top + i0 * sy)  # top edge of first full row
    if i1 + 1 < rh and coverage[i1] != coverage[i1 + 1]:
        frac = (0.5 - coverage[i1]) / (coverage[i1 + 1] - coverage[i1])
        frac = float(np.clip(frac, 0.0, 1.0))
        bot = float(centres[i1] + frac * sy)
    else:
        bot = float(top + (i1 + 1) * sy)
    return surface, bot, centres, widths_px, med_w


def surface_bottom_from_raster(raster, bounds):
    """Public helper: (surface_y, bottom_y) for one raster."""
    s, b, *_ = _row_surface_bottom(np.asarray(raster), bounds)
    return s, b


@dataclass
class LevelSeries:
    """Liquid level time series in frame pixels."""

    t: np.ndarray
    h_px: np.ndarray  # height above the hole (or bottom fallback)
    surface_y: np.ndarray
    bottom_y: float  # median bottom over the clip
    width_profile: dict = field(default_factory=dict)  # {y_px, h_px, width_px}
    frames: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    conf: np.ndarray = field(default_factory=lambda: np.zeros(0))
    fps: float = 30.0
    prompt: str = ""
    hole_y: float = float("nan")
    hole_estimated: bool = False
    hole_method: str = ""
    bottom_series: np.ndarray = field(default_factory=lambda: np.zeros(0))
    median_width_px: float = 0.0
    m_per_px: float | None = None
    warnings: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def h_m(self):
        if self.m_per_px is None:
            return None
        return self.h_px * float(self.m_per_px)

    def summary(self) -> dict:
        """Compact dict for the LLM / agent tool result."""
        if self.n == 0:
            return {"n_frames": 0, "prompt": self.prompt}
        h = np.asarray(self.h_px, dtype=float)
        finite = h[np.isfinite(h)]
        # Monotonicity: fraction of diffs <= small tolerance.
        d = np.diff(finite) if finite.size > 1 else np.zeros(0)
        mono = float(np.mean(d <= 0.5)) if d.size else 1.0
        wp = self.width_profile or {}
        widths = np.asarray(wp.get("width_px", []), dtype=float)
        return {
            "n_frames": int(self.n),
            "frame_range": [int(self.frames[0]), int(self.frames[-1])] if self.n else [],
            "t_range": [round(float(self.t[0]), 4), round(float(self.t[-1]), 4)] if self.n else [],
            "h0_px": round(float(finite[0]), 2) if finite.size else None,
            "h_final_px": round(float(finite[-1]), 2) if finite.size else None,
            "h_max_px": round(float(np.max(finite)), 2) if finite.size else None,
            "surface_y_range": [round(float(np.min(self.surface_y)), 2), round(float(np.max(self.surface_y)), 2)],
            "bottom_y": round(float(self.bottom_y), 2),
            "hole_y": round(float(self.hole_y), 2) if np.isfinite(self.hole_y) else None,
            "hole_estimated": bool(self.hole_estimated),
            "median_width_px": round(float(self.median_width_px), 2),
            "monotonic_fraction": round(mono, 4),
            "width_profile": {
                "n": int(len(widths)),
                "median_width_px": round(float(np.median(widths)), 2) if widths.size else None,
                "min_width_px": round(float(np.min(widths)), 2) if widths.size else None,
                "max_width_px": round(float(np.max(widths)), 2) if widths.size else None,
            },
            "warnings": list(self.warnings),
        }

    def to_json(self) -> dict:
        """JSON-serialisable dict (pipeline `level` shape plus extras)."""
        out = {
            "t": [round(float(v), 6) for v in self.t],
            "h": [round(float(v), 4) for v in self.h_px],
            "h_px": [round(float(v), 4) for v in self.h_px],
            "surface_y": [round(float(v), 4) for v in self.surface_y],
            "bottom_y": float(self.bottom_y),
            "hole_y": float(self.hole_y) if np.isfinite(self.hole_y) else None,
            "hole_estimated": bool(self.hole_estimated),
            "frames": [int(v) for v in self.frames],
            "conf": [round(float(v), 4) for v in self.conf],
            "fps": float(self.fps),
            "prompt": self.prompt,
            "unit": "m" if self.m_per_px is not None else "px",
            "m_per_px": float(self.m_per_px) if self.m_per_px is not None else None,
            "median_width_px": float(self.median_width_px),
            "width_profile": {
                k: [round(float(v), 4) for v in np.asarray(val, dtype=float)]
                for k, val in (self.width_profile or {}).items()
            },
            "warnings": list(self.warnings),
        }
        if self.m_per_px is not None:
            out["h_m"] = [round(float(v), 6) for v in self.h_px * float(self.m_per_px)]
        return out

    def to_csv(self, m_per_px: float | None = None) -> str:
        """CSV text with columns ``t_s, h_px, h_m, surface_y_px``."""
        scale = m_per_px if m_per_px is not None else self.m_per_px
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["t_s", "h_px", "h_m", "surface_y_px"])
        for i in range(self.n):
            h_m = float(self.h_px[i]) * float(scale) if scale is not None else ""
            w.writerow([f"{float(self.t[i]):.6f}", f"{float(self.h_px[i]):.4f}",
                        f"{h_m:.6f}" if scale is not None else "", f"{float(self.surface_y[i]):.4f}"])
        return buf.getvalue()

    def write_csv(self, path, m_per_px: float | None = None) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_csv(m_per_px))
        return p

    # Back-compat aliases used by physics/UI code.
    @property
    def h(self):  # noqa: D102
        return self.h_px


def _resolve_sam_text(video_path, sam_text_or_cache, prompt: str, data_dir: str = "data") -> str:
    """Resolve SAM text from an explicit string/path, or the disk cache."""
    if sam_text_or_cache is not None:
        if isinstance(sam_text_or_cache, Path) and sam_text_or_cache.exists():
            return sam_text_or_cache.read_text()
        if isinstance(sam_text_or_cache, str):
            p = Path(sam_text_or_cache)
            # A path to an existing cache file vs raw SAM text.
            if len(sam_text_or_cache) < 500 and p.exists():
                return p.read_text()
            if "<" in sam_text_or_cache and "id" in sam_text_or_cache:
                return sam_text_or_cache
            if p.exists():
                return p.read_text()
            return sam_text_or_cache
        # dict {(oid, frame): ...} is handled by caller via rasters path.
        raise TypeError("sam_text_or_cache must be SAM text, a path, or None")
    # Cache lookup (no API calls).
    try:
        from physicam.strobe import cached_sam_text
        from physicam.video import sha256_file
        sha = sha256_file(video_path)
        text = cached_sam_text(data_dir, sha, prompt)
        if text is not None:
            return text
    except Exception:
        pass
    raise FileNotFoundError(
        f"no cached SAM text for prompt {prompt!r}; run segment_video first (cache miss would need the API)")


def _parse_rasters_with_conf(text: str):
    """Parse SAM text -> {(oid, frame): (raster, bounds, conf)}.

    Kept for compatibility; rasters come from the parallel mask cache
    (lazily unpacked bit-packed rasters, never a serial re-decode).
    """
    from physicam.maskcache import decoded_masks
    from physicam.sam import mask_cache_key_for_text

    if not text or not text.strip():
        return {}
    ms = decoded_masks(text, mask_cache_key_for_text(text))
    out = {}
    for rec in ms.records:
        if not rec.has_mask:
            continue
        out[(rec.object_id, int(rec.frame))] = (
            ms.raster(rec.index), tuple(float(v) for v in rec.bounds), float(rec.confidence))
    return out


def _estimate_hole_y(t, surface_y, bottom_y, fps):
    """Hole level = final steady surface; returns (hole_y, estimated, method)."""
    n = len(t)
    if n == 0:
        return float(bottom_y), False, "bottom_fallback"
    win = max(int(round(fps * 1.0)), 1)  # at least 1 s
    win = min(win, n)
    tail_surface = surface_y[-win:]
    hole = float(np.median(tail_surface))
    # Steadiness: slope over the trailing window and residual height.
    if win >= 2 and t[-1] > t[-win]:
        slope = float(np.polyfit(t[-win:], surface_y[-win:], 1)[0])  # px/s, surface falling => slope >0 (y down? surface_y increases as level falls)
        # surface_y increases when level falls; steady means |slope| small.
    else:
        slope = 0.0
    h_tail = hole - float(np.median(surface_y[:max(1, min(win, n))]))  # negative-ish magnitude check below
    h0_above_bottom = float(bottom_y - np.median(surface_y[:max(1, min(win, n))]))
    h_final_above_hole = 0.0  # by construction at the median tail
    # Draining still in progress if the surface is still moving in last window
    # or the tail window range is large compared to noise.
    tail_range = float(np.max(tail_surface) - np.min(tail_surface)) if win else 0.0
    still_moving = abs(slope) > 1.0 and tail_range > 3.0  # >1 px/s sustained
    # Also: if the last window is not near a plateau (monotonic trend), not steady.
    if still_moving:
        return float(bottom_y), False, "bottom_fallback_clip_ended_early"
    return hole, True, "final_steady_surface"


def level_series_from_rasters(rasters: dict, fps: float, prompt: str = "",
                              m_per_px: float | None = None,
                              hole_y_px: float | None = None,
                              hole_height_m: float | None = None) -> LevelSeries:
    """Build a LevelSeries from {(oid,frame): (raster,bounds[,conf])}.

    Per frame the largest-area mask wins (this stitches/deduplicates ids
    for a single liquid body). Used by tests with synthetic rasters.
    """
    # Per (oid, frame) measurements.
    per_frame: dict[int, list] = {}
    width_samples_first: list = []
    frames_sorted = sorted({f for _, f in rasters})
    meas: dict[tuple, dict] = {}
    for (oid, frame), val in rasters.items():
        raster = np.asarray(val[0])
        bounds = val[1]
        conf = float(val[2]) if len(val) > 2 else 0.0
        if raster.size == 0 or not np.any(raster):
            continue
        surface, bot, centres, widths_px, med_w = _row_surface_bottom(raster, bounds)
        if not np.isfinite(surface):
            continue
        left, top, right, bottom = (float(v) for v in bounds)
        sx = (right - left) / raster.shape[1] if raster.shape[1] else 0.0
        sy = (bottom - top) / raster.shape[0] if raster.shape[0] else 0.0
        area = float(raster.sum()) * sx * sy
        meas[(oid, int(frame))] = {
            "surface": surface, "bottom": bot, "centres": centres,
            "widths": widths_px, "med_w": med_w, "area": area, "conf": conf,
        }
        per_frame.setdefault(int(frame), []).append(oid)

    frames_list, surfaces, bottoms, confs, med_widths = [], [], [], [], []
    chosen: dict[int, dict] = {}
    for frame in frames_sorted:
        cands = [(oid, meas[(oid, frame)]) for oid in per_frame.get(frame, []) if (oid, frame) in meas]
        if not cands:
            continue
        oid, m = max(cands, key=lambda c: c[1]["area"])
        chosen[frame] = m
        frames_list.append(frame)
        surfaces.append(m["surface"])
        bottoms.append(m["bottom"])
        confs.append(m["conf"])
        med_widths.append(m["med_w"])

    frames_arr = np.asarray(frames_list, dtype=int)
    surface_y = np.asarray(surfaces, dtype=float)
    bottom_series = np.asarray(bottoms, dtype=float)
    bottom_y = float(np.median(bottom_series)) if bottom_series.size else 0.0
    t = frames_arr.astype(float) / float(fps) if fps else frames_arr.astype(float)

    # Width profile on the first frames (full jar): average widths per y-bin.
    width_profile: dict = {"y_px": [], "h_px": [], "width_px": []}
    if frames_list:
        k_first = min(len(frames_list), max(5, int(round(fps)) if fps else 5))
        # Bin by rounded centre y (1 px bins).
        acc: dict[float, list] = {}
        for frame in frames_list[:k_first]:
            m = chosen[frame]
            for cy, w in zip(m["centres"], m["widths"]):
                if w > 0:
                    acc.setdefault(round(float(cy)), []).append(float(w))
        ys = sorted(acc)
        wp_y = np.asarray(ys, dtype=float)
        wp_w = np.asarray([float(np.median(acc[y])) for y in ys], dtype=float)
        width_profile = {"y_px": wp_y, "h_px": bottom_y - wp_y, "width_px": wp_w}

    median_width = float(np.median(med_widths)) if med_widths else 0.0
    return _finalize_level(frames_arr, surface_y, bottom_series, confs, median_width,
                           width_profile, fps, prompt, m_per_px, hole_y_px, hole_height_m)


def _finalize_level(frames_arr, surface_y, bottom_series, confs, median_width,
                    width_profile, fps, prompt, m_per_px, hole_y_px, hole_height_m) -> LevelSeries:
    """Shared hole-level / h_px assembly for raster- and stats-based input."""
    bottom_y = float(np.median(bottom_series)) if len(bottom_series) else 0.0
    t = frames_arr.astype(float) / float(fps) if fps else frames_arr.astype(float)
    warnings: list[str] = []
    if hole_y_px is not None:
        hole_y, hole_est, hole_method = float(hole_y_px), True, "fact_hole_y_px"
    elif hole_height_m is not None and m_per_px:
        hole_y = bottom_y - float(hole_height_m) / float(m_per_px)
        hole_est, hole_method = True, "fact_hole_height_m"
    else:
        hole_y, hole_est, hole_method = _estimate_hole_y(t, surface_y, bottom_y, float(fps))
        if not hole_est:
            warnings.append("clip ends before draining stops: hole level unknown; "
                            "h is above the mask bottom and the Torricelli fit asymptote should be used")
    h_px = np.asarray(hole_y - surface_y, dtype=float) if hole_est else np.asarray(bottom_y - surface_y, dtype=float)
    return LevelSeries(
        t=t, h_px=h_px, surface_y=surface_y, bottom_y=bottom_y,
        width_profile=width_profile, frames=frames_arr,
        conf=np.asarray(confs, dtype=float), fps=float(fps), prompt=prompt,
        hole_y=float(hole_y), hole_estimated=bool(hole_est), hole_method=hole_method,
        bottom_series=np.asarray(bottom_series, dtype=float), median_width_px=float(median_width),
        m_per_px=m_per_px, warnings=warnings)


def level_series_from_maskset(ms, fps: float, prompt: str = "",
                              m_per_px: float | None = None,
                              hole_y_px: float | None = None,
                              hole_height_m: float | None = None) -> LevelSeries:
    """Build a LevelSeries from a MaskSet's precomputed stats (no unpacking).

    Per frame the largest-area mask wins, as in
    :func:`level_series_from_rasters`. The width profile uses the stored
    row-width profiles of the first frames.
    """
    per_frame: dict[int, object] = {}
    for rec in ms.records:
        if not rec.has_mask or not np.isfinite(rec.surface_y):
            continue
        cur = per_frame.get(int(rec.frame))
        if cur is None or rec.area > cur.area:
            per_frame[int(rec.frame)] = rec
    frames_list = sorted(per_frame)
    if not frames_list:
        raise ValueError(f"no mask records for level (prompt {prompt!r})")
    frames_arr = np.asarray(frames_list, dtype=int)
    surface_y = np.asarray([per_frame[f].surface_y for f in frames_list], dtype=float)
    bottom_series = np.asarray([per_frame[f].bottom_y for f in frames_list], dtype=float)
    confs = [per_frame[f].confidence for f in frames_list]
    med_widths = [per_frame[f].median_width_px for f in frames_list]
    bottom_y = float(np.median(bottom_series)) if bottom_series.size else 0.0

    width_profile: dict = {"y_px": [], "h_px": [], "width_px": []}
    k_first = min(len(frames_list), max(5, int(round(fps)) if fps else 5))
    acc: dict[float, list] = {}
    for f in frames_list[:k_first]:
        rec = per_frame[f]
        centres, widths = ms.row_widths(rec.index)
        for cy, w in zip(centres, widths):
            if w > 0:
                acc.setdefault(round(float(cy)), []).append(float(w))
    if acc:
        ys = sorted(acc)
        wp_y = np.asarray(ys, dtype=float)
        wp_w = np.asarray([float(np.median(acc[y])) for y in ys], dtype=float)
        width_profile = {"y_px": wp_y, "h_px": bottom_y - wp_y, "width_px": wp_w}
    median_width = float(np.median(med_widths)) if med_widths else 0.0
    return _finalize_level(frames_arr, surface_y, bottom_series, confs, median_width,
                           width_profile, fps, prompt, m_per_px, hole_y_px, hole_height_m)


def level_series(video_path, sam_text_or_cache=None, fps: float = 30.0, prompt: str = "brown liquid",
                 *, m_per_px: float | None = None, hole_y_px: float | None = None,
                 hole_height_m: float | None = None, data_dir: str = "data") -> LevelSeries:
    """Measure the liquid level for every frame from cached SAM text.

    ``sam_text_or_cache`` may be raw SAM text, a path to a cached SAM text
    file, a dict of rasters (see :func:`level_series_from_rasters`), or
    None to read ``data/cache/sam/`` for *video_path* + *prompt* (no API
    calls). ``video_path`` is only used for the cache key in the last case.
    """
    if isinstance(sam_text_or_cache, dict):
        return level_series_from_rasters(sam_text_or_cache, fps, prompt,
                                         m_per_px=m_per_px, hole_y_px=hole_y_px,
                                         hole_height_m=hole_height_m)
    text = _resolve_sam_text(video_path, sam_text_or_cache, prompt, data_dir=data_dir)
    # Parallel mask cache: stats (surface/bottom/widths) without unpacking.
    from physicam.maskcache import decoded_masks
    from physicam.sam import mask_cache_key, mask_cache_key_for_text

    cache_key = None
    try:
        from pathlib import Path as _P

        from physicam.video import sha256_file

        if video_path is not None and _P(video_path).exists():
            cache_key = mask_cache_key(sha256_file(video_path), prompt)
    except Exception:
        cache_key = None
    ms = decoded_masks(text, cache_key or mask_cache_key_for_text(text), data_dir=data_dir)
    if not any(r.has_mask for r in ms.records):
        raise ValueError(f"no mask rasters in SAM text for prompt {prompt!r}")
    return level_series_from_maskset(ms, fps, prompt, m_per_px=m_per_px,
                                     hole_y_px=hole_y_px, hole_height_m=hole_height_m)


def smooth_series(t, y, window_s: float = 2.0, fps: float | None = None):
    """Optional Savitzky-Golay smoothing (for derivatives/plots; fits use raw)."""
    y = np.asarray(y, dtype=float)
    if len(y) < 5:
        return y.copy()
    if fps is None:
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
        fps = 1.0 / dt if dt > 0 else 30.0
    win = int(round(window_s * float(fps)))
    if win % 2 == 0:
        win += 1
    win = min(win, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if win < 5:
        return y.copy()
    from scipy.signal import savgol_filter
    return savgol_filter(y, window_length=win, polyorder=2)


def calibrate_from_container_diameter(median_width_px: float, container_diameter_m: float):
    """Calibration from the container diameter: m_per_px = D / median width.

    Caveat (systematic): a liquid-filled cylinder refracts horizontally,
    so the apparent liquid width is close to the *outer* diameter while
    vertical distances are not magnified. The returned calibration carries
    this note and a >=5% relative uncertainty.
    """
    from physicam.calibrate import Calibration
    if not median_width_px or median_width_px <= 0:
        raise ValueError("median_width_px must be positive")
    return Calibration(
        m_per_px=float(container_diameter_m) / float(median_width_px),
        rel_uncertainty=0.05, method="container_diameter",
        details=(f"m_per_px=D/median liquid-mask width={container_diameter_m} m/{median_width_px:.1f} px; "
                 "systematic: liquid-filled cylinder refracts horizontally, apparent width ~ outer "
                 "diameter, vertical distances not magnified; prefer a vertical ruler/h0 reference"))


def calibrate_from_h0(h0_px: float, h0_m: float):
    """Calibration from a known initial height: m_per_px = h0_m / h0_px."""
    from physicam.calibrate import Calibration
    if not h0_px or h0_px <= 0:
        raise ValueError("h0_px must be positive")
    return Calibration(
        m_per_px=float(h0_m) / float(h0_px), rel_uncertainty=0.03, method="h0",
        details=f"m_per_px=h0_m/h0_px={h0_m} m/{h0_px:.1f} px (vertical reference, no refraction bias)")
