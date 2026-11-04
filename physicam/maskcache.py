"""Parallel, disk-cached decoding of SAM 3.1 mask rasters.

``decode_mask_to_raster`` from ``meta_sam_parser`` is a pure-Python range
decoder and is the bottleneck for long videos (thousands of frames x
several candidate prompts). This module parses the SAM text once, decodes
every mask raster in a ``ProcessPoolExecutor`` (chunks of ~64 records to
amortize IPC), bit-packs the rasters with ``np.packbits`` and caches the
result in ``data/cache/masks/<key>.npz`` (compressed). Warm calls load
the npz without any mask decoding.

Workers also precompute per-record statistics (area, centroid, equivalent
diameter, liquid-level surface/bottom and the row-width profile), so most
consumers (track building, cross-prompt dedup, level extraction) never
unpack a raster at all. Rasters are unpacked lazily via
:meth:`MaskSet.raster`. SVG contours are *not* part of the main decode:
they are computed only for an explicitly selected subset of records
(:meth:`MaskSet.ensure_contours`, at most a few hundred frames) in the
same worker pool and cached in a sidecar npz.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CHUNK = 8
MAX_WORKERS_CAP = 12
_SERIAL_BELOW = 32  # tiny inputs: pool startup costs more than decoding

_SAFE_KEY_RE = re.compile(r"[^A-Za-z0-9._-]+")


def mask_cache_path(data_dir: str | Path, cache_key: str) -> Path:
    """Path of the npz cache file for *cache_key* under *data_dir*."""
    safe = _SAFE_KEY_RE.sub("_", str(cache_key)).strip("_") or "masks"
    return Path(data_dir) / "cache" / "masks" / f"{safe}.npz"


def _contour_cache_path(data_dir: str | Path, cache_key: str) -> Path:
    p = mask_cache_path(data_dir, cache_key)
    return p.with_name(p.name[:-4] + ".contours.npz")


def _save_contours(path: Path, text_sha: str, contours: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(contours)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, text_sha=np.asarray(text_sha),
                        index=np.asarray(keys, dtype=np.int64),
                        path=np.asarray([contours[k] for k in keys]))
    tmp.replace(path)


def _resolve_text(source) -> str:
    """Resolve *source* (raw SAM text, a path, or a Path) to SAM text."""
    if isinstance(source, Path):
        return source.read_text()
    if isinstance(source, str):
        if "<" in source and "mask" in source:
            return source
        p = Path(source)
        if len(source) < 500 and p.exists():
            return p.read_text()
        if p.exists():
            return p.read_text()
        return source
    raise TypeError("source must be SAM text, a path string, or a Path")


def _oid_key(oid: str):
    return (len(oid), oid)


# ---------------------------------------------------------------------------
# Worker-side decoding (top-level functions: picklable for ProcessPool)
# ---------------------------------------------------------------------------

def _surface_bottom(widths_px: np.ndarray, top: float, sy: float):
    """Sub-pixel surface/bottom from row widths (mirrors physicam.level)."""
    positive = widths_px[widths_px > 0]
    if positive.size == 0:
        return float("nan"), float("nan"), 0.0
    med_w = float(np.median(positive))
    if med_w <= 0:
        return float("nan"), float("nan"), 0.0
    coverage = widths_px / med_w
    full = np.nonzero(coverage >= 0.5)[0]
    if full.size == 0:
        return float("nan"), float("nan"), med_w
    rh = len(widths_px)
    i0, i1 = int(full[0]), int(full[-1])
    if i0 > 0 and coverage[i0] != coverage[i0 - 1]:
        frac = float(np.clip((0.5 - coverage[i0 - 1]) / (coverage[i0] - coverage[i0 - 1]), 0.0, 1.0))
        surface = float(top + (i0 - 1 + 0.5) * sy + frac * sy)
    else:
        surface = float(top + i0 * sy)
    if i1 + 1 < rh and coverage[i1] != coverage[i1 + 1]:
        frac = float(np.clip((0.5 - coverage[i1]) / (coverage[i1 + 1] - coverage[i1]), 0.0, 1.0))
        bot = float(top + (i1 + 0.5) * sy + frac * sy)
    else:
        bot = float(top + (i1 + 1) * sy)
    return surface, bot, med_w


def _contour_from_raster(raster: np.ndarray) -> str:
    """SVG polygon (raster coords) tracing one mask raster's outline.

    Built from per-row foreground extents of the already-decoded raster,
    so selected contours cost no second mask decode. Points are kept
    only where the outline x changes, which keeps paths compact.
    """
    fg = np.asarray(raster) != 0
    rows = np.nonzero(fg.any(axis=1))[0]
    if rows.size == 0:
        return ""
    lefts = np.argmax(fg, axis=1)  # first fg col per row (valid on fg rows)
    rights = fg.shape[1] - 1 - np.argmax(fg[:, ::-1], axis=1)
    pts: list[tuple[float, float]] = []
    for i in rows:  # left edge, top -> bottom (x = left edge of fg run)
        x, y = float(lefts[i]), float(i)
        if not pts or pts[-1][0] != x:
            pts.append((x, y))
    for i in rows[::-1]:  # right edge, bottom -> top (x = right edge + 1)
        x, y = float(rights[i] + 1), float(i + 1)
        if not pts or pts[-1][0] != x:
            pts.append((x, y))
    if len(pts) < 3:
        return ""
    body = "L".join(f"{x:g} {y:g}" for x, y in pts)
    return f"M{body}Z"


def _decode_one(idx, mask, bounds_mask, bounds_parse, want_contour=False):
    """Decode one mask and compute its stats. Runs in worker processes."""
    from meta_sam_parser import decode_mask_to_raster

    h, w = int(mask.height), int(mask.width)
    try:
        raw = decode_mask_to_raster(mask)
    except Exception:
        # Mirror the official parser, which drops invalid mask payloads
        # (its _accept_mask validates by decoding) but keeps the box.
        return (idx, None, h, w, 0.0, 0.0, 0.0, 0.0,
                float("nan"), float("nan"), 0.0, b"", "")
    raster = np.frombuffer(raw, np.uint8).reshape(h, w)
    packed = np.packbits(raster).tobytes()
    x1, y1, x2, y2 = (float(v) for v in bounds_parse)
    sx = (x2 - x1) / w if w else 0.0
    sy = (y2 - y1) / h if h else 0.0
    # Two reduction passes total: row sums serve the centroid, the area
    # and the level row-width profile.
    rowsum = raster.sum(axis=1, dtype=np.float64)
    colsum = raster.sum(axis=0, dtype=np.float64)
    count = float(rowsum.sum())
    if count > 0:
        cx = x1 + float(((np.arange(w) + 0.5) * colsum).sum() / count) * sx
        cy = y1 + float(((np.arange(h) + 0.5) * rowsum).sum() / count) * sy
        area = count * sx * sy
    else:
        cx, cy, area = (x1 + x2) / 2.0, (y1 + y2) / 2.0, 0.0
    eq = math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0
    # Level stats use the mask bounds (the raster's true frame mapping).
    mx1, my1, mx2, _my2 = (float(v) for v in bounds_mask)
    msx = (mx2 - mx1) / w if w else 0.0
    msy = ((_my2 - my1) / h) if h else 0.0
    widths_px = rowsum * msx
    surface, bottom, med_w = _surface_bottom(widths_px, my1, msy)
    contour = _contour_from_raster(raster) if want_contour else ""
    # Widths travel as float32 bytes (compact IPC; stored as float32 too).
    return (idx, packed, h, w, area, cx, cy, eq, surface, bottom, med_w,
            widths_px.astype(np.float32).tobytes(), contour)


def _decode_chunk(jobs):
    """Decode a chunk of (idx, mask, bounds_mask, bounds_parse, want_contour)."""
    return [_decode_one(idx, mask, bm, bp, wc) for idx, mask, bm, bp, wc in jobs]


def _contour_one(idx, mask):
    from meta_sam_parser import decode_mask_to_svg_path

    try:
        return idx, str(decode_mask_to_svg_path(mask))
    except Exception:
        return idx, ""


def _contour_chunk(jobs):
    return [_contour_one(idx, mask) for idx, mask in jobs]


def _chunks(seq, n=CHUNK):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _run_parallel(fn, jobs, max_workers):
    """Run *fn* over chunks of *jobs*, in parallel or serially."""
    if not jobs:
        return []
    workers = max_workers if max_workers is not None else min(os.cpu_count() or 1, MAX_WORKERS_CAP)
    chunked = list(_chunks(jobs))
    if workers <= 1 or len(jobs) < _SERIAL_BELOW:
        out = []
        for c in chunked:
            out.extend(fn(c))
        return out
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        out = []
        # Consume chunks as they finish: under uneven host load this avoids
        # head-of-line blocking behind a slow chunk.
        from concurrent.futures import as_completed

        futures = [pool.submit(fn, c) for c in chunked]
        for fut in as_completed(futures):
            out.extend(fut.result())
    return out


# ---------------------------------------------------------------------------
# Parsing (envelope only; the mask codec itself is never hand-rolled)
# ---------------------------------------------------------------------------
#
# The official parser validates every mask by decoding it inside
# ``_accept_mask`` (``decode_mask_to_raster`` just to check the payload),
# so a serial parse of a long video pays the full pure-Python decode cost
# before any parallelism is possible (~0.25 s/frame on the draining clip).
# The hot path therefore splits the documented line envelope with regexes
# (frame header, object id, box fields, ``data=h,w,payload``) and builds
# ``SegmentationMask`` objects directly; decoding always uses the official
# ``decode_mask_to_raster`` / ``decode_mask_to_svg_path`` in the workers.
# Any line that does not match the envelope exactly falls back to the
# official parser for the whole text.

_LINE_RE = re.compile(r"^<(\d+)f>(.*)$")
_REC_RE = re.compile(
    r"(?:^|,)([0-9]+)<\|box;x1=(-?\d+);y1=(-?\d+);x2=(-?\d+);y2=(-?\d+);w=(\d+);h=(\d+)"
    r"(?:;c=([0-9.eE+-]+))?\|>"
    r"(?:<\|mask;x=0;y=0(?:;c=([0-9.eE+-]+))?;data=(\d+),(\d+),([!~][^|]*)\|>)?")


def _parse_raw_fast(text: str):
    """Fast envelope parse -> (boxes, raw_masks) or None to fall back.

    ``boxes``: {(oid, frame): (x1, y1, x2, y2, conf)} with half-open
    bounds (right/bottom = inclusive + 1, as the official parser emits);
    ``raw_masks``: {(oid, frame): (SegmentationMask, bounds, mask_conf)}.
    """
    from meta_sam_parser import SegmentationMask

    boxes: dict = {}
    raw_masks: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if m is None:
            return None
        frame = int(m.group(1))
        body = m.group(2)
        if body.count("<|box;") != body.count("<|mask;"):
            # Box-only records are not part of the video wire format here;
            # let the official parser handle anything unusual.
            if "<|mask;" not in body:
                return None
        found = 0
        for r in _REC_RE.finditer(body):
            found += 1
            oid = r.group(1)
            x1, y1, x2i, y2i = (int(r.group(k)) for k in (2, 3, 4, 5))
            bounds = (float(x1), float(y1), float(x2i + 1), float(y2i + 1))
            try:
                box_conf = float(r.group(8)) if r.group(8) else 0.0
            except ValueError:
                return None
            boxes[(oid, frame)] = (bounds[0], bounds[1], bounds[2], bounds[3], box_conf)
            if r.group(10) is not None:
                h, w = int(r.group(10)), int(r.group(11))
                payload = r.group(12)
                try:
                    mask_conf = float(r.group(9)) if r.group(9) else box_conf
                except ValueError:
                    return None
                mask = SegmentationMask(
                    encoding="lossless" if payload.startswith("~") else "one_bit",
                    payload=payload, width=w, height=h)
                raw_masks[(oid, frame)] = (mask, bounds, mask_conf)
        if found == 0 or found != body.count("<|box;"):
            return None
    return boxes, raw_masks


def _parse_raw_official(text: str):
    """Official-parser parse -> (boxes, raw_masks). Decodes to validate."""
    from meta_sam_parser import CompletedOutcome, video_segmentation_format

    boxes: dict = {}
    raw_masks: dict = {}
    if not text or not text.strip():
        return boxes, raw_masks
    parser = video_segmentation_format().create_parser()
    parser.push(text if text.endswith("\n") else text + "\n")
    result = parser.finish(CompletedOutcome()).result
    for rec in result.records:
        if rec.kind == "box":
            boxes[(str(rec.object_id), int(rec.frame.frame_index))] = (
                float(rec.left), float(rec.top), float(rec.right), float(rec.bottom),
                float(rec.confidence or 0.0))
        elif rec.kind == "mask":
            b = rec.bounds
            raw_masks[(str(rec.object_id), int(rec.frame.frame_index))] = (
                rec.mask, (float(b.left), float(b.top), float(b.right), float(b.bottom)),
                float(rec.confidence or 0.0))
    return boxes, raw_masks


def parse_records(text: str):
    """Parse SAM text once; return (boxes, masks) without decoding masks.

    ``boxes``: {(oid, frame): (x1, y1, x2, y2, conf)};
    ``masks``: {(oid, frame): (SegmentationMask, bounds, mask_conf)}.
    """
    if not text or not text.strip():
        return {}, {}
    fast = _parse_raw_fast(text)
    if fast is not None:
        return fast
    return _parse_raw_official(text)


def box_records(text: str) -> dict:
    """Box records only: {(oid, frame): (x1, y1, x2, y2, conf)}.

    Splitting boxes out of the line envelope never decodes a mask raster,
    so cross-prompt dedup can run on boxes before any raster decode is
    paid for.
    """
    boxes, _ = parse_records(text)
    return boxes


# ---------------------------------------------------------------------------
# MaskSet
# ---------------------------------------------------------------------------

@dataclass
class MaskRecord:
    """One decoded mask (or box-only) record with precomputed stats."""

    index: int
    object_id: str
    frame: int
    confidence: float
    bounds: tuple  # mask bounds (left, top, right, bottom), frame px
    box: tuple  # SAM box bounds, frame px (== bounds when no box record)
    raster_width: int = 0
    raster_height: int = 0
    has_mask: bool = False
    area: float = 0.0
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    eq_diameter: float = 0.0
    surface_y: float = float("nan")
    bottom_y: float = float("nan")
    median_width_px: float = 0.0


@dataclass
class MaskSet:
    """Decoded masks for one SAM text: packed rasters + per-record stats."""

    records: list = field(default_factory=list)
    cache_key: str = ""
    text_sha: str = ""
    _packed: bytes = b""
    _offsets: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    _widths: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    _contours: dict = field(default_factory=dict)  # index -> raw raster-space SVG path
    _text: str = field(default="", repr=False)
    _data_dir: str = "data"
    _index: dict = field(default_factory=dict)

    def __post_init__(self):
        self._index = {(r.object_id, int(r.frame)): int(r.index) for r in self.records}

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, i):
        return self.records[i]

    # -- lookups ------------------------------------------------------------
    def index_of(self, object_id: str, frame: int):
        """Global record index for (object_id, frame), or None."""
        return self._index.get((str(object_id), int(frame)))

    def record_for(self, object_id: str, frame: int):
        idx = self.index_of(object_id, frame)
        return self.records[idx] if idx is not None else None

    def object_ids(self) -> list:
        seen: list = []
        for r in self.records:
            if r.object_id not in seen:
                seen.append(r.object_id)
        return seen

    def for_object(self, object_id: str) -> list:
        return [r for r in self.records if r.object_id == str(object_id)]

    def by_frame(self) -> dict:
        out: dict = {}
        for r in self.records:
            out.setdefault(int(r.frame), []).append(r)
        return out

    # -- lazy raster unpack ---------------------------------------------------
    def raster(self, i: int) -> np.ndarray:
        """Unpack record *i*'s raster as a (h, w) uint8 0/1 array."""
        r = self.records[int(i)]
        if not r.has_mask or r.raster_height == 0 or r.raster_width == 0:
            return np.zeros((max(r.raster_height, 0), max(r.raster_width, 0)), dtype=np.uint8)
        start, end = int(self._offsets[int(i)]), int(self._offsets[int(i) + 1])
        chunk = np.frombuffer(self._packed[start:end], dtype=np.uint8)
        flat = np.unpackbits(chunk, count=r.raster_height * r.raster_width)
        return flat.reshape(r.raster_height, r.raster_width).copy()

    def raster_for(self, object_id: str, frame: int):
        idx = self.index_of(object_id, frame)
        return self.raster(idx) if idx is not None else None

    def row_widths(self, i: int):
        """(row centre y in frame px, row width in px) for record *i*."""
        r = self.records[int(i)]
        if self._widths.size == 0 or r.raster_height == 0:
            return np.zeros(0), np.zeros(0)
        widths = np.asarray(self._widths[int(i), :r.raster_height], dtype=float)
        _x1, y1, _x2, y2 = (float(v) for v in r.bounds)
        sy = (y2 - y1) / r.raster_height if r.raster_height else 0.0
        centres = y1 + (np.arange(r.raster_height, dtype=float) + 0.5) * sy
        return centres, widths

    # -- contours (selected records only, cached) ------------------------------
    @property
    def contours(self) -> dict:
        """{record index: raw raster-space SVG path} currently available."""
        return dict(self._contours)

    def ensure_contours(self, indices, *, max_workers=None) -> dict:
        """Compute (or load cached) raster-space SVG paths for *indices*.

        Only the requested records are decoded. Results are cached in a
        sidecar npz next to the mask cache. Returns {index: svg_path}.
        """
        wanted = sorted({int(i) for i in indices})
        if not wanted:
            return {}
        cpath = _contour_cache_path(self._data_dir, self.cache_key) if self.cache_key else None
        cached: dict = {}
        if cpath is not None and cpath.exists():
            try:
                with np.load(cpath, allow_pickle=False) as z:
                    if str(z["text_sha"]) == self.text_sha:
                        for idx, path in zip(z["index"].tolist(), z["path"].tolist()):
                            cached[int(idx)] = str(path)
            except Exception:
                cached = {}
        self._contours.update({i: cached[i] for i in wanted if i in cached})
        missing = [i for i in wanted if i not in self._contours]
        if missing:
            if not self._text:
                raise RuntimeError("ensure_contours needs the SAM text (MaskSet loaded without it)")
            _boxes, masks = parse_records(self._text)
            jobs = []
            for i in missing:
                r = self.records[i]
                raw = masks.get((r.object_id, int(r.frame)))
                if raw is not None:
                    jobs.append((i, raw[0]))
            for idx, svg in _run_parallel(_contour_chunk, jobs, max_workers):
                self._contours[int(idx)] = svg
            for i in missing:
                self._contours.setdefault(i, "")
            if cpath is not None:
                try:
                    merged = dict(cached)
                    merged.update(self._contours)
                    _save_contours(cpath, self.text_sha, merged)
                except Exception:
                    pass
        return {i: self._contours.get(i, "") for i in wanted}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plan_contour_indices(boxes: dict, masks: dict, max_per_object: int = 300) -> list[int]:
    """Global record indices selected for contours (every k-th per object).

    Record order matches :func:`decoded_masks`: objects sorted by
    (len, oid), then frame. Keeps at most ``max_per_object`` per object.
    """
    keys = sorted(set(boxes) | set(masks), key=lambda k: (_oid_key(k[0]), k[1]))
    pos = {k: i for i, k in enumerate(keys)}
    by_oid: dict[str, list] = {}
    for oid, frame in keys:
        by_oid.setdefault(oid, []).append(frame)
    out: list[int] = []
    for oid in sorted(by_oid, key=lambda s: (len(s), s)):
        frames = sorted(by_oid[oid])
        stride = max(1, math.ceil(len(frames) / max_per_object))
        for f in frames[::stride]:
            if (oid, f) in masks:
                out.append(pos[(oid, f)])
    return out


def decoded_masks(sam_text_path_or_text, cache_key: str, *, data_dir: str | Path = "data",
                  max_workers: int | None = None, use_cache: bool = True,
                  contour_indices=None) -> MaskSet:
    """Decode all masks in a SAM text, in parallel, with an npz disk cache.

    ``sam_text_path_or_text`` is raw SAM text or a path to it; ``cache_key``
    names the cache file ``data/cache/masks/<cache_key>.npz``. Subsequent
    calls with the same key and unchanged text load the npz (no decoding).
    ``contour_indices`` optionally selects records whose SVG contours are
    traced from their rasters in the same worker pass (and sidecar-cached).
    """
    text = _resolve_text(sam_text_path_or_text)
    text_sha = hashlib.sha256(text.encode()).hexdigest()
    cache_path = mask_cache_path(data_dir, cache_key)

    if use_cache and cache_path.exists():
        try:
            ms = _load_npz(cache_path, cache_key, str(data_dir))
            if ms.text_sha == text_sha:
                ms._text = text
                if contour_indices:
                    ms.ensure_contours(contour_indices, max_workers=max_workers)
                return ms
        except Exception:
            pass  # corrupt/stale cache: fall through and re-decode

    boxes, masks = parse_records(text)
    keys = sorted(set(boxes) | set(masks), key=lambda k: (_oid_key(k[0]), k[1]))

    records: list = []
    jobs: list = []
    contour_set = {int(i) for i in (contour_indices or [])}
    for idx, (oid, frame) in enumerate(keys):
        box = boxes.get((oid, frame))
        raw = masks.get((oid, frame))
        if raw is not None:
            mask_obj, bounds_mask, mask_conf = raw
            conf = float(box[4] if box else mask_conf)
        else:
            mask_obj, bounds_mask = None, (box[0], box[1], box[2], box[3])
            conf = float(box[4]) if box else 0.0
        box_t = (box[0], box[1], box[2], box[3]) if box else bounds_mask
        records.append(MaskRecord(index=idx, object_id=str(oid), frame=int(frame),
                                  confidence=conf, bounds=bounds_mask, box=box_t,
                                  has_mask=raw is not None))
        if raw is not None:
            jobs.append((idx, mask_obj, bounds_mask, box_t, idx in contour_set))

    results = _run_parallel(_decode_chunk, jobs, max_workers)
    by_idx = {r[0]: r for r in results}

    packed_parts: list = []
    offsets = [0]
    max_rows = 0
    widths_rows: list = []
    contours: dict = {}
    for r in records:
        if r.has_mask and r.index in by_idx and by_idx[r.index][1] is not None:
            (_i, packed, h, w, area, cx, cy, eq, surf, bot, med_w, widths, contour) = by_idx[r.index]
            if contour:
                contours[r.index] = contour
            r.raster_height, r.raster_width = int(h), int(w)
            r.area, r.centroid_x, r.centroid_y = float(area), float(cx), float(cy)
            r.eq_diameter = float(eq)
            r.surface_y, r.bottom_y = float(surf), float(bot)
            r.median_width_px = float(med_w)
            packed_parts.append(packed)
            offsets.append(offsets[-1] + len(packed))
            widths_rows.append(widths)  # float32 bytes from the worker
            max_rows = max(max_rows, len(widths) // 4)
        else:
            if r.index in by_idx and by_idx[r.index][1] is None:
                r.has_mask = False  # invalid payload: official parser drops the mask
            x1, y1, x2, y2 = (float(v) for v in r.box)
            r.centroid_x, r.centroid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            r.area = max(0.0, (x2 - x1) * (y2 - y1))
            r.eq_diameter = math.sqrt(4.0 * r.area / math.pi) if r.area > 0 else 0.0
            r.surface_y, r.bottom_y = y1, y2
            r.median_width_px = max(0.0, x2 - x1)
            offsets.append(offsets[-1])
            widths_rows.append(b"")

    widths_arr = np.zeros((len(records), max_rows), dtype=np.float32)
    for i, row in enumerate(widths_rows):
        if row:
            vals = np.frombuffer(row, dtype=np.float32)
            widths_arr[i, :len(vals)] = vals

    ms = MaskSet(records=records, cache_key=str(cache_key), text_sha=text_sha,
                 _packed=b"".join(packed_parts),
                 _offsets=np.asarray(offsets, dtype=np.int64),
                 _widths=widths_arr, _contours=contours,
                 _text=text, _data_dir=str(data_dir))
    if use_cache:
        try:
            _save_npz(cache_path, ms)
        except Exception:
            pass
        if contours:
            try:
                _save_contours(_contour_cache_path(str(data_dir), cache_key), text_sha, contours)
            except Exception:
                pass
    return ms


def _save_npz(path: Path, ms: MaskSet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    recs = ms.records
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        text_sha=np.asarray(ms.text_sha),
        object_id=np.asarray([r.object_id for r in recs]),
        frame=np.asarray([r.frame for r in recs], dtype=np.int32),
        confidence=np.asarray([r.confidence for r in recs], dtype=np.float32),
        bounds=np.asarray([r.bounds for r in recs], dtype=np.float64).reshape(-1, 4),
        box=np.asarray([r.box for r in recs], dtype=np.float64).reshape(-1, 4),
        raster_hw=np.asarray([[r.raster_height, r.raster_width] for r in recs], dtype=np.int32).reshape(-1, 2),
        has_mask=np.asarray([r.has_mask for r in recs], dtype=bool),
        area=np.asarray([r.area for r in recs], dtype=np.float64),
        centroid=np.asarray([[r.centroid_x, r.centroid_y] for r in recs], dtype=np.float64).reshape(-1, 2),
        eq_diameter=np.asarray([r.eq_diameter for r in recs], dtype=np.float64),
        surface_y=np.asarray([r.surface_y for r in recs], dtype=np.float64),
        bottom_y=np.asarray([r.bottom_y for r in recs], dtype=np.float64),
        median_width=np.asarray([r.median_width_px for r in recs], dtype=np.float64),
        offsets=ms._offsets,
        packed=np.frombuffer(ms._packed, dtype=np.uint8),
        widths=ms._widths,
    )
    tmp.replace(path)


def _load_npz(path: Path, cache_key: str, data_dir: str) -> MaskSet:
    with np.load(path, allow_pickle=False) as z:
        # Read each member once: NpzFile.__getitem__ re-decompresses the
        # whole array on every access, so per-record z[...] is O(n^2).
        text_sha = str(z["text_sha"])
        object_id = z["object_id"]
        frame = z["frame"]
        confidence = z["confidence"]
        bounds = z["bounds"]
        box = z["box"]
        raster_hw = z["raster_hw"]
        has_mask = z["has_mask"]
        area = z["area"]
        centroid = z["centroid"]
        eq_diameter = z["eq_diameter"]
        surface_y = z["surface_y"]
        bottom_y = z["bottom_y"]
        median_width = z["median_width"]
        packed = z["packed"].tobytes()
        offsets = np.asarray(z["offsets"], dtype=np.int64)
        widths = np.asarray(z["widths"], dtype=np.float32)
    records = []
    for i in range(len(frame)):
        records.append(MaskRecord(
            index=i, object_id=str(object_id[i]), frame=int(frame[i]),
            confidence=float(confidence[i]),
            bounds=tuple(float(v) for v in bounds[i]),
            box=tuple(float(v) for v in box[i]),
            raster_height=int(raster_hw[i][0]), raster_width=int(raster_hw[i][1]),
            has_mask=bool(has_mask[i]), area=float(area[i]),
            centroid_x=float(centroid[i][0]), centroid_y=float(centroid[i][1]),
            eq_diameter=float(eq_diameter[i]),
            surface_y=float(surface_y[i]), bottom_y=float(bottom_y[i]),
            median_width_px=float(median_width[i])))
    return MaskSet(records=records, cache_key=cache_key, text_sha=text_sha,
                   _packed=packed, _offsets=offsets, _widths=widths,
                   _data_dir=data_dir)


def decode_mask_payloads(payloads, *, max_workers: int | None = None) -> list:
    """Decode raw mask payloads (e.g. from image segmentation).

    ``payloads``: list of (mask, bounds_mask, bounds_parse). Returns a list
    of (raster, stats dict) in input order. Used where a full MaskSet is
    unnecessary (single images with a handful of masks).
    """
    jobs = [(i, m, bm, bp, False) for i, (m, bm, bp) in enumerate(payloads)]
    out: list = [None] * len(jobs)
    for (idx, packed, h, w, area, cx, cy, eq, surf, bot, med_w, _widths, _c) in _run_parallel(_decode_chunk, jobs, max_workers):
        if packed is None:
            out[idx] = (np.zeros((h, w), dtype=np.uint8),
                        {"area": 0.0, "centroid": (0.0, 0.0), "eq_diameter": 0.0,
                         "surface_y": surf, "bottom_y": bot, "median_width_px": med_w})
            continue
        raster = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=h * w).reshape(h, w)
        out[idx] = (raster, {"area": area, "centroid": (cx, cy), "eq_diameter": eq,
                             "surface_y": surf, "bottom_y": bot, "median_width_px": med_w})
    return out
