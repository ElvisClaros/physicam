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


def parse_sam_text(text: str, fps: float, label: str = "") -> list[Track]:
    """Pure parser: SAM output text + fps -> tracks (unstiched, raw SAM ids)."""
    from meta_sam_parser import CompletedOutcome, video_segmentation_format, decode_mask_to_raster, decode_mask_to_svg_path

    if not text or not text.strip():
        return []
    parser = video_segmentation_format().create_parser()
    parser.push(text if text.endswith("\n") else text + "\n")
    result = parser.finish(CompletedOutcome()).result

    boxes: dict[tuple[str, int], tuple[float, float, float, float, float]] = {}
    masks: dict[tuple[str, int], object] = {}
    for rec in result.records:
        if rec.kind == "box":
            boxes[(str(rec.object_id), int(rec.frame.frame_index))] = (
                float(rec.left), float(rec.top), float(rec.right), float(rec.bottom),
                float(rec.confidence or 0.0))
        elif rec.kind == "mask":
            masks[(str(rec.object_id), int(rec.frame.frame_index))] = rec

    obj_ids = sorted({oid for oid, _ in list(boxes.keys()) + list(masks.keys())},
                     key=lambda s: (len(s), s))
    tracks: list[Track] = []
    for oid in obj_ids:
        frame_list = sorted({f for (o, f) in list(boxes.keys()) + list(masks.keys()) if o == oid})
        rows = []
        svg_by_frame: dict[int, str] = {}
        for f in frame_list:
            rec = masks.get((oid, f))
            box = boxes.get((oid, f))
            if rec is None and box is None:
                continue
            if box is not None:
                x1, y1, x2, y2, conf = box
            else:
                b = rec.bounds  # type: ignore[union-attr]
                x1, y1, x2, y2 = float(b.left), float(b.top), float(b.right), float(b.bottom)
                conf = float(rec.confidence or 0.0)  # type: ignore[union-attr]
            if rec is not None:
                m = rec.mask  # type: ignore[union-attr]
                b = rec.bounds  # type: ignore[union-attr]
                raster = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width)
                sx = (x2 - x1) / m.width if m.width else 0.0
                sy = (y2 - y1) / m.height if m.height else 0.0
                count = float(raster.sum())
                if count > 0:
                    jj, ii = np.meshgrid(np.arange(m.width), np.arange(m.height))
                    cx = x1 + float((raster * (jj + 0.5)).sum() / count) * sx
                    cy = y1 + float((raster * (ii + 0.5)).sum() / count) * sy
                    area = count * sx * sy
                else:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    area = 0.0
                try:
                    svg_by_frame[f] = raster_path_to_frame(
                        decode_mask_to_svg_path(m), x1, y1, x2 - x1, y2 - y1, m.width, m.height)
                except Exception:
                    pass
            else:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                area = max(0.0, (x2 - x1) * (y2 - y1))
            rows.append((f, cx, cy, area, x1, y1, x2, y2, conf))
        if not rows:
            continue
        rows.sort()
        frames = np.array([r[0] for r in rows], dtype=int)
        # Contour subsampling: keep every k-th so total <= MAX_CONTOURS.
        stride = max(1, math.ceil(len(rows) / MAX_CONTOURS))
        contours = {int(fr): svg_by_frame[fr] for fr in frames[::stride] if fr in svg_by_frame}
        area_arr = np.array([r[3] for r in rows], dtype=float)
        tracks.append(Track(
            id=f"t{len(tracks)}", label=label or oid,
            frames=frames, t=frames.astype(float) / fps,
            x=np.array([r[1] for r in rows]), y=np.array([r[2] for r in rows]),
            area=area_arr,
            bbox=np.array([[r[4], r[5], r[6], r[7]] for r in rows], dtype=float),
            conf=np.array([r[8] for r in rows], dtype=float),
            eq_diameter=np.sqrt(4.0 * np.maximum(area_arr, 0.0) / math.pi),
            contours=contours, fps=float(fps),
        ))
        # Keep raw SAM id available for stitching determinism via label fallback.
        tracks[-1]._sam_id = oid  # type: ignore[attr-defined]
    return tracks


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


def segment_video(client: "MetaClient", video_path: str | Path, prompt: str,
                  fps: float, *, cache: bool = True) -> list[Track]:
    """Upload, call SAM 3.1, parse, stitch. Raw text cached for free re-runs."""
    from physicam.video import sha256_file

    video_path = Path(video_path)
    sha = sha256_file(video_path)
    cache_path = _sam_cache_path(client.settings.data_dir, sha, prompt)
    text: str | None = None
    if cache and cache_path.exists():
        text = cache_path.read_text()
    if text is None:
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
    raw = parse_sam_text(text, fps, label=prompt)
    return stitch_tracks(raw, fps)


def segment_image(client: "MetaClient", pil_image: Image.Image, prompt: str) -> list[dict]:
    """Segment one image; return boxes, masks info, axis lengths in frame px."""
    import base64
    import io

    from meta_sam_parser import CompletedOutcome, image_segmentation_format, decode_mask_to_raster

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
        item: dict = {"object_id": oid, "bbox": [x1, y1, x2, y2], "conf": conf,
                      "width_px": x2 - x1, "height_px": y2 - y1, "area_px": 0.0,
                      "major_axis_px": max(x2 - x1, y2 - y1), "minor_axis_px": min(x2 - x1, y2 - y1),
                      "longest_px": max(x2 - x1, y2 - y1)}
        if rec is not None:
            m = rec.mask  # type: ignore[union-attr]
            raster = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width).astype(float)
            sx = (x2 - x1) / m.width if m.width else 0.0
            sy = (y2 - y1) / m.height if m.height else 0.0
            count = float(raster.sum())
            if count > 0:
                jj, ii = np.meshgrid(np.arange(m.width), np.arange(m.height))
                cxr = float((raster * jj).sum() / count)
                cyr = float((raster * ii).sum() / count)
                # Second central moments in raster units -> frame px.
                mu20 = float((raster * (jj - cxr) ** 2).sum() / count) * sx * sx
                mu02 = float((raster * (ii - cyr) ** 2).sum() / count) * sy * sy
                mu11 = float((raster * (jj - cxr) * (ii - cyr)).sum() / count) * sx * sy
                cov = np.array([[mu20, mu11], [mu11, mu02]])
                eig = np.linalg.eigvalsh(cov)
                # Axis lengths of the equivalent ellipse (full lengths).
                major = 4.0 * math.sqrt(max(float(eig[1]), 0.0))
                minor = 4.0 * math.sqrt(max(float(eig[0]), 0.0))
                item.update({
                    "area_px": count * sx * sy,
                    "centroid": [x1 + (cxr + 0.5) * sx, y1 + (cyr + 0.5) * sy],
                    "major_axis_px": major, "minor_axis_px": minor,
                    "longest_px": max(major, x2 - x1, y2 - y1),
                    "eq_diameter_px": math.sqrt(4.0 * count * sx * sy / math.pi),
                })
        out.append(item)
    return out
