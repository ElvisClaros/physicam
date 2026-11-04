"""Build ``web/mock/result.json`` (and ``clip.mp4``) from REAL SAM output.

Reads the committed fixture ``tests/fixtures/bounce240_red_ball_f0-199.sam.txt.gz``
(real SAM 3.1 output, 240 fps, 1280x720) with the official ``meta_sam_parser``,
computes mask-weighted sub-pixel centroids and frame-coordinate contours, runs a
simple numpy bounce analysis (impact at the max-y frame, parabolas before and
after it) and writes a result JSON in the exact contract shape from AGENTS.md.

Run from the repo root:  .venv/bin/python web/mock/make_mock.py
No network, no API key needed.
"""

from __future__ import annotations

import gzip
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
from meta_sam_parser import (
    CompletedOutcome,
    decode_mask_to_raster,
    decode_mask_to_svg_path,
    video_segmentation_format,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "bounce240_red_ball_f0-199.sam.txt.gz"
CLIP_SRC = ROOT / "_probe" / "real" / "bounce_clip.mp4"
OUT_DIR = Path(__file__).resolve().parent

FPS = 240.0
FRAME_W, FRAME_H = 1280, 720
BALL_DIAMETER_M = 0.080  # known fact used for the track-size calibration
TRACK_COLOR = "#ff4d4f"


def _transform_path(d: str, left: float, top: float, sx: float, sy: float) -> str:
    """Map an SVG path from raster coords to frame coords.

    ``decode_mask_to_svg_path`` emits paths in raster pixel coordinates; the
    raster maps onto the mask bounds rect in frame pixels (see AGENTS.md).
    Only absolute M/L/Z commands are produced by the parser.
    """
    tokens = re.findall(r"[MLZ]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    out: list[str] = []
    i = 0
    cmd = ""
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("M", "L", "Z"):
            cmd = tok
            out.append(tok)
            i += 1
            if tok == "Z":
                cmd = ""
            continue
        # a coordinate pair for the current M/L command
        x = float(tok)
        y = float(tokens[i + 1])
        out.append(f"{left + x * sx:.1f} {top + y * sy:.1f}")
        i += 2
        if cmd == "M":
            cmd = "L"  # subsequent pairs are implicit lineto
    return _join_path(out)


def _join_path(parts: list[str]) -> str:
    s = ""
    for p in parts:
        if p in ("M", "L", "Z"):
            s += p
        else:
            s += p + " "
    return s.strip().replace("  ", " ")


def parse_fixture() -> dict:
    """Parse the SAM fixture into per-frame centroid/contour records."""
    text = gzip.open(FIXTURE, "rt", encoding="utf-8").read()
    parser = video_segmentation_format().create_parser()
    parser.push(text if text.endswith("\n") else text + "\n")
    result = parser.finish(CompletedOutcome()).result

    frames: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    areas: list[float] = []
    bboxes: list[list[float]] = []
    contours: dict[str, str] = {}

    for rec in result.records:
        if rec.kind != "mask":
            continue
        m, b = rec.mask, rec.bounds
        raster = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width)
        yy, xx = np.nonzero(raster)
        if len(xx) == 0:
            continue
        sx = (b.right - b.left) / m.width
        sy = (b.bottom - b.top) / m.height
        cx = float(b.left + (xx.mean() + 0.5) * sx)
        cy = float(b.top + (yy.mean() + 0.5) * sy)
        fi = int(rec.frame.frame_index)
        frames.append(fi)
        xs.append(cx)
        ys.append(cy)
        # raster area scaled to frame pixels
        areas.append(float(len(xx) * sx * sy))
        bboxes.append([float(b.left), float(b.top), float(b.right), float(b.bottom)])
        if fi % 5 == 0:
            contours[str(fi)] = _transform_path(
                decode_mask_to_svg_path(m), float(b.left), float(b.top), sx, sy
            )

    order = np.argsort(frames)
    return {
        "frames": [frames[i] for i in order],
        "x": [xs[i] for i in order],
        "y": [ys[i] for i in order],
        "area": [areas[i] for i in order],
        "bbox": [bboxes[i] for i in order],
        "contours": contours,
    }


def _parabola(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit y = a t^2 + b t + c; return (coeffs, stderr_of_coeffs, r2)."""
    coeffs, cov = np.polyfit(t, y, 2, cov=True)
    pred = np.polyval(coeffs, t)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    stderr = np.sqrt(np.clip(np.diag(cov), 0, None))
    return coeffs, stderr, r2


def bounce_analysis(track: dict, m_per_px: float, rel_cal: float) -> dict:
    """Impact at max-y frame; parabola per flight segment; pooled g + e."""
    t = np.array(track["frames"], dtype=float) / FPS
    y = np.array(track["y"], dtype=float)
    x = np.array(track["x"], dtype=float)
    imp = int(np.argmax(y))
    t_imp = float(t[imp])

    # exclude 2 frames either side of contact (mask squashes at impact)
    seg_defs = [("fall", 0, max(imp - 2, 3)), ("rebound", min(imp + 3, len(t) - 3), len(t))]
    segments = []
    g_vals: list[tuple[float, float]] = []  # (g_px_s2, weight)
    v_before = v_after = float("nan")
    warnings: list[str] = []

    for name, a, b in seg_defs:
        coeffs, stderr, r2 = _parabola(t[a:b], y[a:b])
        g_px = float(2 * coeffs[0])
        tt = np.linspace(float(t[a]), float(t[b - 1]), 60)
        curve = {"t": [round(float(v), 5) for v in tt],
                 "y": [round(float(v), 2) for v in np.polyval(coeffs, tt)]}
        resid = y[a:b] - np.polyval(coeffs, t[a:b])
        rmse = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else 0.0
        segments.append({
            "name": name, "t_range": [round(float(t[a]), 5), round(float(t[b - 1]), 5)],
            "g_px_s2": round(g_px, 1), "r2": round(float(r2), 4), "rmse_px": round(rmse, 2),
            "curve": curve,
        })
        if g_px > 0:
            g_vals.append((g_px, max(b - a, 1)))
        v_at_imp = float(2 * coeffs[0] * t_imp + coeffs[1])
        if name == "fall":
            v_before = v_at_imp
        else:
            v_after = v_at_imp
        if r2 < 0.98:
            warnings.append(f"{name} segment is poorly parabolic (R² = {r2:.3f}); "
                            "drag or a driven impact may be present")
        if g_px <= 0:
            warnings.append(f"{name} segment curvature is non-positive (g_px = {g_px:.0f} px/s²); "
                            "it is excluded from the pooled g")

    # g from the rebound segment alone is the cleanest here (fall decelerates);
    # fall back to the point-weighted mean of positive-g segments.
    rebound_g = next((s["g_px_s2"] for s in segments if s["name"] == "rebound"), None)
    if rebound_g and rebound_g > 0:
        g_px = float(rebound_g)
        g_seg = next(s for s in segments if s["name"] == "rebound")
        # stderr of g from the rebound fit, recomputed for the error budget
        a, b = seg_defs[1][1], seg_defs[1][2]
        _, stderr, _ = _parabola(t[a:b], y[a:b])
        g_stderr_px = float(2 * stderr[0])
    elif g_vals:
        g_px = float(np.average([g for g, _ in g_vals], weights=[w for _, w in g_vals]))
        g_stderr_px = 0.05 * g_px
        g_seg = segments[-1]
    else:
        g_px, g_stderr_px, g_seg = 0.0, 0.0, segments[-1]

    g = g_px * m_per_px
    g_stderr = math.hypot(g_stderr_px * m_per_px, g * rel_cal)

    e = abs(v_after / v_before) if v_before and math.isfinite(v_before) and v_before != 0 else float("nan")
    e_stderr = 0.04 * e if math.isfinite(e) else 0.0
    if math.isfinite(e) and e > 1.0:
        warnings.append("fitted e > 1: the pre-impact flight is not a clean passive fall, "
                        "so e is an upper bound, not a passive restitution")

    # full-range fitted curve (piecewise, from the two segment parabolas)
    ct: list[float] = []
    cy: list[float] = []
    for s in segments:
        ct.extend(s["curve"]["t"])
        cy.extend(s["curve"]["y"])

    params = {
        "g": {"value": round(g, 3), "stderr": round(g_stderr, 3), "unit": "m/s^2"},
    }
    if math.isfinite(e):
        params["e"] = {"value": round(float(e), 3), "stderr": round(float(e_stderr), 3), "unit": "1"}
    params["v_impact"] = {"value": round(abs(v_before) * m_per_px, 3),
                           "stderr": round(0.05 * abs(v_before) * m_per_px, 3), "unit": "m/s"}

    return {
        "id": "f0", "track_id": "t0", "model": "bounce",
        "t_range": [round(float(t[0]), 5), round(float(t[-1]), 5)],
        "params": params,
        "r2": g_seg["r2"], "rmse_px": g_seg["rmse_px"], "n_points": int(len(t)),
        "curve": {"t": ct, "y": cy},
        "segments": segments,
        "impacts": [{
            "index": 1, "t": round(t_imp, 5), "frame": int(track["frames"][imp]),
            "x": round(float(x[imp]), 1), "y": round(float(y[imp]), 1),
            "e": round(float(e), 3) if math.isfinite(e) else None,
            "e_stderr": round(float(e_stderr), 3),
        }],
        "warnings": warnings,
    }


def main() -> None:
    """Generate result.json and copy the demo clip next to it."""
    track_data = parse_fixture()
    t = np.array(track_data["frames"], dtype=float) / FPS

    # calibration from the tracked ball's known diameter (median equiv. diameter)
    areas = np.array(track_data["area"])
    equiv_diam_px = float(np.median(2.0 * np.sqrt(areas / np.pi)))
    m_per_px = BALL_DIAMETER_M / equiv_diam_px
    rel_cal = 0.03

    fit = bounce_analysis(track_data, m_per_px, rel_cal)
    g = fit["params"]["g"]["value"]
    g_err = fit["params"]["g"]["stderr"]
    e_val = fit["params"].get("e", {}).get("value")

    track = {
        "id": "t0", "label": "red ball", "color": TRACK_COLOR,
        "frames": track_data["frames"],
        "t": [round(float(v), 5) for v in t],
        "x": [round(float(v), 2) for v in track_data["x"]],
        "y": [round(float(v), 2) for v in track_data["y"]],
        "area": [round(float(v), 1) for v in track_data["area"]],
        "bbox": [[round(float(c), 1) for c in bb] for bb in track_data["bbox"]],
        "contours": track_data["contours"],
    }

    headline = f"g = {g:.2f} ± {g_err:.2f} m/s²"
    report = (
        "# Bouncing-ball measurement\n\n"
        "## Method\n\n"
        "The **red ball** was tracked with SAM 3.1 at 240 fps and calibrated from its "
        f"known diameter of {BALL_DIAMETER_M * 100:.0f} cm "
        f"({m_per_px * 1000:.2f} mm/px). The impact frame was located at the maximum "
        "centroid height in image coordinates, and a parabola was fitted to each flight segment.\n\n"
        "## Results\n\n"
        "| Quantity | Value | Unit |\n| --- | --- | --- |\n"
        f"| g (rebound segment) | {g:.2f} ± {g_err:.2f} | m/s² |\n"
        + (f"| restitution e (impact 1) | {e_val:.2f} | 1 |\n" if e_val is not None else "")
        + "\n## Notes\n\n"
        "- The pre-impact flight decelerates in image space, so g is reported from the "
        "rebound segment and the restitution estimate is flagged as an upper bound.\n"
        "- Uncertainties combine the parabola covariance with the calibration uncertainty.\n"
    )

    result = {
        "id": "mock-bounce-240",
        "status": "done",
        "question": "What is g?",
        "video": {
            "url": "mock/clip.mp4", "fps": FPS, "width": FRAME_W, "height": FRAME_H,
            "n_frames": int(len(t)), "duration_s": round(float(t[-1]), 4),
        },
        "plan": {
            "experiment": "Free-flight segments of a bouncing ball; fit a parabola per segment.",
            "objects": [{"sam_prompt": "red ball", "role": "bouncing ball"}],
            "model": "bounce",
            "reference": {"kind": "track_size", "diameter_m": BALL_DIAMETER_M},
            "notes": "Generated offline from the real SAM fixture by web/mock/make_mock.py.",
        },
        "calibration": {
            "m_per_px": round(m_per_px, 7), "rel_uncertainty": rel_cal,
            "method": "track_size",
            "details": f"Ball diameter {BALL_DIAMETER_M:.3f} m = {equiv_diam_px:.1f} px "
                       "(median equivalent diameter).",
        },
        "tracks": [track],
        "fits": [fit],
        "answer": {
            "headline": headline, "value": g, "unit": "m/s^2",
            "uncertainty": g_err,
            "summary": "Gravity from the rebound parabola, calibrated by the ball diameter. "
                       "The fall segment is not cleanly parabolic, so restitution is an upper bound.",
        },
        "report_markdown": report,
        "trace": [
            {"step": 1, "kind": "plan", "name": "watch_video", "args": {"question": "What is g?"},
             "summary": "Muse Spark planned a bounce analysis of the red ball", "ms": 8200},
            {"step": 2, "kind": "tool", "name": "track_objects", "args": {"prompt": "red ball"},
             "summary": f"SAM 3.1 tracked the red ball in {len(t)} frames", "ms": 41000},
            {"step": 3, "kind": "tool", "name": "calibrate_from_track",
             "args": {"track_id": "t0", "real_diameter_m": BALL_DIAMETER_M},
             "summary": f"Calibration {m_per_px * 1000:.2f} mm/px from the ball diameter", "ms": 120},
            {"step": 4, "kind": "tool", "name": "fit_model",
             "args": {"track_id": "t0", "model": "bounce"},
             "summary": f"Bounce fit: {headline}", "ms": 340},
            {"step": 5, "kind": "answer", "name": "finish",
             "args": {"value": g, "unit": "m/s^2"},
             "summary": headline, "ms": 900},
            {"step": 6, "kind": "report", "name": "write_report", "args": {},
             "summary": "Muse Wonder wrote the lab report", "ms": 5200},
        ],
        "cost": {"sam_frames": int(len(t)), "spark_tokens": 8420,
                 "usd_estimate": round(len(t) * 0.20 / 1000 + 0.03, 3)},
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    if CLIP_SRC.exists():
        shutil.copyfile(CLIP_SRC, OUT_DIR / "clip.mp4")
        print(f"copied {CLIP_SRC} -> {OUT_DIR / 'clip.mp4'}")
    else:
        print(f"WARNING: {CLIP_SRC} missing; clip.mp4 not written (video will 404 in mock mode)")
    print(f"wrote {OUT_DIR / 'result.json'}: {len(t)} frames, {headline}, "
          f"e={e_val}, contours={len(track['contours'])}")


if __name__ == "__main__":
    main()
