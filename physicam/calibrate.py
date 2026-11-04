"""Pixel-to-metre calibration from a reference, a track size, gravity, or none."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from physicam.meta_client import MetaClient
    from physicam.sam import Track


@dataclass(frozen=True)
class Calibration:
    """Scale factor with honest relative uncertainty.

    ``kind="constant"`` (default): single ``m_per_px``.
    ``kind="per_frame"``: time-varying scale ``s(t)=D/d_px(t)`` for
    perspective correction; ``t_series``/``s_series``/``d_series`` hold the
    smoothed series, ``(cx, cy)`` is the principal point (image centre).
    """

    m_per_px: float
    rel_uncertainty: float
    method: str  # reference | track_size | gravity | none
    details: str = ""
    kind: str = "constant"  # constant | per_frame
    t_series: tuple[float, ...] | None = None
    s_series: tuple[float, ...] | None = None
    d_series: tuple[float, ...] | None = None
    cx: float = 0.0
    cy: float = 0.0
    real_diameter_m: float | None = None
    frame_width: float | None = None
    frame_height: float | None = None

    def to_dict(self) -> dict:
        d: dict = {"m_per_px": self.m_per_px, "rel_uncertainty": self.rel_uncertainty,
                   "method": self.method, "details": self.details, "kind": self.kind}
        if self.kind == "per_frame":
            d.update({
                "cx": self.cx, "cy": self.cy,
                "real_diameter_m": self.real_diameter_m,
                "frame_width": self.frame_width, "frame_height": self.frame_height,
                "t": [round(float(v), 6) for v in (self.t_series or ())],
                "m_per_px_series": [round(float(v), 8) for v in (self.s_series or ())],
                "diameter_px_smooth": [round(float(v), 3) for v in (self.d_series or ())],
                "depth_ratio": self.depth_ratio_list(),
            })
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            m_per_px=float(d["m_per_px"]), rel_uncertainty=float(d.get("rel_uncertainty", 0.0)),
            method=str(d.get("method", "none")), details=str(d.get("details", "")),
            kind=str(d.get("kind", "constant")),
            t_series=tuple(d["t"]) if d.get("t") else None,
            s_series=tuple(d["m_per_px_series"]) if d.get("m_per_px_series") else None,
            d_series=tuple(d["diameter_px_smooth"]) if d.get("diameter_px_smooth") else None,
            cx=float(d.get("cx", 0.0)), cy=float(d.get("cy", 0.0)),
            real_diameter_m=d.get("real_diameter_m"),
            frame_width=d.get("frame_width"), frame_height=d.get("frame_height"))

    def convert(self, value_px: float, stderr_px: float = 0.0) -> tuple[float, float]:
        """Convert a pixel-space value (+stderr) to metres, combining uncertainties."""
        v = value_px * self.m_per_px
        rel = math.hypot(stderr_px / value_px if value_px else 0.0, self.rel_uncertainty) if value_px else self.rel_uncertainty
        return v, abs(v) * rel if v else stderr_px * self.m_per_px

    def scale_at(self, t) -> np.ndarray:
        """Interpolated ``s(t)`` (m/px) at times *t*; constant scale if not per-frame."""
        tt = np.asarray(t, dtype=float)
        if self.kind != "per_frame" or not self.t_series or not self.s_series:
            return np.full(tt.shape, float(self.m_per_px))
        return np.interp(tt, np.asarray(self.t_series), np.asarray(self.s_series))

    def depth_ratio(self, t) -> np.ndarray:
        """Relative depth ``Z(t)/Z0 = d0/d_px(t)`` with ``d0`` the median smoothed diameter."""
        tt = np.asarray(t, dtype=float)
        if self.kind != "per_frame" or not self.d_series or not self.t_series:
            return np.ones(tt.shape)
        d = np.interp(tt, np.asarray(self.t_series), np.asarray(self.d_series))
        d0 = float(np.median(np.asarray(self.d_series))) or 1.0
        return d0 / np.maximum(d, 1e-9)

    def depth_ratio_list(self) -> list[float]:
        if not self.t_series:
            return []
        return [round(float(v), 5) for v in self.depth_ratio(np.asarray(self.t_series))]

    def to_metric(self, track: "Track") -> tuple[np.ndarray, np.ndarray]:
        """Metric lateral coordinates ``X=(x-cx)s(t)``, ``Y=(y-cy)s(t)`` (metres)."""
        s = self.scale_at(track.t)
        return (np.asarray(track.x, float) - self.cx) * s, (np.asarray(track.y, float) - self.cy) * s


def _apparent_diameters(track: "Track") -> np.ndarray:
    """Apparent diameter per frame from mask area (equivalent diameter).

    The Track stores the mask-area equivalent diameter; a mask-moment minor
    axis would be less motion-blur sensitive but is not stored, and the bbox
    minor axis proved noisier on real footage, so the equivalent diameter
    (explicitly allowed by the spec) is used.
    """
    return np.asarray(track.eq_diameter, dtype=float)


def apparent_size_variation(track: "Track") -> float:
    """Relative spread ``(p90-p10)/median`` of the apparent diameter (0 if unknown)."""
    d = _apparent_diameters(track)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return 0.0
    med = float(np.median(d)) or 1.0
    return float((np.percentile(d, 90) - np.percentile(d, 10)) / med)


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def _smooth_diameter(t: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, float]:
    """Robust local smoothing of *d(t)*; returns (smoothed, residual_rel).

    Median filter (~0.05 s) kills mask outliers, then a Savitzky-Golay
    low-order (quadratic, ~0.4 s) filter tracks the slow perspective trend.
    A single global polynomial was tried first and failed on real footage:
    the depth trend is not global-quadratic, and it left the per-flight g
    spread at 5.8% (local smoothing: <1%).
    """
    good = np.isfinite(d) & (d > 0) & np.isfinite(t)
    tt, dd = t[good], d[good]
    if dd.size == 0:
        raise ValueError("track has no positive apparent diameters")
    if dd.size < 5:
        med = float(np.median(dd))
        return np.full(t.shape, med), 0.0
    if dd.size < 20 or len(t) < 20:
        med = float(np.median(dd))
        smoothed_all = np.full(t.shape, med)
        resid_rel = float(np.median(np.abs(dd - med)) * 1.4826 / (med or 1.0))
        return smoothed_all, resid_rel
    dt = float(np.median(np.diff(tt))) if len(tt) > 1 else 0.01
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.01
    # Work on the full-length array (interpolating any bad samples first).
    full = np.interp(t, tt, dd)
    med_win = max(3, min(15, _odd(int(round(0.05 / dt)))))
    sg_win = max(5, min(_odd(int(round(0.4 / dt))), _odd(len(full) // 2 * 2 - 1) if len(full) > 5 else 5))
    if sg_win >= len(full):
        sg_win = _odd(len(full) - 1) if len(full) % 2 == 0 else len(full)
    try:
        from scipy.signal import medfilt, savgol_filter
        smoothed = medfilt(full, kernel_size=med_win)
        if sg_win >= 5 and len(smoothed) > sg_win:
            smoothed = savgol_filter(smoothed, window_length=sg_win, polyorder=2)
    except Exception:
        smoothed = full
    smoothed = np.maximum(smoothed, 1e-6)
    pred_good = np.interp(tt, t, smoothed)
    resid_rel = float(np.median(np.abs(dd - pred_good)) * 1.4826 / (float(np.median(dd)) or 1.0))
    return smoothed, resid_rel


def none() -> Calibration:
    """No calibration: results stay in pixel units."""
    return Calibration(m_per_px=1.0, rel_uncertainty=0.0, method="none",
                       details="no calibration; values in pixel units")


def from_track_size(track: "Track", real_diameter_m: float, per_frame: bool = False,
                    frame_size: tuple[float, float] | None = None,
                    cx: float | None = None, cy: float | None = None) -> Calibration:
    """Calibrate from a tracked object's known diameter.

    With ``per_frame=True`` returns a time-varying perspective-corrected
    calibration: ``s(t)=D/d_px(t)`` with ``d_px`` robustly smoothed.
    ``frame_size=(width, height)`` sets the principal point to the image
    centre unless ``cx``/``cy`` are given explicitly.
    """
    d_eq = np.asarray(track.eq_diameter, dtype=float)
    d_eq = d_eq[d_eq > 0]
    if d_eq.size == 0:
        raise ValueError("track has no positive equivalent diameters")
    med_eq = float(np.median(d_eq))
    if not per_frame:
        mad = float(np.median(np.abs(d_eq - med_eq))) * 1.4826 if d_eq.size > 1 else 0.0
        spread_rel = (mad / med_eq) if med_eq else 0.0
        rel = max(0.03, spread_rel)
        return Calibration(
            m_per_px=real_diameter_m / med_eq, rel_uncertainty=rel, method="track_size",
            details=f"median eq_diameter={med_eq:.2f} px, MAD-based spread={spread_rel:.3f}, "
                    f"assumed >=3% mask-edge bias; real_diameter={real_diameter_m} m")
    # Per-frame perspective correction.
    t = np.asarray(track.t, dtype=float)
    d_raw = _apparent_diameters(track)
    d_smooth, resid_rel = _smooth_diameter(t, d_raw)
    s_series = float(real_diameter_m) / d_smooth
    med_d = float(np.median(d_smooth)) or med_eq
    if cx is None or cy is None:
        if frame_size is not None:
            cx = float(frame_size[0]) / 2.0 if cx is None else cx
            cy = float(frame_size[1]) / 2.0 if cy is None else cy
        else:
            # No frame size known: fall back to bbox extent centre (documented approximation).
            try:
                bbox = np.asarray(track.bbox, dtype=float)
                cx = float((np.min(bbox[:, 0]) + np.max(bbox[:, 2])) / 2.0) if cx is None else cx
                cy = float((np.min(bbox[:, 1]) + np.max(bbox[:, 3])) / 2.0) if cy is None else cy
            except Exception:
                cx = float(np.median(track.x)) if cx is None else cx
                cy = float(np.median(track.y)) if cy is None else cy
    var = float((np.max(d_smooth) - np.min(d_smooth)) / med_d) if med_d else 0.0
    rel = max(0.03, resid_rel)
    fw = float(frame_size[0]) if frame_size else None
    fh = float(frame_size[1]) if frame_size else None
    return Calibration(
        m_per_px=float(real_diameter_m / med_d), rel_uncertainty=rel, method="track_size",
        details=(f"per-frame perspective correction: D={real_diameter_m} m, "
                 f"d_px smoothed {float(np.min(d_smooth)):.1f}->{float(np.max(d_smooth)):.1f} px "
                 f"(variation {var:.1%}), principal point=({cx:.1f},{cy:.1f}), "
                 f"smoothing residual rel={resid_rel:.3f}"),
        kind="per_frame",
        t_series=tuple(float(v) for v in t),
        s_series=tuple(float(v) for v in s_series),
        d_series=tuple(float(v) for v in d_smooth),
        cx=float(cx), cy=float(cy), real_diameter_m=float(real_diameter_m),
        frame_width=fw, frame_height=fh)


def from_gravity(fit_accel_px: float) -> Calibration:
    """"Gravity as the ruler": set m_per_px so the fitted acceleration equals 9.81."""
    if fit_accel_px == 0:
        raise ValueError("fit_accel_px must be non-zero")
    return Calibration(m_per_px=9.81 / abs(fit_accel_px), rel_uncertainty=0.0,
                       method="gravity",
                       details=f"m_per_px set so |a|={abs(fit_accel_px):.3f} px/s^2 maps to g=9.81 m/s^2; "
                               "absolute scale is assumed, not measured")


def from_reference(client: "MetaClient", video_path, t: float, prompt: str,
                   real_size_m: float, dimension: str = "longest",
                   duration_s: float | None = None) -> Calibration:
    """Calibrate from a reference object of known size segmented on 3 frames.

    *dimension* is one of "longest" | "width" | "height" | "diameter", selecting
    which measured pixel extent corresponds to *real_size_m*. Uncertainty uses
    1 px of edge error per side (2 px total) plus the spread between frames.
    """
    if dimension not in ("longest", "width", "height", "diameter"):
        raise ValueError(f"dimension must be longest|width|height|diameter, got {dimension!r}")
    from physicam.sam import segment_image
    from physicam.video import extract_frame, probe

    if duration_s is None:
        try:
            duration_s = probe(video_path).duration_s
        except Exception:
            duration_s = 0.0
    times = [t] if not duration_s or duration_s <= 0 else [
        max(0.0, t - 0.25 * duration_s), t, min(duration_s, t + 0.25 * duration_s)]
    times = sorted(set(round(x, 3) for x in times))
    sizes: list[float] = []
    for tt in times:
        img = extract_frame(video_path, tt)
        results = segment_image(client, img, prompt)
        if not results:
            continue
        best = max(results, key=lambda r: r.get("conf", 0.0))
        if dimension == "width":
            px = float(best["width_px"])
        elif dimension == "height":
            px = float(best["height_px"])
        elif dimension == "diameter":
            px = float(best.get("eq_diameter_px") or best["longest_px"])
        else:
            px = float(best["longest_px"])
        if px > 0:
            sizes.append(px)
    if not sizes:
        raise RuntimeError(f"reference object {prompt!r} not found in any sampled frame")
    med = float(np.median(sizes))
    spread = float(np.std(sizes)) / med if len(sizes) > 1 and med else 0.0
    edge_rel = 2.0 / med if med else 0.0  # 1 px per side
    rel = math.hypot(edge_rel, spread)
    return Calibration(
        m_per_px=real_size_m / med, rel_uncertainty=rel, method="reference",
        details=f"dimension={dimension}, sizes_px={[round(s,2) for s in sizes]}, median={med:.2f} px, "
                f"edge_rel={edge_rel:.3f}, spread_rel={spread:.3f}, real_size={real_size_m} m")
