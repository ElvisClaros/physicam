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
    """Scale factor with honest relative uncertainty."""

    m_per_px: float
    rel_uncertainty: float
    method: str  # reference | track_size | gravity | none
    details: str = ""

    def to_dict(self) -> dict:
        return {"m_per_px": self.m_per_px, "rel_uncertainty": self.rel_uncertainty,
                "method": self.method, "details": self.details}

    def convert(self, value_px: float, stderr_px: float = 0.0) -> tuple[float, float]:
        """Convert a pixel-space value (+stderr) to metres, combining uncertainties."""
        v = value_px * self.m_per_px
        rel = math.hypot(stderr_px / value_px if value_px else 0.0, self.rel_uncertainty) if value_px else self.rel_uncertainty
        return v, abs(v) * rel if v else stderr_px * self.m_per_px


def none() -> Calibration:
    """No calibration: results stay in pixel units."""
    return Calibration(m_per_px=1.0, rel_uncertainty=0.0, method="none",
                       details="no calibration; values in pixel units")


def from_track_size(track: "Track", real_diameter_m: float) -> Calibration:
    """Calibrate from a tracked object's known diameter (median eq. diameter)."""
    d = np.asarray(track.eq_diameter, dtype=float)
    d = d[d > 0]
    if d.size == 0:
        raise ValueError("track has no positive equivalent diameters")
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) * 1.4826 if d.size > 1 else 0.0
    spread_rel = (mad / med) if med else 0.0
    rel = max(0.03, spread_rel)
    return Calibration(
        m_per_px=real_diameter_m / med, rel_uncertainty=rel, method="track_size",
        details=f"median eq_diameter={med:.2f} px, MAD-based spread={spread_rel:.3f}, "
                f"assumed >=3% mask-edge bias; real_diameter={real_diameter_m} m")


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
