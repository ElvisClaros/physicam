"""Synthetic-mask tests for oriented extents and reference calibration (no network)."""

import numpy as np

from physicam.calibrate import aspect_ratio_check, from_reference
from physicam.sam import mask_extents


def _rect_raster(w=400, h=283):
    return np.ones((h, w), dtype=np.uint8)


def test_axis_aligned_rectangle():
    ext = mask_extents(_rect_raster(), (0.0, 0.0, 400.0, 283.0))
    assert abs(ext["long_extent_px"] - 400.0) / 400.0 < 0.005
    assert abs(ext["short_extent_px"] - 283.0) / 283.0 < 0.005
    # Ellipse major overstates a rectangle side by ~15% (diagnostic only).
    assert ext["ellipse_major_px"] > 1.1 * ext["long_extent_px"]
    assert ext["bbox_width_px"] == 400.0 and ext["bbox_height_px"] == 283.0
    assert abs(ext["robust_long_extent_px"] - 400.0) / 400.0 < 0.005
    assert abs(ext["robust_short_extent_px"] - 283.0) / 283.0 < 0.005
    # Percentile numbers are retained only as diagnostics.
    assert "trimmed_long_extent_px" in ext and "trimmed_short_extent_px" in ext


def test_raster_frame_scale_half():
    # Raster is half the frame size: each raster pixel is 2 frame px.
    ext = mask_extents(_rect_raster(), (0.0, 0.0, 800.0, 566.0))
    assert abs(ext["long_extent_px"] - 800.0) / 800.0 < 0.01
    assert abs(ext["short_extent_px"] - 566.0) / 566.0 < 0.01


def test_rotated_rectangle_30deg():
    size = 800
    yy, xx = np.mgrid[0:size, 0:size]
    ang = np.radians(30.0)
    dx = xx + 0.5 - size / 2
    dy = yy + 0.5 - size / 2
    u = dx * np.cos(ang) + dy * np.sin(ang)
    v = -dx * np.sin(ang) + dy * np.cos(ang)
    raster = ((np.abs(u) <= 200.0) & (np.abs(v) <= 141.5)).astype(np.uint8)
    ext = mask_extents(raster, (0.0, 0.0, float(size), float(size)))
    assert abs(ext["long_extent_px"] - 400.0) / 400.0 < 0.01
    assert abs(ext["short_extent_px"] - 283.0) / 283.0 < 0.01
    assert abs(abs(ext["orientation_deg"]) - 30.0) < 2.0


def test_disc():
    yy, xx = np.mgrid[0:200, 0:200]
    raster = (((xx + 0.5 - 100.0) ** 2 + (yy + 0.5 - 100.0) ** 2) <= 60.0 ** 2).astype(np.uint8)
    ext = mask_extents(raster, (0.0, 0.0, 200.0, 200.0))
    assert abs(ext["eq_diameter_px"] - 120.0) / 120.0 < 0.01
    assert abs(ext["long_extent_px"] - 120.0) / 120.0 < 0.01
    assert abs(ext["short_extent_px"] - 120.0) / 120.0 < 0.01
    # Disc must not be ~4% short in the fields from_reference consumes.
    assert abs(ext["robust_long_extent_px"] - 120.0) / 120.0 < 0.005
    assert abs(ext["robust_short_extent_px"] - 120.0) / 120.0 < 0.005


def test_robust_resists_stray_pixel():
    raster = np.zeros((600, 800), dtype=np.uint8)
    raster[100:383, 100:500] = 1
    clean = mask_extents(raster, (0.0, 0.0, 800.0, 600.0))
    raster[10, 790] = 1
    stray = mask_extents(raster, (0.0, 0.0, 800.0, 600.0))
    assert stray["long_extent_px"] > clean["long_extent_px"] * 1.05
    assert abs(stray["robust_long_extent_px"] - clean["robust_long_extent_px"]) / clean["robust_long_extent_px"] < 0.02


def test_stray_3px_blob_50px_away_does_not_change_robust():
    raster = np.zeros((600, 800), dtype=np.uint8)
    raster[100:383, 100:500] = 1  # 400x283 rectangle
    clean = mask_extents(raster, (0.0, 0.0, 800.0, 600.0))
    # 3-pixel blob whose nearest pixel is 50 px from the rectangle edge.
    raster[100:103, 550] = 1
    stray = mask_extents(raster, (0.0, 0.0, 800.0, 600.0))
    assert abs(stray["robust_long_extent_px"] - clean["robust_long_extent_px"]) / clean["robust_long_extent_px"] < 0.005
    assert abs(stray["robust_short_extent_px"] - clean["robust_short_extent_px"]) / clean["robust_short_extent_px"] < 0.005
    assert abs(stray["robust_long_extent_px"] - 400.0) / 400.0 < 0.005


def test_a4_aspect_check_flags_ratio_125():
    warning, discrepancy = aspect_ratio_check("sheet of paper", 1.25)
    assert warning is not None and "perspective_warning" in warning
    assert discrepancy > 0.08
    ok, _ = aspect_ratio_check("sheet of paper", 1.414)
    assert ok is None


def test_from_reference_shortest_and_perspective_warning(monkeypatch):
    import physicam.sam as sam_mod
    import physicam.video as video_mod
    from PIL import Image

    result = {"conf": 0.9, "robust_long_extent_px": 500.0, "robust_short_extent_px": 400.0,
              "long_extent_px": 500.0, "short_extent_px": 400.0,
              "eq_diameter_px": 450.0, "orientation_deg": 90.0,
              "oriented_width_px": 400.0, "oriented_height_px": 500.0,
              "bbox_width_px": 400.0, "bbox_height_px": 500.0}
    monkeypatch.setattr(sam_mod, "segment_image", lambda client, img, prompt: [result])
    monkeypatch.setattr(video_mod, "extract_frame", lambda path, t: Image.new("RGB", (10, 10)))
    cal = from_reference(None, "unused.mp4", 0.0, "sheet of paper", 0.21, "shortest", duration_s=0.0)
    assert abs(cal.m_per_px - 0.21 / 400.0) < 1e-12
    assert "perspective_warning" in cal.details
    # Base edge uncertainty is 2/400=0.005; perspective adds half the discrepancy.
    assert cal.rel_uncertainty > 0.05
    cal_w = from_reference(None, "unused.mp4", 0.0, "sheet of paper", 0.21, "width", duration_s=0.0)
    assert abs(cal_w.m_per_px - 0.21 / 400.0) < 1e-12
    cal_h = from_reference(None, "unused.mp4", 0.0, "sheet of paper", 0.297, "height", duration_s=0.0)
    assert abs(cal_h.m_per_px - 0.297 / 500.0) < 1e-12
