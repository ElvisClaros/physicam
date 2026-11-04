"""Draining (Torricelli) tests: synthetic Cd recovery, generalized vessel,
and synthetic-mask level extraction. No network."""

from __future__ import annotations

import math

import numpy as np

from physicam.calibrate import Calibration
from physicam.level import level_series_from_rasters
from physicam.physics import (
    fit_draining_arrays,
    fit_generalized_vessel,
    solve_generalized_vessel,
    torricelli_tf_theory,
)

D_M, D_HOLE, G = 0.075, 0.005, 9.81


def _synthetic_torricelli(Cd=0.62, h0=0.0586, plateau=1.0, n=600, noise=0.0002, seed=1):
    r = (D_HOLE / D_M) ** 2
    k = Cd * r * math.sqrt(G / (2 * (1 - r**2)))
    tf = math.sqrt(h0) / k
    t = np.linspace(0, tf * 0.95 + plateau, n)
    dt = np.clip(t - plateau, 0, None)
    h = (np.sqrt(h0) - k * dt) ** 2
    h[t < plateau] = h0
    rng = np.random.default_rng(seed)
    return t, h + rng.normal(0, noise, len(t)), tf


def test_synthetic_torricelli_recovers_cd_within_1pct():
    t, h, _tf = _synthetic_torricelli()
    cal = Calibration(m_per_px=0.00036, rel_uncertainty=0.01, method="reference", details="")
    fit = fit_draining_arrays(t, h, cal, {"hole_diameter_m": D_HOLE, "container_diameter_m": D_M,
                                          "sigma_d_m": 0.0, "sigma_D_m": 0.0})
    # Zero d/D errors here isolate the fit; with defaults Cd err is ~5%.
    assert fit.params["Cd"]["value"] is not None
    assert abs(fit.params["Cd"]["value"] - 0.62) / 0.62 < 0.01
    # Theory matches the course formula and Cd cross-check.
    theory = torricelli_tf_theory(float(fit.params["h0"]["value"]), D_M, D_HOLE)
    assert abs(fit.params["t_f_theory"]["value"] - theory) < 1e-9
    # Analytic vs cubic-Newton times agree closely for clean Torricelli data.
    for frac in (10, 50, 90):
        a = fit.info["times_torricelli"][f"t{frac}"]
        b = fit.info["times_cubic_newton"][f"t{frac}"]
        assert a is not None and b is not None
        assert abs(a - b) < 0.5
    for name in ("quadratic", "cubic", "exponential"):
        assert name in fit.info["empirical"]


def test_generalized_vessel_agrees_with_closed_form_cylinder():
    h0, Cd = 0.0586, 0.62
    t = np.linspace(0, 40, 200)
    hg = solve_generalized_vessel(t, h0, Cd, lambda hv: D_M, D_HOLE, g=G)
    # Closed form of A dh/dt = -Cd a sqrt(2gh) for constant A.
    r = (D_HOLE / D_M) ** 2
    k2 = Cd * r * math.sqrt(2 * G) / 2.0
    hc = np.where(t < math.sqrt(h0) / k2, (np.sqrt(h0) - k2 * t) ** 2, 0.0)
    assert np.max(np.abs(hg - hc)) < 1e-6
    gen = fit_generalized_vessel(t, hg, lambda hv: D_M, D_HOLE, g=G)
    assert abs(gen["Cd"] - Cd) / Cd < 0.01


def _draining_rasters(n=120, fps=10.0):
    """Draining rectangle (raster = frame coords) + meniscus bump + jet blob."""
    rasters = {}
    for f in range(n):
        tt = f / fps
        # Surface raster row: drains from 40 to 150 (hole), then steady.
        surf = int(min(40 + tt * 10, 150))
        raster = np.zeros((220, 120), dtype=np.uint8)
        raster[surf:210, 20:100] = 1  # liquid body, bottom row 210
        if surf > 4:
            raster[surf - 3:surf, 50:70] = 1  # meniscus bump (narrow)
        # Exit-jet blob: narrow detached rows below the body would be
        # outside; instead a narrow blob overlapping the bottom rows.
        raster[205:215, 55:65] = 1
        rasters[("1", f)] = (raster, (0.0, 0.0, 120.0, 220.0), 0.9)
    return rasters, fps


def test_level_extraction_synthetic_mask_sequence():
    rasters, fps = _draining_rasters()
    level = level_series_from_rasters(rasters, fps, "brown liquid")
    assert level.n == 120
    # Bottom is the fixed median bottom (~210-220 frame px).
    assert 205 <= level.bottom_y <= 220
    # Surface falls monotonically (y increases) despite meniscus/jet.
    d = np.diff(level.surface_y)
    assert np.mean(d >= -0.6) > 0.95
    # Hole estimated from the final steady surface; h -> ~0 at the end.
    assert level.hole_estimated
    assert abs(level.h_px[-1]) < 3.0
    assert level.h_px[0] > 80
    # CSV contract.
    lines = level.to_csv().splitlines()
    assert lines[0] == "t_s,h_px,h_m,surface_y_px"
    assert len(lines) == level.n + 1
    js = level.to_json()
    assert js["bottom_y"] == level.bottom_y
    assert len(js["t"]) == level.n


def test_validation_against_manual_csv(tmp_path):
    # Build a fake manual table from the synthetic Torricelli curve itself.
    import csv as _csv
    t, h, _ = _synthetic_torricelli(noise=0.0)
    slices = np.linspace(1, len(t), 27).astype(int)
    path = tmp_path / "ref.csv"
    with open(path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["Slice", "Length", "Tiempo_s"])
        for s in slices:
            w.writerow([int(s), float(h[s - 1] * 100), float(t[s - 1])])
    cal = Calibration(m_per_px=0.001, rel_uncertainty=0.01, method="reference", details="")
    fit = fit_draining_arrays(t, h, cal, {"hole_diameter_m": D_HOLE, "container_diameter_m": D_M,
                                          "reference_csv": str(path),
                                          "_frames": np.arange(len(t), dtype=float)})
    val = fit.info["validation"]
    assert val["n"] == 27
    assert val["rms_norm"] < 0.01
    assert val["rms_cm"] is not None and val["rms_cm"] < 0.1
    assert "t50" in val["times_ref"]
