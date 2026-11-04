"""Physics fits on synthetic numeric trajectories (noise sigma=0.7 px, with gaps)."""

import numpy as np
import pytest

from physicam.calibrate import Calibration
from physicam.physics import fit_model
from physicam.sam import Track

rng_global = np.random.default_rng(42)


def make_track(t, x, y, fps=240.0, track_id="t0"):
    t = np.asarray(t, float)
    n = len(t)
    area = np.full(n, 2000.0)
    return Track(id=track_id, label="ball",
                 frames=np.round(t * fps).astype(int), t=t,
                 x=np.asarray(x, float), y=np.asarray(y, float),
                 area=area, bbox=np.zeros((n, 4)),
                 conf=np.full(n, 0.9), eq_diameter=np.sqrt(4 * area / np.pi),
                 contours={}, fps=fps)


def add_noise(v, sigma=0.7, rng=None):
    rng = rng or np.random.default_rng(0)
    return v + rng.normal(0, sigma, size=np.shape(v))


def with_gaps(t, *arrs, drop_frac=0.15, seed=1):
    rng = np.random.default_rng(seed)
    keep = rng.random(len(t)) > drop_frac
    keep[0] = keep[-1] = True
    return (t[keep],) + tuple(a[keep] for a in arrs)


def test_free_fall_recovers_g_within_1pct():
    rng = np.random.default_rng(7)
    fps, g_px = 240.0, 8000.0  # px/s^2
    t = np.arange(0, 120) / fps
    y = 100 + 0 * t + 0.5 * g_px * t ** 2
    x = 300 + 50 * t
    t, x, y = with_gaps(t, add_noise(x, rng=rng), add_noise(y, rng=rng))
    track = make_track(t, x, y, fps)
    calib = Calibration(m_per_px=9.81 / g_px, rel_uncertainty=0.0, method="reference", details="test")
    fit = fit_model(track, "free_fall", calib)
    g = fit.params["g"]["value"]
    assert abs(g - 9.81) / 9.81 < 0.01, f"g={g}"
    assert fit.r2 > 0.99


def _bounce_trajectory(fps=240.0, g_px=8000.0, e_true=0.8, n_flights=4, seed=3):
    """Piecewise parabolas bouncing on floor y_floor with restitution e."""
    rng = np.random.default_rng(seed)
    y_floor, y0, vy = 600.0, 150.0, 0.0
    ts, ys, xs = [], [], []
    t_now, x_now, vx = 0.0, 200.0, 30.0
    y_now = y0
    for _ in range(n_flights):
        # fall from y_now with initial vy to floor, then rebound
        # solve 0.5 g dt^2 + vy dt + (y_now - y_floor) = 0
        disc = vy * vy - 2 * g_px * (y_now - y_floor)
        dt = (-vy + np.sqrt(max(disc, 0.0))) / g_px
        tt = np.arange(0, dt, 1.0 / fps)
        ts.append(t_now + tt)
        ys.append(y_now + vy * tt + 0.5 * g_px * tt ** 2)
        xs.append(x_now + vx * tt)
        t_now += dt
        x_now += vx * dt
        vy = -e_true * (vy + g_px * dt)  # rebound velocity (up = negative)
        y_now = y_floor
        if abs(vy) < 200:
            break
    t = np.concatenate(ts)
    y = np.concatenate(ys)
    x = np.concatenate(xs)
    return t, add_noise(x, rng=rng), add_noise(y, rng=rng)


def test_bounce_recovers_e_and_g():
    fps, g_px, e_true = 240.0, 8000.0, 0.8
    t, x, y = _bounce_trajectory(fps, g_px, e_true)
    t, x, y = with_gaps(t, x, y, drop_frac=0.1, seed=5)
    track = make_track(t, x, y, fps)
    calib = Calibration(m_per_px=9.81 / g_px, rel_uncertainty=0.0, method="reference", details="test")
    fit = fit_model(track, "bounce", calib)
    g = fit.params["g"]["value"]
    assert abs(g - 9.81) / 9.81 < 0.03, f"g={g}, segments={fit.segments}"
    assert len(fit.restitution) >= 2, fit.restitution
    for entry in fit.restitution:
        if "e" in entry:
            assert abs(entry["e"] - e_true) < 0.06, entry


def test_pendulum_recovers_period_within_half_pct():
    rng = np.random.default_rng(11)
    fps = 60.0
    T_true = 1.6
    omega = 2 * np.pi / T_true
    t = np.arange(0, 600) / fps
    x = 500 + 200 * np.exp(-0.02 * t) * np.cos(omega * t)
    y = 400 + 20 * np.exp(-0.02 * t) * np.cos(omega * t)
    t, x, y = with_gaps(t, add_noise(x, rng=rng), add_noise(y, rng=rng), drop_frac=0.1)
    track = make_track(t, x, y, fps)
    fit = fit_model(track, "pendulum", Calibration(0.001, 0.0, "reference", ""))
    T = fit.params["T"]["value"]
    assert abs(T - T_true) / T_true < 0.005, f"T={T}"


def test_constant_velocity():
    rng = np.random.default_rng(13)
    fps = 60.0
    t = np.arange(0, 120) / fps
    x = 100 + 120 * t
    y = 500 - 30 * t
    t, x, y = with_gaps(t, add_noise(x, rng=rng), add_noise(y, rng=rng))
    track = make_track(t, x, y, fps)
    calib = Calibration(m_per_px=0.01, rel_uncertainty=0.0, method="reference", details="")
    fit = fit_model(track, "constant_velocity", calib)
    speed_m = fit.params["speed"]["value"]
    assert abs(speed_m - np.hypot(120, 30) * 0.01) < 0.05


def test_outlier_rejection():
    rng = np.random.default_rng(17)
    fps, g_px = 240.0, 8000.0
    t = np.arange(0, 120) / fps
    y = 100 + 0.5 * g_px * t ** 2
    x = 300 + 10 * t
    y = add_noise(y, rng=rng)
    x = add_noise(x, rng=rng)
    # Gross outliers on 5% of points.
    idx = rng.choice(len(t), size=6, replace=False)
    y[idx] += rng.choice([-1, 1], size=6) * 40
    track = make_track(t, x, y, fps)
    calib = Calibration(m_per_px=9.81 / g_px, rel_uncertainty=0.0, method="reference", details="")
    fit = fit_model(track, "free_fall", calib)
    assert abs(fit.params["g"]["value"] - 9.81) / 9.81 < 0.02

def test_velocity_stderr_includes_linear_and_covariance():
    import math
    from physicam.physics import _velocity_stderr
    # v = 2*a2*t + a1; Var = (2t)^2 Var(a2) + Var(a1) + 4t Cov(a2,a1)
    cov = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 1.0]]
    got = _velocity_stderr(cov, 0.5)
    assert abs(got - math.sqrt(1.0 * 0.04 + 0.09 + 2.0 * 1.0 * 0.01)) < 1e-12
    # quadratic-only estimate would be 2t*sigma_a2 = 0.2; full must differ
    assert got > 0.2
    assert _velocity_stderr(None, 1.0) == 0.0
