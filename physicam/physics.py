"""Physical model fits in pixel space (image y DOWN), converted via Calibration.

Every fit returns a JSON-serializable Fit. Uncertainties combine the
curve_fit covariance (after one pass of outlier rejection) with the
calibration relative uncertainty in quadrature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

from physicam.calibrate import Calibration
from physicam.sam import Track

G_SI = 9.81


@dataclass
class Fit:
    """Result of one model fit."""

    model: str
    track_id: str
    t_range: list[float]
    params: dict  # name -> {value, stderr, unit}
    r2: float
    rmse_px: float
    n_points: int
    curve: dict = field(default_factory=dict)  # {t, x?, y?}
    residuals: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    restitution: list = field(default_factory=list)
    impact_times: list = field(default_factory=list)
    id: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "model": self.model, "track_id": self.track_id,
                "t_range": self.t_range, "params": self.params, "r2": self.r2,
                "rmse_px": self.rmse_px, "n_points": self.n_points, "curve": self.curve,
                "residuals": self.residuals, "warnings": self.warnings, "segments": self.segments,
                "restitution": self.restitution, "impact_times": self.impact_times}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _slice(track: Track, t_range) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.ones(track.n, dtype=bool)
    if t_range is not None:
        mask = (track.t >= t_range[0]) & (track.t <= t_range[1])
    idx = np.where(mask)[0]
    if len(idx) < 3:
        raise ValueError(f"only {len(idx)} points in t_range {t_range}; need >= 3")
    return track.t[idx], track.x[idx], track.y[idx], idx


def _robust_polyfit(t, y, deg):
    """Polyfit with one pass of MAD-based outlier rejection. Returns (coeffs, kept_mask, cov-ish stderr)."""
    coeffs = np.polyfit(t, y, deg)
    resid = y - np.polyval(coeffs, t)
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) * 1.4826
    if mad > 0:
        keep = np.abs(resid - med) <= 3.5 * mad
        if keep.sum() >= deg + 2 and not keep.all():
            coeffs = np.polyfit(t[keep], y[keep], deg)
            resid = y - np.polyval(coeffs, t)
            return coeffs, keep, resid
    return coeffs, np.ones_like(y, dtype=bool), resid


def _curve_fit_robust(func, t, y, p0, maxfev=20000):
    """curve_fit + one rejection pass. Returns (popt, perr, keep, yfit_full)."""
    popt, pcov = curve_fit(func, t, y, p0=p0, maxfev=maxfev)
    resid = y - func(t, *popt)
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) * 1.4826
    keep = np.ones_like(y, dtype=bool)
    if mad > 0:
        k2 = np.abs(resid - med) <= 3.5 * mad
        if k2.sum() >= len(p0) + 1 and not k2.all():
            keep = k2
            popt, pcov = curve_fit(func, t[keep], y[keep], p0=popt, maxfev=maxfev)
    perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else np.zeros_like(popt)
    return popt, perr, keep, func(t, *popt)


def _r2_rmse(y, yfit):
    ss_res = float(np.sum((y - yfit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2, math.sqrt(ss_res / max(len(y), 1))


def _param(value_px, stderr_px, unit_px, calib: Calibration, unit_m, scale_pow=1):
    """Convert px param to calibrated unit combining covariance + calibration uncertainty."""
    if calib is None or calib.method == "none":
        return {"value": float(value_px), "stderr": float(stderr_px), "unit": unit_px}
    factor = calib.m_per_px ** scale_pow
    v = float(value_px) * factor
    fit_rel = abs(stderr_px / value_px) if value_px else 0.0
    rel = math.hypot(fit_rel, abs(scale_pow) * calib.rel_uncertainty)
    return {"value": v, "stderr": abs(v) * rel, "unit": unit_m}


def _structure_warning(resid, t, warnings):
    if len(resid) >= 8:
        # Lag-1 autocorrelation as a cheap structure detector.
        r = np.asarray(resid, dtype=float)
        if np.std(r) > 0:
            ac = float(np.corrcoef(r[:-1], r[1:])[0, 1])
            if abs(ac) > 0.5:
                warnings.append("large residual structure: maybe air drag, perspective, or wrong model")


# ---------------------------------------------------------------------------
# free fall / projectile
# ---------------------------------------------------------------------------

def fit_projectile(track: Track, calib: Calibration | None = None, t_range=None,
                   options: dict | None = None, model_name: str = "projectile") -> Fit:
    t, x, y, _ = _slice(track, t_range)
    calib = calib or Calibration(1.0, 0.0, "none", "")
    warnings: list[str] = []
    if len(t) < 9:
        warnings.append(f"only {len(t)} points in fit")
    cx, keepx, _ = _robust_polyfit(t, x, 1)
    cy, keepy, _ = _robust_polyfit(t, y, 2)
    # Covariance-based stderrs via linear least squares on kept points.
    def lin_err(tt, yy, deg):
        c, cov = np.polyfit(tt, yy, deg, cov=True)
        return c, np.sqrt(np.diag(cov))
    try:
        cx, ex = lin_err(t[keepx], x[keepx], 1)
        cy, ey = lin_err(t[keepy], y[keepy], 2)
    except Exception:
        ex, ey = np.zeros(2), np.zeros(3)
    yfit = np.polyval(cy, t)
    xfit = np.polyval(cx, t)
    r2, rmse = _r2_rmse(y, yfit)
    resid = y - yfit
    _structure_warning(resid, t, warnings)
    vx_px, vy0_px = float(cx[0]), float(cy[1])
    a_px = float(2.0 * cy[0])  # y down: positive a = accelerating downward
    # stderr of a: coeff is a/2 so stderr_a = 2*stderr_coeff
    a_err = float(2.0 * ey[0]) if len(ey) > 0 else 0.0
    vx_err = float(ex[0]) if len(ex) > 0 else 0.0
    vy_err = float(ey[1]) if len(ey) > 1 else 0.0
    speed_px = math.hypot(vx_px, vy0_px)
    params = {
        "g": _param(a_px, a_err, "px/s^2", calib, "m/s^2", 1),
        "a_px": {"value": a_px, "stderr": a_err, "unit": "px/s^2"},
        "vx": _param(vx_px, vx_err, "px/s", calib, "m/s", 1),
        "vy0": _param(vy0_px, vy_err, "px/s", calib, "m/s", 1),
        "speed": _param(speed_px, math.hypot(vx_err, vy_err), "px/s", calib, "m/s", 1),
        "angle_deg": {"value": math.degrees(math.atan2(-vy0_px, vx_px)), "stderr": 0.0, "unit": "deg"},
        "x0": {"value": float(cx[1]), "stderr": float(ex[1]) if len(ex) > 1 else 0.0, "unit": "px"},
        "y0": {"value": float(cy[2]), "stderr": float(ey[2]) if len(ey) > 2 else 0.0, "unit": "px"},
    }
    if calib.method == "none":
        params["g"] = {"value": a_px, "stderr": a_err, "unit": "px/s^2"}
    tc = np.linspace(t[0], t[-1], 100)
    return Fit(model=model_name, track_id=track.id, t_range=[float(t[0]), float(t[-1])],
               params=params, r2=r2, rmse_px=rmse, n_points=int(len(t)),
               curve={"t": [float(v) for v in tc], "x": [float(v) for v in np.polyval(cx, tc)],
                      "y": [float(v) for v in np.polyval(cy, tc)]},
               residuals=[round(float(v), 4) for v in resid], warnings=warnings)


def fit_free_fall(track, calib=None, t_range=None, options=None) -> Fit:
    return fit_projectile(track, calib, t_range, options, model_name="free_fall")


# ---------------------------------------------------------------------------
# constant velocity / polynomial
# ---------------------------------------------------------------------------

def fit_constant_velocity(track, calib=None, t_range=None, options=None) -> Fit:
    t, x, y, _ = _slice(track, t_range)
    calib = calib or Calibration(1.0, 0.0, "none", "")
    warnings: list[str] = []
    cx, kx, _ = _robust_polyfit(t, x, 1)
    cy, ky, _ = _robust_polyfit(t, y, 1)
    try:
        _, ex = np.polyfit(t[kx], x[kx], 1, cov=True), None
    except Exception:
        ex = None
    # proper stderrs
    def errs(tt, yy, deg):
        try:
            _, cov = np.polyfit(tt, yy, deg, cov=True)
            return np.sqrt(np.diag(cov))
        except Exception:
            return np.zeros(deg + 1)
    exv, eyv = errs(t, x, 1), errs(t, y, 1)
    vx, vy = float(cx[0]), float(cy[0])
    speed = math.hypot(vx, vy)
    serr = math.hypot(float(exv[0]), float(eyv[0]))
    yfit = np.polyval(cy, t)
    r2, rmse = _r2_rmse(y, yfit)
    params = {
        "vx": _param(vx, float(exv[0]), "px/s", calib, "m/s", 1),
        "vy": _param(vy, float(eyv[0]), "px/s", calib, "m/s", 1),
        "speed": _param(speed, serr, "px/s", calib, "m/s", 1),
        "direction_deg": {"value": math.degrees(math.atan2(-vy, vx)), "stderr": 0.0, "unit": "deg"},
    }
    tc = np.linspace(t[0], t[-1], 50)
    return Fit(model="constant_velocity", track_id=track.id, t_range=[float(t[0]), float(t[-1])],
               params=params, r2=r2, rmse_px=rmse, n_points=int(len(t)),
               curve={"t": [float(v) for v in tc], "x": [float(v) for v in np.polyval(cx, tc)],
                      "y": [float(v) for v in np.polyval(cy, tc)]},
               residuals=[round(float(v), 4) for v in (y - yfit)], warnings=warnings)


def fit_polynomial(track, calib=None, t_range=None, options=None) -> Fit:
    options = options or {}
    deg = int(options.get("degree", 2))
    axis = options.get("axis", "y")
    t, x, y, _ = _slice(track, t_range)
    calib = calib or Calibration(1.0, 0.0, "none", "")
    yy = y if axis == "y" else x
    coeffs, keep, resid = _robust_polyfit(t, yy, deg)
    try:
        _, cov = np.polyfit(t[keep], yy[keep], deg, cov=True)
        errs = np.sqrt(np.diag(cov))
    except Exception:
        errs = np.zeros(deg + 1)
    yfit = np.polyval(coeffs, t)
    r2, rmse = _r2_rmse(yy, yfit)
    params = {f"c{i}": {"value": float(coeffs[deg - i]), "stderr": float(errs[deg - i]) if len(errs) > deg - i else 0.0,
                        "unit": f"px/s^{i}"} for i in range(deg + 1)}
    tc = np.linspace(t[0], t[-1], 100)
    curve = {"t": [float(v) for v in tc], axis: [float(v) for v in np.polyval(coeffs, tc)]}
    return Fit(model="polynomial", track_id=track.id, t_range=[float(t[0]), float(t[-1])],
               params=params, r2=r2, rmse_px=rmse, n_points=int(len(t)), curve=curve,
               residuals=[round(float(v), 4) for v in (yy - yfit)], warnings=[])


# ---------------------------------------------------------------------------
# pendulum / oscillator (damped sinusoid)
# ---------------------------------------------------------------------------

def _fit_damped(track, calib, t_range, model_name, options=None) -> Fit:
    options = options or {}
    t, x, y, _ = _slice(track, t_range)
    calib = calib or Calibration(1.0, 0.0, "none", "")
    warnings: list[str] = []
    # Dominant axis = larger range.
    use_x = float(np.ptp(x)) >= float(np.ptp(y))
    s = x if use_x else y
    axis_name = "x" if use_x else "y"
    if options.get("axis") in ("x", "y"):
        use_x = options["axis"] == "x"
        s = x if use_x else y
        axis_name = options["axis"]
    # Initial frequency from FFT.
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.01
    s0 = s - np.mean(s)
    freqs = np.fft.rfftfreq(len(t), d=dt)
    spec = np.abs(np.fft.rfft(s0 * np.hanning(len(t))))
    if len(spec) > 1:
        spec[0] = 0
    f0 = float(freqs[int(np.argmax(spec))]) if len(freqs) else 1.0
    if f0 <= 0:
        f0 = 1.0
    omega0 = 2 * math.pi * f0
    amp0 = float(np.std(s) * math.sqrt(2)) or 1.0
    c0 = float(np.mean(s))

    def model(tt, A, gamma, omega, phi, c):
        return A * np.exp(-gamma * tt) * np.cos(omega * tt + phi) + c

    # Fit in shifted time for stability.
    tshift = t - t[0]
    try:
        popt, perr, keep, sfit = _curve_fit_robust(model, tshift, s, [amp0, 0.05, omega0, 0.0, c0])
    except Exception as exc:
        raise ValueError(f"damped-sinusoid fit failed: {exc}") from exc
    A, gamma, omega, phi, c = [float(v) for v in popt]
    eA, egamma, eomega, ephi, ec = [float(v) for v in perr]
    T = 2 * math.pi / omega if omega else float("inf")
    eT = T * (eomega / omega) if omega else 0.0
    r2, rmse = _r2_rmse(s, sfit)
    resid = s - sfit
    _structure_warning(resid, t, warnings)
    params = {
        "period": _param(T, eT, "s", calib, "s", 0) if False else {"value": float(T), "stderr": float(eT), "unit": "s"},
        "T": {"value": float(T), "stderr": float(eT), "unit": "s"},
        "omega": {"value": float(omega), "stderr": float(eomega), "unit": "rad/s"},
        "gamma": {"value": float(gamma), "stderr": float(egamma), "unit": "1/s"},
        "amplitude": _param(A, eA, "px", calib, "m", 1),
        "axis": {"value": axis_name, "stderr": 0.0, "unit": ""},
    }
    if model_name == "pendulum":
        Q = omega / (2 * gamma) if gamma > 1e-12 else float("inf")
        params["Q"] = {"value": float(Q), "stderr": 0.0, "unit": ""}
        L = options.get("length_m") or options.get("L")
        if L:
            # g = 4 pi^2 L / T^2 with finite-amplitude correction using amplitude angle.
            # Amplitude angle approx: theta0 ~ A_px * m_per_px / L (small-angle geometry).
            g_val = 4 * math.pi ** 2 * float(L) / (T * T)
            rel_T = eT / T if T else 0.0
            rel = math.hypot(2 * rel_T, calib.rel_uncertainty if calib.method != "none" else 0.0)
            theta0 = 0.0
            if calib.method != "none":
                theta0 = abs(A) * calib.m_per_px / float(L)
                corr = 1.0 + theta0 ** 2 / 16.0 + 11.0 * theta0 ** 4 / 3072.0
                g_val = g_val * corr
                params["theta0_rad"] = {"value": float(theta0), "stderr": 0.0, "unit": "rad"}
            params["g"] = {"value": float(g_val), "stderr": float(abs(g_val) * rel), "unit": "m/s^2"}
            params["length"] = {"value": float(L), "stderr": 0.0, "unit": "m"}
    else:  # oscillator
        omega0_fit = math.sqrt(max(omega ** 2 + gamma ** 2, 0.0))
        zeta = gamma / omega0_fit if omega0_fit else 0.0
        params["omega0"] = {"value": float(omega0_fit), "stderr": float(eomega), "unit": "rad/s"}
        params["zeta"] = {"value": float(zeta), "stderr": 0.0, "unit": ""}
        params["k_over_m"] = {"value": float(omega0_fit ** 2), "stderr": float(2 * omega0_fit * eomega), "unit": "1/s^2"}
    tc = np.linspace(t[0], t[-1], 200)
    sc = model(tc - t[0], *popt)
    curve = {"t": [float(v) for v in tc], axis_name: [float(v) for v in sc]}
    return Fit(model=model_name, track_id=track.id, t_range=[float(t[0]), float(t[-1])],
               params=params, r2=r2, rmse_px=rmse, n_points=int(len(t)), curve=curve,
               residuals=[round(float(v), 4) for v in resid], warnings=warnings)


def fit_pendulum(track, calib=None, t_range=None, options=None) -> Fit:
    return _fit_damped(track, calib, t_range, "pendulum", options)


def fit_oscillator(track, calib=None, t_range=None, options=None) -> Fit:
    return _fit_damped(track, calib, t_range, "oscillator", options)


# ---------------------------------------------------------------------------
# bounce
# ---------------------------------------------------------------------------

def detect_impacts(t: np.ndarray, y: np.ndarray, fps: float) -> list[int]:
    """Indices of impacts (local maxima of y, y-down = floor contact).

    Robust to noise via light smoothing and a prominence requirement scaled
    to the trajectory range; requires velocity sign change + -> -.
    """
    n = len(y)
    if n < 5:
        return []
    # Light smoothing (3-pt moving average, handles jitter without shifting peaks much).
    ys = np.convolve(y, np.ones(3) / 3.0, mode="same")
    ys[0], ys[-1] = y[0], y[-1]
    vy = np.gradient(ys, t)
    rng = float(np.ptp(ys)) or 1.0
    impacts: list[int] = []
    # Candidate peaks: local maxima with prominence.
    for i in range(2, n - 2):
        if ys[i] >= ys[i - 1] and ys[i] >= ys[i + 1] and ys[i] >= ys[i - 2] and ys[i] >= ys[i + 2]:
            # Prominence vs neighbourhood minima within +/- 0.5s window.
            win = max(3, int(0.5 * fps))
            lo, hi = max(0, i - win), min(n, i + win + 1)
            prom = ys[i] - max(float(np.min(ys[lo:i + 1])), float(np.min(ys[i:hi]))) if False else ys[i] - min(float(np.min(ys[lo:hi])), ys[i])
            # Simpler prominence: drop to nearest minima on each side.
            left_min = float(np.min(ys[lo:i + 1])) if i > lo else ys[i]
            right_min = float(np.min(ys[i:hi])) if hi > i else ys[i]
            prom = ys[i] - max(left_min, right_min)
            if prom < 0.15 * rng:
                continue
            # Velocity sign: falling before (vy>0), rising after (vy<0).
            vb = float(np.mean(vy[max(0, i - 3):i])) if i > 0 else vy[i]
            va = float(np.mean(vy[i + 1:i + 4])) if i + 1 < n else vy[i]
            if vb > 0 and va < 0:
                impacts.append(i)
    # Merge impacts closer than 0.08 s (keep the deepest y).
    merged: list[int] = []
    for idx in impacts:
        if merged and (t[idx] - t[merged[-1]]) < 0.08:
            if ys[idx] > ys[merged[-1]]:
                merged[-1] = idx
        else:
            merged.append(idx)
    return merged


def fit_bounce(track: Track, calib: Calibration | None = None, t_range=None,
               options: dict | None = None) -> Fit:
    """Per-flight parabolas, per-impact restitution, pooled g."""
    options = options or {}
    calib = calib or Calibration(1.0, 0.0, "none", "")
    t_all, x_all, y_all, _ = _slice(track, t_range)
    fps = track.fps
    warnings: list[str] = []
    impact_idx_local = detect_impacts(t_all, y_all, fps)
    # Impact times from refined parabola vertex: use raw local indices directly.
    impact_times = [float(t_all[i]) for i in impact_idx_local]
    # Flight segments: [start, first impact), between impacts, (last impact, end].
    bounds_t = [float(t_all[0])] + impact_times + [float(t_all[-1])]
    segments: list[dict] = []
    g_vals: list[tuple[float, float]] = []  # (g_px, stderr_px)
    fits_y: list[np.ndarray] = []
    all_resid: list[float] = []
    curve_t: list[float] = []
    curve_y: list[float] = []
    apex_heights: list[float] = []

    for s_idx in range(len(bounds_t) - 1):
        t0, t1 = bounds_t[s_idx], bounds_t[s_idx + 1]
        # Trim 2 frames at impact ends (contact / deformation), keep edges of clip.
        trim = 2.0 / fps if fps else 0.0
        lo = t0 + (trim if s_idx > 0 else 0.0)
        hi = t1 - (trim if s_idx < len(bounds_t) - 2 else 0.0)
        sel = (t_all >= lo) & (t_all <= hi)
        tt, yy = t_all[sel], y_all[sel]
        seg: dict = {"index": s_idx, "t_range": [float(t0), float(t1)], "n_points": int(len(tt))}
        if len(tt) < 5:
            seg["warnings"] = [f"only {len(tt)} points in flight"]
            segments.append(seg)
            continue
        coeffs, keep, resid_all = _robust_polyfit(tt, yy, 2)
        try:
            _, cov = np.polyfit(tt[keep], yy[keep], 2, cov=True)
            errs = np.sqrt(np.diag(cov))
        except Exception:
            errs = np.zeros(3)
        a_px = float(2.0 * coeffs[0])
        a_err = float(2.0 * errs[0]) if len(errs) > 0 else 0.0
        yfit = np.polyval(coeffs, tt)
        r2s, rmses = _r2_rmse(yy, yfit)
        if len(tt) < 9:
            seg.setdefault("warnings", []).append("only 9 points in flight" if len(tt) == 9 else f"only {len(tt)} points in flight")
        seg.update({"g_px": a_px, "g_px_stderr": a_err, "coeffs": [float(c) for c in coeffs],
                    "coeff_stderr": [float(e) for e in errs], "r2": r2s, "rmse_px": rmses,
                    "t_fit_range": [float(tt[0]), float(tt[-1])]})
        # Quality gate: a flight that is not cleanly parabolic (caught ball,
        # rolling tail, perspective-distorted throw) must not poison the pool.
        seg["excluded_from_pool"] = False
        if r2s < 0.98:
            seg["excluded_from_pool"] = True
            seg.setdefault("warnings", []).append(
                f"flight R^2={r2s:.3f} < 0.98: not cleanly ballistic, excluded from pooled g")
        if a_px <= 0:
            seg["excluded_from_pool"] = True
            seg.setdefault("warnings", []).append(
                "fitted acceleration is not downward (a_px <= 0): perspective, camera tilt, or non-ballistic motion")
        # Converted g for this segment.
        seg["g"] = _param(a_px, a_err, "px/s^2", calib, "m/s^2", 1)
        if calib.method == "none":
            seg["g"] = {"value": a_px, "stderr": a_err, "unit": "px/s^2"}
        segments.append(seg)
        if a_err > 0 and not seg["excluded_from_pool"]:
            g_vals.append((a_px, a_err))
        # Apex height: impact-floor y (max at ends) minus apex (min fitted y in flight).
        apex_y = float(np.min(yfit))
        floor_y = float(max(yfit[0], yfit[-1])) if len(yfit) else apex_y
        apex_heights.append(floor_y - apex_y)
        fits_y.append(yfit)
        all_resid.extend([float(v) for v in (yy - yfit)])
        tc = np.linspace(tt[0], tt[-1], max(10, len(tt)))
        curve_t.extend([float(v) for v in tc])
        curve_y.extend([float(v) for v in np.polyval(coeffs, tc)])

    # Restitution per impact: fitted velocities either side; cross-check heights.
    seg_by_index = {s["index"]: s for s in segments}
    restitution: list[dict] = []
    for k, it in enumerate(impact_times):
        before = seg_by_index.get(k)
        after = seg_by_index.get(k + 1)
        entry: dict = {"impact_index": k, "t": it}
        if before and "coeffs" in before and after and "coeffs" in after:
            cb, ca = before["coeffs"], after["coeffs"]
            v_before = float(2 * cb[0] * it + cb[1])  # dy/dt, positive = falling
            v_after = float(2 * ca[0] * it + ca[1])  # negative = rising
            if before.get("excluded_from_pool") or after.get("excluded_from_pool"):
                entry["valid"] = False
                entry["warning"] = "neighbouring flight excluded from pool; restitution unreliable"
            elif v_before > 0 and v_after < 0:
                e = abs(v_after) / v_before
                if e > 1.05:
                    entry["valid"] = False
                    entry["warning"] = f"e={e:.2f} > 1 is unphysical; suspect segment boundary or perspective"
                else:
                    entry["valid"] = True
                # stderr by propagation ignoring covariance between segments.
                eb = before["coeff_stderr"][0] if len(before["coeff_stderr"]) > 0 else 0.0
                ea = after["coeff_stderr"][0] if len(after["coeff_stderr"]) > 0 else 0.0
                vb_err = abs(2 * it) * eb
                va_err = abs(2 * it) * ea
                e_err = e * math.hypot(va_err / abs(v_after) if v_after else 0, vb_err / v_before if v_before else 0)
                entry.update({"e": float(e), "e_stderr": float(e_err),
                              "v_before_px_s": v_before, "v_after_px_s": v_after})
            else:
                entry["warning"] = f"velocity signs unexpected at impact (v_before={v_before:.1f}, v_after={v_after:.1f})"
        # Height cross-check: e_h = sqrt(h_next / h_prev) using adjacent flight apex heights.
        if 0 <= k < len(apex_heights) and 0 <= k + 1 < len(apex_heights) + 1:
            pass
        restitution.append(entry)
    # Attach height-based e where both neighbouring flights have heights.
    # apex_heights[i] corresponds to flight segment i (only for fitted segments, in order).
    fitted_idx = [s["index"] for s in segments if "g_px" in s]
    hmap = {idx: h for idx, h in zip(fitted_idx, apex_heights)}
    for entry in restitution:
        k = entry["impact_index"]
        if k in hmap and (k + 1) in hmap and hmap[k] > 0:
            entry["e_height"] = float(math.sqrt(hmap[k + 1] / hmap[k]))
            if "e" in entry and abs(entry["e"] - entry["e_height"]) > 0.15:
                entry["warning"] = entry.get("warning", "") + " velocity-e and height-e disagree >0.15"

    if not g_vals:
        raise ValueError("bounce: no flight segment with >= 5 points could be fitted")
    w = np.array([1.0 / (e * e) for _, e in g_vals])
    g_pool = float(np.sum(w * np.array([g for g, _ in g_vals])) / np.sum(w))
    g_pool_err = float(math.sqrt(1.0 / np.sum(w)))
    # Consistency check.
    g_arr = np.array([g for g, _ in g_vals])
    if len(g_arr) > 1 and g_pool:
        spread = float((np.max(g_arr) - np.min(g_arr)) / abs(g_pool))
        if spread > 0.05:
            warnings.append(f"per-segment g range {spread:.1%} > 5%: check time base, segment boundaries, or perspective (motion toward/away from camera)")
    if not impact_times:
        warnings.append("no impacts detected; fitted as free fall")
    # Overall r2/rmse across fitted flights.
    rmse_all = float(math.sqrt(np.mean(np.square(all_resid)))) if all_resid else 0.0
    params = {
        "g": _param(g_pool, g_pool_err, "px/s^2", calib, "m/s^2", 1),
        "g_px": {"value": g_pool, "stderr": g_pool_err, "unit": "px/s^2"},
        "n_impacts": {"value": float(len(impact_times)), "stderr": 0.0, "unit": ""},
    }
    if calib.method == "none":
        params["g"] = {"value": g_pool, "stderr": g_pool_err, "unit": "px/s^2"}
    for i, entry in enumerate(restitution):
        if "e" in entry and entry.get("valid", True):
            params[f"e_{i}"] = {"value": entry["e"], "stderr": entry.get("e_stderr", 0.0), "unit": ""}
    order = np.argsort(curve_t)
    curve = {"t": [curve_t[i] for i in order], "y": [curve_y[i] for i in order]} if curve_t else {}
    fit = Fit(model="bounce", track_id=track.id, t_range=[float(t_all[0]), float(t_all[-1])],
              params=params, r2=float(segments[0].get("r2", 0.0)) if segments else 0.0,
              rmse_px=rmse_all, n_points=int(len(t_all)), curve=curve,
              residuals=[round(float(v), 4) for v in all_resid], warnings=warnings,
              segments=segments, restitution=restitution, impact_times=impact_times)
    return fit


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

_DISPATCH = {
    "free_fall": fit_free_fall,
    "projectile": fit_projectile,
    "bounce": fit_bounce,
    "pendulum": fit_pendulum,
    "oscillator": fit_oscillator,
    "constant_velocity": fit_constant_velocity,
    "polynomial": fit_polynomial,
}


def _metric_track_and_calib(track: Track, calib: Calibration):
    """Return (metric Track in metres, identity Calibration) for per-frame calib."""
    X, Y = calib.to_metric(track)
    metric = Track(id=track.id, label=track.label, frames=np.asarray(track.frames),
                   t=np.asarray(track.t, dtype=float), x=X, y=Y,
                   area=np.asarray(track.area, dtype=float), bbox=np.asarray(track.bbox, dtype=float),
                   conf=np.asarray(track.conf, dtype=float),
                   eq_diameter=np.asarray(track.eq_diameter, dtype=float),
                   contours=dict(track.contours), fps=track.fps)
    identity = Calibration(m_per_px=1.0, rel_uncertainty=calib.rel_uncertainty,
                           method=calib.method, details=calib.details)
    return metric, identity


def fit_model(track: Track, model: str, calib: Calibration | None = None,
              t_range=None, options: dict | None = None) -> Fit:
    """Fit *model* to *track* and return a Fit. Raises ValueError on bad model.

    When *calib* is per-frame (perspective-corrected), positions are first
    converted to metric lateral coordinates ``X=(x-cx)s(t)``, ``Y=(y-cy)s(t)``
    and the fit runs in metres with an identity scale.
    """
    key = model.lower().strip()
    if key not in _DISPATCH:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(_DISPATCH)}")
    if calib is not None and getattr(calib, "kind", "constant") == "per_frame":
        metric, identity = _metric_track_and_calib(track, calib)
        fit = _DISPATCH[key](metric, identity, t_range, options)
        fit.warnings = list(fit.warnings) + [
            "per-frame perspective correction applied: fit in metric lateral coordinates X=(x-cx)s(t), Y=(y-cy)s(t); rmse is in metres"]
        # rmse_px now holds metres; keep field for contract compatibility.
        return fit
    fit = _DISPATCH[key](track, calib, t_range, options)
    return fit
