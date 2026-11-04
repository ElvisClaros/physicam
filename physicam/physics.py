"""Physical model fits in pixel space (image y DOWN), converted via Calibration.

Every fit returns a JSON-serializable Fit. Uncertainties combine the
curve_fit covariance (after one pass of outlier rejection) with the
calibration relative uncertainty in quadrature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit, least_squares

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
    curve: dict = field(default_factory=dict)  # {t, x, y}
    residuals: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    restitution: list = field(default_factory=list)
    impact_times: list = field(default_factory=list)
    impacts: list = field(default_factory=list)
    info: dict = field(default_factory=dict)
    id: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "model": self.model, "track_id": self.track_id,
                "t_range": self.t_range, "params": self.params, "r2": self.r2,
                "rmse_px": self.rmse_px, "n_points": self.n_points, "curve": self.curve,
                "residuals": self.residuals, "warnings": self.warnings, "segments": self.segments,
                "restitution": self.restitution, "impact_times": self.impact_times,
                "impacts": self.impacts, "info": self.info}


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


def _residual_inflation(resid, n_params: int) -> tuple[float, float]:
    """Inflation factor for covariance stderrs from structured residuals.

    The point-to-point (high-frequency) scatter of the residuals estimates
    the white centroid noise; the residual RMS is the total noise including
    low-frequency model mismatch.  ``chi2_red = RSS / (dof * sigma_hf**2)``
    is therefore ~1 for a clean white-noise fit and > 1 when residuals are
    structured (model mismatch, drag, perspective).  Returns
    ``(sqrt(max(chi2_red, 1)), chi2_red)``.  Covariance stderrs from
    ``curve_fit``/``polyfit(cov=True)`` already absorb the RMS, so this
    factor only adds the *excess* over white noise.
    """
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 8:
        return 1.0, 1.0
    dof = max(n - int(n_params), 1)
    rss = float(np.sum(r ** 2))
    if rss <= 0:
        return 1.0, 1.0
    d = np.diff(r)
    med = float(np.median(d))
    sigma_hf = 1.4826 * float(np.median(np.abs(d - med))) / math.sqrt(2.0)
    if sigma_hf <= 0:
        sigma_hf = float(np.std(d) / math.sqrt(2.0)) if len(d) else 0.0
    if sigma_hf <= 0:
        return 1.0, 1.0
    chi2_red = rss / (dof * sigma_hf ** 2)
    if not np.isfinite(chi2_red) or chi2_red <= 1.0:
        return 1.0, float(chi2_red) if np.isfinite(chi2_red) else 1.0
    return float(math.sqrt(chi2_red)), float(chi2_red)


def _inflate_errs(errs, resid, n_params: int):
    """Scale *errs* by the residual inflation factor; return (errs, factor, chi2_red)."""
    factor, chi2_red = _residual_inflation(resid, n_params)
    if factor != 1.0:
        errs = np.asarray(errs, dtype=float) * factor
    return errs, factor, chi2_red


def _param(value_px, stderr_px, unit_px, calib: Calibration, unit_m, scale_pow=1):
    """Convert px param to calibrated unit combining covariance + calibration uncertainty."""
    if calib is None or calib.method == "none":
        return {"value": float(value_px), "stderr": float(stderr_px), "unit": unit_px}
    factor = calib.m_per_px ** scale_pow
    v = float(value_px) * factor
    fit_rel = abs(stderr_px / value_px) if value_px else 0.0
    rel = math.hypot(fit_rel, abs(scale_pow) * calib.rel_uncertainty)
    return {"value": v, "stderr": abs(v) * rel, "unit": unit_m}


def _smoothed_other_axis(track: Track, tc: np.ndarray, axis: str) -> list[float]:
    """Track position on *axis* smoothed (5-pt) and interpolated at *tc*."""
    src_t = np.asarray(track.t, dtype=float)
    src = np.asarray(track.x if axis == "x" else track.y, dtype=float)
    if len(src) >= 5:
        kernel = np.ones(5) / 5.0
        sm = np.convolve(src, kernel, mode="same")
        sm[:2], sm[-2:] = src[:2], src[-2:]
    else:
        sm = src
    return [float(v) for v in np.interp(tc, src_t, sm)]


def _track_pos_at(track: Track, t_query: float) -> tuple[float, float]:
    """Interpolated track (x, y) at *t_query*."""
    return (float(np.interp(t_query, track.t, track.x)),
            float(np.interp(t_query, track.t, track.y)))


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
    # Honest errors: inflate covariance stderrs by sqrt(chi2_red) when
    # residuals are structured (chi2_red from residual RMS vs white noise).
    ex, _fx, _cx2 = _inflate_errs(ex, x[keepx] - np.polyval(cx, t[keepx]), 2)
    ey, _fy, _cy2 = _inflate_errs(ey, y[keepy] - np.polyval(cy, t[keepy]), 3)
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
        "angle_deg": {"value": math.degrees(math.atan2(-vy0_px, vx_px)),
                      "stderr": float(math.degrees(math.hypot(vy_err * abs(vx_px), vx_err * abs(vy0_px)) / max(speed_px ** 2, 1e-12))) if speed_px else 0.0, "unit": "deg"},
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
               residuals=[round(float(v), 4) for v in resid], warnings=warnings,
               info={"chi2_red": float(_cy2), "inflation_factor": float(_fy)})


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
    exv, _fxv, _cxv = _inflate_errs(exv, x - np.polyval(cx, t), 2)
    eyv, _fyv, _cyv = _inflate_errs(eyv, y - np.polyval(cy, t), 2)
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
               residuals=[round(float(v), 4) for v in (y - yfit)], warnings=warnings,
               info={"chi2_red": float(_cyv), "inflation_factor": float(_fyv)})


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
    errs, _fpoly, _cpoly = _inflate_errs(errs, yy[keep] - np.polyval(coeffs, t[keep]), deg + 1)
    yfit = np.polyval(coeffs, t)
    r2, rmse = _r2_rmse(yy, yfit)
    params = {f"c{i}": {"value": float(coeffs[deg - i]), "stderr": float(errs[deg - i]) if len(errs) > deg - i else 0.0,
                        "unit": f"px/s^{i}"} for i in range(deg + 1)}
    tc = np.linspace(t[0], t[-1], 100)
    other = "y" if axis == "x" else "x"
    curve = {"t": [float(v) for v in tc], axis: [float(v) for v in np.polyval(coeffs, tc)],
             other: _smoothed_other_axis(track, tc, other)}
    return Fit(model="polynomial", track_id=track.id, t_range=[float(t[0]), float(t[-1])],
               params=params, r2=r2, rmse_px=rmse, n_points=int(len(t)), curve=curve,
               residuals=[round(float(v), 4) for v in (yy - yfit)], warnings=[],
               info={"axis": axis, "chi2_red": float(_cpoly), "inflation_factor": float(_fpoly)})


# ---------------------------------------------------------------------------
# pendulum / oscillator (damped sinusoid)
# ---------------------------------------------------------------------------

def _damped_model(tt, A, gamma, omega, phi, c):
    return A * np.exp(-gamma * tt) * np.cos(omega * tt + phi) + c


def _fit_damped_sinusoid(t: np.ndarray, s: np.ndarray):
    """Fit A*exp(-gamma*t)*cos(omega*t+phi)+c in shifted time.

    Returns (popt, perr, sfit) with popt = [A, gamma, omega, phi, c].
    """
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.01
    s0 = s - np.mean(s)
    freqs = np.fft.rfftfreq(len(t), d=dt)
    spec = np.abs(np.fft.rfft(s0 * np.hanning(len(t))))
    if len(spec) > 1:
        spec[0] = 0
    f0 = float(freqs[int(np.argmax(spec))]) if len(freqs) else 1.0
    if f0 <= 0:
        f0 = 1.0
    p0 = [float(np.std(s) * math.sqrt(2)) or 1.0, 0.05, 2 * math.pi * f0, 0.0, float(np.mean(s))]
    popt, perr, _keep, sfit = _curve_fit_robust(_damped_model, t - t[0], s, p0)
    return popt, perr, sfit


def _half_split_periods(t: np.ndarray, s: np.ndarray, T_full: float) -> list[float] | None:
    """Periods from fitting each half separately, or None.

    Only when each half spans >= 3 periods (duration / T_full >= 3).
    """
    if not (np.isfinite(T_full) and T_full > 0) or len(t) < 12:
        return None
    mid = 0.5 * (float(t[0]) + float(t[-1]))
    halves: list[float] = []
    for mask in (t <= mid, t > mid):
        if int(mask.sum()) < 6:
            return None
        th, sh = t[mask], s[mask]
        if float(th[-1] - th[0]) / T_full < 3.0:
            return None
        try:
            popt_h, _perr_h, _fit_h = _fit_damped_sinusoid(th, sh)
        except Exception:
            return None
        omega_h = float(popt_h[2])
        if not (np.isfinite(omega_h) and omega_h > 0):
            return None
        halves.append(float(2 * math.pi / omega_h))
    return halves


def _honest_period(t: np.ndarray, s: np.ndarray, resid, omega: float, eomega_cov: float,
                   n_params: int = 5) -> dict:
    """Honest period uncertainty: inflated stat + half-split systematic.

    Returns dict with T, sigma_stat (covariance stderr inflated by
    sqrt(chi2_red) when chi2_red > 1), sigma_sys (|T1-T2|/2 from halves,
    0 when halves unavailable), sigma_T = sqrt(sigma_stat^2+sigma_sys^2),
    T_halves, chi2_red and inflation_factor.
    """
    T = float(2 * math.pi / omega) if omega else float("inf")
    eT_cov = float(T * (eomega_cov / omega)) if omega else 0.0
    factor, chi2_red = _residual_inflation(resid, n_params)
    sigma_stat = float(eT_cov * factor)
    halves = _half_split_periods(np.asarray(t, float), np.asarray(s, float), T)
    sigma_sys = float(abs(halves[0] - halves[1]) / 2.0) if halves else 0.0
    sigma_T = float(math.hypot(sigma_stat, sigma_sys))
    return {"T": T, "eT_cov": float(eT_cov), "sigma_stat": sigma_stat,
            "sigma_sys": sigma_sys, "sigma_T": sigma_T,
            "T_halves": [float(v) for v in halves] if halves else None,
            "chi2_red": float(chi2_red), "inflation_factor": float(factor)}


def _fit_circle(x: np.ndarray, y: np.ndarray) -> dict:
    """Circle fit: algebraic Kasa fit, then Gauss-Newton geometric refinement.

    Returns centre (cx, cy), radius R, per-parameter stderrs from the
    geometric-fit covariance, radial residuals, and the angular arc span
    (of theta = atan2(x-cx, y-cy), in degrees).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    a_mat = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b_vec = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    cx, cy, c0 = [float(v) for v in sol]
    r2_init = c0 + cx ** 2 + cy ** 2
    if not np.isfinite(r2_init) or r2_init <= 0:
        raise ValueError("Kasa circle fit gave no positive radius")
    p_init = np.array([cx, cy, math.sqrt(r2_init)])

    def resid(p):
        return np.hypot(x - p[0], y - p[1]) - p[2]

    result = least_squares(resid, p_init, method="lm", max_nfev=100 * len(x) + 100)
    cx, cy, radius = [float(v) for v in result.x]
    if not (np.isfinite(radius) and radius > 0):
        raise ValueError("geometric circle fit gave no positive radius")
    r = resid(result.x)
    dof = max(len(x) - 3, 1)
    s2 = float(np.sum(r ** 2) / dof)
    try:
        cov = s2 * np.linalg.pinv(result.jac.T @ result.jac)
        perr = np.sqrt(np.abs(np.diag(cov)))
    except Exception:
        perr = np.zeros(3)
    theta = np.arctan2(x - cx, y - cy)
    return {"cx": cx, "cy": cy, "R": radius,
            "perr": [float(v) for v in perr], "resid": r,
            "arc_span_deg": float(math.degrees(np.ptp(theta))) if len(theta) else 0.0,
            "theta": theta}


def _opt_float(options: dict, *keys: str) -> float | None:
    for key in keys:
        v = options.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def _fit_damped(track, calib, t_range, model_name, options=None) -> Fit:
    """Oscillator path: damped sinusoid on the dominant axis."""
    options = options or {}
    t, x, y, _ = _slice(track, t_range)
    calib = calib or Calibration(1.0, 0.0, "none", "")
    warnings: list[str] = []
    use_x = float(np.ptp(x)) >= float(np.ptp(y))
    s = x if use_x else y
    axis_name = "x" if use_x else "y"
    if options.get("axis") in ("x", "y"):
        use_x = options["axis"] == "x"
        s = x if use_x else y
        axis_name = options["axis"]
    try:
        popt, perr, sfit = _fit_damped_sinusoid(t, s)
    except Exception as exc:
        raise ValueError(f"damped-sinusoid fit failed: {exc}") from exc
    A, gamma, omega, phi, c = [float(v) for v in popt]
    eA, egamma, eomega, ephi, ec = [float(v) for v in perr]
    T = 2 * math.pi / omega if omega else float("inf")
    r2, rmse = _r2_rmse(s, sfit)
    resid = s - sfit
    _structure_warning(resid, t, warnings)
    honest = _honest_period(t, np.asarray(s, float), resid, omega, eomega)
    eT = float(honest["sigma_T"])
    # Scale omega-family stderrs by the same honest factor as T.
    scale = (eT / honest["eT_cov"]) if honest["eT_cov"] > 0 else 1.0
    eomega = float(eomega * scale)
    eA, egamma = float(eA * honest["inflation_factor"]), float(egamma * honest["inflation_factor"])
    params = {
        "T": {"value": float(T), "stderr": float(eT), "unit": "s"},
        "omega": {"value": float(omega), "stderr": float(eomega), "unit": "rad/s"},
        "gamma": {"value": float(gamma), "stderr": float(egamma), "unit": "1/s"},
        "amplitude": _param(A, eA, "px", calib, "m", 1),
    }
    info = {"axis": axis_name, "T_halves": honest["T_halves"],
            "sigma_stat": float(honest["sigma_stat"]), "sigma_sys": float(honest["sigma_sys"]),
            "chi2_red": float(honest["chi2_red"]), "inflation_factor": float(honest["inflation_factor"])}
    omega0_fit = math.sqrt(max(omega ** 2 + gamma ** 2, 0.0))
    zeta = gamma / omega0_fit if omega0_fit else 0.0
    params["omega0"] = {"value": float(omega0_fit), "stderr": float(eomega), "unit": "rad/s"}
    zeta_err = float(abs(zeta) * math.hypot(egamma / gamma if gamma > 1e-12 else 0.0,
                                            eomega / omega0_fit if omega0_fit else 0.0)) if zeta else float(egamma / omega0_fit) if omega0_fit else 0.0
    params["zeta"] = {"value": float(zeta), "stderr": zeta_err, "unit": ""}
    params["k_over_m"] = {"value": float(omega0_fit ** 2), "stderr": float(2 * omega0_fit * eomega), "unit": "1/s^2"}
    tc = np.linspace(t[0], t[-1], 200)
    sc = _damped_model(tc - t[0], *popt)
    other = "y" if axis_name == "x" else "x"
    curve = {"t": [float(v) for v in tc], axis_name: [float(v) for v in sc],
             other: _smoothed_other_axis(track, tc, other), "axis": axis_name}
    return Fit(model=model_name, track_id=track.id, t_range=[float(t[0]), float(t[-1])],
               params=params, r2=r2, rmse_px=rmse, n_points=int(len(t)), curve=curve,
               residuals=[round(float(v), 4) for v in resid], warnings=warnings, info=info)


def fit_pendulum(track, calib=None, t_range=None, options=None) -> Fit:
    """Pendulum: circle-fit geometry, damped sinusoid on theta(t), then g.

    g priority: (1) options.length_m, (2) calibration with circle-fit
    L_px, (3) neither: T/T0/theta0 plus predicted length for g=9.81.
    """
    options = dict(options or {})
    t, x, y, _ = _slice(track, t_range)
    calib = calib or Calibration(1.0, 0.0, "none", "")
    warnings: list[str] = []
    axis_name = "x" if float(np.ptp(x)) >= float(np.ptp(y)) else "y"
    if options.get("axis") in ("x", "y"):
        axis_name = str(options["axis"])

    # --- circle geometry ---------------------------------------------------
    circle: dict | None = None
    try:
        circle = _fit_circle(x, y)
    except Exception as exc:
        warnings.append(f"circle fit failed ({exc}); L_px unknown")
    arc_span_deg = float(circle["arc_span_deg"]) if circle else None
    pivot_px: list[float] | None = None
    L_px: float | None = None
    L_px_stderr: float | None = None
    if circle is not None:
        cx_c, cy_c, r_c = circle["cx"], circle["cy"], circle["R"]
        if arc_span_deg is not None and arc_span_deg < 10.0:
            warnings.append(
                f"short swing arc: arc spans {arc_span_deg:.1f} deg (< ~10 deg); "
                "circle-fit pivot and radius are ill-conditioned, so L_px is unknown "
                f"(large uncertainty, sigma_L_px~{circle['perr'][2]:.1f} px)")
        elif not cy_c < float(np.median(y)):
            warnings.append("circle-fit pivot is not above the bob (pivot y >= median bob y); L_px unknown")
        else:
            pivot_px = [float(cx_c), float(cy_c)]
            L_px = float(r_c)
            L_px_stderr = float(circle["perr"][2])
            if L_px and L_px_stderr / L_px > 0.10:
                warnings.append(f"circle-fit radius has large uncertainty ({L_px_stderr / L_px:.0%})")

    # --- theta(t) or dominant-axis fit -------------------------------------
    theta0_rad, theta0_err = 0.0, 0.0
    gamma = omega = 0.0
    egamma = eomega = 0.0
    r2 = rmse_px = 0.0
    resid_list: list[float] = []
    curve: dict = {}
    used_theta = False
    honest_signal: np.ndarray | None = None
    honest_resid: np.ndarray | None = None
    extent = max(float(np.ptp(x)), float(np.ptp(y)), 1.0)
    theta_popt = None
    if circle is not None and 0 < circle["R"] <= 50.0 * extent + 1000.0:
        theta = np.asarray(circle["theta"], dtype=float)
        try:
            theta_popt, theta_perr, theta_fit = _fit_damped_sinusoid(t, theta)
            r2_theta, _ = _r2_rmse(theta, theta_fit)
            if r2_theta < 0.5:
                theta_popt = None
            else:
                used_theta = True
                A, gamma, omega, phi, c_off = [float(v) for v in theta_popt]
                eA, egamma, eomega, _ephi, _ec = [float(v) for v in theta_perr]
                theta0_rad, theta0_err = abs(A), abs(eA)
                r2 = r2_theta
                cx_c, cy_c, r_c = circle["cx"], circle["cy"], circle["R"]
                x_fit = cx_c + r_c * np.sin(theta_fit)
                y_fit = cy_c + r_c * np.cos(theta_fit)
                rmse_px = float(math.sqrt(np.mean((x - x_fit) ** 2 + (y - y_fit) ** 2)))
                resid_list = [round(float(v), 4) for v in (r_c * (theta - theta_fit))]
                tc = np.linspace(t[0], t[-1], 200)
                theta_curve = _damped_model(tc - t[0], *theta_popt)
                curve = {"t": [float(v) for v in tc],
                         "x": [float(v) for v in (cx_c + r_c * np.sin(theta_curve))],
                         "y": [float(v) for v in (cy_c + r_c * np.cos(theta_curve))],
                         "axis": axis_name}
                _structure_warning(theta - theta_fit, t, warnings)
                honest_signal = np.asarray(theta, dtype=float)
                honest_resid = np.asarray(theta - theta_fit, dtype=float)
        except Exception:
            theta_popt = None
    if not used_theta:
        s = x if axis_name == "x" else y
        try:
            popt, perr, sfit = _fit_damped_sinusoid(t, np.asarray(s, dtype=float))
        except Exception as exc:
            raise ValueError(f"pendulum fit failed: {exc}") from exc
        A, gamma, omega, phi, c_off = [float(v) for v in popt]
        eA, egamma, eomega, _ephi, _ec = [float(v) for v in perr]
        r2, rmse_px = _r2_rmse(s, sfit)
        resid_list = [round(float(v), 4) for v in (s - sfit)]
        _structure_warning(s - sfit, t, warnings)
        honest_signal = np.asarray(s, dtype=float)
        honest_resid = np.asarray(s - sfit, dtype=float)
        if L_px:
            ratio = min(abs(A) / L_px, 1.0)
            theta0_rad = float(math.asin(ratio))
            theta0_err = float(abs(eA) / (L_px * math.sqrt(max(1.0 - ratio ** 2, 1e-9))))
        elif circle is not None:
            theta0_rad = float(np.ptp(circle["theta"]) / 2.0)
            theta0_err = float(theta0_rad)
            warnings.append("theta0 estimated from the observed angular span; amplitude is uncertain")
        else:
            warnings.append("theta0 unknown without a circle fit")
        tc = np.linspace(t[0], t[-1], 200)
        sc = _damped_model(tc - t[0], *popt)
        other = "y" if axis_name == "x" else "x"
        curve = {"t": [float(v) for v in tc], axis_name: [float(v) for v in sc],
                 other: _smoothed_other_axis(track, tc, other), "axis": axis_name}

    T = 2 * math.pi / omega if omega else float("inf")
    # Honest period uncertainty: inflated stat + half-split systematic.
    if honest_signal is not None and honest_resid is not None and omega:
        honest = _honest_period(t, honest_signal, honest_resid, omega, eomega)
    else:
        honest = {"sigma_stat": float(T * (eomega / omega)) if omega else 0.0,
                  "sigma_sys": 0.0, "sigma_T": float(T * (eomega / omega)) if omega else 0.0,
                  "T_halves": None, "chi2_red": 1.0, "inflation_factor": 1.0, "eT_cov": float(T * (eomega / omega)) if omega else 0.0}
    eT = float(honest["sigma_T"])
    _scale = (eT / honest["eT_cov"]) if honest.get("eT_cov") else 1.0
    eomega = float(eomega * _scale)
    egamma = float(egamma * float(honest["inflation_factor"]))
    theta0_err = float(theta0_err * float(honest["inflation_factor"]))
    # Finite-amplitude correction.
    corr = 1.0 + theta0_rad ** 2 / 16.0 + 11.0 * theta0_rad ** 4 / 3072.0
    dcorr = theta0_rad / 8.0 + 11.0 * theta0_rad ** 3 / 768.0
    sigma_corr = abs(dcorr) * theta0_err
    T0 = T / corr if corr else T
    rel_T0 = math.hypot(eT / T if T else 0.0, sigma_corr / corr if corr else 0.0)
    eT0 = float(abs(T0) * rel_T0)
    # gamma ~ 0 means Q is unbounded; emit null (JSON has no inf) not inf.
    Q = omega / (2 * gamma) if gamma > 1e-12 else None
    q_err = float(abs(Q) * math.hypot(eomega / omega if omega else 0.0,
                                      egamma / gamma if gamma > 1e-12 else 0.0)) if Q is not None else 0.0
    params: dict = {
        "T": {"value": float(T), "stderr": float(eT), "unit": "s"},
        "T0": {"value": float(T0), "stderr": float(eT0), "unit": "s"},
        "theta0_deg": {"value": float(math.degrees(theta0_rad)),
                       "stderr": float(math.degrees(theta0_err)), "unit": "deg"},
        "gamma": {"value": float(gamma), "stderr": float(egamma), "unit": "1/s"},
        "Q": {"value": (float(Q) if Q is not None else None), "stderr": q_err, "unit": ""},
    }

    # --- length and g --------------------------------------------------------
    length_m = _opt_float(options, "length_m", "L_m", "length", "L")
    string_length_m = _opt_float(options, "string_length_m", "string_length")
    length_err_given = _opt_float(options, "length_m_stderr", "length_stderr_m",
                                  "length_uncertainty_m", "length_m_uncertainty")
    refers = str(options.get("length_refers_to") or options.get("refers_to") or "").lower()
    quote = str(options.get("length_quote") or options.get("quote") or "").lower()
    is_string = (string_length_m is not None or bool(options.get("length_is_string"))
                 or bool(options.get("is_string_length"))
                 or str(options.get("length_kind") or options.get("length_type") or "").lower().startswith("string")
                 or "string" in refers or "string" in quote)
    if string_length_m is not None:
        length_m = string_length_m
    bob_d = _opt_float(options, "bob_diameter_m", "bob_diameter", "diameter_m", "object_diameter_m")
    bob_explicit = bob_d is not None

    L_m: float | None = None
    L_m_stderr: float | None = None
    method_g = "none"
    length_source = ""
    systematic_note = ""
    if length_m is not None:
        method_g = "length_m"
        length_source = "string_length" if is_string else "length_m"
        base_err = float(length_err_given) if length_err_given is not None else 0.005
        L_m = float(length_m)
        L_m_stderr = base_err
        if bob_d is not None and (is_string or bob_explicit):
            L_m = float(length_m) + float(bob_d) / 2.0
            length_source += "+bob_radius"
        elif is_string:
            systematic_note = ("length is the string length, not pivot-to-bob-centre-of-mass; "
                               "true length exceeds it by the bob radius (unknown), so g is "
                               "biased low by roughly that fractional amount")
            warnings.append(systematic_note)
    elif calib.method != "none" and L_px is not None:
        method_g = "calibration"
        length_source = "calibration_circle"
        L_m = float(L_px * calib.m_per_px)
        rel_l = math.hypot((L_px_stderr / L_px) if L_px else 0.0, abs(calib.rel_uncertainty))
        L_m_stderr = float(abs(L_m) * rel_l)

    dominant = "period"
    if L_m is not None:
        params["L_m"] = {"value": float(L_m), "stderr": float(L_m_stderr or 0.0), "unit": "m"}
        g_val = 4 * math.pi ** 2 * float(L_m) / (T0 * T0) if T0 else float("inf")
        contrib = {
            "length": (float(L_m_stderr or 0.0) / float(L_m)) if L_m else 0.0,
            "period": 2.0 * (eT / T if T else 0.0),
            "amplitude_correction": 2.0 * (sigma_corr / corr if corr else 0.0),
        }
        dominant = max(contrib, key=lambda k: contrib[k])
        rel_g = math.hypot(contrib["length"], contrib["period"], contrib["amplitude_correction"])
        params["g"] = {"value": float(g_val), "stderr": float(abs(g_val) * rel_g), "unit": "m/s^2"}
        predicted_m = None
    else:
        predicted_m = float(G_SI * T0 * T0 / (4 * math.pi ** 2)) if T0 else 0.0
        pred_err = float(abs(predicted_m) * 2.0 * rel_T0) if predicted_m else 0.0
        params["L_predicted_m"] = {"value": predicted_m, "stderr": pred_err, "unit": "m"}
        warnings.append("no length_m and no calibrated circle length: g not computed; "
                        "L_predicted_m is the length implied by g=9.81 (derived fact, not a measurement of g)")

    info: dict = {
        "axis": axis_name,
        "pivot_px": pivot_px,
        "L_px": float(L_px) if L_px is not None else None,
        "L_px_stderr": float(L_px_stderr) if L_px_stderr is not None else None,
        "arc_span_deg": float(arc_span_deg) if arc_span_deg is not None else None,
        "method_g": method_g,
        "dominant_error": dominant,
        "finite_amplitude_correction": float(corr),
        "theta_fit": bool(used_theta),
        "length_source": length_source,
        "T_halves": honest["T_halves"],
        "sigma_stat": float(honest["sigma_stat"]),
        "sigma_sys": float(honest["sigma_sys"]),
        "chi2_red": float(honest["chi2_red"]),
        "inflation_factor": float(honest["inflation_factor"]),
    }
    if bob_d is not None and length_m is not None and (is_string or bob_explicit):
        info["bob_diameter_m"] = float(bob_d)
    if systematic_note:
        info["length_systematic_note"] = systematic_note
    if L_m is not None:
        info["L_m"] = float(L_m)
    else:
        info["predicted_length_m"] = float(predicted_m)
        info["predicted_length_for_g_m"] = float(predicted_m)
    return Fit(model="pendulum", track_id=track.id, t_range=[float(t[0]), float(t[-1])],
               params=params, r2=float(r2), rmse_px=float(rmse_px), n_points=int(len(t)),
               curve=curve, residuals=resid_list, warnings=warnings, info=info)


def fit_oscillator(track, calib=None, t_range=None, options=None) -> Fit:
    return _fit_damped(track, calib, t_range, "oscillator", options)


# ---------------------------------------------------------------------------
# bounce
# ---------------------------------------------------------------------------

def _velocity_stderr(cov, t: float) -> float:
    """Std-dev of v(t)=2*a2*t+a1 for y=a2*t^2+a1*t+a0, from the full 3x3 covariance.

    Includes the linear-coefficient variance and the a2/a1 covariance term;
    using only (2t*sigma_a2) understates the error whenever sigma_a1 > 0.
    """
    try:
        c = np.asarray(cov, dtype=float)
        if c.shape != (3, 3):
            return 0.0
        var = (2.0 * t) ** 2 * c[0, 0] + c[1, 1] + 2.0 * (2.0 * t) * c[0, 1]
        return float(math.sqrt(max(var, 0.0)))
    except Exception:
        return 0.0


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
            # Prominence: drop to the higher of the nearest minima on each side.
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
    curve_x: list[float] = []
    apex_heights: list[float] = []

    for s_idx in range(len(bounds_t) - 1):
        t0, t1 = bounds_t[s_idx], bounds_t[s_idx + 1]
        # Trim 2 frames at impact ends (contact / deformation), keep edges of clip.
        trim = 2.0 / fps if fps else 0.0
        lo = t0 + (trim if s_idx > 0 else 0.0)
        hi = t1 - (trim if s_idx < len(bounds_t) - 2 else 0.0)
        sel = (t_all >= lo) & (t_all <= hi)
        tt, yy, xx = t_all[sel], y_all[sel], x_all[sel]
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
            cov = np.zeros((3, 3))
            errs = np.zeros(3)
        # Honest segment errors: inflate by sqrt(chi2_red) on structured residuals.
        _seg_factor, _seg_chi2 = _residual_inflation(yy[keep] - np.polyval(coeffs, tt[keep]), 3)
        if _seg_factor != 1.0:
            errs = np.asarray(errs, dtype=float) * _seg_factor
            cov = np.asarray(cov, dtype=float) * (_seg_factor ** 2)
        a_px = float(2.0 * coeffs[0])
        a_err = float(2.0 * errs[0]) if len(errs) > 0 else 0.0
        yfit = np.polyval(coeffs, tt)
        r2s, rmses = _r2_rmse(yy, yfit)
        if len(tt) < 9:
            seg.setdefault("warnings", []).append("only 9 points in flight" if len(tt) == 9 else f"only {len(tt)} points in flight")
        seg.update({"g_px": a_px, "g_px_stderr": a_err, "coeffs": [float(c) for c in coeffs],
                    "coeff_stderr": [float(e) for e in errs],
                    "coeff_cov": [[float(v) for v in row] for row in np.asarray(cov, dtype=float).reshape(3, 3)],
                    "r2": r2s, "rmse_px": rmses, "chi2_red": float(_seg_chi2),
                    "inflation_factor": float(_seg_factor),
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
        try:
            cx_seg, _, _ = _robust_polyfit(tt, xx, 1)
        except Exception:
            cx_seg = np.array([0.0, float(xx[0]) if len(xx) else 0.0])
        seg["curve"] = {"t": [float(v) for v in tc],
                        "x": [float(v) for v in np.polyval(cx_seg, tc)],
                        "y": [float(v) for v in np.polyval(coeffs, tc)]}
        curve_x.extend([float(v) for v in np.polyval(cx_seg, tc)])

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
            if v_before > 0 and v_after < 0:
                e = abs(v_after) / v_before
                if e > 1.05:
                    entry["valid"] = False
                    entry["warning"] = f"e={e:.2f} > 1 is unphysical; suspect segment boundary or perspective"
                else:
                    entry["valid"] = True
                # Full propagation: v=2*a2*t+a1 per segment (segments independent,
                # so no cross-segment covariance), including sigma_a1 and Cov(a2,a1).
                vb_err = _velocity_stderr(before.get("coeff_cov"), it)
                va_err = _velocity_stderr(after.get("coeff_cov"), it)
                if vb_err == 0.0 and va_err == 0.0:  # legacy segments without cov
                    eb = before["coeff_stderr"][0] if len(before.get("coeff_stderr", [])) > 0 else 0.0
                    ea = after["coeff_stderr"][0] if len(after.get("coeff_stderr", [])) > 0 else 0.0
                    vb_err, va_err = abs(2 * it) * eb, abs(2 * it) * ea
                e_err = e * math.hypot(va_err / abs(v_after) if v_after else 0, vb_err / v_before if v_before else 0)
                entry.update({"e": float(e), "e_stderr": float(e_err),
                              "v_before_px_s": v_before, "v_after_px_s": v_after})
                if before.get("excluded_from_pool") or after.get("excluded_from_pool"):
                    entry["valid"] = False
                    entry["warning"] = (entry.get("warning", "") + " neighbouring flight excluded from pool; restitution unreliable").strip()
            else:
                entry["warning"] = f"velocity signs unexpected at impact (v_before={v_before:.1f}, v_after={v_after:.1f})"
        # Height cross-check e_height is attached below from apex_heights.
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
    curve = {"t": [curve_t[i] for i in order], "x": [curve_x[i] for i in order],
             "y": [curve_y[i] for i in order]} if curve_t else {}
    impacts: list[dict] = []
    for entry in restitution:
        it = float(entry["t"])
        ix, iy = _track_pos_at(track, it)
        impacts.append({
            "t": it, "x": ix, "y": iy,
            "e": float(entry["e"]) if "e" in entry else None,
            "e_stderr": float(entry.get("e_stderr", 0.0)) if "e" in entry else None,
            "e_height": float(entry["e_height"]) if "e_height" in entry else None,
            "valid": bool(entry.get("valid", False)) if "e" in entry else False,
        })
    fit = Fit(model="bounce", track_id=track.id, t_range=[float(t_all[0]), float(t_all[-1])],
              params=params, r2=float(segments[0].get("r2", 0.0)) if segments else 0.0,
              rmse_px=rmse_all, n_points=int(len(t_all)), curve=curve,
              residuals=[round(float(v), 4) for v in all_resid], warnings=warnings,
              segments=segments, restitution=restitution, impact_times=impact_times,
              impacts=impacts, info={"axis": "y"})
    return fit


# ---------------------------------------------------------------------------
# draining (Torricelli)
# ---------------------------------------------------------------------------

G_CGS = 981.0
DRAIN_FRACTIONS = (10, 25, 50, 75, 90)


def torricelli_tf_theory(h0_m: float, container_diameter_m: float, hole_diameter_m: float,
                         g: float = G_SI) -> float:
    """Ideal (Cd=1) drain time: sqrt((2h0/g)*(A^2/a^2 - 1)) — the course formula."""
    A = math.pi * (float(container_diameter_m) / 2.0) ** 2
    a = math.pi * (float(hole_diameter_m) / 2.0) ** 2
    return float(math.sqrt((2.0 * float(h0_m) / g) * ((A * A) / (a * a) - 1.0)))


def newton_raphson(f, df, x0: float, tol: float = 1e-10, max_iter: int = 100):
    """Newton-Raphson root find (mirrors the FIUBA course's codigo.py)."""
    x = float(x0)
    for _ in range(max_iter):
        fx, dfx = float(f(x)), float(df(x))
        if dfx == 0:
            break
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return float(x_new)
        x = x_new
    return float(x)


def _torricelli_h(t, h0, k, t0):
    # h = h0 for t < t0 (not yet draining), then (sqrt(h0)-k(t-t0))^2.
    # The plateau is what makes h0 and t0 separately identifiable.
    dt = np.clip(np.asarray(t, dtype=float) - float(t0), 0.0, None)
    u = np.sqrt(max(float(h0), 0.0)) - float(k) * dt
    return np.where(u > 0, u ** 2, 0.0)


def _fit_torricelli(t, h):
    """Fit sqrt(h)=sqrt(h0)-k(t-t0); returns (popt, perr, yfit)."""
    t = np.asarray(t, float); h = np.asarray(h, float)
    hmax = float(np.max(h)) if len(h) else 1.0
    dur = float(t[-1] - t[0]) if len(t) > 1 else 1.0
    p0 = [hmax, math.sqrt(max(hmax, 1e-12)) / max(dur * 0.8, 1e-9), float(t[0])]
    bounds = ([0.2 * hmax, 1e-12, float(t[0]) - max(dur, 1.0)],
              [1.5 * hmax, np.inf, float(t[0]) + max(dur, 1.0)])
    popt, pcov = curve_fit(_torricelli_h, t, h, p0=p0, bounds=bounds, maxfev=40000)
    perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else np.zeros(3)
    # h0/t0 are only separately identifiable when a pre-drain plateau is
    # observed; otherwise the curve depends only on sqrt(h0)+k*t0. Fall
    # back to t0 = t[0] (draining from the first frame) when the 3-param
    # covariance is degenerate.
    if (not np.all(np.isfinite(perr)) or perr[0] > abs(popt[0]) or perr[2] > max(dur, 1.0)):
        t_fixed = float(t[0])

        def _h2(tt, h0_, k_):
            return _torricelli_h(tt, h0_, k_, t_fixed)

        try:
            p2, pc2 = curve_fit(_h2, t, h, p0=[popt[0], popt[1]],
                                bounds=([0.2 * hmax, 1e-12], [1.5 * hmax, np.inf]), maxfev=40000)
            e2 = np.sqrt(np.diag(pc2)) if pc2 is not None and np.all(np.isfinite(pc2)) else np.zeros(2)
            popt = np.array([p2[0], p2[1], t_fixed])
            perr = np.array([e2[0], e2[1], 0.0])
        except Exception:
            pass
    return popt, perr, _torricelli_h(t, *popt)


def _empirical_fits(t, h_norm):
    """Quadratic/cubic/exponential fits on h/h0 with R^2 (course's fits)."""
    out = {}
    for name, deg in (("quadratic", 2), ("cubic", 3)):
        try:
            coeffs = np.polyfit(t, h_norm, deg)
            pred = np.polyval(coeffs, t)
            r2, _ = _r2_rmse(h_norm, pred)
            out[name] = {"coeffs": [float(c) for c in coeffs], "r2": float(r2),
                         "degree": deg}
        except Exception as exc:
            out[name] = {"error": str(exc), "r2": None}
    try:
        pos = h_norm > 1e-6
        coeffs_e = np.polyfit(t[pos], np.log(h_norm[pos]), 1)  # log h = b*t + a
        pred_e = np.exp(np.polyval(coeffs_e, t))
        r2e, _ = _r2_rmse(h_norm, pred_e)
        out["exponential"] = {"coeffs": [float(coeffs_e[1]), float(coeffs_e[0])],
                              "r2": float(r2e), "form": "exp(a+b*t)"}
    except Exception as exc:
        out["exponential"] = {"error": str(exc), "r2": None}
    return out


def _times_from_cubic(coeffs, fractions=DRAIN_FRACTIONS, t_max: float = 100.0):
    """Times to X% drained from a cubic fit on h/h0 via Newton-Raphson."""
    c = np.asarray(coeffs, float)  # highest-degree first (np.polyfit)
    poly = np.poly1d(c)
    dpoly = np.polyder(poly)
    out = {}
    guesses = {10: 2.0, 25: 5.0, 50: 10.0, 75: 20.0, 90: 30.0}
    for frac in fractions:
        target = 1.0 - frac / 100.0
        try:
            root = newton_raphson(lambda x: float(poly(x)) - target,
                                  lambda x: float(dpoly(x)), guesses.get(frac, 5.0))
            if not (0 <= root <= max(t_max * 2, 200)):
                raise ValueError("root out of range")
        except Exception:
            root = None
            # Brent fallback on a scan.
            try:
                from scipy.optimize import brentq
                ts = np.linspace(0, max(t_max, 1.0), 2000)
                vals = poly(ts) - target
                for i in range(len(ts) - 1):
                    if vals[i] == 0 or vals[i] * vals[i + 1] < 0:
                        root = float(brentq(lambda x: float(poly(x)) - target, ts[i], ts[i + 1]))
                        break
            except Exception:
                root = None
        out[f"t{frac}"] = (float(root) if root is not None and np.isfinite(root) else None)
    return out


def solve_generalized_vessel(t_eval, h0, Cd, width_of_h, hole_diameter_m,
                             g: float = G_SI, rtol: float = 1e-8):
    """Solve A(h) dh/dt = -Cd*a*sqrt(2 g h), A=pi*(w(h)/2)^2.

    ``width_of_h(h)`` returns vessel width (same length unit as ``h`` and
    ``hole_diameter_m``). Returns h(t_eval) via solve_ivp (RK45).
    """
    from scipy.integrate import solve_ivp
    a_hole = math.pi * (float(hole_diameter_m) / 2.0) ** 2

    def rhs(tt, yy):
        h = max(float(yy[0]), 0.0)
        w = float(width_of_h(h))
        A = math.pi * (w / 2.0) ** 2 if w > 0 else 0.0
        if A <= 0 or h <= 0:
            return [0.0]
        return [float(-Cd * a_hole * math.sqrt(2.0 * g * h) / A)]

    t_eval = np.asarray(t_eval, float)
    sol = solve_ivp(rhs, [float(t_eval[0]), float(t_eval[-1])], [float(h0)],
                    method="RK45", rtol=rtol, atol=1e-10, t_eval=t_eval)
    return np.maximum(sol.y[0], 0.0)


def fit_generalized_vessel(t, h, width_of_h, hole_diameter_m, g: float = G_SI):
    """Fit (Cd, h0) for the generalized vessel; returns dict."""
    from scipy.optimize import least_squares
    t = np.asarray(t, float); h = np.asarray(h, float)
    hmax = float(np.max(h))

    def resid(p):
        Cd, h0 = float(p[0]), float(p[1])
        try:
            pred = solve_generalized_vessel(t, h0, Cd, width_of_h, hole_diameter_m, g=g)
        except Exception:
            return np.full_like(h, 1e3)
        return pred - h

    res = least_squares(resid, x0=[0.6, hmax], bounds=([0.01, 0.2 * hmax], [2.0, 1.5 * hmax]))
    Cd, h0 = float(res.x[0]), float(res.x[1])
    pred = solve_generalized_vessel(t, h0, Cd, width_of_h, hole_diameter_m, g=g)
    r2, rmse = _r2_rmse(h, pred)
    return {"Cd": Cd, "h0": h0, "curve_h": pred, "r2": float(r2), "rmse": float(rmse)}


def _load_reference_csv(path):
    """Read the students' CSV: Slice (1-based @fps), Length (cm), Tiempo_s."""
    import csv as _csv
    slices, lengths, times = [], [], []
    with open(path, newline="") as fh:
        for row in _csv.DictReader(fh):
            try:
                slices.append(int(float(row["Slice"])))
                lengths.append(float(row["Length"]))
                times.append(float(row["Tiempo_s"]))
            except Exception:
                continue
    return np.asarray(slices), np.asarray(lengths, float), np.asarray(times, float)


def validate_against_reference(t_level, h_level, frames_level, calib, reference_csv,
                               level_fps: float | None = None):
    """Validation vs manual ImageJ data (spec §3.6).

    Interpolates Physicam's h at the students' frames and reports RMS/max
    differences in cm (when calibrated) and in h/h0 (always), plus times
    to X% computed the same way (cubic+Newton) from the 27 manual points.
    """
    slices, length_cm, t_ref = _load_reference_csv(reference_csv)
    if len(slices) == 0:
        return {"error": "empty reference csv", "n": 0}
    frames_level = np.asarray(frames_level, float)
    h_level = np.asarray(h_level, float)
    order = np.argsort(frames_level)
    h_at = np.interp(slices - 1, frames_level[order], h_level[order])  # Slice is 1-based
    # h_level is in metres if calib given else px; convert both ways.
    if calib is not None and getattr(calib, "method", "none") != "none":
        h_at_cm = h_at * 100.0 if np.max(np.abs(h_at)) < 5 else h_at  # already m -> cm
        # h_at is in h_level units: if those are metres, *100; physics passes metres.
        rms_cm = float(np.sqrt(np.mean((h_at_cm - length_cm) ** 2)))
        max_cm = float(np.max(np.abs(h_at_cm - length_cm)))
    else:
        rms_cm = max_cm = None
    norm_auto = h_at / h_at[0] if h_at[0] else h_at * 0
    norm_ref = length_cm / length_cm[0] if length_cm[0] else length_cm * 0
    rms_norm = float(np.sqrt(np.mean((norm_auto - norm_ref) ** 2)))
    max_norm = float(np.max(np.abs(norm_auto - norm_ref)))
    # Times to X% from the students' points: cubic on h/h0 + Newton-Raphson.
    try:
        coeffs_ref = np.polyfit(t_ref, norm_ref, 3)
        times_ref = _times_from_cubic(coeffs_ref, t_max=float(t_ref[-1]) if len(t_ref) else 50.0)
    except Exception:
        times_ref = {}
    return {
        "n": int(len(slices)), "rms_cm": rms_cm, "max_cm": max_cm,
        "rms_norm": rms_norm, "max_norm": max_norm, "rms_h_over_h0": rms_norm, "max_h_over_h0": max_norm,
        "times_ref": times_ref,
        "points": {"t": [float(v) for v in t_ref], "h_cm": [float(v) for v in length_cm],
                   "h_norm": [float(v) for v in norm_ref],
                   "physicam_h_norm": [float(v) for v in norm_auto]},
        "credit": "manual ImageJ measurements by the FIUBA student team",
    }


def fit_draining_arrays(t, h, calib=None, options: dict | None = None,
                        width_profile: dict | None = None, hole_y=None,
                        track_id: str = "level") -> Fit:
    """Torricelli draining fit on (t, h) arrays.

    ``h`` is in metres when *calib* is calibrated (caller converts), else
    px. With calibration the fit runs in metres (documented form 2);
    without it the scale-free normalized form is fitted and Cd needs the
    container/hole diameters plus a scale for h0 (reported unavailable
    otherwise). ``options``: hole_diameter_m, container_diameter_m,
    sigma_d_m (0.0001), sigma_D_m (0.001), reference_csv, g.
    """
    options = dict(options or {})
    calib = calib or Calibration(1.0, 0.0, "none", "")
    calibrated = calib.method != "none" and calib.m_per_px != 1.0 or calib.method not in ("none",)
    # Actually: calibrated iff method != none.
    calibrated = calib.method != "none"
    g = float(options.get("g", G_SI))
    t = np.asarray(t, float); h = np.asarray(h, float)
    ok = np.isfinite(t) & np.isfinite(h)
    t, h = t[ok], h[ok]
    order = np.argsort(t); t, h = t[order], h[order]
    if len(t) < 5:
        raise ValueError("draining: need >= 5 level points")
    warnings: list[str] = []
    # Fit.
    popt, perr, hfit = _fit_torricelli(t, h)
    h0, k, t0 = [float(v) for v in popt]
    eh0, ek, et0 = [float(v) for v in perr]
    resid = h - hfit
    factor, chi2_red = _residual_inflation(resid, 3)
    # curve_fit's covariance is already chi2-inflated (scaled by RSS/dof);
    # `factor` (RMS vs white-noise sigma) is reported as a diagnostic in
    # info, and the split-half term below carries the model-mismatch
    # systematic. Multiplying again would double-count the mismatch.
    perr_i = np.array([eh0, ek, et0])
    r2, rmse = _r2_rmse(h, hfit)
    _structure_warning(resid, t, warnings)
    if not options.get("_hole_estimated", True):
        warnings.append("hole level from the Torricelli asymptote (clip ended early); h0/k absorb the offset")
    # Split-half systematic on k: refit only k on each half with the
    # global h0/t0 fixed (a free h0/t0 per half is degenerate — the
    # second half never sees the plateau or the start of draining).
    sigma_sys_k = 0.0
    halves_k: list[float] = []
    mid = 0.5 * (t[0] + t[-1])
    for mask in (t <= mid, t > mid):
        if int(mask.sum()) >= 5:
            try:
                from scipy.optimize import curve_fit as _cf

                def _hk(tt, k_, _h0=h0, _t0=t0):
                    return _torricelli_h(tt, _h0, k_, _t0)

                pk, _ = _cf(_hk, t[mask], h[mask], p0=[k], bounds=([1e-12], [np.inf]), maxfev=20000)
                halves_k.append(float(pk[0]))
            except Exception:
                pass
    if len(halves_k) == 2:
        sigma_sys_k = abs(halves_k[0] - halves_k[1]) / 2.0
    ek_total = float(math.hypot(ek, sigma_sys_k))

    d_m = _opt_float(options, "hole_diameter_m", "hole_diameter", "d_m")
    D_m = _opt_float(options, "container_diameter_m", "container_diameter", "D_m")
    sigma_d = float(options.get("sigma_d_m", options.get("hole_diameter_stderr_m", 0.0001)))
    sigma_D = float(options.get("sigma_D_m", options.get("container_diameter_stderr_m", 0.001)))

    unit_h = "m" if calibrated else "px"
    sqrt_h0 = math.sqrt(max(h0, 0.0))
    t_f = float(t0 + sqrt_h0 / k) if k > 0 else float("inf")
    # t_f stderr: propagate h0/k/t0 (ignore covariances, inflated already).
    et_f = float(math.hypot(et0, (0.5 / (sqrt_h0 * k) * eh0) if sqrt_h0 and k else 0.0,
                            (sqrt_h0 / (k * k) * ek_total) if k else 0.0)) if np.isfinite(t_f) else 0.0

    Cd_val, Cd_err, Cd_method = None, 0.0, "unavailable_no_scale"
    t_f_theory = None
    if d_m and D_m and d_m > 0 and D_m > 0:
        r_ratio = (float(d_m) / float(D_m)) ** 2  # a/A
        if calibrated:
            # Form 2 in metres: Cd = k / (r*sqrt(g/(2(1-r^2))))
            denom = r_ratio * math.sqrt(g / (2.0 * (1.0 - r_ratio ** 2)))
            Cd_val = float(k / denom) if denom else None
            Cd_method = "metres: Cd=k/(r*sqrt(g/(2(1-r^2))))"
            t_f_theory = torricelli_tf_theory(h0, float(D_m), float(d_m), g)
        else:
            # Scale-free form needs h0 in metres for t_f,theory; without a
            # scale Cd cannot be formed. Documented limitation.
            warnings.append("Cd needs a length scale (calibration or h0_m/container width); "
                            "scale-free t_f reported in px-fit units only")
        if Cd_val is not None and Cd_val == Cd_val:
            rel_k = (ek_total / k) if k else 0.0
            rel_r = 2.0 * math.hypot(sigma_d / float(d_m), sigma_D / float(D_m))
            rel_cal_k = 0.5 * abs(calib.rel_uncertainty) if calibrated else 0.0
            Cd_err = float(abs(Cd_val) * math.sqrt(rel_k ** 2 + rel_r ** 2 + rel_cal_k ** 2))
            # Cross-check scale-free Cd = t_f,theory / t_f.
            if t_f_theory and np.isfinite(t_f) and t_f > 0:
                Cd_scalefree = float(t_f_theory / (t_f - t0 + (t_f_theory * 0))) if False else float(t_f_theory / (float(t0 + sqrt_h0 / k) - 0.0)) if False else None
                # t_f includes t0 offset; theory is a duration. Use durations:
                Cd_sf = float(t_f_theory / (sqrt_h0 / k)) if k else None
                Cd_method += f"; scale-free check Cd=t_f,theory/t_f_duration={Cd_sf:.4f}" if Cd_sf else ""

    # Analytic times to X% drained (height) from the Torricelli fit.
    times_torr, times_torr_err = {}, {}
    dur_tf = (sqrt_h0 / k) if k else float("inf")
    for frac in DRAIN_FRACTIONS:
        tx = float(t0 + dur_tf * (1.0 - math.sqrt(1.0 - frac / 100.0))) if np.isfinite(dur_tf) else None
        times_torr[f"t{frac}"] = tx
        times_torr_err[f"t{frac}"] = float(abs(tx - t0) * (ek_total / k)) if (tx is not None and k) else 0.0

    # Empirical fits on h/h0 + cubic Newton times.
    h_norm = h / h0 if h0 else h * 0
    empirical = _empirical_fits(t, h_norm)
    times_cubic = {}
    if empirical.get("cubic", {}).get("coeffs"):
        times_cubic = _times_from_cubic(empirical["cubic"]["coeffs"], t_max=float(t[-1]))

    # Generalized vessel (when a width profile + hole are available).
    generalized = None
    if width_profile and width_profile.get("width_px") is not None and d_m:
        try:
            wp_h = np.asarray(width_profile.get("h_px", width_profile.get("h", [])), float)
            wp_w = np.asarray(width_profile.get("width_px", []), float)
            good = np.isfinite(wp_h) & np.isfinite(wp_w) & (wp_w > 0)
            if int(good.sum()) >= 3:
                # Work in the same units as h: profile h is in px; convert
                # with calib when the fit is in metres.
                if calibrated:
                    wp_h_m = wp_h * float(calib.m_per_px); wp_w_m = wp_w * float(calib.m_per_px)
                    hh, dd = wp_h_m, float(d_m)
                else:
                    wp_h_m, wp_w_m, hh, dd = wp_h, wp_w, wp_h, float(d_m) / float(calib.m_per_px) if calib.m_per_px else float(d_m)
                    wp_h_m, wp_w_m = wp_h, wp_w
                order_w = np.argsort(wp_h_m[good])
                xs, ws = wp_h_m[good][order_w], wp_w_m[good][order_w]
                # Deduplicate xs for interp.
                uniq, idx_u = np.unique(xs, return_index=True)
                ws_u = ws[idx_u]

                def _width_of_h(hv, _xs=uniq, _ws=ws_u):
                    return float(np.interp(hv, _xs, _ws, left=_ws[0], right=_ws[-1]))

                gen = fit_generalized_vessel(t, h, _width_of_h, float(d_m) if calibrated else float(d_m) / (float(calib.m_per_px) or 1.0) if False else (float(d_m) if calibrated else float(d_m) * 1.0), g=g if calibrated else g)
                # NOTE: in px units g must be in px/s^2; caller converts.
                generalized = {"Cd": gen["Cd"], "h0": gen["h0"], "r2": gen["r2"], "rmse": gen["rmse"]}
        except Exception as exc:
            generalized = {"error": str(exc)}

    # Ideal (Cd=1) theory curve for the UI (dashed).
    theory_curve = None
    if t_f_theory is not None and np.isfinite(t_f_theory):
        tc_th = np.linspace(float(t[0]), float(t[0] + t_f_theory * 1.05), 100)
        # Ideal curve starts at t0 with same h0.
        uth = np.sqrt(max(h0, 0.0)) - (1.0 * (k / Cd_val if Cd_val else k)) * (tc_th - t0) if Cd_val else None
        if uth is not None:
            theory_curve = {"t": [float(v) for v in tc_th],
                            "h": [float(v) for v in np.where(uth > 0, uth ** 2, 0.0)]}

    # Validation vs manual data.
    validation = None
    ref_csv = options.get("reference_csv")
    if ref_csv:
        try:
            frames_opt = options.get("_frames")
            fps_opt = options.get("_fps")
            if frames_opt is not None:
                h_for_val = h  # already in fit units (m when calibrated)
                validation = validate_against_reference(t, h_for_val, np.asarray(frames_opt, float),
                                                        calib if calibrated else None, str(ref_csv))
            else:
                # Fall back to time-based frames at 60 fps convention.
                fps_guess = float(fps_opt or 60.0)
                validation = validate_against_reference(
                    t, h, np.round(t * fps_guess).astype(float), calib if calibrated else None, str(ref_csv))
        except Exception as exc:
            validation = {"error": str(exc), "n": 0}

    params: dict = {
        "Cd": {"value": (float(Cd_val) if Cd_val is not None else None), "stderr": float(Cd_err), "unit": ""},
        "t_f": {"value": float(t_f), "stderr": float(et_f), "unit": "s"},
        "k": {"value": float(k), "stderr": float(ek_total), "unit": f"{unit_h}^0.5/s"},
        "h0": {"value": float(h0), "stderr": float(eh0), "unit": unit_h},
        "t0": {"value": float(t0), "stderr": float(et0), "unit": "s"},
    }
    if t_f_theory is not None:
        params["t_f_theory"] = {"value": float(t_f_theory), "stderr": 0.0, "unit": "s"}
    for frac in DRAIN_FRACTIONS:
        v = times_torr.get(f"t{frac}")
        params[f"t{frac}"] = {"value": (float(v) if v is not None else None),
                              "stderr": float(times_torr_err.get(f"t{frac}", 0.0)), "unit": "s"}

    tc = np.linspace(float(t[0]), float(t[-1]), 200)
    curve = {"t": [float(v) for v in tc], "h": [float(v) for v in _torricelli_h(tc, h0, k, t0)]}
    # rmse in px for the Fit contract: convert when calibrated.
    rmse_px = float(rmse / calib.m_per_px) if (calibrated and calib.m_per_px) else float(rmse)
    info: dict = {
        "axis": "h", "model_form": Cd_method, "chi2_red": float(chi2_red),
        "inflation_factor": float(factor), "k_halves": [float(v) for v in halves_k],
        "sigma_sys_k": float(sigma_sys_k),
        "empirical": empirical, "times_torricelli": times_torr,
        "times_cubic_newton": times_cubic, "t_f_theory": (float(t_f_theory) if t_f_theory is not None else None),
        "generalized_vessel": generalized, "validation": validation,
        "theory_curve": theory_curve, "hole_y": (float(hole_y) if hole_y is not None else None),
        "r_ratio_a_over_A": (float((d_m / D_m) ** 2) if (d_m and D_m) else None),
    }
    return Fit(model="draining", track_id=track_id, t_range=[float(t[0]), float(t[-1])],
               params=params, r2=float(r2), rmse_px=rmse_px, n_points=int(len(t)),
               curve=curve, residuals=[round(float(v), 4) for v in resid],
               warnings=warnings, info=info)


def fit_draining(level, calib=None, t_range=None, options: dict | None = None) -> Fit:
    """Fit the draining model to a LevelSeries (or Track-like level holder).

    Converts px -> m with *calib* when calibrated, so Cd uses the metres
    form; the scale-free check Cd = t_f,theory/t_f is recorded in info.
    The generalized-vessel sub-fit runs in the fit's units (g converted
    to px/s^2 when uncalibrated).
    """
    options = dict(options or {})
    calib = calib or Calibration(1.0, 0.0, "none", "")
    # Duck-typing: LevelSeries has t/h_px/surface_y; a Track fallback uses y.
    if hasattr(level, "h_px"):
        t_all = np.asarray(level.t, float)
        h_px_all = np.asarray(level.h_px, float)
        width_profile = getattr(level, "width_profile", None)
        hole_y = getattr(level, "hole_y", None)
        options.setdefault("_hole_estimated", bool(getattr(level, "hole_estimated", True)))
        options["_frames"] = np.asarray(getattr(level, "frames", np.round(t_all * getattr(level, "fps", 60.0))), float)
        options["_fps"] = float(getattr(level, "fps", 60.0))
        track_id = "level"
    else:
        # Track fallback: h from surface-like y (max-y minus y).
        t_all = np.asarray(level.t, float)
        y = np.asarray(level.y, float)
        h_px_all = float(np.max(y)) - y if len(y) else y
        width_profile = options.get("width_profile")
        hole_y = None
        track_id = getattr(level, "id", "level")
    mask = np.ones(len(t_all), dtype=bool)
    if t_range is not None:
        mask = (t_all >= t_range[0]) & (t_all <= t_range[1])
    t, h_px = t_all[mask], h_px_all[mask]
    if calib.method != "none":
        h_fit = h_px * float(calib.m_per_px)
        fit = fit_draining_arrays(t, h_fit, calib, options, width_profile=width_profile,
                                  hole_y=hole_y, track_id=track_id)
        # Generalized vessel in metres is already consistent (width profile
        # converted inside fit_draining_arrays via calib).
        return fit
    # Uncalibrated: generalized needs g in px/s^2 — handled by re-running
    # the generalized sub-fit here with g_px unknown, so skip it in arrays
    # by not passing a profile that triggers the metres assumption; instead
    # compute it scale-free below if possible. Simplest: pass profile but
    # with g converted using container-width scale when available.
    fit = fit_draining_arrays(t, h_px, calib, options, width_profile=None,
                              hole_y=hole_y, track_id=track_id)
    if width_profile and options.get("hole_diameter_m"):
        try:
            D_m = _opt_float(options, "container_diameter_m", "container_diameter")
            # Estimate px scale from container width if we have a median width.
            med_w = None
            # width_profile widths are px; a cylinder check uses median width.
            wp_w = np.asarray(width_profile.get("width_px", []), float)
            med_w = float(np.median(wp_w[wp_w > 0])) if wp_w.size and np.any(wp_w > 0) else None
            if D_m and med_w:
                scale = float(D_m) / med_w  # m/px (refraction caveat noted by caller)
                g_px = G_SI / scale
                wp_h = np.asarray(width_profile.get("h_px", []), float)
                good = np.isfinite(wp_h) & np.isfinite(wp_w) & (wp_w > 0)
                xs, ws = wp_h[good], wp_w[good]
                order_w = np.argsort(xs); xs, ws = xs[order_w], ws[order_w]
                uniq, idx_u = np.unique(xs, return_index=True); ws_u = ws[idx_u]

                def _w(hv, _xs=uniq, _ws=ws_u):
                    return float(np.interp(hv, _xs, _ws, left=_ws[0], right=_ws[-1]))

                d_px = float(options["hole_diameter_m"]) / scale
                gen = fit_generalized_vessel(t, h_px, _w, d_px, g=g_px)
                fit.info["generalized_vessel"] = {"Cd": gen["Cd"], "h0": gen["h0"],
                                                  "r2": gen["r2"], "rmse": gen["rmse"],
                                                  "note": "px units with g_px from container-width scale"}
        except Exception as exc:
            fit.info["generalized_vessel"] = {"error": str(exc)}
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
    if key == "draining":
        # Level-based fit: *track* may be a LevelSeries or a Track whose
        # options carry a level (options.level / options.level_series).
        opts = dict(options or {})
        level_obj = opts.pop("level", None) or opts.pop("level_series", None) or track
        return fit_draining(level_obj, calib, t_range, opts)
    if key not in _DISPATCH:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(list(_DISPATCH) + ['draining'])}")
    if calib is not None and getattr(calib, "kind", "constant") == "per_frame":
        metric, identity = _metric_track_and_calib(track, calib)
        fit = _DISPATCH[key](metric, identity, t_range, options)
        fit.warnings = list(fit.warnings) + [
            "per-frame perspective correction applied: fit in metric lateral coordinates X=(x-cx)s(t), Y=(y-cy)s(t); rmse is in metres"]
        # rmse_px now holds metres; keep field for contract compatibility.
        return fit
    fit = _DISPATCH[key](track, calib, t_range, options)
    return fit
