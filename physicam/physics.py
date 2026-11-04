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
    other = "y" if axis == "x" else "x"
    curve = {"t": [float(v) for v in tc], axis: [float(v) for v in np.polyval(coeffs, tc)],
             other: _smoothed_other_axis(track, tc, other)}
    return Fit(model="polynomial", track_id=track.id, t_range=[float(t[0]), float(t[-1])],
               params=params, r2=r2, rmse_px=rmse, n_points=int(len(t)), curve=curve,
               residuals=[round(float(v), 4) for v in (yy - yfit)], warnings=[],
               info={"axis": axis})


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
    eT = T * (eomega / omega) if omega else 0.0
    r2, rmse = _r2_rmse(s, sfit)
    resid = s - sfit
    _structure_warning(resid, t, warnings)
    params = {
        "T": {"value": float(T), "stderr": float(eT), "unit": "s"},
        "omega": {"value": float(omega), "stderr": float(eomega), "unit": "rad/s"},
        "gamma": {"value": float(gamma), "stderr": float(egamma), "unit": "1/s"},
        "amplitude": _param(A, eA, "px", calib, "m", 1),
    }
    info = {"axis": axis_name}
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
    eT = float(T * (eomega / omega)) if omega else 0.0
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
        a_px = float(2.0 * coeffs[0])
        a_err = float(2.0 * errs[0]) if len(errs) > 0 else 0.0
        yfit = np.polyval(coeffs, tt)
        r2s, rmses = _r2_rmse(yy, yfit)
        if len(tt) < 9:
            seg.setdefault("warnings", []).append("only 9 points in flight" if len(tt) == 9 else f"only {len(tt)} points in flight")
        seg.update({"g_px": a_px, "g_px_stderr": a_err, "coeffs": [float(c) for c in coeffs],
                    "coeff_stderr": [float(e) for e in errs],
                    "coeff_cov": [[float(v) for v in row] for row in np.asarray(cov, dtype=float).reshape(3, 3)],
                    "r2": r2s, "rmse_px": rmses,
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
