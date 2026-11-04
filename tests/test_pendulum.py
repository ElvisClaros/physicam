"""Synthetic pendulum tests: nonlinear simulation, no network.

Integrates theta'' = -(g/L) sin(theta) - 2*gamma*theta' at 60 fps.
L = 0.80 m, theta0 = 25 deg, pixel noise sigma = 0.8 px unless stated.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from physicam.calibrate import Calibration
from physicam.physics import fit_model
from physicam.sam import Track

G_TRUE = 9.81
L_TRUE = 0.80
FPS = 60.0
M_PER_PX = 0.0016  # L_px = 500 px
PIVOT = (640.0, 80.0)


def _simulate(theta0_deg: float = 25.0, gamma: float = 0.01, duration: float = 12.0,
              seed: int = 7, sigma: float = 0.8):
    theta0 = np.radians(theta0_deg)

    def rhs(t, z):
        return [z[1], -(G_TRUE / L_TRUE) * np.sin(z[0]) - 2.0 * gamma * z[1]]

    t = np.arange(0, int(duration * FPS)) / FPS
    sol = solve_ivp(rhs, [0.0, float(t[-1])], [theta0, 0.0], t_eval=t,
                    rtol=1e-10, atol=1e-12)
    assert sol.success
    theta = sol.y[0]
    l_px = L_TRUE / M_PER_PX
    x = PIVOT[0] + l_px * np.sin(theta)
    y = PIVOT[1] + l_px * np.cos(theta)
    rng = np.random.default_rng(seed)
    x = x + rng.normal(0, sigma, size=x.shape)
    y = y + rng.normal(0, sigma, size=y.shape)
    return t, x, y


def _track(t, x, y):
    n = len(t)
    area = np.full(n, np.pi * 20.0 ** 2)
    return Track(id="t0", label="pendulum bob", frames=np.round(t * FPS).astype(int),
                 t=np.asarray(t, float), x=np.asarray(x, float), y=np.asarray(y, float),
                 area=area, bbox=np.zeros((n, 4)), conf=np.full(n, 0.9),
                 eq_diameter=np.full(n, 40.0), contours={}, fps=FPS)


def test_pendulum_length_m_recovers_g_and_correction_matters():
    t, x, y = _simulate(theta0_deg=25.0)
    fit = fit_model(_track(t, x, y), "pendulum", Calibration(1.0, 0.0, "none", ""),
                    options={"length_m": L_TRUE})
    g = fit.params["g"]["value"]
    err_corr = abs(g - G_TRUE) / G_TRUE
    assert err_corr < 0.005, f"g={g} err={err_corr:.4%} params={fit.params} warnings={fit.warnings}"
    # Without the finite-amplitude correction, g from the raw period is worse.
    period = fit.params["T"]["value"]
    g_uncorr = 4 * np.pi ** 2 * L_TRUE / period ** 2
    err_uncorr = abs(g_uncorr - G_TRUE) / G_TRUE
    assert err_uncorr > err_corr, f"uncorr {err_uncorr:.4%} vs corr {err_corr:.4%}"
    assert err_uncorr > 0.01, f"expected >1% uncorrected error, got {err_uncorr:.4%}"
    assert fit.params["theta0_deg"]["value"] > 15.0
    assert fit.info["method_g"] == "length_m"
    assert fit.info["dominant_error"] == "length"


def test_pendulum_calibration_circle_recovers_length():
    t, x, y = _simulate(theta0_deg=25.0)
    calib = Calibration(M_PER_PX, 0.002, "reference", "synthetic")
    fit = fit_model(_track(t, x, y), "pendulum", calib)
    l_m = fit.params["L_m"]["value"]
    assert abs(l_m - L_TRUE) / L_TRUE < 0.02, f"L_m={l_m} info={fit.info}"
    assert abs(fit.info["L_px"] - L_TRUE / M_PER_PX) / (L_TRUE / M_PER_PX) < 0.02
    pivot = fit.info["pivot_px"]
    assert pivot[1] < float(np.median(y)), "pivot must be above the bob"
    assert abs(pivot[0] - PIVOT[0]) < 20.0 and abs(pivot[1] - PIVOT[1]) < 20.0
    assert abs(fit.params["g"]["value"] - G_TRUE) / G_TRUE < 0.02


def test_pendulum_honest_uncertainty_covers_g():
    import math

    t, x, y = _simulate(theta0_deg=25.0)
    fit = fit_model(_track(t, x, y), "pendulum", Calibration(1.0, 0.0, "none", ""),
                    options={"length_m": L_TRUE})
    g = fit.params["g"]
    assert abs(g["value"] - G_TRUE) < 2 * g["stderr"], (
        f"g={g['value']}±{g['stderr']} does not cover {G_TRUE}")
    # Half-split systematic is reported and combined in quadrature.
    assert fit.info["T_halves"] is not None and len(fit.info["T_halves"]) == 2
    assert fit.info["sigma_sys"] == abs(fit.info["T_halves"][0] - fit.info["T_halves"][1]) / 2
    assert fit.info["sigma_stat"] > 0
    expected = math.hypot(fit.info["sigma_stat"], fit.info["sigma_sys"])
    assert abs(fit.params["T"]["stderr"] - expected) < 1e-12
    # g must use the honest period error (not the covariance-only value):
    # with a known length, period contributes 2*sigma_T/T to g.
    assert g["stderr"] > 0


def test_pendulum_short_arc_warns_and_no_bogus_length():
    t, x, y = _simulate(theta0_deg=4.0, seed=11)
    calib = Calibration(M_PER_PX, 0.002, "reference", "synthetic")
    fit = fit_model(_track(t, x, y), "pendulum", calib)
    text = " ".join(fit.warnings).lower()
    assert "arc" in text, f"warnings={fit.warnings}"
    assert fit.info["L_px"] is None, f"info={fit.info}"
    assert "L_m" not in fit.params and "g" not in fit.params, fit.params


def test_spoken_length_fact_extraction_and_fallback_options():
    from physicam.agent import _pendulum_fit_options
    from physicam.plan import Plan

    plan = Plan.from_dict({
        "experiment": "pendulum", "phenomenon": "pendulum", "moving_objects": [],
        "reference_candidates": [], "spoken_facts": "the string is 80 cm",
        "recommended_model": "pendulum", "target_quantity": {"quantity": "g", "unit": "m/s^2"},
    })
    assert plan.facts_extracted, plan.to_dict()
    fact = plan.facts_extracted[0]
    assert fact["quantity"] == "length" and abs(fact["value_m"] - 0.80) < 1e-9
    assert "string" in fact["refers_to"]
    opts = _pendulum_fit_options(plan, {})
    assert opts is not None and abs(opts["length_m"] - 0.80) < 1e-9
