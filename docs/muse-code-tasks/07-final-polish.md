# Task 07: final polish (honest uncertainties, report figure, cached-cost display)

Read `AGENTS.md`, then `physicam/` and `web/`. Do not start servers on port 8000; use port 8010 if
needed.

1. **Honest uncertainty for periodic fits** (pendulum and oscillator in `physics.py`). The spring
   job reports T = 0.96310 ± 0.00028 s, which is the covariance stderr only. Its R² is 0.966, with
   structured residuals, and the live agent run found window-to-window values of 0.9603, 0.9633 and
   0.9664 s. That stderr is overconfident.
   - Also fit the first and second halves of the window (when each has ≥ 3 periods), and define
     σ_sys = |T₁ − T₂| / 2.
   - Report σ_T = sqrt(σ_stat² + σ_sys²), with `info.T_halves` and `info.sigma_stat` /
     `info.sigma_sys` shown.
   - Inflate σ_stat by sqrt(χ²_red) when χ²_red > 1. Do this for all fits (projectile, bounce
     segments, constant velocity), using the residual RMS as the noise estimate.
   - Propagate the honest σ_T into g for pendulums.
   - Update the tests: the synthetic pendulum must still recover g, and the uncertainty must cover
     the true value (|error| < 2σ).
2. **Report strobe figure** (`web/`): the strobe image at the top of the lab report is huge and
   letterboxed in black. Constrain it to `max-height: 420px`, keep its natural aspect ratio, center
   it, and give it a transparent background, rounded corners and a caption ("Stroboscopic composite
   from SAM 3.1 masks · every k-th frame"). Clicking it opens it full size in a lightbox.
3. **Cost footer for cached runs**: when a job re-ran from cache, the footer says "0 SAM frames · 0
   Spark tokens". Store the original live cost in the result (`cost.original` the first time a
   video+question ran live; carry it over on cached re-runs) and show "≈ $X (original live run) ·
   cached replay".
4. **Answer card**: if `answer.uncertainty` has more than 2 significant digits or the value has
   excess decimals, format as in task 04. Also show a one-line provenance under the headline, e.g.
   "pendulum fit on θ(t) · 18 periods · length from narration (80 cm)", built from the primary fit's
   `info` and the plan facts.
5. Run `.venv/bin/python -m pytest -q` and finish with a summary.
