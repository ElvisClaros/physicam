# Task 06: the pendulum measures g (with the length heard or measured)

Read `AGENTS.md`, `physicam/physics.py`, `physicam/agent.py`, `physicam/plan.py` and
`physicam/calibrate.py`. Another engineer is working on `examples/`, `scripts/`, `README.md` and the
static mode in `web/app.js` in parallel. **Only edit `physicam/physics.py`, `physicam/agent.py`,
`physicam/plan.py`, `physicam/calibrate.py` and tests.** Do not start a server on port 8000.

The headline demo will be a phone video of a home-made pendulum (keys or a small bottle on a string,
hanging from a door frame). In the video the user says the length out loud ("the string is 80 cm").
Physicam must turn that into **g with an honest uncertainty**.

## 1. Pendulum geometry from the trajectory (`physics.py`)
- Fit a **circle to the bob's centroid path** (algebraic Kåsa fit, then Gauss–Newton geometric
  refinement) to get the pivot (cx, cy) and the radius L_px. Its uncertainty comes from the fit
  covariance or residuals. It must be robust when the arc is short (small amplitude): report a
  warning and a large uncertainty when the arc spans less than ~10°, and fall back to "L_px unknown".
  Sanity check: the pivot must be above the bob (smaller y).
- Angle θ(t) = atan2(x − cx, y − cy). Fit the damped sinusoid on **θ(t)** (better than x(t) for
  large amplitudes), giving T, γ and θ0 (amplitude in degrees).
- **Finite-amplitude correction**: T0 = T / (1 + θ0²/16 + 11θ0⁴/3072).
- g in three ways, in priority order:
  1. `options.length_m` given (user fact or spoken fact): g = 4π²·L/T0². Note that the length
     should be from the pivot to the bob's **center of mass**. If the user gives the string length
     and the bob diameter is known, add half of it; otherwise add a systematic-uncertainty note.
  2. Calibration available: L = L_px·m_per_px, then g as above.
  3. Neither: report T, T0 and θ0, plus the **predicted length for g = 9.81** (L = g·T0²/4π²). That
     is the "gravity as a ruler" inverse, shown as a derived fact and never as a measurement of g.
- Uncertainty: propagate σ_T (with T from many cycles, it is tiny), σ_L (user length: assume ±0.5 cm
  unless stated; from pixels: circle fit plus calibration) and the amplitude correction. The result
  will usually be dominated by σ_L, so say so in `info.dominant_error`.
- Emit the `params` `T`, `T0`, `theta0_deg`, `gamma`, `Q`, `L_m` (when known) and `g` (when
  computable), and the `info` fields `pivot_px`, `L_px` and `method_g`. Emit a `curve` with both
  `x` and `y` reconstructed from θ(t) on the circle, so the UI overlay draws the swing arc nicely.

## 2. Plan and agent
- `plan.py`: make sure `spoken_facts` extraction explicitly captures lengths, heights and diameters
  with units, and returns structured `facts_extracted: [{quantity: "length"|"height"|"diameter"|"distance", value_m, refers_to, quote}]`.
  These flow into the agent's initial context.
- `agent.py`: describe in the system prompt and tool descriptions that `fit_model(model="pendulum",
  options={"length_m": L})` computes g, and that spoken or user facts must be passed through. Add a
  worked example of the decision ("user said 80 cm, so pass length_m=0.80") to the system prompt.
- The deterministic fallback must also use `facts_extracted` lengths for pendulum jobs.

## 3. Tests
Synthetic pendulums (no network) with a full nonlinear simulation: integrate θ'' = −(g/L)·sin θ −
2γθ' with scipy at 60 fps, with L = 0.80 m, θ0 = 25° and pixel noise σ = 0.8 px:
- With `length_m` given, g is recovered within 0.5%, and the finite-amplitude correction matters:
  show that without it the error is larger.
- With a calibration (m_per_px), L_px from the circle fit recovers L within 2%.
- With a short arc (θ0 = 4°), the circle fit warns and does not produce a bogus L.

`.venv/bin/python -m pytest -q` passes. Finish with a summary that includes the synthetic-test
numbers.
