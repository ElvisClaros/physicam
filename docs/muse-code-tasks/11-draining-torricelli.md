# Task 11: liquid level and draining (Torricelli), a real FIUBA lab

Read `AGENTS.md`, then `physicam/` and `web/`. Do not start servers on port 8000 (use 8010).

## Context: the problem we are solving
In FIUBA's *Modelación Numérica* course, students film a jar that drains through a small hole and
must extract the water height h(t) **by hand**. In this case they used ImageJ, 27 manually measured
frames in a 41 s video. They then fit models by least squares and use Newton–Raphson to find the
times at which 10/25/50/75/90% of the height has drained, and compare with Torricelli's law. The
real data is in `data/inbox/fiuba/`:
- `agua_5mm.mp4`: 60 fps, 478×850, 2471 frames, transcoded from the original `agua_5mm.avi`. A glass
  jar with **dyed (brown) water** drains through a 5 mm hole near the bottom. A paper strip with cm
  marks is taped to the jar.
- `tabla_agua_5mm.csv`: the students' manual ImageJ measurements. The columns `Slice` (1-based frame
  number at 60 fps), `Length` (water height in **cm**) and `Tiempo_s` are the ground truth for
  validation.
- `codigo.py`: the students' analysis, for reference only; do not import or run it. It uses jar
  diameter D = 7.5 cm (A1 = π·3.75²), hole diameter d = 0.5 cm, g = 981 cm/s², and the theoretical
  t_f = sqrt((2h0/g)·(A1²/A2² − 1)) with h(t)/h0 = (1 − t/t_f)².
- `video_aceite.mp4`: the same experiment with oil (pale yellow, harder to segment). Optional second
  example.
- `diametro_75mm.jpeg`: photo of the jar diameter measured as 75 ± 1 mm.

A probe (`_probe/level.py`) shows how to get the level from SAM masks. Its result is in the lead's
notes at the end of this file.

## 1. Level measurement (`physicam/level.py`, new)
- `level_series(video_path, sam_text_or_cache, fps, prompt)` decodes **mask rasters for every frame**
  of the liquid track from the cached SAM text (same parser as strobe; no new API calls when
  cached).
- For each frame:
  - **surface_y**: the first mask row whose width is ≥ 50% of the median row width. This makes it
    robust to the meniscus, splashes and the exit jet. Refine to sub-pixel by linear interpolation
    of the row coverage fraction across the boundary.
  - **bottom_y**: the last such row. It should be constant: use its median over the clip as the
    fixed bottom, since the liquid bottom is the jar bottom.
  - **width profile** w(y) on the first frames (full jar).
- Track selection: use the largest-area track for the liquid prompt, and stitch ids as elsewhere.
- Output a `LevelSeries` (t, h_px, surface_y, bottom_y, width_profile, frames, conf) with
  `to_json()` and `to_csv()` (columns `t_s, h_px, h_m, surface_y_px`). Use the same subpixel and
  coordinate conventions as `sam.py`.

## 2. Calibration options for level
- From the container diameter (fact `container_diameter_m`): m_per_px = D / (median liquid-mask
  width). **Caveat:** a liquid-filled cylinder refracts horizontally, so the apparent width of the
  liquid is close to the outer diameter, and vertical distances are not magnified. Report this as a
  systematic note.
- From a reference ruler or strip (`calibrate_reference`), or from the initial height fact `h0_m`.
- The scale-free outputs (h/h0, t_f, times to X%) need **no calibration**. Lead with them.

## 3. Physics (`physics.py`): model `draining`
With r = a/A, where a is the hole area and A the container area, and approach-velocity correction
v = sqrt(2gh/(1−r²)):
1. **Torricelli (ideal, Cd = 1)**: dh/dt = −r·sqrt(2gh/(1−r²)). The prediction t_f,theory is
   computed from h0 (it matches the students' formula exactly) and needs no fit.
2. **Torricelli with discharge coefficient**: dh/dt = −Cd·r·sqrt(2gh/(1−r²)), with closed form
   sqrt(h) = sqrt(h0) − k·(t − t0).
   - Fit `h0`, `k` and `t0` by nonlinear least squares on the measured h(t).
   - Report Cd = k / (r·sqrt(g/(2(1−r²)))) when d and D are given, and t_f = t0 + sqrt(h0)/k.
   - The uncertainty includes the fit covariance (with χ² inflation and split-half) plus the
     propagated σ_d and σ_D (defaults ±0.1 mm and ±1 mm, overridable).
   - In pixel units, Cd is still computable from the **scale-free** form: fit h/h0 = (1 − (t − t0)/t_f)²,
     then Cd = t_f,theory(Cd = 1) / t_f, using h0 converted with the calibration. Document which form
     is used.
3. **Generalized vessel**: A(h) from the measured width profile (axisymmetric, A = π(w/2)²). Solve
   A(h)·dh/dt = −Cd·a·sqrt(2gh) with scipy `solve_ivp` (RK45, rtol 1e-8) and fit Cd and h0 by least
   squares. Compare with model 2: for a cylindrical jar they should agree.
4. **The course's empirical fits**: quadratic, cubic and exponential on h/h0, with R², reported
   side by side with the physical models.
5. **Times to 10/25/50/75/90% drained** (of the height, as in the course): from the fitted
   Torricelli model analytically, and from the cubic fit via Newton–Raphson (to mirror the course),
   with the fitted-model times as the primary result.
6. **Validation vs manual data** (when a reference CSV is supplied via
   `options.reference_csv` or the CLI): interpolate Physicam's h at the students' frames and report
   RMS and max differences, in cm when calibrated and in h/h0 always. Compare the times to X% with
   those computed the same way from the students' 27 points.

## 4. Agent, plan, pipeline
- `plan.py`: add `phenomenon = "draining"` (liquid level falling: emptying jar, tank, bottle, hourglass
  sand…). Candidate SAM prompts for liquids are color + liquid ("brown liquid", "water", "yellow
  oil"). Extract facts such as the hole diameter, container diameter and liquid type from the
  question or narration.
- `agent.py`: a new tool `measure_level(prompt, t_start?, t_end?)` returns a summary (frames, h0_px,
  final h, monotonicity, width profile stats). `fit_model(model="draining", options={hole_diameter_m,
  container_diameter_m, ...})` returns params `Cd`, `t_f`, `k`, `h0`, `t10..t90`, plus `t_f_theory`
  and the empirical fits in `info`. Update the system prompt with a worked example.
- Pipeline and result JSON: add `level: {t, h, surface_y, bottom_y, unit}` and `level_csv_url`. The
  server exposes `GET /api/jobs/{id}/level.csv`.

## 5. UI (`web/`)
- When `result.level` exists:
  - Draw on the video overlay a **horizontal surface line** at surface_y for the current frame, the
    liquid mask fill (from contours) and a label `h = 3.21 cm`.
  - Show a chart "Liquid level vs time": measured points, the Torricelli Cd fit, the ideal Cd = 1
    curve dashed, and the students' manual points as hollow markers when `validation` exists. Below
    it, a residuals strip.
  - Show a **"Times to drain" table** (10/25/50/75/90%: Physicam vs reference).
  - Add a **"Download h(t) CSV"** button.
- Answer card: e.g. "C_d = 0.62 ± 0.03 · drains in 41.2 ± 0.4 s".

## 6. Example and validation
- Add the water clip as an example via `examples/media/fiuba-draining-water.mp4` plus a sidecar
  JSON:
  - title: "Draining jar · Torricelli (FIUBA lab)"
  - subtitle: "Real Modelación Numérica lab: 5 mm hole, 7.5 cm jar"
  - question: "What is the discharge coefficient of the hole, and how long does the jar take to drain?"
  - facts: `{container_diameter_m: 0.075, hole_diameter_m: 0.005}`
  - reference_csv: `examples/validation/fiuba_tabla_agua_5mm.csv`

  Copy the CSV into `examples/validation/`, crediting "manual ImageJ measurements by the FIUBA
  student team" in ATTRIBUTION.md. Build it with `examples/build_examples.py`, live (the SAM cost is
  about $0.50 for 2471 frames, which is fine).
- Report the validation numbers in your summary: RMS (cm and h/h0) vs the manual table, Cd, t_f vs
  t_f,theory, and times to X% vs the students'.
- Tests: synthetic Torricelli data recovers Cd within 1%; the generalized-vessel solver agrees with
  the closed form for a cylinder; level extraction works on a synthetic mask sequence (a draining
  rectangle with a meniscus bump and a jet blob).
- `.venv/bin/python -m pytest -q` passes. Summarize.

## Lead's notes (probe results)
Lead's probe on `agua_5mm.mp4` with SAM prompt **"brown liquid"**: 1 track, all 2471 frames, 106 s of
SAM time. The raw SAM text is NOT cached in `data/cache` (the probe bypassed the cache), so you will
re-run it once (about $0.50); after that the cache serves it.
- **h must be measured above the HOLE, not above the jar bottom.** That is Torricelli's h, and it
  is what the students measured. Liquid below the hole stays in the jar: here the surface stopped
  73 px above the mask bottom.
  - Estimate the hole level as the **final steady surface level**: the median surface_y over the
    trailing window where |dh/dt| ≈ 0, at least 1 s.
  - Allow the facts `hole_height_m` and `hole_y_px` to override.
  - If the clip ends before draining stops, use the h→0 asymptote of the Torricelli fit, with a
    warning.
- With h above the hole, the normalized RMS vs the ImageJ table is **0.025·h0** (max 0.05·h0). The
  remaining differences are ±0.2–0.3 cm early, with alternating sign, which points to manual
  measurement noise.
- Level noise is 0.75 px std vs a 2 s Savitzky–Golay smooth, and the smoothed series is monotonic.
  Offer an optional SG smoothing for derivatives (dh/dt and the Q(t) plot), but fit on raw data.
- Quick Torricelli fit in px: h0 = 150 px, k = 0.268 px^0.5/s, t0 = 0.95 s, t_f = 46.7 s, RMS = 4.3 px.
  The model mismatch is visible, so the generalized-vessel fit and residual diagnostics matter.
  Ideal t_f (Cd = 1, h0 = 5.86 cm, D = 7.5, d = 0.5) is 24.6 s, giving **Cd ≈ 0.53**. Typical
  sharp-edged orifices give 0.6, and a hole punched in a thin wall at small heads can be lower, so
  this is plausible. Report it with an honest uncertainty, dominated by d (a 5 mm hole ±0.1 mm gives
  ±4% on Cd) and the h0 scale.
- **Scale discrepancy:** matching h0 = 5.86 cm gives 0.360 mm/px, but liquid-mask width 181 px for
  D = 7.5 cm gives 0.414 mm/px (+15%). The liquid's apparent width is refracted by the curved
  glass and liquid. **Prefer a vertical reference**: segment the paper strip with cm marks taped
  to the jar ("paper strip" / "paper ruler"). Ask Spark vision (`inspect_frames`) for the visible
  mark span, and SAM gives its pixel length. Otherwise use the `h0_m` fact. Report the jar-width
  calibration as a cross-check with the refraction caveat.
- The level probe code is in `_probe/level.py`, with the saved series in
  `data/inbox/fiuba/agua_5mm.brown_liquid.level.npy` (rows t_s, h_px from mask bottom). Reuse its
  ideas, not its code.
