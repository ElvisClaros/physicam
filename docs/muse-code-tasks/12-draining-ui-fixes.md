# Task 12: draining job presentation fixes

Read `web/app.js`, `web/charts.js`, `web/style.css` and `physicam/report.py`. A server runs on port
8000 from the host; do not start one on 8000 (use 8010). Check against
`examples/jobs/fiuba-draining-water/result.json`.

Real-browser screenshots of `#job=fiuba-draining-water` show:
1. **Page width explodes to 16855 px**, because the params "diagnostics" sub-table prints huge
   values (`info.validation` with its arrays of 27 points, `info.empirical` coefficients, etc.)
   on one line. Fix:
   - Diagnostics show scalars and short arrays (≤ 6 numbers) only.
   - Objects and long arrays are summarized, e.g. "validation: n = 27, RMS 0.026·h0", with a
     "details" toggle that shows pretty-printed JSON in a scrollable `<pre>` (max-height 240 px,
     `overflow: auto`, `white-space: pre-wrap`, `word-break: break-all`).
   - Also add CSS so no table cell can widen the page: `table-layout: fixed` or
     `overflow-wrap: anywhere`, and `max-width: 100%` on wrappers.
2. **Empty charts**: "Height vs time", "Horizontal position vs time" and "Speed vs time" show "No data
   yet" for level jobs. Hide those three chart cards when the job has no fitted moving-object
   tracks, or when `result.level` exists and the primary fit is `draining`. The "Liquid level"
   chart and the "Times to drain" table then take the full width.
3. **Overlay label is wrong**: it shows "h = 9064.63 cm" at t = 12 s, where the true level is about
   2.7 cm. Compute it from `result.level.h` interpolated at the current time, in the units given by
   `result.level.unit`, converted correctly: m → cm × 100, px stays px. Check that the value agrees
   with the Liquid level chart at the same time.
4. **Legend**: the level job shows three auxiliary track chips ("brown liquid (aka dark liquid)",
   "water (aka brown water)" twice). For level jobs, show a single chip for the level track
   ("liquid level · brown liquid"), and hide duplicates with the same label.
5. **Timeline badge**: `measure_level` is shown with a SPARK badge. It is a SAM 3.1 tool, so give it
   the SAM badge and track icon. Check the badge mapping for all tools.
6. **Report math**: Muse Wonder wrote LaTeX (`$C_d$`, `\pm`, `\mathrm{...}`, `m$^{0.5}$/s`) that the
   markdown renderer shows raw.
   - In `report.py`, tell Wonder explicitly: no LaTeX, use plain Unicode (±, ², ½, subscripts as
     `C_d` → "Cd", "t_f").
   - Post-process the returned markdown: strip `$…$` delimiters, `\pm`→±, `\times`→×, `\cdot`→·,
     `^{2}`/`^2`→², `\mathrm{X}`→X, `\sqrt{x}`→√(x), `\frac{a}{b}`→a/b, `_{x}`→_x.
   - Then run `python -m physicam rereport examples/jobs/fiuba-draining-water` (a live Wonder call
     is fine) and also re-report the other two examples.
7. `node --check web/*.js` and `.venv/bin/python -m pytest -q` pass. Summarize.
