# Task 08: demo readiness (visual noise and number formatting)

Read `AGENTS.md`, `web/app.js`, `physicam/report.py` and `physicam/pipeline.py`. Real screenshots of
`#job=elastic-pendulum` exposed these issues. Do not start servers on port 8000 (use 8010).

1. **Trail too long**: the overlay draws the entire trajectory history, which on a 17 s pendulum
   becomes a messy filled ellipse. Draw only the last 1.0 s of trail, fading in opacity and width,
   and capped at 240 points. The fitted ghost curve is drawn only within ±1.5 s of the current time
   for long clips (more than 4 s), or in full for short clips.
2. **Auxiliary tracks**: a 10-frame "metal weight" track is shown next to the main "metal ball"
   track. Tracks not referenced by any fit are drawn at 30% opacity, without labels, and appear in
   the legend as toggleable chips that are off by default. Tracks with fewer than 15 frames are
   hidden entirely unless they are referenced by a fit.
3. **Number formatting everywhere in the UI**: `info` values and arrays (`T_halves`, `sigma_stat`,
   `chi2_red`, `inflation_factor`, `pivot_px`…) are shown as raw floats. Format them with 4
   significant digits, and arrays as comma-separated formatted numbers. Give `info` rows friendlier
   labels ("Half-window periods", "σ stat", "σ sys", "χ²ᵣ", "inflation") and put them in a small
   muted "diagnostics" sub-table below the params.
4. **Report numbers**: Muse Wonder copies raw floats ("0.43333333333333335 s",
   "0.9648424115937253 ± 0.005818…").
   - In `report.py`, round every float in the compact JSON sent to Wonder to sensible precision:
     values to the uncertainty's decimal place with 2 significant digits on the uncertainty, or 4
     significant digits when there is no uncertainty. Times go to 3 decimals.
   - Tell Wonder explicitly in the prompt: use proper significant figures, never more than 4
     significant digits, no raw floats.
   - Also post-process the markdown with a regex that rounds any number with more than 6 digits
     after the decimal point to 4 significant digits.
5. **Hash navigation**: changing `location.hash` to `#job=<id>` (without a reload) must load that
   job. Add a `hashchange` listener. Clicking an example also updates the hash.
6. Re-render the report for the two example jobs (`examples/jobs/*/`) and the jobs in `data/jobs/`
   without re-running the pipeline. Add a `python -m physicam rereport <job_dir>` CLI that rewrites
   `report_markdown` in result.json and report.md (a live Wonder call is fine; it is cheap), then
   run it on both examples.
7. `.venv/bin/python -m pytest -q` passes, and `node --check web/*.js`. Summarize.
