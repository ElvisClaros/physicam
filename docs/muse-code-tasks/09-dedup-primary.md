# Task 09: cross-prompt track dedup + primary-fit-focused display

Read `AGENTS.md`, `physicam/sam.py`, `physicam/agent.py`, `physicam/pipeline.py`, `physicam/physics.py`
and `web/app.js`. Do not start servers on port 8000 (use 8010). Check against the real job
`data/jobs/06f3b1539509/` (live bounce run) and `examples/jobs/*`.

## 1. Cross-prompt duplicate tracks (backend bug)
Concurrent pre-tracking runs several candidate prompts ("red ball", "orange ball", "small ball").
They all segment **the same physical ball**, so the result has 3 tracks of one object. The agent then
fits several of them, and the UI draws overlapping trajectories.
- After multi-prompt tracking, deduplicate **across prompts**. Two tracks are the same object when,
  over their overlapping frames (≥ 60% of the shorter track), the median box IoU is ≥ 0.5 or the
  median centroid distance is < 0.5 × the median equivalent diameter. Keep the track with more
  frames (tie-break: higher median confidence). Keep the others' labels in `aliases` on the kept
  track. Frames where only the dropped track has data may fill gaps in the kept track, but only when
  consistent (distance < 1 diameter from the interpolated kept position).
- The `track_objects` tool summary then reports one track ("red ball", aliases: orange ball, small
  ball), so the agent fits one object.
- Add tests with synthetic tracks: duplicates are merged, and two genuinely different objects close
  together are kept.

## 2. Primary-fit-focused display (UI)
- The **primary fit** is the first id of `answer.fit_ids` (fallback: the last fit). The overlay draws
  impact markers, e labels and ghost curves **only for the primary fit** by default. Other fits'
  ghosts are hidden unless "Other fits" is expanded in the params panel; then draw them at 35%
  opacity.
- **Invalid impacts** (`valid === false`, or e > 1.05) are never labeled with an e value on the
  video. Draw them as small grey hollow circles with no text.
- **Restitution chart**: bars only for the primary fit's valid impacts, labeled "impact 1…n" (no
  overlapping text; rotate labels or use short labels like "#1"), with error bars. Show the apex-
  height cross-check as a hollow diamond marker per impact when `e_height` exists.
- Charts: draw only tracks referenced by the primary fit at full opacity. Others are hidden, or at
  25% opacity when the legend chip is toggled on.
- Params table: hide `g_px` when it duplicates `g` (same unit). Show count-like params
  (`n_impacts`) as integers with no "± —".

## 3. Re-run the live bounce job offline from cache to apply the dedup
- Re-run the live bounce job (`data/jobs/06f3b1539509`, video
  `/tmp/claude-1000/-tmp-Firefox-meta-h/924c59a5-0268-4e77-a9a2-7390d4545cc0/scratchpad/demo/bounce_throw.mp4`
  if readable; otherwise skip) and both examples, **live** for the agent (SAM is cached, so the cost
  is small), with `examples/build_examples.py`.
- Make sure the bouncy example's answer is a single deduped track, then run
  `python -m physicam rereport` on both examples.
- `.venv/bin/python -m pytest -q` passes and `node --check web/*.js` passes. Summarize with the new
  example answers.
