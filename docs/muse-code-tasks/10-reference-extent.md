# Task 10: fix reference-object size measurement (15% calibration bias)

Read `physicam/sam.py` (`segment_image`) and `physicam/calibrate.py` (`from_reference`). Do not start
servers on port 8000.

## Bug (found in a live test)
An A4 sheet (portrait, 0.297 m long side) was segmented by SAM 3.1 in a 2048×1152 image, with bbox
528×717 px. `segment_image` reported `longest_px = 820`, which is the **equivalent-ellipse major axis
from second moments**. For a uniform rectangle of side a, that axis is 4a/√12 ≈ 1.155·a, so the
calibration would be biased by about 15% and every metric result from an A4 or credit-card reference
would be wrong. The same issue affects `width`/`height` when the object is rotated.

## Fix
- In `segment_image`, compute the **oriented extents** from the mask raster mapped to frame
  coordinates (use bounds and scale; the raster is smaller than the frame):
  - Principal axes come from second moments (orientation θ). Project all foreground pixel centers
    onto the major and minor axes.
  - `long_extent_px = (max − min) + pixel_size_along_axis` (account for the raster-to-frame scale
    and the half-pixel at each end). Same for `short_extent_px`.
  - Add the robust variant: 0.5th/99.5th percentiles plus edge correction, to resist stray mask
    pixels. Report both, and use the robust one.
  - Also keep `bbox_width_px` and `bbox_height_px` (axis-aligned), `eq_diameter_px` (from area), and
    `ellipse_major_px` / `ellipse_minor_px` (renamed from the old fields, for diagnostics only).
- `from_reference` dimensions:
  - `longest`: robust long extent.
  - `shortest`: robust short extent (new).
  - `width` / `height`: oriented extents assigned by angle to the image axes, so a slightly rotated
    sheet still gets its true side lengths.
  - `diameter`: for round objects, `eq_diameter_px`; if the object is far from round (aspect > 1.15),
    the long extent plus a warning.
  - Add a **shape sanity check**. For a reference whose aspect ratio is known (A4 is 1.414, letter
    1.294, credit card 1.586), compare the measured long/short ratio. When it is off by more than 8%,
    add `perspective_warning` to the calibration details (the sheet is tilted relative to the camera)
    and increase `rel_uncertainty` by half the discrepancy.

    Known-ratio presets live in a small dict keyed by prompt keywords ("a4", "paper", "sheet",
    "letter", "credit card", "card").
- Update the agent tool description of `calibrate_reference` (`dimension` enum includes `shortest`;
  default `longest`).

## Tests (synthetic masks; no network)
Build rasters directly and call the extent function, refactored so it is testable without SAM:
- An axis-aligned rectangle 400×283 px: long extent within 0.5% of 400 and short within 0.5% of
  283. A raster/frame scale of 0.5 (raster half the frame size) still gives frame-pixel extents
  within 1%.
- The same rectangle rotated 30°: extents within 1%.
- A disc of diameter 120 px: eq_diameter and both extents within 1%.
- The A4 aspect check flags a rectangle whose ratio is 1.25.

`.venv/bin/python -m pytest -q` passes. Summarize the before/after on the A4 case: the image is at
`data/test/a4_scene.png`; re-run `segment_image` on it with prompt "sheet of paper" (one cheap SAM
image call) and report the new long/short extents and ratio.
