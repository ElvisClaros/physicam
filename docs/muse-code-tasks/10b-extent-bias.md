# Task 10b: remove the percentile bias in `mask_extents`

In `physicam/sam.py`, `mask_extents()` reports `robust_long_extent_px` and `robust_short_extent_px`
as the 0.5th–99.5th percentile range plus an edge correction, and `from_reference` uses them. That
percentile range is **biased short**:
- For a rectangle, the projection along an axis is uniform, so the 99% range is 0.99·L. On the A4
  test it gave 707.1 px vs 718.4 px exact, 1.6% short.
- For a disc, the projection density is ∝ sqrt(1−x²) and the tails are thin, so the trimmed range
  is about 0.96·D, **~4% short**.

Both would bias every calibrated result.

## Fix
- Make outliers robust **topologically** instead of by trimming. Keep only the largest
  4-connected component of the raster (implement a small BFS/union-find in numpy or pure Python;
  no new dependencies; rasters are small), then use the exact extent `(max − min) +
  pixel_size_along_axis` on that component. Optionally apply a 1-pixel binary opening before
  labeling when the component has thin spurs. It must keep exactness on clean rectangles and discs.
- `robust_*` fields = the exact extents of the largest component. Keep the percentile numbers only
  as `trimmed_*` diagnostics, unused.
- Tests: a 400×283 rectangle and a 120 px disc must be within 0.5% using the fields that
  `from_reference` consumes. A stray 3-pixel blob 50 px away must not change the result. A disc must
  no longer come out ~4% short.
- Re-run `segment_image` on `data/test/a4_scene.png` ("sheet of paper"), one SAM call, and report the
  long/short extents and the ratio.
- `.venv/bin/python -m pytest -q` passes. Summarize.
