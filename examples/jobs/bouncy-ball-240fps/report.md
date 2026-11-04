## Question

What is the coefficient of restitution of this ball, and what does the throw tell us about g?

## What Physicam saw

A boy throws a small red ball across a tiled room, and the ball bounces multiple times on the floor. The video is 1280 × 720, has 960 frames, lasts 4.000 s, and was recorded at 240.0 fps. Physicam tracked the red ball for 863 frames from 0.000 s to 3.996 s. The camera was static, but the scene was backlit, with a dark foreground silhouette and low contrast on the target. No length reference was available.

## Method

The vertical ball position was fitted with a bounce model: parabolic flight between impacts, with the speed reduced at each impact. The selected fit uses 791 points from 0.300 s to 3.996 s. Because calibration used method “none”, with 1.000 m/px nominal scale and 0.000 relative uncertainty in the file, distances, speeds, and acceleration remain in pixel units. Restitution was calculated from the vertical speeds immediately before and after impact, and was also checked against the bounce-height method.

## Results

| Quantity | Result |
|---|---|
| Impacts in selected fit | 2 |
| Valid impact time | 1.538 s |
| Speed before valid impact | 3187 px/s |
| Speed after valid impact | −2222 px/s |
| Coefficient of restitution, e | 0.70 ± 0.12 |
| Height-based e at same impact | 0.8287 |
| Pooled g | 4750 ± 340 px/s² |
| Segment g values used | 4550 ± 670 px/s²; 4810 ± 390 px/s² |
| Fit R² | 0.9952 |
| Fit RMSE | 32.42 px |
| Other impact, at 2.471 s, marked invalid | e = 0.25 ± 0.42; height-based e = 0.6117 |

## Discussion

The only reliable restitution is e = 0.70 ± 0.12 at 1.538 s. It is dimensionless, so it can be interpreted without a length scale: the ball rebounded with about this fraction of its incoming vertical speed. The second fitted impact is not reliable because a neighbouring flight was excluded and the velocity-based and height-based values disagree by more than 0.15, as also seen, to a lesser extent, at the valid impact.

For g, there is no accepted-value comparison in m/s²: with no known length, 4750 ± 340 px/s² cannot be converted to m/s². The throw does support constant-acceleration ballistic flight, with R² = 0.9952, but the residuals are not negligible at 32.42 px RMSE. Dominant error sources are the low-contrast, backlit tracking, possible perspective as the ball moves toward or away from the camera, and uncertain segment boundaries. This is reflected in the 5.6% per-segment g spread and an excluded segment with g = 600 ± 1100 px/s² and R² = 0.5462.

## How to measure better next time

Include a visible object of known length, or a measured floor-tile dimension, in the ball’s plane so px can be converted to m. Keep the flight as side-on and in one plane as possible, improve foreground lighting and contrast, and ensure each bounce and the following flight are clearly visible so no segment must be excluded.