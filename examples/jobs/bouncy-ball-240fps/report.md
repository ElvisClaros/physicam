## Question
What is the coefficient of restitution of this ball, and what does the throw tell us about g?

## What Physicam saw
A boy throws a small red ball across a tiled room and it bounces multiple times on the floor. The video is 1280 × 720 px, 240.0 fps, 4.0 s and 960 frames, with a time window of 0.0–3.5 s. Tracking the red ball gave track t0 over 0.0–3.995833 s and 863 frames. Quality issues were backlit conditions, a dark foreground silhouette and low contrast on the target, although the camera was static.

## Method
Physicam fitted a bounce model to the vertical ball trajectory, splitting it at impacts into free-flight segments. Calibration was none, with m_per_px = 1.0 and rel_uncertainty = 0.0, so lengths and accelerations remain in pixel units. Segment fits with poor behaviour or unphysical restitution were excluded from the pooled result. Fit f0 used the full t0 track; f1 repeated the analysis using free flight only, from 0.3333333333333333–3.9875 s.

## Results
| Quantity | Fit | Value |
|---|---|---|
| Coefficient of restitution, e | f0 | 0.6976269806104375 ± 0.004184273402523289 |
| Coefficient of restitution, e | f1 | 0.696813671342044 ± 0.0041847836812585896 |
| Pooled g | f0 | 4691.602576479591 ± 13.888652268640655 px/s² |
| Pooled g | f1 | 4694.606705794758 ± 13.893007364701203 px/s² |
| R² | f0 | 0.9971087360480302 |
| R² | f1 | 0.9958761287302011 |
| RMSE | f0 | 30.971483206324308 px |
| RMSE | f1 | 32.01353363838601 px |
| Valid impact time | f0/f1 | 1.5375 s |

The headline result is e = 0.6976 ± 0.0042, dimensionless, so no metric scale is required.

## Discussion
No accepted metric value can be compared directly: accepted g is in m/s², whereas this uncalibrated result is only in px/s², and no accepted e for this ball is supplied. Internally, f0 and f1 agree closely for both e and g, supporting the valid impact at 1.5375 s. The impact at 0.325 s gave e = 2.3144670128263884, flagged as unphysical because e > 1, and its neighbouring segment was excluded. The later impact at 2.470833333333333 s was also invalid because velocity-e and height-e disagreed by >0.15.

The throw tells us nothing reliable about g. Its segment fitted g = -2343.983568798512 ± 211.15729443366214 px/s² and was excluded, consistent with a powered throw rather than free fall. Dominant errors are perspective, segment-boundary placement and poor tracking near impacts, reflected in the warning that the per-segment g range was 5.8% > 5%. Despite high R², RMSE around 30.971483206324308–32.01353363838601 px shows substantial pixel residuals, so the formal uncertainties probably underestimate systematic error.

## How to measure better next time
Place a known metric reference in the ball’s plane of motion, improve front lighting and contrast, and keep the trajectory perpendicular to the camera axis to reduce perspective changes. Use cleaner segment boundaries around impacts, exclude the powered throw by design, and retain only free-flight portions for estimating g.