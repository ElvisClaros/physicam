## Question

What is the coefficient of restitution of this ball, and what does the throw tell us about $g$?

## What Physicam saw

A boy throws a small red ball across a tiled room, and it bounces multiple times on the floor. Physicam tracked the red ball over 863 frames from 0.000 s to 3.996 s in a $1280 \times 720$ video recorded at 240.0 fps with a static camera. The footage was backlit, with a dark foreground silhouette and low contrast on the target, making the ball difficult to locate consistently, particularly near segment boundaries.

## Method

A bounce model was fitted to the vertical, $y$-direction, motion of the tracked ball. Flight between impacts was modelled as parabolic, constant-acceleration motion, while impacts were identified from changes between successive flight segments. The coefficient of restitution, $e$, was calculated from the vertical speeds immediately before and after impact. Results below use fit f1, covering 0.300 s to 3.996 s and 791 points. No length reference or known ball size was available, and the calibration method was “none”, so distances, speeds, and acceleration remain in pixels.

## Results

| Quantity | Result |
|---|---:|
| Valid impact time | 1.538 s |
| Speed before impact | 3187 px/s |
| Speed after impact | $-2222$ px/s |
| Coefficient of restitution, $e$ | $0.70 \pm 0.12$ |
| Height-based $e$ at same impact | 0.8287 |
| Pooled $g$ | $4750 \pm 340$ px/s² |
| Fit $R^2$ | 0.9952 |
| Fit RMSE | 32.42 px |
| Impacts in fit f1 | 2.000 |
| Second, invalid impact time | 2.471 s |

Only the impact at 1.538 s was valid. Segment accelerations were $4550 \pm 670$ px/s² and $4810 \pm 390$ px/s²; another segment, with $600 \pm 1100$ px/s² and $R^2 = 0.5462$, was excluded from the pooled result.

## Discussion

The valid bounce gives $e = 0.70 \pm 0.12$, meaning the rebound vertical speed was about this fraction of the incoming speed. There is no universal accepted value for $e$, because it depends on the ball and floor. The height-based estimate, 0.8287, differs noticeably from the velocity-based result, indicating sensitivity to the fitted trajectories and impact boundary.

No accepted-value comparison can be made for $g$ in m/s²: without a length scale, $4750 \pm 340$ px/s² cannot be converted to physical units. The throw nevertheless supports approximately constant-acceleration ballistic flight. The high $R^2$ of 0.9952 shows that parabolas describe most vertical motion well, but the 32.42 px RMSE and 5.6% per-segment spread in $g$ reveal substantial residuals. Likely dominant errors are poor target contrast from backlighting, uncertain segment boundaries near impacts, and perspective if the ball moved toward or away from the camera. These problems explain the excluded segment and invalid restitution at 2.471 s, where velocity- and height-based estimates disagreed by more than the reliability threshold stated in the fit warning.

## How to measure better next time

Place a clearly visible object of known length in the ball’s plane of motion, light the scene from the camera side, and keep the throw parallel to the image plane. Record a cleaner background, mark the floor contact line, and repeat the throw to test whether $e$ and the per-segment $g$ values agree.