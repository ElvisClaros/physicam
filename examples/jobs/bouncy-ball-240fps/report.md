## Question

What is the coefficient of restitution of this ball, and what does the throw tell us about g?

## What Physicam saw

A small red ball was thrown across a tiled room and bounced repeatedly on the floor. The video lasted 4.000 s at 240.0 fps, with a frame size of 1280 by 720. The camera was static. The scene was backlit, with a dark foreground silhouette and low contrast on the target, making the ball difficult to distinguish in some frames.

## Method

Physicam tracked the red ball using the prompts “red ball”, “small ball”, and “red sphere”. The track used for the reported fit, t2, contained 862 frames spanning 0.000 s to 3.996 s. A bounce model was fitted to the vertical motion over 0.333 s to 3.996 s, identifying impacts and estimating the vertical acceleration and coefficient of restitution for each usable impact. No reference object or known size was available, so no length calibration could be performed. Consequently, distances, velocities, and acceleration are reported in pixels, pixel-based velocity units, and px/s² rather than SI units.

## Results

| Quantity | Result |
|---|---:|
| Valid impacts used | 2.000 |
| Points in fit | 782.0 |
| Fit $R^2$ | 0.9959 |
| Fit RMSE | 32.42 px |
| Coefficient of restitution, $e$ | $0.70 \pm 0.12$ |
| Height-based $e$ at valid impact | 0.8294 |
| Vertical acceleration, $g$ | $4760 \pm 340\ \mathrm{px/s^2}$ |
| Segment $g$, first pooled segment | $4550 \pm 730\ \mathrm{px/s^2}$ |
| Segment $g$, second pooled segment | $4810 \pm 380\ \mathrm{px/s^2}$ |

The valid restitution measurement occurred at 1.538 s, with velocities before and after impact of 3189 px/s and −2222 px/s, respectively. Another impact at 2.471 s gave $e = 0.25 \pm 0.42$, but it was marked invalid and excluded from the headline result.

## Discussion

The headline result is $e = 0.70 \pm 0.12$, indicating a moderately bouncy collision, while $g = 4760 \pm 340\ \mathrm{px/s^2}$ cannot be compared with an accepted SI value because the pixel scale is unknown. The high $R^2$ shows that the bounce model captures the overall trajectory, but the RMSE of 32.42 px shows substantial scatter around the fitted path. The velocity-based and height-based restitution estimates also disagree by more than the stated reliability threshold, which weakens confidence in the detailed bounce behaviour.

The dominant error sources are likely poor target visibility from backlighting and low contrast, uncertainty in segment boundaries near impacts, and changing perspective as the ball moves through the room. This is supported by the per-segment $g$ range of 5.6%, above the 5% warning level. The invalid late impact and excluded neighbouring flight further suggest that the later motion was tracked or segmented less reliably.

## How to measure better next time

Film against a brighter, higher-contrast background and light the ball from the camera side. Place a clearly visible object of known length in the plane of motion to calibrate pixels to metres. Keep the ball’s motion as perpendicular to the camera view as possible, mark the floor contact line, and record a cleaner sequence with fewer occlusions so every bounce segment can be included.