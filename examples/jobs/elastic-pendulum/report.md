## Question

What is the oscillation period of this spring pendulum, in which a small metal weight on a hanging coil spring swings side-to-side?

## What Physicam saw

Physicam tracked the metal weight, labelled “metal ball” (track t1), over 491 frames from 0.433 s to 16.767 s. The video contained 504 frames and lasted 16.800 s. The camera was static, but motion blur was noted as a quality issue. No spatial calibration was available, although spatial calibration is not needed to determine a period from times alone.

## Method

The horizontal position, x(t), of the tracked weight was fitted with a damped-sinusoid oscillator model over the full tracked interval. A pendulum fit was also attempted. The oscillator result was checked against an independent refit using only the first half, from 0.433 s to 8.500 s, and by comparing periods fitted separately to the two halves of the full record. The adopted period was taken from the full-range oscillator fit.

## Results

| Quantity | Model / interval | Value |
|---|---|---:|
| Oscillation period, adopted | Oscillator, 0.433–16.767 s | 0.9633 ± 0.0060 s |
| Oscillation period | Oscillator, full range, fit uncertainty | 0.9633 ± 0.0039 s |
| Oscillation period | Oscillator, 0.433–8.500 s | 0.9603 ± 0.0078 s |
| Oscillation period | Pendulum, full range | 0.9648 ± 0.0058 s |
| Angular frequency | Oscillator, full range | 6.522 ± 0.026 rad/s |
| Damping coefficient, γ | Oscillator, full range | 0.024 ± 0.015 1/s |
| Amplitude | Oscillator, full range | 204 ± 25 px |
| k/m | Oscillator, full range | 42.54 ± 0.34 1/s² |
| Fit quality | Oscillator, full range | R² = 0.9653, RMSE = 22.81 px |
| Fit quality | Pendulum, full range | R² = 0.6896, RMSE = 221.9 px |

The half-period checks for the full oscillator fit were 0.961 s and 0.967 s.

## Discussion

There is no independently measured accepted period with which to compare the result. The pendulum fit gave a similar period, but its much poorer R², large RMSE, and pivot warning make it unreliable for geometry. Its predicted length, 0.076 ± 0.025 m, follows from assuming g = 9.81, so it is not a measurement of g or an independent check.

For the oscillator fit, the statistical and systematic period uncertainties were 0.002194 s and 0.003168 s, respectively; “period” was identified as the dominant error source. Agreement between the full-range period, the first-half refit, and the two half-fits supports 0.9633 ± 0.0060 s.

However, the residuals contain substantial structure: the reduced chi-squared was 63.99, and the fit warned of possible air drag, perspective, or an incorrect model. Thus the sinusoid describes the timing well but not every detail of the motion. Motion blur may also shift tracked positions.

## How to measure better next time

Reduce motion blur, keep the swing in a plane parallel to the camera, and include a calibrated length reference and visible pivot. Those improvements would test the pendulum geometry as well as the period, reduce perspective and tracking errors, and allow residual structure to be distinguished from camera or tracking artefacts.