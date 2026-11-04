## Question

What is the oscillation period of this spring pendulum?

## What Physicam saw

Physicam tracked a small shiny metal bob at the bottom of a hanging coil spring, swinging side-to-side. The track, labelled metal ball, contains 491 frames from 0.433 s to 16.767 s. The video has 504 frames, a duration of 16.800 s, and a frame rate of 30.00 frames/s. The camera was static, but motion blur was noted. No spatial calibration or reference length was available, so positions were analysed in pixels; calibration is not required for a period measurement.

## Method

The horizontal position, x, of the bob was tracked over time. Three fits were examined: a pendulum fit over the full track and damped-sinusoid oscillator fits over the full track and its first half. The pendulum circle fit was rejected because its pivot was not above the bob and its residuals were much larger. The reported period was therefore taken from the full-range oscillator fit, with the pendulum fit, first-half oscillator fit, and split-half periods used as consistency checks.

## Results

| Fit | Time range (s) | Period (s) | R² | RMSE (px) | Points |
|---|---|---|---|---|---|
| Pendulum, full range | 0.433–16.767 | 0.9648 ± 0.0058 | 0.6896 | 221.9 | 491 |
| Oscillator, full range | 0.433–16.767 | 0.9633 ± 0.0039 | 0.9653 | 22.81 | 491 |
| Oscillator, first half | 0.433–8.500 | 0.9603 ± 0.0078 | 0.9685 | 23.92 | 243 |

The adopted result is T = 0.963 ± 0.006 s. For the full-range oscillator fit, ω = 6.522 ± 0.026 rad/s, γ = 0.024 ± 0.015 1/s, and amplitude = 204 ± 25 px.

## Discussion

There is no accepted period value in the supplied data for direct comparison. The three period estimates agree closely: 0.9648 s, 0.9633 s, and 0.9603 s. The full-range oscillator split halves gave 0.961 s and 0.967 s, supporting a stable period. The oscillator fit explains the motion far better than the pendulum fit, whose R² is only 0.6896 and whose predicted pivot geometry failed. The length implied using g = 9.81 m/s² was 0.076 ± 0.025 m, but this is a derived fact, not a measurement of g or length; g was not computed because no calibrated length was available.

Period was identified as the dominant error source. Despite the high oscillator R², its reduced chi-square of 63.99 and the flagged large residual structure indicate systematic mismatch, possibly from motion blur, perspective, air drag, or the spring-pendulum motion not being a simple planar pendulum. The damping estimate is also uncertain relative to its size.

## How to measure better next time

Reduce motion blur with brighter lighting or a faster shutter, keep the swing in one plane facing the camera, and make the pivot and full spring visible. Add a calibrated length reference in the swing plane so length and g can be tested rather than assumed. Record a longer, steady oscillation and repeat the measurement to check period stability.