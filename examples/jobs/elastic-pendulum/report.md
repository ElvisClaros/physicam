## Question

What is the oscillation period of this spring pendulum?

## What Physicam saw

A small shiny metal bob/weight at the bottom of a hanging coil spring swinging side-to-side, with motion blur noted. The main track, t1 ("metal ball"), contains 491 frames from 0.433 s to 16.767 s. The video is 30.00 fps, 720 × 1280 px, and 16.800 s long.

## Method

The bob's horizontal position, x(t), was tracked over time. A pendulum fit (f0) and damped-sinusoid oscillator fits (f1, f2) were applied. Because the pendulum circle fit failed—pivot not above the bob and R² = 0.6896—the reported period uses the oscillator refit f2 over 2.000 s to 14.000 s. No spatial calibration was needed for a period measurement.

## Results

| Fit | Range (s) | Period (s) | R² | RMSE (px) |
|---|---|---:|---:|---:|
| f0 pendulum | 0.433–16.767 | 0.9648 ± 0.0058 | 0.6896 | 221.9 |
| f1 oscillator | 0.433–16.767 | 0.9633 ± 0.0039 | 0.9653 | 22.81 |
| f2 oscillator | 2.000–14.000 | 0.9638 ± 0.0036 | 0.9675 | 22.44 |

Headline result: **period = 0.9638 ± 0.0036 s**. For f2, ω = 6.519 ± 0.025 rad/s, amplitude = 195 ± 30 px, and γ = 0.020 ± 0.023 1/s. Its split-half periods are 0.9623 s and 0.9645 s.

## Discussion

There is no accepted reference period in the data to compare against. The pendulum fit also gives L_predicted = 0.076 ± 0.025 m, but this is only the length implied by assuming g = 9.81; g was not computed and this is not a measured length. Agreement among f0, f1, and f2 supports ~0.964 s, with f2 the cleanest. However, residuals say the simple damped sinusoid is incomplete: f2 has reduced chi-squared 69.19 and RMSE 22.44 px despite R² = 0.9675, with a warning of "large residual structure: maybe air drag, perspective, or wrong model." The pendulum fit's dominant error is listed as "period," and a spring pendulum swinging side-to-side need not behave as a simple pendulum. Motion blur is a likely tracking error source.

## How to measure better next time

Use a sharper, higher-contrast bob marker to reduce motion-blur tracking error; keep the camera perpendicular to the swing plane to reduce perspective; add a known length scale and a visible fixed pivot reference; and record a longer, steady oscillation so period, damping, and any spring-specific behaviour can be separated more cleanly.