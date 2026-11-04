## Question

What is the oscillation period of this spring pendulum?

## What Physicam saw

The video shows a small metal weight on a hanging coil spring swinging side-to-side as a spring pendulum. The recording is 720 × 1280 px, 30.0 fps, 504 frames, and 16.8 s long, with a static camera and motion blur noted. The main track, labelled “metal ball,” contains 491 frames from 0.433 to 16.767 s. A second “metal weight” track contains only 10 frames from 16.467 to 16.767 s and was not used for the reported fits.

## Method

Physicam tracked the bob’s side-to-side position over time and fitted that motion with two models. A pendulum model was fitted over the full 0.433–16.767 s window. Because its geometry was unreliable, a damped oscillator model was also fitted to the same position–time data. That oscillator fit was repeated on the restricted 2.0–12.0 s window to check whether the period was stable. The period is determined from timing, so no spatial calibration was needed; the calibration method was “none.”

## Results

| Fit | Window (s) | Points | Period (s) | R² | RMSE (px) |
|---|---:|---:|---:|---:|---:|
| Pendulum, f0 | 0.433–16.767 | 491 | 0.9648 ± 0.0010 | 0.690 | 221.92 |
| Oscillator, f1 | 0.433–16.767 | 491 | 0.9633 ± 0.0003 | 0.965 | 22.81 |
| Oscillator, f2 | 2.0–12.0 | 301 | 0.9638 ± 0.0005 | 0.968 | 22.80 |

The headline result is an oscillation period of approximately **0.963 ± 0.001 s**.

## Discussion

No accepted reference period is provided, so the period cannot be compared with an accepted value. The value 9.81 was used only to derive a predicted pendulum length of 0.076 ± 0.011 m from the pendulum fit; this is not a measurement of $g$, and no length was measured independently.

The oscillator fits agree closely, differing by only about 0.0005 s between the full and restricted windows, which supports a stable period. In contrast, the pendulum fit is poor and carries the warning that the circle-fit pivot is not above the bob, making its length unreliable. Motion blur is a likely tracking-error source. Although the oscillator RMSE of about 22.8 px is far below the pendulum value, its warning of “large residual structure” indicates remaining systematic mismatch, possibly from air drag, perspective, or the wrong model. Thus the formal period uncertainty may understate model-related uncertainty.

## How to measure better next time

Reduce motion blur with better lighting or a faster exposure, keep the pivot visible and above the bob, and provide a measured pendulum length and spatial calibration so the geometry and $g$ can be checked independently. A cleaner, more planar swing and a repeat recording would also help separate true damping from tracking and perspective effects.