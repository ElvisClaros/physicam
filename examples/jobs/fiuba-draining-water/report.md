## Question

What is the discharge coefficient of the hole, and how long does the jar take to drain?

## What Physicam saw

Physicam tracked the brown liquid in a glass jar draining through a small bottom hole, with the level falling against a taped 1-6 scale. The main track contained 2471 frames from 0.000 s to 41.167 s. Independent manual measurements were available for 27 times over the same interval, credited to manual ImageJ measurements by the FIUBA student team.

## Method

The liquid-mask width gave the scale: 0.0004098 m/px from 0.075 m / 183.0 px, with 0.05 relative uncertainty. Physicam measured the falling height h(t), formed h/h_0, and fitted the draining model. In this fit, √(h) falls linearly, with slope k, drain time t_f, and Cd derived from k. A scale-free check, the empirical quadratic and cubic fits, a generalized-vessel fit, and the manual heights were used to test consistency, rather than to replace the main fit.

## Results

| Quantity | Value |
|---|---|
| Cd | 0.559 ± 0.032 |
| t_f | 46.200 ± 0.890 s |
| t_f_theory | 25.810 s |
| k | 0.00550 ± 0.00011 m⁰·⁵/s |
| h_0 | 0.064567 ± 0.000090 m |
| t10 | 2.371 ± 0.046 s |
| t25 | 6.190 ± 0.120 s |
| t50 | 13.530 ± 0.260 s |
| t75 | 23.100 ± 0.450 s |
| t90 | 31.590 ± 0.610 s |

## Discussion

No accepted Cd or drain-time value is supplied in the data, so no external comparison can be made. Internally, the scale-free check gives Cd = 0.5588, essentially identical to the fitted 0.559 ± 0.032, while the generalized-vessel fit gives 0.6009. The ideal-theory time, 25.810 s, is much shorter than the fitted 46.200 ± 0.890 s, as expected when Cd is below 1.

The dominant errors are systematic. The calibration uses the container diameter, but the liquid-filled cylinder refracts horizontally, so its apparent width is close to the outer diameter while vertical distances are not magnified. The 0.05 calibration uncertainty therefore limits absolute heights. The two half-fit k values, 0.005373 and 0.005584 m⁰·⁵/s, also show drift during draining.

The residuals say the simple model is only approximate: reduced chi-squared is 1694, with an inflation factor of 41.15. Against the manual heights, the root-mean-square difference is 0.363 cm and the maximum is 0.8212 cm, or 0.02577 and 0.06759 in h/h_0. The empirical quadratic fits well, with r² = 0.9935, whereas the exponential gives only 0.1376, supporting the draining form over exponential decay.

## How to measure better next time

Use a vertical ruler or the known h_0 as the primary scale reference, as the calibration note recommends, instead of the refracted horizontal diameter. Film until the jar is fully empty, beyond 41.167 s toward the fitted 46.200 s, so t_f is observed rather than extrapolated. Record the hole and jar diameters directly and repeat the run to test the drift between the first and second halves.