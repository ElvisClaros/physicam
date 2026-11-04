## Question

What is the discharge coefficient of the hole, and how long does the jar take to drain?

## What Physicam saw

A glass jar containing brown liquid drained through a small hole in its base. The liquid level fell against a taped 1–6 scale. Physicam tracked the brown liquid over 2471 frames from 0.000 s to 41.167 s. Independent manual measurements covered 0.000 s to 41.317 s, with the height falling from 5.860 cm to 0.032 cm.

## Method

The liquid surface height $h(t)$ was tracked automatically. Calibration used the container diameter: $0.075$ m over $183.0$ px, giving $0.0004098$ m/px, with a relative uncertainty of $0.05000$. A Torricelli draining model was fitted to $h(t)$, yielding the discharge coefficient $C_d$, drain time $t_f$, rate constant $k$ and initial height $h_0$. The fit was checked against 27 manual height measurements and against empirical quadratic, cubic and exponential fits.

## Results

| Quantity | Value | Unit |
|---|---:|---|
| $C_d$ | $0.559 \pm 0.032$ | — |
| $t_f$ | $46.200 \pm 0.890$ | s |
| $t_{f,\mathrm{theory}}$ | $25.810$ | s |
| $k$ | $0.00550 \pm 0.00011$ | m$^{0.5}$/s |
| $h_0$ | $0.06457 \pm 0.00009$ | m |
| $t_{10}$ | $2.371 \pm 0.046$ | s |
| $t_{25}$ | $6.190 \pm 0.120$ | s |
| $t_{50}$ | $13.530 \pm 0.260$ | s |
| $t_{75}$ | $23.100 \pm 0.450$ | s |
| $t_{90}$ | $31.590 \pm 0.610$ | s |

## Discussion

No accepted reference value is supplied in the data, so $C_d$ can only be compared internally. The scale-free check gives $C_d = 0.5588$, consistent with the fitted $0.559 \pm 0.032$, while a generalised-vessel model gives $0.6009$ with $R^2 = 0.9129$. The fitted drain time exceeds the theoretical value by a factor consistent with $C_d < 1$.

The dominant errors are systematic. The reduced chi-squared is $1694$, with an inflation factor of $41.15$, indicating the model or calibration, not random scatter, limits the fit. The value of $k$ drifted between halves, from $0.005373$ to $0.005584$ m$^{0.5}$/s. Against manual measurements, residuals had an RMS of $0.363$ cm and a maximum of $0.8212$ cm, or $0.02577$ and $0.06759$ of $h_0$ respectively — errors growing relatively as the level falls. An empirical quadratic achieved $R^2 = 0.9935$, far better than an exponential at $0.1376$, supporting the Torricelli form but showing residual structure.

## How to measure better next time

Use a vertical ruler for calibration instead of the container diameter, since refraction through the liquid-filled cylinder distorts the apparent width. Record until the jar is fully empty to constrain $t_f$ directly rather than by extrapolation, and sample the low-level stage more densely, where relative residuals are largest.