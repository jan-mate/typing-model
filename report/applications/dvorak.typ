== Is Dvorak faster than QWERTY? <is-dvorak-faster>

=== Introduction
Dvorak claimed the layout minimizes finger movement to increase typing speed compared to QWERTY @dvorak1936. The actual benefit has been debated @noyes1983 @liebowitz1990. A 1956 study retrained typists and found Dvorak no faster than QWERTY @strong1956. The subjects were already biased to know QWERTY. The study used 10 subjects per group. The small sample size leaves the question open.

Kinkead estimated layout speed using fixed movement costs @kinkead1975. The model is very simple. It does not explain much of what affects typing speed. The model suffers from QWERTY practice bias.

This study applies the trained LightGBM typing model to measure the speed difference.

=== Methodology
This study evaluates the Dvorak layout using the trained LightGBM model.
LightGBM is selected for two reasons.
It is the only model with no statistically significant layout bias (@sec-bias), so the result depends least on the bias correction.
It also achieves the best zero-shot Dvorak accuracy (@table-performance).

The model scores 25 000 sentences from the combined Reddit and Wikipedia corpus (@sec-corpus).
For each keystroke it predicts the normalized inter-keystroke interval.
A lower value indicates faster typing.
The same sentences are scored on both QWERTY and Dvorak.
The difference is converted to a percentage speedup using the dataset's mean IKI standard deviation (50.4 ms) and mean IKI (110.5 ms).

=== Results
LightGBM predicts a mean normalized interval of 0.001 for QWERTY and -0.013 for Dvorak. Dvorak is predicted to be faster than QWERTY by a raw advantage of 0.014 standard deviations.

The raw predictions must be adjusted for layout bias. LightGBM exhibits a known bias shift of +0.0015 (@sec-bias). The model slightly underestimates Dvorak speed. Adjusting for this bias yields an advantage of 0.0158 standard deviations, equivalent to a 0.7% typing speedup for Dvorak.

Applying the uncertainty of the bias shift yields an adjusted Dvorak speedup interval ranging from 0.5% to 1.0% (p < 0.001). The entire interval is positive, which indicates a motor advantage for Dvorak.

=== Discussion
This interval understates the true uncertainty. The underlying confidence intervals resample sentences, not participants, so they miss between-typist variation (@sec-limitations). The 0.7% is therefore directional rather than exact.

The analysis would be more precise if the model were fine-tuned on Dvorak typing data.

=== Conclusion
Data-driven modeling indicates a 0.7% typing speedup for Dvorak over QWERTY. The estimate is positive across the bias-shift interval, but the intervals omit between-typist variation, so the 0.7% is an indication rather than a settled result.






