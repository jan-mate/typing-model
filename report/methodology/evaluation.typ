== Evaluation <sec-evaluation>
Each model is an ensemble of 10 submodels, evaluated on QWERTY and zero-shot on Dvorak.

Validation MAE scores every QWERTY keystroke with the one submodel that never trained on it, then pools these out-of-fold predictions and measures the error once.
This is a single-submodel score. Every QWERTY keystroke trained 9 of the 10 submodels, so only one submodel is leakage-free per keystroke. The full ensemble therefore cannot be scored on QWERTY without a held-out set (@sec-limitations).

Dvorak is fully held out, so all 10 submodels are leakage-free on it. This gives two numbers that differ only in the order of two steps: averaging the 10 submodels, and measuring the error.
The single-model Dvorak MAE measures each submodel's error, then averages the 10 errors. It is computed the same way as Validation MAE, so the two are directly comparable and measure cross-layout transfer.
The ensemble Dvorak MAE reverses the order: it averages the 10 submodels into one prediction, then measures that prediction's error. Averaging predictions cancels some error, so the ensemble usually scores lower, and it is the predictor the applications use.

A second metric, bias, is the mean signed error: the average of predicted minus measured IKI. Unlike MAE, it keeps the sign, so it captures whether the model over- or under-predicts on average rather than how far off it is. Bias is the same for the single-model and ensemble predictors, because bias is a linear function of the predictions: the average of the errors equals the error of the average. MAE uses absolute error, which is nonlinear and breaks that equality.

The baseline always predicts zero, the mean of z-score-normalized IKI by construction, so its error marks the performance of a model with no predictive power.

=== Confidence intervals
A bootstrap quantifies the uncertainty in the reported statistics.
The procedure resamples the data with replacement $2 thin 000$ times and takes the 2.5th and 97.5th percentiles of the resampled statistic as a 95% confidence interval.

The resampling unit is the sentence, not the individual keystroke.
Keystrokes within a sentence are correlated: the sliding feature window overlaps between neighbours, and typing rhythm links successive intervals.
Resampling individual keystrokes treats these correlated values as independent, which underestimates the uncertainty and narrows the interval.
Resampling whole sentences preserves the within-sentence correlation, which widens the intervals.

Sentence resampling can also overcorrect. Since the feature window spans only a few keystrokes, keys far apart in a long sentence are nearly independent; whole-sentence clustering discards these degrees of freedom, widening the interval more than correlation alone requires.

A stricter design would also resample participants. Each typist has their own patterns, so keystrokes from the same person are correlated. Resampling sentences leaves this correlation in the data. This narrows the interval.

