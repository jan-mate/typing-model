#import "/styles.typ": *

== Data preparation

=== Participant filtering <participant-filtering>
The pipeline filters the data to fast typists (above 80 WPM) who use 9–10 fingers and are US-based.

The subset is motivated as follows:

- *Reduced variance.* Fast typists vary less in their IKI than slow typists @dhakal2018, which gives the model a cleaner signal.
- *Matching the target users.* The model targets skilled, fast typists, so training on them matches the population it predicts for.
- *Simpler enrichment.* Assuming 9–10 fingers fixes which finger presses each key, simplifying the finger and hand features added during enrichment.
- *Consistent layout.* Restricting to US participants avoids layout variants such as UK-QWERTY or German QWERTZ, which place punctuation and symbols on different keys and would need more complex labeling. The data records country rather than physical layout, so US participants are assumed to use the US QWERTY layout.

The 80 WPM threshold trades typing speed against data volume.
80 WPM is assumed fast enough to represent the target population while retaining enough training data.
Of the $56 thin 469$ US QWERTY participants who use 9–10 fingers, $8 thin 853$ (16%) type above 80 WPM, but only $1 thin 449$ (3%) type above 100 WPM.
A 100 WPM cutoff would leave roughly six times fewer participants, risking too little data to train and validate the models reliably.

A matching subset of fast Dvorak typists serves as a held-out benchmark for cross-layout generalisation.

=== Typos
Real typing contains errors, which must be handled before the data can train a model.
A modified Levenshtein distance identifies five typo types by comparing the raw input against the target sentence:

+ *Substitution* — a wrong key (e.g. "c i t" for "cat").
+ *Insertion* — an extra key (e.g. "c i a t").
+ *Deletion* — a skipped key (e.g. "c t").
+ *Transposition* — two keys swapped (e.g. "c t a").
+ *Proofreading* — any sequence using Backspace to correct a mistake.

When a typo is detected, the pipeline splits the sequence at the error.
The split discards the IKI immediately before and after the typo, and the following correct keystrokes start a new segment.
The model therefore trains only on sequences where the typist executed the intended motor plan.
Of the $4 thin 961 thin 134$ annotated QWERTY keystrokes, the filter removes $194 thin 879$ (3.93%) as typos.

Three reasons motivate this approach:

- *Simplicity.* Removing typos is simpler than modeling them, which would require the model to predict both an IKI and an error probability, alongside the time to recover from each typo type.
- *Sparse, noisy signal.* Typos are only 3.93% of keystrokes, and their recovery timing is erratic, so there is too little consistent data to learn from reliably.
- *Cleaner motor signal.* Discarding error-recovery intervals leaves only the motor flow the model is meant to predict.

The approach rests on two assumptions.
The first is that typing speed and error rate are correlated, so optimizing for speed already avoids error-prone movements.
The second is that typos are layout-independent.
Neither is tested here.
If either fails, removing typos could bias the model: it might rate a fast but error-prone key transition as good, having seen that transition only when typed correctly, or it might skew the cross-layout QWERTY-to-Dvorak comparison.
Mistyped transitions also carry real motor movement, such as the "oz" sequence pressed when "foxes" is mistyped as "fozes", which the approach discards rather than learns from.

For example, a typist intending "foxes" but typing "fozes" results in the keystream shown in @table-typo-example.

#figure(
  table(
    columns: 6,
    align: center + horizon,
    stroke: none,
    fill: flat,
    [*Key*], [f], [o], [z], [e], [s],
    [*Typo?*], [0], [0], [1], [0], [0],
    [*IKI (ms)*], [NaN], [110], [150], [120], [105]
  ),
  caption: [A keystroke sequence containing a typo on the third key.],
) <table-typo-example>

The pipeline splits this to remove the typo, leaving two clean segments shown in @table-typo-split.

#figure(
  grid(
    columns: 2,
    column-gutter: 1em,
    table(
      columns: 3,
      align: center + horizon,
      stroke: none,
      fill: flat,
      [*Key*], [f], [o],
      [*IKI*], [NaN], [110],
    ),
    table(
      columns: 3,
      align: center + horizon,
      stroke: none,
      fill: flat,
      [*Key*], [e], [s],
      [*IKI*], [NaN], [105]
    )
  ),
  caption: [The segments remaining after the typo and its surrounding intervals are removed.],
) <table-typo-split>

=== Outlier detection and z-scores
Keystroke timings contain non-typing noise, such as pauses from external distraction.
An IKI more than 4 standard deviations (SD) above a participant's mean counts as an outlier, and the sequence splits at that point, as with typos.
This keeps 98.98% of IKI.
A per-participant threshold, rather than a fixed cutoff, accounts for differing skill levels.
No cutoff cleanly separates distraction from genuine typing. The 4-SD value is a subjective judgement rather than a data-derived one.

After outlier removal, each participant's IKI becomes a z-score:
$ z = (x - mu_p) / sigma_p $
where $mu_p$ and $sigma_p$ are the mean and standard deviation of that participant's non-outlier IKI.
Per-participant normalization removes absolute speed differences between typists, so the model learns the relative difficulty of key transitions rather than individual typing speed.
It also gives each participant unit variance, so a typist with a wider spread of IKI no longer carries disproportionate weight in the model's training loss.

After all filtering and cleaning, the data comprises $4 thin 766 thin 251$ QWERTY keystrokes from $7 thin 944$ participants and $10 thin 071$ Dvorak keystrokes from $17$ participants.
