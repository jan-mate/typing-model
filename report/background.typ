#import "/styles.typ": *

= Background

== Typing speed metrics
Typing speed has two common units.
Words per minute (WPM) aggregates over a passage.
Inter-Keystroke Interval (IKI) measures the time between two consecutive keypresses, typically in milliseconds.
This project works with IKI.

== Prior models
Early models of typing speed used for layout evaluation relied on heuristics.
Dvorak's original work @dvorak1936 motivated layout design primarily through reduced finger travel and hand alternation.
Later models such as CARPALX @krzywinski-carpalx formalised this into weighted effort scores combining row penalties, finger usage, and bigram costs.
These models share a common limitation: their weights are hand-assigned, not learned from typing data.
They also typically target both speed and ergonomics; this project addresses speed only.

A small body of work used measured typing data.

Kinkead @kinkead1975 conducted an early computational evaluation.
They captured $115 thin 000$ keystrokes from 22 typists during standardised speed typing tests on a QWERTY layout, and classified each keystroke by its relation to the previous one (opposite hand, same hand different finger, same finger, repeated key).
The analysis identified alternate-hand strokes as fastest and same-finger strokes as slowest.
They used mean class times and English digram frequencies to estimate Dvorak speed without recruiting Dvorak typists. The model predicted Dvorak as 2.6% faster than QWERTY and an optimal layout as 7.6% faster.

Hiraga et al. @hiraga1980 took a regression-based approach.
From $302 thin 392$ keystrokes typed by a single professional typist on QWERTY, they fit a multivariate linear model predicting IKI from hand alternation, row transition, finger distance, and keystroke frequency.
Hand alternation was the dominant factor, giving a ≈40 ms advantage for alternate-hand strokes, consistent with Kinkead's findings.

Işeri and Ekşioğlu @iseri2015 measured digraph timings directly through repeated tapping rather than real typing, for the purpose of designing an optimized Turkish layout.
Seven participants tapped each of 241 same-hand digraph pairs in isolation at maximum speed for two minutes per pair, yielding per-digraph timing rates.
Alternating-hand digraph costs were not measured directly; they were derived from the same-hand results using a scaling factor from Salthouse @salthouse1986, making that part a heuristic estimate rather than an empirical measurement.
An ANOVA identified the column (which finger), row, hand, and period as significant factors, with finger column carrying the most explanatory power.
Combining these empirical digraph costs with Turkish digraph frequencies in a quadratic assignment problem, they derived an optimized layout that outperformed the existing Turkish layouts on both the cost objective and Dvorak's heuristic criteria.

Williams et al. @williams2026 took the largest-scale data-driven approach to date.
They trained on real typing data from the 136M Keystrokes dataset @dhakal2018, where $168 thin 000$ participants each typed 15 sentences from a pool of $1 thin 493$ English items.
For each sentence, individual typing speeds were z-scored within participant and averaged across participants, producing a sentence-level "typability" score.
Random forest regression selected eight predictors from 30 candidates spanning linguistic, layout, and biomechanical attributes (most importantly the proportion of lowercase characters, total keystrokes, and syllables per word), which were then combined in a multiple linear regression model.
The model explained 74% of typability variance in training and 68% on a held-out test set.

== Limitations of existing approaches

@limitations-comparison summarises which limitations apply to each prior approach.

#figure(
  table(
    columns: (1.8fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    inset: 6pt,
    align: (col, row) => if col == 0 { left + horizon } else { center + horizon },
    stroke: none,
    fill: zebra,

    [*Limitation*], [*CARPALX*], [*Kinkead*], [*Hiraga*], [*Işeri*], [*Williams*], [*This work*],
    [Sentence-level granularity], [✗], [], [], [], [✗], [],
    [Linear model], [✗], [✗], [✗], [✗], [✗], [],
    [Arbitrary weights], [✗], [], [], [(✗)], [], [],
    [Hand-designed features], [✗], [✗], [✗], [✗], [✗], [(✗)],
    [No anticipation], [(✗)], [✗], [✗], [✗], [], [],
    [Linguistic confounding], [n/a], [✗], [✗], [], [✗], [(✗)],
    [Small sample], [n/a], [✗], [✗], [✗], [], [(✗)],
    [Isolated tapping], [n/a], [], [], [✗], [], [],
    [QWERTY only], [], [✗], [✗], [], [✗], [(✗)],
  ),
  caption: [Limitations of prior approaches. ✗ = present; (✗) = partially present; blank = absent; n/a = not applicable.],
) <limitations-comparison>

*Granularity.* The Typability Index predicts at sentence level, producing one score regardless of how many keystrokes a sentence contains.
IKI is necessary to capture key-to-key transitions directly.

*Linearity.* All previous models combine features linearly.
Linear models cannot easily capture interactions such as a longer reach being especially bad when performed using pinky fingers.

*Arbitrary heuristic weights.* Models like CARPALX and Dvorak's original criteria combine factors using weights chosen by researchers rather than learned from data.
The relative cost of, say, a pinky reach versus an index stretch has not been empirically grounded.

*Hand-designed features.* All data-driven predecessors rely on hand-designed features (finger travel, syllable counts, word frequency).
Whether these capture the relevant motor signal completely is untested; a deep learning model trained on raw data could find patterns no one has labeled.

*Anticipation.* Salthouse @salthouse1986 argues that typing is chunked: typists process upcoming characters in parallel with executing the current keystroke.
If this is correct, IKI depends not only on the preceding key but also on what follows.
The heuristic and bigram-cost models ignore future context entirely.

*Linguistic confounding.* Kinkead's and Hiraga et al.'s keystrokes came from real QWERTY typing tests, where linguistically common bigrams may appear fast because typists have practised them more, not because the underlying movement is easier.
This conflates biomechanical cost with practice.

*Small samples.* Kinkead used 22 typists, Hiraga et al. a single typist, and Işeri and Ekşioğlu seven participants.
Estimates from such small samples may not generalise.

*Isolated tapping.* Işeri and Ekşioğlu avoided the linguistic confound by having participants tap each digraph in isolation, but the resulting measure reflects motor capacity for one repeated motion rather than timing within real text, where surrounding keys influence each transition.

*Generalisation.* All data-driven predecessors were trained and evaluated on QWERTY data with standard English prose.
It remains untested whether models transfer to alternative layouts (Dvorak, Colemak).

Training a model that avoids these limitations is not straightforward.
Keystroke datasets are almost exclusively QWERTY-based, so there is a risk of learning QWERTY-specific patterns rather than general motor principles.
Linguistic and motor cost are entangled: common sequences may be fast because they are practised, not because they are physically easier.
IKI data is also noisy; the same transition typed by the same person in the same context shows variance.

== This project's response
Training on per-keystroke IKI from the 136M Keystrokes dataset @dhakal2018
($168 thin 000$ participants) resolves the granularity issue and some sample-size issues.
A windowed input spanning future keys adds anticipatory context.
Non-linear architectures (gradient-boosted trees and MLPs) capture interactions
that linear models miss.
Learned weights replace hand-assigned heuristics.
Zero-shot evaluation on Dvorak tests cross-layout generalization directly.
The linguistic confound is only partly resolved.
The dataset is real QWERTY typing, so familiarity and motor cost stay entangled.
Explicit linguistic features (word frequency, bigram frequency) let the model assign some of the speed to familiarity rather than motor cost, but cannot separate the two fully.

