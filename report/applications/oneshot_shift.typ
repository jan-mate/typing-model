#import "/styles.typ": *

== One-Shot Shift <oneshot-shift>

=== Introduction
Typing capital letters is slower than typing lowercase letters.
Standard capitalization requires pressing the Shift key and the target letter simultaneously.
One-shot shift (1SS) modifies this: tapping a dedicated one-shot key shifts only the next keystroke, then the modifier releases.
Pressing the one-shot key then `c` outputs `C`.
The mechanism was originally developed to assist typists with motor impairments @vanderheiden1982[p. 147].
It eliminates the simultaneous press but adds an additional keystroke.
The net effect on typing speed is untested.

=== Methodology
This study evaluates 1SS using the trained LightGBM model.
1SS is scored on QWERTY, but the one-shot key is a novel input.
Its motor transition is familiar, a home-row key followed by a letter, but its specific feature encoding is not.
The prediction is therefore a mild extrapolation, which makes robustness the deciding criterion.
LightGBM transfers across layouts with the least bias (@sec-bias), the best available evidence of robustness, so it is selected.

The model scores a corpus sample twice: once with standard capitalization and once with isolated capitals rewritten using 1SS.
An isolated capital is a single capital letter with no adjacent capital.
This includes word-initial capitals like the "F" in "Fox" and mid-word capitals like the "H" in "GitHub", but excludes all-caps runs like "NASA".
The evaluation uses 25 000 sentences from the Reddit and Wikipedia corpus (@sec-corpus), containing 41 154 isolated capitals.
The difference in total predicted time between the 1SS and standard versions represents the cost or saving of 1SS.

Evaluating 1SS requires assigning the one-shot key to a physical slot.
Standard Shift keys lack motor features in the model because the 136M Keystrokes dataset @dhakal2018 does not record which Shift key was used.
The semicolon slot serves as the one-shot key.
It is absent from the evaluation corpus and occupies a home-row position with a low predicted movement cost.

The semicolon never appears in the training data.
Its motor features mirror the `a` key on the opposite home row, so the model can still predict its movement cost.
The linguistic features are the harder part.
This study uses an In-Word encoding: the model treats the one-shot key as the first character of the word.
The one-shot key receives the word-start features, and the capitalized letter becomes the second character.
@oss-encoding compares the features for "Fox" under both systems.

#figure(
  stack(
    dir: ttb,
    spacing: 1em,
  [(a) Standard],
  table(
    columns: 4,
    inset: 5pt,
    align: center + horizon,
    stroke: none,
    fill: zebra,

    [*Key*], [F], [o], [x],
    [*Shift*], [1], [0], [0],
    [*Bigram freq*], [6.06], [5.37], [5.08],
    [*Word freq*], [4.66], [4.66], [4.66],
    [*Word start*], [1], [0], [0],
    [*Rel. pos*], [0.00], [0.50], [1.00],
  ),
  [(b) One-shot shift (In-Word Encoding)],
  table(
    columns: 5,
    inset: 5pt,
    align: center + horizon,
    stroke: none,
    fill: zebra,

    [*Key*], [1SS], [f], [o], [x],
    [*Shift*], [0], [0], [0], [0],
    [*Bigram freq*], [7.58], [6.06], [6.45], [5.08],
    [*Word freq*], [4.66], [4.66], [4.66], [4.66],
    [*Word start*], [1], [0], [0], [0],
    [*Rel. pos*], [0.00], [0.33], [0.67], [1.00],
  ),
  ),
  caption: [Enriched features for "Fox" typed normally (a) and with one-shot shift (b). Each word is preceded by a space, so the first key's bigram frequency is its word-initial (space $arrow.r$ key) value. The In-Word encoding treats the one-shot key as the start of the word, shifting the relative positions of subsequent letters.],
) <oss-encoding>

This encoding is chosen because it keeps the post-space slot occupied by a word-start.
In-Word makes the one-shot key that word-start, so the slot still carries a word frequency.
The Boundary alternative places the one-shot key between the space and the word, where it belongs to no word and has no word frequency.
The model has never seen a key without a word frequency in the post-space slot, making that transition out-of-distribution (OOD).
In-Word is therefore less OOD.
It also simplifies the encoding of a word with a mid-word capital, like the "H" in "GitHub".

A capital at the start of a sentence has no preceding interval in the dataset.
Scoring that interval would unfairly favor the standard version, so the 1SS version drops it too.
Sentence-initial capitals are therefore scored as free under both schemes.
The measured saving comes only from isolated capitals that are not sentence-initial.

=== Results
Over 25 000 sentences, 1SS is 0.71% faster than standard capitalization.
Each non-initial isolated capital saves an average of 18.5 ms.
Extending the same saving to sentence-initial capitals would raise the overall speedup to 1.1%.

=== Discussion
The interpretation is that 1SS yields a speedup on the semicolon slot.
Whether it survives placing 1SS on the Shift key itself is unclear.
The semicolon may be a faster slot, though the two Shift keys allow hand alternation.

The prediction depends on the In-Word feature encoding.
A different encoding of the one-shot key could change the result.
It is unclear which encoding best models how a typist mentally chunks the key.
Without behavioral data, this cannot be confirmed.

LightGBM was chosen as the most robust model, but MLP Main and MLP DL could have been scored on 1SS too.
Their predictions would test whether the saving is robust across architectures or specific to LightGBM.

The 0.71% figure is conservative.
Sentence-initial capitals are scored as zero saving because the dataset segments sentences, removing the first keystroke's preceding interval.
In continuous prose such a capital is preceded by a space, like any word-initial capital, so 1SS should save there too.
The 1.1% estimate assumes it saves the 18.5 ms, which this evaluation cannot confirm.

=== Conclusion
Assuming the typist mentally chunks the modifier as part of the word and the one-shot key occupies a fast position, one-shot shift is faster than standard capitalization.
On QWERTY, 1SS reduces typing time by an average of 18.5 ms per isolated capital, yielding a 0.71% overall speedup, or about 1.1% if the benefit extends to sentence-initial capitals.

