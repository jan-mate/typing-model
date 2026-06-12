== Repeat Key <repeat-key>

=== Introduction
A repeat key is a dedicated key that re-emits the previously typed character.
It converts same-finger repeated keypresses, such as typing "poppy" as `p` `o` `p` `RPT` `y` or "squirrel" as `s` `q` `u` `i` `r` `RPT` `e` `l`, into alternating-finger sequences.
Standard typing requires rapidly lifting and pressing the same finger twice.
The repeat key removes this motion.
The effect on overall typing speed depends on how frequently double letters occur and how fast the alternating sequence is.

=== Methodology
This study evaluates repeat key speed using the trained LightGBM model.
LightGBM transfers to unfamiliar inputs with the least layout bias and the best Dvorak MAE (@sec-bias).
The evaluation uses the 25 000-sentence Reddit and Wikipedia corpus (@sec-corpus).

Evaluating the repeat key requires assigning it to a physical slot.
This study assumes the repeat key (`RPT`) occupies the right-pinky home slot, displacing the semicolon `;`.
The semicolon does not appear in the evaluated sentences, so replacing it with a repeat key leaves them fully typable.
The result depends on this slot choice; the repeat key's value on other slots is untested.

The repeat key introduces sequences that do not exist in standard typing.
Because the `RPT` key lacks real-world bigram data, the evaluation computes its bigram frequencies from the occurrences of literal double letters in the corpus.
For example, the bigram `t` followed by `RPT` takes the same frequency as `tt`.
The word frequency features remain unchanged.
The model scores the corpus twice: once with standard double letters and once with double letters rewritten to use the repeat key.
The difference in predicted time represents the speedup of the mechanical change.

=== Results
Over the 25 000 sentences, using the repeat key for all double letters is 0.26% faster than standard typing.

=== Discussion
The 0.26% rests on a fixed assumption: the repeat key takes the `;` slot.
How it would perform on other slots is untested.

The repeat key needs little extrapolation: its bigram frequencies come from real double letters and the word frequencies are unchanged.
MLP Main scored best on QWERTY, so it may be more accurate here too and might have scored the repeat key differently than LightGBM.
Scoring the repeat key under the other models would test whether the 0.26% holds across architectures and is left for future work.

Another value of the repeat key is that it frees the physical double-tap gesture.
Because the repeat key handles literal double letters, a physical double-press (e.g., tapping `r` twice) is no longer needed for ordinary spelling.
This gesture becomes available for other functions, such as keyboard shortcuts, macros, or outputting a full word.

When these freed double-taps are used for word output, a dictionary of 23 such entries reaches a 1.52% speedup (@sec-abbr-dict).

=== Conclusion
On the right-pinky home slot, the repeat key increases typing speed by 0.26%.
It also unlocks the physical double-tap gesture for secondary functions, such as outputting full words.

