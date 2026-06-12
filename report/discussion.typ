= Discussion

== Interpreting the results

=== Zero-shot evaluation works
Prior data-driven typing models trained and evaluated on QWERTY alone, leaving cross-layout transfer untested.
This model is the first evaluated zero-shot on a different physical layout.
The transfer holds: the single-model MAE rises by at most 0.034 moving from QWERTY validation to Dvorak.

=== LightGBM is the most layout-neutral model
Bias shift measures how much a model's systematic error changes between QWERTY and Dvorak (@sec-bias).
LightGBM's bias shift, +0.0015, is the only one whose confidence interval includes zero, so it is statistically indistinguishable from zero (under 0.08 ms).
The other three models carry a significant shift, though each stays under 1 ms.
All shifts are positive, so the models under-credit Dvorak.
LightGBM therefore transfers across layouts with the least bias.

=== The models learn motor signal
The near-linguistic model (MLP Linguistic) sets a reference for how much accuracy comes from language statistics alone.
The motor models beat it on validation and on Dvorak.
This indicates they learned motor patterns beyond language, and these survive the move to Dvorak.

=== Motor signal is recoverable from raw positions
MLP DL does almost as well as the other motor models from key positions alone, without hand-designed motor features.
The motor signal is therefore partly recoverable from near-raw spatial input.

=== Typing is nonlinear
The nonlinear models beat Linear Regression on both layouts.
Two effects could explain this.
Features may interact: the cost of one movement may depend on another.
Features may also scale nonlinearly: a 2 cm finger movement may not cost twice a 1 cm movement.

=== MLP Main overfits
MLP Main fits the QWERTY training data best but transfers worst.
Its capacity is the likely cause.
Many features and a large network let it memorize QWERTY-specific patterns instead of a general motor rule.
Those patterns fail once the letters move.
MLP DL is simpler on both counts, fewer features and a smaller network, so it has less room to memorize.
It generalizes better.

=== Typing is anticipatory
Window size selection tests how far ahead each model looks (@sec-hpo).
For Linear Regression and LightGBM, looking one key ahead beat looking only backward.
Typing is therefore anticipatory.
The MLPs were not tested without lookahead during HPO.
They showed the same effect during the exploratory phase, so they are presumed to behave the same.
This selection ran on QWERTY only, but the effect likely holds on Dvorak too.

=== A performance ceiling remains
No model exceeds 0.31 $R^2$.
One explanation is aleatoric noise.
Typing timing varies even for the same keystroke in the same context.
Some of each IKI may be unpredictable in principle.
How much of the residual is irreducible noise rather than unmodeled signal is unclear, and this project does not measure it.

== Use cases
The models predict per-keystroke IKI for any sequence of modeled keys on QWERTY, and on Dvorak with a measured bias.
A use case is trustworthy when its keystrokes stay within this domain.
Four applications meet this condition.

+ *Dvorak versus QWERTY speed.* Both layouts have real typing data, and the Dvorak bias is measured (@sec-bias).
+ *Abbreviation dictionaries.* An abbreviation is typed as a QWERTY sequence. Predicting its per-keystroke cost is similar to the task the model was validated on.
+ *One-shot shift.* It encodes motor patterns the model has been trained on.
+ *Repeat key.* So does the repeat key.

Ensemble disagreement also offers a signal for when to distrust a prediction.
On a random layout, LightGBM and MLP DL raise their disagreement by 23% and 18% over QWERTY (@sec-uncertainty).
Both respond to out-of-distribution input.
MLP Main raises its disagreement by only 6%, so it carries a weaker signal.

The signal is partial, not clean.
For the same two models, Dvorak std is slightly lower than QWERTY std, even though Dvorak is also zero-shot (@table-uncertainty).
One interpretation is that Dvorak's letter placement involves fewer high-variance motor patterns than QWERTY.

== Limitations <sec-limitations>

=== Systems it cannot evaluate
Chording presses several keys at once to produce a single output.
The model predicts the interval between sequential single-key presses.
A chord has no interval within it.
Its surrounding transitions never appear in the training data.
Neither is predictable.
The limit is the data, not the method.
The same approach, retrained on recordings of chorded typing, could presumably predict typing speed on chorded keyboards.

A layer key works like Shift: held down, it remaps the other keys to a different set.
For example, holding space could turn the home-row keys `asdf` into `1234`.
The model represents only one such modifier, Shift.
The training data contains no other layer key, so the model cannot learn the cost of holding an arbitrary one.
Evaluating layers would require data that records them.

=== Untested domains
The training data covers standard English prose on standard row-staggered keyboards.
It lacks keystrokes for function keys (F1–F12), rare symbols, and modifiers other than Shift.
Typing patterns may also differ across languages.
Furthermore, the model's performance on alternative physical keyboards, such as ortholinear or split designs, is untested.
Generalization to these domains remains unknown.

=== Other layouts
The model runs on any layout, but its bias is known only for Dvorak.
On Colemak, an optimized layout, or a random one, the bias is unmeasured.
If it matched the small Dvorak bias, the model could rank these layouts.
How reasonable that assumption is stays unknown.
A ranking could therefore be correct without any way to confirm it, and wrong without any way to detect it.
Measuring bias on more layouts would settle this, but that needs typing data on each layout.

=== Data
After filtering to fast, US, 9–10-finger typists, the training data is almost entirely QWERTY: $7 thin 944$ participants against $17$ on Dvorak, or $4 thin 766 thin 251$ keystrokes against $10 thin 071$.
The Dvorak sample is small, which widens the confidence intervals; a larger sample would resolve the bias shifts more precisely.
Dvorak typists are also self-selected: people who chose to learn a non-default layout may differ from the QWERTY population in typing habits or motivation, introducing a confound the data cannot control for.

The data is real typing, so the linguistic confound is not fully removed.
Common sequences may be fast because they are practiced, not because they are physically easier.
Explicit linguistic features mitigate this but cannot eliminate it.

The model targets skilled typists.
Transfer to slower typists is untested.
Slower typists are noisier @dhakal2018.
Many of the features assume standard touch-typing, so training on them would likely be harder.

=== No held-out QWERTY set
A QWERTY dataset should have been held out from every fold, so the ensemble could be scored on QWERTY as it is on Dvorak.
None was, so the QWERTY number is a single-submodel out-of-fold score (@sec-evaluation) and underestimates the deployed ensemble.

=== Validation context leakage
Word-level folds still expose the surrounding keys of an unseen validation word.
The model can use that context to predict the held-out timings.
The QWERTY validation MAE is therefore likely slightly optimistic.
A sentence-level split with word- and bigram-level dropout would reduce this leakage.
The Dvorak evaluation uses fully held-out data and is unaffected.

=== Typos
Typos make up 3.93% of keystrokes and are slow, so discarding them rather than modeling them is a substantive simplification.
The approach assumes that speed and error rate correlate, and that typos are layout-independent.
Neither is tested, so whether discarding them biases the results is unknown.

=== LightGBM inference speed
LightGBM takes 106.3 s per million keystrokes, an order of magnitude slower than the other models (@table-speed).
Layout optimization requires scoring many candidate layouts against a corpus.
The inference speed makes LightGBM impractical for that optimization.

== Future work
The main open problem is training a layout-unbiased model.
Only LightGBM shows no statistically significant cross-layout bias; the other three models carry a small but significant shift.
A speed difference between layouts may be smaller than a model's bias: a 0.5% Dvorak speedup is undetectable if the model carries 1% layout bias.
Two directions could narrow that gap.

=== Better data
Two approaches could provide better data.

Collecting real typing data on several layouts would almost certainly cut bias.
Even one more layout would help considerably.
Three layouts let one serve for training, one for validation, and one for test.
This also simplifies data splitting and feature selection

A cleaner route removes the confounds at the source.
Participants type randomly generated sequences until speed plateaus; the plateau time estimates motor cost.
Typing nonsense removes the linguistic familiarity confound.
Training to plateau removes the layout familiarity confound.
Unlike isolated-digraph tapping @iseri2015, random sequences preserve the influence of surrounding keys.
A model trained on such data estimates motor cost directly, free of the linguistic and practice biases this project can only partly control.
The approach requires a dedicated data-collection study rather than reuse of existing keystroke logs.
How much data such a study needs is unclear.
Learning curves estimated from existing data could bound it.

=== Better architecture
The fixed-window input is overly complex.
Letting some window positions carry fewer features would be simpler, faster, and less prone to overfitting.
This was not done because it is harder to implement.
MLP DL, with its simpler feature set, showed less overfitting than MLP Main, suggesting simpler architectures generalize better.

=== Confidence intervals
The intervals resample sentences, capturing correlation within a sentence. They do not resample participants. Correlation between keystrokes from the same typist is therefore not modeled. Accounting for it is future work.
