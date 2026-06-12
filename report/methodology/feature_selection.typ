== Feature selection
Feature selection simplifies the models, cutting inference time while keeping or even improving generalization.
All selection runs on fold 0 only: per-fold selection would need an arbitrary rule to reconcile disagreements between folds, for negligible gain on a fold of roughly $10^6$ samples.

The method depends on the architecture.
Linear regression uses L1 regularization to drive uninformative coefficients to zero directly.
LightGBM and the MLPs use SHAP-driven ablation against a `random_noise` control feature.

=== Interpreting SHAP
SHAP (SHapley Additive exPlanations) assigns each feature a signed, additive contribution to every prediction.
The mean absolute SHAP value ranks how much the model relies on a feature: it measures the magnitude of influence, not its direction.
Rankings are model-specific, so a feature LightGBM leans on may rank low for an MLP.
A `random_noise` feature (standard normal, no signal) sets an objective cutoff: any real feature ranking below it carries no useful information.
Correlated features split their SHAP credit, so a low value can mean a feature is redundant rather than uninformative.
The reverse also holds: a high-SHAP feature can still be removable if other features carry the same signal.
The ablation therefore tests each removal against validation MAE rather than trusting the ranking alone.

=== Linear regression
The linear baseline uses L1 (Lasso) regularization rather than SHAP-driven ablation.
L1 is intrinsic to the model: the loss that fits the coefficients also drives uninformative ones to exactly zero, handling redundancy and the heavy multicollinearity of the one-hot finger/hand encoding at once.
SHAP ablation ranks features one at a time and would keep both members of a correlated pair.

An `SGDRegressor` (`epsilon_insensitive` loss, $epsilon = 0$, equivalent to minimising MAE) trains on all candidate features with `penalty="l1"` and a fixed window ($W_"back"=2$, $W_"ahead"=1$).
$alpha$ sweeps from $10^(-4)$ to $10^(-1)$ on a log scale (10 values); the $alpha$ minimising validation MAE defines the surviving features (non-zero coefficient in at least one temporal slot).

L1 dropped four one-hot finger columns (`finger_2/4/5/7`), `finger_type_4`, and two of the three one-hot hand columns, keeping most motor and linguistic features.
This left 41 surviving features (listed in @appendix-selected-features).

=== LGBM
Feature selection uses SHAP rather than LightGBM's built-in split-gain importance.
Split-gain importance is biased toward features used in early splits.
SHAP, computed with `shap.TreeExplainer` on $50 thin 000$ validation samples, averages over all possible feature orderings, providing a stable measure of each feature's contribution regardless of split order.

A `LGBMRegressor` trains on all 36 candidate features (plus the `random_noise` control) with `objective='regression_l1'`, `n_estimators=500`, `w_back=2`, `w_ahead=1`, default hyperparameters otherwise, and `finger`/`hand` as categorical features.
Selection runs on fold 0, with SHAP importances summed across all temporal slots ($t-2$ to $t+1$).
The baseline reached MAE 0.5852 [95% CI: 0.5833, 0.5872] (full ranking in @appendix-lgbm-shap).

Six features ranked below `random_noise` (`is_pad`, `same_finger_trigram`, `scissors`, `out_triroll`, `is_word_start`, `in_triroll`) and were removed unconditionally.

Beyond those, `is_word_end` is a binary flag already captured more richly by `word_relative_pos`; `finger_type` correlates with `finger` while ranking lower; and `sequence_length` with `sequence_relative_pos` both approximate information already in `sequence_pos`.
Removing all four gave 26 features at MAE 0.5866 [95% CI: 0.5846, 0.5885], not meaningfully worse.

`unigram_frequency` correlates with both `shift` and `bigram_frequency`, suggesting redundancy.
It also acts as a near-unique identifier for each key on QWERTY, which risks the model memorising per-key biases rather than learning generalisable motor difficulty.
Removing it improved performance: 25 features, MAE 0.5743 [95% CI: 0.5723, 0.5762].

`move_cos` and `move_sin` correlate with `x` and `y` (0.72 and 0.97).
For a linear model this multicollinearity would hurt, but LGBM handles it without penalty.
Removing both, or either alone, consistently worsened results, so all four stay.

`hand` ranked weakest among the rest, and `finger` carries similar information, but removing `hand` gave MAE 0.5791 [95% CI: 0.5771, 0.5810], worse, so it stays.
`is_syllable_start` and `is_syllable_end` correlate at 0.82 with word start/end, a distinction `word_relative_pos` already encodes continuously; removing them improved to 23 features at MAE 0.5746 [95% CI: 0.5727, 0.5766].
`word_index` relates to `word_relative_pos` and `word_length`, but removing it gave MAE 0.5807 [95% CI: 0.5788, 0.5827], worse, so it stays.

SHAP was rerun on the 23-feature set (MAE 0.5771 [95% CI: 0.5751, 0.5790]; ranking in @appendix-lgbm-shap-reduced).
`redirects` now ranked lowest, but removing it gave MAE 0.5791 [95% CI: 0.5771, 0.5811], worse.
The 23 features are final (listed in @appendix-selected-features).

=== MLP Main
Because an MLP learns differently from a tree, it gets its own SHAP analysis.
A multilayer perceptron (512 → 256 → 64 → 1, GELU activations, 0.2 dropout) trains on all 51 candidate features (plus `random_noise`), with `finger`, `hand`, and `finger_type` pre-expanded to one-hot columns.
Training uses the AdamW optimizer, MAE loss, and a learning rate that halves when validation loss plateaus. The inputs are standard-normalized so that features with different scales contribute equally to the gradient updates. It trains with a batch size of $32 thin 768$, up to 15 epochs with early stopping at patience 2.
The window is `w_back=3`, `w_ahead=1`, on fold 0.
SHAP values, computed for the neural network using 300 background and 700 test samples, are summed across all 5 temporal slots.
The baseline reached MAE 0.5758 [95% CI: 0.5737, 0.5778] (full ranking in @appendix-mlp-shap).

Rather than ablating from scratch, the LGBM feature set is the starting point, since both models learn from the same data and their important features overlap.
Expanding `finger` and `hand` to one-hot columns gave 34 features at MAE 0.5737 [95% CI: 0.5717, 0.5757].

The MLP rated `is_syllable_start` and `is_syllable_end` higher than LGBM did.
This is expected: a tree can threshold `word_relative_pos` to recover word boundaries implicitly, whereas an MLP benefits from the explicit binary signal.
Adding both improved to MAE 0.5724 [95% CI: 0.5704, 0.5744].
With the syllable features now encoding position, `word_index` became redundant; removing it improved to 35 features at MAE 0.5714 [95% CI: 0.5695, 0.5734].

Adding `finger_type` alongside the finger one-hot columns, on the reasoning that it encodes finger-category symmetry explicitly, gave 40 features at MAE 0.5734 [95% CI: 0.5714, 0.5754], worse, so it was not added.

Unlike LGBM, MLPs are sensitive to highly correlated inputs, since redundant features can confuse gradient updates.
`move_sin` correlates 0.97 with `y` and `move_cos` correlates 0.72 with `x`.
Removing `move_cos` and `y`, keeping the more symmetric `move_sin` and `x`, gave MAE 0.5729 [95% CI: 0.5710, 0.5749], slightly worse.
Adding `move_cos` back alone gave 0.5758, and `y` back alone gave 0.5733, both worse than 0.5714, so all four stay.

`sequence_pos` ranked low, so `sequence_relative_pos` was tried as a replacement; the result of 0.5724 [95% CI: 0.5704, 0.5744] was not better, so `sequence_pos` stays.

SHAP was rerun on the 35-feature set (ranking in @appendix-mlp-shap-reduced).
`sequence_pos` now sat just above `random_noise`, but removing it gave MAE 0.5756 [95% CI: 0.5736, 0.5776], worse.
`finger_7` ranked near the bottom, but removing it gave 0.5757 [95% CI: 0.5737, 0.5777], worse.
`word_relative_pos` overlaps with the syllable features, but removing it gave 0.5726 [95% CI: 0.5706, 0.5746], worse.
The 35-feature set is final at MAE 0.5714 [95% CI: 0.5695, 0.5734] (listed in @appendix-selected-features).

=== MLP Linguistic
The model keeps the 7 mostly-linguistic features from the MLP Main set (listed in @appendix-selected-features).

An MLP is used rather than LGBM for this variant: exploratory analysis showed LGBM did worse on this linguistic-only subset, so an MLP is better suited to this small, mostly continuous feature set.

This is only an approximation of a pure linguistic baseline.
Many features that would normally count as linguistic are excluded because they correlate too strongly with motor features and would leak motor information back in.
For example, `unigram_frequency` near-uniquely identifies each key on a fixed layout: knowing `unigram_frequency=0.093` is enough to infer the key is "e", at which point the model can learn the motor cost of pressing "e".
The retained features are the linguistic signals with the lowest motor leakage; no clean separation exists, so more linguistic signal is available in principle than this variant can capture without confounding.

=== MLP Deep Learning
The feature set keeps mostly raw inputs (listed in @appendix-selected-features).

The motor inputs are not fully raw.
`hand` is a human-derived heuristic, but it is necessary: the layout encoding is deliberately symmetric, so `x` and `y` do not uniquely identify a key, and two keys (one per hand) can share the same `(x, y)`.
Without `hand` the model cannot tell which finger produced an event, which cripples its ability to learn motor patterns.
The symmetric coordinate encoding is kept rather than switched to absolute positions, because the mirror structure should aid generalisation across layouts that share it.
The variant is therefore not "fully deep learning".
Linguistic features are identical to MLP Linguistic.
