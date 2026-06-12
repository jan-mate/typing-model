#import "/styles.typ": *

#show figure: set block(breakable: true)

= Appendix

== Keyboard Layouts <appendix-layouts>
To evaluate cross-layout generalization, models trained on QWERTY are tested on alternative layouts. Visual representations of these layouts are provided for reference.

#figure(
  image("all_layouts.png", width: 100%),
  caption: [The four layouts evaluated in this study: US QWERTY (top left), Dvorak (top right), Colemak (bottom left), and a procedurally generated Random layout (bottom right) used to test extreme out-of-distribution performance.],
) <appendix-layout-grid>

== SHAP feature importance <shap-values>
SHAP values were computed during feature selection (see the LGBM and MLP subsections). For each model, two rankings are shown: one on the full candidate set and one on the reduced set after ablation. A `random_noise` feature (standard normal, no signal) is included in every run as an objective cutoff — features ranking below it are uninformative.

=== LGBM, all features <appendix-lgbm-shap>
#figure(
  caption: [LGBM SHAP feature importance on all 36 candidate features (plus the `random_noise` control). Computed via `shap.TreeExplainer` on 50,000 validation samples. Validation MAE: 0.5852 [95% CI: 0.5833, 0.5872].],
  table(
    columns: (auto, auto, auto),
    align: (right, left, right),
    inset: 5pt,
    stroke: none,
    fill: zebra,
    table.header[*Rank*][*Feature*][*SHAP*],
    [1], [`bigram_frequency`], [0.22321],
    [2], [`word_frequency`], [0.17523],
    [3], [`finger`], [0.10542],
    [4], [`same_finger`], [0.08113],
    [5], [`unigram_frequency`], [0.07365],
    [6], [`same_hand`], [0.06400],
    [7], [`move_dist`], [0.05593],
    [8], [`repetition`], [0.03150],
    [9], [`same_finger_skipgram`], [0.02739],
    [10], [`move_sin`], [0.02359],
    [11], [`x`], [0.02280],
    [12], [`y`], [0.02278],
    [13], [`word_relative_pos`], [0.02256],
    [14], [`word_length`], [0.02027],
    [15], [`move_cos`], [0.01924],
    [16], [`is_syllable_start`], [0.01722],
    [17], [`double_row_jump`], [0.01553],
    [18], [`out_roll`], [0.01513],
    [19], [`sequence_pos`], [0.01262],
    [20], [`in_roll`], [0.01208],
    [21], [`shift`], [0.01000],
    [22], [`skipgram_repetition`], [0.00962],
    [23], [`word_index`], [0.00907],
    [24], [`is_syllable_end`], [0.00788],
    [25], [`redirects`], [0.00619],
    [26], [`hand`], [0.00617],
    [27], [`sequence_length`], [0.00400],
    [28], [`sequence_relative_pos`], [0.00365],
    [29], [`finger_type`], [0.00244],
    [30], [`is_word_end`], [0.00215],
    [31], [*`random_noise`*], [*0.00178*],
    [32], [`in_triroll`], [0.00074],
    [33], [`is_word_start`], [0.00066],
    [34], [`same_finger_trigram`], [0.00054],
    [35], [`out_triroll`], [0.00053],
    [36], [`scissors`], [0.00039],
    [37], [`is_pad`], [0.00000],
  )
)

=== LGBM, reduced 23 features <appendix-lgbm-shap-reduced>
#figure(
  caption: [LGBM SHAP feature importance after ablation, on the final 23-feature set. Validation MAE: 0.5771 [95% CI: 0.5751, 0.5790].],
  table(
    columns: (auto, auto, auto),
    align: (right, left, right),
    inset: 5pt,
    stroke: none,
    fill: zebra,
    table.header[*Rank*][*Feature*][*SHAP*],
    [1], [`bigram_frequency`], [0.20960],
    [2], [`word_frequency`], [0.17620],
    [3], [`finger`], [0.12619],
    [4], [`same_finger`], [0.08229],
    [5], [`move_dist`], [0.07721],
    [6], [`same_hand`], [0.06852],
    [7], [`y`], [0.03581],
    [8], [`repetition`], [0.03349],
    [9], [`x`], [0.03048],
    [10], [`same_finger_skipgram`], [0.02787],
    [11], [`move_sin`], [0.02783],
    [12], [`word_relative_pos`], [0.02548],
    [13], [`move_cos`], [0.02016],
    [14], [`word_length`], [0.01961],
    [15], [`shift`], [0.01825],
    [16], [`word_index`], [0.01799],
    [17], [`double_row_jump`], [0.01752],
    [18], [`sequence_pos`], [0.01618],
    [19], [`out_roll`], [0.01494],
    [20], [`skipgram_repetition`], [0.01161],
    [21], [`in_roll`], [0.01079],
    [22], [`hand`], [0.00985],
    [23], [`redirects`], [0.00549],
    [24], [*`random_noise`*], [*0.00287*],
  )
)

=== MLP, all features <appendix-mlp-shap>
#figure(
  caption: [MLP SHAP feature importance on all 51 candidate features (plus the `random_noise` control). Computed via `shap.DeepExplainer` with 300 background and 700 test samples. Validation MAE: 0.5758 [95% CI: 0.5737, 0.5778].],
  table(
    columns: (auto, auto, auto),
    align: (right, left, right),
    inset: 5pt,
    stroke: none,
    fill: zebra,
    table.header[*Rank*][*Feature*][*SHAP*],
    [1], [`word_frequency`], [0.20461],
    [2], [`bigram_frequency`], [0.18802],
    [3], [`move_dist`], [0.09535],
    [4], [`shift`], [0.09126],
    [5], [`move_sin`], [0.08026],
    [6], [`same_hand`], [0.07798],
    [7], [`move_cos`], [0.06875],
    [8], [`same_finger`], [0.06404],
    [9], [`x`], [0.06051],
    [10], [`unigram_frequency`], [0.05010],
    [11], [`is_syllable_start`], [0.04729],
    [12], [`word_length`], [0.04547],
    [13], [`repetition`], [0.04444],
    [14], [`is_word_start`], [0.03962],
    [15], [`is_syllable_end`], [0.03879],
    [16], [`hand_0`], [0.03783],
    [17], [`in_roll`], [0.03695],
    [18], [`finger_6`], [0.03592],
    [19], [`word_index`], [0.03581],
    [20], [`hand_1`], [0.03486],
    [21], [`is_word_end`], [0.03305],
    [22], [`finger_type_3`], [0.03291],
    [23], [`hand_2`], [0.03219],
    [24], [`out_roll`], [0.03137],
    [25], [`same_finger_skipgram`], [0.02964],
    [26], [`y`], [0.02960],
    [27], [`double_row_jump`], [0.02921],
    [28], [`finger_4`], [0.02910],
    [29], [`skipgram_repetition`], [0.02894],
    [30], [`finger_type_4`], [0.02726],
    [31], [`finger_2`], [0.02635],
    [32], [`finger_type_2`], [0.02443],
    [33], [`finger_3`], [0.02394],
    [34], [`word_relative_pos`], [0.02382],
    [35], [`same_finger_trigram`], [0.02164],
    [36], [`sequence_relative_pos`], [0.02028],
    [37], [`finger_9`], [0.02007],
    [38], [`finger_type_0`], [0.01840],
    [39], [`redirects`], [0.01838],
    [40], [`in_triroll`], [0.01818],
    [41], [`out_triroll`], [0.01439],
    [42], [`finger_8`], [0.01430],
    [43], [`finger_type_1`], [0.01426],
    [44], [`finger_0`], [0.01284],
    [45], [`sequence_length`], [0.01242],
    [46], [`sequence_pos`], [0.01225],
    [47], [`finger_7`], [0.00985],
    [48], [`finger_1`], [0.00897],
    [49], [*`random_noise`*], [*0.00691*],
    [50], [`scissors`], [0.00641],
    [51], [`is_pad`], [0.00610],
    [52], [`finger_5`], [0.00000],
  )
)

=== MLP, reduced 35 features <appendix-mlp-shap-reduced>
#figure(
  caption: [MLP SHAP feature importance on the reduced 35-feature set. Validation MAE: 0.5714 [95% CI: 0.5695, 0.5734].],
  table(
    columns: (auto, auto, auto),
    align: (right, left, right),
    inset: 5pt,
    stroke: none,
    fill: zebra,
    table.header[*Rank*][*Feature*][*SHAP*],
    [1], [`word_frequency`], [0.21279],
    [2], [`bigram_frequency`], [0.18681],
    [3], [`move_dist`], [0.11895],
    [4], [`shift`], [0.09630],
    [5], [`move_sin`], [0.08676],
    [6], [`move_cos`], [0.07469],
    [7], [`same_hand`], [0.07331],
    [8], [`same_finger`], [0.06606],
    [9], [`finger_4`], [0.05777],
    [10], [`x`], [0.05530],
    [11], [`finger_6`], [0.05373],
    [12], [`word_length`], [0.05194],
    [13], [`hand_0`], [0.04867],
    [14], [`hand_2`], [0.04809],
    [15], [`is_syllable_start`], [0.04415],
    [16], [`hand_1`], [0.04389],
    [17], [`finger_3`], [0.04198],
    [18], [`in_roll`], [0.04178],
    [19], [`repetition`], [0.03776],
    [20], [`y`], [0.03653],
    [21], [`is_syllable_end`], [0.03647],
    [22], [`word_relative_pos`], [0.03643],
    [23], [`out_roll`], [0.03216],
    [24], [`finger_2`], [0.03169],
    [25], [`same_finger_skipgram`], [0.03161],
    [26], [`skipgram_repetition`], [0.03066],
    [27], [`double_row_jump`], [0.02959],
    [28], [`finger_9`], [0.02266],
    [29], [`redirects`], [0.02070],
    [30], [`finger_0`], [0.01918],
    [31], [`finger_8`], [0.01807],
    [32], [`finger_1`], [0.01171],
    [33], [`finger_7`], [0.01108],
    [34], [`sequence_pos`], [0.00967],
    [35], [*`random_noise`*], [*0.00964*],
    [36], [`finger_5`], [0.00000],
  )
)

== Hyperparameter Optimization Details <appendix-hpo>
This section details the search spaces and final hyperparameter configurations.

=== Linear Regression
*Search space (per trial):*
- `penalty`: `l1`, `l2`, or `elasticnet`
- `alpha`: $10^(-5)$ to $10^(-1)$, log-scaled
- `l1_ratio`: 0.0 to 1.0 (only if `penalty="elasticnet"`)
- `learning_rate`: `invscaling` or `adaptive`
- `eta0`: $10^(-3)$ to $10^(-1)$, log-scaled

*Best hyperparameters:*
```python
{
    "w_back": 1, "w_ahead": 1,
    "penalty": "l2",
    "alpha": 0.01216,
    "learning_rate": "adaptive",
    "eta0": 0.00171,
}
```

=== LGBM
*Search space (per trial):*
- `learning_rate`: 0.01 to 0.2, log-scaled
- `num_leaves`: 15 to 255
- `max_depth`: -1 (unlimited) to 15
- `min_child_samples`: 10 to 100
- `subsample`: 0.5 to 1.0
- `colsample_bytree`: 0.5 to 1.0
- `reg_alpha`: $10^(-8)$ to 10.0, log-scaled
- `reg_lambda`: $10^(-8)$ to 10.0, log-scaled

*Best hyperparameters:*
```python
{
    "w_back": 3, "w_ahead": 1,
    "learning_rate": 0.08865,
    "num_leaves": 73,
    "max_depth": 13,
    "min_child_samples": 41,
    "subsample": 0.7485,
    "colsample_bytree": 0.8856,
    "reg_alpha": 0.13770,
    "reg_lambda": 0.03024,
}
```

=== MLP Main
*Search space (per trial):*
- `n_layers`: 1 to 4
- `hidden_dim`: 64, 128, 256, 512, or 1024
- `dropout`: 0.0 to 0.5
- `activation`: `ReLU`, `GELU`, or `SiLU`
- `lr`: $10^(-4)$ to $10^(-2)$, log-scaled
- `weight_decay`: $10^(-6)$ to $10^(-2)$, log-scaled
- `batch_size`: 8192, 16384, 32768, or 65536

*Best hyperparameters:*
```python
{
    "w_back": 2, "w_ahead": 1,
    "n_layers": 2,
    "hidden_dim": 1024,
    "dropout": 0.0800,
    "activation": "GELU",
    "lr": 0.001428,
    "weight_decay": 0.001833,
    "batch_size": 32768,
}
```

=== MLP DL
*Search space (per trial):*
Same as MLP Main, except `n_layers`: 1 to 5.

*Best hyperparameters:*
```python
{
    "w_back": 2, "w_ahead": 1,
    "n_layers": 3,
    "hidden_dim": 256,
    "dropout": 0.0950,
    "activation": "GELU",
    "lr": 0.002053,
    "weight_decay": 0.000344,
    "batch_size": 16384,
}
```

=== MLP Linguistic
*Search space (per trial):*
Same as MLP Main.

*Best hyperparameters:*
```python
{
    "w_back": 2, "w_ahead": 1,
    "n_layers": 2,
    "hidden_dim": 1024,
    "dropout": 0.3861,
    "activation": "SiLU",
    "lr": 0.002592,
    "weight_decay": 0.000824,
    "batch_size": 8192,
}
```

== Selected Features <appendix-selected-features>
This section details the specific features selected for each model.

=== Linear Regression
```python
LINREG_FEATURES = [
    "is_pad", "move_dist", "move_sin", "move_cos", "x", "y", "shift",
    "unigram_frequency", "bigram_frequency", "word_frequency",
    "same_hand", "same_finger", "same_finger_trigram",
    "repetition", "skipgram_repetition", "same_finger_skipgram",
    "in_roll", "out_roll", "in_triroll", "out_triroll",
    "redirects", "double_row_jump", "scissors",
    "word_index", "word_length", "word_relative_pos",
    "is_word_start", "is_word_end", "is_syllable_start", "is_syllable_end",
    "finger_0", "finger_1", "finger_3", "finger_6", "finger_8", "finger_9",
    "finger_type_0", "finger_type_1", "finger_type_2", "finger_type_3",
    "hand_0",
]
```

=== LGBM
```python
LGBM_FEATURES = [
    'move_dist', 'move_cos', 'move_sin', 'x', 'y', 'shift',
    'bigram_frequency', 'word_frequency',
    'same_hand', 'same_finger',
    'repetition', 'skipgram_repetition', 'same_finger_skipgram',
    'in_roll', 'out_roll', 'redirects', 'double_row_jump',
    'sequence_pos', 'word_index', 'word_length', 'word_relative_pos',
    'finger', 'hand'
]
```

=== MLP Main
```python
MLP_MAIN_FEATURES = [
    "move_dist", "move_cos", "move_sin", "x", "y", "shift",
    "bigram_frequency", "word_frequency",
    "repetition", "skipgram_repetition", "same_finger_skipgram",
    "same_finger", "same_hand",
    "in_roll", "out_roll", "redirects", "double_row_jump",
    "sequence_pos", "word_length", "word_relative_pos",
    "is_syllable_start", "is_syllable_end",
    "finger_0", "finger_1", "finger_2", "finger_3", "finger_4",
    "finger_5", "finger_6", "finger_7", "finger_8", "finger_9",
    "hand_0", "hand_1", "hand_2",
]
```

=== MLP Linguistic
```python
MLP_LINGUISTIC_FEATURES = [
    "bigram_frequency", "word_frequency",
    "sequence_pos", "word_length", "word_relative_pos",
    "is_syllable_start", "is_syllable_end",
]
```

=== MLP Deep Learning
```python
MLP_DL_FEATURES = [
    "x", "y", "shift", "hand",
    "bigram_frequency", "word_frequency",
    "sequence_pos", "word_length", "word_relative_pos",
    "is_syllable_start", "is_syllable_end",
]
```


