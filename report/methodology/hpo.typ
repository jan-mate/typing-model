#import "/styles.typ": *

== Hyperparameter optimization <sec-hpo>
All models use Optuna for hyperparameter optimization.
Optuna uses a Tree-structured Parzen Estimator (TPE) sampler, a form of Bayesian optimization: it fits a probabilistic model over past trial results and uses it to propose the next hyperparameters most likely to improve the objective, rather than sampling randomly or searching a grid.
The four non-linear models (LGBM, MLP Main, MLP DL, MLP Linguistic) follow a two-phase strategy: phase 1 runs a small trial budget for each candidate $(W_"back", W_"ahead")$ window in a grid; phase 2 continues the winning window's study to refine the same hyperparameters.
Both phases share a search space and a `MedianPruner`, which stops a trial whose intermediate score falls below the running median, freeing compute for promising configurations.

Window size is the dominant structural decision, since it sets the input dimensionality.
Fixing the best window first lets phase 2 explore one consistent search space instead of spreading the budget across inferior windows.

LinReg uses a single phase: its search space has only five mostly categorical hyperparameters, training is fast, and there are no architectural decisions, so a flat grid over windows with 20 trials per config characterizes it.
The full search spaces and final hyperparameter configurations for all models are listed in @appendix-hpo.

Each window-analysis table reports the first $N$ trials per window in chronological order ($N$ = phase-1 budget), excluding the phase-2 trials appended to the winning window.
`num_completed` counts finished trials; `num_pruned` counts trials the `MedianPruner` stopped early (LinReg has no pruner).
Where the best-MAE window was not chosen, the reasoning follows the table.

=== Linear Regression
The linear baseline uses `SGDRegressor` with `epsilon_insensitive` loss ($epsilon = 0$, equivalent to minimising MAE), `StandardScaler` normalization, and one-hot encoded categorical inputs (`finger`, `hand`, `finger_type`).
The feature set is `LINREG_FEATURES` from the L1-selection step (@appendix-selected-features).
Training uses `max_iter=1000` and `tol=1e-3` with no early stopping.

#figure(
  table(
    columns: 5,
    align: (center, center, right, right, right),
    inset: 7pt,
    stroke: none,
    fill: zebra,
    table.header[*$W_"back"$*][*$W_"ahead"$*][*best MAE*][*avg MAE*][*completed*],
    [3], [1], [0.5924], [0.6122], [20],
    [2], [1], [0.5930], [0.6267], [20],
    [1], [1], [0.5934], [0.6042], [20],
    [3], [0], [0.5940], [0.6282], [20],
    [2], [0], [0.5946], [0.6121], [20],
    [1], [0], [0.5948], [0.6300], [20],
  ),
  caption: [Phase 1 window analysis for Linear Regression (20 trials per window).],
  kind: table,
) <table-hpo-linreg>

@table-hpo-linreg shows that 3/1 had the best MAE, but 1/1 was chosen: its best MAE is within 0.001 of 3/1 (below typical run-to-run noise), it has the lowest avg MAE of any config (0.604, a more robust optimum), and it is the smallest input.

=== LGBM
Gradient-boosted trees on `LGBM_FEATURES`, with `finger` and `hand` passed as categorical features.

*Fixed training settings:* `objective='regression_l1'`, `metric='l1'`, up to `n_estimators=1000` with early stopping at `patience=50` rounds on the validation set.

#figure(
  table(
    columns: 6,
    align: (center, center, right, right, right, right),
    inset: 7pt,
    stroke: none,
    fill: zebra,
    table.header[*$W_"back"$*][*$W_"ahead"$*][*best MAE*][*avg MAE*][*completed*][*pruned*],
    [3], [1], [0.5624], [0.5636], [6], [5],
    [3], [2], [0.5629], [0.5727], [9], [3],
    [2], [2], [0.5633], [0.5647], [5], [7],
    [2], [1], [0.5637], [0.5641], [5], [7],
    [3], [0], [0.5659], [0.5676], [5], [7],
    [1], [1], [0.5661], [0.5732], [5], [6],
    [1], [2], [0.5661], [0.5700], [8], [4],
    [2], [0], [0.5680], [0.5743], [6], [6],
    [1], [0], [0.5685], [0.5727], [8], [4],
  ),
  caption: [Phase 1 window analysis for LGBM (12 trials per window).],
  kind: table,
) <table-hpo-lgbm>

As shown in @table-hpo-lgbm, 3/1 had the lowest best MAE and was selected for phase 2.
The next contenders (3/2 and 2/2) were within 0.001 but added input dimensionality without clear benefit.

=== MLP Main
The main neural model uses a multilayer perceptron with variable depth and width. Training uses the AdamW optimizer, MAE loss, and a learning rate that halves when validation loss plateaus (patience of 1 epoch). Inputs are standard-normalized to aid training stability, and categorical inputs are one-hot encoded.

*Fixed training settings:* up to 30 epochs, early stopping patience 3 epochs.

The window grid was narrowed to ${2, 3} times {1, 2}$ based on LGBM phase 1 and exploratory analysis, which ruled out $W_"back"=1$ and $W_"ahead"=0$ as inferior across all configs, saving five window configs of trials.

#figure(
  table(
    columns: 6,
    align: (center, center, right, right, right, right),
    inset: 7pt,
    stroke: none,
    fill: zebra,
    table.header[*$W_"back"$*][*$W_"ahead"$*][*best MAE*][*avg MAE*][*completed*][*pruned*],
    [2], [1], [0.5595], [0.5637], [8], [4],
    [3], [1], [0.5599], [0.5634], [6], [6],
    [2], [2], [0.5615], [0.5661], [9], [3],
    [3], [2], [0.5629], [0.5660], [7], [5],
  ),
  caption: [Phase 1 window analysis for MLP Main (12 trials per window).],
  kind: table,
) <table-hpo-mlp-main>

@table-hpo-mlp-main indicates that 2/1 had the best MAE and is the smallest input, making the choice straightforward.

*Resulting architecture:* `Linear(140, 1024) -> GELU -> Dropout(0.08) -> Linear(1024, 1024) -> GELU -> Dropout(0.08) -> Linear(1024, 1)`.
Input dimension is $35 "features" times 4 "timesteps" = 140$. Total trainable parameters: ≈1.20M.

=== MLP DL
MLP DL uses the same architecture and pipeline as MLP Main, but on the smaller `MLP_DL_FEATURES` set.
The depth range extends to 1–5 layers so the network has more capacity to learn motor patterns from raw coordinates.
The same narrowed window grid applies.

*Fixed training settings:* same as MLP Main.

#figure(
  table(
    columns: 6,
    align: (center, center, right, right, right, right),
    inset: 7pt,
    stroke: none,
    fill: zebra,
    table.header[*$W_"back"$*][*$W_"ahead"$*][*best MAE*][*avg MAE*][*completed*][*pruned*],
    [2], [2], [0.5630], [0.5727], [9], [3],
    [2], [1], [0.5633], [0.5752], [9], [3],
    [3], [1], [0.5634], [0.5694], [7], [5],
    [3], [2], [0.5641], [0.5780], [8], [3],
  ),
  caption: [Phase 1 window analysis for MLP DL (12 trials per window).],
  kind: table,
) <table-hpo-mlp-dl>

@table-hpo-mlp-dl shows that 2/2 had the best phase-1 MAE, but 2/1 was chosen: the difference (0.0003) is small, and 2/1 is significantly simpler.

*Resulting architecture:* `Linear(44, 256) -> GELU -> Dropout(0.10) -> Linear(256, 256) -> GELU -> Dropout(0.10) -> Linear(256, 256) -> GELU -> Dropout(0.10) -> Linear(256, 1)`.
Input dimension is $11 "features" times 4 "timesteps" = 44$. Total trainable parameters: ≈143k.

=== MLP Linguistic
MLP Linguistic uses the same architecture and pipeline as MLP Main, but on `MLP_LINGUISTIC_FEATURES` only.
Because the feature set is much smaller, the per-window trial budget drops to 8.
The same narrowed window grid applies.

*Fixed training settings:* same as MLP Main.

#figure(
  table(
    columns: 6,
    align: (center, center, right, right, right, right),
    inset: 7pt,
    stroke: none,
    fill: zebra,
    table.header[*$W_"back"$*][*$W_"ahead"$*][*best MAE*][*avg MAE*][*completed*][*pruned*],
    [2], [1], [0.6323], [0.6390], [6], [2],
    [3], [2], [0.6333], [0.6375], [8], [0],
    [2], [2], [0.6341], [0.6428], [6], [2],
    [3], [1], [0.6359], [0.6385], [5], [3],
  ),
  caption: [Phase 1 window analysis for MLP Linguistic (8 trials per window).],
  kind: table,
) <table-hpo-mlp-ling>

@table-hpo-mlp-ling reports that 2/1 had the best MAE and is the smallest input.

*Resulting architecture:* `Linear(28, 1024) -> SiLU -> Dropout(0.39) -> Linear(1024, 1024) -> SiLU -> Dropout(0.39) -> Linear(1024, 1)`.
Input dimension is $7 "features" times 4 "timesteps" = 28$. Total trainable parameters: ≈1.08M.
