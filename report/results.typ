#import "/styles.typ": *

= Results
== Model performance
@table-performance reports QWERTY validation MAE and two zero-shot Dvorak MAEs for each model: the single-model average (with 95% CI) and the deployed ensemble (@sec-evaluation).

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 6pt,
    align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
    stroke: none,
    fill: zebra,
    table.header(
      [*Model*], [*Val MAE*], [*Val $R^2$*], [*Dvorak MAE* \ #text(size: 0.78em, weight: "regular")[(single, 95% CI)]], [*Dvorak MAE* \ #text(size: 0.78em, weight: "regular")[(ensemble, 95% CI)]], [*Dvorak $R^2$*],
    ),
    [Baseline],           [0.7374], [0.0000], [0.7411], [0.7411], [0.0000],
    [Linear Regression],  [0.6014], [0.2329], mae([0.6125], [(0.5972, 0.6269)]), mae([0.6106], [(0.5961, 0.6252)]), [0.2401],
    [LightGBM],           [0.5678], [0.3041], mae([*0.5922*], [(0.5780, 0.6053)]), mae([*0.5884*], [(0.5752, 0.6027)]), [*0.2938*],
    [MLP (Main)],         [*0.5657*], [*0.3051*], mae([0.5998], [(0.5860, 0.6138)]), mae([0.5958], [(0.5819, 0.6098)]), [0.2828],
    [MLP (DL)],           [0.5718], [0.2920], mae([0.6038], [(0.5897, 0.6184)]), mae([0.5987], [(0.5848, 0.6139)]), [0.2711],
    [MLP (Ling)],         [0.6324], [0.1881], mae([0.6231], [(0.6075, 0.6380)]), mae([0.6153], [(0.6007, 0.6304)]), [0.2371],
  ),
  caption: [Model performance on QWERTY validation and zero-shot Dvorak transfer. The single Dvorak MAE is the mean of the 10 submodels' individual MAEs; the ensemble Dvorak MAE scores the averaged prediction (@sec-evaluation). MAE is normalized to participant-wise IKI standard deviation.],
) <table-performance>

All models achieve lower MAE than the baseline. 
MLP Main achieves the best QWERTY validation MAE, with LightGBM marginally behind. 
LightGBM has the smallest gap between its validation and single-model Dvorak MAE (0.024 vs. 0.034 for MLP Main).
The ensemble lowers Dvorak MAE by about 0.004 over the single-model average, a small gain because the submodels nearly agree (fold std ≈0.002).
MLP DL achieves similar MAE to the main models despite fewer engineered features, suggesting the raw coordinate representation captures meaningful motor signal.

== Bias analysis <sec-bias>
@table-bias reports the bias and bias shift for each model.
Bias is the model's mean signed error, the average of predicted minus measured IKI. 
A negative bias means the model predicts faster typing than measured. 
The bias shift is the Dvorak bias minus the validation bias: it measures how much this systematic error changes between layouts, after removing any misprediction common to both. 
A bias shift of zero means the error is identical on both layouts.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 6pt,
    align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
    stroke: none,
    fill: zebra,
    table.header(
      [*Model*], [*Val bias*], [*Dvorak bias*], [*Bias shift* \ #text(size: 0.78em, weight: "regular")[(95% CI)]],
    ),
    [Linear Regression], [-0.1234], [-0.1044], mae([+0.0190], [(+0.0136, +0.0248)]),
    [LightGBM],          [-0.0769], [-0.0754], mae([*+0.0015*], [(-0.0039, +0.0068)]),
    [MLP (Main)],        [*-0.0631*], [*-0.0505*], mae([+0.0126], [(+0.0071, +0.0179)]),
    [MLP (DL)],          [-0.0902], [-0.0823], mae([+0.0079], [(+0.0025, +0.0135)]),
  ),
  caption: [Mean signed error (bias) and its shift across layouts. Units are normalized Z-scores.],
) <table-bias>

All models have negative biases on both QWERTY and Dvorak, meaning they systematically predict lower IKI than measured (faster typing than reality) for both populations. 
MAE optimization on a right-skewed IKI distribution causes this; it is unrelated to layout.

The bias shift captures the layout-specific effect after removing this baseline tendency. 
All shifts are positive, meaning the under-prediction is smaller on Dvorak than on QWERTY: the models predict Dvorak as slightly slower than ground truth warrants. 
The models are biased against Dvorak in layout comparisons.

LightGBM is the only model whose 95% CI includes zero, so its bias shift is not statistically significant.
Its shift is also the smallest, less than 0.08 ms at the dataset's IKI standard deviation of 50.4 ms.

== Ensemble uncertainty <sec-uncertainty>
The 10-fold ensemble provides a natural measure of epistemic uncertainty: the standard deviation of the 10 submodel predictions per keystroke. 
A well-calibrated ensemble disagrees more on unfamiliar inputs and less on familiar ones. 
Values are computed over 25 000 sentences (1 009 622 keystrokes), sampled from a combined Wikipedia/Reddit corpus.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 6pt,
    align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
    stroke: none,
    fill: flat,
    table.header(
      [*Model*], [*QWERTY std*], [*Dvorak std*], [*Colemak std*], [*Random std*],
    ),
    [Linear Regression], [0.4352], [0.4444], [0.4468], [0.4304],
    [LightGBM],          [0.0941], [0.0896], [0.0892], [0.1162],
    [MLP (Main)],        [0.1210], [0.1245], [0.1227], [0.1287],
    [MLP (DL)],          [0.1078], [0.1048], [0.1046], [0.1270],
  ),
  caption: [Mean ensemble standard deviation (epistemic uncertainty) across layouts.],
) <table-uncertainty>

@table-uncertainty reports this statistic across layouts.
The diagnostic question is whether each model's disagreement scales with input familiarity, increasing on a random layout and staying low on structured layouts. 
Comparing Random std to QWERTY std:

- *LightGBM* (0.1162 vs. 0.0941, +23%) and *MLP DL* (0.1270 vs. 0.1078, +18%) show separation.
- *MLP Main* (0.1287 vs. 0.1210, +6%) shows less separation.
- *Linear Regression* (0.4304 vs. 0.4352, -1%) is uninformative as an uncertainty estimator.

== Inference speed
@table-speed reports timing on 1 009 622 keystrokes (25 000 sentences).
Linguistic feature precomputation (16.7 s) is a shared one-time cost not included in the per-model totals.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 6pt,
    align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
    stroke: none,
    fill: flat,
    table.header(
      [*Model*], [*Layout enrich (s)*], [*Predict (s)*], [*Total (s)*],
    ),
    [Linear Regression], [0.777], [9.032],   [9.809],
    [LightGBM],          [0.637], [105.682], [106.319],
    [MLP (Main)],        [0.677], [15.330],  [16.007],
    [MLP (DL)],          [*0.543*], [*4.710*], [*5.253*],
  ),
  caption: [Inference time for 1 million keystrokes on Google Colab (T4 GPU for MLPs, default CPU for Linear Regression and LightGBM).],
) <table-speed>

MLP DL requires 5.3 s, benefiting from a small feature set (≈143k parameters vs. ≈1.2M for MLP Main) and efficient GPU batching. 
Linear Regression requires 9.8 s despite running on CPU, since the model itself is cheap. 
MLP Main is slower due to a larger feature matrix and wider network. 
LightGBM requires 106.3 s: tree traversal does not parallelize efficiently on GPU, and the large ensemble (382 812 leaves) drives most of this cost.
