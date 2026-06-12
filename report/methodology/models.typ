== Models <sec-models>
Five models are trained on the same data, differing in architecture and feature set.

*Linear Regression* is an interpretable baseline.
It sets a reference point: if the other models cannot beat it, their added cost is not justified.
The Typability Index @williams2026 uses the same linear approach, so it also anchors this work against prior art.

*LightGBM* is a feature-based model. A gradient-boosted tree ensemble over the engineered features.

*MLP Main* is the main neural network model, trained on engineered features.

*MLP Linguistic* trains on the linguistic features only.
It aims to estimate how much of the accuracy comes from linguistic structure rather than learned motor patterns.

*MLP DL* takes the deep-learning approach: it strips away most feature engineering and learns from near-raw key positions, so the model must find motor patterns itself rather than being handed them.

All neural models use regularization, specifically dropout and weight decay.
Regularization prevents the networks from simply memorizing the QWERTY training data, forcing them to distinguish generalizable motor patterns from linguistic patterns.

=== Loss function
The models minimise mean absolute error (MAE).
MAE suits a keyboard-optimization goal because it weights every millisecond equally, unlike MSE, which over-penalises large errors and would let a few slow outliers dominate the objective.

=== Temporal window
To capture context, models do not just look at the current keystroke.
They use a temporal window spanning $W_"back"$ preceding keystrokes and $W_"ahead"$ future keystrokes.
If $W_"back"=2$ and $W_"ahead"=1$, the model predicts the time to type the current key at $t$ using the features of keys at $t-2$, $t-1$, $t$, and $t+1$.
This sliding window allows the models to account for the momentum of previous movements and the anticipatory cost of upcoming characters.
The optimal window size is treated as a hyperparameter (@sec-hpo).

=== Exploratory analysis
An initial evaluation tests multiple architectures on 10% of the QWERTY data.
The evaluated models include 1D CNN, LSTM, Transformer, Random Forest, XGBoost, Linear Regression, LightGBM, and MLP.
LightGBM and the MLP architectures achieve the lowest error rates and train faster.
Linear Regression performs well given its simplicity.
It remains as a baseline.
The remaining architectures are discarded.
This exploratory phase also establishes the starting hyperparameters described below.

