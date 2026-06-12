== Final training
Every model trains on the full dataset with 10-fold cross-validation, using the feature set and hyperparameters from the preceding sections.
This produces an ensemble of 10 submodels per architecture, whose predictions average at inference time.

The spread across folds supplies an estimate of epistemic uncertainty.
The cost is proportionally higher inference time and storage.

For the linear model and the MLP variants, a separate standardisation (z-score normalization) is fitted on each fold's training data and stored with its submodel.
This prevents data leakage: the validation fold's statistics never inform the scaler.
LGBM needs no normalization, so it stores no scaler.

