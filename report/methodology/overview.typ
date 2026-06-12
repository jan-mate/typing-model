= Methodology
The goal is to train ML models on QWERTY data that predict inter-keystroke intervals on Dvorak data.

The methodology is a pipeline from raw keystroke logs to evaluated models.
Four stages produce the modeling dataset: selection picks the typists and keystrokes, preparation cleans them, enrichment computes the features, and splitting divides the data into training and validation sets.
The model architectures are introduced next (@sec-models), as context for the modeling stages that follow.
Feature selection, hyperparameter optimization, and final training produce the models.
A last stage defines the metrics and statistical tests used to evaluate them.

All code is available at #link("https://github.com/jan-mate/typing-model")[`github.com/jan-mate/typing-model`].

