# measure ensemble-fold disagreement per layout: average per-keystroke std across
# folds on corpus sentences. real layouts should agree more than `random`
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.wrappers import LGBM as LGBMSpeedModel, LinReg as LinRegSpeedModel, MLP as MLPSpeedModel, MLPDL as MLPDLSpeedModel
from src.config import CORPUS_PATH, model_dir
from src.enrichment.engine import EnrichmentEngine
from src.utils.corpus_eval import predict_corpus, sample_corpus, corpus_stats

N_SENTENCES = 25_000
BOOTSTRAP_SAMPLES = 1000

LAYOUTS = {
    name: {
        "layout_path":     f"{PROJECT_ROOT}/data/layouts/{layout_file}",
        "layout_map_path": f"{PROJECT_ROOT}/data/layouts/layout_map.json",
        "shifts_path":     f"{PROJECT_ROOT}/data/layouts/shifts_us.json",
    }
    for name, layout_file in [
        ("qwerty",  "qwerty_us.json"),
        ("dvorak",  "dvorak.json"),
        ("colemak", "colemak.json"),
        ("random",  "random.json"),
    ]
}

MODELS = [
    ("Linear Regression", LinRegSpeedModel, model_dir("linreg")),
    ("LightGBM",          LGBMSpeedModel,   model_dir("lgbm")),
    ("MLP",               MLPSpeedModel,    model_dir("mlp_main")),
    ("MLP (DL)",          MLPDLSpeedModel,  model_dir("mlp_dl")),
]


def main():
    sentences = sample_corpus(CORPUS_PATH, n_sentences=N_SENTENCES)
    print(f"Sampled {len(sentences)} sentences from {CORPUS_PATH}")

    engine = EnrichmentEngine(
        unigrams_path=f"{PROJECT_ROOT}/data/frequencies/unigrams_zipf.json",
        bigrams_path=f"{PROJECT_ROOT}/data/frequencies/bigrams_zipf.json",
        words_path=f"{PROJECT_ROOT}/data/frequencies/words_zipf.json",
        movement_features_path=f"{PROJECT_ROOT}/data/layouts/movement_features.json",
    )

    rows = []
    for model_name, model_cls, model_path in MODELS:
        if not os.path.isdir(model_path):
            print(f"Skipping {model_name}: {model_path} not found")
            continue

        print(f"\n--- {model_name} ---")
        model = model_cls(model_path)

        for layout_name, layout_paths in LAYOUTS.items():
            means, stds, _ = predict_corpus(model, sentences, engine, layout_paths)
            
            stats = corpus_stats(stds, n_bootstrap=BOOTSTRAP_SAMPLES)
            
            rows.append({
                "Model":        model_name,
                "Layout":       layout_name.capitalize(),
                "Mean Pred (z)": round(float(np.mean(means)), 6),
                "Mean Std Dev":  round(float(stats['mean']), 6),
                "Lower CI":      round(float(stats['ci_low']), 6),
                "Upper CI":      round(float(stats['ci_high']), 6),
                "Mean Variance": round(float(np.mean(stds ** 2)), 6),
            })

    print("\n" + "=" * 120)
    print("ENSEMBLE UNCERTAINTY (WEIRDNESS PENALTY) COMPARISON")
    print("95% Bootstrap Confidence Intervals for the Mean Std Dev (disagreement)")
    print("=" * 120)
    print(pd.DataFrame(rows).to_string(index=False))
    print("=" * 120)


if __name__ == "__main__":
    main()