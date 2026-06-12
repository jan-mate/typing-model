import os
import sys
import time
import warnings

import pandas as pd
import torch

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.wrappers import LGBM as LGBMSpeedModel, LinReg as LinRegSpeedModel, MLP as MLPSpeedModel, MLPDL as MLPDLSpeedModel
from src.config import CORPUS_PATH, model_dir
from src.enrichment.engine import EnrichmentEngine
from src.utils.corpus_eval import predict_corpus, sample_corpus

N_SENTENCES = 25_000

LAYOUT_PATHS = {
    "layout_path":     f"{PROJECT_ROOT}/data/layouts/qwerty_us.json",
    "layout_map_path": f"{PROJECT_ROOT}/data/layouts/layout_map.json",
    "shifts_path":     f"{PROJECT_ROOT}/data/layouts/shifts_us.json",
}

MODELS = [
    ("Linear Regression", LinRegSpeedModel, model_dir("linreg")),
    ("LightGBM",          LGBMSpeedModel,   model_dir("lgbm")),
    ("MLP",               MLPSpeedModel,    model_dir("mlp_main")),
    ("MLP (DL)",          MLPDLSpeedModel,  model_dir("mlp_dl")),
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sentences = sample_corpus(CORPUS_PATH, n_sentences=N_SENTENCES)
    total_chars = sum(len(s) for s in sentences)
    print(f"Sampled {len(sentences)} sentences ({total_chars:,} characters)")

    engine = EnrichmentEngine(
        unigrams_path=f"{PROJECT_ROOT}/data/frequencies/unigrams_zipf.json",
        bigrams_path=f"{PROJECT_ROOT}/data/frequencies/bigrams_zipf.json",
        words_path=f"{PROJECT_ROOT}/data/frequencies/words_zipf.json",
        movement_features_path=f"{PROJECT_ROOT}/data/layouts/movement_features.json",
    )

    from src.utils.inference import build_linguistic_frame, extract_features, compute_costs

    print(f"\n--- Precomputing Linguistic Features ---")
    t0 = time.perf_counter()
    base_dict, offsets = build_linguistic_frame(sentences, engine, w_back=3, w_ahead=1)
    linguistic_time = time.perf_counter() - t0
    print(f"Linguistic setup completed in {linguistic_time:.3f}s")

    results = []
    for model_name, model_cls, model_path in MODELS:
        if not os.path.isdir(model_path):
            print(f"Skipping {model_name}: {model_path} not found")
            continue

        print(f"\n--- Benchmarking {model_name} ---")
        model = model_cls(model_path)

        t1 = time.perf_counter()
        enriched, _ = engine.enrich_layout(base_dict, **LAYOUT_PATHS)
        df_feats = extract_features(enriched, list(model.features))
        layout_time = time.perf_counter() - t1

        t2 = time.perf_counter()
        _ = compute_costs(df_feats, offsets, model)
        predict_time = time.perf_counter() - t2

        total_model_time = layout_time + predict_time

        results.append({
            "Model": model_name,
            "Sentences": len(sentences),
            "Keystrokes": total_chars,
            "Layout (s)": round(layout_time, 3),
            "Predict (s)": round(predict_time, 3),
            "Total (s)": round(total_model_time, 3),
            "Sec / 1M": round(total_model_time * 1_000_000 / max(total_chars, 1), 3),
        })

    print("\n" + "=" * 90)
    print(f"INFERENCE BENCHMARK ({total_chars:,} keystrokes)")
    print(f"Shared Linguistic Time: {linguistic_time:.3f}s (calculated once)")
    print("=" * 90)
    if results:
        print(pd.DataFrame(results).to_string(index=False))
    else:
        print("No models found.")
    print("=" * 90)


if __name__ == "__main__":
    main()
