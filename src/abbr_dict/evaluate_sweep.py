# evaluate the λ-sweep dictionaries and plot results. dicts are optimized with DL but
# judged with LGBM, to avoid optimizing and evaluating with the same model
import glob
import json
import os
import random
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOCAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, LOCAL_ROOT)

from src.config import model_dir, CORPUS_PATH
from src.abbr_dict.config import RPT_KEY, data_path
from src.enrichment.engine import EnrichmentEngine
from src.models.wrappers import LGBM as LGBMSpeedModel
from src.utils.inference import enrich_texts, compute_costs
from src.abbr_dict.speed import TRG_CHAR, RPT_CHAR
from src.abbr_dict.eval_core import (
    run_evaluation,
    build_sentence_pair,
    IKI_MEAN_MS, IKI_STD_MS,
)

DRIVE_BASE     = "/content/drive/MyDrive/typing-model"
STORAGE_ROOT   = DRIVE_BASE if os.path.exists(DRIVE_BASE) else LOCAL_ROOT

SWEEP_DIR      = data_path("sweep_results")
LGBM_MODEL_DIR = model_dir("lgbm")   # judge model (dicts were optimized with DL)
WORDS_ZIPF     = os.path.join(LOCAL_ROOT,   "data/frequencies/words_zipf.json")

N_SENTENCES  = 25_000
MAX_SENT_LEN = 70
MIN_SENT_LEN = 2
BATCH_SIZE   = 50_000
RANDOM_SEED  = 42
# RPT_KEY (from config) must match the sweep dictionaries' regime


def main():
    random.seed(RANDOM_SEED)

    if RPT_KEY:
        layout_paths = {
            "layout_path":     f"{LOCAL_ROOT}/data/layouts/qwerty_us_rpt_trg.json",
            "layout_map_path": f"{LOCAL_ROOT}/data/layouts/layout_map_rpt_trg.json",
            "shifts_path":     f"{LOCAL_ROOT}/data/layouts/shifts_us.json",
        }
        engine_config = {
            "unigrams_path":          f"{LOCAL_ROOT}/data/frequencies/unigrams_zipf_rpt_trg.json",
            "bigrams_path":           f"{LOCAL_ROOT}/data/frequencies/bigrams_zipf_rpt_trg.json",
            "words_path":             f"{LOCAL_ROOT}/data/frequencies/words_zipf.json",
            "movement_features_path": f"{LOCAL_ROOT}/data/layouts/movement_features_rpt_trg.json",
        }
    else:
        layout_paths = {
            "layout_path":     f"{LOCAL_ROOT}/data/layouts/qwerty_us.json",
            "layout_map_path": f"{LOCAL_ROOT}/data/layouts/layout_map.json",
            "shifts_path":     f"{LOCAL_ROOT}/data/layouts/shifts_us.json",
        }
        engine_config = {
            "unigrams_path":          f"{LOCAL_ROOT}/data/frequencies/unigrams_zipf.json",
            "bigrams_path":           f"{LOCAL_ROOT}/data/frequencies/bigrams_zipf.json",
            "words_path":             f"{LOCAL_ROOT}/data/frequencies/words_zipf.json",
            "movement_features_path": f"{LOCAL_ROOT}/data/layouts/movement_features.json",
        }

    print("Loading engine and LGBM model...")
    engine = EnrichmentEngine(**engine_config)
    lgbm   = LGBMSpeedModel(LGBM_MODEL_DIR)
    print(f"LGBM: {len(lgbm.models)} folds  w_back={lgbm.w_back}  w_ahead={lgbm.w_ahead}")

    # real bigram freqs to patch RPT-encoded double letters
    with open(engine_config["bigrams_path"]) as f:
        raw_bigram_map = json.load(f)

    with open(WORDS_ZIPF) as f:
        word_freq_map = json.load(f)
    print(f"\nLoading corpus: {CORPUS_PATH}")
    df_corpus = pd.read_parquet(CORPUS_PATH)
    sentences = []
    for text in df_corpus["body"].dropna():
        for fragment in re.split(r"[.\n!?]+", str(text)):
            fragment = fragment.strip()
            if MIN_SENT_LEN <= len(fragment) <= MAX_SENT_LEN:
                sentences.append(fragment)
    del df_corpus

    random.shuffle(sentences)
    sentences = sentences[:N_SENTENCES]
    print(f"Sampled {len(sentences)} sentences\n")

    # match only bare sweep dicts, not eval outputs like dict_lambda_0.20_eval_dl.json
    pattern = os.path.join(SWEEP_DIR, "dict_lambda_*.json")
    lam_re = re.compile(r"dict_lambda_([0-9.]+)\.json$")
    dict_files = sorted(
        (p for p in glob.glob(pattern) if lam_re.search(p)),
        key=lambda p: float(lam_re.search(p).group(1)),
    )

    if not dict_files:
        print(f"No dict_lambda_*.json files found in {SWEEP_DIR}")
        return

    print(f"Found {len(dict_files)} dictionaries to evaluate\n")

    results = []
    for path in dict_files:
        with open(path) as f:
            dict_data = json.load(f)

        meta   = dict_data["metadata"]
        lam    = meta["lambda"]
        n_dict = meta["dict_size"]
        intuit = meta["avg_intuitiveness"]

        metrics = run_evaluation(dict_data, sentences, engine, layout_paths, lgbm, word_freq_map,
                                 raw_bigram_map=raw_bigram_map)

        print(
            f"λ={lam:.2f}  speedup={metrics['agg_speedup_pct']:+.2f}%"
            f"  coverage={metrics['coverage_pct']:.1f}%"
            f"  intuit={intuit:.3f}  n={n_dict}"
        )

        results.append({
            "lambda":                   lam,
            "dict_size":                n_dict,
            "avg_intuitiveness":        intuit,
            **metrics,
        })

    summary = {"n_sentences": N_SENTENCES, "results": results}
    out_path = os.path.join(SWEEP_DIR, "eval_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")

    _plot(results, SWEEP_DIR)


def _plot(results, output_dir):
    lambdas = [r["lambda"]            for r in results]
    x_vals  = [r["avg_intuitiveness"] for r in results]
    y_vals  = [r["agg_speedup_pct"]   for r in results]
    sizes   = [r["dict_size"]         for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_vals, y_vals, c=lambdas, cmap="viridis", s=80, vmin=0, vmax=1, zorder=3)

    for x, y, lam, sz in zip(x_vals, y_vals, lambdas, sizes):
        ax.annotate(f"λ={lam:.1f}\n(n={sz})", (x, y),
                    textcoords="offset points", xytext=(6, 4), fontsize=7.5)

    ax.set_xlabel("Mean intuitiveness of selected entries")
    ax.set_ylabel("Aggregate speedup (%)")
    ax.set_title("Pareto frontier: LGBM speedup vs intuitiveness (dicts optimized with DL)")
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, "eval_pareto.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()