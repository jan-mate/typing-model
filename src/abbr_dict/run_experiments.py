# batch optimize + evaluate for the report's experiment tables
import os
import random
import re
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import CORPUS_PATH, DATA_ROOT, model_dir
from src.abbr_dict.config import RPT_KEY, data_path
from src.abbr_dict.optimize import load_and_preprocess, run_optimize, DATA_PATH
from src.utils.layout_utils import build_layout_and_engine
from src.models.wrappers import LGBM as LGBMSpeedModel
from src.abbr_dict.eval_core import run_evaluation

LAMBDA       = 0.1
MODEL        = "dl"          # optimize with DL, judge with LGBM
N_SENTENCES  = 25_000
MIN_SENT_LEN = 2
MAX_SENT_LEN = 70
SEED         = 42
SIGMA        = 1.0
OVERLAP      = 1.0
MIN_INTUIT   = 0.30
WORDS_ZIPF   = os.path.join(DATA_ROOT, "data/frequencies/words_zipf.json")


def _experiment_list():
    # tuples of (name, top_k, include_words, include_suffixes, max_size, doubletap, trg_only)
    if RPT_KEY:
        cfgs  = [(f"K{k}", k, True, True, 160, False, False) for k in (3, 5, 8, 10, 12, 15)]
        cfgs += [(f"N{n}", 12, True, True, n, False, False)
                 for n in (10, 20, 40, 80, 160, 320, 640, 1280)]
        cfgs += [
            ("words_only",    12, True,  False, 160, False, False),
            ("suffixes_only", 12, False, True,  160, False, False),
            ("trg_only",      12, True,  True,  160, False, True),
            ("doubletap",     12, True,  True,  160, True,  False),
        ]
        return cfgs
    return [
        ("full",       12, True, True,  160, False, False),
        ("words_only", 12, True, False, 160, False, False),
    ]


def optimize_all():
    outdir = data_path("experiments")
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for name, top_k, inc_w, inc_s, max_size, dbl, trg in _experiment_list():
        items, overlaps = load_and_preprocess(
            DATA_PATH, include_suffixes=inc_s, include_words=inc_w,
            model_name=MODEL, top_k_per_word=top_k, sigma_penalty=SIGMA, lambda_=LAMBDA,
        )
        res = run_optimize(
            items, overlaps, lambda_=LAMBDA, overlap_penalty=OVERLAP,
            min_intuitiveness=MIN_INTUIT, max_dict_size=max_size, rpt_key=RPT_KEY,
            model_name=MODEL, sigma_penalty=SIGMA, doubletap_only=dbl, trg_only=trg,
            verbose=False,
        )
        if res is None:
            print(f"  [optimize] {name}: NO SOLUTION", flush=True)
            continue
        m = res["metadata"]
        path = os.path.join(outdir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(res, f)
        print(f"  [optimize] {name:14s} size={m['dict_size']:4d} "
              f"{m['solver_status']} gap={m['solver_gap_pct']:.1f}% t={m['solver_wall_time_s']:.0f}s",
              flush=True)
        paths.append((name, path))
    return paths


def _load_sentences():
    df = pd.read_parquet(CORPUS_PATH)
    sents = []
    for text in df["body"].dropna():
        for frag in re.split(r"[.\n!?]+", str(text)):
            frag = frag.strip()
            if MIN_SENT_LEN <= len(frag) <= MAX_SENT_LEN:
                sents.append(frag)
    del df
    random.seed(SEED)
    random.shuffle(sents)
    return sents[:N_SENTENCES]


def evaluate_all(paths):
    layout_paths, engine, bigrams_file = build_layout_and_engine(".", True)
    lgbm = LGBMSpeedModel(model_dir("lgbm"))
    with open(bigrams_file) as f:
        raw_bigram_map = json.load(f)
    with open(WORDS_ZIPF) as f:
        word_freq_map = json.load(f)
    sentences = _load_sentences()
    print(f"  [eval] engine+LGBM ready, {len(sentences)} sentences\n", flush=True)

    results = []
    for name, path in paths:
        with open(path) as f:
            d = json.load(f)
        m = run_evaluation(d, sentences, engine, layout_paths, lgbm, word_freq_map,
                           raw_bigram_map=raw_bigram_map)
        row = {
            "name": name,
            "dict_size": m["dict_size"],
            "speedup_pct": m["agg_speedup_pct"],
            "coverage_pct": m["coverage_pct"],
            "intuit": m["dict_mean_intuitiveness"],
            "singleword_ms_per_match": m["singleword_ms_per_match"],
            "suffix_ms_per_match": m["suffix_ms_per_match"],
        }
        results.append(row)
        print(f"  [eval] {name:14s} size={row['dict_size']:4d} "
              f"speedup={row['speedup_pct']:6.2f}%  cov={row['coverage_pct']:5.1f}%  "
              f"intuit={row['intuit']}", flush=True)
    return results


def main():
    regime = "rpt-on" if RPT_KEY else "rpt-off"
    print(f"=== Experiments [{regime}] : optimizing ===", flush=True)
    paths = optimize_all()
    print(f"\n=== Experiments [{regime}] : evaluating ===", flush=True)
    results = evaluate_all(paths)
    out = data_path("experiments_summary.json")
    with open(out, "w") as f:
        json.dump({"regime": regime, "lambda": LAMBDA, "n_sentences": N_SENTENCES,
                   "results": results}, f, indent=2)
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
