import json
import os
import re
import sys
import random
import pandas as pd
import argparse

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.config import STORAGE_ROOT, DATA_ROOT, model_dir, CORPUS_PATH
from src.abbr_dict.config import data_path
from src.utils.layout_utils import build_layout_and_engine
from src.models.wrappers import MLP as MLPSpeedModel, LGBM as LGBMSpeedModel, LinReg as LinRegSpeedModel, MLPDL as DLSpeedModel
from src.abbr_dict.eval_core import run_evaluation, compute_trg_bigram_freqs, display_ks
from src.abbr_dict.eval_display import print_examples, print_summary

import argparse

MLP_MODEL_DIR    = model_dir("mlp_main")
LGBM_MODEL_DIR   = model_dir("lgbm")
LINREG_MODEL_DIR = model_dir("linreg")
DL_MODEL_DIR     = model_dir("mlp_dl")

WORDS_ZIPF       = os.path.join(DATA_ROOT, "data/frequencies/words_zipf.json")

DEFAULT_DICT_PATH = os.path.join(data_path("optimization_results"), "optimized_dict_lgbm.json")
if not os.path.exists(DEFAULT_DICT_PATH):
    DEFAULT_DICT_PATH = DEFAULT_DICT_PATH.replace(STORAGE_ROOT, DATA_ROOT, 1)

DEFAULT_MODEL_TYPE = "lgbm"   # "dl" | "lgbm" | "mlp" | "linreg"
N_SENTENCES  = 25_000
MAX_SENT_LEN = 70
MIN_SENT_LEN = 2
RANDOM_SEED  = 42

RPT_KEY_VARIANT = "semicolon"   # "semicolon" | "j" (diagnostic: RPT at j position)


def _load_model(model_type):
    if model_type == "lgbm":
        return LGBMSpeedModel(LGBM_MODEL_DIR)
    if model_type == "dl":
        return DLSpeedModel(DL_MODEL_DIR)
    if model_type == "linreg":
        return LinRegSpeedModel(LINREG_MODEL_DIR)
    return MLPSpeedModel(MLP_MODEL_DIR)


def main():
    parser = argparse.ArgumentParser(description="Evaluate an abbreviation dictionary.")
    parser.add_argument("--dict", type=str, default=DEFAULT_DICT_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--sentences", type=int, default=N_SENTENCES)
    parser.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args()
    quiet = args.quiet

    def vprint(*a, **kw):
        if not quiet:
            print(*a, **kw)

    random.seed(RANDOM_SEED)

    dict_path = args.dict
    if not os.path.exists(dict_path):
        dict_path = os.path.join(STORAGE_ROOT, args.dict)
    if not os.path.exists(dict_path):
        dict_path = os.path.join(DATA_ROOT, args.dict)

    if not os.path.exists(dict_path):
        print(f"Error: Dictionary not found at {args.dict}")
        print(f"  (Checked locally and in {STORAGE_ROOT})")
        return

    vprint(f"Loading dictionary from {dict_path}...")
    with open(dict_path) as f:
        dict_data = json.load(f)

    meta    = dict_data["metadata"]
    rpt_key = meta["rpt_key"]
    n_sw    = sum(1 for e in dict_data["dictionary"] if e["type"] in ("singleword", "linked"))
    n_suf   = sum(1 for e in dict_data["dictionary"] if e["type"] == "suffix")
    vprint(f"Dict: {len(dict_data['dictionary'])} entries  rpt_key={rpt_key}  lambda={meta.get('lambda', '?')}")
    vprint(f"  Singleword: {n_sw}  Suffix: {n_suf}")

    with open(WORDS_ZIPF) as f:
        word_freq_map = json.load(f)

    trg_bigram_freqs = compute_trg_bigram_freqs(dict_data, word_freq_map)
    if not quiet:
        top_bg = sorted(trg_bigram_freqs.items(), key=lambda x: -x[1])[:10]
        print(f"TRG/RPT bigram freqs computed ({len(trg_bigram_freqs)} bigrams):")
        for bg, zipf in top_bg:
            print(f"  {display_ks(bg)!r:6s}  Zipf={zipf:.3f}")
        print()

    # always the TRG-present layout; the dict's rpt_key only drives keystream encoding
    layout_paths, engine, bigrams_file = build_layout_and_engine(
        ".", True, rpt_variant=RPT_KEY_VARIANT
    )
    with open(bigrams_file) as f:
        raw_bigram_map = json.load(f)

    model = _load_model(args.model)
    vprint(f"{args.model.upper()}: {len(model.models)} folds  w_back={model.w_back}  w_ahead={model.w_ahead}\n")

    vprint(f"Loading corpus: {CORPUS_PATH}")
    df_corpus = pd.read_parquet(CORPUS_PATH)
    sentences = []
    for text in df_corpus["body"].dropna():
        for fragment in re.split(r"[.\n!?]+", str(text)):
            fragment = fragment.strip()
            if MIN_SENT_LEN <= len(fragment) <= MAX_SENT_LEN:
                sentences.append(fragment)
    random.shuffle(sentences)
    sentences = sentences[:args.sentences]
    vprint(f"Sampled {len(sentences)} sentences  avg_len={sum(len(s) for s in sentences)/max(len(sentences),1):.1f} chars\n")

    vprint("Inferring normal costs...")
    vprint("Inferring abbr costs...")
    result, debug = run_evaluation(
        dict_data, sentences, engine, layout_paths, model, word_freq_map,
        return_debug=True, raw_bigram_map=raw_bigram_map,
    )

    if quiet:
        dict_name = os.path.basename(dict_path).replace(".json", "")
        print(f"{dict_name} [{args.model}]: {result['agg_speedup_pct']:.2f}%")
    else:
        print_examples(debug)
        print_summary(result, model_name=args.model)

    eval_path = dict_path.replace(".json", f"_eval_{args.model}.json")
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(result, f, indent=2)
    vprint(f"\nSaved eval results to {eval_path}")


if __name__ == "__main__":
    main()
