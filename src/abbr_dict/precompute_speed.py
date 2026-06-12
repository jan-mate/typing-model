# scores each candidate inside real corpus sentences (not isolated keystreams) with the
# DL and LGBM ensembles
import json
import os
import re
import sys
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

LOCAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(LOCAL_ROOT)

from src.config import STORAGE_ROOT, DATA_ROOT, CORPUS_PATH, model_dir
from src.abbr_dict.config import RPT_KEY, data_path
from src.utils.layout_utils import build_layout_and_engine
from src.models.wrappers import (
    MLPDL as DLSpeedModel, LGBM as LGBMSpeedModel, INFERENCE_FEATURES,
)
from src.utils.inference import enrich_texts
from src.abbr_dict.speed import override_word_frequency
from src.abbr_dict.context_strips import (
    build_word_index, build_suffix_index,
    find_occurrences, build_context_strip,
    weighted_mean_savings, compute_costs_sliced,
)

CANDIDATES_PATH = data_path("candidates_scored.json")
if not os.path.exists(CANDIDATES_PATH):
    CANDIDATES_PATH = CANDIDATES_PATH.replace(STORAGE_ROOT, DATA_ROOT, 1)

WORDS_ZIPF_PATH = os.path.join(DATA_ROOT, "data/frequencies/words_zipf.json")
OUTPUT_PATH     = data_path("speed_savings.json")

DL_MODEL_DIR   = model_dir("mlp_dl")
LGBM_MODEL_DIR = model_dir("lgbm")

# real-context instances sampled per candidate (use all if fewer are found)
N_CONTEXT = 6

MIN_SENT_LEN = 10
MAX_SENT_LEN = 100

SEQ_BATCH_SIZE = 100_000

TEST_MODE   = False
TEST_N_ITEMS = 2000


def _load_corpus_sentences() -> list[str]:
    print(f"Loading corpus: {CORPUS_PATH}")
    df = pd.read_parquet(CORPUS_PATH)
    sents = []
    for text in df["body"].dropna():
        for frag in re.split(r"[.\n!?]+", str(text)):
            frag = frag.strip()
            if MIN_SENT_LEN <= len(frag) <= MAX_SENT_LEN:
                sents.append(frag)
    random.shuffle(sents)
    print(f"  {len(sents):,} sentence fragments")
    return sents


def _collect_strips(items, word_to_sents, suffix_index, word_freq_map, left_len, right_len):
    all_strips = []
    cand_meta   = []   # (item_idx, cand_idx, ks_save)
    cand_ranges = []

    for item_idx, item in enumerate(tqdm(items, desc="Collecting context strips")):
        word  = item["text"]
        wl    = word.lower()
        ctype = item["type"]

        for cand_idx, cand in enumerate(item.get("candidates", [])):
            abbr = cand["abbr"]
            tf   = cand["trigger_form"]

            occs = find_occurrences(wl, ctype, word_to_sents, suffix_index, N_CONTEXT)

            strips = [
                build_context_strip(occ, wl, ctype, abbr, tf,
                                    left_len, right_len, RPT_KEY, word_freq_map)
                for occ in occs
            ]

            # keystrokes saved per match; +1 because the trigger doubles as the
            # word-boundary space (eval_core skips the space-append on a match)
            if strips:
                ks_save = strips[0].word_len + 1 - strips[0].abbr_len
            else:
                from src.abbr_dict.speed import expand_with_rpt, build_abbr_keystream
                ks_save = (
                    len(expand_with_rpt(word, RPT_KEY)) + 1
                    - len(build_abbr_keystream(abbr, tf, RPT_KEY))
                )

            start = len(all_strips)
            all_strips.extend(strips)
            cand_ranges.append((start, len(all_strips)))
            cand_meta.append((item_idx, cand_idx, ks_save))

    return all_strips, cand_meta, cand_ranges


def _score_strips(all_strips, left_len, engine, layout_paths, dl, lgbm, word_freq_map):
    word_kss   = [s.word_ks   for s in all_strips]
    abbr_kss   = [s.abbr_ks   for s in all_strips]
    host_words = [s.host_word for s in all_strips]
    word_lens  = [s.word_len  for s in all_strips]
    abbr_lens  = [s.abbr_len  for s in all_strips]

    max_w_back  = max(dl.w_back,  lgbm.w_back)
    max_w_ahead = max(dl.w_ahead, lgbm.w_ahead)

    results = {}
    for label, kss, lens in [("word", word_kss, word_lens), ("abbr", abbr_kss, abbr_lens)]:
        dl_costs   = []
        lgbm_costs = []

        for i in range(0, len(kss), SEQ_BATCH_SIZE):
            batch_kss   = kss[i : i + SEQ_BATCH_SIZE]
            batch_hosts = host_words[i : i + SEQ_BATCH_SIZE]
            batch_lens  = lens[i : i + SEQ_BATCH_SIZE]

            df_feats, offsets = enrich_texts(
                batch_kss, engine, layout_paths, INFERENCE_FEATURES,
                max_w_back, max_w_ahead,
            )
            override_word_frequency(df_feats, offsets, batch_kss, batch_hosts, word_freq_map)

            if "sequence_pos" in df_feats.columns:
                df_feats["sequence_pos"] += 10.0

            dl_costs.extend(
                compute_costs_sliced(df_feats, offsets, dl, left_len, batch_lens)
            )
            lgbm_costs.extend(
                compute_costs_sliced(df_feats, offsets, lgbm, left_len, batch_lens)
            )

            print(f"  [{label}] {min(i + SEQ_BATCH_SIZE, len(kss))}/{len(kss)}")

        results[label] = {"dl": dl_costs, "lgbm": lgbm_costs}

    return results


def _aggregate_savings(all_strips, cand_ranges, strip_scores):
    word_dl   = strip_scores["word"]["dl"]
    word_lgbm = strip_scores["word"]["lgbm"]
    abbr_dl   = strip_scores["abbr"]["dl"]
    abbr_lgbm = strip_scores["abbr"]["lgbm"]

    savings_dl   = []
    savings_lgbm = []

    for start, end in cand_ranges:
        if start == end:
            savings_dl.append((0.0, 0.0))
            savings_lgbm.append((0.0, 0.0))
            continue

        weights = [all_strips[k].weight for k in range(start, end)]
        dl_sav   = [word_dl[k][0]   - abbr_dl[k][0]   for k in range(start, end)]
        lgbm_sav = [word_lgbm[k][0] - abbr_lgbm[k][0] for k in range(start, end)]

        d_mean = weighted_mean_savings(dl_sav, weights)
        l_mean = weighted_mean_savings(lgbm_sav, weights)

        # propagate std via quadrature over instances (approximate)
        d_std = float(np.sqrt(sum(
            (word_dl[k][1] ** 2 + abbr_dl[k][1] ** 2) * (w ** 2)
            for k, w in zip(range(start, end), weights)
        ) / (sum(weights) ** 2 + 1e-12)))
        l_std = float(np.sqrt(sum(
            (word_lgbm[k][1] ** 2 + abbr_lgbm[k][1] ** 2) * (w ** 2)
            for k, w in zip(range(start, end), weights)
        ) / (sum(weights) ** 2 + 1e-12)))

        savings_dl.append((d_mean, d_std))
        savings_lgbm.append((l_mean, l_std))

    return savings_dl, savings_lgbm


def main():
    print(f"Loading candidates: {CANDIDATES_PATH}")
    print(f"Loading frequencies: {WORDS_ZIPF_PATH}")
    print(f"Saving results to:  {OUTPUT_PATH}")

    with open(CANDIDATES_PATH) as f:
        data = json.load(f)
    with open(WORDS_ZIPF_PATH) as f:
        word_freq_map = json.load(f)

    cand_rpt_key = data.get("rpt_key")
    if cand_rpt_key is None:
        raise ValueError(
            f"{CANDIDATES_PATH} has no `rpt_key` header — regenerate it with the "
            "updated generate_candidates.py before running precompute_speed.py."
        )
    if cand_rpt_key != RPT_KEY:
        raise ValueError(
            f"rpt_key mismatch: candidates file says {cand_rpt_key!r} but "
            f"precompute_speed.py is configured with RPT_KEY={RPT_KEY!r}."
        )

    items = data["items"]
    if TEST_MODE:
        items = items[:TEST_N_ITEMS]
        print(f"[TEST MODE] Using {len(items)} items")
    else:
        print(f"Loaded {len(items)} items")

    layout_paths, engine, _ = build_layout_and_engine(DATA_ROOT, True)

    dl   = DLSpeedModel(DL_MODEL_DIR)
    lgbm = LGBMSpeedModel(LGBM_MODEL_DIR)
    print(f"DL:   {len(dl.models)} folds  w_back={dl.w_back} w_ahead={dl.w_ahead}")
    print(f"LGBM: {len(lgbm.models)} folds  w_back={lgbm.w_back} w_ahead={lgbm.w_ahead}")

    left_len  = max(dl.w_back,  lgbm.w_back)
    right_len = max(dl.w_ahead, lgbm.w_ahead)

    sentences   = _load_corpus_sentences()
    word_to_sents  = build_word_index(sentences)
    suffix_index   = build_suffix_index(word_to_sents)
    print(f"  Word index: {len(word_to_sents):,} entries")
    print(f"  Suffix index: {len(suffix_index):,} entries")

    all_strips, cand_meta, cand_ranges = _collect_strips(
        items, word_to_sents, suffix_index, word_freq_map, left_len, right_len,
    )
    n_no_hits = sum(1 for s, e in cand_ranges if s == e)
    print(f"  Total strips: {len(all_strips):,}  (candidates with no corpus hits: {n_no_hits})")

    strip_scores = _score_strips(all_strips, left_len, engine, layout_paths, dl, lgbm, word_freq_map)

    savings_dl, savings_lgbm = _aggregate_savings(all_strips, cand_ranges, strip_scores)

    # group candidates back by item
    item_candidates: list[list] = [[] for _ in items]
    for ci, (item_idx, cand_idx, ks_save) in enumerate(cand_meta):
        cand = items[item_idx]["candidates"][cand_idx]
        d_mean, d_std = savings_dl[ci]
        l_mean, l_std = savings_lgbm[ci]
        item_candidates[item_idx].append({
            "abbr": cand["abbr"],
            "trigger_form": cand["trigger_form"],
            "intuitiveness": cand.get("intuitiveness"),
            "savings": {
                "dl":         {"mean": round(d_mean, 5), "std": round(d_std, 5)},
                "lgbm":       {"mean": round(l_mean, 5), "std": round(l_std, 5)},
                # keystroke-count length term, used by filter_streaming
                "keystrokes": {"mean": ks_save,          "std": 0.0},
            },
        })

    out_items = [
        {
            "text":      item["text"],
            "type":      item["type"],
            "frequency": item.get("frequency"),
            "candidates": item_candidates[i],
        }
        for i, item in enumerate(items)
    ]

    result = {"rpt_key": RPT_KEY, "items": out_items}
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    all_dl   = np.array([c["savings"]["dl"]["mean"]   for i in out_items for c in i["candidates"]])
    all_lgbm = np.array([c["savings"]["lgbm"]["mean"] for i in out_items for c in i["candidates"]])

    print(f"\nSaved {len(out_items)} items → {OUTPUT_PATH}")
    print(f"DL   savings — mean={all_dl.mean():.4f}  std={all_dl.std():.4f}  "
          f"faster={(all_dl > 0).mean()*100:.1f}%")
    print(f"LGBM savings — mean={all_lgbm.mean():.4f}  std={all_lgbm.std():.4f}  "
          f"faster={(all_lgbm > 0).mean()*100:.1f}%")
    if len(all_dl) > 1:
        print(f"Pearson DL/LGBM: {np.corrcoef(all_dl, all_lgbm)[0, 1]:.4f}")


if __name__ == "__main__":
    main()