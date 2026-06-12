import math
import os
import re
import sys
from collections import defaultdict
import numpy as np

LOCAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, LOCAL_ROOT)

from src.utils.inference import enrich_texts, compute_costs
from src.models.wrappers import INFERENCE_FEATURES
from src.abbr_dict.speed import (
    expand_with_rpt, build_abbr_keystream,
    TRG_CHAR, RPT_CHAR,
)

DISPLAY_TRG = "★"
DISPLAY_RPT = "↻"

IKI_MEAN_MS  = 110.5
IKI_STD_MS   = 50.4
BATCH_SIZE   = 50_000
MIN_ENTRY_USES = 1


def display_ks(text):
    return text.replace(TRG_CHAR, DISPLAY_TRG).replace(RPT_CHAR, DISPLAY_RPT)


_PUNCT_RE = re.compile(r"^([^a-zA-Z0-9']*)(.*?)([^a-zA-Z0-9']*)$", re.DOTALL)

def split_punct(token):
    m = _PUNCT_RE.match(token)
    return (m.group(1), m.group(2), m.group(3)) if m else ("", token, "")


def build_sentence_pair(sentence, singleword_map, suffix_list, has_suffixes, rpt_key):
    tokens = sentence.split()
    normal_chars   = []
    abbr_chars     = []
    freq_overrides = []
    match_keys     = []
    stats = {"singleword": 0, "suffix": 0, "none": 0}
    prev_was_match = False

    for i, token in enumerate(tokens):
        pre, word_body, post = split_punct(token)
        word_lower = word_body.lower()

        normal_token = list(pre) + list(word_body) + list(post)
        abbr_token   = None
        target_word  = None
        match_type   = None

        if word_lower in singleword_map:
            entry = singleword_map[word_lower]
            abbr = entry["abbr"]
            if word_body and word_body[0].isupper() and abbr:
                abbr = abbr[0].upper() + abbr[1:]
            abbr_token  = list(pre) + build_abbr_keystream(abbr, entry["trigger_form"], rpt_key) + list(post)
            target_word = word_lower
            match_type  = "singleword"

        elif has_suffixes:
            for suffix, entry in suffix_list:
                if word_lower.endswith(suffix) and len(word_lower) > len(suffix):
                    stem = word_body[: -len(suffix)]
                    abbr_token = (
                        list(pre)
                        + expand_with_rpt(stem, rpt_key)
                        + build_abbr_keystream(entry["abbr"], entry["trigger_form"], rpt_key)
                        + list(post)
                    )
                    target_word = word_lower
                    match_type  = "suffix"
                    break

        if match_type is not None:
            abbr_start = len(abbr_chars)
            abbr_chars.extend(abbr_token)
            freq_overrides.append((abbr_start, len(abbr_chars), target_word))
            entry_key = suffix if match_type == "suffix" else target_word
            match_keys.append(entry_key)
            stats[match_type] += 1
            prev_was_match = True
        else:
            # unmatched words still use RPT encoding (to measure raw RPT overhead); the
            # freq override restores the original word's Zipf, since "let\x02er" isn't in the dict
            abbr_start = len(abbr_chars)
            expanded = expand_with_rpt(word_body, rpt_key)
            abbr_chars.extend(list(pre) + expanded + list(post))
            if word_lower:
                freq_overrides.append((abbr_start + len(pre), abbr_start + len(pre) + len(expanded), word_lower))
            stats["none"] += 1
            prev_was_match = False

        normal_chars.extend(normal_token)
        if i < len(tokens) - 1:
            normal_chars.append(" ")
            if not prev_was_match:
                abbr_chars.append(" ")

    return "".join(normal_chars), "".join(abbr_chars), freq_overrides, stats, match_keys


def apply_freq_overrides(df_feats, offsets, batch_overrides, word_freq_map):
    freq_col = df_feats.columns.get_loc("word_frequency")
    for sent_idx, overrides in enumerate(batch_overrides):
        if not overrides:
            continue
        seq_start = offsets[sent_idx][0]
        for char_start, char_end, target_word in overrides:
            zipf = float(word_freq_map.get(target_word, 0.0))
            df_feats.iloc[seq_start + char_start : seq_start + char_end, freq_col] = zipf


def compute_trg_bigram_freqs(dict_data, word_freq_map):
    raw_sums = {}
    for entry in dict_data["dictionary"]:
        zipf = word_freq_map.get(entry["text"].lower(), 0.0)
        if zipf <= 0:
            continue
        raw = 10 ** zipf
        tf = entry["trigger_form"]
        if tf == "trg":
            key = entry["abbr"][-1] + TRG_CHAR
            raw_sums[key] = raw_sums.get(key, 0.0) + raw
        elif tf == "rpt_trg":
            k1 = entry["abbr"][0] + RPT_CHAR
            k2 = RPT_CHAR + TRG_CHAR
            raw_sums[k1] = raw_sums.get(k1, 0.0) + raw
            raw_sums[k2] = raw_sums.get(k2, 0.0) + raw
    return {bg: math.log10(r) for bg, r in raw_sums.items() if r > 0}


def _resolve_rpt_char(text, pos):
    while pos >= 0 and text[pos] == RPT_CHAR:
        pos -= 1
    return text[pos] if pos >= 0 else ''


def patch_trg_bigram_freqs(df_feats, offsets, texts, trg_bigram_freqs, raw_bigram_map=None):
    # matched bigrams use trg_bigram_freqs; unmatched RPT bigrams fall back to the
    # original repeated-letter frequency in raw_bigram_map
    bg_col = df_feats.columns.get_loc("bigram_frequency")
    for idx, text in enumerate(texts):
        start, length = offsets[idx]
        for pos in range(1, length):
            prev = text[pos - 1]
            curr = text[pos]
            if TRG_CHAR in (prev, curr) or RPT_CHAR in (prev, curr):
                bigram = prev + curr
                if bigram in trg_bigram_freqs:
                    df_feats.iloc[start + pos, bg_col] = trg_bigram_freqs[bigram]
                elif raw_bigram_map is not None and RPT_CHAR in (prev, curr):
                    real_prev = _resolve_rpt_char(text, pos - 1) if prev == RPT_CHAR else prev
                    real_curr = _resolve_rpt_char(text, pos)     if curr == RPT_CHAR else curr
                    freq = raw_bigram_map.get(real_prev + real_curr)
                    if freq is not None:
                        df_feats.iloc[start + pos, bg_col] = freq


def infer_batch(texts, overrides_list, engine, layout_paths, model, word_freq_map,
                trg_bigram_freqs=None, raw_bigram_map=None):
    df_feats, offsets = enrich_texts(texts, engine, layout_paths, INFERENCE_FEATURES, model.w_back, model.w_ahead)
    if overrides_list is not None:
        apply_freq_overrides(df_feats, offsets, overrides_list, word_freq_map)
    if trg_bigram_freqs or raw_bigram_map:
        patch_trg_bigram_freqs(df_feats, offsets, texts, trg_bigram_freqs or {}, raw_bigram_map)
    costs = compute_costs(df_feats, offsets, model)
    return costs, df_feats, offsets


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield i, lst[i : i + n]


def run_evaluation(dict_data, sentences, engine, layout_paths, model, word_freq_map,
                   return_debug=False, raw_bigram_map=None):
    meta    = dict_data["metadata"]
    rpt_key = meta["rpt_key"]

    singleword_map = {}
    suffix_list    = []
    for entry in dict_data["dictionary"]:
        t = entry["type"]
        if t in ("singleword", "linked"):
            singleword_map[entry["text"].lower()] = entry
        elif t == "suffix":
            suffix_list.append((entry["text"].lower(), entry))
    suffix_list.sort(key=lambda x: -len(x[0]))
    has_suffixes = bool(suffix_list)

    trg_bigram_freqs = compute_trg_bigram_freqs(dict_data, word_freq_map)

    all_data = []
    for sent in sentences:
        normal_text, abbr_text, freq_overrides, stats, match_keys = build_sentence_pair(
            sent, singleword_map, suffix_list, has_suffixes, rpt_key
        )
        all_data.append((sent, normal_text, abbr_text, freq_overrides, stats, match_keys))

    normal_costs = []
    for _start_i, batch in chunks(all_data, BATCH_SIZE):
        costs, _, _ = infer_batch([d[1] for d in batch], None, engine, layout_paths, model, word_freq_map)
        normal_costs.extend(costs)

    abbr_costs      = []
    debug_df_feats  = None
    debug_offsets   = None
    debug_batch_end = 0
    for batch_idx, (start_i, batch) in enumerate(chunks(all_data, BATCH_SIZE)):
        costs, df_f, offs = infer_batch(
            [d[2] for d in batch], [d[3] for d in batch],
            engine, layout_paths, model, word_freq_map,
            trg_bigram_freqs=trg_bigram_freqs,
            raw_bigram_map=raw_bigram_map,
        )
        abbr_costs.extend(costs)
        if batch_idx == 0:
            debug_df_feats  = df_f
            debug_offsets   = offs
            debug_batch_end = len(batch)

    normal_z_arr = np.array([c[0] for c in normal_costs])
    abbr_z_arr   = np.array([c[0] for c in abbr_costs])
    n_norm_arr   = np.array([len(d[1]) for d in all_data])
    n_abbr_arr   = np.array([len(d[2]) for d in all_data])

    t_normal_arr   = np.maximum(0, n_norm_arr - 1) * IKI_MEAN_MS + normal_z_arr * IKI_STD_MS
    t_abbr_arr     = np.maximum(0, n_abbr_arr - 1) * IKI_MEAN_MS + abbr_z_arr   * IKI_STD_MS
    ms_savings_arr = t_normal_arr - t_abbr_arr
    z_savings_arr  = normal_z_arr - abbr_z_arr

    total_t_normal  = float(t_normal_arr.sum())
    total_t_abbr    = float(t_abbr_arr.sum())
    total_ms_saved  = float(ms_savings_arr.sum())
    agg_speedup_pct = total_ms_saved / max(1.0, total_t_normal) * 100.0

    total_tokens = sum(sum(d[4].values()) for d in all_data)
    n_sw  = sum(d[4]["singleword"] for d in all_data)
    n_suf = sum(d[4]["suffix"]     for d in all_data)
    n_cov = n_sw + n_suf

    sw_per_match  = []
    suf_per_match = []
    for i in range(len(all_data)):
        stats      = all_data[i][4]
        n_this_sw  = stats["singleword"]
        n_this_suf = stats["suffix"]
        n_this_tot = n_this_sw + n_this_suf
        if n_this_tot == 0:
            continue
        ms = float(ms_savings_arr[i])
        if n_this_sw > 0:
            sw_per_match.append(ms * n_this_sw / n_this_tot / n_this_sw)
        if n_this_suf > 0:
            suf_per_match.append(ms * n_this_suf / n_this_tot / n_this_suf)

    entry_savings_samples = defaultdict(list)
    entry_normal_samples  = defaultdict(list)
    for i in range(len(all_data)):
        keys      = all_data[i][5]
        n_matches = len(keys)
        if n_matches == 0:
            continue
        ms  = float(ms_savings_arr[i])
        t_n = float(t_normal_arr[i])
        for key in keys:
            entry_savings_samples[key].append(ms / n_matches)
            entry_normal_samples[key].append(t_n / n_matches)

    entry_meta     = {e["text"].lower(): e for e in dict_data["dictionary"]}
    per_entry_stats = []
    for key, samples in entry_savings_samples.items():
        if len(samples) < MIN_ENTRY_USES:
            continue
        mean_sav    = float(np.mean(samples))
        std_sav     = float(np.std(samples))
        mean_norm   = float(np.mean(entry_normal_samples[key]))
        speedup_pct = (mean_sav / mean_norm * 100.0) if mean_norm > 0 else 0.0
        meta_e      = entry_meta.get(key, {})
        per_entry_stats.append({
            "text":          key,
            "abbr":          meta_e.get("abbr", "?"),
            "trigger_form":  meta_e.get("trigger_form", "?"),
            "intuitiveness": meta_e.get("intuitiveness"),
            "uses":          len(samples),
            "mean_ms":       round(mean_sav, 2),
            "std_ms":        round(std_sav, 2),
            "speedup_pct":   round(speedup_pct, 3),
            "total_ms":      round(mean_sav * len(samples), 1),
        })
    per_entry_stats.sort(key=lambda r: r["total_ms"], reverse=True)

    intuits = [e["intuitiveness"] for e in dict_data["dictionary"] if "intuitiveness" in e]
    result = {
        "agg_speedup_pct":          round(agg_speedup_pct, 4),
        "dict_size":                len(dict_data["dictionary"]),
        "dict_mean_intuitiveness":  round(float(np.mean(intuits)), 4) if intuits else None,
        "dict_min_intuitiveness":   round(float(np.min(intuits)),  4) if intuits else None,
        "total_ms_saved":           round(total_ms_saved, 2),
        "total_t_normal_s":         round(total_t_normal / 1000.0, 3),
        "total_t_abbr_s":           round(total_t_abbr   / 1000.0, 3),
        "mean_sentence_savings_ms": round(float(ms_savings_arr.mean()), 4),
        "std_sentence_savings_ms":  round(float(ms_savings_arr.std()),  4),
        "z_savings_mean":           round(float(z_savings_arr.mean()),  6),
        "z_savings_std":            round(float(z_savings_arr.std()),   6),
        "coverage_pct":             round(100.0 * n_cov / max(1, total_tokens), 4),
        "n_sentences":              len(all_data),
        "avg_sentence_len":         round(sum(len(d[0]) for d in all_data) / max(len(all_data), 1), 1),
        "n_singleword_matches":     n_sw,
        "n_suffix_matches":         n_suf,
        "total_tokens":             total_tokens,
        "singleword_ms_per_match":  round(float(np.mean(sw_per_match)),  2) if sw_per_match  else None,
        "suffix_ms_per_match":      round(float(np.mean(suf_per_match)), 2) if suf_per_match else None,
        "per_entry_stats":          per_entry_stats,
    }

    if not return_debug:
        return result

    debug = {
        "all_data":         all_data,
        "normal_costs":     normal_costs,
        "abbr_costs":       abbr_costs,
        "ms_savings_arr":   ms_savings_arr,
        "debug_df_feats":   debug_df_feats,
        "debug_offsets":    debug_offsets,
        "debug_batch_end":  debug_batch_end,
        "has_suffixes":     has_suffixes,
        "trg_bigram_freqs": trg_bigram_freqs,
    }
    return result, debug