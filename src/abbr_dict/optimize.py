import json
import os
import sys
import time
import numpy as np
from ortools.sat.python import cp_model
import argparse

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import STORAGE_ROOT, DATA_ROOT
from src.abbr_dict.config import RPT_KEY, data_path

DATA_PATH = data_path("speed_savings_filtered.json")
OUTPUT_DIR = data_path("optimization_results")

SCALE = 100000

DEFAULT_LAMBDA = 0.5
DEFAULT_OVERLAP_PENALTY = 1.0 # suffix-overlap penalty coefficient
DEFAULT_SIGMA_PENALTY = 1.0   # ensemble-uncertainty penalty coefficient (k_sigma)
DEFAULT_MIN_INTUITIVENESS = 0.30
DEFAULT_MAX_DICT_SIZE = 160
DEFAULT_MODEL = "lgbm"

LINK_IDENTICAL = True
INCLUDE_SUFFIXES = True 


def is_doublechar(abbr):
    return len(abbr) == 2 and abbr[0] == abbr[1]


def _score_key(model_name):
    return f"savings_z_{model_name}"


def _std_key(model_name):
    return f"savings_z_{model_name}_std"


def _effective_savings(item, score_key, std_key, sigma_penalty):
    # uncertainty-penalized savings s_eff = mean - k_sigma * std, down-ranking
    # candidates whose predicted saving is noisy across the ensemble
    return item.get(score_key, 0) - sigma_penalty * item.get(std_key, 0)


def load_and_preprocess(data_path, include_suffixes=True, include_words=True, link_identical=True,
                        max_suffix_len=None, model_name="lgbm", top_k_per_word=1,
                        sigma_penalty=DEFAULT_SIGMA_PENALTY, *, lambda_):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    with open(data_path) as f:
        data = json.load(f)

    raw_items = data["items"]
    if not include_suffixes:
        raw_items = [item for item in raw_items if item["type"] != "suffix"]
    if not include_words:
        raw_items = [item for item in raw_items if item["type"] == "suffix"]
    if RPT_KEY:
        # doublechar "cc"+TRG is keystroke-identical to rpt_trg when the repeat key
        # is on, so it is a duplicate trigger — drop it. (with rpt off, "cc"+TRG =
        # [c,c,TRG] is the only valid form and is kept.)
        raw_items = [it for it in raw_items
                     if not (is_doublechar(it["abbr"]) and it["trigger_form"] == "trg")]
    if max_suffix_len is not None:
        raw_items = [item for item in raw_items if not (item["type"] == "suffix" and len(item["text"]) > max_suffix_len)]

    # group identical (text, abbr, trigger_form) entries that appear under
    # multiple types (e.g. same abbr as both singleword and suffix)
    if link_identical:
        key_to_items = {}
        for item in raw_items:
            key = (item["text"], item["abbr"], item["trigger_form"])
            key_to_items.setdefault(key, []).append(item)
        merged_items = []
        for group in key_to_items.values():
            if len(group) == 1:
                merged_items.append(group[0])
            else:
                main = group[0].copy()
                main["frequency"] = sum(it.get("frequency", 0) for it in group)
                main["is_linked"] = True
                main["types"] = [it["type"] for it in group]
                merged_items.append(main)
        raw_items = merged_items

    # pre-pick top-K forms per entry
    score_key = _score_key(model_name)
    std_key   = _std_key(model_name)

    def _shortlist_key(it):
        # same speed/intuitiveness blend the ILP uses
        s_eff = _effective_savings(it, score_key, std_key, sigma_penalty)
        return (1 - lambda_) * s_eff + lambda_ * 10 * it.get("intuitiveness", 0)

    text_to_cands = {}
    for item in raw_items:
        text_to_cands.setdefault(item["text"], []).append(item)
    processed_items = []
    for cands in text_to_cands.values():
        ranked = sorted(cands, key=_shortlist_key, reverse=True)
        processed_items.extend(ranked[:top_k_per_word])

    # for each entry, check every sub-suffix of its text (e.g. "day" in "today")
    # against the dict's suffixes. greedy matching fires the entry's own rule first,
    # so an overlapped smaller suffix's saving is lost.
    text_to_indices = {}
    for j, item_a in enumerate(processed_items):
        if item_a["type"] == "suffix" and item_a["text"] not in text_to_indices:
            text_to_indices[item_a["text"]] = [j]
    b_to_a_overlaps = {}
    for i, item_b in enumerate(processed_items):
        text_b = item_b["text"]
        for cut in range(1, len(text_b)):
            for j in text_to_indices.get(text_b[cut:], ()):
                if j != i:
                    b_to_a_overlaps.setdefault(i, []).append(j)

    return processed_items, b_to_a_overlaps


def run_optimize(
    processed_items,
    b_to_a_overlaps,
    lambda_,
    overlap_penalty,
    min_intuitiveness,
    max_dict_size,
    rpt_key=True,
    model_name="lgbm",
    solver_timeout=300.0,
    verbose=True,
    top_k_per_item=None,
    doubletap_only=False,
    trg_only=False,
    sigma_penalty=DEFAULT_SIGMA_PENALTY,
):
    def log(msg):
        if verbose:
            print(msg)

    model = cp_model.CpModel()

    score_key = _score_key(model_name)
    std_key   = _std_key(model_name)
    valid_indices = []
    for i, item in enumerate(processed_items):
        s_eff = _effective_savings(item, score_key, std_key, sigma_penalty)
        if item.get("intuitiveness", 0) >= min_intuitiveness and s_eff > 0:
            if doubletap_only:
                if is_doublechar(item["abbr"]) and item["trigger_form"] == "doubletap":
                    valid_indices.append(i)
            elif trg_only:
                if item["trigger_form"] == "trg":
                    valid_indices.append(i)
            else:
                valid_indices.append(i)

    if not valid_indices:
        log("No candidates found meeting the intuitiveness floor.")
        return None

    # x[i] = 1 if candidate i is included
    x = {}
    for i in valid_indices:
        x[i] = model.NewBoolVar(f'x_{i}')

    # at most one abbreviation per word (only matters when linking is off)
    text_to_indices = {}
    for i in valid_indices:
        txt = processed_items[i]["text"]
        if txt not in text_to_indices:
            text_to_indices[txt] = []
        text_to_indices[txt].append(i)

    for txt, indices in text_to_indices.items():
        if len(indices) > 1:
            model.AddAtMostOne(x[i] for i in indices)

    # each (abbr, trigger_form) expands to at most one text
    abbr_to_indices = {}
    for i in valid_indices:
        item = processed_items[i]
        key = (item["abbr"], item["trigger_form"])
        abbr_to_indices.setdefault(key, []).append(i)

    for indices in abbr_to_indices.values():
        if len(indices) > 1:
            model.AddAtMostOne(x[i] for i in indices)

    model.Add(sum(x[i] for i in valid_indices) <= max_dict_size)

    # objective: maximize sum of frequency * speedup_z, blended with intuitiveness by lambda
    obj_terms = []
    for i in valid_indices:
        item = processed_items[i]
        freq = item["frequency"]
        if model_name in ("dl", "lgbm"):
            speedup = _effective_savings(item, score_key, std_key, sigma_penalty)
        else:
            speedup = item.get("savings_z_mean", 0)

        # scale to int for the CP-SAT solver
        reward = int(freq * speedup * SCALE)

        if lambda_ > 0:
            # scale intuitiveness (0..1) to match typical Z-impact (0..20)
            i_reward = int(item.get("intuitiveness", 0) * freq * 10 * SCALE)
            total_reward = int((1 - lambda_) * reward + lambda_ * i_reward)
        else:
            total_reward = reward

        # if both an entry and a suffix it overlaps are picked, subtract freq(b)*savings(a)
        if i in b_to_a_overlaps and overlap_penalty > 0:
            for j in b_to_a_overlaps[i]:
                if j in valid_indices:
                    # z_ij = x_i AND x_j, linearized
                    z_ij = model.NewBoolVar(f'z_{i}_{j}')
                    model.Add(z_ij <= x[i])
                    model.Add(z_ij <= x[j])
                    model.Add(z_ij >= x[i] + x[j] - 1)

                    a_speedup = processed_items[j].get("savings_z_mean", 0)
                    penalty = int(overlap_penalty * freq * a_speedup * SCALE)
                    obj_terms.append(-z_ij * penalty)

        obj_terms.append(x[i] * total_reward)

    model.Maximize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = solver_timeout
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        status_name = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
        obj_value = solver.ObjectiveValue()
        best_bound = solver.BestObjectiveBound()
        gap_pct = (
            0.0 if status == cp_model.OPTIMAL
            else abs(best_bound - obj_value) / max(abs(obj_value), 1e-9) * 100
        )
        log(f"Solver: {status_name}  obj={obj_value:,.0f}  bound={best_bound:,.0f}  gap={gap_pct:.2f}%  time={solver.WallTime():.1f}s")

        selected = [i for i in valid_indices if solver.Value(x[i]) == 1]

        entries = []
        total_z_savings_gross = 0
        total_z_penalty = 0
        total_intuitiveness = 0
        
        for i in selected:
            item = processed_items[i]
            entries.append({
                "text": item["text"],
                "abbr": item["abbr"],
                "trigger_form": item["trigger_form"],
                "type": item["type"],
                "frequency": item["frequency"],
                "savings_z_mean": item.get("savings_z_mean", 0),
                "savings_z_std": item.get(std_key, 0),
                "savings_z_eff": _effective_savings(item, score_key, std_key, sigma_penalty),
                "savings_keystrokes": item.get("savings_keystrokes", 0),
                "intuitiveness": item.get("intuitiveness", 0)
            })
            total_z_savings_gross += item["frequency"] * item.get("savings_z_mean", 0)
            total_intuitiveness += item.get("intuitiveness", 0)
            
            # recompute overlap penalty, for metadata only
            if i in b_to_a_overlaps:
                for j in b_to_a_overlaps[i]:
                    if j in selected:
                        a_speedup = processed_items[j].get("savings_z_mean", 0)
                        total_z_penalty += item["frequency"] * a_speedup

        result = {
            "metadata": {
                "lambda": lambda_,
                "overlap_penalty": overlap_penalty,
                "sigma_penalty": sigma_penalty,
                "min_intuitiveness": min_intuitiveness,
                "max_dict_size": max_dict_size,
                "model_name": model_name,
                "rpt_key": rpt_key,
                "dict_size": len(selected),
                "total_z_savings_gross": total_z_savings_gross,
                "total_z_penalty": total_z_penalty,
                "total_z_savings_net": total_z_savings_gross - total_z_penalty,
                "avg_intuitiveness": total_intuitiveness / len(selected) if selected else 0,
                "solver_status": status_name,
                "solver_gap_pct": gap_pct,
                "solver_wall_time_s": solver.WallTime(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "dictionary": entries
        }
        return result
    else:
        log("No solution found or solver error.")
        return None


def _save_and_print(result, output_dir, output_name):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    m = result["metadata"]
    entries = result["dictionary"]
    print(f"\nSaved {m['dict_size']} entries to {out_path}")
    print(f"Gross Z-Savings: {m['total_z_savings_gross']:,.2f}")
    print(f"Overlap Penalty: {m['total_z_penalty']:,.2f}")
    print(f"NET Z-Savings:   {m['total_z_savings_net']:,.2f}")
    print(f"Avg intuitiveness: {m['avg_intuitiveness']:.3f}")

    entries_sorted = sorted(entries, key=lambda x: x["frequency"] * x["savings_z_mean"], reverse=True)
    for entry in entries_sorted[:10]:
        impact = entry["frequency"] * entry["savings_z_mean"]
        print(f"  {entry['text']:12} -> {entry['abbr']:5} ({entry['trigger_form']:10}) | Z-Impact: {impact:8.1f}")


def main():
    parser = argparse.ArgumentParser(description="Optimize abbreviation dictionary.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--size", type=int, default=DEFAULT_MAX_DICT_SIZE)
    parser.add_argument("--lambda_", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--overlap_penalty", type=float, default=DEFAULT_OVERLAP_PENALTY)
    parser.add_argument("--sigma_penalty", type=float, default=DEFAULT_SIGMA_PENALTY)
    parser.add_argument("--min_i", type=float, default=DEFAULT_MIN_INTUITIVENESS)
    parser.add_argument("--doubletap", action="store_true")
    parser.add_argument("--trg_only", action="store_true")
    parser.add_argument("--words_only", action="store_true")
    parser.add_argument("--suffixes_only", action="store_true")
    parser.add_argument("--top_k_per_word", type=int, default=1)
    parser.add_argument("--output", type=str)

    args = parser.parse_args()

    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    print(f"Loading data from {DATA_PATH}...")
    include_suffixes = INCLUDE_SUFFIXES and not args.words_only
    include_words = not args.suffixes_only
    processed_items, b_to_a_overlaps = load_and_preprocess(
        DATA_PATH, include_suffixes=include_suffixes, include_words=include_words,
        link_identical=LINK_IDENTICAL,
        model_name=args.model, top_k_per_word=args.top_k_per_word,
        sigma_penalty=args.sigma_penalty, lambda_=args.lambda_,
    )
    print(f"After pre-pick (top_k={args.top_k_per_word} per word, model={args.model}): {len(processed_items)} items.")

    output_name = args.output if args.output else f"optimized_dict_{args.model}.json"

    result = run_optimize(
        processed_items, b_to_a_overlaps,
        lambda_=args.lambda_, overlap_penalty=args.overlap_penalty, min_intuitiveness=args.min_i,
        max_dict_size=args.size, rpt_key=RPT_KEY, model_name=args.model,
        doubletap_only=args.doubletap, trg_only=args.trg_only,
        verbose=True, sigma_penalty=args.sigma_penalty,
    )
    if result:
        _save_and_print(result, OUTPUT_DIR, output_name)


if __name__ == "__main__":
    main()
