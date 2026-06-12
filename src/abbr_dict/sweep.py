import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.abbr_dict.optimize import load_and_preprocess, run_optimize
from src.abbr_dict.config import RPT_KEY, data_path

DATA_PATH         = data_path("speed_savings_filtered.json")
OUTPUT_DIR        = data_path("sweep_results")
LAMBDAS           = [round(x * 0.1, 1) for x in range(0, 11)]
OVERLAP_PENALTY   = 1.0
SIGMA_PENALTY     = 1.0
MIN_INTUITIVENESS = 0.30
MAX_DICT_SIZE     = 160
SOLVER_TIMEOUT    = 300.0
MODEL_NAME        = "dl"      # optimize with DL, evaluate with LGBM (anti-Goodhart)
TOP_K             = 12    
MAX_SUFFIX_LEN    = 5
INCLUDE_SUFFIXES  = True


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Sweeping λ = {LAMBDAS} (K={TOP_K}, model={MODEL_NAME})\n")

    summary_results = []

    for lambda_ in LAMBDAS:
        print(f"── λ={lambda_:.1f} ─────────────────────────────────────")
        # re-shortlist per λ so intuitiveness shapes the per-word top-K at high λ
        processed_items, b_to_a_overlaps = load_and_preprocess(
            DATA_PATH,
            include_suffixes=INCLUDE_SUFFIXES,
            model_name=MODEL_NAME,
            top_k_per_word=TOP_K,
            max_suffix_len=MAX_SUFFIX_LEN,
            sigma_penalty=SIGMA_PENALTY,
            lambda_=lambda_,
        )
        result = run_optimize(
            processed_items, b_to_a_overlaps,
            lambda_=lambda_,
            overlap_penalty=OVERLAP_PENALTY,
            min_intuitiveness=MIN_INTUITIVENESS,
            max_dict_size=MAX_DICT_SIZE,
            rpt_key=RPT_KEY,
            model_name=MODEL_NAME,
            solver_timeout=SOLVER_TIMEOUT,
            sigma_penalty=SIGMA_PENALTY,
            verbose=False,
        )
        if result is None:
            print(f"  FAILED\n")
            continue

        m = result["metadata"]
        gap_str = f"gap={m['solver_gap_pct']:.2f}%" if m['solver_status'] != 'OPTIMAL' else "OPTIMAL"
        print(f"  dict_size={m['dict_size']}  net_z={m['total_z_savings_net']:.1f}"
              f"  avg_intuit={m['avg_intuitiveness']:.3f}  t={m['solver_wall_time_s']:.1f}s  [{gap_str}]")

        dict_path = os.path.join(OUTPUT_DIR, f"dict_lambda_{lambda_:.2f}.json")
        with open(dict_path, "w") as f:
            json.dump(result, f, indent=2)

        summary_results.append({
            "lambda":            lambda_,
            "dict_size":         m["dict_size"],
            "net_z_savings":     m["total_z_savings_net"],
            "avg_intuitiveness": m["avg_intuitiveness"],
            "solve_time":        m["solver_wall_time_s"],
            "solver_status":     m["solver_status"],
            "gap_pct":           m["solver_gap_pct"],
        })

    summary = {
        "params": {
            "top_k":             TOP_K,
            "overlap_penalty":   OVERLAP_PENALTY,
            "sigma_penalty":     SIGMA_PENALTY,
            "min_intuitiveness": MIN_INTUITIVENESS,
            "max_dict_size":     MAX_DICT_SIZE,
            "model":             MODEL_NAME,
        },
        "results": summary_results,
    }
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    if summary_results:
        _plot_pareto(summary_results, OUTPUT_DIR)


def _plot_pareto(results, output_dir):
    lambdas  = [r["lambda"]            for r in results]
    x_vals   = [r["avg_intuitiveness"] for r in results]
    y_vals   = [r["net_z_savings"]     for r in results]
    sizes    = [r["dict_size"]         for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(x_vals, y_vals, c=lambdas, cmap="viridis", s=80, vmin=0, vmax=1, zorder=3)
    for x, y, lam, sz in zip(x_vals, y_vals, lambdas, sizes):
        ax.annotate(f"λ={lam:.1f}\n(n={sz})", (x, y),
                    textcoords="offset points", xytext=(6, 4), fontsize=7.5)

    ax.set_xlabel("Mean intuitiveness of selected entries")
    ax.set_ylabel("Net Z-score savings (frequency-weighted)")
    ax.set_title("Pareto frontier: speed savings vs intuitiveness")
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, "pareto.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved Pareto plot to {plot_path}")


if __name__ == "__main__":
    main()