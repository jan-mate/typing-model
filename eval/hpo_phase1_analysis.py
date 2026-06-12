# phase 1 and 2 share a study, so the winning config has phase-2 trials appended and
# an inflated/biased MAE. re-extract phase-1 numbers by taking the first PHASE_1_TRIALS
# trials per (w_back, w_ahead) study, sorted by datetime_start

import json
import os
import sys

import optuna
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from src.config import model_dir

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS = [
    {
        "name": "LinearRegression",
        "dir": model_dir("linreg", subdir=""),
        "db_file": "optuna_linreg_study.db",
        "study_prefix": "linreg_hpo_w",
        "w_back_grid": [1, 2, 3],
        "w_ahead_grid": [0, 1],
        "phase_1_trials": 20,
        "best_params_file": "best_linreg_params.json",
        "single_phase": True,
    },
    {
        "name": "LGBM",
        "dir": model_dir("lgbm", subdir=""),
        "db_file": "optuna_lgbm_study.db",
        "study_prefix": "lgbm_study_w",
        "w_back_grid": [1, 2, 3],
        "w_ahead_grid": [0, 1, 2],
        "phase_1_trials": 12,
        "best_params_file": "best_optuna_lgbm_params.json",
        "single_phase": False,
    },
    {
        "name": "MLP_Main",
        "dir": model_dir("mlp_main", subdir=""),
        "db_file": "mlp_optuna_study.db",
        "study_prefix": "mlp_study_w",
        "w_back_grid": [2, 3],
        "w_ahead_grid": [1, 2],
        "phase_1_trials": 12,
        "best_params_file": "best_optuna_mlp_params.json",
        "single_phase": False,
    },
    {
        "name": "MLP_DL",
        "dir": model_dir("mlp_dl", subdir=""),
        "db_file": "optuna_dl.db",
        "study_prefix": "dl_w",
        "w_back_grid": [2, 3],
        "w_ahead_grid": [1, 2],
        "phase_1_trials": 12,
        "best_params_file": "best_dl_params.json",
        "single_phase": False,
    },
    {
        "name": "MLP_Linguistic",
        "dir": model_dir("mlp_linguistic", subdir=""),
        "db_file": "optuna_linguistic.db",
        "study_prefix": "linguistic_w",
        "w_back_grid": [2, 3],
        "w_ahead_grid": [1, 2],
        "phase_1_trials": 8,
        "best_params_file": "best_linguistic_params.json",
        "single_phase": False,
    },
]


def study_name_for(cfg, w_b, w_a):
    # LinReg uses "linreg_hpo_w{w_b}_a{w_a}", others use "{prefix}{w_b}_a{w_a}"
    return f"{cfg['study_prefix']}{w_b}_a{w_a}"


def phase1_table(cfg):
    db_path = os.path.join(cfg["dir"], cfg["db_file"])
    storage = f"sqlite:///{db_path}"
    if not os.path.exists(db_path):
        print(f"  [skip] db not found: {db_path}")
        return None

    rows = []
    for w_b in cfg["w_back_grid"]:
        for w_a in cfg["w_ahead_grid"]:
            sname = study_name_for(cfg, w_b, w_a)
            try:
                study = optuna.load_study(study_name=sname, storage=storage)
            except KeyError:
                continue

            trials = sorted(
                study.trials,
                key=lambda t: t.datetime_start or pd.Timestamp.min.to_pydatetime(),
            )
            phase1 = trials[: cfg["phase_1_trials"]]

            completed = [
                t for t in phase1
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
            ]
            pruned = [t for t in phase1 if t.state == optuna.trial.TrialState.PRUNED]

            if completed:
                vals = [t.value for t in completed]
                best_mae = min(vals)
                avg_mae = sum(vals) / len(vals)
            else:
                best_mae = float("nan")
                avg_mae = float("nan")

            rows.append({
                "w_back": w_b,
                "w_ahead": w_a,
                "best_mae": best_mae,
                "avg_mae": avg_mae,
                "num_completed": len(completed),
                "num_pruned": len(pruned),
            })

    df = pd.DataFrame(rows).sort_values("best_mae").reset_index(drop=True)
    return df


def best_params_and_window(cfg):
    json_path = os.path.join(cfg["dir"], cfg["best_params_file"])
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f), "from json"

    # Fallback: scan all studies, find the one with the lowest best_value
    db_path = os.path.join(cfg["dir"], cfg["db_file"])
    if not os.path.exists(db_path):
        return None, "no db / no json"

    storage = f"sqlite:///{db_path}"
    best = None
    best_window = None
    for w_b in cfg["w_back_grid"]:
        for w_a in cfg["w_ahead_grid"]:
            sname = study_name_for(cfg, w_b, w_a)
            try:
                study = optuna.load_study(study_name=sname, storage=storage)
            except KeyError:
                continue
            try:
                val = study.best_value
            except ValueError:
                continue
            if best is None or val < best:
                best = val
                best_window = (w_b, w_a, study.best_params)

    if best_window is None:
        return None, "no completed trials"
    w_b, w_a, params = best_window
    params = dict(params)
    params["w_back"] = w_b
    params["w_ahead"] = w_a
    return params, "from study (best across all configs)"


def main():
    for cfg in MODELS:
        print("=" * 70)
        print(f"Group: {cfg['name']}  (phase_1_trials cap = {cfg['phase_1_trials']})")
        if cfg["single_phase"]:
            print("(single-phase HPO: no phase 2)")
        print("=" * 70)

        df = phase1_table(cfg)
        if df is not None:
            print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

        params, src = best_params_and_window(cfg)
        print(f"\nBest params ({src}):")
        if params is None:
            print("  <none found>")
        else:
            print(json.dumps(params, indent=2))
        print()


if __name__ == "__main__":
    main()