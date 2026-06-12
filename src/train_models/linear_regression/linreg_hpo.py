import sys
sys.path.append('/content/typing-model')

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import optuna
import json
import warnings

from src.config import model_dir, ENRICHED_DATA_PATH
from src.models.wrappers import LINREG_FEATURES
from src.utils.data_loader import prepare_sequential_data, get_train_val_masks
from src.utils.training_utils import SEED, seed_everything

warnings.filterwarnings('ignore')

CFG = {
    "data_path": ENRICHED_DATA_PATH,
    "model_save_dir": model_dir("linreg", subdir=""),
    "target": "iki_z",
    "subset": None,
    "n_trials": 20, # Baseline doesn't need huge trial counts
    "fold_id": 1,
    "fold_id": 1,
    "w_back_grid": [1, 2, 3],
    "w_ahead_grid":[0, 1]
}


ACTIVE_FEATURES = LINREG_FEATURES

MAX_W_BACK = max(CFG["w_back_grid"])
MAX_W_AHEAD = max(CFG["w_ahead_grid"])
NUM_FEAT = len(ACTIVE_FEATURES)

def train_eval_fold(X_train, y_train, X_val, y_val, params):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model = SGDRegressor(
        loss='epsilon_insensitive', epsilon=0.0, penalty=params['penalty'],
        alpha=params['alpha'], l1_ratio=params.get('l1_ratio', 0.15),
        learning_rate=params['learning_rate'], eta0=params['eta0'],
        max_iter=1000, tol=1e-3, random_state=42
    )
    model.fit(X_train_scaled, y_train)
    return mean_absolute_error(y_val, model.predict(X_val_scaled))

def objective(trial, X_max, y, f_b, f_w, w_b, w_a):
    penalty = trial.suggest_categorical("penalty",["l2", "l1", "elasticnet"])
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else 0.15
    learning_rate = trial.suggest_categorical("learning_rate", ["invscaling", "adaptive"])
    eta0 = trial.suggest_float("eta0", 1e-3, 1e-1, log=True)
    
    start_idx = (MAX_W_BACK - w_b) * NUM_FEAT
    end_idx = (MAX_W_BACK + 1 + w_a) * NUM_FEAT
    X = X_max[:, start_idx:end_idx]
    
    train_mask, val_mask = get_train_val_masks(f_b, f_w, CFG["fold_id"])
    
    params = {'penalty': penalty, 'alpha': alpha, 'l1_ratio': l1_ratio, 'learning_rate': learning_rate, 'eta0': eta0}
    return train_eval_fold(X[train_mask], y[train_mask], X[val_mask], y[val_mask], params)

def main():
    seed_everything()
    os.makedirs(CFG["model_save_dir"], exist_ok=True)
    df_raw = pd.read_parquet(CFG["data_path"])
    if CFG["subset"]: df_raw = df_raw.head(CFG["subset"]).copy()
        
    print(f"Loading MAX window array into memory (w_back={MAX_W_BACK}, w_ahead={MAX_W_AHEAD})...")
    X_max, y, f_b, f_w = prepare_sequential_data(df_raw, CFG["target"], MAX_W_BACK, MAX_W_AHEAD, ACTIVE_FEATURES)
    del df_raw
    
    db_path = os.path.join(CFG["model_save_dir"], "optuna_linreg_study.db")
    sampler = optuna.samplers.TPESampler(seed=SEED)

    best_overall_mae = float('inf')
    best_overall_params = None

    print(f"\n{'='*50}\nBaseline Linear Regression HPO Grid\n{'='*50}")

    for w_b in CFG["w_back_grid"]:
        for w_a in CFG["w_ahead_grid"]:
            study_name = f"linreg_hpo_w{w_b}_a{w_a}"
            study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{db_path}", load_if_exists=True, direction="minimize", sampler=sampler)
            
            remaining_trials = CFG["n_trials"] - len(study.trials)
            
            if remaining_trials > 0:
                print(f"Resuming {study_name}: {len(study.trials)} existing trials. Running {remaining_trials} more...")
                study.optimize(lambda trial: objective(trial, X_max, y, f_b, f_w, w_b, w_a), n_trials=remaining_trials)
            else:
                print(f"Skipping {study_name}: Already completed {len(study.trials)} trials.")
            
            mae = study.best_value
            print(f"w_back={w_b}, w_ahead={w_a} | Best MAE: {mae:.5f}")
            
            if mae < best_overall_mae:
                best_overall_mae = mae
                best_overall_params = study.best_params
                best_overall_params["w_back"] = w_b
                best_overall_params["w_ahead"] = w_a
                
    json_path = os.path.join(CFG["model_save_dir"], "best_linreg_params.json")
    with open(json_path, "w") as f:
        json.dump(best_overall_params, f, indent=4)
        
    print(f"\nOptimization Complete. Best MAE: {best_overall_mae:.5f}")
    print(f"Saved best global params to {json_path}")

if __name__ == "__main__":
    main()