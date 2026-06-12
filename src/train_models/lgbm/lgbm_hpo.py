import sys
sys.path.append('/content/typing-model')

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import optuna
from optuna.integration import LightGBMPruningCallback
import json
import warnings

from src.config import model_dir, ENRICHED_DATA_PATH
from src.models.wrappers import LGBM_FEATURES
from src.utils.data_loader import prepare_sequential_data, get_train_val_masks, get_categorical_indices
from src.utils.training_utils import SEED, seed_everything

warnings.filterwarnings('ignore')

CFG = {
    "data_path": ENRICHED_DATA_PATH,
    "model_save_dir": model_dir("lgbm", subdir=""),
    "target": "iki_z",
    "subset": None,
    "fold_id": 1,
    
    "RUN_PHASE_2": False,
    "phase_1_trials": 12,
    "w_back_grid": [1, 2, 3],
    "w_ahead_grid": [0, 1, 2],
    
    "phase_2_trials": 38,
    "PHASE_2_W_BACK": 3, 
    "PHASE_2_W_AHEAD": 1,
    
    "n_estimators": 1000,
    "patience": 50
}

ACTIVE_FEATURES = LGBM_FEATURES

NUM_FEAT = len(ACTIVE_FEATURES)
MAX_W_BACK = max(max(CFG["w_back_grid"]), CFG["PHASE_2_W_BACK"])
MAX_W_AHEAD = max(max(CFG["w_ahead_grid"]), CFG["PHASE_2_W_AHEAD"])

def train_eval_fold(X_train, y_train, X_val, y_val, params, cat_indices, trial=None):
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_indices)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=cat_indices)
    
    callbacks =[lgb.early_stopping(stopping_rounds=CFG["patience"], verbose=False)]
    if trial is not None:
        callbacks.append(LightGBMPruningCallback(trial, 'l1'))
        
    lgbm_params = {
        'objective': 'regression_l1', 'metric': 'l1', 'verbosity': -1, 'n_jobs': -1,
        'device': 'gpu',
        'learning_rate': params['learning_rate'], 'num_leaves': params['num_leaves'],
        'max_depth': params['max_depth'], 'min_child_samples': params['min_child_samples'],
        'subsample': params['subsample'], 'colsample_bytree': params['colsample_bytree'],
        'reg_alpha': params['reg_alpha'], 'reg_lambda': params['reg_lambda'],
        'seed': SEED
    }
    
    model = lgb.train(lgbm_params, train_data, num_boost_round=CFG["n_estimators"], valid_sets=[val_data], callbacks=callbacks)
    return mean_absolute_error(y_val, model.predict(X_val))

def objective(trial, X_max, y, f_b, f_w, fixed_w_b, fixed_w_a):
    w_b = trial.suggest_int("w_back", fixed_w_b, fixed_w_b)
    w_a = trial.suggest_int("w_ahead", fixed_w_a, fixed_w_a)
        
    params = {
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int("num_leaves", 15, 255),
        'max_depth': trial.suggest_int("max_depth", -1, 15),
        'min_child_samples': trial.suggest_int("min_child_samples", 10, 100),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
    }
    
    start_idx = (MAX_W_BACK - w_b) * NUM_FEAT
    end_idx = (MAX_W_BACK + 1 + w_a) * NUM_FEAT
    X = X_max[:, start_idx:end_idx]
    
    cat_indices = get_categorical_indices(ACTIVE_FEATURES, w_b, w_a)
    
    train_mask, val_mask = get_train_val_masks(f_b, f_w, CFG["fold_id"])
    return train_eval_fold(X[train_mask], y[train_mask], X[val_mask], y[val_mask], params, cat_indices, trial=trial)

def main():
    seed_everything()
    os.makedirs(CFG["model_save_dir"], exist_ok=True)
    df_raw = pd.read_parquet(CFG["data_path"])
    if CFG["subset"]: df_raw = df_raw.head(CFG["subset"]).copy()
        
    X_max, y, f_b, f_w = prepare_sequential_data(df_raw, CFG["target"], MAX_W_BACK, MAX_W_AHEAD, ACTIVE_FEATURES)
    del df_raw
    
    db_path = os.path.join(CFG["model_save_dir"], "optuna_lgbm_study.db")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    sampler = optuna.samplers.TPESampler(seed=SEED)

    if not CFG["RUN_PHASE_2"]:
        print(f"\n{'='*50}\nPHASE 1: LGBM Grid Search\n{'='*50}")
        for w_b in CFG["w_back_grid"]:
            for w_a in CFG["w_ahead_grid"]:
                study_name = f"lgbm_study_w{w_b}_a{w_a}"
                study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{db_path}", load_if_exists=True, direction="minimize", pruner=pruner, sampler=sampler)
                n_trials_to_run = max(0, CFG["phase_1_trials"] - len(study.trials))
                if n_trials_to_run > 0:
                    study.optimize(lambda trial: objective(trial, X_max, y, f_b, f_w, w_b, w_a), n_trials=n_trials_to_run)
                print(f"w_back={w_b}, w_ahead={w_a} Best MAE: {study.best_value:.5f}")
    else:
        best_w_b, best_w_a = CFG["PHASE_2_W_BACK"], CFG["PHASE_2_W_AHEAD"]
        print(f"\n{'='*50}\nPHASE 2: Deep Dive (w_back={best_w_b}, w_ahead={best_w_a})\n{'='*50}")

        final_study = optuna.create_study(study_name=f"lgbm_study_w{best_w_b}_a{best_w_a}", storage=f"sqlite:///{db_path}", load_if_exists=True, direction="minimize", pruner=pruner, sampler=sampler)
        n_trials_to_run = max(0, CFG["phase_2_trials"] - len(final_study.trials))
        if n_trials_to_run > 0:
            final_study.optimize(lambda trial: objective(trial, X_max, y, f_b, f_w, best_w_b, best_w_a), n_trials=n_trials_to_run)
        
        final_params = final_study.best_params
        final_params.update({"w_back": best_w_b, "w_ahead": best_w_a})
        
        json_path = os.path.join(CFG["model_save_dir"], "best_optuna_lgbm_params.json")
        with open(json_path, "w") as f: json.dump(final_params, f, indent=4)
        print(f"Phase 2 Complete! Saved JSON to {json_path}")

if __name__ == "__main__":
    main()