import sys
sys.path.append('/content/typing-model')

import os
import json
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import optuna

from src.config import model_dir, ENRICHED_DATA_PATH
from src.models.mlp_arch import DynamicMLP
from src.utils.data_loader import MLP_DL_FEATURES, prepare_sequential_data, get_train_val_masks
from src.utils.training_utils import SEED, seed_everything, seed_worker
warnings.filterwarnings('ignore')

CFG = {
    "data_path": ENRICHED_DATA_PATH,
    "model_save_dir": model_dir("mlp_dl", subdir=""),
    "target": "iki_z",
    "subset": None,
    "fold_id": 1,
    
    "RUN_PHASE_2": True,
    "phase_1_trials": 12,
    "w_back_grid":[2, 3],
    "w_ahead_grid": [1, 2],
    "phase_2_trials": 38,
    "PHASE_2_W_BACK": 2,
    "PHASE_2_W_AHEAD": 1,
    
    "epochs": 30,
    "patience": 3,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_workers": 4,
}

ACTIVE_FEATURES = MLP_DL_FEATURES

MAX_W_BACK = max(max(CFG["w_back_grid"]), CFG["PHASE_2_W_BACK"])
MAX_W_AHEAD = max(max(CFG["w_ahead_grid"]), CFG["PHASE_2_W_AHEAD"])

def train_eval_fold(X_train, y_train, X_val, y_val, params, trial=None):
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=params['batch_size'], shuffle=True, pin_memory=True, num_workers=CFG["num_workers"], worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=params['batch_size'], pin_memory=True, num_workers=CFG["num_workers"])
    
    model = DynamicMLP(X_train.shape[1], params['n_layers'], params['hidden_dim'], params['dropout'], params['activation']).to(CFG["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    criterion = nn.L1Loss()
    
    best_val_mae = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(CFG["epochs"]):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(CFG["device"], non_blocking=True), by.to(CFG["device"], non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_maes =[]
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(CFG["device"], non_blocking=True), by.to(CFG["device"], non_blocking=True)
                val_maes.append(criterion(model(bx), by).item())
        avg_val_mae = np.mean(val_maes)
        
        if trial is not None:
            trial.report(avg_val_mae, epoch)
            if trial.should_prune(): raise optuna.TrialPruned()
                
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            epochs_no_improve = 0
        else: epochs_no_improve += 1
            
        scheduler.step(avg_val_mae)
        if epochs_no_improve >= CFG["patience"]: break
    return best_val_mae

def objective(trial, X_max, y, f_b, f_w, fixed_w_b, fixed_w_a):
    w_b = trial.suggest_int("w_back", fixed_w_b, fixed_w_b)
    w_a = trial.suggest_int("w_ahead", fixed_w_a, fixed_w_a)
    
    params = {
        'n_layers': trial.suggest_int("n_layers", 1, 5),
        'hidden_dim': trial.suggest_categorical("hidden_dim",[64, 128, 256, 512, 1024]),
        'dropout': trial.suggest_float("dropout", 0.0, 0.5),
        'activation': trial.suggest_categorical("activation",["ReLU", "GELU", "SiLU"]),
        'lr': trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        'batch_size': trial.suggest_categorical("batch_size",[8192, 16384, 32768, 65536])
    }
    
    start_idx = (MAX_W_BACK - w_b) * len(ACTIVE_FEATURES)
    end_idx = (MAX_W_BACK + 1 + w_a) * len(ACTIVE_FEATURES)
    X = X_max[:, start_idx:end_idx]
    
    train_mask, val_mask = get_train_val_masks(f_b, f_w, CFG["fold_id"])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    return train_eval_fold(X_train, y[train_mask], scaler.transform(X[val_mask]), y[val_mask], params, trial=trial)

def main():
    seed_everything(use_torch=True)
    os.makedirs(CFG["model_save_dir"], exist_ok=True)
    df_raw = pd.read_parquet(CFG["data_path"])
    if CFG["subset"]: df_raw = df_raw.head(CFG["subset"]).copy()

    X_max, y, f_b, f_w = prepare_sequential_data(df_raw, CFG["target"], MAX_W_BACK, MAX_W_AHEAD, ACTIVE_FEATURES)
    del df_raw

    db_path = os.path.join(CFG["model_save_dir"], "optuna_dl.db")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    sampler = optuna.samplers.TPESampler(seed=SEED)

    if not CFG["RUN_PHASE_2"]:
        for w_b in CFG["w_back_grid"]:
            for w_a in CFG["w_ahead_grid"]:
                study = optuna.create_study(study_name=f"dl_w{w_b}_a{w_a}", storage=f"sqlite:///{db_path}", load_if_exists=True, direction="minimize", pruner=pruner, sampler=sampler)

                n_trials_to_run = max(0, CFG["phase_1_trials"] - len(study.trials))
                if n_trials_to_run > 0:
                    study.optimize(lambda trial: objective(trial, X_max, y, f_b, f_w, w_b, w_a), n_trials=n_trials_to_run)

                print(f"w_back={w_b}, w_ahead={w_a} Best MAE: {study.best_value:.5f}")
    else:
        best_w_b, best_w_a = CFG["PHASE_2_W_BACK"], CFG["PHASE_2_W_AHEAD"]
        final_study = optuna.create_study(study_name=f"dl_w{best_w_b}_a{best_w_a}", storage=f"sqlite:///{db_path}", load_if_exists=True, direction="minimize", pruner=pruner, sampler=sampler)
        
        n_trials_to_run = max(0, CFG["phase_2_trials"] - len(final_study.trials))
        if n_trials_to_run > 0:
            final_study.optimize(lambda trial: objective(trial, X_max, y, f_b, f_w, best_w_b, best_w_a), n_trials=n_trials_to_run)
        
        final_params = final_study.best_params
        final_params.update({"w_back": best_w_b, "w_ahead": best_w_a})
        with open(os.path.join(CFG["model_save_dir"], "best_dl_params.json"), "w") as f: json.dump(final_params, f, indent=4)
        print("Phase 2 Complete! Saved JSON.")

if __name__ == "__main__":
    main()