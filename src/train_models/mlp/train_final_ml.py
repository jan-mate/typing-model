import sys
sys.path.append('/content/typing-model')

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

from src.config import model_dir, ENRICHED_DATA_PATH
from src.models.mlp_arch import DynamicMLP
from src.models.wrappers import MLP_MAIN_FEATURES
from src.utils.data_loader import prepare_sequential_data, get_train_val_masks
from src.utils.training_utils import SEED, load_params, seed_everything, seed_worker

CFG = {
    "data_path": ENRICHED_DATA_PATH,
    "model_save_dir": model_dir("mlp_main"),
    "subset": None,
    "epochs": 40,
    "patience": 5,
    "folds_to_train": list(range(10)),
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_workers": 4,
}

ACTIVE_FEATURES = MLP_MAIN_FEATURES


def main():
    seed_everything(use_torch=True)
    json_path = os.path.join(os.path.dirname(CFG["model_save_dir"]), "best_optuna_mlp_params.json")
    best_params = load_params(json_path)

    ensemble_dir = CFG["model_save_dir"]
    os.makedirs(ensemble_dir, exist_ok=True)

    df_raw = pd.read_parquet(CFG["data_path"])
    if CFG["subset"]:
        df_raw = df_raw.head(CFG["subset"]).copy()

    X, y, f_b, f_w = prepare_sequential_data(df_raw, "iki_z", best_params["w_back"], best_params["w_ahead"], ACTIVE_FEATURES)
    del df_raw

    fold_scores = []
    for fold_id in tqdm(CFG["folds_to_train"], desc="Overall Folds Progress"):
        model_path = os.path.join(ensemble_dir, f"final_mlp_main_fold_{fold_id}.pth")
        scaler_path = os.path.join(ensemble_dir, f"final_mlp_main_scaler_fold_{fold_id}.pkl")
        done_path = os.path.join(ensemble_dir, f"final_mlp_main_fold_{fold_id}.done")

        if os.path.exists(done_path):
            continue

        print(f"\n--- Training MLP Main on Fold {fold_id} ---")
        train_mask, val_mask = get_train_val_masks(f_b, f_w, fold_id)

        scaler = StandardScaler()
        X_train, y_train = scaler.fit_transform(X[train_mask]), y[train_mask]
        X_val, y_val = scaler.transform(X[val_mask]), y[val_mask]
        joblib.dump(scaler, scaler_path)

        g = torch.Generator()
        g.manual_seed(SEED)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=best_params["batch_size"], shuffle=True, pin_memory=True, num_workers=CFG["num_workers"],
            worker_init_fn=seed_worker, generator=g,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            batch_size=best_params["batch_size"], pin_memory=True, num_workers=CFG["num_workers"],
        )

        model = DynamicMLP(
            X_train.shape[1], best_params["n_layers"], best_params["hidden_dim"],
            best_params["dropout"], best_params["activation"],
        ).to(CFG["device"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
        criterion = nn.L1Loss()

        best_val_mae = float("inf")
        epochs_no_improve = 0

        for epoch in tqdm(range(CFG["epochs"]), desc=f"Epochs Fold {fold_id}", leave=False):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(CFG["device"], non_blocking=True), by.to(CFG["device"], non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                criterion(model(bx), by).backward()
                optimizer.step()

            model.eval()
            val_maes = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(CFG["device"], non_blocking=True), by.to(CFG["device"], non_blocking=True)
                    val_maes.append(criterion(model(bx), by).item())

            avg_val_mae = np.mean(val_maes)
            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                epochs_no_improve = 0
                torch.save(model.state_dict(), model_path)
            else:
                epochs_no_improve += 1

            scheduler.step(avg_val_mae)
            if epochs_no_improve >= CFG["patience"]:
                break

        fold_scores.append(best_val_mae)
        print(f"Finished Fold {fold_id} - Best Val MAE: {best_val_mae:.6f}")
        with open(done_path, "w") as f:
            f.write("1")

    if fold_scores:
        print(f"\nAverage MLP Main MAE of newly trained folds: {np.mean(fold_scores):.6f}")


if __name__ == "__main__":
    main()
