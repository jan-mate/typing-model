import sys
sys.path.append('/content/typing-model')

import os
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import json
import joblib
import warnings

import optuna

from src.config import model_dir, ENRICHED_DATA_PATH
from src.models.wrappers import LINREG_FEATURES
from src.utils.data_loader import prepare_sequential_data, get_train_val_masks

warnings.filterwarnings("ignore")

CFG = {
    "data_path": ENRICHED_DATA_PATH,
    "model_save_dir": model_dir("linreg"),
    "target": "iki_z",
    "subset": None,
    "folds_to_train": list(range(10)),
    "w_back": 1,
    "w_ahead": 1,
}

ACTIVE_FEATURES = LINREG_FEATURES


def main():
    import pandas as pd

    os.makedirs(CFG["model_save_dir"], exist_ok=True)

    db_path = os.path.join(os.path.dirname(CFG["model_save_dir"]), "optuna_linreg_study.db")
    study_name = f"linreg_hpo_w{CFG['w_back']}_a{CFG['w_ahead']}"
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
    best_params = study.best_params.copy()
    best_params["w_back"] = CFG["w_back"]
    best_params["w_ahead"] = CFG["w_ahead"]
    print(f"Loaded best params from '{study_name}' (MAE: {study.best_value:.6f}): {best_params}")

    ensemble_dir = CFG["model_save_dir"]
    os.makedirs(ensemble_dir, exist_ok=True)

    df_raw = pd.read_parquet(CFG["data_path"])
    if CFG["subset"]:
        df_raw = df_raw.head(CFG["subset"]).copy()

    X, y, f_b, f_w = prepare_sequential_data(
        df_raw, CFG["target"], best_params["w_back"], best_params["w_ahead"], ACTIVE_FEATURES
    )
    del df_raw

    for fold_id in tqdm(CFG["folds_to_train"], desc="Overall Training Progress"):
        model_path = os.path.join(ensemble_dir, f"final_linreg_fold_{fold_id}.pkl")
        scaler_path = os.path.join(ensemble_dir, f"final_linreg_scaler_fold_{fold_id}.pkl")
        done_path = f"{model_path}.done"

        if os.path.exists(done_path):
            continue

        train_mask, val_mask = get_train_val_masks(f_b, f_w, fold_id)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[train_mask])
        X_val_scaled = scaler.transform(X[val_mask])

        model = SGDRegressor(
            loss="epsilon_insensitive", epsilon=0.0, penalty=best_params["penalty"],
            alpha=best_params["alpha"], l1_ratio=best_params.get("l1_ratio", 0.15),
            learning_rate=best_params["learning_rate"], eta0=best_params["eta0"],
            max_iter=1000, tol=1e-3, random_state=42,
        )
        model.fit(X_train_scaled, y[train_mask])

        val_mae = mean_absolute_error(y[val_mask], model.predict(X_val_scaled))
        print(f"Finished Fold {fold_id} - Val MAE: {val_mae:.6f}")

        joblib.dump(scaler, scaler_path)
        joblib.dump(model, model_path)
        with open(done_path, "w") as f:
            f.write("1")

    json_path = os.path.join(os.path.dirname(CFG["model_save_dir"]), "best_linreg_params.json")
    with open(json_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Saved params to {json_path}")


if __name__ == "__main__":
    main()
