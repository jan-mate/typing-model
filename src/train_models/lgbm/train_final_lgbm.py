import sys
sys.path.append('/content/typing-model')

import os
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import warnings

from src.config import model_dir, ENRICHED_DATA_PATH
from src.models.wrappers import LGBM_FEATURES
from src.utils.data_loader import prepare_sequential_data, get_train_val_masks, get_categorical_indices
from src.utils.training_utils import SEED, load_params, seed_everything

warnings.filterwarnings("ignore")

CFG = {
    "data_path": ENRICHED_DATA_PATH,
    "model_save_dir": model_dir("lgbm"),
    "target": "iki_z",
    "subset": None,
    "n_estimators": 1000,
    "patience": 50,
    "folds_to_train": list(range(10)),
}

ACTIVE_FEATURES = LGBM_FEATURES


def main():
    import pandas as pd
    seed_everything()
    os.makedirs(CFG["model_save_dir"], exist_ok=True)

    json_path = os.path.join(os.path.dirname(CFG["model_save_dir"]), "best_optuna_lgbm_params.json")
    best_params = load_params(json_path)

    df_raw = pd.read_parquet(CFG["data_path"])
    if CFG["subset"]:
        df_raw = df_raw.head(CFG["subset"]).copy()

    X, y, f_b, f_w = prepare_sequential_data(df_raw, CFG["target"], best_params["w_back"], best_params["w_ahead"], ACTIVE_FEATURES)
    del df_raw

    cat_indices = get_categorical_indices(ACTIVE_FEATURES, best_params["w_back"], best_params["w_ahead"])

    lgbm_params = {
        "objective": "regression_l1", "metric": "l1", "verbosity": -1, "n_jobs": -1,
        "device": "gpu",
        "learning_rate": best_params["learning_rate"], "num_leaves": best_params["num_leaves"],
        "max_depth": best_params["max_depth"], "min_child_samples": best_params["min_child_samples"],
        "subsample": best_params["subsample"], "colsample_bytree": best_params["colsample_bytree"],
        "reg_alpha": best_params["reg_alpha"], "reg_lambda": best_params["reg_lambda"],
        "seed": SEED,
    }

    fold_scores = []
    for fold_id in tqdm(CFG["folds_to_train"], desc="Overall Training Progress"):
        model_path = os.path.join(CFG["model_save_dir"], f"final_lgbm_fold_{fold_id}.txt")
        done_path = f"{model_path}.done"
        if os.path.exists(done_path):
            continue

        train_mask, val_mask = get_train_val_masks(f_b, f_w, fold_id)
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        model = lgb.train(
            lgbm_params,
            lgb.Dataset(X_train, label=y_train, categorical_feature=cat_indices),
            num_boost_round=CFG["n_estimators"],
            valid_sets=[lgb.Dataset(X_val, label=y_val, categorical_feature=cat_indices)],
            callbacks=[lgb.early_stopping(CFG["patience"], verbose=False)],
        )

        fold_mae = mean_absolute_error(y_val, model.predict(X_val))
        fold_scores.append(fold_mae)
        print(f"Finished Fold {fold_id} - Val MAE: {fold_mae:.4f}")

        model.save_model(model_path)
        with open(done_path, "w") as f:
            f.write("1")

    if fold_scores:
        print(f"\nAverage LGBM MAE across newly trained folds: {np.mean(fold_scores):.6f}")


if __name__ == "__main__":
    main()
