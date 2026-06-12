import sys
sys.path.append('/content/typing-model')

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import warnings

from src.config import ENRICHED_DATA_PATH
from src.utils.data_loader import get_all_features, prepare_sequential_data, get_train_val_masks
from src.utils.training_utils import mae_with_ci, seed_everything

warnings.filterwarnings("ignore")

CFG = {
    "data_path": ENRICHED_DATA_PATH,
    "target": "iki_z",
    "w_back": 2,
    "w_ahead": 1,
    "fold_id": 0,
    "alpha_grid": np.logspace(-4, -1, 10),
    "subset": None,
}

ACTIVE_FEATURES = get_all_features()


def fit_l1(X_train, y_train, X_val, y_val, alpha):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = SGDRegressor(
        loss="epsilon_insensitive", epsilon=0.0, penalty="l1",
        alpha=alpha, learning_rate="invscaling", eta0=0.01,
        max_iter=1000, tol=1e-3, random_state=42,
    )
    model.fit(X_train_s, y_train)
    preds = model.predict(X_val_s)
    mae, ci_low, ci_high = mae_with_ci(y_val, preds)
    return model, mae, ci_low, ci_high


def selected_features_from_coef(coef, features, w_back, w_ahead):
    num_steps = w_back + w_ahead + 1
    num_features = len(features)
    selected = []
    for i, feat in enumerate(features):
        cols = [i + step * num_features for step in range(num_steps)]
        if np.any(coef[cols] != 0.0):
            selected.append(feat)
    return selected


def main():
    seed_everything()

    df_raw = pd.read_parquet(CFG["data_path"])
    if CFG["subset"]:
        df_raw = df_raw.head(CFG["subset"]).copy()

    X, y, f_b, f_w = prepare_sequential_data(
        df_raw, CFG["target"], CFG["w_back"], CFG["w_ahead"], ACTIVE_FEATURES
    )
    del df_raw

    train_mask, val_mask = get_train_val_masks(f_b, f_w, CFG["fold_id"])
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"\n{'='*60}")
    print(f"L1 Feature Selection (fold {CFG['fold_id']}, w_back={CFG['w_back']}, w_ahead={CFG['w_ahead']})")
    print(f"Total candidate features: {len(ACTIVE_FEATURES)}")
    print(f"{'='*60}\n")

    print(f"{'alpha':>10s} | {'val MAE':>10s} | {'CI95 low':>10s} | {'CI95 high':>10s} | {'n_selected':>10s}")
    print("-" * 64)

    results = []
    for alpha in CFG["alpha_grid"]:
        model, mae, ci_low, ci_high = fit_l1(X_train, y_train, X_val, y_val, alpha)
        selected = selected_features_from_coef(model.coef_, ACTIVE_FEATURES, CFG["w_back"], CFG["w_ahead"])
        results.append({"alpha": alpha, "mae": mae, "ci_low": ci_low, "ci_high": ci_high, "selected": selected, "model": model})
        print(f"{alpha:10.5f} | {mae:10.5f} | {ci_low:10.5f} | {ci_high:10.5f} | {len(selected):10d}")

    best = min(results, key=lambda r: r["mae"])
    selected = best["selected"]
    dropped = [f for f in ACTIVE_FEATURES if f not in selected]

    print(f"\n{'='*60}")
    print(f"Best alpha: {best['alpha']:.5f}  |  Val MAE: {best['mae']:.5f}  [95% CI: {best['ci_low']:.5f}, {best['ci_high']:.5f}]  |  n_selected: {len(selected)}/{len(ACTIVE_FEATURES)}")
    print(f"{'='*60}")

    print(f"\nDropped features ({len(dropped)}):")
    for f in dropped:
        print(f"  - {f}")

    print(f"\n--- Copy-paste into src/models/wrappers.py as LINREG_FEATURES ---\n")
    print("LINREG_FEATURES = [")
    for f in selected:
        print(f'    "{f}",')
    print("]")


if __name__ == "__main__":
    main()